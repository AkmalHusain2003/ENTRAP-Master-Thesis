from google.colab import drive
drive.mount('/content/drive')

import os
import re
import time
import warnings
import pandas as pd
import pyvo


_pyvo_version = tuple(int(x) for x in pyvo.__version__.split(".")[:2])
if _pyvo_version < (1, 4):
    warnings.warn(
        f"pyvo versi {pyvo.__version__} terdeteksi. "
        "Direkomendasikan >= 1.4 untuk async TAP yang stabil. "
        "Install: !pip install 'pyvo>=1.4'"
    )

print(f"pyvo versi: {pyvo.__version__}")
print("Library siap.")


CENTERS_PATH   = '/content/drive/MyDrive/Tesis/cluster_centers.csv'
FOLDER_RESULTS = '/content/drive/MyDrive/Tesis/Results_Mock/'

os.makedirs(FOLDER_RESULTS, exist_ok=True)

centers_data = pd.read_csv(CENTERS_PATH)

# Hapus kolom index Pandas yang tersimpan jika ada
if 'Unnamed: 0' in centers_data.columns:
    centers_data = centers_data.drop('Unnamed: 0', axis=1)

# Validasi kolom wajib
required = {'cluster_name', 'ra', 'dec', 'radius'}
missing  = required - set(centers_data.columns)
if missing:
    raise ValueError(f"Kolom kurang di cluster_centers.csv: {missing}")

print(f"Jumlah cluster: {len(centers_data)}")
print(centers_data)


_GAVO_TAP_URL = "https://dc.g-vo.org/tap"

try:
    TAP_SERVICE = pyvo.dal.TAPService(_GAVO_TAP_URL)
    # Verifikasi koneksi dengan query minimal
    _test = TAP_SERVICE.run_sync(
        "SELECT TOP 1 source_id FROM gedr3mock.main",
        maxrec=1
    )
    print(f"[OK] TAP service GAVO terhubung: {_GAVO_TAP_URL}")
    print(f"     Test query: {len(_test)} baris diterima.")
except Exception as exc:
    raise RuntimeError(
        f"Gagal terhubung ke TAP service GAVO.\n"
        f"  URL  : {_GAVO_TAP_URL}\n"
        f"  Error: {exc}\n"
        "Periksa koneksi Colab dan status: https://gea.esac.esa.int/gaiastatus/"
    ) from exc



def _safe_name(cluster_name: str) -> str:
    """
    Sanitasi nama cluster menjadi string aman untuk nama file.
    HARUS identik dengan fungsi yang sama di pipeline.py:
      re.sub(r"[^\\w\\-]", "_", cluster_name)
    """
    return re.sub(r"[^\w\-]", "_", cluster_name)


_GEDR3_COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "parallax",
    "pmra",
    "pmdec",
    "radial_velocity",   
    "l",                 
    "b",                 
    "phot_g_mean_mag",
    "bp_rp",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
]

_EXCLUDE_POPID = 11


_MAXREC = 10_000_000  


def query_single_cluster(cluster_name: str,
                          center_ra: float,
                          center_dec: float,
                          radius_deg: float,
                          tap_service: pyvo.dal.TAPService,
                          out_dir: str,
                          max_retries: int = 3,
                          sleep_between: float = 2.0) -> pd.DataFrame:
    """
    Cone search GeDR3Mock untuk satu cluster. Hasil di-cache ke parquet.

    Parameters
    ----------
    cluster_name   : Nama cluster (digunakan untuk nama file cache).
    center_ra      : Right Ascension pusat cone search (derajat, ICRS).
    center_dec     : Declination pusat cone search (derajat, ICRS).
    radius_deg     : Radius cone search (derajat).
    tap_service    : Objek pyvo.dal.TAPService yang sudah diinisialisasi.
    out_dir        : Direktori output untuk file parquet cache.
    max_retries    : Jumlah maksimum percobaan ulang jika server error.
    sleep_between  : Detik tunggu antar query sukses; back-off berlipat
                     per attempt: sleep_between * attempt saat retry.

    Returns
    -------
    pd.DataFrame berisi kolom _GEDR3_COLUMNS.

    Notes
    -----
    Kolom Output (identik dengan Gaia DR3 query):
      source_id, ra, dec, parallax, pmra, pmdec, radial_velocity,
      l, b, phot_g_mean_mag, bp_rp, phot_bp_mean_mag, phot_rp_mean_mag

    Eksklusi Open Cluster:
      Filter WHERE popid != 11 mengeluarkan bintang-bintang yang
      di-assign ke stellar clusters (Robin+2003 Besancon model).
      popid tidak dimasukkan ke kolom output agar konsisten dengan
      schema Gaia DR3.

    Pola ADQL (CTE -- direkomendasikan GAVO untuk gedr3mock.main):
      WITH sample AS (
          SELECT * FROM gedr3mock.main
          WHERE CONTAINS(
              POINT('ICRS', ra, dec),
              CIRCLE('ICRS', ra0, dec0, r)
          ) = 1
      )
      SELECT <cols> FROM sample
      WHERE popid != 11

      CTE memastikan indeks spasial digunakan di subquery dalam,
      kemudian filter popid diterapkan pada hasil spatial yang lebih
      kecil -- ini jauh lebih efisien daripada filter AND langsung.

    Strategi Query:
      1. Coba submit_job (async TAP, tidak ada batas baris, lebih
         sesuai untuk query besar pada gedr3mock.main).
      2. Jika gagal, fallback ke run_sync (sinkron, lebih toleran
         terhadap beberapa konfigurasi server).
      3. Kedua strategi di-wrap dalam retry loop hingga max_retries
         dengan exponential back-off (sleep_between * attempt detik).

    Nama File Cache:
      gaia_ref_{safe_name}.parquet
      Konvensi identik dengan pipeline.py agar file langsung terbaca.

    execution_duration:
      Async job diberi batas waktu 7200 detik (2 jam) untuk
      mengakomodasi cone search besar pada tabel ~1.5 miliar bintang.
    """
    sname    = _safe_name(cluster_name)
    out_path = os.path.join(out_dir, f"gaia_ref_{sname}.parquet")

    # --- Cek cache ---
    if os.path.exists(out_path):
        df_cache = pd.read_parquet(out_path)
        print(
            f"  [Cache] {cluster_name}: {len(df_cache):,} bintang "
            f"({os.path.basename(out_path)}) -- dilewati."
        )
        return df_cache

    print(
        f"  [Query] {cluster_name}: "
        f"ra={center_ra:.4f} dec={center_dec:.4f} r={radius_deg:.4f} deg ..."
    )

    
    select_cols = ", ".join(_GEDR3_COLUMNS)
    adql = (
        f"WITH sample AS ( "
        f"  SELECT * FROM gedr3mock.main "
        f"  WHERE CONTAINS( "
        f"    POINT('ICRS', ra, dec), "
        f"    CIRCLE('ICRS', {center_ra:.8f}, {center_dec:.8f}, {radius_deg:.8f})"
        f"  ) = 1 "
        f") "
        f"SELECT {select_cols} "
        f"FROM sample "
        f"WHERE popid != {_EXCLUDE_POPID}"
    )

    result     = None
    last_error = None

    
    for attempt in range(1, max_retries + 1):

        if attempt > 1:
            wait = sleep_between * attempt   # 4s, 6s, 8s untuk sleep_between=2
            print(f"    [Retry {attempt}/{max_retries}] tunggu {wait:.0f}s ...")
            time.sleep(wait)

       
        try:
            job = tap_service.submit_job(adql, maxrec=_MAXREC)
            job.execution_duration = 7200  
            job.wait(phases=["COMPLETED", "ERROR", "ABORTED"],
                     timeout=7200)
     
            job.raise_if_error()
            result = job.fetch_result()
            break  

        except Exception as exc_async:
            warnings.warn(
                f"  submit_job (async) gagal (attempt {attempt}/{max_retries}): "
                f"{exc_async}. Mencoba run_sync ..."
            )

            try:
                result = tap_service.run_sync(adql, maxrec=_MAXREC)
                break   

            except Exception as exc_sync:
                last_error = (exc_async, exc_sync)
                warnings.warn(
                    f"  run_sync juga gagal (attempt {attempt}/{max_retries}): "
                    f"{exc_sync}."
                )

    if result is None:
        exc_async, exc_sync = last_error
        raise RuntimeError(
            f"Query gagal setelah {max_retries} percobaan untuk '{cluster_name}'.\n"
            f"  Async terakhir : {exc_async}\n"
            f"  Sync  terakhir : {exc_sync}\n"
            "Periksa koneksi dan status GAVO: https://dc.g-vo.org/\n"
            f"  ADQL yang dieksekusi:\n{adql}"
        )


    df = result.to_table().to_pandas()


    df.columns = [c.lower() for c in df.columns]

    if len(df) == 0:
        warnings.warn(
            f"Query mengembalikan 0 bintang untuk '{cluster_name}'. "
            "Periksa koordinat center dan radius, atau mungkin area ini "
            "kosong di GeDR3Mock (misalnya di luar footprint model)."
        )

        empty = pd.DataFrame(columns=_GEDR3_COLUMNS)
        empty.to_parquet(out_path, index=False)
        return empty

    missing_cols = set(_GEDR3_COLUMNS) - set(df.columns)
    if missing_cols:
        warnings.warn(
            f"Kolom hilang di hasil query '{cluster_name}': {missing_cols}. "
            "Periksa apakah _GEDR3_COLUMNS sesuai dengan schema gedr3mock.main."
        )

    df.to_parquet(out_path, index=False)
    print(
        f"  [Simpan] {cluster_name}: {len(df):,} bintang "
        f"(popid=11 dikecualikan) -> {os.path.basename(out_path)}"
    )

    time.sleep(sleep_between)
    return df


summary = []

for idx, row in centers_data.iterrows():
    cluster_name = str(row["cluster_name"])
    center_ra    = float(row["ra"])
    center_dec   = float(row["dec"])
    radius_deg   = float(row["radius"])

    try:
        df_result = query_single_cluster(
            cluster_name  = cluster_name,
            center_ra     = center_ra,
            center_dec    = center_dec,
            radius_deg    = radius_deg,
            tap_service   = TAP_SERVICE,
            out_dir       = FOLDER_RESULTS,
            max_retries   = 3,       
            sleep_between = 2.0,     
        )
        summary.append({
            "cluster" : cluster_name,
            "n_gedr3" : len(df_result),
            "status"  : "OK",
            "file"    : f"gaia_ref_{_safe_name(cluster_name)}.parquet",
        })
    except Exception as exc:
        print(f"  [ERROR] {cluster_name}: {exc}")
        summary.append({
            "cluster" : cluster_name,
            "n_gedr3" : 0,
            "status"  : f"ERROR: {exc}",
            "file"    : "",
        })


df_summary = pd.DataFrame(summary)
print("\n" + "=" * 60)
print("RINGKASAN QUERY GeDR3MOCK (popid=11 dikecualikan)")
print("=" * 60)
print(df_summary.to_string(index=False))

n_ok    = int((df_summary["status"] == "OK").sum())
n_total = len(centers_data)
print(f"\nBerhasil : {n_ok}/{n_total} cluster")
print(f"Gagal    : {n_total - n_ok}/{n_total} cluster")
print(f"Hasil disimpan di: {FOLDER_RESULTS}")
print()
print("Catatan:")
print("  - File parquet diberi nama 'gaia_ref_*.parquet' untuk")
print("    kompatibilitas langsung dengan pipeline.py.")
print("  - popid=11 (open clusters, Robin+2003 Besancon model)")
print("    telah dikecualikan dari semua hasil.")
print("  - Kolom output identik dengan Gaia DR3 query:")
print(f"    {_GEDR3_COLUMNS}")