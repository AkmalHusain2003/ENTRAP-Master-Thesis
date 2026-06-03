import os
import math
import time
import tracemalloc
import warnings
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from entrap import ENTRAP  


# =============================================================================
# KONFIGURASI
# =============================================================================

BASE_PATH   = "/home/akmal/Tesis/Mock_Data_Testing/New/Gaia_Final_Mock/"
OUTPUT_PATH = "/home/akmal/Tesis/Mock_Data_Testing/New/ENTRAP_Results/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

FEATURE_COLS = ["ra", "dec", "pmra", "pmdec", "parallax"]

ALL_COMBOS = list(itertools.product([3, 4], [3, 4])) 


# =============================================================================
# UTILITY
# =============================================================================

def load_file(full_path: str) -> pd.DataFrame:
    if full_path.endswith(".parquet"):
        return pd.read_parquet(full_path)
    return pd.read_csv(full_path)


def split_ground_truth(df: pd.DataFrame):
    return df[df["region"] == "background"].copy(), df[df["region"] != "background"].copy()


def compute_cluster_centroid(df_cluster: pd.DataFrame) -> np.ndarray:
    # Mean aritmetik 5-D — estimator tidak-bias untuk profil King/Plummer
    return df_cluster[FEATURE_COLS].mean().to_numpy()


def find_best_matching_cluster(data_arr: np.ndarray, labels: np.ndarray, gt_centroid: np.ndarray) -> int:
    # Pilih label cluster dengan centroid Euclidean 5-D terdekat ke ground-truth
    cluster_labels = np.unique(labels)
    cluster_labels = cluster_labels[cluster_labels >= 0]  # buang noise (-1)

    if len(cluster_labels) == 0:
        return -1

    best_label, min_dist = -1, np.inf
    for lbl in cluster_labels:
        dist = np.linalg.norm(data_arr[labels == lbl].mean(axis=0) - gt_centroid)
        if dist < min_dist:
            min_dist, best_label = dist, lbl

    return best_label


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Positif = anggota cluster (non-background), Negatif = background/noise
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "precision": precision, "recall": recall, "f1_score": f1, "accuracy": accuracy}


def compute_tidal_recovery(df: pd.DataFrame, labels: np.ndarray, best_label: int) -> dict:
    # Recovery total non-background dan per sub-region (core, edge, tidal)
    pred_mask = (labels == best_label) if best_label >= 0 else np.zeros(len(df), dtype=bool)

    total_nonbg = (df["region"] != "background").sum()
    result = {
        "tidal_recovery_pct": (
            100.0 * ((df["region"] != "background") & pred_mask).sum() / total_nonbg
            if total_nonbg > 0 else 0.0
        )
    }
    for region in ["core", "edge", "tidal"]:
        mask  = df["region"] == region
        total = mask.sum()
        result[f"recovery_{region}_pct"] = 100.0 * (mask & pred_mask).sum() / total if total > 0 else 0.0
        result[f"n_{region}"] = int(total)

    return result


# =============================================================================
# GRID SEARCH — tanpa tracemalloc agar secepat mungkin
# =============================================================================

def run_entrap_fast(data_arr: np.ndarray, min_samples: int, min_cluster_size: int) -> np.ndarray:
    return ENTRAP(min_samples=min_samples, min_cluster_size=min_cluster_size,
                  enable_tracking=True).fit_predict(data_arr)


def grid_search(df, data_arr, gt_centroid, y_true, pbar_grid) -> tuple:
    # Skor gabungan: F1 + tidal_recovery/100 (keduanya dalam [0,1], bobot sama)
    best_score, best_params, trial_rows = -np.inf, {"min_samples": 3, "min_cluster_size": 4}, []

    for ms, mcs in ALL_COMBOS:
        t0 = time.perf_counter()
        try:
            labels = run_entrap_fast(data_arr, ms, mcs)
        except Exception as e:
            trial_rows.append({"min_samples": ms, "min_cluster_size": mcs,
                                "score": 0.0, "f1": 0.0, "tidal_pct": 0.0,
                                "elapsed_sec": 0.0, "error": str(e)})
            pbar_grid.update(1)
            continue

        elapsed  = time.perf_counter() - t0
        best_lbl = find_best_matching_cluster(data_arr, labels, gt_centroid)
        y_pred   = (labels == best_lbl).astype(int) if best_lbl >= 0 else np.zeros(len(y_true), dtype=int)

        clf   = compute_classification_metrics(y_true, y_pred)
        tidal = compute_tidal_recovery(df, labels, best_lbl)
        score = clf["f1_score"] + tidal["tidal_recovery_pct"] / 100.0

        trial_rows.append({"min_samples": ms, "min_cluster_size": mcs, "score": score,
                            "f1": clf["f1_score"], "tidal_pct": tidal["tidal_recovery_pct"],
                            "elapsed_sec": elapsed, "error": ""})

        if score > best_score:
            best_score, best_params = score, {"min_samples": ms, "min_cluster_size": mcs}

        pbar_grid.set_postfix({"ms": ms, "mcs": mcs,
                                "F1": f"{clf['f1_score']:.3f}",
                                "TR%": f"{tidal['tidal_recovery_pct']:.1f}",
                                "score": f"{score:.3f}"})
        pbar_grid.update(1)

    return best_params, trial_rows


# =============================================================================
# EVALUASI FINAL
# =============================================================================

def run_entrap_profiled(data_arr, min_samples, min_cluster_size) -> tuple:
    tracemalloc.start()
    t_start = time.perf_counter()
    labels  = ENTRAP(min_samples=min_samples, min_cluster_size=min_cluster_size,
                     enable_tracking=True).fit_predict(data_arr)
    elapsed = time.perf_counter() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return labels, elapsed, peak


def evaluate_best_params(df, data_arr, gt_centroid, best_params) -> dict:
    y_true = (df["region"] != "background").astype(int).to_numpy()
    N      = len(df)

    labels, elapsed_sec, peak_mem_b = run_entrap_profiled(
        data_arr, best_params["min_samples"], best_params["min_cluster_size"]
    )

    best_lbl = find_best_matching_cluster(data_arr, labels, gt_centroid)
    y_pred   = (labels == best_lbl).astype(int) if best_lbl >= 0 else np.zeros(len(y_true), dtype=int)

    clf_m   = compute_classification_metrics(y_true, y_pred)
    tidal_m = compute_tidal_recovery(df, labels, best_lbl)

    return {
        "N_total":                   N,
        "N_cluster_gt":              int((df["region"] != "background").sum()),
        "N_background_gt":           int((df["region"] == "background").sum()),
        "best_min_samples":          best_params["min_samples"],
        "best_min_cluster_size":     best_params["min_cluster_size"],
        "n_clusters_found":          int(np.sum(np.unique(labels) >= 0)),
        "n_noise_points":            int(np.sum(labels == -1)),
        "matched_cluster_label":     int(best_lbl),
        **clf_m,
        **tidal_m,
        "elapsed_sec":               elapsed_sec,
        "peak_memory_bytes":         peak_mem_b,
        "peak_memory_MB":            peak_mem_b / (1024 ** 2),
        "log10_N":                   math.log10(N) if N > 0 else 0.0,
        "log10_elapsed_sec":         math.log10(elapsed_sec) if elapsed_sec > 0 else float("-inf"),
        "log10_peak_memory_bytes":   math.log10(peak_mem_b) if peak_mem_b > 0 else 0.0,
        "time_per_point_us":         (elapsed_sec / N) * 1e6 if N > 0 else 0.0,
        "memory_per_point_bytes":    peak_mem_b / N if N > 0 else 0.0,
    }


# =============================================================================
# PIPELINE PER FILE
# =============================================================================

def process_single_file(file: str, pbar_files: tqdm):
    full_path    = os.path.join(BASE_PATH, file)
    cluster_name = os.path.splitext(file)[0].replace("mock_final_", "")

    pbar_files.set_postfix({"cluster": cluster_name, "step": "load"})

    try:
        df = load_file(full_path)
    except Exception as e:
        print(f"\n[ERROR] Gagal baca {file}: {e}")
        return None

    missing = [c for c in FEATURE_COLS + ["region"] if c not in df.columns]
    if missing:
        print(f"\n[SKIP] {file} — kolom hilang: {missing}")
        return None

    _, df_cluster = split_ground_truth(df)
    if len(df_cluster) == 0:
        print(f"\n[SKIP] {file} — tidak ada titik cluster.")
        return None

    data_arr    = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    gt_centroid = compute_cluster_centroid(df_cluster)
    y_true      = (df["region"] != "background").astype(int).to_numpy()

    # Grid search
    pbar_files.set_postfix({"cluster": cluster_name, "step": "grid"})
    with tqdm(total=len(ALL_COMBOS), desc=f"  Grid [{cluster_name}]",
              leave=False, unit="combo", colour="cyan") as pbar_grid:
        best_params, trial_rows = grid_search(df, data_arr, gt_centroid, y_true, pbar_grid)

    pd.DataFrame(trial_rows).assign(cluster_name=cluster_name).to_parquet(
        os.path.join(OUTPUT_PATH, f"{cluster_name}_grid_trials.parquet"), index=False
    )

    # Evaluasi final + profiling
    pbar_files.set_postfix({"cluster": cluster_name, "step": "eval+profile"})
    try:
        result = evaluate_best_params(df, data_arr, gt_centroid, best_params)
    except Exception as e:
        print(f"\n[ERROR] Evaluasi final {file}: {e}")
        return None

    result.update({
        "cluster_name":    cluster_name,
        "source_file":     file,
        "grid_best_score": max(r["score"] for r in trial_rows),
    })

    row_df = pd.DataFrame([result])
    row_df.to_parquet(os.path.join(OUTPUT_PATH, f"{cluster_name}_results.parquet"), index=False)

    pbar_files.set_postfix({
        "cluster": cluster_name,
        "ms":      best_params["min_samples"],
        "mcs":     best_params["min_cluster_size"],
        "F1":      f"{result['f1_score']:.3f}",
        "TR%":     f"{result['tidal_recovery_pct']:.1f}",
        "t(s)":    f"{result['elapsed_sec']:.2f}",
    })

    return row_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  ENTRAP Benchmark v2  |  Grid Search 3x3  |  No Optuna")
    print(f"  Base  : {BASE_PATH}")
    print(f"  Output: {OUTPUT_PATH}")
    print("=" * 70)

    file_list = sorted([f for f in os.listdir(BASE_PATH)
                        if f.endswith(".parquet") or f.endswith(".csv")])
    if not file_list:
        print("[ERROR] Tidak ada file ditemukan.")
        return

    print(f"  {len(file_list)} file mock ditemukan.\n")
    all_results = []

    with tqdm(file_list, desc="File Mock", unit="file", colour="green", dynamic_ncols=True) as pbar_files:
        for file in pbar_files:
            result_df = process_single_file(file, pbar_files)
            if result_df is not None:
                all_results.append(result_df)

    if not all_results:
        print("\n[WARNING] Tidak ada hasil terkumpul.")
        return

    col_order = [
        "cluster_name", "source_file",
        "N_total", "N_cluster_gt", "N_background_gt",
        "best_min_samples", "best_min_cluster_size",
        "n_clusters_found", "n_noise_points", "matched_cluster_label",
        "TP", "TN", "FP", "FN",
        "accuracy", "precision", "recall", "f1_score",
        "tidal_recovery_pct",
        "recovery_core_pct", "recovery_edge_pct", "recovery_tidal_pct",
        "n_core", "n_edge", "n_tidal",
        "elapsed_sec", "peak_memory_bytes", "peak_memory_MB",
        "log10_N", "log10_elapsed_sec", "log10_peak_memory_bytes",
        "time_per_point_us", "memory_per_point_bytes",
        "grid_best_score",
    ]

    summary_df = pd.concat(all_results, ignore_index=True)
    col_order  = [c for c in col_order if c in summary_df.columns]
    remaining  = [c for c in summary_df.columns if c not in col_order]
    summary_df = summary_df[col_order + remaining]

    out = os.path.join(OUTPUT_PATH, "entrap_benchmark_summary.parquet")
    summary_df.to_parquet(out, index=False)

    print("\n" + "=" * 70)
    print("  RINGKASAN")
    print("=" * 70)
    print(summary_df[[
        "f1_score", "accuracy", "precision", "recall",
        "tidal_recovery_pct", "elapsed_sec", "peak_memory_MB",
    ]].describe().round(4).to_string())
    print("=" * 70)
    print(f"\n  Summary : {out}  |  Cluster: {len(summary_df)}\n")


if __name__ == "__main__":
    main()