"""
N-Body Simulation of Open Clusters using AMUSE + MASC
======================================================
Initial conditions are derived from observations (backward DataFrame) following:

  M_init  : Lamers et al. (2005, A&A 441, 117) — numerical inversion
  r_h,init: Marks & Kroupa (2012, A&A 543, A8)  — embedded cluster M–R relation
  N_init  : Kroupa (2001, MNRAS 322, 231)        — canonical IMF, <m> ≈ 0.4 M_sun

Cluster stars are generated with amuse.ext.masc (new_star_cluster),
which samples the Kroupa IMF and places stars in a Plummer profile.

Each cluster is integrated FORWARD from its birth position/velocity
(x0,y0,z0 / vx0,vy0,vz0 from backward orbit integration) for
tend = Age[Myr] inside the MWPotential2014 (Bovy 2015, ApJS 216, 29).

Galactic potential
------------------
MWPotential2014 is a three-component axisymmetric model fitted to
kinematic observations of the Milky Way:
  • PowerSphericalPotentialwCutoff  — bulge
  • MiyamotoNagaiPotential           — disk
  • NFWPotential                     — halo
Calibrated at ro = 8.0 kpc, vo = 220 km/s (Bovy 2015).

Softening
---------
eps = r_h * N^(-0.3) — Athanassoula et al. (2000, MNRAS 314, 475).
Optimal untuk distribusi Plummer dengan tree code (MISE minimization).

Kicking / ejection prescription  (Kos 2024, A&A 691, A28)
----------------------------------------------------------
Separuh bintang mendapat satu kick acak selama simulasi.
  v_eject(t) = σ_init × (1 − t/τ)^0.5   [km/s]
    σ_init = sqrt(G M_init / (6 r_h))     — dispersi kecepatan Plummer
  P_e(t)  = 2 (1 − t/τ) / τ              — laju ejeksi menurun linier
    Sampling: t_eject = τ × (1 − sqrt(1 − U)), U ~ Uniform(0,1)
  Arah kick: isotropik seragam di seluruh bola
Referensi formula v_eject: mengaproksimasi Moyano Loyola & Hurley (2013)
yang menunjukkan kecepatan escape menurun seiring massa cluster berkurang.

Usage
-----
    results, df_stars = run_all_clusters(backward)

References
----------
Lamers et al. 2005, A&A 441, 117
Marks & Kroupa 2012, A&A 543, A8
Kroupa 2001, MNRAS 322, 231
Baumgardt & Makino 2003, MNRAS 340, 227
Asplund et al. 2009, ARA&A 47, 481  (Z_sun = 0.014)
Bovy 2015, ApJS 216, 29             (MWPotential2014)
Athanassoula et al. 2000, MNRAS 314, 475  (softening)
Kos 2024, A&A 691, A28             (v_eject, sample_kicking)
Moyano Loyola & Hurley 2013, MNRAS 434, 2509  (escape velocity evolution)
"""

import warnings
import numpy as np
import pandas as pd

from galpy.potential import MWPotential2014, to_amuse
from amuse.lab import *
from amuse.units import units
from amuse.couple import bridge
from amuse.ext.masc import new_star_cluster

warnings.filterwarnings("ignore")


# ============================================================
# KONSTANTA FISIKA
# ============================================================
# G dalam satuan pc M_sun^-1 (km/s)^2
# Derivasi: G = 6.674e-11 m^3 kg^-1 s^-2
#             = 4.302e-3 pc M_sun^-1 (km/s)^2
G_PC_MSUN_KMS2 = 4.302e-3


# ============================================================
# GALACTIC CONSTANTS  (MWPotential2014 calibration, Bovy 2015)
# ============================================================
R0_KPC = 8.0     # kpc   — Sun galactocentric distance
V0_KMS = 220.0   # km/s  — local circular speed


# ============================================================
# GALACTIC POTENTIAL
# ============================================================
# to_amuse() wraps MWPotential2014 ke AMUSE-compatible field code.
# MWPotential2014 time-independent (axisymmetric, tanpa bar),
# sehingga galpy mengevaluasi gaya pada seluruh array partikel sekaligus.
# ro=R0_KPC dan vo=V0_KMS harus konsisten dengan kalibrasi MWPotential2014.
mw_amuse = to_amuse(MWPotential2014, ro=R0_KPC, vo=V0_KMS)


# ============================================================
# INITIAL PARAMETER DERIVATION FUNCTIONS
# ============================================================

def compute_M_init(M_obs_msun, age_Myr, R_gal_kpc,
                   gamma=0.62, t0_ref_Myr=810.0):
    """
    Recover M_init dengan inversi numerik persamaan evolusi massa
    Lamers et al. (2005, A&A 441, 117), Eq. 7:

        M(t) = [M_init^(1-γ) − (1-γ)·t/t0]^(1/(1-γ)) × μ_ev(t)

    Parameters
    ----------
    M_obs_msun  : float — present-day observed mass [M_sun]
    age_Myr     : float — cluster age [Myr]
    R_gal_kpc   : float — galactocentric radius at birth [kpc]
    gamma       : float — tidal dissolution exponent
                          0.62 dari Baumgardt & Makino (2003) N-body
    t0_ref_Myr  : float — dissolution timescale di R=8.5 kpc [Myr]
                          810 Myr dari Lamers et al. (2005)

    Returns
    -------
    M_init : float [M_sun]

    Notes
    -----
    mu_ev(t) = 1 - 0.07*(t/Gyr)^0.255  (Lamers+ 2005 Eq. 2;
    clamped di 0.50 untuk cluster sangat tua).
    t0 scales linier dengan R_gal: tidal field lebih kuat mendekati pusat.
    Bisection 200 iterasi; konvergen ke < 1e-7 relative error pada M_init.
    """
    t_9   = age_Myr * 1e6 / 1e9
    mu_ev = max(0.50, 1.0 - 0.07 * t_9**0.255)
    t0    = t0_ref_Myr * (R_gal_kpc / 8.5)

    def M_evolved(M_init):
        arg = M_init**(1.0 - gamma) - (1.0 - gamma) * age_Myr / t0
        if arg <= 0.0:
            return 0.0
        return arg**(1.0 / (1.0 - gamma)) * mu_ev

    lo = float(M_obs_msun)
    hi = float(M_obs_msun) * 100.0

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if M_evolved(mid) >= float(M_obs_msun):
            hi = mid
        else:
            lo = mid
        if hi > 0 and (hi - lo) / hi < 1e-7:
            break

    return 0.5 * (lo + hi)


def compute_r_h_init(M_init_msun):
    """
    Initial half-mass radius dari Marks & Kroupa (2012, A&A 543, A8):

        r_h = 0.1 × (M_ecl / M_sun)^0.13  [pc]

    Diturunkan dari distribusi binding energy bintang-bintang biner
    di open cluster yang mengkonstrain densitas lahir embedded cluster.

    Returns
    -------
    r_h : float [pc]
    """
    return 0.1 * float(M_init_msun)**0.13


def compute_N_init(M_init_msun, mean_mass_msun=0.4):
    """
    Initial number of stars: N_init = M_init / <m>.

    <m> ~ 0.38–0.40 M_sun untuk Kroupa (2001) canonical two-part
    power-law IMF diintegrasikan pada [0.08, 100] M_sun pada t=0.
    Jangan gunakan <m> = 0.5 M_sun (itu untuk populasi evolved).

    Returns
    -------
    N_init : int (minimum 50 untuk numerical stability)
    """
    return max(50, int(round(float(M_init_msun) / mean_mass_msun)))


def feh_to_Z(feh_dex, Z_sun=0.014):
    """
    Konversi [Fe/H] ke absolute metallicity Z.

        Z = Z_sun × 10^([Fe/H])

    Z_sun = 0.014 (Asplund et al. 2009, ARA&A 47, 481).
    Di-clamp ke [1e-4, 0.05] untuk numerical stability stellar models.
    """
    Z = Z_sun * 10.0**float(feh_dex)
    return float(np.clip(Z, 1e-4, 0.05))


# ============================================================
# SOFTENING LENGTH  (Athanassoula et al. 2000)
# ============================================================

def compute_softening_eps(r_h_pc, N_stars):
    """
    Plummer softening length untuk BHTree + distribusi Plummer.

    Formula: Athanassoula et al. (2000, MNRAS 314, 475):
        eps = r_h × N^(-0.3)

    Diturunkan dari minimisasi Mean Integrated Square Error (MISE)
    antara gaya N-body ter-smoothing dan gaya kontinu sejati distribusi
    Plummer. Scaling N^{-0.3} empiris untuk tree code, berlaku N = 10^3–10^5.

    Konteks: valid untuk studi orbit/tidal stripping (BHTree sebagai
    collisionless integrator). Tidak valid untuk studi internal dynamics.

    Bounds
    ------
    atas : 0.1 × r_h  — eps > 0.1 r_h menghaluskan gravitasi inti terlalu agresif
    bawah: 0.001 pc   — di bawah ini timestep BHTree tidak praktis
    """
    eps_pc = r_h_pc * float(N_stars) ** (-0.30)
    return float(np.clip(eps_pc, 0.001, 0.10 * r_h_pc))


# ============================================================
# KICKING PRESCRIPTION  (Kos 2024, A&A 691, A28)
# ============================================================

def compute_sigma_init(M_init_msun, r_h_pc):
    """
    Dispersi kecepatan 1D Plummer sphere dalam virial equilibrium:

        σ_init = sqrt(G × M_init / (6 × r_h))  [km/s]

    Skala kecepatan karakteristik cluster; digunakan sebagai amplitudo
    dasar v_eject pada t=0.

    Parameters
    ----------
    M_init_msun : float — initial cluster mass [M_sun]
    r_h_pc      : float — initial half-mass radius [pc]

    Returns
    -------
    sigma : float [km/s]

    Notes
    -----
    G = 4.302e-3 pc M_sun^-1 (km/s)^2.
    Untuk M=500 M_sun, r_h=0.25 pc: σ ≈ 1.2 km/s, konsisten dengan
    Moyano Loyola & Hurley (2013) μ ≈ 2.1 km/s untuk binary escapers.
    """
    return float(np.sqrt(G_PC_MSUN_KMS2 * M_init_msun / (6.0 * r_h_pc)))


def v_eject_magnitude(t_Myr, tau_Myr, sigma_init_kms, beta=0.5):
    """
    Amplitudo kecepatan kick sebagai fungsi waktu (Kos 2024).

    Mengaproksimasi evolusi kecepatan escape dari Moyano Loyola &
    Hurley (2013, MNRAS 434, 2509):

        v_eject(t) = σ_init × (1 − t/τ)^β

    Pada t=0: kick = σ_init (cluster masif, penuh energi).
    Pada t=τ: kick → 0 (cluster hampir terlarut).
    Penurunan (1-t/τ)^0.5 mencerminkan v_escape = sqrt(M(t)/r_t(t)).

    Parameters
    ----------
    t_Myr         : float — waktu ejeksi [Myr]
    tau_Myr       : float — usia cluster [Myr]
    sigma_init_kms: float — dispersi kecepatan awal [km/s]
    beta          : float — eksponen penurunan temporal (default 0.5)

    Returns
    -------
    v_kick : float [km/s]
    """
    if t_Myr >= tau_Myr:
        return 0.0
    return float(sigma_init_kms * (1.0 - t_Myr / tau_Myr) ** beta)


def sample_kicking(N_stars, tau_Myr, sigma_init_kms,
                   kicked_fraction=0.5, beta=0.5, random_seed=42):
    """
    Pre-assign ejection kicks ke sebagian bintang (Kos 2024, A&A 691, A28).

    DESKRIPSI METODE
    ----------------
    50% bintang dipilih acak, masing-masing mendapat SATU kick pada waktu
    yang disampling dari P_e(t). Mensimulasikan ejeksi dinamis (binary
    hardening, close encounters) yang tidak di-resolve BHTree.

    DISTRIBUSI WAKTU EJEKSI
    -----------------------
    P_e(t) = 2(1 − t/τ) / τ   — menurun linier, lebih banyak ejeksi awal.
    Integral 0→τ = 1 (ternormalisasi).

    Inverse CDF:
        F(t) = 2(t/τ) − (t/τ)^2
        t_eject = τ × (1 − sqrt(1 − U)),  U ~ Uniform(0,1)

    ARAH KICK
    ---------
    Isotropik: φ ~ Uniform(0, 2π), cos(θ) ~ Uniform(−1, 1).

    Parameters
    ----------
    N_stars        : int   — jumlah total bintang
    tau_Myr        : float — usia cluster [Myr]
    sigma_init_kms : float — dispersi kecepatan awal [km/s]
    kicked_fraction: float — fraksi bintang yang mendapat kick (default 0.5)
    beta           : float — eksponen temporal v_eject (default 0.5)
    random_seed    : int   — seed reproducibility

    Returns
    -------
    kick_schedule : dict
        key   : int — index bintang (0-based)
        value : tuple (t_kick_Myr, direction_3d, v_mag_kms)

    Notes
    -----
    Seed dioffset +1 dari seed MASC untuk menghindari korelasi posisi-kick.
    """
    rng = np.random.default_rng(random_seed + 1)

    n_kicked = int(round(kicked_fraction * N_stars))
    n_kicked = max(1, min(n_kicked, N_stars))

    kicked_indices = rng.choice(N_stars, size=n_kicked, replace=False)

    U      = rng.uniform(0.0, 1.0, size=n_kicked)
    t_kick = tau_Myr * (1.0 - np.sqrt(np.clip(1.0 - U, 0.0, 1.0)))

    phi   = rng.uniform(0.0, 2.0 * np.pi, size=n_kicked)
    costh = rng.uniform(-1.0, 1.0, size=n_kicked)
    sinth = np.sqrt(np.clip(1.0 - costh**2, 0.0, 1.0))

    directions = np.column_stack([
        sinth * np.cos(phi),
        sinth * np.sin(phi),
        costh,
    ])

    v_mags = np.array([
        v_eject_magnitude(t, tau_Myr, sigma_init_kms, beta)
        for t in t_kick
    ])

    kick_schedule = {
        int(kicked_indices[i]): (
            float(t_kick[i]),
            directions[i],
            float(v_mags[i]),
        )
        for i in range(n_kicked)
    }

    return kick_schedule


# ============================================================
# CLUSTER SETUP (MASC)
# ============================================================

def setup_cluster_masc(row, random_seed=42, mean_mass_msun=0.4):
    """
    Derive initial conditions untuk satu cluster dan generate stars
    dengan amuse.ext.masc.new_star_cluster.

    Kolom yang dibutuhkan dari backward row
    ----------------------------------------
    Mass_[Msun]  : M_obs (present-day)
    Age_[Myr]    : cluster age
    x0[kpc], y0[kpc], z0[kpc]        : birth position
    vx0[km/s], vy0[km/s], vz0[km/s]  : birth velocity
    FeH_[dex]    : metallicity (opsional, default 0.0)

    Returns
    -------
    stars   : AMUSE Particles
    M_init  : float [M_sun]
    r_h     : float [pc]
    N_init  : int
    params  : dict

    Notes on MASC
    -------------
    stellar_mass TIDAK BOLEH dipass ke new_star_cluster.
    Bug MASC 2023.9.0: UnboundLocalError di generate_single_stars()
    karena number_of_single_stars tidak di-assign dalam if-stellar_mass branch.
    Gunakan number_of_stars=N_init saja.
    """
    M_obs   = float(row['Mass_[Msun]'])
    age_Myr = float(row['Age_[Myr]'])
    x0      = float(row['x0[kpc]'])
    y0      = float(row['y0[kpc]'])
    z0      = float(row['z0[kpc]'])
    vx0     = float(row['vx0[km/s]'])
    vy0     = float(row['vy0[km/s]'])
    vz0     = float(row['vz0[km/s]'])
    feh     = float(row['FeH_[dex]']) if 'FeH_[dex]' in row.index else 0.0

    R_gal  = float(np.sqrt(x0**2 + y0**2))
    M_init = compute_M_init(M_obs, age_Myr, R_gal)
    r_h    = compute_r_h_init(M_init)
    N_init = compute_N_init(M_init, mean_mass_msun)
    Z      = feh_to_Z(feh)

    params = dict(
        name    = row.get('name', str(row.name)),
        M_obs   = M_obs,
        age_Myr = age_Myr,
        R_gal   = R_gal,
        M_init  = M_init,
        r_h     = r_h,
        N_init  = N_init,
        Z       = Z,
        x0=x0, y0=y0, z0=z0,
        vx0=vx0, vy0=vy0, vz0=vz0,
    )

    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  Cluster : {params['name']}")
    print(f"  {sep[2:]}")
    print(f"  OBSERVED (input)")
    print(f"    M_obs   = {M_obs:>10.1f}  M_sun")
    print(f"    Age     = {age_Myr:>10.1f}  Myr")
    print(f"    [Fe/H]  = {feh:>10.3f}  dex  ->  Z = {Z:.4f}")
    print(f"  DERIVED INITIAL CONDITIONS")
    print(f"    R_gal   = {R_gal:>10.3f}  kpc  (birth position)")
    print(f"    M_init  = {M_init:>10.1f}  M_sun  (Lamers+ 2005)")
    print(f"    r_h     = {r_h:>10.4f}  pc   (Marks & Kroupa 2012)")
    print(f"    N_init  = {N_init:>10d}       (Kroupa 2001 IMF)")
    print(f"    M_init/M_obs = {M_init/M_obs:.2f}x")
    print(f"  BIRTH PHASE SPACE")
    print(f"    pos = ({x0:+.4f}, {y0:+.4f}, {z0:+.4f}) kpc")
    print(f"    vel = ({vx0:+.2f}, {vy0:+.2f}, {vz0:+.2f}) km/s")
    print(f"{sep}\n")

    stars = new_star_cluster(
        number_of_stars       = N_init,
        initial_mass_function = "kroupa",
        upper_mass_limit      = 100.0 | units.MSun,
        lower_mass_limit      = 0.08  | units.MSun,
        effective_radius      = r_h   | units.parsec,
        star_distribution     = "plummer",
        star_metallicity      = Z,
        random_seed           = random_seed,
    )

    stars.x  += x0  | units.kpc
    stars.y  += y0  | units.kpc
    stars.z  += z0  | units.kpc
    stars.vx += vx0 | units.km / units.s
    stars.vy += vy0 | units.km / units.s
    stars.vz += vz0 | units.km / units.s

    return stars, M_init, r_h, N_init, params


# ============================================================
# SINGLE CLUSTER N-BODY SIMULATION
# ============================================================

def run_cluster_simulation(row, dt_Myr=0.001, dtout_Myr=0.001,
                           random_seed=42, n_workers=8,
                           opening_angle=0.6,
                           kicked_fraction=0.5, kick_beta=0.5):
    """
    Jalankan simulasi N-body penuh untuk satu cluster.

    ALUR SIMULASI
    -------------
    1. Derive initial conditions (M_init, r_h, N) dari backward row
    2. Generate stars dengan MASC (Kroupa IMF, Plummer profile)
    3. Pre-assign kick schedule: 50% bintang, waktu dari P_e(t),
       amplitudo v(t) = σ×(1-t/τ)^0.5  (Kos 2024)
    4. Setup BHTree + MWPotential2014 bridge
    5. Integration loop:
       a. Evolve bridge maju dt
       b. Apply scheduled kicks yang waktunya sudah tiba;
          push vx,vy,vz kembali ke integrator via to_code
       c. Record CoM, r_h, N_bound setiap dtout

    Parameters
    ----------
    row            : pandas Series — satu baris dari backward DataFrame
    dt_Myr         : float — bridge timestep [Myr]
    dtout_Myr      : float — output cadence [Myr]
    random_seed    : int   — seed untuk MASC dan kick sampling
    n_workers      : int   — BHTree parallel workers
    opening_angle  : float — BHTree Barnes-Hut opening angle theta
    kicked_fraction: float — fraksi bintang yang mendapat kick
    kick_beta      : float — eksponen temporal v_eject(t) = σ×(1-t/τ)^β

    Returns
    -------
    dict:
        params          — dict parameter turunan
        com_track       — array (K,4): [t_Myr, x_kpc, y_kpc, z_kpc]
        rh_track        — array (K,): half-mass radius [pc]
        N_bound_track   — array (K,): jumlah bintang dalam 2*r_h
        eps_pc          — float: softening length [pc]
        sigma_init_kms  — float: dispersi kecepatan awal [km/s]
        n_kicked        — int: jumlah bintang yang mendapat kick
        kick_times_Myr  — array: waktu kick [Myr]
        stars           — AMUSE Particles: snapshot akhir
    """
    # ----------------------------------------------------------------
    # 1. Initial conditions
    # ----------------------------------------------------------------
    stars, M_init, r_h, N_init, params = setup_cluster_masc(
        row, random_seed=random_seed
    )
    N = len(stars)

    # ----------------------------------------------------------------
    # 2. Softening  (Athanassoula et al. 2000)
    # ----------------------------------------------------------------
    eps_pc = compute_softening_eps(r_h, N)
    print(f"  Softening  : eps = {eps_pc:.4f} pc  "
          f"(= {eps_pc/r_h:.3f} x r_h,  r_h = {r_h:.4f} pc,  N = {N})")

    # ----------------------------------------------------------------
    # 3. Kick schedule  (Kos 2024)
    # ----------------------------------------------------------------
    sigma_init = compute_sigma_init(M_init, r_h)
    tau_Myr    = float(params['age_Myr'])

    kick_schedule = sample_kicking(
        N_stars         = N,
        tau_Myr         = tau_Myr,
        sigma_init_kms  = sigma_init,
        kicked_fraction = kicked_fraction,
        beta            = kick_beta,
        random_seed     = random_seed,
    )

    n_kicked       = len(kick_schedule)
    kick_times_arr = np.array([v[0] for v in kick_schedule.values()])
    v_kick_arr     = np.array([v[2] for v in kick_schedule.values()])

    print(f"  Kicking    : σ_init = {sigma_init:.3f} km/s  |  "
          f"n_kicked = {n_kicked}/{N}  ({kicked_fraction*100:.0f}%)")
    print(f"             : v_kick ∈ [{v_kick_arr.min():.3f}, "
          f"{v_kick_arr.max():.3f}] km/s  "
          f"(mean = {v_kick_arr.mean():.3f} km/s)")
    print(f"             : t_kick ∈ [{kick_times_arr.min():.3f}, "
          f"{kick_times_arr.max():.3f}] Myr  "
          f"(P_e(t) ∝ (1-t/τ) distribution)\n")

    applied_kicks = set()

    # ----------------------------------------------------------------
    # 4. BHTree integrator
    # ----------------------------------------------------------------
    converter = nbody_system.nbody_to_si(M_init | units.MSun,
                                         r_h    | units.parsec)

    cluster_code = BHTree(converter, number_of_workers=n_workers)
    cluster_code.parameters.epsilon_squared = (eps_pc | units.parsec) ** 2
    cluster_code.parameters.opening_angle   = opening_angle
    cluster_code.parameters.timestep        = dt_Myr | units.Myr

    cluster_code.particles.add_particles(stars)

    # Channel dua arah:
    # from_code: integrator -> stars (tarik posisi/kecepatan terbaru)
    # to_code  : stars -> integrator (push kick, hanya vx vy vz)
    from_code = cluster_code.particles.new_channel_to(stars)
    to_code   = stars.new_channel_to(cluster_code.particles)

    # ----------------------------------------------------------------
    # 5. Bridge: cluster_code + MWPotential2014
    # ----------------------------------------------------------------
    # MWPotential2014 time-independent, tidak perlu reset_time() antar cluster.
    # Bridge timestep = dt/2 untuk leapfrog stability.
    gravity = bridge.Bridge(use_threading=False)
    gravity.add_system(cluster_code, (mw_amuse,))
    gravity.add_system(mw_amuse)
    gravity.timestep = (dt_Myr / 2.0) | units.Myr

    # ----------------------------------------------------------------
    # 6. Evolution loop
    # ----------------------------------------------------------------
    tend       = tau_Myr   | units.Myr
    dt         = dt_Myr    | units.Myr
    dtout      = dtout_Myr | units.Myr

    time       = 0.0 | units.Myr
    t_next_out = 0.0 | units.Myr

    com_track     = []
    rh_track      = []
    N_bound_track = []

    while time < tend:
        gravity.evolve_model(time + dt)
        from_code.copy()
        time = gravity.model_time

        # --- Apply kicks ---
        t_now             = time.value_in(units.Myr)
        kicks_applied_now = []

        for idx, (t_k, direction, v_mag) in kick_schedule.items():
            if idx in applied_kicks:
                continue
            if t_now >= t_k:
                stars[idx].vx += (direction[0] * v_mag) | units.km / units.s
                stars[idx].vy += (direction[1] * v_mag) | units.km / units.s
                stars[idx].vz += (direction[2] * v_mag) | units.km / units.s
                kicks_applied_now.append(idx)

        if kicks_applied_now:
            applied_kicks.update(kicks_applied_now)
            # Push hanya kecepatan — posisi tidak diubah
            # (menghindari inkonsistensi dengan leapfrog step)
            to_code.copy_attributes(['vx', 'vy', 'vz'])

        # --- Output tracking ---
        if time >= t_next_out:
            com = stars.center_of_mass()

            dx = stars.x - com.x
            dy = stars.y - com.y
            dz = stars.z - com.z
            r  = (dx**2 + dy**2 + dz**2).sqrt()

            rh_now  = np.median(r.value_in(units.parsec))
            N_bound = int(np.sum(r.value_in(units.parsec) <= 2.0 * rh_now))

            com_track.append([
                time.value_in(units.Myr),
                com.x.value_in(units.kpc),
                com.y.value_in(units.kpc),
                com.z.value_in(units.kpc),
            ])
            rh_track.append(rh_now)
            N_bound_track.append(N_bound)

            t_next_out += dtout

    gravity.stop()

    return dict(
        params         = params,
        com_track      = np.array(com_track),
        rh_track       = np.array(rh_track),
        N_bound_track  = np.array(N_bound_track),
        eps_pc         = eps_pc,
        sigma_init_kms = sigma_init,
        n_kicked       = n_kicked,
        kick_times_Myr = kick_times_arr,
        stars          = stars,
    )


# ============================================================
# EXPORT CSV
# ============================================================

def results_to_csv(results, save_path="nbody_stars_all.csv"):
    """
    Gabungkan snapshot bintang akhir semua cluster ke satu CSV.

    Setiap baris = satu bintang. Kolom:
        cluster_name  — nama cluster
        star_idx      — index bintang dalam cluster (0-based)
        mass_msun     — massa bintang [M_sun]
        x_kpc         — posisi X galaktik akhir [kpc]
        y_kpc         — posisi Y galaktik akhir [kpc]
        z_kpc         — posisi Z galaktik akhir [kpc]
        vx_kms        — kecepatan VX akhir [km/s]
        vy_kms        — kecepatan VY akhir [km/s]
        vz_kms        — kecepatan VZ akhir [km/s]
        age_Myr       — usia cluster [Myr]
        M_init_msun   — massa awal cluster [M_sun]
        r_h_pc        — half-mass radius awal [pc]
        R_gal_kpc     — galactocentric radius saat lahir [kpc]
        eps_pc        — softening length [pc]
        sigma_kms     — dispersi kecepatan awal cluster [km/s]
        n_kicked      — jumlah bintang yang mendapat kick
        N_init        — jumlah bintang awal cluster

    Parameters
    ----------
    results   : list of dict — output dari run_cluster_simulation
    save_path : str — path file CSV output

    Returns
    -------
    df : pandas DataFrame
    """
    rows = []

    for res in results:
        if res is None:
            continue

        p     = res['params']
        stars = res['stars']

        x_kpc  = stars.x.value_in(units.kpc)
        y_kpc  = stars.y.value_in(units.kpc)
        z_kpc  = stars.z.value_in(units.kpc)
        vx_kms = stars.vx.value_in(units.km / units.s)
        vy_kms = stars.vy.value_in(units.km / units.s)
        vz_kms = stars.vz.value_in(units.km / units.s)
        mass   = stars.mass.value_in(units.MSun)

        n = len(stars)
        for i in range(n):
            rows.append({
                'cluster_name' : p['name'],
                'star_idx'     : i,
                'mass_msun'    : mass[i],
                'x_kpc'        : x_kpc[i],
                'y_kpc'        : y_kpc[i],
                'z_kpc'        : z_kpc[i],
                'vx_kms'       : vx_kms[i],
                'vy_kms'       : vy_kms[i],
                'vz_kms'       : vz_kms[i],
                'age_Myr'      : p['age_Myr'],
                'M_init_msun'  : p['M_init'],
                'r_h_pc'       : p['r_h'],
                'R_gal_kpc'    : p['R_gal'],
                'eps_pc'       : res['eps_pc'],
                'sigma_kms'    : res['sigma_init_kms'],
                'n_kicked'     : res['n_kicked'],
                'N_init'       : p['N_init'],
            })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, float_format='%.6f')
    print(f"  Saved {len(df)} stars dari {df['cluster_name'].nunique()} "
          f"cluster -> {save_path}")
    return df


# ============================================================
# BATCH SIMULATION
# ============================================================

def run_all_clusters(backward, output_csv="/home/akmal/nbody_stars_all.csv",
                     dt_Myr=0.001, dtout_Myr=0.001,
                     random_seed=42, n_workers=8,
                     opening_angle=0.6,
                     kicked_fraction=0.5, kick_beta=0.5):
    """
    Jalankan simulasi N-body untuk seluruh cluster dalam backward DataFrame.

    Cluster dijalankan secara sekuensial. Jika satu cluster gagal,
    error di-log dan dilanjutkan ke cluster berikutnya.

    Parameters
    ----------
    backward       : pandas DataFrame — output compute_birth_locations()
    output_csv     : str   — path file CSV output
    dt_Myr         : float — bridge timestep [Myr]
    dtout_Myr      : float — output cadence [Myr]
    random_seed    : int   — base seed; tiap cluster mendapat seed+idx
    n_workers      : int   — BHTree parallel workers
    opening_angle  : float — Barnes-Hut opening angle
    kicked_fraction: float — fraksi bintang yang mendapat kick
    kick_beta      : float — eksponen temporal v_eject

    Returns
    -------
    results : list of dict (None untuk cluster yang gagal)
    df_all  : pandas DataFrame — tabel yang disimpan ke CSV
    """
    n       = len(backward)
    results = []

    print(f"\n{'='*60}")
    print(f"  BATCH N-BODY SIMULATION")
    print(f"  Total clusters  : {n}")
    print(f"  dt / dtout      : {dt_Myr} / {dtout_Myr} Myr")
    print(f"  n_workers       : {n_workers}")
    print(f"  kicked_fraction : {kicked_fraction}")
    print(f"  output          : {output_csv}")
    print(f"{'='*60}\n")

    # Cek cluster dengan usia tidak valid sebelum mulai
    bad = backward[backward['Age_[Myr]'] <= 0]
    if len(bad) > 0:
        print(f"  [WARN] {len(bad)} cluster dengan Age_[Myr] <= 0 "
              f"akan dilewati:\n  {list(bad['name'])}\n")

    for idx, (_, row) in enumerate(backward.iterrows()):
        cluster_name = row.get('name', f'cluster_{idx}')

        if row['Age_[Myr]'] <= 0:
            print(f"[{idx+1}/{n}]  {cluster_name}  — SKIP (Age <= 0)")
            results.append(None)
            continue

        print(f"\n[{idx+1}/{n}]  {cluster_name}")

        # Seed unik per cluster: reprodusibel tapi berbeda antar cluster
        cluster_seed = random_seed + idx

        try:
            res = run_cluster_simulation(
                row,
                dt_Myr          = dt_Myr,
                dtout_Myr       = dtout_Myr,
                random_seed     = cluster_seed,
                n_workers       = n_workers,
                opening_angle   = opening_angle,
                kicked_fraction = kicked_fraction,
                kick_beta       = kick_beta,
            )
            results.append(res)
            print(f"  OK  |  Final r_h = {res['rh_track'][-1]:.3f} pc  "
                  f"|  N_bound = {res['N_bound_track'][-1]}")

        except Exception as exc:
            print(f"  [FAIL] {cluster_name}: {exc}")
            results.append(None)

    n_ok   = sum(1 for r in results if r is not None)
    n_fail = n - n_ok

    print(f"\n{'='*60}")
    print(f"  Selesai : {n_ok} berhasil, {n_fail} gagal")
    print(f"{'='*60}\n")

    df_all = results_to_csv(
        [r for r in results if r is not None],
        save_path=output_csv,
    )

    return results, df_all


# ============================================================s
# EKSEKUSI
# ============================================================

backward = pd.read_csv("/home/akmal/data_result_backward.csv")

print(f"Loaded {len(backward)} clusters dari backward CSV.")
print(backward[['name', 'Age_[Myr]', 'Mass_[Msun]',
                'x0[kpc]', 'y0[kpc]', 'z0[kpc]']].to_string(index=False))

results, df_stars = run_all_clusters(
    backward,
    output_csv      = "/home/akmal/nbody_stars_all.csv",
    dt_Myr          = 0.01,
    dtout_Myr       = 0.01,
    random_seed     = 42,
    n_workers       = 8,
    opening_angle   = 0.6,
    kicked_fraction = 0.5,
    kick_beta       = 0.5,
)

print(f"\nTotal bintang   : {len(df_stars)}")
print(f"Total cluster   : {df_stars['cluster_name'].nunique()}")
print(df_stars.head(10).to_string(index=False))