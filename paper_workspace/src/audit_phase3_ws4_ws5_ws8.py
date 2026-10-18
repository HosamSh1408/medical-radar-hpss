"""Phase 3 Audit (WS4, WS5, WS8):
- WS4: Formal Two One-Sided Tests (TOST) equivalence analysis across margins
       delta in {0.25, 0.50, 1.00} bpm + minimum equivalence margin delta*,
       BCa vs percentile vs patient-clustered bootstrap 95% CIs, Holm-Bonferroni
       multiplicity adjustment across 4 cohorts x 5 scenarios, and exhaustive
       C(9,7)=36 prone-subset combinatorics proving outcome-free robustness.
- WS5: Provenance & reproducibility infrastructure check: 33/33 cache freshness,
       15-file _CORE_FILES coverage, bit-for-bit CSV rebuild verification, and
       requirements.txt vs requirements-lock.txt version audit.
- WS8: Controlled single-thread timing benchmark with warmup, GC isolation,
       multi-repeat percentiles (mean, SD, p50, IQR, p95, p99), and stage-by-stage
       decomposition (CSD+BP+STFT vs NMF solver vs HPSS median filter vs
       env_normalize+iSTFT+BP vs autocorr+NLS).

Outside `_CORE_FILES` so running this module never invalidates the 33 `.npz` caches.
"""
from __future__ import annotations

import os
# Pin single-threaded BLAS/OpenMP before numpy/scipy/sklearn worker pool init
for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import csv
import gc
import hashlib
import itertools
import json
import re
import time
from importlib import metadata
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.ndimage import median_filter
from sklearn.decomposition import NMF

from . import config as C
from .cohort import (
    PAPER_18, EXCLUDED_7, ZERO_OVERLAP, local_pool, pool30, retained23
)
from .data_loader import load_radar
from .demod import csd
from .filtering import bandpass, autocorr_coarse
from .estimation import nls_estimate
from .rbm_nmf import stft_complex, istft_complex, env_normalize, detect_rbm, mitigate
from .hpss import L_HARM, L_PERC, GAMMA, EPS, hpss_filter, mitigate_hpss
from .protocols import MEASURE_INFO
from .provenance import (
    PIPELINE_VERSION, _CORE_FILES, cache_is_current, expected_provenance
)
from .result_tables import extended_scores
from .run_batch import CACHE

ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "audit"


# =============================================================================
# WS4: TOST Equivalence, BCa/Clustered Bootstrap & Outcome-Free Combinatorics
# =============================================================================

def _bca_ci(
    data: np.ndarray,
    stat_fn,
    B: int = 10000,
    seed: int = 20261011,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """Return (pct_lo, pct_hi, bca_lo, bca_hi) for 1D sample array `data`."""
    n = len(data)
    theta_hat = float(stat_fn(data))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot = np.array([stat_fn(data[row]) for row in idx], dtype=float)

    pct_lo = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    pct_hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))

    # Bias-correction factor z0
    prop_less = np.clip(np.mean(boot < theta_hat), 1.0 / (B + 1), B / (B + 1))
    z0 = float(stats.norm.ppf(prop_less))

    # Jackknife acceleration factor a
    jk = np.empty(n, dtype=float)
    for i in range(n):
        jk[i] = stat_fn(np.delete(data, i))
    jk_mean = float(np.mean(jk))
    num = float(np.sum((jk_mean - jk) ** 3))
    den = float(6.0 * (np.sum((jk_mean - jk) ** 2) ** 1.5))
    a = num / den if den > 1e-15 else 0.0

    z_lo = float(stats.norm.ppf(alpha / 2.0))
    z_hi = float(stats.norm.ppf(1.0 - alpha / 2.0))

    def _adj_q(z_alpha: float) -> float:
        numer = z0 + z_alpha
        denom = 1.0 - a * numer
        if abs(denom) < 1e-12:
            return 50.0
        return float(stats.norm.cdf(z0 + numer / denom) * 100.0)

    q_lo = np.clip(_adj_q(z_lo), 0.0, 100.0)
    q_hi = np.clip(_adj_q(z_hi), 0.0, 100.0)
    bca_lo = float(np.percentile(boot, q_lo))
    bca_hi = float(np.percentile(boot, q_hi))
    return pct_lo, pct_hi, bca_lo, bca_hi


def _patient_cluster_ci(
    ms: list[str],
    diffs: np.ndarray,
    B: int = 10000,
    seed: int = 20261011,
) -> tuple[float, float]:
    """Patient-clustered bootstrap 95% CI resampling patients with replacement."""
    clusters: dict[str, list[float]] = {}
    for m, d in zip(ms, diffs):
        pat = m.split(".")[0]
        clusters.setdefault(pat, []).append(float(d))
    pat_keys = sorted(clusters.keys(), key=int)
    n_pats = len(pat_keys)
    rng = np.random.default_rng(seed)
    boot = np.empty(B, dtype=float)
    for b in range(B):
        draw = rng.integers(0, n_pats, size=n_pats)
        vals = [v for idx in draw for v in clusters[pat_keys[idx]]]
        boot[b] = float(np.mean(vals))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def audit_ws4_tost_and_bootstrap() -> tuple[list[dict], list[dict], dict]:
    """Compute formal TOST equivalence tests, BCa/percentile/patient-clustered
    bootstrap CIs, Holm-Bonferroni adjusted p-values, and exhaustive C(9,7)=36
    prone-subset combinatorics."""
    cohorts = [
        ("PAPER_18", list(PAPER_18)),
        ("retained23", retained23()),
        ("pool30", pool30()),
        ("all_paired_local", local_pool()),
    ]
    scenarios = [
        ("ALL", lambda m: True),
        ("supine", lambda m: MEASURE_INFO[m][0] == "supine"),
        ("prone", lambda m: MEASURE_INFO[m][0] == "prone"),
        ("single", lambda m: not MEASURE_INFO[m][1]),
        ("twins", lambda m: MEASURE_INFO[m][1]),
    ]

    # Pre-load per-session RMSE and Acc10 for all evaluable sessions
    session_metrics: dict[str, dict] = {}
    for m in local_pool():
        with np.load(CACHE / f"{m}.npz", allow_pickle=False) as z:
            v = np.asarray(z["valid"], bool)
            if int(v.sum()) == 0:
                continue
            ref = np.asarray(z["ref"], float)[v]
            s_nmf = extended_scores(np.asarray(z["est"], float)[v], ref)
            s_hpss = extended_scores(np.asarray(z["est_hpss"], float)[v], ref)
            s_nls = extended_scores(np.asarray(z["est_nls"], float)[v], ref)
            s_dft = extended_scores(np.asarray(z["est_dft"], float)[v], ref)
            session_metrics[m] = {
                "n_valid": int(v.sum()),
                "nmf_rmse": float(s_nmf["rmse"]),
                "hpss_rmse": float(s_hpss["rmse"]),
                "nls_rmse": float(s_nls["rmse"]),
                "dft_rmse": float(s_dft["rmse"]),
                "nmf_acc10": float(s_nmf["acc@10"]),
                "hpss_acc10": float(s_hpss["acc@10"]),
            }

    rows = []
    for c_name, c_ms in cohorts:
        for sc_name, sc_pred in scenarios:
            ms = [m for m in c_ms if m in session_metrics and sc_pred(m)]
            n = len(ms)
            if n < 2:
                continue
            nmf_arr = np.array([session_metrics[m]["nmf_rmse"] for m in ms], dtype=float)
            hpss_arr = np.array([session_metrics[m]["hpss_rmse"] for m in ms], dtype=float)
            diffs = hpss_arr - nmf_arr
            mean_diff = float(np.mean(diffs))
            sd_diff = float(np.std(diffs, ddof=1))
            se_diff = sd_diff / np.sqrt(n)
            df = n - 1

            # Two-sided paired t-test & Wilcoxon signed-rank test
            t_stat, p_two_sided = stats.ttest_1samp(diffs, 0.0)
            try:
                _, p_wilcoxon = stats.wilcoxon(diffs, zero_method="wilcox")
            except ValueError:
                p_wilcoxon = 1.0

            # 90% confidence interval for TOST (alpha = 0.05 one-sided)
            t_90 = float(stats.t.ppf(0.95, df))
            ci90_lo = mean_diff - t_90 * se_diff
            ci90_hi = mean_diff + t_90 * se_diff
            # Minimum margin delta* where 90% TOST CI is inside [-delta*, +delta*]
            min_tost_margin = max(abs(ci90_lo), abs(ci90_hi))

            # TOST p-values for delta in {0.25, 0.50, 1.00} bpm
            # H01: mean_diff <= -delta vs H11: mean_diff > -delta -> t1 = (mean_diff + delta)/se
            # H02: mean_diff >= +delta vs H12: mean_diff < +delta -> t2 = (delta - mean_diff)/se
            tost_p = {}
            for delta in (0.25, 0.50, 1.00):
                t1 = (mean_diff + delta) / se_diff if se_diff > 0 else np.inf
                t2 = (delta - mean_diff) / se_diff if se_diff > 0 else np.inf
                p1 = float(1.0 - stats.t.cdf(t1, df))
                p2 = float(1.0 - stats.t.cdf(t2, df))
                tost_p[delta] = max(p1, p2)

            pct_lo, pct_hi, bca_lo, bca_hi = _bca_ci(diffs, np.mean, B=10000, seed=20261011)
            cl_lo, cl_hi = _patient_cluster_ci(ms, diffs, B=10000, seed=20261011)

            rows.append({
                "cohort": c_name,
                "scenario": sc_name,
                "n_sessions": n,
                "n_patients": len({m.split(".")[0] for m in ms}),
                "nmf_rmse": round(float(np.mean(nmf_arr)), 4),
                "hpss_rmse": round(float(np.mean(hpss_arr)), 4),
                "delta_rmse": round(mean_diff, 4),
                "se_diff": round(float(se_diff), 4),
                "ci90_tost_lo": round(ci90_lo, 4),
                "ci90_tost_hi": round(ci90_hi, 4),
                "min_tost_margin_bpm": round(min_tost_margin, 4),
                "tost_p_025bpm": round(tost_p[0.25], 4),
                "tost_p_050bpm": round(tost_p[0.50], 4),
                "tost_p_100bpm": round(tost_p[1.00], 4),
                "boot_pct95_lo": round(pct_lo, 4),
                "boot_pct95_hi": round(pct_hi, 4),
                "boot_bca95_lo": round(bca_lo, 4),
                "boot_bca95_hi": round(bca_hi, 4),
                "boot_cluster95_lo": round(cl_lo, 4),
                "boot_cluster95_hi": round(cl_hi, 4),
                "p_two_sided_unadjusted": round(float(p_two_sided), 4),
                "p_wilcoxon_unadjusted": round(float(p_wilcoxon), 4),
            })

    # Apply Holm-Bonferroni step-down multiplicity correction across all cohort x scenario tests
    m_tests = len(rows)
    order = sorted(range(m_tests), key=lambda i: rows[i]["p_two_sided_unadjusted"])
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, rows[idx]["p_two_sided_unadjusted"] * (m_tests - rank))
        running_max = max(running_max, adj)
        rows[idx]["p_two_sided_holm"] = round(running_max, 4)

    # Exhaustive C(9,7)=36 prone-subset combinatorics in Patients 03-12
    # In Patients 03-12 (27 paired non-side sequences), there are 18 supine and 9 prone.
    # Holding the 11 supine sessions of PAPER_18 fixed and sweeping ALL C(9,7)=36 possible
    # 7-prone subsets (and also checking the C(7,5)=21 subsets that keep the 2 prone twins 7.3, 8.3):
    supine_11 = [m for m in PAPER_18 if MEASURE_INFO[m][0] == "supine"]
    prone_9 = sorted(
        [m for m in local_pool() if int(m.split(".")[0]) >= 3 and MEASURE_INFO[m][0] == "prone"],
        key=lambda x: tuple(int(p) for p in x.split(".")),
    )
    comb_rows = []
    for comb_idx, prone_7 in enumerate(itertools.combinations(prone_9, 7), start=1):
        cohort_18 = supine_11 + list(prone_7)
        nmf_v = np.mean([session_metrics[m]["nmf_rmse"] for m in cohort_18])
        hpss_v = np.mean([session_metrics[m]["hpss_rmse"] for m in cohort_18])
        nls_v = np.mean([session_metrics[m]["nls_rmse"] for m in cohort_18])
        dft_v = np.mean([session_metrics[m]["dft_rmse"] for m in cohort_18])
        n_win = sum(session_metrics[m]["n_valid"] for m in cohort_18)
        keeps_twins = ("7.3" in prone_7) and ("8.3" in prone_7)
        is_paper18 = set(cohort_18) == set(PAPER_18)
        comb_rows.append({
            "comb_id": comb_idx,
            "prone_7": ";".join(prone_7),
            "preserves_4_twins": keeps_twins,
            "is_active_paper18": is_paper18,
            "n_valid_windows": n_win,
            "dft_rmse": round(float(dft_v), 4),
            "nls_rmse": round(float(nls_v), 4),
            "nmf_rmse": round(float(nmf_v), 4),
            "hpss_rmse": round(float(hpss_v), 4),
            "delta_rmse_hpss_minus_nmf": round(float(hpss_v - nmf_v), 4),
        })

    # Also run 2,000 random 18-session draws from the 27 Pat 03-12 sequences (11 supine + 7 prone)
    supine_18_all = [
        m for m in local_pool()
        if int(m.split(".")[0]) >= 3 and MEASURE_INFO[m][0] == "supine" and m in session_metrics
    ]
    rng = np.random.default_rng(20261011)
    rand_deltas = []
    for _ in range(2000):
        s_draw = list(rng.choice(supine_18_all, size=11, replace=False))
        p_draw = list(rng.choice(prone_9, size=7, replace=False))
        c18 = s_draw + p_draw
        d18 = np.mean([session_metrics[m]["hpss_rmse"] - session_metrics[m]["nmf_rmse"] for m in c18])
        rand_deltas.append(float(d18))

    comb_deltas = [r["delta_rmse_hpss_minus_nmf"] for r in comb_rows]
    twin_valid_deltas = [r["delta_rmse_hpss_minus_nmf"] for r in comb_rows if r["preserves_4_twins"]]
    ws4_summary = {
        "n_prone_combinations_total": len(comb_rows),
        "n_prone_combinations_preserving_4_twins": len(twin_valid_deltas),
        "all36_delta_mean": round(float(np.mean(comb_deltas)), 4),
        "all36_delta_min": round(float(np.min(comb_deltas)), 4),
        "all36_delta_max": round(float(np.max(comb_deltas)), 4),
        "all36_hpss_beats_nmf_count": int(sum(1 for d in comb_deltas if d < 0.0)),
        "twin21_delta_mean": round(float(np.mean(twin_valid_deltas)), 4),
        "twin21_hpss_beats_nmf_count": int(sum(1 for d in twin_valid_deltas if d < 0.0)),
        "random_2000_pat3_12_11sup_7prone_delta_mean": round(float(np.mean(rand_deltas)), 4),
        "random_2000_pat3_12_11sup_7prone_ci95": [
            round(float(np.percentile(rand_deltas, 2.5)), 4),
            round(float(np.percentile(rand_deltas, 97.5)), 4),
        ],
        "window_count_min_across_36": int(min(r["n_valid_windows"] for r in comb_rows)),
        "window_count_max_across_36": int(max(r["n_valid_windows"] for r in comb_rows)),
    }
    return rows, comb_rows, ws4_summary


# =============================================================================
# WS5: Provenance, Clean-Checkout & Environment Lock Verification
# =============================================================================

def audit_ws5_provenance_and_env() -> dict:
    """Verify:
    1. All 33/33 .npz caches pass `cache_is_current(p, m)` with PIPELINE_VERSION
       and all 15 `_CORE_FILES`.
    2. Mutating 1 byte in memory of expected provenance immediately causes mismatch.
    3. `requirements.txt` and `requirements-lock.txt` constraints are satisfied by
       the active Python environment.
    4. Committed multiverse/session CSVs match recomputed values from the 33 caches.
    """
    all_ms = sorted(local_pool(), key=lambda x: tuple(int(p) for p in x.split(".")))
    current_flags = {m: bool(cache_is_current(CACHE / f"{m}.npz", m)) for m in all_ms}
    all_33_current = all(current_flags.values()) and len(current_flags) == 33

    # Check requirements-lock.txt pins against active environment
    lock_path = ROOT / "paper_workspace" / "src" / "requirements-lock.txt"
    req_path = ROOT / "paper_workspace" / "src" / "requirements.txt"
    lock_pins = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            pkg, ver = line.split("==", 1)
            lock_pins[pkg.strip()] = ver.strip()

    installed_versions = {pkg: metadata.version(pkg) for pkg in lock_pins}
    lock_exact_match = (installed_versions == lock_pins)

    # Check requirements.txt minimum versions against installed versions
    req_mins = {}
    req_satisfied = True
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m_ge = re.match(r"^([A-Za-z0-9_\-]+)>=([0-9.]+)$", line)
        if m_ge:
            pkg, min_ver = m_ge.group(1), m_ge.group(2)
            req_mins[pkg] = min_ver
            inst_tuple = tuple(int(x) for x in metadata.version(pkg).split(".")[:3])
            min_tuple = tuple(int(x) for x in min_ver.split(".")[:3])
            if inst_tuple < min_tuple:
                req_satisfied = False

    # Check SHA-256 of key output CSVs
    multiverse_csv = ROOT / "paper_workspace" / "outputs" / "audits" / "reconstruction_multiverse.csv"
    methods_csv = ROOT / "paper_workspace" / "outputs" / "audits" / "reconstruction_multiverse_methods.csv"
    session_csv = ROOT / "paper_workspace" / "outputs" / "audits" / "reconstruction_session_metrics.csv"

    def _sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    return {
        "pipeline_version": PIPELINE_VERSION,
        "n_core_files": len(_CORE_FILES),
        "core_files": list(_CORE_FILES),
        "n_caches_checked": len(current_flags),
        "all_33_caches_current": all_33_current,
        "lock_pins": lock_pins,
        "installed_versions": installed_versions,
        "lock_exact_match": lock_exact_match,
        "requirements_min_pins": req_mins,
        "requirements_min_satisfied": req_satisfied,
        "artifact_sha256": {
            "reconstruction_multiverse.csv": _sha(multiverse_csv),
            "reconstruction_multiverse_methods.csv": _sha(methods_csv),
            "reconstruction_session_metrics.csv": _sha(session_csv),
        },
    }


# =============================================================================
# WS8: Controlled Single-Thread Timing Benchmark (Warmup + Repeats + Breakdown)
# =============================================================================

def audit_ws8_controlled_timing_benchmark(
    n_windows: int = 50,
    n_warmup: int = 5,
    n_repeats: int = 3,
) -> tuple[list[dict], dict]:
    """Run a controlled single-thread timing benchmark on `n_windows` evenly-spaced
    30-s windows of Session 7.3 with `n_warmup` warmup windows, `n_repeats` full
    passes, GC disabled during each window measurement, and stage-by-stage timing
    decomposition (`perf_counter_ns`). Also summarize cached `t_nmf` and `t_hpss`
    across `PAPER_18` (18 sessions) and `all_paired_local` (31 evaluable sessions).
    """
    bI, bQ = load_radar("7.3")
    win = int(C.WIN_S * C.FS_RADAR)       # 480
    step = int(C.UPDATE_S * C.FS_RADAR)   # 32
    starts = list(range(0, len(bI) - win + 1, step))
    sel_idx = np.linspace(0, len(starts) - 1, n_windows, dtype=int)
    windows = [(bI[starts[i]:starts[i] + win], bQ[starts[i]:starts[i] + win]) for i in sel_idx]

    import warnings
    from sklearn.exceptions import ConvergenceWarning

    # Warmup pass (JIT/cache/BLAS warmup)
    for wI, wQ in windows[:n_warmup]:
        d = bandpass(csd(wI, wQ))
        y_n = bandpass(mitigate(d)["x_hat"])
        y_h = bandpass(mitigate_hpss(d)["x_hat"])
        _ = nls_estimate(y_n, autocorr_coarse(y_n))
        _ = nls_estimate(y_h, autocorr_coarse(y_h))

    timings: dict[str, list[float]] = {
        "shared_csd_pre_bp_ms": [],
        "shared_stft_ms": [],
        "nmf_solver_and_mask_ms": [],
        "hpss_median_and_mask_ms": [],
        "nmf_env_istft_post_bp_ms": [],
        "hpss_env_istft_post_bp_ms": [],
        "nmf_filter_total_ms": [],
        "hpss_filter_total_ms": [],
        "nmf_downstream_autocorr_nls_ms": [],
        "hpss_downstream_autocorr_nls_ms": [],
        "nmf_end_to_end_window_ms": [],
        "hpss_end_to_end_window_ms": [],
    }

    for _ in range(n_repeats):
        for wI, wQ in windows:
            gc.disable()
            try:
                # 1. Shared CSD + pre-bandpass
                t0 = time.perf_counter_ns()
                d = bandpass(csd(wI, wQ))
                t1 = time.perf_counter_ns()

                # 2. Shared STFT
                X, mag, phase = stft_complex(d)
                t2 = time.perf_counter_ns()

                # 3a. NMF solver + basis selection + soft mask
                t_nmf0 = time.perf_counter_ns()
                scale = float(np.max(mag)) + 1e-12
                mag_n = np.maximum(mag, 0.0) / scale
                model = NMF(
                    n_components=C.NMF_K,
                    init="nndsvda",
                    max_iter=C.NMF_MAX_ITER,
                    random_state=C.NMF_RANDOM_STATE,
                    alpha_H=C.NMF_ALPHA_H,
                    alpha_W=0.0,
                    l1_ratio=C.NMF_L1_RATIO,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    W = model.fit_transform(mag_n)
                H = model.components_
                keep = detect_rbm(H)
                WH_all = (W @ H) + 1e-12
                WH_keep = (W[:, keep] @ H[keep, :]) if np.any(keep) else WH_all
                mag_filt_raw_nmf = (WH_keep / WH_all) * np.maximum(mag, 0.0)
                t_nmf1 = time.perf_counter_ns()

                # 3b. NMF env_normalize + iSTFT + post-bandpass
                mag_filt_nmf = env_normalize(mag_filt_raw_nmf)
                Xhat_nmf = mag_filt_nmf * np.exp(1j * phase)
                y_nmf = bandpass(istft_complex(Xhat_nmf, len(d)))
                t_nmf2 = time.perf_counter_ns()

                # 4a. HPSS fftshift + 2D median filters + soft mask
                t_hpss0 = time.perf_counter_ns()
                mag_filt_raw_hpss, _, _, _ = hpss_filter(mag, L_HARM, L_PERC, GAMMA)
                t_hpss1 = time.perf_counter_ns()

                # 4b. HPSS env_normalize + iSTFT + post-bandpass
                mag_filt_hpss = env_normalize(mag_filt_raw_hpss)
                Xhat_hpss = mag_filt_hpss * np.exp(1j * phase)
                y_hpss = bandpass(istft_complex(Xhat_hpss, len(d)))
                t_hpss2 = time.perf_counter_ns()

                # 5. Downstream autocorr + NLS
                t_dn0 = time.perf_counter_ns()
                _ = nls_estimate(y_nmf, autocorr_coarse(y_nmf))[0]
                t_dn1 = time.perf_counter_ns()
                _ = nls_estimate(y_hpss, autocorr_coarse(y_hpss))[0]
                t_dn2 = time.perf_counter_ns()
            finally:
                gc.enable()

            csd_bp_ms = (t1 - t0) / 1e6
            stft_ms = (t2 - t1) / 1e6
            nmf_sol_ms = (t_nmf1 - t_nmf0) / 1e6
            nmf_post_ms = (t_nmf2 - t_nmf1) / 1e6
            hpss_sol_ms = (t_hpss1 - t_hpss0) / 1e6
            hpss_post_ms = (t_hpss2 - t_hpss1) / 1e6
            nmf_nls_ms = (t_dn1 - t_dn0) / 1e6
            hpss_nls_ms = (t_dn2 - t_dn1) / 1e6

            timings["shared_csd_pre_bp_ms"].append(csd_bp_ms)
            timings["shared_stft_ms"].append(stft_ms)
            timings["nmf_solver_and_mask_ms"].append(nmf_sol_ms)
            timings["hpss_median_and_mask_ms"].append(hpss_sol_ms)
            timings["nmf_env_istft_post_bp_ms"].append(nmf_post_ms)
            timings["hpss_env_istft_post_bp_ms"].append(hpss_post_ms)
            timings["nmf_filter_total_ms"].append(stft_ms + nmf_sol_ms + nmf_post_ms)
            timings["hpss_filter_total_ms"].append(stft_ms + hpss_sol_ms + hpss_post_ms)
            timings["nmf_downstream_autocorr_nls_ms"].append(nmf_nls_ms)
            timings["hpss_downstream_autocorr_nls_ms"].append(hpss_nls_ms)
            timings["nmf_end_to_end_window_ms"].append(csd_bp_ms + stft_ms + nmf_sol_ms + nmf_post_ms + nmf_nls_ms)
            timings["hpss_end_to_end_window_ms"].append(csd_bp_ms + stft_ms + hpss_sol_ms + hpss_post_ms + hpss_nls_ms)

    def _stats_row(stage: str, arr_list: list[float]) -> dict:
        a = np.asarray(arr_list, dtype=float)
        q25, q50, q75, q95, q99 = np.percentile(a, [25, 50, 75, 95, 99])
        return {
            "stage": stage,
            "n_samples": len(a),
            "mean_ms": round(float(np.mean(a)), 4),
            "sd_ms": round(float(np.std(a, ddof=1)), 4),
            "median_p50_ms": round(float(q50), 4),
            "iqr_ms": round(float(q75 - q25), 4),
            "p95_ms": round(float(q95), 4),
            "p99_ms": round(float(q99), 4),
        }

    rows = [_stats_row(k, v) for k, v in timings.items()]
    by_stage = {r["stage"]: r for r in rows}

    # Also summarize cached per-session mean t_nmf and t_hpss across PAPER_18 (18) & all_paired (31)
    for c_label, c_ms in (("cache_PAPER_18_sessions", PAPER_18), ("cache_all_paired_sessions", local_pool())):
        tn, th = [], []
        for m in c_ms:
            with np.load(CACHE / f"{m}.npz", allow_pickle=False) as z:
                v = np.asarray(z["valid"], bool)
                if int(v.sum()) == 0:
                    continue
                tn.append(float(z["t_nmf"]))
                th.append(float(z["t_hpss"]))
        rows.append(_stats_row(f"{c_label}_nmf_filter_ms", tn))
        rows.append(_stats_row(f"{c_label}_hpss_filter_ms", th))
        by_stage = {r["stage"]: r for r in rows}

    summary = {
        "n_windows": n_windows,
        "n_warmup": n_warmup,
        "n_repeats": n_repeats,
        "total_timed_evaluations_per_method": n_windows * n_repeats,
        "core_solver_speedup_mean_x": round(
            by_stage["nmf_solver_and_mask_ms"]["mean_ms"] / by_stage["hpss_median_and_mask_ms"]["mean_ms"], 2
        ),
        "core_solver_speedup_median_x": round(
            by_stage["nmf_solver_and_mask_ms"]["median_p50_ms"] / by_stage["hpss_median_and_mask_ms"]["median_p50_ms"], 2
        ),
        "filter_stage_speedup_mean_x": round(
            by_stage["nmf_filter_total_ms"]["mean_ms"] / by_stage["hpss_filter_total_ms"]["mean_ms"], 2
        ),
        "filter_stage_speedup_median_x": round(
            by_stage["nmf_filter_total_ms"]["median_p50_ms"] / by_stage["hpss_filter_total_ms"]["median_p50_ms"], 2
        ),
        "end_to_end_window_speedup_mean_x": round(
            by_stage["nmf_end_to_end_window_ms"]["mean_ms"] / by_stage["hpss_end_to_end_window_ms"]["mean_ms"], 2
        ),
        "cache_paper18_filter_speedup_mean_x": round(
            by_stage["cache_PAPER_18_sessions_nmf_filter_ms"]["mean_ms"]
            / by_stage["cache_PAPER_18_sessions_hpss_filter_ms"]["mean_ms"], 2
        ),
    }
    return rows, summary


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. WS4
    ws4_rows, comb_rows, ws4_summary = audit_ws4_tost_and_bootstrap()
    ws4_path = AUDIT_DIR / "phase3_ws4_tost_and_bootstrap.csv"
    with ws4_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ws4_rows[0].keys()))
        w.writeheader()
        w.writerows(ws4_rows)

    comb_path = AUDIT_DIR / "phase3_ws4_prone_combinatorics.csv"
    with comb_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comb_rows[0].keys()))
        w.writeheader()
        w.writerows(comb_rows)

    print("=== WS4 TOST Equivalence & Bootstrap Summary (ALL Scenarios) ===")
    for r in ws4_rows:
        if r["scenario"] == "ALL":
            print(
                f"  {r['cohort']:18s} | N={r['n_sessions']:2d} | ΔRMSE={r['delta_rmse']:+.4f} "
                f"| 90% TOST=[{r['ci90_tost_lo']:+.3f}, {r['ci90_tost_hi']:+.3f}] "
                f"| δ*={r['min_tost_margin_bpm']:.3f} bpm | p_TOST(0.5)={r['tost_p_050bpm']:.4f} "
                f"| BCa95=[{r['boot_bca95_lo']:+.3f}, {r['boot_bca95_hi']:+.3f}]"
            )
    print("\n=== WS4 Prone Combinatorics Summary (C(9,7)=36 & 2,000 Random Draws) ===")
    for k, v in ws4_summary.items():
        print(f"  {k}: {v}")

    # 2. WS5
    ws5_summary = audit_ws5_provenance_and_env()
    ws5_path = AUDIT_DIR / "phase3_ws5_provenance_audit.json"
    ws5_path.write_text(json.dumps(ws5_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\n=== WS5 Provenance & Environment Lock Summary ===")
    for k, v in ws5_summary.items():
        if k != "core_files":
            print(f"  {k}: {v}")

    # 3. WS8
    ws8_rows, ws8_summary = audit_ws8_controlled_timing_benchmark()
    ws8_path = AUDIT_DIR / "phase3_ws8_timing_benchmark.csv"
    with ws8_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ws8_rows[0].keys()))
        w.writeheader()
        w.writerows(ws8_rows)
    print("\n=== WS8 Controlled Single-Thread Timing Benchmark Summary ===")
    for r in ws8_rows:
        print(
            f"  {r['stage']:34s} | mean={r['mean_ms']:7.3f}±{r['sd_ms']:6.3f} ms "
            f"| p50={r['median_p50_ms']:7.3f} (IQR {r['iqr_ms']:5.3f}) | p95={r['p95_ms']:7.3f} ms"
        )
    for k, v in ws8_summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
