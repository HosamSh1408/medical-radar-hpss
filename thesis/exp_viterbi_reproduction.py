"""Baseline-anchored reproduction & rigorous evaluation of score-based HR-BR Viterbi tracking.

Reproduces and audits the attached manuscript on top of the locked Beltrao et al. (2022)
+ HPSS baseline (`main-v3-audit-fix-2`) without mutating any frozen baseline code or cache.

Execution:
  MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /opt/homebrew/bin/python3.10 -m thesis.exp_viterbi_reproduction --workers 4
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import itertools
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from paper_workspace.src import config as C
from paper_workspace.src import evaluation as EV
from paper_workspace.src.data_loader import MEASURE_TO_RADAR, load_radar
from paper_workspace.src.demod import csd
from paper_workspace.src.estimation import dft_estimate, nls_estimate, psd_fft
from paper_workspace.src.filtering import autocorr_coarse, bandpass
from paper_workspace.src.hpss import mitigate_hpss
from paper_workspace.src.protocols import MEASURE_INFO
from paper_workspace.src.provenance import expected_provenance
from paper_workspace.src.rbm_nmf import mitigate
from thesis import edf_extractor as EDFX
from thesis import io_utils as IO
from thesis import viterbi_tracker as VT

SCHEMA_VERSION = "viterbi-repro-v2"
OUTPUT_DIR = IO.ROOT / "thesis" / "outputs" / "viterbi_repro"
OBS_CACHE_DIR = OUTPUT_DIR / "cache"
TRACKER_PATH = IO.ROOT / "thesis" / "viterbi_tracker.py"

BUNDLE_JSON = IO.BUNDLE_DIR / "metrics_viterbi_repro.json"
AUDIT_SUMMARY_CSV = IO.AUDIT_DIR / "viterbi_repro_summary.csv"
AUDIT_SUBGROUPS_CSV = IO.AUDIT_DIR / "viterbi_repro_subgroups.csv"
AUDIT_SESSIONS_CSV = IO.AUDIT_DIR / "viterbi_repro_sessions.csv"
AUDIT_RECON_MD = IO.AUDIT_DIR / "viterbi_repro_reconciliation.md"

COHORT_ORDER = ("pool30_sync", "retained23", "PAPER_18", "all_paired")
REGIME_ORDER = (
    "branch_neutral_raw_nls_lag",
    "pure_metadata_clock_lag",
    "cached_nmf_lag",
)

# Pre-registered narrow LOPO hyperparameter grid for C6a-C6c (16 x 1D + 48 x 2D = 64 configs)
LOPO_SIGMAS = (2.0, 4.0, 6.0, 8.0)
LOPO_EPS_FLOORS = (1e-300, 1e-3, 1e-2, 0.05)
LOPO_W_BR_2D = (0.5, 1.0, 2.0)

# Expanded unpinned LOPO grid for C6d/C6e (36 x 1D + 48 x 2D = 84 configs)
LOPO_WIDE_SIGMAS_1D = (2.0, 4.0, 6.0, 8.0, 12.0, 16.0)
LOPO_WIDE_EPS_1D = (1e-3, 1e-2, 0.05)
LOPO_WIDE_WHQI = (0.0, 4.0)
LOPO_WIDE_SIGMAS_2D = (4.0, 8.0, 12.0, 16.0)
LOPO_WIDE_EPS_2D = (1e-2, 0.05)
LOPO_WIDE_W_BR_2D = (0.5, 2.0, 4.0)

FIXED_METHODS = (
    "A1_DFT_Locked",
    "A2_RawNLS_Locked",
    "A3_Beltrao_NMF_NLS",
    "A4_HPSS_NLS",
    "B1_Clean_WelchPeak_Raw",
    "B2_Clean_WelchPeak_NMF",
    "B3_Clean_WelchPeak_HPSS",
    "B4_HarmNLS_FullGrid_NMF",
    "B5_HarmNLS_FullGrid_HPSS",
    "B5a_Anchored_HarmNLS_Peak_HPSS",
    "B6_Diag_Legacy_AD_NaiveFFT",
    "B7_Diag_Legacy_AD_RawNMF",
    "C1a_1D_Vit_NMF",
    "C1b_1D_Vit_HPSS",
    "C1m_1D_Vit_HPSS_MatchedW05",
    "C2a_1D_HQI_NMF",
    "C2b_1D_HQI_HPSS",
    "C2m_1D_HQI_HPSS_MatchedW05",
    "C3a_2D_JointVit_NMF",
    "C3b_2D_JointVit_HPSS",
    "C3c_2D_JointVit_HPSS_HPSSHR",
    "C3d_2D_JointVit_NMF_ADHR_Diag",
    "C4a_2D_HarmNLS_Vit_NMF",
    "C4b_2D_HarmNLS_Vit_HPSS",
    "C4c_1D_Anchored_HarmNLS_Vit_HPSS",
    "C4d_2D_Anchored_HarmNLS_Vit_HPSS",
    "C5a_2D_JointVit_TransSt_NMF",
    "C5b_2D_JointVit_TransSt_HPSS",
)

LOPO_FAMILIES = (
    ("C6a_LOPO_Cal_Vit_HPSS", "P_br_hpss_2bpm", VT.BR_GRID_2BPM, "A4_HPSS_NLS", "narrow"),
    ("C6b_LOPO_Cal_HarmNLS_HPSS", "S_nls_hpss_1bpm", VT.BR_GRID_1BPM, "A4_HPSS_NLS", "narrow"),
    ("C6c_LOPO_Cal_HarmNLS_NMF", "S_nls_nmf_1bpm", VT.BR_GRID_1BPM, "A3_Beltrao_NMF_NLS", "narrow"),
    ("C6d_LOPO_Wide_AnchoredNLS_HPSS", "S_anc_hpss_1bpm", VT.BR_GRID_1BPM, "A4_HPSS_NLS", "wide"),
    ("C6e_LOPO_Wide_AnchoredNLS_NMF", "S_anc_nmf_1bpm", VT.BR_GRID_1BPM, "A3_Beltrao_NMF_NLS", "wide"),
)

ALL_METHODS = FIXED_METHODS + tuple(f[0] for f in LOPO_FAMILIES)

ATTACHED_DRAFT_REFERENCE = {
    "cohort": "retained23",
    "n_windows_1s_grid": 12397,
    "naive_fft_rmse_bpm": 17.47,
    "naive_fft_acc10_pct": 50.6,
    "no_joint_rmse_bpm": 17.40,
    "no_joint_acc10_pct": 50.7,
    "proposed_rmse_bpm": 9.34,
    "proposed_acc10_pct": 77.7,
    "prone_rmse_bpm": 7.76,
    "prone_acc10_pct": 84.8,
}


def _sharpness_proxy(d_clean: np.ndarray, f_nls_bpm: float, f_coarse_hz: float) -> np.ndarray:
    """Compute bounded [0, 1] candidate weights for {NLS, DFT, Autocorr}."""
    freqs, psd = psd_fft(d_clean, nfft=int(C.NLS_NFFT), fs=float(C.FS_RADAR))
    band = (freqs >= C.BAND_LOW_HZ) & (freqs <= C.BAND_HIGH_HZ)
    s_band = psd[band]
    mean_s = float(np.mean(s_band)) + 1e-12 if s_band.size else 1.0
    f_dft_bpm = float(freqs[band][int(np.argmax(s_band))] * 60.0) if s_band.size else float("nan")
    p_nls_val = float(np.interp(f_nls_bpm / 60.0, freqs, psd, left=0.0, right=0.0)) if np.isfinite(f_nls_bpm) else 0.0
    p_dft_val = float(np.max(s_band)) if s_band.size else 0.0

    pc_nls = float(np.clip((p_nls_val / mean_s - 1.0) / 10.0, 0.0, 1.0)) if np.isfinite(f_nls_bpm) else 0.0
    pc_dft = float(np.clip((p_dft_val / mean_s - 1.0) / 10.0, 0.0, 1.0)) if np.isfinite(f_dft_bpm) else 0.0
    fc_bpm = float(f_coarse_hz * 60.0) if np.isfinite(f_coarse_hz) else float("nan")
    pc_coarse = 0.5 if np.isfinite(fc_bpm) else 0.0
    return np.array(
        [[f_nls_bpm, pc_nls], [f_dft_bpm, pc_dft], [fc_bpm, pc_coarse]],
        dtype=np.float64,
    )


def _obs_cache_is_valid(path: Path, measurement: str, tracker_sha: str) -> bool:
    if not path.exists():
        return False
    try:
        radar_sha = expected_provenance(measurement)["input_sha256"]["radar_csv"]
        with np.load(path, allow_pickle=False) as z:
            prov = json.loads(str(z["provenance_json"].item()))
            return (
                prov.get("schema_version") == SCHEMA_VERSION
                and prov.get("measurement") == measurement
                and prov.get("tracker_sha256") == tracker_sha
                and prov.get("radar_sha256") == radar_sha
            )
    except Exception:
        return False


def _extract_one_session(task: tuple[str, str, bool]) -> tuple[str, str]:
    measurement, tracker_sha, force = task
    OBS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OBS_CACHE_DIR / f"{measurement}.npz"
    if not force and _obs_cache_is_valid(out_path, measurement, tracker_sha):
        return measurement, "cached"

    b_i, b_q = load_radar(measurement)
    w_len = int(C.WIN_SAMPLES)
    step = int(C.UPDATE_S * C.FS_RADAR)
    starts = list(range(0, len(b_i) - w_len + 1, step))
    n_win = len(starts)

    p_br_raw = np.zeros((n_win, VT.BR_GRID_2BPM.size), dtype=np.float64)
    p_br_nmf = np.zeros((n_win, VT.BR_GRID_2BPM.size), dtype=np.float64)
    p_br_hpss = np.zeros((n_win, VT.BR_GRID_2BPM.size), dtype=np.float64)
    p_br_hpss_n480 = np.zeros((n_win, VT.BR_GRID_2BPM.size), dtype=np.float64)
    s_nls_nmf = np.zeros((n_win, VT.BR_GRID_1BPM.size), dtype=np.float64)
    s_nls_hpss = np.zeros((n_win, VT.BR_GRID_1BPM.size), dtype=np.float64)
    s_anc_nmf = np.zeros((n_win, VT.BR_GRID_1BPM.size), dtype=np.float64)
    s_anc_hpss = np.zeros((n_win, VT.BR_GRID_1BPM.size), dtype=np.float64)
    p_hr_csd = np.zeros((n_win, VT.HR_GRID_2BPM.size), dtype=np.float64)
    p_hr_hpss = np.zeros((n_win, VT.HR_GRID_2BPM.size), dtype=np.float64)
    p_hr_dacm = np.zeros((n_win, VT.HR_GRID_2BPM.size), dtype=np.float64)
    p_hr_notched = np.zeros((n_win, VT.HR_GRID_2BPM.size), dtype=np.float64)
    p_hr_csd_n480 = np.zeros((n_win, VT.HR_GRID_2BPM.size), dtype=np.float64)

    aux_nmf = np.full((n_win, 3, 2), np.nan, dtype=np.float64)
    aux_hpss = np.full((n_win, 3, 2), np.nan, dtype=np.float64)

    f_dft_fresh = np.full(n_win, np.nan, dtype=np.float64)
    f_nls_fresh = np.full(n_win, np.nan, dtype=np.float64)
    f_nmf_fresh = np.full(n_win, np.nan, dtype=np.float64)
    f_hpss_fresh = np.full(n_win, np.nan, dtype=np.float64)
    t_nmf_arr = np.zeros(n_win, dtype=np.float64)
    t_hpss_arr = np.zeros(n_win, dtype=np.float64)

    for idx, s0 in enumerate(starts):
        x = csd(b_i[s0 : s0 + w_len], b_q[s0 : s0 + w_len])
        d_raw = bandpass(x)

        t0 = time.perf_counter()
        rbm = mitigate(d_raw)
        t_nmf_arr[idx] = time.perf_counter() - t0
        d_hat = bandpass(rbm["x_hat"])

        t1 = time.perf_counter()
        hpss = mitigate_hpss(d_raw)
        t_hpss_arr[idx] = time.perf_counter() - t1
        d_hpss = bandpass(hpss["x_hat"])

        f_dft_fresh[idx], _, _ = dft_estimate(d_raw)
        f_nls_fresh[idx], _, _ = nls_estimate(d_raw, autocorr_coarse(d_raw))
        fc_n = autocorr_coarse(d_hat)
        f_nmf_fresh[idx], _, _ = nls_estimate(d_hat, fc_n)
        fc_h = autocorr_coarse(d_hpss)
        f_hpss_fresh[idx], _, _ = nls_estimate(d_hpss, fc_h)

        p_br_raw[idx] = VT.extract_br_welch_spectrum(d_raw, VT.BR_GRID_2BPM)
        p_br_nmf[idx] = VT.extract_br_welch_spectrum(d_hat, VT.BR_GRID_2BPM)
        p_br_hpss[idx] = VT.extract_br_welch_spectrum(d_hpss, VT.BR_GRID_2BPM)
        p_br_hpss_n480[idx] = VT.extract_br_welch_spectrum(d_hpss, VT.BR_GRID_2BPM, nperseg=480, noverlap=0)
        s_nls_nmf[idx] = VT.extract_br_harmonic_nls_spectrum(d_hat, VT.BR_GRID_1BPM)
        s_nls_hpss[idx] = VT.extract_br_harmonic_nls_spectrum(d_hpss, VT.BR_GRID_1BPM)
        s_anc_nmf[idx] = VT.extract_br_anchored_nls_spectrum(d_hat, fc_n, VT.BR_GRID_1BPM)
        s_anc_hpss[idx] = VT.extract_br_anchored_nls_spectrum(d_hpss, fc_h, VT.BR_GRID_1BPM)

        p_hr_csd[idx] = VT.extract_hr_csd_velocity_spectrum(x, VT.HR_GRID_2BPM, use_hpss=False)
        p_hr_hpss[idx] = VT.extract_hr_csd_velocity_spectrum(x, VT.HR_GRID_2BPM, use_hpss=True)
        p_hr_dacm[idx] = VT.extract_hr_dacm_velocity_spectrum(x, VT.HR_GRID_2BPM)
        p_hr_notched[idx] = VT.extract_hr_br_notched_velocity_spectrum(x, f_hpss_fresh[idx], VT.HR_GRID_2BPM)
        p_hr_csd_n480[idx] = VT.extract_hr_csd_velocity_spectrum(
            x, VT.HR_GRID_2BPM, use_hpss=False, nperseg=480, noverlap=0
        )

        aux_nmf[idx] = _sharpness_proxy(d_hat, f_nmf_fresh[idx], fc_n)
        aux_hpss[idx] = _sharpness_proxy(d_hpss, f_hpss_fresh[idx], fc_h)

    ad_diag = VT.legacy_ad_session_pipeline(b_i, b_q)
    ad_sens_raw_nndsvd = VT.legacy_ad_sensitivity_pipeline(
        b_i, b_q, detrend_before_nmf=False, bandpass_before_nmf=False, nmf_init="nndsvd"
    )["diag_ad_nmf_sens"]
    ad_sens_detrend_bp_nndsvd = VT.legacy_ad_sensitivity_pipeline(
        b_i, b_q, detrend_before_nmf=True, bandpass_before_nmf=True, nmf_init="nndsvd"
    )["diag_ad_nmf_sens"]
    radar_sha = expected_provenance(measurement)["input_sha256"]["radar_csv"]
    prov_str = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "measurement": measurement,
            "tracker_sha256": tracker_sha,
            "radar_sha256": radar_sha,
        },
        sort_keys=True,
    )

    tmp_path = OBS_CACHE_DIR / f".{measurement}.tmp.npz"
    np.savez_compressed(
        tmp_path,
        P_br_raw_2bpm=p_br_raw,
        P_br_nmf_2bpm=p_br_nmf,
        P_br_hpss_2bpm=p_br_hpss,
        P_br_hpss_n480_2bpm=p_br_hpss_n480,
        S_nls_nmf_1bpm=s_nls_nmf,
        S_nls_hpss_1bpm=s_nls_hpss,
        S_anc_nmf_1bpm=s_anc_nmf,
        S_anc_hpss_1bpm=s_anc_hpss,
        P_hr_csd_2bpm=p_hr_csd,
        P_hr_hpss_2bpm=p_hr_hpss,
        P_hr_dacm_2bpm=p_hr_dacm,
        P_hr_notched_2bpm=p_hr_notched,
        P_hr_csd_n480_2bpm=p_hr_csd_n480,
        aux_cands_nmf=aux_nmf,
        aux_cands_hpss=aux_hpss,
        f_dft_fresh=f_dft_fresh,
        f_nls_fresh=f_nls_fresh,
        f_nmf_fresh=f_nmf_fresh,
        f_hpss_fresh=f_hpss_fresh,
        t_nmf_ms=np.array(float(np.mean(t_nmf_arr) * 1000.0)),
        t_hpss_ms=np.array(float(np.mean(t_hpss_arr) * 1000.0)),
        diag_ad_naive_fft=ad_diag["diag_ad_naive_fft"],
        diag_ad_nmf_nojv=ad_diag["diag_ad_nmf_nojv"],
        diag_ad_nmf_raw_nndsvd=ad_sens_raw_nndsvd,
        diag_ad_nmf_detrend_bp_nndsvd=ad_sens_detrend_bp_nndsvd,
        P_hr_ad_diag_2bpm=ad_diag["P_hr_ad_diag_2bpm"],
        provenance_json=np.array(prov_str),
    )
    tmp_path.replace(out_path)
    return measurement, "computed"


def _load_obs_cache(measurement: str) -> dict[str, np.ndarray]:
    with np.load(OBS_CACHE_DIR / f"{measurement}.npz", allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def _build_aux_prior_seq(aux_cands: np.ndarray, p_hr_seq: np.ndarray, br_grid: np.ndarray) -> np.ndarray:
    n_t = aux_cands.shape[0]
    out = np.zeros((n_t, br_grid.size), dtype=np.float64)
    for t in range(n_t):
        p_hr = p_hr_seq[t]
        hr_rough = float(VT.HR_GRID_2BPM[int(np.argmax(p_hr))]) if float(np.max(p_hr)) > 0.0 else float("nan")
        cands = [(float(aux_cands[t, k, 0]), float(aux_cands[t, k, 1])) for k in range(aux_cands.shape[1])]
        out[t] = VT.compute_transparent_aux_prior(cands, hr_rough, br_grid)
    return out


def _argmax_grid(spec_seq: np.ndarray, grid: np.ndarray) -> np.ndarray:
    maxv = np.max(spec_seq, axis=1)
    idx = np.argmax(spec_seq, axis=1)
    return np.where(maxv > 0.0, grid[idx], grid[0]).astype(np.float64)


def _decode_fixed_session(measurement: str) -> dict[str, Any]:
    """Decode all fixed Group A/B/C methods and precomputed LOPO grid candidates for one session."""
    cache = IO.load_locked_cache(measurement)
    obs = _load_obs_cache(measurement)

    p_br_raw = obs["P_br_raw_2bpm"]
    p_br_nmf = obs["P_br_nmf_2bpm"]
    p_br_hpss = obs["P_br_hpss_2bpm"]
    p_br_hpss_n480 = obs["P_br_hpss_n480_2bpm"]
    s_nls_nmf = obs["S_nls_nmf_1bpm"]
    s_nls_hpss = obs["S_nls_hpss_1bpm"]
    s_anc_nmf = obs["S_anc_nmf_1bpm"]
    s_anc_hpss = obs["S_anc_hpss_1bpm"]
    p_hr_csd = obs["P_hr_csd_2bpm"]
    p_hr_hpss = obs["P_hr_hpss_2bpm"]
    p_hr_ad = obs["P_hr_ad_diag_2bpm"]
    p_hr_dacm = obs["P_hr_dacm_2bpm"]
    p_hr_notched = obs["P_hr_notched_2bpm"]
    p_hr_csd_n480 = obs["P_hr_csd_n480_2bpm"]

    n_t = p_br_hpss.shape[0]
    hqi2_csd = VT.precompute_hqi_sequence(p_hr_csd, VT.HR_GRID_2BPM, VT.BR_GRID_2BPM)
    hqi1_csd = VT.precompute_hqi_sequence(p_hr_csd, VT.HR_GRID_2BPM, VT.BR_GRID_1BPM)
    hqi2_hpss = VT.precompute_hqi_sequence(p_hr_hpss, VT.HR_GRID_2BPM, VT.BR_GRID_2BPM)
    hqi2_ad = VT.precompute_hqi_sequence(p_hr_ad, VT.HR_GRID_2BPM, VT.BR_GRID_2BPM)
    hqi2_csd_n480 = VT.precompute_hqi_sequence(p_hr_csd_n480, VT.HR_GRID_2BPM, VT.BR_GRID_2BPM)

    hqi1d_csd = np.zeros((n_t, VT.BR_GRID_2BPM.size), dtype=np.float64)
    for t in range(n_t):
        if float(np.max(p_hr_csd[t])) > 0.0:
            hqi1d_csd[t] = hqi2_csd[t, int(np.argmax(p_hr_csd[t])), :]

    aux_prior_nmf = _build_aux_prior_seq(obs["aux_cands_nmf"], p_hr_csd, VT.BR_GRID_2BPM)
    aux_prior_hpss = _build_aux_prior_seq(obs["aux_cands_hpss"], p_hr_csd, VT.BR_GRID_2BPM)

    # Fixed methods (Groups A, B, C1-C5 + de-confounded matched & anchored variants)
    est: dict[str, np.ndarray] = {
        "A1_DFT_Locked": np.asarray(cache["est_dft"], dtype=np.float64),
        "A2_RawNLS_Locked": np.asarray(cache["est_nls"], dtype=np.float64),
        "A3_Beltrao_NMF_NLS": np.asarray(cache["est"], dtype=np.float64),
        "A4_HPSS_NLS": np.asarray(cache["est_hpss"], dtype=np.float64),
        "B1_Clean_WelchPeak_Raw": _argmax_grid(p_br_raw, VT.BR_GRID_2BPM),
        "B2_Clean_WelchPeak_NMF": _argmax_grid(p_br_nmf, VT.BR_GRID_2BPM),
        "B3_Clean_WelchPeak_HPSS": _argmax_grid(p_br_hpss, VT.BR_GRID_2BPM),
        "B4_HarmNLS_FullGrid_NMF": _argmax_grid(s_nls_nmf, VT.BR_GRID_1BPM),
        "B5_HarmNLS_FullGrid_HPSS": _argmax_grid(s_nls_hpss, VT.BR_GRID_1BPM),
        "B5a_Anchored_HarmNLS_Peak_HPSS": _argmax_grid(s_anc_hpss, VT.BR_GRID_1BPM),
        "B6_Diag_Legacy_AD_NaiveFFT": np.asarray(obs["diag_ad_naive_fft"], dtype=np.float64),
        "B7_Diag_Legacy_AD_RawNMF": np.asarray(obs["diag_ad_nmf_nojv"], dtype=np.float64),
        "C1a_1D_Vit_NMF": VT.decode_viterbi_1d(p_br_nmf, VT.BR_GRID_2BPM)[0],
        "C1b_1D_Vit_HPSS": VT.decode_viterbi_1d(p_br_hpss, VT.BR_GRID_2BPM)[0],
        "C1m_1D_Vit_HPSS_MatchedW05": VT.decode_viterbi_1d(
            p_br_hpss, VT.BR_GRID_2BPM, sigma_br=4.0, w_br=0.5
        )[0],
        "C2a_1D_HQI_NMF": VT.decode_viterbi_1d(p_br_nmf, VT.BR_GRID_2BPM, hqi_1d_seq=hqi1d_csd)[0],
        "C2b_1D_HQI_HPSS": VT.decode_viterbi_1d(p_br_hpss, VT.BR_GRID_2BPM, hqi_1d_seq=hqi1d_csd)[0],
        "C2m_1D_HQI_HPSS_MatchedW05": VT.decode_viterbi_1d(
            p_br_hpss, VT.BR_GRID_2BPM, sigma_br=4.0, w_br=0.5, hqi_1d_seq=hqi1d_csd, w_hqi=4.0
        )[0],
    }

    c3a_br, _, _ = VT.decode_viterbi_2d_joint(p_br_nmf, p_hr_csd, VT.BR_GRID_2BPM, VT.HR_GRID_2BPM, hqi_matrix_seq=hqi2_csd)
    c3b_br, c3b_hr, _ = VT.decode_viterbi_2d_joint(p_br_hpss, p_hr_csd, VT.BR_GRID_2BPM, VT.HR_GRID_2BPM, hqi_matrix_seq=hqi2_csd)
    c3c_br, _, _ = VT.decode_viterbi_2d_joint(p_br_hpss, p_hr_hpss, VT.BR_GRID_2BPM, VT.HR_GRID_2BPM, hqi_matrix_seq=hqi2_hpss)
    c3d_br, _, _ = VT.decode_viterbi_2d_joint(p_br_nmf, p_hr_ad, VT.BR_GRID_2BPM, VT.HR_GRID_2BPM, hqi_matrix_seq=hqi2_ad)
    c4a_br, _, _ = VT.decode_viterbi_2d_joint(s_nls_nmf, p_hr_csd, VT.BR_GRID_1BPM, VT.HR_GRID_2BPM, hqi_matrix_seq=hqi1_csd)
    c4b_br, c4b_hr, _ = VT.decode_viterbi_2d_joint(s_nls_hpss, p_hr_csd, VT.BR_GRID_1BPM, VT.HR_GRID_2BPM, hqi_matrix_seq=hqi1_csd)
    c4c_br = VT.decode_viterbi_1d(s_anc_hpss, VT.BR_GRID_1BPM, sigma_br=4.0, w_br=1.0)[0]
    c4d_br, c4d_hr, _ = VT.decode_viterbi_2d_joint(s_anc_hpss, p_hr_csd, VT.BR_GRID_1BPM, VT.HR_GRID_2BPM, hqi_matrix_seq=hqi1_csd)
    c5a_br, _, _ = VT.decode_viterbi_2d_joint(
        p_br_nmf, p_hr_csd, VT.BR_GRID_2BPM, VT.HR_GRID_2BPM, aux_prior_seq=aux_prior_nmf, hqi_matrix_seq=hqi2_csd
    )
    c5b_br, _, _ = VT.decode_viterbi_2d_joint(
        p_br_hpss, p_hr_csd, VT.BR_GRID_2BPM, VT.HR_GRID_2BPM, aux_prior_seq=aux_prior_hpss, hqi_matrix_seq=hqi2_csd
    )

    est.update({
        "C3a_2D_JointVit_NMF": c3a_br,
        "C3b_2D_JointVit_HPSS": c3b_br,
        "C3c_2D_JointVit_HPSS_HPSSHR": c3c_br,
        "C3d_2D_JointVit_NMF_ADHR_Diag": c3d_br,
        "C4a_2D_HarmNLS_Vit_NMF": c4a_br,
        "C4b_2D_HarmNLS_Vit_HPSS": c4b_br,
        "C4c_1D_Anchored_HarmNLS_Vit_HPSS": c4c_br,
        "C4d_2D_Anchored_HarmNLS_Vit_HPSS": c4d_br,
        "C5a_2D_JointVit_TransSt_NMF": c5a_br,
        "C5b_2D_JointVit_TransSt_HPSS": c5b_br,
    })

    # Precompute LOPO grid candidates for C6a-C6c (narrow) and C6d-C6e (wide) once per session
    hqi1d_1bpm = np.zeros((n_t, VT.BR_GRID_1BPM.size), dtype=np.float64)
    for t in range(n_t):
        if float(np.max(p_hr_csd[t])) > 0.0:
            hqi1d_1bpm[t] = hqi1_csd[t, int(np.argmax(p_hr_csd[t])), :]

    spec_map = {
        "P_br_hpss_2bpm": (p_br_hpss, VT.BR_GRID_2BPM, hqi2_csd, hqi1d_csd),
        "S_nls_hpss_1bpm": (s_nls_hpss, VT.BR_GRID_1BPM, hqi1_csd, hqi1d_1bpm),
        "S_nls_nmf_1bpm": (s_nls_nmf, VT.BR_GRID_1BPM, hqi1_csd, hqi1d_1bpm),
        "S_anc_hpss_1bpm": (s_anc_hpss, VT.BR_GRID_1BPM, hqi1_csd, hqi1d_1bpm),
        "S_anc_nmf_1bpm": (s_anc_nmf, VT.BR_GRID_1BPM, hqi1_csd, hqi1d_1bpm),
    }
    lopo_grid_paths: dict[str, dict[tuple, np.ndarray]] = {}
    for fam_name, spec_key, grid, _, grid_mode in LOPO_FAMILIES:
        p_seq, _, hqi2, hqi1d = spec_map[spec_key]
        fam_paths: dict[tuple, np.ndarray] = {}
        if grid_mode == "narrow":
            for sigma_br, eps_p in itertools.product(LOPO_SIGMAS, LOPO_EPS_FLOORS):
                fam_paths[("1d", sigma_br, eps_p, 1.0)] = VT.decode_viterbi_1d(
                    p_seq, grid, sigma_br=sigma_br, w_br=1.0, eps_p=eps_p, hqi_1d_seq=hqi1d
                )[0]
                for w_br in LOPO_W_BR_2D:
                    fam_paths[("2d", sigma_br, eps_p, w_br)] = VT.decode_viterbi_2d_joint(
                        p_seq,
                        p_hr_csd,
                        grid,
                        VT.HR_GRID_2BPM,
                        w_br=w_br,
                        w_hqi=4.0,
                        sigma_br=sigma_br,
                        sigma_hr=12.0,
                        eps_p=eps_p,
                        hqi_matrix_seq=hqi2,
                    )[0]
        else:
            for sigma_br, eps_p, w_hqi in itertools.product(
                LOPO_WIDE_SIGMAS_1D, LOPO_WIDE_EPS_1D, LOPO_WIDE_WHQI
            ):
                fam_paths[("1d", sigma_br, eps_p, 1.0, w_hqi)] = VT.decode_viterbi_1d(
                    p_seq,
                    grid,
                    sigma_br=sigma_br,
                    w_br=1.0,
                    eps_p=eps_p,
                    hqi_1d_seq=(hqi1d if w_hqi > 0.0 else None),
                    w_hqi=w_hqi,
                )[0]
            for sigma_br, eps_p, w_br, w_hqi in itertools.product(
                LOPO_WIDE_SIGMAS_2D, LOPO_WIDE_EPS_2D, LOPO_WIDE_W_BR_2D, LOPO_WIDE_WHQI
            ):
                use_hqi = w_hqi > 0.0
                fam_paths[("2d", sigma_br, eps_p, w_br, w_hqi)] = VT.decode_viterbi_2d_joint(
                    p_seq,
                    p_hr_csd,
                    grid,
                    VT.HR_GRID_2BPM,
                    w_br=w_br,
                    w_hqi=w_hqi,
                    sigma_br=sigma_br,
                    sigma_hr=12.0,
                    eps_p=eps_p,
                    use_hqi=use_hqi,
                    hqi_matrix_seq=(hqi2 if use_hqi else None),
                )[0]
        lopo_grid_paths[fam_name] = fam_paths

    # Stress-test decoded sequences (Welch n480 + detrended Legacy-AD sensitivity)
    c3b_br_n480, _, _ = VT.decode_viterbi_2d_joint(
        p_br_hpss_n480, p_hr_csd, VT.BR_GRID_2BPM, VT.HR_GRID_2BPM, hqi_matrix_seq=hqi2_csd
    )
    c3b_both_n480, _, _ = VT.decode_viterbi_2d_joint(
        p_br_hpss_n480, p_hr_csd_n480, VT.BR_GRID_2BPM, VT.HR_GRID_2BPM, hqi_matrix_seq=hqi2_csd_n480
    )
    stress_est = {
        "B3_Clean_WelchPeak_HPSS_n480": _argmax_grid(p_br_hpss_n480, VT.BR_GRID_2BPM),
        "C3b_2D_JointVit_HPSS_br_n480": c3b_br_n480,
        "C3b_2D_JointVit_HPSS_both_n480": c3b_both_n480,
        "B7_sens_raw_nndsvd": np.asarray(obs["diag_ad_nmf_raw_nndsvd"], dtype=np.float64),
        "B7_sens_detrend_bp_nndsvd": np.asarray(obs["diag_ad_nmf_detrend_bp_nndsvd"], dtype=np.float64),
    }

    parity = {
        "dft_max": float(np.max(np.abs(obs["f_dft_fresh"] - cache["est_dft"]))),
        "nls_max": float(np.max(np.abs(obs["f_nls_fresh"] - cache["est_nls"]))),
        "nmf_max": float(np.max(np.abs(obs["f_nmf_fresh"] - cache["est"]))),
        "hpss_max": float(np.max(np.abs(obs["f_hpss_fresh"] - cache["est_hpss"]))),
        "t_nmf_ms": float(obs["t_nmf_ms"]),
        "t_hpss_ms": float(obs["t_hpss_ms"]),
    }

    return {
        "measurement": measurement,
        "est": est,
        "stress_est": stress_est,
        "hr_paths": {
            "hr_peak_csd_vel": _argmax_grid(p_hr_csd, VT.HR_GRID_2BPM),
            "hr_peak_hpss_vel": _argmax_grid(p_hr_hpss, VT.HR_GRID_2BPM),
            "hr_peak_dacm_vel": _argmax_grid(p_hr_dacm, VT.HR_GRID_2BPM),
            "hr_peak_br_notched_vel": _argmax_grid(p_hr_notched, VT.HR_GRID_2BPM),
            "hr_peak_csd_vel_n480": _argmax_grid(p_hr_csd_n480, VT.HR_GRID_2BPM),
            "hr_peak_ad_diag": _argmax_grid(p_hr_ad, VT.HR_GRID_2BPM),
            "hr_vit_c3b": c3b_hr,
            "hr_vit_c4b": c4b_hr,
            "hr_vit_c4d": c4d_hr,
        },
        "lopo_grid_paths": lopo_grid_paths,
        "parity": parity,
    }


def _session_metrics(est: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    e = np.asarray(est, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    ae = np.abs(e)
    r_val = float(np.corrcoef(est, ref)[0, 1]) if (np.std(est) > 0.0 and np.std(ref) > 0.0) else float("nan")
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(ae)),
        "bias": float(np.mean(e)),
        "acc3": float(100.0 * np.mean(ae < 3.0)),
        "acc5": float(100.0 * np.mean(ae < 5.0)),
        "acc6": float(100.0 * np.mean(ae < 6.0)),
        "acc10": float(100.0 * np.mean(ae < 10.0)),
        "acc20": float(100.0 * np.mean(ae < 20.0)),
        "gross_gt10_pct": float(100.0 * np.mean(ae > 10.0)),
        "r": r_val,
        "est_mean": float(np.mean(est)),
        "ref_mean": float(np.mean(ref)),
        "ref_sd": float(np.std(ref)),
    }


def _session_level_bland_altman_and_icc(est_means: np.ndarray, ref_means: np.ndarray) -> dict[str, float]:
    n = len(est_means)
    if n < 2:
        return {k: float("nan") for k in ("bias", "sd", "loa_lo", "loa_up", "prop_slope", "prop_p", "icc_1_1")}
    diff = est_means - ref_means
    avg = 0.5 * (est_means + ref_means)
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    reg = stats.linregress(avg, diff) if np.std(avg) > 0.0 else None

    # One-way random-effects ICC(1,1) across k=2 methods per session
    grand = float(np.mean(np.concatenate([est_means, ref_means])))
    row_means = avg
    bms = 2.0 * float(np.sum((row_means - grand) ** 2)) / float(n - 1)
    wms = float(np.sum((est_means - row_means) ** 2 + (ref_means - row_means) ** 2)) / float(n)
    icc = float((bms - wms) / (bms + wms)) if (bms + wms) > 0.0 else float("nan")
    return {
        "bias": bias,
        "sd": sd,
        "loa_lo": bias - 1.96 * sd,
        "loa_up": bias + 1.96 * sd,
        "prop_slope": float(reg.slope) if reg is not None else float("nan"),
        "prop_p": float(reg.pvalue) if reg is not None else float("nan"),
        "icc_1_1": icc,
    }


def _is_upper_pinned(sel_cfg: tuple, grid_mode: str) -> bool:
    mode, sigma_br, _, w_br = sel_cfg[:4]
    if grid_mode == "narrow":
        return float(sigma_br) == float(max(LOPO_SIGMAS)) or (
            mode == "2d" and float(w_br) == float(max(LOPO_W_BR_2D))
        )
    max_sig = float(max(LOPO_WIDE_SIGMAS_2D if mode == "2d" else LOPO_WIDE_SIGMAS_1D))
    return float(sigma_br) == max_sig or (mode == "2d" and float(w_br) == float(max(LOPO_WIDE_W_BR_2D)))


def _is_any_boundary_pinned(sel_cfg: tuple, grid_mode: str) -> bool:
    mode, sigma_br, eps_p, w_br = sel_cfg[:4]
    if grid_mode == "narrow":
        sigmas, epss, wbrs = LOPO_SIGMAS, LOPO_EPS_FLOORS, LOPO_W_BR_2D
    else:
        sigmas = LOPO_WIDE_SIGMAS_2D if mode == "2d" else LOPO_WIDE_SIGMAS_1D
        epss = LOPO_WIDE_EPS_2D if mode == "2d" else LOPO_WIDE_EPS_1D
        wbrs = LOPO_WIDE_W_BR_2D
    pinned = (
        float(sigma_br) in (float(min(sigmas)), float(max(sigmas)))
        or float(eps_p) in (float(min(epss)), float(max(epss)))
    )
    if mode == "2d":
        pinned = pinned or float(w_br) in (float(min(wbrs)), float(max(wbrs)))
    return bool(pinned)


def _build_lopo_predictions(
    cohort: str,
    regime: str,
    decoded: dict[str, dict[str, Any]],
    ref_valid_map: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[dict]]]:
    """Run strictly patient-disjoint LOPO selection for C6a-C6e."""
    sessions = [m for m in IO.get_cohort_sessions(cohort, include_zero_valid=False) if np.any(ref_valid_map[(m, regime)][1])]
    folds = list(IO.iter_lopo_folds(cohort))

    out_preds: dict[str, dict[str, np.ndarray]] = {f[0]: {} for f in LOPO_FAMILIES}
    fold_logs: dict[str, list[dict]] = {f[0]: [] for f in LOPO_FAMILIES}

    for fam_name, _, _, anchor_name, grid_mode in LOPO_FAMILIES:
        grid_keys = list(next(iter(decoded.values()))["lopo_grid_paths"][fam_name].keys())
        for fold in folds:
            train_ms = [m for m in fold.train_sessions if m in sessions]
            test_ms = [m for m in fold.test_sessions if m in sessions]
            if not test_ms:
                continue

            # Anchor within-session rmcorr on training patients only
            base_est_list = []
            ref_list = []
            for m in train_ms:
                ref_arr, v_arr = ref_valid_map[(m, regime)]
                base_est_list.append(decoded[m]["est"][anchor_name][v_arr])
                ref_list.append(ref_arr[v_arr])
            base_rmc = IO.rmcorr_ancova(base_est_list, ref_list)["r_within"]
            min_r = 0.95 * base_rmc if np.isfinite(base_rmc) else -np.inf

            best_elig = None
            best_fallback = None
            for cfg in grid_keys:
                tr_ests = []
                tr_rmses = []
                for m, rf in zip(train_ms, ref_list):
                    _, v_arr = ref_valid_map[(m, regime)]
                    pe = decoded[m]["lopo_grid_paths"][fam_name][cfg][v_arr]
                    tr_ests.append(pe)
                    tr_rmses.append(float(np.sqrt(np.mean((pe - rf) ** 2))))
                mean_rmse = float(np.mean(tr_rmses))
                rmc_val = IO.rmcorr_ancova(tr_ests, ref_list)["r_within"]
                cand_fb = (-(rmc_val if np.isfinite(rmc_val) else -1.0), mean_rmse, cfg, rmc_val)
                if best_fallback is None or cand_fb < best_fallback:
                    best_fallback = cand_fb
                if np.isfinite(rmc_val) and rmc_val >= min_r:
                    cand_el = (mean_rmse, -rmc_val, cfg, rmc_val)
                    if best_elig is None or cand_el < best_elig:
                        best_elig = cand_el

            fallback_used = best_elig is None
            if not fallback_used:
                train_rmse_sel, _, sel_cfg, train_r_sel = best_elig
            else:
                _, train_rmse_sel, sel_cfg, train_r_sel = best_fallback

            for m in test_ms:
                out_preds[fam_name][m] = decoded[m]["lopo_grid_paths"][fam_name][sel_cfg]

            w_hqi_val = float(sel_cfg[4]) if len(sel_cfg) >= 5 else 4.0
            fold_logs[fam_name].append({
                "patient_id": int(fold.patient_id),
                "test_sessions": list(test_ms),
                "grid_mode": grid_mode,
                "selected_mode": sel_cfg[0],
                "selected_sigma_br": float(sel_cfg[1]),
                "selected_eps_floor": float(sel_cfg[2]),
                "selected_w_br": float(sel_cfg[3]),
                "selected_w_hqi": w_hqi_val,
                "upper_pinned": _is_upper_pinned(sel_cfg, grid_mode),
                "any_boundary_pinned": _is_any_boundary_pinned(sel_cfg, grid_mode),
                "train_rmse_sess": float(train_rmse_sel),
                "train_r_within": float(train_r_sel),
                "train_anchor_r_within": float(base_rmc),
                "fallback_used": bool(fallback_used),
            })

    return out_preds, fold_logs


def _aggregate_method_metrics(
    sessions: list[str],
    est_by_sess: dict[str, np.ndarray],
    ref_valid_map: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    regime: str,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Compute equal-session, pooled, tracking, and Bland-Altman metrics."""
    per_sess: dict[str, dict[str, float]] = {}
    est_valid_list, ref_valid_list = [], []
    est_means, ref_means = [], []

    for m in sessions:
        ref_arr, v_arr = ref_valid_map[(m, regime)]
        if not np.any(v_arr):
            continue
        ev = np.asarray(est_by_sess[m][v_arr], dtype=np.float64)
        rv = np.asarray(ref_arr[v_arr], dtype=np.float64)
        ok = np.isfinite(ev) & np.isfinite(rv)
        ev, rv = ev[ok], rv[ok]
        if ev.size == 0:
            continue
        sc = _session_metrics(ev, rv)
        per_sess[m] = sc
        est_valid_list.append(ev)
        ref_valid_list.append(rv)
        est_means.append(sc["est_mean"])
        ref_means.append(sc["ref_mean"])

    if not per_sess:
        return {"n_sessions": 0, "n_windows": 0}, per_sess

    all_e = np.concatenate(est_valid_list)
    all_r = np.concatenate(ref_valid_list)
    pool_sc = _session_metrics(all_e, all_r)
    rmc = IO.rmcorr_ancova(est_valid_list, ref_valid_list)
    ba_sess = _session_level_bland_altman_and_icc(np.array(est_means), np.array(ref_means))
    ba_clust = IO.clustered_bland_altman(est_valid_list, ref_valid_list)

    r_vals = [sc["r"] for sc in per_sess.values() if np.isfinite(sc["r"])]
    agg = {
        "n_sessions": len(per_sess),
        "n_windows": int(all_r.size),
        "rmse_sess": float(np.mean([sc["rmse"] for sc in per_sess.values()])),
        "rmse_sess_median": float(np.median([sc["rmse"] for sc in per_sess.values()])),
        "mae_sess": float(np.mean([sc["mae"] for sc in per_sess.values()])),
        "acc3_sess": float(np.mean([sc["acc3"] for sc in per_sess.values()])),
        "acc5_sess": float(np.mean([sc["acc5"] for sc in per_sess.values()])),
        "acc6_sess": float(np.mean([sc["acc6"] for sc in per_sess.values()])),
        "acc10_sess": float(np.mean([sc["acc10"] for sc in per_sess.values()])),
        "acc20_sess": float(np.mean([sc["acc20"] for sc in per_sess.values()])),
        "gross_gt10_sess_pct": float(np.mean([sc["gross_gt10_pct"] for sc in per_sess.values()])),
        "rmse_pool": pool_sc["rmse"],
        "mae_pool": pool_sc["mae"],
        "acc5_pool": pool_sc["acc5"],
        "acc10_pool": pool_sc["acc10"],
        "acc20_pool": pool_sc["acc20"],
        "gross_gt10_pool_pct": pool_sc["gross_gt10_pct"],
        "r_within_rmcorr": rmc["r_within"],
        "r_within_rmcorr_df": rmc["df"],
        "r_within_rmcorr_p": rmc["p_value"],
        "r_within_sess_mean": float(np.mean(r_vals)) if r_vals else float("nan"),
        "sessions_r_lt_0_3": int(np.sum(np.array(r_vals) < 0.3)) if r_vals else 0,
        "ba_session_level": ba_sess,
        "ba_clustered_anova": ba_clust,
    }
    return agg, per_sess


def _safe_wilcoxon_p(deltas: np.ndarray) -> float:
    arr = np.asarray(deltas, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2 or np.allclose(arr, 0.0, atol=1e-15):
        return 1.0
    try:
        p = float(stats.wilcoxon(arr).pvalue)
        return p if np.isfinite(p) else 1.0
    except ValueError:
        return 1.0


def _constant_baselines(
    sessions: list[str],
    ref_valid_map: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    regime: str,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    ref_list = []
    sess_ids = []
    pids = []
    for m in sessions:
        ref_arr, v_arr = ref_valid_map[(m, regime)]
        if np.any(v_arr):
            ref_list.append(ref_arr[v_arr])
            sess_ids.append(m)
            pids.append(IO.patient_id_of(m))
    if not ref_list:
        return {
            "const_loso": float("nan"),
            "const_lopo_median": float("nan"),
            "const_lopo_mean": float("nan"),
            "const_global_oracle": float("nan"),
            "session_mean_oracle": float("nan"),
        }, {}

    pooled = np.concatenate(ref_list)
    gmean = float(np.mean(pooled))
    c_loso, c_lopo_med, c_lopo_mean, c_glob, s_sd = [], [], [], [], []
    per_sess_const: dict[str, dict[str, float]] = {}
    for idx, (m, pid, rf) in enumerate(zip(sess_ids, pids, ref_list)):
        others = np.concatenate([ref_list[j] for j in range(len(ref_list)) if j != idx]) if len(ref_list) > 1 else rf
        med_other = float(np.median(others))
        r_loso = float(np.sqrt(np.mean((rf - med_other) ** 2)))

        other_pat = [ref_list[j] for j in range(len(ref_list)) if pids[j] != pid]
        pat_pool = np.concatenate(other_pat) if other_pat else rf
        r_lopo_med = float(np.sqrt(np.mean((rf - float(np.median(pat_pool))) ** 2)))
        r_lopo_mean = float(np.sqrt(np.mean((rf - float(np.mean(pat_pool))) ** 2)))

        r_glob = float(np.sqrt(np.mean((rf - gmean) ** 2)))
        sd_val = float(np.std(rf))
        c_loso.append(r_loso)
        c_lopo_med.append(r_lopo_med)
        c_lopo_mean.append(r_lopo_mean)
        c_glob.append(r_glob)
        s_sd.append(sd_val)
        per_sess_const[m] = {
            "const_loso_rmse": r_loso,
            "const_lopo_median_rmse": r_lopo_med,
            "const_lopo_mean_rmse": r_lopo_mean,
            "const_global_rmse": r_glob,
            "oracle_sd": sd_val,
        }
    return {
        "const_loso": float(np.mean(c_loso)),
        "const_lopo_median": float(np.mean(c_lopo_med)),
        "const_lopo_mean": float(np.mean(c_lopo_mean)),
        "const_global_oracle": float(np.mean(c_glob)),
        "session_mean_oracle": float(np.mean(s_sd)),
    }, per_sess_const


def _paired_tests(
    sessions: list[str],
    per_sess_method: dict[str, dict[str, float]],
    per_sess_anchor: dict[str, dict[str, float]],
    anchor_rmse_key: str = "rmse",
) -> dict[str, Any]:
    common = [m for m in sessions if m in per_sess_method and m in per_sess_anchor]
    if len(common) < 2:
        return {
            "mean_delta": float("nan"),
            "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
            "p_wilcoxon": float("nan"),
            "p_wilcoxon_patient": float("nan"),
        }
    deltas = np.array([per_sess_method[m]["rmse"] - per_sess_anchor[m][anchor_rmse_key] for m in common], dtype=np.float64)
    by_pat: dict[int, list[float]] = {}
    for m, d in zip(common, deltas):
        by_pat.setdefault(IO.patient_id_of(m), []).append(float(d))
    pat_map = {p: np.asarray(vals, dtype=np.float64) for p, vals in by_pat.items()}
    boot = IO.patient_clustered_bootstrap(pat_map, n_boot=10000, seed=IO.RANDOM_SEED)
    p_wil_sess = _safe_wilcoxon_p(deltas)
    pat_deltas = np.array([float(np.mean(vals)) for vals in pat_map.values()], dtype=np.float64)
    p_wil_pat = _safe_wilcoxon_p(pat_deltas)
    return {
        "n_sessions": len(common),
        "n_patients": len(pat_map),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "wins": f"{int(np.sum(deltas < 0.0))}/{len(common)}",
        "patient_wins": f"{int(np.sum(pat_deltas < 0.0))}/{len(pat_deltas)}",
        "ci95_lo": boot["ci95_lo"],
        "ci95_hi": boot["ci95_hi"],
        "p_wilcoxon": p_wil_sess,
        "p_wilcoxon_patient": p_wil_pat,
    }


def _evaluate_hr_branch(
    decoded: dict[str, dict[str, Any]],
    edf_map: dict[str, EDFX.EdfRecording],
    lags_map: dict[str, dict[str, float | int | None]],
) -> dict[str, Any]:
    """Validate all 8 radar HR peak & 2D Viterbi HR trajectories against EDF 1024 Hz R-Trigger."""
    sessions = IO.get_cohort_sessions("pool30_sync", include_zero_valid=False)
    hr_keys = (
        "hr_peak_csd_vel",
        "hr_peak_hpss_vel",
        "hr_peak_dacm_vel",
        "hr_peak_br_notched_vel",
        "hr_peak_csd_vel_n480",
        "hr_peak_ad_diag",
        "hr_vit_c3b",
        "hr_vit_c4b",
        "hr_vit_c4d",
    )
    per_key_sess: dict[str, list[dict[str, float]]] = {k: [] for k in hr_keys}
    per_key_est: dict[str, list[np.ndarray]] = {k: [] for k in hr_keys}
    per_key_ref: dict[str, list[np.ndarray]] = {k: [] for k in hr_keys}

    for m in sessions:
        lag_val = int(lags_map[m]["neutral_raw_nls_lag_s"])
        n_win = len(decoded[m]["est"]["A4_HPSS_NLS"])
        centers = np.arange(n_win, dtype=float) * C.UPDATE_S + C.WIN_S / 2.0
        ext = EDFX.extract_window_references(edf_map[m], centers, lag_s=lag_val)
        ok = (
            ext["valid_vwin"]
            & np.isfinite(ext["hr_rtrig_bpm"])
            & np.isfinite(ext["hr_1hz_30s"])
            & (ext["signaliq_mean"] >= 50.0)
            & (ext["hr_1hz_30s"] > 50.0)
        )
        if int(np.sum(ok)) < 5:
            continue
        hr_ref = ext["hr_rtrig_bpm"][ok]
        for k in hr_keys:
            hr_est = decoded[m]["hr_paths"][k][ok]
            err = hr_est - hr_ref
            per_key_est[k].append(hr_est)
            per_key_ref[k].append(hr_ref)
            per_key_sess[k].append({
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "within_10_pct": float(100.0 * np.mean(np.abs(err) <= 10.0)),
            })

    out: dict[str, Any] = {}
    for k in hr_keys:
        srows = per_key_sess[k]
        all_e = np.concatenate(per_key_est[k])
        all_r = np.concatenate(per_key_ref[k])
        rmc = IO.rmcorr_ancova(per_key_est[k], per_key_ref[k])
        w10_sess = float(np.mean([r["within_10_pct"] for r in srows]))
        out[k] = {
            "n_sessions": len(srows),
            "n_windows": int(all_r.size),
            "mae_sess": float(np.mean([r["mae"] for r in srows])),
            "rmse_sess": float(np.mean([r["rmse"] for r in srows])),
            "pct_within_10bpm_sess": w10_sess,
            "pct_within_10bpm_pool": float(100.0 * np.mean(np.abs(all_e - all_r) <= 10.0)),
            "r_within_rmcorr": rmc["r_within"],
            "passes_gate_g2_50pct": bool(w10_sess >= 50.0),
        }
    return out


def _phase_scramble_session(task: tuple[str, int, int]) -> tuple[str, dict[int, dict[str, np.ndarray]]]:
    """Run deterministic multi-seed full-session FFT phase scrambling for one radar session."""
    m, n_seeds, base_seed = task
    b_i, b_q = load_radar(m)
    x_full = csd(b_i, b_q)
    xf = np.fft.fft(x_full)
    mag_xf = np.abs(xf)
    w_len = int(C.WIN_SAMPLES)
    step = int(C.UPDATE_S * C.FS_RADAR)
    starts = list(range(0, x_full.size - w_len + 1, step))
    n_w = len(starts)

    pid = IO.patient_id_of(m)
    s_idx = int(m.split(".")[1])
    out_by_seed: dict[int, dict[str, np.ndarray]] = {}
    for seed_idx in range(n_seeds):
        rng_s = np.random.default_rng(int(base_seed) + seed_idx * 1009 + pid * 31 + s_idx)
        rand_phases = rng_s.uniform(0.0, 2.0 * np.pi, size=xf.size)
        rand_phases[0] = 0.0
        x_scr = np.fft.ifft(mag_xf * np.exp(1j * rand_phases))

        f_h = np.empty(n_w, dtype=np.float64)
        p2 = np.zeros((n_w, VT.BR_GRID_2BPM.size), dtype=np.float64)
        s1 = np.zeros((n_w, VT.BR_GRID_1BPM.size), dtype=np.float64)
        s1_anc = np.zeros((n_w, VT.BR_GRID_1BPM.size), dtype=np.float64)
        phr = np.zeros((n_w, VT.HR_GRID_2BPM.size), dtype=np.float64)
        for idx, s0 in enumerate(starts):
            xw = x_scr[s0 : s0 + w_len]
            dh = bandpass(mitigate_hpss(bandpass(xw))["x_hat"])
            fc_h = autocorr_coarse(dh)
            f_h[idx], _, _ = nls_estimate(dh, fc_h)
            p2[idx] = VT.extract_br_welch_spectrum(dh, VT.BR_GRID_2BPM)
            s1[idx] = VT.extract_br_harmonic_nls_spectrum(dh, VT.BR_GRID_1BPM)
            s1_anc[idx] = VT.extract_br_anchored_nls_spectrum(dh, fc_h, VT.BR_GRID_1BPM)
            phr[idx] = VT.extract_hr_csd_velocity_spectrum(xw, VT.HR_GRID_2BPM)

        out_by_seed[seed_idx] = {
            "HPSS_NLS": f_h,
            "C1b_1D_Vit_HPSS": VT.decode_viterbi_1d(p2, VT.BR_GRID_2BPM)[0],
            "C3b_2D_JointVit_HPSS": VT.decode_viterbi_2d_joint(p2, phr, VT.BR_GRID_2BPM, VT.HR_GRID_2BPM)[0],
            "C4b_2D_HarmNLS_Vit_HPSS": VT.decode_viterbi_2d_joint(s1, phr, VT.BR_GRID_1BPM, VT.HR_GRID_2BPM)[0],
            "C4d_2D_Anchored_HarmNLS_Vit_HPSS": VT.decode_viterbi_2d_joint(
                s1_anc, phr, VT.BR_GRID_1BPM, VT.HR_GRID_2BPM
            )[0],
        }
    return m, out_by_seed


def _run_pillar4_controls(
    decoded: dict[str, dict[str, Any]],
    ref_valid_map: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    workers: int = 4,
    skip_surrogates: bool = False,
) -> dict[str, Any]:
    """Run all 5 mandatory Pillar-IV controls (with multi-seed surrogate distributions)."""
    regime = "branch_neutral_raw_nls_lag"
    p30 = IO.get_cohort_sessions("pool30_sync", include_zero_valid=False)

    # 1. Stiff-HMM Anti-Inertia control (sigma_br = 0.25 bpm)
    stiff_results = {}
    for label, spec_key, grid, is_2d in (
        ("stiff_1d_welch_hpss", "P_br_hpss_2bpm", VT.BR_GRID_2BPM, False),
        ("stiff_2d_harmnls_hpss", "S_nls_hpss_1bpm", VT.BR_GRID_1BPM, True),
        ("stiff_2d_anchored_nls_hpss", "S_anc_hpss_1bpm", VT.BR_GRID_1BPM, True),
    ):
        est_map = {}
        for m in p30:
            obs = _load_obs_cache(m)
            p_seq = obs[spec_key]
            if not is_2d:
                est_map[m] = VT.decode_viterbi_1d(p_seq, grid, sigma_br=0.25)[0]
            else:
                est_map[m] = VT.decode_viterbi_2d_joint(
                    p_seq, obs["P_hr_csd_2bpm"], grid, VT.HR_GRID_2BPM, sigma_br=0.25
                )[0]
        agg, _ = _aggregate_method_metrics(p30, est_map, ref_valid_map, regime)
        stiff_results[label] = {
            "rmse_sess": agg["rmse_sess"],
            "acc10_sess": agg["acc10_sess"],
            "r_within_rmcorr": agg["r_within_rmcorr"],
            "r_within_sess_mean": agg["r_within_sess_mean"],
        }

    # 2a. Time-shuffled reference surrogate (20 seeds)
    n_time_seeds = 20
    shuf_methods = (
        "A4_HPSS_NLS",
        "C1b_1D_Vit_HPSS",
        "C3b_2D_JointVit_HPSS",
        "C4b_2D_HarmNLS_Vit_HPSS",
        "C4d_2D_Anchored_HarmNLS_Vit_HPSS",
    )
    time_shuf_runs: dict[str, list[tuple[float, float]]] = {mname: [] for mname in shuf_methods}
    for s_idx in range(n_time_seeds):
        rng_t = np.random.default_rng(IO.RANDOM_SEED + s_idx)
        shuf_ref_map: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        for m in p30:
            rf, vv = ref_valid_map[(m, regime)]
            rf_s = rf.copy()
            rf_s[vv] = rng_t.permutation(rf[vv])
            shuf_ref_map[(m, regime)] = (rf_s, vv)
        for mname in shuf_methods:
            est_map = {m: decoded[m]["est"][mname] for m in p30}
            agg_s, _ = _aggregate_method_metrics(p30, est_map, shuf_ref_map, regime)
            time_shuf_runs[mname].append((agg_s["rmse_sess"], agg_s["r_within_rmcorr"]))

    time_shuf = {}
    for mname, pairs in time_shuf_runs.items():
        rmses = np.array([p[0] for p in pairs], dtype=np.float64)
        rwins = np.array([p[1] for p in pairs], dtype=np.float64)
        time_shuf[mname] = {
            "n_seeds": n_time_seeds,
            "rmse_sess": float(np.mean(rmses)),
            "rmse_sess_sd": float(np.std(rmses, ddof=1)),
            "r_within_rmcorr": float(np.mean(rwins)),
            "r_within_rmcorr_sd": float(np.std(rwins, ddof=1)),
        }

    # 2b. Phase-scrambled radar surrogate (5 seeds, parallelized across sessions)
    n_scr_seeds = 5
    phase_scr: dict[str, Any] = {"skipped": bool(skip_surrogates), "n_seeds": n_scr_seeds}
    if not skip_surrogates:
        scr_tasks = [(m, n_scr_seeds, IO.RANDOM_SEED) for m in p30]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                scr_res_list = list(ex.map(_phase_scramble_session, scr_tasks))
        else:
            scr_res_list = [_phase_scramble_session(t) for t in scr_tasks]
        scr_by_sess = dict(scr_res_list)

        scr_methods = (
            "HPSS_NLS",
            "C1b_1D_Vit_HPSS",
            "C3b_2D_JointVit_HPSS",
            "C4b_2D_HarmNLS_Vit_HPSS",
            "C4d_2D_Anchored_HarmNLS_Vit_HPSS",
        )
        scr_runs: dict[str, list[tuple[float, float, float]]] = {k: [] for k in scr_methods}
        for seed_idx in range(n_scr_seeds):
            for mname in scr_methods:
                emap = {m: scr_by_sess[m][seed_idx][mname] for m in p30}
                agg_p, _ = _aggregate_method_metrics(p30, emap, ref_valid_map, regime)
                scr_runs[mname].append((agg_p["rmse_sess"], agg_p["acc10_sess"], agg_p["r_within_rmcorr"]))

        for mname, triples in scr_runs.items():
            rmses = np.array([x[0] for x in triples], dtype=np.float64)
            acc10s = np.array([x[1] for x in triples], dtype=np.float64)
            rwins = np.array([x[2] for x in triples], dtype=np.float64)
            phase_scr[mname] = {
                "n_seeds": n_scr_seeds,
                "rmse_sess": float(np.mean(rmses)),
                "rmse_sess_sd": float(np.std(rmses, ddof=1)),
                "acc10_sess": float(np.mean(acc10s)),
                "acc10_sess_sd": float(np.std(acc10s, ddof=1)),
                "r_within_rmcorr": float(np.mean(rwins)),
                "r_within_rmcorr_sd": float(np.std(rwins, ddof=1)),
            }

    # 3. Complex WGN noise floor (N=200 windows x 480 samples, seed=20261011)
    rng_wgn = np.random.default_rng(IO.RANDOM_SEED)
    n_wgn = 200
    f_hpss_wgn = np.empty(n_wgn, dtype=np.float64)
    p2_wgn = np.zeros((n_wgn, VT.BR_GRID_2BPM.size), dtype=np.float64)
    s1_wgn = np.zeros((n_wgn, VT.BR_GRID_1BPM.size), dtype=np.float64)
    s1_anc_wgn = np.zeros((n_wgn, VT.BR_GRID_1BPM.size), dtype=np.float64)
    phr_wgn = np.zeros((n_wgn, VT.HR_GRID_2BPM.size), dtype=np.float64)
    for k in range(n_wgn):
        xw = rng_wgn.standard_normal(C.WIN_SAMPLES) + 1j * rng_wgn.standard_normal(C.WIN_SAMPLES)
        dh = bandpass(mitigate_hpss(bandpass(xw))["x_hat"])
        fc_w = autocorr_coarse(dh)
        f_hpss_wgn[k], _, _ = nls_estimate(dh, fc_w)
        p2_wgn[k] = VT.extract_br_welch_spectrum(dh, VT.BR_GRID_2BPM)
        s1_wgn[k] = VT.extract_br_harmonic_nls_spectrum(dh, VT.BR_GRID_1BPM)
        s1_anc_wgn[k] = VT.extract_br_anchored_nls_spectrum(dh, fc_w, VT.BR_GRID_1BPM)
        phr_wgn[k] = VT.extract_hr_csd_velocity_spectrum(xw, VT.HR_GRID_2BPM)

    wgn_paths = {
        "HPSS_NLS": f_hpss_wgn,
        "B3_Clean_WelchPeak_HPSS": _argmax_grid(p2_wgn, VT.BR_GRID_2BPM),
        "C1b_1D_Vit_HPSS": VT.decode_viterbi_1d(p2_wgn, VT.BR_GRID_2BPM)[0],
        "C3b_2D_JointVit_HPSS": VT.decode_viterbi_2d_joint(p2_wgn, phr_wgn, VT.BR_GRID_2BPM, VT.HR_GRID_2BPM)[0],
        "C4b_2D_HarmNLS_Vit_HPSS": VT.decode_viterbi_2d_joint(s1_wgn, phr_wgn, VT.BR_GRID_1BPM, VT.HR_GRID_2BPM)[0],
        "C4d_2D_Anchored_HarmNLS_Vit_HPSS": VT.decode_viterbi_2d_joint(
            s1_anc_wgn, phr_wgn, VT.BR_GRID_1BPM, VT.HR_GRID_2BPM
        )[0],
    }
    wgn_summary = {}
    for k, arr in wgn_paths.items():
        wgn_summary[k] = {
            "median_bpm": float(np.median(arr)),
            "iqr_bpm": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
            "pct_in_paper18_range_25_8_to_81_5": float(100.0 * np.mean((arr >= 25.8) & (arr <= 81.5))),
        }

    # 4. Mathematical group invariances (Session 7.3 first 10 windows, including anchored NLS & DACM/notched HR)
    b_i, b_q = load_radar("7.3")
    seg_i, seg_q = b_i[: 480 + 9 * 32], b_q[: 480 + 9 * 32]

    def _run_inv_slice(arr_i: np.ndarray, arr_q: np.ndarray) -> dict[str, np.ndarray]:
        p2_s, s1_s, s1_anc_s, phr_s, phr_dacm_s, phr_notch_s = [], [], [], [], [], []
        for s0 in range(0, 10 * 32, 32):
            xw = csd(arr_i[s0 : s0 + 480], arr_q[s0 : s0 + 480])
            dh = bandpass(mitigate_hpss(bandpass(xw))["x_hat"])
            fc_i = autocorr_coarse(dh)
            fh_i, _, _ = nls_estimate(dh, fc_i)
            p2_s.append(VT.extract_br_welch_spectrum(dh, VT.BR_GRID_2BPM))
            s1_s.append(VT.extract_br_harmonic_nls_spectrum(dh, VT.BR_GRID_1BPM))
            s1_anc_s.append(VT.extract_br_anchored_nls_spectrum(dh, fc_i, VT.BR_GRID_1BPM))
            phr_s.append(VT.extract_hr_csd_velocity_spectrum(xw, VT.HR_GRID_2BPM))
            phr_dacm_s.append(VT.extract_hr_dacm_velocity_spectrum(xw, VT.HR_GRID_2BPM))
            phr_notch_s.append(VT.extract_hr_br_notched_velocity_spectrum(xw, fh_i, VT.HR_GRID_2BPM))
        p2_a = np.asarray(p2_s)
        s1_a = np.asarray(s1_s)
        s1_anc_a = np.asarray(s1_anc_s)
        phr_a = np.asarray(phr_s)
        phr_dacm_a = np.asarray(phr_dacm_s)
        phr_notch_a = np.asarray(phr_notch_s)
        return {
            "Welch_BR_HPSS": _argmax_grid(p2_a, VT.BR_GRID_2BPM),
            "HarmNLS_BR_HPSS": _argmax_grid(s1_a, VT.BR_GRID_1BPM),
            "Anchored_HarmNLS_BR_HPSS": _argmax_grid(s1_anc_a, VT.BR_GRID_1BPM),
            "CSD_Vel_HR": _argmax_grid(phr_a, VT.HR_GRID_2BPM),
            "DACM_HR": _argmax_grid(phr_dacm_a, VT.HR_GRID_2BPM),
            "BR_Notched_HR": _argmax_grid(phr_notch_a, VT.HR_GRID_2BPM),
            "C1b_1D_Vit_HPSS": VT.decode_viterbi_1d(p2_a, VT.BR_GRID_2BPM)[0],
            "C3b_2D_JointVit_HPSS": VT.decode_viterbi_2d_joint(p2_a, phr_a, VT.BR_GRID_2BPM, VT.HR_GRID_2BPM)[0],
            "C4b_2D_HarmNLS_Vit_HPSS": VT.decode_viterbi_2d_joint(s1_a, phr_a, VT.BR_GRID_1BPM, VT.HR_GRID_2BPM)[0],
            "C4d_2D_Anchored_HarmNLS_Vit_HPSS": VT.decode_viterbi_2d_joint(
                s1_anc_a, phr_a, VT.BR_GRID_1BPM, VT.HR_GRID_2BPM
            )[0],
        }

    ref_inv = _run_inv_slice(seg_i, seg_q)
    z_rot = (seg_i + 1j * seg_q) * np.exp(1j * np.pi / 3.0)
    invariance_summary: dict[str, dict[str, float]] = {}
    for tname, (ti, tq) in {
        "scale_3p7x": (3.7 * seg_i, 3.7 * seg_q),
        "phase_rot_pi_over_3": (z_rot.real, z_rot.imag),
        "iq_sign_flip": (-seg_i, -seg_q),
        "iq_channel_swap": (seg_q, seg_i),
    }.items():
        alt = _run_inv_slice(ti, tq)
        invariance_summary[tname] = {k: float(np.max(np.abs(ref_inv[k] - alt[k]))) for k in ref_inv}

    # 5. Deterministic synthetic ground-truth recovery
    fs = float(C.FS_RADAR)
    dur_s = 120.0
    t = np.arange(int(dur_s * fs), dtype=np.float64) / fs
    synth_summary = {}
    for cname, chirp in (("stationary_45bpm_with_150hr", False), ("chirp_35_to_65bpm_rbm_150hr", True)):
        if not chirp:
            f_inst = np.full_like(t, 45.0 / 60.0)
            phase_resp = 2.0 * np.pi * f_inst * t
            d_m = 0.0012 * np.sin(phase_resp) + 0.00012 * np.sin(2.0 * np.pi * 2.5 * t)
            x_syn = np.exp(1j * (0.4 + 4.0 * np.pi * d_m / C.LAMBDA_M))
            x_syn = x_syn + 0.01 * (rng_wgn.standard_normal(t.size) + 1j * rng_wgn.standard_normal(t.size))
        else:
            f_inst = (35.0 + 30.0 * (t / dur_s)) / 60.0
            phase_resp = 2.0 * np.pi * ((35.0 / 60.0) * t + 0.5 * (30.0 / 60.0 / dur_s) * (t ** 2))
            d_m = 0.0010 * np.sin(phase_resp) + 0.00010 * np.sin(2.0 * np.pi * 2.5 * t)
            x_syn = np.exp(1j * (0.4 + 4.0 * np.pi * d_m / C.LAMBDA_M))
            rbm_mask = (t >= 50.0) & (t <= 54.0)
            x_syn[rbm_mask] *= 8.0
            x_syn = x_syn + 0.01 * (rng_wgn.standard_normal(t.size) + 1j * rng_wgn.standard_normal(t.size))

        starts = list(range(0, t.size - int(C.WIN_SAMPLES) + 1, int(C.UPDATE_S * C.FS_RADAR)))
        truth_bpm = np.array([f_inst[s0 + int(C.WIN_SAMPLES) // 2] * 60.0 for s0 in starts], dtype=np.float64)
        f_h, p2_s, s1_s, s1_anc_s, phr_s = [], [], [], [], []
        for s0 in starts:
            xw = x_syn[s0 : s0 + int(C.WIN_SAMPLES)]
            dh = bandpass(mitigate_hpss(bandpass(xw))["x_hat"])
            fc_s = autocorr_coarse(dh)
            fh, _, _ = nls_estimate(dh, fc_s)
            f_h.append(fh)
            p2_s.append(VT.extract_br_welch_spectrum(dh, VT.BR_GRID_2BPM))
            s1_s.append(VT.extract_br_harmonic_nls_spectrum(dh, VT.BR_GRID_1BPM))
            s1_anc_s.append(VT.extract_br_anchored_nls_spectrum(dh, fc_s, VT.BR_GRID_1BPM))
            phr_s.append(VT.extract_hr_csd_velocity_spectrum(xw, VT.HR_GRID_2BPM))
        p2_a, s1_a, s1_anc_a, phr_a = (
            np.asarray(p2_s),
            np.asarray(s1_s),
            np.asarray(s1_anc_s),
            np.asarray(phr_s),
        )
        cands = {
            "HPSS_NLS": np.asarray(f_h),
            "C1b_1D_Vit_HPSS": VT.decode_viterbi_1d(p2_a, VT.BR_GRID_2BPM)[0],
            "C3b_2D_JointVit_HPSS": VT.decode_viterbi_2d_joint(p2_a, phr_a, VT.BR_GRID_2BPM, VT.HR_GRID_2BPM)[0],
            "C4b_2D_HarmNLS_Vit_HPSS": VT.decode_viterbi_2d_joint(s1_a, phr_a, VT.BR_GRID_1BPM, VT.HR_GRID_2BPM)[0],
            "C4d_2D_Anchored_HarmNLS_Vit_HPSS": VT.decode_viterbi_2d_joint(
                s1_anc_a, phr_a, VT.BR_GRID_1BPM, VT.HR_GRID_2BPM
            )[0],
        }
        synth_summary[cname] = {
            k: {
                "mae_bpm": float(np.mean(np.abs(v - truth_bpm))),
                "rmse_bpm": float(np.sqrt(np.mean((v - truth_bpm) ** 2))),
            }
            for k, v in cands.items()
        }

    return {
        "stiff_hmm_anti_inertia": stiff_results,
        "surrogate_time_shuffled_ref": time_shuf,
        "surrogate_phase_scrambled_radar": phase_scr,
        "complex_wgn_noise_floor": wgn_summary,
        "mathematical_group_invariances_max_diff": invariance_summary,
        "synthetic_ground_truth": synth_summary,
    }


def _run_audit_stress_tests(
    decoded: dict[str, dict[str, Any]],
    ref_valid_map: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Run Welch 30-s resolution and Legacy-AD drift/init sensitivity stress-tests."""
    regime = "branch_neutral_raw_nls_lag"
    p30 = IO.get_cohort_sessions("pool30_sync", include_zero_valid=False)
    r23 = IO.get_cohort_sessions("retained23", include_zero_valid=False)

    welch_variants = {
        "B3_Clean_WelchPeak_HPSS_n256": {m: decoded[m]["est"]["B3_Clean_WelchPeak_HPSS"] for m in p30},
        "B3_Clean_WelchPeak_HPSS_n480": {m: decoded[m]["stress_est"]["B3_Clean_WelchPeak_HPSS_n480"] for m in p30},
        "C3b_2D_JointVit_HPSS_n256": {m: decoded[m]["est"]["C3b_2D_JointVit_HPSS"] for m in p30},
        "C3b_2D_JointVit_HPSS_br_n480": {m: decoded[m]["stress_est"]["C3b_2D_JointVit_HPSS_br_n480"] for m in p30},
        "C3b_2D_JointVit_HPSS_both_n480": {m: decoded[m]["stress_est"]["C3b_2D_JointVit_HPSS_both_n480"] for m in p30},
    }
    welch_out = {}
    for k, emap in welch_variants.items():
        agg, _ = _aggregate_method_metrics(p30, emap, ref_valid_map, regime)
        welch_out[k] = {
            "rmse_sess": agg["rmse_sess"],
            "rmse_pool": agg["rmse_pool"],
            "acc10_sess": agg["acc10_sess"],
            "r_within_rmcorr": agg["r_within_rmcorr"],
        }

    legacy_out: dict[str, dict[str, Any]] = {}
    for coh_name, sess_list in (("retained23", r23), ("pool30_sync", p30)):
        ad_variants = {
            "B7_raw_nndsvda_baseline": {m: decoded[m]["est"]["B7_Diag_Legacy_AD_RawNMF"] for m in sess_list},
            "B7_sens_raw_nndsvd": {m: decoded[m]["stress_est"]["B7_sens_raw_nndsvd"] for m in sess_list},
            "B7_sens_detrend_bp_nndsvd": {m: decoded[m]["stress_est"]["B7_sens_detrend_bp_nndsvd"] for m in sess_list},
        }
        legacy_out[coh_name] = {}
        for k, emap in ad_variants.items():
            agg, _ = _aggregate_method_metrics(sess_list, emap, ref_valid_map, regime)
            legacy_out[coh_name][k] = {
                "rmse_sess": agg["rmse_sess"],
                "rmse_pool": agg["rmse_pool"],
                "acc10_sess": agg["acc10_sess"],
                "r_within_rmcorr": agg["r_within_rmcorr"],
            }

    return {
        "welch_n480_sensitivity_pool30_sync": welch_out,
        "legacy_ad_sensitivity": legacy_out,
    }


def _subgroup_splits(
    cohort: str,
    sessions: list[str],
    est_by_method: dict[str, dict[str, np.ndarray]],
    ref_valid_map: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    regime: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    if cohort == "retained23":
        groups = {
            "prone": [m for m in sessions if MEASURE_INFO.get(m, ("?",))[0] == "prone"],
            "non_prone": [m for m in sessions if MEASURE_INFO.get(m, ("?",))[0] != "prone"],
        }
    elif cohort == "PAPER_18":
        groups = {
            "supine": [m for m in sessions if MEASURE_INFO.get(m, ("?",))[0] == "supine"],
            "prone": [m for m in sessions if MEASURE_INFO.get(m, ("?",))[0] == "prone"],
            "single": [m for m in sessions if not MEASURE_INFO.get(m, ("?", False))[1]],
            "twins": [m for m in sessions if MEASURE_INFO.get(m, ("?", False))[1]],
        }
    else:
        return {}

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for gname, g_ms in groups.items():
        out[gname] = {}
        for mname, emap in est_by_method.items():
            agg, _ = _aggregate_method_metrics(g_ms, emap, ref_valid_map, regime)
            out[gname][mname] = agg
    return out


def _write_reconciliation_markdown(master: dict[str, Any]) -> None:
    """Write concise structured reconciliation report mapping draft vs audited vs Viterbi."""
    p_ret_neu = master["results"]["branch_neutral_raw_nls_lag"]["retained23"]
    p_p30_neu = master["results"]["branch_neutral_raw_nls_lag"]["pool30_sync"]
    p_18_cached = master["results"]["cached_nmf_lag"]["PAPER_18"]

    lines = [
        "# Viterbi Reproduction & Baseline Reconciliation Audit",
        "",
        f"**Pipeline Anchor:** `main-v3-audit-fix-2` (`33/33` SHA-256 locked baseline caches verified)  ",
        f"**Schema:** `{master['provenance']['schema_version']}` | **Git Head:** `{master['provenance']['git']['head_commit']}` (`{master['provenance']['git']['branch']}`) | **Python:** `{master['provenance']['git']['python_version']}`",
        "",
        "## 1. Locked Baseline Parity Check (Fresh Python 3.10 vs Frozen Python 3.9 Caches)",
        f"- Max window deviation across all 33 sessions — `DFT`: `{master['parity_audit']['max_abs_diff_dft']:.2e}` bpm, "
        f"`Raw NLS`: `{master['parity_audit']['max_abs_diff_nls']:.2e}` bpm, "
        f"`NMF+NLS`: `{master['parity_audit']['max_abs_diff_nmf']:.2e}` bpm, "
        f"`HPSS+NLS`: `{master['parity_audit']['max_abs_diff_hpss']:.2e}` bpm.",
        f"- `PAPER_18` (`cached_nmf_lag`) locked baseline reproduction: "
        f"`NLS+NMF` = `{p_18_cached['methods']['A3_Beltrao_NMF_NLS']['rmse_sess']:.3f}` bpm (`{p_18_cached['methods']['A3_Beltrao_NMF_NLS']['acc10_sess']:.1f}%`), "
        f"`HPSS+NLS` = `{p_18_cached['methods']['A4_HPSS_NLS']['rmse_sess']:.3f}` bpm (`{p_18_cached['methods']['A4_HPSS_NLS']['acc10_sess']:.1f}%`).",
        "",
        "## 2. Primary Confirmatory Cohort (`pool30_sync`, Branch-Neutral Synchronization)",
        f"- Constant baselines: `const_loso` = `{p_p30_neu['constants']['const_loso']:.3f}` bpm, "
        f"`const_lopo_median` = `{p_p30_neu['constants']['const_lopo_median']:.3f}` bpm, "
        f"`const_lopo_mean` = `{p_p30_neu['constants']['const_lopo_mean']:.3f}` bpm, "
        f"`const_global_oracle` = `{p_p30_neu['constants']['const_global_oracle']:.3f}` bpm, "
        f"`session_mean_oracle` = `{p_p30_neu['constants']['session_mean_oracle']:.3f}` bpm.",
        "",
        "| Method | S | Windows | RMSE_sess | RMSE_pool | MAE_sess | Acc@10 | r_within (rmcorr) | r_sess_mean |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for mname in ALL_METHODS:
        d = p_p30_neu["methods"][mname]
        lines.append(
            f"| `{mname}` | {d['n_sessions']} | {d['n_windows']} | {d['rmse_sess']:.3f} | {d['rmse_pool']:.3f} | "
            f"{d['mae_sess']:.3f} | {d['acc10_sess']:.1f}% | {d['r_within_rmcorr']:.3f} | {d['r_within_sess_mean']:.3f} |"
        )

    ref_d = master["_definitions"]["attached_draft_reference"]
    lines.extend([
        "",
        "## 3. Attached Draft Bridge on `retained23` (Branch-Neutral Synchronization, 6,447 Valid 2-s Windows)",
        f"- Attached draft Table 1 claimed (`0.5 Hz` ref interpolation / `1-s` grid = `{ref_d['n_windows_1s_grid']}` windows, dynamic per-method lag): "
        f"`Naive FFT = {ref_d['naive_fft_rmse_bpm']:.2f} bpm ({ref_d['naive_fft_acc10_pct']:.1f}% Acc10)`, "
        f"`No Joint 2D Viterbi = {ref_d['no_joint_rmse_bpm']:.2f} bpm ({ref_d['no_joint_acc10_pct']:.1f}% Acc10)`, "
        f"`Proposed = {ref_d['proposed_rmse_bpm']:.2f} bpm ({ref_d['proposed_acc10_pct']:.1f}% Acc10)`, "
        f"`Prone = {ref_d['prone_rmse_bpm']:.2f} bpm ({ref_d['prone_acc10_pct']:.1f}% Acc10)`.",
        "",
        "| Method | RMSE_sess | RMSE_pool | Acc@10_sess | r_within | BA Bias [LoA] (Session) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    bridge_methods = (
        "B6_Diag_Legacy_AD_NaiveFFT",
        "B7_Diag_Legacy_AD_RawNMF",
        "C3d_2D_JointVit_NMF_ADHR_Diag",
        "B1_Clean_WelchPeak_Raw",
        "B2_Clean_WelchPeak_NMF",
        "B3_Clean_WelchPeak_HPSS",
        "B5a_Anchored_HarmNLS_Peak_HPSS",
        "A1_DFT_Locked",
        "A2_RawNLS_Locked",
        "A3_Beltrao_NMF_NLS",
        "A4_HPSS_NLS",
        "C1b_1D_Vit_HPSS",
        "C1m_1D_Vit_HPSS_MatchedW05",
        "C2b_1D_HQI_HPSS",
        "C3b_2D_JointVit_HPSS",
        "C4b_2D_HarmNLS_Vit_HPSS",
        "C4c_1D_Anchored_HarmNLS_Vit_HPSS",
        "C4d_2D_Anchored_HarmNLS_Vit_HPSS",
        "C6a_LOPO_Cal_Vit_HPSS",
        "C6b_LOPO_Cal_HarmNLS_HPSS",
        "C6c_LOPO_Cal_HarmNLS_NMF",
        "C6d_LOPO_Wide_AnchoredNLS_HPSS",
        "C6e_LOPO_Wide_AnchoredNLS_NMF",
    )
    for mname in bridge_methods:
        d = p_ret_neu["methods"][mname]
        ba = d["ba_session_level"]
        lines.append(
            f"| `{mname}` | {d['rmse_sess']:.3f} | {d['rmse_pool']:.3f} | {d['acc10_sess']:.1f}% | "
            f"{d['r_within_rmcorr']:.3f} | {ba['bias']:+.2f} [{ba['loa_lo']:+.2f}, {ba['loa_up']:+.2f}] |"
        )

    lines.extend([
        "",
        "## 4. Secondary HR Validation Against EDF 1024 Hz R-Trigger (`SignalIQ >= 50`)",
        "| HR Estimator | Windows | MAE_sess (bpm) | RMSE_sess (bpm) | <=10 bpm (sess) | r_within | Gate G2 (>=50%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for hk, hv in master["hr_validation"].items():
        lines.append(
            f"| `{hk}` | {hv['n_windows']} | {hv['mae_sess']:.2f} | {hv['rmse_sess']:.2f} | "
            f"{hv['pct_within_10bpm_sess']:.1f}% | {hv['r_within_rmcorr']:.3f} | {hv['passes_gate_g2_50pct']} |"
        )

    lines.extend([
        "",
        "## 5. LOPO Hyperparameter Selection & Pinning Diagnostics (`pool30_sync`, `branch_neutral_raw_nls_lag`)",
        "| LOPO Family | Grid | Folds | Fallback Folds | Upper-Pinned Folds | Any-Boundary Folds |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for fam, ls in p_p30_neu["lopo_summary"].items():
        lines.append(
            f"| `{fam}` | {ls['grid_mode']} | {ls['n_folds']} | {ls['n_fallback']} | {ls['n_upper_pinned']} | {ls['n_any_boundary_pinned']} |"
        )

    lines.extend([
        "",
        "## 6. Physics & Multi-Seed Surrogate Stress-Tests (`branch_neutral_raw_nls_lag`)",
        "### Welch 30-s (`nperseg=480`) vs 16-s (`nperseg=256`) on `pool30_sync`",
        "| Variant | RMSE_sess | RMSE_pool | Acc@10_sess | r_within |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for k, d in master["audit_stress_tests"]["welch_n480_sensitivity_pool30_sync"].items():
        lines.append(
            f"| `{k}` | {d['rmse_sess']:.3f} | {d['rmse_pool']:.3f} | {d['acc10_sess']:.1f}% | {d['r_within_rmcorr']:.3f} |"
        )

    lines.extend([
        "",
        "### Legacy AD + NMF Drift & Init Sensitivity",
        "| Cohort | Variant | RMSE_sess | RMSE_pool | Acc@10_sess | r_within |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])
    for coh_n, cdict in master["audit_stress_tests"]["legacy_ad_sensitivity"].items():
        for k, d in cdict.items():
            lines.append(
                f"| `{coh_n}` | `{k}` | {d['rmse_sess']:.3f} | {d['rmse_pool']:.3f} | {d['acc10_sess']:.1f}% | {d['r_within_rmcorr']:.3f} |"
            )

    AUDIT_RECON_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(workers: int = 4, force_cache: bool = False, skip_surrogates: bool = False) -> dict[str, Any]:
    """Execute complete extraction, ablation ladder, LOPO, HR check, and Pillar-IV controls."""
    IO.verify_all_baseline_caches()
    tracker_sha = IO.sha256_file(TRACKER_PATH)
    all_sessions = IO.get_cohort_sessions("all_paired", include_zero_valid=True)

    # Stage 1: Observation extraction & cache parity
    tasks = [(m, tracker_sha, force_cache) for m in all_sessions]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            statuses = list(ex.map(_extract_one_session, tasks))
    else:
        statuses = [_extract_one_session(t) for t in tasks]
    n_computed = sum(1 for _, s in statuses if s == "computed")
    print(f"Stage 1 observation cache: {len(statuses)} sessions ({n_computed} computed, {len(statuses) - n_computed} cached).")

    # Stage 2a: Decode all fixed methods + LOPO grid candidates in parallel across sessions
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            dec_list = list(ex.map(_decode_fixed_session, all_sessions))
    else:
        dec_list = [_decode_fixed_session(m) for m in all_sessions]
    decoded = {d["measurement"]: d for d in dec_list}

    parity_audit = {
        "max_abs_diff_dft": max(d["parity"]["dft_max"] for d in dec_list),
        "max_abs_diff_nls": max(d["parity"]["nls_max"] for d in dec_list),
        "max_abs_diff_nmf": max(d["parity"]["nmf_max"] for d in dec_list),
        "max_abs_diff_hpss": max(d["parity"]["hpss_max"] for d in dec_list),
        "mean_t_nmf_ms": float(np.mean([d["parity"]["t_nmf_ms"] for d in dec_list])),
        "mean_t_hpss_ms": float(np.mean([d["parity"]["t_hpss_ms"] for d in dec_list])),
    }
    print("Stage 1 parity max abs diff:", parity_audit)

    # Load EDF references and align under all 3 lag regimes
    lags_map = IO.load_locked_neutral_lags()
    edf_map = {m: EDFX.load_edf_recording(m) for m in all_sessions}
    ref_valid_map: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    for m in all_sessions:
        cache = IO.load_locked_cache(m)
        edf = edf_map[m]
        n_win = len(cache["est"])
        centers = np.arange(n_win, dtype=float) * C.UPDATE_S + C.WIN_S / 2.0
        finite_est = np.isfinite(cache["est"]) & np.isfinite(cache["est_hpss"])
        lag_info = lags_map[m]
        reg_lags = {
            "branch_neutral_raw_nls_lag": int(lag_info["neutral_raw_nls_lag_s"]),
            "pure_metadata_clock_lag": int(lag_info["pure_metadata_lag_s"]),
            "cached_nmf_lag": int(lag_info["cached_nmf_lag_s"]),
        }
        for reg, lval in reg_lags.items():
            if reg == "cached_nmf_lag":
                ref_valid_map[(m, reg)] = (np.asarray(cache["ref"], dtype=np.float64), np.asarray(cache["valid"], bool))
            else:
                ref_w, vwin = EV.window_ref_and_gating(
                    edf.resp_256, edf.resp_rate_1hz, centers, lval, fs_raw=edf.fs_resp
                )
                v_arr = vwin & np.isfinite(ref_w) & finite_est
                ref_valid_map[(m, reg)] = (ref_w, v_arr)

    # Evaluate Cohort x Regime matrix + LOPO + Subgroups + Paired Stats
    results: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []

    pool30_set = set(IO.get_cohort_sessions("pool30_sync"))
    ret23_set = set(IO.get_cohort_sessions("retained23"))
    p18_set = set(IO.get_cohort_sessions("PAPER_18"))

    for reg in REGIME_ORDER:
        results[reg] = {}
        for cohort in COHORT_ORDER:
            raw_ids = IO.get_cohort_sessions(cohort, include_zero_valid=False)
            eval_ids = [m for m in raw_ids if np.any(ref_valid_map[(m, reg)][1])]
            lopo_preds, lopo_folds = _build_lopo_predictions(cohort, reg, decoded, ref_valid_map)

            lopo_summary: dict[str, dict[str, Any]] = {}
            for fam_name, _, _, _, grid_mode in LOPO_FAMILIES:
                frows = lopo_folds[fam_name]
                lopo_summary[fam_name] = {
                    "grid_mode": grid_mode,
                    "n_folds": len(frows),
                    "n_fallback": int(sum(1 for f in frows if f["fallback_used"])),
                    "n_upper_pinned": int(sum(1 for f in frows if f["upper_pinned"])),
                    "n_any_boundary_pinned": int(sum(1 for f in frows if f["any_boundary_pinned"])),
                }

            est_by_method: dict[str, dict[str, np.ndarray]] = {
                mname: {m: decoded[m]["est"][mname] for m in eval_ids}
                for mname in FIXED_METHODS
            }
            for fam_name, _, _, _, _ in LOPO_FAMILIES:
                est_by_method[fam_name] = lopo_preds[fam_name]

            constants, per_sess_const = _constant_baselines(eval_ids, ref_valid_map, reg)
            methods_agg: dict[str, Any] = {}
            per_sess_by_method: dict[str, dict[str, dict[str, float]]] = {}
            for mname in ALL_METHODS:
                agg, p_sess = _aggregate_method_metrics(eval_ids, est_by_method[mname], ref_valid_map, reg)
                methods_agg[mname] = agg
                per_sess_by_method[mname] = p_sess

                summary_rows.append({
                    "regime": reg,
                    "cohort": cohort,
                    "method": mname,
                    "n_sessions": agg.get("n_sessions", 0),
                    "n_windows": agg.get("n_windows", 0),
                    "rmse_sess": agg.get("rmse_sess", float("nan")),
                    "rmse_pool": agg.get("rmse_pool", float("nan")),
                    "mae_sess": agg.get("mae_sess", float("nan")),
                    "acc3_sess": agg.get("acc3_sess", float("nan")),
                    "acc5_sess": agg.get("acc5_sess", float("nan")),
                    "acc6_sess": agg.get("acc6_sess", float("nan")),
                    "acc10_sess": agg.get("acc10_sess", float("nan")),
                    "acc20_sess": agg.get("acc20_sess", float("nan")),
                    "gross_gt10_sess_pct": agg.get("gross_gt10_sess_pct", float("nan")),
                    "r_within_rmcorr": agg.get("r_within_rmcorr", float("nan")),
                    "r_within_sess_mean": agg.get("r_within_sess_mean", float("nan")),
                    "ba_sess_bias": agg.get("ba_session_level", {}).get("bias", float("nan")),
                    "ba_sess_loa_lo": agg.get("ba_session_level", {}).get("loa_lo", float("nan")),
                    "ba_sess_loa_up": agg.get("ba_session_level", {}).get("loa_up", float("nan")),
                    "ba_sess_prop_p": agg.get("ba_session_level", {}).get("prop_p", float("nan")),
                    "icc_1_1": agg.get("ba_session_level", {}).get("icc_1_1", float("nan")),
                    "const_loso": constants["const_loso"],
                    "const_lopo_median": constants["const_lopo_median"],
                    "const_lopo_mean": constants["const_lopo_mean"],
                })

            # Paired comparisons against NMF+NLS, HPSS+NLS, Const-LOSO, and Const-LOPO-Median with Holm-Bonferroni
            paired_vs_nmf: dict[str, Any] = {}
            paired_vs_hpss: dict[str, Any] = {}
            paired_vs_const: dict[str, Any] = {}
            paired_vs_const_lopo: dict[str, Any] = {}
            viterbi_names = [m for m in ALL_METHODS if m.startswith("C")]
            for mname in ALL_METHODS:
                paired_vs_nmf[mname] = _paired_tests(eval_ids, per_sess_by_method[mname], per_sess_by_method["A3_Beltrao_NMF_NLS"])
                paired_vs_hpss[mname] = _paired_tests(eval_ids, per_sess_by_method[mname], per_sess_by_method["A4_HPSS_NLS"])
                paired_vs_const[mname] = _paired_tests(eval_ids, per_sess_by_method[mname], per_sess_const, anchor_rmse_key="const_loso_rmse")
                paired_vs_const_lopo[mname] = _paired_tests(
                    eval_ids, per_sess_by_method[mname], per_sess_const, anchor_rmse_key="const_lopo_median_rmse"
                )

            for anchor_dict in (paired_vs_nmf, paired_vs_hpss, paired_vs_const, paired_vs_const_lopo):
                pvals = [anchor_dict[m]["p_wilcoxon"] for m in viterbi_names]
                padj = IO.holm_bonferroni([p if np.isfinite(p) else 1.0 for p in pvals])
                pvals_pat = [anchor_dict[m]["p_wilcoxon_patient"] for m in viterbi_names]
                padj_pat = IO.holm_bonferroni([p if np.isfinite(p) else 1.0 for p in pvals_pat])
                for m, pa, pa_pat in zip(viterbi_names, padj, padj_pat):
                    anchor_dict[m]["p_wilcoxon_holm"] = pa
                    anchor_dict[m]["p_wilcoxon_patient_holm"] = pa_pat

            subgroups = _subgroup_splits(cohort, eval_ids, est_by_method, ref_valid_map, reg)
            for gname, g_dict in subgroups.items():
                for mname, d in g_dict.items():
                    subgroup_rows.append({
                        "regime": reg,
                        "cohort": cohort,
                        "subgroup": gname,
                        "method": mname,
                        "n_sessions": d.get("n_sessions", 0),
                        "n_windows": d.get("n_windows", 0),
                        "rmse_sess": d.get("rmse_sess", float("nan")),
                        "rmse_pool": d.get("rmse_pool", float("nan")),
                        "mae_sess": d.get("mae_sess", float("nan")),
                        "acc5_sess": d.get("acc5_sess", float("nan")),
                        "acc10_sess": d.get("acc10_sess", float("nan")),
                        "acc20_sess": d.get("acc20_sess", float("nan")),
                        "r_within_rmcorr": d.get("r_within_rmcorr", float("nan")),
                    })

            # Emit cohort-scoped session rows for EVERY cohort so LOPO scatter matches exact cohort folds
            for m in eval_ids:
                for mname in ALL_METHODS:
                    if m not in per_sess_by_method[mname]:
                        continue
                    sc = per_sess_by_method[mname][m]
                    pos, twins, _ = MEASURE_INFO.get(m, ("?", False, "?"))
                    session_rows.append({
                        "regime": reg,
                        "cohort": cohort,
                        "measurement": m,
                        "patient_id": IO.patient_id_of(m),
                        "position": pos,
                        "cobedding_twins": twins,
                        "in_pool30_sync": m in pool30_set,
                        "in_retained23": m in ret23_set,
                        "in_paper18": m in p18_set,
                        "method": mname,
                        "n_valid": int(np.sum(ref_valid_map[(m, reg)][1])),
                        "rmse": sc["rmse"],
                        "mae": sc["mae"],
                        "bias": sc["bias"],
                        "acc5": sc["acc5"],
                        "acc10": sc["acc10"],
                        "r": sc["r"],
                        "est_mean": sc["est_mean"],
                        "ref_mean": sc["ref_mean"],
                    })

            results[reg][cohort] = {
                "sessions": eval_ids,
                "constants": constants,
                "methods": methods_agg,
                "paired_vs_nmf_nls": paired_vs_nmf,
                "paired_vs_hpss_nls": paired_vs_hpss,
                "paired_vs_const_loso": paired_vs_const,
                "paired_vs_const_lopo_median": paired_vs_const_lopo,
                "subgroups": subgroups,
                "lopo_folds": lopo_folds,
                "lopo_summary": lopo_summary,
            }

    # Stage 3: HR Validation + Pillar-IV 5 Controls + Physics Audit Stress-Tests
    hr_validation = _evaluate_hr_branch(decoded, edf_map, lags_map)
    controls = _run_pillar4_controls(decoded, ref_valid_map, workers=workers, skip_surrogates=skip_surrogates)
    audit_stress_tests = _run_audit_stress_tests(decoded, ref_valid_map)

    # Write CSV exports
    IO.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(AUDIT_SUMMARY_CSV, index=False)
    pd.DataFrame(subgroup_rows).to_csv(AUDIT_SUBGROUPS_CSV, index=False)
    pd.DataFrame(session_rows).to_csv(AUDIT_SESSIONS_CSV, index=False)

    master = {
        "_definitions": {
            "scope": "Baseline-anchored reproduction of score-based HR-BR Viterbi tracking on top of locked main-v3-audit-fix-2.",
            "regimes": {
                "branch_neutral_raw_nls_lag": "Primary confirmatory regime: one shared Raw-NLS lag per session.",
                "pure_metadata_clock_lag": "Zero-search header clock offset sensitivity regime.",
                "cached_nmf_lag": "Exact v1.0-student-baseline lag & gating parity bridge.",
            },
            "acc_thresholds": "Strict `< tau` bpm error per notation_glossary.tex (`Acc@3`, `Acc@5`, `Acc@6`, `Acc@10`, `Acc@20`).",
            "lopo_rule": "Minimizes training-patient RMSE_sess subject to r_within_rmcorr_train >= 0.95 * anchor_r_within_train.",
            "attached_draft_reference": ATTACHED_DRAFT_REFERENCE,
        },
        "provenance": {
            "schema_version": SCHEMA_VERSION,
            "tracker_sha256": tracker_sha,
            "runner_sha256": IO.sha256_file(Path(__file__).resolve()),
            "git": IO.git_head_info(),
            "bootstrap_seed": IO.RANDOM_SEED,
        },
        "parity_audit": parity_audit,
        "results": results,
        "hr_validation": hr_validation,
        "controls": controls,
        "audit_stress_tests": audit_stress_tests,
    }

    IO.write_json_locked(BUNDLE_JSON, master)
    _write_reconciliation_markdown(master)
    print(f"Wrote {BUNDLE_JSON}, {AUDIT_SUMMARY_CSV}, {AUDIT_SUBGROUPS_CSV}, {AUDIT_SESSIONS_CSV}, {AUDIT_RECON_MD}")
    return master


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline-anchored Viterbi tracking reproduction.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes for observation extraction and decoding.")
    parser.add_argument("--force-cache", action="store_true", help="Recompute observation caches from scratch.")
    parser.add_argument("--skip-surrogates", action="store_true", help="Skip phase-scrambled full-session surrogate pass.")
    args = parser.parse_args()
    run_experiment(workers=max(1, args.workers), force_cache=args.force_cache, skip_surrogates=args.skip_surrogates)


if __name__ == "__main__":
    main()
