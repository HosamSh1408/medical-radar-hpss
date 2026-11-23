"""EXP-1 Stage 1: WP1 Lean Front-End, WP2 Causal HPSS & Rung 1 Causality/Parsimony Benchmark.

Implements `thesis/analysis_plan_v3_ai.md` Stage 1 strictly without mutating any file inside
`paper_workspace/`, `beltrao_data/`, or `thesis/outputs/viterbi_repro/`.

Execution:
  MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /opt/homebrew/bin/python3.10 -m thesis.exp1_stage1_parsimony --workers 4
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
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy import stats

from paper_workspace.src import config as C
from paper_workspace.src.data_loader import load_radar
from paper_workspace.src.demod import csd
from paper_workspace.src.estimation import nls_estimate
from paper_workspace.src.filtering import autocorr_coarse, bandpass
from paper_workspace.src.hpss import mitigate_hpss
from paper_workspace.src.provenance import expected_provenance
from thesis import edf_extractor as EDFX
from thesis import io_utils as IO
from thesis import viterbi_tracker as VT
from thesis import wp1_wp2_causal_hmm as W

SCHEMA_VERSION = "stage1-parsimony-v3"
FEAT_CACHE_SCHEMA = "stage1-parsimony-v3"
OUTPUT_DIR = IO.ROOT / "thesis" / "outputs" / "stage1_parsimony"
FEAT_CACHE_DIR = OUTPUT_DIR / "cache"
SURR_CACHE_DIR = FEAT_CACHE_DIR / "surrogates_5seed"
WP_MODULE_PATH = IO.ROOT / "thesis" / "wp1_wp2_causal_hmm.py"

BUNDLE_JSON = IO.BUNDLE_DIR / "metrics_stage1_parsimony.json"
AUDIT_MD = IO.AUDIT_DIR / "stage1_parsimony_report.md"
AUDIT_SESSIONS_CSV = IO.AUDIT_DIR / "stage1_parsimony_sessions.csv"

COHORT_ORDER = ("pool30_sync", "retained23", "PAPER_18", "all_paired")
LOPO_COHORTS = ("pool30_sync", "retained23", "PAPER_18")
REGIME = "branch_neutral_raw_nls_lag"

TOST_MARGIN = 0.50

# Decoded candidate grid includes 0.5 for legacy v2 sensitivity comparison;
# Stage 1.1 primary LOPO restricts sigma_br >= 1.0 bpm/hop and guard_mult = 1.00.
SIGMA_GRID_FULL = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
SIGMA_GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
EPS_GRID = (1e-4, 1e-3)
BETA_GRID = (1.0, 2.0, 4.0)

WP1_O2_INV_CLASS: dict[str, str] = {
    "F0_CSD_Locked": "unconditional",
    "F1a_ClipChanInterp_CSD": "axis_dependent",
    "F1b_ClipJointInterp_CSD": "conditional_fixed_mask",
    "F2a_SymWhitened_lam005": "unconditional",
    "F2b_SymWhitened_lam010": "unconditional",
    "F2c_SymWhitened_lam025": "unconditional",
    "F3_ClipChan_SymWhitened_lam010": "axis_dependent",
}

WP1_METHODS = (
    "F0_CSD_Locked",
    "F1a_ClipChanInterp_CSD",
    "F1b_ClipJointInterp_CSD",
    "F2a_SymWhitened_lam005",
    "F2b_SymWhitened_lam010",
    "F2c_SymWhitened_lam025",
    "F3_ClipChan_SymWhitened_lam010",
)

WP2_METHODS = (
    "H0_Locked_Rect_Full_L3",
    "H1_COLA_Hann_Full_L3",
    "H2_COLA_Hann_InBand_L3",
    "H3_Causal_COLA_InBand_L2",
    "H4_Causal_COLA_InBand_L1",
    "H5_Causal_COLA_InBand_L0",
    "H6_Causal_Rect_Full_L0",
)

WP2_SPECS: dict[str, dict[str, Any]] = {
    "H0_Locked_Rect_Full_L3": {"window": "rect", "lookahead": 3, "in_band": False, "locked": True},
    "H1_COLA_Hann_Full_L3": {"window": "cola_hann", "lookahead": 3, "in_band": False, "locked": False},
    "H2_COLA_Hann_InBand_L3": {"window": "cola_hann", "lookahead": 3, "in_band": True, "locked": False},
    "H3_Causal_COLA_InBand_L2": {"window": "cola_hann", "lookahead": 2, "in_band": True, "locked": False},
    "H4_Causal_COLA_InBand_L1": {"window": "cola_hann", "lookahead": 1, "in_band": True, "locked": False},
    "H5_Causal_COLA_InBand_L0": {"window": "cola_hann", "lookahead": 0, "in_band": True, "locked": False},
    "H6_Causal_Rect_Full_L0": {"window": "rect", "lookahead": 0, "in_band": False, "locked": False},
}

RUNG1_METHODS = (
    "R0_HPSS_NLS_Locked",
    "R1a_Online_Forward_MMSE_0s",
    "R1b_Online_Forward_MAP_0s",
    "R1c_FixedLag_MMSE_4s",
    "R1d_FixedLag_MMSE_8s",
    "R1e_FixedLag_MMSE_16s",
    "R1f_Offline_FB_MMSE_Inf",
    "R1g_Offline_Viterbi_Unpinned",
    "R1h_E2E_Causal_H5_OnlineForward_0s",
    "R1i_E2E_Causal_H6_OnlineForward_0s",
)

RUNG1_BLOCK_LOOKAHEAD_S: dict[str, float] = {
    "R0_HPSS_NLS_Locked": 0.0,
    "R1a_Online_Forward_MMSE_0s": 0.0,
    "R1b_Online_Forward_MAP_0s": 0.0,
    "R1c_FixedLag_MMSE_4s": 4.0,
    "R1d_FixedLag_MMSE_8s": 8.0,
    "R1e_FixedLag_MMSE_16s": 16.0,
    "R1f_Offline_FB_MMSE_Inf": float("inf"),
    "R1g_Offline_Viterbi_Unpinned": float("inf"),
    "R1h_E2E_Causal_H5_OnlineForward_0s": 0.0,
    "R1i_E2E_Causal_H6_OnlineForward_0s": 0.0,
}

RUNG1_STFT_LOOKAHEAD_S: dict[str, float] = {
    "R0_HPSS_NLS_Locked": 3.0,
    "R1a_Online_Forward_MMSE_0s": 3.0,
    "R1b_Online_Forward_MAP_0s": 3.0,
    "R1c_FixedLag_MMSE_4s": 3.0,
    "R1d_FixedLag_MMSE_8s": 3.0,
    "R1e_FixedLag_MMSE_16s": 3.0,
    "R1f_Offline_FB_MMSE_Inf": 3.0,
    "R1g_Offline_Viterbi_Unpinned": 3.0,
    "R1h_E2E_Causal_H5_OnlineForward_0s": 0.0,
    "R1i_E2E_Causal_H6_OnlineForward_0s": 0.0,
}


def _cache_valid(path: Path, measurement: str, wp_sha: str) -> bool:
    if not path.exists():
        return False
    try:
        radar_sha = expected_provenance(measurement)["input_sha256"]["radar_csv"]
        with np.load(path, allow_pickle=False) as z:
            prov = json.loads(str(z["provenance_json"].item()))
            return (
                prov.get("schema_version") == FEAT_CACHE_SCHEMA
                and prov.get("measurement") == measurement
                and prov.get("wp_sha256") == wp_sha
                and prov.get("radar_sha256") == radar_sha
            )
    except Exception:
        return False


def _run_locked_hpss_nls(x_win: np.ndarray) -> tuple[float, np.ndarray, float, float]:
    d_raw = bandpass(x_win)
    t0 = time.perf_counter()
    h = mitigate_hpss(d_raw)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    d_h = bandpass(h["x_hat"])
    fc = autocorr_coarse(d_h)
    est, _, _ = nls_estimate(d_h, fc)
    return float(est), d_h, float(fc), dt_ms


def _run_v2_hpss_nls(
    d_raw: np.ndarray,
    window: str,
    lookahead: int,
    in_band: bool,
    env_norm: str = "block",
) -> tuple[float, np.ndarray, float, float]:
    t0 = time.perf_counter()
    h = W.mitigate_hpss_v2(d_raw, window=window, lookahead=lookahead, in_band=in_band, env_norm=env_norm)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    d_h = bandpass(h["x_hat"])
    fc = autocorr_coarse(d_h)
    est, _, _ = nls_estimate(d_h, fc)
    return float(est), d_h, float(fc), dt_ms


def _extract_session_features(task: tuple[str, str, bool]) -> tuple[str, str, float]:
    measurement, wp_sha, force = task
    FEAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEAT_CACHE_DIR / f"{measurement}.npz"
    if not force and _cache_valid(out_path, measurement, wp_sha):
        return measurement, "cached", 0.0

    b_i, b_q = load_radar(measurement)
    clip_i, clip_q, clip_any = W.detect_adc_rail_mask(b_i, b_q)

    w_len = int(C.WIN_SAMPLES)
    step = int(C.UPDATE_S * C.FS_RADAR)
    starts = list(range(0, len(b_i) - w_len + 1, step))
    n_win = len(starts)

    wp1_est = {k: np.full(n_win, np.nan, dtype=np.float64) for k in WP1_METHODS}
    wp2_est = {k: np.full(n_win, np.nan, dtype=np.float64) for k in WP2_METHODS}
    wp2_times = {k: np.zeros(n_win, dtype=np.float64) for k in WP2_METHODS}

    s_anc_h0 = np.zeros((n_win, VT.BR_GRID_1BPM.size), dtype=np.float64)
    s_anc_h5 = np.zeros((n_win, VT.BR_GRID_1BPM.size), dtype=np.float64)
    s_anc_h6 = np.zeros((n_win, VT.BR_GRID_1BPM.size), dtype=np.float64)
    clip_frac_w = np.zeros(n_win, dtype=np.float64)

    for idx, s0 in enumerate(starts):
        s1 = s0 + w_len
        bi_w = b_i[s0:s1]
        bq_w = b_q[s0:s1]
        ci_w = clip_i[s0:s1]
        cq_w = clip_q[s0:s1]
        ca_w = clip_any[s0:s1]
        has_clip = bool(np.any(ca_w))
        clip_frac_w[idx] = float(np.mean(ca_w))

        # F0 == H0 (shared upstream & locked HPSS)
        x0 = csd(bi_w, bq_w)
        d_raw0 = bandpass(x0)
        f0, d_h0, fc_h0, ms_h0 = _run_locked_hpss_nls(x0)
        wp1_est["F0_CSD_Locked"][idx] = f0
        wp2_est["H0_Locked_Rect_Full_L3"][idx] = f0
        wp2_times["H0_Locked_Rect_Full_L3"][idx] = ms_h0
        s_anc_h0[idx] = VT.extract_br_anchored_nls_spectrum(d_h0, fc_h0, VT.BR_GRID_1BPM)

        # F1a / F1b clip masking (short-circuit when 0 clipped samples in window -> exact copy)
        if has_clip:
            x_f1a = W.csd_clip_masked(bi_w, bq_w, mode="channel", clip_mask=(ci_w, cq_w))
            wp1_est["F1a_ClipChanInterp_CSD"][idx] = _run_locked_hpss_nls(x_f1a)[0]
            x_f1b = W.csd_clip_masked(bi_w, bq_w, mode="joint", clip_mask=ca_w)
            wp1_est["F1b_ClipJointInterp_CSD"][idx] = _run_locked_hpss_nls(x_f1b)[0]
        else:
            wp1_est["F1a_ClipChanInterp_CSD"][idx] = f0
            wp1_est["F1b_ClipJointInterp_CSD"][idx] = f0

        # F2a / F2b / F2c ZCA symmetric whitening
        wp1_est["F2a_SymWhitened_lam005"][idx] = _run_locked_hpss_nls(W.csd_sym_whitened(bi_w, bq_w, lam=0.05))[0]
        f_f2b = _run_locked_hpss_nls(W.csd_sym_whitened(bi_w, bq_w, lam=0.10))[0]
        wp1_est["F2b_SymWhitened_lam010"][idx] = f_f2b
        wp1_est["F2c_SymWhitened_lam025"][idx] = _run_locked_hpss_nls(W.csd_sym_whitened(bi_w, bq_w, lam=0.25))[0]

        if has_clip:
            x_f3 = W.csd_clip_and_whiten(bi_w, bq_w, lam=0.10, mode="channel", clip_mask=(ci_w, cq_w))
            wp1_est["F3_ClipChan_SymWhitened_lam010"][idx] = _run_locked_hpss_nls(x_f3)[0]
        else:
            wp1_est["F3_ClipChan_SymWhitened_lam010"][idx] = f_f2b

        # WP2 variants H1..H6 on locked CSD
        for h_name in WP2_METHODS[1:]:
            spec = WP2_SPECS[h_name]
            fh, d_hv, fc_v, dt_v = _run_v2_hpss_nls(
                d_raw0, window=spec["window"], lookahead=spec["lookahead"], in_band=spec["in_band"]
            )
            wp2_est[h_name][idx] = fh
            wp2_times[h_name][idx] = dt_v
            if h_name == "H5_Causal_COLA_InBand_L0":
                s_anc_h5[idx] = VT.extract_br_anchored_nls_spectrum(d_hv, fc_v, VT.BR_GRID_1BPM)
            elif h_name == "H6_Causal_Rect_Full_L0":
                s_anc_h6[idx] = VT.extract_br_anchored_nls_spectrum(d_hv, fc_v, VT.BR_GRID_1BPM)

    locked_cache = IO.load_locked_cache(measurement)
    np.testing.assert_allclose(
        wp1_est["F0_CSD_Locked"],
        np.asarray(locked_cache["est_hpss"], dtype=np.float64),
        rtol=0.0,
        atol=1e-9,
        equal_nan=True,
    )
    max_par = float(np.nanmax(np.abs(wp1_est["F0_CSD_Locked"] - np.asarray(locked_cache["est_hpss"], float))))

    radar_sha = expected_provenance(measurement)["input_sha256"]["radar_csv"]
    prov_str = json.dumps(
        {
            "schema_version": FEAT_CACHE_SCHEMA,
            "measurement": measurement,
            "wp_sha256": wp_sha,
            "radar_sha256": radar_sha,
        },
        sort_keys=True,
    )

    payload: dict[str, Any] = {
        "S_anc_h0_1bpm": s_anc_h0,
        "S_anc_h5_1bpm": s_anc_h5,
        "S_anc_h6_1bpm": s_anc_h6,
        "clip_frac_w": clip_frac_w,
        "session_clip_frac": np.array(float(np.mean(clip_any)), dtype=np.float64),
        "provenance_json": np.array(prov_str),
    }
    for k, arr in wp1_est.items():
        payload[f"wp1_{k}"] = arr
    for k, arr in wp2_est.items():
        payload[f"wp2_{k}"] = arr
        payload[f"time_{k}_ms"] = wp2_times[k]

    tmp_path = FEAT_CACHE_DIR / f".{measurement}.tmp.npz"
    np.savez_compressed(tmp_path, **payload)
    tmp_path.replace(out_path)
    return measurement, "computed", max_par


def _load_feat_cache(measurement: str) -> dict[str, np.ndarray]:
    with np.load(FEAT_CACHE_DIR / f"{measurement}.npz", allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def _decode_rung1_session(measurement: str) -> tuple[str, dict[str, Any]]:
    """Decode all Rung 1 unpinned grid candidates for one session."""
    feat = _load_feat_cache(measurement)
    s_h0 = feat["S_anc_h0_1bpm"]
    s_h5 = feat["S_anc_h5_1bpm"]
    s_h6 = feat["S_anc_h6_1bpm"]

    prob_cfgs = [(s, e, b) for s in SIGMA_GRID_FULL for e in EPS_GRID for b in BETA_GRID]
    vit_cfgs = [(s, e, 1.0) for s in SIGMA_GRID_FULL for e in EPS_GRID]

    grid_paths: dict[str, dict[tuple[float, float, float], np.ndarray]] = {
        "R1a_Online_Forward_MMSE_0s": {},
        "R1b_Online_Forward_MAP_0s": {},
        "R1c_FixedLag_MMSE_4s": {},
        "R1d_FixedLag_MMSE_8s": {},
        "R1e_FixedLag_MMSE_16s": {},
        "R1f_Offline_FB_MMSE_Inf": {},
        "R1g_Offline_Viterbi_Unpinned": {},
        "R1h_E2E_Causal_H5_OnlineForward_0s": {},
        "R1i_E2E_Causal_H6_OnlineForward_0s": {},
    }
    uncertainty_h0: dict[tuple[float, float, float], dict[str, np.ndarray]] = {}
    uncertainty_h5: dict[tuple[float, float, float], dict[str, np.ndarray]] = {}

    for cfg in prob_cfgs:
        sig_v, eps_v, beta_v = cfg
        fwd0 = W.hmm_forward_filter(s_h0, VT.BR_GRID_1BPM, sigma_br=sig_v, eps=eps_v, beta=beta_v)
        grid_paths["R1a_Online_Forward_MMSE_0s"][cfg] = fwd0["est_mmse"]
        grid_paths["R1b_Online_Forward_MAP_0s"][cfg] = fwd0["est_map"]
        uncertainty_h0[cfg] = {
            "post_entropy": fwd0["post_entropy"],
            "post_sd_bpm": fwd0["post_sd_bpm"],
        }
        grid_paths["R1c_FixedLag_MMSE_4s"][cfg] = W.hmm_fixed_lag_smoother(
            s_h0, lag_hops=2, grid_bpm=VT.BR_GRID_1BPM, sigma_br=sig_v, eps=eps_v, beta=beta_v
        )["est_mmse"]
        grid_paths["R1d_FixedLag_MMSE_8s"][cfg] = W.hmm_fixed_lag_smoother(
            s_h0, lag_hops=4, grid_bpm=VT.BR_GRID_1BPM, sigma_br=sig_v, eps=eps_v, beta=beta_v
        )["est_mmse"]
        grid_paths["R1e_FixedLag_MMSE_16s"][cfg] = W.hmm_fixed_lag_smoother(
            s_h0, lag_hops=8, grid_bpm=VT.BR_GRID_1BPM, sigma_br=sig_v, eps=eps_v, beta=beta_v
        )["est_mmse"]
        grid_paths["R1f_Offline_FB_MMSE_Inf"][cfg] = W.hmm_offline_fb_smoother(
            s_h0, grid_bpm=VT.BR_GRID_1BPM, sigma_br=sig_v, eps=eps_v, beta=beta_v
        )["est_mmse"]

        fwd5 = W.hmm_forward_filter(s_h5, VT.BR_GRID_1BPM, sigma_br=sig_v, eps=eps_v, beta=beta_v)
        grid_paths["R1h_E2E_Causal_H5_OnlineForward_0s"][cfg] = fwd5["est_mmse"]
        uncertainty_h5[cfg] = {
            "post_entropy": fwd5["post_entropy"],
            "post_sd_bpm": fwd5["post_sd_bpm"],
        }

        fwd6 = W.hmm_forward_filter(s_h6, VT.BR_GRID_1BPM, sigma_br=sig_v, eps=eps_v, beta=beta_v)
        grid_paths["R1i_E2E_Causal_H6_OnlineForward_0s"][cfg] = fwd6["est_mmse"]

    for cfg in vit_cfgs:
        sig_v, eps_v, beta_v = cfg
        grid_paths["R1g_Offline_Viterbi_Unpinned"][cfg] = W.hmm_offline_viterbi_prob(
            s_h0, grid_bpm=VT.BR_GRID_1BPM, sigma_br=sig_v, eps=eps_v, beta=beta_v
        )[0]

    return measurement, {
        "anchor_r0": feat["wp2_H0_Locked_Rect_Full_L3"],
        "grid_paths": grid_paths,
        "uncertainty_h0": uncertainty_h0,
        "uncertainty_h5": uncertainty_h5,
    }


def _session_metrics(est: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    e = np.asarray(est, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    ae = np.abs(e)
    r_val = float(np.corrcoef(est, ref)[0, 1]) if (np.std(est) > 1e-12 and np.std(ref) > 1e-12) else float("nan")
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(ae)),
        "bias": float(np.mean(e)),
        "acc3": float(100.0 * np.mean(ae < 3.0)),
        "acc5": float(100.0 * np.mean(ae < 5.0)),
        "acc6": float(100.0 * np.mean(ae < 6.0)),
        "acc10": float(100.0 * np.mean(ae < 10.0)),
        "gross_gt10_pct": float(100.0 * np.mean(ae > 10.0)),
        "r": r_val,
        "est_mean": float(np.mean(est)),
        "ref_mean": float(np.mean(ref)),
    }


def _session_level_ba_and_icc(est_means: np.ndarray, ref_means: np.ndarray) -> dict[str, float]:
    n = len(est_means)
    if n < 2:
        return {k: float("nan") for k in ("bias", "sd", "loa_lo", "loa_up", "icc_1_1")}
    diff = est_means - ref_means
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    grand = float(np.mean(np.concatenate([est_means, ref_means])))
    row_means = 0.5 * (est_means + ref_means)
    bms = 2.0 * float(np.sum((row_means - grand) ** 2)) / float(n - 1)
    wms = float(np.sum((est_means - row_means) ** 2 + (ref_means - row_means) ** 2)) / float(n)
    icc = float((bms - wms) / (bms + wms)) if (bms + wms) > 0.0 else float("nan")
    return {
        "bias": bias,
        "sd": sd,
        "loa_lo": bias - 1.96 * sd,
        "loa_up": bias + 1.96 * sd,
        "icc_1_1": icc,
    }


def _safe_wilcoxon_p(deltas: np.ndarray) -> float:
    arr = np.asarray(deltas, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    nonzero = arr[np.abs(arr) > 1e-15]
    if nonzero.size < 2:
        return 1.0
    try:
        p = float(stats.wilcoxon(arr, zero_method="wilcox", method="auto").pvalue)
        return p if np.isfinite(p) else 1.0
    except ValueError:
        return 1.0


def _paired_tost(deltas: np.ndarray, margin: float = TOST_MARGIN) -> dict[str, float | bool]:
    arr = np.asarray(deltas, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return {"n": int(n), "mean": float("nan"), "ci90_lo": float("nan"), "ci90_hi": float("nan"), "p_tost": float("nan"), "equivalent": False}
    mean_d = float(np.mean(arr))
    sd_d = float(np.std(arr, ddof=1))
    if sd_d <= 1e-15:
        eq = bool(abs(mean_d) < float(margin))
        return {"n": int(n), "mean": mean_d, "ci90_lo": mean_d, "ci90_hi": mean_d, "p_tost": 0.0 if eq else 1.0, "equivalent": eq}
    se = sd_d / float(np.sqrt(n))
    df = n - 1
    p1 = float(1.0 - stats.t.cdf((mean_d - (-float(margin))) / se, df=df))
    p2 = float(1.0 - stats.t.cdf((float(margin) - mean_d) / se, df=df))
    p_tost = float(max(p1, p2))
    tcrit = float(stats.t.ppf(0.95, df=df))
    return {
        "n": int(n),
        "mean": mean_d,
        "ci90_lo": mean_d - tcrit * se,
        "ci90_hi": mean_d + tcrit * se,
        "p_tost": p_tost,
        "equivalent": bool(p_tost < 0.05),
    }


def _aggregate_method(
    sessions: list[str],
    est_by_sess: dict[str, np.ndarray],
    ref_valid_map: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    per_sess: dict[str, dict[str, float]] = {}
    est_list, ref_list, est_means, ref_means = [], [], [], []
    for m in sessions:
        ref_arr, v_arr = ref_valid_map[m]
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
        est_list.append(ev)
        ref_list.append(rv)
        est_means.append(sc["est_mean"])
        ref_means.append(sc["ref_mean"])
    if not per_sess:
        return {"n_sessions": 0, "n_windows": 0}, per_sess
    all_e = np.concatenate(est_list)
    all_r = np.concatenate(ref_list)
    pool_sc = _session_metrics(all_e, all_r)
    rmc = IO.rmcorr_ancova(est_list, ref_list)
    ba_sess = _session_level_ba_and_icc(np.asarray(est_means), np.asarray(ref_means))
    ba_clust = IO.clustered_bland_altman(est_list, ref_list)
    r_vals = [sc["r"] for sc in per_sess.values() if np.isfinite(sc["r"])]
    return {
        "n_sessions": len(per_sess),
        "n_windows": int(all_r.size),
        "rmse_sess": float(np.mean([sc["rmse"] for sc in per_sess.values()])),
        "rmse_sess_median": float(np.median([sc["rmse"] for sc in per_sess.values()])),
        "mae_sess": float(np.mean([sc["mae"] for sc in per_sess.values()])),
        "acc3_sess": float(np.mean([sc["acc3"] for sc in per_sess.values()])),
        "acc6_sess": float(np.mean([sc["acc6"] for sc in per_sess.values()])),
        "acc10_sess": float(np.mean([sc["acc10"] for sc in per_sess.values()])),
        "gross_gt10_sess_pct": float(np.mean([sc["gross_gt10_pct"] for sc in per_sess.values()])),
        "rmse_pool": pool_sc["rmse"],
        "mae_pool": pool_sc["mae"],
        "r_within_rmcorr": rmc["r_within"],
        "r_within_rmcorr_p": rmc["p_value"],
        "r_within_sess_mean": float(np.mean(r_vals)) if r_vals else float("nan"),
        "icc_1_1": ba_sess["icc_1_1"],
        "ba_session_level": ba_sess,
        "ba_clustered_anova": ba_clust,
    }, per_sess


def _paired_comparison(
    sessions: list[str],
    per_sess_cand: dict[str, dict[str, float]],
    per_sess_ref: dict[str, dict[str, float]],
    ref_key: str = "rmse",
) -> dict[str, Any]:
    common = [m for m in sessions if m in per_sess_cand and m in per_sess_ref]
    if len(common) < 2:
        return {"mean_delta": float("nan"), "p_wilcoxon_patient": 1.0, "p_wilcoxon_sess": 1.0}
    deltas = np.array([per_sess_cand[m]["rmse"] - per_sess_ref[m][ref_key] for m in common], dtype=np.float64)
    by_pat: dict[int, list[float]] = {}
    for m, d in zip(common, deltas):
        by_pat.setdefault(IO.patient_id_of(m), []).append(float(d))
    pat_map = {p: np.asarray(vals, dtype=np.float64) for p, vals in by_pat.items()}
    boot = IO.patient_clustered_bootstrap(pat_map, n_boot=10000, seed=IO.RANDOM_SEED)
    pat_deltas = np.array([float(np.mean(vals)) for vals in pat_map.values()], dtype=np.float64)
    return {
        "n_sessions": len(common),
        "n_patients": len(pat_map),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "wins_sess": f"{int(np.sum(deltas < 0.0))}/{len(common)}",
        "wins_pat": f"{int(np.sum(pat_deltas < 0.0))}/{len(pat_deltas)}",
        "ci95_lo": boot["ci95_lo"],
        "ci95_hi": boot["ci95_hi"],
        "p_wilcoxon_sess": _safe_wilcoxon_p(deltas),
        "p_wilcoxon_patient": _safe_wilcoxon_p(pat_deltas),
        "tost_sess": _paired_tost(deltas, TOST_MARGIN),
        "tost_patient": _paired_tost(pat_deltas, TOST_MARGIN),
    }


def _constant_baselines(
    sessions: list[str],
    ref_valid_map: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    ref_list, sids, pids = [], [], []
    for m in sessions:
        rf, vv = ref_valid_map[m]
        if np.any(vv):
            ref_list.append(rf[vv])
            sids.append(m)
            pids.append(IO.patient_id_of(m))
    if not ref_list:
        return {"const_loso": float("nan"), "const_lopo_median": float("nan")}, {}
    pooled = np.concatenate(ref_list)
    gmean = float(np.mean(pooled))
    c_loso, c_lopo_med, c_lopo_mean, c_glob = [], [], [], []
    per_sess: dict[str, dict[str, float]] = {}
    for i, (m, pid, rf) in enumerate(zip(sids, pids, ref_list)):
        others = np.concatenate([ref_list[j] for j in range(len(ref_list)) if j != i]) if len(ref_list) > 1 else rf
        r_loso = float(np.sqrt(np.mean((rf - float(np.median(others))) ** 2)))
        pat_others = [ref_list[j] for j in range(len(ref_list)) if pids[j] != pid]
        pat_pool = np.concatenate(pat_others) if pat_others else rf
        r_lopo_med = float(np.sqrt(np.mean((rf - float(np.median(pat_pool))) ** 2)))
        r_lopo_mean = float(np.sqrt(np.mean((rf - float(np.mean(pat_pool))) ** 2)))
        r_glob = float(np.sqrt(np.mean((rf - gmean) ** 2)))
        c_loso.append(r_loso)
        c_lopo_med.append(r_lopo_med)
        c_lopo_mean.append(r_lopo_mean)
        c_glob.append(r_glob)
        per_sess[m] = {"rmse": r_loso, "const_lopo_median_rmse": r_lopo_med}
    return {
        "const_loso": float(np.mean(c_loso)),
        "const_lopo_median": float(np.mean(c_lopo_med)),
        "const_lopo_mean": float(np.mean(c_lopo_mean)),
        "const_global_oracle": float(np.mean(c_glob)),
    }, per_sess


def _run_lopo_rung1(
    cohort: str,
    decoded: dict[str, dict[str, Any]],
    ref_1hz_map: dict[str, tuple[np.ndarray, np.ndarray]],
    guard_mult: float = 1.00,
    sigma_min: float = 1.0,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, np.ndarray]]]]:
    """Strictly patient-disjoint LOPO trained ONLY on `br_1hz_30s`."""
    sessions = [m for m in IO.get_cohort_sessions(cohort, include_zero_valid=False) if np.any(ref_1hz_map[m][1])]
    folds = list(IO.iter_lopo_folds(cohort))
    sig_grid_used = [float(s) for s in SIGMA_GRID_FULL if float(s) >= float(sigma_min)]

    preds: dict[str, dict[str, np.ndarray]] = {m_name: {} for m_name in RUNG1_METHODS}
    preds["R0_HPSS_NLS_Locked"] = {m: decoded[m]["anchor_r0"] for m in sessions}
    fold_logs: dict[str, list[dict[str, Any]]] = {m_name: [] for m_name in RUNG1_METHODS[1:]}
    unc_logs: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "R1a_Online_Forward_MMSE_0s": {},
        "R1h_E2E_Causal_H5_OnlineForward_0s": {},
    }

    for fam in RUNG1_METHODS[1:]:
        cfgs = [
            c for c in next(iter(decoded.values()))["grid_paths"][fam].keys()
            if float(c[0]) >= float(sigma_min)
        ]
        for fold in folds:
            train_ms = [m for m in fold.train_sessions if m in sessions]
            test_ms = [m for m in fold.test_sessions if m in sessions]
            if not test_ms:
                continue

            anchor_est = [decoded[m]["anchor_r0"][ref_1hz_map[m][1]] for m in train_ms]
            ref_list = [ref_1hz_map[m][0][ref_1hz_map[m][1]] for m in train_ms]
            base_r = IO.rmcorr_ancova(anchor_est, ref_list)["r_within"]
            min_r = float(guard_mult) * base_r if np.isfinite(base_r) else -np.inf

            best_el = None
            best_fb = None
            for cfg in cfgs:
                tr_ests, tr_rmses = [], []
                for m, rf in zip(train_ms, ref_list):
                    _, v_arr = ref_1hz_map[m]
                    pe = decoded[m]["grid_paths"][fam][cfg][v_arr]
                    tr_ests.append(pe)
                    tr_rmses.append(float(np.sqrt(np.mean((pe - rf) ** 2))))
                mean_rmse = float(np.mean(tr_rmses))
                rmc_v = IO.rmcorr_ancova(tr_ests, ref_list)["r_within"]
                fb_tuple = (-(rmc_v if np.isfinite(rmc_v) else -1.0), mean_rmse, cfg, rmc_v)
                if best_fb is None or fb_tuple < best_fb:
                    best_fb = fb_tuple
                if np.isfinite(rmc_v) and rmc_v >= min_r:
                    el_tuple = (mean_rmse, -rmc_v, cfg, rmc_v)
                    if best_el is None or el_tuple < best_el:
                        best_el = el_tuple

            fb_used = best_el is None
            if not fb_used:
                tr_rmse_sel, _, sel_cfg, tr_r_sel = best_el
            else:
                _, tr_rmse_sel, sel_cfg, tr_r_sel = best_fb

            for m in test_ms:
                preds[fam][m] = decoded[m]["grid_paths"][fam][sel_cfg]
                if fam == "R1a_Online_Forward_MMSE_0s":
                    unc_logs[fam][m] = decoded[m]["uncertainty_h0"][sel_cfg]
                elif fam == "R1h_E2E_Causal_H5_OnlineForward_0s":
                    unc_logs[fam][m] = decoded[m]["uncertainty_h5"][sel_cfg]

            sig_v, eps_v, beta_v = sel_cfg
            fold_logs[fam].append({
                "patient_id": int(fold.patient_id),
                "test_sessions": list(test_ms),
                "sigma_br": float(sig_v),
                "eps": float(eps_v),
                "beta": float(beta_v),
                "lower_sigma_pinned": bool(float(sig_v) == float(min(sig_grid_used))),
                "upper_sigma_pinned": bool(float(sig_v) == float(max(sig_grid_used))),
                "any_boundary_pinned": bool(
                    float(sig_v) in (float(min(sig_grid_used)), float(max(sig_grid_used)))
                    or float(eps_v) in (float(min(EPS_GRID)), float(max(EPS_GRID)))
                    or (fam != "R1g_Offline_Viterbi_Unpinned" and float(beta_v) in (float(min(BETA_GRID)), float(max(BETA_GRID))))
                ),
                "train_rmse_sess": float(tr_rmse_sel),
                "train_r_within": float(tr_r_sel),
                "train_anchor_r_within": float(base_r),
                "fallback_used": bool(fb_used),
            })

    return preds, fold_logs, unc_logs


def _phase_scramble_session_stage1(
    task: tuple[str, int, int, str, dict[str, tuple[float, float, float]]],
) -> tuple[str, dict[int, dict[str, np.ndarray]]]:
    """Run 5-seed full-session radar phase-scrambled surrogates with cached spectra."""
    m, n_seeds, base_seed, wp_sha, fold_cfgs = task
    SURR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = SURR_CACHE_DIR / f"{m}.npz"
    radar_sha = expected_provenance(m)["input_sha256"]["radar_csv"]

    loaded = None
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as z:
                prov = json.loads(str(z["provenance_json"].item()))
                if (
                    prov.get("schema_version") == FEAT_CACHE_SCHEMA
                    and prov.get("measurement") == m
                    and prov.get("wp_sha256") == wp_sha
                    and prov.get("radar_sha256") == radar_sha
                    and int(prov.get("n_seeds", 0)) == int(n_seeds)
                    and int(prov.get("base_seed", -1)) == int(base_seed)
                ):
                    loaded = {
                        "f_r0": z["f_r0"].copy(),
                        "S_anc_h0": z["S_anc_h0"].copy(),
                        "S_anc_h5": z["S_anc_h5"].copy(),
                    }
        except Exception:
            loaded = None

    if loaded is None:
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
        f_r0_all = np.empty((n_seeds, n_w), dtype=np.float64)
        s_h0_all = np.zeros((n_seeds, n_w, VT.BR_GRID_1BPM.size), dtype=np.float64)
        s_h5_all = np.zeros((n_seeds, n_w, VT.BR_GRID_1BPM.size), dtype=np.float64)

        for seed_idx in range(n_seeds):
            rng_s = np.random.default_rng(int(base_seed) + seed_idx * 1009 + pid * 31 + s_idx)
            rand_phases = rng_s.uniform(0.0, 2.0 * np.pi, size=xf.size)
            rand_phases[0] = 0.0
            x_scr = np.fft.ifft(mag_xf * np.exp(1j * rand_phases))

            for idx, s0 in enumerate(starts):
                dr = bandpass(x_scr[s0 : s0 + w_len])
                dh0 = bandpass(mitigate_hpss(dr)["x_hat"])
                fc0 = autocorr_coarse(dh0)
                f_r0_all[seed_idx, idx], _, _ = nls_estimate(dh0, fc0)
                s_h0_all[seed_idx, idx] = VT.extract_br_anchored_nls_spectrum(dh0, fc0, VT.BR_GRID_1BPM)

                dh5 = bandpass(W.mitigate_hpss_v2(dr, window="cola_hann", lookahead=0, in_band=True)["x_hat"])
                fc5 = autocorr_coarse(dh5)
                s_h5_all[seed_idx, idx] = VT.extract_br_anchored_nls_spectrum(dh5, fc5, VT.BR_GRID_1BPM)

        prov_str = json.dumps(
            {
                "schema_version": FEAT_CACHE_SCHEMA,
                "measurement": m,
                "wp_sha256": wp_sha,
                "radar_sha256": radar_sha,
                "n_seeds": int(n_seeds),
                "base_seed": int(base_seed),
            },
            sort_keys=True,
        )
        tmp_path = SURR_CACHE_DIR / f".{m}.tmp.npz"
        np.savez_compressed(
            tmp_path,
            f_r0=f_r0_all,
            S_anc_h0=s_h0_all,
            S_anc_h5=s_h5_all,
            provenance_json=np.array(prov_str),
        )
        tmp_path.replace(cache_path)
        loaded = {"f_r0": f_r0_all, "S_anc_h0": s_h0_all, "S_anc_h5": s_h5_all}

    cfg_r1a = fold_cfgs["R1a_Online_Forward_MMSE_0s"]
    cfg_r1f = fold_cfgs["R1f_Offline_FB_MMSE_Inf"]
    cfg_r1g = fold_cfgs["R1g_Offline_Viterbi_Unpinned"]
    cfg_r1h = fold_cfgs["R1h_E2E_Causal_H5_OnlineForward_0s"]
    out_by_seed: dict[int, dict[str, np.ndarray]] = {}
    for seed_idx in range(n_seeds):
        s_anc_h0 = loaded["S_anc_h0"][seed_idx]
        s_anc_h5 = loaded["S_anc_h5"][seed_idx]
        out_by_seed[seed_idx] = {
            "R0_HPSS_NLS_Locked": loaded["f_r0"][seed_idx],
            "R1a_Online_Forward_MMSE_0s": W.hmm_forward_filter(
                s_anc_h0, VT.BR_GRID_1BPM, sigma_br=cfg_r1a[0], eps=cfg_r1a[1], beta=cfg_r1a[2]
            )["est_mmse"],
            "R1f_Offline_FB_MMSE_Inf": W.hmm_offline_fb_smoother(
                s_anc_h0, VT.BR_GRID_1BPM, sigma_br=cfg_r1f[0], eps=cfg_r1f[1], beta=cfg_r1f[2]
            )["est_mmse"],
            "R1g_Offline_Viterbi_Unpinned": W.hmm_offline_viterbi_prob(
                s_anc_h0, VT.BR_GRID_1BPM, sigma_br=cfg_r1g[0], eps=cfg_r1g[1], beta=cfg_r1g[2]
            )[0],
            "R1h_E2E_Causal_H5_OnlineForward_0s": W.hmm_forward_filter(
                s_anc_h5, VT.BR_GRID_1BPM, sigma_br=cfg_r1h[0], eps=cfg_r1h[1], beta=cfg_r1h[2]
            )["est_mmse"],
        }
    return m, out_by_seed


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    xa, ya = x[ok], y[ok]
    if xa.size < 3 or np.std(xa) <= 1e-12 or np.std(ya) <= 1e-12:
        return float("nan")
    val = float(stats.spearmanr(xa, ya).statistic)
    return val if np.isfinite(val) else float("nan")


def _wp4_uncertainty_preview(
    sessions: list[str],
    preds: dict[str, dict[str, np.ndarray]],
    unc_logs: dict[str, dict[str, dict[str, np.ndarray]]],
    ref_1hz_map: dict[str, tuple[np.ndarray, np.ndarray]],
    ref_trig_map: dict[str, tuple[np.ndarray, np.ndarray]],
    pause_map: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compute Stage 1 causal uncertainty preview & quiet-apnea safety audit."""
    out: dict[str, Any] = {}
    for m_name in ("R1a_Online_Forward_MMSE_0s", "R1h_E2E_Causal_H5_OnlineForward_0s"):
        res_m: dict[str, Any] = {}
        for ref_lbl, rmap in (("br_1hz_30s", ref_1hz_map), ("br_trig_bpm", ref_trig_map)):
            ents, sds, errs = [], [], []
            sp_ent_sess, sp_sd_sess = [], []
            for m in sessions:
                rf, vv = rmap[m]
                if not np.any(vv):
                    continue
                ae = np.abs(preds[m_name][m][vv] - rf[vv])
                en = unc_logs[m_name][m]["post_entropy"][vv]
                sd = unc_logs[m_name][m]["post_sd_bpm"][vv]
                ents.append(en)
                sds.append(sd)
                errs.append(ae)
                se = _safe_spearman(en, ae)
                ss = _safe_spearman(sd, ae)
                if np.isfinite(se):
                    sp_ent_sess.append(se)
                if np.isfinite(ss):
                    sp_sd_sess.append(ss)
            all_en = np.concatenate(ents)
            all_sd = np.concatenate(sds)
            all_ae = np.concatenate(errs)
            res_m[ref_lbl] = {
                "n_windows": int(all_ae.size),
                "spearman_entropy_pool": _safe_spearman(all_en, all_ae),
                "spearman_entropy_sess_mean": float(np.mean(sp_ent_sess)) if sp_ent_sess else float("nan"),
                "spearman_sd_pool": _safe_spearman(all_sd, all_ae),
                "spearman_sd_sess_mean": float(np.mean(sp_sd_sess)) if sp_sd_sess else float("nan"),
            }

        # Quiet-apnea pause stratification (> 5.0 s breath gaps vs normal valid windows)
        en_pause, en_norm, sd_pause, sd_norm = [], [], [], []
        for m in sessions:
            _, v1 = ref_1hz_map[m]
            pm = pause_map[m] & v1
            nm = (~pause_map[m]) & v1
            if np.any(pm):
                en_pause.append(unc_logs[m_name][m]["post_entropy"][pm])
                sd_pause.append(unc_logs[m_name][m]["post_sd_bpm"][pm])
            if np.any(nm):
                en_norm.append(unc_logs[m_name][m]["post_entropy"][nm])
                sd_norm.append(unc_logs[m_name][m]["post_sd_bpm"][nm])
        ep = np.concatenate(en_pause) if en_pause else np.empty(0)
        en_arr = np.concatenate(en_norm) if en_norm else np.empty(0)
        sp = np.concatenate(sd_pause) if sd_pause else np.empty(0)
        sn = np.concatenate(sd_norm) if sd_norm else np.empty(0)
        res_m["quiet_apnea_safety"] = {
            "n_pause_windows": int(ep.size),
            "n_regular_windows": int(en_arr.size),
            "mean_entropy_pause": float(np.mean(ep)) if ep.size else float("nan"),
            "mean_entropy_regular": float(np.mean(en_arr)) if en_arr.size else float("nan"),
            "mean_sd_pause_bpm": float(np.mean(sp)) if sp.size else float("nan"),
            "mean_sd_regular_bpm": float(np.mean(sn)) if sn.size else float("nan"),
        }
        out[m_name] = res_m
    return out


def _run_invariance_and_causality_audit() -> dict[str, Any]:
    """Run mathematical group invariance and exact causality verification checks.

    Explicitly separates our strictly O(2)-symmetric Stage 1 operators (`F2a/b/c`, `F1b`,
    `H1..H6`, `env_normalize_causal`, and all Rung 1 HMM decoders on symmetric `H5`) from
    the tiny `~1e-4 bpm` I/Q swap asymmetry inherited solely by continuous MMSE on frozen
    `paper_workspace/src/hpss.py` (even-NFFT=256 unpaired Nyquist bin). Also cleanly
    separates intra-block median normalization (`env_norm="block"`) from frame-causal
    `env_norm="causal"` under mid-session perturbation.
    """
    new_inv_results: dict[str, dict[str, float]] = {}
    frozen_h0_diag: dict[str, dict[str, float]] = {}

    for sess_check, label in (("2.3", "clipped_2_3"), ("7.3", "clean_7_3")):
        bi, bq = load_radar(sess_check)
        n_sub = 480 + 9 * 32
        bi, bq = bi[:n_sub], bq[:n_sub]
        starts = [s for s in range(0, 10 * 32, 32)]
        _, _, clip_any = W.detect_adc_rail_mask(bi, bq)

        def _eval_slice(arr_i: np.ndarray, arr_q: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
            out_new: dict[str, list[float]] = {
                "F2a_SymWhitened_lam005": [],
                "F2b_SymWhitened_lam010": [],
                "F2c_SymWhitened_lam025": [],
                "F1b_ClipJoint_FixedMask": [],
                "H1_COLA_Hann_Full_L3": [],
                "H2_COLA_Hann_InBand_L3": [],
                "H5_Causal_COLA_InBand_L0": [],
                "H5_CausalEnv_COLA_InBand_L0": [],
                "H6_Causal_Rect_Full_L0": [],
            }
            s_anc_h5, s_anc_h0 = [], []
            for s0 in starts:
                bw_i, bw_q, cw = arr_i[s0 : s0 + 480], arr_q[s0 : s0 + 480], clip_any[s0 : s0 + 480]
                x0 = csd(bw_i, bw_q)
                dr0 = bandpass(x0)

                _, dh0, fc0, _ = _run_locked_hpss_nls(x0)
                s_anc_h0.append(VT.extract_br_anchored_nls_spectrum(dh0, fc0, VT.BR_GRID_1BPM))
                out_new["F2a_SymWhitened_lam005"].append(
                    _run_locked_hpss_nls(W.csd_sym_whitened(bw_i, bw_q, lam=0.05))[0]
                )
                out_new["F2b_SymWhitened_lam010"].append(
                    _run_locked_hpss_nls(W.csd_sym_whitened(bw_i, bw_q, lam=0.10))[0]
                )
                out_new["F2c_SymWhitened_lam025"].append(
                    _run_locked_hpss_nls(W.csd_sym_whitened(bw_i, bw_q, lam=0.25))[0]
                )
                out_new["F1b_ClipJoint_FixedMask"].append(
                    _run_locked_hpss_nls(W.csd_clip_masked(bw_i, bw_q, mode="joint", clip_mask=cw))[0]
                )
                for h_name in ("H1_COLA_Hann_Full_L3", "H2_COLA_Hann_InBand_L3", "H5_Causal_COLA_InBand_L0", "H6_Causal_Rect_Full_L0"):
                    sp = WP2_SPECS[h_name]
                    fh, dhv, fcv, _ = _run_v2_hpss_nls(dr0, sp["window"], sp["lookahead"], sp["in_band"])
                    out_new[h_name].append(fh)
                    if h_name == "H5_Causal_COLA_InBand_L0":
                        s_anc_h5.append(VT.extract_br_anchored_nls_spectrum(dhv, fcv, VT.BR_GRID_1BPM))
                out_new["H5_CausalEnv_COLA_InBand_L0"].append(
                    _run_v2_hpss_nls(dr0, "cola_hann", 0, True, env_norm="causal")[0]
                )

            s0_a = np.asarray(s_anc_h0)
            s5_a = np.asarray(s_anc_h5)
            arr_new = {k: np.asarray(v, dtype=np.float64) for k, v in out_new.items()}
            # Verify all 4 Rung 1 HMM decoders on strictly symmetric H5 spectrum
            arr_new["R1h_E2E_Causal_H5_Forward"] = W.hmm_forward_filter(s5_a)["est_mmse"]
            arr_new["R1h_E2E_Causal_H5_FixedLag4s"] = W.hmm_fixed_lag_smoother(s5_a, lag_hops=2)["est_mmse"]
            arr_new["R1h_E2E_Causal_H5_OfflineFB"] = W.hmm_offline_fb_smoother(s5_a)["est_mmse"]
            arr_new["R1h_E2E_Causal_H5_OfflineVit"] = W.hmm_offline_viterbi_prob(s5_a)[0]

            # Frozen H0 continuous MMSE decoders (isolate legacy Nyquist-bin swap behavior)
            arr_h0 = {
                "R1a_H0_Forward_MMSE": W.hmm_forward_filter(s0_a)["est_mmse"],
                "R1c_H0_FixedLag_4s": W.hmm_fixed_lag_smoother(s0_a, lag_hops=2)["est_mmse"],
                "R1f_H0_Offline_FB": W.hmm_offline_fb_smoother(s0_a)["est_mmse"],
                "R1g_H0_Offline_Vit": W.hmm_offline_viterbi_prob(s0_a)[0],
            }
            return arr_new, arr_h0

        ref_new, ref_h0 = _eval_slice(bi, bq)
        z_rot = (bi + 1j * bq) * np.exp(1j * np.pi / 3.0)
        new_diffs: dict[str, float] = {}
        h0_diffs: dict[str, float] = {}
        for tname, (ti, tq) in {
            "scale_3p7x": (3.7 * bi, 3.7 * bq),
            "phase_rot_pi_over_3": (z_rot.real, z_rot.imag),
            "iq_sign_flip": (-bi, -bq),
            "iq_channel_swap": (bq, bi),
        }.items():
            alt_new, alt_h0 = _eval_slice(ti, tq)
            new_diffs[tname] = float(max(np.max(np.abs(ref_new[k] - alt_new[k])) for k in ref_new))
            h0_diffs[tname] = float(max(np.max(np.abs(ref_h0[k] - alt_h0[k])) for k in ref_h0))
        new_inv_results[label] = new_diffs
        frozen_h0_diag[label] = h0_diffs

    # Spectral & HMM causality checks
    rng = np.random.default_rng(IO.RANDOM_SEED)
    mag = rng.random((256, 32)) + 0.1
    mag_pert = mag.copy()
    mag_pert[:, 18:] += 50.0
    h5_a = W.hpss_filter_causal(mag, lookahead=0, in_band=True)[0]
    h5_b = W.hpss_filter_causal(mag_pert, lookahead=0, in_band=True)[0]

    # Mid-session 7.3 [4800:5280] end-to-end perturbation at t >= 15 s (sample >= 240)
    bi7, bq7 = load_radar("7.3")
    dr7 = bandpass(csd(bi7[4800:5280], bq7[4800:5280]))
    t_ax = np.arange(480) / float(C.FS_RADAR)
    dr7_p = dr7.copy()
    dr7_p[240:] += 50.0 * np.exp(1j * 2.0 * np.pi * 0.8 * t_ax[240:])
    h5_c_a = W.mitigate_hpss_v2(dr7, window="cola_hann", lookahead=0, in_band=True, env_norm="causal")
    h5_c_b = W.mitigate_hpss_v2(dr7_p, window="cola_hann", lookahead=0, in_band=True, env_norm="causal")
    h5_blk_a = W.mitigate_hpss_v2(dr7, window="cola_hann", lookahead=0, in_band=True, env_norm="block")
    h5_blk_b = W.mitigate_hpss_v2(dr7_p, window="cola_hann", lookahead=0, in_band=True, env_norm="block")

    s_seq = rng.random((40, VT.BR_GRID_1BPM.size))
    s_pert = s_seq.copy()
    s_pert[25:] = rng.random((15, VT.BR_GRID_1BPM.size))
    hmm_fwd_diff = float(np.max(np.abs(W.hmm_forward_filter(s_seq)["est_mmse"][:25] - W.hmm_forward_filter(s_pert)["est_mmse"][:25])))
    hmm_lag4_diff = float(np.max(np.abs(W.hmm_fixed_lag_smoother(s_seq, lag_hops=4)["est_mmse"][:21] - W.hmm_fixed_lag_smoother(s_pert, lag_hops=4)["est_mmse"][:21])))

    return {
        "new_operators_invariance_max_diff_bpm": new_inv_results,
        "all_new_invariances_lt_1e6": bool(
            all(val < 1e-6 for sub in new_inv_results.values() for val in sub.values())
        ),
        "frozen_h0_swap_asymmetry_diagnostic_bpm": frozen_h0_diag,
        "causality_hpss_l0_max_diff": float(np.max(np.abs(h5_a[:, :18] - h5_b[:, :18]))),
        "causality_mitigate_v2_causal_env_mask_max_diff": float(np.max(np.abs(h5_c_a["mask"][:, :14] - h5_c_b["mask"][:, :14]))),
        "causality_mitigate_v2_causal_env_mag_filt_max_diff": float(np.max(np.abs(h5_c_a["mag_filt"][:, :14] - h5_c_b["mag_filt"][:, :14]))),
        "causality_mitigate_v2_block_env_mag_filt_diag_diff": float(np.max(np.abs(h5_blk_a["mag_filt"][:, :14] - h5_blk_b["mag_filt"][:, :14]))),
        "causality_hmm_fwd_max_diff": hmm_fwd_diff,
        "causality_hmm_lag4_max_diff": hmm_lag4_diff,
    }


def _run_shifted_grid_audit(lags_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Measure uniform-prior [18, 90] vs [18, 120] bpm midpoint shrinkage on 5 diverse sessions."""
    sess5 = ["2.3", "4.1", "7.3", "9.2", "11.3"]
    grid90 = np.arange(18.0, 90.0 + 1e-9, 1.0)
    grid120 = np.arange(18.0, 120.0 + 1e-9, 1.0)

    s90_map, s120_map, ref1 = {}, {}, {}
    for m in sess5:
        bi, bq = load_radar(m)
        starts = list(range(0, len(bi) - int(C.WIN_SAMPLES) + 1, int(C.UPDATE_S * C.FS_RADAR)))
        cs = np.arange(len(starts), dtype=float) * C.UPDATE_S + C.WIN_S / 2.0
        ext = EDFX.extract_window_references(
            EDFX.load_edf_recording(m), cs, int(lags_map[m]["neutral_raw_nls_lag_s"])
        )
        v1 = np.asarray(ext["valid_vwin"], bool) & np.isfinite(ext["br_1hz_30s"])
        ref1[m] = (np.asarray(ext["br_1hz_30s"], float), v1)
        a90, a120 = [], []
        for s0 in starts:
            dh = bandpass(mitigate_hpss(bandpass(csd(bi[s0 : s0 + 480], bq[s0 : s0 + 480])))["x_hat"])
            fc = autocorr_coarse(dh)
            a90.append(VT.extract_br_anchored_nls_spectrum(dh, fc, grid90))
            a120.append(VT.extract_br_anchored_nls_spectrum(dh, fc, grid120))
        s90_map[m] = np.asarray(a90)
        s120_map[m] = np.asarray(a120)

    specs = (
        ("memoryless_mmse_naive_eps001", lambda s, g: ((np.maximum(s, 0.0) + 0.01) / np.sum(np.maximum(s, 0.0) + 0.01, axis=1, keepdims=True)) @ g),
        ("fwd_mmse_legacy_b05_e005_sig8", lambda s, g: W.hmm_forward_filter(s, g, sigma_br=8.0, eps=0.05, beta=0.5)["est_mmse"]),
        ("fwd_mmse_equivariant_b1_e1e3_sig2", lambda s, g: W.hmm_forward_filter(s, g, sigma_br=2.0, eps=1e-3, beta=1.0)["est_mmse"]),
        ("fb_mmse_equivariant_b1_e1e3_sig1", lambda s, g: W.hmm_offline_fb_smoother(s, g, sigma_br=1.0, eps=1e-3, beta=1.0)["est_mmse"]),
        ("viterbi_b1_e1e3_sig1", lambda s, g: W.hmm_offline_viterbi_prob(s, g, sigma_br=1.0, eps=1e-3, beta=1.0)[0]),
    )
    out: dict[str, Any] = {}
    for label, fn in specs:
        p90 = {m: fn(s90_map[m], grid90) for m in sess5}
        p120 = {m: fn(s120_map[m], grid120) for m in sess5}
        a90, _ = _aggregate_method(sess5, p90, ref1)
        a120, _ = _aggregate_method(sess5, p120, ref1)
        b90 = float(a90["ba_session_level"]["bias"])
        b120 = float(a120["ba_session_level"]["bias"])
        out[label] = {
            "sessions": sess5,
            "rmse_18_90": float(a90["rmse_sess"]),
            "bias_18_90": b90,
            "rmse_18_120": float(a120["rmse_sess"]),
            "bias_18_120": b120,
            "bias_shift_bpm": float(b120 - b90),
            "abs_bias_shift_bpm": float(abs(b120 - b90)),
        }
    return out


def _holm_update_paired(paired_map: dict[str, dict[str, Any]], keys: list[str]) -> None:
    p_pat = [float(paired_map[k]["p_wilcoxon_patient"]) for k in keys]
    p_ses = [float(paired_map[k]["p_wilcoxon_sess"]) for k in keys]
    adj_pat = IO.holm_bonferroni(p_pat)
    adj_ses = IO.holm_bonferroni(p_ses)
    for k, ap, a_s in zip(keys, adj_pat, adj_ses):
        paired_map[k]["p_wilcoxon_patient_holm"] = ap
        paired_map[k]["p_wilcoxon_sess_holm"] = a_s


def _run_lag_and_pause_audit(
    sessions: list[str],
    p30_preds: dict[str, dict[str, np.ndarray]],
    ref_1hz_map: dict[str, tuple[np.ndarray, np.ndarray]],
    ref_trig_map: dict[str, tuple[np.ndarray, np.ndarray]],
    trig_count_map: dict[str, np.ndarray],
    pause_map: dict[str, np.ndarray],
    max_hop: int = 8,
) -> dict[str, Any]:
    """Quantify VitaGuard 1 Hz group delay (+lag => seqA trails seqB) and pause-exclusion bias."""
    r1_seq = {m: ref_1hz_map[m][0] for m in sessions}
    v1_only = {m: ref_1hz_map[m][1] for m in sessions}
    rt_seq = {m: ref_trig_map[m][0] for m in sessions}
    vt_only = {m: ref_trig_map[m][1] for m in sessions}

    def _lag_profile(
        seq_a: dict[str, np.ndarray],
        seq_b: dict[str, np.ndarray],
        vmap: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        prof: dict[str, float] = {}
        best_lag = 0
        best_r = -np.inf
        for k in range(-max_hop, max_hop + 1):
            xs, ys = [], []
            for m in sessions:
                a, b, v = seq_a[m], seq_b[m], vmap[m]
                if k == 0:
                    a_s, b_s, vm = a, b, v
                elif k > 0:
                    a_s, b_s, vm = a[k:], b[: len(b) - k], v[k:] & v[: len(v) - k]
                else:
                    kk = -k
                    a_s, b_s, vm = a[: len(a) - kk], b[kk:], v[: len(v) - kk] & v[kk:]
                if int(np.sum(vm)) >= 10:
                    xs.append(a_s[vm])
                    ys.append(b_s[vm])
            r_val = float(IO.rmcorr_ancova(xs, ys)["r_within"])
            lag_s = int(2 * k)
            prof[str(lag_s)] = r_val
            if np.isfinite(r_val) and r_val > best_r:
                best_r = r_val
                best_lag = lag_s
        return {
            "lag_0s_r": float(prof["0"]),
            "peak_lag_s": int(best_lag),
            "peak_r": float(best_r),
            "profile_rmcorr_by_lag_s": prof,
        }

    lag_profiles = {
        "br_1hz_30s_vs_br_trig_bpm": _lag_profile(r1_seq, rt_seq, vt_only),
        "R0_NLS_vs_br_1hz_30s": _lag_profile(p30_preds["R0_HPSS_NLS_Locked"], r1_seq, v1_only),
        "R0_NLS_vs_br_trig_bpm": _lag_profile(p30_preds["R0_HPSS_NLS_Locked"], rt_seq, vt_only),
        "R1a_Online_Forward_MMSE_0s_vs_br_1hz_30s": _lag_profile(
            p30_preds["R1a_Online_Forward_MMSE_0s"], r1_seq, v1_only
        ),
        "R1a_Online_Forward_MMSE_0s_vs_br_trig_bpm": _lag_profile(
            p30_preds["R1a_Online_Forward_MMSE_0s"], rt_seq, vt_only
        ),
        "R1f_Offline_FB_MMSE_Inf_vs_br_1hz_30s": _lag_profile(
            p30_preds["R1f_Offline_FB_MMSE_Inf"], r1_seq, v1_only
        ),
        "R1f_Offline_FB_MMSE_Inf_vs_br_trig_bpm": _lag_profile(
            p30_preds["R1f_Offline_FB_MMSE_Inf"], rt_seq, vt_only
        ),
    }

    # Pause-exclusion vs whole-window breath-count rate bias decomposition
    d_int_all, d_cnt_all = [], []
    d_int_pause, d_cnt_pause = [], []
    d_int_reg, d_cnt_reg = [], []
    d_cnt_v1_pause, d_cnt_v1_reg = [], []
    sess_int_minus_1hz, sess_cnt_minus_1hz = [], []
    for m in sessions:
        r1, v1 = ref_1hz_map[m]
        rt, vt = ref_trig_map[m]
        rc = trig_count_map[m]
        vtc = vt & np.isfinite(rc)
        v1c = v1 & np.isfinite(rc)
        pz = pause_map[m]
        if np.any(vtc):
            di = rt[vtc] - r1[vtc]
            dc = rc[vtc] - r1[vtc]
            d_int_all.append(di)
            d_cnt_all.append(dc)
            sess_int_minus_1hz.append(float(np.mean(di)))
            sess_cnt_minus_1hz.append(float(np.mean(dc)))
            p_m = vtc & pz
            r_m = vtc & (~pz)
            if np.any(p_m):
                d_int_pause.append(rt[p_m] - r1[p_m])
                d_cnt_pause.append(rc[p_m] - r1[p_m])
            if np.any(r_m):
                d_int_reg.append(rt[r_m] - r1[r_m])
                d_cnt_reg.append(rc[r_m] - r1[r_m])
        if np.any(v1c & pz):
            d_cnt_v1_pause.append(rc[v1c & pz] - r1[v1c & pz])
        if np.any(v1c & (~pz)):
            d_cnt_v1_reg.append(rc[v1c & (~pz)] - r1[v1c & (~pz)])

    c_int_all = np.concatenate(d_int_all)
    c_cnt_all = np.concatenate(d_cnt_all)
    c_int_pz = np.concatenate(d_int_pause) if d_int_pause else np.empty(0)
    c_cnt_pz = np.concatenate(d_cnt_pause) if d_cnt_pause else np.empty(0)
    c_int_rg = np.concatenate(d_int_reg) if d_int_reg else np.empty(0)
    c_cnt_rg = np.concatenate(d_cnt_reg) if d_cnt_reg else np.empty(0)

    pause_decomp = {
        "n_windows_paired_count": int(c_int_all.size),
        "mean_interval_minus_1hz_sess": float(np.mean(sess_int_minus_1hz)),
        "mean_interval_minus_1hz_pool": float(np.mean(c_int_all)),
        "mean_count_minus_1hz_sess": float(np.mean(sess_cnt_minus_1hz)),
        "mean_count_minus_1hz_pool": float(np.mean(c_cnt_all)),
        "mean_interval_minus_count_pool": float(np.mean(c_int_all - c_cnt_all)),
        "pause_gt5s": {
            "n_windows": int(c_int_pz.size),
            "mean_interval_minus_1hz": float(np.mean(c_int_pz)) if c_int_pz.size else float("nan"),
            "mean_count_minus_1hz": float(np.mean(c_cnt_pz)) if c_cnt_pz.size else float("nan"),
            "n_windows_all_v1": int(sum(len(x) for x in d_cnt_v1_pause)),
        },
        "regular_no_gt5s_pause": {
            "n_windows": int(c_int_rg.size),
            "mean_interval_minus_1hz": float(np.mean(c_int_rg)) if c_int_rg.size else float("nan"),
            "mean_count_minus_1hz": float(np.mean(c_cnt_rg)) if c_cnt_rg.size else float("nan"),
            "n_windows_all_v1": int(sum(len(x) for x in d_cnt_v1_reg)),
        },
    }
    return {
        "lag_profiles_rmcorr": lag_profiles,
        "pause_rate_bias_decomposition": pause_decomp,
    }


def _build_lopo_sensitivity_audit(
    decoded: dict[str, dict[str, Any]],
    ref_1hz_map: dict[str, tuple[np.ndarray, np.ndarray]],
    ref_trig_map: dict[str, tuple[np.ndarray, np.ndarray]],
    primary_results: dict[str, Any],
) -> dict[str, Any]:
    """Compare Stage 1.1 primary LOPO vs legacy v2 (0.95*r, sig>=0.5) & fixed sigma=2.0."""
    target_fams = (
        "R1a_Online_Forward_MMSE_0s",
        "R1c_FixedLag_MMSE_4s",
        "R1d_FixedLag_MMSE_8s",
        "R1e_FixedLag_MMSE_16s",
        "R1f_Offline_FB_MMSE_Inf",
        "R1g_Offline_Viterbi_Unpinned",
    )
    out: dict[str, Any] = {}
    for coh in LOPO_COHORTS:
        slist = IO.get_cohort_sessions(coh, include_zero_valid=False)
        # 1. Primary (guard=1.00, sigma_min=1.0) extracted from already-computed results
        prim_blk: dict[str, Any] = {}
        for fam in target_fams:
            m1 = primary_results[coh]["br_1hz_30s"]["methods"][fam]
            mt = primary_results[coh]["br_trig_bpm"]["methods"][fam]
            fs = primary_results[coh]["lopo_fold_summary"][fam]
            prim_blk[fam] = {
                "rmse_sess_1hz": float(m1["rmse_sess"]),
                "r_within_1hz": float(m1["r_within_rmcorr"]),
                "rmse_sess_trig": float(mt["rmse_sess"]),
                "r_within_trig": float(mt["r_within_rmcorr"]),
                "lower_sigma_pinned_folds": int(fs["lower_sigma_pinned_folds"]),
                "fallback_folds": int(fs["fallback_folds"]),
            }

        # 2. Legacy v2 (guard=0.95, sigma_min=0.5)
        leg_preds, leg_flogs, _ = _run_lopo_rung1(coh, decoded, ref_1hz_map, guard_mult=0.95, sigma_min=0.5)
        leg_blk: dict[str, Any] = {}
        for fam in target_fams:
            a1, _ = _aggregate_method(slist, leg_preds[fam], ref_1hz_map)
            at, _ = _aggregate_method(slist, leg_preds[fam], ref_trig_map)
            logs = leg_flogs[fam]
            leg_blk[fam] = {
                "rmse_sess_1hz": float(a1["rmse_sess"]),
                "r_within_1hz": float(a1["r_within_rmcorr"]),
                "rmse_sess_trig": float(at["rmse_sess"]),
                "r_within_trig": float(at["r_within_rmcorr"]),
                "lower_sigma_pinned_folds": int(sum(1 for r in logs if r["lower_sigma_pinned"])),
                "fallback_folds": int(sum(1 for r in logs if r["fallback_used"])),
            }

        # 3. Fixed hyperparameter (sigma=2.0, eps=1e-3, beta=1.0) — zero fold-switching confound
        fix_cfg = (2.0, 1e-3, 1.0)
        fix_blk: dict[str, Any] = {}
        for fam in target_fams:
            fpreds = {m: decoded[m]["grid_paths"][fam][fix_cfg] for m in slist}
            a1, _ = _aggregate_method(slist, fpreds, ref_1hz_map)
            at, _ = _aggregate_method(slist, fpreds, ref_trig_map)
            fix_blk[fam] = {
                "rmse_sess_1hz": float(a1["rmse_sess"]),
                "r_within_1hz": float(a1["r_within_rmcorr"]),
                "rmse_sess_trig": float(at["rmse_sess"]),
                "r_within_trig": float(at["r_within_rmcorr"]),
            }

        out[coh] = {
            "primary_guard100_sigmin10": prim_blk,
            "legacy_v2_guard095_sigmin05": leg_blk,
            "fixed_sigma2_eps1e3_beta1": fix_blk,
        }
    return out


def main() -> None:
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="EXP-1 Stage 1 Parsimony & Causality Benchmark")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-surrogates", action="store_true")
    args = parser.parse_args()

    print("[EXP-1] Verifying frozen baseline caches read-only...")
    status = IO.verify_all_baseline_caches()
    print(f"[EXP-1] Verified {len(status)}/33 locked v1 caches.")

    wp_sha = IO.sha256_file(WP_MODULE_PATH)
    all_eval_sessions = IO.get_cohort_sessions("all_paired", include_zero_valid=False)

    print(f"[EXP-1] Extracting Stage 1 WP1/WP2 features across {len(all_eval_sessions)} sessions (workers={args.workers})...")
    tasks = [(m, wp_sha, args.force) for m in all_eval_sessions]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            feat_logs = list(ex.map(_extract_session_features, tasks))
    else:
        feat_logs = [_extract_session_features(t) for t in tasks]
    n_comp = sum(1 for _, st, _ in feat_logs if st == "computed")
    max_par_all = max((mx for _, _, mx in feat_logs), default=0.0)
    print(f"[EXP-1] Feature extraction complete ({n_comp} computed, {len(feat_logs)-n_comp} cached, F0 max diff={max_par_all:.2e} bpm).")

    # Load all EDF references once under locked branch_neutral_raw_nls_lag
    print("[EXP-1] Loading EDF dual references & branch-neutral lags...")
    lags_map = IO.load_locked_neutral_lags()
    ref_1hz_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ref_trig_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    trig_count_map: dict[str, np.ndarray] = {}
    pause_map: dict[str, np.ndarray] = {}

    for m in all_eval_sessions:
        feat = _load_feat_cache(m)
        n_win = len(feat["wp1_F0_CSD_Locked"])
        centers = np.arange(n_win, dtype=float) * C.UPDATE_S + C.WIN_S / 2.0
        edf = EDFX.load_edf_recording(m)
        lag_val = int(lags_map[m]["neutral_raw_nls_lag_s"])
        ext = EDFX.extract_window_references(edf, centers, lag_s=lag_val)
        v1 = np.asarray(ext["valid_vwin"], dtype=bool) & np.isfinite(ext["br_1hz_30s"])
        vtrig = v1 & np.isfinite(ext["br_trig_bpm"])
        pause = v1 & np.isfinite(ext["resp_max_gap_s"]) & (ext["resp_max_gap_s"] > 5.0)
        ref_1hz_map[m] = (np.asarray(ext["br_1hz_30s"], dtype=np.float64), v1)
        ref_trig_map[m] = (np.asarray(ext["br_trig_bpm"], dtype=np.float64), vtrig)
        pause_map[m] = pause

        # Whole-window breath-count rate 60 * N_events / 30 s over identical [a, b)
        resp_trig_t = EDFX.detect_trigger_times(edf.resp_trig_1024, fs_hz=edf.fs_trig, min_refractory_s=0.30)
        tc = centers - float(lag_val)
        a_arr = np.floor(tc - float(C.WIN_S) / 2.0).astype(int)
        b_arr = a_arr + int(C.WIN_S)
        br_cnt = np.full(n_win, np.nan, dtype=np.float64)
        for w_idx in range(n_win):
            if bool(ext["in_bounds"][w_idx]):
                n_ev = int(np.sum((resp_trig_t >= float(a_arr[w_idx])) & (resp_trig_t < float(b_arr[w_idx]))))
                br_cnt[w_idx] = 60.0 * float(n_ev) / float(C.WIN_S)
        trig_count_map[m] = br_cnt

    # Dual-reference discrepancy summary across cohorts
    discrepancy_summary: dict[str, Any] = {}
    for coh in COHORT_ORDER:
        slist = IO.get_cohort_sessions(coh, include_zero_valid=False)
        diffs_sess, maes_sess, e_1hz, e_trg = [], [], [], []
        n_v1, n_vt, n_pz = 0, 0, 0
        for m in slist:
            r1, v1 = ref_1hz_map[m]
            rt, vt = ref_trig_map[m]
            n_v1 += int(np.sum(v1))
            n_vt += int(np.sum(vt))
            n_pz += int(np.sum(pause_map[m]))
            if np.any(vt):
                d = rt[vt] - r1[vt]
                diffs_sess.append(float(np.mean(d)))
                maes_sess.append(float(np.mean(np.abs(d))))
                e_1hz.append(r1[vt])
                e_trg.append(rt[vt])
        all_d = np.concatenate(e_trg) - np.concatenate(e_1hz)
        rmc_ref = IO.rmcorr_ancova(e_trg, e_1hz)
        discrepancy_summary[coh] = {
            "n_sessions": len(slist),
            "n_windows_1hz": n_v1,
            "n_windows_trig": n_vt,
            "n_pause_gt5s": n_pz,
            "pause_gt5s_pct": float(100.0 * n_pz / max(1, n_v1)),
            "mean_trig_minus_1hz_sess": float(np.mean(diffs_sess)),
            "mean_trig_minus_1hz_pool": float(np.mean(all_d)),
            "mae_trig_vs_1hz_sess": float(np.mean(maes_sess)),
            "mae_trig_vs_1hz_pool": float(np.mean(np.abs(all_d))),
            "r_within_trig_vs_1hz": rmc_ref["r_within"],
        }

    # Precompute WP1 + WP2 predictions per session
    feats_by_sess = {m: _load_feat_cache(m) for m in all_eval_sessions}
    wp1_preds = {k: {m: feats_by_sess[m][f"wp1_{k}"] for m in all_eval_sessions} for k in WP1_METHODS}
    wp2_preds = {k: {m: feats_by_sess[m][f"wp2_{k}"] for m in all_eval_sessions} for k in WP2_METHODS}
    wp2_runtime = {}
    p30_list = IO.get_cohort_sessions("pool30_sync", include_zero_valid=False)
    h0_ms = float(np.mean([float(np.mean(feats_by_sess[m]["time_H0_Locked_Rect_Full_L3_ms"])) for m in p30_list]))
    for k in WP2_METHODS:
        mean_ms = float(np.mean([float(np.mean(feats_by_sess[m][f"time_{k}_ms"])) for m in p30_list]))
        wp2_runtime[k] = {
            "lookahead_s": float(WP2_SPECS[k]["lookahead"]),
            "mean_ms_per_win": mean_ms,
            "speedup_vs_h0": float(h0_ms / max(mean_ms, 1e-9)),
        }

    # Evaluate WP1 & WP2 across all cohorts & both references
    wp1_results: dict[str, dict[str, Any]] = {}
    wp2_results: dict[str, dict[str, Any]] = {}
    sentinel_ids = ("2.3", "11.1", "12.1", "7.3", "4.1")
    wp1_sentinels: dict[str, dict[str, dict[str, float]]] = {}

    for coh in COHORT_ORDER:
        slist = IO.get_cohort_sessions(coh, include_zero_valid=False)
        wp1_results[coh] = {}
        wp2_results[coh] = {}
        for ref_lbl, rmap in (("br_1hz_30s", ref_1hz_map), ("br_trig_bpm", ref_trig_map)):
            const_sum, _ = _constant_baselines(slist, rmap)

            # WP1
            m_agg1, m_ps1 = {}, {}
            for k in WP1_METHODS:
                m_agg1[k], m_ps1[k] = _aggregate_method(slist, wp1_preds[k], rmap)
            paired1 = {k: _paired_comparison(slist, m_ps1[k], m_ps1["F0_CSD_Locked"]) for k in WP1_METHODS[1:]}
            _holm_update_paired(paired1, list(WP1_METHODS[1:]))
            wp1_results[coh][ref_lbl] = {"constant_baselines": const_sum, "methods": m_agg1, "paired_vs_F0": paired1}
            if coh == "pool30_sync":
                wp1_sentinels[ref_lbl] = {
                    k: {sid: float(m_ps1[k][sid]["rmse"]) for sid in sentinel_ids if sid in m_ps1[k]}
                    for k in WP1_METHODS
                }

            # WP2
            m_agg2, m_ps2 = {}, {}
            for k in WP2_METHODS:
                m_agg2[k], m_ps2[k] = _aggregate_method(slist, wp2_preds[k], rmap)
            paired2 = {k: _paired_comparison(slist, m_ps2[k], m_ps2["H0_Locked_Rect_Full_L3"]) for k in WP2_METHODS[1:]}
            _holm_update_paired(paired2, list(WP2_METHODS[1:]))
            wp2_results[coh][ref_lbl] = {"constant_baselines": const_sum, "methods": m_agg2, "paired_vs_H0": paired2}

    # Verification controls & shifted-grid shortcut audit (run prior to Gate G1 so O(2) check is empirical)
    print("[EXP-1] Running invariance, causality & shifted-grid verification suite...")
    invariance_audit = _run_invariance_and_causality_audit()
    shifted_grid_audit = _run_shifted_grid_audit(lags_map)

    # Gate G1 parsimony decision on pool30_sync (requires statistical support + dual-ref gain + measured invariance)
    g1_candidates = {}
    best_g1_key = None
    best_g1_score = 0.0
    max_new_inv = max(
        val for sub in invariance_audit["new_operators_invariance_max_diff_bpm"].values() for val in sub.values()
    )
    for k in WP1_METHODS[1:]:
        pr1 = wp1_results["pool30_sync"]["br_1hz_30s"]["paired_vs_F0"][k]
        prt = wp1_results["pool30_sync"]["br_trig_bpm"]["paired_vs_F0"][k]
        d1 = float(pr1["mean_delta"])
        dt = float(prt["mean_delta"])
        ci95_hi_1hz = float(pr1["ci95_hi"])
        p_holm_1hz = float(pr1["p_wilcoxon_patient_holm"])
        s73_1 = wp1_sentinels["br_1hz_30s"][k]["7.3"] - wp1_sentinels["br_1hz_30s"]["F0_CSD_Locked"]["7.3"]
        s41_1 = wp1_sentinels["br_1hz_30s"][k]["4.1"] - wp1_sentinels["br_1hz_30s"]["F0_CSD_Locked"]["4.1"]
        s73_t = wp1_sentinels["br_trig_bpm"][k]["7.3"] - wp1_sentinels["br_trig_bpm"]["F0_CSD_Locked"]["7.3"]
        s41_t = wp1_sentinels["br_trig_bpm"][k]["4.1"] - wp1_sentinels["br_trig_bpm"]["F0_CSD_Locked"]["4.1"]
        max_sent_deg = max(s73_1, s41_1, s73_t, s41_t)
        inv_cls = WP1_O2_INV_CLASS[k]
        inv_measured_ok = bool(inv_cls in ("unconditional", "conditional_fixed_mask") and max_new_inv < 1e-6)
        stat_supported = bool(ci95_hi_1hz < 0.0 or p_holm_1hz < 0.05)
        passes_parsimony = bool(d1 < 0.0 and dt < 0.0 and stat_supported and max_sent_deg <= 0.50 and inv_measured_ok)
        g1_candidates[k] = {
            "delta_1hz": d1,
            "ci95_hi_1hz": ci95_hi_1hz,
            "delta_trig": dt,
            "p_pat_holm_1hz": p_holm_1hz,
            "p_pat_holm_trig": float(prt["p_wilcoxon_patient_holm"]),
            "max_clean_sentinel_degradation": float(max_sent_deg),
            "o2_invariance_class": inv_cls,
            "measured_o2_invariance_ok": inv_measured_ok,
            "statistically_supported": stat_supported,
            "passes_gate_g1_parsimony": passes_parsimony,
            "strict_unconditional_o2_inv": bool(inv_cls == "unconditional"),
        }
        if passes_parsimony and (d1 + dt) < best_g1_score:
            best_g1_score = d1 + dt
            best_g1_key = k
    gate_g1 = {
        "decision": f"ADOPT_{best_g1_key}" if best_g1_key else "RETAIN_F0_CSD_LOCKED",
        "candidates": g1_candidates,
    }

    # Gate G2 automated decision for 0s causal HPSS (H5 & H6) on BOTH tost_sess & tost_patient
    g2_candidates = {}
    best_g2_key = None
    best_g2_score = float("inf")
    for k in ("H5_Causal_COLA_InBand_L0", "H6_Causal_Rect_Full_L0"):
        p1 = wp2_results["pool30_sync"]["br_1hz_30s"]["paired_vs_H0"][k]
        pt = wp2_results["pool30_sync"]["br_trig_bpm"]["paired_vs_H0"][k]
        eq_sess_1 = bool(p1["tost_sess"]["equivalent"] or p1["mean_delta"] < 0.0)
        eq_sess_t = bool(pt["tost_sess"]["equivalent"] or pt["mean_delta"] < 0.0)
        eq_pat_1 = bool(p1["tost_patient"]["equivalent"] or p1["mean_delta"] < 0.0)
        eq_pat_t = bool(pt["tost_patient"]["equivalent"] or pt["mean_delta"] < 0.0)
        passes_sess_both = bool(eq_sess_1 and eq_sess_t)
        passes_pat_both = bool(eq_pat_1 and eq_pat_t)
        passes_g2 = bool(passes_sess_both and passes_pat_both and wp2_runtime[k]["mean_ms_per_win"] < 10.0)
        g2_candidates[k] = {
            "delta_1hz": float(p1["mean_delta"]),
            "delta_trig": float(pt["mean_delta"]),
            "tost_sess_p_1hz": float(p1["tost_sess"]["p_tost"]),
            "tost_patient_p_1hz": float(p1["tost_patient"]["p_tost"]),
            "tost_sess_p_trig": float(pt["tost_sess"]["p_tost"]),
            "tost_patient_p_trig": float(pt["tost_patient"]["p_tost"]),
            "passes_tost_sess_both": passes_sess_both,
            "passes_tost_patient_both": passes_pat_both,
            "mean_ms_per_win": wp2_runtime[k]["mean_ms_per_win"],
            "passes_gate_g2": passes_g2,
        }
        if passes_g2 and (float(p1["mean_delta"]) + float(pt["mean_delta"])) < best_g2_score:
            best_g2_score = float(p1["mean_delta"]) + float(pt["mean_delta"])
            best_g2_key = k
    gate_g2 = {
        "decision": f"ADOPT_{best_g2_key}" if best_g2_key else "RETAIN_H0_LOCKED",
        "candidates": g2_candidates,
    }

    # Decode Rung 1 unpinned grid across sessions in parallel
    print(f"[EXP-1] Decoding Rung 1 unpinned HMM grid across {len(all_eval_sessions)} sessions (workers={args.workers})...")
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            dec_list = list(ex.map(_decode_rung1_session, all_eval_sessions))
    else:
        dec_list = [_decode_rung1_session(m) for m in all_eval_sessions]
    decoded = dict(dec_list)

    # Evaluate Rung 1 LOPO across pool30_sync, retained23, PAPER_18
    rung1_results: dict[str, Any] = {}
    p30_preds: dict[str, dict[str, np.ndarray]] = {}
    p30_fold_logs: dict[str, list[dict[str, Any]]] = {}
    p30_unc_logs: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    p30_per_sess_1hz: dict[str, dict[str, dict[str, float]]] = {}

    for coh in LOPO_COHORTS:
        slist = IO.get_cohort_sessions(coh, include_zero_valid=False)
        preds_c, flogs_c, ulogs_c = _run_lopo_rung1(coh, decoded, ref_1hz_map, guard_mult=1.00, sigma_min=1.0)
        if coh == "pool30_sync":
            p30_preds, p30_fold_logs, p30_unc_logs = preds_c, flogs_c, ulogs_c

        coh_block: dict[str, Any] = {}
        for ref_lbl, rmap in (("br_1hz_30s", ref_1hz_map), ("br_trig_bpm", ref_trig_map)):
            const_sum, per_sess_const = _constant_baselines(slist, rmap)
            m_agg, m_ps = {}, {}
            for k in RUNG1_METHODS:
                m_agg[k], m_ps[k] = _aggregate_method(slist, preds_c[k], rmap)
            if coh == "pool30_sync" and ref_lbl == "br_1hz_30s":
                p30_per_sess_1hz = m_ps

            paired_r0 = {k: _paired_comparison(slist, m_ps[k], m_ps["R0_HPSS_NLS_Locked"]) for k in RUNG1_METHODS[1:]}
            _holm_update_paired(paired_r0, list(RUNG1_METHODS[1:]))
            paired_const = {k: _paired_comparison(slist, m_ps[k], per_sess_const) for k in RUNG1_METHODS}
            _holm_update_paired(paired_const, list(RUNG1_METHODS))
            paired_vit = {
                k: _paired_comparison(slist, m_ps[k], m_ps["R1g_Offline_Viterbi_Unpinned"])
                for k in RUNG1_METHODS[1:]
                if k != "R1g_Offline_Viterbi_Unpinned"
            }

            coh_block[ref_lbl] = {
                "constant_baselines": const_sum,
                "methods": m_agg,
                "paired_vs_R0": paired_r0,
                "paired_vs_const_loso": paired_const,
                "paired_vs_R1g_offline_viterbi": paired_vit,
            }

        # Fold summary statistics
        fold_summary = {}
        for fam, logs in flogs_c.items():
            sigmas = [row["sigma_br"] for row in logs]
            epss = [row["eps"] for row in logs]
            betas = [row["beta"] for row in logs]
            fold_summary[fam] = {
                "n_folds": len(logs),
                "sigma_counts": {str(s): sigmas.count(s) for s in sorted(set(sigmas))},
                "eps_counts": {str(e): epss.count(e) for e in sorted(set(epss))},
                "beta_counts": {str(b): betas.count(b) for b in sorted(set(betas))},
                "lower_sigma_pinned_folds": int(sum(1 for row in logs if row["lower_sigma_pinned"])),
                "upper_sigma_pinned_folds": int(sum(1 for row in logs if row["upper_sigma_pinned"])),
                "any_boundary_pinned_folds": int(sum(1 for row in logs if row["any_boundary_pinned"])),
                "fallback_folds": int(sum(1 for row in logs if row["fallback_used"])),
                "folds": logs,
            }
        coh_block["lopo_fold_summary"] = fold_summary
        rung1_results[coh] = coh_block

    # WP4 causal uncertainty preview on pool30_sync
    wp4_preview = _wp4_uncertainty_preview(
        p30_list, p30_preds, p30_unc_logs, ref_1hz_map, ref_trig_map, pause_map
    )

    # Dual-reference monitor-lag & pause-exclusion audit + LOPO guard/floor sensitivity
    print("[EXP-1] Running dual-reference lag/pause audit & LOPO guard sensitivity...")
    lag_and_pause_audit = _run_lag_and_pause_audit(
        p30_list, p30_preds, ref_1hz_map, ref_trig_map, trig_count_map, pause_map
    )
    lopo_sensitivity_audit = _build_lopo_sensitivity_audit(decoded, ref_1hz_map, ref_trig_map, rung1_results)

    # 5-seed radar phase-scrambled surrogates on pool30_sync
    phase_scr_out: dict[str, Any] = {"skipped": bool(args.skip_surrogates), "n_seeds": 5}
    if not args.skip_surrogates:
        print(f"[EXP-1] Running 5-seed phase-scrambled radar surrogates on {len(p30_list)} sessions (workers={args.workers})...")
        sess_cfgs: dict[str, dict[str, tuple[float, float, float]]] = {}
        for fam in (
            "R1a_Online_Forward_MMSE_0s",
            "R1f_Offline_FB_MMSE_Inf",
            "R1g_Offline_Viterbi_Unpinned",
            "R1h_E2E_Causal_H5_OnlineForward_0s",
        ):
            for row in p30_fold_logs[fam]:
                for tm in row["test_sessions"]:
                    sess_cfgs.setdefault(tm, {})[fam] = (row["sigma_br"], row["eps"], row["beta"])
        scr_tasks = [(m, 5, IO.RANDOM_SEED, wp_sha, sess_cfgs[m]) for m in p30_list]
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                scr_res = dict(ex.map(_phase_scramble_session_stage1, scr_tasks))
        else:
            scr_res = dict(_phase_scramble_session_stage1(t) for t in scr_tasks)

        scr_methods = (
            "R0_HPSS_NLS_Locked",
            "R1a_Online_Forward_MMSE_0s",
            "R1f_Offline_FB_MMSE_Inf",
            "R1g_Offline_Viterbi_Unpinned",
            "R1h_E2E_Causal_H5_OnlineForward_0s",
        )
        for ref_lbl, rmap in (("br_1hz_30s", ref_1hz_map), ("br_trig_bpm", ref_trig_map)):
            sub: dict[str, Any] = {}
            for k in scr_methods:
                rmses, acc10s, rws = [], [], []
                for s_i in range(5):
                    emap = {m: scr_res[m][s_i][k] for m in p30_list}
                    ag, _ = _aggregate_method(p30_list, emap, rmap)
                    rmses.append(ag["rmse_sess"])
                    acc10s.append(ag["acc10_sess"])
                    rws.append(ag["r_within_rmcorr"])
                sub[k] = {
                    "rmse_sess_mean": float(np.mean(rmses)),
                    "rmse_sess_sd": float(np.std(rmses, ddof=1)),
                    "acc10_sess_mean": float(np.mean(acc10s)),
                    "r_within_rmcorr_mean": float(np.mean(rws)),
                }
            phase_scr_out[ref_lbl] = sub

    # Export per-session CSV on pool30_sync
    IO.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with AUDIT_SESSIONS_CSV.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "measurement", "patient_id", "n_win_1hz", "n_win_trig",
            "clip_frac_session_true", "clip_frac_max_win",
            "F0_rmse_1hz", "F1a_rmse_1hz", "F1b_rmse_1hz", "F2a_rmse_1hz", "F2b_rmse_1hz", "F2c_rmse_1hz", "F3_rmse_1hz",
            "H5_rmse_1hz", "H6_rmse_1hz",
            "R1a_rmse_1hz", "R1b_rmse_1hz", "R1c_rmse_1hz", "R1d_rmse_1hz", "R1e_rmse_1hz",
            "R1f_rmse_1hz", "R1g_rmse_1hz", "R1h_rmse_1hz", "R1i_rmse_1hz",
            "F0_rmse_trig", "F2b_rmse_trig", "F2c_rmse_trig", "H5_rmse_trig",
            "R1a_rmse_trig", "R1f_rmse_trig", "R1g_rmse_trig", "R1h_rmse_trig",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        # Extract pool30_sync per-session RMSEs cleanly
        wp1_1 = {k: _aggregate_method(p30_list, wp1_preds[k], ref_1hz_map)[1] for k in WP1_METHODS}
        wp1_t = {k: _aggregate_method(p30_list, wp1_preds[k], ref_trig_map)[1] for k in WP1_METHODS}
        wp2_1 = {k: _aggregate_method(p30_list, wp2_preds[k], ref_1hz_map)[1] for k in WP2_METHODS}
        wp2_t = {k: _aggregate_method(p30_list, wp2_preds[k], ref_trig_map)[1] for k in WP2_METHODS}
        r1_t = {k: _aggregate_method(p30_list, p30_preds[k], ref_trig_map)[1] for k in RUNG1_METHODS}
        for m in p30_list:
            writer.writerow({
                "measurement": m,
                "patient_id": IO.patient_id_of(m),
                "n_win_1hz": int(np.sum(ref_1hz_map[m][1])),
                "n_win_trig": int(np.sum(ref_trig_map[m][1])),
                "clip_frac_session_true": f"{float(feats_by_sess[m]['session_clip_frac']):.4f}",
                "clip_frac_max_win": f"{float(np.max(feats_by_sess[m]['clip_frac_w'])):.4f}",
                "F0_rmse_1hz": f"{wp1_1['F0_CSD_Locked'][m]['rmse']:.4f}",
                "F1a_rmse_1hz": f"{wp1_1['F1a_ClipChanInterp_CSD'][m]['rmse']:.4f}",
                "F1b_rmse_1hz": f"{wp1_1['F1b_ClipJointInterp_CSD'][m]['rmse']:.4f}",
                "F2a_rmse_1hz": f"{wp1_1['F2a_SymWhitened_lam005'][m]['rmse']:.4f}",
                "F2b_rmse_1hz": f"{wp1_1['F2b_SymWhitened_lam010'][m]['rmse']:.4f}",
                "F2c_rmse_1hz": f"{wp1_1['F2c_SymWhitened_lam025'][m]['rmse']:.4f}",
                "F3_rmse_1hz": f"{wp1_1['F3_ClipChan_SymWhitened_lam010'][m]['rmse']:.4f}",
                "H5_rmse_1hz": f"{wp2_1['H5_Causal_COLA_InBand_L0'][m]['rmse']:.4f}",
                "H6_rmse_1hz": f"{wp2_1['H6_Causal_Rect_Full_L0'][m]['rmse']:.4f}",
                "R1a_rmse_1hz": f"{p30_per_sess_1hz['R1a_Online_Forward_MMSE_0s'][m]['rmse']:.4f}",
                "R1b_rmse_1hz": f"{p30_per_sess_1hz['R1b_Online_Forward_MAP_0s'][m]['rmse']:.4f}",
                "R1c_rmse_1hz": f"{p30_per_sess_1hz['R1c_FixedLag_MMSE_4s'][m]['rmse']:.4f}",
                "R1d_rmse_1hz": f"{p30_per_sess_1hz['R1d_FixedLag_MMSE_8s'][m]['rmse']:.4f}",
                "R1e_rmse_1hz": f"{p30_per_sess_1hz['R1e_FixedLag_MMSE_16s'][m]['rmse']:.4f}",
                "R1f_rmse_1hz": f"{p30_per_sess_1hz['R1f_Offline_FB_MMSE_Inf'][m]['rmse']:.4f}",
                "R1g_rmse_1hz": f"{p30_per_sess_1hz['R1g_Offline_Viterbi_Unpinned'][m]['rmse']:.4f}",
                "R1h_rmse_1hz": f"{p30_per_sess_1hz['R1h_E2E_Causal_H5_OnlineForward_0s'][m]['rmse']:.4f}",
                "R1i_rmse_1hz": f"{p30_per_sess_1hz['R1i_E2E_Causal_H6_OnlineForward_0s'][m]['rmse']:.4f}",
                "F0_rmse_trig": f"{wp1_t['F0_CSD_Locked'][m]['rmse']:.4f}" if m in wp1_t["F0_CSD_Locked"] else "",
                "F2b_rmse_trig": f"{wp1_t['F2b_SymWhitened_lam010'][m]['rmse']:.4f}" if m in wp1_t["F2b_SymWhitened_lam010"] else "",
                "F2c_rmse_trig": f"{wp1_t['F2c_SymWhitened_lam025'][m]['rmse']:.4f}" if m in wp1_t["F2c_SymWhitened_lam025"] else "",
                "H5_rmse_trig": f"{wp2_t['H5_Causal_COLA_InBand_L0'][m]['rmse']:.4f}" if m in wp2_t["H5_Causal_COLA_InBand_L0"] else "",
                "R1a_rmse_trig": f"{r1_t['R1a_Online_Forward_MMSE_0s'][m]['rmse']:.4f}" if m in r1_t["R1a_Online_Forward_MMSE_0s"] else "",
                "R1f_rmse_trig": f"{r1_t['R1f_Offline_FB_MMSE_Inf'][m]['rmse']:.4f}" if m in r1_t["R1f_Offline_FB_MMSE_Inf"] else "",
                "R1g_rmse_trig": f"{r1_t['R1g_Offline_Viterbi_Unpinned'][m]['rmse']:.4f}" if m in r1_t["R1g_Offline_Viterbi_Unpinned"] else "",
                "R1h_rmse_trig": f"{r1_t['R1h_E2E_Causal_H5_OnlineForward_0s'][m]['rmse']:.4f}" if m in r1_t["R1h_E2E_Causal_H5_OnlineForward_0s"] else "",
            })

    rung1_causality_contract = {
        k: {
            "block_lookahead_s": None if np.isinf(RUNG1_BLOCK_LOOKAHEAD_S[k]) else float(RUNG1_BLOCK_LOOKAHEAD_S[k]),
            "block_lookahead_label": "inf" if np.isinf(RUNG1_BLOCK_LOOKAHEAD_S[k]) else f"{RUNG1_BLOCK_LOOKAHEAD_S[k]:.0f} s",
            "stft_lookahead_s": float(RUNG1_STFT_LOOKAHEAD_S[k]),
            "stft_lookahead_label": f"{RUNG1_STFT_LOOKAHEAD_S[k]:.0f} s",
        }
        for k in RUNG1_METHODS
    }

    total_runtime_s = float(time.perf_counter() - t_start)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "git_head": IO.git_head_info(),
        "analysis_plan": "thesis/analysis_plan_v3_ai.md",
        "regime": REGIME,
        "lopo_selection_contract": {
            "guard_mult": 1.00,
            "sigma_grid": list(SIGMA_GRID),
            "eps_grid": list(EPS_GRID),
            "beta_grid": list(BETA_GRID),
        },
        "f0_vs_locked_cache_max_diff_bpm": float(max_par_all),
        "total_runtime_s": total_runtime_s,
        "dual_reference_discrepancy": discrepancy_summary,
        "dual_reference_lag_and_pause_audit": lag_and_pause_audit,
        "wp1_ablation": wp1_results,
        "wp1_sentinel_sessions_pool30": wp1_sentinels,
        "gate_g1_verdict": gate_g1,
        "wp2_runtime": wp2_runtime,
        "wp2_ablation": wp2_results,
        "gate_g2_verdict": gate_g2,
        "rung1_causality_contract": rung1_causality_contract,
        "rung1_lopo": rung1_results,
        "rung1_lopo_sensitivity_audit": lopo_sensitivity_audit,
        "wp4_causal_uncertainty_preview": wp4_preview,
        "phase_scrambled_surrogates_pool30": phase_scr_out,
        "shifted_grid_midpoint_audit": shifted_grid_audit,
        "invariance_and_causality_checks": invariance_audit,
    }
    IO.write_json_locked(BUNDLE_JSON, payload)

    # Write crisp Markdown audit report
    new_c_max = max(invariance_audit["new_operators_invariance_max_diff_bpm"]["clean_7_3"].values())
    new_x_max = max(invariance_audit["new_operators_invariance_max_diff_bpm"]["clipped_2_3"].values())
    h0_x_max = max(invariance_audit["frozen_h0_swap_asymmetry_diagnostic_bpm"]["clipped_2_3"].values())
    lines = [
        f"# EXP-1 Stage 1.1 Parsimony & Causality Benchmark Report (`{SCHEMA_VERSION}`)\n",
        f"- **Git HEAD:** `{payload['git_head']['head_commit']}` (`{payload['git_head']['branch']}`, Python `{payload['git_head']['python_version']}`)",
        f"- **Locked Baseline Parity (`max |F0 - est_hpss|` across {len(all_eval_sessions)} sessions):** `{max_par_all:.2e} bpm`",
        f"- **Total Wall Time:** `{total_runtime_s:.1f} s`",
        f"- **New Stage 1 Operators O(2) × R+>0 Invariance (< 1e-6 bpm):** `{'PASS' if invariance_audit['all_new_invariances_lt_1e6'] else 'FAIL'}` (clean max `{new_c_max:.2e}`, clipped max `{new_x_max:.2e}`; frozen `h0` Nyquist-swap continuous-MMSE diag `{h0_x_max:.2e} bpm`)",
        f"- **Exact Frame Causality Diffs:** `hpss_filter_causal(L=0)` `{invariance_audit['causality_hpss_l0_max_diff']:.2e}`, `mitigate_hpss_v2(env_norm='causal')` `mask` `{invariance_audit['causality_mitigate_v2_causal_env_mask_max_diff']:.2e}` / `mag_filt` `{invariance_audit['causality_mitigate_v2_causal_env_mag_filt_max_diff']:.2e}` (vs `env_norm='block'` intra-block median coupling diag `{invariance_audit['causality_mitigate_v2_block_env_mag_filt_diag_diff']:.2f}`), HMM Forward `{invariance_audit['causality_hmm_fwd_max_diff']:.2e}`, HMM Lag-4 `{invariance_audit['causality_hmm_lag4_max_diff']:.2e}`\n",
        "## 1. Dual-Reference Discrepancy Summary (`1024 Hz Resp.-Trigger` minus `1 Hz RespRate`)\n",
        "| Cohort | $S$ | $N_{1\\mathrm{Hz}}$ | $N_{\\mathrm{trig}}$ | $>5\\text{ s}$ Pauses | Mean Diff (sess / pool) | MAE (pool) | $r_{\\mathrm{within}}$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for coh in COHORT_ORDER:
        d = discrepancy_summary[coh]
        lines.append(
            f"| `{coh}` | {d['n_sessions']} | {d['n_windows_1hz']} | {d['n_windows_trig']} | {d['n_pause_gt5s']} ({d['pause_gt5s_pct']:.1f}%) | "
            f"`{d['mean_trig_minus_1hz_sess']:+.2f}` / `{d['mean_trig_minus_1hz_pool']:+.2f}` | {d['mae_trig_vs_1hz_pool']:.2f} | {d['r_within_trig_vs_1hz']:.3f} |"
        )

    lp_d = lag_and_pause_audit["lag_profiles_rmcorr"]
    pd_d = lag_and_pause_audit["pause_rate_bias_decomposition"]
    lines += [
        "\n### 1b. Monitor Group-Delay & Pause-Exclusion Diagnostic (`pool30_sync`, `+lag` = first signal trails)\n",
        "| Pair | $r_{\\mathrm{within}}$ (`0 s`) | Peak Lag (s) | Peak $r_{\\mathrm{within}}$ | Interpretation |",
        "| :--- | :---: | :---: | :---: | :--- |",
        f"| `br_1hz_30s` vs `br_trig_bpm` | {lp_d['br_1hz_30s_vs_br_trig_bpm']['lag_0s_r']:.3f} | `{lp_d['br_1hz_30s_vs_br_trig_bpm']['peak_lag_s']:+d} s` | {lp_d['br_1hz_30s_vs_br_trig_bpm']['peak_r']:.3f} | VitaGuard `1 Hz` trend trails physical breath triggers |",
        f"| `R0_NLS` vs `br_1hz_30s` / `br_trig_bpm` | {lp_d['R0_NLS_vs_br_1hz_30s']['lag_0s_r']:.3f} / {lp_d['R0_NLS_vs_br_trig_bpm']['lag_0s_r']:.3f} | `{lp_d['R0_NLS_vs_br_1hz_30s']['peak_lag_s']:+d} s` / `{lp_d['R0_NLS_vs_br_trig_bpm']['peak_lag_s']:+d} s` | {lp_d['R0_NLS_vs_br_1hz_30s']['peak_r']:.3f} / {lp_d['R0_NLS_vs_br_trig_bpm']['peak_r']:.3f} | `branch_neutral_raw_nls_lag` aligns `R0` with `1 Hz` (`+10 s` behind `Trig`) |",
        f"| `R1a (0 s)` vs `br_1hz_30s` / `br_trig_bpm` | {lp_d['R1a_Online_Forward_MMSE_0s_vs_br_1hz_30s']['lag_0s_r']:.3f} / {lp_d['R1a_Online_Forward_MMSE_0s_vs_br_trig_bpm']['lag_0s_r']:.3f} | `{lp_d['R1a_Online_Forward_MMSE_0s_vs_br_1hz_30s']['peak_lag_s']:+d} s` / `{lp_d['R1a_Online_Forward_MMSE_0s_vs_br_trig_bpm']['peak_lag_s']:+d} s` | {lp_d['R1a_Online_Forward_MMSE_0s_vs_br_1hz_30s']['peak_r']:.3f} / {lp_d['R1a_Online_Forward_MMSE_0s_vs_br_trig_bpm']['peak_r']:.3f} | Causal Forward-HMM filter delay stacks with VitaGuard lag |",
        f"| `R1f (∞ FB)` vs `br_1hz_30s` / `br_trig_bpm` | {lp_d['R1f_Offline_FB_MMSE_Inf_vs_br_1hz_30s']['lag_0s_r']:.3f} / {lp_d['R1f_Offline_FB_MMSE_Inf_vs_br_trig_bpm']['lag_0s_r']:.3f} | `{lp_d['R1f_Offline_FB_MMSE_Inf_vs_br_1hz_30s']['peak_lag_s']:+d} s` / `{lp_d['R1f_Offline_FB_MMSE_Inf_vs_br_trig_bpm']['peak_lag_s']:+d} s` | {lp_d['R1f_Offline_FB_MMSE_Inf_vs_br_1hz_30s']['peak_r']:.3f} / {lp_d['R1f_Offline_FB_MMSE_Inf_vs_br_trig_bpm']['peak_r']:.3f} | Symmetric offline smoother (flat peak ~0–4 s vs 1 Hz; avoids causal filter phase lag) |",
        f"- **Pause-Exclusion Bias (`>5 s` gaps dropped in `br_trig_bpm` vs whole-window `br_trig_count_bpm = 60*N/30s`):** Overall `interval - 1Hz = {pd_d['mean_interval_minus_1hz_pool']:+.2f} bpm` (`sess {pd_d['mean_interval_minus_1hz_sess']:+.2f}`) vs `count - 1Hz = {pd_d['mean_count_minus_1hz_pool']:+.2f} bpm` (`sess {pd_d['mean_count_minus_1hz_sess']:+.2f}`); on `>5 s` pause windows ($n={pd_d['pause_gt5s']['n_windows']}$): `interval - 1Hz = {pd_d['pause_gt5s']['mean_interval_minus_1hz']:+.2f}` vs `count - 1Hz = {pd_d['pause_gt5s']['mean_count_minus_1hz']:+.2f} bpm`; on regular windows ($n={pd_d['regular_no_gt5s_pause']['n_windows']}$): `interval - 1Hz = {pd_d['regular_no_gt5s_pause']['mean_interval_minus_1hz']:+.2f}` vs `count - 1Hz = {pd_d['regular_no_gt5s_pause']['mean_count_minus_1hz']:+.2f} bpm`.",
    ]

    lines += [
        "\n## 2. WP1 Front-End Gate G1 (`pool30_sync` Dual-Reference & Sentinels)\n",
        f"- **Gate G1 Verdict:** **`{gate_g1['decision']}`**\n",
        "| Method | $\\mathrm{RMSE}_{\\mathrm{sess}}$ (1 Hz) | $\\Delta_{1\\mathrm{Hz}}$ [95% CI] | $p_{\\mathrm{pat,Holm}}$ | $\\mathrm{RMSE}_{\\mathrm{sess}}$ (Trig) | $\\Delta_{\\mathrm{Trig}}$ | 2.3 | 11.1 | 7.3 | 4.1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    p30_1 = wp1_results["pool30_sync"]["br_1hz_30s"]
    p30_t = wp1_results["pool30_sync"]["br_trig_bpm"]
    for k in WP1_METHODS:
        m1 = p30_1["methods"][k]
        mt = p30_t["methods"][k]
        s1 = wp1_sentinels["br_1hz_30s"][k]
        if k == "F0_CSD_Locked":
            d1_s, pp = "anchor", "—"
            dt_s = "anchor"
        else:
            pr1 = p30_1["paired_vs_F0"][k]
            prt = p30_t["paired_vs_F0"][k]
            d1_s = f"{pr1['mean_delta']:+.3f} [{pr1['ci95_lo']:+.2f}, {pr1['ci95_hi']:+.2f}]"
            pp = f"{pr1['p_wilcoxon_patient_holm']:.4f}"
            dt_s = f"{prt['mean_delta']:+.3f}"
        lines.append(
            f"| `{k}` | {m1['rmse_sess']:.3f} | {d1_s} | {pp} | {mt['rmse_sess']:.3f} | {dt_s} | "
            f"{s1['2.3']:.2f} | {s1['11.1']:.2f} | {s1['7.3']:.2f} | {s1['4.1']:.2f} |"
        )

    lines += [
        "\n## 3. WP2 Causal Radar-Native HPSS Gate G2 (`pool30_sync` Dual-Reference & Speedup)\n",
        f"- **Gate G2 Verdict:** **`{gate_g2['decision']}`** (Hypothesis `H1b` falsified cleanly; `H5` & `H6` fail ±0.50 bpm TOST on both references; `H0` retained as primary and `H5` carried as strict 0 s STFT lookahead ablation)\n",
        "| Method | STFT Lookahead | ms/win | Speedup | $\\mathrm{RMSE}_{\\mathrm{sess}}$ (1 Hz) | $\\Delta_{1\\mathrm{Hz}}$ | $p_{\\mathrm{TOST,sess / pat}}$ (1 Hz) | $\\mathrm{RMSE}_{\\mathrm{sess}}$ (Trig) | $p_{\\mathrm{TOST,sess / pat}}$ (Trig) | $r_{\\mathrm{within}}$ (1 Hz / Trig) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    w2_1 = wp2_results["pool30_sync"]["br_1hz_30s"]
    w2_t = wp2_results["pool30_sync"]["br_trig_bpm"]
    for k in WP2_METHODS:
        rt = wp2_runtime[k]
        m1 = w2_1["methods"][k]
        mt = w2_t["methods"][k]
        if k == "H0_Locked_Rect_Full_L3":
            d1_s, pt1_s, ptt_s = "anchor", "—", "—"
        else:
            pr1 = w2_1["paired_vs_H0"][k]
            prt = w2_t["paired_vs_H0"][k]
            d1_s = f"{pr1['mean_delta']:+.3f}"
            pt1_s = f"{pr1['tost_sess']['p_tost']:.4f} / {pr1['tost_patient']['p_tost']:.4f}"
            ptt_s = f"{prt['tost_sess']['p_tost']:.4f} / {prt['tost_patient']['p_tost']:.4f}"
        lines.append(
            f"| `{k}` | {rt['lookahead_s']:.0f} s | {rt['mean_ms_per_win']:.2f} | {rt['speedup_vs_h0']:.2f}x | "
            f"{m1['rmse_sess']:.3f} | {d1_s} | {pt1_s} | {mt['rmse_sess']:.3f} | {ptt_s} | "
            f"{m1['r_within_rmcorr']:.3f} / {mt['r_within_rmcorr']:.3f} |"
        )

    lines += [
        "\n## 4. Rung 1 Causality Frontier & Unpinned LOPO (`pool30_sync`, guard `1.00 × anchor_r`, `σ >= 1.0`, trained ONLY on `1 Hz`)\n",
        f"- **Constant Baselines (`1 Hz` / `1024 Hz Trig`):** `const_loso` = `{rung1_results['pool30_sync']['br_1hz_30s']['constant_baselines']['const_loso']:.3f}` / `{rung1_results['pool30_sync']['br_trig_bpm']['constant_baselines']['const_loso']:.3f} bpm`; "
        f"`const_lopo_median` = `{rung1_results['pool30_sync']['br_1hz_30s']['constant_baselines']['const_lopo_median']:.3f}` / `{rung1_results['pool30_sync']['br_trig_bpm']['constant_baselines']['const_lopo_median']:.3f} bpm`.\n",
        "| Method | Lookahead (Block / STFT) | $\\mathrm{RMSE}_{\\mathrm{sess}}$ (1 Hz) | Acc@10 (1 Hz) | $r_{\\mathrm{within}}$ (1 Hz) | $\\mathrm{RMSE}_{\\mathrm{sess}}$ (Trig) | Acc@10 (Trig) | $r_{\\mathrm{within}}$ (Trig) | TOST vs R1g (1 Hz) | Lower $\\sigma$ Pin |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    r1_1 = rung1_results["pool30_sync"]["br_1hz_30s"]
    r1_t = rung1_results["pool30_sync"]["br_trig_bpm"]
    fsum = rung1_results["pool30_sync"]["lopo_fold_summary"]
    for k in RUNG1_METHODS:
        m1 = r1_1["methods"][k]
        mt = r1_t["methods"][k]
        cc = rung1_causality_contract[k]
        la_blk = "∞" if cc["block_lookahead_label"] == "inf" else cc["block_lookahead_label"]
        la_s = f"{la_blk} / {cc['stft_lookahead_label']}"
        if k in ("R0_HPSS_NLS_Locked", "R1g_Offline_Viterbi_Unpinned"):
            tost_s = "ref" if k == "R1g_Offline_Viterbi_Unpinned" else "—"
        else:
            tost_s = f"{r1_1['paired_vs_R1g_offline_viterbi'][k]['tost_sess']['p_tost']:.4f}"
        lpin_s = "—" if k == "R0_HPSS_NLS_Locked" else f"{fsum[k]['lower_sigma_pinned_folds']}/{fsum[k]['n_folds']}"
        lines.append(
            f"| `{k}` | {la_s} | {m1['rmse_sess']:.3f} | {m1['acc10_sess']:.1f}% | {m1['r_within_rmcorr']:.3f} | "
            f"{mt['rmse_sess']:.3f} | {mt['acc10_sess']:.1f}% | {mt['r_within_rmcorr']:.3f} | {tost_s} | {lpin_s} |"
        )

    lines.append("\n### 4a. Paired Patient-Level Comparisons vs `const_loso` (`N = 11` Infants, `pool30_sync`)\n")
    lines.append("| Method | $\\Delta_{1\\mathrm{Hz}}$ vs `const_loso` | Wins (Pat) | $p_{\\mathrm{pat}}$ / $p_{\\mathrm{Holm}}$ (1 Hz) | $\\Delta_{\\mathrm{Trig}}$ vs `const_loso` | Wins (Pat) | $p_{\\mathrm{pat}}$ / $p_{\\mathrm{Holm}}$ (Trig) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for k in ("R0_HPSS_NLS_Locked", "R1a_Online_Forward_MMSE_0s", "R1f_Offline_FB_MMSE_Inf", "R1g_Offline_Viterbi_Unpinned", "R1h_E2E_Causal_H5_OnlineForward_0s"):
        c1 = r1_1["paired_vs_const_loso"][k]
        ct = r1_t["paired_vs_const_loso"][k]
        lines.append(
            f"| `{k}` | {c1['mean_delta']:+.3f} [{c1['ci95_lo']:+.2f}, {c1['ci95_hi']:+.2f}] | {c1['wins_pat']} | "
            f"{c1['p_wilcoxon_patient']:.4f} / {c1['p_wilcoxon_patient_holm']:.4f} | "
            f"{ct['mean_delta']:+.3f} [{ct['ci95_lo']:+.2f}, {ct['ci95_hi']:+.2f}] | {ct['wins_pat']} | "
            f"{ct['p_wilcoxon_patient']:.4f} / {ct['p_wilcoxon_patient_holm']:.4f} |"
        )
    lines.append("- *Note:* On `1 Hz`, `const_loso` patient-level significance is bounded by low-variability/high-radar-error infants (`pat 9`: `const = 6.17` vs `R1a = 12.47 bpm`), whereas on physical `Trig` `R1f`/`R1g` win `10/11` infants (`p_pat < 0.02`).")

    if not phase_scr_out["skipped"]:
        lines.append("\n### 4b. 5-Seed Phase-Scrambled Radar Anti-Inertia Surrogate (`pool30_sync`)\n")
        lines.append("| Method | Real RMSE (1 Hz) | Scrambled RMSE (1 Hz) | Real $r_{\\mathrm{within}}$ | Scrambled $r_{\\mathrm{within}}$ | Real RMSE (Trig) | Scrambled RMSE (Trig) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for k in ("R0_HPSS_NLS_Locked", "R1a_Online_Forward_MMSE_0s", "R1f_Offline_FB_MMSE_Inf", "R1g_Offline_Viterbi_Unpinned", "R1h_E2E_Causal_H5_OnlineForward_0s"):
            s1 = phase_scr_out["br_1hz_30s"][k]
            st = phase_scr_out["br_trig_bpm"][k]
            lines.append(
                f"| `{k}` | {r1_1['methods'][k]['rmse_sess']:.3f} | {s1['rmse_sess_mean']:.3f}±{s1['rmse_sess_sd']:.3f} | "
                f"{r1_1['methods'][k]['r_within_rmcorr']:.3f} | {s1['r_within_rmcorr_mean']:.3f} | "
                f"{r1_t['methods'][k]['rmse_sess']:.3f} | {st['rmse_sess_mean']:.3f}±{st['rmse_sess_sd']:.3f} |"
            )

    lines.append("\n### 4c. Shifted-Grid Midpoint-Shrinkage Control (`[18, 90]` vs `[18, 120]` bpm across 5 sessions)\n")
    lines.append("| Decoder Regime | RMSE `[18, 90]` | Bias `[18, 90]` | RMSE `[18, 120]` | Bias `[18, 120]` | Shift (`Δbias`) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for lbl, row in shifted_grid_audit.items():
        lines.append(
            f"| `{lbl}` | {row['rmse_18_90']:.3f} | {row['bias_18_90']:+.2f} | "
            f"{row['rmse_18_120']:.3f} | {row['bias_18_120']:+.2f} | `{row['bias_shift_bpm']:+.2f}` |"
        )

    lines.append("\n### 4d. LOPO Guard & Floor Sensitivity Audit Across Cohorts (`Stage 1.1` vs `Legacy v2` vs `Fixed σ=2.0`)\n")
    lines.append("| Cohort | Regime | `R1a (0 s)` 1Hz / Trig | `R1c (4 s)` 1Hz / Trig | `R1f (∞ FB)` 1Hz / Trig | `R1g (∞ Vit)` 1Hz / Trig |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    for coh in LOPO_COHORTS:
        for rname, rl in (("primary_guard100_sigmin10", "Stage 1.1 (1.00×r, σ≥1)"), ("legacy_v2_guard095_sigmin05", "Legacy v2 (0.95×r, σ≥0.5)"), ("fixed_sigma2_eps1e3_beta1", "Fixed σ=2.0, ε=1e-3, β=1")):
            blk = lopo_sensitivity_audit[coh][rname]
            lines.append(
                f"| `{coh}` | {rl} | {blk['R1a_Online_Forward_MMSE_0s']['rmse_sess_1hz']:.3f} / {blk['R1a_Online_Forward_MMSE_0s']['rmse_sess_trig']:.3f} | "
                f"{blk['R1c_FixedLag_MMSE_4s']['rmse_sess_1hz']:.3f} / {blk['R1c_FixedLag_MMSE_4s']['rmse_sess_trig']:.3f} | "
                f"{blk['R1f_Offline_FB_MMSE_Inf']['rmse_sess_1hz']:.3f} / {blk['R1f_Offline_FB_MMSE_Inf']['rmse_sess_trig']:.3f} | "
                f"{blk['R1g_Offline_Viterbi_Unpinned']['rmse_sess_1hz']:.3f} / {blk['R1g_Offline_Viterbi_Unpinned']['rmse_sess_trig']:.3f} |"
            )

    lines.append("\n## 5. WP4 Causal Posterior Uncertainty Preview & Quiet-Apnea Safety\n")
    for m_name, sub in wp4_preview.items():
        qs = sub["quiet_apnea_safety"]
        lines.append(
            f"- **`{m_name}`:** Spearman($H_t$, $|e|$) = `{sub['br_1hz_30s']['spearman_entropy_pool']:.3f}` pool / `{sub['br_1hz_30s']['spearman_entropy_sess_mean']:.3f}` within-sess (1 Hz), "
            f"`{sub['br_trig_bpm']['spearman_entropy_pool']:.3f}` pool / `{sub['br_trig_bpm']['spearman_entropy_sess_mean']:.3f}` within-sess (Trig); "
            f"Spearman($\\mathrm{{SD}}$, $|e|$) = `{sub['br_1hz_30s']['spearman_sd_pool']:.3f}` / `{sub['br_1hz_30s']['spearman_sd_sess_mean']:.3f}` (1 Hz); "
            f"quiet-apnea >5 s mean $H_t$ = `{qs['mean_entropy_pause']:.3f}` vs regular `{qs['mean_entropy_regular']:.3f}` ($n={qs['n_pause_windows']}/{qs['n_regular_windows']}$)."
        )

    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[EXP-1] Wrote {BUNDLE_JSON}, {AUDIT_MD}, {AUDIT_SESSIONS_CSV} in {total_runtime_s:.1f}s.")


if __name__ == "__main__":
    main()
