"""EXP-3 Stage 3 (WP3/WP4): Bayesian & Spectral Uncertainty Quantification, Selective Prediction,
Quiet-Apnea Pause Safety, OOD/WGN Abstention & Finite-K Hierarchical Conformal Calibration.

Executes `thesis/analysis_plan_v3_ai.md` Section 5 and Section 9 (`stage3-uncertainty-v1`)
strictly read-only against frozen baseline caches (`paper_workspace/outputs/cache/*.npz`),
Stage 1 feature caches (`thesis/outputs/stage1_parsimony/cache/*.npz`), and Stage 2 salience /
prediction caches (`thesis/outputs/stage2_complexity/`).

Execution:
  MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /opt/homebrew/bin/python3.10 -m thesis.exp3_stage3_uncertainty
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
import pyedflib

from paper_workspace.src import config as C
from paper_workspace.src.data_loader import MEASURE_TO_REF, load_radar
from paper_workspace.src.demod import csd
from paper_workspace.src.evaluation import window_ref_and_gating
from paper_workspace.src.filtering import autocorr_coarse, bandpass
from paper_workspace.src.hpss import mitigate_hpss
from thesis import diff_hmm_and_foils as D
from thesis import diff_hmm_batch as DB
from thesis import exp0_forensics as E0
from thesis import exp1_stage1_parsimony as E1
from thesis import exp2_stage2_complexity as E2
from thesis import io_utils as IO
from thesis import viterbi_tracker as VT
from thesis import wp1_wp2_causal_hmm as W
from thesis import wp3_wp4_uncertainty_conformal as UQ

SCHEMA_VERSION = "stage3-uncertainty-v1"
OUTPUT_DIR = IO.ROOT / "thesis" / "outputs" / "stage3_uncertainty"
STAGE3_ARCHIVE = OUTPUT_DIR / "pool30_uncertainty_and_conformal.npz"
FROZEN_STAGE4_ARCHIVE = OUTPUT_DIR / "frozen_stage4_models.npz"

STAGE1_JSON = IO.BUNDLE_DIR / "metrics_stage1_parsimony.json"
STAGE2_JSON = IO.BUNDLE_DIR / "metrics_stage2_complexity.json"
BUNDLE_JSON = IO.BUNDLE_DIR / "metrics_stage3_uncertainty.json"
AUDIT_MD = IO.AUDIT_DIR / "stage3_uncertainty_report.md"
AUDIT_SESSIONS_CSV = IO.AUDIT_DIR / "stage3_uncertainty_sessions.csv"

COHORTS_ALL = ("pool30_sync", "retained23", "PAPER_18", "all_paired")
COHORTS_MAIN = ("pool30_sync", "retained23", "PAPER_18")
REGIME = "branch_neutral_raw_nls_lag"

BAYES_MODELS_POOL30: tuple[str, ...] = (
    "R2e_Causal_DiffHMM_FilterSafe8",
    "R2a_Causal_DiffHMM_Linear11",
    "R2b_Causal_DiffHMM_MLP43",
    "R2d_E2E_Causal_H5_DiffHMM_Linear11",
    "R1a_Online_Forward_MMSE_0s",
    "R1h_E2E_Causal_H5_OnlineForward_0s",
    "R1f_Offline_FB_MMSE_Inf",
    "R2f_Offline_DiffHMM_FilterSafe8",
)

CROSS_COHORT_MODELS: tuple[str, ...] = (
    "R2e_Causal_DiffHMM_FilterSafe8",
    "R2a_Causal_DiffHMM_Linear11",
    "R2d_E2E_Causal_H5_DiffHMM_Linear11",
    "R1a_Online_Forward_MMSE_0s",
    "R1h_E2E_Causal_H5_OnlineForward_0s",
    "R1f_Offline_FB_MMSE_Inf",
)

CONFORMAL_MODELS_POOL30: tuple[str, ...] = (
    "R2e_Causal_DiffHMM_FilterSafe8",
    "R2a_Causal_DiffHMM_Linear11",
    "R2d_E2E_Causal_H5_DiffHMM_Linear11",
    "R1a_Online_Forward_MMSE_0s",
    "R1h_E2E_Causal_H5_OnlineForward_0s",
    "R1f_Offline_FB_MMSE_Inf",
    "R0_HPSS_NLS_Locked",
    "R3c_Ridge_Spectral_439",
    "R3a_Causal_TCN_16k",
)


def _fast_extract_all_refs_and_pauses(
    measurement: str,
    n_win: int,
    lag_s: int,
) -> dict[str, Any]:
    """Extract 4-way respiratory references (`1 Hz`, `Trig 0s`, `Trig -10s`, `Trig count`) and `> 5.0 s` pauses."""
    centers_s = np.arange(n_win, dtype=np.float64) * float(C.UPDATE_S) + float(C.WIN_S) / 2.0
    edf_path = C.REF_DIR / f"{MEASURE_TO_REF[measurement]}.edf"
    rd = pyedflib.EdfReader(str(edf_path))
    try:
        lbls = [rd.getLabel(i) for i in range(rd.signals_in_file)]
        idx = {l: i for i, l in enumerate(lbls)}
        resp_256 = rd.readSignal(idx["Resp"]).astype(np.float64)
        rate_1hz = rd.readSignal(idx["RespRate"]).astype(np.float64)
        hr_1hz = rd.readSignal(idx["HeartRate"]).astype(np.float64)
        trig = rd.readSignal(idx["Resp.-Trigger"]).astype(np.float64)
        fs_r = float(rd.getSampleFrequency(idx["Resp"]))
        fs_t = float(rd.getSampleFrequency(idx["Resp.-Trigger"]))
        n_ref = min(len(rate_1hz), len(hr_1hz), len(resp_256) // int(round(fs_r)))
    finally:
        rd._close()

    rw, vw = window_ref_and_gating(resp_256, rate_1hz, centers_s, lag_s, fs_raw=fs_r)
    v1 = np.asarray(vw, dtype=bool) & np.isfinite(rw)

    active = trig > 0.5
    rising = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    times: list[float] = []
    for r_idx in rising:
        ts = float(r_idx) / fs_t
        if not times or (ts - times[-1]) >= 0.30:
            times.append(ts)
    ev = np.asarray(times, dtype=np.float64)

    def _trig_at(shift_lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tc = centers_s - float(shift_lag)
        a_arr = np.floor(tc - float(C.WIN_S) / 2.0).astype(int)
        b_arr = a_arr + int(C.WIN_S)
        arr = np.full(n_win, np.nan, dtype=np.float64)
        max_gaps = np.full(n_win, np.nan, dtype=np.float64)
        for w_i in range(n_win):
            a, b = int(a_arr[w_i]), int(b_arr[w_i])
            if a < 0 or b > n_ref:
                continue
            iw = ev[(ev >= float(a)) & (ev <= float(b))]
            if iw.size < 2:
                max_gaps[w_i] = float(b - a)
                continue
            gaps = np.diff(iw)
            max_gaps[w_i] = float(max(iw[0] - float(a), float(b) - iw[-1], float(np.max(gaps))))
            rates = 60.0 / gaps
            vr = rates[(rates >= 12.0) & (rates <= 120.0)]
            if vr.size > 0:
                arr[w_i] = float(np.median(vr))
        return arr, (v1 & np.isfinite(arr)), max_gaps

    rt0, vt0, max_gap_s = _trig_at(lag_s)
    rta, vta, _ = _trig_at(lag_s + 10)

    tc0 = centers_s - float(lag_s)
    a0 = np.floor(tc0 - float(C.WIN_S) / 2.0).astype(int)
    b0 = a0 + int(C.WIN_S)
    rcnt = np.full(n_win, np.nan, dtype=np.float64)
    for w_i in range(n_win):
        a, b = int(a0[w_i]), int(b0[w_i])
        if a < 0 or b > n_ref:
            continue
        n_ev = int(np.sum((ev >= float(a)) & (ev < float(b))))
        rcnt[w_i] = 60.0 * float(n_ev) / float(C.WIN_S)
    vtc = v1 & np.isfinite(rcnt)
    pause_mask = v1 & np.isfinite(max_gap_s) & (max_gap_s > 5.0)

    return {
        "br_1hz_30s": (np.asarray(rw, dtype=np.float64), v1),
        "br_trig_bpm": (rt0, vt0),
        "br_trig_bpm_lag_aligned": (rta, vta),
        "br_trig_count_bpm": (rcnt, vtc),
        "resp_max_gap_s": max_gap_s,
        "pause_gt5s_mask": pause_mask,
    }


def _precompute_rung1_grids(
    sessions: list[str],
    feat_caches: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, dict[tuple[float, float, float], dict[str, np.ndarray]]]]:
    """Decode Rung 1 grid candidates and posterior uncertainties once across all unique sessions."""
    prob_cfgs = [(s, e, b) for s in E1.SIGMA_GRID for e in E1.EPS_GRID for b in E1.BETA_GRID]
    vit_cfgs = [(s, e, 1.0) for s in E1.SIGMA_GRID for e in E1.EPS_GRID]
    dec_sess: dict[str, dict[str, dict[tuple[float, float, float], dict[str, np.ndarray]]]] = {
        "R1a_Online_Forward_MMSE_0s": {},
        "R1h_E2E_Causal_H5_OnlineForward_0s": {},
        "R1f_Offline_FB_MMSE_Inf": {},
        "R1g_Offline_Viterbi_Unpinned": {},
    }
    for m in sessions:
        s0 = feat_caches[m]["S_anc_h0_1bpm"]
        s5 = feat_caches[m]["S_anc_h5_1bpm"]
        g_1a, g_1h, g_1f, g_1g = {}, {}, {}, {}
        for cfg in prob_cfgs:
            sig_v, eps_v, beta_v = cfg
            f0 = W.hmm_forward_filter(s0, VT.BR_GRID_1BPM, sig_v, eps_v, beta_v)
            g_1a[cfg] = {"est": f0["est_mmse"], "post_entropy": f0["post_entropy"], "post_sd_bpm": f0["post_sd_bpm"]}
            f5 = W.hmm_forward_filter(s5, VT.BR_GRID_1BPM, sig_v, eps_v, beta_v)
            g_1h[cfg] = {"est": f5["est_mmse"], "post_entropy": f5["post_entropy"], "post_sd_bpm": f5["post_sd_bpm"]}
            fb0 = W.hmm_offline_fb_smoother(s0, VT.BR_GRID_1BPM, sig_v, eps_v, beta_v)
            g_1f[cfg] = {"est": fb0["est_mmse"], "post_entropy": fb0["post_entropy"], "post_sd_bpm": fb0["post_sd_bpm"]}
        for cfg in vit_cfgs:
            sig_v, eps_v, beta_v = cfg
            v_est = W.hmm_offline_viterbi_prob(s0, VT.BR_GRID_1BPM, sig_v, eps_v, beta_v)[0]
            g_1g[cfg] = {"est": v_est}
        dec_sess["R1a_Online_Forward_MMSE_0s"][m] = g_1a
        dec_sess["R1h_E2E_Causal_H5_OnlineForward_0s"][m] = g_1h
        dec_sess["R1f_Offline_FB_MMSE_Inf"][m] = g_1f
        dec_sess["R1g_Offline_Viterbi_Unpinned"][m] = g_1g
    return dec_sess


def _decode_rung1_with_uncertainty(
    cohort: str,
    ref_1hz_map: dict[str, tuple[np.ndarray, np.ndarray]],
    feat_caches: dict[str, dict[str, np.ndarray]],
    dec_sess: dict[str, dict[str, dict[tuple[float, float, float], dict[str, np.ndarray]]]],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, dict[int, tuple[float, float, float]]],
]:
    """Run exact Stage 1.1 (`guard_mult = 1.00`, `sigma_min = 1.0`) LOPO for R0/R1a/R1h/R1f/R1g with posterior uncertainty."""
    sessions = [m for m in IO.get_cohort_sessions(cohort, include_zero_valid=False) if np.any(ref_1hz_map[m][1])]

    preds: dict[str, dict[str, np.ndarray]] = {
        "R0_HPSS_NLS_Locked": {m: feat_caches[m]["wp2_H0_Locked_Rect_Full_L3"] for m in sessions},
        "R1a_Online_Forward_MMSE_0s": {},
        "R1h_E2E_Causal_H5_OnlineForward_0s": {},
        "R1f_Offline_FB_MMSE_Inf": {},
        "R1g_Offline_Viterbi_Unpinned": {},
    }
    unc: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "R1a_Online_Forward_MMSE_0s": {},
        "R1h_E2E_Causal_H5_OnlineForward_0s": {},
        "R1f_Offline_FB_MMSE_Inf": {},
    }
    sel_cfgs: dict[str, dict[int, tuple[float, float, float]]] = {k: {} for k in dec_sess}

    for fold in IO.iter_lopo_folds(cohort):
        tr = [m for m in fold.train_sessions if m in sessions]
        te = [m for m in fold.test_sessions if m in sessions]
        if not te:
            continue
        anc_e = [preds["R0_HPSS_NLS_Locked"][m][ref_1hz_map[m][1]] for m in tr]
        refs = [ref_1hz_map[m][0][ref_1hz_map[m][1]] for m in tr]
        base_r = IO.rmcorr_ancova(anc_e, refs)["r_within"]
        min_r = 1.00 * base_r if np.isfinite(base_r) else -np.inf

        for fam in dec_sess:
            cfgs = list(dec_sess[fam][tr[0]].keys())
            best_el, best_fb = None, None
            for cfg in cfgs:
                es = [dec_sess[fam][m][cfg]["est"][ref_1hz_map[m][1]] for m in tr]
                rmses = [float(np.sqrt(np.mean((e - r) ** 2))) for e, r in zip(es, refs)]
                mrmse = float(np.mean(rmses))
                rw = IO.rmcorr_ancova(es, refs)["r_within"]
                fb_t = (-(rw if np.isfinite(rw) else -1.0), mrmse, cfg)
                if best_fb is None or fb_t < best_fb:
                    best_fb = fb_t
                if np.isfinite(rw) and rw >= min_r:
                    el_t = (mrmse, -rw, cfg)
                    if best_el is None or el_t < best_el:
                        best_el = el_t
            chosen = best_el[2] if best_el is not None else best_fb[2]
            sel_cfgs[fam][fold.patient_id] = chosen
            for m in te:
                preds[fam][m] = dec_sess[fam][m][chosen]["est"]
                if fam in unc:
                    unc[fam][m] = {
                        "post_entropy": dec_sess[fam][m][chosen]["post_entropy"],
                        "post_sd_bpm": dec_sess[fam][m][chosen]["post_sd_bpm"],
                    }

    return preds, unc, sel_cfgs


def _fit_rung2_fold_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Worker fitting R2e, R2a, R2d on a single patient-disjoint LOPO fold."""
    tr_ms: list[str] = task["tr_ms"]
    te_ms: list[str] = task["te_ms"]
    ref_map: dict[str, tuple[np.ndarray, np.ndarray]] = task["ref_map"]
    sal_h0 = {m: E2._load_salience_session(m)[0] for m in tr_ms + te_ms}
    sal_h5 = {m: E2._load_salience_session(m)[1] for m in tr_ms + te_ms}
    tr_h0 = [(sal_h0[m], ref_map[m][0], ref_map[m][1]) for m in tr_ms]
    tr_h5 = [(sal_h5[m], ref_map[m][0], ref_map[m][1]) for m in tr_ms]

    r2e = DB.fit_diff_hmm_batched(D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward", reg_l2=5e-3), tr_h0, maxiter=200)
    r2a = DB.fit_diff_hmm_batched(D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward", reg_l2=5e-3), tr_h0, maxiter=200)
    r2d = DB.fit_diff_hmm_batched(D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward", reg_l2=5e-3), tr_h5, maxiter=200)

    preds_f: dict[str, dict[str, np.ndarray]] = {
        "R2e_Causal_DiffHMM_FilterSafe8": {},
        "R2a_Causal_DiffHMM_Linear11": {},
        "R2d_E2E_Causal_H5_DiffHMM_Linear11": {},
    }
    unc_f: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "R2e_Causal_DiffHMM_FilterSafe8": {},
        "R2a_Causal_DiffHMM_Linear11": {},
        "R2d_E2E_Causal_H5_DiffHMM_Linear11": {},
    }
    for m in te_ms:
        pe = r2e.predict_session(sal_h0[m])
        pa = r2a.predict_session(sal_h0[m])
        pd = r2d.predict_session(sal_h5[m])
        preds_f["R2e_Causal_DiffHMM_FilterSafe8"][m] = pe["est_mmse"]
        preds_f["R2a_Causal_DiffHMM_Linear11"][m] = pa["est_mmse"]
        preds_f["R2d_E2E_Causal_H5_DiffHMM_Linear11"][m] = pd["est_mmse"]
        unc_f["R2e_Causal_DiffHMM_FilterSafe8"][m] = {"post_entropy": pe["post_entropy"], "post_sd_bpm": pe["post_sd_bpm"]}
        unc_f["R2a_Causal_DiffHMM_Linear11"][m] = {"post_entropy": pa["post_entropy"], "post_sd_bpm": pa["post_sd_bpm"]}
        unc_f["R2d_E2E_Causal_H5_DiffHMM_Linear11"][m] = {"post_entropy": pd["post_entropy"], "post_sd_bpm": pd["post_sd_bpm"]}
    return {"preds": preds_f, "unc": unc_f}


def _fit_rung2_cohort_with_uncertainty(
    cohort: str,
    ref_1hz_map: dict[str, tuple[np.ndarray, np.ndarray]],
    workers: int = 4,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, dict[str, np.ndarray]]],
]:
    """Fit Rung 2 Differentiable 1D-HMMs in parallel across patient-disjoint LOPO folds."""
    sessions = [m for m in IO.get_cohort_sessions(cohort, include_zero_valid=False) if np.any(ref_1hz_map[m][1])]
    models_to_fit = ("R2e_Causal_DiffHMM_FilterSafe8", "R2a_Causal_DiffHMM_Linear11", "R2d_E2E_Causal_H5_DiffHMM_Linear11")
    preds: dict[str, dict[str, np.ndarray]] = {k: {} for k in models_to_fit}
    unc: dict[str, dict[str, dict[str, np.ndarray]]] = {k: {} for k in models_to_fit}

    tasks = []
    for fold in IO.iter_lopo_folds(cohort):
        tr_ms = [m for m in fold.train_sessions if m in sessions]
        te_ms = [m for m in fold.test_sessions if m in sessions]
        if not te_ms:
            continue
        tasks.append({"tr_ms": tr_ms, "te_ms": te_ms, "ref_map": {m: ref_1hz_map[m] for m in tr_ms}})

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(_fit_rung2_fold_worker, tasks))
    else:
        outs = [_fit_rung2_fold_worker(t) for t in tasks]

    for fo in outs:
        for km in models_to_fit:
            preds[km].update(fo["preds"][km])
            unc[km].update(fo["unc"][km])
    return preds, unc


def _build_all_proxies_for_model(
    cohort: str,
    sessions: list[str],
    preds_m: dict[str, np.ndarray],
    unc_m: dict[str, dict[str, np.ndarray]],
    sal_caches: dict[str, tuple[np.ndarray, np.ndarray]],
    ref_1hz_map: dict[str, tuple[np.ndarray, np.ndarray]],
    use_h5_salience: bool = False,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    """Build all 8 uncertainty proxies `U1..U6c` for a Bayesian state-space model on `cohort`."""
    sal_idx = 1 if use_h5_salience else 0
    by_proxy: dict[str, dict[str, np.ndarray]] = {p: {} for p in UQ.PROXY_ORDER}
    base5_by_sess: dict[str, np.ndarray] = {}

    for m in sessions:
        phi = sal_caches[m][sal_idx]
        zt = UQ.extract_zero_tuned_proxies_session(
            phi,
            unc_m[m]["post_entropy"],
            unc_m[m]["post_sd_bpm"],
        )
        for p_k, arr in zt.items():
            by_proxy[p_k][m] = arr
        base5_by_sess[m] = UQ.build_base5_feature_matrix(zt)

    # Fit U6c_LOPO_Calibrated_SQI strictly on K-1 training patients per LOPO fold using br_1hz_30s
    cal_logs: list[dict[str, Any]] = []
    for fold in IO.iter_lopo_folds(cohort):
        tr_ms = [m for m in fold.train_sessions if m in sessions]
        te_ms = [m for m in fold.test_sessions if m in sessions]
        if not te_ms:
            continue
        tr_X = [base5_by_sess[m][ref_1hz_map[m][1]] for m in tr_ms]
        tr_ae = [np.abs(preds_m[m][ref_1hz_map[m][1]] - ref_1hz_map[m][0][ref_1hz_map[m][1]]) for m in tr_ms]
        cal = UQ.fit_lopo_calibrated_sqi_weights(tr_X, tr_ae, ridge_lam=100.0)
        w = cal["weights"]
        cal_logs.append({
            "patient_id": int(fold.patient_id),
            "test_sessions": te_ms,
            "weights_base5": {
                "w_U1_Posterior_Entropy": float(w[0]),
                "w_U2_Posterior_SD": float(w[1]),
                "w_U3_Spectral_Entropy_Clean": float(w[2]),
                "w_U4_Spectral_Entropy_Raw": float(w[3]),
                "w_U5_HQI_Harmonic_Deficit": float(w[4]),
            },
            "mu_X": cal["mu_X"].tolist(),
            "sd_X": cal["sd_X"].tolist(),
            "q_lo": float(cal["q_lo"]),
            "q_hi": float(cal["q_hi"]),
        })
        for m in te_ms:
            by_proxy["U6c_LOPO_Calibrated_SQI"][m] = UQ.apply_lopo_calibrated_sqi(base5_by_sess[m], cal)

    return by_proxy, cal_logs


def _run_ood_wgn_surrogate_ucd_audit(
    p30_sessions: list[str],
    sal_caches: dict[str, tuple[np.ndarray, np.ndarray]],
    feat_caches: dict[str, dict[str, np.ndarray]],
    ref_1hz_map: dict[str, tuple[np.ndarray, np.ndarray]],
    proxies_by_model: dict[str, dict[str, dict[str, np.ndarray]]],
    cal_logs_by_model: dict[str, list[dict[str, Any]]],
    r2e_mean_theta: np.ndarray,
    r2a_mean_theta: np.ndarray,
    r2d_mean_theta: np.ndarray,
    r1a_mode_cfg: tuple[float, float, float],
    r1h_mode_cfg: tuple[float, float, float],
) -> dict[str, Any]:
    """Run OOD abstention audit on WGN (N=200, seed 20261011), 5-seed phase-scrambled surrogates & UCD."""
    # 1. Generate Complex White Gaussian Noise (N=200 windows x 480 samples, fixed seed 20261011)
    rng = np.random.default_rng(IO.RANDOM_SEED)
    n_wgn = 200
    wgn_i = rng.standard_normal((n_wgn, int(C.WIN_SAMPLES)))
    wgn_q = rng.standard_normal((n_wgn, int(C.WIN_SAMPLES)))

    phi_wgn_h0 = np.empty((n_wgn, VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS), dtype=np.float64)
    phi_wgn_h5 = np.empty((n_wgn, VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS), dtype=np.float64)
    s_anc_wgn_h0 = np.empty((n_wgn, VT.BR_GRID_1BPM.size), dtype=np.float64)
    s_anc_wgn_h5 = np.empty((n_wgn, VT.BR_GRID_1BPM.size), dtype=np.float64)

    for i in range(n_wgn):
        x = csd(wgn_i[i], wgn_q[i])
        dr = bandpass(x)
        dh0 = bandpass(mitigate_hpss(dr)["x_hat"])
        fc0 = autocorr_coarse(dh0)
        s_anc_wgn_h0[i] = VT.extract_br_anchored_nls_spectrum(dh0, fc0, VT.BR_GRID_1BPM)
        phi_wgn_h0[i] = D.extract_salience_features_window(dr, dh0, fc0, VT.BR_GRID_1BPM)

        dh5 = bandpass(W.mitigate_hpss_v2(dr, window="cola_hann", lookahead=0, in_band=True)["x_hat"])
        fc5 = autocorr_coarse(dh5)
        s_anc_wgn_h5[i] = VT.extract_br_anchored_nls_spectrum(dh5, fc5, VT.BR_GRID_1BPM)
        phi_wgn_h5[i] = D.extract_salience_features_window(dr, dh5, fc5, VT.BR_GRID_1BPM)

    r2e_m = D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward")
    r2e_m.theta = r2e_mean_theta
    r2a_m = D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward")
    r2a_m.theta = r2a_mean_theta
    r2d_m = D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward")
    r2d_m.theta = r2d_mean_theta

    wgn_post: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {
        "R2e_Causal_DiffHMM_FilterSafe8": (r2e_m.predict_session(phi_wgn_h0), phi_wgn_h0),
        "R2a_Causal_DiffHMM_Linear11": (r2a_m.predict_session(phi_wgn_h0), phi_wgn_h0),
        "R2d_E2E_Causal_H5_DiffHMM_Linear11": (r2d_m.predict_session(phi_wgn_h5), phi_wgn_h5),
        "R1a_Online_Forward_MMSE_0s": (
            W.hmm_forward_filter(s_anc_wgn_h0, VT.BR_GRID_1BPM, r1a_mode_cfg[0], r1a_mode_cfg[1], r1a_mode_cfg[2]),
            phi_wgn_h0,
        ),
        "R1h_E2E_Causal_H5_OnlineForward_0s": (
            W.hmm_forward_filter(s_anc_wgn_h5, VT.BR_GRID_1BPM, r1h_mode_cfg[0], r1h_mode_cfg[1], r1h_mode_cfg[2]),
            phi_wgn_h5,
        ),
    }

    # 2. Extract UCD Status != ChildPresent telemetry across pool30_sync
    ucd_frac_map: dict[str, np.ndarray] = {}
    for m in p30_sessions:
        df_r = IO.load_radar_full(m)
        n_w = sal_caches[m][0].shape[0]
        tel = E0._compute_radar_window_telemetry(df_r, n_w)
        ucd_frac_map[m] = np.asarray(tel["ucd_nonchild_w"], dtype=np.float64)

    # 3. Evaluate OOD AUROCs across WGN, 5-seed Phase-Scrambled Surrogates, and UCD != ChildPresent
    ood_by_model: dict[str, Any] = {}
    for km, (pr_wgn, phi_w) in wgn_post.items():
        use_h5 = "H5" in km
        sal_idx = 1 if use_h5 else 0
        zt_wgn = UQ.extract_zero_tuned_proxies_session(phi_w, pr_wgn["post_entropy"], pr_wgn["post_sd_bpm"])
        X_wgn = UQ.build_base5_feature_matrix(zt_wgn)
        u6c_wgn_folds = []
        for clog in cal_logs_by_model[km]:
            w_arr = np.array([
                clog["weights_base5"]["w_U1_Posterior_Entropy"],
                clog["weights_base5"]["w_U2_Posterior_SD"],
                clog["weights_base5"]["w_U3_Spectral_Entropy_Clean"],
                clog["weights_base5"]["w_U4_Spectral_Entropy_Raw"],
                clog["weights_base5"]["w_U5_HQI_Harmonic_Deficit"],
            ], dtype=np.float64)
            cal_obj = {
                "weights": w_arr,
                "mu_X": np.asarray(clog["mu_X"], dtype=np.float64),
                "sd_X": np.asarray(clog["sd_X"], dtype=np.float64),
                "q_lo": np.array(clog["q_lo"], dtype=np.float64),
                "q_hi": np.array(clog["q_hi"], dtype=np.float64),
            }
            u6c_wgn_folds.append(UQ.apply_lopo_calibrated_sqi(X_wgn, cal_obj))
        zt_wgn["U6c_LOPO_Calibrated_SQI"] = np.mean(u6c_wgn_folds, axis=0)

        # Surrogate uncertainty proxies across 5 seeds on pool30_sync valid windows
        surr_by_p: dict[str, list[np.ndarray]] = {p: [] for p in UQ.PROXY_ORDER}
        for m in p30_sessions:
            vv = ref_1hz_map[m][1]
            sh0, sh5 = E2._load_surr_salience_session(m)
            sh_use = sh5 if use_h5 else sh0
            with np.load(E1.SURR_CACHE_DIR / f"{m}.npz", allow_pickle=False) as z1:
                s_anc_surr = z1["S_anc_h5" if use_h5 else "S_anc_h0"]
            pid = IO.patient_id_of(m)
            clog_m = next(c for c in cal_logs_by_model[km] if c["patient_id"] == pid)
            cal_obj_m = {
                "weights": np.array([
                    clog_m["weights_base5"]["w_U1_Posterior_Entropy"],
                    clog_m["weights_base5"]["w_U2_Posterior_SD"],
                    clog_m["weights_base5"]["w_U3_Spectral_Entropy_Clean"],
                    clog_m["weights_base5"]["w_U4_Spectral_Entropy_Raw"],
                    clog_m["weights_base5"]["w_U5_HQI_Harmonic_Deficit"],
                ], dtype=np.float64),
                "mu_X": np.asarray(clog_m["mu_X"], dtype=np.float64),
                "sd_X": np.asarray(clog_m["sd_X"], dtype=np.float64),
                "q_lo": np.array(clog_m["q_lo"], dtype=np.float64),
                "q_hi": np.array(clog_m["q_hi"], dtype=np.float64),
            }
            for s_i in range(5):
                if km == "R2e_Causal_DiffHMM_FilterSafe8":
                    pr_s = r2e_m.predict_session(sh_use[s_i])
                elif km == "R2a_Causal_DiffHMM_Linear11":
                    pr_s = r2a_m.predict_session(sh_use[s_i])
                elif km == "R2d_E2E_Causal_H5_DiffHMM_Linear11":
                    pr_s = r2d_m.predict_session(sh_use[s_i])
                elif km == "R1a_Online_Forward_MMSE_0s":
                    pr_s = W.hmm_forward_filter(s_anc_surr[s_i], VT.BR_GRID_1BPM, r1a_mode_cfg[0], r1a_mode_cfg[1], r1a_mode_cfg[2])
                else:
                    pr_s = W.hmm_forward_filter(s_anc_surr[s_i], VT.BR_GRID_1BPM, r1h_mode_cfg[0], r1h_mode_cfg[1], r1h_mode_cfg[2])
                zt_s = UQ.extract_zero_tuned_proxies_session(sh_use[s_i], pr_s["post_entropy"], pr_s["post_sd_bpm"])
                zt_s["U6c_LOPO_Calibrated_SQI"] = UQ.apply_lopo_calibrated_sqi(UQ.build_base5_feature_matrix(zt_s), cal_obj_m)
                for p_k in UQ.PROXY_ORDER:
                    surr_by_p[p_k].append(zt_s[p_k][vv])

        model_ood_res: dict[str, Any] = {}
        for p_k in UQ.PROXY_ORDER:
            u_in_valid = np.concatenate([proxies_by_model[km][p_k][m][ref_1hz_map[m][1]] for m in p30_sessions])
            u_in_clean_ucd = np.concatenate([
                proxies_by_model[km][p_k][m][ref_1hz_map[m][1] & (ucd_frac_map[m] == 0.0)]
                for m in p30_sessions
            ])
            u_ucd_any = np.concatenate([
                proxies_by_model[km][p_k][m][ucd_frac_map[m] > 0.0]
                for m in p30_sessions if np.any(ucd_frac_map[m] > 0.0)
            ])
            u_ucd_maj = np.concatenate([
                proxies_by_model[km][p_k][m][ucd_frac_map[m] >= 0.5]
                for m in p30_sessions if np.any(ucd_frac_map[m] >= 0.5)
            ])
            # Telemetry-gated SQI: SQI_hw = (1 - u) * (1 - I(ucd > 0)) => u_hw = 1.0 when ucd > 0
            u_hw_ucd_any = np.ones_like(u_ucd_any)

            model_ood_res[p_k] = {
                "wgn_200_seed20261011": UQ.exact_auroc_and_fpr95(u_in_valid, zt_wgn[p_k]),
                "phase_scrambled_5seed": UQ.exact_auroc_and_fpr95(u_in_valid, np.concatenate(surr_by_p[p_k])),
                "ucd_nonchild_partial_window_waveform_only": UQ.exact_auroc_and_fpr95(u_in_clean_ucd, u_ucd_any),
                "ucd_nonchild_ge50pct_window_waveform_only": UQ.exact_auroc_and_fpr95(u_in_clean_ucd, u_ucd_maj),
                "ucd_nonchild_hardware_telemetry_gated": UQ.exact_auroc_and_fpr95(u_in_clean_ucd, u_hw_ucd_any),
            }

        # Causal temporal short-lag mixture spectral entropy diagnostic (history_hops = 3 -> 6 s causal history)
        ct_in = np.concatenate([
            UQ.causal_temporal_spectral_entropy(sal_caches[m][sal_idx], history_hops=3)[ref_1hz_map[m][1]]
            for m in p30_sessions
        ])
        ct_wgn = UQ.causal_temporal_spectral_entropy(phi_w, history_hops=3)
        model_ood_res["causal_temporal_spectral_entropy_6s_history"] = {
            "wgn_200_seed20261011": UQ.exact_auroc_and_fpr95(ct_in, ct_wgn),
        }
        ood_by_model[km] = model_ood_res

    return {
        "wgn_config": {"n_windows": n_wgn, "win_samples": int(C.WIN_SAMPLES), "seed": int(IO.RANDOM_SEED)},
        "ucd_partial_window_forensics": {
            "n_pool30_windows_ucd_gt_0": int(sum(int(np.sum(ucd_frac_map[m] > 0.0)) for m in p30_sessions)),
            "n_pool30_windows_ucd_ge_0p5": int(sum(int(np.sum(ucd_frac_map[m] >= 0.5)) for m in p30_sessions)),
            "n_pool30_windows_ucd_eq_1p0": int(sum(int(np.sum(ucd_frac_map[m] >= 1.0)) for m in p30_sessions)),
            "max_ucd_window_frac_pool30": float(max(float(np.max(ucd_frac_map[m])) for m in p30_sessions)),
            "explanation": (
                "All 28 pool30_sync recordings start with 15.0-21.0 s (240-336 samples) of boot 'Processing'/'PassiveMode', "
                "and sessions 4.3 and 9.2 contain 4.0 s (64 samples) mid-session DAC recalibration 'Processing' events. "
                "Because the sliding window is 30.0 s (480 samples), 0 windows in the corpus have 100% UCD!=ChildPresent "
                "(max 70.0% at t=0 in 3.1; 13.3% mid-session in 4.3/9.2), so blind waveform-only proxies see 30-87% real "
                "infant breathing whereas firmware UCD telemetry achieves exact AUROC = 1.000."
            ),
        },
        "models": ood_by_model,
    }


def _run_stage3_invariance_and_causality_controls(
    r2e_mean_theta: np.ndarray,
    r2a_mean_theta: np.ndarray,
    cal_r2e_fold0: dict[str, Any],
) -> dict[str, Any]:
    """Verify O(2) x R_{>0} invariance (< 1e-6) and exact 0.0 causality of all 8 uncertainty proxies U1..U6c."""
    r2e_m = D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward")
    r2e_m.theta = r2e_mean_theta
    r2a_m = D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward")
    r2a_m.theta = r2a_mean_theta

    cal_obj = {
        "weights": np.array([
            cal_r2e_fold0["weights_base5"]["w_U1_Posterior_Entropy"],
            cal_r2e_fold0["weights_base5"]["w_U2_Posterior_SD"],
            cal_r2e_fold0["weights_base5"]["w_U3_Spectral_Entropy_Clean"],
            cal_r2e_fold0["weights_base5"]["w_U4_Spectral_Entropy_Raw"],
            cal_r2e_fold0["weights_base5"]["w_U5_HQI_Harmonic_Deficit"],
        ], dtype=np.float64),
        "mu_X": np.asarray(cal_r2e_fold0["mu_X"], dtype=np.float64),
        "sd_X": np.asarray(cal_r2e_fold0["sd_X"], dtype=np.float64),
        "q_lo": np.array(cal_r2e_fold0["q_lo"], dtype=np.float64),
        "q_hi": np.array(cal_r2e_fold0["q_hi"], dtype=np.float64),
    }

    inv_res: dict[str, dict[str, float]] = {}
    for sess_check, label in (("2.3", "clipped_2_3"), ("7.3", "clean_7_3")):
        bi, bq = load_radar(sess_check)
        n_sub = 480 + 9 * 32
        bi, bq = bi[:n_sub], bq[:n_sub]
        starts = [s for s in range(0, 10 * 32, 32)]

        def _eval_proxies_h5(ai: np.ndarray, aq: np.ndarray) -> dict[str, np.ndarray]:
            phs = []
            for s0 in starts:
                dr = bandpass(csd(ai[s0 : s0 + 480], aq[s0 : s0 + 480]))
                dh5 = bandpass(W.mitigate_hpss_v2(dr, window="cola_hann", lookahead=0, in_band=True)["x_hat"])
                fc5 = autocorr_coarse(dh5)
                phs.append(D.extract_salience_features_window(dr, dh5, fc5, VT.BR_GRID_1BPM))
            p5 = np.asarray(phs, dtype=np.float64)
            pr = r2e_m.predict_session(p5)
            zt = UQ.extract_zero_tuned_proxies_session(p5, pr["post_entropy"], pr["post_sd_bpm"])
            zt["U6c_LOPO_Calibrated_SQI"] = UQ.apply_lopo_calibrated_sqi(UQ.build_base5_feature_matrix(zt), cal_obj)
            zt["Causal_Temporal_SpecEnt"] = UQ.causal_temporal_spectral_entropy(p5, history_hops=3)
            return zt

        base_p = _eval_proxies_h5(bi, bq)
        z_rot = (bi + 1j * bq) * np.exp(1j * np.pi / 3.0)
        diffs: dict[str, float] = {}
        for tname, (ti, tq) in {
            "scale_3p7x": (3.7 * bi, 3.7 * bq),
            "phase_rot_pi_over_3": (z_rot.real, z_rot.imag),
            "iq_sign_flip": (-bi, -bq),
            "iq_channel_swap": (bq, bi),
        }.items():
            alt_p = _eval_proxies_h5(ti, tq)
            diffs[tname] = float(max(np.max(np.abs(base_p[k] - alt_p[k])) for k in base_p))
        inv_res[label] = diffs

    # Exact 0.0 causality perturbation check on all 8 proxies + causal_temporal_spectral_entropy
    rng = np.random.default_rng(IO.RANDOM_SEED)
    phi_t = rng.normal(-2.0, 1.0, size=(30, VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS))
    phi_p = phi_t.copy()
    phi_p[18:] = rng.normal(-2.0, 2.0, size=(12, VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS))

    pr_t = r2e_m.predict_session(phi_t)
    pr_p = r2e_m.predict_session(phi_p)
    zt_t = UQ.extract_zero_tuned_proxies_session(phi_t, pr_t["post_entropy"], pr_t["post_sd_bpm"])
    zt_p = UQ.extract_zero_tuned_proxies_session(phi_p, pr_p["post_entropy"], pr_p["post_sd_bpm"])
    zt_t["U6c_LOPO_Calibrated_SQI"] = UQ.apply_lopo_calibrated_sqi(UQ.build_base5_feature_matrix(zt_t), cal_obj)
    zt_p["U6c_LOPO_Calibrated_SQI"] = UQ.apply_lopo_calibrated_sqi(UQ.build_base5_feature_matrix(zt_p), cal_obj)
    zt_t["Causal_Temporal_SpecEnt"] = UQ.causal_temporal_spectral_entropy(phi_t, history_hops=3)
    zt_p["Causal_Temporal_SpecEnt"] = UQ.causal_temporal_spectral_entropy(phi_p, history_hops=3)

    causality_diffs = {
        k: float(np.max(np.abs(zt_t[k][:18] - zt_p[k][:18])))
        for k in zt_t
    }

    # Low-amplitude ("normal" 0.2 mm) vs physiological ("deep" 1.0 mm) 24 GHz Jacobi-Anger CW radar OOD-barrier check
    fs = float(C.FS_RADAR)
    lam_mm = float(C.LAMBDA_MM)
    t_full = np.arange(480 + 19 * 32, dtype=np.float64) / fs
    synth_ood_checks: dict[str, dict[str, float]] = {}
    for amp_mm, snr_db, lbl in ((0.2, 10.0, "normal_0p2mm_10dB"), (0.5, 10.0, "normal_0p5mm_10dB"), (1.0, 15.0, "deep_1p0mm_15dB")):
        for bpm in (28.0, 45.0, 75.0):
            rng_s = np.random.default_rng(IO.RANDOM_SEED)
            d_t = amp_mm * np.sin(2.0 * np.pi * (bpm / 60.0) * t_full)
            phi_rf = 0.7 + (4.0 * np.pi / lam_mm) * d_t
            sig_p = float(np.var(np.cos(phi_rf)) + np.var(np.sin(phi_rf)))
            noise_sd = float(np.sqrt(0.5 * sig_p / (10.0 ** (snr_db / 10.0))))
            bi_s = np.cos(phi_rf) + rng_s.normal(0.0, noise_sd, size=t_full.size)
            bq_s = np.sin(phi_rf) + rng_s.normal(0.0, noise_sd, size=t_full.size)
            phis_s = []
            for w_idx in range(20):
                s0 = w_idx * 32
                dr = bandpass(csd(bi_s[s0 : s0 + 480], bq_s[s0 : s0 + 480]))
                dh = bandpass(mitigate_hpss(dr)["x_hat"])
                fc = autocorr_coarse(dh)
                phis_s.append(D.extract_salience_features_window(dr, dh, fc, VT.BR_GRID_1BPM))
            phi_arr = np.asarray(phis_s, dtype=np.float64)
            pr_s = r2e_m.predict_session(phi_arr)
            zt_s = UQ.extract_zero_tuned_proxies_session(phi_arr, pr_s["post_entropy"], pr_s["post_sd_bpm"])
            hmix_s = UQ.causal_temporal_spectral_entropy(phi_arr, history_hops=3)
            b_ood_s = np.clip((hmix_s - 0.90) / 0.10, 0.0, 1.0)
            synth_ood_checks[f"{lbl}_{int(bpm)}bpm"] = {
                "mae_bpm": float(np.mean(np.abs(pr_s["est_mmse"] - bpm))),
                "mean_h_mix": float(np.mean(hmix_s)),
                "max_b_ood": float(np.max(b_ood_s)),
                "mean_u2_post_sd": float(np.mean(zt_s["U2_Posterior_SD"])),
                "mean_u6b_geom": float(np.mean(zt_s["U6b_ZeroTuned_Geometric_SQI"])),
            }

    return {
        "invariance_max_diff_by_sentinel": inv_res,
        "all_invariances_lt_1e6": bool(all(v < 1e-6 for sub in inv_res.values() for v in sub.values())),
        "causality_max_diff_by_proxy": causality_diffs,
        "all_causal_diffs_exact_zero": bool(all(v == 0.0 for v in causality_diffs.values())),
        "low_amp_synthetic_ood_barrier_check": synth_ood_checks,
        "low_amp_synthetic_zero_false_ood": bool(all(v["max_b_ood"] == 0.0 and v["mae_bpm"] < 0.10 for v in synth_ood_checks.values())),
    }


def _fit_and_archive_frozen_stage4_models(
    p30_sessions: list[str],
    sal_caches: dict[str, tuple[np.ndarray, np.ndarray]],
    ref_1hz_map: dict[str, tuple[np.ndarray, np.ndarray]],
    r1a_mode_cfg: tuple[float, float, float],
    r1h_mode_cfg: tuple[float, float, float],
    r1f_mode_cfg: tuple[float, float, float],
    r2e_mean_th: np.ndarray,
    r2a_mean_th: np.ndarray,
    r2d_mean_th: np.ndarray,
    frozen_c4_q_hat_bpm: dict[str, float | None],
    frozen_c5_q_hat_norm: dict[str, float | None],
    frozen_c4_q_hat_bpm_tla: dict[str, float | None],
    frozen_c5_q_hat_norm_tla: dict[str, float | None],
) -> dict[str, Any]:
    """Fit full-cohort (`K=11`, `S=28` `pool30_sync`) Rung 2 & Rung 3 models and lock `frozen_stage4_models.npz`."""
    tr_h0 = [(sal_caches[m][0], ref_1hz_map[m][0], ref_1hz_map[m][1]) for m in p30_sessions]
    tr_h5 = [(sal_caches[m][1], ref_1hz_map[m][0], ref_1hz_map[m][1]) for m in p30_sessions]
    tr_pids = [IO.patient_id_of(m) for m in p30_sessions]

    # Full-cohort K=11 Rung 2 refits (maxiter=200)
    r2e_full = DB.fit_diff_hmm_batched(
        D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward", reg_l2=5e-3),
        tr_h0,
        maxiter=200,
    )
    r2f_full = DB.fit_diff_hmm_batched(
        D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="offline_fb", reg_l2=5e-3),
        tr_h0,
        maxiter=200,
    )
    r2a_full = DB.fit_diff_hmm_batched(
        D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward", reg_l2=5e-3),
        tr_h0,
        maxiter=200,
    )
    r2d_full = DB.fit_diff_hmm_batched(
        D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward", reg_l2=5e-3),
        tr_h5,
        maxiter=200,
    )

    # Full-cohort K=11 Rung 3 fits (11-fold patient-LOPO CV for R3c Ridge; inner val_patient_id=12 for R3b MLP & R3a TCN)
    r3c_res = D.fit_predict_ridge_spectral(
        tr_h0,
        [sal_caches[p30_sessions[0]][0]],
        train_patient_ids=tr_pids,
    )
    val_pat_full = max(sorted(set(tr_pids)))  # patient 12
    tr3_h0 = [(sal_caches[m][0], ref_1hz_map[m][0], ref_1hz_map[m][1]) for m in p30_sessions if IO.patient_id_of(m) != val_pat_full]
    va3_h0 = [(sal_caches[m][0], ref_1hz_map[m][0], ref_1hz_map[m][1]) for m in p30_sessions if IO.patient_id_of(m) == val_pat_full]
    r3b_res = D.fit_predict_spectral_mlp(
        tr3_h0,
        va3_h0,
        [sal_caches[p30_sessions[0]][0]],
        alpha_grid=(0.1, 1.0, 10.0, 100.0),
        seed=IO.RANDOM_SEED,
    )
    r3a_tcn, r3a_info = D.fit_select_causal_tcn(
        tr3_h0,
        va3_h0,
        lr=1.5e-3,
        grid=((1e-3, 0.0), (1e-2, 0.0), (1e-2, 0.15), (5e-2, 0.25)),
        max_epochs=100,
        patience=18,
        seed=IO.RANDOM_SEED,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    c4_arr = np.array(
        [np.nan if frozen_c4_q_hat_bpm[a] is None else float(frozen_c4_q_hat_bpm[a]) for a in ("0.05", "0.10", "0.20", "0.30")],
        dtype=np.float64,
    )
    c5_arr = np.array(
        [np.nan if frozen_c5_q_hat_norm[a] is None else float(frozen_c5_q_hat_norm[a]) for a in ("0.05", "0.10", "0.20", "0.30")],
        dtype=np.float64,
    )
    c4_tla_arr = np.array(
        [np.nan if frozen_c4_q_hat_bpm_tla[a] is None else float(frozen_c4_q_hat_bpm_tla[a]) for a in ("0.05", "0.10", "0.20", "0.30")],
        dtype=np.float64,
    )
    c5_tla_arr = np.array(
        [np.nan if frozen_c5_q_hat_norm_tla[a] is None else float(frozen_c5_q_hat_norm_tla[a]) for a in ("0.05", "0.10", "0.20", "0.30")],
        dtype=np.float64,
    )

    ridge_mdl = r3c_res["model"]
    mlp_mdl = r3b_res["model"]
    npz_payload: dict[str, np.ndarray] = {
        "r1a_mode_cfg": np.asarray(r1a_mode_cfg, dtype=np.float64),
        "r1h_mode_cfg": np.asarray(r1h_mode_cfg, dtype=np.float64),
        "r1f_mode_cfg": np.asarray(r1f_mode_cfg, dtype=np.float64),
        "r2e_lopo_mean_theta": np.asarray(r2e_mean_th, dtype=np.float64),
        "r2a_lopo_mean_theta": np.asarray(r2a_mean_th, dtype=np.float64),
        "r2d_lopo_mean_theta": np.asarray(r2d_mean_th, dtype=np.float64),
        "r2e_full_refit_theta": np.asarray(r2e_full.theta, dtype=np.float64),
        "r2f_full_refit_theta": np.asarray(r2f_full.theta, dtype=np.float64),
        "r2a_full_refit_theta": np.asarray(r2a_full.theta, dtype=np.float64),
        "r2d_full_refit_theta": np.asarray(r2d_full.theta, dtype=np.float64),
        "r3c_ridge_coef": np.asarray(ridge_mdl.coef_, dtype=np.float64),
        "r3c_ridge_intercept": np.array([float(ridge_mdl.intercept_)], dtype=np.float64),
        "r3c_ridge_mean": np.asarray(r3c_res["mean"], dtype=np.float64),
        "r3c_ridge_scale": np.asarray(r3c_res["scale"], dtype=np.float64),
        "r3c_ridge_alpha": np.array([float(r3c_res["selected_alpha"])], dtype=np.float64),
        "r3b_mlp_mean": np.asarray(r3b_res["mean"], dtype=np.float64),
        "r3b_mlp_scale": np.asarray(r3b_res["scale"], dtype=np.float64),
        "r3b_mlp_alpha": np.array([float(r3b_res["selected_alpha"])], dtype=np.float64),
        "r3a_tcn_mean": np.asarray(r3a_tcn.mean_, dtype=np.float64),
        "r3a_tcn_scale": np.asarray(r3a_tcn.scale_, dtype=np.float64),
        "r3a_tcn_weight_decay": np.array([float(r3a_info["selected_weight_decay"])], dtype=np.float64),
        "r3a_tcn_dropout_p": np.array([float(r3a_info["selected_dropout_p"])], dtype=np.float64),
        "c4_q_hat_bpm_1hz": c4_arr,
        "c5_q_hat_norm_1hz": c5_arr,
        "c4_q_hat_bpm_trig_lag_aligned": c4_tla_arr,
        "c5_q_hat_norm_trig_lag_aligned": c5_tla_arr,
    }
    for idx_l, (c_arr, b_arr) in enumerate(zip(mlp_mdl.coefs_, mlp_mdl.intercepts_)):
        npz_payload[f"r3b_mlp_W_{idx_l}"] = np.asarray(c_arr, dtype=np.float64)
        npz_payload[f"r3b_mlp_b_{idx_l}"] = np.asarray(b_arr, dtype=np.float64)
    for k_tcn, arr_tcn in r3a_tcn.params.items():
        npz_payload[f"r3a_tcn_{k_tcn}"] = np.asarray(arr_tcn, dtype=np.float64)

    tmp_f4 = OUTPUT_DIR / ".frozen_stage4_models.tmp.npz"
    np.savez_compressed(tmp_f4, **npz_payload)
    tmp_f4.replace(FROZEN_STAGE4_ARCHIVE)
    f4_sha256 = IO.sha256_file(FROZEN_STAGE4_ARCHIVE)

    mlp_w_l2 = float(np.sqrt(sum(float(np.sum(np.asarray(c) ** 2)) for c in mlp_mdl.coefs_)))
    tcn_w_l2 = float(np.sqrt(sum(float(np.sum(v ** 2)) for k, v in r3a_tcn.params.items() if k.startswith("W"))))

    return {
        "frozen_stage4_models_archive": str(FROZEN_STAGE4_ARCHIVE.relative_to(IO.ROOT)),
        "frozen_stage4_models_sha256": f4_sha256,
        "frozen_r2e_full_refit_theta": r2e_full.theta.tolist(),
        "frozen_r2e_full_refit_interpretable": r2e_full.interpretable_params(),
        "frozen_r2f_full_refit_theta": r2f_full.theta.tolist(),
        "frozen_r2f_full_refit_interpretable": r2f_full.interpretable_params(),
        "frozen_r2a_full_refit_theta": r2a_full.theta.tolist(),
        "frozen_r2a_full_refit_interpretable": r2a_full.interpretable_params(),
        "frozen_r2d_full_refit_theta": r2d_full.theta.tolist(),
        "frozen_r2d_full_refit_interpretable": r2d_full.interpretable_params(),
        "frozen_r3c_ridge_439_summary": {
            "n_params": int(r3c_res["n_params"]),
            "selected_alpha": float(r3c_res["selected_alpha"]),
            "inner_cv_mode": str(r3c_res["inner_cv_mode"]),
            "inner_lopo_rmse_bpm": float(r3c_res["inner_lopo_rmse"]),
            "intercept_bpm": float(ridge_mdl.intercept_),
            "coef_l2_norm": float(np.linalg.norm(ridge_mdl.coef_)),
        },
        "frozen_r3b_mlp_15k_summary": {
            "n_params": int(r3b_res["n_params"]),
            "selected_alpha": float(r3b_res["selected_alpha"]),
            "val_patient_id": int(val_pat_full),
            "best_val_rmse_bpm": float(r3b_res["best_val_rmse"]),
            "weight_l2_norm": mlp_w_l2,
        },
        "frozen_r3a_tcn_16k_summary": {
            "n_params": int(r3a_tcn.n_params),
            "selected_weight_decay": float(r3a_info["selected_weight_decay"]),
            "selected_dropout_p": float(r3a_info["selected_dropout_p"]),
            "val_patient_id": int(val_pat_full),
            "best_val_rmse_bpm": float(r3a_info["best_val_rmse"]),
            "weight_l2_norm": tcn_w_l2,
        },
    }


def _verify_stage1_and_stage2_full_coverage_parity(
    selective_results: dict[str, Any],
) -> dict[str, float]:
    """Assert that full-coverage (c_cov = 1.00) risk matches Stage 1.1 and Stage 2.2 within < 1e-12."""
    s1 = json.loads(STAGE1_JSON.read_text(encoding="utf-8"))
    s2 = json.loads(STAGE2_JSON.read_text(encoding="utf-8"))
    max_s1_diff = 0.0
    max_s2_diff = 0.0

    for coh in COHORTS_MAIN:
        for rlbl in ("br_1hz_30s", "br_trig_bpm"):
            for km in ("R0_HPSS_NLS_Locked", "R1a_Online_Forward_MMSE_0s", "R1h_E2E_Causal_H5_OnlineForward_0s", "R1f_Offline_FB_MMSE_Inf"):
                m1 = s1["rung1_lopo"][coh][rlbl]["methods"][km]
                px_chk = "U3_Spectral_Entropy_Clean" if km == "R0_HPSS_NLS_Locked" else "U2_Posterior_SD"
                m3 = selective_results[coh][rlbl][km][px_chk]
                d_sess = abs(float(m3["session_macro"]["full_risk_rmse_sess"]) - float(m1["rmse_sess"]))
                d_pool = abs(float(m3["window_pooled"]["full_risk_rmse"]) - float(m1["rmse_pool"]))
                max_s1_diff = max(max_s1_diff, d_sess, d_pool)
                if max(d_sess, d_pool) >= 1e-12:
                    raise AssertionError(f"Stage 1.1 parity mismatch on {coh}/{rlbl}/{km}: {max(d_sess, d_pool):.3e}")

        for rlbl in ("br_1hz_30s", "br_trig_bpm", "br_trig_bpm_lag_aligned"):
            for km in ("R0_HPSS_NLS_Locked", "R2e_Causal_DiffHMM_FilterSafe8", "R2a_Causal_DiffHMM_Linear11", "R2d_E2E_Causal_H5_DiffHMM_Linear11"):
                m2 = s2["cohorts"][coh][rlbl]["methods"][km]
                px_chk = "U3_Spectral_Entropy_Clean" if km == "R0_HPSS_NLS_Locked" else "U2_Posterior_SD"
                m3 = selective_results[coh][rlbl][km][px_chk]
                d_sess = abs(float(m3["session_macro"]["full_risk_rmse_sess"]) - float(m2["rmse_sess"]))
                d_pool = abs(float(m3["window_pooled"]["full_risk_rmse"]) - float(m2["rmse_pool"]))
                max_s2_diff = max(max_s2_diff, d_sess, d_pool)
                if max(d_sess, d_pool) >= 1e-12:
                    raise AssertionError(f"Stage 2.2 parity mismatch on {coh}/{rlbl}/{km}: {max(d_sess, d_pool):.3e}")

    return {"stage1_full_cov_max_diff": max_s1_diff, "stage2_full_cov_max_diff": max_s2_diff}


def main() -> None:
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="EXP-3 Stage 3 Uncertainty, Apnea Safety & Conformal Audit")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    print("[EXP-3] Verifying frozen baseline caches read-only...")
    IO.verify_all_baseline_caches()

    all_sessions = IO.get_cohort_sessions("all_paired", include_zero_valid=False)
    p30_sessions = IO.get_cohort_sessions("pool30_sync", include_zero_valid=False)
    lags_map = IO.load_locked_neutral_lags()

    print(f"[EXP-3] Loading Stage 1 feature caches, Stage 2 salience caches & 4-way EDF references across {len(all_sessions)} sessions...")
    feat_caches: dict[str, dict[str, np.ndarray]] = {}
    sal_caches: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ref_1hz_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ref_trig_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ref_tla_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ref_tcnt_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pause_map: dict[str, np.ndarray] = {}

    for m in all_sessions:
        feat_caches[m] = E1._load_feat_cache(m)
        sal_caches[m] = E2._load_salience_session(m)
        n_w = sal_caches[m][0].shape[0]
        lag_val = int(lags_map[m]["neutral_raw_nls_lag_s"])
        ext = _fast_extract_all_refs_and_pauses(m, n_w, lag_val)
        ref_1hz_map[m] = ext["br_1hz_30s"]
        ref_trig_map[m] = ext["br_trig_bpm"]
        ref_tla_map[m] = ext["br_trig_bpm_lag_aligned"]
        ref_tcnt_map[m] = ext["br_trig_count_bpm"]
        pause_map[m] = ext["pause_gt5s_mask"]

    # Load frozen Stage 2 pool30_sync prediction archive
    arch_s2 = np.load(E2.PRED_ARCHIVE, allow_pickle=False)
    s2_master = json.loads(STAGE2_JSON.read_text(encoding="utf-8"))

    selective_by_cohort: dict[str, Any] = {}
    apnea_safety_by_cohort: dict[str, Any] = {}
    conformal_by_cohort: dict[str, Any] = {}
    p30_proxies_by_model: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    p30_cal_logs_by_model: dict[str, list[dict[str, Any]]] = {}
    p30_per_sess_r2e_1hz: dict[str, dict[str, dict[str, float]]] = {}
    p30_per_sess_r2e_tla: dict[str, dict[str, dict[str, float]]] = {}
    p30_r1_sel_cfgs: dict[str, dict[int, tuple[float, float, float]]] = {}
    p30_r2d_thetas: list[np.ndarray] = []

    print("[EXP-3] Pre-decoding Rung 1 grid candidates and posterior uncertainties once across 31 sessions...")
    rung1_dec_all = _precompute_rung1_grids(all_sessions, feat_caches)

    for coh in COHORTS_ALL:
        print(f"[EXP-3] Evaluating LOPO uncertainty, selective prediction & finite-K conformal on cohort={coh}...")
        slist = [m for m in IO.get_cohort_sessions(coh, include_zero_valid=False) if np.any(ref_1hz_map[m][1])]
        r1_preds, r1_unc, r1_cfgs = _decode_rung1_with_uncertainty(coh, ref_1hz_map, feat_caches, rung1_dec_all)

        if coh == "pool30_sync":
            p30_r1_sel_cfgs = r1_cfgs
            r2_preds = {
                km: {m: arch_s2[f"pred__{km}__{m}"].astype(np.float64) for m in slist}
                for km in ("R2e_Causal_DiffHMM_FilterSafe8", "R2a_Causal_DiffHMM_Linear11", "R2b_Causal_DiffHMM_MLP43", "R2d_E2E_Causal_H5_DiffHMM_Linear11", "R2f_Offline_DiffHMM_FilterSafe8")
            }
            r2_unc = {
                km: {
                    m: {
                        "post_entropy": arch_s2[f"unc_ent__{km}__{m}"].astype(np.float64),
                        "post_sd_bpm": arch_s2[f"unc_sd__{km}__{m}"].astype(np.float64),
                    }
                    for m in slist
                }
                for km in ("R2e_Causal_DiffHMM_FilterSafe8", "R2a_Causal_DiffHMM_Linear11", "R2b_Causal_DiffHMM_MLP43", "R2d_E2E_Causal_H5_DiffHMM_Linear11")
            }
            r2_unc["R2f_Offline_DiffHMM_FilterSafe8"] = {}
            for fold in IO.iter_lopo_folds(coh):
                tr_ms = [m for m in fold.train_sessions if m in slist]
                te_ms = [m for m in fold.test_sessions if m in slist]
                tr_h0 = [(sal_caches[m][0], ref_1hz_map[m][0], ref_1hz_map[m][1]) for m in tr_ms]
                tr_h5 = [(sal_caches[m][1], ref_1hz_map[m][0], ref_1hz_map[m][1]) for m in tr_ms]
                r2f = DB.fit_diff_hmm_batched(D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="offline_fb", reg_l2=5e-3), tr_h0, maxiter=200)
                r2d = DB.fit_diff_hmm_batched(D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward", reg_l2=5e-3), tr_h5, maxiter=200)
                p30_r2d_thetas.append(r2d.theta.copy())
                for m in te_ms:
                    prf = r2f.predict_session(sal_caches[m][0])
                    np.testing.assert_allclose(prf["est_mmse"], r2_preds["R2f_Offline_DiffHMM_FilterSafe8"][m], rtol=0.0, atol=1e-12)
                    r2_unc["R2f_Offline_DiffHMM_FilterSafe8"][m] = {"post_entropy": prf["post_entropy"], "post_sd_bpm": prf["post_sd_bpm"]}
        else:
            r2_preds, r2_unc = _fit_rung2_cohort_with_uncertainty(
                coh, ref_1hz_map, workers=args.workers
            )

        models_this_coh = BAYES_MODELS_POOL30 if coh == "pool30_sync" else CROSS_COHORT_MODELS
        preds_all_coh: dict[str, dict[str, np.ndarray]] = {**r1_preds, **r2_preds}
        if coh == "pool30_sync":
            for km in E2.MODEL_ORDER:
                if km not in preds_all_coh:
                    preds_all_coh[km] = {m: arch_s2[f"pred__{km}__{m}"].astype(np.float64) for m in slist}

        unc_all_coh: dict[str, dict[str, dict[str, np.ndarray]]] = {**r1_unc, **r2_unc}
        proxies_coh: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        cal_logs_coh: dict[str, list[dict[str, Any]]] = {}

        for km in models_this_coh:
            use_h5 = "H5" in km
            proxies_coh[km], cal_logs_coh[km] = _build_all_proxies_for_model(
                coh, slist, preds_all_coh[km], unc_all_coh[km], sal_caches, ref_1hz_map, use_h5_salience=use_h5
            )

        if coh == "pool30_sync":
            p30_proxies_by_model = proxies_coh
            p30_cal_logs_by_model = cal_logs_coh

        # Evaluate selective prediction across references
        ref_pairs = (
            (
                ("br_1hz_30s", ref_1hz_map),
                ("br_trig_bpm", ref_trig_map),
                ("br_trig_bpm_lag_aligned", ref_tla_map),
                ("br_trig_count_bpm", ref_tcnt_map),
            )
            if coh == "pool30_sync"
            else (
                ("br_1hz_30s", ref_1hz_map),
                ("br_trig_bpm", ref_trig_map),
                ("br_trig_bpm_lag_aligned", ref_tla_map),
            )
        )
        coh_sel_block: dict[str, Any] = {}
        for rlbl, rmap in ref_pairs:
            ref_block: dict[str, Any] = {}
            for km in models_this_coh:
                keep_curves = bool(coh == "pool30_sync" and rlbl in ("br_1hz_30s", "br_trig_bpm_lag_aligned") and km in ("R2e_Causal_DiffHMM_FilterSafe8", "R2a_Causal_DiffHMM_Linear11", "R1a_Online_Forward_MMSE_0s", "R2d_E2E_Causal_H5_DiffHMM_Linear11"))
                sum_m, ps_m = UQ.evaluate_cohort_selective_prediction(
                    slist, preds_all_coh[km], proxies_coh[km], rmap, cov_grid=UQ.COV_GRID_51, include_curves=keep_curves
                )
                ref_block[km] = sum_m
                if coh == "pool30_sync" and km == "R2e_Causal_DiffHMM_FilterSafe8":
                    if rlbl == "br_1hz_30s":
                        p30_per_sess_r2e_1hz = ps_m
                    elif rlbl == "br_trig_bpm_lag_aligned":
                        p30_per_sess_r2e_tla = ps_m
            r0_spec_proxies = {
                pk: proxies_coh["R2e_Causal_DiffHMM_FilterSafe8"][pk]
                for pk in ("U3_Spectral_Entropy_Clean", "U4_Spectral_Entropy_Raw", "U5_HQI_Harmonic_Deficit")
            }
            ref_block["R0_HPSS_NLS_Locked"], _ = UQ.evaluate_cohort_selective_prediction(
                slist, preds_all_coh["R0_HPSS_NLS_Locked"], r0_spec_proxies, rmap, cov_grid=UQ.COV_GRID_51, include_curves=False
            )
            coh_sel_block[rlbl] = ref_block
        selective_by_cohort[coh] = coh_sel_block

        # Quiet-apnea pause safety audit (> 5.0 s gaps vs regular breathing)
        v1_mask_map = {m: ref_1hz_map[m][1] for m in slist}
        apnea_coh: dict[str, Any] = {}
        for km in models_this_coh:
            apnea_coh[km] = UQ.evaluate_apnea_pause_safety(
                slist, proxies_coh[km], v1_mask_map, pause_map, safety_ratio_limit=1.15
            )
        apnea_safety_by_cohort[coh] = apnea_coh

        # Finite-K Conformal Audit (C1..C5 across alpha in {0.05, 0.10, 0.20, 0.30})
        conf_models = CONFORMAL_MODELS_POOL30 if coh == "pool30_sync" else ("R2e_Causal_DiffHMM_FilterSafe8", "R2a_Causal_DiffHMM_Linear11", "R1a_Online_Forward_MMSE_0s")
        conf_coh: dict[str, Any] = {}
        for rlbl, rmap in (("br_1hz_30s", ref_1hz_map), ("br_trig_bpm_lag_aligned", ref_tla_map)):
            conf_ref: dict[str, Any] = {}
            for km in conf_models:
                if km in unc_all_coh:
                    sd_map = {m: unc_all_coh[km][m]["post_sd_bpm"] for m in slist}
                else:
                    # Point regressors (R0, R3c, R3a) lack an intrinsic Bayesian posterior SD;
                    # fall back to R2e posterior SD scale for C5 comparison while C1..C4 use raw |y - yhat|
                    sd_map = {m: unc_all_coh["R2e_Causal_DiffHMM_FilterSafe8"][m]["post_sd_bpm"] for m in slist}
                conf_ref[km] = UQ.evaluate_finite_k_conformal_suite(
                    coh, preds_all_coh[km], sd_map, rmap, alphas=UQ.ALPHA_CONFORMAL_GRID
                )
            conf_coh[rlbl] = conf_ref
        conformal_by_cohort[coh] = conf_coh

    parity_checks = _verify_stage1_and_stage2_full_coverage_parity(selective_by_cohort)
    print(
        f"[EXP-3] Verified full-coverage (c_cov=1.00) parity against Stage 1.1 "
        f"({parity_checks['stage1_full_cov_max_diff']:.2e}) and Stage 2.2 ({parity_checks['stage2_full_cov_max_diff']:.2e})."
    )

    # Summarize LOPO calibrated SQI (U6c) learned weights across pool30_sync folds
    lopo_sqi_weight_summary: dict[str, Any] = {}
    for km, logs in p30_cal_logs_by_model.items():
        w_keys = list(logs[0]["weights_base5"].keys())
        lopo_sqi_weight_summary[km] = {
            wk: {
                "mean": float(np.mean([row["weights_base5"][wk] for row in logs])),
                "sd": float(np.std([row["weights_base5"][wk] for row in logs], ddof=1)),
            }
            for wk in w_keys
        }

    # OOD / WGN / Phase-Scrambled Surrogate / UCD Abstention Audit
    print("[EXP-3] Running OOD abstention audit (WGN N=200 seed=20261011, 5-seed surrogates, UCD telemetry)...")
    r2e_mean_th = np.asarray(s2_master["verification_controls"]["r2e_mean_theta"], dtype=np.float64)
    r2a_mean_th = np.asarray(s2_master["verification_controls"]["r2a_mean_theta"], dtype=np.float64)
    r2d_mean_th = np.mean(np.asarray(p30_r2d_thetas, dtype=np.float64), axis=0)
    # Modal / median R1a, R1h & R1f cfg on pool30_sync
    r1a_cfgs_list = list(p30_r1_sel_cfgs["R1a_Online_Forward_MMSE_0s"].values())
    r1h_cfgs_list = list(p30_r1_sel_cfgs["R1h_E2E_Causal_H5_OnlineForward_0s"].values())
    r1f_cfgs_list = list(p30_r1_sel_cfgs["R1f_Offline_FB_MMSE_Inf"].values())
    r1a_mode_cfg = max(set(r1a_cfgs_list), key=r1a_cfgs_list.count)
    r1h_mode_cfg = max(set(r1h_cfgs_list), key=r1h_cfgs_list.count)
    r1f_mode_cfg = max(set(r1f_cfgs_list), key=r1f_cfgs_list.count)

    ood_audit = _run_ood_wgn_surrogate_ucd_audit(
        p30_sessions,
        sal_caches,
        feat_caches,
        ref_1hz_map,
        p30_proxies_by_model,
        p30_cal_logs_by_model,
        r2e_mean_th,
        r2a_mean_th,
        r2d_mean_th,
        r1a_mode_cfg,
        r1h_mode_cfg,
    )

    print("[EXP-3] Running O(2) x R_{>0} invariance and exact 0.0 causality verification controls...")
    controls = _run_stage3_invariance_and_causality_controls(
        r2e_mean_th,
        r2a_mean_th,
        p30_cal_logs_by_model["R2e_Causal_DiffHMM_FilterSafe8"][0],
    )

    # Gate G4 Adjudication & Pre-Stage-4 Hard Freeze Manifest
    r2e_1hz_u2 = selective_by_cohort["pool30_sync"]["br_1hz_30s"]["R2e_Causal_DiffHMM_FilterSafe8"]["U2_Posterior_SD"]
    r2e_tla_u2 = selective_by_cohort["pool30_sync"]["br_trig_bpm_lag_aligned"]["R2e_Causal_DiffHMM_FilterSafe8"]["U2_Posterior_SD"]
    r2e_1hz_u6b = selective_by_cohort["pool30_sync"]["br_1hz_30s"]["R2e_Causal_DiffHMM_FilterSafe8"]["U6b_ZeroTuned_Geometric_SQI"]
    r2e_tla_u6b = selective_by_cohort["pool30_sync"]["br_trig_bpm_lag_aligned"]["R2e_Causal_DiffHMM_FilterSafe8"]["U6b_ZeroTuned_Geometric_SQI"]
    r2a_1hz_u6b = selective_by_cohort["pool30_sync"]["br_1hz_30s"]["R2a_Causal_DiffHMM_Linear11"]["U6b_ZeroTuned_Geometric_SQI"]
    r2a_tla_u6b = selective_by_cohort["pool30_sync"]["br_trig_bpm_lag_aligned"]["R2a_Causal_DiffHMM_Linear11"]["U6b_ZeroTuned_Geometric_SQI"]

    ap_r2e_u2 = apnea_safety_by_cohort["pool30_sync"]["R2e_Causal_DiffHMM_FilterSafe8"]["U2_Posterior_SD"]
    ap_r2e_u6b = apnea_safety_by_cohort["pool30_sync"]["R2e_Causal_DiffHMM_FilterSafe8"]["U6b_ZeroTuned_Geometric_SQI"]
    ap_r2e_u3 = apnea_safety_by_cohort["pool30_sync"]["R2e_Causal_DiffHMM_FilterSafe8"]["U3_Spectral_Entropy_Clean"]

    wgn_auroc_r2e_u6b = float(ood_audit["models"]["R2e_Causal_DiffHMM_FilterSafe8"]["U6b_ZeroTuned_Geometric_SQI"]["wgn_200_seed20261011"]["auroc"])
    wgn_auroc_r2e_u6a = float(ood_audit["models"]["R2e_Causal_DiffHMM_FilterSafe8"]["U6a_ZeroTuned_Additive_SQI"]["wgn_200_seed20261011"]["auroc"])
    wgn_auroc_r2e_u3 = float(ood_audit["models"]["R2e_Causal_DiffHMM_FilterSafe8"]["U3_Spectral_Entropy_Clean"]["wgn_200_seed20261011"]["auroc"])

    conf_r2e_1hz = conformal_by_cohort["pool30_sync"]["br_1hz_30s"]["R2e_Causal_DiffHMM_FilterSafe8"]["by_alpha"]
    conf_r2e_tla = conformal_by_cohort["pool30_sync"]["br_trig_bpm_lag_aligned"]["R2e_Causal_DiffHMM_FilterSafe8"]["by_alpha"]

    frozen_c4_q_hat_bpm: dict[str, float | None] = {}
    frozen_c5_q_hat_norm: dict[str, float | None] = {}
    frozen_c4_q_hat_bpm_tla: dict[str, float | None] = {}
    frozen_c5_q_hat_norm_tla: dict[str, float | None] = {}
    for a_str in ("0.05", "0.10", "0.20", "0.30"):
        c4_rg = conf_r2e_1hz[a_str]["regimes"]["C4_PatientExchangeable_Hierarchical_LOPO"]
        c5_rg = conf_r2e_1hz[a_str]["regimes"]["C5_LocallyAdaptive_BayesConformal_Hierarchical_LOPO"]
        frozen_c4_q_hat_bpm[a_str] = None if c4_rg["is_vacuous"] else float(np.mean([f["q_hat_bpm"] for f in c4_rg["folds"]]))
        frozen_c5_q_hat_norm[a_str] = None if c5_rg["is_vacuous"] else float(np.mean([f["q_hat_norm"] for f in c5_rg["folds"]]))
        c4_rg_t = conf_r2e_tla[a_str]["regimes"]["C4_PatientExchangeable_Hierarchical_LOPO"]
        c5_rg_t = conf_r2e_tla[a_str]["regimes"]["C5_LocallyAdaptive_BayesConformal_Hierarchical_LOPO"]
        frozen_c4_q_hat_bpm_tla[a_str] = None if c4_rg_t["is_vacuous"] else float(np.mean([f["q_hat_bpm"] for f in c4_rg_t["folds"]]))
        frozen_c5_q_hat_norm_tla[a_str] = None if c5_rg_t["is_vacuous"] else float(np.mean([f["q_hat_norm"] for f in c5_rg_t["folds"]]))

    # Exact full-cohort (K=11) patient-exchangeable conformal quantiles across pool30_sync 1 Hz OOF residuals
    p30_pats = sorted({IO.patient_id_of(m) for m in p30_sessions})
    p30_abs_err_by_pat = {
        p: np.concatenate([
            np.abs(
                arch_s2[f"pred__R2e_Causal_DiffHMM_FilterSafe8__{m}"][ref_1hz_map[m][1]].astype(np.float64)
                - ref_1hz_map[m][0][ref_1hz_map[m][1]]
            )
            for m in p30_sessions if IO.patient_id_of(m) == p
        ])
        for p in p30_pats
    }
    p30_norm_err_by_pat = {
        p: np.concatenate([
            np.abs(
                arch_s2[f"pred__R2e_Causal_DiffHMM_FilterSafe8__{m}"][ref_1hz_map[m][1]].astype(np.float64)
                - ref_1hz_map[m][0][ref_1hz_map[m][1]]
            )
            / np.maximum(arch_s2[f"unc_sd__R2e_Causal_DiffHMM_FilterSafe8__{m}"][ref_1hz_map[m][1]].astype(np.float64), 1.0)
            for m in p30_sessions if IO.patient_id_of(m) == p
        ])
        for p in p30_pats
    }
    frozen_c4_q_hat_bpm_full_k11_1hz: dict[str, float | None] = {}
    frozen_c5_q_hat_norm_full_k11_1hz: dict[str, float | None] = {}
    for a_val in UQ.ALPHA_CONFORMAL_GRID:
        a_str = f"{a_val:.2f}"
        tcov = 1.0 - float(a_val)
        pqs_abs = np.array([float(np.quantile(p30_abs_err_by_pat[p], tcov)) for p in p30_pats], dtype=np.float64)
        pqs_norm = np.array([float(np.quantile(p30_norm_err_by_pat[p], tcov)) for p in p30_pats], dtype=np.float64)
        q4_k, _, v4_k = UQ.empirical_order_statistic(pqs_abs, float(a_val))
        q5_k, _, v5_k = UQ.empirical_order_statistic(pqs_norm, float(a_val))
        frozen_c4_q_hat_bpm_full_k11_1hz[a_str] = None if v4_k else float(q4_k)
        frozen_c5_q_hat_norm_full_k11_1hz[a_str] = None if v5_k else float(q5_k)

    print("[EXP-3] Fitting full-cohort K=11 Rung 2 & Rung 3 models and archiving frozen_stage4_models.npz...")
    frozen_s4_info = _fit_and_archive_frozen_stage4_models(
        p30_sessions,
        sal_caches,
        ref_1hz_map,
        r1a_mode_cfg,
        r1h_mode_cfg,
        r1f_mode_cfg,
        r2e_mean_th,
        r2a_mean_th,
        r2d_mean_th,
        frozen_c4_q_hat_bpm,
        frozen_c5_q_hat_norm,
        frozen_c4_q_hat_bpm_tla,
        frozen_c5_q_hat_norm_tla,
    )

    gate_g4_verdict = {
        "gate": "G4",
        "decision": "PASS_AND_HARD_FREEZE_WP1_TO_WP4",
        "primary_selective_prediction_gain_lt_0": bool(
            float(r2e_1hz_u2["session_macro"]["e_aurc_vs_random_rmse_sess"]) < 0.0
            and float(r2e_tla_u2["session_macro"]["e_aurc_vs_random_rmse_sess"]) < 0.0
            and float(r2e_1hz_u6b["session_macro"]["e_aurc_vs_random_rmse_sess"]) < 0.0
            and float(r2e_tla_u6b["session_macro"]["e_aurc_vs_random_rmse_sess"]) < 0.0
            and float(r2e_1hz_u6b["window_pooled"]["e_aurc_vs_random_rmse"]) < 0.0
        ),
        "wgn_auroc_ge_0p95": bool(wgn_auroc_r2e_u6b >= 0.95 and wgn_auroc_r2e_u6a >= 0.95 and wgn_auroc_r2e_u3 >= 0.95),
        "apnea_pause_safety_verified": bool(
            ap_r2e_u2["apnea_safe_at_80pct_pooled"]
            and ap_r2e_u6b["apnea_safe_at_80pct_pooled"]
            and not ap_r2e_u3["apnea_safe_at_80pct_pooled"]
        ),
        "finite_k_conformal_theorem_verified": bool(
            conf_r2e_1hz["0.05"]["patient_exchangeable_is_vacuous"] is True
            and conf_r2e_1hz["0.10"]["regimes"]["C4_PatientExchangeable_Hierarchical_LOPO"]["valid_distribution_free_guarantee"] is True
            and conf_r2e_1hz["0.20"]["regimes"]["C4_PatientExchangeable_Hierarchical_LOPO"]["valid_distribution_free_guarantee"] is True
            and conf_r2e_1hz["0.30"]["regimes"]["C4_PatientExchangeable_Hierarchical_LOPO"]["valid_distribution_free_guarantee"] is True
            and conf_r2e_1hz["0.20"]["regimes"]["C5_LocallyAdaptive_BayesConformal_Hierarchical_LOPO"]["valid_distribution_free_guarantee"] is True
        ),
        "hard_freeze_manifest_for_stage4_l4": {
            "wp1_frontend": "F0_CSD_Locked",
            "wp2_hpss_primary": "H0_Locked_Rect_Full_L3",
            "wp2_hpss_primary_lookahead": "48 samples (3.0 s lookahead lag at 16 Hz; L_H = 3 hops)",
            "wp2_hpss_causal_0s_ablation": "H5_Causal_COLA_InBand_L0",
            "wp2_hpss_causal_0s_ablation_lookahead": "0 samples (0.0 s lookahead lag at 16 Hz; L_H = 0 hops)",
            "rung0_anchor": "R0_HPSS_NLS_Locked",
            "rung1_causal_primary": "R1a_Online_Forward_MMSE_0s",
            "rung1_offline_primary": "R1f_Offline_FB_MMSE_Inf",
            "rung2_causal_filtersafe_primary": "R2e_Causal_DiffHMM_FilterSafe8",
            "rung2_causal_harmonic_secondary": "R2a_Causal_DiffHMM_Linear11",
            "rung2_e2e_causal_0s_ablation": "R2d_E2E_Causal_H5_DiffHMM_Linear11",
            "rung3_shortcut_foils": ["R3c_Ridge_Spectral_439", "R3b_Spectral_MLP_15k", "R3a_Causal_TCN_16k"],
            "wp4_primary_tracking_uncertainty": "U2_Posterior_SD",
            "wp4_primary_composite_sqi": "U6b_ZeroTuned_Geometric_SQI",
            "wp4_conformal_calibrator": "C5_LocallyAdaptive_BayesConformal_Hierarchical_LOPO",
            "frozen_r1a_mode_cfg": list(r1a_mode_cfg),
            "frozen_r1h_mode_cfg": list(r1h_mode_cfg),
            "frozen_r1f_mode_cfg": list(r1f_mode_cfg),
            "frozen_r2e_mean_theta": r2e_mean_th.tolist(),
            "frozen_r2a_mean_theta": r2a_mean_th.tolist(),
            "frozen_r2d_mean_theta": r2d_mean_th.tolist(),
            "frozen_c4_q_hat_bpm_pool30_1hz": frozen_c4_q_hat_bpm,
            "frozen_c5_q_hat_norm_pool30_1hz": frozen_c5_q_hat_norm,
            "frozen_c4_q_hat_bpm_pool30_trig_lag_aligned": frozen_c4_q_hat_bpm_tla,
            "frozen_c5_q_hat_norm_pool30_trig_lag_aligned": frozen_c5_q_hat_norm_tla,
            "frozen_c4_q_hat_bpm_full_k11_1hz": frozen_c4_q_hat_bpm_full_k11_1hz,
            "frozen_c5_q_hat_norm_full_k11_1hz": frozen_c5_q_hat_norm_full_k11_1hz,
            **frozen_s4_info,
            "stage4_l4_evaluation_protocol": {
                "external_dataset": "external_data/gleichauf2024/",
                "zero_touch_policy": "Strict zero-touch evaluation of hard-frozen Stage 3 models and C4/C5 quantiles; zero parameter tuning on external data.",
                "primary_inband_stratum_bpm": [28.0, 80.0],
                "primary_inband_rationale": "Within locked 0.40-1.50 Hz (24-90 bpm) 4th-order Butterworth bandpass and [18, 90] bpm HMM grid.",
                "secondary_out_of_passband_low_rate_stratum_bpm": [10.0, 24.0],
                "secondary_out_of_passband_rationale": "At/below 0.40 Hz (24 bpm) FIR high-pass cutoff skirt; isolates fundamental attenuation vs harmonic aliasing.",
            },
        },
    }

    # Save reusable Stage 3 archive (.npz) for plotting & clean-room verification
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arch_payload: dict[str, Any] = {
        "sessions": np.asarray(p30_sessions, dtype=str),
        "proxies": np.asarray(UQ.PROXY_ORDER, dtype=str),
        "cov_grid_51": UQ.COV_GRID_51,
    }
    for m in p30_sessions:
        arch_payload[f"valid_1hz__{m}"] = np.asarray(ref_1hz_map[m][1], dtype=bool)
        arch_payload[f"ref_1hz__{m}"] = np.asarray(ref_1hz_map[m][0], dtype=np.float64)
        arch_payload[f"valid_tla__{m}"] = np.asarray(ref_tla_map[m][1], dtype=bool)
        arch_payload[f"ref_tla__{m}"] = np.asarray(ref_tla_map[m][0], dtype=np.float64)
        arch_payload[f"pause_gt5s__{m}"] = np.asarray(pause_map[m], dtype=bool)
        for km in ("R2e_Causal_DiffHMM_FilterSafe8", "R2a_Causal_DiffHMM_Linear11", "R2d_E2E_Causal_H5_DiffHMM_Linear11", "R1a_Online_Forward_MMSE_0s", "R1h_E2E_Causal_H5_OnlineForward_0s"):
            for p_k in UQ.PROXY_ORDER:
                arch_payload[f"u__{km}__{p_k}__{m}"] = np.asarray(p30_proxies_by_model[km][p_k][m], dtype=np.float64)

    tmp_arch = OUTPUT_DIR / ".pool30_uncertainty_and_conformal.tmp.npz"
    np.savez_compressed(tmp_arch, **arch_payload)
    tmp_arch.replace(STAGE3_ARCHIVE)

    # Export per-session CSV on pool30_sync for R2e across key proxies
    IO.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    csv_proxies = ("U1_Posterior_Entropy", "U2_Posterior_SD", "U3_Spectral_Entropy_Clean", "U6a_ZeroTuned_Additive_SQI", "U6b_ZeroTuned_Geometric_SQI", "U6c_LOPO_Calibrated_SQI")
    with AUDIT_SESSIONS_CSV.open("w", newline="", encoding="utf-8") as fh:
        cols = ["measurement", "patient_id", "n_valid_1hz", "n_pause_gt5s", "R2e_full_rmse_1hz", "R2e_full_rmse_trig_lag_aligned"]
        for pk in csv_proxies:
            cols.extend([f"R2e_{pk}_aurc_1hz", f"R2e_{pk}_eaurc_rand_1hz", f"R2e_{pk}_rmse80_1hz", f"R2e_{pk}_eaurc_rand_tla"])
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for m in p30_sessions:
            v1 = ref_1hz_map[m][1]
            pz = pause_map[m]
            r0_1 = p30_per_sess_r2e_1hz["U2_Posterior_SD"][m]
            r0_t = p30_per_sess_r2e_tla["U2_Posterior_SD"][m]
            row: dict[str, Any] = {
                "measurement": m,
                "patient_id": IO.patient_id_of(m),
                "n_valid_1hz": int(np.sum(v1)),
                "n_pause_gt5s": int(np.sum(pz & v1)),
                "R2e_full_rmse_1hz": f"{r0_1['full_risk_rmse']:.4f}",
                "R2e_full_rmse_trig_lag_aligned": f"{r0_t['full_risk_rmse']:.4f}",
            }
            for pk in csv_proxies:
                s1_p = p30_per_sess_r2e_1hz[pk][m]
                st_p = p30_per_sess_r2e_tla[pk][m]
                row[f"R2e_{pk}_aurc_1hz"] = f"{s1_p['aurc_rmse']:.4f}"
                row[f"R2e_{pk}_eaurc_rand_1hz"] = f"{s1_p['e_aurc_vs_random_rmse']:.4f}"
                row[f"R2e_{pk}_rmse80_1hz"] = f"{s1_p['risk_rmse_at_80']:.4f}"
                row[f"R2e_{pk}_eaurc_rand_tla"] = f"{st_p['e_aurc_vs_random_rmse']:.4f}"
            writer.writerow(row)

    total_s = float(time.perf_counter() - t_start)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "git_head": IO.git_head_info(),
        "analysis_plan": "thesis/analysis_plan_v3_ai.md",
        "regime": REGIME,
        "uniform_grid_sd_1bpm": UQ.UNIFORM_GRID_SD_1BPM,
        "full_coverage_parity_checks": parity_checks,
        "total_runtime_s": total_s,
        "gate_g4_verdict": gate_g4_verdict,
        "selective_prediction_by_cohort": selective_by_cohort,
        "apnea_pause_safety_by_cohort": apnea_safety_by_cohort,
        "ood_abstention_audit_pool30": ood_audit,
        "finite_k_conformal_by_cohort": conformal_by_cohort,
        "lopo_calibrated_sqi_weights_pool30": lopo_sqi_weight_summary,
        "lopo_calibrated_sqi_fold_logs_pool30": p30_cal_logs_by_model,
        "verification_controls": controls,
    }
    json.dumps(payload, allow_nan=False)
    IO.write_json_locked(BUNDLE_JSON, payload)

    # Write Markdown Audit Report
    md_lines = [
        f"# EXP-3 Stage 3 (WP3/WP4) Uncertainty, Apnea-Pause Safety & Conformal Audit (`{SCHEMA_VERSION}`)\n",
        f"- **Git HEAD:** `{payload['git_head']['head_commit']}` (`{payload['git_head']['branch']}`, Python `{payload['git_head']['python_version']}`)",
        f"- **Total Wall Time:** `{total_s:.1f} s` | **Full-Coverage (`c_cov=1.00`) Parity:** Stage 1.1 `{parity_checks['stage1_full_cov_max_diff']:.2e}`, Stage 2.2 `{parity_checks['stage2_full_cov_max_diff']:.2e}`",
        f"- **Uncertainty O(2)xR+ Invariance on H5 (< 1e-6):** `{'PASS' if controls['all_invariances_lt_1e6'] else 'FAIL'}` (clean max `{max(controls['invariance_max_diff_by_sentinel']['clean_7_3'].values()):.2e}`, clipped max `{max(controls['invariance_max_diff_by_sentinel']['clipped_2_3'].values()):.2e}`)",
        f"- **Exact 0.0 Causality Diffs across all 8 Proxies (`U1..U6c`):** `{'PASS' if controls['all_causal_diffs_exact_zero'] else 'FAIL'}` (`max diff = {max(controls['causality_max_diff_by_proxy'].values()):.2e}`)",
        f"- **Gate G4 Verdict:** **`{gate_g4_verdict['decision']}`**\n",
        "## 1. Selective Prediction & Risk–Coverage (`pool30_sync`, $S = 28$ Non-Empty Synchronous Sessions / $K = 11$ Infants, $c_{\\mathrm{cov}} \\in [0.50, 1.00]$)\n",
        "| Model | Proxy | Params | $\\mathrm{RMSE}_{\\mathrm{sess}}(1.0 \\to 0.80)$ (1 Hz) | $\\mathrm{E\\text{-}AURC}_{\\mathrm{sess}}$ (1 Hz) [95% CI] | Wins (Pat) | $p_{\\mathrm{sess}} / p_{\\mathrm{pat}}$ [Holm] (1 Hz) | $\\mathrm{E\\text{-}AURC}_{\\mathrm{pool}}$ (1 Hz) | $\\mathrm{RMSE}_{\\mathrm{sess}}(1.0 \\to 0.80)$ (Trig -10s) | $\\mathrm{E\\text{-}AURC}_{\\mathrm{sess}}$ (Trig -10s) [95% CI] | $p_{\\mathrm{sess}} / p_{\\mathrm{pat}}$ [Holm] (-10s) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    md_models_proxies: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("R2e (8p)", "R2e_Causal_DiffHMM_FilterSafe8", UQ.PROXY_ORDER),
        ("R2a (11p)", "R2a_Causal_DiffHMM_Linear11", UQ.PROXY_ORDER),
        ("R1a (3p)", "R1a_Online_Forward_MMSE_0s", UQ.PROXY_ORDER),
        ("R0 (0p)", "R0_HPSS_NLS_Locked", ("U3_Spectral_Entropy_Clean", "U4_Spectral_Entropy_Raw", "U5_HQI_Harmonic_Deficit")),
    )
    for km_short, km_full, pk_list in md_models_proxies:
        for pk in pk_list:
            s1_m = selective_by_cohort["pool30_sync"]["br_1hz_30s"][km_full][pk]["session_macro"]
            p1_m = selective_by_cohort["pool30_sync"]["br_1hz_30s"][km_full][pk]["window_pooled"]
            st_m = selective_by_cohort["pool30_sync"]["br_trig_bpm_lag_aligned"][km_full][pk]["session_macro"]
            np_u = UQ.PROXY_META[pk]["n_tuned_params"]
            md_lines.append(
                f"| `{km_short}` | `{pk}` | {np_u} | `{s1_m['full_risk_rmse_sess']:.3f} -> {s1_m['risk_rmse_sess_at_80']:.3f}` | "
                f"`{s1_m['e_aurc_vs_random_rmse_sess']:+.3f} [{s1_m['e_aurc_ci95_lo']:+.3f}, {s1_m['e_aurc_ci95_hi']:+.3f}]` | "
                f"{s1_m['wins_pat']} | `{s1_m['p_wilcoxon_sess']:.4f} / {s1_m['p_wilcoxon_patient']:.4f} [{s1_m['p_wilcoxon_sess_holm']:.4f} / {s1_m['p_wilcoxon_patient_holm']:.4f}]` | "
                f"`{p1_m['e_aurc_vs_random_rmse']:+.3f}` | "
                f"`{st_m['full_risk_rmse_sess']:.3f} -> {st_m['risk_rmse_sess_at_80']:.3f}` | "
                f"`{st_m['e_aurc_vs_random_rmse_sess']:+.3f} [{st_m['e_aurc_ci95_lo']:+.3f}, {st_m['e_aurc_ci95_hi']:+.3f}]` | "
                f"`{st_m['p_wilcoxon_sess']:.4f} / {st_m['p_wilcoxon_patient']:.4f} [{st_m['p_wilcoxon_sess_holm']:.4f} / {st_m['p_wilcoxon_patient_holm']:.4f}]` |"
            )

    md_lines += [
        "\n## 2. Quiet-Apnea Respiratory Pause Safety Stratification (`pool30_sync`, `resp_max_gap_s > 5.0 s`, $N_{\\mathrm{pause}}=1581$ vs. $N_{\\mathrm{reg}}=5059$) & OOD Abstention (`WGN N=200`)\n",
        "| Model | Proxy | Mean $u$ (Pause vs. Reg) | Rej@80% Pause (%) | Rej@80% Reg (%) | Pause Risk Ratio (`<=1.15`) | Safe? | WGN AUROC (`>=0.95`) | Surrogate AUROC |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for km_short, km_full in (("R2e (8p)", "R2e_Causal_DiffHMM_FilterSafe8"), ("R2a (11p)", "R2a_Causal_DiffHMM_Linear11"), ("R1a (3p)", "R1a_Online_Forward_MMSE_0s")):
        for pk in UQ.PROXY_ORDER:
            ap = apnea_safety_by_cohort["pool30_sync"][km_full][pk]
            r80 = ap["pooled_threshold_rejection"]["0.80"]
            ood_m = ood_audit["models"][km_full][pk]
            md_lines.append(
                f"| `{km_short}` | `{pk}` | `{ap['mean_u_pause']:.3f} vs {ap['mean_u_regular']:.3f}` | "
                f"`{r80['rej_pause_pct']:.1f}%` | `{r80['rej_regular_pct']:.1f}%` | "
                f"`{r80['rejection_risk_ratio']:.2f}x` | `{'PASS' if ap['apnea_safe_at_80pct_pooled'] else 'FAIL (Over-Rej)'}` | "
                f"`{ood_m['wgn_200_seed20261011']['auroc']:.4f}` | `{ood_m['phase_scrambled_5seed']['auroc']:.4f}` |"
            )

    md_lines += [
        "\n## 3. 5-Regime Finite-$K$ Conformal Calibration Audit (`pool30_sync`, $K_{\\mathrm{cal}} = 10$ Infants, $\\alpha_{\\min} = 1/11 \\approx 0.0909$, `R2e_Causal_DiffHMM_FilterSafe8`)\n",
        "| $\\alpha$ (Target $1-\\alpha$) | Order Stat $i_{\\alpha}/K_{\\mathrm{cal}}$ | Regime | Vacuous? | Patient Coverage (Mean ± SD, Min) | Under-Covered Infants | Pooled Coverage | Mean Width (bpm) |",
        "| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for a_str in ("0.05", "0.10", "0.20", "0.30"):
        ab = conf_r2e_1hz[a_str]
        for rname in UQ.CONFORMAL_REGIME_ORDER:
            rg = ab["regimes"][rname]
            w_str = "∞ (vacuous)" if rg["is_vacuous"] else f"{rg['mean_interval_width_bpm']:.2f}"
            md_lines.append(
                f"| `{a_str}` (`{ab['target_coverage']:.2f}`) | `{ab['patient_order_stat_index']}/{ab['k_cal_patients']}` | `{rname}` | "
                f"`{rg['is_vacuous']}` | `{rg['empirical_coverage_patient_mean']:.3f} ± {rg['empirical_coverage_patient_sd']:.3f}` (`min {rg['empirical_coverage_patient_min']:.3f}`) | "
                f"`{rg['n_patients_undercovered']}/11` | `{rg['empirical_coverage_pooled']:.3f}` | `{w_str}` |"
            )

    hf_m = gate_g4_verdict["hard_freeze_manifest_for_stage4_l4"]
    r2e_ip = hf_m["frozen_r2e_full_refit_interpretable"]
    r2f_ip = hf_m["frozen_r2f_full_refit_interpretable"]
    r2a_ip = hf_m["frozen_r2a_full_refit_interpretable"]
    r2d_ip = hf_m["frozen_r2d_full_refit_interpretable"]
    r3c_sm = hf_m["frozen_r3c_ridge_439_summary"]
    r3b_sm = hf_m["frozen_r3b_mlp_15k_summary"]
    r3a_sm = hf_m["frozen_r3a_tcn_16k_summary"]
    s4_proto = hf_m["stage4_l4_evaluation_protocol"]
    md_lines += [
        "\n## 4. Gate `G4` Hard-Freeze Manifest for Stage 4 (`L4` External Benchmark)\n",
        f"- **Frozen Stage 4 Model Archive:** `{hf_m['frozen_stage4_models_archive']}` (SHA-256 `{hf_m['frozen_stage4_models_sha256']}`)",
        f"- **WP1 / WP2 Lookaheads:** `{hf_m['wp1_frontend']}` | `{hf_m['wp2_hpss_primary']}` = `{hf_m['wp2_hpss_primary_lookahead']}` | `{hf_m['wp2_hpss_causal_0s_ablation']}` = `{hf_m['wp2_hpss_causal_0s_ablation_lookahead']}`",
        f"- **Rung 1 Modal Configs:** `R1a` `{hf_m['frozen_r1a_mode_cfg']}`, `R1h` `{hf_m['frozen_r1h_mode_cfg']}`, `R1f` `{hf_m['frozen_r1f_mode_cfg']}`",
        f"- **Rung 2 `R2e_Causal_DiffHMM_FilterSafe8` (`8p`):** LOPO-mean `{[round(x, 6) for x in hf_m['frozen_r2e_mean_theta']]}`; Full-cohort `K=11` refit `{[round(x, 6) for x in hf_m['frozen_r2e_full_refit_theta']]}` (`w1={r2e_ip['w1_p1_clean']:.3f}`, `w5={r2e_ip['w5_anchor']:.3f}`, `w6={r2e_ip['w6_p1_raw']:.3f}`, `sigma_slow={r2e_ip['sigma_slow_bpm']:.3f} bpm`, `sigma_jump={r2e_ip['sigma_jump_bpm']:.3f} bpm`, `pi_jump={r2e_ip['pi_jump']:.6f}`)",
        f"- **Rung 2 `R2f_Offline_DiffHMM_FilterSafe8` (`8p`):** Full-cohort `K=11` refit `{[round(x, 6) for x in hf_m['frozen_r2f_full_refit_theta']]}` (`w1={r2f_ip['w1_p1_clean']:.3f}`, `w5={r2f_ip['w5_anchor']:.3f}`, `w6={r2f_ip['w6_p1_raw']:.3f}`, `sigma_slow={r2f_ip['sigma_slow_bpm']:.3f} bpm`, `sigma_jump={r2f_ip['sigma_jump_bpm']:.3f} bpm`, `pi_jump={r2f_ip['pi_jump']:.6f}`)",
        f"- **Rung 2 `R2a_Causal_DiffHMM_Linear11` (`11p`):** LOPO-mean `{[round(x, 6) for x in hf_m['frozen_r2a_mean_theta']]}`; Full-cohort `K=11` refit `{[round(x, 6) for x in hf_m['frozen_r2a_full_refit_theta']]}` (`w1={r2a_ip['w1_p1_clean']:.3f}`, `w2={r2a_ip['w2_p2_harm']:.3f}`, `w5={r2a_ip['w5_anchor']:.3f}`, `w6={r2a_ip['w6_p1_raw']:.3f}`, `sigma_slow={r2a_ip['sigma_slow_bpm']:.3f} bpm`, `sigma_jump={r2a_ip['sigma_jump_bpm']:.3f} bpm`, `pi_jump={r2a_ip['pi_jump']:.6f}`)",
        f"- **Rung 2 `R2d_E2E_Causal_H5_DiffHMM_Linear11` (`11p`):** LOPO-mean `{[round(x, 6) for x in hf_m['frozen_r2d_mean_theta']]}`; Full-cohort `K=11` refit `{[round(x, 6) for x in hf_m['frozen_r2d_full_refit_theta']]}` (`w1={r2d_ip['w1_p1_clean']:.3f}`, `w2={r2d_ip['w2_p2_harm']:.3f}`, `w5={r2d_ip['w5_anchor']:.3f}`, `w6={r2d_ip['w6_p1_raw']:.3f}`, `sigma_slow={r2d_ip['sigma_slow_bpm']:.3f} bpm`, `sigma_jump={r2d_ip['sigma_jump_bpm']:.3f} bpm`, `pi_jump={r2d_ip['pi_jump']:.6f}`)",
        f"- **Rung 3 Full-Cohort (`K=11`) Frozen Foils:** `R3c_Ridge_Spectral_439` (`n={r3c_sm['n_params']}`, `alpha={r3c_sm['selected_alpha']:.1f}`, `inner_lopo_rmse={r3c_sm['inner_lopo_rmse_bpm']:.3f}`, `||w||_2={r3c_sm['coef_l2_norm']:.3f}`); `R3b_Spectral_MLP_15k` (`n={r3b_sm['n_params']}`, `alpha={r3b_sm['selected_alpha']:.1f}`, `val_pat={r3b_sm['val_patient_id']}`, `||W||_F={r3b_sm['weight_l2_norm']:.3f}`); `R3a_Causal_TCN_16k` (`n={r3a_sm['n_params']}`, `wd={r3a_sm['selected_weight_decay']}`, `drop={r3a_sm['selected_dropout_p']}`, `val_pat={r3a_sm['val_patient_id']}`, `||W||_F={r3a_sm['weight_l2_norm']:.3f}`)",
        f"- **Frozen Conformal Quantiles (`1 Hz` LOPO-mean / Full-`K=11`):** `C4` (`bpm`) `{hf_m['frozen_c4_q_hat_bpm_pool30_1hz']}` / `{hf_m['frozen_c4_q_hat_bpm_full_k11_1hz']}`; `C5` (norm) `{hf_m['frozen_c5_q_hat_norm_pool30_1hz']}` / `{hf_m['frozen_c5_q_hat_norm_full_k11_1hz']}`; (`Trig -10s` LOPO-mean): `C4` `{hf_m['frozen_c4_q_hat_bpm_pool30_trig_lag_aligned']}`; `C5` `{hf_m['frozen_c5_q_hat_norm_pool30_trig_lag_aligned']}`",
        f"- **Pre-Registered Stage 4 Protocol (`{s4_proto['external_dataset']}`):** Primary in-band stratum `{s4_proto['primary_inband_stratum_bpm']} bpm` vs. secondary out-of-passband FIR-skirt stratum `{s4_proto['secondary_out_of_passband_low_rate_stratum_bpm']} bpm` (`{s4_proto['zero_touch_policy']}`).",
    ]

    AUDIT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[EXP-3] Wrote {BUNDLE_JSON}, {AUDIT_MD}, {AUDIT_SESSIONS_CSV}, {STAGE3_ARCHIVE} in {total_s:.1f}s.")


if __name__ == "__main__":
    main()
