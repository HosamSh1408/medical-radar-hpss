"""EXP-2 Stage 2: Complexity Ladder (Rung 0-3), Differentiable 1D-HMM & Shortcut-Learning Audit.

Trains all Rung 1, Rung 2 (`11-43` param zero-offset symmetric Differentiable 1D-HMMs),
and Rung 3 (`439-15,817` param unconstrained Ridge / Spectral MLP / Causal TCN foils)
strictly under patient-disjoint LOPO using ONLY `br_1hz_30s` (`1 Hz RespRate`),
then evaluates blindly on physical `br_trig_bpm` (`1024 Hz Resp.-Trigger`), out-of-range
`20-80 bpm` 24 GHz synthetic frequency sweeps, shifted `[18, 90]` vs `[18, 120]` grids,
5-seed phase-scrambled radar surrogates, O(2)xR+ invariance, and strict causality.

Execution:
  MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /opt/homebrew/bin/python3.10 -m thesis.exp2_stage2_complexity --workers 4
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
from paper_workspace.src.filtering import autocorr_coarse, bandpass
from paper_workspace.src.hpss import mitigate_hpss
from paper_workspace.src.provenance import expected_provenance
from thesis import diff_hmm_and_foils as D
from thesis import diff_hmm_batch as DB
from thesis import edf_extractor as EDFX
from thesis import exp1_stage1_parsimony as E1
from thesis import io_utils as IO
from thesis import viterbi_tracker as VT
from thesis import wp1_wp2_causal_hmm as W

SCHEMA_VERSION = "stage2-complexity-v4"
OUTPUT_DIR = IO.ROOT / "thesis" / "outputs" / "stage2_complexity"
SAL_CACHE_DIR = OUTPUT_DIR / "cache"
SURR_CACHE_DIR = SAL_CACHE_DIR / "surrogates_5seed"
PRED_ARCHIVE = OUTPUT_DIR / "pool30_predictions.npz"
DIFF_MODULE_PATH = IO.ROOT / "thesis" / "diff_hmm_and_foils.py"
WP_MODULE_PATH = IO.ROOT / "thesis" / "wp1_wp2_causal_hmm.py"

STAGE1_BUNDLE_JSON = IO.BUNDLE_DIR / "metrics_stage1_parsimony.json"
BUNDLE_JSON = IO.BUNDLE_DIR / "metrics_stage2_complexity.json"
AUDIT_MD = IO.AUDIT_DIR / "stage2_complexity_report.md"
AUDIT_SESSIONS_CSV = IO.AUDIT_DIR / "stage2_complexity_sessions.csv"

COHORTS = ("pool30_sync", "retained23", "PAPER_18")
REGIME = "branch_neutral_raw_nls_lag"

MODEL_ORDER = (
    "R0_HPSS_NLS_Locked",
    "R1a_Online_Forward_MMSE_0s",
    "R1f_Offline_FB_MMSE_Inf",
    "R1g_Offline_Viterbi_Unpinned",
    "R2e_Causal_DiffHMM_FilterSafe8",
    "R2a_Causal_DiffHMM_Linear11",
    "R2b_Causal_DiffHMM_MLP43",
    "R2f_Offline_DiffHMM_FilterSafe8",
    "R2c_Offline_DiffHMM_Linear11",
    "R2d_E2E_Causal_H5_DiffHMM_Linear11",
    "R3c_Ridge_Spectral_439",
    "R3b_Spectral_MLP_15k",
    "R3a_Causal_TCN_16k",
)

MODEL_META: dict[str, dict[str, Any]] = {
    "R0_HPSS_NLS_Locked": {"rung": 0, "n_params": 0, "block_lookahead_s": 0.0, "block_lookahead_label": "0 s", "stft_lookahead_s": 3.0, "shift_equivariant": True, "o2_invariant": True},
    "R1a_Online_Forward_MMSE_0s": {"rung": 1, "n_params": 3, "block_lookahead_s": 0.0, "block_lookahead_label": "0 s", "stft_lookahead_s": 3.0, "shift_equivariant": True, "o2_invariant": True},
    "R1f_Offline_FB_MMSE_Inf": {"rung": 1, "n_params": 3, "block_lookahead_s": None, "block_lookahead_label": "inf", "stft_lookahead_s": 3.0, "shift_equivariant": True, "o2_invariant": True},
    "R1g_Offline_Viterbi_Unpinned": {"rung": 1, "n_params": 2, "block_lookahead_s": None, "block_lookahead_label": "inf", "stft_lookahead_s": 3.0, "shift_equivariant": True, "o2_invariant": True},
    "R2e_Causal_DiffHMM_FilterSafe8": {"rung": 2, "n_params": 8, "block_lookahead_s": 0.0, "block_lookahead_label": "0 s", "stft_lookahead_s": 3.0, "shift_equivariant": True, "o2_invariant": True},
    "R2a_Causal_DiffHMM_Linear11": {"rung": 2, "n_params": 11, "block_lookahead_s": 0.0, "block_lookahead_label": "0 s", "stft_lookahead_s": 3.0, "shift_equivariant": True, "o2_invariant": True},
    "R2b_Causal_DiffHMM_MLP43": {"rung": 2, "n_params": 43, "block_lookahead_s": 0.0, "block_lookahead_label": "0 s", "stft_lookahead_s": 3.0, "shift_equivariant": True, "o2_invariant": True},
    "R2f_Offline_DiffHMM_FilterSafe8": {"rung": 2, "n_params": 8, "block_lookahead_s": None, "block_lookahead_label": "inf", "stft_lookahead_s": 3.0, "shift_equivariant": True, "o2_invariant": True},
    "R2c_Offline_DiffHMM_Linear11": {"rung": 2, "n_params": 11, "block_lookahead_s": None, "block_lookahead_label": "inf", "stft_lookahead_s": 3.0, "shift_equivariant": True, "o2_invariant": True},
    "R2d_E2E_Causal_H5_DiffHMM_Linear11": {"rung": 2, "n_params": 11, "block_lookahead_s": 0.0, "block_lookahead_label": "0 s", "stft_lookahead_s": 0.0, "shift_equivariant": True, "o2_invariant": True},
    "R3c_Ridge_Spectral_439": {"rung": 3, "n_params": 439, "block_lookahead_s": 0.0, "block_lookahead_label": "0 s", "stft_lookahead_s": 3.0, "shift_equivariant": False, "o2_invariant": True},
    "R3b_Spectral_MLP_15k": {"rung": 3, "n_params": 14593, "block_lookahead_s": 0.0, "block_lookahead_label": "0 s", "stft_lookahead_s": 3.0, "shift_equivariant": False, "o2_invariant": True},
    "R3a_Causal_TCN_16k": {"rung": 3, "n_params": 15817, "block_lookahead_s": 0.0, "block_lookahead_label": "0 s", "stft_lookahead_s": 3.0, "shift_equivariant": False, "o2_invariant": True},
}


def _sal_cache_valid(path: Path, measurement: str, wp_sha: str, diff_sha: str) -> bool:
    if not path.exists():
        return False
    try:
        radar_sha = expected_provenance(measurement)["input_sha256"]["radar_csv"]
        with np.load(path, allow_pickle=False) as z:
            prov = json.loads(str(z["provenance_json"].item()))
            sv = prov.get("schema_version")
            diff_ok = (prov.get("diff_sha256") == diff_sha) if sv == SCHEMA_VERSION else True
            return (
                sv in ("stage2-complexity-v2", "stage2-complexity-v3", SCHEMA_VERSION)
                and prov.get("measurement") == measurement
                and prov.get("wp_sha256") == wp_sha
                and prov.get("radar_sha256") == radar_sha
                and diff_ok
            )
    except Exception:
        return False


def _extract_salience_session(task: tuple[str, str, str, bool]) -> tuple[str, str]:
    m, wp_sha, diff_sha, force = task
    SAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAL_CACHE_DIR / f"{m}.npz"
    if not force and _sal_cache_valid(out_path, m, wp_sha, diff_sha):
        return m, "cached"

    bi, bq = load_radar(m)
    w_len = int(C.WIN_SAMPLES)
    step = int(C.UPDATE_S * C.FS_RADAR)
    starts = list(range(0, len(bi) - w_len + 1, step))
    n_win = len(starts)

    phi_h0 = np.empty((n_win, VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS), dtype=np.float32)
    phi_h5 = np.empty((n_win, VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS), dtype=np.float32)

    for idx, s0 in enumerate(starts):
        x0 = csd(bi[s0 : s0 + w_len], bq[s0 : s0 + w_len])
        dr = bandpass(x0)

        dh0 = bandpass(mitigate_hpss(dr)["x_hat"])
        fc0 = autocorr_coarse(dh0)
        phi_h0[idx] = D.extract_salience_features_window(dr, dh0, fc0, VT.BR_GRID_1BPM).astype(np.float32)

        dh5 = bandpass(W.mitigate_hpss_v2(dr, window="cola_hann", lookahead=0, in_band=True)["x_hat"])
        fc5 = autocorr_coarse(dh5)
        phi_h5[idx] = D.extract_salience_features_window(dr, dh5, fc5, VT.BR_GRID_1BPM).astype(np.float32)

    radar_sha = expected_provenance(m)["input_sha256"]["radar_csv"]
    prov_str = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "measurement": m,
            "wp_sha256": wp_sha,
            "diff_sha256": diff_sha,
            "radar_sha256": radar_sha,
        },
        sort_keys=True,
    )
    tmp_path = SAL_CACHE_DIR / f".{m}.tmp.npz"
    np.savez_compressed(tmp_path, phi_h0=phi_h0, phi_h5=phi_h5, provenance_json=np.array(prov_str))
    tmp_path.replace(out_path)
    return m, "computed"


def _load_salience_session(m: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(SAL_CACHE_DIR / f"{m}.npz", allow_pickle=False) as z:
        return z["phi_h0"].astype(np.float64), z["phi_h5"].astype(np.float64)


def _extract_surr_salience_session(task: tuple[str, int, int, str, str, bool]) -> tuple[str, str]:
    m, n_seeds, base_seed, wp_sha, diff_sha, force = task
    SURR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SURR_CACHE_DIR / f"{m}.npz"
    radar_sha = expected_provenance(m)["input_sha256"]["radar_csv"]
    if not force and out_path.exists():
        try:
            with np.load(out_path, allow_pickle=False) as z:
                prov = json.loads(str(z["provenance_json"].item()))
                sv = prov.get("schema_version")
                diff_ok = (prov.get("diff_sha256") == diff_sha) if sv == SCHEMA_VERSION else True
                if (
                    sv in ("stage2-complexity-v2", "stage2-complexity-v3", SCHEMA_VERSION)
                    and prov.get("measurement") == m
                    and prov.get("wp_sha256") == wp_sha
                    and prov.get("radar_sha256") == radar_sha
                    and int(prov.get("n_seeds", 0)) == int(n_seeds)
                    and diff_ok
                ):
                    return m, "cached"
        except Exception:
            pass

    bi, bq = load_radar(m)
    x_full = csd(bi, bq)
    xf = np.fft.fft(x_full)
    mag_xf = np.abs(xf)
    w_len = int(C.WIN_SAMPLES)
    step = int(C.UPDATE_S * C.FS_RADAR)
    starts = list(range(0, x_full.size - w_len + 1, step))
    n_w = len(starts)

    pid = IO.patient_id_of(m)
    s_idx = int(m.split(".")[1])
    phi_h0_all = np.empty((n_seeds, n_w, VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS), dtype=np.float32)
    phi_h5_all = np.empty((n_seeds, n_w, VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS), dtype=np.float32)

    for seed_idx in range(n_seeds):
        rng_s = np.random.default_rng(int(base_seed) + seed_idx * 1009 + pid * 31 + s_idx)
        rand_phases = rng_s.uniform(0.0, 2.0 * np.pi, size=xf.size)
        rand_phases[0] = 0.0
        x_scr = np.fft.ifft(mag_xf * np.exp(1j * rand_phases))

        for idx, s0 in enumerate(starts):
            dr = bandpass(x_scr[s0 : s0 + w_len])
            dh0 = bandpass(mitigate_hpss(dr)["x_hat"])
            fc0 = autocorr_coarse(dh0)
            phi_h0_all[seed_idx, idx] = D.extract_salience_features_window(dr, dh0, fc0, VT.BR_GRID_1BPM).astype(np.float32)

            dh5 = bandpass(W.mitigate_hpss_v2(dr, window="cola_hann", lookahead=0, in_band=True)["x_hat"])
            fc5 = autocorr_coarse(dh5)
            phi_h5_all[seed_idx, idx] = D.extract_salience_features_window(dr, dh5, fc5, VT.BR_GRID_1BPM).astype(np.float32)

    prov_str = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "measurement": m,
            "wp_sha256": wp_sha,
            "diff_sha256": diff_sha,
            "radar_sha256": radar_sha,
            "n_seeds": int(n_seeds),
            "base_seed": int(base_seed),
        },
        sort_keys=True,
    )
    tmp_path = SURR_CACHE_DIR / f".{m}.tmp.npz"
    np.savez_compressed(tmp_path, phi_h0=phi_h0_all, phi_h5=phi_h5_all, provenance_json=np.array(prov_str))
    tmp_path.replace(out_path)
    return m, "computed"


def _load_surr_salience_session(m: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(SURR_CACHE_DIR / f"{m}.npz", allow_pickle=False) as z:
        return z["phi_h0"].astype(np.float64), z["phi_h5"].astype(np.float64)


def _make_clinical_shift_salience_pair(
    f0: float = 36.0,
    delta: float = 10.0,
    n_t: int = 15,
    grid: np.ndarray = VT.BR_GRID_1BPM,
) -> tuple[np.ndarray, np.ndarray]:
    """Build `[18, 90]` salience probe pair matching `extract_salience_features_window` (`ch4` >= 0.05 floor)."""
    p_b = np.zeros((n_t, grid.size, D.N_SAL_CHANNELS), dtype=np.float64)
    p_s = np.zeros((n_t, grid.size, D.N_SAL_CHANNELS), dtype=np.float64)
    for ch in range(D.N_SAL_CHANNELS):
        if ch == 4:
            g_b = np.maximum(0.05, np.exp(-0.5 * ((grid[None, :] - f0) / float(C.NLS_RADIUS_BPM)) ** 2))
            g_s = np.maximum(0.05, np.exp(-0.5 * ((grid[None, :] - (f0 + delta)) / float(C.NLS_RADIUS_BPM)) ** 2))
        else:
            g_b = np.exp(-0.5 * ((grid[None, :] - f0) / 2.5) ** 2)
            g_s = np.exp(-0.5 * ((grid[None, :] - (f0 + delta)) / 2.5) ** 2)
        p_b[:, :, ch] = np.log(g_b + 1e-3)
        p_s[:, :, ch] = np.log(g_s + 1e-3)
    return p_b, p_s


def _generate_synthetic_suite() -> dict[str, dict[str, Any]]:
    """Generate zero-touch 24 GHz CW synthetic frequency sweep (20-80 bpm) + chirp + shift probe."""
    fs = float(C.FS_RADAR)
    dur_s = 120.0
    t = np.arange(int(dur_s * fs), dtype=np.float64) / fs
    rng = np.random.default_rng(IO.RANDOM_SEED)
    w_len = int(C.WIN_SAMPLES)
    step = int(C.UPDATE_S * C.FS_RADAR)
    starts = list(range(0, t.size - w_len + 1, step))

    scenarios: dict[str, dict[str, Any]] = {}
    cases = (
        ("stat_28bpm", False, 28.0),
        ("stat_35bpm", False, 35.0),
        ("stat_50bpm", False, 50.0),
        ("stat_65bpm", False, 65.0),
        ("stat_80bpm", False, 80.0),
        ("chirp_30_to_75bpm", True, 30.0),
        ("stat_20bpm_kaiser_edge", False, 20.0),
        ("probe_shift_36bpm", False, 36.0),
        ("probe_shift_46bpm", False, 46.0),
    )
    for cname, is_chirp, f0_bpm in cases:
        if not is_chirp:
            f_inst = np.full_like(t, f0_bpm / 60.0)
            phase_resp = 2.0 * np.pi * f_inst * t
            d_m = 0.0012 * np.sin(phase_resp) + 0.00012 * np.sin(2.0 * np.pi * 2.5 * t)
            x_syn = np.exp(1j * (0.4 + 4.0 * np.pi * d_m / C.LAMBDA_M))
            x_syn = x_syn + 0.01 * (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size))
        else:
            # Linear sweep 30 -> 75 bpm with 4-s RBM amplitude burst at t=[50, 54]s
            f_inst = (30.0 + 45.0 * (t / dur_s)) / 60.0
            phase_resp = 2.0 * np.pi * ((30.0 / 60.0) * t + 0.5 * (45.0 / 60.0 / dur_s) * (t ** 2))
            d_m = 0.0010 * np.sin(phase_resp) + 0.00010 * np.sin(2.0 * np.pi * 2.5 * t)
            x_syn = np.exp(1j * (0.4 + 4.0 * np.pi * d_m / C.LAMBDA_M))
            rbm_mask = (t >= 50.0) & (t <= 54.0)
            x_syn[rbm_mask] *= 8.0
            x_syn = x_syn + 0.01 * (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size))

        truth = np.array([f_inst[s0 + w_len // 2] * 60.0 for s0 in starts], dtype=np.float64)
        phi_h0 = np.empty((len(starts), VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS), dtype=np.float64)
        phi_h5 = np.empty((len(starts), VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS), dtype=np.float64)
        s_anc_h0 = np.empty((len(starts), VT.BR_GRID_1BPM.size), dtype=np.float64)
        f_r0 = np.empty(len(starts), dtype=np.float64)

        for idx, s0 in enumerate(starts):
            dr = bandpass(x_syn[s0 : s0 + w_len])
            f0_est, dh0, fc0, _ = E1._run_locked_hpss_nls(x_syn[s0 : s0 + w_len])
            f_r0[idx] = f0_est
            s_anc_h0[idx] = VT.extract_br_anchored_nls_spectrum(dh0, fc0, VT.BR_GRID_1BPM)
            phi_h0[idx] = D.extract_salience_features_window(dr, dh0, fc0, VT.BR_GRID_1BPM)

            dh5 = bandpass(W.mitigate_hpss_v2(dr, window="cola_hann", lookahead=0, in_band=True)["x_hat"])
            fc5 = autocorr_coarse(dh5)
            phi_h5[idx] = D.extract_salience_features_window(dr, dh5, fc5, VT.BR_GRID_1BPM)

        scenarios[cname] = {
            "truth_bpm": truth,
            "f_r0": f_r0,
            "s_anc_h0": s_anc_h0,
            "phi_h0": phi_h0,
            "phi_h5": phi_h5,
        }
    return scenarios


def _precompute_stage1_decoded(sessions: list[str]) -> dict[str, dict[str, Any]]:
    """Decode locked Stage 1 R0/R1a/R1f/R1g grids once per unique session."""
    decoded_mini: dict[str, dict[str, Any]] = {}
    prob_cfgs = [(s, e, b) for s in E1.SIGMA_GRID for e in E1.EPS_GRID for b in E1.BETA_GRID]
    vit_cfgs = [(s, e, 1.0) for s in E1.SIGMA_GRID for e in E1.EPS_GRID]

    for m in sessions:
        feat = E1._load_feat_cache(m)
        s0 = feat["S_anc_h0_1bpm"]
        r1a_g, r1f_g, r1g_g = {}, {}, {}
        for cfg in prob_cfgs:
            sig, eps, beta = cfg
            r1a_g[cfg] = W.hmm_forward_filter(s0, VT.BR_GRID_1BPM, sig, eps, beta)["est_mmse"]
            r1f_g[cfg] = W.hmm_offline_fb_smoother(s0, VT.BR_GRID_1BPM, sig, eps, beta)["est_mmse"]
        for cfg in vit_cfgs:
            sig, eps, beta = cfg
            r1g_g[cfg] = W.hmm_offline_viterbi_prob(s0, VT.BR_GRID_1BPM, sig, eps, beta)[0]
        decoded_mini[m] = {"anchor_r0": feat["wp2_H0_Locked_Rect_Full_L3"], "grid": {"R1a": r1a_g, "R1f": r1f_g, "R1g": r1g_g}}
    return decoded_mini


def _compute_stage1_r0_r1(
    cohort: str,
    ref_1hz_map: dict[str, tuple[np.ndarray, np.ndarray]],
    decoded_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute Stage 1 R0/R1a/R1f/R1g exactly from frozen Stage 1 feature caches."""
    sessions = IO.get_cohort_sessions(cohort, include_zero_valid=False)
    decoded_mini = decoded_cache if decoded_cache is not None else _precompute_stage1_decoded(sessions)

    folds = list(IO.iter_lopo_folds(cohort))
    out: dict[str, dict[str, np.ndarray]] = {
        "R0_HPSS_NLS_Locked": {m: decoded_mini[m]["anchor_r0"] for m in sessions},
        "R1a_Online_Forward_MMSE_0s": {},
        "R1f_Offline_FB_MMSE_Inf": {},
        "R1g_Offline_Viterbi_Unpinned": {},
    }
    fold_r1a_cfgs: dict[int, tuple[float, float, float]] = {}

    for fold in folds:
        tr = [m for m in fold.train_sessions if m in sessions]
        te = [m for m in fold.test_sessions if m in sessions]
        if not te:
            continue
        anc_e = [decoded_mini[m]["anchor_r0"][ref_1hz_map[m][1]] for m in tr]
        refs = [ref_1hz_map[m][0][ref_1hz_map[m][1]] for m in tr]
        base_r = IO.rmcorr_ancova(anc_e, refs)["r_within"]
        min_r = 1.00 * base_r if np.isfinite(base_r) else -np.inf

        for short_k, full_k in (
            ("R1a", "R1a_Online_Forward_MMSE_0s"),
            ("R1f", "R1f_Offline_FB_MMSE_Inf"),
            ("R1g", "R1g_Offline_Viterbi_Unpinned"),
        ):
            cfgs = list(decoded_mini[tr[0]]["grid"][short_k].keys())
            best_el, best_fb = None, None
            for cfg in cfgs:
                es = [decoded_mini[m]["grid"][short_k][cfg][ref_1hz_map[m][1]] for m in tr]
                rmses = [float(np.sqrt(np.mean((e - r) ** 2))) for e, r in zip(es, refs)]
                mrmse = float(np.mean(rmses))
                rw = IO.rmcorr_ancova(es, refs)["r_within"]
                fb = (-(rw if np.isfinite(rw) else -1.0), mrmse, cfg)
                if best_fb is None or fb < best_fb:
                    best_fb = fb
                if np.isfinite(rw) and rw >= min_r:
                    el = (mrmse, -rw, cfg)
                    if best_el is None or el < best_el:
                        best_el = el
            sel_cfg = best_el[2] if best_el is not None else best_fb[2]
            for m in te:
                out[full_k][m] = decoded_mini[m]["grid"][short_k][sel_cfg]
            if full_k == "R1a_Online_Forward_MMSE_0s":
                fold_r1a_cfgs[fold.patient_id] = sel_cfg

    out["_r1a_fold_cfgs"] = fold_r1a_cfgs  # type: ignore[assignment]
    return out


def _fit_eval_fold_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Train all Rung 2 and Rung 3 models strictly on one patient-disjoint LOPO fold."""
    fold_idx = int(task["fold_idx"])
    patient_id = int(task["patient_id"])
    train_ms: list[str] = task["train_ms"]
    test_ms: list[str] = task["test_ms"]
    ref_map: dict[str, tuple[np.ndarray, np.ndarray]] = task["ref_map"]
    run_extra: bool = bool(task["run_extra"])

    phi_h0 = {m: _load_salience_session(m)[0] for m in train_ms + test_ms}
    phi_h5 = {m: _load_salience_session(m)[1] for m in train_ms + test_ms}

    tr_h0 = [(phi_h0[m], ref_map[m][0], ref_map[m][1]) for m in train_ms]
    tr_h5 = [(phi_h5[m], ref_map[m][0], ref_map[m][1]) for m in train_ms]

    # Deterministic inner validation patient (never touches test patient)
    tr_pats = sorted({IO.patient_id_of(m) for m in train_ms})
    val_pat = tr_pats[fold_idx % len(tr_pats)]
    tr3_ms = [m for m in train_ms if IO.patient_id_of(m) != val_pat]
    va3_ms = [m for m in train_ms if IO.patient_id_of(m) == val_pat]
    tr3_h0 = [(phi_h0[m], ref_map[m][0], ref_map[m][1]) for m in tr3_ms]
    va3_h0 = [(phi_h0[m], ref_map[m][0], ref_map[m][1]) for m in va3_ms]

    # Fit Rung 2 equivariant models (all <= 43 params, zero offset) via exact batched BLAS BPTT (`maxiter=200`)
    r2e = DB.fit_diff_hmm_batched(D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward", reg_l2=5e-3), tr_h0, maxiter=200)
    r2a = DB.fit_diff_hmm_batched(D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward", reg_l2=5e-3), tr_h0, maxiter=200)
    r2b = DB.fit_diff_hmm_batched(D.DifferentiableSymmetricHMM(mode="mlp", decoder="causal_forward", reg_l2=5e-3, seed=IO.RANDOM_SEED + patient_id), tr_h0, maxiter=200)
    r2f = DB.fit_diff_hmm_batched(D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="offline_fb", reg_l2=5e-3), tr_h0, maxiter=200)
    r2c = DB.fit_diff_hmm_batched(D.DifferentiableSymmetricHMM(mode="linear", decoder="offline_fb", reg_l2=5e-3), tr_h0, maxiter=200)
    r2d = DB.fit_diff_hmm_batched(D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward", reg_l2=5e-3), tr_h5, maxiter=200)

    # Fit Rung 3 unconstrained ERM foils (inner patient-LOPO Ridge 439, inner-val Spectral MLP 14.6k & Causal TCN 15.8k)
    ridge_res = D.fit_predict_ridge_spectral(
        tr_h0,
        [phi_h0[m] for m in test_ms],
        train_patient_ids=[IO.patient_id_of(m) for m in train_ms],
    )
    mlp_res = D.fit_predict_spectral_mlp(
        tr3_h0,
        va3_h0,
        [phi_h0[m] for m in test_ms],
        alpha_grid=(0.1, 1.0, 10.0, 100.0),
        seed=IO.RANDOM_SEED + patient_id,
    )
    tcn, tcn_info = D.fit_select_causal_tcn(
        tr3_h0,
        va3_h0,
        lr=1.5e-3,
        grid=((1e-3, 0.0), (1e-2, 0.0), (1e-2, 0.15), (5e-2, 0.25)),
        max_epochs=100,
        patience=18,
        seed=IO.RANDOM_SEED + patient_id,
    )

    test_preds: dict[str, dict[str, np.ndarray]] = {k: {} for k in MODEL_ORDER if k.startswith(("R2", "R3"))}
    test_unc: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "R2e_Causal_DiffHMM_FilterSafe8": {},
        "R2a_Causal_DiffHMM_Linear11": {},
        "R2b_Causal_DiffHMM_MLP43": {},
        "R2d_E2E_Causal_H5_DiffHMM_Linear11": {},
    }

    for idx_m, m in enumerate(test_ms):
        pr2e = r2e.predict_session(phi_h0[m])
        pr2a = r2a.predict_session(phi_h0[m])
        pr2b = r2b.predict_session(phi_h0[m])
        pr2f = r2f.predict_session(phi_h0[m])
        pr2c = r2c.predict_session(phi_h0[m])
        pr2d = r2d.predict_session(phi_h5[m])

        test_preds["R2e_Causal_DiffHMM_FilterSafe8"][m] = pr2e["est_mmse"]
        test_preds["R2a_Causal_DiffHMM_Linear11"][m] = pr2a["est_mmse"]
        test_preds["R2b_Causal_DiffHMM_MLP43"][m] = pr2b["est_mmse"]
        test_preds["R2f_Offline_DiffHMM_FilterSafe8"][m] = pr2f["est_mmse"]
        test_preds["R2c_Offline_DiffHMM_Linear11"][m] = pr2c["est_mmse"]
        test_preds["R2d_E2E_Causal_H5_DiffHMM_Linear11"][m] = pr2d["est_mmse"]
        test_preds["R3c_Ridge_Spectral_439"][m] = ridge_res["preds"][idx_m]
        test_preds["R3b_Spectral_MLP_15k"][m] = mlp_res["preds"][idx_m]
        test_preds["R3a_Causal_TCN_16k"][m] = tcn.predict_session(phi_h0[m])

        test_unc["R2e_Causal_DiffHMM_FilterSafe8"][m] = {"post_entropy": pr2e["post_entropy"], "post_sd_bpm": pr2e["post_sd_bpm"]}
        test_unc["R2a_Causal_DiffHMM_Linear11"][m] = {"post_entropy": pr2a["post_entropy"], "post_sd_bpm": pr2a["post_sd_bpm"]}
        test_unc["R2b_Causal_DiffHMM_MLP43"][m] = {"post_entropy": pr2b["post_entropy"], "post_sd_bpm": pr2b["post_sd_bpm"]}
        test_unc["R2d_E2E_Causal_H5_DiffHMM_Linear11"][m] = {"post_entropy": pr2d["post_entropy"], "post_sd_bpm": pr2d["post_sd_bpm"]}

    out: dict[str, Any] = {
        "patient_id": patient_id,
        "val_patient_id": int(val_pat),
        "test_ms": test_ms,
        "test_preds": test_preds,
        "test_unc": test_unc,
        "r2e_theta": r2e.theta.tolist(),
        "r2a_theta": r2a.theta.tolist(),
        "r2b_theta": r2b.theta.tolist(),
        "r2e_interp": r2e.interpretable_params(),
        "r2a_interp": r2a.interpretable_params(),
        "r2b_interp": r2b.interpretable_params(),
        "r2f_interp": r2f.interpretable_params(),
        "r2c_interp": r2c.interpretable_params(),
        "r2d_interp": r2d.interpretable_params(),
        "r2e_nit": r2e.train_history_["nit"],
        "r2a_nit": r2a.train_history_["nit"],
        "r2b_nit": r2b.train_history_["nit"],
        "r2f_nit": r2f.train_history_["nit"],
        "r2c_nit": r2c.train_history_["nit"],
        "r2d_nit": r2d.train_history_["nit"],
        "r2_success": {
            "r2e": bool(r2e.train_history_["success"]),
            "r2a": bool(r2a.train_history_["success"]),
            "r2b": bool(r2b.train_history_["success"]),
            "r2f": bool(r2f.train_history_["success"]),
            "r2c": bool(r2c.train_history_["success"]),
            "r2d": bool(r2d.train_history_["success"]),
        },
        "r3_hyperparams": {
            "ridge_alpha": float(ridge_res["selected_alpha"]),
            "ridge_inner_cv_mode": str(ridge_res["inner_cv_mode"]),
            "ridge_inner_lopo_rmse": float(ridge_res["inner_lopo_rmse"]),
            "mlp_alpha": float(mlp_res["selected_alpha"]),
            "mlp_val_rmse": float(mlp_res["best_val_rmse"]),
            "tcn_weight_decay": float(tcn_info["selected_weight_decay"]),
            "tcn_dropout_p": float(tcn_info["selected_dropout_p"]),
            "tcn_val_rmse": float(tcn_info["best_val_rmse"]),
        },
    }

    if run_extra:
        # 0. Calibrated [18, 90] +10 bpm salience shift probe matching ch4 >= 0.05 floor (contrasting R2 vs R3 foils)
        p_b, p_s = _make_clinical_shift_salience_pair(f0=36.0, delta=10.0, n_t=15, grid=VT.BR_GRID_1BPM)
        pb_r = np.clip(ridge_res["model"].predict((D.flatten_salience_tensor(p_b) - ridge_res["mean"]) / ridge_res["scale"]), 12.0, 120.0)
        ps_r = np.clip(ridge_res["model"].predict((D.flatten_salience_tensor(p_s) - ridge_res["mean"]) / ridge_res["scale"]), 12.0, 120.0)
        pb_m = np.clip(mlp_res["model"].predict((D.flatten_salience_tensor(p_b) - mlp_res["mean"]) / mlp_res["scale"]), 12.0, 120.0)
        ps_m = np.clip(mlp_res["model"].predict((D.flatten_salience_tensor(p_s) - mlp_res["mean"]) / mlp_res["scale"]), 12.0, 120.0)
        pb_t = tcn.predict_session(p_b)
        ps_t = tcn.predict_session(p_s)
        out["r3_clinical_shift_err_bpm"] = {
            "R3c_Ridge_Spectral_439": {"max_abs_err_bpm": float(np.max(np.abs((ps_r - pb_r) - 10.0))), "mean_shift_bpm": float(np.mean(ps_r - pb_r))},
            "R3b_Spectral_MLP_15k": {"max_abs_err_bpm": float(np.max(np.abs((ps_m - pb_m) - 10.0))), "mean_shift_bpm": float(np.mean(ps_m - pb_m))},
            "R3a_Causal_TCN_16k": {"max_abs_err_bpm": float(np.max(np.abs((ps_t - pb_t) - 10.0))), "mean_shift_bpm": float(np.mean(ps_t - pb_t))},
        }
        # 1. Synthetic sweep (eval this fold's frozen models on generated 20..80 bpm + chirp + waveform shift probe)
        synth_suite = task["synth_suite"]
        r1a_cfg = task["r1a_cfg"]
        synth_fold: dict[str, dict[str, np.ndarray]] = {}
        for cname, sdata in synth_suite.items():
            ph0_s, ph5_s = sdata["phi_h0"], sdata["phi_h5"]
            synth_fold[cname] = {
                "R0_HPSS_NLS_Locked": sdata["f_r0"],
                "R1a_Online_Forward_MMSE_0s": W.hmm_forward_filter(
                    sdata["s_anc_h0"], VT.BR_GRID_1BPM, sigma_br=r1a_cfg[0], eps=r1a_cfg[1], beta=r1a_cfg[2]
                )["est_mmse"],
                "R2e_Causal_DiffHMM_FilterSafe8": r2e.predict_session(ph0_s)["est_mmse"],
                "R2a_Causal_DiffHMM_Linear11": r2a.predict_session(ph0_s)["est_mmse"],
                "R2b_Causal_DiffHMM_MLP43": r2b.predict_session(ph0_s)["est_mmse"],
                "R2d_E2E_Causal_H5_DiffHMM_Linear11": r2d.predict_session(ph5_s)["est_mmse"],
                "R3c_Ridge_Spectral_439": np.clip(
                    ridge_res["model"].predict((D.flatten_salience_tensor(ph0_s) - ridge_res["mean"]) / ridge_res["scale"]), 12.0, 120.0
                ),
                "R3b_Spectral_MLP_15k": np.clip(
                    mlp_res["model"].predict((D.flatten_salience_tensor(ph0_s) - mlp_res["mean"]) / mlp_res["scale"]), 12.0, 120.0
                ),
                "R3a_Causal_TCN_16k": tcn.predict_session(ph0_s),
            }
        out["synth_fold"] = synth_fold

        # 2. 5-seed phase-scrambled test predictions under this fold's frozen models
        if not task["skip_surrogates"]:
            surr_fold: dict[str, dict[int, dict[str, np.ndarray]]] = {}
            for m in test_ms:
                sh0, sh5 = _load_surr_salience_session(m)
                per_seed: dict[int, dict[str, np.ndarray]] = {}
                for s_i in range(5):
                    ph0_m, ph5_m = sh0[s_i], sh5[s_i]
                    per_seed[s_i] = {
                        "R2e_Causal_DiffHMM_FilterSafe8": r2e.predict_session(ph0_m)["est_mmse"],
                        "R2a_Causal_DiffHMM_Linear11": r2a.predict_session(ph0_m)["est_mmse"],
                        "R2b_Causal_DiffHMM_MLP43": r2b.predict_session(ph0_m)["est_mmse"],
                        "R2d_E2E_Causal_H5_DiffHMM_Linear11": r2d.predict_session(ph5_m)["est_mmse"],
                        "R3c_Ridge_Spectral_439": np.clip(
                            ridge_res["model"].predict((D.flatten_salience_tensor(ph0_m) - ridge_res["mean"]) / ridge_res["scale"]), 12.0, 120.0
                        ),
                        "R3b_Spectral_MLP_15k": np.clip(
                            mlp_res["model"].predict((D.flatten_salience_tensor(ph0_m) - mlp_res["mean"]) / mlp_res["scale"]), 12.0, 120.0
                        ),
                        "R3a_Causal_TCN_16k": tcn.predict_session(ph0_m),
                    }
                surr_fold[m] = per_seed
            out["surr_fold"] = surr_fold

    return out


def _lag_profile(
    sessions: list[str],
    seq_a: dict[str, np.ndarray],
    seq_b: dict[str, np.ndarray],
    vmap: dict[str, np.ndarray],
    max_hop: int = 8,
) -> dict[str, Any]:
    """Cross-correlation rmcorr profile over `[-2*max_hop, +2*max_hop]` seconds (`+lag` => `seq_a` trails `seq_b`)."""
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


def _run_stage2_invariance_causality_shift(
    fold_r2e_thetas: list[np.ndarray],
    fold_r2a_thetas: list[np.ndarray],
    fold_r2b_thetas: list[np.ndarray],
    lags_map: dict[str, dict[str, Any]],
    r3_shift_folds: list[dict[str, dict[str, float]]],
    radar_shift_probe: dict[str, Any],
) -> dict[str, Any]:
    """Verify O(2)xR+ invariance, strict causality, operator vs clinical-grid shift equivariance & [18,120] shift."""
    # 1. Exact O(2)xR+ invariance on E2E causal R2 (H5 salience + DiffHMM) on clipped 2.3 & clean 7.3
    mean_th_e = np.mean(np.asarray(fold_r2e_thetas, dtype=np.float64), axis=0)
    mean_th_a = np.mean(np.asarray(fold_r2a_thetas, dtype=np.float64), axis=0)
    mean_th_b = np.mean(np.asarray(fold_r2b_thetas, dtype=np.float64), axis=0)
    r2e_mean = D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward")
    r2e_mean.theta = mean_th_e
    r2a_mean = D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward")
    r2a_mean.theta = mean_th_a
    r2b_mean = D.DifferentiableSymmetricHMM(mode="mlp", decoder="causal_forward")
    r2b_mean.theta = mean_th_b

    inv_res: dict[str, dict[str, float]] = {}
    frozen_h0_diag: dict[str, dict[str, float]] = {}
    for sess_check, label in (("2.3", "clipped_2_3"), ("7.3", "clean_7_3")):
        bi, bq = load_radar(sess_check)
        n_sub = 480 + 9 * 32
        bi, bq = bi[:n_sub], bq[:n_sub]
        starts = [s for s in range(0, 10 * 32, 32)]

        def _eval_r2_slice(ai: np.ndarray, aq: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
            phs_h5, phs_h0 = [], []
            for s0 in starts:
                dr = bandpass(csd(ai[s0 : s0 + 480], aq[s0 : s0 + 480]))
                dh5 = bandpass(W.mitigate_hpss_v2(dr, window="cola_hann", lookahead=0, in_band=True)["x_hat"])
                fc5 = autocorr_coarse(dh5)
                phs_h5.append(D.extract_salience_features_window(dr, dh5, fc5, VT.BR_GRID_1BPM))
                dh0 = bandpass(mitigate_hpss(dr)["x_hat"])
                fc0 = autocorr_coarse(dh0)
                phs_h0.append(D.extract_salience_features_window(dr, dh0, fc0, VT.BR_GRID_1BPM))
            p5 = np.asarray(phs_h5)
            p0 = np.asarray(phs_h0)
            return (
                {
                    "R2e_H5_FilterSafe8": r2e_mean.predict_session(p5)["est_mmse"],
                    "R2a_H5_Linear11": r2a_mean.predict_session(p5)["est_mmse"],
                    "R2b_H5_MLP43": r2b_mean.predict_session(p5)["est_mmse"],
                },
                {
                    "R2e_H0_FilterSafe8": r2e_mean.predict_session(p0)["est_mmse"],
                    "R2a_H0_Linear11": r2a_mean.predict_session(p0)["est_mmse"],
                    "R2b_H0_MLP43": r2b_mean.predict_session(p0)["est_mmse"],
                },
            )

        base_h5, base_h0 = _eval_r2_slice(bi, bq)
        z_rot = (bi + 1j * bq) * np.exp(1j * np.pi / 3.0)
        diffs_h5: dict[str, float] = {}
        diffs_h0: dict[str, float] = {}
        for tname, (ti, tq) in {
            "scale_3p7x": (3.7 * bi, 3.7 * bq),
            "phase_rot_pi_over_3": (z_rot.real, z_rot.imag),
            "iq_sign_flip": (-bi, -bq),
            "iq_channel_swap": (bq, bi),
        }.items():
            alt_h5, alt_h0 = _eval_r2_slice(ti, tq)
            diffs_h5[tname] = float(max(np.max(np.abs(base_h5[k] - alt_h5[k])) for k in base_h5))
            diffs_h0[tname] = float(max(np.max(np.abs(base_h0[k] - alt_h0[k])) for k in base_h0))
        inv_res[label] = diffs_h5
        frozen_h0_diag[label] = diffs_h0

    # 2. Exact strict causality (t > t0 perturbation == 0.0 diff)
    rng = np.random.default_rng(IO.RANDOM_SEED)
    phi_t = rng.normal(-2.0, 1.0, size=(30, VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS))
    phi_p = phi_t.copy()
    phi_p[18:] = rng.normal(-2.0, 2.0, size=(12, VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS))
    r2e_cdiff = float(np.max(np.abs(r2e_mean.predict_session(phi_t)["est_mmse"][:18] - r2e_mean.predict_session(phi_p)["est_mmse"][:18])))
    r2a_cdiff = float(np.max(np.abs(r2a_mean.predict_session(phi_t)["est_mmse"][:18] - r2a_mean.predict_session(phi_p)["est_mmse"][:18])))
    r2b_cdiff = float(np.max(np.abs(r2b_mean.predict_session(phi_t)["est_mmse"][:18] - r2b_mean.predict_session(phi_p)["est_mmse"][:18])))

    tcn_chk = D.CausalDilatedTCN(in_dim=438, channels=24, dilations=(1, 2, 4), seed=1)
    tcn_cdiff = float(np.max(np.abs(tcn_chk.predict_session(phi_t)[:18] - tcn_chk.predict_session(phi_p)[:18])))

    # 3a. Exact operator frequency-shift equivariance (+10 bpm on wide interior grid [-60, 180], eps=1e-12)
    wide_grid = np.arange(-60.0, 180.0 + 1e-9, 1.0)
    gw0 = np.full(15, 40.0)
    gw1 = gw0 + 10.0
    pw_base = np.zeros((15, wide_grid.size, D.N_SAL_CHANNELS))
    pw_shift = np.zeros((15, wide_grid.size, D.N_SAL_CHANNELS))
    for ch in range(D.N_SAL_CHANNELS):
        if ch == 4:
            gb = np.maximum(0.05, np.exp(-0.5 * ((wide_grid[None, :] - gw0[:, None]) / float(C.NLS_RADIUS_BPM)) ** 2))
            gs = np.maximum(0.05, np.exp(-0.5 * ((wide_grid[None, :] - gw1[:, None]) / float(C.NLS_RADIUS_BPM)) ** 2))
        else:
            gb = np.exp(-0.5 * ((wide_grid[None, :] - gw0[:, None]) / 2.5) ** 2)
            gs = np.exp(-0.5 * ((wide_grid[None, :] - gw1[:, None]) / 2.5) ** 2)
        pw_base[:, :, ch] = np.log(gb + 1e-12)
        pw_shift[:, :, ch] = np.log(gs + 1e-12)
    op_eq_diff_e = float(
        np.max(np.abs((r2e_mean.predict_session(pw_shift, grid_bpm=wide_grid)["est_mmse"] - r2e_mean.predict_session(pw_base, grid_bpm=wide_grid)["est_mmse"]) - 10.0))
    )
    op_eq_diff_a = float(
        np.max(np.abs((r2a_mean.predict_session(pw_shift, grid_bpm=wide_grid)["est_mmse"] - r2a_mean.predict_session(pw_base, grid_bpm=wide_grid)["est_mmse"]) - 10.0))
    )
    op_eq_diff_b = float(
        np.max(np.abs((r2b_mean.predict_session(pw_shift, grid_bpm=wide_grid)["est_mmse"] - r2b_mean.predict_session(pw_base, grid_bpm=wide_grid)["est_mmse"]) - 10.0))
    )

    # 3b. Calibrated [18, 90] interior shift diagnostic (matching ch4 >= 0.05 floor)
    p_base, p_shift = _make_clinical_shift_salience_pair(f0=36.0, delta=10.0, n_t=15, grid=VT.BR_GRID_1BPM)
    pe_0 = r2e_mean.predict_session(p_base)["est_mmse"]
    pe_1 = r2e_mean.predict_session(p_shift)["est_mmse"]
    pa_0 = r2a_mean.predict_session(p_base)["est_mmse"]
    pa_1 = r2a_mean.predict_session(p_shift)["est_mmse"]
    pb_0 = r2b_mean.predict_session(p_base)["est_mmse"]
    pb_1 = r2b_mean.predict_session(p_shift)["est_mmse"]
    clinical_shift_diag: dict[str, Any] = {
        "R2e_Causal_DiffHMM_FilterSafe8": {
            "max_abs_err_bpm": float(np.max(np.abs((pe_1 - pe_0) - 10.0))),
            "mean_shift_bpm": float(np.mean(pe_1 - pe_0)),
        },
        "R2a_Causal_DiffHMM_Linear11": {
            "max_abs_err_bpm": float(np.max(np.abs((pa_1 - pa_0) - 10.0))),
            "mean_shift_bpm": float(np.mean(pa_1 - pa_0)),
        },
        "R2b_Causal_DiffHMM_MLP43": {
            "max_abs_err_bpm": float(np.max(np.abs((pb_1 - pb_0) - 10.0))),
            "mean_shift_bpm": float(np.mean(pb_1 - pb_0)),
        },
    }
    if r3_shift_folds:
        for km in ("R3c_Ridge_Spectral_439", "R3b_Spectral_MLP_15k", "R3a_Causal_TCN_16k"):
            clinical_shift_diag[km] = {
                "mean_max_abs_err_bpm": float(np.mean([row[km]["max_abs_err_bpm"] for row in r3_shift_folds])),
                "mean_shift_bpm": float(np.mean([row[km]["mean_shift_bpm"] for row in r3_shift_folds])),
                "sd_shift_bpm": float(np.std([row[km]["mean_shift_bpm"] for row in r3_shift_folds], ddof=1)) if len(r3_shift_folds) > 1 else 0.0,
            }

    # 4. Shifted-grid midpoint check [18, 90] vs [18, 120] on trained R2e, R2a & R2b across 5 sessions
    sess5 = ["2.3", "4.1", "7.3", "9.2", "11.3"]
    grid90 = np.arange(18.0, 90.0 + 1e-9, 1.0)
    grid120 = np.arange(18.0, 120.0 + 1e-9, 1.0)
    p90_s, p120_s, ref1_s = {}, {}, {}
    for m in sess5:
        bi, bq = load_radar(m)
        starts = list(range(0, len(bi) - 480 + 1, 32))
        cs = np.arange(len(starts), dtype=float) * C.UPDATE_S + C.WIN_S / 2.0
        ext = EDFX.extract_window_references(EDFX.load_edf_recording(m), cs, int(lags_map[m]["neutral_raw_nls_lag_s"]))
        v1 = np.asarray(ext["valid_vwin"], bool) & np.isfinite(ext["br_1hz_30s"])
        ref1_s[m] = (np.asarray(ext["br_1hz_30s"], float), v1)
        a90, a120 = [], []
        for s0 in starts:
            dr = bandpass(csd(bi[s0 : s0 + 480], bq[s0 : s0 + 480]))
            dh = bandpass(mitigate_hpss(dr)["x_hat"])
            fc = autocorr_coarse(dh)
            a90.append(D.extract_salience_features_window(dr, dh, fc, grid90))
            a120.append(D.extract_salience_features_window(dr, dh, fc, grid120))
        p90_s[m] = np.asarray(a90)
        p120_s[m] = np.asarray(a120)

    shift_grid_out = {}
    for lbl, mdl in (
        ("R2e_Causal_DiffHMM_FilterSafe8", r2e_mean),
        ("R2a_Causal_DiffHMM_Linear11", r2a_mean),
        ("R2b_Causal_DiffHMM_MLP43", r2b_mean),
    ):
        pr90 = {m: mdl.predict_session(p90_s[m], grid_bpm=grid90)["est_mmse"] for m in sess5}
        pr120 = {m: mdl.predict_session(p120_s[m], grid_bpm=grid120)["est_mmse"] for m in sess5}
        ag90, _ = E1._aggregate_method(sess5, pr90, ref1_s)
        ag120, _ = E1._aggregate_method(sess5, pr120, ref1_s)
        b90 = float(ag90["ba_session_level"]["bias"])
        b120 = float(ag120["ba_session_level"]["bias"])
        shift_grid_out[lbl] = {
            "rmse_18_90": float(ag90["rmse_sess"]),
            "bias_18_90": b90,
            "rmse_18_120": float(ag120["rmse_sess"]),
            "bias_18_120": b120,
            "abs_bias_shift_bpm": float(abs(b120 - b90)),
        }

    return {
        "invariance_r2_max_diff_bpm": inv_res,
        "all_r2_invariances_lt_1e6": bool(all(val < 1e-6 for sub in inv_res.values() for val in sub.values())),
        "frozen_h0_swap_asymmetry_diagnostic_bpm": frozen_h0_diag,
        "causality_r2e_max_diff": r2e_cdiff,
        "causality_r2a_max_diff": r2a_cdiff,
        "causality_r2b_max_diff": r2b_cdiff,
        "causality_r3a_tcn_max_diff": tcn_cdiff,
        "freq_shift_equivariance_operator_max_diff_bpm": {"R2e": op_eq_diff_e, "R2a": op_eq_diff_a, "R2b": op_eq_diff_b},
        "freq_shift_equivariance_clinical_grid_bpm": clinical_shift_diag,
        "radar_waveform_shift_probe_36_to_46_bpm": radar_shift_probe,
        "shifted_grid_18_90_vs_18_120": shift_grid_out,
        "r2e_mean_theta": mean_th_e.tolist(),
        "r2a_mean_theta": mean_th_a.tolist(),
        "r2b_mean_theta": mean_th_b.tolist(),
    }


def _verify_stage1_parity(cohort_results: dict[str, Any]) -> float:
    """Assert exact < 1e-12 parity against locked Stage 1.1 (`stage1-parsimony-v3`) R0/R1 metrics."""
    s1 = json.loads(STAGE1_BUNDLE_JSON.read_text(encoding="utf-8"))
    if s1.get("schema_version") != "stage1-parsimony-v3":
        raise RuntimeError(f"Expected Stage 1.1 schema stage1-parsimony-v3, got {s1.get('schema_version')!r}")
    max_diff = 0.0
    for coh in COHORTS:
        for rlbl in ("br_1hz_30s", "br_trig_bpm"):
            for km in (
                "R0_HPSS_NLS_Locked",
                "R1a_Online_Forward_MMSE_0s",
                "R1f_Offline_FB_MMSE_Inf",
                "R1g_Offline_Viterbi_Unpinned",
            ):
                m1 = s1["rung1_lopo"][coh][rlbl]["methods"][km]
                m2 = cohort_results[coh][rlbl]["methods"][km]
                for fld in ("rmse_sess", "rmse_pool", "r_within_rmcorr"):
                    diff = abs(float(m2[fld]) - float(m1[fld]))
                    if diff > max_diff:
                        max_diff = diff
                    if diff >= 1e-12:
                        raise AssertionError(
                            f"Stage 1.1 parity mismatch on {coh}/{rlbl}/{km}/{fld}: "
                            f"stage2={m2[fld]} vs stage1={m1[fld]} (diff={diff:.3e})"
                        )
    return float(max_diff)


def main() -> None:
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="EXP-2 Stage 2 Complexity Ladder & Shortcut Audit")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-surrogates", action="store_true")
    args = parser.parse_args()

    print("[EXP-2] Verifying frozen baseline caches read-only...")
    IO.verify_all_baseline_caches()

    wp_sha = IO.sha256_file(WP_MODULE_PATH)
    diff_sha = IO.sha256_file(DIFF_MODULE_PATH)
    all_eval_sessions = IO.get_cohort_sessions("all_paired", include_zero_valid=False)

    print(f"[EXP-2] Extracting Stage 2 6-channel salience tensors across {len(all_eval_sessions)} sessions (workers={args.workers})...")
    sal_tasks = [(m, wp_sha, diff_sha, args.force) for m in all_eval_sessions]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            sal_logs = list(ex.map(_extract_salience_session, sal_tasks))
    else:
        sal_logs = [_extract_salience_session(t) for t in sal_tasks]
    print(f"[EXP-2] Salience tensors ready ({sum(1 for _, s in sal_logs if s == 'computed')} computed, {sum(1 for _, s in sal_logs if s == 'cached')} cached).")

    lags_map = IO.load_locked_neutral_lags()
    p30_list = IO.get_cohort_sessions("pool30_sync", include_zero_valid=False)
    p30_set = set(p30_list)
    ref_1hz_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ref_trig_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ref_trig_lag_aligned_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ref_trig_count_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pause_map: dict[str, np.ndarray] = {}

    for m in all_eval_sessions:
        phi0, _ = _load_salience_session(m)
        n_win = phi0.shape[0]
        centers = np.arange(n_win, dtype=float) * C.UPDATE_S + C.WIN_S / 2.0
        edf = EDFX.load_edf_recording(m)
        lag_val = int(lags_map[m]["neutral_raw_nls_lag_s"])
        ext = EDFX.extract_window_references(edf, centers, lag_s=lag_val)
        v1 = np.asarray(ext["valid_vwin"], dtype=bool) & np.isfinite(ext["br_1hz_30s"])
        vt = v1 & np.isfinite(ext["br_trig_bpm"])
        pz = v1 & np.isfinite(ext["resp_max_gap_s"]) & (ext["resp_max_gap_s"] > 5.0)
        ref_1hz_map[m] = (np.asarray(ext["br_1hz_30s"], dtype=np.float64), v1)
        ref_trig_map[m] = (np.asarray(ext["br_trig_bpm"], dtype=np.float64), vt)
        pause_map[m] = pz

        # Lag-compensated Trig across ALL sessions (lag_val + 10 removes VitaGuard's empirical +10 s monitor delay)
        ext_m10 = EDFX.extract_window_references(edf, centers, lag_s=lag_val + 10)
        vta = v1 & np.isfinite(ext_m10["br_trig_bpm"])
        ref_trig_lag_aligned_map[m] = (np.asarray(ext_m10["br_trig_bpm"], dtype=np.float64), vta)

        if m in p30_set:
            # Whole-window breath-count rate 60 * N_events / 30 s over identical [a, b) at lag_val
            resp_trig_t = EDFX.detect_trigger_times(edf.resp_trig_1024, fs_hz=edf.fs_trig, min_refractory_s=0.30)
            tc = centers - float(lag_val)
            a_arr = np.floor(tc - float(C.WIN_S) / 2.0).astype(int)
            b_arr = a_arr + int(C.WIN_S)
            br_cnt = np.full(n_win, np.nan, dtype=np.float64)
            for w_idx in range(n_win):
                if bool(ext["in_bounds"][w_idx]):
                    n_ev = int(np.sum((resp_trig_t >= float(a_arr[w_idx])) & (resp_trig_t < float(b_arr[w_idx]))))
                    br_cnt[w_idx] = 60.0 * float(n_ev) / float(C.WIN_S)
            vtc = v1 & np.isfinite(br_cnt)
            ref_trig_count_map[m] = (br_cnt, vtc)

    if not args.skip_surrogates:
        print(f"[EXP-2] Preparing 5-seed phase-scrambled salience tensors for pool30_sync (workers={args.workers})...")
        stasks = [(m, 5, IO.RANDOM_SEED, wp_sha, diff_sha, args.force) for m in p30_list]
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                slogs = list(ex.map(_extract_surr_salience_session, stasks))
        else:
            slogs = [_extract_surr_salience_session(t) for t in stasks]
        print(f"[EXP-2] Surrogate salience ready ({sum(1 for _, s in slogs if s == 'computed')} computed, {sum(1 for _, s in slogs if s == 'cached')} cached).")

    print("[EXP-2] Generating zero-touch 24 GHz synthetic frequency sweep (20-80 bpm + chirp)...")
    synth_suite = _generate_synthetic_suite()

    print("[EXP-2] Decoding locked Stage 1 R0/R1 grid once across union of cohort sessions...")
    union_coh_sessions = sorted({m for coh in COHORTS for m in IO.get_cohort_sessions(coh, include_zero_valid=False)})
    stage1_decoded_cache = _precompute_stage1_decoded(union_coh_sessions)

    cohort_results: dict[str, Any] = {}
    p30_preds_all: dict[str, dict[str, np.ndarray]] = {}
    p30_unc_all: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    p30_synth_folds: list[dict[str, dict[str, np.ndarray]]] = []
    p30_surr_by_sess: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    p30_r2e_thetas: list[np.ndarray] = []
    p30_r2a_thetas: list[np.ndarray] = []
    p30_r2b_thetas: list[np.ndarray] = []
    p30_r2e_interps: list[dict[str, float]] = []
    p30_r2a_interps: list[dict[str, float]] = []
    p30_r2d_interps: list[dict[str, float]] = []
    p30_r3_shift_folds: list[dict[str, dict[str, float]]] = []
    p30_fold_logs: list[dict[str, Any]] = []

    for coh in COHORTS:
        print(f"[EXP-2] Running LOPO complexity ladder on cohort={coh} (workers={args.workers})...")
        slist = [m for m in IO.get_cohort_sessions(coh, include_zero_valid=False) if np.any(ref_1hz_map[m][1])]
        r1_out = _compute_stage1_r0_r1(coh, ref_1hz_map, decoded_cache=stage1_decoded_cache)
        r1a_cfgs = r1_out.pop("_r1a_fold_cfgs")

        folds = list(IO.iter_lopo_folds(coh))
        fold_tasks = []
        for f_idx, fold in enumerate(folds):
            tr_ms = [m for m in fold.train_sessions if m in slist]
            te_ms = [m for m in fold.test_sessions if m in slist]
            if not te_ms:
                continue
            fold_tasks.append({
                "fold_idx": f_idx,
                "patient_id": int(fold.patient_id),
                "train_ms": tr_ms,
                "test_ms": te_ms,
                "ref_map": {m: ref_1hz_map[m] for m in tr_ms},
                "run_extra": bool(coh == "pool30_sync"),
                "synth_suite": synth_suite if coh == "pool30_sync" else None,
                "r1a_cfg": r1a_cfgs[fold.patient_id] if coh == "pool30_sync" else None,
                "skip_surrogates": bool(args.skip_surrogates),
            })

        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                f_outs = list(ex.map(_fit_eval_fold_worker, fold_tasks))
        else:
            f_outs = [_fit_eval_fold_worker(t) for t in fold_tasks]

        preds_coh: dict[str, dict[str, np.ndarray]] = {k: dict(r1_out[k]) if k in r1_out else {} for k in MODEL_ORDER}
        unc_coh: dict[str, dict[str, dict[str, np.ndarray]]] = {
            "R2e_Causal_DiffHMM_FilterSafe8": {},
            "R2a_Causal_DiffHMM_Linear11": {},
            "R2b_Causal_DiffHMM_MLP43": {},
            "R2d_E2E_Causal_H5_DiffHMM_Linear11": {},
        }
        for fo in f_outs:
            for k, m_dict in fo["test_preds"].items():
                preds_coh[k].update(m_dict)
            for k, u_dict in fo["test_unc"].items():
                unc_coh[k].update(u_dict)

        if coh == "pool30_sync":
            p30_preds_all = preds_coh
            p30_unc_all = unc_coh
            for fo in f_outs:
                p30_r2e_thetas.append(np.asarray(fo["r2e_theta"], dtype=np.float64))
                p30_r2a_thetas.append(np.asarray(fo["r2a_theta"], dtype=np.float64))
                p30_r2b_thetas.append(np.asarray(fo["r2b_theta"], dtype=np.float64))
                p30_r2e_interps.append(fo["r2e_interp"])
                p30_r2a_interps.append(fo["r2a_interp"])
                p30_r2d_interps.append(fo["r2d_interp"])
                p30_fold_logs.append({
                    "patient_id": fo["patient_id"],
                    "val_patient_id": fo["val_patient_id"],
                    "test_sessions": fo["test_ms"],
                    "r2e_nit": fo["r2e_nit"],
                    "r2a_nit": fo["r2a_nit"],
                    "r2b_nit": fo["r2b_nit"],
                    "r2f_nit": fo["r2f_nit"],
                    "r2c_nit": fo["r2c_nit"],
                    "r2d_nit": fo["r2d_nit"],
                    "r2_success": fo["r2_success"],
                    "r2e_interp": fo["r2e_interp"],
                    "r2a_interp": fo["r2a_interp"],
                    "r2b_interp": fo["r2b_interp"],
                    "r2f_interp": fo["r2f_interp"],
                    "r2c_interp": fo["r2c_interp"],
                    "r2d_interp": fo["r2d_interp"],
                    "r3_hyperparams": fo["r3_hyperparams"],
                })
                if "r3_clinical_shift_err_bpm" in fo:
                    p30_r3_shift_folds.append(fo["r3_clinical_shift_err_bpm"])
                if "synth_fold" in fo:
                    p30_synth_folds.append(fo["synth_fold"])
                if "surr_fold" in fo:
                    p30_surr_by_sess.update(fo["surr_fold"])

        coh_block: dict[str, Any] = {}
        ref_pairs = (
            (
                ("br_1hz_30s", ref_1hz_map),
                ("br_trig_bpm", ref_trig_map),
                ("br_trig_bpm_lag_aligned", ref_trig_lag_aligned_map),
                ("br_trig_count_bpm", ref_trig_count_map),
            )
            if coh == "pool30_sync"
            else (
                ("br_1hz_30s", ref_1hz_map),
                ("br_trig_bpm", ref_trig_map),
                ("br_trig_bpm_lag_aligned", ref_trig_lag_aligned_map),
            )
        )
        for ref_lbl, rmap in ref_pairs:
            const_sum, per_sess_const = E1._constant_baselines(slist, rmap)
            m_agg, m_ps = {}, {}
            for k in MODEL_ORDER:
                m_agg[k], m_ps[k] = E1._aggregate_method(slist, preds_coh[k], rmap)
            challengers = [k for k in MODEL_ORDER if k != "R0_HPSS_NLS_Locked"]
            paired_r0 = {k: E1._paired_comparison(slist, m_ps[k], m_ps["R0_HPSS_NLS_Locked"]) for k in challengers}
            E1._holm_update_paired(paired_r0, challengers)

            r2_r3_keys = [k for k in MODEL_ORDER if MODEL_META[k]["rung"] >= 2]
            paired_r1a = {k: E1._paired_comparison(slist, m_ps[k], m_ps["R1a_Online_Forward_MMSE_0s"]) for k in r2_r3_keys}
            E1._holm_update_paired(paired_r1a, r2_r3_keys)

            paired_const = {k: E1._paired_comparison(slist, m_ps[k], per_sess_const) for k in MODEL_ORDER}
            E1._holm_update_paired(paired_const, list(MODEL_ORDER))

            coh_block[ref_lbl] = {
                "constant_baselines": const_sum,
                "methods": m_agg,
                "paired_vs_R0": paired_r0,
                "paired_vs_R1a": paired_r1a,
                "paired_vs_const_loso": paired_const,
            }
            if coh == "pool30_sync":
                coh_block[ref_lbl]["_per_sess"] = m_ps  # strip later for JSON cleanliness

        cohort_results[coh] = coh_block

    stage1_parity_max_diff = _verify_stage1_parity(cohort_results)
    print(f"[EXP-2] Verified exact Stage 1.1 parity across 3 cohorts x 2 refs x 4 anchors (max diff={stage1_parity_max_diff:.2e}).")

    # Compute [-16, +16] s rmcorr lag profiles & variance-collapse ratio on pool30_sync
    r1_seq = {m: ref_1hz_map[m][0] for m in p30_list}
    v1_only = {m: ref_1hz_map[m][1] for m in p30_list}
    rt_seq = {m: ref_trig_map[m][0] for m in p30_list}
    vt_only = {m: ref_trig_map[m][1] for m in p30_list}
    rta_seq = {m: ref_trig_lag_aligned_map[m][0] for m in p30_list}
    vta_only = {m: ref_trig_lag_aligned_map[m][1] for m in p30_list}

    r0_vs_trig_lag_aligned = _lag_profile(p30_list, p30_preds_all["R0_HPSS_NLS_Locked"], rta_seq, vta_only)
    if abs(int(r0_vs_trig_lag_aligned["peak_lag_s"])) > 2:
        raise AssertionError(f"Expected R0 vs br_trig_bpm_lag_aligned peak_lag_s within +-2 s, got {r0_vs_trig_lag_aligned['peak_lag_s']} s")

    lag_profiles_pool30: dict[str, Any] = {
        "br_1hz_30s_vs_br_trig_bpm": _lag_profile(p30_list, r1_seq, rt_seq, vt_only),
        "br_1hz_30s_vs_br_trig_bpm_lag_aligned": _lag_profile(p30_list, r1_seq, rta_seq, vta_only),
        "R0_vs_br_trig_bpm_lag_aligned": r0_vs_trig_lag_aligned,
        "vs_br_1hz_30s": {},
        "vs_br_trig_bpm": {},
    }
    pred_sd_ratio_1hz: dict[str, float] = {}
    pred_sd_ratio_tla: dict[str, float] = {}
    for k in MODEL_ORDER:
        lag_profiles_pool30["vs_br_1hz_30s"][k] = _lag_profile(p30_list, p30_preds_all[k], r1_seq, v1_only)
        lag_profiles_pool30["vs_br_trig_bpm"][k] = _lag_profile(p30_list, p30_preds_all[k], rt_seq, vt_only)
        ratios_1hz, ratios_tla = [], []
        for m in p30_list:
            rv, vv = ref_1hz_map[m]
            sd_r = float(np.std(rv[vv]))
            if sd_r > 1e-6:
                ratios_1hz.append(float(np.std(p30_preds_all[k][m][vv]) / sd_r))
            rt, vt_m = ref_trig_lag_aligned_map[m]
            sd_t = float(np.std(rt[vt_m]))
            if sd_t > 1e-6:
                ratios_tla.append(float(np.std(p30_preds_all[k][m][vt_m]) / sd_t))
        pred_sd_ratio_1hz[k] = float(np.mean(ratios_1hz))
        pred_sd_ratio_tla[k] = float(np.mean(ratios_tla))

    mean_sd_1hz = float(np.mean([float(np.std(ref_1hz_map[m][0][ref_1hz_map[m][1]])) for m in p30_list]))
    mean_sd_tla = float(np.mean([float(np.std(ref_trig_lag_aligned_map[m][0][ref_trig_lag_aligned_map[m][1]])) for m in p30_list]))
    ref_sd_summary = {
        "mean_sd_1hz": mean_sd_1hz,
        "mean_sd_trig_lag_aligned": mean_sd_tla,
        "ratio_1hz_over_trig_lag_aligned": float(mean_sd_1hz / mean_sd_tla),
    }

    # Assemble 4-reference shortcut-learning diagnostic across rungs on pool30_sync
    shortcut_summary: dict[str, Any] = {}
    m1_p30 = cohort_results["pool30_sync"]["br_1hz_30s"]["methods"]
    mt_p30 = cohort_results["pool30_sync"]["br_trig_bpm"]["methods"]
    mta_p30 = cohort_results["pool30_sync"]["br_trig_bpm_lag_aligned"]["methods"]
    mtc_p30 = cohort_results["pool30_sync"]["br_trig_count_bpm"]["methods"]

    rank_1hz = {k: i + 1 for i, k in enumerate(sorted(MODEL_ORDER, key=lambda x: m1_p30[x]["rmse_sess"]))}
    rank_trig = {k: i + 1 for i, k in enumerate(sorted(MODEL_ORDER, key=lambda x: mt_p30[x]["rmse_sess"]))}
    rank_tla = {k: i + 1 for i, k in enumerate(sorted(MODEL_ORDER, key=lambda x: mta_p30[x]["rmse_sess"]))}
    rank_tcnt = {k: i + 1 for i, k in enumerate(sorted(MODEL_ORDER, key=lambda x: mtc_p30[x]["rmse_sess"]))}
    for k in MODEL_ORDER:
        shortcut_summary[k] = {
            **MODEL_META[k],
            "rmse_1hz": float(m1_p30[k]["rmse_sess"]),
            "rmse_trig": float(mt_p30[k]["rmse_sess"]),
            "rmse_trig_lag_aligned": float(mta_p30[k]["rmse_sess"]),
            "rmse_trig_count": float(mtc_p30[k]["rmse_sess"]),
            "trig_minus_1hz_rmse": float(mt_p30[k]["rmse_sess"] - m1_p30[k]["rmse_sess"]),
            "trig_lag_aligned_minus_1hz_rmse": float(mta_p30[k]["rmse_sess"] - m1_p30[k]["rmse_sess"]),
            "trig_count_minus_1hz_rmse": float(mtc_p30[k]["rmse_sess"] - m1_p30[k]["rmse_sess"]),
            "bias_1hz": float(m1_p30[k]["ba_session_level"]["bias"]),
            "bias_trig": float(mt_p30[k]["ba_session_level"]["bias"]),
            "bias_trig_lag_aligned": float(mta_p30[k]["ba_session_level"]["bias"]),
            "bias_trig_count": float(mtc_p30[k]["ba_session_level"]["bias"]),
            "r_within_1hz": float(m1_p30[k]["r_within_rmcorr"]),
            "r_within_trig": float(mt_p30[k]["r_within_rmcorr"]),
            "r_within_trig_lag_aligned": float(mta_p30[k]["r_within_rmcorr"]),
            "r_within_trig_count": float(mtc_p30[k]["r_within_rmcorr"]),
            "pred_sd_ratio_1hz": float(pred_sd_ratio_1hz[k]),
            "pred_sd_ratio_trig_lag_aligned": float(pred_sd_ratio_tla[k]),
            "peak_lag_vs_1hz_s": int(lag_profiles_pool30["vs_br_1hz_30s"][k]["peak_lag_s"]),
            "peak_r_vs_1hz": float(lag_profiles_pool30["vs_br_1hz_30s"][k]["peak_r"]),
            "peak_lag_vs_trig_s": int(lag_profiles_pool30["vs_br_trig_bpm"][k]["peak_lag_s"]),
            "peak_r_vs_trig": float(lag_profiles_pool30["vs_br_trig_bpm"][k]["peak_r"]),
            "rank_1hz": rank_1hz[k],
            "rank_trig": rank_trig[k],
            "rank_trig_lag_aligned": rank_tla[k],
            "rank_trig_count": rank_tcnt[k],
            "rank_drop_1hz_to_trig": int(rank_trig[k] - rank_1hz[k]),
            "rank_drop_1hz_to_trig_lag_aligned": int(rank_tla[k] - rank_1hz[k]),
        }

    # Summarize interpretable R2e, R2a & R2d weights across 11 pool30_sync folds
    interp_summary = {}
    for lbl, lst in (
        ("R2e_Causal_DiffHMM_FilterSafe8", p30_r2e_interps),
        ("R2a_Causal_DiffHMM_Linear11", p30_r2a_interps),
        ("R2d_E2E_Causal_H5_DiffHMM_Linear11", p30_r2d_interps),
    ):
        keys = list(lst[0].keys())
        interp_summary[lbl] = {
            kp: {
                "mean": float(np.mean([row[kp] for row in lst])),
                "sd": float(np.std([row[kp] for row in lst], ddof=1)),
            }
            for kp in keys
        }

    # Unsupervised bandpass filter-edge profile across all valid pool30_sync windows
    phi_p30_valid = np.concatenate([_load_salience_session(m)[0][ref_1hz_map[m][1]] for m in p30_list], axis=0)
    w3_m = float(interp_summary["R2a_Causal_DiffHMM_Linear11"]["w3_p3_harm"]["mean"])
    w4_m = float(interp_summary["R2a_Causal_DiffHMM_Linear11"]["w4_phalf_sub"]["mean"])
    prof_freqs = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    prof_rows: dict[str, Any] = {}
    for f_bpm in prof_freqs:
        idx_f = int(np.argmin(np.abs(VT.BR_GRID_1BPM - float(f_bpm))))
        ch_means = [float(np.mean(phi_p30_valid[:, idx_f, ch])) for ch in range(D.N_SAL_CHANNELS)]
        prof_rows[str(f_bpm)] = {
            "ch0_1f": ch_means[0],
            "ch1_2f": ch_means[1],
            "ch2_3f": ch_means[2],
            "ch3_0p5f": ch_means[3],
            "ch4_anchor": ch_means[4],
            "ch5_raw": ch_means[5],
            "r2a_w3_ch2_plus_w4_ch3": float(w3_m * ch_means[2] + w4_m * ch_means[3]),
        }
    filter_edge_profile = {
        "n_windows": int(phi_p30_valid.shape[0]),
        "by_freq_bpm": prof_rows,
        "r2a_cliff_50_to_60_bpm": float(prof_rows["60"]["r2a_w3_ch2_plus_w4_ch3"] - prof_rows["50"]["r2a_w3_ch2_plus_w4_ch3"]),
        "r2a_cliff_20_to_90_bpm": float(prof_rows["90"]["r2a_w3_ch2_plus_w4_ch3"] - prof_rows["20"]["r2a_w3_ch2_plus_w4_ch3"]),
        "r2e_filter_edge_cliff_bpm": 0.0,
        "ref_within_session_sd_summary": ref_sd_summary,
    }

    r2_convergence_summary = {
        mk: {
            "maxiter": 200,
            "n_converged": int(sum(1 for row in p30_fold_logs if row["r2_success"][mk])),
            "n_at_maxiter": int(sum(1 for row in p30_fold_logs if int(row[f"{mk}_nit"]) >= 200)),
            "median_nit": float(np.median([int(row[f"{mk}_nit"]) for row in p30_fold_logs])),
            "max_nit": int(max(int(row[f"{mk}_nit"]) for row in p30_fold_logs)),
        }
        for mk in ("r2e", "r2a", "r2b", "r2f", "r2c", "r2d")
    }

    # Summarize synthetic out-of-range frequency sweep & real-waveform 36 -> 46 bpm shift probe (11 folds)
    synth_models = (
        "R0_HPSS_NLS_Locked",
        "R1a_Online_Forward_MMSE_0s",
        "R2e_Causal_DiffHMM_FilterSafe8",
        "R2a_Causal_DiffHMM_Linear11",
        "R2b_Causal_DiffHMM_MLP43",
        "R2d_E2E_Causal_H5_DiffHMM_Linear11",
        "R3c_Ridge_Spectral_439",
        "R3b_Spectral_MLP_15k",
        "R3a_Causal_TCN_16k",
    )
    synth_sweep_cases = (
        "stat_28bpm",
        "stat_35bpm",
        "stat_50bpm",
        "stat_65bpm",
        "stat_80bpm",
        "chirp_30_to_75bpm",
        "stat_20bpm_kaiser_edge",
    )
    synth_summary: dict[str, Any] = {}
    for cname in synth_sweep_cases:
        sdata = synth_suite[cname]
        truth = sdata["truth_bpm"]
        sub: dict[str, Any] = {}
        for km in synth_models:
            rmses, maes, biases, means_pred = [], [], [], []
            for f_dict in p30_synth_folds:
                pr = f_dict[cname][km]
                e = pr - truth
                rmses.append(float(np.sqrt(np.mean(e ** 2))))
                maes.append(float(np.mean(np.abs(e))))
                biases.append(float(np.mean(e)))
                means_pred.append(float(np.mean(pr)))
            sub[km] = {
                "rmse_mean": float(np.mean(rmses)),
                "rmse_sd": float(np.std(rmses, ddof=1)),
                "mae_mean": float(np.mean(maes)),
                "bias_mean": float(np.mean(biases)),
                "pred_mean": float(np.mean(means_pred)),
                "pred_sd_across_folds": float(np.std(means_pred, ddof=1)),
            }
        synth_summary[cname] = sub

    radar_shift_summary: dict[str, Any] = {}
    for km in synth_models:
        p36_means = [float(np.mean(fd["probe_shift_36bpm"][km])) for fd in p30_synth_folds]
        p46_means = [float(np.mean(fd["probe_shift_46bpm"][km])) for fd in p30_synth_folds]
        fold_shifts = [float(np.mean(fd["probe_shift_46bpm"][km] - fd["probe_shift_36bpm"][km])) for fd in p30_synth_folds]
        fold_max_errs = [float(np.max(np.abs((fd["probe_shift_46bpm"][km] - fd["probe_shift_36bpm"][km]) - 10.0))) for fd in p30_synth_folds]
        m_shift = float(np.mean(fold_shifts))
        radar_shift_summary[km] = {
            "pred_36_mean": float(np.mean(p36_means)),
            "pred_46_mean": float(np.mean(p46_means)),
            "mean_shift_bpm": m_shift,
            "sd_shift_across_folds_bpm": float(np.std(fold_shifts, ddof=1)) if len(fold_shifts) > 1 else 0.0,
            "abs_err_from_10_bpm": float(abs(m_shift - 10.0)),
            "mean_max_window_err_bpm": float(np.mean(fold_max_errs)),
        }

    # Summarize 5-seed phase-scrambled surrogates for R2/R3 models
    surr_summary: dict[str, Any] = {"skipped": bool(args.skip_surrogates), "n_seeds": 5}
    if not args.skip_surrogates:
        surr_keys = (
            "R2e_Causal_DiffHMM_FilterSafe8",
            "R2a_Causal_DiffHMM_Linear11",
            "R2b_Causal_DiffHMM_MLP43",
            "R2d_E2E_Causal_H5_DiffHMM_Linear11",
            "R3c_Ridge_Spectral_439",
            "R3b_Spectral_MLP_15k",
            "R3a_Causal_TCN_16k",
        )
        for ref_lbl, rmap in (("br_1hz_30s", ref_1hz_map), ("br_trig_bpm", ref_trig_map)):
            sub_r: dict[str, Any] = {}
            for km in surr_keys:
                rmses, rws = [], []
                for s_i in range(5):
                    emap = {m: p30_surr_by_sess[m][s_i][km] for m in p30_list}
                    ag, _ = E1._aggregate_method(p30_list, emap, rmap)
                    rmses.append(ag["rmse_sess"])
                    rws.append(ag["r_within_rmcorr"])
                sub_r[km] = {
                    "rmse_sess_mean": float(np.mean(rmses)),
                    "rmse_sess_sd": float(np.std(rmses, ddof=1)),
                    "r_within_rmcorr_mean": float(np.mean(rws)),
                }
            surr_summary[ref_lbl] = sub_r

    # WP4 causal uncertainty preview (pool & unweighted within-session Spearman) for R2e, R2a, R2b, R2d
    wp4_r2_preview: dict[str, Any] = {}
    for km in (
        "R2e_Causal_DiffHMM_FilterSafe8",
        "R2a_Causal_DiffHMM_Linear11",
        "R2b_Causal_DiffHMM_MLP43",
        "R2d_E2E_Causal_H5_DiffHMM_Linear11",
    ):
        res_k: dict[str, Any] = {}
        for ref_lbl, rmap in (("br_1hz_30s", ref_1hz_map), ("br_trig_bpm", ref_trig_map)):
            ents, sds, errs = [], [], []
            sp_ent_sess, sp_sd_sess = [], []
            for m in p30_list:
                rf, vv = rmap[m]
                if not np.any(vv):
                    continue
                ae = np.abs(p30_preds_all[km][m][vv] - rf[vv])
                en = p30_unc_all[km][m]["post_entropy"][vv]
                sd = p30_unc_all[km][m]["post_sd_bpm"][vv]
                ents.append(en)
                sds.append(sd)
                errs.append(ae)
                se = E1._safe_spearman(en, ae)
                ss = E1._safe_spearman(sd, ae)
                if np.isfinite(se):
                    sp_ent_sess.append(se)
                if np.isfinite(ss):
                    sp_sd_sess.append(ss)
            res_k[ref_lbl] = {
                "spearman_entropy_pool": E1._safe_spearman(np.concatenate(ents), np.concatenate(errs)),
                "spearman_entropy_sess_mean": float(np.mean(sp_ent_sess)),
                "spearman_sd_pool": E1._safe_spearman(np.concatenate(sds), np.concatenate(errs)),
                "spearman_sd_sess_mean": float(np.mean(sp_sd_sess)),
            }
        ep = np.concatenate([p30_unc_all[km][m]["post_entropy"][pause_map[m] & ref_1hz_map[m][1]] for m in p30_list if np.any(pause_map[m] & ref_1hz_map[m][1])])
        er = np.concatenate([p30_unc_all[km][m]["post_entropy"][(~pause_map[m]) & ref_1hz_map[m][1]] for m in p30_list if np.any((~pause_map[m]) & ref_1hz_map[m][1])])
        sp = np.concatenate([p30_unc_all[km][m]["post_sd_bpm"][pause_map[m] & ref_1hz_map[m][1]] for m in p30_list if np.any(pause_map[m] & ref_1hz_map[m][1])])
        sr = np.concatenate([p30_unc_all[km][m]["post_sd_bpm"][(~pause_map[m]) & ref_1hz_map[m][1]] for m in p30_list if np.any((~pause_map[m]) & ref_1hz_map[m][1])])
        res_k["quiet_apnea_safety"] = {
            "n_pause": int(ep.size),
            "n_regular": int(er.size),
            "mean_entropy_pause": float(np.mean(ep)),
            "mean_entropy_regular": float(np.mean(er)),
            "mean_sd_pause_bpm": float(np.mean(sp)),
            "mean_sd_regular_bpm": float(np.mean(sr)),
        }
        wp4_r2_preview[km] = res_k

    # Verification controls (O(2)xR+ invariance, causality, freq-shift equivariance, [18,120] grid shift)
    print("[EXP-2] Running invariance, causality & equivariance checks...")
    controls = _run_stage2_invariance_causality_shift(
        p30_r2e_thetas, p30_r2a_thetas, p30_r2b_thetas, lags_map, p30_r3_shift_folds, radar_shift_summary
    )

    # Save reusable pool30_sync prediction, uncertainty & synthetic sweep archive
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_dict: dict[str, Any] = {
        "sessions": np.asarray(p30_list, dtype=str),
        "models": np.asarray(MODEL_ORDER, dtype=str),
    }
    for k in MODEL_ORDER:
        for m in p30_list:
            archive_dict[f"pred__{k}__{m}"] = np.asarray(p30_preds_all[k][m], dtype=np.float64)
    for km in (
        "R2e_Causal_DiffHMM_FilterSafe8",
        "R2a_Causal_DiffHMM_Linear11",
        "R2b_Causal_DiffHMM_MLP43",
        "R2d_E2E_Causal_H5_DiffHMM_Linear11",
    ):
        for m in p30_list:
            archive_dict[f"unc_ent__{km}__{m}"] = np.asarray(p30_unc_all[km][m]["post_entropy"], dtype=np.float64)
            archive_dict[f"unc_sd__{km}__{m}"] = np.asarray(p30_unc_all[km][m]["post_sd_bpm"], dtype=np.float64)
    for cname, sdata in synth_suite.items():
        archive_dict[f"synth_truth__{cname}"] = np.asarray(sdata["truth_bpm"], dtype=np.float64)
        for km in synth_models:
            stack = np.asarray([f_dict[cname][km] for f_dict in p30_synth_folds], dtype=np.float64)
            archive_dict[f"synth_mean__{cname}__{km}"] = np.mean(stack, axis=0)
            archive_dict[f"synth_sd__{cname}__{km}"] = np.std(stack, axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros(stack.shape[1], dtype=np.float64)
    tmp_arch = OUTPUT_DIR / ".pool30_predictions.tmp.npz"
    np.savez_compressed(tmp_arch, **archive_dict)
    tmp_arch.replace(PRED_ARCHIVE)

    # Write per-session CSV on pool30_sync (1 Hz, Trig, Lag-Aligned Trig)
    IO.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    ps_1hz = cohort_results["pool30_sync"]["br_1hz_30s"].pop("_per_sess")
    ps_trig = cohort_results["pool30_sync"]["br_trig_bpm"].pop("_per_sess")
    ps_tla = cohort_results["pool30_sync"]["br_trig_bpm_lag_aligned"].pop("_per_sess")
    cohort_results["pool30_sync"]["br_trig_count_bpm"].pop("_per_sess")
    with AUDIT_SESSIONS_CSV.open("w", newline="", encoding="utf-8") as fh:
        cols = (
            ["measurement", "patient_id"]
            + [f"{k}_rmse_1hz" for k in MODEL_ORDER]
            + [f"{k}_rmse_trig" for k in MODEL_ORDER]
            + [f"{k}_rmse_trig_lag_aligned" for k in MODEL_ORDER]
        )
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for m in p30_list:
            row: dict[str, Any] = {"measurement": m, "patient_id": IO.patient_id_of(m)}
            for k in MODEL_ORDER:
                row[f"{k}_rmse_1hz"] = f"{ps_1hz[k][m]['rmse']:.4f}"
                row[f"{k}_rmse_trig"] = f"{ps_trig[k][m]['rmse']:.4f}" if m in ps_trig[k] else ""
                row[f"{k}_rmse_trig_lag_aligned"] = f"{ps_tla[k][m]['rmse']:.4f}" if m in ps_tla[k] else ""
            writer.writerow(row)

    total_s = float(time.perf_counter() - t_start)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "git_head": IO.git_head_info(),
        "analysis_plan": "thesis/analysis_plan_v3_ai.md",
        "regime": REGIME,
        "stage1_parity_max_diff": stage1_parity_max_diff,
        "total_runtime_s": total_s,
        "cohorts": cohort_results,
        "shortcut_learning_audit_pool30": shortcut_summary,
        "lag_profiles_pool30": lag_profiles_pool30,
        "filter_edge_channel_profile_pool30": filter_edge_profile,
        "r2_convergence_summary": r2_convergence_summary,
        "interpretable_params_pool30": interp_summary,
        "pool30_fold_logs": p30_fold_logs,
        "synthetic_frequency_sweep_pool30": synth_summary,
        "phase_scrambled_surrogates_pool30": surr_summary,
        "wp4_causal_uncertainty_preview": wp4_r2_preview,
        "verification_controls": controls,
    }
    json.dumps(payload, allow_nan=False)
    IO.write_json_locked(BUNDLE_JSON, payload)

    # Generate concise Markdown audit report
    lines = [
        f"# EXP-2 Stage 2 Complexity Ladder & Shortcut-Learning Audit (`{SCHEMA_VERSION}`)\n",
        f"- **Git HEAD:** `{payload['git_head']['head_commit']}` (`{payload['git_head']['branch']}`, Python `{payload['git_head']['python_version']}`)",
        f"- **Total Wall Time:** `{total_s:.1f} s` | **Stage 1.1 Anchor Parity (`R0/R1a/R1f/R1g`):** `max diff = {stage1_parity_max_diff:.2e}`",
        f"- **R2 O(2)xR+ Invariance on H5 (< 1e-6 bpm):** `{'PASS' if controls['all_r2_invariances_lt_1e6'] else 'FAIL'}` (clean max `{max(controls['invariance_r2_max_diff_bpm']['clean_7_3'].values()):.2e}`, clipped max `{max(controls['invariance_r2_max_diff_bpm']['clipped_2_3'].values()):.2e}`; frozen H0 swap diag `{controls['frozen_h0_swap_asymmetry_diagnostic_bpm']['clean_7_3']['iq_channel_swap']:.2e}`)",
        f"- **Strict 0.0 Causality Diffs:** `R2e={controls['causality_r2e_max_diff']:.2e}`, `R2a={controls['causality_r2a_max_diff']:.2e}`, `R2b={controls['causality_r2b_max_diff']:.2e}`, `R3a_TCN={controls['causality_r3a_tcn_max_diff']:.2e}`",
        f"- **Exact Operator +10 bpm Equivariance (`[-60, 180]`):** `R2e={controls['freq_shift_equivariance_operator_max_diff_bpm']['R2e']:.2e}`, `R2a={controls['freq_shift_equivariance_operator_max_diff_bpm']['R2a']:.2e}`, `R2b={controls['freq_shift_equivariance_operator_max_diff_bpm']['R2b']:.2e} bpm`",
        f"- **Calibrated `[18, 90]` +10 bpm Shift Err (`36->46`):** `R2e={controls['freq_shift_equivariance_clinical_grid_bpm']['R2e_Causal_DiffHMM_FilterSafe8']['max_abs_err_bpm']:.3e}`, `R2a={controls['freq_shift_equivariance_clinical_grid_bpm']['R2a_Causal_DiffHMM_Linear11']['max_abs_err_bpm']:.3e}` vs `R3c={controls['freq_shift_equivariance_clinical_grid_bpm']['R3c_Ridge_Spectral_439']['mean_max_abs_err_bpm']:.2f}`, `R3b={controls['freq_shift_equivariance_clinical_grid_bpm']['R3b_Spectral_MLP_15k']['mean_max_abs_err_bpm']:.2f}`, `R3a={controls['freq_shift_equivariance_clinical_grid_bpm']['R3a_Causal_TCN_16k']['mean_max_abs_err_bpm']:.2f} bpm`",
        f"- **24 GHz Waveform `+10 bpm` Shift (`36->46`):** `R2e={radar_shift_summary['R2e_Causal_DiffHMM_FilterSafe8']['mean_shift_bpm']:.3f}`, `R2a={radar_shift_summary['R2a_Causal_DiffHMM_Linear11']['mean_shift_bpm']:.3f}` vs `R3c={radar_shift_summary['R3c_Ridge_Spectral_439']['mean_shift_bpm']:.3f}`, `R3b={radar_shift_summary['R3b_Spectral_MLP_15k']['mean_shift_bpm']:.3f}`, `R3a={radar_shift_summary['R3a_Causal_TCN_16k']['mean_shift_bpm']:.3f} bpm`",
        f"- **Shifted-Grid `[18, 90]` vs `[18, 120]` Bias Shift:** `R2e={controls['shifted_grid_18_90_vs_18_120']['R2e_Causal_DiffHMM_FilterSafe8']['abs_bias_shift_bpm']:.4f}`, `R2a={controls['shifted_grid_18_90_vs_18_120']['R2a_Causal_DiffHMM_Linear11']['abs_bias_shift_bpm']:.4f}`, `R2b={controls['shifted_grid_18_90_vs_18_120']['R2b_Causal_DiffHMM_MLP43']['abs_bias_shift_bpm']:.4f} bpm`",
        f"- **R2 L-BFGS-B Convergence (`maxiter=200`):** `R2e={r2_convergence_summary['r2e']['n_converged']}/11 (med {r2_convergence_summary['r2e']['median_nit']:.0f})`, `R2a={r2_convergence_summary['r2a']['n_converged']}/11`, `R2b={r2_convergence_summary['r2b']['n_converged']}/11 (med {r2_convergence_summary['r2b']['median_nit']:.0f})`, `R2f={r2_convergence_summary['r2f']['n_converged']}/11`, `R2c={r2_convergence_summary['r2c']['n_converged']}/11`\n",
        "## 1. 4-Rung Complexity Ladder (`pool30_sync`, trained ONLY on `1 Hz RespRate`, 4-way blind test)\n",
        "| Rung | Method | Params | Lookahead (Block / STFT) | $\\mathrm{RMSE}_{\\mathrm{sess}}$ (1 Hz) | Rank (1 Hz) | $r_{\\mathrm{within}}$ (1 Hz) | $\\mathrm{SD}/\\mathrm{SD}_{1\\mathrm{Hz}}$ | $\\mathrm{SD}/\\mathrm{SD}_{\\mathrm{T10}}$ | $\\mathrm{RMSE}_{\\mathrm{sess}}$ (Trig 0s) | $\\mathrm{RMSE}_{\\mathrm{sess}}$ (Trig -10s) | Rank (-10s) | $r_{\\mathrm{within}}$ (-10s) |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k in MODEL_ORDER:
        s = shortcut_summary[k]
        la_lbl = f"{s['block_lookahead_label']} / {s['stft_lookahead_s']:.0f} s"
        lines.append(
            f"| R{s['rung']} | `{k}` | {s['n_params']:,} | {la_lbl} | {s['rmse_1hz']:.3f} | #{s['rank_1hz']} | {s['r_within_1hz']:.3f} | "
            f"{s['pred_sd_ratio_1hz']:.2f} | {s['pred_sd_ratio_trig_lag_aligned']:.2f} | {s['rmse_trig']:.3f} | {s['rmse_trig_lag_aligned']:.3f} | #{s['rank_trig_lag_aligned']} | {s['r_within_trig_lag_aligned']:.3f} |"
        )

    lines += [
        "\n## 2. Interpretable Rung 2 Learned Parameters (`8` & `11` params mean ± SD across 11 LOPO folds)\n",
        "| Parameter | `R2e_Causal_DiffHMM_FilterSafe8` | `R2a_Causal_DiffHMM_Linear11` | `R2d_E2E_Causal_H5_DiffHMM_Linear11` |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for kp in interp_summary["R2a_Causal_DiffHMM_Linear11"]:
        ve = interp_summary["R2e_Causal_DiffHMM_FilterSafe8"][kp]
        va = interp_summary["R2a_Causal_DiffHMM_Linear11"][kp]
        vd = interp_summary["R2d_E2E_Causal_H5_DiffHMM_Linear11"][kp]
        lines.append(f"| `{kp}` | `{ve['mean']:+.3f} ± {ve['sd']:.3f}` | `{va['mean']:+.3f} ± {va['sd']:.3f}` | `{vd['mean']:+.3f} ± {vd['sd']:.3f}` |")

    lines += [
        "\n## 3. Out-of-Range 24 GHz Passband Synthetic Frequency Sweep (`28–80 bpm`, zero synthetic tuning)\n",
        "| Method | `28 bpm` MAE | `35 bpm` MAE | `50 bpm` MAE | `65 bpm` MAE | `80 bpm` MAE | Chirp `30->75` RMSE |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for km in synth_models:
        lines.append(
            f"| `{km}` | {synth_summary['stat_28bpm'][km]['mae_mean']:.2f} | "
            f"{synth_summary['stat_35bpm'][km]['mae_mean']:.2f} | "
            f"{synth_summary['stat_50bpm'][km]['mae_mean']:.2f} | "
            f"{synth_summary['stat_65bpm'][km]['mae_mean']:.2f} | "
            f"{synth_summary['stat_80bpm'][km]['mae_mean']:.2f} | "
            f"{synth_summary['chirp_30_to_75bpm'][km]['rmse_mean']:.2f} |"
        )
    s20 = synth_summary["stat_20bpm_kaiser_edge"]
    lines.append(
        f"- *Kaiser Low-Edge Roll-off Diagnostic (`20 bpm = 0.333 Hz`, 64-tap `BAND_LOW_HZ=0.30 Hz` skirt attenuates `1f` below `2f=40 bpm`):* "
        f"`R0={s20['R0_HPSS_NLS_Locked']['mae_mean']:.2f}`, `R2e={s20['R2e_Causal_DiffHMM_FilterSafe8']['mae_mean']:.2f}`, `R2a={s20['R2a_Causal_DiffHMM_Linear11']['mae_mean']:.2f}`, `R3a={s20['R3a_Causal_TCN_16k']['mae_mean']:.2f} bpm` MAE."
    )

    if not surr_summary["skipped"]:
        lines += [
            "\n## 4. 5-Seed Phase-Scrambled Radar Anti-Inertia Surrogates (`pool30_sync`)\n",
            "| Method | Real RMSE (1 Hz) | Scrambled RMSE (1 Hz) | Real $r_{\\mathrm{within}}$ | Scrambled $r_{\\mathrm{within}}$ |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for km in surr_keys:
            sr = surr_summary["br_1hz_30s"][km]
            lines.append(
                f"| `{km}` | {m1_p30[km]['rmse_sess']:.3f} | {sr['rmse_sess_mean']:.3f}±{sr['rmse_sess_sd']:.3f} | "
                f"{m1_p30[km]['r_within_rmcorr']:.3f} | {sr['r_within_rmcorr_mean']:.3f} |"
            )

    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[EXP-2] Wrote {BUNDLE_JSON}, {AUDIT_MD}, {AUDIT_SESSIONS_CSV}, {PRED_ARCHIVE} in {total_s:.1f}s.")


if __name__ == "__main__":
    main()
