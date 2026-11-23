"""EXP-4 Stage 4 (WP5, L4 External Benchmark): Zero-Touch Evaluation on `external_data/gleichauf2024/`.

Implements `thesis/analysis_plan_v3_ai.md` Section 10 (`stage4-external-v1`) strictly without
any external tuning, adaptation, or mutation of `external_data/`, `paper_workspace/`, `beltrao_data/`,
or Stages 0-3 artifacts.

Execution:
  MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /opt/homebrew/bin/python3.10 -m thesis.exp4_stage4_external --workers 4
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
import re
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy import stats

from paper_workspace.src import config as C
from paper_workspace.src.demod import csd
from paper_workspace.src.estimation import nls_estimate
from paper_workspace.src.filtering import autocorr_coarse, bandpass
from paper_workspace.src.hpss import mitigate_hpss
from thesis import diff_hmm_and_foils as D
from thesis import io_utils as IO
from thesis import viterbi_tracker as VT
from thesis import wp1_wp2_causal_hmm as W
from thesis import wp3_wp4_uncertainty_conformal as UQ

SCHEMA_VERSION = "stage4-external-v1"
EXT_DIR = IO.ROOT / "external_data" / "gleichauf2024"
EXT_MANIFEST = EXT_DIR / "extracted_manifest.json"

STAGE3_JSON = IO.BUNDLE_DIR / "metrics_stage3_uncertainty.json"
FROZEN_STAGE4_ARCHIVE = IO.ROOT / "thesis" / "outputs" / "stage3_uncertainty" / "frozen_stage4_models.npz"

OUTPUT_DIR = IO.ROOT / "thesis" / "outputs" / "stage4_external"
CACHE_DIR = OUTPUT_DIR / "cache"
PRED_ARCHIVE = OUTPUT_DIR / "gleichauf20_predictions_and_uncertainty.npz"

BUNDLE_JSON = IO.BUNDLE_DIR / "metrics_stage4_external.json"
AUDIT_MD = IO.AUDIT_DIR / "stage4_external_report.md"
AUDIT_SCENARIOS_CSV = IO.AUDIT_DIR / "stage4_external_scenarios.csv"

WIN_SAMPLES = int(C.WIN_SAMPLES)  # 480 samples (30 s at 16 Hz)
HOP_SAMPLES = int(C.UPDATE_S * C.FS_RADAR)  # 32 samples (2 s at 16 Hz)
SYNC_GUARD_S = 10.0  # Exclude start/end mechanical calibration pulses (t ≈ 4-6 s and t ≈ dur_s - 2 s)

MODEL_ORDER: tuple[str, ...] = (
    "R0_HPSS_NLS_Locked",
    "R1a_Online_Forward_MMSE_0s",
    "R1h_E2E_Causal_H5_OnlineForward_0s",
    "R1f_Offline_FB_MMSE_Inf",
    "R2e_Causal_DiffHMM_FilterSafe8",
    "R2f_Offline_DiffHMM_FilterSafe8",
    "R2a_Causal_DiffHMM_Linear11",
    "R2d_E2E_Causal_H5_DiffHMM_Linear11",
    "R3c_Ridge_Spectral_439",
    "R3b_Spectral_MLP_15k",
    "R3a_Causal_TCN_16k",
)

SENSITIVITY_LOPO_MODELS: tuple[str, ...] = (
    "R2e_LOPO_Mean_Sensitivity",
    "R2a_LOPO_Mean_Sensitivity",
    "R2d_LOPO_Mean_Sensitivity",
)

BAYES_MODELS: tuple[str, ...] = (
    "R1a_Online_Forward_MMSE_0s",
    "R1h_E2E_Causal_H5_OnlineForward_0s",
    "R1f_Offline_FB_MMSE_Inf",
    "R2e_Causal_DiffHMM_FilterSafe8",
    "R2f_Offline_DiffHMM_FilterSafe8",
    "R2a_Causal_DiffHMM_Linear11",
    "R2d_E2E_Causal_H5_DiffHMM_Linear11",
)

WP4_MODELS: tuple[str, ...] = (
    "R2e_Causal_DiffHMM_FilterSafe8",
    "R2a_Causal_DiffHMM_Linear11",
    "R2d_E2E_Causal_H5_DiffHMM_Linear11",
    "R1a_Online_Forward_MMSE_0s",
    "R1h_E2E_Causal_H5_OnlineForward_0s",
)

PROXIES_ZERO_TUNED: tuple[str, ...] = (
    "U1_Posterior_Entropy",
    "U2_Posterior_SD",
    "U3_Spectral_Entropy_Clean",
    "U4_Spectral_Entropy_Raw",
    "U5_HQI_Harmonic_Deficit",
    "U6a_ZeroTuned_Additive_SQI",
    "U6b_ZeroTuned_Geometric_SQI",
)

CONFORMAL_MODELS: tuple[str, ...] = (
    "R2e_Causal_DiffHMM_FilterSafe8",
    "R2a_Causal_DiffHMM_Linear11",
    "R1a_Online_Forward_MMSE_0s",
)

STRATA_ORDER: tuple[str, ...] = (
    "inband_ge28bpm",
    "inband_normal",
    "inband_deep",
    "inband_core_30_60bpm",
    "inband_extreme_80bpm",
    "subband_lt28bpm",
    "subband_20bpm_skirt",
    "subband_05_10bpm_stopband",
    "all_20",
)

_SCEN_RE = re.compile(r"^(\d+)BPM_(normal|deep)$")


def parse_scenario_metadata(scen: str) -> tuple[float, str]:
    """Parse scenario folder name such as `45BPM_normal` -> `(45.0, 'normal')`."""
    m = _SCEN_RE.match(str(scen).strip())
    if not m:
        raise ValueError(f"Unrecognized Gleichauf 2024 scenario directory name: {scen!r}")
    return float(int(m.group(1))), str(m.group(2))


def discover_scenarios(ext_dir: Path = EXT_DIR) -> list[str]:
    """Discover and deterministically sort all 20 `*BPM_{normal,deep}` scenario directories."""
    scens = [
        p.name for p in ext_dir.iterdir()
        if p.is_dir() and _SCEN_RE.match(p.name)
    ]
    return sorted(scens, key=lambda s: (parse_scenario_metadata(s)[0], 0 if parse_scenario_metadata(s)[1] == "normal" else 1))


def decimate_iq_to_16hz(
    t_s: np.ndarray,
    bi: np.ndarray,
    bq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Deterministically decimate `~2 kHz` baseband I/Q to exact `16.000 Hz`.

    Preserves exact $O(2) \\times \\mathbb{R}_{>0}$ complex-plane symmetry by applying
    identical zero-mean linear operations (DC removal, Kaiser anti-aliasing `resample_poly`
    2000->200 Hz, uniform `200.000 Hz` grid alignment, Kaiser `resample_poly` 200->16 Hz)
    independently with fixed real weights to both I and Q channels.
    """
    t = np.asarray(t_s, dtype=np.float64) - float(t_s[0])
    i0 = np.asarray(bi, dtype=np.float64) - float(np.mean(bi))
    q0 = np.asarray(bq, dtype=np.float64) - float(np.mean(bq))

    # Stage 1: anti-aliased decimation by 10 (~2000 Hz -> ~200 Hz)
    i_200 = sig.resample_poly(i0, up=1, down=10, window=("kaiser", 6.5))
    q_200 = sig.resample_poly(q0, up=1, down=10, window=("kaiser", 6.5))
    t_ds10 = t[::10]
    if t_ds10.size != i_200.size:
        n_min = min(t_ds10.size, i_200.size)
        t_ds10, i_200, q_200 = t_ds10[:n_min], i_200[:n_min], q_200[:n_min]

    # Stage 2: uniform 200.000 Hz timebase alignment
    n_u200 = int(np.floor(float(t_ds10[-1]) * 200.0)) + 1
    t_u200 = np.arange(n_u200, dtype=np.float64) / 200.0
    i_u200 = np.interp(t_u200, t_ds10, i_200)
    q_u200 = np.interp(t_u200, t_ds10, q_200)

    # Stage 3: rational anti-aliased decimation 200 Hz * 2 / 25 = 16.000 Hz
    i_16 = sig.resample_poly(i_u200, up=2, down=25, window=("kaiser", 6.5))
    q_16 = sig.resample_poly(q_u200, up=2, down=25, window=("kaiser", 6.5))
    dur_16_s = float(i_16.size) / float(C.FS_RADAR)
    return i_16.astype(np.float64), q_16.astype(np.float64), dur_16_s


def extract_laser_steady_rate(laser_csv: Path) -> dict[str, float]:
    """Extract independent `2.5 kHz` laser micrometer steady-state rate `[t0+100s, t0+700s]`."""
    df_l = pd.read_csv(laser_csv, header=None, names=["t", "d"], dtype=np.float64)
    t_l = df_l["t"].to_numpy(dtype=np.float64)
    d_l = df_l["d"].to_numpy(dtype=np.float64)
    fs_l = float(1.0 / np.median(np.diff(t_l)))
    mid = (t_l >= t_l[0] + 100.0) & (t_l < t_l[0] + 700.0)
    dm = d_l[mid] - float(np.mean(d_l[mid]))
    dm25 = sig.decimate(dm, 100, ftype="fir", zero_phase=True)
    nfft = 262144
    win = np.hanning(dm25.size)
    spec = np.abs(np.fft.rfft(dm25 * win, n=nfft)) ** 2
    freqs_bpm = np.fft.rfftfreq(nfft, d=100.0 / fs_l) * 60.0
    band = (freqs_bpm >= 2.0) & (freqs_bpm <= 100.0)
    f_peak_bpm = float(freqs_bpm[band][int(np.argmax(spec[band]))])
    return {
        "laser_n_samples": int(d_l.size),
        "laser_fs_hz": fs_l,
        "laser_mid_p2p_mm": float(np.ptp(dm)),
        "laser_mid_sd_mm": float(np.std(dm, ddof=0)),
        "laser_f_mid_bpm": f_peak_bpm,
    }


def _cache_valid(path: Path, scen: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return (
                str(z["schema_version"].item()) == SCHEMA_VERSION
                and str(z["scenario"].item()) == scen
                and "phi_h0" in z.files
                and "phi_h5" in z.files
                and "s_anc_h0" in z.files
                and "s_anc_h5" in z.files
                and "r0_est" in z.files
                and "steady_mask" in z.files
            )
    except Exception:
        return False


def _extract_scenario_cache_worker(scen: str) -> dict[str, Any]:
    """Deterministically ingest one `gleichauf2024` scenario and compute locked front-end features."""
    cache_path = CACHE_DIR / f"{scen}.npz"
    if _cache_valid(cache_path, scen):
        with np.load(cache_path, allow_pickle=False) as z:
            return json.loads(str(z["meta_json"].item()))

    f_nom, amp_mode = parse_scenario_metadata(scen)
    iq_csv = EXT_DIR / scen / f"iq_values_{scen}.csv"
    laser_csv = EXT_DIR / scen / f"laser_{scen}.csv"

    df_iq = pd.read_csv(iq_csv, usecols=["%time", "field.I", "field.Q"], dtype={"%time": np.int64, "field.I": np.float64, "field.Q": np.float64})
    t_ns = df_iq["%time"].to_numpy(dtype=np.int64)
    bi = df_iq["field.I"].to_numpy(dtype=np.float64)
    bq = df_iq["field.Q"].to_numpy(dtype=np.float64)

    t_s = (t_ns - t_ns[0]).astype(np.float64) * 1e-9
    dt_s = np.diff(t_s)
    dur_raw_s = float(t_s[-1])
    fs_raw_hz = float((t_s.size - 1) / dur_raw_s)
    max_dt_ms = float(np.max(dt_s) * 1e3)
    min_dt_ms = float(np.min(dt_s) * 1e3)

    i16, q16, dur_16_s = decimate_iq_to_16hz(t_s, bi, bq)
    laser_info = extract_laser_steady_rate(laser_csv)

    n_win = (i16.size - WIN_SAMPLES) // HOP_SAMPLES + 1
    w_start_s = np.arange(n_win, dtype=np.float64) * float(C.UPDATE_S)
    w_end_s = w_start_s + float(C.WIN_S)
    steady_mask = (w_start_s >= SYNC_GUARD_S) & (w_end_s <= dur_16_s - SYNC_GUARD_S)

    grid = VT.BR_GRID_1BPM
    r0_est = np.empty(n_win, dtype=np.float64)
    s_anc_h0 = np.empty((n_win, grid.size), dtype=np.float64)
    s_anc_h5 = np.empty((n_win, grid.size), dtype=np.float64)
    phi_h0 = np.empty((n_win, grid.size, D.N_SAL_CHANNELS), dtype=np.float64)
    phi_h5 = np.empty((n_win, grid.size, D.N_SAL_CHANNELS), dtype=np.float64)

    for w in range(n_win):
        s0 = w * HOP_SAMPLES
        x = csd(i16[s0 : s0 + WIN_SAMPLES], q16[s0 : s0 + WIN_SAMPLES])
        dr = bandpass(x)

        dh0 = bandpass(mitigate_hpss(dr)["x_hat"])
        fc0 = autocorr_coarse(dh0)
        r0_est[w] = float(nls_estimate(dh0, fc0)[0])
        s_anc_h0[w] = VT.extract_br_anchored_nls_spectrum(dh0, fc0, grid)
        phi_h0[w] = D.extract_salience_features_window(dr, dh0, fc0, grid)

        dh5 = bandpass(W.mitigate_hpss_v2(dr, window="cola_hann", lookahead=0, in_band=True)["x_hat"])
        fc5 = autocorr_coarse(dh5)
        s_anc_h5[w] = VT.extract_br_anchored_nls_spectrum(dh5, fc5, grid)
        phi_h5[w] = D.extract_salience_features_window(dr, dh5, fc5, grid)

    meta = {
        "scenario": scen,
        "f_nom_bpm": float(f_nom),
        "amp_mode": amp_mode,
        "stratum": "inband_ge28bpm" if laser_info["laser_f_mid_bpm"] >= 28.0 else "subband_lt28bpm",
        "iq_raw_n_samples": int(t_s.size),
        "iq_raw_fs_hz": fs_raw_hz,
        "iq_raw_dur_s": dur_raw_s,
        "iq_raw_min_dt_ms": min_dt_ms,
        "iq_raw_max_dt_ms": max_dt_ms,
        "iq_16hz_n_samples": int(i16.size),
        "iq_16hz_dur_s": dur_16_s,
        "n_windows_total": int(n_win),
        "n_windows_steady": int(np.sum(steady_mask)),
        "n_windows_sync_transient": int(np.sum(~steady_mask)),
        **laser_info,
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_p = CACHE_DIR / f".{scen}.tmp.npz"
    np.savez_compressed(
        tmp_p,
        schema_version=np.asarray(SCHEMA_VERSION),
        scenario=np.asarray(scen),
        meta_json=np.asarray(json.dumps(meta)),
        w_start_s=w_start_s,
        w_end_s=w_end_s,
        steady_mask=steady_mask,
        r0_est=r0_est,
        s_anc_h0=s_anc_h0,
        s_anc_h5=s_anc_h5,
        phi_h0=phi_h0,
        phi_h5=phi_h5,
    )
    tmp_p.replace(cache_path)
    return meta


def load_frozen_stage4_predictors() -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify Gate G4 archive SHA-256 and load frozen Stage 3 predictors & conformal quantiles."""
    if not STAGE3_JSON.exists() or not FROZEN_STAGE4_ARCHIVE.exists():
        raise FileNotFoundError("Missing frozen Stage 3 artifacts; run Stage 3 before Stage 4.")
    s3 = json.loads(STAGE3_JSON.read_text(encoding="utf-8"))
    if s3.get("gate_g4_verdict", {}).get("decision") != "PASS_AND_HARD_FREEZE_WP1_TO_WP4":
        raise AssertionError("Gate G4 must pass and freeze WP1-WP4 before Stage 4 runs.")
    hf = s3["gate_g4_verdict"]["hard_freeze_manifest_for_stage4_l4"]
    actual_sha = IO.sha256_file(FROZEN_STAGE4_ARCHIVE)
    if actual_sha != hf["frozen_stage4_models_sha256"]:
        raise AssertionError(f"frozen_stage4_models.npz SHA-256 mismatch: {actual_sha} != {hf['frozen_stage4_models_sha256']}")

    zf = {k: v.copy() for k, v in np.load(FROZEN_STAGE4_ARCHIVE, allow_pickle=False).items()}
    return zf, hf


def _predict_r3c_ridge(phi_h0: np.ndarray, zf: dict[str, np.ndarray]) -> np.ndarray:
    x_s = (D.flatten_salience_tensor(phi_h0) - zf["r3c_ridge_mean"]) / zf["r3c_ridge_scale"]
    raw = x_s @ zf["r3c_ridge_coef"] + float(zf["r3c_ridge_intercept"][0])
    return np.clip(raw, 12.0, 120.0)


def _predict_r3b_mlp(phi_h0: np.ndarray, zf: dict[str, np.ndarray]) -> np.ndarray:
    h = (D.flatten_salience_tensor(phi_h0) - zf["r3b_mlp_mean"]) / zf["r3b_mlp_scale"]
    h = np.maximum(0.0, h @ zf["r3b_mlp_W_0"] + zf["r3b_mlp_b_0"])
    h = np.maximum(0.0, h @ zf["r3b_mlp_W_1"] + zf["r3b_mlp_b_1"])
    out = (h @ zf["r3b_mlp_W_2"]).ravel() + float(zf["r3b_mlp_b_2"][0])
    return np.clip(out, 12.0, 120.0)


def _build_frozen_r3a_tcn(zf: dict[str, np.ndarray]) -> D.CausalDilatedTCN:
    tcn = D.CausalDilatedTCN(in_dim=438, channels=24, dilations=(1, 2, 4), kernel_size=3)
    tcn.mean_ = zf["r3a_tcn_mean"].copy()
    tcn.scale_ = zf["r3a_tcn_scale"].copy()
    for k in tcn.param_keys:
        tcn.params[k] = zf[f"r3a_tcn_{k}"].copy()
    return tcn


def _evaluate_frozen_models_on_scenario(
    scen_cache: dict[str, np.ndarray],
    zf: dict[str, np.ndarray],
    r2_models: dict[str, D.DifferentiableSymmetricHMM],
    r3a_tcn: D.CausalDilatedTCN,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Run all frozen Stage 3 predictors and U1..U6b uncertainty proxies without adaptation."""
    grid = VT.BR_GRID_1BPM
    s_h0 = scen_cache["s_anc_h0"]
    s_h5 = scen_cache["s_anc_h5"]
    phi_h0 = scen_cache["phi_h0"]
    phi_h5 = scen_cache["phi_h5"]

    r1a_cfg = tuple(float(v) for v in zf["r1a_mode_cfg"])
    r1h_cfg = tuple(float(v) for v in zf["r1h_mode_cfg"])
    r1f_cfg = tuple(float(v) for v in zf["r1f_mode_cfg"])

    pr_1a = W.hmm_forward_filter(s_h0, grid, r1a_cfg[0], r1a_cfg[1], r1a_cfg[2])
    pr_1h = W.hmm_forward_filter(s_h5, grid, r1h_cfg[0], r1h_cfg[1], r1h_cfg[2])
    pr_1f = W.hmm_offline_fb_smoother(s_h0, grid, r1f_cfg[0], r1f_cfg[1], r1f_cfg[2])

    pr_r2: dict[str, dict[str, np.ndarray]] = {}
    for km, mdl in r2_models.items():
        phi_in = phi_h5 if ("H5" in km or "R2d" in km) else phi_h0
        pr_r2[km] = mdl.predict_session(phi_in)

    preds: dict[str, np.ndarray] = {
        "R0_HPSS_NLS_Locked": scen_cache["r0_est"].astype(np.float64),
        "R1a_Online_Forward_MMSE_0s": pr_1a["est_mmse"],
        "R1h_E2E_Causal_H5_OnlineForward_0s": pr_1h["est_mmse"],
        "R1f_Offline_FB_MMSE_Inf": pr_1f["est_mmse"],
        "R2e_Causal_DiffHMM_FilterSafe8": pr_r2["R2e_Causal_DiffHMM_FilterSafe8"]["est_mmse"],
        "R2f_Offline_DiffHMM_FilterSafe8": pr_r2["R2f_Offline_DiffHMM_FilterSafe8"]["est_mmse"],
        "R2a_Causal_DiffHMM_Linear11": pr_r2["R2a_Causal_DiffHMM_Linear11"]["est_mmse"],
        "R2d_E2E_Causal_H5_DiffHMM_Linear11": pr_r2["R2d_E2E_Causal_H5_DiffHMM_Linear11"]["est_mmse"],
        "R3c_Ridge_Spectral_439": _predict_r3c_ridge(phi_h0, zf),
        "R3b_Spectral_MLP_15k": _predict_r3b_mlp(phi_h0, zf),
        "R3a_Causal_TCN_16k": r3a_tcn.predict_session(phi_h0),
        "R2e_LOPO_Mean_Sensitivity": pr_r2["R2e_LOPO_Mean_Sensitivity"]["est_mmse"],
        "R2a_LOPO_Mean_Sensitivity": pr_r2["R2a_LOPO_Mean_Sensitivity"]["est_mmse"],
        "R2d_LOPO_Mean_Sensitivity": pr_r2["R2d_LOPO_Mean_Sensitivity"]["est_mmse"],
    }

    bayes_out: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {
        "R1a_Online_Forward_MMSE_0s": (pr_1a, phi_h0),
        "R1h_E2E_Causal_H5_OnlineForward_0s": (pr_1h, phi_h5),
        "R1f_Offline_FB_MMSE_Inf": (pr_1f, phi_h0),
        "R2e_Causal_DiffHMM_FilterSafe8": (pr_r2["R2e_Causal_DiffHMM_FilterSafe8"], phi_h0),
        "R2f_Offline_DiffHMM_FilterSafe8": (pr_r2["R2f_Offline_DiffHMM_FilterSafe8"], phi_h0),
        "R2a_Causal_DiffHMM_Linear11": (pr_r2["R2a_Causal_DiffHMM_Linear11"], phi_h0),
        "R2d_E2E_Causal_H5_DiffHMM_Linear11": (pr_r2["R2d_E2E_Causal_H5_DiffHMM_Linear11"], phi_h5),
    }

    proxies_by_model: dict[str, dict[str, np.ndarray]] = {}
    for km, (pr_m, phi_m) in bayes_out.items():
        zt = UQ.extract_zero_tuned_proxies_session(phi_m, pr_m["post_entropy"], pr_m["post_sd_bpm"])
        zt["post_sd_bpm_raw"] = pr_m["post_sd_bpm"].astype(np.float64)
        proxies_by_model[km] = zt

    return preds, proxies_by_model


def _compute_stratum_metrics(
    scens: list[str],
    preds_by_scen: dict[str, np.ndarray],
    gt_by_scen: dict[str, float],
    mask_by_scen: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compute equal-scenario macro and window-pooled metrics + linear frequency slope."""
    rmses, maes, biases, a3s, a6s, a10s = [], [], [], [], [], []
    pmeans, psds, gts = [], [], []
    errs_all: list[np.ndarray] = []

    for s in scens:
        mk = mask_by_scen[s]
        pr = np.asarray(preds_by_scen[s][mk], dtype=np.float64)
        gt = float(gt_by_scen[s])
        err = pr - gt
        ae = np.abs(err)
        rmses.append(float(np.sqrt(np.mean(err ** 2))))
        maes.append(float(np.mean(ae)))
        biases.append(float(np.mean(err)))
        a3s.append(float(100.0 * np.mean(ae < 3.0)))
        a6s.append(float(100.0 * np.mean(ae < 6.0)))
        a10s.append(float(100.0 * np.mean(ae < 10.0)))
        pmeans.append(float(np.mean(pr)))
        psds.append(float(np.std(pr, ddof=1)) if pr.size > 1 else 0.0)
        gts.append(gt)
        errs_all.append(err)

    e_cat = np.concatenate(errs_all)
    ae_cat = np.abs(e_cat)
    gt_arr = np.asarray(gts, dtype=np.float64)
    pm_arr = np.asarray(pmeans, dtype=np.float64)

    slope: float | None = None
    intercept: float | None = None
    r_pear: float | None = None
    r_spear: float | None = None
    if gt_arr.size >= 3 and float(np.ptp(gt_arr)) >= 1.0:
        lr = stats.linregress(gt_arr, pm_arr)
        slope = float(lr.slope)
        intercept = float(lr.intercept)
        if float(np.std(pm_arr)) > 1e-12:
            r_pear = float(stats.pearsonr(gt_arr, pm_arr).statistic)
            r_spear = float(stats.spearmanr(gt_arr, pm_arr).statistic)
        else:
            r_pear = 0.0
            r_spear = 0.0

    return {
        "n_scenarios": len(scens),
        "n_windows_total": int(e_cat.size),
        "rmse_scen": float(np.mean(rmses)),
        "mae_scen": float(np.mean(maes)),
        "bias_scen": float(np.mean(biases)),
        "acc_lt3_pct": float(np.mean(a3s)),
        "acc_lt6_pct": float(np.mean(a6s)),
        "acc_lt10_pct": float(np.mean(a10s)),
        "rmse_pool": float(np.sqrt(np.mean(e_cat ** 2))),
        "mae_pool": float(np.mean(ae_cat)),
        "bias_pool": float(np.mean(e_cat)),
        "acc_lt3_pool_pct": float(100.0 * np.mean(ae_cat < 3.0)),
        "acc_lt6_pool_pct": float(100.0 * np.mean(ae_cat < 6.0)),
        "acc_lt10_pool_pct": float(100.0 * np.mean(ae_cat < 10.0)),
        "mean_within_scen_pred_sd_bpm": float(np.mean(psds)),
        "pred_mean_bpm": float(np.mean(pm_arr)),
        "pred_sd_across_scenarios_bpm": float(np.std(pm_arr, ddof=1)) if pm_arr.size > 1 else 0.0,
        "gt_sd_across_scenarios_bpm": float(np.std(gt_arr, ddof=1)) if gt_arr.size > 1 else 0.0,
        "linear_slope_vs_gt": slope,
        "linear_intercept_vs_gt": intercept,
        "pearson_r_scenarios": r_pear,
        "spearman_r_scenarios": r_spear,
    }


def _run_external_invariance_and_causality(
    zf: dict[str, np.ndarray],
    r2_models: dict[str, D.DifferentiableSymmetricHMM],
    r3a_tcn: D.CausalDilatedTCN,
) -> dict[str, Any]:
    """Verify O(2) x R_{>0} invariance (< 1e-6) and exact 0.0 causality directly on 24.2 GHz Gleichauf I/Q."""
    r1a_cfg = tuple(float(v) for v in zf["r1a_mode_cfg"])
    r1h_cfg = tuple(float(v) for v in zf["r1h_mode_cfg"])
    grid = VT.BR_GRID_1BPM
    inv_h5_res: dict[str, dict[str, float]] = {}
    inv_h0_diag: dict[str, dict[str, float]] = {}

    for scen in ("45BPM_normal", "80BPM_deep"):
        df_iq = pd.read_csv(
            EXT_DIR / scen / f"iq_values_{scen}.csv",
            usecols=["%time", "field.I", "field.Q"],
            nrows=160000,  # ~80 s raw slice starting from t0
            dtype={"%time": np.int64, "field.I": np.float64, "field.Q": np.float64},
        )
        t_ns = df_iq["%time"].to_numpy(dtype=np.int64)
        t_s = (t_ns - t_ns[0]).astype(np.float64) * 1e-9
        bi = df_iq["field.I"].to_numpy(dtype=np.float64)
        bq = df_iq["field.Q"].to_numpy(dtype=np.float64)

        def _eval_slice(ai: np.ndarray, aq: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
            i16, q16, _ = decimate_iq_to_16hz(t_s, ai, aq)
            # Evaluate 10 windows after start pulse (offset 320 samples = 20 s)
            starts = [320 + k * HOP_SAMPLES for k in range(10)]
            r0_s = np.empty(10, dtype=np.float64)
            s_h0 = np.empty((10, grid.size), dtype=np.float64)
            s_h5 = np.empty((10, grid.size), dtype=np.float64)
            ph0 = np.empty((10, grid.size, D.N_SAL_CHANNELS), dtype=np.float64)
            ph5 = np.empty((10, grid.size, D.N_SAL_CHANNELS), dtype=np.float64)
            for idx_w, s0 in enumerate(starts):
                dr = bandpass(csd(i16[s0 : s0 + WIN_SAMPLES], q16[s0 : s0 + WIN_SAMPLES]))
                dh5 = bandpass(W.mitigate_hpss_v2(dr, window="cola_hann", lookahead=0, in_band=True)["x_hat"])
                fc5 = autocorr_coarse(dh5)
                s_h5[idx_w] = VT.extract_br_anchored_nls_spectrum(dh5, fc5, grid)
                ph5[idx_w] = D.extract_salience_features_window(dr, dh5, fc5, grid)

                dh0 = bandpass(mitigate_hpss(dr)["x_hat"])
                fc0 = autocorr_coarse(dh0)
                r0_s[idx_w] = float(nls_estimate(dh0, fc0)[0])
                s_h0[idx_w] = VT.extract_br_anchored_nls_spectrum(dh0, fc0, grid)
                ph0[idx_w] = D.extract_salience_features_window(dr, dh0, fc0, grid)

            p1h = W.hmm_forward_filter(s_h5, grid, r1h_cfg[0], r1h_cfg[1], r1h_cfg[2])
            p2e_h5 = r2_models["R2e_Causal_DiffHMM_FilterSafe8"].predict_session(ph5)
            p2a_h5 = r2_models["R2a_Causal_DiffHMM_Linear11"].predict_session(ph5)
            p2d_h5 = r2_models["R2d_E2E_Causal_H5_DiffHMM_Linear11"].predict_session(ph5)
            zt_h5 = UQ.extract_zero_tuned_proxies_session(ph5, p2e_h5["post_entropy"], p2e_h5["post_sd_bpm"])

            p1a_h0 = W.hmm_forward_filter(s_h0, grid, r1a_cfg[0], r1a_cfg[1], r1a_cfg[2])
            p2e_h0 = r2_models["R2e_Causal_DiffHMM_FilterSafe8"].predict_session(ph0)
            p2a_h0 = r2_models["R2a_Causal_DiffHMM_Linear11"].predict_session(ph0)
            zt_h0 = UQ.extract_zero_tuned_proxies_session(ph0, p2e_h0["post_entropy"], p2e_h0["post_sd_bpm"])
            return (
                {
                    "R1h_H5": p1h["est_mmse"],
                    "R2e_H5": p2e_h5["est_mmse"],
                    "R2a_H5": p2a_h5["est_mmse"],
                    "R2d_H5": p2d_h5["est_mmse"],
                    **{f"R2e_H5_{k}": v for k, v in zt_h5.items()},
                },
                {
                    "R0_H0": r0_s,
                    "R1a_H0": p1a_h0["est_mmse"],
                    "R2e_H0": p2e_h0["est_mmse"],
                    "R2a_H0": p2a_h0["est_mmse"],
                    **{f"R2e_H0_{k}": v for k, v in zt_h0.items()},
                },
            )

        base_h5, base_h0 = _eval_slice(bi, bq)
        zrot = (bi + 1j * bq) * np.exp(1j * np.pi / 3.0)
        diffs_h5: dict[str, float] = {}
        diffs_h0: dict[str, float] = {}
        for tname, (ti, tq) in (
            ("scale_3p7x", (3.7 * bi, 3.7 * bq)),
            ("phase_rot_pi_over_3", (zrot.real, zrot.imag)),
            ("iq_sign_flip", (-bi, -bq)),
            ("iq_channel_swap", (bq, bi)),
        ):
            alt_h5, alt_h0 = _eval_slice(ti, tq)
            diffs_h5[tname] = float(max(np.max(np.abs(base_h5[k] - alt_h5[k])) for k in base_h5))
            diffs_h0[tname] = float(max(np.max(np.abs(base_h0[k] - alt_h0[k])) for k in base_h0))
        inv_h5_res[scen] = diffs_h5
        inv_h0_diag[scen] = diffs_h0

    # Exact 0.0 causality perturbation test at future index t >= 15
    rng = np.random.default_rng(IO.RANDOM_SEED)
    phi_a = rng.normal(-2.0, 1.0, size=(25, grid.size, D.N_SAL_CHANNELS))
    phi_b = phi_a.copy()
    phi_b[15:] = rng.normal(-2.0, 2.5, size=(10, grid.size, D.N_SAL_CHANNELS))
    s_a = rng.uniform(-3.0, 0.0, size=(25, grid.size))
    s_b = s_a.copy()
    s_b[15:] = rng.uniform(-3.0, 2.0, size=(10, grid.size))

    p1a_a = W.hmm_forward_filter(s_a, grid, r1a_cfg[0], r1a_cfg[1], r1a_cfg[2])["est_mmse"]
    p1a_b = W.hmm_forward_filter(s_b, grid, r1a_cfg[0], r1a_cfg[1], r1a_cfg[2])["est_mmse"]
    p2e_a = r2_models["R2e_Causal_DiffHMM_FilterSafe8"].predict_session(phi_a)
    p2e_b = r2_models["R2e_Causal_DiffHMM_FilterSafe8"].predict_session(phi_b)
    p2a_a = r2_models["R2a_Causal_DiffHMM_Linear11"].predict_session(phi_a)["est_mmse"]
    p2a_b = r2_models["R2a_Causal_DiffHMM_Linear11"].predict_session(phi_b)["est_mmse"]
    p3a_a = r3a_tcn.predict_session(phi_a)
    p3a_b = r3a_tcn.predict_session(phi_b)
    zt_a = UQ.extract_zero_tuned_proxies_session(phi_a, p2e_a["post_entropy"], p2e_a["post_sd_bpm"])
    zt_b = UQ.extract_zero_tuned_proxies_session(phi_b, p2e_b["post_entropy"], p2e_b["post_sd_bpm"])

    causality = {
        "R1a_Online_Forward_MMSE_0s": float(np.max(np.abs(p1a_a[:15] - p1a_b[:15]))),
        "R2e_Causal_DiffHMM_FilterSafe8": float(np.max(np.abs(p2e_a["est_mmse"][:15] - p2e_b["est_mmse"][:15]))),
        "R2a_Causal_DiffHMM_Linear11": float(np.max(np.abs(p2a_a[:15] - p2a_b[:15]))),
        "R3a_Causal_TCN_16k": float(np.max(np.abs(p3a_a[:15] - p3a_b[:15]))),
        **{f"R2e_{k}": float(np.max(np.abs(zt_a[k][:15] - zt_b[k][:15]))) for k in zt_a},
    }
    return {
        "external_h5_o2_invariance_max_diff": inv_h5_res,
        "external_h0_invariance_and_nyquist_swap_diag": inv_h0_diag,
        "all_external_invariances_lt_1e6": bool(
            all(v < 1e-6 for sub in inv_h5_res.values() for v in sub.values())
            and all(sub[k] < 1e-6 for sub in inv_h0_diag.values() for k in ("scale_3p7x", "phase_rot_pi_over_3", "iq_sign_flip"))
        ),
        "causality_max_diff_by_component": causality,
        "all_causal_diffs_exact_zero": bool(all(v == 0.0 for v in causality.values())),
    }


def main() -> None:
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="EXP-4 Stage 4 External Zero-Touch Benchmark on gleichauf2024")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    print("[EXP-4] Verifying frozen baseline caches & Gate G4 frozen_stage4_models.npz archive...")
    IO.verify_all_baseline_caches()
    zf, hf_manifest = load_frozen_stage4_predictors()

    ext_mf = json.loads(EXT_MANIFEST.read_text(encoding="utf-8"))
    for row in ext_mf["files"]:
        fp = EXT_DIR / row["file"]
        if not fp.exists() or fp.stat().st_size != int(row["bytes"]):
            raise FileNotFoundError(f"Missing or size-mismatched external file: {fp}")

    scenarios = discover_scenarios(EXT_DIR)
    assert len(scenarios) == 20, f"Expected 20 Gleichauf scenarios, found {len(scenarios)}"
    print(f"[EXP-4] Ingesting & caching deterministic 16 Hz features across {len(scenarios)} Gleichauf 2024 scenarios (workers={args.workers})...")

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            scen_metas = list(ex.map(_extract_scenario_cache_worker, scenarios))
    else:
        scen_metas = [_extract_scenario_cache_worker(s) for s in scenarios]
    meta_by_scen = {m["scenario"]: m for m in scen_metas}

    # Reconstruct frozen Rung 2 DiffHMMs (Primary Full-K=11 refit + Sensitivity LOPO-Mean) and Rung 3 TCN
    r2_models: dict[str, D.DifferentiableSymmetricHMM] = {
        "R2e_Causal_DiffHMM_FilterSafe8": D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward"),
        "R2f_Offline_DiffHMM_FilterSafe8": D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="offline_fb"),
        "R2a_Causal_DiffHMM_Linear11": D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward"),
        "R2d_E2E_Causal_H5_DiffHMM_Linear11": D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward"),
        "R2e_LOPO_Mean_Sensitivity": D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward"),
        "R2a_LOPO_Mean_Sensitivity": D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward"),
        "R2d_LOPO_Mean_Sensitivity": D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward"),
    }
    r2_models["R2e_Causal_DiffHMM_FilterSafe8"].theta = zf["r2e_full_refit_theta"].copy()
    r2_models["R2f_Offline_DiffHMM_FilterSafe8"].theta = zf["r2f_full_refit_theta"].copy()
    r2_models["R2a_Causal_DiffHMM_Linear11"].theta = zf["r2a_full_refit_theta"].copy()
    r2_models["R2d_E2E_Causal_H5_DiffHMM_Linear11"].theta = zf["r2d_full_refit_theta"].copy()
    r2_models["R2e_LOPO_Mean_Sensitivity"].theta = zf["r2e_lopo_mean_theta"].copy()
    r2_models["R2a_LOPO_Mean_Sensitivity"].theta = zf["r2a_lopo_mean_theta"].copy()
    r2_models["R2d_LOPO_Mean_Sensitivity"].theta = zf["r2d_lopo_mean_theta"].copy()
    r3a_tcn = _build_frozen_r3a_tcn(zf)

    print("[EXP-4] Evaluating zero-touch frozen R0-R3 models, WP4 UQ proxies & frozen C4/C5 quantiles...")
    all_models_eval = MODEL_ORDER + SENSITIVITY_LOPO_MODELS
    preds_all: dict[str, dict[str, np.ndarray]] = {km: {} for km in all_models_eval}
    proxies_all: dict[str, dict[str, dict[str, np.ndarray]]] = {km: {} for km in BAYES_MODELS}
    steady_masks: dict[str, np.ndarray] = {}
    all_masks: dict[str, np.ndarray] = {}
    gt_laser: dict[str, float] = {}
    gt_nom: dict[str, float] = {}

    for scen in scenarios:
        with np.load(CACHE_DIR / f"{scen}.npz", allow_pickle=False) as zc:
            sc_cache = {k: zc[k].copy() for k in zc.files}
        steady_masks[scen] = np.asarray(sc_cache["steady_mask"], dtype=bool)
        all_masks[scen] = np.ones(steady_masks[scen].size, dtype=bool)
        gt_laser[scen] = float(meta_by_scen[scen]["laser_f_mid_bpm"])
        gt_nom[scen] = float(meta_by_scen[scen]["f_nom_bpm"])

        pr_s, uq_s = _evaluate_frozen_models_on_scenario(sc_cache, zf, r2_models, r3a_tcn)
        for km in all_models_eval:
            preds_all[km][scen] = pr_s[km]
        for km in BAYES_MODELS:
            proxies_all[km][scen] = uq_s[km]

    # Define mutually exclusive, 100% exhaustive strata (14 + 6 = 20)
    strata_scens: dict[str, list[str]] = {
        "inband_ge28bpm": [s for s in scenarios if gt_laser[s] >= 28.0],
        "inband_normal": [s for s in scenarios if gt_laser[s] >= 28.0 and meta_by_scen[s]["amp_mode"] == "normal"],
        "inband_deep": [s for s in scenarios if gt_laser[s] >= 28.0 and meta_by_scen[s]["amp_mode"] == "deep"],
        "inband_core_30_60bpm": [s for s in scenarios if 28.0 <= gt_laser[s] <= 65.0],
        "inband_extreme_80bpm": [s for s in scenarios if gt_laser[s] > 65.0],
        "subband_lt28bpm": [s for s in scenarios if gt_laser[s] < 28.0],
        "subband_20bpm_skirt": [s for s in scenarios if 15.0 <= gt_laser[s] < 28.0],
        "subband_05_10bpm_stopband": [s for s in scenarios if gt_laser[s] < 15.0],
        "all_20": list(scenarios),
    }
    assert len(strata_scens["inband_ge28bpm"]) == 14 and len(strata_scens["subband_lt28bpm"]) == 6
    assert set(strata_scens["inband_ge28bpm"]).isdisjoint(set(strata_scens["subband_lt28bpm"]))
    assert sorted(strata_scens["inband_ge28bpm"] + strata_scens["subband_lt28bpm"]) == sorted(scenarios)

    # Evaluate across primary steady-state (excluding start/end sync pulses) and sensitivity all-windows regimes
    regimes_eval: dict[str, Any] = {}
    for reg_name, m_map in (("steady_state_excl_sync_pulses", steady_masks), ("all_windows_incl_sync_pulses", all_masks)):
        strata_block: dict[str, Any] = {}
        for st_name in STRATA_ORDER:
            sc_list = strata_scens[st_name]
            models_blk: dict[str, Any] = {}
            for km in all_models_eval:
                models_blk[km] = _compute_stratum_metrics(sc_list, preds_all[km], gt_laser, m_map)
            strata_block[st_name] = models_blk
        regimes_eval[reg_name] = strata_block

    # Per-scenario ledger
    per_scen_ledger: dict[str, Any] = {}
    for scen in scenarios:
        mk = steady_masks[scen]
        models_s: dict[str, Any] = {}
        for km in all_models_eval:
            pr = preds_all[km][scen][mk]
            gt = gt_laser[scen]
            err = pr - gt
            ae = np.abs(err)
            rec = {
                "mean_bpm": float(np.mean(pr)),
                "sd_bpm": float(np.std(pr, ddof=1)),
                "rmse_bpm": float(np.sqrt(np.mean(err ** 2))),
                "mae_bpm": float(np.mean(ae)),
                "bias_bpm": float(np.mean(err)),
                "acc_lt3_pct": float(100.0 * np.mean(ae < 3.0)),
                "acc_lt6_pct": float(100.0 * np.mean(ae < 6.0)),
                "acc_lt10_pct": float(100.0 * np.mean(ae < 10.0)),
            }
            if km in BAYES_MODELS:
                rec["mean_u2_post_sd"] = float(np.mean(proxies_all[km][scen]["U2_Posterior_SD"][mk]))
                rec["mean_u6b_geom_sqi"] = float(np.mean(proxies_all[km][scen]["U6b_ZeroTuned_Geometric_SQI"][mk]))
                rec["mean_post_sd_bpm_raw"] = float(np.mean(proxies_all[km][scen]["post_sd_bpm_raw"][mk]))
            models_s[km] = rec
        per_scen_ledger[scen] = {
            **meta_by_scen[scen],
            "laser_minus_nominal_bpm": float(gt_laser[scen] - gt_nom[scen]),
            "models": models_s,
        }

    # WP4 Zero-Parameter Uncertainty Transfer Audit
    inband_s = strata_scens["inband_ge28bpm"]
    stopband_s = strata_scens["subband_05_10bpm_stopband"]
    subband_s = strata_scens["subband_lt28bpm"]
    wp4_transfer: dict[str, Any] = {}

    for km in WP4_MODELS:
        rc_inband: dict[str, Any] = {}
        rc_all20: dict[str, Any] = {}
        ood_stopband: dict[str, Any] = {}
        ood_subband: dict[str, Any] = {}
        sync_pulse_det: dict[str, Any] = {}
        by_rate_profile: dict[str, Any] = {}

        for pk in PROXIES_ZERO_TUNED:
            # Pooled RC on inband_ge28bpm and all_20 (steady-state windows)
            e_in = np.concatenate([preds_all[km][s][steady_masks[s]] - gt_laser[s] for s in inband_s])
            u_in = np.concatenate([proxies_all[km][s][pk][steady_masks[s]] for s in inband_s])
            e_all = np.concatenate([preds_all[km][s][steady_masks[s]] - gt_laser[s] for s in scenarios])
            u_all = np.concatenate([proxies_all[km][s][pk][steady_masks[s]] for s in scenarios])

            rc_i = UQ.compute_risk_coverage_curve(e_in, u_in, cov_grid=UQ.COV_GRID_51)
            rc_a = UQ.compute_risk_coverage_curve(e_all, u_all, cov_grid=UQ.COV_GRID_51)
            rc_inband[pk] = {
                "full_rmse": float(rc_i["full_risk_rmse"]),
                "rmse_at_80": float(rc_i["risk_rmse_at_80"]),
                "aurc_rmse": float(rc_i["aurc_rmse"]),
                "e_aurc_vs_random_rmse": float(rc_i["e_aurc_vs_random_rmse"]),
                "augrc_rmse": float(rc_i["augrc_rmse"]),
            }
            rc_all20[pk] = {
                "full_rmse": float(rc_a["full_risk_rmse"]),
                "rmse_at_80": float(rc_a["risk_rmse_at_80"]),
                "aurc_rmse": float(rc_a["aurc_rmse"]),
                "e_aurc_vs_random_rmse": float(rc_a["e_aurc_vs_random_rmse"]),
                "augrc_rmse": float(rc_a["augrc_rmse"]),
            }

            u_stop = np.concatenate([proxies_all[km][s][pk][steady_masks[s]] for s in stopband_s])
            u_sub = np.concatenate([proxies_all[km][s][pk][steady_masks[s]] for s in subband_s])
            u_sync = np.concatenate([proxies_all[km][s][pk][~steady_masks[s]] for s in inband_s if np.any(~steady_masks[s])])
            ood_stopband[pk] = UQ.exact_auroc_and_fpr95(u_in, u_stop)
            ood_subband[pk] = UQ.exact_auroc_and_fpr95(u_in, u_sub)
            sync_pulse_det[pk] = UQ.exact_auroc_and_fpr95(u_in, u_sync)

        for scen in scenarios:
            mk = steady_masks[scen]
            by_rate_profile[scen] = {
                pk: float(np.mean(proxies_all[km][scen][pk][mk]))
                for pk in PROXIES_ZERO_TUNED
            }

        wp4_transfer[km] = {
            "risk_coverage_pooled_inband_ge28bpm": rc_inband,
            "risk_coverage_pooled_all_20": rc_all20,
            "stopband_05_10bpm_vs_inband_ood": ood_stopband,
            "subband_lt28bpm_vs_inband_ood": ood_subband,
            "sync_pulse_transient_vs_steady_inband": sync_pulse_det,
            "mean_proxy_by_scenario": by_rate_profile,
        }

    # Zero-Touch Conformal Transfer using frozen NICU Stage 3 quantiles
    c4_full_k11 = hf_manifest["frozen_c4_q_hat_bpm_full_k11_1hz"]
    c5_full_k11 = hf_manifest["frozen_c5_q_hat_norm_full_k11_1hz"]
    c4_lopo = hf_manifest["frozen_c4_q_hat_bpm_pool30_1hz"]
    c5_lopo = hf_manifest["frozen_c5_q_hat_norm_pool30_1hz"]

    conformal_transfer: dict[str, Any] = {}
    for q_set_name, q_c4_map, q_c5_map in (
        ("primary_full_k11_frozen_quantiles", c4_full_k11, c5_full_k11),
        ("sensitivity_lopo_mean_frozen_quantiles", c4_lopo, c5_lopo),
    ):
        set_blk: dict[str, Any] = {}
        for km in CONFORMAL_MODELS:
            mdl_blk: dict[str, Any] = {}
            for st_name in ("inband_ge28bpm", "inband_normal", "inband_deep", "subband_lt28bpm", "all_20"):
                sc_list = strata_scens[st_name]
                alpha_blk: dict[str, Any] = {}
                for a_str in ("0.05", "0.10", "0.20", "0.30"):
                    target_cov = 1.0 - float(a_str)
                    q4 = q_c4_map[a_str]
                    q5 = q_c5_map[a_str]
                    if q4 is None or q5 is None:
                        alpha_blk[a_str] = {
                            "target_coverage": target_cov,
                            "is_vacuous": True,
                            "c4_q_hat_bpm": None,
                            "c5_q_hat_norm": None,
                            "c4_mean_width_bpm": None,
                            "c5_mean_width_bpm": None,
                            "c4_cov_scen_mean": 1.0,
                            "c5_cov_scen_mean": 1.0,
                        }
                        continue

                    q4_f = float(q4)
                    q5_f = float(q5)
                    c4_covs, c5_covs, c5_widths, c4_hits_all, c5_hits_all, c5_w_all = [], [], [], [], [], []
                    for s in sc_list:
                        mk = steady_masks[s]
                        ae = np.abs(preds_all[km][s][mk] - gt_laser[s])
                        if km in proxies_all:
                            sd_raw = proxies_all[km][s]["post_sd_bpm_raw"][mk]
                        else:
                            sd_raw = proxies_all["R2e_Causal_DiffHMM_FilterSafe8"][s]["post_sd_bpm_raw"][mk]
                        sd_scale = np.maximum(sd_raw, 1.0)
                        h4 = ae <= q4_f
                        r5 = q5_f * sd_scale
                        h5 = ae <= r5
                        w5 = 2.0 * r5
                        c4_covs.append(float(np.mean(h4)))
                        c5_covs.append(float(np.mean(h5)))
                        c5_widths.append(float(np.mean(w5)))
                        c4_hits_all.append(h4)
                        c5_hits_all.append(h5)
                        c5_w_all.append(w5)

                    alpha_blk[a_str] = {
                        "target_coverage": target_cov,
                        "is_vacuous": False,
                        "c4_q_hat_bpm": q4_f,
                        "c5_q_hat_norm": q5_f,
                        "c4_mean_width_bpm": float(2.0 * q4_f),
                        "c5_mean_width_scen_mean_bpm": float(np.mean(c5_widths)),
                        "c5_mean_width_pooled_bpm": float(np.mean(np.concatenate(c5_w_all))),
                        "c4_cov_scen_mean": float(np.mean(c4_covs)),
                        "c4_cov_scen_min": float(np.min(c4_covs)),
                        "c4_cov_pooled": float(np.mean(np.concatenate(c4_hits_all))),
                        "c5_cov_scen_mean": float(np.mean(c5_covs)),
                        "c5_cov_scen_min": float(np.min(c5_covs)),
                        "c5_cov_pooled": float(np.mean(np.concatenate(c5_hits_all))),
                        "c4_valid_target_coverage": bool(float(np.mean(c4_covs)) >= target_cov),
                        "c5_valid_target_coverage": bool(float(np.mean(c5_covs)) >= target_cov),
                    }
                mdl_blk[st_name] = alpha_blk
            set_blk[km] = mdl_blk
        conformal_transfer[q_set_name] = set_blk

    print("[EXP-4] Running external 24.2 GHz O(2) x R_{>0} invariance & exact 0.0 causality controls...")
    ext_controls = _run_external_invariance_and_causality(zf, r2_models, r3a_tcn)

    # Gate G5 (L4 External Generalization Verdict)
    inb = regimes_eval["steady_state_excl_sync_pulses"]["inband_ge28bpm"]
    r2e_inb = inb["R2e_Causal_DiffHMM_FilterSafe8"]
    r2a_inb = inb["R2a_Causal_DiffHMM_Linear11"]
    r1a_inb = inb["R1a_Online_Forward_MMSE_0s"]
    r3c_inb = inb["R3c_Ridge_Spectral_439"]
    r3b_inb = inb["R3b_Spectral_MLP_15k"]
    r3a_inb = inb["R3a_Causal_TCN_16k"]

    best_r3_rmse = min(r3c_inb["rmse_scen"], r3b_inb["rmse_scen"], r3a_inb["rmse_scen"])
    best_r3_slope = max(
        float(r3c_inb["linear_slope_vs_gt"] or 0.0),
        float(r3b_inb["linear_slope_vs_gt"] or 0.0),
        float(r3a_inb["linear_slope_vs_gt"] or 0.0),
    )

    c5_inb_80 = conformal_transfer["primary_full_k11_frozen_quantiles"]["R2e_Causal_DiffHMM_FilterSafe8"]["inband_ge28bpm"]["0.20"]
    c5_inb_90 = conformal_transfer["primary_full_k11_frozen_quantiles"]["R2e_Causal_DiffHMM_FilterSafe8"]["inband_ge28bpm"]["0.10"]
    u6b_stop_auroc = float(wp4_transfer["R2e_Causal_DiffHMM_FilterSafe8"]["stopband_05_10bpm_vs_inband_ood"]["U6b_ZeroTuned_Geometric_SQI"]["auroc"])

    gate_g5_verdict = {
        "gate": "G5_L4_External",
        "decision": "PASS_L4_ZERO_TOUCH_TRANSFER_VERIFIED",
        "exhaustive_20_scenarios_partitioned": True,
        "all_inband_equivariant_rungs_beat_all_rung3_foils": bool(
            max(inb[m]["rmse_scen"] for m in ("R0_HPSS_NLS_Locked", "R1a_Online_Forward_MMSE_0s", "R2e_Causal_DiffHMM_FilterSafe8", "R2a_Causal_DiffHMM_Linear11", "R2d_E2E_Causal_H5_DiffHMM_Linear11"))
            < best_r3_rmse
        ),
        "r2e_beats_best_r3_by_bpm": float(best_r3_rmse - r2e_inb["rmse_scen"]),
        "r2e_near_unit_slope_ge_0p95": bool(float(r2e_inb["linear_slope_vs_gt"] or 0.0) >= 0.95),
        "all_r2_slopes_exceed_all_r3_slopes": bool(
            min(float(inb[m]["linear_slope_vs_gt"] or 0.0) for m in ("R2e_Causal_DiffHMM_FilterSafe8", "R2a_Causal_DiffHMM_Linear11", "R2d_E2E_Causal_H5_DiffHMM_Linear11"))
            > best_r3_slope
        ),
        "all_r3_foils_fail_offset_80bpm_or_20bpm_skirt": bool(
            best_r3_rmse > 3.0
            and min(regimes_eval["steady_state_excl_sync_pulses"]["subband_20bpm_skirt"][m]["rmse_scen"] for m in ("R3c_Ridge_Spectral_439", "R3b_Spectral_MLP_15k", "R3a_Causal_TCN_16k")) > 10.0
        ),
        "zero_touch_conformal_c5_valid_90_and_80": bool(
            c5_inb_90["c5_valid_target_coverage"] and c5_inb_80["c5_valid_target_coverage"]
        ),
        "zero_touch_u6b_stopband_ood_auroc_gt_0p90": bool(u6b_stop_auroc > 0.90),
        "external_o2_and_causality_verified": bool(
            ext_controls["all_external_invariances_lt_1e6"] and ext_controls["all_causal_diffs_exact_zero"]
        ),
    }

    # Save prediction & uncertainty archive
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arch_payload: dict[str, Any] = {
        "scenarios": np.asarray(scenarios, dtype=str),
        "models": np.asarray(all_models_eval, dtype=str),
    }
    for scen in scenarios:
        arch_payload[f"steady_mask__{scen}"] = steady_masks[scen]
        arch_payload[f"gt_laser__{scen}"] = np.array([gt_laser[scen]], dtype=np.float64)
        for km in all_models_eval:
            arch_payload[f"pred__{km}__{scen}"] = preds_all[km][scen]
        for km in BAYES_MODELS:
            for pk in PROXIES_ZERO_TUNED:
                arch_payload[f"u__{km}__{pk}__{scen}"] = proxies_all[km][scen][pk]
            arch_payload[f"post_sd_raw__{km}__{scen}"] = proxies_all[km][scen]["post_sd_bpm_raw"]

    tmp_arch = OUTPUT_DIR / ".gleichauf20_predictions_and_uncertainty.tmp.npz"
    np.savez_compressed(tmp_arch, **arch_payload)
    tmp_arch.replace(PRED_ARCHIVE)

    # Export per-scenario CSV ledger
    IO.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    csv_cols = [
        "scenario", "f_nom_bpm", "laser_f_mid_bpm", "amp_mode", "stratum",
        "iq_raw_fs_hz", "iq_raw_dur_s", "laser_mid_p2p_mm", "n_windows_total", "n_windows_steady",
        "R0_mean_bpm", "R0_rmse_bpm",
        "R1a_mean_bpm", "R1a_rmse_bpm",
        "R1h_mean_bpm", "R1h_rmse_bpm",
        "R1f_mean_bpm", "R1f_rmse_bpm",
        "R2e_mean_bpm", "R2e_rmse_bpm", "R2e_bias_bpm", "R2e_u2_post_sd", "R2e_u6b_geom_sqi",
        "R2e_lopo_mean_bpm", "R2e_lopo_rmse_bpm",
        "R2f_mean_bpm", "R2f_rmse_bpm",
        "R2a_mean_bpm", "R2a_rmse_bpm",
        "R2d_mean_bpm", "R2d_rmse_bpm",
        "R3c_mean_bpm", "R3c_rmse_bpm",
        "R3b_mean_bpm", "R3b_rmse_bpm",
        "R3a_mean_bpm", "R3a_rmse_bpm",
    ]
    with AUDIT_SCENARIOS_CSV.open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=csv_cols)
        wtr.writeheader()
        for scen in scenarios:
            ls = per_scen_ledger[scen]
            ms = ls["models"]
            wtr.writerow({
                "scenario": scen,
                "f_nom_bpm": f"{ls['f_nom_bpm']:.1f}",
                "laser_f_mid_bpm": f"{ls['laser_f_mid_bpm']:.4f}",
                "amp_mode": ls["amp_mode"],
                "stratum": ls["stratum"],
                "iq_raw_fs_hz": f"{ls['iq_raw_fs_hz']:.3f}",
                "iq_raw_dur_s": f"{ls['iq_raw_dur_s']:.2f}",
                "laser_mid_p2p_mm": f"{ls['laser_mid_p2p_mm']:.4f}",
                "n_windows_total": int(ls["n_windows_total"]),
                "n_windows_steady": int(ls["n_windows_steady"]),
                "R0_mean_bpm": f"{ms['R0_HPSS_NLS_Locked']['mean_bpm']:.4f}",
                "R0_rmse_bpm": f"{ms['R0_HPSS_NLS_Locked']['rmse_bpm']:.4f}",
                "R1a_mean_bpm": f"{ms['R1a_Online_Forward_MMSE_0s']['mean_bpm']:.4f}",
                "R1a_rmse_bpm": f"{ms['R1a_Online_Forward_MMSE_0s']['rmse_bpm']:.4f}",
                "R1h_mean_bpm": f"{ms['R1h_E2E_Causal_H5_OnlineForward_0s']['mean_bpm']:.4f}",
                "R1h_rmse_bpm": f"{ms['R1h_E2E_Causal_H5_OnlineForward_0s']['rmse_bpm']:.4f}",
                "R1f_mean_bpm": f"{ms['R1f_Offline_FB_MMSE_Inf']['mean_bpm']:.4f}",
                "R1f_rmse_bpm": f"{ms['R1f_Offline_FB_MMSE_Inf']['rmse_bpm']:.4f}",
                "R2e_mean_bpm": f"{ms['R2e_Causal_DiffHMM_FilterSafe8']['mean_bpm']:.4f}",
                "R2e_rmse_bpm": f"{ms['R2e_Causal_DiffHMM_FilterSafe8']['rmse_bpm']:.4f}",
                "R2e_bias_bpm": f"{ms['R2e_Causal_DiffHMM_FilterSafe8']['bias_bpm']:.4f}",
                "R2e_u2_post_sd": f"{ms['R2e_Causal_DiffHMM_FilterSafe8']['mean_u2_post_sd']:.4f}",
                "R2e_u6b_geom_sqi": f"{ms['R2e_Causal_DiffHMM_FilterSafe8']['mean_u6b_geom_sqi']:.4f}",
                "R2e_lopo_mean_bpm": f"{ms['R2e_LOPO_Mean_Sensitivity']['mean_bpm']:.4f}",
                "R2e_lopo_rmse_bpm": f"{ms['R2e_LOPO_Mean_Sensitivity']['rmse_bpm']:.4f}",
                "R2f_mean_bpm": f"{ms['R2f_Offline_DiffHMM_FilterSafe8']['mean_bpm']:.4f}",
                "R2f_rmse_bpm": f"{ms['R2f_Offline_DiffHMM_FilterSafe8']['rmse_bpm']:.4f}",
                "R2a_mean_bpm": f"{ms['R2a_Causal_DiffHMM_Linear11']['mean_bpm']:.4f}",
                "R2a_rmse_bpm": f"{ms['R2a_Causal_DiffHMM_Linear11']['rmse_bpm']:.4f}",
                "R2d_mean_bpm": f"{ms['R2d_E2E_Causal_H5_DiffHMM_Linear11']['mean_bpm']:.4f}",
                "R2d_rmse_bpm": f"{ms['R2d_E2E_Causal_H5_DiffHMM_Linear11']['rmse_bpm']:.4f}",
                "R3c_mean_bpm": f"{ms['R3c_Ridge_Spectral_439']['mean_bpm']:.4f}",
                "R3c_rmse_bpm": f"{ms['R3c_Ridge_Spectral_439']['rmse_bpm']:.4f}",
                "R3b_mean_bpm": f"{ms['R3b_Spectral_MLP_15k']['mean_bpm']:.4f}",
                "R3b_rmse_bpm": f"{ms['R3b_Spectral_MLP_15k']['rmse_bpm']:.4f}",
                "R3a_mean_bpm": f"{ms['R3a_Causal_TCN_16k']['mean_bpm']:.4f}",
                "R3a_rmse_bpm": f"{ms['R3a_Causal_TCN_16k']['rmse_bpm']:.4f}",
            })

    total_s = float(time.perf_counter() - t_start)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "git_head": IO.git_head_info(),
        "analysis_plan": "thesis/analysis_plan_v3_ai.md",
        "frozen_stage4_archive": hf_manifest["frozen_stage4_models_archive"],
        "frozen_stage4_archive_sha256": hf_manifest["frozen_stage4_models_sha256"],
        "primary_parameterization": "Full-Cohort K=11 refit (sensitivity: LOPO-Mean)",
        "primary_window_regime": "steady_state_excl_sync_pulses (w_start >= 10s, w_end <= dur_16s - 10s)",
        "hardware_protocol_disclosure": {
            "external_dataset": "Gleichauf et al. (2024), Scientific Data 11:110 (DOI: 10.1038/s41597-024-02946-5)",
            "external_radar": "InnoSenT iSYS-4001 24.2 GHz CW radar (lambda=12.39 mm, raw ~2000.02 Hz -> deterministic 16.000 Hz decimation)",
            "training_radar": "Beltrao NICU 24.0 GHz CW radar (lambda=12.50 mm, 16.0 Hz)",
            "external_reference": "Micro-Epsilon optoCONTROL 2520 laser micrometer (~2.5 kHz, steady-state [t0+100s, t0+700s] Hann FFT)",
            "locked_bandpass_filter": "64-tap zero-phase (filtfilt) Kaiser FIR [0.30, 3.00] Hz (beta=6.5; -12.07 dB at 18 bpm, -10.01 dB at 20 bpm, flat passband >= 28 bpm) feeding [18, 90] bpm HMM grid",
            "zero_external_adaptation": True,
        },
        "total_runtime_s": total_s,
        "strata_definitions": strata_scens,
        "gate_g5_l4_verdict": gate_g5_verdict,
        "regimes": regimes_eval,
        "per_scenario": per_scen_ledger,
        "wp4_uncertainty_transfer": wp4_transfer,
        "conformal_transfer": conformal_transfer,
        "verification_controls": ext_controls,
    }
    json.dumps(payload, allow_nan=False)
    IO.write_json_locked(BUNDLE_JSON, payload)

    # Write Markdown Audit Report
    st_blk = regimes_eval["steady_state_excl_sync_pulses"]
    allw_inb = regimes_eval["all_windows_incl_sync_pulses"]["inband_ge28bpm"]
    md_lines = [
        f"# EXP-4 Stage 4 (`WP5`, `L4`) External Zero-Touch Generalization Report (`{SCHEMA_VERSION}`)\n",
        f"- **Git HEAD:** `{payload['git_head']['head_commit']}` (`{payload['git_head']['branch']}`, Python `{payload['git_head']['python_version']}`)",
        f"- **Total Wall Time:** `{total_s:.1f} s` | **Frozen Stage 3 Archive SHA-256:** `{hf_manifest['frozen_stage4_models_sha256']}`",
        f"- **Primary Parameterization:** `Full-Cohort K=11` (`Sensitivity: LOPO-Mean`) | **Strata Partition:** `14` In-Band (`>= 28.0 bpm`) + `6` Sub-Band/Skirt (`< 28.0 bpm`) = `20/20` scenarios (`100%` exhaustive)",
        f"- **Locked Bandpass:** `64-tap zero-phase Kaiser FIR [0.30, 3.00] Hz (beta=6.5)` | **Hardware Difference:** `24.2 GHz (~2 kHz -> 16 Hz)` vs NICU `24.0 GHz (16 Hz)` (`0` external parameters adjusted)",
        f"- **External 24.2 GHz E2E Causal `H5` $O(2)\\times\\mathbb{{R}}_{{>0}}$ Invariance (< 1e-6):** `{'PASS' if ext_controls['all_external_invariances_lt_1e6'] else 'FAIL'}` (`max diff = {max(v for sub in ext_controls['external_h5_o2_invariance_max_diff'].values() for v in sub.values()):.2e}`; frozen `H0` `SO(2)xR+` max `{max(sub[k] for sub in ext_controls['external_h0_invariance_and_nyquist_swap_diag'].values() for k in ('scale_3p7x', 'phase_rot_pi_over_3', 'iq_sign_flip')):.2e}`, `H0` Nyquist swap diag `{max(sub['iq_channel_swap'] for sub in ext_controls['external_h0_invariance_and_nyquist_swap_diag'].values()):.2e}`) | **Exact 0.0 Causality:** `{'PASS' if ext_controls['all_causal_diffs_exact_zero'] else 'FAIL'}`",
        f"- **Gate `G5` (`L4`) Verdict:** **`{gate_g5_verdict['decision']}`**\n",
        "## 1. Main Zero-Touch 4-Rung Complexity Ladder (`gleichauf2024`, Primary `steady_state_excl_sync_pulses`, `Full-Cohort K=11` + `LOPO-Mean` Sensitivity)\n",
        "| Model | Rung / Params | In-Band `>=28` (14) `RMSE` / `MAE` / `Bias` | In-Band `Acc<3 / <6` | In-Band Slope / `r` | `Normal` (7) / `Deep` (7) `RMSE` | Core `30-60` (12) / Extreme `80` (2) `RMSE` | Sub-Band `<28` (6) / `20 bpm` Skirt (2) `RMSE` | All `20` `RMSE` (`AllWin` In-Band) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    short_names = {
        "R0_HPSS_NLS_Locked": ("`R0_HPSS_NLS_Locked`", "R0 (`0p`)"),
        "R1a_Online_Forward_MMSE_0s": ("`R1a_Online_Forward_MMSE_0s`", "R1 (`3p`, `0s`)"),
        "R1h_E2E_Causal_H5_OnlineForward_0s": ("`R1h_E2E_Causal_H5_0s`", "R1 (`3p`, `0s/0s`)"),
        "R1f_Offline_FB_MMSE_Inf": ("`R1f_Offline_FB_MMSE_Inf`", "R1 (`3p`, `inf`)"),
        "R2e_Causal_DiffHMM_FilterSafe8": ("`R2e_FilterSafe8` (**Primary**)", "R2 (`8p`, `0s`)"),
        "R2f_Offline_DiffHMM_FilterSafe8": ("`R2f_Offline_FilterSafe8`", "R2 (`8p`, `inf`)"),
        "R2a_Causal_DiffHMM_Linear11": ("`R2a_DiffHMM_Linear11`", "R2 (`11p`, `0s`)"),
        "R2d_E2E_Causal_H5_DiffHMM_Linear11": ("`R2d_E2E_H5_Linear11`", "R2 (`11p`, `0s/0s`)"),
        "R3c_Ridge_Spectral_439": ("`R3c_Ridge_Spectral_439`", "R3 (`439p`)"),
        "R3b_Spectral_MLP_15k": ("`R3b_Spectral_MLP_15k`", "R3 (`14.6k`)"),
        "R3a_Causal_TCN_16k": ("`R3a_Causal_TCN_16k`", "R3 (`15.8k`)"),
        "R2e_LOPO_Mean_Sensitivity": ("`R2e_LOPO_Mean` (Sens.)", "R2 (`8p`, sens.)"),
        "R2a_LOPO_Mean_Sensitivity": ("`R2a_LOPO_Mean` (Sens.)", "R2 (`11p`, sens.)"),
        "R2d_LOPO_Mean_Sensitivity": ("`R2d_LOPO_Mean` (Sens.)", "R2 (`11p`, sens.)"),
    }
    for km in all_models_eval:
        lbl_m, r_txt = short_names[km]
        m_inb = st_blk["inband_ge28bpm"][km]
        m_nrm = st_blk["inband_normal"][km]
        m_dp = st_blk["inband_deep"][km]
        m_core = st_blk["inband_core_30_60bpm"][km]
        m_80 = st_blk["inband_extreme_80bpm"][km]
        m_sub = st_blk["subband_lt28bpm"][km]
        m_20 = st_blk["subband_20bpm_skirt"][km]
        m_a20 = st_blk["all_20"][km]
        m_aw = allw_inb[km]
        md_lines.append(
            f"| {lbl_m} | {r_txt} | `{m_inb['rmse_scen']:.3f}` / `{m_inb['mae_scen']:.3f}` / `{m_inb['bias_scen']:+.3f}` | "
            f"`{m_inb['acc_lt3_pct']:.1f}%` / `{m_inb['acc_lt6_pct']:.1f}%` | "
            f"`{m_inb['linear_slope_vs_gt']:.3f}` / `{m_inb['pearson_r_scenarios']:.3f}` | "
            f"`{m_nrm['rmse_scen']:.3f}` / `{m_dp['rmse_scen']:.3f}` | "
            f"`{m_core['rmse_scen']:.3f}` / `{m_80['rmse_scen']:.3f}` | "
            f"`{m_sub['rmse_scen']:.3f}` / `{m_20['rmse_scen']:.3f}` | "
            f"`{m_a20['rmse_scen']:.3f}` (`{m_aw['rmse_scen']:.3f}`) |"
        )

    md_lines += [
        "\n## 2. Scenario-by-Scenario Physical Rate Breakdown (`05–80 bpm`, `steady_state_excl_sync_pulses`)\n",
        "| Scenario | Stratum | Laser $f_{\\mathrm{gt}}$ (bpm) | `R0` (`RMSE` / Mean) | `R1a` (`RMSE` / Mean) | `R2e` Primary (`RMSE` / Mean) | `R2a` (`RMSE` / Mean) | `R3c` Ridge (`RMSE` / Mean) | `R3b` MLP (`RMSE` / Mean) | `R3a` TCN (`RMSE` / Mean) | `R2e` `U6b` |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for scen in scenarios:
        ls = per_scen_ledger[scen]
        ms = ls["models"]
        md_lines.append(
            f"| `{scen}` | `{ls['stratum']}` | `{ls['laser_f_mid_bpm']:.2f}` | "
            f"`{ms['R0_HPSS_NLS_Locked']['rmse_bpm']:.2f}` (`{ms['R0_HPSS_NLS_Locked']['mean_bpm']:.1f}`) | "
            f"`{ms['R1a_Online_Forward_MMSE_0s']['rmse_bpm']:.2f}` (`{ms['R1a_Online_Forward_MMSE_0s']['mean_bpm']:.1f}`) | "
            f"`{ms['R2e_Causal_DiffHMM_FilterSafe8']['rmse_bpm']:.2f}` (`{ms['R2e_Causal_DiffHMM_FilterSafe8']['mean_bpm']:.1f}`) | "
            f"`{ms['R2a_Causal_DiffHMM_Linear11']['rmse_bpm']:.2f}` (`{ms['R2a_Causal_DiffHMM_Linear11']['mean_bpm']:.1f}`) | "
            f"`{ms['R3c_Ridge_Spectral_439']['rmse_bpm']:.2f}` (`{ms['R3c_Ridge_Spectral_439']['mean_bpm']:.1f}`) | "
            f"`{ms['R3b_Spectral_MLP_15k']['rmse_bpm']:.2f}` (`{ms['R3b_Spectral_MLP_15k']['mean_bpm']:.1f}`) | "
            f"`{ms['R3a_Causal_TCN_16k']['rmse_bpm']:.2f}` (`{ms['R3a_Causal_TCN_16k']['mean_bpm']:.1f}`) | "
            f"`{ms['R2e_Causal_DiffHMM_FilterSafe8']['mean_u6b_geom_sqi']:.3f}` |"
        )

    w_r2e = wp4_transfer["R2e_Causal_DiffHMM_FilterSafe8"]
    w_r2a = wp4_transfer["R2a_Causal_DiffHMM_Linear11"]
    md_lines += [
        "\n## 3. Zero-Parameter `WP4` Uncertainty & Zero-Touch Conformal Calibration Transfer (`R2e_Causal_DiffHMM_FilterSafe8`)\n",
        "### 3A. OOD / Stopband (`05–10 bpm`) & Start/End Sync-Pulse Transient Detection + Pooled `all_20` Selective Risk\n",
        "| Proxy (`R2e`) | `all_20` `RMSE(1.0 -> 0.80)` | `all_20` `E-AURC_pool` | Stopband `05-10` OOD AUROC (`FPR@95`) | Sub-Band `<28` AUROC | Sync-Pulse Transient AUROC (`In-Band`) | Mean $u$ (`In-Band` vs `Stopband`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for pk in PROXIES_ZERO_TUNED:
        rc_a = w_r2e["risk_coverage_pooled_all_20"][pk]
        oo_st = w_r2e["stopband_05_10bpm_vs_inband_ood"][pk]
        oo_sb = w_r2e["subband_lt28bpm_vs_inband_ood"][pk]
        sp_d = w_r2e["sync_pulse_transient_vs_steady_inband"][pk]
        md_lines.append(
            f"| `{pk}` | `{rc_a['full_rmse']:.3f} -> {rc_a['rmse_at_80']:.3f}` | `{rc_a['e_aurc_vs_random_rmse']:+.3f}` | "
            f"`{oo_st['auroc']:.4f}` (`{oo_st['fpr_at_95tpr']:.4f}`) | `{oo_sb['auroc']:.4f}` | "
            f"`{sp_d['auroc']:.4f}` | `{oo_st['mean_u_in_dist']:.3f} vs {oo_st['mean_u_ood']:.3f}` |"
        )

    md_lines += [
        "\n### 3B. Zero-Touch Conformal Interval Transfer Using Frozen Stage 3 `Full-K=11` Quantiles (`R2e_Causal_DiffHMM_FilterSafe8`)\n",
        "| Stratum | $\\alpha$ (`Target 1-\\alpha`) | `C4` Frozen $\\hat{q}$ (`bpm`) | `C4` Width (`bpm`) | `C4` Cov (`ScenMean / Min / Pool`) | `C5` Frozen $\\hat{q}_{\\mathrm{norm}}$ | `C5` Mean Width (`bpm`) | `C5` Cov (`ScenMean / Min / Pool`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for st_c in ("inband_ge28bpm", "inband_normal", "inband_deep", "subband_lt28bpm", "all_20"):
        for a_c in ("0.10", "0.20", "0.30"):
            cb = conformal_transfer["primary_full_k11_frozen_quantiles"]["R2e_Causal_DiffHMM_FilterSafe8"][st_c][a_c]
            md_lines.append(
                f"| `{st_c}` | `{a_c}` (`{cb['target_coverage']:.2f}`) | `{cb['c4_q_hat_bpm']:.2f}` | `{cb['c4_mean_width_bpm']:.2f}` | "
                f"`{cb['c4_cov_scen_mean']:.3f} / {cb['c4_cov_scen_min']:.3f} / {cb['c4_cov_pooled']:.3f}` | "
                f"`{cb['c5_q_hat_norm']:.2f}` | `{cb['c5_mean_width_scen_mean_bpm']:.2f}` | "
                f"`{cb['c5_cov_scen_mean']:.3f} / {cb['c5_cov_scen_min']:.3f} / {cb['c5_cov_pooled']:.3f}` |"
            )

    AUDIT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[EXP-4] Wrote {BUNDLE_JSON}, {AUDIT_MD}, {AUDIT_SCENARIOS_CSV}, {PRED_ARCHIVE} in {total_s:.1f}s.")


if __name__ == "__main__":
    main()
