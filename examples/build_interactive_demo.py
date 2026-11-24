#!/usr/bin/env python3
"""Build a zero-install, offline interactive HTML5 dashboard for NICU Session 7.3.

Usage:
  python3 examples/build_interactive_demo.py                    # Rebuilds HTML from tracked fixtures + live pipeline
  python3 examples/build_interactive_demo.py --refresh-fixture  # Also refreshes LOPO fixture from local Stage 2-3 caches

Outputs:
  examples/session_7_3_live_demo.html (single self-contained file, zero network/CDN deps)
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_workspace.src import config as C
from paper_workspace.src.data_loader import load_radar
from paper_workspace.src.demod import csd
from paper_workspace.src.estimation import nls_estimate
from paper_workspace.src.filtering import autocorr_coarse, bandpass
from paper_workspace.src.hpss import mitigate_hpss
from thesis import diff_hmm_and_foils as D
from thesis import io_utils as IO
from thesis import viterbi_tracker as VT
from thesis import wp1_wp2_causal_hmm as W
from thesis import wp3_wp4_uncertainty_conformal as UQ

FROZEN_ARCHIVE_PATH = ROOT / "thesis" / "outputs" / "stage3_uncertainty" / "frozen_stage4_models.npz"
EXPECTED_FROZEN_SHA = "af33a83e157f7d2efb0f47e50c9544ee19bacb7da88ab0e0235ccead029cbcc4"
FIXTURE_PATH = ROOT / "examples" / "session_7_3_lopo_fixture.npz"
DEFAULT_OUTPUT_PATH = ROOT / "examples" / "session_7_3_live_demo.html"

LOPO_MODELS: tuple[tuple[str, str], ...] = (
    ("pred_R0", "R0_HPSS_NLS_Locked"),
    ("pred_R1a", "R1a_Online_Forward_MMSE_0s"),
    ("pred_R1f", "R1f_Offline_FB_MMSE_Inf"),
    ("pred_R2e_lopo", "R2e_Causal_DiffHMM_FilterSafe8"),
    ("pred_R3c", "R3c_Ridge_Spectral_439"),
    ("pred_R3b", "R3b_Spectral_MLP_15k"),
    ("pred_R3a", "R3a_Causal_TCN_16k"),
)


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_bundle() -> dict[str, np.ndarray]:
    if not FROZEN_ARCHIVE_PATH.is_file():
        raise FileNotFoundError(f"Missing frozen Stage 4 archive: {FROZEN_ARCHIVE_PATH}")
    actual_sha = _sha256_bytes(FROZEN_ARCHIVE_PATH)
    if actual_sha != EXPECTED_FROZEN_SHA:
        raise AssertionError(f"Archive SHA mismatch: {actual_sha} != {EXPECTED_FROZEN_SHA}")
    with np.load(FROZEN_ARCHIVE_PATH, allow_pickle=False) as zf:
        return {k: zf[k].copy() for k in zf.files}


def _extract_fixture_from_full_caches() -> dict[str, np.ndarray]:
    p2_path = ROOT / "thesis" / "outputs" / "stage2_complexity" / "pool30_predictions.npz"
    p3_path = ROOT / "thesis" / "outputs" / "stage3_uncertainty" / "pool30_uncertainty_and_conformal.npz"
    cache_73 = IO.load_locked_cache("7.3")
    with np.load(p2_path, allow_pickle=False) as p2, np.load(p3_path, allow_pickle=False) as p3:
        out: dict[str, np.ndarray] = {
            "valid": np.asarray(cache_73["valid"], dtype=bool),
            "ref_1hz": np.asarray(p3["ref_1hz__7.3"], dtype=np.float64),
            "ref_tla": np.asarray(p3["ref_tla__7.3"], dtype=np.float64),
            "pause_gt5s": np.asarray(p3["pause_gt5s__7.3"], dtype=bool),
            "unc_sd_R2e_lopo": np.asarray(p2["unc_sd__R2e_Causal_DiffHMM_FilterSafe8__7.3"], dtype=np.float64),
            "u2_R2e_lopo": np.asarray(p3["u__R2e_Causal_DiffHMM_FilterSafe8__U2_Posterior_SD__7.3"], dtype=np.float64),
            "u6b_R2e_lopo": np.asarray(p3["u__R2e_Causal_DiffHMM_FilterSafe8__U6b_ZeroTuned_Geometric_SQI__7.3"], dtype=np.float64),
            "u6c_R2e_lopo": np.asarray(p3["u__R2e_Causal_DiffHMM_FilterSafe8__U6c_LOPO_Calibrated_SQI__7.3"], dtype=np.float64),
        }
        for short_key, full_name in LOPO_MODELS:
            out[short_key] = np.asarray(p2[f"pred__{full_name}__7.3"], dtype=np.float64)
    return out


def load_or_refresh_session73_lopo(refresh_fixture: bool) -> dict[str, np.ndarray]:
    p2_path = ROOT / "thesis" / "outputs" / "stage2_complexity" / "pool30_predictions.npz"
    p3_path = ROOT / "thesis" / "outputs" / "stage3_uncertainty" / "pool30_uncertainty_and_conformal.npz"
    has_full = p2_path.is_file() and p3_path.is_file()

    if refresh_fixture:
        if not has_full:
            raise FileNotFoundError("Cannot --refresh-fixture without Stage 2-3 .npz caches.")
        data = _extract_fixture_from_full_caches()
        np.savez_compressed(FIXTURE_PATH, **data)
        return data

    if FIXTURE_PATH.is_file():
        with np.load(FIXTURE_PATH, allow_pickle=False) as fx:
            data = {k: fx[k].copy() for k in fx.files}
        if has_full:
            ref_data = _extract_fixture_from_full_caches()
            for k, arr in ref_data.items():
                if arr.dtype == bool:
                    assert np.array_equal(data[k], arr), f"Fixture mismatch on {k}"
                else:
                    assert np.allclose(data[k], arr, atol=1e-12, equal_nan=True), f"Fixture mismatch on {k}"
        return data

    if has_full:
        data = _extract_fixture_from_full_caches()
        np.savez_compressed(FIXTURE_PATH, **data)
        return data

    raise FileNotFoundError(f"Missing both {FIXTURE_PATH} and local Stage 2-3 archives.")


def _extract_iq_windows(bi: np.ndarray, bq: np.ndarray, n_win: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = int(C.UPDATE_S * C.FS_RADAR)
    win_len = int(C.WIN_SAMPLES)
    grid = VT.BR_GRID_1BPM
    phi = np.empty((n_win, grid.size, D.N_SAL_CHANNELS), dtype=np.float64)
    s_anc = np.empty((n_win, grid.size), dtype=np.float64)
    r0_est = np.empty(n_win, dtype=np.float64)
    for idx in range(n_win):
        s0 = idx * step
        x_win = csd(bi[s0 : s0 + win_len], bq[s0 : s0 + win_len])
        d_raw = bandpass(x_win)
        d_clean = bandpass(mitigate_hpss(d_raw)["x_hat"])
        f_coarse = autocorr_coarse(d_clean)
        r0_est[idx] = float(nls_estimate(d_clean, f_coarse)[0])
        s_anc[idx] = VT.extract_br_anchored_nls_spectrum(d_clean, f_coarse, grid)
        phi[idx] = D.extract_salience_features_window(d_raw, d_clean, f_coarse, grid)
    return phi, s_anc, r0_est


def _quantize_rows_u8_b64(mat: np.ndarray) -> str:
    row_max = np.maximum(np.max(mat, axis=1, keepdims=True), 1e-12)
    normed = np.clip(mat / row_max, 0.0, 1.0)
    q = np.rint(normed * 255.0).astype(np.uint8)
    return base64.b64encode(q.tobytes(order="C")).decode("ascii")


def compute_live_hmm_matrices(
    zf: dict[str, np.ndarray],
    lopo_data: dict[str, np.ndarray],
) -> tuple[str, str, list[float], dict[str, float]]:
    """Run live signal extraction on tracked Session 7.3 & return salience/posterior base64 matrices."""
    bi, bq = load_radar("7.3")
    n_win = int(lopo_data["ref_1hz"].size)
    phi_73, _, r0_live = _extract_iq_windows(bi, bq, n_win)
    assert np.allclose(r0_live, lopo_data["pred_R0"], atol=1e-6), "Live R0 diverged from locked fixture"

    grid = VT.BR_GRID_1BPM
    r2e_mean = D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward", grid_bpm=grid)
    r2e_mean.theta = np.asarray(zf["r2e_lopo_mean_theta"], dtype=np.float64).copy()
    lss, lsj, lpi = r2e_mean._trans_params(r2e_mean.theta)
    t_mat, _ = D.build_mixture_transition_and_grad(grid, lss, lsj, lpi)
    ell, _ = D.hmm_emission_and_grad(phi_73, r2e_mean.theta, mode="linear_filtersafe")
    _, f_cache = D.forward_hmm_predictions_and_bptt(ell, t_mat, grid, decoder="causal_forward")

    # Full-refit frozen FilterSafe-8 parameters for display
    r2e_full = D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward", grid_bpm=grid)
    r2e_full.theta = np.asarray(zf["r2e_full_refit_theta"], dtype=np.float64).copy()

    c5_q80 = float(zf["c5_q_hat_norm_1hz"][2])
    c5_half = np.round(c5_q80 * np.maximum(lopo_data["unc_sd_R2e_lopo"], 1.0), 3).tolist()
    return (
        _quantize_rows_u8_b64(f_cache["e_mat"]),
        _quantize_rows_u8_b64(f_cache["alpha"]),
        c5_half,
        r2e_full.interpretable_params(),
    )


def _synth_cw(rate_bpm: float, duration_s: float = 54.0, disp_mm: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(int(duration_s * C.FS_RADAR), dtype=np.float64) / float(C.FS_RADAR)
    d_t = disp_mm * 1e-3 * np.sin(2.0 * math.pi * (rate_bpm / 60.0) * t)
    phase = 0.45 + (4.0 * math.pi / float(C.LAMBDA_M)) * d_t
    z = 1.25 * np.exp(1j * phase)
    return z.real, z.imag


def _eval_foils_and_r2e(
    phi: np.ndarray,
    s_anc: np.ndarray,
    r0_est: np.ndarray,
    zf: dict[str, np.ndarray],
    r2e: D.DifferentiableSymmetricHMM,
    tcn: D.CausalDilatedTCN,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    grid = VT.BR_GRID_1BPM
    r1a_cfg = tuple(float(v) for v in zf["r1a_mode_cfg"])
    pr_1a = W.hmm_forward_filter(s_anc, grid, r1a_cfg[0], r1a_cfg[1], r1a_cfg[2])
    pr_r2e = r2e.predict_session(phi)

    x_flat = D.flatten_salience_tensor(phi)
    x_ridge = (x_flat - zf["r3c_ridge_mean"]) / zf["r3c_ridge_scale"]
    pred_r3c = np.clip(x_ridge @ zf["r3c_ridge_coef"] + float(zf["r3c_ridge_intercept"][0]), 12.0, 120.0)

    h = (x_flat - zf["r3b_mlp_mean"]) / zf["r3b_mlp_scale"]
    h = np.maximum(0.0, h @ zf["r3b_mlp_W_0"] + zf["r3b_mlp_b_0"])
    h = np.maximum(0.0, h @ zf["r3b_mlp_W_1"] + zf["r3b_mlp_b_1"])
    pred_r3b = np.clip((h @ zf["r3b_mlp_W_2"]).ravel() + float(zf["r3b_mlp_b_2"][0]), 12.0, 120.0)

    pred_r3a = tcn.predict_session(phi)
    uq = UQ.extract_zero_tuned_proxies_session(phi, pr_r2e["post_entropy"], pr_r2e["post_sd_bpm"])
    return (
        {
            "R0": r0_est,
            "R1a": pr_1a["est_mmse"],
            "R2e": pr_r2e["est_mmse"],
            "R3c": pred_r3c,
            "R3b": pred_r3b,
            "R3a": pred_r3a,
        },
        {
            "U2": uq["U2_Posterior_SD"],
            "U6b": uq["U6b_ZeroTuned_Geometric_SQI"],
        },
    )


def compute_stress_benchmarks(zf: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compute live frozen Stage 4 14-72 bpm frequency sweep and 20-bpm -> WGN dropout stress."""
    grid = VT.BR_GRID_1BPM
    r2e = D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward", grid_bpm=grid)
    r2e.theta = np.asarray(zf["r2e_full_refit_theta"], dtype=np.float64).copy()

    tcn = D.CausalDilatedTCN(in_dim=438, channels=24, dilations=(1, 2, 4), kernel_size=3)
    tcn.mean_ = zf["r3a_tcn_mean"].copy()
    tcn.scale_ = zf["r3a_tcn_scale"].copy()
    for k in tcn.param_keys:
        tcn.params[k] = zf[f"r3a_tcn_{k}"].copy()

    rates = list(range(14, 74, 2))
    sweep: dict[str, Any] = {"rates_bpm": rates, "R0": [], "R1a": [], "R2e": [], "R3c": [], "R3b": [], "R3a": [], "U2": [], "U6b": []}
    n_win_sweep = 13
    for rate in rates:
        bi, bq = _synth_cw(float(rate), duration_s=54.0)
        phi, s_anc, r0_est = _extract_iq_windows(bi, bq, n_win_sweep)
        preds, uq = _eval_foils_and_r2e(phi, s_anc, r0_est, zf, r2e, tcn)
        for mk in ("R0", "R1a", "R2e", "R3c", "R3b", "R3a"):
            sweep[mk].append(round(float(np.mean(preds[mk][3:])), 2))
        for uk in ("U2", "U6b"):
            sweep[uk].append(round(float(np.mean(uq[uk][3:])), 3))

    # 36-window stress replay: 12 windows @ 20 bpm -> 12 windows WGN -> 12 windows @ 20 bpm
    bi20, bq20 = _synth_cw(20.0, duration_s=54.0)
    phi20_all, sanc20_all, r020_all = _extract_iq_windows(bi20, bq20, n_win_sweep)
    phi20, sanc20, r020 = phi20_all[-12:], sanc20_all[-12:], r020_all[-12:]

    rng = np.random.default_rng(2026)
    phi_w = np.empty((12, grid.size, D.N_SAL_CHANNELS), dtype=np.float64)
    sanc_w = np.empty((12, grid.size), dtype=np.float64)
    r0_w = np.empty(12, dtype=np.float64)
    for k in range(12):
        wgn = rng.normal(0.0, 1.0, size=int(C.WIN_SAMPLES)) + 1j * rng.normal(0.0, 1.0, size=int(C.WIN_SAMPLES))
        dr = bandpass(wgn)
        dc = bandpass(mitigate_hpss(dr)["x_hat"])
        fc = autocorr_coarse(dc)
        r0_w[k] = float(nls_estimate(dc, fc)[0])
        sanc_w[k] = VT.extract_br_anchored_nls_spectrum(dc, fc, grid)
        phi_w[k] = D.extract_salience_features_window(dr, dc, fc, grid)

    phi_seq = np.concatenate([phi20, phi_w, phi20], axis=0)
    sanc_seq = np.concatenate([sanc20, sanc_w, sanc20], axis=0)
    r0_seq = np.concatenate([r020, r0_w, r020], axis=0)
    sp_preds, sp_uq = _eval_foils_and_r2e(phi_seq, sanc_seq, r0_seq, zf, r2e, tcn)

    replay = {
        "n_win": 36,
        "regime_labels": ["20 bpm Bradypnea"] * 12 + ["WGN Dropout"] * 12 + ["20 bpm Recovery"] * 12,
        "gt_bpm": [20.0] * 12 + [None] * 12 + [20.0] * 12,
        "R0": np.round(sp_preds["R0"], 2).tolist(),
        "R2e": np.round(sp_preds["R2e"], 2).tolist(),
        "R3b": np.round(sp_preds["R3b"], 2).tolist(),
        "R3a": np.round(sp_preds["R3a"], 2).tolist(),
        "U2": np.round(sp_uq["U2"], 3).tolist(),
        "U6b": np.round(sp_uq["U6b"], 3).tolist(),
    }
    return {"sweep": sweep, "dropout_replay": replay}


def load_audited_ladder(lopo_data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    raw_dir = ROOT / "paper_orchestra_bundle" / "raw_materials"
    s2 = json.loads((raw_dir / "metrics_stage2_complexity.json").read_text(encoding="utf-8"))
    s4 = json.loads((raw_dir / "metrics_stage4_external.json").read_text(encoding="utf-8"))

    m1 = s2["cohorts"]["pool30_sync"]["br_1hz_30s"]["methods"]
    mt = s2["cohorts"]["pool30_sync"]["br_trig_bpm_lag_aligned"]["methods"]
    p1 = s2["cohorts"]["pool30_sync"]["br_1hz_30s"]["paired_vs_R0"]
    inb = s4["regimes"]["steady_state_excl_sync_pulses"]["inband_ge28bpm"]
    skt = s4["regimes"]["steady_state_excl_sync_pulses"]["subband_20bpm_skirt"]

    valid = lopo_data["valid"]
    ref_1hz = lopo_data["ref_1hz"]
    rows_spec = [
        ("R0 HPSS+NLS", "0", "0 / 3 s", "pred_R0", "R0_HPSS_NLS_Locked", False),
        ("R1a Fwd HMM", "3", "0 / 3 s", "pred_R1a", "R1a_Online_Forward_MMSE_0s", False),
        ("R1f Offline FB HMM", "3", "inf / 3 s", "pred_R1f", "R1f_Offline_FB_MMSE_Inf", False),
        ("R2e FilterSafe-8 (Ours)", "8", "0 / 3 s", "pred_R2e_lopo", "R2e_Causal_DiffHMM_FilterSafe8", True),
        ("R3c Spectral Ridge", "439", "0 / 3 s", "pred_R3c", "R3c_Ridge_Spectral_439", False),
        ("R3b Spectral MLP", "14,569", "0 / 3 s", "pred_R3b", "R3b_Spectral_MLP_15k", False),
        ("R3a Causal TCN", "15,817", "0 / 3 s", "pred_R3a", "R3a_Causal_TCN_16k", False),
    ]
    rows: list[dict[str, Any]] = []
    for label, params, lookahead, lopo_key, mkey, is_ours in rows_spec:
        err73 = lopo_data[lopo_key][valid] - ref_1hz[valid]
        rmse73 = float(np.sqrt(np.mean(err73 ** 2)))
        ph = f"{p1[mkey]['p_wilcoxon_patient_holm']:.4f}" if mkey in p1 else "baseline"
        rows.append(
            {
                "label": label,
                "params": params,
                "lookahead": lookahead,
                "rmse_73": f"{rmse73:.3f}",
                "nicu_1hz": f"{m1[mkey]['rmse_sess']:.3f}",
                "p_holm": ph,
                "nicu_trig": f"{mt[mkey]['rmse_sess']:.3f}",
                "ext_inband": f"{inb[mkey]['rmse_scen']:.4f}",
                "ext_slope": f"{inb[mkey]['linear_slope_vs_gt']:.4f}",
                "ext_skirt": f"{skt[mkey]['rmse_scen']:.4f}",
                "is_ours": is_ours,
            }
        )
    return rows


def build_payload(refresh_fixture: bool) -> dict[str, Any]:
    zf = load_frozen_bundle()
    lopo = load_or_refresh_session73_lopo(refresh_fixture=refresh_fixture)
    sal_b64, post_b64, c5_half, ip = compute_live_hmm_matrices(zf, lopo)
    stress = compute_stress_benchmarks(zf)
    ladder = load_audited_ladder(lopo)

    n_win = int(lopo["ref_1hz"].size)
    t_min = np.round((np.arange(n_win, dtype=np.float64) * float(C.UPDATE_S) + float(C.WIN_S) / 2.0) / 60.0, 3).tolist()
    return {
        "meta": {
            "session": "7.3",
            "n_win": n_win,
            "valid_count": int(np.sum(lopo["valid"])),
            "pause_count": int(np.sum(lopo["pause_gt5s"] & lopo["valid"])),
            "grid_min": int(VT.BR_GRID_1BPM[0]),
            "grid_max": int(VT.BR_GRID_1BPM[-1]),
            "grid_len": int(VT.BR_GRID_1BPM.size),
            "archive_sha_short": EXPECTED_FROZEN_SHA[:16],
            "r2e_params": {k: round(float(v), 4) for k, v in ip.items()},
        },
        "s73": {
            "t_min": t_min,
            "valid": lopo["valid"].astype(int).tolist(),
            "pause": lopo["pause_gt5s"].astype(int).tolist(),
            "ref_1hz": np.round(lopo["ref_1hz"], 3).tolist(),
            "pred_R0": np.round(lopo["pred_R0"], 3).tolist(),
            "pred_R1a": np.round(lopo["pred_R1a"], 3).tolist(),
            "pred_R1f": np.round(lopo["pred_R1f"], 3).tolist(),
            "pred_R2e": np.round(lopo["pred_R2e_lopo"], 3).tolist(),
            "pred_R3c": np.round(lopo["pred_R3c"], 3).tolist(),
            "pred_R3b": np.round(lopo["pred_R3b"], 3).tolist(),
            "pred_R3a": np.round(lopo["pred_R3a"], 3).tolist(),
            "c5_half_80": c5_half,
            "u2": np.round(lopo["u2_R2e_lopo"], 4).tolist(),
            "u6b": np.round(lopo["u6b_R2e_lopo"], 4).tolist(),
            "u6c": np.round(lopo["u6c_R2e_lopo"], 4).tolist(),
            "sal_u8_b64": sal_b64,
            "post_u8_b64": post_b64,
        },
        "stress": stress,
        "ladder": ladder,
    }


def render_html_document(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light dark" />
<title>FilterSafe-8 &amp; Causal UQ — Live NICU Session 7.3 Demo</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg-light: #f6f8fb;
    --bg-dark: #0e131b;
    --card-light: #ffffff;
    --card-dark: #161d29;
    --surface2-light: #edf1f7;
    --surface2-dark: #1e2736;
    --border-light: #d3dbe6;
    --border-dark: #2c3749;
    --fg-light: #121926;
    --fg-dark: #eef2f8;
    --muted-light: #4a576d;
    --muted-dark: #9fb0c8;
    --brand-light: #006c4f;
    --brand-dark: #10c492;
    --accent-light: #005896;
    --accent-dark: #56b4e9;
    --warn-light: #b54500;
    --warn-dark: #ff8c42;
    --danger-light: #a61d56;
    --danger-dark: #f0629e;
    --purple-light: #6d28d9;
    --purple-dark: #b694f9;

    --bg: var(--bg-light);
    --card: var(--card-light);
    --surface2: var(--surface2-light);
    --border: var(--border-light);
    --fg: var(--fg-light);
    --muted: var(--muted-light);
    --brand-color: var(--brand-light);
    --accent: var(--accent-light);
    --warn: var(--warn-light);
    --danger: var(--danger-light);
    --purple: var(--purple-light);

    --color-scrollbar-track: #e6ebf2;
    --color-scrollbar-thumb: #6b7280;
    scrollbar-color: var(--color-scrollbar-thumb) var(--color-scrollbar-track);
    accent-color: var(--brand-color);
  }}

  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: var(--bg-dark);
      --card: var(--card-dark);
      --surface2: var(--surface2-dark);
      --border: var(--border-dark);
      --fg: var(--fg-dark);
      --muted: var(--muted-dark);
      --brand-color: var(--brand-dark);
      --accent: var(--accent-dark);
      --warn: var(--warn-dark);
      --danger: var(--danger-dark);
      --purple: var(--purple-dark);
      --color-scrollbar-track: #1b2230;
      --color-scrollbar-thumb: #8a99ad;
    }}
  }}

  @supports (color: light-dark(white, black)) {{
    :root {{
      --bg: light-dark(var(--bg-light), var(--bg-dark));
      --card: light-dark(var(--card-light), var(--card-dark));
      --surface2: light-dark(var(--surface2-light), var(--surface2-dark));
      --border: light-dark(var(--border-light), var(--border-dark));
      --fg: light-dark(var(--fg-light), var(--fg-dark));
      --muted: light-dark(var(--muted-light), var(--muted-dark));
      --brand-color: light-dark(var(--brand-light), var(--brand-dark));
      --accent: light-dark(var(--accent-light), var(--accent-dark));
      --warn: light-dark(var(--warn-light), var(--warn-dark));
      --danger: light-dark(var(--danger-light), var(--danger-dark));
      --purple: light-dark(var(--purple-light), var(--purple-dark));
      --color-scrollbar-track: light-dark(#e6ebf2, #1b2230);
      --color-scrollbar-thumb: light-dark(#6b7280, #8a99ad);
    }}
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 0;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--fg);
    line-height: 1.45;
    font-size: 14px;
  }}

  .app-shell {{
    max-width: 1380px;
    margin: 0 auto;
    padding: 16px 20px 28px;
  }}

  .topbar {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 14px;
  }}

  .title-wrap h1 {{
    margin: 0 0 4px;
    font-size: 1.28rem;
    font-weight: 700;
    letter-spacing: -0.01em;
  }}
  .subtitle {{
    margin: 0;
    color: var(--muted);
    font-size: 0.86rem;
  }}

  .badge-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
  }}
  .pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--card);
    font-size: 0.78rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }}
  .pill-brand {{ border-color: var(--brand-color); color: var(--brand-color); }}

  /* Tabs guidance: single horizontal row, shape indicator ::after, no panel padding/bg */
  .tabs {{
    width: 100%;
  }}
  .tablist {{
    display: flex;
    overflow-x: auto;
    border-bottom: 1px solid var(--border);
    margin-bottom: 14px;
    gap: 4px;
  }}
  [role="tab"] {{
    position: relative;
    flex: none;
    appearance: none;
    background: transparent;
    border: 0;
    padding: 10px 14px;
    font: inherit;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--muted);
    cursor: pointer;
    white-space: nowrap;
  }}
  [role="tab"]:hover {{ color: var(--fg); }}
  [role="tab"]:focus-visible {{
    outline: 2px solid var(--accent);
    outline-offset: -3px;
    border-radius: 4px;
  }}
  [role="tab"][aria-selected="true"] {{
    color: var(--accent);
  }}
  [role="tab"][aria-selected="true"]::after {{
    content: "";
    position: absolute;
    inset: auto 0 0;
    border-block-end: 3px solid var(--accent);
  }}

  /* Panel spacing strictly on inner wrapper per tabs guide */
  .panel-inner {{
    display: flex;
    flex-direction: column;
    gap: 14px;
  }}

  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
  }}

  .card-header {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }}
  .card-title {{
    margin: 0;
    font-size: 0.96rem;
    font-weight: 700;
  }}
  .card-note {{
    margin: 0;
    font-size: 0.8rem;
    color: var(--muted);
  }}

  .grid-main {{
    display: grid;
    grid-template-columns: minmax(0, 2.2fr) minmax(300px, 1fr);
    gap: 14px;
    align-items: start;
  }}
  .grid-two {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
  }}
  @media (max-width: 980px) {{
    .grid-main, .grid-two {{ grid-template-columns: 1fr; }}
  }}

  .controls-bar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px 14px;
  }}
  .btn {{
    appearance: none;
    border: 1px solid var(--border);
    background: var(--surface2);
    color: var(--fg);
    border-radius: 7px;
    padding: 6px 12px;
    font: inherit;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    font-variant-numeric: tabular-nums;
  }}
  .btn:hover {{ border-color: var(--accent); }}
  .btn:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
  .btn-primary {{
    background: var(--brand-color);
    border-color: var(--brand-color);
    color: #ffffff;
  }}
  @media (prefers-color-scheme: dark) {{
    .btn-primary {{ color: #062119; }}
  }}

  select.sel {{
    background: var(--surface2);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 5px 9px;
    font: inherit;
    font-size: 0.82rem;
    font-weight: 600;
  }}

  .scrubber-wrap {{
    flex: 1 1 260px;
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 220px;
  }}
  .mono-val {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-variant-numeric: tabular-nums;
    font-size: 0.82rem;
  }}

  /* Range Slider styling per slider guide (separate WebKit/Firefox blocks) */
  input[type="range"] {{
    --slider-track-size: 0.45rem;
    --slider-thumb-size: 1.15rem;
    --slider-track-radius: 999px;
    --slider-thumb-radius: 50%;
    --slider-track-color: var(--border);
    --slider-fill-color: var(--brand-color);
    --slider-thumb-color: var(--slider-fill-color);
    --slider-direction: to right;
    --attr-min: attr(min type(<number>), 0);
    --attr-max: attr(max type(<number>), 100);
    --control-value: attr(value type(<number>), 50);
    --slider-progress: calc(progress(var(--control-value), var(--attr-min), var(--attr-max)) * 100%);
    appearance: none;
    inline-size: 100%;
    block-size: 2.5rem;
    margin: 0;
    padding: 0;
    border: 0;
    border-radius: 0;
    background: transparent;
    cursor: pointer;
  }}
  @supports not (opacity: progress(1, 0, 100)) {{
    input[type="range"] {{
      --slider-progress: calc((var(--control-value) - var(--attr-min)) / (var(--attr-max) - var(--attr-min)) * 100%);
    }}
  }}
  input[type="range"]::-webkit-slider-runnable-track {{
    block-size: var(--slider-track-size);
    border: 0;
    border-radius: var(--slider-track-radius);
    background: linear-gradient(
      var(--slider-direction),
      var(--slider-fill-color) var(--slider-progress),
      var(--slider-track-color) var(--slider-progress)
    );
  }}
  input[type="range"]::-webkit-slider-thumb {{
    -webkit-appearance: none;
    appearance: none;
    inline-size: var(--slider-thumb-size);
    block-size: var(--slider-thumb-size);
    margin-block-start: calc((var(--slider-track-size) - var(--slider-thumb-size)) / 2);
    border: 0;
    border-radius: var(--slider-thumb-radius);
    background: var(--slider-thumb-color);
  }}
  input[type="range"]::-moz-range-track {{
    block-size: var(--slider-track-size);
    border: 0;
    border-radius: var(--slider-track-radius);
    background: var(--slider-track-color);
  }}
  input[type="range"]::-moz-range-progress {{
    block-size: var(--slider-track-size);
    border-radius: var(--slider-track-radius);
    background: var(--slider-fill-color);
  }}
  input[type="range"]::-moz-range-thumb {{
    inline-size: var(--slider-thumb-size);
    block-size: var(--slider-thumb-size);
    border: 0;
    border-radius: var(--slider-thumb-radius);
    background: var(--slider-thumb-color);
  }}
  input[type="range"]:focus-visible {{
    outline: 2px solid var(--slider-fill-color);
    outline-offset: 4px;
  }}

  /* Checkboxes per brand-consistent-forms guide */
  .toggle-list {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px 14px;
    margin-top: 8px;
  }}
  .chk-label {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    user-select: none;
  }}
  .swatch {{
    width: 11px;
    height: 3px;
    border-radius: 2px;
    display: inline-block;
  }}
  @supports not (accent-color: var(--brand-color)) {{
    .chk-label input[type="checkbox"].visually-hidden {{
      position: absolute !important;
      clip-path: inset(50%) !important;
      overflow: hidden !important;
      width: 1px !important;
      height: 1px !important;
      margin: -1px !important;
      padding: 0 !important;
      border: 0 !important;
      white-space: nowrap !important;
    }}
    .chk-label {{ position: relative; padding-left: 1.5rem; }}
    .chk-label input[type="checkbox"] + span::before {{
      content: "";
      position: absolute;
      left: 0;
      top: 50%;
      transform: translateY(-50%);
      width: 1rem;
      height: 1rem;
      border: 2px solid var(--border);
      background: var(--card);
      border-radius: 3px;
      box-sizing: border-box;
    }}
    .chk-label input[type="checkbox"]:focus-visible + span::before {{
      outline: 2px solid var(--fg);
      outline-offset: 2px;
    }}
    .chk-label input[type="checkbox"]:checked + span::before {{
      background-color: var(--brand-color);
      border-color: var(--brand-color);
    }}
    .chk-label input[type="checkbox"]:checked + span::after {{
      content: "✓";
      position: absolute;
      left: 0.2rem;
      top: 50%;
      transform: translateY(-50%);
      color: #ffffff;
      font-weight: 700;
      font-size: 0.78rem;
    }}
  }}

  .canvas-box {{
    width: 100%;
    position: relative;
  }}
  canvas {{
    display: block;
    width: 100%;
    border-radius: 6px;
  }}

  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
  }}
  .kpi {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
  }}
  .kpi-k {{
    font-size: 0.75rem;
    color: var(--muted);
    display: block;
  }}
  .kpi-v {{
    font-size: 0.96rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }}

  /* Responsive sticky table per responsive-table guide */
  .table-wrapper {{
    overflow: auto;
    max-inline-size: 100%;
    max-block-size: min(560px, 75vh);
    container-type: inline-size;
    border: 1px solid var(--border);
    border-radius: 8px;
  }}
  .table-wrapper:focus-visible {{
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }}
  table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.84rem;
    font-variant-numeric: tabular-nums;
    --label-2: "Params";
    --label-3: "Lookahead";
    --label-4: "7.3 Held-Out";
    --label-5: "NICU 1 Hz";
    --label-6: "p_Holm";
    --label-7: "NICU Trig-10s";
    --label-8: "Ext >=28 bpm";
    --label-9: "Ext Slope";
    --label-10: "20 bpm Skirt";
  }}
  caption {{
    text-align: left;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 10px 12px;
    background: var(--card);
    border-bottom: 1px solid var(--border);
  }}
  thead {{
    position: sticky;
    inset-block-start: 0;
    z-index: 2;
  }}
  thead th {{
    background: var(--surface2);
    color: var(--fg);
    text-align: left;
    padding: 9px 10px;
    border-bottom: 1px solid var(--border);
    font-weight: 700;
    white-space: nowrap;
  }}
  tbody th {{
    position: sticky;
    inset-inline-start: 0;
    z-index: 1;
    background: var(--card);
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    font-weight: 600;
  }}
  tbody td {{
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  tbody tr.row-ours th,
  tbody tr.row-ours td {{
    background: var(--surface2);
    font-weight: 700;
  }}

  @container (width < 600px) {{
    table, thead, tbody, tr, th, td {{ display: block; }}
    thead {{ display: none; }}
    tbody tr {{
      margin-block-end: 0.85rem;
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
    }}
    tbody th {{
      position: sticky;
      inset-block-start: 0;
      z-index: 2;
      background: var(--surface2);
    }}
    tbody td {{
      padding: 6px 10px;
      white-space: normal;
    }}
    td:nth-child(2)::before {{ content: var(--label-2) ": "; font-weight: 700; color: var(--muted); }}
    td:nth-child(3)::before {{ content: var(--label-3) ": "; font-weight: 700; color: var(--muted); }}
    td:nth-child(4)::before {{ content: var(--label-4) ": "; font-weight: 700; color: var(--muted); }}
    td:nth-child(5)::before {{ content: var(--label-5) ": "; font-weight: 700; color: var(--muted); }}
    td:nth-child(6)::before {{ content: var(--label-6) ": "; font-weight: 700; color: var(--muted); }}
    td:nth-child(7)::before {{ content: var(--label-7) ": "; font-weight: 700; color: var(--muted); }}
    td:nth-child(8)::before {{ content: var(--label-8) ": "; font-weight: 700; color: var(--muted); }}
    td:nth-child(9)::before {{ content: var(--label-9) ": "; font-weight: 700; color: var(--muted); }}
    td:nth-child(10)::before {{ content: var(--label-10) ": "; font-weight: 700; color: var(--muted); }}
  }}
</style>
</head>
<body>
<div class="app-shell">
  <header class="topbar">
    <div class="title-wrap">
      <h1>NICU Anchor Session 7.3 — Live Causal <code>FilterSafe-8</code> &amp; UQ Replay</h1>
      <p class="subtitle">Master's Thesis Demo (Hosam Shibly, BTU M.Sc. AI Module 13600) &middot; Offline Zero-Install &middot; Strict Held-Out LOPO Infant 7 vs. Frozen Stage 4 Models</p>
    </div>
    <div class="badge-row">
      <span class="pill pill-brand">R2e FilterSafe-8: 8 params</span>
      <span class="pill">Archive SHA: <code>{payload["meta"]["archive_sha_short"]}</code></span>
      <label class="pill" for="theme-select">
        Theme
        <select id="theme-select" class="sel" aria-label="Color scheme">
          <option value="light dark">Auto</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </label>
    </div>
  </header>

  <div class="tabs">
    <div class="tablist" focusgroup="tablist nomemory" role="tablist" aria-label="Interactive thesis demo views">
      <button type="button" role="tab" id="tab-session" aria-selected="true" focusgroupstart>1. Session 7.3 Live Replay &amp; UQ</button>
      <button type="button" role="tab" id="tab-stress">2. 20-bpm Skirt &amp; WGN Stress Test</button>
      <button type="button" role="tab" id="tab-ladder">3. Complexity Ladder &amp; Audited Tables</button>
    </div>

    <div id="panel-session" role="tabpanel" aria-labelledby="tab-session" tabindex="0">
      <div class="panel-inner">
        <div class="card">
          <div class="controls-bar">
            <button type="button" id="btn-play" class="btn btn-primary">Play Live</button>
            <button type="button" id="btn-step-back" class="btn" aria-label="Step backward 1 window">-2 s</button>
            <button type="button" id="btn-step-fwd" class="btn" aria-label="Step forward 1 window">+2 s</button>
            <label class="mono-val" for="speed-select">
              Speed
              <select id="speed-select" class="sel">
                <option value="1">1x (2s/tick)</option>
                <option value="5" selected>5x</option>
                <option value="15">15x</option>
                <option value="30">30x</option>
              </select>
            </label>
            <div class="scrubber-wrap">
              <label for="time-slider" class="mono-val" style="white-space:nowrap;">Time</label>
              <input id="time-slider" type="range" min="0" max="{payload["meta"]["n_win"] - 1}" step="1" value="180" />
              <span id="time-readout" class="mono-val" style="min-width:128px;text-align:right;">t = 0.0 min</span>
            </div>
            <label class="mono-val" for="bg-select">
              Heatmap
              <select id="bg-select" class="sel">
                <option value="posterior" selected>Causal Posterior α_t(r)</option>
                <option value="salience">Emission Salience e_t(r)</option>
                <option value="none">Clean Background</option>
              </select>
            </label>
          </div>
          <div class="toggle-list" role="group" aria-label="Model trajectory toggles">
            <label class="chk-label" for="chk-ref"><input type="checkbox" id="chk-ref" class="visually-hidden" checked /><span><i class="swatch" style="background:currentColor;"></i> 1 Hz Monitor Ref</span></label>
            <label class="chk-label" for="chk-r2e"><input type="checkbox" id="chk-r2e" class="visually-hidden" checked /><span><i class="swatch" style="background:#009E73;"></i> R2e FilterSafe-8 (8p)</span></label>
            <label class="chk-label" for="chk-c5"><input type="checkbox" id="chk-c5" class="visually-hidden" checked /><span><i class="swatch" style="background:#009E73;opacity:0.45;height:8px;"></i> 80% C5 Conformal Tube</span></label>
            <label class="chk-label" for="chk-r0"><input type="checkbox" id="chk-r0" class="visually-hidden" checked /><span><i class="swatch" style="background:#56B4E9;"></i> R0 HPSS+NLS (0p)</span></label>
            <label class="chk-label" for="chk-r3a"><input type="checkbox" id="chk-r3a" class="visually-hidden" checked /><span><i class="swatch" style="background:#CC79A7;"></i> R3a Causal TCN (15.8k p)</span></label>
            <label class="chk-label" for="chk-r1a"><input type="checkbox" id="chk-r1a" class="visually-hidden" /><span><i class="swatch" style="background:#0072B2;"></i> R1a Fwd HMM (3p)</span></label>
            <label class="chk-label" for="chk-r3c"><input type="checkbox" id="chk-r3c" class="visually-hidden" /><span><i class="swatch" style="background:#E69F00;"></i> R3c Ridge (439p)</span></label>
            <label class="chk-label" for="chk-r3b"><input type="checkbox" id="chk-r3b" class="visually-hidden" /><span><i class="swatch" style="background:#D55E00;"></i> R3b MLP (14.6k p)</span></label>
            <label class="chk-label" for="chk-r1f"><input type="checkbox" id="chk-r1f" class="visually-hidden" /><span><i class="swatch" style="background:#6d28d9;"></i> R1f Offline FB HMM</span></label>
          </div>
        </div>

        <div class="grid-main">
          <div style="display:flex;flex-direction:column;gap:14px;">
            <div class="card">
              <div class="card-header">
                <h2 class="card-title">Tracking Trajectory (Held-Out LOPO Infant 7 &middot; 25.0 min, 736 Windows)</h2>
                <p class="card-note">Click or drag anywhere on the plot to jump cursor t &middot; Pink markers = clinical pauses &gt;5 s</p>
              </div>
              <div class="canvas-box">
                <canvas id="cv-traj" height="305" aria-label="Respiratory rate trajectory canvas"></canvas>
              </div>
            </div>

            <div class="card">
              <div class="card-header">
                <h2 class="card-title">Causal Uncertainty Timeline: U2 Posterior SD vs. U6b Barrier SQI</h2>
                <p class="card-note">Vertical cursor = live playback time &middot; Dashed line = selective abstention threshold for current coverage</p>
              </div>
              <div class="canvas-box">
                <canvas id="cv-uq" height="165" aria-label="Causal uncertainty timeline canvas"></canvas>
              </div>
            </div>
          </div>

          <div style="display:flex;flex-direction:column;gap:14px;">
            <div class="card">
              <div class="card-header">
                <h2 class="card-title">Instantaneous State Belief at Cursor t</h2>
                <p class="card-note" id="slice-subtitle">Comparing memoryless emission e_t(r) vs. causal posterior α_t(r)</p>
              </div>
              <div class="canvas-box">
                <canvas id="cv-slice" height="205" aria-label="Spectral emission and posterior distribution slice"></canvas>
              </div>
              <div class="kpi-grid" style="margin-top:10px;">
                <div class="kpi"><span class="kpi-k">1 Hz Ref / Pause</span><span id="kpi-ref" class="kpi-v">--</span></div>
                <div class="kpi"><span class="kpi-k">R2e FilterSafe-8 &plusmn; C5</span><span id="kpi-r2e" class="kpi-v">--</span></div>
                <div class="kpi"><span class="kpi-k">R0 (0p) vs. R3a TCN (16k)</span><span id="kpi-foils" class="kpi-v">--</span></div>
                <div class="kpi"><span class="kpi-k">U2 SD / U6b Barrier</span><span id="kpi-uq" class="kpi-v">--</span></div>
              </div>
            </div>

            <div class="card">
              <div class="card-header">
                <h2 class="card-title">Interactive Selective Abstention Gate</h2>
                <label class="mono-val" for="proxy-select">
                  Proxy
                  <select id="proxy-select" class="sel">
                    <option value="u6b" selected>U6b Zero-Tuned Barrier</option>
                    <option value="u2">U2 Posterior SD (0p)</option>
                  </select>
                </label>
              </div>
              <div style="display:flex;align-items:center;gap:10px;">
                <label for="cov-slider" class="mono-val" style="white-space:nowrap;">Coverage</label>
                <input id="cov-slider" type="range" min="50" max="100" step="1" value="80" />
                <span id="cov-readout" class="mono-val" style="min-width:46px;text-align:right;">80%</span>
              </div>
              <div class="kpi-grid" style="margin-top:6px;">
                <div class="kpi"><span class="kpi-k">Retained R2e 1 Hz RMSE</span><span id="kpi-ret-rmse" class="kpi-v">--</span></div>
                <div class="kpi"><span class="kpi-k">Delta vs. 100% Coverage</span><span id="kpi-ret-delta" class="kpi-v">--</span></div>
                <div class="kpi"><span class="kpi-k">80% C5 Empirical Cov.</span><span id="kpi-ret-c5" class="kpi-v">--</span></div>
                <div class="kpi"><span class="kpi-k">Retained Clinical Pauses</span><span id="kpi-ret-pause" class="kpi-v">--</span></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div id="panel-stress" role="tabpanel" aria-labelledby="tab-stress" hidden="until-found">
      <div class="panel-inner">
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">Live Frozen Stage 4 Stress Test: Why Physics Constraints Prevent Out-of-Band &amp; Dropout Hallucination</h2>
            <p class="card-note">Computed live from SHA-256 verified <code>frozen_stage4_models.npz</code> (<code>af33a83e157f7d2e</code>) on synthetic 24 GHz CW I/Q waveforms</p>
          </div>
          <div class="grid-two">
            <div>
              <h3 style="margin:0 0 6px;font-size:0.88rem;">A. Frequency Sweep Across 28 bpm High-Pass Cutoff (14–72 bpm)</h3>
              <div class="canvas-box"><canvas id="cv-sweep" height="295" aria-label="Frequency sweep invariance plot"></canvas></div>
            </div>
            <div>
              <h3 style="margin:0 0 6px;font-size:0.88rem;">B. 20 bpm Bradypnea &rarr; Pure WGN Dropout &rarr; 20 bpm Recovery</h3>
              <div class="canvas-box"><canvas id="cv-dropout" height="295" aria-label="Bradypnea to WGN dropout stress replay"></canvas></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div id="panel-ladder" role="tabpanel" aria-labelledby="tab-ladder" hidden="until-found">
      <div class="panel-inner">
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">Audited Complexity Ladder: Anchor Session 7.3 vs. Full NICU Cohort (K=11, S=28) &amp; External 24.2 GHz L4</h2>
            <p class="card-note">Cohort &amp; external values loaded verbatim from <code>metrics_stage2_complexity.json</code> and <code>metrics_stage4_external.json</code></p>
          </div>
          <div tabindex="0" role="region" aria-labelledby="ladder-caption" class="table-wrapper">
            <table>
              <caption id="ladder-caption">Held-out NICU Session 7.3 vs. full cohort &amp; zero-touch external hardware RMSE (bpm)</caption>
              <thead>
                <tr>
                  <th>Model Family</th>
                  <th>Params</th>
                  <th>Lookahead (Blk / STFT)</th>
                  <th>Session 7.3 LOPO 1 Hz</th>
                  <th>NICU 1 Hz (S=28)</th>
                  <th>p_Holm vs R0</th>
                  <th>NICU Trig-10s</th>
                  <th>Ext >=28 bpm</th>
                  <th>Ext Slope</th>
                  <th>Ext 20-bpm Skirt</th>
                </tr>
              </thead>
              <tbody id="ladder-tbody"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<script id="demo-data" type="application/json">{data_json}</script>
<script>
(() => {{
  "use strict";
  const DATA = JSON.parse(document.getElementById("demo-data").textContent);
  const S73 = DATA.s73;
  const N_WIN = DATA.meta.n_win;
  const GRID_MIN = DATA.meta.grid_min;
  const GRID_LEN = DATA.meta.grid_len;

  function decodeU8Matrix(b64) {{
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }}
  const SAL_U8 = decodeU8Matrix(S73.sal_u8_b64);
  const POST_U8 = decodeU8Matrix(S73.post_u8_b64);

  const STATE = {{
    tIdx: Math.min(240, N_WIN - 1),
    playing: false,
    speed: 5,
    lastTick: 0,
    bgMode: "posterior",
    proxy: "u6b",
    coveragePct: 80,
    dragging: false,
    toggles: {{
      ref: true, r2e: true, c5: true, r0: true, r3a: true,
      r1a: false, r3c: false, r3b: false, r1f: false,
    }},
  }};

  const COLORS = {{
    ref: "fg",
    r2e: "#009E73",
    r0: "#56B4E9",
    r3a: "#CC79A7",
    r1a: "#0072B2",
    r3c: "#E69F00",
    r3b: "#D55E00",
    r1f: "#7C3AED",
  }};

  function getThemeColors() {{
    const cs = getComputedStyle(document.documentElement);
    return {{
      bg: cs.getPropertyValue("--card").trim() || "#ffffff",
      surface2: cs.getPropertyValue("--surface2").trim() || "#edf1f7",
      border: cs.getPropertyValue("--border").trim() || "#d3dbe6",
      fg: cs.getPropertyValue("--fg").trim() || "#121926",
      muted: cs.getPropertyValue("--muted").trim() || "#4a576d",
      brand: cs.getPropertyValue("--brand-color").trim() || "#006c4f",
      accent: cs.getPropertyValue("--accent").trim() || "#005896",
      warn: cs.getPropertyValue("--warn").trim() || "#b54500",
      danger: cs.getPropertyValue("--danger").trim() || "#a61d56",
    }};
  }}

  function setupCanvas(cv, cssH) {{
    const dpr = Math.max(window.devicePixelRatio || 1, 1);
    const rect = cv.getBoundingClientRect();
    const cssW = Math.max(rect.width || cv.parentElement.clientWidth || 640, 280);
    cv.style.height = cssH + "px";
    const targetW = Math.round(cssW * dpr);
    const targetH = Math.round(cssH * dpr);
    if (cv.width !== targetW || cv.height !== targetH) {{
      cv.width = targetW;
      cv.height = targetH;
    }}
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return {{ ctx, w: cssW, h: cssH }};
  }}

  function syncRangeVars(slider) {{
    slider.style.setProperty("--attr-min", slider.min || 0);
    slider.style.setProperty("--attr-max", slider.max || 100);
    slider.style.setProperty("--control-value", slider.value);
  }}

  function computeAbstentionState() {{
    const arr = STATE.proxy === "u6b" ? S73.u6b : S73.u2;
    const validIdx = [];
    for (let i = 0; i < N_WIN; i++) {{
      if (S73.valid[i]) validIdx.push(i);
    }}
    const sorted = validIdx.map((i) => arr[i]).sort((a, b) => a - b);
    const frac = STATE.coveragePct / 100.0;
    const thrIdx = Math.min(sorted.length - 1, Math.max(0, Math.floor(frac * (sorted.length - 1))));
    const thr = sorted[thrIdx];

    let seAll = 0, nAll = 0, seRet = 0, nRet = 0, covHits = 0;
    let pausesAll = 0, pausesRet = 0;
    const kept = new Uint8Array(N_WIN);
    for (let i = 0; i < N_WIN; i++) {{
      const isKept = STATE.coveragePct >= 100 ? true : arr[i] <= thr;
      if (isKept) kept[i] = 1;
      if (!S73.valid[i]) continue;
      const err = S73.pred_R2e[i] - S73.ref_1hz[i];
      seAll += err * err;
      nAll++;
      if (S73.pause[i]) pausesAll++;
      if (isKept) {{
        seRet += err * err;
        nRet++;
        if (Math.abs(err) <= S73.c5_half_80[i]) covHits++;
        if (S73.pause[i]) pausesRet++;
      }}
    }}
    const rmseAll = Math.sqrt(seAll / Math.max(1, nAll));
    const rmseRet = Math.sqrt(seRet / Math.max(1, nRet));
    return {{
      thr,
      kept,
      rmseAll,
      rmseRet,
      delta: rmseRet - rmseAll,
      c5Cov: covHits / Math.max(1, nRet),
      pausesRet,
      pausesAll,
      nRet,
      nAll,
    }};
  }}

  function drawTrajectory(ab) {{
    const cv = document.getElementById("cv-traj");
    const {{ ctx, w, h }} = setupCanvas(cv, 305);
    const tc = getThemeColors();
    const pad = {{ l: 46, r: 16, t: 16, b: 30 }};
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;
    const yMin = 18, yMax = 90;
    const xOf = (i) => pad.l + (i / (N_WIN - 1)) * pw;
    const yOf = (y) => pad.t + ((yMax - Math.min(yMax, Math.max(yMin, y))) / (yMax - yMin)) * ph;

    ctx.fillStyle = tc.bg;
    ctx.fillRect(0, 0, w, h);

    // Optional background heatmap (e_t or alpha_t)
    if (STATE.bgMode !== "none") {{
      const mat = STATE.bgMode === "posterior" ? POST_U8 : SAL_U8;
      const cols = Math.min(N_WIN, 240);
      const cellW = pw / cols;
      const cellH = ph / GRID_LEN;
      for (let c = 0; c < cols; c++) {{
        const ti = Math.round((c / (cols - 1)) * (N_WIN - 1));
        const x = pad.l + c * cellW;
        for (let b = 0; b < GRID_LEN; b++) {{
          const v = mat[ti * GRID_LEN + b] / 255.0;
          if (v < 0.08) continue;
          ctx.fillStyle = STATE.bgMode === "posterior"
            ? `rgba(0, 158, 115, ${{(v * 0.42).toFixed(3)}})`
            : `rgba(86, 180, 233, ${{(v * 0.36).toFixed(3)}})`;
          ctx.fillRect(x, pad.t + ((GRID_LEN - 1 - b) / GRID_LEN) * ph, cellW + 0.6, cellH + 0.6);
        }}
      }}
    }}

    // Shade invalid/abstained regions + highlight >5s pause ticks
    for (let i = 0; i < N_WIN; i++) {{
      if (S73.valid[i] && !ab.kept[i]) {{
        ctx.fillStyle = "rgba(213, 94, 0, 0.11)";
        ctx.fillRect(xOf(i) - 1, pad.t, pw / N_WIN + 1.5, ph);
      }}
      if (S73.pause[i] && S73.valid[i]) {{
        ctx.fillStyle = tc.danger;
        ctx.fillRect(xOf(i) - 1, pad.t, Math.max(2, pw / N_WIN), 5);
      }}
    }}

    // Grid lines
    ctx.strokeStyle = tc.border;
    ctx.lineWidth = 1;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = tc.muted;
    ctx.textAlign = "right";
    for (let br = 20; br <= 90; br += 10) {{
      const yy = yOf(br);
      ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + pw, yy); ctx.stroke();
      ctx.fillText(String(br), pad.l - 6, yy + 4);
    }}
    ctx.textAlign = "center";
    for (let m = 0; m <= 25; m += 5) {{
      const xx = pad.l + (m / S73.t_min[N_WIN - 1]) * pw;
      ctx.fillText(m + "m", xx, h - 8);
    }}
    ctx.save();
    ctx.translate(14, pad.t + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("Rate (bpm)", 0, 0);
    ctx.restore();

    // 80% C5 Conformal tube around R2e
    if (STATE.toggles.c5 && STATE.toggles.r2e) {{
      ctx.beginPath();
      for (let i = 0; i < N_WIN; i++) {{
        const x = xOf(i), y = yOf(S73.pred_R2e[i] + S73.c5_half_80[i]);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }}
      for (let i = N_WIN - 1; i >= 0; i--) {{
        ctx.lineTo(xOf(i), yOf(S73.pred_R2e[i] - S73.c5_half_80[i]));
      }}
      ctx.closePath();
      ctx.fillStyle = "rgba(0, 158, 115, 0.22)";
      ctx.fill();
    }}

    const drawSeries = (arr, color, lw, alpha = 1.0, dash = []) => {{
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.globalAlpha = alpha;
      ctx.setLineDash(dash);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < N_WIN; i++) {{
        if (!Number.isFinite(arr[i])) {{ started = false; continue; }}
        const x = xOf(i), y = yOf(arr[i]);
        if (!started) {{ ctx.moveTo(x, y); started = true; }} else ctx.lineTo(x, y);
      }}
      ctx.stroke();
      ctx.restore();
    }};

    if (STATE.toggles.r0) drawSeries(S73.pred_R0, COLORS.r0, 1.15, 0.78);
    if (STATE.toggles.r3b) drawSeries(S73.pred_R3b, COLORS.r3b, 1.2, 0.78);
    if (STATE.toggles.r3c) drawSeries(S73.pred_R3c, COLORS.r3c, 1.25, 0.85);
    if (STATE.toggles.r3a) drawSeries(S73.pred_R3a, COLORS.r3a, 1.35, 0.85);
    if (STATE.toggles.r1a) drawSeries(S73.pred_R1a, COLORS.r1a, 1.35, 0.88);
    if (STATE.toggles.r1f) drawSeries(S73.pred_R1f, COLORS.r1f, 1.35, 0.88, [4, 3]);
    if (STATE.toggles.ref) drawSeries(S73.ref_1hz, tc.fg, 1.75, 0.92);
    if (STATE.toggles.r2e) drawSeries(S73.pred_R2e, COLORS.r2e, 2.0, 1.0);

    // Live playback cursor
    const cx = xOf(STATE.tIdx);
    ctx.strokeStyle = tc.accent;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(cx, pad.t); ctx.lineTo(cx, pad.t + ph); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COLORS.r2e;
    ctx.beginPath(); ctx.arc(cx, yOf(S73.pred_R2e[STATE.tIdx]), 4.5, 0, Math.PI * 2); ctx.fill();
  }}

  function drawUncertaintyTimeline(ab) {{
    const cv = document.getElementById("cv-uq");
    const {{ ctx, w, h }} = setupCanvas(cv, 165);
    const tc = getThemeColors();
    const pad = {{ l: 46, r: 16, t: 14, b: 26 }};
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;
    const xOf = (i) => pad.l + (i / (N_WIN - 1)) * pw;
    const yOf = (u) => pad.t + (1.0 - Math.min(1.0, Math.max(0.0, u))) * ph;

    ctx.fillStyle = tc.bg;
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = tc.border;
    ctx.lineWidth = 1;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = tc.muted;
    ctx.textAlign = "right";
    for (const v of [0.0, 0.25, 0.5, 0.75, 1.0]) {{
      const yy = yOf(v);
      ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + pw, yy); ctx.stroke();
      ctx.fillText(v.toFixed(2), pad.l - 6, yy + 4);
    }}

    // Abstention highlight
    for (let i = 0; i < N_WIN; i++) {{
      if (S73.valid[i] && !ab.kept[i]) {{
        ctx.fillStyle = "rgba(213, 94, 0, 0.14)";
        ctx.fillRect(xOf(i) - 1, pad.t, pw / N_WIN + 1.5, ph);
      }}
    }}

    const plotCurve = (arr, color, lw, dash = []) => {{
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.setLineDash(dash);
      ctx.beginPath();
      for (let i = 0; i < N_WIN; i++) {{
        const x = xOf(i), y = yOf(arr[i]);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }}
      ctx.stroke();
      ctx.restore();
    }};
    plotCurve(S73.u2, COLORS.r1a, 1.4);
    plotCurve(S73.u6b, COLORS.r3b, 1.5);

    if (STATE.coveragePct < 100) {{
      const yt = yOf(ab.thr);
      ctx.save();
      ctx.strokeStyle = tc.fg;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(pad.l, yt); ctx.lineTo(pad.l + pw, yt); ctx.stroke();
      ctx.restore();
    }}

    const cx = xOf(STATE.tIdx);
    ctx.strokeStyle = tc.accent;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(cx, pad.t); ctx.lineTo(cx, pad.t + ph); ctx.stroke();
    ctx.setLineDash([]);
  }}

  function drawSlice() {{
    const cv = document.getElementById("cv-slice");
    const {{ ctx, w, h }} = setupCanvas(cv, 205);
    const tc = getThemeColors();
    const pad = {{ l: 38, r: 12, t: 18, b: 28 }};
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;
    const ti = STATE.tIdx;

    ctx.fillStyle = tc.bg;
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = tc.border;
    ctx.lineWidth = 1;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = tc.muted;
    ctx.textAlign = "right";
    for (const v of [0, 0.5, 1.0]) {{
      const yy = pad.t + (1 - v) * ph;
      ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + pw, yy); ctx.stroke();
      ctx.fillText(v.toFixed(1), pad.l - 5, yy + 4);
    }}
    ctx.textAlign = "center";
    for (let br = 20; br <= 90; br += 10) {{
      const xx = pad.l + ((br - GRID_MIN) / (GRID_LEN - 1)) * pw;
      ctx.fillText(String(br), xx, h - 8);
    }}

    const xGrid = (b) => pad.l + (b / (GRID_LEN - 1)) * pw;
    const yNorm = (v) => pad.t + (1 - v) * ph;

    // Emission bars (light blue)
    ctx.fillStyle = "rgba(86, 180, 233, 0.45)";
    const bw = Math.max(1.5, pw / GRID_LEN - 0.5);
    for (let b = 0; b < GRID_LEN; b++) {{
      const v = SAL_U8[ti * GRID_LEN + b] / 255.0;
      const y = yNorm(v);
      ctx.fillRect(xGrid(b) - bw / 2, y, bw, pad.t + ph - y);
    }}

    // Causal posterior curve
    ctx.strokeStyle = COLORS.r2e;
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    for (let b = 0; b < GRID_LEN; b++) {{
      const v = POST_U8[ti * GRID_LEN + b] / 255.0;
      const x = xGrid(b), y = yNorm(v);
      b === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }}
    ctx.stroke();

    // Vertical markers for Ref, R0, R2e
    const drawMarker = (val, color, label, dy) => {{
      if (!Number.isFinite(val)) return;
      const xx = pad.l + ((Math.min(90, Math.max(18, val)) - GRID_MIN) / (GRID_LEN - 1)) * pw;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(xx, pad.t); ctx.lineTo(xx, pad.t + ph); ctx.stroke();
      ctx.fillStyle = color;
      ctx.textAlign = xx > w - 45 ? "right" : "left";
      ctx.fillText(label, xx + (xx > w - 45 ? -4 : 4), pad.t + dy);
    }};
    drawMarker(S73.pred_R0[ti], COLORS.r0, "R0", 12);
    drawMarker(S73.ref_1hz[ti], tc.fg, "Ref", 25);
    drawMarker(S73.pred_R2e[ti], COLORS.r2e, "R2e", 38);
  }}

  function drawStressTabs() {{
    const cv1 = document.getElementById("cv-sweep");
    const cv2 = document.getElementById("cv-dropout");
    if (!cv1 || !cv2) return;
    const tc = getThemeColors();

    // A. Sweep plot
    {{
      const {{ ctx, w, h }} = setupCanvas(cv1, 295);
      const sw = DATA.stress.sweep;
      const pad = {{ l: 46, r: 16, t: 22, b: 34 }};
      const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
      const xOf = (r) => pad.l + ((r - 14) / (72 - 14)) * pw;
      const yOf = (y) => pad.t + ((85 - Math.min(85, Math.max(10, y))) / (85 - 10)) * ph;

      ctx.fillStyle = tc.bg; ctx.fillRect(0, 0, w, h);
      // 28 bpm cutoff shading (<28 bpm skirt)
      ctx.fillStyle = "rgba(213, 94, 0, 0.09)";
      ctx.fillRect(pad.l, pad.t, xOf(28) - pad.l, ph);
      ctx.strokeStyle = tc.warn; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(xOf(28), pad.t); ctx.lineTo(xOf(28), pad.t + ph); ctx.stroke();
      ctx.setLineDash([]);

      ctx.strokeStyle = tc.border; ctx.lineWidth = 1;
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillStyle = tc.muted; ctx.textAlign = "right";
      for (const y of [15, 30, 45, 60, 75]) {{
        const yy = yOf(y); ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + pw, yy); ctx.stroke();
        ctx.fillText(String(y), pad.l - 6, yy + 4);
      }}
      ctx.textAlign = "center";
      for (const r of [14, 20, 28, 40, 56, 72]) {{
        ctx.fillText(String(r), xOf(r), h - 10);
      }}
      ctx.fillStyle = tc.warn;
      ctx.fillText("<28 bpm Skirt", (pad.l + xOf(28)) / 2, pad.t + 14);

      // Identity line
      ctx.save(); ctx.strokeStyle = tc.fg; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(xOf(14), yOf(14)); ctx.lineTo(xOf(72), yOf(72)); ctx.stroke(); ctx.restore();

      for (const [k, color, lw] of [["R3c", COLORS.r3c, 1.5], ["R3b", COLORS.r3b, 1.6], ["R3a", COLORS.r3a, 1.8], ["R0", COLORS.r0, 1.5], ["R2e", COLORS.r2e, 2.3]]) {{
        ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath();
        sw.rates_bpm.forEach((r, idx) => {{
          const x = xOf(r), y = yOf(sw[k][idx]);
          idx === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }});
        ctx.stroke();
      }}
      // Compact inline legend
      let lx = pad.l + 8;
      for (const [lbl, c] of [["Identity", tc.fg], ["R2e (8p)", COLORS.r2e], ["R3a TCN", COLORS.r3a], ["R3b MLP", COLORS.r3b], ["R3c Ridge", COLORS.r3c]]) {{
        ctx.fillStyle = c; ctx.fillRect(lx, h - 29, 10, 3);
        ctx.fillStyle = tc.muted; ctx.textAlign = "left"; ctx.fillText(lbl, lx + 13, h - 24);
        lx += ctx.measureText(lbl).width + 24;
      }}
    }}

    // B. 20 bpm -> WGN -> 20 bpm Dropout plot
    {{
      const {{ ctx, w, h }} = setupCanvas(cv2, 295);
      const rp = DATA.stress.dropout_replay;
      const n = rp.n_win;
      const pad = {{ l: 46, r: 42, t: 22, b: 34 }};
      const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
      const xOf = (i) => pad.l + (i / (n - 1)) * pw;
      const yOf = (y) => pad.t + ((85 - Math.min(85, Math.max(10, y))) / (85 - 10)) * ph;
      const yU = (u) => pad.t + (1 - u) * ph;

      ctx.fillStyle = tc.bg; ctx.fillRect(0, 0, w, h);
      // Shade WGN windows 12..23
      ctx.fillStyle = "rgba(166, 29, 86, 0.12)";
      ctx.fillRect(xOf(11.5), pad.t, xOf(23.5) - xOf(11.5), ph);

      ctx.strokeStyle = tc.border; ctx.lineWidth = 1;
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillStyle = tc.muted; ctx.textAlign = "right";
      for (const y of [20, 40, 60, 80]) {{
        const yy = yOf(y); ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + pw, yy); ctx.stroke();
        ctx.fillText(String(y), pad.l - 6, yy + 4);
      }}
      ctx.textAlign = "left";
      for (const u of [0.0, 0.5, 1.0]) {{
        ctx.fillText(u.toFixed(1), pad.l + pw + 6, yU(u) + 4);
      }}
      ctx.textAlign = "center";
      ctx.fillStyle = tc.danger;
      ctx.fillText("Pure WGN Dropout", (xOf(12) + xOf(23)) / 2, pad.t + 14);
      ctx.fillStyle = tc.muted;
      ctx.fillText("20 bpm Skirt", xOf(5.5), h - 10);
      ctx.fillText("Recovery (20 bpm)", xOf(29.5), h - 10);

      // Reference 20 bpm segments
      ctx.save(); ctx.strokeStyle = tc.fg; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(xOf(0), yOf(20)); ctx.lineTo(xOf(11), yOf(20));
      ctx.moveTo(xOf(24), yOf(20)); ctx.lineTo(xOf(35), yOf(20)); ctx.stroke(); ctx.restore();

      for (const [k, color, lw, dash] of [["R3a", COLORS.r3a, 1.7, []], ["R3b", COLORS.r3b, 1.5, []], ["R2e", COLORS.r2e, 2.3, []]]) {{
        ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash);
        ctx.beginPath();
        for (let i = 0; i < n; i++) {{
          const x = xOf(i), y = yOf(rp[k][i]);
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }}
        ctx.stroke(); ctx.restore();
      }}
      // U6b dashed overlay (right axis [0,1])
      ctx.save(); ctx.strokeStyle = tc.warn; ctx.lineWidth = 1.7; ctx.setLineDash([5, 3]);
      ctx.beginPath();
      for (let i = 0; i < n; i++) {{
        const x = xOf(i), y = yU(rp.U6b[i]);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }}
      ctx.stroke(); ctx.restore();

      let lx = pad.l + 8;
      for (const [lbl, c] of [["R2e (bpm)", COLORS.r2e], ["R3a TCN", COLORS.r3a], ["R3b MLP", COLORS.r3b], ["U6b SQI (right)", tc.warn]]) {{
        ctx.fillStyle = c; ctx.fillRect(lx, pad.t + 4, 10, 3);
        ctx.fillStyle = tc.muted; ctx.textAlign = "left"; ctx.fillText(lbl, lx + 13, pad.t + 9);
        lx += ctx.measureText(lbl).width + 22;
      }}
    }}
  }}

  function populateLadderTable() {{
    const tb = document.getElementById("ladder-tbody");
    tb.innerHTML = DATA.ladder.map((r) => `
      <tr class="${{r.is_ours ? "row-ours" : ""}}">
        <th>${{r.label}}</th>
        <td>${{r.params}}</td>
        <td>${{r.lookahead}}</td>
        <td>${{r.rmse_73}}</td>
        <td>${{r.nicu_1hz}}</td>
        <td>${{r.p_holm}}</td>
        <td>${{r.nicu_trig}}</td>
        <td>${{r.ext_inband}}</td>
        <td>${{r.ext_slope}}</td>
        <td>${{r.ext_skirt}}</td>
      </tr>
    `).join("");
  }}

  function updateKPIs(ab) {{
    const i = STATE.tIdx;
    const tStr = S73.t_min[i].toFixed(2);
    document.getElementById("time-readout").textContent = `t = ${{tStr}} m (${{i + 1}}/${{N_WIN}})`;
    const ts = document.getElementById("time-slider");
    ts.value = String(i);
    ts.setAttribute("aria-valuetext", `${{tStr}} minutes`);
    syncRangeVars(ts);

    const cs = document.getElementById("cov-slider");
    document.getElementById("cov-readout").textContent = `${{STATE.coveragePct}}%`;
    cs.setAttribute("aria-valuetext", `${{STATE.coveragePct}} percent retained`);
    syncRangeVars(cs);

    const isP = S73.pause[i] ? " | PAUSE >5s" : "";
    const isV = S73.valid[i] ? "" : " | Invalid";
    document.getElementById("kpi-ref").textContent = `${{S73.ref_1hz[i].toFixed(1)}} bpm${{isP}}${{isV}}`;
    document.getElementById("kpi-r2e").textContent = `${{S73.pred_R2e[i].toFixed(1)}} \u00B1 ${{S73.c5_half_80[i].toFixed(1)}} bpm`;
    document.getElementById("kpi-foils").textContent = `${{S73.pred_R0[i].toFixed(1)}} / ${{S73.pred_R3a[i].toFixed(1)}} bpm`;
    document.getElementById("kpi-uq").textContent = `${{S73.u2[i].toFixed(3)}} / ${{S73.u6b[i].toFixed(3)}}`;

    document.getElementById("kpi-ret-rmse").textContent = `${{ab.rmseRet.toFixed(3)}} bpm (${{ab.nRet}}/${{ab.nAll}})`;
    const sign = ab.delta <= 0 ? "" : "+";
    document.getElementById("kpi-ret-delta").textContent = `${{sign}}${{ab.delta.toFixed(3)}} bpm`;
    document.getElementById("kpi-ret-c5").textContent = `${{(ab.c5Cov * 100).toFixed(1)}}% (target 80%)`;
    document.getElementById("kpi-ret-pause").textContent = `${{ab.pausesRet}} / ${{ab.pausesAll}} (${{((ab.pausesRet / Math.max(1, ab.pausesAll)) * 100).toFixed(0)}}%)`;
  }}

  function redrawAll() {{
    const ab = computeAbstentionState();
    updateKPIs(ab);
    drawTrajectory(ab);
    drawUncertaintyTimeline(ab);
    drawSlice();
    drawStressTabs();
  }}

  // Playback loop
  function tick(now) {{
    if (STATE.playing) {{
      const intervalMs = 1000.0 / STATE.speed;
      if (now - STATE.lastTick >= intervalMs) {{
        STATE.lastTick = now;
        STATE.tIdx = (STATE.tIdx + 1) % N_WIN;
        redrawAll();
      }}
      requestAnimationFrame(tick);
    }}
  }}

  // Wire controls
  const btnPlay = document.getElementById("btn-play");
  btnPlay.addEventListener("click", () => {{
    STATE.playing = !STATE.playing;
    btnPlay.textContent = STATE.playing ? "Pause" : "Play Live";
    if (STATE.playing) {{
      STATE.lastTick = performance.now();
      requestAnimationFrame(tick);
    }}
  }});
  document.getElementById("btn-step-back").addEventListener("click", () => {{
    STATE.tIdx = Math.max(0, STATE.tIdx - 1); redrawAll();
  }});
  document.getElementById("btn-step-fwd").addEventListener("click", () => {{
    STATE.tIdx = Math.min(N_WIN - 1, STATE.tIdx + 1); redrawAll();
  }});
  document.getElementById("speed-select").addEventListener("change", (e) => {{
    STATE.speed = Number(e.target.value);
  }});
  document.getElementById("bg-select").addEventListener("change", (e) => {{
    STATE.bgMode = e.target.value; redrawAll();
  }});
  document.getElementById("proxy-select").addEventListener("change", (e) => {{
    STATE.proxy = e.target.value; redrawAll();
  }});
  document.getElementById("time-slider").addEventListener("input", (e) => {{
    STATE.tIdx = Number(e.target.value); redrawAll();
  }});
  document.getElementById("cov-slider").addEventListener("input", (e) => {{
    STATE.coveragePct = Number(e.target.value); redrawAll();
  }});
  document.getElementById("theme-select").addEventListener("change", (e) => {{
    document.documentElement.style.colorScheme = e.target.value;
    requestAnimationFrame(redrawAll);
  }});

  for (const k of Object.keys(STATE.toggles)) {{
    const el = document.getElementById("chk-" + k);
    el.addEventListener("change", () => {{ STATE.toggles[k] = el.checked; redrawAll(); }});
  }}

  // Click / drag scrubbing on trajectory & UQ canvases
  function attachScrub(cv) {{
    const jump = (ev) => {{
      const rect = cv.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left - 46) / Math.max(1, rect.width - 62)));
      STATE.tIdx = Math.round(frac * (N_WIN - 1));
      redrawAll();
    }};
    cv.addEventListener("pointerdown", (ev) => {{ STATE.dragging = true; cv.setPointerCapture(ev.pointerId); jump(ev); }});
    cv.addEventListener("pointermove", (ev) => {{ if (STATE.dragging) jump(ev); }});
    cv.addEventListener("pointerup", () => {{ STATE.dragging = false; }});
  }}
  attachScrub(document.getElementById("cv-traj"));
  attachScrub(document.getElementById("cv-uq"));

  // Tabs logic with find-in-page beforematch & fallback keyboard navigation
  const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
  const panels = Array.from(document.querySelectorAll('[role="tabpanel"]'));
  function activateTab(targetTab, focusTab = false) {{
    for (const tab of tabs) {{
      const sel = tab === targetTab;
      tab.ariaSelected = sel ? "true" : "false";
      tab.toggleAttribute("focusgroupstart", sel);
      if (!Object.hasOwn(HTMLElement.prototype, "focusGroup")) {{
        tab.tabIndex = sel ? 0 : -1;
      }}
    }}
    for (const panel of panels) {{
      const show = panel.getAttribute("aria-labelledby") === targetTab.id;
      if (show) {{
        panel.removeAttribute("hidden");
        panel.tabIndex = 0;
      }} else {{
        panel.setAttribute("hidden", "until-found");
        panel.removeAttribute("tabindex");
      }}
    }}
    targetTab.scrollIntoView({{ block: "nearest", inline: "nearest" }});
    if (focusTab) targetTab.focus();
    requestAnimationFrame(redrawAll);
  }}
  tabs.forEach((tab) => {{
    tab.addEventListener("click", () => activateTab(tab));
    tab.addEventListener("focus", () => {{ if (tab.ariaSelected !== "true") activateTab(tab); }});
  }});
  panels.forEach((panel) => {{
    panel.addEventListener("beforematch", () => {{
      const owner = tabs.find((t) => t.id === panel.getAttribute("aria-labelledby"));
      if (owner) activateTab(owner);
    }});
  }});
  if (!Object.hasOwn(HTMLElement.prototype, "focusGroup")) {{
    tabs.forEach((tab, idx) => {{
      tab.tabIndex = idx === 0 ? 0 : -1;
      tab.addEventListener("keydown", (e) => {{
        let next = -1;
        if (e.key === "ArrowRight") next = (idx + 1) % tabs.length;
        else if (e.key === "ArrowLeft") next = (idx - 1 + tabs.length) % tabs.length;
        else if (e.key === "Home") next = 0;
        else if (e.key === "End") next = tabs.length - 1;
        if (next >= 0) {{
          e.preventDefault();
          activateTab(tabs[next], true);
        }}
      }});
    }});
  }}

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => requestAnimationFrame(redrawAll));
  window.addEventListener("resize", () => requestAnimationFrame(redrawAll));

  populateLadderTable();
  redrawAll();
}})();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline interactive Session 7.3 HTML demo.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output HTML path.")
    parser.add_argument(
        "--refresh-fixture",
        action="store_true",
        help="Refresh examples/session_7_3_lopo_fixture.npz from local Stage 2-3 archives.",
    )
    args = parser.parse_args()
    payload = build_payload(refresh_fixture=args.refresh_fixture)
    html = render_html_document(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024.0:.1f} KB, N={payload['meta']['n_win']} windows).")


if __name__ == "__main__":
    main()
