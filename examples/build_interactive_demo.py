#!/usr/bin/env python3
"""Build a zero-install, offline academic interactive HTML5 supplement for NICU Session 7.3.

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


def _to_json_list(arr: np.ndarray, ndigits: int) -> list[float | None]:
    """Convert numeric array to strict RFC-8259 JSON list (non-finite -> None/null)."""
    out: list[float | None] = []
    for raw in np.asarray(arr, dtype=np.float64).ravel():
        val = float(raw)
        out.append(round(val, ndigits) if math.isfinite(val) else None)
    return out


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
) -> tuple[str, str, list[float | None], dict[str, float]]:
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

    r2e_full = D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward", grid_bpm=grid)
    r2e_full.theta = np.asarray(zf["r2e_full_refit_theta"], dtype=np.float64).copy()

    c5_q80 = float(zf["c5_q_hat_norm_1hz"][2])
    c5_half = _to_json_list(c5_q80 * np.maximum(lopo_data["unc_sd_R2e_lopo"], 1.0), 3)
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
            val = float(np.mean(preds[mk][3:]))
            sweep[mk].append(round(val, 2) if math.isfinite(val) else None)
        for uk in ("U2", "U6b"):
            val = float(np.mean(uq[uk][3:]))
            sweep[uk].append(round(val, 3) if math.isfinite(val) else None)

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
        "gt_bpm": [20.0] * 12 + [None] * 12 + [20.0] * 12,
        "R0": _to_json_list(sp_preds["R0"], 2),
        "R2e": _to_json_list(sp_preds["R2e"], 2),
        "R3b": _to_json_list(sp_preds["R3b"], 2),
        "R3a": _to_json_list(sp_preds["R3a"], 2),
        "U2": _to_json_list(sp_uq["U2"], 3),
        "U6b": _to_json_list(sp_uq["U6b"], 3),
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
    eval_mask = valid & np.isfinite(ref_1hz)
    rows_spec = [
        ("R0 HPSS+NLS", "0", "0 / 3 s", "pred_R0", "R0_HPSS_NLS_Locked", False),
        ("R1a Forward HMM", "3", "0 / 3 s", "pred_R1a", "R1a_Online_Forward_MMSE_0s", False),
        ("R1f Offline FB HMM", "3", "inf / 3 s", "pred_R1f", "R1f_Offline_FB_MMSE_Inf", False),
        ("R2e FilterSafe-8 (Ours)", "8", "0 / 3 s", "pred_R2e_lopo", "R2e_Causal_DiffHMM_FilterSafe8", True),
        ("R3c Spectral Ridge", "439", "0 / 3 s", "pred_R3c", "R3c_Ridge_Spectral_439", False),
        ("R3b Spectral MLP", "14,569", "0 / 3 s", "pred_R3b", "R3b_Spectral_MLP_15k", False),
        ("R3a Causal TCN", "15,817", "0 / 3 s", "pred_R3a", "R3a_Causal_TCN_16k", False),
    ]
    rows: list[dict[str, Any]] = []
    for label, params, lookahead, lopo_key, mkey, is_ours in rows_spec:
        err73 = lopo_data[lopo_key][eval_mask] - ref_1hz[eval_mask]
        rmse73 = float(np.sqrt(np.mean(err73 ** 2)))
        assert math.isfinite(rmse73), f"Non-finite Session 7.3 RMSE for {lopo_key}"
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
    t_min = _to_json_list(
        (np.arange(n_win, dtype=np.float64) * float(C.UPDATE_S) + float(C.WIN_S) / 2.0) / 60.0,
        3,
    )
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
            "ref_1hz": _to_json_list(lopo["ref_1hz"], 3),
            "pred_R0": _to_json_list(lopo["pred_R0"], 3),
            "pred_R1a": _to_json_list(lopo["pred_R1a"], 3),
            "pred_R1f": _to_json_list(lopo["pred_R1f"], 3),
            "pred_R2e": _to_json_list(lopo["pred_R2e_lopo"], 3),
            "pred_R3c": _to_json_list(lopo["pred_R3c"], 3),
            "pred_R3b": _to_json_list(lopo["pred_R3b"], 3),
            "pred_R3a": _to_json_list(lopo["pred_R3a"], 3),
            "c5_half_80": c5_half,
            "u2": _to_json_list(lopo["u2_R2e_lopo"], 4),
            "u6b": _to_json_list(lopo["u6b_R2e_lopo"], 4),
            "u6c": _to_json_list(lopo["u6c_R2e_lopo"], 4),
            "sal_u8_b64": sal_b64,
            "post_u8_b64": post_b64,
        },
        "stress": stress,
        "ladder": ladder,
    }


def render_html_document(payload: dict[str, Any]) -> str:
    # Strict RFC-8259 serialization: allow_nan=False prevents bare NaN from breaking JSON.parse
    data_json = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light dark" />
<title>Companion Supplement — Causal FilterSafe-8 &amp; Uncertainty Quantification (Session 7.3)</title>
<style>
  :root {{
    color-scheme: light dark;
    --font-serif: "Palatino Linotype", Palatino, "Book Antiqua", Georgia, Cambria, "Times New Roman", serif;
    --font-sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --font-mono: ui-monospace, "SFMono-Regular", Menlo, Monaco, Consolas, monospace;

    --bg-light: #fafaf7;
    --bg-dark: #14171c;
    --card-light: #ffffff;
    --card-dark: #1b1f26;
    --surface2-light: #f2f4f7;
    --surface2-dark: #232831;
    --border-light: #cfd5df;
    --border-dark: #333b48;
    --rule-strong-light: #1f2937;
    --rule-strong-dark: #cbd5e1;
    --fg-light: #141821;
    --fg-dark: #e6e9ef;
    --muted-light: #4b5565;
    --muted-dark: #9ba5b5;
    --brand-light: #1b4d3e;
    --brand-dark: #3ca385;
    --accent-light: #1b365d;
    --accent-dark: #6ba4e8;
    --warn-light: #9a3412;
    --warn-dark: #e88a45;
    --danger-light: #881337;
    --danger-dark: #e06c96;

    --bg: var(--bg-light);
    --card: var(--card-light);
    --surface2: var(--surface2-light);
    --border: var(--border-light);
    --rule-strong: var(--rule-strong-light);
    --fg: var(--fg-light);
    --muted: var(--muted-light);
    --brand-color: var(--brand-light);
    --accent: var(--accent-light);
    --warn: var(--warn-light);
    --danger: var(--danger-light);

    --color-scrollbar-track: #eaedf2;
    --color-scrollbar-thumb: #6b7280;
    scrollbar-color: var(--color-scrollbar-thumb) var(--color-scrollbar-track);
    accent-color: var(--brand-color);
  }}

  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: var(--bg-dark);
      --card: var(--card-dark);
      --surface2: var(--surface2-dark);
      --border: var(--border-dark);
      --rule-strong: var(--rule-strong-dark);
      --fg: var(--fg-dark);
      --muted: var(--muted-dark);
      --brand-color: var(--brand-dark);
      --accent: var(--accent-dark);
      --warn: var(--warn-dark);
      --danger: var(--danger-dark);
      --color-scrollbar-track: #1b1f26;
      --color-scrollbar-thumb: #7b8596;
    }}
  }}

  @supports (color: light-dark(white, black)) {{
    :root:not([data-theme]) {{
      --bg: light-dark(var(--bg-light), var(--bg-dark));
      --card: light-dark(var(--card-light), var(--card-dark));
      --surface2: light-dark(var(--surface2-light), var(--surface2-dark));
      --border: light-dark(var(--border-light), var(--border-dark));
      --rule-strong: light-dark(var(--rule-strong-light), var(--rule-strong-dark));
      --fg: light-dark(var(--fg-light), var(--fg-dark));
      --muted: light-dark(var(--muted-light), var(--muted-dark));
      --brand-color: light-dark(var(--brand-light), var(--brand-dark));
      --accent: light-dark(var(--accent-light), var(--accent-dark));
      --warn: light-dark(var(--warn-light), var(--warn-dark));
      --danger: light-dark(var(--danger-light), var(--danger-dark));
      --color-scrollbar-track: light-dark(#eaedf2, #1b1f26);
      --color-scrollbar-thumb: light-dark(#6b7280, #7b8596);
    }}
  }}

  html[data-theme="light"] {{
    color-scheme: light;
    --bg: var(--bg-light);
    --card: var(--card-light);
    --surface2: var(--surface2-light);
    --border: var(--border-light);
    --rule-strong: var(--rule-strong-light);
    --fg: var(--fg-light);
    --muted: var(--muted-light);
    --brand-color: var(--brand-light);
    --accent: var(--accent-light);
    --warn: var(--warn-light);
    --danger: var(--danger-light);
    --color-scrollbar-track: #eaedf2;
    --color-scrollbar-thumb: #6b7280;
  }}

  html[data-theme="dark"] {{
    color-scheme: dark;
    --bg: var(--bg-dark);
    --card: var(--card-dark);
    --surface2: var(--surface2-dark);
    --border: var(--border-dark);
    --rule-strong: var(--rule-strong-dark);
    --fg: var(--fg-dark);
    --muted: var(--muted-dark);
    --brand-color: var(--brand-dark);
    --accent: var(--accent-dark);
    --warn: var(--warn-dark);
    --danger: var(--danger-dark);
    --color-scrollbar-track: #1b1f26;
    --color-scrollbar-thumb: #7b8596;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 0;
    font-family: var(--font-sans);
    background: var(--bg);
    color: var(--fg);
    line-height: 1.48;
    font-size: 13.5px;
  }}

  .monograph-shell {{
    max-width: 1320px;
    margin: 0 auto;
    padding: 18px 24px 32px;
  }}

  /* Monograph Masthead */
  .masthead {{
    border-top: 2.5px solid var(--rule-strong);
    border-bottom: 1.5px solid var(--rule-strong);
    padding: 12px 0 10px;
    margin-bottom: 14px;
  }}
  .rubric {{
    font-family: var(--font-sans);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0 0 4px;
  }}
  .masthead-title {{
    font-family: var(--font-serif);
    font-size: 1.42rem;
    font-weight: 700;
    line-height: 1.24;
    margin: 0 0 3px;
    color: var(--fg);
  }}
  .masthead-sub {{
    font-family: var(--font-serif);
    font-style: italic;
    font-size: 0.96rem;
    color: var(--muted);
    margin: 0 0 9px;
  }}
  .meta-bar {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 8px 16px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
    font-size: 0.8rem;
    color: var(--muted);
  }}
  .meta-items {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px 14px;
    font-variant-numeric: tabular-nums;
  }}
  .meta-items code {{
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--fg);
  }}

  /* Tabs guidance: single horizontal row, shape ::after indicator, no panel padding/bg */
  .tabs {{ width: 100%; }}
  .tablist {{
    display: flex;
    overflow-x: auto;
    border-bottom: 1px solid var(--border);
    margin-bottom: 14px;
    gap: 6px;
  }}
  [role="tab"] {{
    position: relative;
    flex: none;
    appearance: none;
    background: transparent;
    border: 0;
    padding: 9px 12px;
    font-family: var(--font-serif);
    font-size: 0.94rem;
    font-weight: 600;
    color: var(--muted);
    cursor: pointer;
    white-space: nowrap;
  }}
  [role="tab"]:hover {{ color: var(--fg); }}
  [role="tab"]:focus-visible {{
    outline: 2px solid var(--accent);
    outline-offset: -3px;
  }}
  [role="tab"][aria-selected="true"] {{ color: var(--accent); }}
  [role="tab"][aria-selected="true"]::after {{
    content: "";
    position: absolute;
    inset: auto 0 0;
    border-block-end: 2.5px solid var(--accent);
  }}

  .panel-inner {{
    display: flex;
    flex-direction: column;
    gap: 14px;
  }}

  .plate {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 13px 15px;
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
    border-radius: 3px;
    padding: 5px 11px;
    font: inherit;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    font-variant-numeric: tabular-nums;
  }}
  .btn:hover {{ border-color: var(--accent); }}
  .btn:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
  .btn-primary {{
    background: var(--accent);
    border-color: var(--accent);
    color: #ffffff;
  }}
  html[data-theme="dark"] .btn-primary {{ color: #0c1118; }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) .btn-primary {{ color: #0c1118; }}
  }}

  select.sel {{
    background: var(--surface2);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 4px 8px;
    font: inherit;
    font-size: 0.8rem;
  }}

  .scrubber-wrap {{
    flex: 1 1 260px;
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 220px;
  }}
  .mono-val {{
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: 0.8rem;
  }}

  /* Range Slider styling per slider guide (separate WebKit/Firefox blocks) */
  input[type="range"] {{
    --slider-track-size: 0.35rem;
    --slider-thumb-size: 1.05rem;
    --slider-track-radius: 2px;
    --slider-thumb-radius: 50%;
    --slider-track-color: var(--border);
    --slider-fill-color: var(--accent);
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
    border: 1.5px solid var(--card);
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
    border: 1.5px solid var(--card);
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
    gap: 7px 14px;
    padding-top: 9px;
    border-top: 1px solid var(--border);
    margin-top: 4px;
  }}
  .chk-label {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.79rem;
    cursor: pointer;
    user-select: none;
  }}
  .swatch {{
    width: 14px;
    height: 2.5px;
    display: inline-block;
    vertical-align: middle;
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
    .chk-label {{ position: relative; padding-left: 1.4rem; }}
    .chk-label input[type="checkbox"] + span::before {{
      content: "";
      position: absolute;
      left: 0;
      top: 50%;
      transform: translateY(-50%);
      width: 0.95rem;
      height: 0.95rem;
      border: 1.5px solid var(--border);
      background: var(--card);
      border-radius: 2px;
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
      left: 0.16rem;
      top: 50%;
      transform: translateY(-50%);
      color: #ffffff;
      font-weight: 700;
      font-size: 0.75rem;
    }}
  }}

  .grid-main {{
    display: grid;
    grid-template-columns: minmax(0, 2.15fr) minmax(320px, 1fr);
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

  .fig-title {{
    margin: 0 0 2px;
    font-family: var(--font-serif);
    font-size: 0.96rem;
    font-weight: 700;
  }}
  .fig-sub {{
    margin: 0 0 8px;
    font-size: 0.78rem;
    color: var(--muted);
  }}
  .fig-caption {{
    margin: 9px 0 0;
    font-family: var(--font-serif);
    font-size: 0.84rem;
    color: var(--muted);
    line-height: 1.42;
  }}
  .fig-caption strong {{ color: var(--fg); }}

  .canvas-box {{ width: 100%; position: relative; }}
  canvas {{ display: block; width: 100%; }}

  /* Booktabs quantitative readout tables */
  table.booktabs {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.81rem;
    font-variant-numeric: tabular-nums;
    margin-top: 8px;
  }}
  table.booktabs thead th {{
    border-top: 1.5px solid var(--rule-strong);
    border-bottom: 1px solid var(--rule-strong);
    padding: 5px 6px;
    text-align: left;
    font-weight: 700;
    font-size: 0.76rem;
    color: var(--muted);
  }}
  table.booktabs tbody td,
  table.booktabs tbody th {{
    padding: 5px 6px;
    border-bottom: 1px solid var(--border);
    text-align: left;
    font-weight: 400;
  }}
  table.booktabs tbody tr:last-child td,
  table.booktabs tbody tr:last-child th {{
    border-bottom: 1.5px solid var(--rule-strong);
  }}
  table.booktabs .num {{
    text-align: right;
    font-family: var(--font-mono);
    font-weight: 600;
  }}

  /* Responsive sticky comparison table (Table 1) */
  .table-wrapper {{
    overflow: auto;
    max-inline-size: 100%;
    max-block-size: min(560px, 75vh);
    container-type: inline-size;
    border-top: 1.5px solid var(--rule-strong);
    border-bottom: 1.5px solid var(--rule-strong);
  }}
  .table-wrapper:focus-visible {{
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }}
  table.ladder {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.82rem;
    font-variant-numeric: tabular-nums;
    --label-2: "Params";
    --label-3: "Lookahead";
    --label-4: "Session 7.3 LOPO";
    --label-5: "NICU 1 Hz (S=28)";
    --label-6: "p_Holm";
    --label-7: "NICU Trig-10s";
    --label-8: "Ext >=28 bpm";
    --label-9: "Ext Slope";
    --label-10: "20-bpm Skirt";
  }}
  table.ladder caption {{
    text-align: left;
    font-family: var(--font-serif);
    font-weight: 700;
    font-size: 0.9rem;
    padding: 8px 6px;
    background: var(--card);
    border-bottom: 1px solid var(--border);
  }}
  table.ladder thead {{
    position: sticky;
    inset-block-start: 0;
    z-index: 2;
  }}
  table.ladder thead th {{
    background: var(--surface2);
    color: var(--fg);
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid var(--rule-strong);
    font-weight: 700;
    white-space: nowrap;
  }}
  table.ladder tbody th {{
    position: sticky;
    inset-inline-start: 0;
    z-index: 1;
    background: var(--card);
    text-align: left;
    padding: 7px 10px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    font-weight: 600;
  }}
  table.ladder tbody td {{
    padding: 7px 10px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    font-family: var(--font-mono);
  }}
  table.ladder tbody tr.row-ours th,
  table.ladder tbody tr.row-ours td {{
    background: var(--surface2);
    font-weight: 700;
  }}

  @container (width < 600px) {{
    table.ladder, table.ladder thead, table.ladder tbody, table.ladder tr, table.ladder th, table.ladder td {{ display: block; }}
    table.ladder thead {{ display: none; }}
    table.ladder tbody tr {{
      margin-block-end: 0.85rem;
      border: 1px solid var(--border);
    }}
    table.ladder tbody th {{
      position: sticky;
      inset-block-start: 0;
      z-index: 2;
      background: var(--surface2);
    }}
    table.ladder tbody td {{ padding: 5px 10px; white-space: normal; }}
    table.ladder td:nth-child(2)::before {{ content: var(--label-2) ": "; font-family: var(--font-sans); font-weight: 700; color: var(--muted); }}
    table.ladder td:nth-child(3)::before {{ content: var(--label-3) ": "; font-family: var(--font-sans); font-weight: 700; color: var(--muted); }}
    table.ladder td:nth-child(4)::before {{ content: var(--label-4) ": "; font-family: var(--font-sans); font-weight: 700; color: var(--muted); }}
    table.ladder td:nth-child(5)::before {{ content: var(--label-5) ": "; font-family: var(--font-sans); font-weight: 700; color: var(--muted); }}
    table.ladder td:nth-child(6)::before {{ content: var(--label-6) ": "; font-family: var(--font-sans); font-weight: 700; color: var(--muted); }}
    table.ladder td:nth-child(7)::before {{ content: var(--label-7) ": "; font-family: var(--font-sans); font-weight: 700; color: var(--muted); }}
    table.ladder td:nth-child(8)::before {{ content: var(--label-8) ": "; font-family: var(--font-sans); font-weight: 700; color: var(--muted); }}
    table.ladder td:nth-child(9)::before {{ content: var(--label-9) ": "; font-family: var(--font-sans); font-weight: 700; color: var(--muted); }}
    table.ladder td:nth-child(10)::before {{ content: var(--label-10) ": "; font-family: var(--font-sans); font-weight: 700; color: var(--muted); }}
  }}
</style>
</head>
<body>
<div class="monograph-shell">
  <header class="masthead">
    <p class="rubric">Brandenburg University of Technology Cottbus–Senftenberg &middot; Faculty 1 &middot; M.Sc. Artificial Intelligence (Module 13600)</p>
    <h1 class="masthead-title">Physics-Constrained Machine Learning for Contactless Neonatal Radar Respiratory Monitoring</h1>
    <p class="masthead-sub">Interactive Companion Supplement — Causal <code>FilterSafe-8</code> Tracking, Selective Abstention, and Out-of-Band Generalization</p>
    <div class="meta-bar">
      <div class="meta-items">
        <span><strong>Author:</strong> Hosam Shibly</span>
        <span>&middot; <strong>Protocol:</strong> Held-Out LOPO Infant 7 (Session 7.3, <i>N</i> = {payload["meta"]["n_win"]}, 25.0 min)</span>
        <span>&middot; <strong>Sealed Core:</strong> <code>5e458c1</code></span>
        <span>&middot; <strong>Archive SHA-256:</strong> <code>{payload["meta"]["archive_sha_short"]}</code></span>
      </div>
      <label for="theme-select">
        Appearance
        <select id="theme-select" class="sel" aria-label="Color appearance">
          <option value="auto">System (Auto)</option>
          <option value="light" selected>Paper (Light)</option>
          <option value="dark">Slate (Dark)</option>
        </select>
      </label>
    </div>
  </header>

  <div class="tabs">
    <div class="tablist" focusgroup="tablist nomemory" role="tablist" aria-label="Interactive monograph figures and tables">
      <button type="button" role="tab" id="tab-session" aria-selected="true" focusgroupstart>Figure 1 &middot; Session 7.3 Causal Tracking &amp; Selective UQ</button>
      <button type="button" role="tab" id="tab-stress">Figure 2 &middot; Sub-Band Skirt (20 bpm) &amp; Sensor-Dropout Stress</button>
      <button type="button" role="tab" id="tab-ladder">Table 1 &middot; Cohort LOPO &amp; External Transfer Summary</button>
    </div>

    <div id="panel-session" role="tabpanel" aria-labelledby="tab-session" tabindex="0">
      <div class="panel-inner">
        <div class="plate">
          <div class="controls-bar">
            <button type="button" id="btn-play" class="btn btn-primary">&#9654; Play</button>
            <button type="button" id="btn-step-back" class="btn" aria-label="Step backward 2 seconds">&minus;2 s</button>
            <button type="button" id="btn-step-fwd" class="btn" aria-label="Step forward 2 seconds">+2 s</button>
            <label class="mono-val" for="speed-select">
              Rate
              <select id="speed-select" class="sel">
                <option value="1">1&times; (0.5 Hz)</option>
                <option value="5" selected>5&times;</option>
                <option value="15">15&times;</option>
                <option value="30">30&times;</option>
              </select>
            </label>
            <div class="scrubber-wrap">
              <label for="time-slider" class="mono-val" style="white-space:nowrap;">Cursor <i>t</i></label>
              <input id="time-slider" type="range" min="0" max="{payload["meta"]["n_win"] - 1}" step="1" value="240" />
              <span id="time-readout" class="mono-val" style="min-width:136px;text-align:right;"><i>t</i> = 0.00 min</span>
            </div>
            <label class="mono-val" for="bg-select">
              Density Field
              <select id="bg-select" class="sel">
                <option value="posterior" selected>Causal Posterior p(r_t | y_1:t)</option>
                <option value="salience">Emission Salience e_t(r)</option>
                <option value="none">None (White/Slate)</option>
              </select>
            </label>
          </div>
          <div class="toggle-list" role="group" aria-label="Estimator overlays">
            <label class="chk-label" for="chk-ref"><input type="checkbox" id="chk-ref" class="visually-hidden" checked /><span><i class="swatch" style="background:currentColor;"></i> 1 Hz Reference</span></label>
            <label class="chk-label" for="chk-r2e"><input type="checkbox" id="chk-r2e" class="visually-hidden" checked /><span><i class="swatch" style="background:#009E73;"></i> R2e <code>FilterSafe-8</code> (<i>P</i> = 8)</span></label>
            <label class="chk-label" for="chk-c5"><input type="checkbox" id="chk-c5" class="visually-hidden" checked /><span><i class="swatch" style="background:#009E73;opacity:0.45;height:8px;"></i> 80% <i>C</i><sub>5</sub> Conformal Interval</span></label>
            <label class="chk-label" for="chk-r0"><input type="checkbox" id="chk-r0" class="visually-hidden" checked /><span><i class="swatch" style="background:#56B4E9;"></i> R0 HPSS+NLS (<i>P</i> = 0)</span></label>
            <label class="chk-label" for="chk-r3a"><input type="checkbox" id="chk-r3a" class="visually-hidden" checked /><span><i class="swatch" style="background:#CC79A7;"></i> R3a Causal TCN (<i>P</i> = 15,817)</span></label>
            <label class="chk-label" for="chk-r1a"><input type="checkbox" id="chk-r1a" class="visually-hidden" /><span><i class="swatch" style="background:#0072B2;"></i> R1a Fwd HMM (<i>P</i> = 3)</span></label>
            <label class="chk-label" for="chk-r3c"><input type="checkbox" id="chk-r3c" class="visually-hidden" /><span><i class="swatch" style="background:#E69F00;"></i> R3c Ridge (<i>P</i> = 439)</span></label>
            <label class="chk-label" for="chk-r3b"><input type="checkbox" id="chk-r3b" class="visually-hidden" /><span><i class="swatch" style="background:#D55E00;"></i> R3b MLP (<i>P</i> = 14,569)</span></label>
            <label class="chk-label" for="chk-r1f"><input type="checkbox" id="chk-r1f" class="visually-hidden" /><span><i class="swatch" style="background:#6d28d9;"></i> R1f Offline FB</span></label>
          </div>
        </div>

        <div class="grid-main">
          <div style="display:flex;flex-direction:column;gap:14px;">
            <div class="plate">
              <h2 class="fig-title">(a) Causal Respiratory Rate Tracking Trajectory (Session 7.3)</h2>
              <p class="fig-sub">Click or drag horizontally across axes (a) or (b) to scrub cursor <i>t</i>; shaded vertical columns indicate abstained windows and top ticks mark &gt;5 s pauses.</p>
              <div class="canvas-box">
                <canvas id="cv-traj" height="310" aria-label="Respiratory rate trajectory plot"></canvas>
              </div>
              <h2 class="fig-title" style="margin-top:14px;">(b) Normalized Causal Uncertainty Signals: <i>U</i><sub>2</sub> vs. <i>U</i><sub>6b</sub></h2>
              <div class="canvas-box">
                <canvas id="cv-uq" height="165" aria-label="Uncertainty signal timeline"></canvas>
              </div>
              <p class="fig-caption"><strong>Figure 1a–b.</strong> Out-of-sample LOPO trajectory on preterm Anchor Session <code>7.3</code> (25.0 min, 736 overlapping 30 s windows at 2 s stride). Memoryless <code>R0</code> jumps onto harmonic/motion clutter while the unconstrained neural foil <code>R3a</code> (Causal TCN, RMSE 6.353 bpm) degrades below <code>R0</code> (4.789 bpm). <code>FilterSafe-8</code> (<code>R2e</code>, 4.117 bpm) maintains continuous Bayesian posterior coherence and finite-<i>K</i> conformal coverage (Chapter 4–5).</p>
            </div>
          </div>

          <div style="display:flex;flex-direction:column;gap:14px;">
            <div class="plate">
              <h2 class="fig-title">(c) Instantaneous State Density at Cursor <i>t</i></h2>
              <p class="fig-sub">Bars: likelihood <i>e<sub>t</sub></i>(<i>r</i>); curve: causal forward posterior <i>p</i>(<i>r<sub>t</sub></i> | <b>y</b><sub>1:<i>t</i></sub>).</p>
              <div class="canvas-box">
                <canvas id="cv-slice" height="205" aria-label="Spectral emission and posterior distribution slice"></canvas>
              </div>
              <table class="booktabs" aria-label="Instantaneous cursor telemetry">
                <thead>
                  <tr><th>Signal / Estimator</th><th class="num">Value at <i>t</i></th></tr>
                </thead>
                <tbody>
                  <tr><th>1 Hz Reference (Clinical State)</th><td id="kpi-ref" class="num">--</td></tr>
                  <tr><th><code>R2e</code> <code>FilterSafe-8</code> &plusmn; 80% <i>C</i><sub>5</sub></th><td id="kpi-r2e" class="num">--</td></tr>
                  <tr><th><code>R0</code> (0p) / <code>R3a</code> TCN (15.8k)</th><td id="kpi-foils" class="num">--</td></tr>
                  <tr><th><i>U</i><sub>2</sub> Post. SD / <i>U</i><sub>6b</sub> Barrier SQI</th><td id="kpi-uq" class="num">--</td></tr>
                </tbody>
              </table>
            </div>

            <div class="plate">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;">
                <h2 class="fig-title" style="margin:0;">Selective Abstention Analysis</h2>
                <label class="mono-val" for="proxy-select">
                  Proxy
                  <select id="proxy-select" class="sel">
                    <option value="u6b" selected>U6b Barrier SQI</option>
                    <option value="u2">U2 Posterior SD</option>
                  </select>
                </label>
              </div>
              <div style="display:flex;align-items:center;gap:10px;margin-top:4px;">
                <label for="cov-slider" class="mono-val" style="white-space:nowrap;">Target Coverage</label>
                <input id="cov-slider" type="range" min="50" max="100" step="1" value="80" />
                <span id="cov-readout" class="mono-val" style="min-width:44px;text-align:right;">80%</span>
              </div>
              <table class="booktabs" aria-label="Selective risk-coverage summary">
                <thead>
                  <tr><th>Retained Cohort Metric</th><th class="num">Result</th></tr>
                </thead>
                <tbody>
                  <tr><th>Retained <code>R2e</code> 1 Hz RMSE</th><td id="kpi-ret-rmse" class="num">--</td></tr>
                  <tr><th>&Delta; RMSE vs. 100% Coverage</th><td id="kpi-ret-delta" class="num">--</td></tr>
                  <tr><th>Empirical 80% <i>C</i><sub>5</sub> Coverage</th><td id="kpi-ret-c5" class="num">--</td></tr>
                  <tr><th>Retained Apnea Pauses (&gt;5 s)</th><td id="kpi-ret-pause" class="num">--</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div id="panel-stress" role="tabpanel" aria-labelledby="tab-stress" hidden="until-found">
      <div class="panel-inner">
        <div class="plate">
          <div class="grid-two">
            <div>
              <h2 class="fig-title">(a) Frequency Sweep Across 28 bpm Cutoff (14–72 bpm)</h2>
              <p class="fig-sub">Zero-touch evaluation of frozen Stage 4 weights (<code>af33a83e</code>) on unit-SNR 24 GHz CW simulations.</p>
              <div class="canvas-box"><canvas id="cv-sweep" height="305" aria-label="Frequency sweep invariance plot"></canvas></div>
            </div>
            <div>
              <h2 class="fig-title">(b) 20 bpm Bradypnea Skirt &rarr; Pure WGN Dropout &rarr; Recovery</h2>
              <p class="fig-sub">36 causal windows testing spectral skirt integrity and zero-tuned <i>U</i><sub>6b</sub> out-of-distribution rejection.</p>
              <div class="canvas-box"><canvas id="cv-dropout" height="305" aria-label="Bradypnea to WGN dropout stress replay"></canvas></div>
            </div>
          </div>
          <p class="fig-caption"><strong>Figure 2a–b.</strong> Because <code>FilterSafe-8</code> operates strictly via <i>SO</i>(2)&times;&#8477;<sub>&gt;0</sub>-invariant folded log-salience channels with zero learned absolute-frequency weights, it maintains unit slope across the 28 bpm high-pass boundary and holds 20 bpm bradypnea accurately while <i>U</i><sub>6b</sub> rises sharply during white Gaussian noise (WGN) dropout. High-capacity regressors (<code>R3a</code>, <code>R3b</code>, <code>R3c</code>) collapse outside training support and hallucinate rates under noise (Chapter 6).</p>
        </div>
      </div>
    </div>

    <div id="panel-ladder" role="tabpanel" aria-labelledby="tab-ladder" hidden="until-found">
      <div class="panel-inner">
        <div class="plate">
          <div tabindex="0" role="region" aria-labelledby="ladder-caption" class="table-wrapper">
            <table class="ladder">
              <caption id="ladder-caption">Table 1. Multi-stage complexity ladder: Session 7.3 held-out LOPO vs. NICU cohort (<i>K</i> = 11, <i>S</i> = 28) &amp; zero-touch 24.2 GHz transfer (RMSE in bpm).</caption>
              <thead>
                <tr>
                  <th>Model Family</th>
                  <th>Params</th>
                  <th>Lookahead (Blk / STFT)</th>
                  <th>Session 7.3 LOPO 1 Hz</th>
                  <th>NICU 1 Hz (<i>S</i> = 28)</th>
                  <th><i>p</i><sub>Holm</sub> vs. R0</th>
                  <th>NICU Trig-10s</th>
                  <th>Ext &ge;28 bpm</th>
                  <th>Ext Slope</th>
                  <th>Ext 20-bpm Skirt</th>
                </tr>
              </thead>
              <tbody id="ladder-tbody"></tbody>
            </table>
          </div>
          <p class="fig-caption"><strong>Table 1 Notes.</strong> Session <code>7.3</code> LOPO RMSE is evaluated over 593 finite valid 30 s reference windows (matching the 100% coverage cohort in Figure 1); full-cohort NICU and zero-touch Stage 4 values are read directly from immutable JSON archives <code>metrics_stage2_complexity.json</code> and <code>metrics_stage4_external.json</code> (Chapters 4 and 6).</p>
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

  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const fmt = (v, d = 1, unit = "") => (isNum(v) ? v.toFixed(d) + unit : "\u2014 (artifact)");

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
    r2e: "#009E73",
    r0: "#56B4E9",
    r3a: "#CC79A7",
    r1a: "#0072B2",
    r3c: "#E69F00",
    r3b: "#D55E00",
    r1f: "#6d28d9",
  }};

  // Set default initial appearance cleanly to Paper (Light)
  document.documentElement.dataset.theme = "light";

  function getThemeColors() {{
    const cs = getComputedStyle(document.documentElement);
    return {{
      bg: cs.getPropertyValue("--card").trim() || "#ffffff",
      surface2: cs.getPropertyValue("--surface2").trim() || "#f2f4f7",
      border: cs.getPropertyValue("--border").trim() || "#cfd5df",
      ruleStrong: cs.getPropertyValue("--rule-strong").trim() || "#1f2937",
      fg: cs.getPropertyValue("--fg").trim() || "#141821",
      muted: cs.getPropertyValue("--muted").trim() || "#4b5565",
      brand: cs.getPropertyValue("--brand-color").trim() || "#1b4d3e",
      accent: cs.getPropertyValue("--accent").trim() || "#1b365d",
      warn: cs.getPropertyValue("--warn").trim() || "#9a3412",
      danger: cs.getPropertyValue("--danger").trim() || "#881337",
    }};
  }}

  function setupCanvas(cv, cssH) {{
    const dpr = Math.max(window.devicePixelRatio || 1, 1);
    const rect = cv.getBoundingClientRect();
    const parentW = cv.parentElement ? cv.parentElement.clientWidth : 640;
    const cssW = Math.max(rect.width || parentW || 640, 280);
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
      if (S73.valid[i] && isNum(S73.ref_1hz[i]) && isNum(S73.pred_R2e[i]) && isNum(arr[i])) {{
        validIdx.push(i);
      }}
    }}
    const sorted = validIdx.map((i) => arr[i]).sort((a, b) => a - b);
    const frac = STATE.coveragePct / 100.0;
    const thrIdx = sorted.length > 0
      ? Math.min(sorted.length - 1, Math.max(0, Math.floor(frac * (sorted.length - 1))))
      : 0;
    const thr = sorted.length > 0 ? sorted[thrIdx] : 1.0;

    let seAll = 0, nAll = 0, seRet = 0, nRet = 0, covHits = 0;
    let pausesAll = 0, pausesRet = 0;
    const kept = new Uint8Array(N_WIN);
    for (let i = 0; i < N_WIN; i++) {{
      const uVal = arr[i];
      const isKept = STATE.coveragePct >= 100 || (isNum(uVal) && uVal <= thr);
      if (isKept) kept[i] = 1;
      if (!S73.valid[i] || !isNum(S73.ref_1hz[i]) || !isNum(S73.pred_R2e[i])) continue;
      const err = S73.pred_R2e[i] - S73.ref_1hz[i];
      seAll += err * err;
      nAll++;
      if (S73.pause[i]) pausesAll++;
      if (isKept) {{
        seRet += err * err;
        nRet++;
        if (isNum(S73.c5_half_80[i]) && Math.abs(err) <= S73.c5_half_80[i]) covHits++;
        if (S73.pause[i]) pausesRet++;
      }}
    }}
    const rmseAll = Math.sqrt(seAll / Math.max(1, nAll));
    const rmseRet = Math.sqrt(seRet / Math.max(1, nRet));
    return {{
      thr, kept, rmseAll, rmseRet, delta: rmseRet - rmseAll,
      c5Cov: covHits / Math.max(1, nRet), pausesRet, pausesAll, nRet, nAll,
    }};
  }}

  function drawFrameAndTicks(ctx, pad, pw, ph, tc) {{
    ctx.strokeStyle = tc.ruleStrong;
    ctx.lineWidth = 1.0;
    ctx.strokeRect(pad.l, pad.t, pw, ph);
  }}

  function drawTrajectory(ab) {{
    const cv = document.getElementById("cv-traj");
    const {{ ctx, w, h }} = setupCanvas(cv, 310);
    const tc = getThemeColors();
    const pad = {{ l: 52, r: 14, t: 14, b: 36 }};
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;
    const yMin = 18, yMax = 90;
    const xOf = (i) => pad.l + (i / (N_WIN - 1)) * pw;
    const yOf = (y) => pad.t + ((yMax - Math.min(yMax, Math.max(yMin, y))) / (yMax - yMin)) * ph;

    ctx.fillStyle = tc.bg;
    ctx.fillRect(0, 0, w, h);

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
            ? `rgba(0, 158, 115, ${{(v * 0.34).toFixed(3)}})`
            : `rgba(86, 180, 233, ${{(v * 0.30).toFixed(3)}})`;
          ctx.fillRect(x, pad.t + ((GRID_LEN - 1 - b) / GRID_LEN) * ph, cellW + 0.6, cellH + 0.6);
        }}
      }}
    }}

    for (let i = 0; i < N_WIN; i++) {{
      if (S73.valid[i] && !ab.kept[i]) {{
        ctx.fillStyle = "rgba(154, 52, 18, 0.10)";
        ctx.fillRect(xOf(i) - 0.5, pad.t, pw / N_WIN + 1.0, ph);
      }}
      if (S73.pause[i] && S73.valid[i]) {{
        ctx.fillStyle = tc.danger;
        ctx.fillRect(xOf(i) - 0.5, pad.t + 1, Math.max(2, pw / N_WIN), 5);
      }}
    }}

    // Subtle dashed horizontal gridlines + ticks
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = tc.muted;
    ctx.textAlign = "right";
    for (let br = 20; br <= 90; br += 10) {{
      const yy = yOf(br);
      ctx.save();
      ctx.strokeStyle = tc.border;
      ctx.lineWidth = 0.8;
      ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + pw, yy); ctx.stroke();
      ctx.restore();
      ctx.strokeStyle = tc.ruleStrong;
      ctx.beginPath(); ctx.moveTo(pad.l - 4, yy); ctx.lineTo(pad.l, yy); ctx.stroke();
      ctx.fillText(String(br), pad.l - 7, yy + 4);
    }}
    ctx.textAlign = "center";
    const tMaxMin = isNum(S73.t_min[N_WIN - 1]) ? S73.t_min[N_WIN - 1] : 25.0;
    for (let m = 0; m <= 25; m += 5) {{
      const xx = pad.l + (m / tMaxMin) * pw;
      ctx.strokeStyle = tc.ruleStrong;
      ctx.beginPath(); ctx.moveTo(xx, pad.t + ph); ctx.lineTo(xx, pad.t + ph + 4); ctx.stroke();
      ctx.fillText(String(m), xx, pad.t + ph + 16);
    }}
    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText("Recording Time t (min)", pad.l + pw / 2, h - 4);
    ctx.save();
    ctx.translate(14, pad.t + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Rate (breaths/min)", 0, 0);
    ctx.restore();

    if (STATE.toggles.c5 && STATE.toggles.r2e) {{
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < N_WIN; i++) {{
        if (!isNum(S73.pred_R2e[i]) || !isNum(S73.c5_half_80[i])) continue;
        const x = xOf(i), y = yOf(S73.pred_R2e[i] + S73.c5_half_80[i]);
        if (!started) {{ ctx.moveTo(x, y); started = true; }} else ctx.lineTo(x, y);
      }}
      for (let i = N_WIN - 1; i >= 0; i--) {{
        if (!isNum(S73.pred_R2e[i]) || !isNum(S73.c5_half_80[i])) continue;
        ctx.lineTo(xOf(i), yOf(S73.pred_R2e[i] - S73.c5_half_80[i]));
      }}
      ctx.closePath();
      ctx.fillStyle = "rgba(0, 158, 115, 0.20)";
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
        const v = arr[i];
        if (!isNum(v)) {{ started = false; continue; }}
        const x = xOf(i), y = yOf(v);
        if (!started) {{ ctx.moveTo(x, y); started = true; }} else ctx.lineTo(x, y);
      }}
      ctx.stroke();
      ctx.restore();
    }};

    if (STATE.toggles.r0) drawSeries(S73.pred_R0, COLORS.r0, 1.15, 0.80);
    if (STATE.toggles.r3b) drawSeries(S73.pred_R3b, COLORS.r3b, 1.2, 0.80);
    if (STATE.toggles.r3c) drawSeries(S73.pred_R3c, COLORS.r3c, 1.25, 0.85);
    if (STATE.toggles.r3a) drawSeries(S73.pred_R3a, COLORS.r3a, 1.35, 0.88);
    if (STATE.toggles.r1a) drawSeries(S73.pred_R1a, COLORS.r1a, 1.35, 0.90);
    if (STATE.toggles.r1f) drawSeries(S73.pred_R1f, COLORS.r1f, 1.35, 0.88, [4, 3]);
    if (STATE.toggles.ref) drawSeries(S73.ref_1hz, tc.fg, 1.7, 0.92);
    if (STATE.toggles.r2e) drawSeries(S73.pred_R2e, COLORS.r2e, 2.0, 1.0);

    drawFrameAndTicks(ctx, pad, pw, ph, tc);

    const cx = xOf(STATE.tIdx);
    ctx.strokeStyle = tc.accent;
    ctx.lineWidth = 1.4;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(cx, pad.t); ctx.lineTo(cx, pad.t + ph); ctx.stroke();
    ctx.setLineDash([]);
    if (isNum(S73.pred_R2e[STATE.tIdx])) {{
      ctx.fillStyle = COLORS.r2e;
      ctx.beginPath(); ctx.arc(cx, yOf(S73.pred_R2e[STATE.tIdx]), 4.2, 0, Math.PI * 2); ctx.fill();
    }}
  }}

  function drawUncertaintyTimeline(ab) {{
    const cv = document.getElementById("cv-uq");
    const {{ ctx, w, h }} = setupCanvas(cv, 165);
    const tc = getThemeColors();
    const pad = {{ l: 52, r: 14, t: 12, b: 32 }};
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;
    const xOf = (i) => pad.l + (i / (N_WIN - 1)) * pw;
    const yOf = (u) => pad.t + (1.0 - Math.min(1.0, Math.max(0.0, u))) * ph;

    ctx.fillStyle = tc.bg;
    ctx.fillRect(0, 0, w, h);

    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = tc.muted;
    ctx.textAlign = "right";
    for (const v of [0.0, 0.25, 0.5, 0.75, 1.0]) {{
      const yy = yOf(v);
      ctx.save();
      ctx.strokeStyle = tc.border;
      ctx.lineWidth = 0.8;
      ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + pw, yy); ctx.stroke();
      ctx.restore();
      ctx.fillText(v.toFixed(2), pad.l - 7, yy + 4);
    }}
    ctx.textAlign = "center";
    const tMaxMin = isNum(S73.t_min[N_WIN - 1]) ? S73.t_min[N_WIN - 1] : 25.0;
    for (let m = 0; m <= 25; m += 5) {{
      ctx.fillText(String(m), pad.l + (m / tMaxMin) * pw, pad.t + ph + 14);
    }}
    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText("Recording Time t (min)", pad.l + pw / 2, h - 3);
    ctx.save();
    ctx.translate(14, pad.t + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("SQI / SD [0, 1]", 0, 0);
    ctx.restore();

    for (let i = 0; i < N_WIN; i++) {{
      if (S73.valid[i] && !ab.kept[i]) {{
        ctx.fillStyle = "rgba(154, 52, 18, 0.12)";
        ctx.fillRect(xOf(i) - 0.5, pad.t, pw / N_WIN + 1.0, ph);
      }}
    }}

    const plotCurve = (arr, color, lw) => {{
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < N_WIN; i++) {{
        if (!isNum(arr[i])) {{ started = false; continue; }}
        const x = xOf(i), y = yOf(arr[i]);
        if (!started) {{ ctx.moveTo(x, y); started = true; }} else ctx.lineTo(x, y);
      }}
      ctx.stroke();
      ctx.restore();
    }};
    plotCurve(S73.u2, COLORS.r1a, 1.35);
    plotCurve(S73.u6b, COLORS.r3b, 1.5);

    if (STATE.coveragePct < 100 && isNum(ab.thr)) {{
      const yt = yOf(ab.thr);
      ctx.save();
      ctx.strokeStyle = tc.ruleStrong;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(pad.l, yt); ctx.lineTo(pad.l + pw, yt); ctx.stroke();
      ctx.restore();
    }}

    drawFrameAndTicks(ctx, pad, pw, ph, tc);

    const cx = xOf(STATE.tIdx);
    ctx.strokeStyle = tc.accent;
    ctx.lineWidth = 1.4;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(cx, pad.t); ctx.lineTo(cx, pad.t + ph); ctx.stroke();
    ctx.setLineDash([]);
  }}

  function drawSlice() {{
    const cv = document.getElementById("cv-slice");
    const {{ ctx, w, h }} = setupCanvas(cv, 205);
    const tc = getThemeColors();
    const pad = {{ l: 40, r: 12, t: 16, b: 32 }};
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;
    const ti = STATE.tIdx;

    ctx.fillStyle = tc.bg;
    ctx.fillRect(0, 0, w, h);

    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = tc.muted;
    ctx.textAlign = "right";
    for (const v of [0, 0.5, 1.0]) {{
      const yy = pad.t + (1 - v) * ph;
      ctx.save();
      ctx.strokeStyle = tc.border;
      ctx.lineWidth = 0.8;
      ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + pw, yy); ctx.stroke();
      ctx.restore();
      ctx.fillText(v.toFixed(1), pad.l - 5, yy + 4);
    }}
    ctx.textAlign = "center";
    for (let br = 20; br <= 90; br += 10) {{
      const xx = pad.l + ((br - GRID_MIN) / (GRID_LEN - 1)) * pw;
      ctx.fillText(String(br), xx, pad.t + ph + 14);
    }}
    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText("Candidate Rate r (breaths/min)", pad.l + pw / 2, h - 3);

    const xGrid = (b) => pad.l + (b / (GRID_LEN - 1)) * pw;
    const yNorm = (v) => pad.t + (1 - v) * ph;

    ctx.fillStyle = "rgba(86, 180, 233, 0.45)";
    const bw = Math.max(1.5, pw / GRID_LEN - 0.5);
    for (let b = 0; b < GRID_LEN; b++) {{
      const v = SAL_U8[ti * GRID_LEN + b] / 255.0;
      const y = yNorm(v);
      ctx.fillRect(xGrid(b) - bw / 2, y, bw, pad.t + ph - y);
    }}

    ctx.strokeStyle = COLORS.r2e;
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    for (let b = 0; b < GRID_LEN; b++) {{
      const v = POST_U8[ti * GRID_LEN + b] / 255.0;
      const x = xGrid(b), y = yNorm(v);
      b === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }}
    ctx.stroke();

    drawFrameAndTicks(ctx, pad, pw, ph, tc);

    const drawMarker = (val, color, label, dy) => {{
      if (!isNum(val)) return;
      const xx = pad.l + ((Math.min(90, Math.max(18, val)) - GRID_MIN) / (GRID_LEN - 1)) * pw;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(xx, pad.t); ctx.lineTo(xx, pad.t + ph); ctx.stroke();
      ctx.fillStyle = color;
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = xx > w - 45 ? "right" : "left";
      ctx.fillText(label, xx + (xx > w - 45 ? -4 : 4), pad.t + dy);
    }};
    drawMarker(S73.pred_R0[ti], COLORS.r0, "R0", 13);
    drawMarker(S73.ref_1hz[ti], tc.fg, "Ref", 27);
    drawMarker(S73.pred_R2e[ti], COLORS.r2e, "R2e", 41);
  }}

  function drawStressTabs() {{
    const cv1 = document.getElementById("cv-sweep");
    const cv2 = document.getElementById("cv-dropout");
    if (!cv1 || !cv2) return;
    const tc = getThemeColors();

    {{
      const {{ ctx, w, h }} = setupCanvas(cv1, 305);
      const sw = DATA.stress.sweep;
      const pad = {{ l: 50, r: 16, t: 22, b: 36 }};
      const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
      const xOf = (r) => pad.l + ((r - 14) / (72 - 14)) * pw;
      const yOf = (y) => pad.t + ((85 - Math.min(85, Math.max(10, y))) / (85 - 10)) * ph;

      ctx.fillStyle = tc.bg; ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "rgba(154, 52, 18, 0.08)";
      ctx.fillRect(pad.l, pad.t, xOf(28) - pad.l, ph);
      ctx.strokeStyle = tc.warn; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(xOf(28), pad.t); ctx.lineTo(xOf(28), pad.t + ph); ctx.stroke();
      ctx.setLineDash([]);

      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillStyle = tc.muted; ctx.textAlign = "right";
      for (const y of [15, 30, 45, 60, 75]) {{
        const yy = yOf(y);
        ctx.save(); ctx.strokeStyle = tc.border; ctx.lineWidth = 0.8; ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + pw, yy); ctx.stroke(); ctx.restore();
        ctx.fillText(String(y), pad.l - 6, yy + 4);
      }}
      ctx.textAlign = "center";
      for (const r of [14, 20, 28, 40, 56, 72]) ctx.fillText(String(r), xOf(r), pad.t + ph + 15);
      ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText("Simulated Respiratory Rate (breaths/min)", pad.l + pw / 2, h - 4);
      ctx.fillStyle = tc.warn;
      ctx.fillText("Skirt (<28 bpm)", (pad.l + xOf(28)) / 2, pad.t + 14);

      ctx.save(); ctx.strokeStyle = tc.fg; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(xOf(14), yOf(14)); ctx.lineTo(xOf(72), yOf(72)); ctx.stroke(); ctx.restore();

      for (const [k, color, lw] of [["R3c", COLORS.r3c, 1.5], ["R3b", COLORS.r3b, 1.6], ["R3a", COLORS.r3a, 1.8], ["R0", COLORS.r0, 1.5], ["R2e", COLORS.r2e, 2.3]]) {{
        ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath();
        let started = false;
        sw.rates_bpm.forEach((r, idx) => {{
          const v = sw[k][idx];
          if (!isNum(v)) {{ started = false; return; }}
          const x = xOf(r), y = yOf(v);
          if (!started) {{ ctx.moveTo(x, y); started = true; }} else ctx.lineTo(x, y);
        }});
        ctx.stroke();
      }}
      drawFrameAndTicks(ctx, pad, pw, ph, tc);

      let lx = pad.l + 8;
      ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
      for (const [lbl, c] of [["Identity", tc.fg], ["R2e (8p)", COLORS.r2e], ["R3a TCN", COLORS.r3a], ["R3b MLP", COLORS.r3b], ["R3c Ridge", COLORS.r3c]]) {{
        ctx.fillStyle = c; ctx.fillRect(lx, h - 49, 10, 3);
        ctx.fillStyle = tc.muted; ctx.textAlign = "left"; ctx.fillText(lbl, lx + 13, h - 44);
        lx += ctx.measureText(lbl).width + 22;
      }}
    }}

    {{
      const {{ ctx, w, h }} = setupCanvas(cv2, 305);
      const rp = DATA.stress.dropout_replay;
      const n = rp.n_win;
      const pad = {{ l: 50, r: 44, t: 22, b: 36 }};
      const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
      const xOf = (i) => pad.l + (i / (n - 1)) * pw;
      const yOf = (y) => pad.t + ((85 - Math.min(85, Math.max(10, y))) / (85 - 10)) * ph;
      const yU = (u) => pad.t + (1 - u) * ph;

      ctx.fillStyle = tc.bg; ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "rgba(136, 19, 55, 0.10)";
      ctx.fillRect(xOf(11.5), pad.t, xOf(23.5) - xOf(11.5), ph);

      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillStyle = tc.muted; ctx.textAlign = "right";
      for (const y of [20, 40, 60, 80]) {{
        const yy = yOf(y);
        ctx.save(); ctx.strokeStyle = tc.border; ctx.lineWidth = 0.8; ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + pw, yy); ctx.stroke(); ctx.restore();
        ctx.fillText(String(y), pad.l - 6, yy + 4);
      }}
      ctx.textAlign = "left";
      for (const u of [0.0, 0.5, 1.0]) ctx.fillText(u.toFixed(1), pad.l + pw + 6, yU(u) + 4);
      ctx.textAlign = "center";
      ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
      ctx.fillStyle = tc.danger; ctx.fillText("WGN Dropout", (xOf(12) + xOf(23)) / 2, pad.t + 14);
      ctx.fillStyle = tc.muted;
      ctx.fillText("20 bpm Skirt", xOf(5.5), pad.t + ph + 15);
      ctx.fillText("Recovery (20 bpm)", xOf(29.5), pad.t + ph + 15);
      ctx.fillText("Window Sequence Index (2 s stride)", pad.l + pw / 2, h - 4);

      ctx.save(); ctx.strokeStyle = tc.fg; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(xOf(0), yOf(20)); ctx.lineTo(xOf(11), yOf(20));
      ctx.moveTo(xOf(24), yOf(20)); ctx.lineTo(xOf(35), yOf(20)); ctx.stroke(); ctx.restore();

      for (const [k, color, lw] of [["R3a", COLORS.r3a, 1.7], ["R3b", COLORS.r3b, 1.5], ["R2e", COLORS.r2e, 2.3]]) {{
        ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = lw;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < n; i++) {{
          if (!isNum(rp[k][i])) {{ started = false; continue; }}
          const x = xOf(i), y = yOf(rp[k][i]);
          if (!started) {{ ctx.moveTo(x, y); started = true; }} else ctx.lineTo(x, y);
        }}
        ctx.stroke(); ctx.restore();
      }}
      ctx.save(); ctx.strokeStyle = tc.warn; ctx.lineWidth = 1.7; ctx.setLineDash([5, 3]);
      ctx.beginPath();
      let uStarted = false;
      for (let i = 0; i < n; i++) {{
        if (!isNum(rp.U6b[i])) {{ uStarted = false; continue; }}
        const x = xOf(i), y = yU(rp.U6b[i]);
        if (!uStarted) {{ ctx.moveTo(x, y); uStarted = true; }} else ctx.lineTo(x, y);
      }}
      ctx.stroke(); ctx.restore();
      drawFrameAndTicks(ctx, pad, pw, ph, tc);

      let lx = pad.l + 8;
      for (const [lbl, c] of [["R2e (bpm)", COLORS.r2e], ["R3a TCN", COLORS.r3a], ["R3b MLP", COLORS.r3b], ["U6b SQI (right)", tc.warn]]) {{
        ctx.fillStyle = c; ctx.fillRect(lx, h - 49, 10, 3);
        ctx.fillStyle = tc.muted; ctx.textAlign = "left"; ctx.fillText(lbl, lx + 13, h - 44);
        lx += ctx.measureText(lbl).width + 20;
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
    const tVal = isNum(S73.t_min[i]) ? S73.t_min[i].toFixed(2) : "0.00";
    document.getElementById("time-readout").innerHTML = `<i>t</i> = ${{tVal}} m (${{i + 1}}/${{N_WIN}})`;
    const ts = document.getElementById("time-slider");
    ts.value = String(i);
    ts.setAttribute("aria-valuetext", `${{tVal}} minutes`);
    syncRangeVars(ts);

    const cs = document.getElementById("cov-slider");
    document.getElementById("cov-readout").textContent = `${{STATE.coveragePct}}%`;
    cs.setAttribute("aria-valuetext", `${{STATE.coveragePct}} percent retained`);
    syncRangeVars(cs);

    const stateTag = S73.pause[i] ? " [PAUSE >5s]" : (S73.valid[i] ? "" : " [Invalid]");
    document.getElementById("kpi-ref").textContent = `${{fmt(S73.ref_1hz[i], 1, " bpm")}}${{stateTag}}`;
    const r2eStr = fmt(S73.pred_R2e[i], 1, "");
    const c5Str = isNum(S73.c5_half_80[i]) ? S73.c5_half_80[i].toFixed(1) : "--";
    document.getElementById("kpi-r2e").textContent = `${{r2eStr}} \u00B1 ${{c5Str}} bpm`;
    document.getElementById("kpi-foils").textContent = `${{fmt(S73.pred_R0[i], 1, "")}} / ${{fmt(S73.pred_R3a[i], 1, " bpm")}}`;
    document.getElementById("kpi-uq").textContent = `${{fmt(S73.u2[i], 3, "")}} / ${{fmt(S73.u6b[i], 3, "")}}`;

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

  const btnPlay = document.getElementById("btn-play");
  btnPlay.addEventListener("click", () => {{
    STATE.playing = !STATE.playing;
    btnPlay.innerHTML = STATE.playing ? "&#9208; Pause" : "&#9654; Play";
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
    const val = e.target.value;
    if (val === "auto") {{
      delete document.documentElement.dataset.theme;
      document.documentElement.style.colorScheme = "light dark";
    }} else {{
      document.documentElement.dataset.theme = val;
      document.documentElement.style.colorScheme = val;
    }}
    requestAnimationFrame(redrawAll);
  }});

  for (const k of Object.keys(STATE.toggles)) {{
    const el = document.getElementById("chk-" + k);
    el.addEventListener("change", () => {{ STATE.toggles[k] = el.checked; redrawAll(); }});
  }}

  function attachScrub(cv) {{
    const jump = (ev) => {{
      const rect = cv.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left - 52) / Math.max(1, rect.width - 66)));
      STATE.tIdx = Math.round(frac * (N_WIN - 1));
      redrawAll();
    }};
    cv.addEventListener("pointerdown", (ev) => {{ STATE.dragging = true; cv.setPointerCapture(ev.pointerId); jump(ev); }});
    cv.addEventListener("pointermove", (ev) => {{ if (STATE.dragging) jump(ev); }});
    cv.addEventListener("pointerup", () => {{ STATE.dragging = false; }});
  }}
  attachScrub(document.getElementById("cv-traj"));
  attachScrub(document.getElementById("cv-uq"));

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
