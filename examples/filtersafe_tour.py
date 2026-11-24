#!/usr/bin/env python3
"""Minimal, self-contained supervisor walkthrough of FilterSafe-8 & safety-aware UQ.

Demonstrates all core methodological claims from Chapters 4-5 in ~5 seconds using the
immutable modules in `paper_workspace/src/` and `thesis/` plus tracked Anchor Session 7.3
(no unbundled raw dataset files required):
  1. Loads the frozen 8-parameter FilterSafe-8 (R2e) weights (archive SHA af33a83e...).
  2. Verifies exact analytical BPTT gradients against central finite differences.
  3. Verifies SO(2) x R_{>0} I/Q invariance, strict causal 0.0-s future isolation, and
     +10 bpm operator translation equivariance on synthetic 24 GHz radar waveforms.
  4. Executes causal FilterSafe-8 + U2/U6b uncertainty scoring live on preterm NICU
     Anchor Session 7.3 vs. out-of-distribution white Gaussian noise (WGN).
  5. Demonstrates exact finite-K conformal vacuity at K_cal=10 (95%) vs K_cal>=19.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_workspace.src import config as C
from paper_workspace.src.data_loader import load_radar
from paper_workspace.src.demod import csd
from paper_workspace.src.filtering import autocorr_coarse, bandpass
from paper_workspace.src.hpss import mitigate_hpss
from thesis import diff_hmm_and_foils as D
from thesis import io_utils as IO
from thesis import viterbi_tracker as VT
from thesis import wp3_wp4_uncertainty_conformal as UQ


def load_frozen_filter_safe8() -> tuple[D.DifferentiableSymmetricHMM, np.ndarray]:
    arch = ROOT / "thesis" / "outputs" / "stage3_uncertainty" / "frozen_stage4_models.npz"
    sha = hashlib.sha256(arch.read_bytes()).hexdigest()
    assert sha.startswith("af33a83e"), f"Unexpected archive SHA-256: {sha}"
    bundle = np.load(arch, allow_pickle=True)
    model = D.DifferentiableSymmetricHMM(
        mode="linear_filtersafe", decoder="causal_forward", grid_bpm=VT.BR_GRID_1BPM
    )
    model.theta = np.asarray(bundle["r2e_full_refit_theta"], dtype=np.float64).copy()
    c5_q = float(bundle["c5_q_hat_norm_1hz"][2])  # alpha = 0.20 (80% target)
    return model, c5_q


def extract_session_salience(measurement: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    bi, bq = load_radar(measurement)
    cache = IO.load_locked_cache(measurement)
    n_win = int(len(cache["est_hpss"]))
    step = int(C.UPDATE_S * C.FS_RADAR)
    win_len = int(C.WIN_SAMPLES)
    phi_seq = np.empty((n_win, VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS), dtype=np.float64)
    for idx in range(n_win):
        s0 = idx * step
        x_win = csd(bi[s0 : s0 + win_len], bq[s0 : s0 + win_len])
        d_raw = bandpass(x_win)
        d_clean = bandpass(mitigate_hpss(d_raw)["x_hat"])
        f_coarse = autocorr_coarse(d_clean)
        phi_seq[idx] = D.extract_salience_features_window(
            d_raw, d_clean, f_coarse, VT.BR_GRID_1BPM
        )
    return phi_seq, cache


def synth_cw_radar(rate_bpm: float, duration_s: float = 40.0, disp_mm: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(int(duration_s * C.FS_RADAR), dtype=np.float64) / float(C.FS_RADAR)
    d_t = disp_mm * 1e-3 * np.sin(2.0 * math.pi * (rate_bpm / 60.0) * t)
    phase = 0.45 + (4.0 * math.pi / float(C.LAMBDA_M)) * d_t
    z = 1.25 * np.exp(1j * phase)
    return z.real, z.imag


def main() -> None:
    print("=" * 72)
    print("FilterSafe-8 (R2e) & Safety-Aware UQ -- Executable Supervisor Tour")
    print("=" * 72)

    # 1. Load frozen R2e parameters
    model, c5_q_80 = load_frozen_filter_safe8()
    assert model.n_params == 8
    ip = model.interpretable_params()
    print("\n[1] Frozen FilterSafe-8 (R2e) parameters (archive SHA af33a83e, full K=11 refit):")
    print(
        f"    w1(clean)={ip['w1_p1_clean']:+.3f}, w5(anchor)={ip['w5_anchor']:+.3f}, "
        f"w6(raw)={ip['w6_p1_raw']:+.3f}, harmonics(w2,w3,w4)=clamped 0"
    )
    print(
        f"    sigma_slow={ip['sigma_slow_bpm']:.3f} bpm, sigma_jump={ip['sigma_jump_bpm']:.3f} bpm, "
        f"pi_jump={ip['pi_jump']:.4f}, k1={ip['k1']:.3f}, k2={ip['k2']:.3f}"
    )

    # 2. Analytical BPTT vs. finite differences
    rng = np.random.default_rng(2026)
    phi_toy = rng.normal(-2.0, 0.8, size=(12, VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS))
    ref_toy = rng.uniform(28.0, 62.0, size=12)
    vm_toy = np.ones(12, dtype=bool)
    th = np.clip(model.theta.copy(), [b[0] + 1e-3 for b in model.bounds], [b[1] - 1e-3 for b in model.bounds])
    _, g_ana = model.loss_and_grad(th, [(phi_toy, ref_toy, vm_toy)])
    eps = 1e-6
    g_fd = np.zeros_like(th)
    for j in range(th.size):
        tp, tm = th.copy(), th.copy()
        tp[j] += eps
        tm[j] -= eps
        lp, _ = model.loss_and_grad(tp, [(phi_toy, ref_toy, vm_toy)])
        lm, _ = model.loss_and_grad(tm, [(phi_toy, ref_toy, vm_toy)])
        g_fd[j] = (lp - lm) / (2.0 * eps)
    grad_err = float(np.max(np.abs(g_ana - g_fd)))
    assert grad_err < 1e-7
    print(f"\n[2] Exact BPTT gradient check vs. central FD: max_abs_diff = {grad_err:.3e} (< 1e-7 PASS)")

    # 3. Physical symmetries: SO(2) x R_{>0}, causal future isolation, and +10 bpm equivariance
    bi36, bq36 = synth_cw_radar(36.0)
    starts = [k * 32 for k in range(5)]

    def eval_iq(ai: np.ndarray, aq: np.ndarray) -> np.ndarray:
        seq = []
        for s0 in starts:
            x0 = csd(ai[s0 : s0 + 480], aq[s0 : s0 + 480])
            dr = bandpass(x0)
            dc = bandpass(mitigate_hpss(dr)["x_hat"])
            seq.append(D.extract_salience_features_window(dr, dc, autocorr_coarse(dc), VT.BR_GRID_1BPM))
        return model.predict_session(np.asarray(seq))["est_mmse"]

    base = eval_iq(bi36, bq36)
    z_rot = (bi36 + 1j * bq36) * np.exp(1j * math.pi / 3.0)
    scale_diff = float(np.max(np.abs(base - eval_iq(3.7 * bi36, 3.7 * bq36))))
    rot_diff = float(np.max(np.abs(base - eval_iq(z_rot.real, z_rot.imag))))
    assert scale_diff < 1e-10 and rot_diff < 1e-10

    wide_grid = np.arange(-60.0, 180.0 + 1e-9, 1.0)
    pb = np.zeros((8, wide_grid.size, D.N_SAL_CHANNELS))
    ps = np.zeros((8, wide_grid.size, D.N_SAL_CHANNELS))
    for ch in range(D.N_SAL_CHANNELS):
        sw = 2.5 + 0.5 * (ch == 4)
        pb[:, :, ch] = np.log(np.exp(-0.5 * ((wide_grid[None, :] - 36.0) / sw) ** 2) + 1e-12)
        ps[:, :, ch] = np.log(np.exp(-0.5 * ((wide_grid[None, :] - 46.0) / sw) ** 2) + 1e-12)
    eq_diff = float(
        np.max(np.abs((model.predict_session(ps, wide_grid)["est_mmse"] - model.predict_session(pb, wide_grid)["est_mmse"]) - 10.0))
    )
    assert eq_diff < 1e-12

    phi_c = phi_toy.copy()
    phi_c[7:] = rng.normal(-2.0, 2.0, size=(5, VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS))
    caus_diff = float(
        np.max(np.abs(model.predict_session(phi_toy)["est_mmse"][:7] - model.predict_session(phi_c)["est_mmse"][:7]))
    )
    assert caus_diff == 0.0
    print(
        f"[3] Physical symmetries & causality:\n"
        f"    SO(2)xR+ scale/rot max diff = {max(scale_diff, rot_diff):.3e} bpm | "
        f"+10 bpm equivariance err = {eq_diff:.3e} bpm | "
        f"Causal future diff = {caus_diff:.1f}"
    )

    # 4. Live execution on preterm NICU Anchor Session 7.3 vs WGN OOD
    phi_73, cache_73 = extract_session_salience("7.3")
    valid = np.asarray(cache_73["valid"], dtype=bool)
    ref_1hz = np.asarray(cache_73["ref"], dtype=np.float64)
    r0_pred = np.asarray(cache_73["est_hpss"], dtype=np.float64)

    preds_73 = model.predict_session(phi_73)
    r2e_pred = preds_73["est_mmse"]
    rmse_r0 = float(np.sqrt(np.mean((r0_pred[valid] - ref_1hz[valid]) ** 2)))
    rmse_r2e = float(np.sqrt(np.mean((r2e_pred[valid] - ref_1hz[valid]) ** 2)))
    u_73 = UQ.extract_zero_tuned_proxies_session(phi_73, preds_73["post_entropy"], preds_73["post_sd_bpm"])

    # Synthetic white Gaussian noise (OOD)
    phi_wgn = np.empty((20, VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS), dtype=np.float64)
    for k in range(20):
        wgn = rng.normal(0.0, 1.0, size=480) + 1j * rng.normal(0.0, 1.0, size=480)
        dr = bandpass(wgn)
        dc = bandpass(mitigate_hpss(dr)["x_hat"])
        phi_wgn[k] = D.extract_salience_features_window(dr, dc, autocorr_coarse(dc), VT.BR_GRID_1BPM)
    preds_wgn = model.predict_session(phi_wgn)
    u_wgn = UQ.extract_zero_tuned_proxies_session(phi_wgn, preds_wgn["post_entropy"], preds_wgn["post_sd_bpm"])

    print(f"\n[4] Live run on NICU Anchor Session 7.3 ({int(np.sum(valid))}/{len(valid)} valid 30-s windows):")
    print(f"    1 Hz RMSE: R0 (0p HPSS+NLS) = {rmse_r0:.3f} bpm  ->  R2e (8p FilterSafe-8) = {rmse_r2e:.3f} bpm")
    print(
        f"    U6b uncertainty: mean in-dist (7.3) = {float(np.mean(u_73['U6b_ZeroTuned_Geometric_SQI'][valid])):.3f} "
        f"vs. WGN OOD = {float(np.mean(u_wgn['U6b_ZeroTuned_Geometric_SQI'])):.3f}"
    )

    # 5. Finite-K conformal theorem illustration
    k10_95, vac10_95 = UQ.finite_sample_conformal_order_index(10, 0.05)
    k19_95, vac19_95 = UQ.finite_sample_conformal_order_index(19, 0.05)
    k10_80, vac10_80 = UQ.finite_sample_conformal_order_index(10, 0.20)
    assert vac10_95 is True and vac19_95 is False and vac10_80 is False
    mean_80_width_73 = float(2.0 * c5_q_80 * np.mean(np.maximum(preds_73["post_sd_bpm"][valid], 1.0)))
    print(
        f"\n[5] Finite-K patient-exchangeable conformal order index:\n"
        f"    95% @ K_cal=10 -> k={k10_95} > 10 (vacuous={vac10_95}) | "
        f"95% @ K_cal=19 -> k={k19_95} (vacuous={vac19_95}) | "
        f"80% @ K_cal=10 -> k={k10_80} (Session 7.3 mean C5 width = {mean_80_width_73:.2f} bpm)"
    )
    print("\nSupervisor tour completed successfully.")


if __name__ == "__main__":
    main()
