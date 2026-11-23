"""Unit and contract tests for Stage 1 Parsimony, Causality & Literature Parity."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from paper_workspace.src.data_loader import load_radar
from paper_workspace.src.demod import csd
from paper_workspace.src.estimation import nls_estimate
from paper_workspace.src.filtering import autocorr_coarse, bandpass
from thesis import io_utils as IO
from thesis import viterbi_tracker as VT
from thesis import wp1_wp2_causal_hmm as W


def test_cola_hann_stft_exact_reconstruction() -> None:
    """COLA Hann analysis -> LSEE-MSTFTM iSTFT must reconstruct arbitrary 480-sample signals."""
    rng = np.random.default_rng(2026)
    x = rng.standard_normal(480) + 1j * rng.standard_normal(480)
    X, _, _, pad_l = W.stft_complex_windowed(x, window="cola_hann", pad_edges=True)
    rec = W.istft_complex_windowed(X, len(x), window="cola_hann", pad_left=pad_l)
    assert np.max(np.abs(rec - x)) < 1e-13

    # Also verify non-480 arbitrary signal length
    x2 = rng.standard_normal(513) + 1j * rng.standard_normal(513)
    X2, _, _, pad_l2 = W.stft_complex_windowed(x2, window="cola_hann", pad_edges=True)
    rec2 = W.istft_complex_windowed(X2, len(x2), window="cola_hann", pad_left=pad_l2)
    assert np.max(np.abs(rec2 - x2)) < 1e-13


def test_hpss_lookahead_and_transition_identities() -> None:
    """Verify centered HPSS median parity, transition row stochasticity, and lag limits."""
    rng = np.random.default_rng(17)
    mag = rng.uniform(0.1, 5.0, (256, 32))
    filt, _, s_h, _ = W.hpss_filter_causal(mag, l_harm=7, l_perc=21, lookahead=3, in_band=False)
    ref_sh = np.fft.ifftshift(median_filter(np.fft.fftshift(mag, axes=0), size=(1, 7), mode="nearest"), axes=0)
    assert np.max(np.abs(s_h - ref_sh)) < 1e-12
    assert np.all(np.isfinite(filt))

    # Transition matrix stochasticity
    for sig in (0.5, 1.0, 2.0, 8.0):
        log_T = W.build_gaussian_log_transition(VT.BR_GRID_1BPM, sigma_br=sig)
        row_sums = np.sum(np.exp(log_T), axis=1)
        assert np.max(np.abs(row_sums - 1.0)) < 1e-13

    # Fixed lag extremes: lag=0 == forward filter, lag>=T == offline FB, and scaled == logsumexp
    S = rng.random((24, VT.BR_GRID_1BPM.size))
    S[10, :] = 0.0  # degenerate flat frame
    fwd = W.hmm_forward_filter(S, sigma_br=1.5, eps=1e-2, beta=1.5)
    lag0 = W.hmm_fixed_lag_smoother(S, lag_hops=0, sigma_br=1.5, eps=1e-2, beta=1.5)
    lag_inf = W.hmm_fixed_lag_smoother(S, lag_hops=100, sigma_br=1.5, eps=1e-2, beta=1.5)
    fb = W.hmm_offline_fb_smoother(S, sigma_br=1.5, eps=1e-2, beta=1.5)

    assert np.max(np.abs(lag0["est_mmse"] - fwd["est_mmse"])) < 1e-12
    assert np.max(np.abs(lag_inf["est_mmse"] - fb["est_mmse"])) < 1e-12

    # Log-space exact path (eps=1e-13 < 1e-12 triggers logsumexp) AND scaled-vs-logsumexp parity
    fwd_log = W.hmm_forward_filter(S, sigma_br=1.0, eps=1e-13, beta=2.0)
    assert np.all(np.isfinite(fwd_log["est_mmse"]))
    assert np.all((fwd_log["post_entropy"] >= 0.0) & (fwd_log["post_entropy"] <= 1.0))

    orig_can_scale = W._can_use_scaled_hmm
    f_s = W.hmm_forward_filter(S, sigma_br=0.5, eps=1e-4, beta=4.0)
    l_s = W.hmm_fixed_lag_smoother(S, lag_hops=4, sigma_br=0.5, eps=1e-4, beta=4.0)
    b_s = W.hmm_offline_fb_smoother(S, sigma_br=0.5, eps=1e-4, beta=4.0)
    try:
        W._can_use_scaled_hmm = lambda e, b, n: False
        f_l = W.hmm_forward_filter(S, sigma_br=0.5, eps=1e-4, beta=4.0)
        l_l = W.hmm_fixed_lag_smoother(S, lag_hops=4, sigma_br=0.5, eps=1e-4, beta=4.0)
        b_l = W.hmm_offline_fb_smoother(S, sigma_br=0.5, eps=1e-4, beta=4.0)
    finally:
        W._can_use_scaled_hmm = orig_can_scale

    assert np.max(np.abs(f_s["est_mmse"] - f_l["est_mmse"])) < 1e-12
    assert np.max(np.abs(f_s["post_entropy"] - f_l["post_entropy"])) < 1e-12
    assert np.max(np.abs(f_s["post_sd_bpm"] - f_l["post_sd_bpm"])) < 1e-12
    assert np.max(np.abs(f_s["log_evidence"] - f_l["log_evidence"])) < 1e-12
    assert np.max(np.abs(l_s["est_mmse"] - l_l["est_mmse"])) < 1e-12
    assert np.max(np.abs(b_s["est_mmse"] - b_l["est_mmse"])) < 1e-12


def test_strict_causality_perturbation() -> None:
    """Perturbing future inputs past lookahead must cause 0.0 diff on causal outputs."""
    rng = np.random.default_rng(99)
    mag = rng.uniform(0.1, 3.0, (256, 32))
    t0 = 15
    for la in (0, 1, 2):
        mag_p = mag.copy()
        mag_p[:, t0 + la + 1 :] += 100.0
        f_a, m_a, sh_a, sp_a = W.hpss_filter_causal(mag, lookahead=la, in_band=True)
        f_b, m_b, sh_b, sp_b = W.hpss_filter_causal(mag_p, lookahead=la, in_band=True)
        assert np.max(np.abs(f_a[:, : t0 + 1] - f_b[:, : t0 + 1])) == 0.0
        assert np.max(np.abs(m_a[:, : t0 + 1] - m_b[:, : t0 + 1])) == 0.0
        assert np.max(np.abs(sh_a[:, : t0 + 1] - sh_b[:, : t0 + 1])) == 0.0
        assert np.max(np.abs(sp_a[:, : t0 + 1] - sp_b[:, : t0 + 1])) == 0.0

    S = rng.random((35, VT.BR_GRID_1BPM.size))
    S_p = S.copy()
    S_p[20:] = rng.random((15, VT.BR_GRID_1BPM.size))
    assert np.max(np.abs(W.hmm_forward_filter(S)["est_mmse"][:20] - W.hmm_forward_filter(S_p)["est_mmse"][:20])) == 0.0
    for lag in (2, 4, 8):
        a = W.hmm_fixed_lag_smoother(S, lag_hops=lag)["est_mmse"][: 20 - lag]
        b = W.hmm_fixed_lag_smoother(S_p, lag_hops=lag)["est_mmse"][: 20 - lag]
        assert np.max(np.abs(a - b)) == 0.0


def test_env_normalize_causal_and_block_causality_distinction() -> None:
    """Verify env_norm='causal' eliminates intra-block future leakage and preserves O(2) x R+."""
    bi, bq = load_radar("7.3")
    dr0 = bandpass(csd(bi[4800:5280], bq[4800:5280]))
    t_ax = np.arange(480, dtype=np.float64) / 16.0
    dr_p = dr0.copy()
    dr_p[240:] += 50.0 * np.exp(1j * 2.0 * np.pi * 0.8 * t_ax[240:])

    h_ca = W.mitigate_hpss_v2(dr0, window="cola_hann", lookahead=0, in_band=True, env_norm="causal")
    h_cb = W.mitigate_hpss_v2(dr_p, window="cola_hann", lookahead=0, in_band=True, env_norm="causal")
    assert np.max(np.abs(h_ca["mag_filt"][:, :14] - h_cb["mag_filt"][:, :14])) == 0.0

    # Block-level median env_norm couples earlier frames in the same 30s block
    h_ba = W.mitigate_hpss_v2(dr0, window="cola_hann", lookahead=0, in_band=True, env_norm="block")
    h_bb = W.mitigate_hpss_v2(dr_p, window="cola_hann", lookahead=0, in_band=True, env_norm="block")
    assert np.max(np.abs(h_ba["mag_filt"][:, :14] - h_bb["mag_filt"][:, :14])) > 1.0

    # O(2) x R+ invariance of env_normalize_causal inside mitigate_hpss_v2
    wi, wq = bi[4800:5280], bq[4800:5280]
    def _eval_c(ai: np.ndarray, aq: np.ndarray) -> np.ndarray:
        dr = bandpass(csd(ai, aq))
        dh = bandpass(W.mitigate_hpss_v2(dr, window="cola_hann", lookahead=0, in_band=True, env_norm="causal")["x_hat"])
        return np.array([nls_estimate(dh, autocorr_coarse(dh))[0]], dtype=np.float64)

    ref_v = _eval_c(wi, wq)
    zr = (wi + 1j * wq) * np.exp(1j * 1.23)
    for ti, tq in ((3.7 * wi, 3.7 * wq), (zr.real, zr.imag), (-wi, -wq), (wq, wi)):
        assert np.max(np.abs(ref_v - _eval_c(ti, tq))) < 1e-12


def test_group_invariances_stage1_operators() -> None:
    """F2 ZCA, F1b (fixed hardware mask), WP2 H1..H6 & Rung 1 HMM must be O(2)xR+ invariant."""
    bi, bq = load_radar("2.3")
    bi = bi[: 480 + 5 * 32]
    bq = bq[: 480 + 5 * 32]
    starts = [s for s in range(0, 6 * 32, 32)]
    _, _, clip_any = W.detect_adc_rail_mask(bi, bq)

    def _run(arr_i: np.ndarray, arr_q: np.ndarray) -> dict[str, np.ndarray]:
        f2_out, f1b_out, h1_out, h5_out, s5_list = [], [], [], [], []
        for s0 in starts:
            wi, wq, cw = arr_i[s0 : s0 + 480], arr_q[s0 : s0 + 480], clip_any[s0 : s0 + 480]
            xw = W.csd_sym_whitened(wi, wq, lam=0.10)
            dw = bandpass(W.mitigate_hpss_v2(bandpass(xw), window="rect", lookahead=3, in_band=False)["x_hat"])
            f2_out.append(nls_estimate(dw, autocorr_coarse(dw))[0])

            xm = W.csd_clip_masked(wi, wq, mode="joint", clip_mask=cw)
            dm = bandpass(W.mitigate_hpss_v2(bandpass(xm), window="rect", lookahead=3, in_band=False)["x_hat"])
            f1b_out.append(nls_estimate(dm, autocorr_coarse(dm))[0])

            dr0 = bandpass(csd(wi, wq))
            dh1 = bandpass(W.mitigate_hpss_v2(dr0, window="cola_hann", lookahead=3, in_band=False)["x_hat"])
            h1_out.append(nls_estimate(dh1, autocorr_coarse(dh1))[0])

            dh5 = bandpass(W.mitigate_hpss_v2(dr0, window="cola_hann", lookahead=0, in_band=True)["x_hat"])
            fc5 = autocorr_coarse(dh5)
            h5_out.append(nls_estimate(dh5, fc5)[0])
            s5_list.append(VT.extract_br_anchored_nls_spectrum(dh5, fc5, VT.BR_GRID_1BPM))

        s5 = np.asarray(s5_list)
        return {
            "f2": np.asarray(f2_out),
            "f1b": np.asarray(f1b_out),
            "h1": np.asarray(h1_out),
            "h5": np.asarray(h5_out),
            "fwd": W.hmm_forward_filter(s5)["est_mmse"],
            "vit": W.hmm_offline_viterbi_prob(s5)[0],
        }

    base = _run(bi, bq)
    z_rot = (bi + 1j * bq) * np.exp(1j * 1.23)
    for ti, tq in (
        (3.7 * bi, 3.7 * bq),
        (z_rot.real, z_rot.imag),
        (-bi, -bq),
        (bq, bi),
    ):
        alt = _run(ti, tq)
        for k in base:
            assert np.max(np.abs(base[k] - alt[k])) < 1e-6


def test_bibliography_and_lit_matrix_parity() -> None:
    """Ensure exact 71-entry 1-to-1 bib_key parity between seed_references.bib and lit_review_matrix.csv."""
    bib_text = (IO.BUNDLE_DIR / "seed_references.bib").read_text(encoding="utf-8")
    bib_keys = re.findall(r"^@\w+\{([^,]+),", bib_text, flags=re.M)
    with (IO.BUNDLE_DIR / "lit_review_matrix.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    csv_keys = [r["bib_key"] for r in rows]

    expected_new = {
        "krbec2024emerging",
        "schellenberger2020dataset",
        "mensch2018differentiable",
        "bittner2017deep",
        "riou2023pesto",
        "geifman2019selectivenet",
        "geirhos2020shortcut",
        "dunn2023distribution",
        "barber2023conformal",
        "gibbs2024conformal",
        "angelopoulos2024conformal",
    }
    assert len(bib_keys) == len(set(bib_keys)) == 71
    assert len(csv_keys) == len(set(csv_keys)) == 71
    assert set(bib_keys) == set(csv_keys)
    assert expected_new.issubset(set(bib_keys))


def test_stage1_json_schema_parity_and_lopo_disjointness() -> None:
    """Verify Stage 1.1 JSON export, F0==est_hpss parity, dual references, lag audit, and strict LOPO."""
    jpath = IO.BUNDLE_DIR / "metrics_stage1_parsimony.json"
    assert jpath.exists(), f"Run thesis.exp1_stage1_parsimony first: missing {jpath}"
    data = json.loads(jpath.read_text(encoding="utf-8"))

    assert data["schema_version"] == "stage1-parsimony-v3"
    assert data["f0_vs_locked_cache_max_diff_bpm"] < 1e-9
    assert data["gate_g1_verdict"]["decision"] == "RETAIN_F0_CSD_LOCKED"
    assert data["gate_g2_verdict"]["decision"] == "RETAIN_H0_LOCKED"
    assert data["shifted_grid_midpoint_audit"]["fwd_mmse_equivariant_b1_e1e3_sig2"]["abs_bias_shift_bpm"] < 0.05
    assert data["shifted_grid_midpoint_audit"]["memoryless_mmse_naive_eps001"]["abs_bias_shift_bpm"] > 1.0
    assert data["invariance_and_causality_checks"]["all_new_invariances_lt_1e6"] is True
    assert data["invariance_and_causality_checks"]["causality_hpss_l0_max_diff"] == 0.0
    assert data["invariance_and_causality_checks"]["causality_mitigate_v2_causal_env_mag_filt_max_diff"] == 0.0
    assert data["invariance_and_causality_checks"]["causality_mitigate_v2_block_env_mag_filt_diag_diff"] > 1.0
    assert data["invariance_and_causality_checks"]["causality_hmm_fwd_max_diff"] == 0.0
    assert data["invariance_and_causality_checks"]["causality_hmm_lag4_max_diff"] == 0.0

    # Two-axis causality contract & lag audit & R1f surrogate entry
    cc = data["rung1_causality_contract"]
    assert cc["R0_HPSS_NLS_Locked"]["block_lookahead_s"] == 0.0
    assert cc["R0_HPSS_NLS_Locked"]["stft_lookahead_s"] == 3.0
    assert cc["R1a_Online_Forward_MMSE_0s"]["block_lookahead_s"] == 0.0
    assert cc["R1a_Online_Forward_MMSE_0s"]["stft_lookahead_s"] == 3.0
    assert cc["R1h_E2E_Causal_H5_OnlineForward_0s"]["stft_lookahead_s"] == 0.0
    assert data["dual_reference_lag_and_pause_audit"]["lag_profiles_rmcorr"]["br_1hz_30s_vs_br_trig_bpm"]["peak_lag_s"] >= 6
    assert "R1f_Offline_FB_MMSE_Inf" in data["phase_scrambled_surrogates_pool30"]["br_1hz_30s"]
    assert "R1f_Offline_FB_MMSE_Inf" in data["phase_scrambled_surrogates_pool30"]["br_trig_bpm"]

    # Check dual references populated for pool30_sync
    for ref_key in ("br_1hz_30s", "br_trig_bpm"):
        assert ref_key in data["wp1_ablation"]["pool30_sync"]
        assert ref_key in data["wp2_ablation"]["pool30_sync"]
        assert ref_key in data["rung1_lopo"]["pool30_sync"]
        assert data["rung1_lopo"]["pool30_sync"][ref_key]["methods"]["R0_HPSS_NLS_Locked"]["n_sessions"] == 28

    # Check LOPO fold disjointness on pool30_sync (11 patients)
    flogs = data["rung1_lopo"]["pool30_sync"]["lopo_fold_summary"]["R1a_Online_Forward_MMSE_0s"]["folds"]
    assert len(flogs) == 11
    pids = [f["patient_id"] for f in flogs]
    assert len(set(pids)) == 11
    for f in flogs:
        for tm in f["test_sessions"]:
            assert IO.patient_id_of(tm) == f["patient_id"]
