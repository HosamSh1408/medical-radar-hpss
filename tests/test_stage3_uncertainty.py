"""Unit and contract tests for Stage 3 (WP3/WP4) Uncertainty, Apnea-Pause Safety & Finite-K Conformal."""
from __future__ import annotations

import json

import numpy as np

import pytest
from paper_workspace.src import config as C
from paper_workspace.src.data_loader import load_radar
from paper_workspace.src.demod import csd
from paper_workspace.src.filtering import autocorr_coarse, bandpass
from paper_workspace.src.hpss import mitigate_hpss
from thesis import diff_hmm_and_foils as D
from thesis import io_utils as IO
from thesis import viterbi_tracker as VT
from thesis import wp1_wp2_causal_hmm as W
from thesis import wp3_wp4_uncertainty_conformal as UQ


def test_spectral_entropy_hqi_and_fast_bootstrap_identities() -> None:
    """Verify uniform/Dirac spectral entropy limits, HQI bounds, empty/2D handling & bit-exact fast bootstrap."""
    n_b = VT.BR_GRID_1BPM.size
    eps0 = 1e-3

    # 1. Flat uniform spectrum -> normalized Shannon entropy == 1.0
    flat_log = np.full((5, n_b), np.log(1.0 + eps0), dtype=np.float64)
    ent_flat = UQ.spectral_shannon_entropy_from_log_salience(flat_log, eps0=eps0)
    assert np.allclose(ent_flat, 1.0, atol=1e-12)

    # 2. Dirac spike at single bin -> normalized Shannon entropy == 0.0
    dirac_p = np.zeros((4, n_b), dtype=np.float64)
    dirac_p[:, 20] = 1.0
    dirac_log = np.log(dirac_p + eps0)
    ent_dirac = UQ.spectral_shannon_entropy_from_log_salience(dirac_log, eps0=eps0)
    assert np.allclose(ent_dirac, 0.0, atol=1e-12)

    # 3. HQI on sharp fundamental + 2nd harmonic vs flat noise
    phi_sharp = np.full((3, n_b, D.N_SAL_CHANNELS), np.log(eps0), dtype=np.float64)
    phi_sharp[:, 22, 0] = np.log(1.0 + eps0)
    phi_sharp[:, 22, 1] = np.log(1.0 + eps0)
    phi_flat = np.full((3, n_b, D.N_SAL_CHANNELS), np.log(1.0 + eps0), dtype=np.float64)
    hqi_sharp = UQ.harmonic_quality_index_from_salience(phi_sharp, eps0=eps0)
    hqi_flat = UQ.harmonic_quality_index_from_salience(phi_flat, eps0=eps0)
    assert np.all(hqi_sharp > 0.55)
    assert np.all(hqi_flat < 0.10)
    assert np.all(hqi_sharp > hqi_flat)

    # 4. Empty array & 2D single-window edge cases
    assert UQ.spectral_shannon_entropy_from_log_salience(np.empty((0, n_b))).size == 0
    assert UQ.harmonic_quality_index_from_salience(np.empty((0, n_b, 6))).size == 0
    assert UQ.causal_temporal_spectral_entropy(np.empty((0, n_b, 6))).size == 0

    single_2d = phi_sharp[0]  # (N_B, 6)
    zt_1 = UQ.extract_zero_tuned_proxies_session(single_2d, np.array([0.25]), np.array([2.0]))
    assert all(v.shape == (1,) and np.all((v >= 0.0) & (v <= 1.0)) for v in zt_1.values())
    with pytest.raises(ValueError, match="must match phi_seq length"):
        UQ.extract_zero_tuned_proxies_session(single_2d, np.array([0.25, 0.30]), np.array([2.0]))

    # 5. Fast vectorized patient-clustered bootstrap bit-exact parity (< 1e-15) against IO.patient_clustered_bootstrap
    pv = {p: np.random.default_rng(100 + p).normal(-0.25, 0.15, size=(p % 4) + 1) for p in range(2, 13)}
    b_ref = IO.patient_clustered_bootstrap(pv, n_boot=10000, seed=IO.RANDOM_SEED)
    b_fast = UQ.fast_patient_clustered_bootstrap(pv, n_boot=10000, seed=IO.RANDOM_SEED)
    for k in b_ref:
        assert abs(float(b_ref[k]) - float(b_fast[k])) < 1e-15

    # Empty patient slice is safely ignored without NaN or warning
    pv_with_empty = {1: np.empty(0, dtype=np.float64), **pv}
    b_fast_empty = UQ.fast_patient_clustered_bootstrap(pv_with_empty, n_boot=10000, seed=IO.RANDOM_SEED)
    for k in b_ref:
        assert abs(float(b_ref[k]) - float(b_fast_empty[k])) < 1e-15


def test_risk_coverage_aurc_eaurc_augrc_identities_and_monotonic_oracle() -> None:
    """Verify Oracle E-AURC == 0, selective efficiency == 1, adversarial E-AURC > 0 & Traub AUGRC identity."""
    rng = np.random.default_rng(2026)
    err = rng.normal(0.0, 7.5, size=400)
    ae = np.abs(err)

    # 1. Oracle uncertainty (u == |err|) -> E-AURC_vs_oracle == 0.0, efficiency == 1.0, monotone curve
    rc_ora = UQ.compute_risk_coverage_curve(err, ae, cov_grid=UQ.COV_GRID_51)
    assert abs(float(rc_ora["e_aurc_vs_oracle_rmse"])) < 1e-12
    assert abs(float(rc_ora["e_aurc_vs_oracle_mae"])) < 1e-12
    assert abs(float(rc_ora["selective_efficiency_rmse"]) - 1.0) < 1e-12
    assert abs(float(rc_ora["selective_efficiency_mae"]) - 1.0) < 1e-12
    assert float(rc_ora["e_aurc_vs_random_rmse"]) < -2.0
    assert float(rc_ora["e_augrc_vs_random_rmse"]) < -1.0
    assert np.all(np.diff(np.asarray(rc_ora["risk_rmse"])) >= -1e-12)
    assert np.all(np.diff(np.asarray(rc_ora["risk_mae"])) >= -1e-12)

    # 2. Full coverage identity (c_cov = 1.00)
    assert abs(float(rc_ora["risk_rmse"][-1]) - float(np.sqrt(np.mean(err ** 2)))) < 1e-12
    assert abs(float(rc_ora["gen_risk_rmse"][-1]) - float(np.sqrt(np.mean(err ** 2)))) < 1e-12
    assert abs(float(rc_ora["risk_mae"][-1]) - float(np.mean(ae))) < 1e-12
    assert abs(float(rc_ora["gen_risk_mae"][-1]) - float(np.mean(ae))) < 1e-12

    # 3. Generalized Risk identity: GR(c) == (k(c)/N) * R(c) at every grid point
    n = err.size
    k_arr = np.clip(np.round(UQ.COV_GRID_51 * float(n)).astype(int), 1, n)
    eff_cov = k_arr.astype(np.float64) / float(n)
    assert np.max(np.abs(np.asarray(rc_ora["gen_risk_rmse"]) - eff_cov * np.asarray(rc_ora["risk_rmse"]))) < 1e-12
    assert np.max(np.abs(np.asarray(rc_ora["gen_risk_mae"]) - eff_cov * np.asarray(rc_ora["risk_mae"]))) < 1e-12

    # 4. Adversarial inverted uncertainty (u == -|err|) -> E-AURC_vs_random > 0
    rc_adv = UQ.compute_risk_coverage_curve(err, -ae, cov_grid=UQ.COV_GRID_51)
    assert float(rc_adv["e_aurc_vs_random_rmse"]) > 1.0
    assert float(rc_adv["selective_efficiency_rmse"]) < 0.0


def test_finite_k_conformal_vacuity_theorem_and_order_statistics() -> None:
    """Verify exact finite-sample order statistics and alpha_min = 1 / (K_cal + 1) vacuity theorem."""
    # For K_cal = 10 (pool30_sync LOPO): alpha_min = 1/11 = 0.0909...
    idx_05, vac_05 = UQ.finite_sample_conformal_order_index(10, 0.05)
    assert idx_05 == 11 and vac_05 is True

    idx_10, vac_10 = UQ.finite_sample_conformal_order_index(10, 0.10)
    assert idx_10 == 10 and vac_10 is False

    idx_20, vac_20 = UQ.finite_sample_conformal_order_index(10, 0.20)
    assert idx_20 == 9 and vac_20 is False

    idx_30, vac_30 = UQ.finite_sample_conformal_order_index(10, 0.30)
    assert idx_30 == 8 and vac_30 is False

    # For K_cal = 9 (retained23 & PAPER_18 LOPO): alpha_min = 1/10 = 0.100
    idx9_05, vac9_05 = UQ.finite_sample_conformal_order_index(9, 0.05)
    assert idx9_05 == 10 and vac9_05 is True
    idx9_10, vac9_10 = UQ.finite_sample_conformal_order_index(9, 0.10)
    assert idx9_10 == 9 and vac9_10 is False

    # Minimum K_cal required for alpha = 0.05 (95% coverage) is K_cal = 19 (K_total = 20 infants)
    _, vac18_05 = UQ.finite_sample_conformal_order_index(18, 0.05)
    idx19_05, vac19_05 = UQ.finite_sample_conformal_order_index(19, 0.05)
    assert vac18_05 is True
    assert idx19_05 == 19 and vac19_05 is False

    scores = np.arange(1.0, 11.0, dtype=np.float64)
    q_vac, _, is_v = UQ.empirical_order_statistic(scores, 0.05)
    assert is_v is True and np.isinf(q_vac)
    q_10, _, is_v10 = UQ.empirical_order_statistic(scores, 0.10)
    assert is_v10 is False and q_10 == 10.0
    q_20, _, is_v20 = UQ.empirical_order_statistic(scores, 0.20)
    assert is_v20 is False and q_20 == 9.0


def test_uncertainty_proxies_o2_scale_invariance_and_strict_causality() -> None:
    """All 8 uncertainty proxies (U1..U6c) and causal temporal entropy must be O(2)xR+ invariant and 0.0 causal."""
    bi, bq = load_radar("2.3")
    bi, bq = bi[: 480 + 4 * 32], bq[: 480 + 4 * 32]
    starts = [s for s in range(0, 5 * 32, 32)]
    grid = VT.BR_GRID_1BPM
    r2e = D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward")

    # Dummy frozen LOPO calibrator for U6c test
    cal_dummy = {
        "weights": np.array([0.05, 0.25, 0.35, 0.10, 0.25], dtype=np.float64),
        "mu_X": np.full(5, 0.35, dtype=np.float64),
        "sd_X": np.full(5, 0.15, dtype=np.float64),
        "q_lo": np.array(-1.5),
        "q_hi": np.array(1.5),
    }

    def _eval_slice(ai: np.ndarray, aq: np.ndarray) -> np.ndarray:
        phs = []
        for s0 in starts:
            dr = bandpass(csd(ai[s0 : s0 + 480], aq[s0 : s0 + 480]))
            dh5 = bandpass(W.mitigate_hpss_v2(dr, window="cola_hann", lookahead=0, in_band=True)["x_hat"])
            fc5 = autocorr_coarse(dh5)
            phs.append(D.extract_salience_features_window(dr, dh5, fc5, grid))
        pa = np.asarray(phs, dtype=np.float64)
        pr = r2e.predict_session(pa)
        zt = UQ.extract_zero_tuned_proxies_session(pa, pr["post_entropy"], pr["post_sd_bpm"])
        zt["U6c_LOPO_Calibrated_SQI"] = UQ.apply_lopo_calibrated_sqi(UQ.build_base5_feature_matrix(zt), cal_dummy)
        zt["Causal_Temporal_SpecEnt"] = UQ.causal_temporal_spectral_entropy(pa, history_hops=3)
        return np.concatenate([zt[k] for k in sorted(zt.keys())])

    base = _eval_slice(bi, bq)
    zrot = (bi + 1j * bq) * np.exp(1j * np.pi / 3.0)
    for ti, tq in ((3.7 * bi, 3.7 * bq), (zrot.real, zrot.imag), (-bi, -bq), (bq, bi)):
        assert np.max(np.abs(base - _eval_slice(ti, tq))) < 1e-11

    # Strict 0.0 causality under future perturbation t >= 12
    rng = np.random.default_rng(123)
    phi_t = rng.normal(-2.0, 1.0, size=(20, grid.size, D.N_SAL_CHANNELS))
    phi_p = phi_t.copy()
    phi_p[12:] = rng.normal(-2.0, 2.0, size=(8, grid.size, D.N_SAL_CHANNELS))
    pr_t = r2e.predict_session(phi_t)
    pr_p = r2e.predict_session(phi_p)
    zt_t = UQ.extract_zero_tuned_proxies_session(phi_t, pr_t["post_entropy"], pr_t["post_sd_bpm"])
    zt_p = UQ.extract_zero_tuned_proxies_session(phi_p, pr_p["post_entropy"], pr_p["post_sd_bpm"])
    zt_t["U6c_LOPO_Calibrated_SQI"] = UQ.apply_lopo_calibrated_sqi(UQ.build_base5_feature_matrix(zt_t), cal_dummy)
    zt_p["U6c_LOPO_Calibrated_SQI"] = UQ.apply_lopo_calibrated_sqi(UQ.build_base5_feature_matrix(zt_p), cal_dummy)
    zt_t["Causal_Temporal_SpecEnt"] = UQ.causal_temporal_spectral_entropy(phi_t, history_hops=3)
    zt_p["Causal_Temporal_SpecEnt"] = UQ.causal_temporal_spectral_entropy(phi_p, history_hops=3)
    for k in zt_t:
        assert np.max(np.abs(zt_t[k][:12] - zt_p[k][:12])) == 0.0


def test_stage3_json_contract_stage1_stage2_parity_and_audit_gates() -> None:
    """Verify Stage 3 JSON artifact (`stage3-uncertainty-v1`), Stage 1.1/2.2 parity, Gate G4, and 7/7 audit pass."""
    s1_path = IO.BUNDLE_DIR / "metrics_stage1_parsimony.json"
    s2_path = IO.BUNDLE_DIR / "metrics_stage2_complexity.json"
    s3_path = IO.BUNDLE_DIR / "metrics_stage3_uncertainty.json"
    a3_path = IO.AUDIT_DIR / "stage3_godmode_audit.json"
    assert s3_path.exists(), f"Run thesis.exp3_stage3_uncertainty first: missing {s3_path}"
    assert a3_path.exists(), f"Run thesis.audit_stage3_uncertainty first: missing {a3_path}"

    s1 = json.loads(s1_path.read_text(encoding="utf-8"))
    s2 = json.loads(s2_path.read_text(encoding="utf-8"))
    s3 = json.loads(s3_path.read_text(encoding="utf-8"))
    a3 = json.loads(a3_path.read_text(encoding="utf-8"))

    assert s3["schema_version"] == "stage3-uncertainty-v1"
    assert s3["full_coverage_parity_checks"]["stage1_full_cov_max_diff"] < 1e-12
    assert s3["full_coverage_parity_checks"]["stage2_full_cov_max_diff"] < 1e-12
    assert s3["gate_g4_verdict"]["decision"] == "PASS_AND_HARD_FREEZE_WP1_TO_WP4"
    assert s3["gate_g4_verdict"]["primary_selective_prediction_gain_lt_0"] is True
    assert s3["gate_g4_verdict"]["wgn_auroc_ge_0p95"] is True
    assert s3["gate_g4_verdict"]["apnea_pause_safety_verified"] is True
    assert s3["gate_g4_verdict"]["finite_k_conformal_theorem_verified"] is True
    assert s3["verification_controls"]["low_amp_synthetic_zero_false_ood"] is True

    hf = s3["gate_g4_verdict"]["hard_freeze_manifest_for_stage4_l4"]
    assert hf["frozen_c4_q_hat_bpm_pool30_1hz"]["0.05"] is None
    assert hf["frozen_c5_q_hat_norm_pool30_1hz"]["0.05"] is None
    assert float(hf["frozen_c4_q_hat_bpm_pool30_1hz"]["0.20"]) > 5.0
    assert float(hf["frozen_c5_q_hat_norm_pool30_1hz"]["0.20"]) > 1.0
    assert hf["wp2_hpss_primary_lookahead"] == "48 samples (3.0 s lookahead lag at 16 Hz; L_H = 3 hops)"
    assert hf["stage4_l4_evaluation_protocol"]["primary_inband_stratum_bpm"] == [28.0, 80.0]

    f4_path = IO.ROOT / hf["frozen_stage4_models_archive"]
    assert f4_path.exists()
    assert IO.sha256_file(f4_path) == hf["frozen_stage4_models_sha256"]
    assert len(hf["frozen_r2e_full_refit_theta"]) == 8
    assert len(hf["frozen_r2a_full_refit_theta"]) == 11
    assert len(hf["frozen_r2d_full_refit_theta"]) == 11
    assert hf["frozen_r3c_ridge_439_summary"]["n_params"] == 439
    assert hf["frozen_r3b_mlp_15k_summary"]["n_params"] > 10000
    assert hf["frozen_r3a_tcn_16k_summary"]["n_params"] == 15817

    for coh in ("pool30_sync", "retained23", "PAPER_18"):
        for rlbl in ("br_1hz_30s", "br_trig_bpm"):
            m1_r0 = s1["rung1_lopo"][coh][rlbl]["methods"]["R0_HPSS_NLS_Locked"]
            m3_r0 = s3["selective_prediction_by_cohort"][coh][rlbl]["R0_HPSS_NLS_Locked"]["U3_Spectral_Entropy_Clean"]
            assert abs(float(m3_r0["session_macro"]["full_risk_rmse_sess"]) - float(m1_r0["rmse_sess"])) < 1e-12
            assert abs(float(m3_r0["window_pooled"]["full_risk_rmse"]) - float(m1_r0["rmse_pool"])) < 1e-12
        for rlbl in ("br_1hz_30s", "br_trig_bpm", "br_trig_bpm_lag_aligned"):
            for km in ("R0_HPSS_NLS_Locked", "R2e_Causal_DiffHMM_FilterSafe8", "R2a_Causal_DiffHMM_Linear11", "R2d_E2E_Causal_H5_DiffHMM_Linear11"):
                m2 = s2["cohorts"][coh][rlbl]["methods"][km]
                px_chk = "U3_Spectral_Entropy_Clean" if km == "R0_HPSS_NLS_Locked" else "U2_Posterior_SD"
                m3 = s3["selective_prediction_by_cohort"][coh][rlbl][km][px_chk]
                assert abs(float(m3["session_macro"]["full_risk_rmse_sess"]) - float(m2["rmse_sess"])) < 1e-12
                assert abs(float(m3["window_pooled"]["full_risk_rmse"]) - float(m2["rmse_pool"])) < 1e-12

    assert a3["all_passed"] is True
    assert all(g["passed"] for g in a3["gates"].values())
    assert a3["gates"]["gate7_gate_g4_hard_freeze_and_json_contract"]["frozen_stage4_sha256_match"] is True
    assert a3["gates"]["gate4_cleanroom_ood_wgn_abstention"]["max_surrogate_auroc_diff"] < 1e-12
    assert a3["gates"]["gate4_cleanroom_ood_wgn_abstention"]["max_ucd_auroc_diff"] < 1e-12
    assert a3["gates"]["gate4_cleanroom_ood_wgn_abstention"]["ucd_window_counts_verified"] is True
    assert a3["gates"]["gate5_cleanroom_finite_k_conformal"]["cross_cohort_vacuity_verified"] is True
    assert a3["gates"]["gate6_invariance_causality_and_lopo_disjointness"]["max_low_amp_0p2mm_b_ood"] == 0.0
