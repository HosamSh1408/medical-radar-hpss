"""Unit and contract tests for Stage 2 Complexity Ladder, Differentiable 1D-HMM & Foils."""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import Ridge

from paper_workspace.src.data_loader import load_radar
from paper_workspace.src.demod import csd
from paper_workspace.src.filtering import autocorr_coarse, bandpass
from thesis import diff_hmm_and_foils as D
from thesis import diff_hmm_batch as DB
from thesis import io_utils as IO
from thesis import viterbi_tracker as VT
from thesis import wp1_wp2_causal_hmm as W


def test_r2_differentiable_hmm_finite_diff_gradients_and_bounds() -> None:
    """Analytical BPTT gradients must match central finite differences (< 1e-6) and enforce sigma_slow >= 1.0."""
    rng = np.random.default_rng(2026)
    grid = VT.BR_GRID_1BPM
    phi = rng.normal(-2.0, 0.8, size=(10, grid.size, D.N_SAL_CHANNELS))
    ref = rng.uniform(25.0, 60.0, size=10)
    vm = np.array([True] * 9 + [False], dtype=bool)
    sess = [(phi, ref, vm)]

    for mode in ("linear", "linear_filtersafe", "mlp"):
        for dec in ("causal_forward", "offline_fb"):
            mdl = D.DifferentiableSymmetricHMM(mode=mode, decoder=dec, grid_bpm=grid, reg_l2=5e-3, seed=11)
            if mode == "linear_filtersafe":
                assert mdl.n_params == 8
                ip = mdl.interpretable_params()
                assert ip["w2_p2_harm"] == 0.0 and ip["w3_p3_harm"] == 0.0 and ip["w4_phalf_sub"] == 0.0
            assert np.isclose(mdl.bounds[0][0], 2.0)
            assert mdl.bounds[-5] == (0.0, 0.5) and mdl.bounds[-4] == (0.0, 0.5)
            assert np.isclose(np.exp(mdl.bounds[-3][0]), 1.0)
            assert np.isclose(mdl.bounds[-1][1], -4.5)
            th = mdl.theta + rng.normal(0.0, 0.04, size=mdl.theta.size)
            th = np.clip(th, [b[0] + 1e-3 for b in mdl.bounds], [b[1] - 1e-3 for b in mdl.bounds])
            _, g_ana = mdl.loss_and_grad(th, sess)
            h = 1e-6
            for i in range(th.size):
                tp = th.copy()
                tp[i] += h
                tm = th.copy()
                tm[i] -= h
                lp, _ = mdl.loss_and_grad(tp, sess)
                lm, _ = mdl.loss_and_grad(tm, sess)
                assert abs(float(g_ana[i]) - float((lp - lm) / (2.0 * h))) < 1e-6

    # Variable-length batched BPTT exact parity (< 1e-12) across all 4 mode/decoder combinations
    sess_var = [
        (rng.normal(-2.0, 0.8, size=(7, grid.size, D.N_SAL_CHANNELS)), rng.uniform(25.0, 60.0, size=7), np.array([False, True, True, True, False, True, False])),
        (rng.normal(-2.0, 0.8, size=(12, grid.size, D.N_SAL_CHANNELS)), rng.uniform(25.0, 60.0, size=12), np.array([True] * 10 + [False, True])),
        (rng.normal(-2.0, 0.8, size=(9, grid.size, D.N_SAL_CHANNELS)), rng.uniform(25.0, 60.0, size=9), np.array([True] * 9)),
    ]
    batch_var = DB.prepare_hmm_batch(sess_var)
    for mode in ("linear", "linear_filtersafe", "mlp"):
        for dec in ("causal_forward", "offline_fb"):
            mdl = D.DifferentiableSymmetricHMM(mode=mode, decoder=dec, grid_bpm=grid, reg_l2=5e-3, seed=29)
            th = np.clip(mdl.theta + rng.normal(0.0, 0.05, size=mdl.theta.size), [b[0] + 1e-3 for b in mdl.bounds], [b[1] - 1e-3 for b in mdl.bounds])
            l_ref, g_ref = mdl.loss_and_grad(th, sess_var)
            l_b, g_b = DB.batch_loss_and_grad(mdl, th, batch_var)
            assert abs(l_b - l_ref) < 1e-12
            assert np.max(np.abs(g_b - g_ref)) < 1e-12


def test_r2_symmetry_causality_shift_equivariance_and_arbitrary_grid() -> None:
    """Verify O(2) x R+ invariance, strict 0.0 causality, +10 bpm equivariance & [18, 120] arbitrary grid."""
    bi, bq = load_radar("2.3")
    bi, bq = bi[: 480 + 4 * 32], bq[: 480 + 4 * 32]
    starts = [s for s in range(0, 5 * 32, 32)]
    grid = VT.BR_GRID_1BPM

    r2e = D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward")
    r2a = D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward")
    r2b = D.DifferentiableSymmetricHMM(mode="mlp", decoder="causal_forward", seed=5)

    def _eval_slice(ai: np.ndarray, aq: np.ndarray) -> np.ndarray:
        phs = []
        for s0 in starts:
            dr = bandpass(csd(ai[s0 : s0 + 480], aq[s0 : s0 + 480]))
            dh5 = bandpass(W.mitigate_hpss_v2(dr, window="cola_hann", lookahead=0, in_band=True)["x_hat"])
            fc5 = autocorr_coarse(dh5)
            phs.append(D.extract_salience_features_window(dr, dh5, fc5, grid))
        pa = np.asarray(phs)
        return np.concatenate([r2e.predict_session(pa)["est_mmse"], r2a.predict_session(pa)["est_mmse"], r2b.predict_session(pa)["est_mmse"]])

    base = _eval_slice(bi, bq)
    zrot = (bi + 1j * bq) * np.exp(1j * np.pi / 3.0)
    for ti, tq in ((3.7 * bi, 3.7 * bq), (zrot.real, zrot.imag), (-bi, -bq), (bq, bi)):
        assert np.max(np.abs(base - _eval_slice(ti, tq))) < 1e-10

    # Strict 0.0 causality
    rng = np.random.default_rng(91)
    phi_t = rng.normal(-2.0, 1.0, size=(20, grid.size, D.N_SAL_CHANNELS))
    phi_p = phi_t.copy()
    phi_p[12:] = rng.normal(-2.0, 2.0, size=(8, grid.size, D.N_SAL_CHANNELS))
    assert np.max(np.abs(r2e.predict_session(phi_t)["est_mmse"][:12] - r2e.predict_session(phi_p)["est_mmse"][:12])) == 0.0
    assert np.max(np.abs(r2a.predict_session(phi_t)["est_mmse"][:12] - r2a.predict_session(phi_p)["est_mmse"][:12])) == 0.0
    assert np.max(np.abs(r2b.predict_session(phi_t)["est_mmse"][:12] - r2b.predict_session(phi_p)["est_mmse"][:12])) == 0.0

    # Operator +10 bpm shift equivariance on wide grid [-60, 180]
    wide_grid = np.arange(-60.0, 180.0 + 1e-9, 1.0)
    pb_w = np.zeros((10, wide_grid.size, D.N_SAL_CHANNELS))
    ps_w = np.zeros((10, wide_grid.size, D.N_SAL_CHANNELS))
    for ch in range(D.N_SAL_CHANNELS):
        sw = 2.5 + 0.5 * (ch == 4)
        pb_w[:, :, ch] = np.log(np.exp(-0.5 * ((wide_grid[None, :] - 40.0) / sw) ** 2) + 1e-12)
        ps_w[:, :, ch] = np.log(np.exp(-0.5 * ((wide_grid[None, :] - 50.0) / sw) ** 2) + 1e-12)
    assert np.max(np.abs((r2e.predict_session(ps_w, grid_bpm=wide_grid)["est_mmse"] - r2e.predict_session(pb_w, grid_bpm=wide_grid)["est_mmse"]) - 10.0)) < 1e-9
    assert np.max(np.abs((r2a.predict_session(ps_w, grid_bpm=wide_grid)["est_mmse"] - r2a.predict_session(pb_w, grid_bpm=wide_grid)["est_mmse"]) - 10.0)) < 1e-9
    assert np.max(np.abs((r2b.predict_session(ps_w, grid_bpm=wide_grid)["est_mmse"] - r2b.predict_session(pb_w, grid_bpm=wide_grid)["est_mmse"]) - 10.0)) < 1e-9

    # Arbitrary [18, 120] grid check
    grid120 = np.arange(18.0, 120.0 + 1e-9, 1.0)
    phi120 = rng.normal(-2.0, 0.5, size=(8, grid120.size, D.N_SAL_CHANNELS))
    pr120 = r2a.predict_session(phi120, grid_bpm=grid120)["est_mmse"]
    assert pr120.shape == (8,) and np.all(np.isfinite(pr120))


def test_causal_tcn_gradients_causality_dropout_and_selection() -> None:
    """Verify TCN analytical gradients, strict causality, deterministic dropout & grid selection."""
    rng = np.random.default_rng(13)
    xs = rng.normal(0.0, 1.0, size=(8, 438))
    ref = rng.uniform(25.0, 60.0, size=8)
    vm = np.array([True] * 7 + [False], dtype=bool)
    tcn = D.CausalDilatedTCN(in_dim=438, channels=24, dilations=(1, 2, 4), seed=7)

    _, g_tcn = tcn.session_mse_loss_and_grad([(xs, ref, vm)], weight_decay=1e-2, dropout_p=0.0, rng=None)
    for k_p in ("Wo", "bo", "Wk_0", "Wk_2", "W0", "b0"):
        flat = tcn.params[k_p].ravel()
        gf = g_tcn[k_p].ravel()
        for idx in rng.choice(flat.size, size=min(8, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + 1e-6
            lp, _ = tcn.session_mse_loss_and_grad([(xs, ref, vm)], weight_decay=1e-2, dropout_p=0.0, rng=None)
            flat[idx] = old - 1e-6
            lm, _ = tcn.session_mse_loss_and_grad([(xs, ref, vm)], weight_decay=1e-2, dropout_p=0.0, rng=None)
            flat[idx] = old
            assert abs(float(gf[idx]) - float((lp - lm) / 2e-6)) < 1e-6

    # Strict 0.0 causality
    phi_t = rng.normal(-2.0, 1.0, size=(16, VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS))
    phi_p = phi_t.copy()
    phi_p[10:] = rng.normal(-2.0, 2.0, size=(6, VT.BR_GRID_1BPM.size, D.N_SAL_CHANNELS))
    assert np.max(np.abs(tcn.predict_session(phi_t)[:10] - tcn.predict_session(phi_p)[:10])) == 0.0

    # Deterministic dropout reproducibility & difference vs dropout=0
    tr_s = [(phi_t, np.linspace(30.0, 45.0, 16), np.ones(16, dtype=bool)) for _ in range(2)]
    va_s = [(phi_t, np.linspace(32.0, 42.0, 16), np.ones(16, dtype=bool))]
    t1 = D.CausalDilatedTCN(in_dim=438, channels=24, dilations=(1, 2, 4), seed=41).fit(
        tr_s, va_s, lr=1e-3, weight_decay=1e-2, dropout_p=0.25, max_epochs=5, patience=5
    )
    t2 = D.CausalDilatedTCN(in_dim=438, channels=24, dilations=(1, 2, 4), seed=41).fit(
        tr_s, va_s, lr=1e-3, weight_decay=1e-2, dropout_p=0.25, max_epochs=5, patience=5
    )
    t0 = D.CausalDilatedTCN(in_dim=438, channels=24, dilations=(1, 2, 4), seed=41).fit(
        tr_s, va_s, lr=1e-3, weight_decay=1e-2, dropout_p=0.0, max_epochs=5, patience=5
    )
    assert np.max(np.abs(t1.predict_session(phi_t) - t2.predict_session(phi_t))) == 0.0
    assert np.max(np.abs(t1.predict_session(phi_t) - t0.predict_session(phi_t))) > 1e-6

    sel_mdl, sel_info = D.fit_select_causal_tcn(
        tr_s, va_s, lr=1e-3, grid=((1e-3, 0.0), (1e-2, 0.2)), max_epochs=5, patience=5, seed=41
    )
    assert len(sel_info["grid_val_rmses"]) == 2
    assert np.isfinite(sel_mdl.best_val_rmse_)


def test_ridge_svd_inner_lopo_and_mlp_alpha_selection() -> None:
    """Verify exact SVD weighted Ridge parity vs sklearn.Ridge and patient-LOPO / MLP selection."""
    rng = np.random.default_rng(101)
    xr = rng.normal(0.0, 1.0, size=(40, 20))
    yr = rng.normal(38.0, 5.0, size=40)
    wr = rng.uniform(0.4, 2.5, size=40)
    for a_v, (coef_s, inter_s) in zip((0.1, 10.0, 1000.0), D._svd_weighted_ridge_paths(xr, yr, wr, np.array([0.1, 10.0, 1000.0]))):
        sk = Ridge(alpha=a_v).fit(xr, yr, sample_weight=wr)
        assert np.max(np.abs(coef_s - sk.coef_)) < 1e-10
        assert abs(float(inter_s) - float(sk.intercept_)) < 1e-10

    n_b = VT.BR_GRID_1BPM.size
    tr_sess = [
        (rng.normal(-2.0, 1.0, size=(20, n_b, D.N_SAL_CHANNELS)), rng.uniform(28.0, 52.0, size=20), np.ones(20, dtype=bool))
        for _ in range(3)
    ]
    te_phi = [rng.normal(-2.0, 1.0, size=(10, n_b, D.N_SAL_CHANNELS))]
    ridge_res = D.fit_predict_ridge_spectral(tr_sess, te_phi, train_patient_ids=[2, 3, 4], alphas=np.array([1.0, 100.0, 1e4]))
    assert ridge_res["inner_cv_mode"] == "patient_lopo"
    assert ridge_res["selected_alpha"] in (1.0, 100.0, 1e4)
    assert len(ridge_res["preds"][0]) == 10

    mlp_res = D.fit_predict_spectral_mlp(tr_sess[:2], tr_sess[2:], te_phi, alpha_grid=(0.1, 10.0), seed=12)
    assert mlp_res["selected_alpha"] in (0.1, 10.0)
    assert len(mlp_res["grid_val_rmses"]) == 2
    assert len(mlp_res["preds"][0]) == 10


def test_stage2_json_contract_stage1_parity_and_audit_gates() -> None:
    """Verify Stage 2.2 JSON artifact (`stage2-complexity-v4`), Stage 1.1 parity, and 7/7 audit pass."""
    s1_path = IO.BUNDLE_DIR / "metrics_stage1_parsimony.json"
    s2_path = IO.BUNDLE_DIR / "metrics_stage2_complexity.json"
    a2_path = IO.AUDIT_DIR / "stage2_godmode_audit.json"
    assert s2_path.exists(), f"Run thesis.exp2_stage2_complexity first: missing {s2_path}"
    assert a2_path.exists(), f"Run thesis.audit_stage2_complexity first: missing {a2_path}"

    s1 = json.loads(s1_path.read_text(encoding="utf-8"))
    s2 = json.loads(s2_path.read_text(encoding="utf-8"))
    a2 = json.loads(a2_path.read_text(encoding="utf-8"))

    assert s2["schema_version"] == "stage2-complexity-v4"
    assert s2["stage1_parity_max_diff"] < 1e-12
    for coh in ("pool30_sync", "retained23", "PAPER_18"):
        assert "br_trig_bpm_lag_aligned" in s2["cohorts"][coh]
        assert len(s2["cohorts"][coh]["br_1hz_30s"]["methods"]) == 13
        for rlbl in ("br_1hz_30s", "br_trig_bpm"):
            for km in ("R0_HPSS_NLS_Locked", "R1a_Online_Forward_MMSE_0s", "R1f_Offline_FB_MMSE_Inf", "R1g_Offline_Viterbi_Unpinned"):
                for fld in ("rmse_sess", "rmse_pool", "r_within_rmcorr"):
                    assert abs(float(s2["cohorts"][coh][rlbl]["methods"][km][fld]) - float(s1["rung1_lopo"][coh][rlbl]["methods"][km][fld])) < 1e-12

    assert "br_trig_count_bpm" in s2["cohorts"]["pool30_sync"]
    assert "pred_sd_ratio_trig_lag_aligned" in s2["shortcut_learning_audit_pool30"]["R2e_Causal_DiffHMM_FilterSafe8"]
    assert "radar_waveform_shift_probe_36_to_46_bpm" in s2["verification_controls"]
    assert "filter_edge_channel_profile_pool30" in s2
    assert abs(int(s2["lag_profiles_pool30"]["R0_vs_br_trig_bpm_lag_aligned"]["peak_lag_s"])) <= 2
    assert len(s2["pool30_fold_logs"]) == 11
    assert a2["all_passed"] is True
    assert all(row["r2a_interp"]["sigma_slow_bpm"] >= 1.0 - 1e-9 for row in s2["pool30_fold_logs"])
    assert a2["all_passed"] is True
    assert all(g["passed"] for g in a2["gates"].values())
