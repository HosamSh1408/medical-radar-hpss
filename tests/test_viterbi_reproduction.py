"""Automated verification suite for the baseline-anchored Viterbi reproduction."""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from paper_workspace.src import config as C
from paper_workspace.src.data_loader import load_radar
from paper_workspace.src.demod import csd
from paper_workspace.src.estimation import nls_estimate
from paper_workspace.src.filtering import autocorr_coarse, bandpass
from paper_workspace.src.hpss import mitigate_hpss
from paper_workspace.src.rbm_nmf import mitigate
from thesis import audit_viterbi_repro as AUDIT
from thesis import exp_viterbi_reproduction as EXP
from thesis import io_utils as IO
from thesis import viterbi_tracker as VT


def test_frozen_baseline_untouched() -> None:
    """All 33 locked v1 NPZ caches must pass SHA-256 provenance verification."""
    status = IO.verify_all_baseline_caches()
    assert len(status) == 33
    assert all(status.values())


def test_separable_2d_viterbi_exact_parity() -> None:
    """Separable 2-pass 2D Viterbi must match unfactored brute-force 2D to atol=1e-12."""
    br_g = np.arange(20.0, 80.0 + 1e-9, 5.0)
    hr_g = np.arange(100.0, 200.0 + 1e-9, 10.0)
    for seed in (0, 17, 2026):
        rng = np.random.default_rng(seed)
        p_br = rng.uniform(0.0, 1.0, (15, br_g.size))
        p_hr = rng.uniform(0.0, 1.0, (15, hr_g.size))
        p_br[3, :] = 0.0  # test numerical floor edge case
        aux = rng.uniform(0.0, 1.5, (15, br_g.size))
        for use_hqi in (False, True):
            for use_aux in (None, aux):
                hqis = VT.precompute_hqi_sequence(p_hr, hr_g, br_g) if use_hqi else None
                b_sep, h_sep, s_sep = VT.decode_viterbi_2d_joint(
                    p_br,
                    p_hr,
                    br_g,
                    hr_g,
                    w_br=0.5,
                    w_hqi=4.0,
                    sigma_br=4.0,
                    sigma_hr=12.0,
                    eps_p=1e-300,
                    aux_prior_seq=use_aux,
                    use_hqi=use_hqi,
                    hqi_matrix_seq=hqis,
                )
                b_ref, h_ref, s_ref = VT.decode_viterbi_2d_bruteforce(
                    p_br,
                    p_hr,
                    br_g,
                    hr_g,
                    w_br=0.5,
                    w_hqi=4.0,
                    sigma_br=4.0,
                    sigma_hr=12.0,
                    eps_p=1e-300,
                    aux_prior_seq=use_aux,
                    use_hqi=use_hqi,
                )
                assert abs(s_sep - s_ref) < 1e-12
                assert np.array_equal(b_sep, b_ref)
                assert np.array_equal(h_sep, h_ref)


def test_hqi_eq5_exact_properties() -> None:
    """Verify exact Eq. 5 formula values, edge cases, contrast clipping, and scale bounds."""
    hr = np.array([100.0, 120.0, 145.0], dtype=np.float64)
    br = np.array([40.0, 60.0], dtype=np.float64)
    p_hr = np.array([0.2, 1.0, 0.5], dtype=np.float64)

    h_mat = VT.compute_hqi_matrix(p_hr, hr_grid_bpm=hr, br_grid_bpm=br)
    assert h_mat.shape == (3, 2)

    # Hand-calculated check:
    # Q_hr = 1.0 / (mean([0.2, 1.0, 0.5]) + 1e-10) = 1.0 / 0.5666666667 = 1.7647 <= 2 => clip=0 => A_t = 0.05
    q_hr = 1.0 / (np.mean(p_hr) + 1e-10)
    a_t = (5.0 + 20.0 * np.clip((q_hr - 2.0) / 18.0, 0.0, 1.0)) * 0.001 * 10.0
    assert abs(a_t - 0.05) < 1e-12

    # For br=40: r = [2.5, 3.0, 3.625] -> only r=3.0 is within 0.10 of {2,3,4,5}
    # p_harm = 1.0, p_nh = max(0.2, 0.5) = 0.5 => C_t = 0.5
    # At (i=1, j=0): r=3.0 => sum exp = ~1.0 => H[1, 0] ~ 0.5 * 0.05 * 1.0 = 0.025
    assert abs(h_mat[1, 0] - 0.025) < 1e-6

    # For br=60: r = [1.667, 2.0, 2.417] -> only r=2.0 is harmonic => p_harm=1.0, p_nh=0.5 => H[1, 1] ~ 0.025
    assert abs(h_mat[1, 1] - 0.025) < 1e-6

    # Empty harmonic neighborhood edge case: br=80 vs hr=[100, 120] -> r=[1.25, 1.5]
    h_empty = VT.compute_hqi_matrix(np.array([0.5, 1.0]), np.array([100.0, 120.0]), np.array([80.0]))
    assert np.all(np.isfinite(h_empty)) and np.max(h_empty) < 1e-6

    # Zero HR input edge case
    h_zero = VT.compute_hqi_matrix(np.zeros(3), hr, br)
    assert np.all(h_zero == 0.0)


def test_mathematical_group_invariances() -> None:
    """BR/HR spectral and Viterbi BPM estimates must be invariant (< 1e-6 bpm) to I/Q symmetries."""
    b_i, b_q = load_radar("7.3")
    seg_i = b_i[: 480 + 9 * 32]
    seg_q = b_q[: 480 + 9 * 32]

    def _run_slice(arr_i: np.ndarray, arr_q: np.ndarray) -> dict[str, np.ndarray]:
        p2_s, s1_s, phr_s = [], [], []
        for s0 in range(0, 10 * 32, 32):
            xw = csd(arr_i[s0 : s0 + 480], arr_q[s0 : s0 + 480])
            dh = bandpass(mitigate_hpss(bandpass(xw))["x_hat"])
            p2_s.append(VT.extract_br_welch_spectrum(dh, VT.BR_GRID_2BPM))
            s1_s.append(VT.extract_br_harmonic_nls_spectrum(dh, VT.BR_GRID_1BPM))
            phr_s.append(VT.extract_hr_csd_velocity_spectrum(xw, VT.HR_GRID_2BPM))
        p2_a, s1_a, phr_a = np.asarray(p2_s), np.asarray(s1_s), np.asarray(phr_s)
        return {
            "welch_br": VT.BR_GRID_2BPM[np.argmax(p2_a, axis=1)],
            "harmnls_br": VT.BR_GRID_1BPM[np.argmax(s1_a, axis=1)],
            "csd_vel_hr": VT.HR_GRID_2BPM[np.argmax(phr_a, axis=1)],
            "c1b_1d": VT.decode_viterbi_1d(p2_a, VT.BR_GRID_2BPM)[0],
            "c3b_2d": VT.decode_viterbi_2d_joint(p2_a, phr_a, VT.BR_GRID_2BPM, VT.HR_GRID_2BPM)[0],
            "c4b_2d_nls": VT.decode_viterbi_2d_joint(s1_a, phr_a, VT.BR_GRID_1BPM, VT.HR_GRID_2BPM)[0],
        }

    ref = _run_slice(seg_i, seg_q)
    z_rot = (seg_i + 1j * seg_q) * np.exp(1j * np.pi / 3.0)
    for ti, tq in (
        (3.7 * seg_i, 3.7 * seg_q),
        (z_rot.real, z_rot.imag),
        (-seg_i, -seg_q),
        (seg_q, seg_i),
    ):
        alt = _run_slice(ti, tq)
        for key in ref:
            assert float(np.max(np.abs(ref[key] - alt[key]))) < 1e-6


def test_synthetic_recovery() -> None:
    """Synthetic stationary + chirp/RBM/cardiac CW radar sequences must be accurately tracked."""
    fs = float(C.FS_RADAR)
    dur_s = 120.0
    t = np.arange(int(dur_s * fs), dtype=np.float64) / fs
    rng = np.random.default_rng(20261011)
    starts = list(range(0, t.size - int(C.WIN_SAMPLES) + 1, int(C.UPDATE_S * C.FS_RADAR)))

    # 1. Stationary 45 bpm breathing + 150 bpm cardiac
    d_st = 0.0008 * np.sin(2.0 * np.pi * 0.75 * t) + 0.00015 * np.sin(2.0 * np.pi * 2.5 * t)
    x_st = np.exp(1j * (0.4 + 4.0 * np.pi * d_st / C.LAMBDA_M))
    x_st = x_st + 0.01 * (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size))
    s1_st, phr_st = [], []
    for s0 in starts:
        xw = x_st[s0 : s0 + int(C.WIN_SAMPLES)]
        dh = bandpass(mitigate_hpss(bandpass(xw))["x_hat"])
        s1_st.append(VT.extract_br_harmonic_nls_spectrum(dh, VT.BR_GRID_1BPM))
        phr_st.append(VT.extract_hr_csd_velocity_spectrum(xw, VT.HR_GRID_2BPM))
    br_st, hr_st, _ = VT.decode_viterbi_2d_joint(
        np.asarray(s1_st), np.asarray(phr_st), VT.BR_GRID_1BPM, VT.HR_GRID_2BPM
    )
    assert float(np.mean(np.abs(br_st - 45.0))) < 1.0
    assert float(np.mean(np.abs(hr_st - 150.0))) < 2.0

    # 2. Chirp 35 -> 65 bpm + 150 bpm cardiac + 8x RBM burst
    f_inst = (35.0 + 30.0 * (t / dur_s)) / 60.0
    phase_resp = 2.0 * np.pi * ((35.0 / 60.0) * t + 0.5 * (30.0 / 60.0 / dur_s) * (t ** 2))
    d_m = 0.0010 * np.sin(phase_resp) + 0.00010 * np.sin(2.0 * np.pi * 2.5 * t)
    x_syn = np.exp(1j * (0.4 + 4.0 * np.pi * d_m / C.LAMBDA_M))
    x_syn[(t >= 50.0) & (t <= 54.0)] *= 8.0
    x_syn = x_syn + 0.01 * (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size))

    truth = np.array([f_inst[s0 + int(C.WIN_SAMPLES) // 2] * 60.0 for s0 in starts])
    s1_s, phr_s = [], []
    for s0 in starts:
        xw = x_syn[s0 : s0 + int(C.WIN_SAMPLES)]
        dh = bandpass(mitigate_hpss(bandpass(xw))["x_hat"])
        s1_s.append(VT.extract_br_harmonic_nls_spectrum(dh, VT.BR_GRID_1BPM))
        phr_s.append(VT.extract_hr_csd_velocity_spectrum(xw, VT.HR_GRID_2BPM))

    br_path, _, _ = VT.decode_viterbi_2d_joint(
        np.asarray(s1_s), np.asarray(phr_s), VT.BR_GRID_1BPM, VT.HR_GRID_2BPM
    )
    assert float(np.mean(np.abs(br_path - truth))) < 2.5


def test_exported_metrics_and_baseline_parity() -> None:
    """All exported artifacts must exist, reproduce locked v1 baselines, and pass controls."""
    metrics_path = IO.BUNDLE_DIR / "metrics_viterbi_repro.json"
    summary_csv = IO.AUDIT_DIR / "viterbi_repro_summary.csv"
    sub_csv = IO.AUDIT_DIR / "viterbi_repro_subgroups.csv"
    sess_csv = IO.AUDIT_DIR / "viterbi_repro_sessions.csv"
    recon_md = IO.AUDIT_DIR / "viterbi_repro_reconciliation.md"

    for p in (metrics_path, summary_csv, sub_csv, sess_csv, recon_md):
        assert p.exists(), f"Missing artifact: {p}"

    for stem in (
        "viterbi_repro_ablation_ladder",
        "viterbi_repro_bland_altman",
        "viterbi_repro_tracking_examples",
        "viterbi_repro_forensic_bridge",
    ):
        assert (IO.BUNDLE_FIG_DIR / f"{stem}.pdf").exists()
        assert (IO.BUNDLE_FIG_DIR / f"{stem}.png").exists()

    master = json.loads(metrics_path.read_text(encoding="utf-8"))
    indep = json.loads((IO.AUDIT_DIR / "final_review_independent.json").read_text(encoding="utf-8"))

    # Exact locked baseline parity on cached_nmf_lag across all 4 cohorts
    cached_res = master["results"]["cached_nmf_lag"]
    for ck, ik in (("PAPER_18", "PAPER_18"), ("retained23", "retained23"), ("pool30_sync", "pool30"), ("all_paired", "all_paired")):
        assert abs(cached_res[ck]["methods"]["A1_DFT_Locked"]["rmse_sess"] - indep["baselines"][ik]["dft"]) < 1e-3
        assert abs(cached_res[ck]["methods"]["A2_RawNLS_Locked"]["rmse_sess"] - indep["baselines"][ik]["nls"]) < 1e-3
        assert abs(cached_res[ck]["methods"]["A3_Beltrao_NMF_NLS"]["rmse_sess"] - indep["baselines"][ik]["nmf"]) < 1e-3
        assert abs(cached_res[ck]["methods"]["A4_HPSS_NLS"]["rmse_sess"] - indep["baselines"][ik]["hpss"]) < 1e-3
        assert abs(cached_res[ck]["constants"]["const_loso"] - indep["baselines"][ik]["const_loso"]) < 1e-3

    # Fresh Python 3.10 parity audit check
    assert master["parity_audit"]["max_abs_diff_hpss"] < 1e-3
    assert master["parity_audit"]["max_abs_diff_nmf"] < 0.5

    # Check retained23 prone count = 8
    ret_sub = master["results"]["branch_neutral_raw_nls_lag"]["retained23"]["subgroups"]
    assert ret_sub["prone"]["A4_HPSS_NLS"]["n_sessions"] == 8

    # Verify LOPO fold completeness and patient disjointness on pool30_sync
    p30_folds = master["results"]["branch_neutral_raw_nls_lag"]["pool30_sync"]["lopo_folds"]["C6b_LOPO_Cal_HarmNLS_HPSS"]
    assert len(p30_folds) == 11
    seen_pats = {f["patient_id"] for f in p30_folds}
    assert seen_pats == set(range(2, 13))

    # Verify HR validation & all 5 Pillar-IV controls are finite
    assert np.isfinite(master["hr_validation"]["hr_peak_csd_vel"]["pct_within_10bpm_sess"])
    assert "hr_peak_dacm_vel" in master["hr_validation"]
    assert "hr_peak_br_notched_vel" in master["hr_validation"]
    ctrl = master["controls"]
    assert np.isfinite(ctrl["stiff_hmm_anti_inertia"]["stiff_2d_harmnls_hpss"]["r_within_rmcorr"])
    assert np.isfinite(ctrl["surrogate_time_shuffled_ref"]["C4b_2D_HarmNLS_Vit_HPSS"]["rmse_sess"])
    assert np.isfinite(ctrl["complex_wgn_noise_floor"]["C4b_2D_HarmNLS_Vit_HPSS"]["median_bpm"])
    for trans_dict in ctrl["mathematical_group_invariances_max_diff"].values():
        assert max(trans_dict.values()) < 1e-6

    df_sum = pd.read_csv(summary_csv)
    assert not df_sum.empty
    df_sess = pd.read_csv(sess_csv, dtype={"measurement": str})
    assert len(df_sess["measurement"].unique()) == 31
    assert set(df_sess["cohort"].unique()) == {"pool30_sync", "retained23", "PAPER_18", "all_paired"}


def test_extended_invariances_anchored_dacm_notched_c2_c5_nmf() -> None:
    """Extended pipelines (Anchored NLS, DACM/notched HR, C2, C5, NMF) stay < 1e-6 bpm invariant."""
    b_i, b_q = load_radar("7.3")
    seg_i = b_i[: 480 + 7 * 32]
    seg_q = b_q[: 480 + 7 * 32]

    def _run_ext(arr_i: np.ndarray, arr_q: np.ndarray) -> dict[str, np.ndarray]:
        p2_h, p2_n, s_anc_h, phr_c, phr_d, phr_n = [], [], [], [], [], []
        aux_seq = []
        for s0 in range(0, 8 * 32, 32):
            xw = csd(arr_i[s0 : s0 + 480], arr_q[s0 : s0 + 480])
            draw = bandpass(xw)
            dh = bandpass(mitigate_hpss(draw)["x_hat"])
            dn = bandpass(mitigate(draw)["x_hat"])
            fc_h = autocorr_coarse(dh)
            fh_h, _, _ = nls_estimate(dh, fc_h)
            p2_h.append(VT.extract_br_welch_spectrum(dh, VT.BR_GRID_2BPM))
            p2_n.append(VT.extract_br_welch_spectrum(dn, VT.BR_GRID_2BPM))
            s_anc_h.append(VT.extract_br_anchored_nls_spectrum(dh, fc_h, VT.BR_GRID_1BPM))
            phr_c.append(VT.extract_hr_csd_velocity_spectrum(xw, VT.HR_GRID_2BPM))
            phr_d.append(VT.extract_hr_dacm_velocity_spectrum(xw, VT.HR_GRID_2BPM))
            phr_n.append(VT.extract_hr_br_notched_velocity_spectrum(xw, fh_h, VT.HR_GRID_2BPM))
            cands = EXP._sharpness_proxy(dh, fh_h, fc_h)
            aux_seq.append(cands)
        p2_ha = np.asarray(p2_h)
        p2_na = np.asarray(p2_n)
        s_anca = np.asarray(s_anc_h)
        phr_ca = np.asarray(phr_c)
        phr_da = np.asarray(phr_d)
        phr_na = np.asarray(phr_n)
        hqi1d = np.array(
            [VT.compute_hqi_1d_penalty(phr_ca[t], VT.HR_GRID_2BPM, VT.BR_GRID_2BPM) for t in range(len(p2_h))]
        )
        aux_prior = EXP._build_aux_prior_seq(np.asarray(aux_seq), phr_ca, VT.BR_GRID_2BPM)
        return {
            "anch_nls": VT.BR_GRID_1BPM[np.argmax(s_anca, axis=1)],
            "dacm_hr": VT.HR_GRID_2BPM[np.argmax(phr_da, axis=1)],
            "notched_hr": VT.HR_GRID_2BPM[np.argmax(phr_na, axis=1)],
            "c2_1d_hqi": VT.decode_viterbi_1d(p2_ha, VT.BR_GRID_2BPM, hqi_1d_seq=hqi1d)[0],
            "c5_aux": VT.decode_viterbi_2d_joint(
                p2_ha, phr_ca, VT.BR_GRID_2BPM, VT.HR_GRID_2BPM, aux_prior_seq=aux_prior
            )[0],
            "nmf_c1": VT.decode_viterbi_1d(p2_na, VT.BR_GRID_2BPM)[0],
            "c4d_anch_2d": VT.decode_viterbi_2d_joint(s_anca, phr_ca, VT.BR_GRID_1BPM, VT.HR_GRID_2BPM)[0],
        }

    ref = _run_ext(seg_i, seg_q)
    z_rot = (seg_i + 1j * seg_q) * np.exp(1j * np.pi / 3.0)
    for ti, tq in (
        (3.7 * seg_i, 3.7 * seg_q),
        (z_rot.real, z_rot.imag),
        (-seg_i, -seg_q),
        (seg_q, seg_i),
    ):
        alt = _run_ext(ti, tq)
        for key in ref:
            assert float(np.max(np.abs(ref[key] - alt[key]))) < 1e-6


def test_paired_wilcoxon_zero_delta_warning_free() -> None:
    """Self-comparison in _paired_tests must return p=1.0 cleanly without RuntimeWarning."""
    sessions = ["2.3", "3.1", "4.1"]
    per_sess = {m: {"rmse": 7.5 + 0.3 * i} for i, m in enumerate(sessions)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = EXP._paired_tests(sessions, per_sess, per_sess)
    assert res["p_wilcoxon"] == 1.0
    assert res["p_wilcoxon_patient"] == 1.0
    runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warns) == 0


def test_v2_schema_cohort_scatter_and_cleanroom_audit() -> None:
    """Verify v2 schema provenance, cohort-scoped Figure 2 scatter, and 7/7 cleanroom audit."""
    ginfo = IO.git_head_info()
    assert ginfo["head_commit"] not in ("", "unknown")

    master = json.loads((IO.BUNDLE_DIR / "metrics_viterbi_repro.json").read_text(encoding="utf-8"))
    assert master["provenance"]["schema_version"] == "viterbi-repro-v2"
    assert master["provenance"]["git"]["head_commit"] not in ("", "unknown")

    p30 = master["results"]["branch_neutral_raw_nls_lag"]["pool30_sync"]
    assert len(p30["methods"]) == 33
    assert "const_lopo_median" in p30["constants"]
    assert "audit_stress_tests" in master
    assert master["controls"]["surrogate_phase_scrambled_radar"]["n_seeds"] == 5

    df_sess = pd.read_csv(IO.AUDIT_DIR / "viterbi_repro_sessions.csv", dtype={"measurement": str})
    for coh in ("retained23", "pool30_sync"):
        for meth in ("A4_HPSS_NLS", "C6d_LOPO_Wide_AnchoredNLS_HPSS", "C6e_LOPO_Wide_AnchoredNLS_NMF"):
            sub = df_sess[
                (df_sess["regime"] == "branch_neutral_raw_nls_lag")
                & (df_sess["cohort"] == coh)
                & (df_sess["method"] == meth)
            ]
            jba = master["results"]["branch_neutral_raw_nls_lag"][coh]["methods"][meth]["ba_session_level"]
            diffs = sub["est_mean"].to_numpy(float) - sub["ref_mean"].to_numpy(float)
            assert abs(float(np.mean(diffs)) - float(jba["bias"])) < 1e-12
            assert abs(float(np.std(diffs, ddof=1)) - float(jba["sd"])) < 1e-12

    audit_rep = AUDIT.run_audit()
    assert audit_rep["all_passed"] is True
    assert all(g["passed"] for g in audit_rep["gates"].values())
