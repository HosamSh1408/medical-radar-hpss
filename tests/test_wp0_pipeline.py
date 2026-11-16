"""Automated verification suite for Phase 0 (analysis_plan_v2, edf_extractor, EXP-0)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from paper_workspace.src import config as C
from paper_workspace.src import evaluation as EV
from paper_workspace.src.data_loader import load_reference
from thesis import edf_extractor as EDFX
from thesis import exp0_forensics as EXP0
from thesis import io_utils as IO


def test_frozen_baseline_untouched() -> None:
    """All 33 locked v1 NPZ caches must pass SHA-256 provenance verification."""
    status = IO.verify_all_baseline_caches()
    assert len(status) == 33
    assert all(status.values())


def test_analysis_plan_v2_and_lopo_folds() -> None:
    """Pre-registered YAML schema and patient-disjoint LOPO folds must hold."""
    plan_path = IO.METADATA_DIR / "analysis_plan_v2.yaml"
    assert plan_path.exists()
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    assert plan["cohorts"]["primary_confirmatory"] == "pool30_sync"
    assert plan["synchronization"]["primary_regime"] == "branch_neutral_raw_nls_lag"
    assert plan["cohorts"]["cross_validation"]["hard_freeze_gate"] == "G4"

    folds_p30 = list(IO.iter_lopo_folds("pool30_sync"))
    assert len(folds_p30) == 11  # Patients 2..12
    covered_p30 = []
    for f in folds_p30:
        train_pats = {IO.patient_id_of(m) for m in f.train_sessions}
        test_pats = {IO.patient_id_of(m) for m in f.test_sessions}
        assert test_pats == {f.patient_id}
        assert train_pats.isdisjoint(test_pats)
        covered_p30.extend(f.test_sessions)
    assert len(covered_p30) == 28
    assert len(set(covered_p30)) == 28

    folds_all = list(IO.iter_lopo_folds("all_paired"))
    assert len(folds_all) == 12  # Patients 1..12
    assert sum(len(f.test_sessions) for f in folds_all) == 31


def test_rmcorr_ancova_simpson_separation() -> None:
    """Bland-Altman ANCOVA rmcorr must detect positive within-subject trend despite Simpson reversal."""
    t = np.linspace(-3, 3, 25)
    est_s1 = 30.0 + t + 0.1 * np.sin(t)
    ref_s1 = 30.0 + t
    # Second session shifted higher on ref, lower on est offset -> naive pooled r < 0
    est_s2 = 15.0 + t - 0.1 * np.sin(t)
    ref_s2 = 55.0 + t
    pooled_r = np.corrcoef(np.r_[est_s1, est_s2], np.r_[ref_s1, ref_s2])[0, 1]
    assert pooled_r < 0.0
    rmc = IO.rmcorr_ancova([est_s1, est_s2], [ref_s1, ref_s2])
    assert rmc["r_within"] > 0.99
    assert rmc["df"] == 50 - 2 - 1


def test_edf_extractor_synthetic_recovery() -> None:
    """Synthetic respiratory & ECG QRS waveforms/triggers must recover rates within < 0.5 bpm."""
    dur_s = 30.0
    fs_256 = 256.0
    fs_1024 = 1024.0
    t256 = np.arange(int(dur_s * fs_256)) / fs_256

    # 45 bpm breathing sinusoid
    resp_bpm_true = 45.0
    resp_sig = 0.8 * np.sin(2.0 * np.pi * (resp_bpm_true / 60.0) * t256)
    br_spec, rms = EDFX.spectral_resp_rate_256hz(resp_sig, fs_raw=fs_256)
    assert abs(br_spec - resp_bpm_true) < 0.5
    assert rms > 0.4

    # 1024 Hz Breath triggers at exact 45 bpm period
    period_b = 60.0 / resp_bpm_true
    b_trig = np.zeros(int(dur_s * fs_1024))
    for k in np.arange(0.5, dur_s - 0.1, period_b):
        idx = int(round(k * fs_1024))
        b_trig[idx:idx + 4] = 1.0
    b_times = EDFX.detect_trigger_times(b_trig, fs_hz=fs_1024, min_refractory_s=0.30)
    br_med, _, _, cnt = EDFX.interval_rate_in_window(b_times, 0.0, dur_s, 12.0, 120.0)
    assert cnt >= 20
    assert abs(br_med - resp_bpm_true) < 0.25

    # 150 bpm QRS envelope synthetic ECG
    hr_true = 150.0
    period_hr = 60.0 / hr_true
    ecg_sig = np.zeros_like(t256)
    r_trig = np.zeros(int(dur_s * fs_1024))
    for beat_t in np.arange(0.3, dur_s - 0.2, period_hr):
        # Gaussian-windowed 20 Hz QRS complex
        ecg_sig += 1.5 * np.exp(-0.5 * ((t256 - beat_t) / 0.015) ** 2) * np.cos(2.0 * np.pi * 20.0 * (t256 - beat_t))
        idx = int(round(beat_t * fs_1024))
        r_trig[idx:idx + 4] = 1.0
    env = EDFX.ecg_qrs_envelope_hr(ecg_sig, fs_ecg=fs_256)
    hr_spec = EDFX.spectral_hr_from_qrs_envelope(env, fs_raw=fs_256)
    r_times = EDFX.detect_trigger_times(r_trig, fs_hz=fs_1024, min_refractory_s=0.22)
    hr_med, _, _, _ = EDFX.interval_rate_in_window(r_times, 0.0, dur_s, 60.0, 230.0)
    assert abs(hr_spec - hr_true) < 0.5
    assert abs(hr_med - hr_true) < 0.5


def test_edf_extractor_real_session_7_3() -> None:
    """Verify exact window gating parity with evaluation.py and high-rate concordance on 7.3."""
    edf = EDFX.load_edf_recording("7.3")
    resp_raw, resp_rate, fs_raw, _ = load_reference("7.3")
    cache = IO.load_locked_cache("7.3")
    centers = np.arange(len(cache["est"]), dtype=float) * C.UPDATE_S + C.WIN_S / 2.0
    lag_s = int(cache["lag"])

    ext = EDFX.extract_window_references(edf, centers, lag_s=lag_s)
    ref_ev, vw_ev = EV.window_ref_and_gating(resp_raw, resp_rate, centers, lag_s, fs_raw=fs_raw)
    assert np.array_equal(ext["valid_vwin"], vw_ev)
    assert np.allclose(ext["br_1hz_30s"][vw_ev], ref_ev[vw_ev], atol=1e-12)

    ok_br = vw_ev & np.isfinite(ext["br_trig_bpm"])
    assert np.mean(np.abs(ext["br_trig_bpm"][ok_br] - ref_ev[ok_br])) < 3.0
    ok_hr = vw_ev & (ext["signaliq_mean"] >= 50) & np.isfinite(ext["hr_rtrig_bpm"])
    assert np.mean(np.abs(ext["hr_rtrig_bpm"][ok_hr] - ext["hr_1hz_30s"][ok_hr])) < 2.5


def test_exp0_outputs_reproduce_locked_v1_audit() -> None:
    """EXP-0 read-only exports must match locked v1 baselines identically and pass completeness checks."""
    metrics_path = IO.BUNDLE_DIR / "metrics_wp0_forensics.json"
    sess_csv = IO.AUDIT_DIR / "forensics_31sessions.csv"
    attr_csv = IO.BUNDLE_DIR / "forensics_attribution.csv"
    dense_md = IO.BUNDLE_DIR / "idea_dense_ch2_wp0.md"

    for p in (metrics_path, sess_csv, attr_csv, dense_md):
        assert p.exists(), f"Missing EXP-0 deliverable: {p}"

    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    indep = json.loads((IO.AUDIT_DIR / "final_review_independent.json").read_text(encoding="utf-8"))
    track = json.loads((IO.AUDIT_DIR / "final_review_tracking_and_noise.json").read_text(encoding="utf-8"))

    reg_c = summary["regimes"]["cached_nmf_lag"]
    for src_key, dst_key in (("PAPER_18", "PAPER_18"), ("retained23", "retained23"), ("pool30_sync", "pool30"), ("all_paired", "all_paired")):
        assert abs(reg_c[src_key]["rmse_sess_hpss"] - indep["baselines"][dst_key]["hpss"]) < 1e-3
        assert abs(reg_c[src_key]["rmse_sess_nmf"] - indep["baselines"][dst_key]["nmf"]) < 1e-3
        assert abs(reg_c[src_key]["const_loso"] - indep["baselines"][dst_key]["const_loso"]) < 1e-3
        assert abs(reg_c[src_key]["r_within_sess_mean_hpss"] - track[dst_key]["within_session_pearson_r_hpss_mean"]) < 1e-3

    # Ensure ANCOVA rmcorr is distinct and computed cleanly
    p30_n = summary["regimes"]["branch_neutral_raw_nls_lag"]["pool30_sync"]
    assert np.isfinite(p30_n["r_within_rmcorr_hpss"])
    assert p30_n["r_within_rmcorr_df"] > 6000

    df_s = pd.read_csv(sess_csv, dtype={"measurement": str})
    assert len(df_s) == 33
    assert set(df_s.loc[df_s["is_zero_valid"], "measurement"]) == {"1.3", "5.1"}
    assert set(df_s.loc[df_s["is_zero_overlap"], "measurement"]) == {"1.1", "2.1", "2.2"}
    assert df_s["iq_sd_ratio_session"].isna().sum() == 0
    assert df_s["ucd_nonchild_frac_session"].isna().sum() == 0

    for stem in ("exp0_session_error_decomposition", "exp0_physics_error_drivers", "exp0_edf256_reference_agreement"):
        assert (IO.BUNDLE_FIG_DIR / f"{stem}.pdf").exists()
        assert (IO.BUNDLE_FIG_DIR / f"{stem}.png").exists()
