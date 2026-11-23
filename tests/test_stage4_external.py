"""Unit and contract tests for Stage 4 (WP5, L4 External Benchmark on gleichauf2024)."""
from __future__ import annotations

import json

import numpy as np
from scipy import signal as sig

from paper_workspace.src import config as C
from paper_workspace.src import filtering as F
from thesis import exp4_stage4_external as E4
from thesis import io_utils as IO


def test_decimate_iq_to_16hz_frequency_preservation_and_o2_symmetry() -> None:
    """Verify deterministic ~2 kHz -> 16 Hz decimation preserves frequency (< 0.01 bpm) and O(2)xR+ symmetry."""
    fs = 2000.02
    t = np.arange(int(913.0 * fs), dtype=np.float64) / fs
    for bpm in (45.0, 80.0):
        omega = 2.0 * np.pi * (bpm / 60.0)
        bi = 1234.5 + 3.0 * np.cos(omega * t)
        bq = -876.2 + 3.0 * np.sin(omega * t)
        i16, q16, dur = E4.decimate_iq_to_16hz(t, bi, bq)
        assert abs(dur - i16.size / 16.0) < 1e-12
        assert abs(dur - 913.0) < 0.2

        spec = np.abs(np.fft.fft((i16 + 1j * q16) * np.hanning(i16.size), n=65536)) ** 2
        freqs = np.fft.fftfreq(65536, d=1.0 / 16.0) * 60.0
        f_peak = float(freqs[int(np.argmax(spec))])
        assert abs(f_peak - bpm) < 0.01

        z = (bi + 1j * bq) * np.exp(1j * np.pi / 3.0)
        ir, qr, _ = E4.decimate_iq_to_16hz(t, 3.7 * z.real, 3.7 * z.imag)
        z_exp = 3.7 * (i16 + 1j * q16) * np.exp(1j * np.pi / 3.0)
        assert float(np.max(np.abs((ir + 1j * qr) - z_exp))) < 1e-11

        isw, qsw, _ = E4.decimate_iq_to_16hz(t, bq, bi)
        assert float(np.max(np.abs(isw - q16))) < 1e-12
        assert float(np.max(np.abs(qsw - i16))) < 1e-12


def test_locked_kaiser_bandpass_provenance_and_scenario_partition() -> None:
    """Verify locked 64-tap Kaiser [0.30, 3.00] Hz FIR filter and exhaustive 14 + 6 = 20 partition."""
    assert int(C.FIR_TAPS) == 64
    assert float(C.BAND_LOW_HZ) == 0.30
    assert float(C.BAND_HIGH_HZ) == 3.00
    assert float(C.KAISER_BETA) == 6.5

    w, h = sig.freqz(F._TAPS, worN=8192, fs=C.FS_RADAR)
    mag2 = np.abs(h) ** 2
    idx_18 = int(np.argmin(np.abs(w - 0.300)))
    idx_45 = int(np.argmin(np.abs(w - 0.750)))
    assert mag2[idx_18] < 0.30  # Attenuated on low-frequency transition skirt (-12 dB)
    assert mag2[idx_45] > 0.98  # Flat unity passband in core breathing range

    scens = E4.discover_scenarios()
    assert len(scens) == 20
    inb = [s for s in scens if E4.parse_scenario_metadata(s)[0] >= 28.0]
    sub = [s for s in scens if E4.parse_scenario_metadata(s)[0] < 28.0]
    assert len(inb) == 14 and len(sub) == 6
    assert set(inb).isdisjoint(set(sub))


def test_stage4_json_contract_primary_sensitivity_and_audit_gates() -> None:
    """Verify Stage 4 JSON (`stage4-external-v1`), zero-touch L4 transfer & 7/7 clean-room gates."""
    s3_path = IO.BUNDLE_DIR / "metrics_stage3_uncertainty.json"
    s4_path = IO.BUNDLE_DIR / "metrics_stage4_external.json"
    a4_path = IO.AUDIT_DIR / "stage4_godmode_audit.json"
    assert s4_path.exists(), f"Run thesis.exp4_stage4_external first: missing {s4_path}"
    assert a4_path.exists(), f"Run thesis.audit_stage4_external first: missing {a4_path}"

    s3 = json.loads(s3_path.read_text(encoding="utf-8"))
    s4 = json.loads(s4_path.read_text(encoding="utf-8"))
    a4 = json.loads(a4_path.read_text(encoding="utf-8"))

    assert s4["schema_version"] == "stage4-external-v1"
    hf_sha = s3["gate_g4_verdict"]["hard_freeze_manifest_for_stage4_l4"]["frozen_stage4_models_sha256"]
    assert s4["frozen_stage4_archive_sha256"] == hf_sha
    assert s4["hardware_protocol_disclosure"]["zero_external_adaptation"] is True

    g5 = s4["gate_g5_l4_verdict"]
    assert g5["decision"] == "PASS_L4_ZERO_TOUCH_TRANSFER_VERIFIED"
    assert g5["exhaustive_20_scenarios_partitioned"] is True
    assert g5["all_inband_equivariant_rungs_beat_all_rung3_foils"] is True
    assert g5["r2e_near_unit_slope_ge_0p95"] is True
    assert g5["zero_touch_conformal_c5_valid_90_and_80"] is True
    assert g5["zero_touch_u6b_stopband_ood_auroc_gt_0p90"] is True
    assert g5["external_o2_and_causality_verified"] is True

    inb = s4["regimes"]["steady_state_excl_sync_pulses"]["inband_ge28bpm"]
    for km in E4.MODEL_ORDER + E4.SENSITIVITY_LOPO_MODELS:
        assert km in inb
    assert float(inb["R2e_Causal_DiffHMM_FilterSafe8"]["rmse_scen"]) < 0.10
    assert float(inb["R2e_LOPO_Mean_Sensitivity"]["rmse_scen"]) < 0.10
    assert float(inb["R3c_Ridge_Spectral_439"]["rmse_scen"]) > 2.0
    assert float(inb["R3a_Causal_TCN_16k"]["rmse_scen"]) > 2.0

    # Verify both publication figures exist on disk
    assert (IO.BUNDLE_FIG_DIR / "stage4_external_frequency_transfer.pdf").exists()
    assert (IO.BUNDLE_FIG_DIR / "stage4_external_frequency_transfer.png").exists()
    assert (IO.BUNDLE_FIG_DIR / "stage4_external_uncertainty_and_conformal.pdf").exists()
    assert (IO.BUNDLE_FIG_DIR / "stage4_external_uncertainty_and_conformal.png").exists()

    assert a4["all_passed"] is True
    assert all(g["passed"] for g in a4["gates"].values())
    assert a4["gates"]["gate1_frozen_and_external_sha256_integrity"]["bad_external_files"] == []
    assert a4["gates"]["gate1_frozen_and_external_sha256_integrity"]["checked_external_files"] == 49
    assert a4["gates"]["gate3_cleanroom_raw_csv_to_model_reexecution"]["max_reexecution_diff_bpm"] < 1e-10
