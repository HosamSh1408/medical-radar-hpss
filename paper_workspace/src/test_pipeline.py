"""Smoke + unit tests: synthetic breathing and one real window."""
import numpy as np
from . import config as C
from .demod import csd
from .rbm_nmf import mitigate
from .filtering import bandpass, autocorr_coarse
from .estimation import dft_estimate, nls_estimate


def test_csd_preserves_fundamental():
    fs = C.FS_RADAR
    t = np.arange(30 * fs) / fs
    f0 = 0.75  # 45 bpm
    bI = np.cos(2 * np.pi * f0 * t)
    bQ = np.sin(2 * np.pi * f0 * t)
    x = csd(bI, bQ)
    d = bandpass(x)
    f, _, _ = dft_estimate(d)
    assert abs(f - 45) < 1.0, f
    print("csd/dft ok:", f)


def test_nmf_removes_spike():
    fs = C.FS_RADAR
    t = np.arange(60 * fs) / fs
    f0 = 42 / 60
    clean = np.exp(1j * 2 * np.pi * f0 * t)
    x = clean.copy()
    x[50 * fs:52 * fs] *= 8  # sparse strong RBM like Fig.4
    out = mitigate(x)
    assert out["n_removed"] >= 1, out["n_removed"]
    d = bandpass(out["x_hat"])
    fc = autocorr_coarse(d)
    fest, _, _ = nls_estimate(d, fc)
    assert abs(fest - 42) < 2.0, fest
    print("nmf/nls ok:", fest, "removed:", out["n_removed"])


def test_real_window():
    from .data_loader import load_radar
    bI, bQ = load_radar("7.3")
    segI, segQ = bI[:480], bQ[:480]
    x = csd(segI, segQ)
    out = mitigate(x)
    d = bandpass(out["x_hat"])
    fc = autocorr_coarse(d)
    fest, _, _ = nls_estimate(d, fc)
    assert np.isfinite(fest) and 18 <= fest <= 180, fest
    print("real window ok:", fest)


def test_stft_roundtrip():
    from .rbm_nmf import stft_complex, istft_complex
    rng = np.random.default_rng(0)
    x = rng.standard_normal(480) + 1j * rng.standard_normal(480)
    X, _, _ = stft_complex(x)
    xh = istft_complex(X, len(x))
    assert np.max(np.abs(x - xh)) < 1e-9
    print("stft roundtrip ok")


def test_hpss_geometry():
    from .hpss import hpss_filter
    mag = np.ones((40, 30)) * 0.1
    mag[10, :] += 2.0  # horizontal harmonic ridge
    mag[:, 15] += 5.0  # vertical transient bar
    filt, mask, _, _ = hpss_filter(mag)
    assert filt[10, 5] / mag[10, 5] > 0.9, "ridge must be kept"
    assert filt[20, 15] / mag[20, 15] < 0.2, "transient must be suppressed"
    print("hpss geometry ok")


def test_hpss_removes_spike():
    from .hpss import mitigate_hpss
    fs = C.FS_RADAR
    t = np.arange(60 * fs) / fs
    f0 = 42 / 60
    x = np.exp(1j * 2 * np.pi * f0 * t)
    x[50 * fs:52 * fs] *= 8
    out = mitigate_hpss(x)
    d = bandpass(out["x_hat"])
    fc = autocorr_coarse(d)
    fest, _, _ = nls_estimate(d, fc)
    assert abs(fest - 42) < 2.0, fest
    print("hpss/nls ok:", fest, "kept:", round(out["kept_frac"], 2))


def test_hoyer_and_env_normalize():
    from .rbm_nmf import hoyer_sparsity, detect_rbm, env_normalize
    assert hoyer_sparsity(np.array([5.0] + [0.0] * 7)) > 0.9
    assert hoyer_sparsity(np.ones(8)) < 0.1
    assert hoyer_sparsity(np.zeros(8)) == 0.0
    H = np.ones((11, 28))  # flat activations: no basis should be flagged as RBM
    assert detect_rbm(H).all()
    mag = np.ones((32, 10))
    mag[:, 5] = 0.6  # dip frame within [0.5, 2.0] gain clip range
    normed = env_normalize(mag)
    assert np.allclose(normed.max(axis=0), 1.0)
    print("hoyer/env_normalize ok")


def test_evaluation_sync_and_gating():
    from .evaluation import window_ref_and_gating, sync_arrays, scores, bland_altman
    fs_ref = 25.0
    resp_raw = np.sin(np.linspace(0, 100, int(120 * fs_ref)))
    rate_1hz = np.full(120, 48.0)
    centers = np.array([15.0, 45.0, 75.0])
    rw, vw = window_ref_and_gating(resp_raw, rate_1hz, centers, lag_s=0, fs_raw=fs_ref)
    assert vw.all() and np.allclose(rw, 48.0)
    est = np.linspace(35.0, 55.0, 30)
    centers_sync = np.arange(30) * 2.0 + 15.0
    rate_sync = np.interp(np.arange(120, dtype=float), centers_sync, est)
    est_a, ref_a, lag, corr = sync_arrays(est, rate_sync, metadata_lag_s=0, search_radius_s=4)
    assert abs(lag) <= 4 and corr > 0.95
    sc = scores(est_a, ref_a)
    ba = bland_altman(est_a, ref_a)
    assert sc["rmse"] < 2.0 and sc["acc@10"] == 100.0 and np.isfinite(ba["bias"])
    print("evaluation/sync ok")


def test_four_way_window_runner():
    from .run_measurement import process_window
    fs = C.FS_RADAR
    t = np.arange(C.WIN_SAMPLES) / fs
    f0 = 48.0 / 60.0  # 48 bpm
    bI = np.cos(2 * np.pi * f0 * t)
    bQ = np.sin(2 * np.pi * f0 * t)
    bI[10 * fs:12 * fs] *= 8.0  # inject transient RBM burst
    bQ[10 * fs:12 * fs] *= 8.0
    res = process_window(bI, bQ)
    assert abs(res["f_nls_nmf"] - 48.0) < 2.0, res["f_nls_nmf"]
    assert abs(res["f_hpss"] - 48.0) < 2.0, res["f_hpss"]
    assert res["t_nmf"] > 0 and res["t_hpss"] > 0
    print("4-way window runner ok")


def test_cache_provenance_is_deterministic():
    from .provenance import expected_provenance, provenance_json
    first = expected_provenance("7.3")
    second = expected_provenance("7.3")
    assert first == second
    assert '"pipeline_version":"main-v3-audit-fix-2"' in provenance_json("7.3")
    assert len(first["source_sha256"]) == 15
    assert "protocols.py" in first["source_sha256"]  # audit A6
    assert set(first["input_sha256"]) == {"radar_csv", "reference_edf"}
    print("cache provenance ok")


# --- Golden-value regression tests (audit Phase 0) -------------------------
# These bind the documented/spec claims to concrete code, inputs and artifacts
# so a silent drift in band, sampling rate, cohort membership, metrics or the
# multiverse table fails loudly instead of shipping stale numbers.

def test_golden_filter_matches_paper_spec():
    from scipy import signal as sig
    from .filtering import design_kaiser
    assert (C.BAND_LOW_HZ, C.BAND_HIGH_HZ) == (0.3, 3.0)
    assert (C.KAISER_BETA, C.FIR_TAPS) == (6.5, 64)
    taps = design_kaiser()
    assert taps.shape == (64,)
    w, h = sig.freqz(taps, worN=65536, fs=C.FS_RADAR)

    def mag(f):
        return float(np.abs(h[np.argmin(np.abs(w - f))]))

    assert mag(1.0) > 0.95, mag(1.0)
    assert mag(4.0) < 0.01, mag(4.0)
    # NOTE: this 64-tap Kaiser design has a wide transition band; DC rejection
    # is weak (|H(0)| ~ 0.14) and the 0.3/3.0 Hz edges sit near -6 dB. Pinned so
    # any revisit of filter shape/taps (WS1) is deliberate.
    assert mag(0.0) < 0.2, mag(0.0)
    print("golden filter ok: H(1Hz)=%.4f H(4Hz)=%.4e H(0Hz)=%.4f"
          % (mag(1.0), mag(4.0), mag(0.0)))


def test_golden_reference_sampling_rates():
    from .data_loader import load_reference
    assert (C.FS_REF_RAW, C.FS_REF_RATE) == (256, 1)
    resp, rate, fs_raw, fs_rate = load_reference("7.3")
    assert fs_raw == 256.0 and fs_rate == 1.0, (fs_raw, fs_rate)
    assert len(rate) > 0 and len(resp) > len(rate)
    print("golden reference rates ok: resp=%d@%.0f rate=%d@%.0f"
          % (len(resp), fs_raw, len(rate), fs_rate))


def test_golden_cohort_counts():
    from .cohort import PAPER_18, local_pool, pool30, retained23
    from .run_batch import CACHE
    from .provenance import cache_is_current
    assert (len(local_pool()), len(PAPER_18), len(pool30()), len(retained23())) == (33, 18, 30, 23)
    expect = {"all_paired_local": 7105, "pool30_no_zero_overlap": 6642,
              "retained23_named_exclusions": 6447,
              "current_reference_validity_ranked_18": 5466}
    groups = {"all_paired_local": local_pool(), "pool30_no_zero_overlap": pool30(),
              "retained23_named_exclusions": retained23(),
              "current_reference_validity_ranked_18": PAPER_18}
    for name, ms in groups.items():
        total = 0
        for m in ms:
            p = CACHE / f"{m}.npz"
            assert p.exists() and cache_is_current(p, m), \
                f"cache missing/stale: {m}; run `python3 -m paper_workspace.src.run_batch --all-paired --force`"
            with np.load(p) as z:
                total += int(np.asarray(z["valid"], bool).sum())
        assert total == expect[name], (name, total, expect[name])
    print("golden cohort counts ok:", expect)


def test_golden_paper18_metrics():
    from .cohort import PAPER_18
    from .run_batch import CACHE
    from .result_tables import extended_scores
    from .provenance import cache_is_current
    nmf, hpss = [], []
    for m in PAPER_18:
        p = CACHE / f"{m}.npz"
        assert p.exists() and cache_is_current(p, m), m
        with np.load(p) as z:
            v = np.asarray(z["valid"], bool)
            ref = np.asarray(z["ref"], float)
            nmf.append(extended_scores(np.asarray(z["est"])[v], ref[v])["rmse"])
            hpss.append(extended_scores(np.asarray(z["est_hpss"])[v], ref[v])["rmse"])
    nmf_m, hpss_m = float(np.mean(nmf)), float(np.mean(hpss))
    assert abs(nmf_m - 6.8628) < 0.01, nmf_m
    assert abs(hpss_m - 6.7980) < 0.01, hpss_m
    assert hpss_m < nmf_m
    print("golden PAPER_18 metrics ok: NMF=%.4f HPSS=%.4f" % (nmf_m, hpss_m))


def test_golden_multiverse_csv_matches_caches():
    import csv
    from pathlib import Path
    from .cohort import PAPER_18, local_pool, pool30, retained23
    from .run_batch import CACHE
    from .result_tables import extended_scores
    from .provenance import cache_is_current
    csv_path = (Path(__file__).resolve().parents[1]
                / "outputs" / "audits" / "reconstruction_multiverse.csv")
    rows = {r["variant"]: r for r in csv.DictReader(csv_path.open())}
    groups = {"all_paired_local": local_pool(), "pool30_no_zero_overlap": pool30(),
              "retained23_named_exclusions": retained23(),
              "current_reference_validity_ranked_18": PAPER_18}
    for name, ms in groups.items():
        n_w, nmf, hpss = 0, [], []
        for m in ms:
            p = CACHE / f"{m}.npz"
            assert cache_is_current(p, m), m
            with np.load(p) as z:
                v = np.asarray(z["valid"], bool)
                ref = np.asarray(z["ref"], float)
                n_w += int(v.sum())
                nmf.append(extended_scores(np.asarray(z["est"])[v], ref[v])["rmse"])
                hpss.append(extended_scores(np.asarray(z["est_hpss"])[v], ref[v])["rmse"])
        r = rows[name]
        assert n_w == int(r["n_valid_windows"]), (name, n_w, r["n_valid_windows"])
        nmf_f = [x for x in nmf if np.isfinite(x)]
        hpss_f = [x for x in hpss if np.isfinite(x)]
        assert abs(float(np.mean(nmf_f)) - float(r["nmf_rmse"])) < 1e-6, name
        assert abs(float(np.mean(hpss_f)) - float(r["hpss_rmse"])) < 1e-6, name
    print("golden multiverse CSV matches caches")


if __name__ == "__main__":
    test_csd_preserves_fundamental()
    test_nmf_removes_spike()
    test_real_window()
    test_stft_roundtrip()
    test_hpss_geometry()
    test_hpss_removes_spike()
    test_hoyer_and_env_normalize()
    test_evaluation_sync_and_gating()
    test_four_way_window_runner()
    test_cache_provenance_is_deterministic()
    test_golden_filter_matches_paper_spec()
    test_golden_reference_sampling_rates()
    test_golden_cohort_counts()
    test_golden_paper18_metrics()
    test_golden_multiverse_csv_matches_caches()
    print("ALL TESTS PASSED")

