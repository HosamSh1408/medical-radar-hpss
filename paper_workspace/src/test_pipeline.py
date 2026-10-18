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
    assert '"pipeline_version":"main-v3-audit-fix-1"' in provenance_json("7.3")
    assert len(first["source_sha256"]) == 14
    assert set(first["input_sha256"]) == {"radar_csv", "reference_edf"}
    print("cache provenance ok")


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
    print("ALL TESTS PASSED")

