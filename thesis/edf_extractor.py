"""High-rate (256 Hz / 1024 Hz) EDF cardiopulmonary reference extractor.

Reads all 11 channels from `beltrao_data/reference/*.edf` cleanly:
  - `ECG` (256 Hz, mV) and `R-Trigger` (1024 Hz binary pulse train)
  - `Resp` (256 Hz, Ohm) and `Resp.-Trigger` (1024 Hz binary pulse train)
  - `HeartRate`, `RespRate`, `SignalIQ`, `Impedance` (1 Hz)

Extracts non-quantized physical breath/cardiac rates and pause/saturation features
aligned to any locked session lag `lag_s` without touching `paper_workspace/src/`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyedflib
from scipy import signal

from paper_workspace.src import config as C
from paper_workspace.src.data_loader import MEASURE_TO_REF
from paper_workspace.src.evaluation import window_ref_and_gating


@dataclass(frozen=True)
class EdfRecording:
    measurement: str
    start_datetime: datetime | None
    duration_s: float
    ecg_256: np.ndarray
    resp_256: np.ndarray
    r_trig_1024: np.ndarray
    resp_trig_1024: np.ndarray
    signaliq_1hz: np.ndarray
    resp_rate_1hz: np.ndarray
    heart_rate_1hz: np.ndarray
    impedance_1hz: np.ndarray
    fs_ecg: float
    fs_resp: float
    fs_trig: float
    fs_rate: float


def load_edf_recording(measurement: str, ref_dir: Path | None = None) -> EdfRecording:
    """Read raw waveforms, triggers, and 1 Hz monitor trends from an EDF file."""
    rdir = Path(ref_dir) if ref_dir else C.REF_DIR
    if measurement not in MEASURE_TO_REF:
        raise ValueError(f"No reference EDF mapped for {measurement!r}")
    edf_path = rdir / f"{MEASURE_TO_REF[measurement]}.edf"
    reader = pyedflib.EdfReader(str(edf_path))
    try:
        labels = [reader.getLabel(i) for i in range(reader.signals_in_file)]
        idx = {lbl: i for i, lbl in enumerate(labels)}
        req = ("ECG", "Resp", "R-Trigger", "Resp.-Trigger", "SignalIQ", "RespRate", "Impedance", "HeartRate")
        missing = [k for k in req if k not in idx]
        if missing:
            raise KeyError(f"{edf_path.name} missing channels: {missing}")

        start_dt = reader.getStartdatetime()
        if getattr(start_dt, "tzinfo", None) is not None:
            start_dt = start_dt.replace(tzinfo=None)

        return EdfRecording(
            measurement=measurement,
            start_datetime=start_dt,
            duration_s=float(reader.file_duration),
            ecg_256=reader.readSignal(idx["ECG"]).astype(np.float64),
            resp_256=reader.readSignal(idx["Resp"]).astype(np.float64),
            r_trig_1024=reader.readSignal(idx["R-Trigger"]).astype(np.float64),
            resp_trig_1024=reader.readSignal(idx["Resp.-Trigger"]).astype(np.float64),
            signaliq_1hz=reader.readSignal(idx["SignalIQ"]).astype(np.float64),
            resp_rate_1hz=reader.readSignal(idx["RespRate"]).astype(np.float64),
            heart_rate_1hz=reader.readSignal(idx["HeartRate"]).astype(np.float64),
            impedance_1hz=reader.readSignal(idx["Impedance"]).astype(np.float64),
            fs_ecg=float(reader.getSampleFrequency(idx["ECG"])),
            fs_resp=float(reader.getSampleFrequency(idx["Resp"])),
            fs_trig=float(reader.getSampleFrequency(idx["R-Trigger"])),
            fs_rate=float(reader.getSampleFrequency(idx["RespRate"])),
        )
    finally:
        reader._close()


def detect_trigger_times(
    trigger_sig: np.ndarray,
    fs_hz: float = 1024.0,
    min_refractory_s: float = 0.22,
    threshold: float = 0.5,
) -> np.ndarray:
    """Detect rising-edge timestamps (seconds) with a physiological refractory guard."""
    active = np.asarray(trigger_sig, dtype=float) > threshold
    if active.size == 0:
        return np.empty(0, dtype=float)
    rising = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    if rising.size == 0:
        return np.empty(0, dtype=float)
    times = rising.astype(float) / float(fs_hz)
    kept = [times[0]]
    for t in times[1:]:
        if (t - kept[-1]) >= min_refractory_s:
            kept.append(t)
    return np.asarray(kept, dtype=float)


def interval_rate_in_window(
    event_times_s: np.ndarray,
    t0_s: float,
    t1_s: float,
    min_bpm: float,
    max_bpm: float,
) -> tuple[float, float, float, int]:
    """Compute robust median BPM, IQR, and max inter-event pause gap inside [t0_s, t1_s]."""
    in_win = event_times_s[(event_times_s >= t0_s) & (event_times_s <= t1_s)]
    if in_win.size < 2:
        max_gap = float(t1_s - t0_s)
        return float("nan"), float("nan"), max_gap, int(in_win.size)

    gaps = np.diff(in_win)
    lead_gap = float(in_win[0] - t0_s)
    trail_gap = float(t1_s - in_win[-1])
    max_gap = float(max(lead_gap, trail_gap, float(gaps.max())))

    rates = 60.0 / gaps
    valid_rates = rates[(rates >= min_bpm) & (rates <= max_bpm)]
    if valid_rates.size == 0:
        return float("nan"), float("nan"), max_gap, int(in_win.size)

    med = float(np.median(valid_rates))
    iqr = float(np.percentile(valid_rates, 75) - np.percentile(valid_rates, 25))
    return med, iqr, max_gap, int(in_win.size)


def _parabolic_peak_hz(freqs: np.ndarray, psd: np.ndarray, band_hz: tuple[float, float]) -> float:
    """Quadratic interpolation around the dominant PSD peak inside `band_hz`."""
    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    idx_band = np.flatnonzero(mask)
    if idx_band.size < 3:
        return float("nan")
    sub = psd[idx_band]
    local = int(np.argmax(sub))
    k = int(idx_band[local])
    if 0 < local < len(sub) - 1:
        y0, y1, y2 = float(psd[k - 1]), float(psd[k]), float(psd[k + 1])
        denom = y0 - 2.0 * y1 + y2
        if denom < -1e-18:
            delta = 0.5 * (y0 - y2) / denom
            delta = float(np.clip(delta, -0.5, 0.5))
            df = float(freqs[1] - freqs[0])
            return float(freqs[k] + delta * df)
    return float(freqs[k])


def spectral_resp_rate_256hz(
    resp_seg: np.ndarray,
    fs_raw: float = 256.0,
    band_hz: tuple[float, float] = (0.25, 2.0),
    nfft: int = 32768,
) -> tuple[float, float]:
    """Return dominant respiratory rate (bpm) and RMS excursion from a 256 Hz `Resp` window."""
    seg = np.asarray(resp_seg, dtype=float)
    if seg.size < int(round(fs_raw * 5)):
        return float("nan"), float("nan")
    centered = seg - float(np.mean(seg))
    rms = float(np.sqrt(np.mean(centered ** 2)))
    if rms < 1e-9:
        return float("nan"), 0.0
    detrended = signal.detrend(centered, type="linear")
    win = np.hanning(detrended.size)
    spec = np.fft.rfft(detrended * win, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs_raw)
    peak_hz = _parabolic_peak_hz(freqs, np.abs(spec) ** 2, band_hz)
    return (float(peak_hz * 60.0) if np.isfinite(peak_hz) else float("nan")), rms


def ecg_qrs_envelope_hr(
    ecg_256: np.ndarray,
    fs_ecg: float = 256.0,
) -> np.ndarray:
    """Compute squared-QRS bandpass envelope over an entire session for linear-phase R-R PSD checks."""
    x = np.asarray(ecg_256, dtype=float)
    x = np.nan_to_num(x - float(np.median(x)), nan=0.0)
    nyq = 0.5 * fs_ecg
    sos = signal.butter(4, [12.0 / nyq, min(35.0, nyq * 0.9) / nyq], btype="bandpass", output="sos")
    qrs = signal.sosfiltfilt(sos, x)
    # Smooth squared QRS energy (~120 ms boxcar) to expose fundamental beat repetition rate
    k_len = max(3, int(round(0.12 * fs_ecg)))
    kernel = np.ones(k_len, dtype=float) / float(k_len)
    return np.convolve(qrs ** 2, kernel, mode="same")


def spectral_hr_from_qrs_envelope(
    env_seg: np.ndarray,
    fs_raw: float = 256.0,
    band_hz: tuple[float, float] = (1.0, 4.0),
    nfft: int = 32768,
) -> float:
    """Extract fundamental heart rate (bpm) from a 30-s QRS energy envelope segment."""
    seg = np.asarray(env_seg, dtype=float)
    if seg.size < int(round(fs_raw * 5)) or float(np.std(seg)) < 1e-12:
        return float("nan")
    centered = seg - float(np.mean(seg))
    win = np.hanning(centered.size)
    spec = np.fft.rfft(centered * win, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs_raw)
    peak_hz = _parabolic_peak_hz(freqs, np.abs(spec) ** 2, band_hz)
    return float(peak_hz * 60.0) if np.isfinite(peak_hz) else float("nan")


def extract_window_references(
    edf: EdfRecording,
    centers_s: np.ndarray,
    lag_s: int | float,
    win_s: int = C.WIN_S,
    sat_thresh: float = 2.99,
) -> dict[str, np.ndarray]:
    """Align 1 Hz, 256 Hz, and 1024 Hz EDF reference channels to radar analysis windows.

    Matches `evaluation.window_ref_and_gating` index boundaries exactly (`floor(tc - win_s/2)`).
    """
    centers = np.asarray(centers_s, dtype=float)
    n_win = len(centers)

    br_1hz_30s, vwin_eval = window_ref_and_gating(
        edf.resp_256,
        edf.resp_rate_1hz,
        centers,
        lag_s,
        fs_raw=edf.fs_resp,
        win_s=win_s,
        sat_thresh=sat_thresh,
    )

    resp_trig_t = detect_trigger_times(edf.resp_trig_1024, fs_hz=edf.fs_trig, min_refractory_s=0.30)
    r_trig_t = detect_trigger_times(edf.r_trig_1024, fs_hz=edf.fs_trig, min_refractory_s=0.22)
    qrs_env = ecg_qrs_envelope_hr(edf.ecg_256, fs_ecg=edf.fs_ecg)

    fs_resp = int(round(edf.fs_resp))
    fs_ecg = int(round(edf.fs_ecg))
    n_resp_sec = len(edf.resp_256) // fs_resp
    n_ref = min(len(edf.resp_rate_1hz), len(edf.heart_rate_1hz), n_resp_sec)

    out: dict[str, np.ndarray] = {
        "br_1hz_30s": br_1hz_30s,
        "valid_vwin": vwin_eval,
        "br_trig_bpm": np.full(n_win, np.nan, dtype=float),
        "br_trig_iqr": np.full(n_win, np.nan, dtype=float),
        "br_256_spec_bpm": np.full(n_win, np.nan, dtype=float),
        "resp_256_rms": np.full(n_win, np.nan, dtype=float),
        "resp_sat_frac": np.full(n_win, np.nan, dtype=float),
        "resp_max_gap_s": np.full(n_win, np.nan, dtype=float),
        "resp_rate_slope_bpm_s": np.full(n_win, np.nan, dtype=float),
        "hr_1hz_30s": np.full(n_win, np.nan, dtype=float),
        "hr_rtrig_bpm": np.full(n_win, np.nan, dtype=float),
        "hr_ecg256_bpm": np.full(n_win, np.nan, dtype=float),
        "signaliq_mean": np.full(n_win, np.nan, dtype=float),
        "impedance_mean": np.full(n_win, np.nan, dtype=float),
        "in_bounds": np.zeros(n_win, dtype=bool),
    }

    tc = centers - float(lag_s)
    half = win_s / 2.0
    a_arr = np.floor(tc - half).astype(int)
    b_arr = a_arr + int(win_s)

    for w_idx in range(n_win):
        a = int(a_arr[w_idx])
        b = int(b_arr[w_idx])
        if a < 0 or b > n_ref:
            continue
        out["in_bounds"][w_idx] = True

        # 1 Hz trends inside [a, b)
        rr_seg = edf.resp_rate_1hz[a:b]
        hr_seg = edf.heart_rate_1hz[a:b]
        siq_seg = edf.signaliq_1hz[a:b]
        imp_seg = edf.impedance_1hz[a:b]

        out["hr_1hz_30s"][w_idx] = float(np.mean(hr_seg))
        out["signaliq_mean"][w_idx] = float(np.mean(siq_seg))
        out["impedance_mean"][w_idx] = float(np.mean(imp_seg))
        t_axis = np.arange(len(rr_seg), dtype=float)
        if np.std(rr_seg) > 0:
            slope, _ = np.polyfit(t_axis, rr_seg, 1)
            out["resp_rate_slope_bpm_s"][w_idx] = float(slope)
        else:
            out["resp_rate_slope_bpm_s"][w_idx] = 0.0

        # 256 Hz waveforms inside [a * fs, b * fs)
        s_r, e_r = a * fs_resp, b * fs_resp
        resp_seg = edf.resp_256[s_r:e_r]
        out["resp_sat_frac"][w_idx] = float(np.mean(np.abs(resp_seg) >= sat_thresh))
        br_spec, rms_val = spectral_resp_rate_256hz(resp_seg, fs_raw=edf.fs_resp)
        out["br_256_spec_bpm"][w_idx] = br_spec
        out["resp_256_rms"][w_idx] = rms_val

        s_e, e_e = a * fs_ecg, b * fs_ecg
        out["hr_ecg256_bpm"][w_idx] = spectral_hr_from_qrs_envelope(
            qrs_env[s_e:e_e], fs_raw=edf.fs_ecg
        )

        # 1024 Hz trigger interval rates
        br_med, br_iqr, br_gap, _ = interval_rate_in_window(
            resp_trig_t, float(a), float(b), min_bpm=12.0, max_bpm=120.0
        )
        out["br_trig_bpm"][w_idx] = br_med
        out["br_trig_iqr"][w_idx] = br_iqr
        out["resp_max_gap_s"][w_idx] = br_gap

        hr_med, _, _, _ = interval_rate_in_window(
            r_trig_t, float(a), float(b), min_bpm=60.0, max_bpm=230.0
        )
        out["hr_rtrig_bpm"][w_idx] = hr_med

    return out
