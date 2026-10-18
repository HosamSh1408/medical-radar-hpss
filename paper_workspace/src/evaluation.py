"""Sync + metrics (Eq.3, accuracy, Bland-Altman)."""
import numpy as np
from . import config as C


def sync_arrays(radar_bpm, ref_bpm_1hz, fs_radar_est=1 / C.UPDATE_S, fs_ref=1.0,
                metadata_lag_s=None, search_radius_s=60, resp_raw=None, fs_raw=256.0):
    """Maximise Pearson correlation over lag (paper offline sync).

    radar_bpm: estimates every UPDATE_S (0.5 Hz). ref: 1 Hz.
    If metadata_lag_s is provided, search is anchored to -metadata_lag_s +- search_radius_s.
    When resp_raw is provided, correlation is evaluated on saturation-free 30-s windows
    against the 30-s window-averaged reference rate (preventing saturated reference
    valleys from pulling the alignment lag).
    Returns (radar_aligned, ref_aligned, lag_s, corr).
    """
    radar_bpm = np.asarray(radar_bpm, dtype=float)
    ref = np.asarray(ref_bpm_1hz, dtype=float)
    tr = np.arange(len(radar_bpm)) * C.UPDATE_S + C.WIN_S / 2.0
    tref = np.arange(len(ref)) * 1.0
    r_on_ref = np.interp(tref, tr, radar_bpm, left=np.nan, right=np.nan)
    best_corr, best_lag = -np.inf, 0

    if metadata_lag_s is not None:
        c0 = int(-round(metadata_lag_s))
        lags = range(c0 - search_radius_s, c0 + search_radius_s + 1)
    else:
        maxlag = int(C.SYNC_MAX_LAG_S)
        lags = range(-maxlag, maxlag + 1)

    if resp_raw is not None:
        for min_win in (60, 15):
            for lag in lags:
                ref_w, vwin = window_ref_and_gating(resp_raw, ref, tr, lag, fs_raw=fs_raw)
                ok = vwin & np.isfinite(ref_w) & np.isfinite(radar_bpm)
                if ok.sum() < min_win:
                    continue
                c = np.corrcoef(radar_bpm[ok], ref_w[ok])[0, 1]
                if np.isfinite(c) and c > best_corr:
                    best_corr, best_lag = c, lag
            if np.isfinite(best_corr):
                break

    if not np.isfinite(best_corr):
        for lag in lags:
            if lag < 0:
                a = r_on_ref[:lag]
                b = ref[-lag:]
            elif lag > 0:
                a = r_on_ref[lag:]
                b = ref[:-lag] if lag < len(ref) else ref[:0]
            else:
                a, b = r_on_ref, ref
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 30:
                continue
            c = np.corrcoef(a[m], b[m])[0, 1]
            if np.isfinite(c) and c > best_corr:
                best_corr, best_lag = c, lag

    if best_lag < 0:
        a = r_on_ref[:best_lag]
        b = ref[-best_lag:]
    elif best_lag > 0:
        a = r_on_ref[best_lag:]
        b = ref[:-best_lag]
    else:
        a, b = r_on_ref, ref
    return a, b, best_lag, best_corr


def window_ref_and_gating(resp_raw, resp_rate, centers_s, lag_s, fs_raw=256.0,
                          win_s=C.WIN_S, sat_thresh=2.99):
    """Compute 30-s window-average reference rate and whole-window saturation gating.

    Follows paper: 30-s processing windows corresponding to non-saturated
    reference device segments (|V_resp| < sat_thresh everywhere in the 30-s window).
    Target ground truth is the average breathing frequency over that 30 s (Fig.4g).
    """
    rate = np.asarray(resp_rate, dtype=float)
    raw = np.asarray(resp_raw, dtype=float)
    fsr = int(round(fs_raw))
    n_sec = len(raw) // fsr
    sat_sec = (np.abs(raw[:n_sec * fsr]) >= sat_thresh).reshape(n_sec, fsr).any(axis=1)

    n_ref = min(len(rate), len(sat_sec))
    tc = centers_s - lag_s
    half = win_s / 2.0
    a = np.floor(tc - half).astype(int)
    b = a + int(win_s)
    inb = (a >= 0) & (b <= n_ref)

    cs_sat = np.r_[0, np.cumsum(sat_sec[:n_ref])]
    cs_rate = np.r_[0, np.cumsum(rate[:n_ref])]

    ref_avg = np.full(len(centers_s), np.nan)
    vwin = np.zeros(len(centers_s), dtype=bool)

    ai, bi = a[inb], b[inb]
    ref_avg[inb] = (cs_rate[bi] - cs_rate[ai]) / float(win_s)
    vwin[inb] = (cs_sat[bi] - cs_sat[ai]) == 0

    return ref_avg, vwin



def ref_saturation_mask(resp_rate, flat_s=3, drop_bpm=15):
    """Detect unreliable reference segments (paper Fig.5 valleys).

    Marks invalid where flat (diff==0) or abrupt drop>drop_bpm.
    Returns boolean valid mask.
    """
    r = np.asarray(resp_rate, dtype=float)
    valid = np.ones(len(r), dtype=bool)
    d = np.abs(np.diff(r, prepend=r[0]))
    # flat
    flat = np.zeros(len(r), dtype=bool)
    run = 0
    for i in range(1, len(r)):
        if r[i] == r[i - 1]:
            run += 1
        else:
            if run >= flat_s:
                flat[i - run - 1:i] = True
            run = 0
    # flush trailing flat run (bug fix: previously dropped if file ends saturated)
    if run >= flat_s:
        flat[len(r) - run - 1:len(r)] = True
    valid[flat] = False
    # abrupt valleys: drop then quick recovery -> invalidate +-5 s around large negative jumps
    for i in range(1, len(r)):
        if (r[i - 1] - r[i]) > drop_bpm * 0.5:
            valid[max(0, i - 5):i + 6] = False
    # also invalidate non-physiological
    valid[(r < 10) | (r > 120)] = False
    return valid


def ref_valid_from_raw(resp_raw, resp_rate, fs_raw=256.0,
                        flat_tol=1e-9, clip_frac=0.5, std_thresh=1e-3):
    """Raw-ADC-based validity (paper: ADC clips at max magnitude -> constant amplitude).

    For each 1Hz RespRate sample j, inspect raw segment j*fs_raw:(j+1)*fs_raw:
    invalid if raw std < std_thresh (flat/clipped) or >clip_frac of samples
    within flat_tol of segment min/max plateau. Falls back to rate-based mask
    via ref_saturation_mask. Returns boolean valid (len == len(resp_rate)).
    """
    raw = np.asarray(resp_raw, dtype=float)
    rate = np.asarray(resp_rate, dtype=float)
    n_rate = len(rate)
    # resample-safe: raw length should be fs_raw * n_rate (256*1536); trim/pad
    valid_raw = np.ones(n_rate, dtype=bool)
    sfr = int(round(fs_raw))
    for j in range(n_rate):
        s = j * sfr
        e = min(s + sfr, len(raw))
        seg = raw[s:e] if e > s else raw[-sfr:]
        if len(seg) < 8:
            valid_raw[j] = False
            continue
        sd = float(np.std(seg))
        if sd < std_thresh:
            valid_raw[j] = False
            continue
        # plateau fraction near min or max (clipping)
        mn, mx = float(np.min(seg)), float(np.max(seg))
        span = mx - mn
        if span < 1e-12:
            valid_raw[j] = False
            continue
        lo_frac = np.mean(np.abs(seg - mn) <= flat_tol * max(1.0, abs(span)))
        hi_frac = np.mean(np.abs(seg - mx) <= flat_tol * max(1.0, abs(span)))
        if max(lo_frac, hi_frac) > clip_frac:
            valid_raw[j] = False
    valid_rate = ref_saturation_mask(rate)
    return valid_raw & valid_rate


def scores(est, ref):
    est = np.asarray(est, dtype=float)
    ref = np.asarray(ref, dtype=float)
    m = np.isfinite(est) & np.isfinite(ref)
    est, ref = est[m], ref[m]
    err = np.abs(est - ref)
    out = {}
    for t in C.ACC_THRESHOLDS:
        out[f"acc@{t}"] = float(np.mean(err < t) * 100) if len(err) else float("nan")
    out["rmse"] = float(np.sqrt(np.mean((est - ref) ** 2))) if len(err) else float("nan")
    out["bias"] = float(np.mean(est - ref)) if len(err) else float("nan")
    out["n"] = int(len(err))
    return out


def bland_altman(est, ref):
    est = np.asarray(est, dtype=float)
    ref = np.asarray(ref, dtype=float)
    m = np.isfinite(est) & np.isfinite(ref)
    est, ref = est[m], ref[m]
    mean = (est + ref) / 2
    diff = est - ref
    bias = float(np.mean(diff))
    sd = float(np.std(diff))
    return {"mean": mean, "diff": diff, "bias": bias, "sd": sd,
            "loa_up": bias + 1.96 * sd, "loa_lo": bias - 1.96 * sd}
