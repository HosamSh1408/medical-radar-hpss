"""P0 dual-lag harness (attached §3.2). No estimator changes.

Dynamic: per-config lag in ±120 s maximizing 0.65*rhoBR + 0.35*rhoHR.
HR inputs are optional stubs until the phase-velocity HR branch (P1) exists;
with hr=None the harness reduces to BR-only Pearson (current behavior),
documented as such.
Metadata-only: lag from file-start timestamps/headers, shared across configs.
"""
import numpy as np
from datetime import datetime
from pathlib import Path
import pyedflib
from . import config as C
from .data_loader import MEASURE_TO_RADAR, MEASURE_TO_REF


def joint_dynamic_lag(br_est, br_ref, hr_est=None, hr_ref=None,
                     maxlag=C.SYNC_MAX_LAG_S, w_br=0.65, w_hr=0.35,
                     ref_valid=None):
    """Grid-search lag maximizing w_br*corr(BR) + w_hr*corr(HR).

    ref_valid: optional 1s-resolution boolean mask on br_ref grid; when given,
    only valid samples enter each lag's correlation (attached scores valid
    windows; unmasked search lets saturation valleys/holds steer the lag).
    """
    br_est = np.asarray(br_est, dtype=float)
    br_ref = np.asarray(br_ref, dtype=float)
    tr = np.arange(len(br_est)) * C.UPDATE_S + C.WIN_S / 2.0
    tref = np.arange(len(br_ref)) * 1.0
    r_on = np.interp(tref, tr, br_est, left=np.nan, right=np.nan)
    if ref_valid is not None:
        ref_valid = np.asarray(ref_valid, dtype=bool)
    best, best_lag, best_br = -np.inf, 0, np.nan
    for lag in range(-int(maxlag), int(maxlag) + 1):
        if lag < 0:
            a, b = r_on[:lag], br_ref[-lag:]
            vv = ref_valid[-lag:] if ref_valid is not None else None
        elif lag > 0:
            a, b = r_on[lag:], br_ref[:-lag]
            vv = (ref_valid[:-lag] if (ref_valid is not None and lag < len(ref_valid))
                  else None)
        else:
            a, b = r_on, br_ref
            vv = ref_valid
        m = np.isfinite(a) & np.isfinite(b)
        if vv is not None:
            m = m & vv[:len(m)]
        if m.sum() < 30:
            continue
        cbr = np.corrcoef(a[m], b[m])[0, 1]
        score = cbr  # BR-only until HR branch exists (P1)
        if hr_est is not None and hr_ref is not None:
            he = np.asarray(hr_est, float)
            hr = np.asarray(hr_ref, float)
            he_on = np.interp(tref, tr, he, left=np.nan, right=np.nan)
            if lag < 0:
                ha, hb = he_on[:lag], hr[-lag:]
            elif lag > 0:
                ha, hb = he_on[lag:], hr[:-lag]
            else:
                ha, hb = he_on, hr
            mh = np.isfinite(ha) & np.isfinite(hb)
            if mh.sum() >= 30:
                chr_ = np.corrcoef(ha[mh], hb[mh])[0, 1]
                if np.isfinite(chr_):
                    score = w_br * cbr + w_hr * chr_
        if np.isfinite(score) and score > best:
            best, best_lag, best_br = score, lag, cbr
    return best_lag, float(best), (float(best_br) if np.isfinite(best_br) else float("nan"))


def hybrid_lag(dynamic_lag, metadata_lag_s, bound=100):
    """Predeclared R2 rule: dynamic ±120 search cannot reach true offsets
    beyond its range (proven: 3.1 true -435, search pinned at +120 boundary).
    If |dynamic| >= bound (search failed at boundary), use negated metadata
    file-start lag; else dynamic. Regressions possible (12.2, 5.2) and reported.
    Returns int lag or dynamic when metadata unavailable."""
    if metadata_lag_s is None:
        return int(dynamic_lag)
    if abs(int(dynamic_lag)) >= bound:
        return -round(metadata_lag_s)
    return int(dynamic_lag)


def _parse_radar_start(stem):
    # '20191120T122632.6722028_1' -> datetime
    try:
        core = stem.split("_")[0]
        return datetime.strptime(core.split(".")[0], "%Y%m%dT%H%M%S")
    except Exception:
        return None


def _parse_edf_start(name):
    # '..._L12122019_121140' -> datetime; header preferred (see below)
    try:
        tail = name.rsplit("_L", 1)[1]
        return datetime.strptime(tail, "%d%m%Y_%H%M%S")
    except Exception:
        return None


def metadata_lag(measurement, radar_dir=None, ref_dir=None):
    """lag = radar_start - edf_start in seconds (file-start timestamps/headers).

    Returns dict(radar_start, edf_header_start, edf_name_start, lag_s).
    Attached sensitivity subset range: -76..-36 s (11 sessions)."""
    from .config import RADAR_DIR, REF_DIR
    radar_dir = Path(radar_dir) if radar_dir else RADAR_DIR
    ref_dir = Path(ref_dir) if ref_dir else REF_DIR
    rname = MEASURE_TO_RADAR.get(measurement)
    ename = MEASURE_TO_REF.get(measurement)
    out = {"radar_start": None, "edf_header_start": None,
           "edf_name_start": None, "lag_s": None}
    if rname is None or ename is None:
        return out
    out["radar_start"] = _parse_radar_start(rname)
    out["edf_name_start"] = _parse_edf_start(ename)
    try:
        r = pyedflib.EdfReader(str(ref_dir / (ename + ".edf")))
        try:
            out["edf_header_start"] = r.getStartDatetime()
        finally:
            r._close()
    except Exception:
        pass
    base = out["edf_header_start"] or out["edf_name_start"]
    if out["radar_start"] is not None and base is not None:
        # header may be tz-naive vs aware; compare naive
        b = base.replace(tzinfo=None) if getattr(base, "tzinfo", None) else base
        out["lag_s"] = (out["radar_start"] - b).total_seconds()
    return out
