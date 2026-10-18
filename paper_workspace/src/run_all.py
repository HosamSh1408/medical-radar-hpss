"""Batch runner over paired measurements + metrics."""
import numpy as np
from .data_loader import load_radar, valid_paired_measurements, load_reference
from .run_measurement import run
from . import evaluation as E
from . import config as C


from . import lags as LAG


def run_one(measurement):
    bI, bQ = load_radar(measurement)
    resp_raw, resp_rate, fs_raw, fs_rate = load_reference(measurement)
    wins = run(bI, bQ)
    est = np.array([w["f_nls_nmf"] for w in wins])
    est_dft = np.array([w["f_dft"] for w in wins])
    est_nls = np.array([w["f_nls"] for w in wins])
    est_hpss = np.array([w["f_hpss"] for w in wins])
    nrem = np.array([w["n_removed"] for w in wins])
    nmf_converged = np.array([w["nmf_converged"] for w in wins], dtype=bool)
    t_nmf = float(np.mean([w["t_nmf"] for w in wins]))
    t_hpss = float(np.mean([w["t_hpss"] for w in wins]))

    centers_s = np.arange(len(est)) * C.UPDATE_S + C.WIN_S / 2.0
    meta_info = LAG.metadata_lag(measurement)
    meta_lag = meta_info.get("lag_s")

    # Sync using saturation-masked correlation anchored to metadata clock offset
    _, _, lag, corr = E.sync_arrays(
        est, resp_rate, metadata_lag_s=meta_lag, resp_raw=resp_raw, fs_raw=fs_raw
    )

    # Window-level reference averaging and whole-window saturation gating (Beltrão 2022)
    ref_win, vwin = E.window_ref_and_gating(resp_raw, resp_rate, centers_s, lag, fs_raw=fs_raw)
    ok = vwin & np.isfinite(ref_win) & np.isfinite(est)
    mm = ok & (nrem <= C.MIN_MOVE_MAXBASES)

    res = {
        "measurement": measurement, "lag_s": lag, "corr": float(corr),
        "n_windows": len(est),
        "t_nmf_ms": t_nmf * 1000.0, "t_hpss_ms": t_hpss * 1000.0,
        "full_nls_nmf": E.scores(est[ok], ref_win[ok]),
        "full_nls": E.scores(est_nls[ok], ref_win[ok]),
        "full_dft": E.scores(est_dft[ok], ref_win[ok]),
        "full_hpss": E.scores(est_hpss[ok], ref_win[ok]),
        "minmove_nls_nmf": E.scores(est[mm], ref_win[mm]),
        "est": est, "est_dft": est_dft, "est_nls": est_nls, "est_hpss": est_hpss,
        "ref": ref_win, "valid": ok, "minmove": mm, "centers_s": centers_s,
        "n_removed": nrem, "nmf_converged": nmf_converged,
    }
    return res



def main(measurements=None):
    ms = measurements or valid_paired_measurements()
    for m in ms:
        try:
            r = run_one(m)
            print(f"{m}: lag={r['lag_s']} corr={r['corr']:.2f} "
                  f"RMSE nmf={r['full_nls_nmf']['rmse']:.2f} "
                  f"acc6={r['full_nls_nmf']['acc@6']:.1f}% n={r['full_nls_nmf']['n']}")
        except Exception as e:
            print(f"{m}: FAILED {e}")


if __name__ == "__main__":
    main(["7.3"])
