"""Sequential batch runner with NPZ caching for the local paired cohort.

Single-process by design (audit A5): earlier documentation described an
"8-worker multiprocessing pool" that never existed in this module.
"""
import numpy as np
from pathlib import Path
import sys
from .run_all import run_one
from .protocols import MEASURE_INFO
from .cohort import PAPER_18
from .provenance import cache_is_current, provenance_json

CACHE = Path(__file__).resolve().parents[1] / "outputs" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

TOP18 = PAPER_18



def run_and_cache(measurements, force=False):
    failures = []
    for m in measurements:
        p = CACHE / f"{m}.npz"
        if p.exists() and not force:
            if cache_is_current(p, m):
                print(f"{m}: current cache, skip")
                continue
            print(f"{m}: stale or unverifiable cache, recompute")
        try:
            r = run_one(m)
            np.savez_compressed(
                p, est=r["est"], est_dft=r["est_dft"], est_nls=r["est_nls"],
                est_hpss=r["est_hpss"],
                ref=r["ref"], valid=r["valid"], minmove=r["minmove"],
                nmf_converged=r["nmf_converged"],
                lag=np.array(r["lag_s"]), corr=np.array(r["corr"]),
                t_nmf=np.array(r["t_nmf_ms"]), t_hpss=np.array(r["t_hpss_ms"]),
                provenance_json=np.array(provenance_json(m)))
            s = r["full_nls_nmf"]
            sh = r["full_hpss"]
            print(f"{m}: NMF RMSE={s['rmse']:.2f} acc6={s['acc@6']:.1f} | "
                  f"HPSS RMSE={sh['rmse']:.2f} acc6={sh['acc@6']:.1f} n={s['n']} "
                  f"tNMF={r['t_nmf_ms']:.1f}ms tHPSS={r['t_hpss_ms']:.1f}ms "
                  f"nmf_conv={100*np.mean(r['nmf_converged']):.1f}% "
                  f"pos={MEASURE_INFO.get(m,('?','?'))[0]}", flush=True)
        except Exception as e:
            failures.append(m)
            print(f"{m}: FAILED {e}", flush=True)
    if failures:
        raise RuntimeError(f"Batch failed for {len(failures)} measurement(s): {failures}")
    return True


def aggregate(measurements=None):
    ms = measurements or TOP18
    E, R, V, M, G = [], [], [], [], []
    per = []
    for m in ms:
        p = CACHE / f"{m}.npz"
        if not p.exists() or not cache_is_current(p, m):
            raise RuntimeError(
                f"{m}: cache missing or stale. Run `python3 main.py --batch --force` "
                "before generating aggregate results."
            )
        z = np.load(p)
        e, d, n, r, v, mm = (z["est"], z["est_dft"], z["est_nls"],
                            z["ref"], z["valid"], z["minmove"])
        E.append(e[v]); R.append(r[v]); M.append(mm[v])
        G.append(np.full(v.sum(), m))
        # per-measurement scores
        from . import evaluation as EV
        per.append((m, EV.scores(e[v], r[v]), EV.scores(e[v & mm], r[v & mm])))
    E, R, M = np.concatenate(E), np.concatenate(R), np.concatenate(M)
    from . import evaluation as EV
    return {"overall": EV.scores(E, R),
            "minimal": EV.scores(E[M], R[M]),
            "per": per, "n_meas": len(per), "n_win": len(E)}


if __name__ == "__main__":
    flags = {"--force", "--all-paired"}
    args = [a for a in sys.argv[1:] if a not in flags]
    force = "--force" in sys.argv[1:]
    if args:
        ms = args
    elif "--all-paired" in sys.argv[1:]:
        from .data_loader import valid_paired_measurements
        ms = valid_paired_measurements()  # all 33 paired sessions (audit A14)
    else:
        ms = TOP18
    run_and_cache(ms, force=force)
