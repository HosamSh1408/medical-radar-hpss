"""Paper-style aggregate tables: scenario x method with proposal + paper metrics."""
import numpy as np
from .run_batch import CACHE, TOP18
from .provenance import cache_is_current
from .protocols import MEASURE_INFO
from . import evaluation as EV

METHODS = [("DFT", "est_dft"), ("NLS", "est_nls"),
           ("NLS+NMF", "est"), ("HPSS", "est_hpss")]
SCENARIOS = [
    ("ALL", lambda m: True),
    ("supine", lambda m: MEASURE_INFO[m][0] == "supine"),
    ("prone", lambda m: MEASURE_INFO[m][0] == "prone"),
    ("single", lambda m: not MEASURE_INFO[m][1]),
    ("twins", lambda m: MEASURE_INFO[m][1]),
]


def extended_scores(e, r):
    e = np.asarray(e, float)
    r = np.asarray(r, float)
    m = np.isfinite(e) & np.isfinite(r)
    e, r = e[m], r[m]
    s = EV.scores(e, r)
    err = np.abs(e - r)
    n = len(err)
    s["acc@1"] = float((err < 1).mean() * 100) if n else float("nan")
    s["acc@5"] = float((err < 5).mean() * 100) if n else float("nan")
    s["mae"] = float(err.mean()) if n else float("nan")
    return s


def build_table(measurements=None, agg="permeas", minimal=False):
    """Build scenario x method table.
    agg='permeas': average per-measurement metrics (Beltrão Eq. 3 / Fig. 6 protocol).
    agg='pooled': concatenate all windows globally.
    minimal=True: restrict to minimal-movement windows (Beltrão Fig. 7).
    """
    ms = measurements or TOP18
    data = {}
    for m in ms:
        p = CACHE / f"{m}.npz"
        if not p.exists() or not cache_is_current(p, m):
            raise RuntimeError(
                f"{m}: cache missing or stale. Run `python3 main.py --batch --force` "
                "before generating tables or figures."
            )
        z = np.load(p)
        mask = z["valid"]
        if minimal and "minmove" in z.files:
            mask = mask & z["minmove"]
        data[m] = {
            "mask": np.array(mask, bool),
            "ref": np.array(z["ref"]),
            "est_dft": np.array(z["est_dft"]),
            "est_nls": np.array(z["est_nls"]),
            "est": np.array(z["est"]),
            "est_hpss": np.array(z["est_hpss"]),
        }

    lines = []
    for scen, filt in SCENARIOS:
        sm = [m for m in data if filt(m)]
        for label, key in METHODS:
            if agg == "permeas":
                per_s = []
                total_n = 0
                for m in sm:
                    v = data[m]["mask"]
                    e = data[m][key][v]
                    r = data[m]["ref"][v]
                    if len(e) > 0 and np.isfinite(e).any():
                        s = extended_scores(e, r)
                        per_s.append(s)
                        total_n += s["n"]
                if not per_s:
                    continue
                avg_s = {
                    "acc@1": float(np.mean([x["acc@1"] for x in per_s])),
                    "acc@3": float(np.mean([x["acc@3"] for x in per_s])),
                    "acc@5": float(np.mean([x["acc@5"] for x in per_s])),
                    "acc@6": float(np.mean([x["acc@6"] for x in per_s])),
                    "acc@10": float(np.mean([x["acc@10"] for x in per_s])),
                    "rmse": float(np.mean([x["rmse"] for x in per_s])),
                    "mae": float(np.mean([x["mae"] for x in per_s])),
                    "bias": float(np.mean([x["bias"] for x in per_s])),
                    "n": total_n,
                }
                lines.append((scen, label, len(per_s), avg_s))
            else:
                E, R = [], []
                for m in sm:
                    v = data[m]["mask"]
                    e = data[m][key][v]
                    r = data[m]["ref"][v]
                    E.append(e)
                    R.append(r)
                if not E:
                    continue
                E_all = np.concatenate(E)
                R_all = np.concatenate(R)
                s = extended_scores(E_all, R_all)
                lines.append((scen, label, len(sm), s))
    return lines, data


def print_table(measurements=None, agg="permeas", minimal=False, title=None):
    lines, _ = build_table(measurements, agg=agg, minimal=minimal)
    t = title or f"Aggregate Table (agg={agg}, minimal={minimal})"
    print(f"\n=== {t} ===")
    print(f"{'scen':<8}{'method':<10}{'nM':>4} {'acc1':>6}{'acc3':>6}{'acc5':>6}"
          f"{'acc6':>6}{'acc10':>7} {'rmse':>6}{'mae':>6} {'n':>7}")
    for scen, label, nm, s in lines:
        print(f"{scen:<8}{label:<10}{nm:>4} {s['acc@1']:>6.1f}{s['acc@3']:>6.1f}"
              f"{s['acc@5']:>6.1f}{s['acc@6']:>6.1f}{s['acc@10']:>7.1f} "
              f"{s['rmse']:>6.2f}{s['mae']:>6.2f} {s['n']:>7}")


def main(measurements=None):
    ms = measurements or TOP18
    # 1. Paper-inspired protocol: per-measurement scenario averages (Figure 6)
    print_table(ms, agg="permeas", minimal=False,
                title="Paper-Inspired Protocol: Per-Measurement Scenario Mean (Fig. 6)")
    # 2. Minimal-movement subset (Figure 7)
    print_table(ms, agg="permeas", minimal=True,
                title="Minimal Movement Windows (Fig. 7, Per-Measurement Mean)")
    # 3. Pooled window-level aggregate
    print_table(ms, agg="pooled", minimal=False,
                title="Global Pooled Windows Aggregate")


if __name__ == "__main__":
    main()
