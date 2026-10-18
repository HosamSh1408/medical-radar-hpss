"""Pilot A/B comparison on one full measurement with proposal metrics."""
import numpy as np
from .run_all import run_one

KEYMAP = {"full_dft": "est_dft", "full_nls": "est_nls",
          "full_nls_nmf": "est", "full_hpss": "est_hpss"}


def report(m="7.3"):
    r = run_one(m)
    print(f"== {m} lag={r['lag_s']} corr={r['corr']:.2f} n={r['n_windows']}")
    for k in ["full_dft", "full_nls", "full_nls_nmf", "full_hpss"]:
        s = r[k]
        e = r[KEYMAP[k]][r["valid"]]
        ref = r["ref"][r["valid"]]
        msk = np.isfinite(e) & np.isfinite(ref)
        err = np.abs(e[msk] - ref[msk])
        print(f"{k}: acc1 {100*(err<1).mean():.1f} acc3 {100*(err<3).mean():.1f} "
              f"acc5 {100*(err<5).mean():.1f} acc6 {s['acc@6']:.1f} "
              f"acc10 {s['acc@10']:.1f} rmse {s['rmse']:.2f} "
              f"mae {err.mean():.2f} bias {s['bias']:.2f} n={s['n']}")


if __name__ == "__main__":
    report("7.3")
