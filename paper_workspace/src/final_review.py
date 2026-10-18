"""Phase 5 final independent review: regenerate the two audit JSON artifacts.

Regenerates (in audit/):
  * final_review_independent.json      -- baselines, paired stats, TOST, window-level
  * final_review_tracking_and_noise.json -- within-session tracking, noise floor

Every quantity is recomputed from the SHA-256 provenance-locked .npz caches
(no EDF re-reads) with explicit, documented definitions recorded in the
"_definitions" block of each output file. Run:  python3 -m paper_workspace.src.final_review

Note: this module is NOT part of provenance._CORE_FILES; it only reads caches
and never mutates the pipeline, so the 33 locked caches remain valid.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from . import config as C
from .cohort import PAPER_18, local_pool, pool30, retained23
from .evaluation import scores  # noqa: F401  (kept for API parity; not used)
from .filtering import autocorr_coarse, bandpass
from .estimation import nls_estimate
from .hpss import mitigate_hpss

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2]
CACHE = _HERE.parents[1] / "outputs" / "cache"
AUDIT = ROOT / "audit"

TOST_MARGIN = 0.50  # bpm, pre-specified (WS4)
NOISE_SEED = 20261011  # matches reconstruction_multiverse.py bootstrap seed
N_NOISE_WINDOWS = 200

COHORTS = {
    "PAPER_18": list(PAPER_18),
    "retained23": retained23(),
    "pool30": pool30(),
    "all_paired": local_pool(),
}


def _load_cohort(measurements):
    """Return list of per-session dicts with valid-masked arrays."""
    out = []
    for m in measurements:
        z = np.load(CACHE / f"{m}.npz", allow_pickle=False)
        v = np.asarray(z["valid"], bool)
        out.append({
            "id": m,
            "est": np.asarray(z["est"], float)[v],
            "est_hpss": np.asarray(z["est_hpss"], float)[v],
            "est_nls": np.asarray(z["est_nls"], float)[v],
            "est_dft": np.asarray(z["est_dft"], float)[v],
            "ref": np.asarray(z["ref"], float)[v],
            "minmove": np.asarray(z["minmove"], bool)[v],
            "lag": int(z["lag"]),
        })
    return [s for s in out if len(s["ref"]) > 0]


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _per_session_rmse(sess, key):
    return np.array([_rmse(s[key], s["ref"]) for s in sess])


def _patient_of(measurement):
    return measurement.split(".")[0]


def _tost_ci90(d, margin=TOST_MARGIN):
    """Paired one-sample TOST at +-margin; returns (p, ci90_lo, ci90_hi)."""
    d = np.asarray(d, float)
    n = len(d)
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    t1 = (mean + margin) / se
    t2 = (margin - mean) / se
    p = max(1 - stats.t.cdf(t1, n - 1), 1 - stats.t.cdf(t2, n - 1))
    tq = stats.t.ppf(0.95, n - 1)
    return float(p), float(mean - tq * se), float(mean + tq * se)


def _sign_test(better, total):
    return float(stats.binomtest(better, total, 0.5).pvalue)


def build_independent():
    defs = {
        "scope": "PAPER_18=18-session benchmark; retained23/pool30/all_paired=cohort multiverse; sessions with zero valid windows are excluded from all per-measurement aggregates.",
        "per_session_rmse": "RMSE_k = sqrt(mean((est-ref)^2)) over the session's valid 30-s windows; sessions weighted equally (per-measurement mean = unweighted mean of RMSE_k).",
        "dft_nls_nmf_hpss": "Per-measurement means of per-session RMSE for each branch.",
        "const_global_oracle": "Per-measurement mean of per-session RMSE of (ref vs the cohort-pooled mean of all valid refs) -- in-sample constant predictor.",
        "const_loso": "Per-measurement mean of per-session RMSE of (ref vs the MEDIAN of the OTHER sessions' pooled valid refs) -- leave-one-session-out constant predictor.",
        "session_mean_oracle": "Mean across sessions of the per-session reference SD -- lower bound for within-session tracking RMSE.",
        "paired": "delta_k = RMSE_hpss_k - RMSE_nmf_k; stats over sessions. loo_range = min/max of the mean delta when each single session is left out. largest_abs_deltas = top-3 |delta_k| with session id.",
        "tost_0p5": "Paired one-sample TOST at margin +-0.50 bpm: p = max(P(t>t1), P(t<t2)); ci90 = mean +- t_{0.95,n-1}*SE (sample SD). patient_level = per-patient mean of session deltas. max_abs_patient_ci90_bound = largest |CI bound|; patient-level equivalence holds iff this is < 0.50 (it is 0.508 for all_paired -> patient-level TOST p=0.0528, i.e. marginal failure at the widest cohort; the pre-registered analysis unit is the session).",
        "window_level_PAPER18": "Pooled over PAPER_18 valid windows. harmonic bins: est/ref ratio in [0.4,0.6] (half) or [1.8,2.2] (double); gross = |err|>10 bpm. Bland-Altman: bias +- 1.96*sample_sd(ddof=1) of (est-ref). heavy_move = windows with n_removed>2 (minmove False).",
        "mitigation_vs_raw": "Per-measurement paired comparison of each mitigated branch vs Raw NLS; wins = sessions with lower RMSE; two-sided Wilcoxon signed-rank.",
        "lags": "Synchronization lag (s) stored in the provenance-locked caches (evaluation.sync_arrays, metadata-anchored +-60 s search).",
    }
    baselines = {}
    paired = {}
    tost = {}
    mit = {}
    lags = {}
    for name, ms in COHORTS.items():
        sess = _load_cohort(ms)
        n = len(sess)
        refs = [s["ref"] for s in sess]
        pooled = np.concatenate(refs)
        gmean = float(pooled.mean())

        r_nmf = _per_session_rmse(sess, "est")
        r_hpss = _per_session_rmse(sess, "est_hpss")
        r_nls = _per_session_rmse(sess, "est_nls")
        r_dft = _per_session_rmse(sess, "est_dft")

        const_global = []
        const_loso = []
        sess_sd = []
        for i, s in enumerate(sess):
            const_global.append(_rmse(s["ref"], np.full(len(s["ref"]), gmean)))
            others = np.concatenate([refs[j] for j in range(n) if j != i])
            const_loso.append(_rmse(s["ref"], np.full(len(s["ref"]), np.median(others))))
            sess_sd.append(float(s["ref"].std()))

        baselines[name] = {
            "dft": round(float(r_dft.mean()), 3),
            "nls": round(float(r_nls.mean()), 3),
            "nmf": round(float(r_nmf.mean()), 4),
            "hpss": round(float(r_hpss.mean()), 4),
            "const_loso": round(float(np.mean(const_loso)), 3),
            "const_global_oracle": round(float(np.mean(const_global)), 3),
            "session_mean_oracle": round(float(np.mean(sess_sd)), 3),
            "n": n,
        }

        delta = r_hpss - r_nmf
        loo = [float(np.delete(delta, i).mean()) for i in range(n)]
        order = np.argsort(-np.abs(delta))[:3]
        paired[name] = {
            "mean_delta": round(float(delta.mean()), 4),
            "median_delta": round(float(np.median(delta)), 4),
            "hpss_better_sessions": f"{int((delta < 0).sum())}/{n}",
            "sign_test_p": round(_sign_test(int((delta < 0).sum()), n), 4),
            "loo_range": [round(min(loo), 4), round(max(loo), 4)],
            "largest_abs_deltas": [[sess[i]["id"], round(float(delta[i]), 3)] for i in order],
        }

        ps, lo, hi = _tost_ci90(delta)
        pats = {}
        for s, d in zip(sess, delta):
            pats.setdefault(_patient_of(s["id"]), []).append(d)
        pat_d = np.array([np.mean(v) for v in pats.values()])
        pp, plo, phi = _tost_ci90(pat_d)
        tost[name] = {
            "session_level": {"n": n, "p": round(ps, 4), "ci90": [round(lo, 3), round(hi, 3)]},
            "patient_level": {"n": len(pat_d), "p": round(pp, 4), "ci90": [round(plo, 3), round(phi, 3)]},
            "max_abs_patient_ci90_bound": round(max(abs(plo), abs(phi)), 3),
        }

        raw = r_nls
        d_nmf = r_nmf - raw
        d_hpss = r_hpss - raw
        mit[name] = {
            "nmf_minus_raw": round(float(d_nmf.mean()), 3),
            "nmf_wins": f"{int((d_nmf < 0).sum())}/{n}",
            "p_wilcoxon_nmf": round(float(stats.wilcoxon(d_nmf).pvalue), 4),
            "hpss_minus_raw": round(float(d_hpss.mean()), 3),
            "hpss_wins": f"{int((d_hpss < 0).sum())}/{n}",
            "p_wilcoxon_hpss": round(float(stats.wilcoxon(d_hpss).pvalue), 4),
        }

    # Window-level PAPER_18 pooled
    sess = _load_cohort(COHORTS["PAPER_18"])
    E = np.concatenate([s["est"] for s in sess])
    H = np.concatenate([s["est_hpss"] for s in sess])
    N = np.concatenate([s["est_nls"] for s in sess])
    R = np.concatenate([s["ref"] for s in sess])
    MM = np.concatenate([s["minmove"] for s in sess])
    err_n, err_h = E - R, H - R
    heavy = ~MM
    window_level = {
        "n_windows": int(len(R)),
        "ref_range_bpm": [round(float(R.min()), 1), round(float(R.max()), 1)],
        "ref_mean_sd": [round(float(R.mean()), 2), round(float(R.std(ddof=1)), 2)],
        "pct_windows_nmf_hpss_within_0.5bpm": round(float(100 * np.mean(np.abs(E - H) <= 0.5)), 2),
        "corr_err_nmf_vs_hpss": round(float(np.corrcoef(err_n, err_h)[0, 1]), 4),
        "harmonic_errors_nmf_pct": {
            "half(0.4-0.6)": round(float(100 * np.mean((E / R >= 0.4) & (E / R <= 0.6))), 2),
            "double(1.8-2.2)": round(float(100 * np.mean((E / R >= 1.8) & (E / R <= 2.2))), 2),
            "gross_|err|>10": round(float(100 * np.mean(np.abs(err_n) > 10)), 2),
        },
        "harmonic_errors_hpss_pct": {
            "half(0.4-0.6)": round(float(100 * np.mean((H / R >= 0.4) & (H / R <= 0.6))), 2),
            "double(1.8-2.2)": round(float(100 * np.mean((H / R >= 1.8) & (H / R <= 2.2))), 2),
            "gross_|err|>10": round(float(100 * np.mean(np.abs(err_h) > 10)), 2),
        },
        "bland_altman_pooled_nmf(bias,lo,hi)": [
            round(float(err_n.mean()), 2),
            round(float(err_n.mean() - 1.96 * err_n.std(ddof=1)), 2),
            round(float(err_n.mean() + 1.96 * err_n.std(ddof=1)), 2),
        ],
        "bland_altman_pooled_hpss(bias,lo,hi)": [
            round(float(err_h.mean()), 2),
            round(float(err_h.mean() - 1.96 * err_h.std(ddof=1)), 2),
            round(float(err_h.mean() + 1.96 * err_h.std(ddof=1)), 2),
        ],
        "pooled_rmse_nmf_hpss_rawnls": [
            round(_rmse(E, R), 3), round(_rmse(H, R), 3), round(_rmse(N, R), 3),
        ],
        "heavy_move_windows": int(heavy.sum()),
        "heavy_move_rmse_nmf_hpss_rawnls": [
            round(_rmse(E[heavy], R[heavy]), 3),
            round(_rmse(H[heavy], R[heavy]), 3),
            round(_rmse(N[heavy], R[heavy]), 3),
        ],
    }

    z73 = np.load(CACHE / "7.3.npz", allow_pickle=False)
    z121 = np.load(CACHE / "12.1.npz", allow_pickle=False)
    z31 = np.load(CACHE / "3.1.npz", allow_pickle=False)
    lags = {"7.3": int(z73["lag"]), "12.1": int(z121["lag"]), "3.1": int(z31["lag"])}

    return {
        "_definitions": defs,
        "baselines": baselines,
        "paired": paired,
        "tost_0p5": tost,
        "window_level_PAPER18": window_level,
        "mitigation_vs_raw": mit,
        "lags": lags,
    }


def _noise_pipeline(d):
    """Raw NLS and HPSS+NLS on a bandpassed complex window (mirrors run_measurement)."""
    f_raw, _, _ = nls_estimate(d, autocorr_coarse(d))
    xh = mitigate_hpss(d)["x_hat"]
    d_h = bandpass(xh)
    f_hpss, _, _ = nls_estimate(d_h, autocorr_coarse(d_h))
    return f_raw, f_hpss


def build_tracking_and_noise():
    defs = {
        "within_session_pearson_r": "Pearson r between per-window estimates and reference rates within each session (valid windows only); mean across sessions.",
        "rmse_*_per_window": "Per-measurement mean of per-session RMSE (same as audit/final_review_independent.json).",
        "rmse_*_session_median": "Median across sessions of the per-session RMSE (typical-session value, robust to outlier sessions).",
        "mae_*_session_median": "Median across sessions of the per-session mean absolute error.",
        "mean_within_session_ref_sd": "Mean across sessions of the per-session reference SD (tracking difficulty floor).",
        "noise_floor": (
            "Complex white Gaussian noise, 200 windows x 480 samples (seed 20261011), passed through the "
            "identical downstream chain: Raw = autocorr_coarse->nls_estimate on bandpassed noise; HPSS = "
            "mitigate_hpss->bandpass->autocorr_coarse->nls_estimate. pct in [25.8, 81.5] uses the PAPER_18 "
            "reference range. rmse_hpss_vs_shuffled = RMS(est_noise_hpss - ref_shuffled) over 200 i.i.d. "
            "draws from PAPER_18 pooled valid refs -- the RMSE a pipeline outputting plausible-looking but "
            "uncorrelated rates would score."
        ),
        "prior_artifact_note": (
            "This file regenerates audit/final_review_tracking_and_noise.json. The earlier untracked version "
            "contained 'rmse_*_session_median_of_radar' whose generating code was not committed and whose "
            "definition could not be recovered; it is replaced by the explicitly defined "
            "'rmse_*_session_median' / 'mae_*_session_median' fields."
        ),
    }
    out = {}
    for name, ms in COHORTS.items():
        sess = _load_cohort(ms)
        r_nmf = _per_session_rmse(sess, "est")
        r_hpss = _per_session_rmse(sess, "est_hpss")
        rs_n, rs_h, sds, maes_n, maes_h = [], [], [], [], []
        for s in sess:
            e, h, r = s["est"], s["est_hpss"], s["ref"]
            if e.std() > 0 and r.std() > 0:
                rs_n.append(float(np.corrcoef(e, r)[0, 1]))
            if h.std() > 0 and r.std() > 0:
                rs_h.append(float(np.corrcoef(h, r)[0, 1]))
            sds.append(float(r.std()))
            maes_n.append(float(np.mean(np.abs(e - r))))
            maes_h.append(float(np.mean(np.abs(h - r))))
        out[name] = {
            "within_session_pearson_r_nmf_mean": round(float(np.mean(rs_n)), 3),
            "within_session_pearson_r_hpss_mean": round(float(np.mean(rs_h)), 3),
            "sessions_r_hpss_lt_0.3": int(np.sum(np.array(rs_h) < 0.3)),
            "n": len(sess),
            "rmse_nmf_per_window": round(float(r_nmf.mean()), 3),
            "rmse_hpss_per_window": round(float(r_hpss.mean()), 3),
            "rmse_nmf_session_median": round(float(np.median(r_nmf)), 3),
            "rmse_hpss_session_median": round(float(np.median(r_hpss)), 3),
            "mae_nmf_session_median": round(float(np.median(maes_n)), 3),
            "mae_hpss_session_median": round(float(np.median(maes_h)), 3),
            "mean_within_session_ref_sd": round(float(np.mean(sds)), 3),
        }

    rng = np.random.default_rng(NOISE_SEED)
    f_raw = []
    f_hpss = []
    for _ in range(N_NOISE_WINDOWS):
        x = rng.standard_normal(C.WIN_S * C.FS_RADAR) + 1j * rng.standard_normal(C.WIN_S * C.FS_RADAR)
        a, b = _noise_pipeline(bandpass(x))
        f_raw.append(a)
        f_hpss.append(b)
    f_raw = np.asarray(f_raw)
    f_hpss = np.asarray(f_hpss)

    sess18 = _load_cohort(COHORTS["PAPER_18"])
    pooled_ref = np.concatenate([s["ref"] for s in sess18])
    ref_shuf = rng.choice(pooled_ref, size=N_NOISE_WINDOWS, replace=True)
    rmse_shuf = _rmse(f_hpss, ref_shuf)

    def mq(x):
        return [round(float(np.median(x)), 1), round(float(np.percentile(x, 25)), 1),
                round(float(np.percentile(x, 75)), 1)]

    out["noise_floor"] = {
        "rawnls_on_noise_median_iqr": mq(f_raw),
        "hpss_on_noise_median_iqr": mq(f_hpss),
        "rmse_noise_hpss_vs_shuffled_PAPER18_refs": round(float(rmse_shuf), 2),
        "pct_noise_estimates_in_25_82bpm": round(
            float(100 * np.mean((f_hpss >= 25.8) & (f_hpss <= 81.5))), 1),
    }
    return {"_definitions": defs, **out}


def _check(expectations, obj, label):
    """Print PASS/FAIL for headline values (tolerance from table)."""
    print(f"\n== Verification: {label} ==")
    fails = 0
    for path, exp, tol in expectations:
        node = obj
        keys = path.split(".") if isinstance(path, str) else path
        for k in keys:
            if isinstance(node, list):
                node = node[int(k)]
            else:
                node = node[k]
        ok = abs(float(node) - exp) <= tol
        fails += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {path} = {node} (expected ~{exp} +-{tol})")
    return fails


def main():
    ind = build_independent()
    trk = build_tracking_and_noise()
    (AUDIT / "final_review_independent.json").write_text(json.dumps(ind, indent=1) + "\n")
    (AUDIT / "final_review_tracking_and_noise.json").write_text(json.dumps(trk, indent=1) + "\n")
    print("Wrote audit/final_review_independent.json and audit/final_review_tracking_and_noise.json")

    exp_ind = [
        ("paired.PAPER_18.mean_delta", -0.0648, 5e-4),
        ("paired.retained23.mean_delta", 0.0249, 5e-4),
        ("paired.pool30.mean_delta", -0.0956, 5e-4),
        ("paired.all_paired.mean_delta", 0.0117, 5e-4),
        ("tost_0p5.PAPER_18.session_level.p", 0.0001, 1e-4),
        ("tost_0p5.all_paired.session_level.p", 0.0171, 1e-4),
        ("tost_0p5.PAPER_18.patient_level.p", 0.0062, 1e-4),
        ("tost_0p5.PAPER_18.max_abs_patient_ci90_bound", 0.348, 1e-3),
        ("tost_0p5.all_paired.max_abs_patient_ci90_bound", 0.508, 1e-3),
        ("tost_0p5.all_paired.patient_level.p", 0.0528, 1e-4),
        ("baselines.PAPER_18.const_global_oracle", 8.023, 1e-3),
        ("baselines.PAPER_18.const_loso", 8.209, 1e-3),
        ("baselines.all_paired.const_loso", 8.194, 1e-3),
        ("baselines.PAPER_18.session_mean_oracle", 4.934, 1e-3),
        ("mitigation_vs_raw.PAPER_18.nmf_minus_raw", -0.625, 1e-3),
        ("mitigation_vs_raw.all_paired.hpss_minus_raw", -0.888, 1e-3),
        ("window_level_PAPER18.pooled_rmse_nmf_hpss_rawnls.0", 7.243, 1e-3),
        ("window_level_PAPER18.pooled_rmse_nmf_hpss_rawnls.1", 7.217, 1e-3),
        ("window_level_PAPER18.heavy_move_windows", 998, 0),
        ("window_level_PAPER18.heavy_move_rmse_nmf_hpss_rawnls.0", 8.323, 1e-3),
    ]
    exp_trk = [
        ("PAPER_18.within_session_pearson_r_nmf_mean", 0.460, 1e-3),
        ("PAPER_18.within_session_pearson_r_hpss_mean", 0.465, 1e-3),
        (("all_paired", "sessions_r_hpss_lt_0.3"), 9, 0),
        ("PAPER_18.rmse_hpss_per_window", 6.798, 1e-3),
        ("PAPER_18.mean_within_session_ref_sd", 4.934, 1e-3),
    ]
    f1 = _check(exp_ind, ind, "final_review_independent.json")
    f2 = _check(exp_trk, trk, "final_review_tracking_and_noise.json")
    if f1 + f2:
        raise SystemExit(f"{f1 + f2} verification check(s) FAILED")
    print("\nAll final-review self-checks passed.")


if __name__ == "__main__":
    main()
