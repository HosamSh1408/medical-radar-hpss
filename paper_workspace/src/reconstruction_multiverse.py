"""Evaluate locked NMF versus HPSS across defensible local cohort variants.

This is deliberately a cohort multiverse first. It consumes only provenance-
valid active caches and does not tune HPSS or select a favorable variant from
the results. Protocol variants that require raw recomputation remain listed in
the unresolved-protocol ledger until separately implemented.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .cohort import PAPER_18, local_pool, pool30, retained23
from .data_loader import valid_paired_measurements
from .provenance import cache_is_current
from .run_batch import CACHE
from .result_tables import extended_scores

OUT = Path(__file__).resolve().parents[1] / "outputs" / "audits"
BOOTSTRAP_SEED = 20261011


def measurement_sort(items):
    return sorted(items, key=lambda x: tuple(int(p) for p in x.split(".")))


def bootstrap_ci(values, n_resamples=2000):
    """Return a deterministic percentile CI for session-level paired deltas."""
    values = np.asarray(values, float)
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED + values.size)
    samples = rng.choice(values, size=(n_resamples, values.size), replace=True)
    means = samples.mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def finite_mean(field, rows):
    values = [r[field] for r in rows if np.isfinite(r[field])]
    return float(np.mean(values)) if values else float("nan")


def cohorts():
    return {
        "all_paired_local": {
            "measurements": measurement_sort(local_pool()),
            "selection_risk": "predefined_all_paired",
        },
        "pool30_no_zero_overlap": {
            "measurements": measurement_sort(pool30()),
            "selection_risk": "predefined_metadata_overlap_exclusion",
        },
        "retained23_named_exclusions": {
            "measurements": measurement_sort(retained23()),
            "selection_risk": "predefined_named_exclusions",
        },
        "current_reference_validity_ranked_18": {
            "measurements": measurement_sort(PAPER_18),
            "selection_risk": "outcome_dependent_reference_validity_ranked",
        },
    }


def load_session(m):
    path = CACHE / f"{m}.npz"
    if not path.exists() or not cache_is_current(path, m):
        raise RuntimeError(
            f"{m}: missing or stale cache. Run `python3 -m paper_workspace.src.run_batch "
            f"--force {m}` before the multiverse."
        )
    with np.load(path, allow_pickle=False) as z:
        mask = np.asarray(z["valid"], bool)
        ref = np.asarray(z["ref"], float)
        out = {}
        for method, key in (("NMF", "est"), ("HPSS", "est_hpss")):
            out[method] = extended_scores(np.asarray(z[key])[mask], ref[mask])
        out["t_nmf_ms"] = float(z["t_nmf"])
        out["t_hpss_ms"] = float(z["t_hpss"])
        out["n_valid"] = int(mask.sum())
        out["patient_group"] = m.split(".")[0]
        return out


def summarize(variant, info, session_rows):
    by_patient = defaultdict(list)
    by_measurement = defaultdict(list)
    for row in session_rows:
        by_patient[row["patient_group"]].append(row)
        by_measurement[row["measurement"]].append(row)

    def mean(field, rows):
        values = [r[field] for r in rows if np.isfinite(r[field])]
        return float(np.mean(values)) if values else float("nan")

    nmf = [r for r in session_rows if r["method"] == "NMF"]
    hpss = [r for r in session_rows if r["method"] == "HPSS"]
    patient_delta_rmse = []
    patient_delta_acc10 = []
    for rows in by_patient.values():
        n = [r for r in rows if r["method"] == "NMF"]
        h = [r for r in rows if r["method"] == "HPSS"]
        n_rmse = mean("rmse", n)
        h_rmse = mean("rmse", h)
        n_acc10 = mean("acc10", n)
        h_acc10 = mean("acc10", h)
        if np.isfinite(n_rmse) and np.isfinite(h_rmse):
            patient_delta_rmse.append(h_rmse - n_rmse)
        if np.isfinite(n_acc10) and np.isfinite(h_acc10):
            patient_delta_acc10.append(h_acc10 - n_acc10)

    session_delta_rmse = []
    session_delta_acc10 = []
    for rows in by_measurement.values():
        n = next((r for r in rows if r["method"] == "NMF"), None)
        h = next((r for r in rows if r["method"] == "HPSS"), None)
        if n and h and np.isfinite(n["rmse"]) and np.isfinite(h["rmse"]):
            session_delta_rmse.append(h["rmse"] - n["rmse"])
        if n and h and np.isfinite(n["acc10"]) and np.isfinite(h["acc10"]):
            session_delta_acc10.append(h["acc10"] - n["acc10"])

    rmse_ci = bootstrap_ci(session_delta_rmse)
    acc10_ci = bootstrap_ci(session_delta_acc10)
    patient_rmse_ci = bootstrap_ci(patient_delta_rmse)
    patient_acc10_ci = bootstrap_ci(patient_delta_acc10)

    return {
        "variant": variant,
        "selection_risk": info["selection_risk"],
        "protocol_variant": "active_main_v3_locked_cache",
        "n_measurements": len(nmf),
        "n_evaluable_measurements": len(session_delta_rmse),
        "n_patient_groups": len(by_patient),
        "n_valid_windows": sum(r["n_valid"] for r in nmf),
        "nmf_rmse": mean("rmse", nmf),
        "hpss_rmse": mean("rmse", hpss),
        "delta_rmse_hpss_minus_nmf": mean("rmse", hpss) - mean("rmse", nmf),
        "nmf_acc6": mean("acc6", nmf),
        "hpss_acc6": mean("acc6", hpss),
        "delta_acc6_hpss_minus_nmf": mean("acc6", hpss) - mean("acc6", nmf),
        "nmf_acc10": mean("acc10", nmf),
        "hpss_acc10": mean("acc10", hpss),
        "delta_acc10_hpss_minus_nmf": mean("acc10", hpss) - mean("acc10", nmf),
        "session_delta_rmse_ci_low": rmse_ci[0],
        "session_delta_rmse_ci_high": rmse_ci[1],
        "session_delta_acc10_ci_low": acc10_ci[0],
        "session_delta_acc10_ci_high": acc10_ci[1],
        "patient_mean_delta_rmse": float(np.mean(patient_delta_rmse)) if patient_delta_rmse else float("nan"),
        "patient_mean_delta_acc10": float(np.mean(patient_delta_acc10)) if patient_delta_acc10 else float("nan"),
        "patient_delta_rmse_ci_low": patient_rmse_ci[0],
        "patient_delta_rmse_ci_high": patient_rmse_ci[1],
        "patient_delta_acc10_ci_low": patient_acc10_ci[0],
        "patient_delta_acc10_ci_high": patient_acc10_ci[1],
        "uncertainty_unit": "paired_measurement_or_patient_bootstrap",
        "bootstrap_resamples": 2000,
        "mean_t_nmf_ms": mean("t_nmf_ms", nmf),
        "mean_t_hpss_ms": mean("t_hpss_ms", hpss),
    }


def main():
    all_measurements = sorted(set(m for c in cohorts().values() for m in c["measurements"]),
                               key=lambda x: tuple(int(p) for p in x.split(".")))
    loaded = {m: load_session(m) for m in all_measurements}
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    detail_rows = []
    method_rows = []
    for variant, info in cohorts().items():
        rows = []
        for m in info["measurements"]:
            s = loaded[m]
            for method in ("NMF", "HPSS"):
                row = {
                    "variant": variant,
                    "measurement": m,
                    "patient_group": s["patient_group"],
                    "method": method,
                    "n_valid": s["n_valid"],
                    "rmse": s[method]["rmse"],
                    "acc6": s[method]["acc@6"],
                    "acc10": s[method]["acc@10"],
                    "mae": s[method]["mae"],
                    "bias": s[method]["bias"],
                    "t_nmf_ms": s["t_nmf_ms"],
                    "t_hpss_ms": s["t_hpss_ms"],
                }
                rows.append(row)
                detail_rows.append(row)
        summary_rows.append(summarize(variant, info, rows))
        for method in ("NMF", "HPSS"):
            method_session_rows = [r for r in rows if r["method"] == method]
            method_rows.append({
                "variant": variant,
                "selection_risk": info["selection_risk"],
                "protocol_variant": "active_main_v3_locked_cache",
                "method": method,
                "n_measurements": len(method_session_rows),
                "n_evaluable_measurements": sum(np.isfinite(r["rmse"]) for r in method_session_rows),
                "n_valid_windows": sum(r["n_valid"] for r in method_session_rows),
                "rmse": finite_mean("rmse", method_session_rows),
                "acc6": finite_mean("acc6", method_session_rows),
                "acc10": finite_mean("acc10", method_session_rows),
                "mae": finite_mean("mae", method_session_rows),
                "bias": finite_mean("bias", method_session_rows),
                "mean_runtime_ms": finite_mean(
                    "t_nmf_ms" if method == "NMF" else "t_hpss_ms", method_session_rows
                ),
            })

    def write(path, rows):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write(OUT / "reconstruction_multiverse.csv", summary_rows)
    write(OUT / "reconstruction_multiverse_methods.csv", method_rows)
    write(OUT / "reconstruction_session_metrics.csv", detail_rows)
    print(f"wrote {OUT / 'reconstruction_multiverse.csv'} ({len(summary_rows)} variants)")
    print(f"wrote {OUT / 'reconstruction_multiverse_methods.csv'} ({len(method_rows)} variant-method rows)")
    for row in summary_rows:
        print(f"{row['variant']}: ΔRMSE={row['delta_rmse_hpss_minus_nmf']:.3f} "
              f"ΔAcc10={row['delta_acc10_hpss_minus_nmf']:+.2f}pp "
              f"sessions={row['n_measurements']} patients={row['n_patient_groups']} "
              f"risk={row['selection_risk']}")


if __name__ == "__main__":
    main()
