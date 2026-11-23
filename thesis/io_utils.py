"""Read-only I/O, LOPO cohort folds, repeated-measures correlation & clustered bootstrap.

All symbol conventions adhere strictly to `paper_orchestra_bundle/raw_materials/notation_glossary.tex`.
Never mutates files inside `paper_workspace/src/`, `paper_workspace/outputs/cache/`, or `beltrao_data/`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy import stats

from paper_workspace.src import config as C
from paper_workspace.src import evaluation as EV
from paper_workspace.src.cohort import PAPER_18, local_pool, pool30, retained23
from paper_workspace.src.data_loader import MEASURE_TO_RADAR, MEASURE_TO_REF
from paper_workspace.src.lags import metadata_lag
from paper_workspace.src.provenance import PIPELINE_VERSION, expected_provenance

ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = ROOT / "paper_workspace" / "metadata"
AUDIT_DIR = ROOT / "audit"
CACHE_DIR = ROOT / "paper_workspace" / "outputs" / "cache"
BUNDLE_DIR = ROOT / "paper_orchestra_bundle" / "raw_materials"
BUNDLE_FIG_DIR = BUNDLE_DIR / "figures"

RANDOM_SEED = 20261011
ZERO_OVERLAP_IDS = ("1.1", "2.1", "2.2")
ZERO_VALID_IDS = ("1.3", "5.1")


@dataclass(frozen=True)
class LopoFold:
    patient_id: int
    train_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head_info() -> dict[str, str]:
    def _run(args: list[str]) -> str:
        try:
            res = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            return res.stdout.strip()
        except Exception:
            return ""

    head_commit = _run(["rev-parse", "--short", "HEAD"])
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])

    if not head_commit or not branch:
        git_dir = ROOT / ".git"
        head_file = git_dir / "HEAD"
        if head_file.exists():
            raw = head_file.read_text(encoding="utf-8").strip()
            if raw.startswith("ref: "):
                ref_rel = raw[5:].strip()
                if not branch:
                    branch = ref_rel.removeprefix("refs/heads/")
                ref_path = git_dir / ref_rel
                if ref_path.exists():
                    head_commit = ref_path.read_text(encoding="utf-8").strip()[:7]
                else:
                    packed = git_dir / "packed-refs"
                    if packed.exists():
                        for line in packed.read_text(encoding="utf-8").splitlines():
                            if not line or line.startswith("#") or line.startswith("^"):
                                continue
                            parts = line.split()
                            if len(parts) >= 2 and parts[1] == ref_rel:
                                head_commit = parts[0][:7]
                                break
            elif len(raw) >= 7:
                head_commit = raw[:7]
                if not branch:
                    branch = "HEAD"

    return {
        "head_commit": head_commit or "unknown",
        "branch": branch or "unknown",
        "python_version": platform.python_version(),
    }


def load_cohort_ledger() -> pd.DataFrame:
    return pd.read_csv(METADATA_DIR / "cohort_ledger.csv", dtype={"measurement": str})


def patient_id_of(measurement: str) -> int:
    return int(str(measurement).split(".")[0])


def get_cohort_sessions(cohort: str, include_zero_valid: bool = False) -> list[str]:
    """Return ordered session IDs for named cohort tier."""
    if cohort == "pool30_sync":
        ids = pool30()
    elif cohort == "all_paired":
        ids = local_pool()
    elif cohort == "retained23":
        ids = retained23()
    elif cohort == "PAPER_18":
        ids = list(PAPER_18)
    else:
        raise ValueError(f"Unknown cohort name: {cohort!r}")

    if not include_zero_valid:
        ids = [m for m in ids if m not in ZERO_VALID_IDS]
    return sorted(ids, key=lambda x: tuple(int(p) for p in x.split(".")))


def iter_lopo_folds(cohort: str = "pool30_sync") -> Iterator[LopoFold]:
    """Yield strictly patient-disjoint Leave-One-Patient-Out folds."""
    sessions = get_cohort_sessions(cohort, include_zero_valid=False)
    patients = sorted({patient_id_of(m) for m in sessions})
    for pat in patients:
        test = tuple(m for m in sessions if patient_id_of(m) == pat)
        train = tuple(m for m in sessions if patient_id_of(m) != pat)
        yield LopoFold(patient_id=pat, train_sessions=train, test_sessions=test)


def verify_all_baseline_caches(measurements: list[str] | None = None) -> dict[str, bool]:
    """Assert that every locked v1 `.npz` cache matches frozen SHA-256 source/input hashes.

    Verified under the Dual Python Contract: Python 3.10.19 consumers validate exact
    `source_sha256` (15 `_CORE_FILES`), `input_sha256`, `pipeline_version`, and arrays
    while allowing the stored baseline `python` (`3.9.6`) and package tags to differ
    from the analysis environment (`3.10.19`).
    """
    ms = measurements or local_pool()
    required = {
        "est", "est_dft", "est_nls", "est_hpss", "ref", "valid",
        "minmove", "nmf_converged", "lag", "corr", "t_nmf", "t_hpss",
        "provenance_json",
    }
    status = {}
    for m in ms:
        p = CACHE_DIR / f"{m}.npz"
        ok = False
        if p.exists():
            exp = expected_provenance(m)
            with np.load(p, allow_pickle=False) as z:
                if required.issubset(set(z.files)):
                    stored = json.loads(str(z["provenance_json"].item()))
                    ok = (
                        stored.get("pipeline_version") == PIPELINE_VERSION
                        and stored.get("measurement") == m
                        and str(stored.get("python", "")).startswith("3.9")
                        and stored.get("source_sha256") == exp["source_sha256"]
                        and stored.get("input_sha256") == exp["input_sha256"]
                    )
        status[m] = ok
        if not ok:
            raise RuntimeError(f"Frozen baseline cache missing or stale for {m}: {p}")
    return status


def load_locked_cache(measurement: str) -> dict[str, np.ndarray]:
    """Load read-only arrays from locked `paper_workspace/outputs/cache/<m>.npz`."""
    p = CACHE_DIR / f"{measurement}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing cache: {p}")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def load_radar_full(measurement: str) -> pd.DataFrame:
    """Load full 42-column radar CSV telemetry without modifying `beltrao_data/`."""
    if measurement not in MEASURE_TO_RADAR:
        raise ValueError(f"No radar file for measurement {measurement!r}")
    csv_path = C.RADAR_DIR / f"{MEASURE_TO_RADAR[measurement]}.csv"
    return pd.read_csv(csv_path, skiprows=2)


def load_locked_neutral_lags() -> dict[str, dict[str, float | int | None]]:
    """Load or verify pre-registered branch-neutral and pure-metadata session lags."""
    out: dict[str, dict[str, float | int | None]] = {}
    ws3_path = AUDIT_DIR / "phase2_ws3_session_lags.csv"
    ws3_rows: dict[str, dict[str, str]] = {}
    if ws3_path.exists():
        with ws3_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                ws3_rows[row["measurement"]] = row

    for m in get_cohort_sessions("all_paired", include_zero_valid=True):
        if m in ws3_rows:
            r = ws3_rows[m]
            meta_s = float(r["meta_clock_lag_s"])
            out[m] = {
                "meta_clock_lag_s": meta_s,
                "pure_metadata_lag_s": int(-round(meta_s)),
                "neutral_raw_nls_lag_s": int(float(r["lag_raw_nls_s"])),
                "cached_nmf_lag_s": int(float(r["lag_nmf_s"])),
                "hpss_own_lag_s": int(float(r["lag_hpss_s"])),
            }
        else:
            mlag = metadata_lag(m).get("lag_s")
            out[m] = {
                "meta_clock_lag_s": mlag,
                "pure_metadata_lag_s": int(-round(mlag)) if mlag is not None else 0,
                "neutral_raw_nls_lag_s": int(load_locked_cache(m)["lag"]),
                "cached_nmf_lag_s": int(load_locked_cache(m)["lag"]),
                "hpss_own_lag_s": int(load_locked_cache(m)["lag"]),
            }
    return out


def rmcorr_ancova(est_list: list[np.ndarray], ref_list: list[np.ndarray]) -> dict[str, float]:
    """Bland & Altman (1995) ANCOVA pooled within-session correlation (`\\rWithin`).

    Strictly distinguishes `\\rWithin` (weighted by within-session sums of squares)
    from `\\rWithinSessMean` (unweighted session Pearson average).
    """
    ss_x = 0.0
    ss_y = 0.0
    sp_xy = 0.0
    n_total = 0
    k_used = 0
    for e, r in zip(est_list, ref_list):
        ea = np.asarray(e, dtype=float)
        ra = np.asarray(r, dtype=float)
        ok = np.isfinite(ea) & np.isfinite(ra)
        ea, ra = ea[ok], ra[ok]
        if len(ea) < 2:
            continue
        dx = ea - ea.mean()
        dy = ra - ra.mean()
        ss_x += float(np.dot(dx, dx))
        ss_y += float(np.dot(dy, dy))
        sp_xy += float(np.dot(dx, dy))
        n_total += len(ea)
        k_used += 1

    if ss_x <= 0.0 or ss_y <= 0.0 or n_total <= k_used + 1:
        return {"r_within": float("nan"), "df": 0, "p_value": float("nan")}

    r_val = float(np.clip(sp_xy / np.sqrt(ss_x * ss_y), -1.0, 1.0))
    df = int(n_total - k_used - 1)
    if abs(r_val) >= 1.0 - 1e-15:
        p_val = 0.0
    else:
        t_stat = r_val * np.sqrt(df / max(1e-15, 1.0 - r_val * r_val))
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df))
    return {"r_within": r_val, "df": df, "p_value": p_val}


def clustered_bland_altman(est_list: list[np.ndarray], ref_list: list[np.ndarray]) -> dict[str, float]:
    """One-way random-effects ANOVA Bland-Altman (Bland & Altman 2007) for repeated obs."""
    diffs = []
    for e, r in zip(est_list, ref_list):
        d = np.asarray(e, float) - np.asarray(r, float)
        d = d[np.isfinite(d)]
        if len(d) > 0:
            diffs.append(d)
    if not diffs:
        return {"bias": float("nan"), "sd_total": float("nan"), "loa_lo": float("nan"), "loa_up": float("nan")}

    k = len(diffs)
    ns = np.array([len(d) for d in diffs], dtype=float)
    n_tot = float(ns.sum())
    all_d = np.concatenate(diffs)
    grand_mean = float(all_d.mean())

    ss_between = float(sum(len(d) * (d.mean() - grand_mean) ** 2 for d in diffs))
    ss_within = float(sum(((d - d.mean()) ** 2).sum() for d in diffs))
    df_within = max(1.0, n_tot - k)
    ms_within = ss_within / df_within
    if k > 1 and n_tot > 0:
        ms_between = ss_between / (k - 1)
        n0 = (n_tot - float((ns ** 2).sum()) / n_tot) / (k - 1)
        var_between = max(0.0, (ms_between - ms_within) / max(1e-12, n0))
    else:
        var_between = 0.0

    sd_total = float(np.sqrt(ms_within + var_between))
    return {
        "bias": grand_mean,
        "sd_within": float(np.sqrt(ms_within)),
        "sd_between": float(np.sqrt(var_between)),
        "sd_total": sd_total,
        "loa_lo": grand_mean - 1.96 * sd_total,
        "loa_up": grand_mean + 1.96 * sd_total,
    }


def patient_clustered_bootstrap(
    patient_values: dict[int, np.ndarray],
    n_boot: int = 10000,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Cluster-bootstrap mean & 95% CI resampling patients with replacement."""
    pats = sorted(patient_values.keys())
    if not pats:
        return {"mean": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    pat_means = np.array([float(np.mean(patient_values[p])) for p in pats], dtype=float)
    # Equal-session weighted estimate inside sampled patient cluster
    rng = np.random.default_rng(seed)
    n_p = len(pats)
    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_p, size=n_p)
        sampled = np.concatenate([patient_values[pats[i]] for i in idx])
        boot_means[b] = float(sampled.mean())
    obs = float(np.concatenate([patient_values[p] for p in pats]).mean())
    return {
        "mean": obs,
        "patient_mean_of_means": float(pat_means.mean()),
        "ci95_lo": float(np.percentile(boot_means, 2.5)),
        "ci95_hi": float(np.percentile(boot_means, 97.5)),
    }


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni step-down adjusted p-values in original order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = float(p_values[idx]) * (m - rank)
        running_max = max(running_max, min(1.0, val))
        adj[idx] = running_max
    return [float(x) for x in adj]


def write_json_locked(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path
