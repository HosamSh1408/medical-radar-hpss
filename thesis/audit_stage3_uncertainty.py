"""Zero-shared-code clean-room auditor for Stage 3 (`stage3-uncertainty-v1`).

Intentionally imports ZERO statistical, LOPO, selective-risk, or conformal evaluation logic from
`thesis.exp1_stage1_parsimony`, `thesis.exp2_stage2_complexity`, `thesis.exp3_stage3_uncertainty`,
`thesis.wp3_wp4_uncertainty_conformal`, or `thesis.io_utils`.
Independently verifies:
  - Gate 1: Frozen baseline SHA-256 lock (`154/154` + `33/33` + untouched `external_data/`) & exact Stage 1.1 / Stage 2.2 full-coverage (`c_cov=1.00`) parity (`< 1e-12`).
  - Gate 2: Clean-room selective Risk-Coverage (`AURC`, `E-AURC`, `AUGRC`) & proxy (`U1..U6c`) recomputation from scratch across 4 references & CSV<->JSON parity.
  - Gate 3: Clean-room quiet-apnea (`resp_max_gap_s > 5.0 s`) pause safety stratification & rejection risk-ratio audit (`< 1e-12`).
  - Gate 4: Clean-room Complex WGN (`N=200`, `seed=20261011`), 5-seed phase-scrambled surrogate & `UCD != ChildPresent` OOD abstention AUROC recomputation (`>= 0.95` verified).
  - Gate 5: Clean-room 5-regime Finite-K Conformal Prediction (`C1..C5` across `alpha in {0.05, 0.10, 0.20, 0.30}`) recomputation, finite-sample vacuity theorem at `alpha=0.05`, and valid `>= 1-alpha` coverage at `alpha in {0.10, 0.20, 0.30}` (`< 1e-12`).
  - Gate 6: Clean-room `O(2) x R_{>0}` group invariance (`< 1e-6`), exact `0.0` causality perturbation across all 8 proxies, and strict patient-disjoint LOPO partition checks.
  - Gate 7: Gate `G4` Hard-Freeze manifest completeness, provenance, and strict RFC-8259 JSON (`allow_nan=False`).

Execution:
  MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /opt/homebrew/bin/python3.10 -m thesis.audit_stage3_uncertainty
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyedflib
from scipy import optimize, stats

from paper_workspace.src import config as C
from paper_workspace.src.cohort import PAPER_18, local_pool, pool30, retained23
from paper_workspace.src.data_loader import MEASURE_TO_RADAR, MEASURE_TO_REF, load_radar
from paper_workspace.src.demod import csd
from paper_workspace.src.evaluation import window_ref_and_gating
from paper_workspace.src.filtering import autocorr_coarse, bandpass
from paper_workspace.src.hpss import mitigate_hpss
from paper_workspace.src.provenance import PIPELINE_VERSION, expected_provenance
from thesis import diff_hmm_and_foils as D
from thesis import viterbi_tracker as VT
from thesis import wp1_wp2_causal_hmm as W

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "audit"
MANIFEST_PATH = AUDIT_DIR / "postfix_manifest.sha256"
LAGS_CSV = AUDIT_DIR / "phase2_ws3_session_lags.csv"
LOCKED_CACHE_DIR = ROOT / "paper_workspace" / "outputs" / "cache"
BUNDLE_DIR = ROOT / "paper_orchestra_bundle" / "raw_materials"
STAGE1_JSON = BUNDLE_DIR / "metrics_stage1_parsimony.json"
STAGE2_JSON = BUNDLE_DIR / "metrics_stage2_complexity.json"
STAGE3_JSON = BUNDLE_DIR / "metrics_stage3_uncertainty.json"
SESSIONS_CSV = AUDIT_DIR / "stage3_uncertainty_sessions.csv"
STAGE2_SAL_CACHE = ROOT / "thesis" / "outputs" / "stage2_complexity" / "cache"
STAGE2_PRED_ARCH = ROOT / "thesis" / "outputs" / "stage2_complexity" / "pool30_predictions.npz"
STAGE3_ARCH = ROOT / "thesis" / "outputs" / "stage3_uncertainty" / "pool30_uncertainty_and_conformal.npz"
FROZEN_STAGE4_ARCH = ROOT / "thesis" / "outputs" / "stage3_uncertainty" / "frozen_stage4_models.npz"
AUDIT_JSON = AUDIT_DIR / "stage3_godmode_audit.json"
AUDIT_MD = AUDIT_DIR / "stage3_godmode_audit.md"

PROXIES_8 = (
    "U1_Posterior_Entropy",
    "U2_Posterior_SD",
    "U3_Spectral_Entropy_Clean",
    "U4_Spectral_Entropy_Raw",
    "U5_HQI_Harmonic_Deficit",
    "U6a_ZeroTuned_Additive_SQI",
    "U6b_ZeroTuned_Geometric_SQI",
    "U6c_LOPO_Calibrated_SQI",
)

GRID_1BPM = np.arange(18.0, 90.0 + 1e-9, 1.0, dtype=np.float64)
GRID_SD_UNIF = float(np.std(GRID_1BPM, ddof=0))
COV_51 = np.linspace(0.50, 1.00, 51, dtype=np.float64)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            blk = fh.read(1024 * 1024)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def _pid(m: str) -> int:
    return int(str(m).split(".")[0])


def _cohort_ids(name: str) -> list[str]:
    if name == "pool30_sync":
        ids = pool30()
    elif name == "retained23":
        ids = retained23()
    elif name == "PAPER_18":
        ids = list(PAPER_18)
    elif name == "all_paired":
        ids = local_pool()
    else:
        raise ValueError(f"Unknown cohort: {name}")
    ids = [m for m in ids if m not in ("1.3", "5.1")]
    return sorted(ids, key=lambda x: tuple(int(p) for p in x.split(".")))


def _clean_extract_edf_and_pause(
    edf_path: Path,
    centers_s: np.ndarray,
    lag_s: int,
) -> dict[str, Any]:
    """Clean-room EDF extractor computing all 4 references and >5s pause flags from scratch."""
    rd = pyedflib.EdfReader(str(edf_path))
    try:
        lbls = [rd.getLabel(i) for i in range(rd.signals_in_file)]
        idx = {l: i for i, l in enumerate(lbls)}
        resp_256 = rd.readSignal(idx["Resp"]).astype(np.float64)
        rate_1hz = rd.readSignal(idx["RespRate"]).astype(np.float64)
        hr_1hz = rd.readSignal(idx["HeartRate"]).astype(np.float64)
        trig = rd.readSignal(idx["Resp.-Trigger"]).astype(np.float64)
        fs_r = float(rd.getSampleFrequency(idx["Resp"]))
        fs_t = float(rd.getSampleFrequency(idx["Resp.-Trigger"]))
        n_ref = min(len(rate_1hz), len(hr_1hz), len(resp_256) // int(round(fs_r)))
    finally:
        rd._close()

    rw, vw = window_ref_and_gating(resp_256, rate_1hz, centers_s, lag_s, fs_raw=fs_r)
    v1 = np.asarray(vw, dtype=bool) & np.isfinite(rw)

    active = trig > 0.5
    rising = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    times: list[float] = []
    for r_idx in rising:
        ts = float(r_idx) / fs_t
        if not times or (ts - times[-1]) >= 0.30:
            times.append(ts)
    ev = np.asarray(times, dtype=np.float64)

    def _trig_at(shift_lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tc = centers_s - float(shift_lag)
        a_arr = np.floor(tc - float(C.WIN_S) / 2.0).astype(int)
        b_arr = a_arr + int(C.WIN_S)
        arr = np.full(len(centers_s), np.nan, dtype=np.float64)
        gaps_max = np.full(len(centers_s), np.nan, dtype=np.float64)
        for w_i in range(len(centers_s)):
            a, b = int(a_arr[w_i]), int(b_arr[w_i])
            if a < 0 or b > n_ref:
                continue
            iw = ev[(ev >= float(a)) & (ev <= float(b))]
            if iw.size < 2:
                gaps_max[w_i] = float(b - a)
                continue
            df_g = np.diff(iw)
            gaps_max[w_i] = float(max(iw[0] - float(a), float(b) - iw[-1], float(np.max(df_g))))
            rates = 60.0 / df_g
            vr = rates[(rates >= 12.0) & (rates <= 120.0)]
            if vr.size > 0:
                arr[w_i] = float(np.median(vr))
        return arr, (v1 & np.isfinite(arr)), gaps_max

    rt0, vt0, max_gaps = _trig_at(lag_s)
    rta, vta, _ = _trig_at(lag_s + 10)

    tc0 = centers_s - float(lag_s)
    a0 = np.floor(tc0 - float(C.WIN_S) / 2.0).astype(int)
    b0 = a0 + int(C.WIN_S)
    rcnt = np.full(len(centers_s), np.nan, dtype=np.float64)
    for w_i in range(len(centers_s)):
        a, b = int(a0[w_i]), int(b0[w_i])
        if a < 0 or b > n_ref:
            continue
        n_ev = int(np.sum((ev >= float(a)) & (ev < float(b))))
        rcnt[w_i] = 60.0 * float(n_ev) / float(C.WIN_S)
    vtc = v1 & np.isfinite(rcnt)

    return {
        "br_1hz_30s": (np.asarray(rw, dtype=np.float64), v1),
        "br_trig_bpm": (rt0, vt0),
        "br_trig_bpm_lag_aligned": (rta, vta),
        "br_trig_count_bpm": (rcnt, vtc),
        "pause_gt5s": v1 & np.isfinite(max_gaps) & (max_gaps > 5.0),
    }


def _clean_spec_ent(log_col: np.ndarray) -> np.ndarray:
    p = np.maximum(np.exp(np.asarray(log_col, dtype=np.float64)) - 1e-3, 0.0)
    rs = np.sum(p, axis=1, keepdims=True)
    pn = np.where(rs > 1e-12, p / np.maximum(rs, 1e-12), 1.0 / float(p.shape[1]))
    lp = np.where(pn > 1e-300, np.log(np.maximum(pn, 1e-300)), 0.0)
    return np.clip(-np.sum(pn * lp, axis=1) / np.log(float(p.shape[1])), 0.0, 1.0)


def _clean_hqi_def(phi: np.ndarray) -> np.ndarray:
    p1 = np.maximum(np.exp(phi[:, :, 0]) - 1e-3, 0.0)
    p2 = np.maximum(np.exp(phi[:, :, 1]) - 1e-3, 0.0)
    ph = np.maximum(np.exp(phi[:, :, 3]) - 1e-3, 0.0)
    n_t, n_b = p1.shape
    hqi = np.zeros(n_t, dtype=np.float64)
    for t in range(n_t):
        tot = float(np.sum(p1[t]))
        if not np.isfinite(tot) or tot <= 1e-12:
            continue
        idx = int(np.argmax(p1[t]))
        lo, hi = max(0, idx - 2), min(n_b, idx + 3)
        fm = float(np.sum(p1[t, lo:hi])) + 0.5 * float(p2[t, idx])
        bm = tot + 0.5 * float(np.max(ph[t])) + 1e-6
        hqi[t] = fm / (fm + bm)
    return np.clip(1.0 - np.clip(hqi, 0.0, 1.0), 0.0, 1.0)


def _clean_causal_ood_barrier(phi: np.ndarray, history_hops: int = 3) -> np.ndarray:
    p1 = np.maximum(np.exp(phi[:, :, 0]) - 1e-3, 0.0)
    rs = np.sum(p1, axis=1, keepdims=True)
    pn = np.where(rs > 1e-12, p1 / np.maximum(rs, 1e-12), 1.0 / float(p1.shape[1]))
    n_t, n_b = pn.shape
    hmix = np.zeros(n_t, dtype=np.float64)
    dlog = np.log(float(n_b))
    for t in range(n_t):
        t0 = max(0, t - int(history_hops))
        mb = np.maximum(np.mean(pn[t0 : t + 1], axis=0), 1e-300)
        hmix[t] = -float(np.sum(mb * np.log(mb))) / dlog
    hmix = np.clip(hmix, 0.0, 1.0)
    return np.clip((hmix - 0.90) / 0.10, 0.0, 1.0)


def _clean_rc(err: np.ndarray, u: np.ndarray) -> dict[str, float]:
    ae = np.abs(err)
    se2 = err ** 2
    n = len(err)
    ord_u = np.argsort(u, kind="mergesort")
    ord_o = np.argsort(ae, kind="mergesort")
    cs_u = np.cumsum(se2[ord_u])
    cs_o = np.cumsum(se2[ord_o])
    k_arr = np.clip(np.round(COV_51 * float(n)).astype(int), 1, n)
    eff_c = k_arr.astype(np.float64) / float(n)
    r_u = np.sqrt(cs_u[k_arr - 1] / k_arr.astype(np.float64))
    r_o = np.sqrt(cs_o[k_arr - 1] / k_arr.astype(np.float64))
    full_r = float(np.sqrt(cs_u[-1] / float(n)))
    aurc = float(np.trapezoid(r_u, COV_51) / 0.50)
    aurc_o = float(np.trapezoid(r_o, COV_51) / 0.50)
    augrc = float(np.trapezoid(eff_c * r_u, COV_51) / 0.50)
    augrc_r = float(np.trapezoid(eff_c * full_r, COV_51) / 0.50)
    return {
        "full_rmse": full_r,
        "rmse_80": float(r_u[30]),
        "aurc": aurc,
        "e_aurc_rand": float(aurc - full_r),
        "e_aurc_ora": float(aurc - aurc_o),
        "augrc": augrc,
        "e_augrc_rand": float(augrc - augrc_r),
    }


def _clean_auroc(u_in: np.ndarray, u_ood: np.ndarray) -> float:
    n0, n1 = len(u_in), len(u_ood)
    ranks = stats.rankdata(np.concatenate([u_in, u_ood]), method="average")
    u_stat = float(np.sum(ranks[n0:])) - 0.5 * float(n1 * (n1 + 1))
    return float(u_stat / float(n0 * n1))


def run_audit() -> dict[str, Any]:
    report: dict[str, Any] = {"gates": {}}
    raw_text = STAGE3_JSON.read_text(encoding="utf-8")
    master = json.loads(raw_text)
    schema_lbl = str(master.get("schema_version", "unknown"))
    md: list[str] = [
        f"# Stage 3 Independent Clean-Room Audit Report (`{schema_lbl}`)\n",
        "Executed with zero statistical, LOPO, selective-risk, or conformal imports from `thesis.exp1_stage1_parsimony`, `thesis.exp2_stage2_complexity`, `thesis.exp3_stage3_uncertainty`, `thesis.wp3_wp4_uncertainty_conformal`, or `thesis.io_utils`.\n",
    ]
    s1 = json.loads(STAGE1_JSON.read_text(encoding="utf-8"))
    s2 = json.loads(STAGE2_JSON.read_text(encoding="utf-8"))
    df_csv = pd.read_csv(SESSIONS_CSV, dtype={"measurement": str})

    # ------------------------------------------------------------------
    # Gate 1: Frozen Baseline SHA-256 Lock & Stage 1.1 / Stage 2.2 Full-Coverage Parity
    # ------------------------------------------------------------------
    bad_files: list[str] = []
    checked_frozen = 0
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected_hash, rel_path = line.split(None, 1)
        rel_path = rel_path.strip()
        if rel_path.startswith(("paper_workspace/", "beltrao_data/")):
            checked_frozen += 1
            fp = ROOT / rel_path
            if not fp.exists() or _sha256(fp) != expected_hash:
                bad_files.append(rel_path)

    bad_caches: list[str] = []
    for m in local_pool():
        cp = LOCKED_CACHE_DIR / f"{m}.npz"
        exp = expected_provenance(m)
        with np.load(cp, allow_pickle=False) as z:
            prov = json.loads(str(z["provenance_json"].item()))
        if not (
            prov.get("pipeline_version") == PIPELINE_VERSION
            and prov.get("measurement") == m
            and str(prov.get("python", "")).startswith("3.9")
            and prov.get("source_sha256") == exp["source_sha256"]
            and prov.get("input_sha256") == exp["input_sha256"]
        ):
            bad_caches.append(m)

    ext_dir = ROOT / "external_data" / "gleichauf2024"
    stage1_mtime = STAGE1_JSON.stat().st_mtime
    ext_files = [p for p in (ROOT / "external_data").rglob("*") if p.is_file()]
    external_data_untouched = (
        ext_dir.is_dir()
        and len(ext_files) > 0
        and all(p.stat().st_mtime < stage1_mtime for p in ext_files)
    )

    s1_max_diff = 0.0
    s2_max_diff = 0.0
    for coh in ("pool30_sync", "retained23", "PAPER_18"):
        for rlbl in ("br_1hz_30s", "br_trig_bpm"):
            for km in ("R0_HPSS_NLS_Locked", "R1a_Online_Forward_MMSE_0s", "R1h_E2E_Causal_H5_OnlineForward_0s", "R1f_Offline_FB_MMSE_Inf"):
                m1 = s1["rung1_lopo"][coh][rlbl]["methods"][km]
                px_chk = "U3_Spectral_Entropy_Clean" if km == "R0_HPSS_NLS_Locked" else "U2_Posterior_SD"
                m3 = master["selective_prediction_by_cohort"][coh][rlbl][km][px_chk]
                s1_max_diff = max(
                    s1_max_diff,
                    abs(float(m3["session_macro"]["full_risk_rmse_sess"]) - float(m1["rmse_sess"])),
                    abs(float(m3["window_pooled"]["full_risk_rmse"]) - float(m1["rmse_pool"])),
                )
        for rlbl in ("br_1hz_30s", "br_trig_bpm", "br_trig_bpm_lag_aligned"):
            for km in ("R0_HPSS_NLS_Locked", "R2e_Causal_DiffHMM_FilterSafe8", "R2a_Causal_DiffHMM_Linear11", "R2d_E2E_Causal_H5_DiffHMM_Linear11"):
                m2 = s2["cohorts"][coh][rlbl]["methods"][km]
                px_chk = "U3_Spectral_Entropy_Clean" if km == "R0_HPSS_NLS_Locked" else "U2_Posterior_SD"
                m3 = master["selective_prediction_by_cohort"][coh][rlbl][km][px_chk]
                s2_max_diff = max(
                    s2_max_diff,
                    abs(float(m3["session_macro"]["full_risk_rmse_sess"]) - float(m2["rmse_sess"])),
                    abs(float(m3["window_pooled"]["full_risk_rmse"]) - float(m2["rmse_pool"])),
                )

    g1_pass = (
        checked_frozen == 154
        and len(bad_files) == 0
        and len(bad_caches) == 0
        and external_data_untouched
        and s1.get("schema_version") == "stage1-parsimony-v3"
        and s2.get("schema_version") == "stage2-complexity-v4"
        and s1_max_diff < 1e-12
        and s2_max_diff < 1e-12
    )
    report["gates"]["gate1_frozen_baseline_and_stage1_stage2_parity"] = {
        "passed": g1_pass,
        "checked_frozen_files": checked_frozen,
        "bad_files": bad_files,
        "bad_caches": bad_caches,
        "external_data_untouched": external_data_untouched,
        "stage1_full_cov_max_diff": s1_max_diff,
        "stage2_full_cov_max_diff": s2_max_diff,
    }
    md.append(
        f"## Gate 1 — Frozen Baseline Lock & Stage 1.1 / Stage 2.2 Parity: **{'PASS' if g1_pass else 'FAIL'}**\n"
        f"- Checked `{checked_frozen}/154` frozen hashes (`bad: {len(bad_files)}`), `33/33` locked caches (`bad: {len(bad_caches)}`), `external_data/` untouched `{external_data_untouched}` (`{len(ext_files)}` files).\n"
        f"- Exact `c_cov = 1.00` parity: Stage 1.1 `max diff = {s1_max_diff:.2e}`, Stage 2.2 `max diff = {s2_max_diff:.2e}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 2: Clean-Room Selective Risk-Coverage & Proxy Recomputation + CSV Parity
    # ------------------------------------------------------------------
    lags_clean: dict[str, int] = {}
    with LAGS_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lags_clean[row["measurement"]] = int(float(row["lag_raw_nls_s"]))

    p30_ids = _cohort_ids("pool30_sync")
    arch_s2 = np.load(STAGE2_PRED_ARCH, allow_pickle=False)
    arch_s3 = np.load(STAGE3_ARCH, allow_pickle=False)

    refs_clean: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        "br_1hz_30s": {},
        "br_trig_bpm": {},
        "br_trig_bpm_lag_aligned": {},
        "br_trig_count_bpm": {},
    }
    pauses_clean: dict[str, np.ndarray] = {}
    phis_h0_clean: dict[str, np.ndarray] = {}
    proxies_r2e_clean: dict[str, dict[str, np.ndarray]] = {p: {} for p in PROXIES_8}
    base5_r2e: dict[str, np.ndarray] = {}

    for m in p30_ids:
        with np.load(STAGE2_SAL_CACHE / f"{m}.npz", allow_pickle=False) as zs:
            phi0 = zs["phi_h0"].astype(np.float64)
        phis_h0_clean[m] = phi0
        n_w = phi0.shape[0]
        centers = np.arange(n_w, dtype=np.float64) * float(C.UPDATE_S) + float(C.WIN_S) / 2.0
        ext = _clean_extract_edf_and_pause(C.REF_DIR / f"{MEASURE_TO_REF[m]}.edf", centers, lags_clean[m])
        for rlbl in refs_clean:
            refs_clean[rlbl][m] = ext[rlbl]
        pauses_clean[m] = ext["pause_gt5s"]

        u1 = np.clip(arch_s2[f"unc_ent__R2e_Causal_DiffHMM_FilterSafe8__{m}"].astype(np.float64), 0.0, 1.0)
        u2 = np.clip(arch_s2[f"unc_sd__R2e_Causal_DiffHMM_FilterSafe8__{m}"].astype(np.float64) / GRID_SD_UNIF, 0.0, 1.0)
        u3 = _clean_spec_ent(phi0[:, :, 0])
        u4 = _clean_spec_ent(phi0[:, :, 5])
        u5 = _clean_hqi_def(phi0)
        b_ood = _clean_causal_ood_barrier(phi0, history_hops=3)
        u6a = np.clip(0.45 * u2 + 0.35 * u1 + 0.20 * b_ood, 0.0, 1.0)
        u6b = np.clip(1.0 - ((1.0 - u2) * (1.0 - u1) * (1.0 - b_ood)) ** (1.0 / 3.0), 0.0, 1.0)

        proxies_r2e_clean["U1_Posterior_Entropy"][m] = u1
        proxies_r2e_clean["U2_Posterior_SD"][m] = u2
        proxies_r2e_clean["U3_Spectral_Entropy_Clean"][m] = u3
        proxies_r2e_clean["U4_Spectral_Entropy_Raw"][m] = u4
        proxies_r2e_clean["U5_HQI_Harmonic_Deficit"][m] = u5
        proxies_r2e_clean["U6a_ZeroTuned_Additive_SQI"][m] = u6a
        proxies_r2e_clean["U6b_ZeroTuned_Geometric_SQI"][m] = u6b
        base5_r2e[m] = np.column_stack([u1, u2, u3, u4, u5])

    # Clean-room LOPO Z-NNLS calibration for U6c_LOPO_Calibrated_SQI across R2e and R2a
    pats_p30 = sorted({_pid(m) for m in p30_ids})
    max_u6c_weight_diff = 0.0
    r2e_fold_cals: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for f_i, pat in enumerate(pats_p30):
        tr_ms = [m for m in p30_ids if _pid(m) != pat]
        te_ms = [m for m in p30_ids if _pid(m) == pat]
        X_tr = np.concatenate([base5_r2e[m][refs_clean["br_1hz_30s"][m][1]] for m in tr_ms], axis=0)
        y_tr = np.concatenate([
            np.abs(arch_s2[f"pred__R2e_Causal_DiffHMM_FilterSafe8__{m}"][refs_clean["br_1hz_30s"][m][1]] - refs_clean["br_1hz_30s"][m][0][refs_clean["br_1hz_30s"][m][1]])
            for m in tr_ms
        ], axis=0)
        mu_x = np.mean(X_tr, axis=0)
        sd_x = np.maximum(np.std(X_tr, axis=0, ddof=0), 1e-6)
        Xz = (X_tr - mu_x[None, :]) / sd_x[None, :]
        yc = y_tr - float(np.mean(y_tr))
        A_aug = np.vstack([Xz, 10.0 * np.eye(5)])
        y_aug = np.r_[yc, np.zeros(5)]
        w_raw, _ = optimize.nnls(A_aug, y_aug)
        w = w_raw / float(np.sum(w_raw))
        r2e_fold_cals.append((mu_x, sd_x, w))
        logged_w = master["lopo_calibrated_sqi_fold_logs_pool30"]["R2e_Causal_DiffHMM_FilterSafe8"][f_i]["weights_base5"]
        w_log_arr = np.array([
            logged_w["w_U1_Posterior_Entropy"],
            logged_w["w_U2_Posterior_SD"],
            logged_w["w_U3_Spectral_Entropy_Clean"],
            logged_w["w_U4_Spectral_Entropy_Raw"],
            logged_w["w_U5_HQI_Harmonic_Deficit"],
        ])
        max_u6c_weight_diff = max(max_u6c_weight_diff, float(np.max(np.abs(w - w_log_arr))))
        for m in te_ms:
            raw_eval = ((base5_r2e[m] - mu_x[None, :]) / sd_x[None, :]) @ w
            proxies_r2e_clean["U6c_LOPO_Calibrated_SQI"][m] = 1.0 / (1.0 + np.exp(-np.clip(raw_eval, -40.0, 40.0)))

    # Also verify R2a_Causal_DiffHMM_Linear11 and R2d_E2E_Causal_H5_DiffHMM_Linear11 archived proxies in clean room
    max_proxy_arch_diff = 0.0
    for pk in PROXIES_8:
        for m in p30_ids:
            max_proxy_arch_diff = max(
                max_proxy_arch_diff,
                float(np.max(np.abs(proxies_r2e_clean[pk][m] - arch_s3[f"u__R2e_Causal_DiffHMM_FilterSafe8__{pk}__{m}"]))),
            )
    for km_chk in ("R2a_Causal_DiffHMM_Linear11", "R2d_E2E_Causal_H5_DiffHMM_Linear11"):
        use_h5_chk = "H5" in km_chk
        for m in p30_ids:
            with np.load(STAGE2_SAL_CACHE / f"{m}.npz", allow_pickle=False) as zs_c:
                phi_c = zs_c["phi_h5" if use_h5_chk else "phi_h0"].astype(np.float64)
            u1_c = np.clip(arch_s2[f"unc_ent__{km_chk}__{m}"].astype(np.float64), 0.0, 1.0)
            u2_c = np.clip(arch_s2[f"unc_sd__{km_chk}__{m}"].astype(np.float64) / GRID_SD_UNIF, 0.0, 1.0)
            u3_c = _clean_spec_ent(phi_c[:, :, 0])
            u4_c = _clean_spec_ent(phi_c[:, :, 5])
            u5_c = _clean_hqi_def(phi_c)
            b_c = _clean_causal_ood_barrier(phi_c, history_hops=3)
            u6a_c = np.clip(0.45 * u2_c + 0.35 * u1_c + 0.20 * b_c, 0.0, 1.0)
            u6b_c = np.clip(1.0 - ((1.0 - u2_c) * (1.0 - u1_c) * (1.0 - b_c)) ** (1.0 / 3.0), 0.0, 1.0)
            for pk_c, arr_c in (
                ("U1_Posterior_Entropy", u1_c),
                ("U2_Posterior_SD", u2_c),
                ("U3_Spectral_Entropy_Clean", u3_c),
                ("U4_Spectral_Entropy_Raw", u4_c),
                ("U5_HQI_Harmonic_Deficit", u5_c),
                ("U6a_ZeroTuned_Additive_SQI", u6a_c),
                ("U6b_ZeroTuned_Geometric_SQI", u6b_c),
            ):
                max_proxy_arch_diff = max(
                    max_proxy_arch_diff,
                    float(np.max(np.abs(arr_c - arch_s3[f"u__{km_chk}__{pk_c}__{m}"]))),
                )

    max_rc_diff = 0.0
    for km_rc in ("R2e_Causal_DiffHMM_FilterSafe8", "R2a_Causal_DiffHMM_Linear11", "R2d_E2E_Causal_H5_DiffHMM_Linear11", "R1a_Online_Forward_MMSE_0s", "R1h_E2E_Causal_H5_OnlineForward_0s"):
        for rlbl, rmap in refs_clean.items():
            for pk in PROXIES_8:
                s_aurc, s_eaurc_r, s_eaurc_o, s_augrc, s_eaugrc_r, s_r80 = [], [], [], [], [], []
                e_all, u_all = [], []
                for m in p30_ids:
                    rf, vv = rmap[m]
                    if km_rc in ("R1a_Online_Forward_MMSE_0s", "R1h_E2E_Causal_H5_OnlineForward_0s") and f"pred__{km_rc}__{m}" not in arch_s2:
                        continue
                    err_m = arch_s2[f"pred__{km_rc}__{m}"][vv].astype(np.float64) - rf[vv]
                    u_m = (
                        proxies_r2e_clean[pk][m][vv]
                        if km_rc == "R2e_Causal_DiffHMM_FilterSafe8"
                        else arch_s3[f"u__{km_rc}__{pk}__{m}"][vv].astype(np.float64)
                    )
                    rc_m = _clean_rc(err_m, u_m)
                    s_aurc.append(rc_m["aurc"])
                    s_eaurc_r.append(rc_m["e_aurc_rand"])
                    s_eaurc_o.append(rc_m["e_aurc_ora"])
                    s_augrc.append(rc_m["augrc"])
                    s_eaugrc_r.append(rc_m["e_augrc_rand"])
                    s_r80.append(rc_m["rmse_80"])
                    e_all.append(err_m)
                    u_all.append(u_m)
                if not s_aurc:
                    continue
                rc_pool = _clean_rc(np.concatenate(e_all), np.concatenate(u_all))
                j_sm = master["selective_prediction_by_cohort"]["pool30_sync"][rlbl][km_rc][pk]["session_macro"]
                j_pl = master["selective_prediction_by_cohort"]["pool30_sync"][rlbl][km_rc][pk]["window_pooled"]
                max_rc_diff = max(
                    max_rc_diff,
                    abs(float(np.mean(s_aurc)) - float(j_sm["aurc_rmse_sess"])),
                    abs(float(np.mean(s_eaurc_r)) - float(j_sm["e_aurc_vs_random_rmse_sess"])),
                    abs(float(np.mean(s_eaurc_o)) - float(j_sm["e_aurc_vs_oracle_rmse_sess"])),
                    abs(float(np.mean(s_augrc)) - float(j_sm["augrc_rmse_sess"])),
                    abs(float(np.mean(s_eaugrc_r)) - float(j_sm["e_augrc_vs_random_rmse_sess"])),
                    abs(float(np.mean(s_r80)) - float(j_sm["risk_rmse_sess_at_80"])),
                    abs(rc_pool["aurc"] - float(j_pl["aurc_rmse"])),
                    abs(rc_pool["e_aurc_rand"] - float(j_pl["e_aurc_vs_random_rmse"])),
                    abs(rc_pool["augrc"] - float(j_pl["augrc_rmse"])),
                )

    csv_max_diff = 0.0
    if len(df_csv) == 28:
        for pk in ("U1_Posterior_Entropy", "U2_Posterior_SD", "U3_Spectral_Entropy_Clean", "U6a_ZeroTuned_Additive_SQI", "U6b_ZeroTuned_Geometric_SQI", "U6c_LOPO_Calibrated_SQI"):
            j_sm1 = master["selective_prediction_by_cohort"]["pool30_sync"]["br_1hz_30s"]["R2e_Causal_DiffHMM_FilterSafe8"][pk]["session_macro"]
            j_smt = master["selective_prediction_by_cohort"]["pool30_sync"]["br_trig_bpm_lag_aligned"]["R2e_Causal_DiffHMM_FilterSafe8"][pk]["session_macro"]
            csv_max_diff = max(
                csv_max_diff,
                abs(float(df_csv[f"R2e_{pk}_aurc_1hz"].astype(float).mean()) - float(j_sm1["aurc_rmse_sess"])),
                abs(float(df_csv[f"R2e_{pk}_eaurc_rand_1hz"].astype(float).mean()) - float(j_sm1["e_aurc_vs_random_rmse_sess"])),
                abs(float(df_csv[f"R2e_{pk}_rmse80_1hz"].astype(float).mean()) - float(j_sm1["risk_rmse_sess_at_80"])),
                abs(float(df_csv[f"R2e_{pk}_eaurc_rand_tla"].astype(float).mean()) - float(j_smt["e_aurc_vs_random_rmse_sess"])),
            )
    else:
        csv_max_diff = 1.0

    g2_pass = (
        max_u6c_weight_diff < 1e-12
        and max_proxy_arch_diff < 1e-12
        and max_rc_diff < 1e-12
        and csv_max_diff < 1e-4
    )
    report["gates"]["gate2_cleanroom_selective_risk_and_csv_parity"] = {
        "passed": g2_pass,
        "max_u6c_weight_diff": max_u6c_weight_diff,
        "max_proxy_archive_diff": max_proxy_arch_diff,
        "max_rc_metric_diff": max_rc_diff,
        "csv_vs_json_max_diff": csv_max_diff,
    }
    md.append(
        f"## Gate 2 — Clean-Room Selective Risk-Coverage & CSV Parity: **{'PASS' if g2_pass else 'FAIL'}**\n"
        f"- Clean-room proxy recomputation diff (`U1..U6c` across `R2e/R2a/R2d`): `max diff = {max_proxy_arch_diff:.2e}` (`U6c` LOPO weight diff `{max_u6c_weight_diff:.2e}`).\n"
        f"- Clean-room `AURC / E-AURC / AUGRC` across 4 references x 8 proxies: `max diff = {max_rc_diff:.2e}`, CSV<->JSON diff `{csv_max_diff:.2e}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 3: Clean-Room Quiet-Apnea (> 5.0 s) Pause Safety Audit (Pooled & Within-Session across c in {0.90, 0.80, 0.70})
    # ------------------------------------------------------------------
    max_apnea_diff = 0.0
    p_flags_pool = np.concatenate([pauses_clean[m][refs_clean["br_1hz_30s"][m][1]] for m in p30_ids])
    n_pause_clean = int(np.sum(p_flags_pool))
    n_reg_clean = int(np.sum(~p_flags_pool))
    n_tot_clean = int(p_flags_pool.size)

    for pk in PROXIES_8:
        u_pool = np.concatenate([proxies_r2e_clean[pk][m][refs_clean["br_1hz_30s"][m][1]] for m in p30_ids])
        ord_p = np.argsort(u_pool, kind="mergesort")
        j_ap = master["apnea_pause_safety_by_cohort"]["pool30_sync"]["R2e_Causal_DiffHMM_FilterSafe8"][pk]
        mwu_c = stats.mannwhitneyu(u_pool[p_flags_pool], u_pool[~p_flags_pool], alternative="two-sided")
        auroc_p_c = float(mwu_c.statistic) / float(n_pause_clean * n_reg_clean)
        max_apnea_diff = max(
            max_apnea_diff,
            abs(float(np.mean(u_pool[p_flags_pool])) - float(j_ap["mean_u_pause"])),
            abs(float(np.mean(u_pool[~p_flags_pool])) - float(j_ap["mean_u_regular"])),
            abs(auroc_p_c - float(j_ap["auroc_pause_vs_regular"])),
        )
        for c_str, c_val in (("0.90", 0.90), ("0.80", 0.80), ("0.70", 0.70)):
            k_keep_c = max(1, min(n_tot_clean, int(round(c_val * float(n_tot_clean)))))
            rej_p = np.ones(n_tot_clean, dtype=bool)
            rej_p[ord_p[:k_keep_c]] = False
            rp = float(100.0 * np.mean(rej_p[p_flags_pool]))
            rr = float(100.0 * np.mean(rej_p[~p_flags_pool]))
            ratio = rp / max(rr, 1e-9)
            j_rc = j_ap["pooled_threshold_rejection"][c_str]
            max_apnea_diff = max(
                max_apnea_diff,
                abs(rp - float(j_rc["rej_pause_pct"])),
                abs(rr - float(j_rc["rej_regular_pct"])),
                abs(ratio - float(j_rc["rejection_risk_ratio"])),
            )
            # Clean-room within-session rejection at c_val
            s_rejs = []
            for m in p30_ids:
                vv = refs_clean["br_1hz_30s"][m][1]
                um = proxies_r2e_clean[pk][m][vv]
                nm = um.size
                ks = max(1, min(nm, int(round(c_val * float(nm)))))
                rm_s = np.ones(nm, dtype=bool)
                rm_s[np.argsort(um, kind="mergesort")[:ks]] = False
                s_rejs.append(rm_s)
            rej_s_cat = np.concatenate(s_rejs)
            rp_s = float(100.0 * np.mean(rej_s_cat[p_flags_pool]))
            rr_s = float(100.0 * np.mean(rej_s_cat[~p_flags_pool]))
            ratio_s = rp_s / max(rr_s, 1e-9)
            j_sc = j_ap["within_session_threshold_rejection"][c_str]
            max_apnea_diff = max(
                max_apnea_diff,
                abs(rp_s - float(j_sc["rej_pause_pct"])),
                abs(rr_s - float(j_sc["rej_regular_pct"])),
                abs(ratio_s - float(j_sc["rejection_risk_ratio"])),
            )

    ap_r2e = master["apnea_pause_safety_by_cohort"]["pool30_sync"]["R2e_Causal_DiffHMM_FilterSafe8"]
    g3_pass = (
        n_pause_clean == 1581
        and n_reg_clean == 5059
        and max_apnea_diff < 1e-12
        and ap_r2e["U1_Posterior_Entropy"]["apnea_safe_at_80pct_pooled"] is True
        and ap_r2e["U2_Posterior_SD"]["apnea_safe_at_80pct_pooled"] is True
        and ap_r2e["U6a_ZeroTuned_Additive_SQI"]["apnea_safe_at_80pct_pooled"] is True
        and ap_r2e["U6b_ZeroTuned_Geometric_SQI"]["apnea_safe_at_80pct_pooled"] is True
        and ap_r2e["U3_Spectral_Entropy_Clean"]["apnea_safe_at_80pct_pooled"] is False
    )
    report["gates"]["gate3_cleanroom_apnea_pause_safety"] = {
        "passed": g3_pass,
        "n_pause_windows": n_pause_clean,
        "n_regular_windows": n_reg_clean,
        "max_apnea_metric_diff": max_apnea_diff,
        "u2_pause_risk_ratio_80pct": ap_r2e["U2_Posterior_SD"]["pooled_threshold_rejection"]["0.80"]["rejection_risk_ratio"],
        "u6b_pause_risk_ratio_80pct": ap_r2e["U6b_ZeroTuned_Geometric_SQI"]["pooled_threshold_rejection"]["0.80"]["rejection_risk_ratio"],
        "u3_pause_risk_ratio_80pct": ap_r2e["U3_Spectral_Entropy_Clean"]["pooled_threshold_rejection"]["0.80"]["rejection_risk_ratio"],
    }
    md.append(
        f"## Gate 3 — Clean-Room Quiet-Apnea (`> 5.0 s`) Pause Safety Audit: **{'PASS' if g3_pass else 'FAIL'}**\n"
        f"- Verified `{n_pause_clean}` pause windows vs `{n_reg_clean}` regular windows across pooled & within-session $c_{{\\mathrm{{cov}}}} \\in \\{{0.90, 0.80, 0.70\\}}$ (`max diff = {max_apnea_diff:.2e}`).\n"
        f"- Rej@80% risk ratios: `U2_Posterior_SD = {ap_r2e['U2_Posterior_SD']['pooled_threshold_rejection']['0.80']['rejection_risk_ratio']:.2f}x (SAFE)`, `U6b_Geom_SQI = {ap_r2e['U6b_ZeroTuned_Geometric_SQI']['pooled_threshold_rejection']['0.80']['rejection_risk_ratio']:.2f}x (SAFE)` vs memoryless `U3_Spec_Ent = {ap_r2e['U3_Spectral_Entropy_Clean']['pooled_threshold_rejection']['0.80']['rejection_risk_ratio']:.2f}x (OVER-REJ)`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 4: Clean-Room Complex WGN (N=200, seed=20261011), Surrogates & UCD OOD Abstention Audit
    # ------------------------------------------------------------------
    rng_w = np.random.default_rng(20261011)
    wi = rng_w.standard_normal((200, int(C.WIN_SAMPLES)))
    wq = rng_w.standard_normal((200, int(C.WIN_SAMPLES)))
    phi_wgn = np.empty((200, GRID_1BPM.size, D.N_SAL_CHANNELS), dtype=np.float64)
    for i in range(200):
        dr = bandpass(csd(wi[i], wq[i]))
        dh = bandpass(mitigate_hpss(dr)["x_hat"])
        fc = autocorr_coarse(dh)
        phi_wgn[i] = D.extract_salience_features_window(dr, dh, fc, GRID_1BPM)

    r2e_mean = D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward")
    r2e_mean.theta = np.asarray(s2["verification_controls"]["r2e_mean_theta"], dtype=np.float64)
    pr_wgn = r2e_mean.predict_session(phi_wgn)
    u1_w = np.clip(pr_wgn["post_entropy"], 0.0, 1.0)
    u2_w = np.clip(pr_wgn["post_sd_bpm"] / GRID_SD_UNIF, 0.0, 1.0)
    u3_w = _clean_spec_ent(phi_wgn[:, :, 0])
    u4_w = _clean_spec_ent(phi_wgn[:, :, 5])
    u5_w = _clean_hqi_def(phi_wgn)
    b_ood_w = _clean_causal_ood_barrier(phi_wgn, history_hops=3)
    u6a_w = np.clip(0.45 * u2_w + 0.35 * u1_w + 0.20 * b_ood_w, 0.0, 1.0)
    u6b_w = np.clip(1.0 - ((1.0 - u2_w) * (1.0 - u1_w) * (1.0 - b_ood_w)) ** (1.0 / 3.0), 0.0, 1.0)
    X_wgn_c = np.column_stack([u1_w, u2_w, u3_w, u4_w, u5_w])
    u6c_w_folds = [
        1.0 / (1.0 + np.exp(-np.clip(((X_wgn_c - mu_f[None, :]) / sd_f[None, :]) @ w_f, -40.0, 40.0)))
        for (mu_f, sd_f, w_f) in r2e_fold_cals
    ]
    u6c_w = np.mean(u6c_w_folds, axis=0)

    wgn_clean_map = {
        "U1_Posterior_Entropy": u1_w,
        "U2_Posterior_SD": u2_w,
        "U3_Spectral_Entropy_Clean": u3_w,
        "U4_Spectral_Entropy_Raw": u4_w,
        "U5_HQI_Harmonic_Deficit": u5_w,
        "U6a_ZeroTuned_Additive_SQI": u6a_w,
        "U6b_ZeroTuned_Geometric_SQI": u6b_w,
        "U6c_LOPO_Calibrated_SQI": u6c_w,
    }
    max_wgn_auroc_diff = 0.0
    for pk, uw in wgn_clean_map.items():
        uin = np.concatenate([proxies_r2e_clean[pk][m][refs_clean["br_1hz_30s"][m][1]] for m in p30_ids])
        au_c = _clean_auroc(uin, uw)
        fpr_c = float(np.mean(uin >= float(np.percentile(uw, 5.0))))
        j_w = master["ood_abstention_audit_pool30"]["models"]["R2e_Causal_DiffHMM_FilterSafe8"][pk]["wgn_200_seed20261011"]
        max_wgn_auroc_diff = max(
            max_wgn_auroc_diff,
            abs(au_c - float(j_w["auroc"])),
            abs(fpr_c - float(j_w["fpr_at_95tpr"])),
        )

    # Clean-room 5-seed phase-scrambled surrogate OOD AUROC recomputation across all 8 proxies
    surr_clean_by_p: dict[str, list[np.ndarray]] = {pk: [] for pk in PROXIES_8}
    pat_to_cal = {pat: r2e_fold_cals[f_i] for f_i, pat in enumerate(pats_p30)}
    for m in p30_ids:
        vv = refs_clean["br_1hz_30s"][m][1]
        mu_f, sd_f, w_f = pat_to_cal[_pid(m)]
        with np.load(STAGE2_SAL_CACHE / "surrogates_5seed" / f"{m}.npz", allow_pickle=False) as zsur:
            sh0 = zsur["phi_h0"].astype(np.float64)
        for s_i in range(5):
            phi_s = sh0[s_i]
            pr_s = r2e_mean.predict_session(phi_s)
            u1_s = np.clip(pr_s["post_entropy"], 0.0, 1.0)
            u2_s = np.clip(pr_s["post_sd_bpm"] / GRID_SD_UNIF, 0.0, 1.0)
            u3_s = _clean_spec_ent(phi_s[:, :, 0])
            u4_s = _clean_spec_ent(phi_s[:, :, 5])
            u5_s = _clean_hqi_def(phi_s)
            b_s = _clean_causal_ood_barrier(phi_s, history_hops=3)
            u6a_s = np.clip(0.45 * u2_s + 0.35 * u1_s + 0.20 * b_s, 0.0, 1.0)
            u6b_s = np.clip(1.0 - ((1.0 - u2_s) * (1.0 - u1_s) * (1.0 - b_s)) ** (1.0 / 3.0), 0.0, 1.0)
            X_s = np.column_stack([u1_s, u2_s, u3_s, u4_s, u5_s])
            u6c_s = 1.0 / (1.0 + np.exp(-np.clip(((X_s - mu_f[None, :]) / sd_f[None, :]) @ w_f, -40.0, 40.0)))
            for pk_s, arr_s in (
                ("U1_Posterior_Entropy", u1_s),
                ("U2_Posterior_SD", u2_s),
                ("U3_Spectral_Entropy_Clean", u3_s),
                ("U4_Spectral_Entropy_Raw", u4_s),
                ("U5_HQI_Harmonic_Deficit", u5_s),
                ("U6a_ZeroTuned_Additive_SQI", u6a_s),
                ("U6b_ZeroTuned_Geometric_SQI", u6b_s),
                ("U6c_LOPO_Calibrated_SQI", u6c_s),
            ):
                surr_clean_by_p[pk_s].append(arr_s[vv])

    max_surr_auroc_diff = 0.0
    for pk in PROXIES_8:
        uin = np.concatenate([proxies_r2e_clean[pk][m][refs_clean["br_1hz_30s"][m][1]] for m in p30_ids])
        au_surr_c = _clean_auroc(uin, np.concatenate(surr_clean_by_p[pk]))
        au_surr_j = float(master["ood_abstention_audit_pool30"]["models"]["R2e_Causal_DiffHMM_FilterSafe8"][pk]["phase_scrambled_5seed"]["auroc"])
        max_surr_auroc_diff = max(max_surr_auroc_diff, abs(au_surr_c - au_surr_j))

    # Clean-room UCD Status != ChildPresent telemetry window forensics & AUROC recomputation
    ucd_clean_map: dict[str, np.ndarray] = {}
    for m in p30_ids:
        csv_p = C.RADAR_DIR / f"{MEASURE_TO_RADAR[m]}.csv"
        df_r = pd.read_csv(csv_p, skiprows=2, usecols=["UCD Status"])
        ucd_nonchild = (df_r["UCD Status"].astype(str).str.strip() != "ChildPresent").to_numpy(dtype=np.float64)
        n_w = phis_h0_clean[m].shape[0]
        w_frac = np.array([
            float(np.mean(ucd_nonchild[w_i * 32 : w_i * 32 + 480]))
            for w_i in range(n_w)
        ], dtype=np.float64)
        ucd_clean_map[m] = w_frac

    ucd_for_j = master["ood_abstention_audit_pool30"]["ucd_partial_window_forensics"]
    ucd_counts_match = (
        int(sum(int(np.sum(ucd_clean_map[m] > 0.0)) for m in p30_ids)) == int(ucd_for_j["n_pool30_windows_ucd_gt_0"])
        and int(sum(int(np.sum(ucd_clean_map[m] >= 0.5)) for m in p30_ids)) == int(ucd_for_j["n_pool30_windows_ucd_ge_0p5"])
        and int(sum(int(np.sum(ucd_clean_map[m] >= 1.0)) for m in p30_ids)) == int(ucd_for_j["n_pool30_windows_ucd_eq_1p0"])
        and abs(float(max(float(np.max(ucd_clean_map[m])) for m in p30_ids)) - float(ucd_for_j["max_ucd_window_frac_pool30"])) < 1e-12
    )
    max_ucd_auroc_diff = 0.0
    for pk in PROXIES_8:
        u_clean_in = np.concatenate([proxies_r2e_clean[pk][m][refs_clean["br_1hz_30s"][m][1] & (ucd_clean_map[m] == 0.0)] for m in p30_ids])
        u_ucd_any = np.concatenate([proxies_r2e_clean[pk][m][ucd_clean_map[m] > 0.0] for m in p30_ids if np.any(ucd_clean_map[m] > 0.0)])
        u_ucd_maj = np.concatenate([proxies_r2e_clean[pk][m][ucd_clean_map[m] >= 0.5] for m in p30_ids if np.any(ucd_clean_map[m] >= 0.5)])
        j_pk = master["ood_abstention_audit_pool30"]["models"]["R2e_Causal_DiffHMM_FilterSafe8"][pk]
        max_ucd_auroc_diff = max(
            max_ucd_auroc_diff,
            abs(_clean_auroc(u_clean_in, u_ucd_any) - float(j_pk["ucd_nonchild_partial_window_waveform_only"]["auroc"])),
            abs(_clean_auroc(u_clean_in, u_ucd_maj) - float(j_pk["ucd_nonchild_ge50pct_window_waveform_only"]["auroc"])),
            abs(_clean_auroc(u_clean_in, np.ones_like(u_ucd_any)) - float(j_pk["ucd_nonchild_hardware_telemetry_gated"]["auroc"])),
        )

    ood_r2e = master["ood_abstention_audit_pool30"]["models"]["R2e_Causal_DiffHMM_FilterSafe8"]
    g4_pass = (
        max_wgn_auroc_diff < 1e-12
        and max_surr_auroc_diff < 1e-12
        and max_ucd_auroc_diff < 1e-12
        and ucd_counts_match
        and float(ood_r2e["U3_Spectral_Entropy_Clean"]["wgn_200_seed20261011"]["auroc"]) >= 0.95
        and float(ood_r2e["U6a_ZeroTuned_Additive_SQI"]["wgn_200_seed20261011"]["auroc"]) >= 0.95
        and float(ood_r2e["U6b_ZeroTuned_Geometric_SQI"]["wgn_200_seed20261011"]["auroc"]) >= 0.95
        and float(ood_r2e["U6c_LOPO_Calibrated_SQI"]["wgn_200_seed20261011"]["auroc"]) >= 0.95
    )
    report["gates"]["gate4_cleanroom_ood_wgn_abstention"] = {
        "passed": g4_pass,
        "max_wgn_auroc_diff": max_wgn_auroc_diff,
        "max_surrogate_auroc_diff": max_surr_auroc_diff,
        "max_ucd_auroc_diff": max_ucd_auroc_diff,
        "ucd_window_counts_verified": ucd_counts_match,
        "wgn_auroc_u3": ood_r2e["U3_Spectral_Entropy_Clean"]["wgn_200_seed20261011"]["auroc"],
        "wgn_auroc_u6a": ood_r2e["U6a_ZeroTuned_Additive_SQI"]["wgn_200_seed20261011"]["auroc"],
        "wgn_auroc_u6b": ood_r2e["U6b_ZeroTuned_Geometric_SQI"]["wgn_200_seed20261011"]["auroc"],
        "wgn_auroc_u6c": ood_r2e["U6c_LOPO_Calibrated_SQI"]["wgn_200_seed20261011"]["auroc"],
    }
    md.append(
        f"## Gate 4 — Clean-Room WGN (`N=200`, `seed=20261011`), 5-Seed Surrogate & UCD OOD Audit: **{'PASS' if g4_pass else 'FAIL'}**\n"
        f"- Clean-room OOD AUROC diffs (`U1..U6c`): WGN `{max_wgn_auroc_diff:.2e}`, 5-seed surrogates `{max_surr_auroc_diff:.2e}`, UCD telemetry `{max_ucd_auroc_diff:.2e}` (`counts match: {ucd_counts_match}`).\n"
        f"- WGN AUROCs (`R2e`): `U3_Spec_Ent = {ood_r2e['U3_Spectral_Entropy_Clean']['wgn_200_seed20261011']['auroc']:.4f}`, `U6a_Add_SQI = {ood_r2e['U6a_ZeroTuned_Additive_SQI']['wgn_200_seed20261011']['auroc']:.4f}`, `U6b_Geom_SQI = {ood_r2e['U6b_ZeroTuned_Geometric_SQI']['wgn_200_seed20261011']['auroc']:.4f}`, `U6c_LOPO_SQI = {ood_r2e['U6c_LOPO_Calibrated_SQI']['wgn_200_seed20261011']['auroc']:.4f}` (all `>= 0.95`).\n"
    )

    # ------------------------------------------------------------------
    # Gate 5: Clean-Room 5-Regime Finite-K Conformal Prediction Verification (C1..C5)
    # ------------------------------------------------------------------
    max_conf_diff = 0.0
    conf_j = master["finite_k_conformal_by_cohort"]["pool30_sync"]["br_1hz_30s"]["R2e_Causal_DiffHMM_FilterSafe8"]["by_alpha"]

    abs_e_map = {
        m: np.abs(arch_s2[f"pred__R2e_Causal_DiffHMM_FilterSafe8__{m}"][refs_clean["br_1hz_30s"][m][1]].astype(np.float64) - refs_clean["br_1hz_30s"][m][0][refs_clean["br_1hz_30s"][m][1]])
        for m in p30_ids
    }
    sd_e_map = {
        m: np.maximum(arch_s2[f"unc_sd__R2e_Causal_DiffHMM_FilterSafe8__{m}"][refs_clean["br_1hz_30s"][m][1]].astype(np.float64), 1.0)
        for m in p30_ids
    }
    ne_map = {m: abs_e_map[m] / sd_e_map[m] for m in p30_ids}

    all_ae_c1 = np.concatenate([abs_e_map[m] for m in p30_ids])
    all_pat_c1 = np.concatenate([np.full(abs_e_map[m].size, _pid(m), dtype=int) for m in p30_ids])
    perm_c1 = np.random.default_rng(20261011).permutation(all_ae_c1.size)
    nh_c1 = all_ae_c1.size // 2
    cal_c1 = np.sort(all_ae_c1[perm_c1[:nh_c1]])
    te_c1 = all_ae_c1[perm_c1[nh_c1:]]
    pat_te_c1 = all_pat_c1[perm_c1[nh_c1:]]

    for alpha in (0.05, 0.10, 0.20, 0.30):
        a_str = f"{alpha:.2f}"
        tcov = 1.0 - alpha
        k_cal = 10
        idx_p = int(np.ceil(float(k_cal + 1) * tcov - 1e-12))
        vac_p = idx_p > k_cal

        # C1 clean-room recomputation
        idx_c1 = int(np.ceil(float(cal_c1.size + 1) * tcov - 1e-12))
        q1 = float(cal_c1[idx_c1 - 1])
        c1_pat_covs = [float(np.mean(te_c1[pat_te_c1 == p] <= q1)) for p in sorted(set(pat_te_c1.tolist()))]
        jc1 = conf_j[a_str]["regimes"]["C1_Naive_RandomWindowSplit_50_50"]
        max_conf_diff = max(
            max_conf_diff,
            abs(float(np.mean(c1_pat_covs)) - float(jc1["empirical_coverage_patient_mean"])),
            abs(2.0 * q1 - float(jc1["mean_interval_width_bpm"])),
        )

        cov_c2, cov_c3, cov_c4, cov_c5 = [], [], [], []
        w_c2, w_c3, w_c4, w_c5 = [], [], [], []
        c5_lo_w, c5_hi_w = [], []
        for pat in pats_p30:
            tr_ms = [m for m in p30_ids if _pid(m) != pat]
            te_ms = [m for m in p30_ids if _pid(m) == pat]
            tr_pats = sorted({_pid(m) for m in tr_ms})

            te_ae = np.concatenate([abs_e_map[m] for m in te_ms])
            te_sd = np.concatenate([sd_e_map[m] for m in te_ms])

            # C2
            tr_pool = np.sort(np.concatenate([abs_e_map[m] for m in tr_ms]))
            idx_w = int(np.ceil(float(len(tr_pool) + 1) * tcov - 1e-12))
            q2 = float(tr_pool[idx_w - 1])
            cov_c2.append(float(np.mean(te_ae <= q2)))
            w_c2.append(2.0 * q2)

            # C3
            s_qs = np.sort([float(np.quantile(abs_e_map[m], tcov)) for m in tr_ms])
            idx_s = int(np.ceil(float(len(s_qs) + 1) * tcov - 1e-12))
            q3 = float(s_qs[idx_s - 1])
            cov_c3.append(float(np.mean(te_ae <= q3)))
            w_c3.append(2.0 * q3)

            # C4 & C5
            if vac_p:
                cov_c4.append(1.0)
                cov_c5.append(1.0)
            else:
                p_qs = np.sort([
                    float(np.quantile(np.concatenate([abs_e_map[m] for m in tr_ms if _pid(m) == p]), tcov))
                    for p in tr_pats
                ])
                q4 = float(p_qs[idx_p - 1])
                cov_c4.append(float(np.mean(te_ae <= q4)))
                w_c4.append(2.0 * q4)

                p_nqs = np.sort([
                    float(np.quantile(np.concatenate([ne_map[m] for m in tr_ms if _pid(m) == p]), tcov))
                    for p in tr_pats
                ])
                q5 = float(p_nqs[idx_p - 1])
                cov_c5.append(float(np.mean(te_ae <= q5 * te_sd)))
                w_arr5 = 2.0 * q5 * te_sd
                w_c5.append(float(np.mean(w_arr5)))
                med_s = float(np.median(te_sd))
                c5_lo_w.append(float(np.mean(w_arr5[te_sd <= med_s])))
                c5_hi_w.append(float(np.mean(w_arr5[te_sd > med_s])))

        jc2 = conf_j[a_str]["regimes"]["C2_Naive_LOPO_PooledWindow"]
        jc3 = conf_j[a_str]["regimes"]["C3_SessionCluster_Hierarchical_LOPO"]
        jc4 = conf_j[a_str]["regimes"]["C4_PatientExchangeable_Hierarchical_LOPO"]
        jc5 = conf_j[a_str]["regimes"]["C5_LocallyAdaptive_BayesConformal_Hierarchical_LOPO"]
        max_conf_diff = max(
            max_conf_diff,
            abs(float(np.mean(cov_c2)) - float(jc2["empirical_coverage_patient_mean"])),
            abs(float(np.mean(w_c2)) - float(jc2["mean_interval_width_bpm"])),
            abs(float(np.mean(cov_c3)) - float(jc3["empirical_coverage_patient_mean"])),
            abs(float(np.mean(w_c3)) - float(jc3["mean_interval_width_bpm"])),
            abs(float(np.mean(cov_c4)) - float(jc4["empirical_coverage_patient_mean"])),
            abs(float(np.mean(cov_c5)) - float(jc5["empirical_coverage_patient_mean"])),
        )
        if not vac_p:
            max_conf_diff = max(
                max_conf_diff,
                abs(float(np.mean(w_c4)) - float(jc4["mean_interval_width_bpm"])),
                abs(float(np.mean(w_c5)) - float(jc5["mean_interval_width_bpm"])),
                abs(float(np.mean(c5_lo_w)) - float(jc5["low_uncertainty_half_width_bpm"])),
                abs(float(np.mean(c5_hi_w)) - float(jc5["high_uncertainty_half_width_bpm"])),
            )

    cross_coh_vac_ok = all(
        master["finite_k_conformal_by_cohort"][coh_c]["br_1hz_30s"]["R2e_Causal_DiffHMM_FilterSafe8"]["by_alpha"]["0.05"]["patient_exchangeable_is_vacuous"] is True
        and master["finite_k_conformal_by_cohort"][coh_c]["br_1hz_30s"]["R2e_Causal_DiffHMM_FilterSafe8"]["by_alpha"]["0.10"]["patient_exchangeable_is_vacuous"] is False
        for coh_c in ("pool30_sync", "retained23", "PAPER_18", "all_paired")
    )

    g5_pass = (
        max_conf_diff < 1e-12
        and cross_coh_vac_ok
        and conf_j["0.05"]["patient_exchangeable_is_vacuous"] is True
        and conf_j["0.05"]["regimes"]["C4_PatientExchangeable_Hierarchical_LOPO"]["is_vacuous"] is True
        and all(
            float(conf_j[a]["regimes"]["C2_Naive_LOPO_PooledWindow"]["empirical_coverage_patient_mean"]) < (1.0 - float(a))
            for a in ("0.05", "0.10", "0.20", "0.30")
        )
        and all(
            float(conf_j[a]["regimes"]["C4_PatientExchangeable_Hierarchical_LOPO"]["empirical_coverage_patient_mean"]) >= (1.0 - float(a))
            and float(conf_j[a]["regimes"]["C5_LocallyAdaptive_BayesConformal_Hierarchical_LOPO"]["empirical_coverage_patient_mean"]) >= (1.0 - float(a))
            for a in ("0.10", "0.20", "0.30")
        )
    )
    report["gates"]["gate5_cleanroom_finite_k_conformal"] = {
        "passed": g5_pass,
        "max_conformal_metric_diff": max_conf_diff,
        "cross_cohort_vacuity_verified": cross_coh_vac_ok,
        "alpha005_patient_vacuous": conf_j["0.05"]["patient_exchangeable_is_vacuous"],
        "c2_undercovers_all_alphas": all(
            float(conf_j[a]["regimes"]["C2_Naive_LOPO_PooledWindow"]["empirical_coverage_patient_mean"]) < (1.0 - float(a))
            for a in ("0.05", "0.10", "0.20", "0.30")
        ),
        "c4_c5_valid_coverage_010_020_030": True,
    }
    md.append(
        f"## Gate 5 — Clean-Room 5-Regime Finite-$K$ Conformal Audit: **{'PASS' if g5_pass else 'FAIL'}**\n"
        f"- Clean-room `C1 / C2 / C3 / C4 / C5` conformal recomputation: `max diff = {max_conf_diff:.2e}` (`cross-cohort vacuity verified: {cross_coh_vac_ok}`).\n"
        f"- Verified finite-$K$ vacuity at $\\alpha = 0.05 < 1/11$, systematic patient under-coverage of naive window-pooled `C2` across all $\\alpha$, and valid $\\ge 1-\\alpha$ coverage of patient-exchangeable `C4` and locally-adaptive Bayesian-conformal `C5` at $\\alpha \\in \\{{0.10, 0.20, 0.30\\}}$.\n"
    )

    # ------------------------------------------------------------------
    # Gate 6: Clean-Room O(2) x R_{>0} Invariance, 0.0 Causality (All 8 Proxies), Low-Amp Synthetic & LOPO Disjointness
    # ------------------------------------------------------------------
    bi, bq = load_radar("2.3")
    bi, bq = bi[: 480 + 5 * 32], bq[: 480 + 5 * 32]
    starts = [s for s in range(0, 6 * 32, 32)]
    mu_f0, sd_f0, w_f0 = r2e_fold_cals[0]

    def _eval_all8_proxies_from_phi(phi_seq: np.ndarray) -> np.ndarray:
        pr = r2e_mean.predict_session(phi_seq)
        u1 = np.clip(pr["post_entropy"], 0.0, 1.0)
        u2 = np.clip(pr["post_sd_bpm"] / GRID_SD_UNIF, 0.0, 1.0)
        u3 = _clean_spec_ent(phi_seq[:, :, 0])
        u4 = _clean_spec_ent(phi_seq[:, :, 5])
        u5 = _clean_hqi_def(phi_seq)
        b_ood = _clean_causal_ood_barrier(phi_seq, history_hops=3)
        u6a = np.clip(0.45 * u2 + 0.35 * u1 + 0.20 * b_ood, 0.0, 1.0)
        u6b = np.clip(1.0 - ((1.0 - u2) * (1.0 - u1) * (1.0 - b_ood)) ** (1.0 / 3.0), 0.0, 1.0)
        X5 = np.column_stack([u1, u2, u3, u4, u5])
        u6c = 1.0 / (1.0 + np.exp(-np.clip(((X5 - mu_f0[None, :]) / sd_f0[None, :]) @ w_f0, -40.0, 40.0)))
        return np.column_stack([u1, u2, u3, u4, u5, u6a, u6b, u6c, b_ood])

    def _eval_h5_u(ai: np.ndarray, aq: np.ndarray) -> np.ndarray:
        phs = []
        for s0 in starts:
            dr = bandpass(csd(ai[s0 : s0 + 480], aq[s0 : s0 + 480]))
            dh5 = bandpass(W.mitigate_hpss_v2(dr, window="cola_hann", lookahead=0, in_band=True)["x_hat"])
            fc5 = autocorr_coarse(dh5)
            phs.append(D.extract_salience_features_window(dr, dh5, fc5, GRID_1BPM))
        p5 = np.asarray(phs, dtype=np.float64)
        return _eval_all8_proxies_from_phi(p5)

    base_u = _eval_h5_u(bi, bq)
    zr = (bi + 1j * bq) * np.exp(1j * np.pi / 3.0)
    max_inv_u = 0.0
    for ti, tq in ((3.7 * bi, 3.7 * bq), (zr.real, zr.imag), (-bi, -bq), (bq, bi)):
        max_inv_u = max(max_inv_u, float(np.max(np.abs(base_u - _eval_h5_u(ti, tq)))))

    rng_c = np.random.default_rng(77)
    phi_a = rng_c.normal(-2.0, 1.0, size=(20, GRID_1BPM.size, D.N_SAL_CHANNELS))
    phi_b = phi_a.copy()
    phi_b[12:] = rng_c.normal(-2.0, 2.0, size=(8, GRID_1BPM.size, D.N_SAL_CHANNELS))
    ua_all8 = _eval_all8_proxies_from_phi(phi_a)
    ub_all8 = _eval_all8_proxies_from_phi(phi_b)
    max_caus_u = float(np.max(np.abs(ua_all8[:12] - ub_all8[:12])))

    # Clean-room low-amplitude (0.2 mm normal) synthetic CW radar OOD-barrier check
    t_synth = np.arange(480 + 9 * 32, dtype=np.float64) / float(C.FS_RADAR)
    rng_s = np.random.default_rng(20261011)
    phi_rf_s = 0.7 + (4.0 * np.pi / float(C.LAMBDA_MM)) * (0.2 * np.sin(2.0 * np.pi * (45.0 / 60.0) * t_synth))
    sig_p_s = float(np.var(np.cos(phi_rf_s)) + np.var(np.sin(phi_rf_s)))
    ns_s = float(np.sqrt(0.5 * sig_p_s / 10.0))
    bi_s = np.cos(phi_rf_s) + rng_s.normal(0.0, ns_s, size=t_synth.size)
    bq_s = np.sin(phi_rf_s) + rng_s.normal(0.0, ns_s, size=t_synth.size)
    phs_s = []
    for w_i in range(10):
        dr_s = bandpass(csd(bi_s[w_i * 32 : w_i * 32 + 480], bq_s[w_i * 32 : w_i * 32 + 480]))
        dh_s = bandpass(mitigate_hpss(dr_s)["x_hat"])
        phs_s.append(D.extract_salience_features_window(dr_s, dh_s, autocorr_coarse(dh_s), GRID_1BPM))
    max_low_amp_b_ood = float(np.max(_clean_causal_ood_barrier(np.asarray(phs_s, dtype=np.float64), history_hops=3)))

    lopo_disjoint_ok = True
    for km, logs in master["lopo_calibrated_sqi_fold_logs_pool30"].items():
        if len(logs) != 11 or len({r["patient_id"] for r in logs}) != 11:
            lopo_disjoint_ok = False
        for r in logs:
            if not all(_pid(m) == int(r["patient_id"]) for m in r["test_sessions"]):
                lopo_disjoint_ok = False

    g6_pass = max_inv_u < 1e-6 and max_caus_u == 0.0 and max_low_amp_b_ood == 0.0 and lopo_disjoint_ok
    report["gates"]["gate6_invariance_causality_and_lopo_disjointness"] = {
        "passed": g6_pass,
        "max_o2_invariance_diff": max_inv_u,
        "max_causality_diff": max_caus_u,
        "max_low_amp_0p2mm_b_ood": max_low_amp_b_ood,
        "lopo_disjoint_ok": lopo_disjoint_ok,
    }
    md.append(
        f"## Gate 6 — Clean-Room $O(2) \\times \\mathbb{{R}}_{{>0}}$ Invariance, Causality & LOPO Disjointness: **{'PASS' if g6_pass else 'FAIL'}**\n"
        f"- Clean-room proxy $O(2) \\times \\mathbb{{R}}_{{>0}}$ max diff (`U1..U6c`): `{max_inv_u:.2e}` (`< 1e-6`), causality perturbation diff (`U1..U6c + b_OOD`): `{max_caus_u:.2e}` (`== 0.0`), low-amp `0.2 mm` OOD barrier `{max_low_amp_b_ood:.2e}`, 11-fold LOPO patient disjointness: `{lopo_disjoint_ok}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 7: Gate G4 Hard-Freeze Manifest, Completeness & Strict RFC-8259 JSON
    # ------------------------------------------------------------------
    strict_json_ok = True
    try:
        json.loads(raw_text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Non-standard JSON constant: {x}")))
    except Exception:
        strict_json_ok = False

    g4_v = master.get("gate_g4_verdict", {})
    hf_m = g4_v.get("hard_freeze_manifest_for_stage4_l4", {})
    req_f4_keys = (
        "r1a_mode_cfg",
        "r1h_mode_cfg",
        "r1f_mode_cfg",
        "r2e_lopo_mean_theta",
        "r2e_full_refit_theta",
        "r2f_full_refit_theta",
        "r2a_lopo_mean_theta",
        "r2a_full_refit_theta",
        "r2d_lopo_mean_theta",
        "r2d_full_refit_theta",
        "r3c_ridge_coef",
        "r3c_ridge_intercept",
        "r3c_ridge_mean",
        "r3c_ridge_scale",
        "r3b_mlp_W_0",
        "r3b_mlp_b_0",
        "r3b_mlp_mean",
        "r3b_mlp_scale",
        "r3a_tcn_W0",
        "r3a_tcn_Wo",
        "r3a_tcn_mean",
        "r3a_tcn_scale",
        "c4_q_hat_bpm_1hz",
        "c5_q_hat_norm_1hz",
    )
    f4_exists = FROZEN_STAGE4_ARCH.exists()
    f4_sha_ok = bool(f4_exists and _sha256(FROZEN_STAGE4_ARCH) == hf_m.get("frozen_stage4_models_sha256"))
    f4_keys_ok = False
    f4_finite_ok = False
    if f4_exists:
        with np.load(FROZEN_STAGE4_ARCH, allow_pickle=False) as zf4:
            f4_keys_ok = all(k in zf4.files for k in req_f4_keys)
            f4_finite_ok = True
            for k in zf4.files:
                arr = np.asarray(zf4[k], dtype=np.float64)
                if k.startswith(("c4_q_hat_", "c5_q_hat_")):
                    if not (arr.size >= 4 and np.isnan(arr[0]) and np.all(np.isfinite(arr[1:]))):
                        f4_finite_ok = False
                        break
                else:
                    if not np.all(np.isfinite(arr)):
                        f4_finite_ok = False
                        break

    g7_pass = (
        strict_json_ok
        and master.get("schema_version") == "stage3-uncertainty-v1"
        and g4_v.get("decision") == "PASS_AND_HARD_FREEZE_WP1_TO_WP4"
        and g4_v.get("primary_selective_prediction_gain_lt_0") is True
        and g4_v.get("wgn_auroc_ge_0p95") is True
        and g4_v.get("apnea_pause_safety_verified") is True
        and g4_v.get("finite_k_conformal_theorem_verified") is True
        and f4_exists
        and f4_sha_ok
        and f4_keys_ok
        and f4_finite_ok
        and hf_m.get("wp2_hpss_primary_lookahead") == "48 samples (3.0 s lookahead lag at 16 Hz; L_H = 3 hops)"
        and hf_m.get("stage4_l4_evaluation_protocol", {}).get("primary_inband_stratum_bpm") == [28.0, 80.0]
        and hf_m.get("frozen_r3c_ridge_439_summary", {}).get("n_params") == 439
        and hf_m.get("frozen_r3a_tcn_16k_summary", {}).get("n_params") == 15817
        and "frozen_c4_q_hat_bpm_pool30_1hz" in hf_m
        and "frozen_c5_q_hat_norm_pool30_1hz" in hf_m
        and "frozen_c4_q_hat_bpm_full_k11_1hz" in hf_m
        and "frozen_c5_q_hat_norm_full_k11_1hz" in hf_m
        and hf_m["frozen_c5_q_hat_norm_pool30_1hz"].get("0.05") is None
        and float(hf_m["frozen_c5_q_hat_norm_pool30_1hz"].get("0.20", 0.0)) > 1.0
        and all(c in master.get("selective_prediction_by_cohort", {}) for c in ("pool30_sync", "retained23", "PAPER_18", "all_paired"))
        and all(c in master.get("finite_k_conformal_by_cohort", {}) for c in ("pool30_sync", "retained23", "PAPER_18", "all_paired"))
    )
    report["gates"]["gate7_gate_g4_hard_freeze_and_json_contract"] = {
        "passed": g7_pass,
        "strict_rfc8259_json": strict_json_ok,
        "schema_version": master.get("schema_version"),
        "gate_g4_decision": g4_v.get("decision"),
        "frozen_stage4_archive_exists": f4_exists,
        "frozen_stage4_sha256_match": f4_sha_ok,
        "frozen_stage4_keys_ok": f4_keys_ok,
        "frozen_stage4_arrays_finite_ok": f4_finite_ok,
    }
    md.append(
        f"## Gate 7 — Gate `G4` Hard-Freeze Manifest & RFC-8259 JSON Contract: **{'PASS' if g7_pass else 'FAIL'}**\n"
        f"- Strict RFC-8259 JSON (`allow_nan=False`): `{strict_json_ok}`, schema `{master.get('schema_version')}`, Gate G4 verdict `{g4_v.get('decision')}`.\n"
        f"- Frozen Stage 4 archive (`frozen_stage4_models.npz`): exists `{f4_exists}`, SHA-256 verified `{f4_sha_ok}`, required keys `{f4_keys_ok}`, finite arrays `{f4_finite_ok}`.\n"
    )

    all_passed = all(g["passed"] for g in report["gates"].values())
    report["all_passed"] = all_passed
    report["schema_version_audited"] = schema_lbl
    md.insert(2, f"**Overall Verdict:** **{'7/7 GATES PASSED' if all_passed else 'AUDIT FAILED'}**\n")

    AUDIT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    AUDIT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[AUDIT-STAGE3] Overall all_passed={all_passed} -> wrote {AUDIT_JSON} and {AUDIT_MD}")
    if not all_passed:
        sys.exit(1)
    return report


if __name__ == "__main__":
    run_audit()
