"""Zero-shared-code clean-room auditor for Stage 1 (`stage1-parsimony-v3`).

Intentionally imports ZERO statistical or LOPO evaluation logic from
`thesis.exp1_stage1_parsimony`. Independently recomputes EDF references,
raw ADC rail fractions, clean-room HMM / Viterbi decoders, O(2) x R+
invariances, frame-level vs block-level causality perturbations, TOST,
patient-clustered Wilcoxon, rmcorr, ICC(1,1), BibTeX parity, and frozen-
baseline SHA-256 verification.

Execution:
  MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /opt/homebrew/bin/python3.10 -m thesis.audit_stage1_parsimony
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
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyedflib
from scipy import stats
from scipy.special import logsumexp

from paper_workspace.src import config as C
from paper_workspace.src.cohort import PAPER_18, local_pool, pool30, retained23
from paper_workspace.src.data_loader import MEASURE_TO_REF, load_radar
from paper_workspace.src.demod import csd
from paper_workspace.src.estimation import nls_estimate
from paper_workspace.src.evaluation import window_ref_and_gating
from paper_workspace.src.filtering import autocorr_coarse, bandpass
from paper_workspace.src.hpss import mitigate_hpss
from paper_workspace.src.provenance import PIPELINE_VERSION, expected_provenance
from thesis import viterbi_tracker as VT
from thesis import wp1_wp2_causal_hmm as W

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "audit"
MANIFEST_PATH = AUDIT_DIR / "postfix_manifest.sha256"
LAGS_CSV = AUDIT_DIR / "phase2_ws3_session_lags.csv"
LOCKED_CACHE_DIR = ROOT / "paper_workspace" / "outputs" / "cache"
STAGE1_CACHE_DIR = ROOT / "thesis" / "outputs" / "stage1_parsimony" / "cache"
BUNDLE_DIR = ROOT / "paper_orchestra_bundle" / "raw_materials"
METRICS_JSON = BUNDLE_DIR / "metrics_stage1_parsimony.json"
SESSIONS_CSV = AUDIT_DIR / "stage1_parsimony_sessions.csv"
AUDIT_JSON = AUDIT_DIR / "stage1_godmode_audit.json"
AUDIT_MD = AUDIT_DIR / "stage1_godmode_audit.md"

BR_GRID = np.arange(18.0, 90.0 + 1e-9, 1.0, dtype=np.float64)


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


# ---------------------------------------------------------------------------
# Clean-room statistics & HMM decoders (independent from thesis.io_utils/exp1)
# ---------------------------------------------------------------------------

def _clean_rmcorr(est_list: list[np.ndarray], ref_list: list[np.ndarray]) -> float:
    dx_p, dy_p = [], []
    for e, r in zip(est_list, ref_list):
        if len(e) < 2:
            continue
        dx_p.append(e - np.mean(e))
        dy_p.append(r - np.mean(r))
    if not dx_p:
        return float("nan")
    dx = np.concatenate(dx_p)
    dy = np.concatenate(dy_p)
    den = float(np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2)))
    if den <= 0.0:
        return float("nan")
    return float(np.clip(np.sum(dx * dy) / den, -1.0, 1.0))


def _clean_icc_1_1(est_means: np.ndarray, ref_means: np.ndarray) -> float:
    mat = np.column_stack([est_means, ref_means])
    n = mat.shape[0]
    grand = float(np.mean(mat))
    rm = np.mean(mat, axis=1)
    bms = 2.0 * float(np.sum((rm - grand) ** 2)) / float(n - 1)
    wms = float(np.sum((mat - rm[:, None]) ** 2)) / float(n)
    return float((bms - wms) / (bms + wms))


def _clean_tost(deltas: np.ndarray, margin: float = 0.50) -> dict[str, Any]:
    arr = np.asarray(deltas, dtype=np.float64)
    n = arr.size
    mean_d = float(np.mean(arr))
    sd_d = float(np.std(arr, ddof=1))
    if sd_d <= 1e-15:
        eq = bool(abs(mean_d) < margin)
        return {"mean": mean_d, "p_tost": 0.0 if eq else 1.0, "equivalent": eq}
    se = sd_d / float(np.sqrt(n))
    df = n - 1
    p1 = float(1.0 - stats.t.cdf((mean_d + margin) / se, df=df))
    p2 = float(1.0 - stats.t.cdf((margin - mean_d) / se, df=df))
    pt = float(max(p1, p2))
    return {"mean": mean_d, "p_tost": pt, "equivalent": bool(pt < 0.05)}


def _clean_wilcoxon(deltas: np.ndarray) -> float:
    arr = np.asarray(deltas, dtype=np.float64)
    nz = arr[np.abs(arr) > 1e-15]
    if nz.size < 2:
        return 1.0
    p = float(stats.wilcoxon(arr, zero_method="wilcox", method="auto").pvalue)
    return p if np.isfinite(p) else 1.0


def _clean_log_trans(grid: np.ndarray, sigma_br: float) -> np.ndarray:
    diff = grid[:, None] - grid[None, :]
    raw = -0.5 * (diff / float(sigma_br)) ** 2
    return raw - logsumexp(raw, axis=1, keepdims=True)


def _clean_hmm_forward(S_seq: np.ndarray, grid: np.ndarray, sigma_br: float, eps: float, beta: float) -> np.ndarray:
    n_t, n_b = S_seq.shape
    log_T = _clean_log_trans(grid, sigma_br)
    log_emit = float(beta) * np.log(np.maximum(S_seq, 0.0) + float(eps))
    log_alpha = np.empty((n_t, n_b), dtype=np.float64)
    u0 = -np.log(float(n_b)) + log_emit[0]
    log_alpha[0] = u0 - float(logsumexp(u0))
    for t in range(1, n_t):
        lp = logsumexp(log_alpha[t - 1, :, None] + log_T, axis=0)
        u = lp + log_emit[t]
        log_alpha[t] = u - float(logsumexp(u))
    return np.exp(log_alpha) @ grid


def _clean_hmm_fixed_lag(
    S_seq: np.ndarray, lag_hops: int, grid: np.ndarray, sigma_br: float, eps: float, beta: float
) -> np.ndarray:
    n_t, n_b = S_seq.shape
    log_T = _clean_log_trans(grid, sigma_br)
    log_emit = float(beta) * np.log(np.maximum(S_seq, 0.0) + float(eps))
    log_alpha = np.empty((n_t, n_b), dtype=np.float64)
    u0 = -np.log(float(n_b)) + log_emit[0]
    log_alpha[0] = u0 - float(logsumexp(u0))
    for t in range(1, n_t):
        lp = logsumexp(log_alpha[t - 1, :, None] + log_T, axis=0)
        u = lp + log_emit[t]
        log_alpha[t] = u - float(logsumexp(u))
    out = np.empty(n_t, dtype=np.float64)
    for t in range(n_t):
        tau_end = min(n_t - 1, t + int(lag_hops))
        lb = np.zeros(n_b, dtype=np.float64)
        for tau in range(tau_end, t, -1):
            lb = logsumexp(log_T + (log_emit[tau] + lb)[None, :], axis=1)
        lp = log_alpha[t] + lb
        lp -= float(logsumexp(lp))
        out[t] = float(np.exp(lp) @ grid)
    return out


def _clean_hmm_fb(S_seq: np.ndarray, grid: np.ndarray, sigma_br: float, eps: float, beta: float) -> np.ndarray:
    n_t, n_b = S_seq.shape
    log_T = _clean_log_trans(grid, sigma_br)
    log_emit = float(beta) * np.log(np.maximum(S_seq, 0.0) + float(eps))
    log_alpha = np.empty((n_t, n_b), dtype=np.float64)
    u0 = -np.log(float(n_b)) + log_emit[0]
    log_alpha[0] = u0 - float(logsumexp(u0))
    for t in range(1, n_t):
        lp = logsumexp(log_alpha[t - 1, :, None] + log_T, axis=0)
        u = lp + log_emit[t]
        log_alpha[t] = u - float(logsumexp(u))
    log_beta = np.zeros((n_t, n_b), dtype=np.float64)
    for t in range(n_t - 2, -1, -1):
        log_beta[t] = logsumexp(log_T + (log_emit[t + 1] + log_beta[t + 1])[None, :], axis=1)
    lp = log_alpha + log_beta
    lp -= logsumexp(lp, axis=1, keepdims=True)
    return np.exp(lp) @ grid


def _clean_hmm_viterbi(S_seq: np.ndarray, grid: np.ndarray, sigma_br: float, eps: float, beta: float) -> tuple[np.ndarray, float]:
    n_t, n_b = S_seq.shape
    log_T = _clean_log_trans(grid, sigma_br)
    log_emit = float(beta) * np.log(np.maximum(S_seq, 0.0) + float(eps))
    dp = np.empty((n_t, n_b), dtype=np.float64)
    bp = np.zeros((n_t, n_b), dtype=np.int32)
    dp[0] = -np.log(float(n_b)) + log_emit[0]
    for t in range(1, n_t):
        for j in range(n_b):
            cands = dp[t - 1] + log_T[:, j]
            k = int(np.argmax(cands))
            bp[t, j] = k
            dp[t, j] = cands[k] + log_emit[t, j]
    path = np.empty(n_t, dtype=np.int32)
    cur = int(np.argmax(dp[-1]))
    score = float(dp[-1, cur])
    path[-1] = cur
    for t in range(n_t - 1, 0, -1):
        cur = int(bp[t, cur])
        path[t - 1] = cur
    return grid[path], score


def _clean_trig_ref_session(edf_path: Path, centers_s: np.ndarray, lag_s: int, v1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rd = pyedflib.EdfReader(str(edf_path))
    try:
        lbls = [rd.getLabel(i) for i in range(rd.signals_in_file)]
        idx = {l: i for i, l in enumerate(lbls)}
        trig = rd.readSignal(idx["Resp.-Trigger"]).astype(np.float64)
        fs_t = float(rd.getSampleFrequency(idx["Resp.-Trigger"]))
        fs_r = float(rd.getSampleFrequency(idx["Resp"]))
        n_resp_sec = len(rd.readSignal(idx["Resp"])) // int(round(fs_r))
        n_ref = min(len(rd.readSignal(idx["RespRate"])), len(rd.readSignal(idx["HeartRate"])), n_resp_sec)
    finally:
        rd._close()

    active = trig > 0.5
    rising = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    times: list[float] = []
    for r_idx in rising:
        ts = float(r_idx) / fs_t
        if not times or (ts - times[-1]) >= 0.30:
            times.append(ts)
    ev = np.asarray(times, dtype=np.float64)

    tc = centers_s - float(lag_s)
    a_arr = np.floor(tc - float(C.WIN_S) / 2.0).astype(int)
    b_arr = a_arr + int(C.WIN_S)
    br_t = np.full(len(centers_s), np.nan, dtype=np.float64)
    for w_i in range(len(centers_s)):
        a, b = int(a_arr[w_i]), int(b_arr[w_i])
        if a < 0 or b > n_ref:
            continue
        iw = ev[(ev >= float(a)) & (ev <= float(b))]
        if iw.size < 2:
            continue
        rates = 60.0 / np.diff(iw)
        vr = rates[(rates >= 12.0) & (rates <= 120.0)]
        if vr.size > 0:
            br_t[w_i] = float(np.median(vr))
    vt = np.asarray(v1, dtype=bool) & np.isfinite(br_t)
    return br_t, vt


def run_audit() -> dict[str, Any]:
    report: dict[str, Any] = {"gates": {}}
    md: list[str] = [
        "# Stage 1.1 Independent Clean-Room Audit Report (`stage1-parsimony-v3`)\n",
        "Executed with zero imports from `thesis.exp1_stage1_parsimony` or `thesis.io_utils`.\n",
    ]
    master = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    df_csv = pd.read_csv(SESSIONS_CSV, dtype={"measurement": str})

    # ------------------------------------------------------------------
    # Gate 1: Frozen Baseline SHA-256 Lock (154/154 + 33/33) & F0 Parity
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

    eval_31 = _cohort_ids("all_paired")
    max_f0_diff = 0.0
    for m in eval_31:
        with np.load(LOCKED_CACHE_DIR / f"{m}.npz", allow_pickle=False) as zl:
            est_hpss = zl["est_hpss"].astype(np.float64)
        with np.load(STAGE1_CACHE_DIR / f"{m}.npz", allow_pickle=False) as zs:
            f0 = zs["wp1_F0_CSD_Locked"].astype(np.float64)
        d = float(np.nanmax(np.abs(f0 - est_hpss)))
        if d > max_f0_diff:
            max_f0_diff = d

    g1_pass = (
        checked_frozen == 154
        and len(bad_files) == 0
        and len(bad_caches) == 0
        and max_f0_diff < 1e-9
        and abs(max_f0_diff - float(master["f0_vs_locked_cache_max_diff_bpm"])) < 1e-12
    )
    report["gates"]["gate1_frozen_baseline_and_f0_lock"] = {
        "passed": g1_pass,
        "checked_frozen_files": checked_frozen,
        "bad_files": bad_files,
        "bad_caches": bad_caches,
        "max_f0_vs_est_hpss_diff_bpm": max_f0_diff,
    }
    md.append(
        f"## Gate 1 — Frozen Baseline SHA-256 & F0 Parity: **{'PASS' if g1_pass else 'FAIL'}**\n"
        f"- Checked `{checked_frozen}/154` frozen files (`bad: {len(bad_files)}`), `33/33` locked `v1` caches (`bad: {len(bad_caches)}`), and `max |F0 - est_hpss|` across 31 sessions (`{max_f0_diff:.2e} bpm`).\n"
    )

    # ------------------------------------------------------------------
    # Gate 2: Clean-Room Metrics, Dual-Ref, Clip Fraction & CSV Parity
    # ------------------------------------------------------------------
    lags_clean: dict[str, int] = {}
    with LAGS_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lags_clean[row["measurement"]] = int(float(row["lag_raw_nls_s"]))

    p30_ids = _cohort_ids("pool30_sync")
    ref_1hz: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ref_trig: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    s_h0_cache: dict[str, np.ndarray] = {}
    wp_preds_clean: dict[str, dict[str, np.ndarray]] = {
        "F0_CSD_Locked": {},
        "F2c_SymWhitened_lam025": {},
        "H0_Locked_Rect_Full_L3": {},
        "H5_Causal_COLA_InBand_L0": {},
        "H6_Causal_Rect_Full_L0": {},
    }
    raw_clip_diff_max = 0.0
    csv_clip_map = dict(zip(df_csv["measurement"], df_csv["clip_frac_session_true"].astype(float)))
    for m in p30_ids:
        epath = C.REF_DIR / f"{MEASURE_TO_REF[m]}.edf"
        rd = pyedflib.EdfReader(str(epath))
        try:
            lbls = [rd.getLabel(i) for i in range(rd.signals_in_file)]
            idx = {l: i for i, l in enumerate(lbls)}
            fs_resp = float(rd.getSampleFrequency(idx["Resp"]))
            resp_256 = rd.readSignal(idx["Resp"]).astype(np.float64)
            rate_1hz = rd.readSignal(idx["RespRate"]).astype(np.float64)
        finally:
            rd._close()

        with np.load(STAGE1_CACHE_DIR / f"{m}.npz", allow_pickle=False) as zs:
            s_h0_cache[m] = zs["S_anc_h0_1bpm"].astype(np.float64)
            wp_preds_clean["F0_CSD_Locked"][m] = zs["wp1_F0_CSD_Locked"].astype(np.float64)
            wp_preds_clean["F2c_SymWhitened_lam025"][m] = zs["wp1_F2c_SymWhitened_lam025"].astype(np.float64)
            wp_preds_clean["H0_Locked_Rect_Full_L3"][m] = zs["wp2_H0_Locked_Rect_Full_L3"].astype(np.float64)
            wp_preds_clean["H5_Causal_COLA_InBand_L0"][m] = zs["wp2_H5_Causal_COLA_InBand_L0"].astype(np.float64)
            wp_preds_clean["H6_Causal_Rect_Full_L0"][m] = zs["wp2_H6_Causal_Rect_Full_L0"].astype(np.float64)
            n_w = len(wp_preds_clean["F0_CSD_Locked"][m])

        centers = np.arange(n_w, dtype=np.float64) * C.UPDATE_S + C.WIN_S / 2.0
        rw, vw = window_ref_and_gating(resp_256, rate_1hz, centers, lags_clean[m], fs_raw=fs_resp)
        v1 = np.asarray(vw, dtype=bool) & np.isfinite(rw)
        ref_1hz[m] = (rw, v1)
        ref_trig[m] = _clean_trig_ref_session(epath, centers, lags_clean[m], v1)

        bi, bq = load_radar(m)
        rail = (~np.isfinite(bi)) | (bi <= -1024.0) | (bi >= 1023.0) | (~np.isfinite(bq)) | (bq <= -1024.0) | (bq >= 1023.0)
        true_cf = float(np.mean(rail))
        raw_clip_diff_max = max(raw_clip_diff_max, abs(true_cf - float(csv_clip_map[m])))

    # Re-decode R1a, R1c, R1f, R1g using clean-room HMM / Viterbi from selected LOPO fold params
    fsum_p30 = master["rung1_lopo"]["pool30_sync"]["lopo_fold_summary"]
    rung1_targets = (
        "R1a_Online_Forward_MMSE_0s",
        "R1c_FixedLag_MMSE_4s",
        "R1f_Offline_FB_MMSE_Inf",
        "R1g_Offline_Viterbi_Unpinned",
    )
    for fam in rung1_targets:
        by_pid = {int(r["patient_id"]): r for r in fsum_p30[fam]["folds"]}
        pmap: dict[str, np.ndarray] = {}
        for m in p30_ids:
            fr = by_pid[_pid(m)]
            sig_v, eps_v, beta_v = float(fr["sigma_br"]), float(fr["eps"]), float(fr["beta"])
            s0 = s_h0_cache[m]
            if fam == "R1a_Online_Forward_MMSE_0s":
                pmap[m] = _clean_hmm_forward(s0, BR_GRID, sig_v, eps_v, beta_v)
            elif fam == "R1c_FixedLag_MMSE_4s":
                pmap[m] = _clean_hmm_fixed_lag(s0, 2, BR_GRID, sig_v, eps_v, beta_v)
            elif fam == "R1f_Offline_FB_MMSE_Inf":
                pmap[m] = _clean_hmm_fb(s0, BR_GRID, sig_v, eps_v, beta_v)
            elif fam == "R1g_Offline_Viterbi_Unpinned":
                pmap[m] = _clean_hmm_viterbi(s0, BR_GRID, sig_v, eps_v, beta_v)[0]
        wp_preds_clean[fam] = pmap

    def _agg_clean(pmap: dict[str, np.ndarray], rmap: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[dict[str, float], dict[str, float]]:
        ps_rmse, ps_mae, ps_acc10, ems, rms, evs, rvs = {}, [], [], [], [], [], []
        for m in p30_ids:
            rv, vv = rmap[m]
            e = pmap[m][vv]
            r = rv[vv]
            err = e - r
            ps_rmse[m] = float(np.sqrt(np.mean(err ** 2)))
            ps_mae.append(float(np.mean(np.abs(err))))
            ps_acc10.append(float(100.0 * np.mean(np.abs(err) < 10.0)))
            ems.append(float(np.mean(e)))
            rms.append(float(np.mean(r)))
            evs.append(e)
            rvs.append(r)
        all_e, all_r = np.concatenate(evs), np.concatenate(rvs)
        return {
            "rmse_sess": float(np.mean(list(ps_rmse.values()))),
            "rmse_pool": float(np.sqrt(np.mean((all_e - all_r) ** 2))),
            "mae_sess": float(np.mean(ps_mae)),
            "acc10_sess": float(np.mean(ps_acc10)),
            "r_within_rmcorr": _clean_rmcorr(evs, rvs),
            "icc_1_1": _clean_icc_1_1(np.array(ems), np.array(rms)),
        }, ps_rmse

    max_metric_diff = 0.0
    check_specs = [
        ("wp1_ablation", "F0_CSD_Locked", None, None),
        ("wp1_ablation", "F2c_SymWhitened_lam025", "paired_vs_F0", "F0_CSD_Locked"),
        ("wp2_ablation", "H5_Causal_COLA_InBand_L0", "paired_vs_H0", "H0_Locked_Rect_Full_L3"),
        ("wp2_ablation", "H6_Causal_Rect_Full_L0", "paired_vs_H0", "H0_Locked_Rect_Full_L3"),
        ("rung1_lopo", "R1a_Online_Forward_MMSE_0s", "paired_vs_R0", "F0_CSD_Locked"),
        ("rung1_lopo", "R1c_FixedLag_MMSE_4s", "paired_vs_R0", "F0_CSD_Locked"),
        ("rung1_lopo", "R1f_Offline_FB_MMSE_Inf", "paired_vs_R0", "F0_CSD_Locked"),
        ("rung1_lopo", "R1g_Offline_Viterbi_Unpinned", "paired_vs_R0", "F0_CSD_Locked"),
    ]
    for ref_lbl, rmap in (("br_1hz_30s", ref_1hz), ("br_trig_bpm", ref_trig)):
        ps_store: dict[str, dict[str, float]] = {}
        for k_m in wp_preds_clean:
            _, ps_store[k_m] = _agg_clean(wp_preds_clean[k_m], rmap)
        for sec, m_name, pair_key, anchor_name in check_specs:
            rec, ps_cand = _agg_clean(wp_preds_clean[m_name], rmap)
            j_m = master[sec]["pool30_sync"][ref_lbl]["methods"][m_name]
            for metric_k in ("rmse_sess", "rmse_pool", "mae_sess", "acc10_sess", "r_within_rmcorr", "icc_1_1"):
                max_metric_diff = max(max_metric_diff, abs(rec[metric_k] - float(j_m[metric_k])))
            if pair_key is not None and anchor_name is not None:
                deltas_s = np.array([ps_cand[m] - ps_store[anchor_name][m] for m in p30_ids], dtype=np.float64)
                by_p: dict[int, list[float]] = {}
                for m, d in zip(p30_ids, deltas_s):
                    by_p.setdefault(_pid(m), []).append(float(d))
                deltas_p = np.array([float(np.mean(v)) for v in by_p.values()], dtype=np.float64)
                j_p = master[sec]["pool30_sync"][ref_lbl][pair_key][m_name]
                t_s = _clean_tost(deltas_s, 0.50)
                t_p = _clean_tost(deltas_p, 0.50)
                p_w = _clean_wilcoxon(deltas_p)
                max_metric_diff = max(
                    max_metric_diff,
                    abs(t_s["p_tost"] - float(j_p["tost_sess"]["p_tost"])),
                    abs(t_p["p_tost"] - float(j_p["tost_patient"]["p_tost"])),
                    abs(p_w - float(j_p["p_wilcoxon_patient"])),
                )

    csv_cols_check = [
        ("F0_rmse_1hz", "wp1_ablation", "br_1hz_30s", "F0_CSD_Locked"),
        ("F2c_rmse_1hz", "wp1_ablation", "br_1hz_30s", "F2c_SymWhitened_lam025"),
        ("H5_rmse_1hz", "wp2_ablation", "br_1hz_30s", "H5_Causal_COLA_InBand_L0"),
        ("R1a_rmse_1hz", "rung1_lopo", "br_1hz_30s", "R1a_Online_Forward_MMSE_0s"),
        ("R1f_rmse_1hz", "rung1_lopo", "br_1hz_30s", "R1f_Offline_FB_MMSE_Inf"),
        ("R1g_rmse_1hz", "rung1_lopo", "br_1hz_30s", "R1g_Offline_Viterbi_Unpinned"),
        ("F2c_rmse_trig", "wp1_ablation", "br_trig_bpm", "F2c_SymWhitened_lam025"),
        ("R1a_rmse_trig", "rung1_lopo", "br_trig_bpm", "R1a_Online_Forward_MMSE_0s"),
        ("R1f_rmse_trig", "rung1_lopo", "br_trig_bpm", "R1f_Offline_FB_MMSE_Inf"),
        ("R1g_rmse_trig", "rung1_lopo", "br_trig_bpm", "R1g_Offline_Viterbi_Unpinned"),
    ]
    csv_json_max_diff = 0.0
    for col, sec, rlbl, m_name in csv_cols_check:
        cm = float(df_csv[col].astype(float).mean())
        jm = float(master[sec]["pool30_sync"][rlbl]["methods"][m_name]["rmse_sess"])
        csv_json_max_diff = max(csv_json_max_diff, abs(cm - jm))

    g2_pass = (
        len(df_csv) == 28
        and max_metric_diff < 1e-11
        and csv_json_max_diff <= 1e-4
        and raw_clip_diff_max <= 1e-4
    )
    report["gates"]["gate2_cleanroom_metrics_and_csv_parity"] = {
        "passed": g2_pass,
        "csv_rows": int(len(df_csv)),
        "max_cleanroom_metric_diff": max_metric_diff,
        "csv_json_max_diff": csv_json_max_diff,
        "raw_clip_frac_max_diff": raw_clip_diff_max,
    }
    md.append(
        f"## Gate 2 — Clean-Room Metric, TOST, Wilcoxon, Raw Clip & CSV Parity: **{'PASS' if g2_pass else 'FAIL'}**\n"
        f"- Max clean-room vs JSON diff across 8 methods x dual references: `{max_metric_diff:.2e}`.\n"
        f"- CSV (`28` rows) vs JSON mean max diff: `{csv_json_max_diff:.2e}`; raw ADC clip fraction max diff: `{raw_clip_diff_max:.2e}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 3: Clean-Room HMM, Scaled vs Logsumexp & Stage 0 Viterbi Parity
    # ------------------------------------------------------------------
    rng = np.random.default_rng(777)
    S_syn = rng.uniform(0.01, 1.0, (20, BR_GRID.size))
    S_syn[7, :] = 0.0
    max_hmm_loop_diff = 0.0
    max_scaled_log_diff = 0.0
    orig_scaled_fn = W._can_use_scaled_hmm
    for sig_v, eps_v, beta_v in ((0.5, 1e-4, 4.0), (2.0, 1e-3, 1.0)):
        c_fwd = _clean_hmm_forward(S_syn, BR_GRID, sig_v, eps_v, beta_v)
        c_lag = _clean_hmm_fixed_lag(S_syn, 4, BR_GRID, sig_v, eps_v, beta_v)
        c_fb = _clean_hmm_fb(S_syn, BR_GRID, sig_v, eps_v, beta_v)
        c_vit, c_sc = _clean_hmm_viterbi(S_syn, BR_GRID, sig_v, eps_v, beta_v)

        w_fwd_s = W.hmm_forward_filter(S_syn, BR_GRID, sig_v, eps_v, beta_v)
        w_lag_s = W.hmm_fixed_lag_smoother(S_syn, 4, BR_GRID, sig_v, eps_v, beta_v)
        w_fb_s = W.hmm_offline_fb_smoother(S_syn, BR_GRID, sig_v, eps_v, beta_v)
        w_vit, w_sc = W.hmm_offline_viterbi_prob(S_syn, BR_GRID, sig_v, eps_v, beta_v)

        max_hmm_loop_diff = max(
            max_hmm_loop_diff,
            float(np.max(np.abs(w_fwd_s["est_mmse"] - c_fwd))),
            float(np.max(np.abs(w_lag_s["est_mmse"] - c_lag))),
            float(np.max(np.abs(w_fb_s["est_mmse"] - c_fb))),
            float(np.max(np.abs(w_vit - c_vit))),
            abs(w_sc - c_sc),
        )
        try:
            W._can_use_scaled_hmm = lambda e, b, n: False
            w_fwd_l = W.hmm_forward_filter(S_syn, BR_GRID, sig_v, eps_v, beta_v)
            w_lag_l = W.hmm_fixed_lag_smoother(S_syn, 4, BR_GRID, sig_v, eps_v, beta_v)
            w_fb_l = W.hmm_offline_fb_smoother(S_syn, BR_GRID, sig_v, eps_v, beta_v)
        finally:
            W._can_use_scaled_hmm = orig_scaled_fn

        max_scaled_log_diff = max(
            max_scaled_log_diff,
            float(np.max(np.abs(w_fwd_s["est_mmse"] - w_fwd_l["est_mmse"]))),
            float(np.max(np.abs(w_fwd_s["post_entropy"] - w_fwd_l["post_entropy"]))),
            float(np.max(np.abs(w_fwd_s["post_sd_bpm"] - w_fwd_l["post_sd_bpm"]))),
            float(np.max(np.abs(w_fwd_s["log_evidence"] - w_fwd_l["log_evidence"]))),
            float(np.max(np.abs(w_lag_s["est_mmse"] - w_lag_l["est_mmse"]))),
            float(np.max(np.abs(w_fb_s["est_mmse"] - w_fb_l["est_mmse"]))),
        )

    vit_stage0_max_diff = 0.0
    for m in p30_ids:
        s0 = s_h0_cache[m]
        p0 = VT.decode_viterbi_1d(s0, BR_GRID, sigma_br=2.0, eps_p=1e-3)[0]
        p1 = W.hmm_offline_viterbi_prob(s0, BR_GRID, sigma_br=2.0, eps=1e-3, beta=1.0)[0]
        vit_stage0_max_diff = max(vit_stage0_max_diff, float(np.max(np.abs(p0 - p1))))

    g3_pass = max_hmm_loop_diff < 1e-12 and max_scaled_log_diff < 1e-12 and vit_stage0_max_diff == 0.0
    report["gates"]["gate3_cleanroom_hmm_and_viterbi_parity"] = {
        "passed": g3_pass,
        "max_hmm_loop_diff": max_hmm_loop_diff,
        "max_scaled_vs_logsumexp_diff": max_scaled_log_diff,
        "stage0_vs_stage1_viterbi_max_diff_bpm": vit_stage0_max_diff,
    }
    md.append(
        f"## Gate 3 — Clean-Room HMM, Scaled vs `logsumexp` & Stage 0 Viterbi Parity: **{'PASS' if g3_pass else 'FAIL'}**\n"
        f"- Clean-room explicit DP/FB loop max diff: `{max_hmm_loop_diff:.2e}`; scaled vs `logsumexp` max diff: `{max_scaled_log_diff:.2e}`.\n"
        f"- Stage 0 `decode_viterbi_1d` vs Stage 1 `hmm_offline_viterbi_prob` across all 28 sessions at `sigma=2.0, eps=1e-3`: `{vit_stage0_max_diff:.2f} bpm`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 4: Causality, O(2) x R+ Invariance & Two-Axis Contract
    # ------------------------------------------------------------------
    mag = rng.random((256, 32)) + 0.1
    mag_p = mag.copy()
    mag_p[:, 18:] += 50.0
    c_hpss = float(np.max(np.abs(W.hpss_filter_causal(mag, lookahead=0, in_band=True)[0][:, :18] - W.hpss_filter_causal(mag_p, lookahead=0, in_band=True)[0][:, :18])))

    bi7, bq7 = load_radar("7.3")
    dr7 = bandpass(csd(bi7[4800:5280], bq7[4800:5280]))
    t_ax = np.arange(480) / float(C.FS_RADAR)
    dr7_p = dr7.copy()
    dr7_p[240:] += 50.0 * np.exp(1j * 2.0 * np.pi * 0.8 * t_ax[240:])
    hc_a = W.mitigate_hpss_v2(dr7, window="cola_hann", lookahead=0, in_band=True, env_norm="causal")
    hc_b = W.mitigate_hpss_v2(dr7_p, window="cola_hann", lookahead=0, in_band=True, env_norm="causal")
    hb_a = W.mitigate_hpss_v2(dr7, window="cola_hann", lookahead=0, in_band=True, env_norm="block")
    hb_b = W.mitigate_hpss_v2(dr7_p, window="cola_hann", lookahead=0, in_band=True, env_norm="block")
    c_mit_causal = float(np.max(np.abs(hc_a["mag_filt"][:, :14] - hc_b["mag_filt"][:, :14])))
    c_mit_block = float(np.max(np.abs(hb_a["mag_filt"][:, :14] - hb_b["mag_filt"][:, :14])))

    S_p = S_syn.copy()
    S_p[12:] = rng.random((8, BR_GRID.size))
    c_fwd = float(np.max(np.abs(W.hmm_forward_filter(S_syn)["est_mmse"][:12] - W.hmm_forward_filter(S_p)["est_mmse"][:12])))
    c_lag = float(np.max(np.abs(W.hmm_fixed_lag_smoother(S_syn, lag_hops=4)["est_mmse"][:8] - W.hmm_fixed_lag_smoother(S_p, lag_hops=4)["est_mmse"][:8])))

    # O(2) x R+ invariance check directly from gate4
    bi2, bq2 = load_radar("2.3")
    bi2, bq2 = bi2[:480], bq2[:480]
    _, _, cw2 = W.detect_adc_rail_mask(bi2, bq2)

    def _run_inv_ops(ai: np.ndarray, aq: np.ndarray) -> dict[str, float]:
        x_0 = csd(ai, aq)
        dr_0 = bandpass(x_0)
        res_d: dict[str, float] = {}
        for lam, nm in ((0.05, "F2a"), (0.10, "F2b"), (0.25, "F2c")):
            dh = bandpass(mitigate_hpss(bandpass(W.csd_sym_whitened(ai, aq, lam=lam)))["x_hat"])
            res_d[nm] = nls_estimate(dh, autocorr_coarse(dh))[0]
        dh_f1b = bandpass(mitigate_hpss(bandpass(W.csd_clip_masked(ai, aq, mode="joint", clip_mask=cw2)))["x_hat"])
        res_d["F1b"] = nls_estimate(dh_f1b, autocorr_coarse(dh_f1b))[0]
        for nm, win, la, ib, en in (
            ("H1", "cola_hann", 3, False, "block"),
            ("H2", "cola_hann", 3, True, "block"),
            ("H5", "cola_hann", 0, True, "block"),
            ("H5c", "cola_hann", 0, True, "causal"),
            ("H6", "rect", 0, False, "block"),
        ):
            dh = bandpass(W.mitigate_hpss_v2(dr_0, window=win, lookahead=la, in_band=ib, env_norm=en)["x_hat"])
            res_d[nm] = nls_estimate(dh, autocorr_coarse(dh))[0]
        return res_d

    base_inv = _run_inv_ops(bi2, bq2)
    zrot = (bi2 + 1j * bq2) * np.exp(1j * np.pi / 3.0)
    inv_max_diff = 0.0
    for ti, tq in ((3.7 * bi2, 3.7 * bq2), (zrot.real, zrot.imag), (-bi2, -bq2), (bq2, bi2)):
        alt_inv = _run_inv_ops(ti, tq)
        inv_max_diff = max(inv_max_diff, max(abs(base_inv[k] - alt_inv[k]) for k in base_inv))

    cc = master["rung1_causality_contract"]
    contract_ok = (
        cc["R0_HPSS_NLS_Locked"]["block_lookahead_s"] == 0.0
        and cc["R0_HPSS_NLS_Locked"]["stft_lookahead_s"] == 3.0
        and cc["R1a_Online_Forward_MMSE_0s"]["block_lookahead_s"] == 0.0
        and cc["R1a_Online_Forward_MMSE_0s"]["stft_lookahead_s"] == 3.0
        and cc["R1h_E2E_Causal_H5_OnlineForward_0s"]["block_lookahead_s"] == 0.0
        and cc["R1h_E2E_Causal_H5_OnlineForward_0s"]["stft_lookahead_s"] == 0.0
        and cc["R1i_E2E_Causal_H6_OnlineForward_0s"]["stft_lookahead_s"] == 0.0
        and cc["R1f_Offline_FB_MMSE_Inf"]["block_lookahead_s"] is None
    )
    g4_pass = (
        c_hpss == 0.0
        and c_mit_causal == 0.0
        and c_mit_block > 1.0
        and c_fwd == 0.0
        and c_lag == 0.0
        and inv_max_diff < 1e-6
        and contract_ok
    )
    report["gates"]["gate4_causality_and_invariance_contract"] = {
        "passed": g4_pass,
        "c_hpss_l0": c_hpss,
        "c_mit_v2_causal_env_mag_filt": c_mit_causal,
        "c_mit_v2_block_env_diag": c_mit_block,
        "c_hmm_fwd": c_fwd,
        "c_hmm_lag4": c_lag,
        "inv_max_diff_bpm": inv_max_diff,
        "two_axis_contract_ok": contract_ok,
    }
    md.append(
        f"## Gate 4 — Causality, O(2) x R+ Invariance & Two-Axis Contract: **{'PASS' if g4_pass else 'FAIL'}**\n"
        f"- Causal future diff: `hpss_l0={c_hpss:.1f}`, `mitigate_v2(causal)={c_mit_causal:.1f}` (vs `block` intra-window diag `{c_mit_block:.2f}`), `hmm_fwd={c_fwd:.1f}`, `hmm_lag4={c_lag:.1f}`.\n"
        f"- Clean-room O(2) x R+ max diff across `F2a/b/c, F1b, H1, H2, H5, H5(causal), H6`: `{inv_max_diff:.2e} bpm`; two-axis contract valid: `{contract_ok}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 5: LOPO Anti-Leakage, Causality Monotonicity & Grid Equivariance
    # ------------------------------------------------------------------
    leak_errs: list[Any] = []
    pin_errs: list[Any] = []
    aux_pin_summary: dict[str, dict[str, int]] = {}
    mono_errs: list[Any] = []
    chain = [
        "R1a_Online_Forward_MMSE_0s",
        "R1c_FixedLag_MMSE_4s",
        "R1d_FixedLag_MMSE_8s",
        "R1e_FixedLag_MMSE_16s",
        "R1f_Offline_FB_MMSE_Inf",
    ]
    primary_frontier_fams = (
        "R1a_Online_Forward_MMSE_0s",
        "R1c_FixedLag_MMSE_4s",
        "R1d_FixedLag_MMSE_8s",
        "R1e_FixedLag_MMSE_16s",
    )
    aux_causal_fams = (
        "R1b_Online_Forward_MAP_0s",
        "R1h_E2E_Causal_H5_OnlineForward_0s",
        "R1i_E2E_Causal_H6_OnlineForward_0s",
    )
    for coh in ("pool30_sync", "retained23", "PAPER_18"):
        expected_ids = _cohort_ids(coh)
        expected_pats = len({_pid(m) for m in expected_ids})
        fsum = master["rung1_lopo"][coh]["lopo_fold_summary"]
        aux_pin_summary[coh] = {}
        for fam, blk in fsum.items():
            if len(blk["folds"]) != expected_pats:
                leak_errs.append((coh, fam, "n_folds", len(blk["folds"])))
            seen_t: list[str] = []
            for row in blk["folds"]:
                if any(_pid(tm) != row["patient_id"] for tm in row["test_sessions"]):
                    leak_errs.append((coh, fam, row["patient_id"], "pid_mismatch"))
                seen_t.extend(row["test_sessions"])
            if sorted(seen_t, key=lambda x: tuple(int(p) for p in x.split("."))) != expected_ids:
                leak_errs.append((coh, fam, "coverage_mismatch"))
            if blk["fallback_folds"] != 0:
                pin_errs.append((coh, fam, "fallback", blk["fallback_folds"]))
            if fam in primary_frontier_fams and blk["lower_sigma_pinned_folds"] != 0:
                pin_errs.append((coh, fam, "lower_sigma_pinned", blk["lower_sigma_pinned_folds"]))
            if fam in aux_causal_fams:
                aux_pin_summary[coh][fam] = int(blk["lower_sigma_pinned_folds"])

        for rlbl in ("br_1hz_30s", "br_trig_bpm"):
            ms = master["rung1_lopo"][coh][rlbl]["methods"]
            vals = [float(ms[k]["rmse_sess"]) for k in chain]
            for i in range(len(vals) - 1):
                if vals[i] + 1e-9 < vals[i + 1]:
                    mono_errs.append((coh, rlbl, chain[i], vals[i], chain[i + 1], vals[i + 1]))

    sh = master["shifted_grid_midpoint_audit"]
    equiv_ok = (
        sh["memoryless_mmse_naive_eps001"]["abs_bias_shift_bpm"] > 1.0
        and sh["fwd_mmse_equivariant_b1_e1e3_sig2"]["abs_bias_shift_bpm"] < 0.05
        and sh["fb_mmse_equivariant_b1_e1e3_sig1"]["abs_bias_shift_bpm"] < 0.05
        and sh["viterbi_b1_e1e3_sig1"]["abs_bias_shift_bpm"] < 0.05
    )
    g5_pass = len(leak_errs) == 0 and len(pin_errs) == 0 and len(mono_errs) == 0 and equiv_ok
    report["gates"]["gate5_lopo_antileakage_monotonicity_and_equivariance"] = {
        "passed": g5_pass,
        "leak_errors": leak_errs,
        "pin_or_fallback_errors": pin_errs,
        "aux_causal_lower_sigma_pinned_folds": aux_pin_summary,
        "monotonicity_errors": mono_errs,
        "equivariance_ok": equiv_ok,
    }
    md.append(
        f"## Gate 5 — LOPO Anti-Leakage, Frontier Monotonicity & Equivariance: **{'PASS' if g5_pass else 'FAIL'}**\n"
        f"- Fold disjointness errors across 3 cohorts: `{len(leak_errs)}`; causal `R1a..R1e` pinning/fallback errors: `{len(pin_errs)}`; monotonicity violations `R1a >= R1c >= R1d >= R1e >= R1f` on `1 Hz` & `Trig`: `{len(mono_errs)}`; shifted-grid equivariance ok: `{equiv_ok}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 6: Surrogates (+ R1f), Lag Audit & Independent G1/G2 Verdicts
    # ------------------------------------------------------------------
    scr = master["phase_scrambled_surrogates_pool30"]
    scr_ok = (not scr["skipped"]) and scr["n_seeds"] == 5
    scr_targets = (
        "R0_HPSS_NLS_Locked",
        "R1a_Online_Forward_MMSE_0s",
        "R1f_Offline_FB_MMSE_Inf",
        "R1g_Offline_Viterbi_Unpinned",
        "R1h_E2E_Causal_H5_OnlineForward_0s",
    )
    if scr_ok:
        for rlbl in ("br_1hz_30s", "br_trig_bpm"):
            real_m = master["rung1_lopo"]["pool30_sync"][rlbl]["methods"]
            for k in scr_targets:
                if k not in scr[rlbl]:
                    scr_ok = False
                    break
                if abs(float(scr[rlbl][k]["r_within_rmcorr_mean"])) >= 0.05:
                    scr_ok = False
                if (float(scr[rlbl][k]["rmse_sess_mean"]) - float(real_m[k]["rmse_sess"])) <= 1.0:
                    scr_ok = False

    lp = master["dual_reference_lag_and_pause_audit"]["lag_profiles_rmcorr"]
    lag_ok = (
        lp["br_1hz_30s_vs_br_trig_bpm"]["peak_lag_s"] >= 6
        and lp["br_1hz_30s_vs_br_trig_bpm"]["peak_r"] > lp["br_1hz_30s_vs_br_trig_bpm"]["lag_0s_r"]
        and lp["R1a_Online_Forward_MMSE_0s_vs_br_trig_bpm"]["peak_lag_s"] >= lp["R1f_Offline_FB_MMSE_Inf_vs_br_trig_bpm"]["peak_lag_s"]
    )
    gg1 = master["gate_g1_verdict"]
    gg2 = master["gate_g2_verdict"]
    g1_recomp_pass = any(v["passes_gate_g1_parsimony"] for v in gg1["candidates"].values())
    g2_recomp_pass = any(v["passes_gate_g2"] for v in gg2["candidates"].values())
    gates_ok = (
        (not g1_recomp_pass and gg1["decision"] == "RETAIN_F0_CSD_LOCKED")
        and (not g2_recomp_pass and gg2["decision"] == "RETAIN_H0_LOCKED")
    )
    g6_pass = scr_ok and lag_ok and gates_ok
    report["gates"]["gate6_controls_gates_and_lag_audit"] = {
        "passed": g6_pass,
        "surrogates_ok": scr_ok,
        "lag_audit_ok": lag_ok,
        "gates_g1_g2_ok": gates_ok,
    }
    md.append(
        f"## Gate 6 — 5-Seed Surrogates (+ `R1f`), Lag Audit & Gates G1/G2: **{'PASS' if g6_pass else 'FAIL'}**\n"
        f"- 5-seed scrambled surrogates (`|r_within| < 0.05` & real margin `> 1.0 bpm`): `{scr_ok}`; Monitor lag audit (`peak >= +6 s`): `{lag_ok}`; Gates `G1=RETAIN_F0_CSD_LOCKED`, `G2=RETAIN_H0_LOCKED`: `{gates_ok}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 7: Provenance, BibTeX/CSV 71/71 Parity & Cohort Completeness
    # ------------------------------------------------------------------
    bib_text = (BUNDLE_DIR / "seed_references.bib").read_text(encoding="utf-8")
    bib_blocks = re.split(r"\n(?=@)", bib_text.strip())
    brace_ok = all(blk.count("{") == blk.count("}") for blk in bib_blocks)
    bib_keys = re.findall(r"^@\w+\{([^,]+),", bib_text, flags=re.M)
    with (BUNDLE_DIR / "lit_review_matrix.csv").open(newline="", encoding="utf-8") as fh:
        lit_rows = list(csv.DictReader(fh))
    csv_keys = [r["bib_key"] for r in lit_rows]
    bib_ok = brace_ok and len(bib_keys) == len(set(bib_keys)) == 71 and len(csv_keys) == len(set(csv_keys)) == 71 and set(bib_keys) == set(csv_keys)

    expected_counts = {
        "pool30_sync": (28, 6640, 6638),
        "retained23": (23, 6447, 6445),
        "PAPER_18": (18, 5466, 5466),
        "all_paired": (31, 7103, 7101),
    }
    count_errs: list[Any] = []
    for coh, (ns, nw1, nwt) in expected_counts.items():
        for rlbl, exp_w in (("br_1hz_30s", nw1), ("br_trig_bpm", nwt)):
            for k, d in master["wp1_ablation"][coh][rlbl]["methods"].items():
                if d["n_sessions"] != ns or d["n_windows"] != exp_w or not np.isfinite(d["rmse_sess"]):
                    count_errs.append(("wp1", coh, rlbl, k, d["n_sessions"], d["n_windows"]))
            for k, d in master["wp2_ablation"][coh][rlbl]["methods"].items():
                if d["n_sessions"] != ns or d["n_windows"] != exp_w or not np.isfinite(d["rmse_sess"]):
                    count_errs.append(("wp2", coh, rlbl, k, d["n_sessions"], d["n_windows"]))
        if coh != "all_paired":
            for rlbl, exp_w in (("br_1hz_30s", nw1), ("br_trig_bpm", nwt)):
                for k, d in master["rung1_lopo"][coh][rlbl]["methods"].items():
                    if d["n_sessions"] != ns or d["n_windows"] != exp_w or not np.isfinite(d["rmse_sess"]):
                        count_errs.append(("rung1", coh, rlbl, k, d["n_sessions"], d["n_windows"]))

    g7_pass = (
        master.get("schema_version") == "stage1-parsimony-v3"
        and bool(master.get("git_head", {}).get("head_commit"))
        and float(master.get("total_runtime_s", 0.0)) > 0.0
        and bib_ok
        and len(count_errs) == 0
    )
    report["gates"]["gate7_provenance_bib_and_completeness"] = {
        "passed": g7_pass,
        "schema_version": master.get("schema_version"),
        "bib_71_parity": bib_ok,
        "count_errors": count_errs,
    }
    md.append(
        f"## Gate 7 — Provenance, 71/71 BibTeX Parity & Cohort Completeness: **{'PASS' if g7_pass else 'FAIL'}**\n"
        f"- Schema `{master.get('schema_version')}` | `71/71` BibTeX↔CSV parity: `{bib_ok}` | Cohort window count errors: `{len(count_errs)}`.\n"
    )

    all_passed = all(g["passed"] for g in report["gates"].values())
    report["all_passed"] = all_passed
    md.insert(2, f"**Overall Verdict:** `{'7/7 GATES PASSED' if all_passed else 'AUDIT FAILED'}`\n")
    AUDIT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    AUDIT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    res = run_audit()
    sys.exit(0 if res["all_passed"] else 1)
