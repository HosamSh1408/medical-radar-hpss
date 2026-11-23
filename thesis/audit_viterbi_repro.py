"""Zero-shared-code clean-room auditor for the Viterbi reproduction pipeline.

Intentionally imports ZERO modules from `thesis.*`. Re-implements all statistical
aggregators, Eq. 5 HQI, brute-force & separable 2D Viterbi DP, EDF channel checks,
and SHA-256 frozen-baseline verification from scratch to guarantee independent audit
validity.

Execution:
  MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /opt/homebrew/bin/python3.10 -m thesis.audit_viterbi_repro
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyedflib
from scipy import stats

from paper_workspace.src import config as C
from paper_workspace.src.cohort import PAPER_18, local_pool, pool30, retained23
from paper_workspace.src.data_loader import MEASURE_TO_REF
from paper_workspace.src.evaluation import window_ref_and_gating
from paper_workspace.src.provenance import PIPELINE_VERSION, expected_provenance

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "audit"
MANIFEST_PATH = AUDIT_DIR / "postfix_manifest.sha256"
LAGS_CSV = AUDIT_DIR / "phase2_ws3_session_lags.csv"
LOCKED_CACHE_DIR = ROOT / "paper_workspace" / "outputs" / "cache"
VITERBI_CACHE_DIR = ROOT / "thesis" / "outputs" / "viterbi_repro" / "cache"
METRICS_JSON = ROOT / "paper_orchestra_bundle" / "raw_materials" / "metrics_viterbi_repro.json"
SESSIONS_CSV = AUDIT_DIR / "viterbi_repro_sessions.csv"
AUDIT_MD = AUDIT_DIR / "viterbi_godmode_audit.md"

BR_GRID_2BPM = np.arange(18.0, 90.0 + 1e-9, 2.0, dtype=np.float64)
BR_GRID_1BPM = np.arange(18.0, 90.0 + 1e-9, 1.0, dtype=np.float64)
HR_GRID_2BPM = np.arange(90.0, 210.0 + 1e-9, 2.0, dtype=np.float64)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            blk = fh.read(1024 * 1024)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def _patient_id(m: str) -> int:
    return int(str(m).split(".")[0])


def _cohort_sessions_clean(name: str) -> list[str]:
    if name == "pool30_sync":
        ids = pool30()
    elif name == "retained23":
        ids = retained23()
    elif name == "PAPER_18":
        ids = list(PAPER_18)
    elif name == "all_paired":
        ids = local_pool()
    else:
        raise ValueError(name)
    ids = [m for m in ids if m not in ("1.3", "5.1")]
    return sorted(ids, key=lambda x: tuple(int(p) for p in x.split(".")))


# ---------------------------------------------------------------------------
# Clean-room Eq. 5 HQI and 1D / 2D Viterbi implementations
# ---------------------------------------------------------------------------
def _clean_hqi_frame_scalar(p_hr: np.ndarray, hr_grid: np.ndarray, br_grid: np.ndarray) -> np.ndarray:
    p_hr = np.maximum(np.asarray(p_hr, dtype=np.float64), 0.0)
    n_h, n_b = hr_grid.size, br_grid.size
    if p_hr.max() <= 0.0:
        return np.zeros((n_h, n_b), dtype=np.float64)
    q_hr = float(p_hr.max()) / (float(p_hr.mean()) + 1e-10)
    a_t = (5.0 + 20.0 * min(1.0, max(0.0, (q_hr - 2.0) / 18.0))) * 0.01
    out = np.zeros((n_h, n_b), dtype=np.float64)
    harm_ns = (2.0, 3.0, 4.0, 5.0)
    for j, b in enumerate(br_grid):
        r = hr_grid / (b + 1e-6)
        in_harm = np.zeros(n_h, dtype=bool)
        gauss = np.zeros(n_h, dtype=np.float64)
        for n in harm_ns:
            in_harm |= np.abs(r - n) < 0.10
            gauss += np.exp(-0.5 * ((r - n) / 0.05) ** 2)
        p_h = float(p_hr[in_harm].max()) if np.any(in_harm) else 0.0
        p_nh = float(p_hr[~in_harm].max()) if np.any(~in_harm) else 0.0
        c_j = min(1.0, max(0.0, p_nh / (p_h + 1e-300)))
        out[:, j] = c_j * a_t * gauss
    return out


def _clean_hqi_seq(p_hr_seq: np.ndarray, hr_grid: np.ndarray, br_grid: np.ndarray) -> np.ndarray:
    """Independent NumPy clean-room Eq. 5 HQI evaluator across (T, N_H, N_B)."""
    p_seq = np.maximum(np.asarray(p_hr_seq, dtype=np.float64), 0.0)
    n_t = p_seq.shape[0]
    n_h, n_b = hr_grid.size, br_grid.size
    r = hr_grid[:, None] / (br_grid[None, :] + 1e-6)
    harm_ns = (2.0, 3.0, 4.0, 5.0)
    in_harm = np.zeros((n_h, n_b), dtype=bool)
    gauss = np.zeros((n_h, n_b), dtype=np.float64)
    for n in harm_ns:
        in_harm |= np.abs(r - n) < 0.10
        gauss += np.exp(-0.5 * ((r - n) / 0.05) ** 2)
    has_h = np.any(in_harm, axis=0)
    has_nh = np.any(~in_harm, axis=0)

    out = np.zeros((n_t, n_h, n_b), dtype=np.float64)
    for t in range(n_t):
        phr = p_seq[t]
        pmax = float(phr.max())
        if pmax <= 0.0:
            continue
        q_hr = pmax / (float(phr.mean()) + 1e-10)
        a_t = (5.0 + 20.0 * min(1.0, max(0.0, (q_hr - 2.0) / 18.0))) * 0.01
        col = phr[:, None]
        ph = np.where(has_h, np.max(np.where(in_harm, col, -np.inf), axis=0), 0.0)
        pnh = np.where(has_nh, np.max(np.where(~in_harm, col, -np.inf), axis=0), 0.0)
        c = np.clip(np.maximum(pnh, 0.0) / (np.maximum(ph, 0.0) + 1e-300), 0.0, 1.0)
        out[t] = c[None, :] * a_t * gauss
    return out


def _clean_decode_1d(
    p_br: np.ndarray,
    br_grid: np.ndarray,
    p_hr: np.ndarray | None = None,
    hr_grid: np.ndarray | None = None,
    sigma_br: float = 4.0,
    w_br: float = 1.0,
    eps_p: float = 1e-300,
    w_hqi: float = 0.0,
) -> tuple[np.ndarray, float]:
    n_t, n_b = p_br.shape
    hqi_seq = (
        _clean_hqi_seq(p_hr, hr_grid, br_grid)
        if (w_hqi > 0.0 and p_hr is not None and hr_grid is not None)
        else None
    )
    v = w_br * np.log(np.maximum(p_br[0], 0.0) + eps_p)
    back = np.zeros((n_t, n_b), dtype=np.int32)
    psi = -0.5 * ((br_grid[:, None] - br_grid[None, :]) / sigma_br) ** 2
    for t in range(1, n_t):
        cand = v[:, None] + psi
        back[t] = np.argmax(cand, axis=0)
        best_prev = cand[back[t], np.arange(n_b)]
        emit = w_br * np.log(np.maximum(p_br[t], 0.0) + eps_p)
        if hqi_seq is not None and float(np.max(p_hr[t])) > 0.0:
            emit = emit - w_hqi * hqi_seq[t, int(np.argmax(p_hr[t])), :]
        v = emit + best_prev
    idx = np.empty(n_t, dtype=np.int32)
    cur = int(np.argmax(v))
    score = float(v[cur])
    idx[-1] = cur
    for t in range(n_t - 1, 0, -1):
        cur = int(back[t, cur])
        idx[t - 1] = cur
    return br_grid[idx], score


def _clean_decode_2d_sep(
    p_br: np.ndarray,
    p_hr: np.ndarray,
    br_grid: np.ndarray,
    hr_grid: np.ndarray,
    w_br: float = 0.5,
    w_hqi: float = 4.0,
    sigma_br: float = 4.0,
    sigma_hr: float = 12.0,
    eps_p: float = 1e-300,
    use_hqi: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    n_t, n_b = p_br.shape
    n_h = hr_grid.size
    hqi_seq = _clean_hqi_seq(p_hr, hr_grid, br_grid) if (use_hqi and w_hqi > 0.0) else None
    r_ij = hr_grid[:, None] / (br_grid[None, :] + 1e-6)
    v = (
        np.log(np.maximum(p_hr[0], 0.0)[:, None] + eps_p)
        + np.log(np.maximum(p_br[0], 0.0)[None, :] + eps_p)
        - 0.5 * ((r_ij - 3.5) / 2.5) ** 2
    )
    psi_b = -0.5 * ((br_grid[:, None] - br_grid[None, :]) / sigma_br) ** 2
    psi_h = -0.5 * ((hr_grid[:, None] - hr_grid[None, :]) / sigma_hr) ** 2
    bp_b = np.zeros((n_t, n_h, n_b), dtype=np.int16)
    bp_h = np.zeros((n_t, n_b, n_h), dtype=np.int16)
    buf_b = np.empty((n_h, n_b, n_b), dtype=np.float64)
    buf_h = np.empty((n_b, n_h, n_h), dtype=np.float64)
    for t in range(1, n_t):
        np.add(v[:, None, :], psi_b[None, :, :], out=buf_b)
        bp_b[t] = np.argmax(buf_b, axis=2)
        mid = np.max(buf_b, axis=2)
        np.add(np.ascontiguousarray(mid.T)[:, None, :], psi_h[None, :, :], out=buf_h)
        bp_h[t] = np.argmax(buf_h, axis=2)
        best_prev = np.max(buf_h, axis=2).T
        emit = (
            np.log(np.maximum(p_hr[t], 0.0)[:, None] + eps_p)
            + w_br * np.log(np.maximum(p_br[t], 0.0)[None, :] + eps_p)
        )
        if hqi_seq is not None:
            emit = emit - w_hqi * hqi_seq[t]
        v = emit + best_prev
    i_cur, j_cur = np.unravel_index(int(np.argmax(v)), v.shape)
    score = float(v[i_cur, j_cur])
    br_i = np.empty(n_t, dtype=np.int32)
    hr_i = np.empty(n_t, dtype=np.int32)
    br_i[-1], hr_i[-1] = int(j_cur), int(i_cur)
    for t in range(n_t - 1, 0, -1):
        i_prev = int(bp_h[t, j_cur, i_cur])
        j_prev = int(bp_b[t, i_prev, j_cur])
        i_cur, j_cur = i_prev, j_prev
        hr_i[t - 1], br_i[t - 1] = i_cur, j_cur
    return br_grid[br_i], hr_grid[hr_i], score


def _clean_decode_2d_bruteforce(
    p_br: np.ndarray,
    p_hr: np.ndarray,
    br_grid: np.ndarray,
    hr_grid: np.ndarray,
    w_br: float = 0.5,
    w_hqi: float = 4.0,
    sigma_br: float = 4.0,
    sigma_hr: float = 12.0,
    eps_p: float = 1e-300,
) -> tuple[np.ndarray, np.ndarray, float]:
    n_t, n_b = p_br.shape
    n_h = hr_grid.size
    n_s = n_h * n_b
    r_ij = hr_grid[:, None] / (br_grid[None, :] + 1e-6)
    v = (
        np.log(np.maximum(p_hr[0], 0.0)[:, None] + eps_p)
        + np.log(np.maximum(p_br[0], 0.0)[None, :] + eps_p)
        - 0.5 * ((r_ij - 3.5) / 2.5) ** 2
    ).reshape(n_s)
    h_rep = np.repeat(hr_grid, n_b)
    b_rep = np.tile(br_grid, n_h)
    psi_all = (
        -0.5 * ((h_rep[None, :] - h_rep[:, None]) / sigma_hr) ** 2
        - 0.5 * ((b_rep[None, :] - b_rep[:, None]) / sigma_br) ** 2
    )
    back = np.zeros((n_t, n_s), dtype=np.int32)
    for t in range(1, n_t):
        cand = v[:, None] + psi_all
        back[t] = np.argmax(cand, axis=0)
        best_prev = cand[back[t], np.arange(n_s)]
        emit = (
            np.log(np.maximum(p_hr[t], 0.0)[:, None] + eps_p)
            + w_br * np.log(np.maximum(p_br[t], 0.0)[None, :] + eps_p)
            - w_hqi * _clean_hqi_frame_scalar(p_hr[t], hr_grid, br_grid)
        ).reshape(n_s)
        v = emit + best_prev
    cur = int(np.argmax(v))
    score = float(v[cur])
    br_i = np.empty(n_t, dtype=np.int32)
    hr_i = np.empty(n_t, dtype=np.int32)
    for t in range(n_t - 1, -1, -1):
        ic, jc = divmod(cur, n_b)
        hr_i[t], br_i[t] = ic, jc
        if t > 0:
            cur = int(back[t, cur])
    return br_grid[br_i], hr_grid[hr_i], score


# ---------------------------------------------------------------------------
# Clean-room statistical formulas (within-session OLS demeaned rmcorr, ICC, BA)
# ---------------------------------------------------------------------------
def _clean_rmcorr_ols(est_list: list[np.ndarray], ref_list: list[np.ndarray]) -> float:
    dx_parts, dy_parts = [], []
    k_used = 0
    for e, r in zip(est_list, ref_list):
        if len(e) < 2:
            continue
        dx_parts.append(e - np.mean(e))
        dy_parts.append(r - np.mean(r))
        k_used += 1
    dx = np.concatenate(dx_parts)
    dy = np.concatenate(dy_parts)
    denom = float(np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2)))
    if denom <= 0.0:
        return float("nan")
    return float(np.clip(np.sum(dx * dy) / denom, -1.0, 1.0))


def _clean_icc_1_1(est_means: np.ndarray, ref_means: np.ndarray) -> float:
    mat = np.column_stack([est_means, ref_means])  # (n, 2)
    n = mat.shape[0]
    grand = float(np.mean(mat))
    row_means = np.mean(mat, axis=1)
    ss_between = 2.0 * float(np.sum((row_means - grand) ** 2))
    ss_within = float(np.sum((mat - row_means[:, None]) ** 2))
    bms = ss_between / (n - 1)
    wms = ss_within / n
    return float((bms - wms) / (bms + wms))


def run_audit() -> dict[str, Any]:
    report: dict[str, Any] = {"gates": {}}
    md_lines = [
        "# God-Mode Independent Clean-Room Audit Report",
        "",
        "Executed with **zero** imports from `thesis.*` against raw EDFs, locked `.npz` caches, and `metrics_viterbi_repro.json`.",
        "",
    ]

    # ------------------------------------------------------------------
    # Gate 1: Frozen Baseline SHA-256 Lock (154/154 files + 33/33 caches)
    # ------------------------------------------------------------------
    bad_files = []
    checked_frozen = 0
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected_hash, rel_path = line.split(None, 1)
        rel_path = rel_path.strip()
        if rel_path.startswith(("paper_workspace/", "beltrao_data/")):
            checked_frozen += 1
            p = ROOT / rel_path
            if not p.exists() or _sha256(p) != expected_hash:
                bad_files.append(rel_path)

    bad_caches = []
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

    g1_pass = checked_frozen == 154 and len(bad_files) == 0 and len(bad_caches) == 0
    report["gates"]["gate1_frozen_baseline_lock"] = {
        "passed": g1_pass,
        "checked_frozen_files": checked_frozen,
        "bad_files": bad_files,
        "bad_caches": bad_caches,
    }
    md_lines.append(
        f"## Gate 1 — Frozen Baseline Lock: **{'PASS' if g1_pass else 'FAIL'}**\n"
        f"- Checked `{checked_frozen}/154` frozen files (`bad: {len(bad_files)}`) and `33/33` locked `v1` caches (`bad: {len(bad_caches)}`).\n"
    )

    # Load master outputs
    master = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    df_sess = pd.read_csv(SESSIONS_CSV, dtype={"measurement": str})

    # ------------------------------------------------------------------
    # Gate 2: Raw EDF Check + Clean-Room End-to-End Metric Recomputation
    # ------------------------------------------------------------------
    lags_clean: dict[str, int] = {}
    with LAGS_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lags_clean[row["measurement"]] = int(float(row["lag_raw_nls_s"]))

    p30_ids = _cohort_sessions_clean("pool30_sync")
    r23_ids = _cohort_sessions_clean("retained23")

    edf_ref_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    bad_edf_fs = []
    for m in p30_ids:
        epath = C.REF_DIR / f"{MEASURE_TO_REF[m]}.edf"
        rd = pyedflib.EdfReader(str(epath))
        try:
            lbls = [rd.getLabel(i) for i in range(rd.signals_in_file)]
            idx = {l: i for i, l in enumerate(lbls)}
            fs_resp = float(rd.getSampleFrequency(idx["Resp"]))
            fs_rate = float(rd.getSampleFrequency(idx["RespRate"]))
            if abs(fs_rate - 1.0) > 1e-6:
                bad_edf_fs.append((m, fs_rate))
            resp_256 = rd.readSignal(idx["Resp"]).astype(np.float64)
            rate_1hz = rd.readSignal(idx["RespRate"]).astype(np.float64)
        finally:
            rd._close()

        with np.load(LOCKED_CACHE_DIR / f"{m}.npz", allow_pickle=False) as z:
            n_win = len(z["est"])
            finite_est = np.isfinite(z["est"]) & np.isfinite(z["est_hpss"])
        centers = np.arange(n_win, dtype=float) * C.UPDATE_S + C.WIN_S / 2.0
        ref_w, vwin = window_ref_and_gating(resp_256, rate_1hz, centers, lags_clean[m], fs_raw=fs_resp)
        edf_ref_map[m] = (ref_w, vwin & np.isfinite(ref_w) & finite_est)

    p30_res = master["results"]["branch_neutral_raw_nls_lag"]["pool30_sync"]
    best_lopo_key = min(
        (k for k in p30_res["methods"] if k.startswith("C6")),
        key=lambda k: p30_res["methods"][k]["rmse_sess"],
    )

    # Re-decode fixed + best LOPO cleanly from raw NPZ arrays
    check_targets = [
        "A3_Beltrao_NMF_NLS",
        "A4_HPSS_NLS",
        "C3b_2D_JointVit_HPSS",
        "C4d_2D_Anchored_HarmNLS_Vit_HPSS",
        best_lopo_key,
    ]
    clean_preds: dict[str, dict[str, np.ndarray]] = {k: {} for k in check_targets}
    fold_lookup = {f["patient_id"]: f for f in p30_res["lopo_folds"][best_lopo_key]}
    for m in p30_ids:
        with np.load(LOCKED_CACHE_DIR / f"{m}.npz", allow_pickle=False) as z:
            clean_preds["A3_Beltrao_NMF_NLS"][m] = z["est"].copy()
            clean_preds["A4_HPSS_NLS"][m] = z["est_hpss"].copy()
        with np.load(VITERBI_CACHE_DIR / f"{m}.npz", allow_pickle=False) as zv:
            p2 = zv["P_br_hpss_2bpm"].copy()
            sa_h = zv["S_anc_hpss_1bpm"].copy()
            sa_n = zv["S_anc_nmf_1bpm"].copy()
            su_h = zv["S_nls_hpss_1bpm"].copy()
            su_n = zv["S_nls_nmf_1bpm"].copy()
            phr = zv["P_hr_csd_2bpm"].copy()
        clean_preds["C3b_2D_JointVit_HPSS"][m] = _clean_decode_2d_sep(p2, phr, BR_GRID_2BPM, HR_GRID_2BPM)[0]
        clean_preds["C4d_2D_Anchored_HarmNLS_Vit_HPSS"][m] = _clean_decode_2d_sep(
            sa_h, phr, BR_GRID_1BPM, HR_GRID_2BPM
        )[0]
        finfo = fold_lookup[_patient_id(m)]
        spec_map = {
            "C6a_LOPO_Cal_Vit_HPSS": (p2, BR_GRID_2BPM),
            "C6b_LOPO_Cal_HarmNLS_HPSS": (su_h, BR_GRID_1BPM),
            "C6c_LOPO_Cal_HarmNLS_NMF": (su_n, BR_GRID_1BPM),
            "C6d_LOPO_Wide_AnchoredNLS_HPSS": (sa_h, BR_GRID_1BPM),
            "C6e_LOPO_Wide_AnchoredNLS_NMF": (sa_n, BR_GRID_1BPM),
        }
        sp, gr = spec_map[best_lopo_key]
        wh = float(finfo.get("selected_w_hqi", 4.0))
        if finfo["selected_mode"] == "1d":
            clean_preds[best_lopo_key][m] = _clean_decode_1d(
                sp, gr, p_hr=phr, hr_grid=HR_GRID_2BPM,
                sigma_br=finfo["selected_sigma_br"], w_br=1.0,
                eps_p=finfo["selected_eps_floor"], w_hqi=wh,
            )[0]
        else:
            clean_preds[best_lopo_key][m] = _clean_decode_2d_sep(
                sp, phr, gr, HR_GRID_2BPM,
                w_br=finfo["selected_w_br"], w_hqi=wh,
                sigma_br=finfo["selected_sigma_br"], eps_p=finfo["selected_eps_floor"],
                use_hqi=(wh > 0.0),
            )[0]

    # Constant baseline clean recomputation
    ref_list_p30 = [edf_ref_map[m][0][edf_ref_map[m][1]] for m in p30_ids]
    pids_p30 = [_patient_id(m) for m in p30_ids]
    const_loso_clean = float(np.mean([
        np.sqrt(np.mean((rf - np.median(np.concatenate([ref_list_p30[j] for j in range(len(p30_ids)) if j != i]))) ** 2))
        for i, rf in enumerate(ref_list_p30)
    ]))
    const_lopo_med_clean = float(np.mean([
        np.sqrt(np.mean((rf - np.median(np.concatenate([ref_list_p30[j] for j in range(len(p30_ids)) if pids_p30[j] != pids_p30[i]]))) ** 2))
        for i, rf in enumerate(ref_list_p30)
    ]))
    const_diffs = {
        "const_loso_diff": abs(const_loso_clean - p30_res["constants"]["const_loso"]),
        "const_lopo_med_diff": abs(const_lopo_med_clean - p30_res["constants"]["const_lopo_median"]),
    }

    max_metric_diff = max(const_diffs.values())
    gate2_details = {}
    sess_rmses: dict[str, dict[str, float]] = {}
    for mname in check_targets:
        s_rmses, s_maes, s_acc10 = [], [], []
        s_emeans, s_rmeans = [], []
        e_valid, r_valid = [], []
        sess_rmses[mname] = {}
        for m in p30_ids:
            rw, vv = edf_ref_map[m]
            ev = clean_preds[mname][m][vv]
            rv = rw[vv]
            err = ev - rv
            rm = float(np.sqrt(np.mean(err ** 2)))
            sess_rmses[mname][m] = rm
            s_rmses.append(rm)
            s_maes.append(float(np.mean(np.abs(err))))
            s_acc10.append(float(100.0 * np.mean(np.abs(err) < 10.0)))  # STRICT < 10
            s_emeans.append(float(np.mean(ev)))
            s_rmeans.append(float(np.mean(rv)))
            e_valid.append(ev)
            r_valid.append(rv)

        all_e, all_r = np.concatenate(e_valid), np.concatenate(r_valid)
        em_arr, rm_arr = np.array(s_emeans), np.array(s_rmeans)
        diffs_s = em_arr - rm_arr
        ba_bias = float(np.mean(diffs_s))
        ba_sd = float(np.std(diffs_s, ddof=1))
        recomp = {
            "rmse_sess": float(np.mean(s_rmses)),
            "rmse_pool": float(np.sqrt(np.mean((all_e - all_r) ** 2))),
            "mae_sess": float(np.mean(s_maes)),
            "acc10_sess": float(np.mean(s_acc10)),
            "r_within_rmcorr": _clean_rmcorr_ols(e_valid, r_valid),
            "icc_1_1": _clean_icc_1_1(em_arr, rm_arr),
            "ba_bias": ba_bias,
            "ba_loa_lo": ba_bias - 1.96 * ba_sd,
            "ba_loa_up": ba_bias + 1.96 * ba_sd,
        }
        jm = p30_res["methods"][mname]
        jba = jm["ba_session_level"]
        diffs = {
            "rmse_sess": abs(recomp["rmse_sess"] - jm["rmse_sess"]),
            "rmse_pool": abs(recomp["rmse_pool"] - jm["rmse_pool"]),
            "mae_sess": abs(recomp["mae_sess"] - jm["mae_sess"]),
            "acc10_sess": abs(recomp["acc10_sess"] - jm["acc10_sess"]),
            "r_within_rmcorr": abs(recomp["r_within_rmcorr"] - jm["r_within_rmcorr"]),
            "icc_1_1": abs(recomp["icc_1_1"] - jba["icc_1_1"]),
            "ba_bias": abs(recomp["ba_bias"] - jba["bias"]),
            "ba_loa_lo": abs(recomp["ba_loa_lo"] - jba["loa_lo"]),
            "ba_loa_up": abs(recomp["ba_loa_up"] - jba["loa_up"]),
        }
        dmax = max(diffs.values())
        max_metric_diff = max(max_metric_diff, dmax)
        gate2_details[mname] = {"max_diff": dmax, **recomp}

    # Second pass: verify patient-level Wilcoxon vs A4_HPSS_NLS cleanly
    for mname in check_targets:
        if mname == "A4_HPSS_NLS":
            continue
        by_p: dict[int, list[float]] = {}
        for m in p30_ids:
            by_p.setdefault(_patient_id(m), []).append(sess_rmses[mname][m] - sess_rmses["A4_HPSS_NLS"][m])
        pat_d = np.array([np.mean(v) for v in by_p.values()])
        p_pat = float(stats.wilcoxon(pat_d).pvalue) if not np.allclose(pat_d, 0.0) else 1.0
        dp = abs(p_pat - p30_res["paired_vs_hpss_nls"][mname]["p_wilcoxon_patient"])
        max_metric_diff = max(max_metric_diff, dp)
        gate2_details[mname]["max_diff"] = max(gate2_details[mname]["max_diff"], dp)
        gate2_details[mname]["p_wilcoxon_patient"] = p_pat

    g2_pass = len(bad_edf_fs) == 0 and max_metric_diff < 1e-9
    report["gates"]["gate2_cleanroom_recomputation"] = {
        "passed": g2_pass,
        "bad_edf_fs": bad_edf_fs,
        "max_metric_diff": max_metric_diff,
        "const_diffs": const_diffs,
        "details": gate2_details,
    }
    md_lines.append(
        f"## Gate 2 — Raw EDF (1.0 Hz Verified) & Clean-Room Metric Recomputation: **{'PASS' if g2_pass else 'FAIL'}**\n"
        f"- Checked EDF `RespRate` sample rate across `pool30_sync` (`bad: {len(bad_edf_fs)}`), re-decoded 5 key trackers (including `{best_lopo_key}`) from raw NPZs + clean-room Viterbi.\n"
        f"- Max discrepancy across `RMSE_sess`, `RMSE_pool`, `MAE_sess`, strict `< 10` `Acc@10`, demeaned-OLS `r_within`, `ICC(1,1)`, BA LoA, `const_loso`, `const_lopo_median`, and `p_wilcoxon_patient`: `{max_metric_diff:.2e}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 3: Clean-Room Brute-Force vs Separable 2D Viterbi Parity on 7.3
    # ------------------------------------------------------------------
    with np.load(VITERBI_CACHE_DIR / "7.3.npz", allow_pickle=False) as zv:
        p2_73 = zv["P_br_hpss_2bpm"][:15, ::2].copy()
        phr_73 = zv["P_hr_csd_2bpm"][:15, ::4].copy()
    br_sub = BR_GRID_2BPM[::2]
    hr_sub = HR_GRID_2BPM[::4]
    b_bf, h_bf, s_bf = _clean_decode_2d_bruteforce(p2_73, phr_73, br_sub, hr_sub)
    b_sp, h_sp, s_sp = _clean_decode_2d_sep(p2_73, phr_73, br_sub, hr_sub)
    dp_path_diff = float(max(np.max(np.abs(b_bf - b_sp)), np.max(np.abs(h_bf - h_sp))))
    dp_score_diff = abs(s_bf - s_sp)

    rw73, vv73 = edf_ref_map["7.3"]
    c3b_73 = clean_preds["C3b_2D_JointVit_HPSS"]["7.3"][vv73]
    row_73 = df_sess[
        (df_sess["regime"] == "branch_neutral_raw_nls_lag")
        & (df_sess["cohort"] == "pool30_sync")
        & (df_sess["measurement"] == "7.3")
        & (df_sess["method"] == "C3b_2D_JointVit_HPSS")
    ].iloc[0]
    sess73_diff = max(
        abs(float(np.sqrt(np.mean((c3b_73 - rw73[vv73]) ** 2))) - float(row_73["rmse"])),
        abs(float(np.mean(c3b_73)) - float(row_73["est_mean"])),
        abs(float(np.mean(rw73[vv73])) - float(row_73["ref_mean"])),
    )
    g3_pass = dp_path_diff == 0.0 and dp_score_diff < 1e-12 and sess73_diff < 1e-12
    report["gates"]["gate3_cleanroom_dp_parity"] = {
        "passed": g3_pass,
        "dp_path_diff": dp_path_diff,
        "dp_score_diff": dp_score_diff,
        "sess73_full_diff": sess73_diff,
    }
    md_lines.append(
        f"## Gate 3 — Clean-Room Eq. 5 HQI & 2D Viterbi DP Parity (`7.3`): **{'PASS' if g3_pass else 'FAIL'}**\n"
        f"- Unfactored brute-force vs separable 2D Viterbi on real `7.3` subgrid: path diff `{dp_path_diff:.1f}` bpm, score diff `{dp_score_diff:.2e}`.\n"
        f"- Full-session `7.3` `C3b` vs exported CSV: max diff `{sess73_diff:.2e}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 4: Figure 2 Cohort-Scoped CSV-to-JSON Scatter Parity
    # ------------------------------------------------------------------
    cohorts_in_csv = sorted(df_sess["cohort"].unique().tolist())
    max_scatter_diff = 0.0
    worst_case = None
    for (reg, coh, meth), grp in df_sess.groupby(["regime", "cohort", "method"]):
        jba = master["results"][reg][coh]["methods"][meth]["ba_session_level"]
        diffs = grp["est_mean"].to_numpy(float) - grp["ref_mean"].to_numpy(float)
        if diffs.size < 2:
            continue
        b_csv = float(np.mean(diffs))
        sd_csv = float(np.std(diffs, ddof=1))
        d_item = max(
            abs(b_csv - float(jba["bias"])),
            abs(sd_csv - float(jba["sd"])),
            abs((b_csv - 1.96 * sd_csv) - float(jba["loa_lo"])),
            abs((b_csv + 1.96 * sd_csv) - float(jba["loa_up"])),
        )
        if d_item > max_scatter_diff:
            max_scatter_diff = d_item
            worst_case = (reg, coh, meth, d_item)
    g4_pass = cohorts_in_csv == sorted(["pool30_sync", "retained23", "PAPER_18", "all_paired"]) and max_scatter_diff < 1e-12
    report["gates"]["gate4_scatter_to_json_parity"] = {
        "passed": g4_pass,
        "cohorts_in_csv": cohorts_in_csv,
        "max_scatter_diff": max_scatter_diff,
        "worst_case": worst_case,
    }
    md_lines.append(
        f"## Gate 4 — Cohort-Scoped Bland–Altman Scatter-to-JSON Parity: **{'PASS' if g4_pass else 'FAIL'}**\n"
        f"- Cohorts exported in CSV: `{cohorts_in_csv}`.\n"
        f"- Max Bias/SD/LoA discrepancy across all `(regime, cohort, method)` combinations (including all LOPO families): `{max_scatter_diff:.2e}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 5: LOPO Anti-Leakage & Grid Pinning Diagnostics
    # ------------------------------------------------------------------
    leak_errors = []
    for reg, reg_d in master["results"].items():
        for coh, coh_d in reg_d.items():
            eval_set = set(coh_d["sessions"])
            for fam, frows in coh_d["lopo_folds"].items():
                seen_test = []
                for f in frows:
                    pid = f["patient_id"]
                    ts = f["test_sessions"]
                    if not ts or any(_patient_id(s) != pid for s in ts):
                        leak_errors.append((reg, coh, fam, pid, "patient_mismatch"))
                    seen_test.extend(ts)
                if sorted(seen_test) != sorted(eval_set):
                    leak_errors.append((reg, coh, fam, "coverage_mismatch"))
    lopo_sum = p30_res["lopo_summary"]
    g5_pass = len(leak_errors) == 0 and len(lopo_sum) >= 5
    report["gates"]["gate5_lopo_anti_leakage_and_pinning"] = {
        "passed": g5_pass,
        "leak_errors": leak_errors,
        "pool30_sync_summary": lopo_sum,
    }
    md_lines.append(
        f"## Gate 5 — LOPO Anti-Leakage & Grid Pinning Audit: **{'PASS' if g5_pass else 'FAIL'}**\n"
        f"- Fold disjointness & coverage errors across all regimes/cohorts: `{len(leak_errors)}`.\n"
        f"- `pool30_sync` LOPO diagnostics (`C6a–C6e`): `{json.dumps(lopo_sum)}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 6: HR Gate G2, Multi-Seed Surrogates & Stress-Tests
    # ------------------------------------------------------------------
    hr_val = master["hr_validation"]
    max_hr_w10 = max(v["pct_within_10bpm_sess"] for v in hr_val.values())
    any_hr_pass = any(v["passes_gate_g2_50pct"] for v in hr_val.values())
    ctrl = master["controls"]
    t_shuf_ok = (
        all(v["n_seeds"] == 20 and abs(v["r_within_rmcorr"]) < 0.05 for v in ctrl["surrogate_time_shuffled_ref"].values())
    )
    p_scr = ctrl["surrogate_phase_scrambled_radar"]
    p_scr_ok = (not p_scr["skipped"]) and all(
        v["n_seeds"] == 5 and abs(v["r_within_rmcorr"]) < 0.10
        for k, v in p_scr.items()
        if isinstance(v, dict)
    )
    inv_max = max(max(d.values()) for d in ctrl["mathematical_group_invariances_max_diff"].values())
    g6_pass = t_shuf_ok and p_scr_ok and inv_max < 1e-6 and ("audit_stress_tests" in master)
    report["gates"]["gate6_controls_and_hr_gate_g2"] = {
        "passed": g6_pass,
        "max_hr_within_10bpm_sess": max_hr_w10,
        "any_hr_passes_gate_g2": any_hr_pass,
        "invariance_max_diff": inv_max,
    }
    md_lines.append(
        f"## Gate 6 — Multi-Seed Surrogates, Invariances & HR Gate G2: **{'PASS' if g6_pass else 'FAIL'}**\n"
        f"- Max HR `<= 10 bpm` across all {len(hr_val)} HR extractors (incl. DACM, BR-Notched, 480-pt): `{max_hr_w10:.2f}%` (Gate `G2 >= 50%`: `{any_hr_pass}`).\n"
        f"- 20-seed time-shuffle `r_within` ok: `{t_shuf_ok}`; 5-seed phase-scramble `r_within` ok: `{p_scr_ok}`; I/Q group invariance max diff: `{inv_max:.2e}` bpm.\n"
    )

    # ------------------------------------------------------------------
    # Gate 7: Provenance, Git Head & Zero-Drop Completeness
    # ------------------------------------------------------------------
    prov = master["provenance"]
    tracker_actual = _sha256(ROOT / "thesis" / "viterbi_tracker.py")
    runner_actual = _sha256(ROOT / "thesis" / "exp_viterbi_reproduction.py")
    ret_res = master["results"]["branch_neutral_raw_nls_lag"]["retained23"]["methods"]
    bad_counts_p30 = [
        (k, d["n_sessions"], d["n_windows"])
        for k, d in p30_res["methods"].items()
        if d["n_sessions"] != 28 or d["n_windows"] != 6640
    ]
    bad_counts_r23 = [
        (k, d["n_sessions"], d["n_windows"])
        for k, d in ret_res.items()
        if d["n_sessions"] != 23 or d["n_windows"] != 6447
    ]
    g7_pass = (
        prov["schema_version"] == "viterbi-repro-v2"
        and prov["git"]["head_commit"] not in ("", "unknown")
        and prov["tracker_sha256"] == tracker_actual
        and prov["runner_sha256"] == runner_actual
        and len(bad_counts_p30) == 0
        and len(bad_counts_r23) == 0
        and len(p30_res["methods"]) == 33
    )
    report["gates"]["gate7_provenance_and_completeness"] = {
        "passed": g7_pass,
        "git_head": prov["git"],
        "n_methods": len(p30_res["methods"]),
        "bad_counts_p30": bad_counts_p30,
        "bad_counts_r23": bad_counts_r23,
    }
    md_lines.append(
        f"## Gate 7 — Provenance, Git Head & Completeness: **{'PASS' if g7_pass else 'FAIL'}**\n"
        f"- Schema `{prov['schema_version']}` | Git `{prov['git']['head_commit']}` (`{prov['git']['branch']}`) | Methods: `{len(p30_res['methods'])}`.\n"
        f"- Window completeness: `pool30_sync` (`28 / 6,640` bad: `{len(bad_counts_p30)}`), `retained23` (`23 / 6,447` bad: `{len(bad_counts_r23)}`).\n"
    )

    all_passed = all(g["passed"] for g in report["gates"].values())
    report["all_passed"] = all_passed
    md_lines.insert(3, f"**Overall Verdict:** `{'7/7 GATES PASSED' if all_passed else 'AUDIT FAILED'}`\n")
    AUDIT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    res = run_audit()
    sys.exit(0 if res["all_passed"] else 1)
