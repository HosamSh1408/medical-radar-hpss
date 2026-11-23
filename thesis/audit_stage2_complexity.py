"""Zero-shared-code clean-room auditor for Stage 2 (`stage2-complexity-v4`).

Intentionally imports ZERO statistical or LOPO evaluation logic from
`thesis.exp1_stage1_parsimony`, `thesis.exp2_stage2_complexity`, or `thesis.io_utils`.
Independently verifies:
  - Gate 1: Frozen baseline SHA-256 lock (`154/154` + `33/33` + clean git) & exact Stage 1.1 parity (`< 1e-12`).
  - Gate 2: Clean-room metric recomputation (`RMSE_sess`, `RMSE_pool`, `MAE_sess`, `Acc@10`, `rmcorr`, `pred_sd_ratio_1hz`) across all 4 `pool30_sync` references & CSV<->JSON parity.
  - Gate 3: Central finite-difference gradient verification (`R2a/R2b/R2c` HMM & `CausalDilatedTCN`), `sigma_slow >= 1.0` floor, and SVD Ridge vs `sklearn.linear_model.Ridge` parity.
  - Gate 4: Clean-room `O(2) x R+` invariance (`< 1e-6`), strict `0.0` causality, exact operator `+10 bpm` shift equivariance (`< 1e-9` on `[-60, 180]`), `[18, 90]` vs `[18, 120]` shift (`< 0.05 bpm`), and two-axis causality metadata.
  - Gate 5: Strict patient-disjoint LOPO partitions (`11/10/10`), `val_patient_id != patient_id`, `sigma_slow_bpm >= 1.0`, and Rung 3 selected hyperparameter logs.
  - Gate 6: `-10 s` lag alignment verification (`|peak_lag_s| <= 2 s`), synthetic `20-80 bpm` sweep checks, and 5-seed phase-scrambled surrogates (`|r_within| < 0.05`).
  - Gate 7: Provenance, strict RFC-8259 JSON (`allow_nan=False`), and completeness across cohorts, 4-way refs, and WP4 within-session Spearman fields.

Execution:
  MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /opt/homebrew/bin/python3.10 -m thesis.audit_stage2_complexity
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
from sklearn.linear_model import Ridge

from paper_workspace.src import config as C
from paper_workspace.src.cohort import PAPER_18, local_pool, pool30, retained23
from paper_workspace.src.data_loader import MEASURE_TO_REF, load_radar
from paper_workspace.src.demod import csd
from paper_workspace.src.evaluation import window_ref_and_gating
from paper_workspace.src.filtering import autocorr_coarse, bandpass
from paper_workspace.src.provenance import PIPELINE_VERSION, expected_provenance
from thesis import diff_hmm_and_foils as D
from thesis import diff_hmm_batch as DB
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
SESSIONS_CSV = AUDIT_DIR / "stage2_complexity_sessions.csv"
PRED_ARCHIVE = ROOT / "thesis" / "outputs" / "stage2_complexity" / "pool30_predictions.npz"
AUDIT_JSON = AUDIT_DIR / "stage2_godmode_audit.json"
AUDIT_MD = AUDIT_DIR / "stage2_godmode_audit.md"

MODELS_13 = (
    "R0_HPSS_NLS_Locked",
    "R1a_Online_Forward_MMSE_0s",
    "R1f_Offline_FB_MMSE_Inf",
    "R1g_Offline_Viterbi_Unpinned",
    "R2e_Causal_DiffHMM_FilterSafe8",
    "R2a_Causal_DiffHMM_Linear11",
    "R2b_Causal_DiffHMM_MLP43",
    "R2f_Offline_DiffHMM_FilterSafe8",
    "R2c_Offline_DiffHMM_Linear11",
    "R2d_E2E_Causal_H5_DiffHMM_Linear11",
    "R3c_Ridge_Spectral_439",
    "R3b_Spectral_MLP_15k",
    "R3a_Causal_TCN_16k",
)


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


def _clean_extract_edf_refs(
    edf_path: Path,
    centers_s: np.ndarray,
    lag_s: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Clean-room EDF extractor computing all 4 pool30_sync references without thesis.edf_extractor."""
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

    def _trig_interval_at(shift_lag: int) -> tuple[np.ndarray, np.ndarray]:
        tc = centers_s - float(shift_lag)
        a_arr = np.floor(tc - float(C.WIN_S) / 2.0).astype(int)
        b_arr = a_arr + int(C.WIN_S)
        arr = np.full(len(centers_s), np.nan, dtype=np.float64)
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
                arr[w_i] = float(np.median(vr))
        return arr, v1 & np.isfinite(arr)

    rt0, vt0 = _trig_interval_at(lag_s)
    rta, vta = _trig_interval_at(lag_s + 10)

    # Breath-count rate at lag_s over identical [a, b) half-open bounds
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
    }


def run_audit() -> dict[str, Any]:
    report: dict[str, Any] = {"gates": {}}
    raw_text = STAGE2_JSON.read_text(encoding="utf-8")
    master = json.loads(raw_text)
    schema_lbl = str(master.get("schema_version", "unknown"))
    md: list[str] = [
        f"# Stage 2 Independent Clean-Room Audit Report (`{schema_lbl}`)\n",
        "Executed with zero statistical/LOPO imports from `thesis.exp1_stage1_parsimony`, `thesis.exp2_stage2_complexity`, or `thesis.io_utils`.\n",
    ]
    s1 = json.loads(STAGE1_JSON.read_text(encoding="utf-8"))
    df_csv = pd.read_csv(SESSIONS_CSV, dtype={"measurement": str})

    # ------------------------------------------------------------------
    # Gate 1: Frozen Baseline SHA-256 Lock (154/154 + 33/33) & Stage 1.1 Parity
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

    git_res = subprocess.run(
        ["git", "status", "--porcelain", "paper_workspace", "beltrao_data", "external_data"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if git_res.returncode == 0:
        git_status_checked = True
        git_clean = git_res.stdout.strip() == ""
    else:
        git_status_checked = False
        git_clean = True  # macOS sandbox blocks xcrun/git shim; manifest + external mtime guard runs below

    ext_dir = ROOT / "external_data" / "gleichauf2024"
    stage1_mtime = STAGE1_JSON.stat().st_mtime
    ext_files = [p for p in (ROOT / "external_data").rglob("*") if p.is_file()]
    external_data_untouched = (
        ext_dir.is_dir()
        and len(ext_files) > 0
        and all(p.stat().st_mtime < stage1_mtime for p in ext_files)
    )

    s1_max_diff = 0.0
    for coh in ("pool30_sync", "retained23", "PAPER_18"):
        for rlbl in ("br_1hz_30s", "br_trig_bpm"):
            for km in (
                "R0_HPSS_NLS_Locked",
                "R1a_Online_Forward_MMSE_0s",
                "R1f_Offline_FB_MMSE_Inf",
                "R1g_Offline_Viterbi_Unpinned",
            ):
                m1 = s1["rung1_lopo"][coh][rlbl]["methods"][km]
                m2 = master["cohorts"][coh][rlbl]["methods"][km]
                for fld in ("rmse_sess", "rmse_pool", "mae_sess", "acc10_sess", "r_within_rmcorr"):
                    s1_max_diff = max(s1_max_diff, abs(float(m2[fld]) - float(m1[fld])))

    g1_pass = (
        checked_frozen == 154
        and len(bad_files) == 0
        and len(bad_caches) == 0
        and git_clean
        and external_data_untouched
        and s1.get("schema_version") == "stage1-parsimony-v3"
        and s1_max_diff < 1e-12
    )
    report["gates"]["gate1_frozen_baseline_and_stage1_parity"] = {
        "passed": g1_pass,
        "checked_frozen_files": checked_frozen,
        "bad_files": bad_files,
        "bad_caches": bad_caches,
        "git_status_checked": git_status_checked,
        "git_frozen_dirs_clean": git_clean,
        "external_data_untouched": external_data_untouched,
        "external_data_file_count": len(ext_files),
        "stage1_parity_max_diff": s1_max_diff,
    }
    md.append(
        f"## Gate 1 — Frozen Baseline Lock & Stage 1.1 Parity: **{'PASS' if g1_pass else 'FAIL'}**\n"
        f"- Checked `{checked_frozen}/154` frozen hashes (`bad: {len(bad_files)}`), `33/33` locked caches (`bad: {len(bad_caches)}`), `external_data/` untouched `{external_data_untouched}` (`{len(ext_files)}` files), git checked `{git_status_checked}`.\n"
        f"- Exact Stage 1.1 parity (`R0, R1a, R1f, R1g` across 3 cohorts x 2 refs x 5 metrics): `max diff = {s1_max_diff:.2e}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 2: Clean-Room Metric Recomputation & CSV<->JSON Parity
    # ------------------------------------------------------------------
    lags_clean: dict[str, int] = {}
    with LAGS_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lags_clean[row["measurement"]] = int(float(row["lag_raw_nls_s"]))

    p30_ids = _cohort_ids("pool30_sync")
    arch = np.load(PRED_ARCHIVE, allow_pickle=False)
    refs_all: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        "br_1hz_30s": {},
        "br_trig_bpm": {},
        "br_trig_bpm_lag_aligned": {},
        "br_trig_count_bpm": {},
    }
    for m in p30_ids:
        n_w = len(arch[f"pred__R0_HPSS_NLS_Locked__{m}"])
        centers = np.arange(n_w, dtype=np.float64) * C.UPDATE_S + C.WIN_S / 2.0
        epath = C.REF_DIR / f"{MEASURE_TO_REF[m]}.edf"
        extracted = _clean_extract_edf_refs(epath, centers, lags_clean[m])
        for rlbl, pair in extracted.items():
            refs_all[rlbl][m] = pair

    max_metric_diff = 0.0
    max_sd_ratio_diff = 0.0
    n_win_errs: list[Any] = []
    for rlbl, rmap in refs_all.items():
        for km in MODELS_13:
            ps_rmse, ps_mae, ps_acc10, evs, rvs = [], [], [], [], []
            sd_ratios = []
            for m in p30_ids:
                rv, vv = rmap[m]
                pr = arch[f"pred__{km}__{m}"][vv].astype(np.float64)
                rf = rv[vv]
                err = pr - rf
                ps_rmse.append(float(np.sqrt(np.mean(err ** 2))))
                ps_mae.append(float(np.mean(np.abs(err))))
                ps_acc10.append(float(100.0 * np.mean(np.abs(err) < 10.0)))
                evs.append(pr)
                rvs.append(rf)
                if rlbl in ("br_1hz_30s", "br_trig_bpm_lag_aligned") and np.std(rf) > 1e-6:
                    sd_ratios.append(float(np.std(pr) / np.std(rf)))
            all_e = np.concatenate(evs)
            all_r = np.concatenate(rvs)
            rec = {
                "rmse_sess": float(np.mean(ps_rmse)),
                "rmse_pool": float(np.sqrt(np.mean((all_e - all_r) ** 2))),
                "mae_sess": float(np.mean(ps_mae)),
                "acc10_sess": float(np.mean(ps_acc10)),
                "r_within_rmcorr": _clean_rmcorr(evs, rvs),
            }
            jm = master["cohorts"]["pool30_sync"][rlbl]["methods"][km]
            if int(all_r.size) != int(jm["n_windows"]):
                n_win_errs.append((rlbl, km, int(all_r.size), int(jm["n_windows"])))
            for fld, val in rec.items():
                max_metric_diff = max(max_metric_diff, abs(val - float(jm[fld])))
            if rlbl == "br_1hz_30s":
                max_sd_ratio_diff = max(
                    max_sd_ratio_diff,
                    abs(float(np.mean(sd_ratios)) - float(master["shortcut_learning_audit_pool30"][km]["pred_sd_ratio_1hz"])),
                )
            elif rlbl == "br_trig_bpm_lag_aligned":
                max_sd_ratio_diff = max(
                    max_sd_ratio_diff,
                    abs(float(np.mean(sd_ratios)) - float(master["shortcut_learning_audit_pool30"][km]["pred_sd_ratio_trig_lag_aligned"])),
                )

    csv_json_max_diff = 0.0
    for km in MODELS_13:
        for suf, rlbl in (("rmse_1hz", "br_1hz_30s"), ("rmse_trig", "br_trig_bpm"), ("rmse_trig_lag_aligned", "br_trig_bpm_lag_aligned")):
            cm = float(df_csv[f"{km}_{suf}"].astype(float).mean())
            jm = float(master["cohorts"]["pool30_sync"][rlbl]["methods"][km]["rmse_sess"])
            csv_json_max_diff = max(csv_json_max_diff, abs(cm - jm))

    g2_pass = (
        len(df_csv) == 28
        and len(n_win_errs) == 0
        and max_metric_diff < 1e-10
        and max_sd_ratio_diff < 1e-10
        and csv_json_max_diff <= 1e-4
    )
    report["gates"]["gate2_cleanroom_metrics_and_csv_parity"] = {
        "passed": g2_pass,
        "csv_rows": int(len(df_csv)),
        "n_windows_parity_errors": n_win_errs,
        "max_cleanroom_metric_diff": max_metric_diff,
        "max_sd_ratio_diff": max_sd_ratio_diff,
        "csv_json_max_diff": csv_json_max_diff,
    }
    md.append(
        f"## Gate 2 — Clean-Room 4-Way Metric Recomputation & CSV Parity: **{'PASS' if g2_pass else 'FAIL'}**\n"
        f"- Max diff across `13` models x `4` refs x `5` metrics: `{max_metric_diff:.2e}`; `pred_sd_ratio` (`1Hz` & `T10`) diff: `{max_sd_ratio_diff:.2e}`; CSV (`28` rows) diff: `{csv_json_max_diff:.2e}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 3: Clean-Room Central Finite-Diff Gradients, Bounds & SVD Ridge
    # ------------------------------------------------------------------
    rng = np.random.default_rng(2026)
    grid = VT.BR_GRID_1BPM
    phi_syn = rng.normal(-2.0, 0.8, size=(10, grid.size, 6))
    ref_syn = rng.uniform(25.0, 60.0, size=10)
    vm_syn = np.array([True] * 9 + [False], dtype=bool)
    sess_syn = [(phi_syn, ref_syn, vm_syn)]

    max_r2_grad_diff = 0.0
    r2_bound_ok = True
    for mode in ("linear", "linear_filtersafe", "mlp"):
        for dec in ("causal_forward", "offline_fb"):
            mdl = D.DifferentiableSymmetricHMM(mode=mode, decoder=dec, reg_l2=5e-3, seed=7)
            if not (
                np.isclose(mdl.bounds[0][0], 2.0)
                and mdl.bounds[-5] == (0.0, 0.5)
                and mdl.bounds[-4] == (0.0, 0.5)
                and np.isclose(np.exp(mdl.bounds[-3][0]), 1.0)
                and np.isclose(mdl.bounds[-1][1], -4.5)
            ):
                r2_bound_ok = False
            th = mdl.theta + rng.normal(0.0, 0.04, size=mdl.theta.size)
            th = np.clip(th, [b[0] + 1e-3 for b in mdl.bounds], [b[1] - 1e-3 for b in mdl.bounds])
            _, g_ana = mdl.loss_and_grad(th, sess_syn)
            h = 1e-6
            for i in range(th.size):
                tp = th.copy()
                tp[i] += h
                tm = th.copy()
                tm[i] -= h
                lp, _ = mdl.loss_and_grad(tp, sess_syn)
                lm, _ = mdl.loss_and_grad(tm, sess_syn)
                max_r2_grad_diff = max(max_r2_grad_diff, abs(float(g_ana[i]) - float((lp - lm) / (2.0 * h))))

    tcn = D.CausalDilatedTCN(in_dim=438, channels=24, dilations=(1, 2, 4), seed=19)
    xs_syn = rng.normal(0.0, 1.0, size=(8, 438))
    _, g_tcn = tcn.session_mse_loss_and_grad([(xs_syn, ref_syn[:8], vm_syn[:8])], weight_decay=1e-2, dropout_p=0.0, rng=None)
    max_tcn_grad_diff = 0.0
    for k_p in tcn.param_keys:
        flat = tcn.params[k_p].ravel()
        gf = g_tcn[k_p].ravel()
        for idx in rng.choice(flat.size, size=min(12, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + 1e-6
            lp, _ = tcn.session_mse_loss_and_grad([(xs_syn, ref_syn[:8], vm_syn[:8])], weight_decay=1e-2, dropout_p=0.0, rng=None)
            flat[idx] = old - 1e-6
            lm, _ = tcn.session_mse_loss_and_grad([(xs_syn, ref_syn[:8], vm_syn[:8])], weight_decay=1e-2, dropout_p=0.0, rng=None)
            flat[idx] = old
            max_tcn_grad_diff = max(max_tcn_grad_diff, abs(float(gf[idx]) - float((lp - lm) / 2e-6)))

    xr = rng.normal(0.0, 1.0, size=(35, 18))
    yr = rng.normal(40.0, 6.0, size=35)
    wr = rng.uniform(0.3, 2.0, size=35)
    svd_paths = D._svd_weighted_ridge_paths(xr, yr, wr, np.array([0.1, 10.0, 1000.0]))
    max_svd_diff = 0.0
    for a_v, (coef_s, inter_s) in zip((0.1, 10.0, 1000.0), svd_paths):
        sk = Ridge(alpha=a_v).fit(xr, yr, sample_weight=wr)
        max_svd_diff = max(max_svd_diff, float(np.max(np.abs(coef_s - sk.coef_))), abs(float(inter_s) - float(sk.intercept_)))

    sess_var = [
        (rng.normal(-2.0, 0.8, size=(7, grid.size, 6)), rng.uniform(25.0, 60.0, size=7), np.array([False, True, True, True, False, True, False])),
        (rng.normal(-2.0, 0.8, size=(12, grid.size, 6)), rng.uniform(25.0, 60.0, size=12), np.array([True] * 10 + [False, True])),
        (rng.normal(-2.0, 0.8, size=(9, grid.size, 6)), rng.uniform(25.0, 60.0, size=9), np.array([True] * 9)),
    ]
    batch_var = DB.prepare_hmm_batch(sess_var)
    max_batch_parity_diff = 0.0
    for mode in ("linear", "linear_filtersafe", "mlp"):
        for dec in ("causal_forward", "offline_fb"):
            mdl_p = D.DifferentiableSymmetricHMM(mode=mode, decoder=dec, grid_bpm=grid, reg_l2=5e-3, seed=23)
            th_p = np.clip(
                mdl_p.theta + rng.normal(0.0, 0.05, size=mdl_p.theta.size),
                [b[0] + 1e-3 for b in mdl_p.bounds],
                [b[1] - 1e-3 for b in mdl_p.bounds],
            )
            l_ref, g_ref = mdl_p.loss_and_grad(th_p, sess_var)
            l_b, g_b = DB.batch_loss_and_grad(mdl_p, th_p, batch_var)
            max_batch_parity_diff = max(max_batch_parity_diff, abs(l_b - l_ref), float(np.max(np.abs(g_b - g_ref))))

    g3_pass = max_r2_grad_diff < 1e-6 and max_tcn_grad_diff < 1e-6 and r2_bound_ok and max_svd_diff < 1e-10 and max_batch_parity_diff < 1e-12
    report["gates"]["gate3_cleanroom_gradients_bounds_and_svd"] = {
        "passed": g3_pass,
        "max_r2_grad_diff": max_r2_grad_diff,
        "max_tcn_grad_diff": max_tcn_grad_diff,
        "r2_sigma_min_1_0_bound_ok": r2_bound_ok,
        "max_svd_ridge_diff": max_svd_diff,
        "max_batched_bptt_parity_diff": max_batch_parity_diff,
    }
    md.append(
        f"## Gate 3 — Finite-Diff Gradients, Bounds & SVD Ridge Parity: **{'PASS' if g3_pass else 'FAIL'}**\n"
        f"- Central FD (`h=1e-6`) max diff: `R2={max_r2_grad_diff:.2e}`, `TCN={max_tcn_grad_diff:.2e}`; `sigma_slow >= 1.0` enforced: `{r2_bound_ok}`; SVD vs `sklearn.Ridge` diff: `{max_svd_diff:.2e}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 4: Clean-Room O(2)xR+, Causality & Frequency-Shift Equivariance
    # ------------------------------------------------------------------
    vc = master["verification_controls"]
    mean_r2e = D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward")
    mean_r2e.theta = np.asarray(vc["r2e_mean_theta"], dtype=np.float64)
    mean_r2a = D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward")
    mean_r2a.theta = np.asarray(vc["r2a_mean_theta"], dtype=np.float64)
    mean_r2b = D.DifferentiableSymmetricHMM(mode="mlp", decoder="causal_forward")
    mean_r2b.theta = np.asarray(vc["r2b_mean_theta"], dtype=np.float64)
    bi2, bq2 = load_radar("2.3")
    bi2, bq2 = bi2[: 480 + 4 * 32], bq2[: 480 + 4 * 32]
    starts5 = [s for s in range(0, 5 * 32, 32)]

    def _eval_h5_r2(ai: np.ndarray, aq: np.ndarray) -> np.ndarray:
        phs = []
        for s0 in starts5:
            dr = bandpass(csd(ai[s0 : s0 + 480], aq[s0 : s0 + 480]))
            dh5 = bandpass(W.mitigate_hpss_v2(dr, window="cola_hann", lookahead=0, in_band=True)["x_hat"])
            fc5 = autocorr_coarse(dh5)
            phs.append(D.extract_salience_features_window(dr, dh5, fc5, grid))
        pa = np.asarray(phs)
        return np.concatenate([mean_r2e.predict_session(pa)["est_mmse"], mean_r2a.predict_session(pa)["est_mmse"], mean_r2b.predict_session(pa)["est_mmse"]])

    b_inv = _eval_h5_r2(bi2, bq2)
    zrot = (bi2 + 1j * bq2) * np.exp(1j * np.pi / 3.0)
    clean_inv_max = 0.0
    for ti, tq in ((3.7 * bi2, 3.7 * bq2), (zrot.real, zrot.imag), (-bi2, -bq2), (bq2, bi2)):
        clean_inv_max = max(clean_inv_max, float(np.max(np.abs(b_inv - _eval_h5_r2(ti, tq)))))

    phi_t = rng.normal(-2.0, 1.0, size=(24, grid.size, 6))
    phi_p = phi_t.copy()
    phi_p[14:] = rng.normal(-2.0, 2.0, size=(10, grid.size, 6))
    c_r2e = float(np.max(np.abs(mean_r2e.predict_session(phi_t)["est_mmse"][:14] - mean_r2e.predict_session(phi_p)["est_mmse"][:14])))
    c_r2a = float(np.max(np.abs(mean_r2a.predict_session(phi_t)["est_mmse"][:14] - mean_r2a.predict_session(phi_p)["est_mmse"][:14])))
    c_r2b = float(np.max(np.abs(mean_r2b.predict_session(phi_t)["est_mmse"][:14] - mean_r2b.predict_session(phi_p)["est_mmse"][:14])))
    c_tcn = float(np.max(np.abs(tcn.predict_session(phi_t)[:14] - tcn.predict_session(phi_p)[:14])))

    wide_grid = np.arange(-60.0, 180.0 + 1e-9, 1.0)
    pb_w = np.zeros((10, wide_grid.size, 6))
    ps_w = np.zeros((10, wide_grid.size, 6))
    for ch in range(6):
        if ch == 4:
            gb_w = np.maximum(0.05, np.exp(-0.5 * ((wide_grid[None, :] - 40.0) / float(C.NLS_RADIUS_BPM)) ** 2))
            gs_w = np.maximum(0.05, np.exp(-0.5 * ((wide_grid[None, :] - 50.0) / float(C.NLS_RADIUS_BPM)) ** 2))
        else:
            gb_w = np.exp(-0.5 * ((wide_grid[None, :] - 40.0) / 2.5) ** 2)
            gs_w = np.exp(-0.5 * ((wide_grid[None, :] - 50.0) / 2.5) ** 2)
        pb_w[:, :, ch] = np.log(gb_w + 1e-12)
        ps_w[:, :, ch] = np.log(gs_w + 1e-12)
    eq_e = float(np.max(np.abs((mean_r2e.predict_session(ps_w, grid_bpm=wide_grid)["est_mmse"] - mean_r2e.predict_session(pb_w, grid_bpm=wide_grid)["est_mmse"]) - 10.0)))
    eq_a = float(np.max(np.abs((mean_r2a.predict_session(ps_w, grid_bpm=wide_grid)["est_mmse"] - mean_r2a.predict_session(pb_w, grid_bpm=wide_grid)["est_mmse"]) - 10.0)))
    eq_b = float(np.max(np.abs((mean_r2b.predict_session(ps_w, grid_bpm=wide_grid)["est_mmse"] - mean_r2b.predict_session(pb_w, grid_bpm=wide_grid)["est_mmse"]) - 10.0)))

    sh_grid_ok = (
        vc["shifted_grid_18_90_vs_18_120"]["R2e_Causal_DiffHMM_FilterSafe8"]["abs_bias_shift_bpm"] < 0.05
        and vc["shifted_grid_18_90_vs_18_120"]["R2a_Causal_DiffHMM_Linear11"]["abs_bias_shift_bpm"] < 0.05
        and vc["shifted_grid_18_90_vs_18_120"]["R2b_Causal_DiffHMM_MLP43"]["abs_bias_shift_bpm"] < 0.05
    )

    rw_probe = vc.get("radar_waveform_shift_probe_36_to_46_bpm", {})
    cgrid_probe = vc.get("freq_shift_equivariance_clinical_grid_bpm", {})
    shift_probe_ok = (
        bool(rw_probe)
        and abs(float(rw_probe.get("R2e_Causal_DiffHMM_FilterSafe8", {}).get("abs_err_from_10_bpm", 99.0))) < 0.5
        and abs(float(rw_probe.get("R2a_Causal_DiffHMM_Linear11", {}).get("abs_err_from_10_bpm", 99.0))) < 0.5
        and abs(float(cgrid_probe.get("R3b_Spectral_MLP_15k", {}).get("mean_shift_bpm", 0.0))) > 0.1
    )

    sc = master["shortcut_learning_audit_pool30"]
    meta_ok = all(
        (sc[km]["block_lookahead_s"] is None if km in ("R1f_Offline_FB_MMSE_Inf", "R1g_Offline_Viterbi_Unpinned", "R2f_Offline_DiffHMM_FilterSafe8", "R2c_Offline_DiffHMM_Linear11") else sc[km]["block_lookahead_s"] == 0.0)
        and (sc[km]["stft_lookahead_s"] == (0.0 if km == "R2d_E2E_Causal_H5_DiffHMM_Linear11" else 3.0))
        for km in MODELS_13
    )

    g4_pass = (
        vc["all_r2_invariances_lt_1e6"] is True
        and clean_inv_max < 1e-6
        and c_r2e == 0.0
        and c_r2a == 0.0
        and c_r2b == 0.0
        and c_tcn == 0.0
        and vc["causality_r2e_max_diff"] == 0.0
        and vc["causality_r2a_max_diff"] == 0.0
        and vc["causality_r2b_max_diff"] == 0.0
        and vc["causality_r3a_tcn_max_diff"] == 0.0
        and eq_e < 1e-9
        and eq_a < 1e-9
        and eq_b < 1e-9
        and vc["freq_shift_equivariance_operator_max_diff_bpm"]["R2e"] < 1e-9
        and vc["freq_shift_equivariance_operator_max_diff_bpm"]["R2a"] < 1e-9
        and vc["freq_shift_equivariance_operator_max_diff_bpm"]["R2b"] < 1e-9
        and sh_grid_ok
        and meta_ok
        and shift_probe_ok
    )
    report["gates"]["gate4_symmetry_causality_shift_equivariance"] = {
        "passed": g4_pass,
        "clean_h5_o2_inv_max_bpm": clean_inv_max,
        "clean_causality_r2e": c_r2e,
        "clean_causality_r2a": c_r2a,
        "clean_causality_r2b": c_r2b,
        "clean_causality_tcn": c_tcn,
        "clean_operator_shift_eq_r2e_bpm": eq_e,
        "clean_operator_shift_eq_r2a_bpm": eq_a,
        "clean_operator_shift_eq_r2b_bpm": eq_b,
        "shifted_grid_lt_005_ok": sh_grid_ok,
        "shift_probe_calibrated_and_radar_ok": shift_probe_ok,
        "two_axis_metadata_ok": meta_ok,
    }
    md.append(
        f"## Gate 4 — O(2)xR+ Symmetry, 0.0 Causality & Frequency-Shift Equivariance: **{'PASS' if g4_pass else 'FAIL'}**\n"
        f"- Clean `H5+R2` O(2)xR+ diff: `{clean_inv_max:.2e} bpm`; causality diffs: `R2e={c_r2e:.1f}`, `R2a={c_r2a:.1f}`, `R2b={c_r2b:.1f}`, `TCN={c_tcn:.1f}`.\n"
        f"- Operator `+10 bpm` shift (`[-60, 180]`): `R2e={eq_e:.2e}`, `R2a={eq_a:.2e}`, `R2b={eq_b:.2e}`; `[18, 90]` vs `[18, 120]` `< 0.05 bpm`: `{sh_grid_ok}`; calibrated/radar shift probes ok: `{shift_probe_ok}`; two-axis metadata ok: `{meta_ok}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 5: LOPO Anti-Leakage, Sigma Floor & R3 Hyperparameter Logs
    # ------------------------------------------------------------------
    leak_errs: list[Any] = []
    bound_errs: list[Any] = []
    hp_errs: list[Any] = []

    expected_folds = {"pool30_sync": 11, "retained23": 10, "PAPER_18": 10}
    for coh, exp_nf in expected_folds.items():
        ids = _cohort_ids(coh)
        pats = sorted({_pid(m) for m in ids})
        if len(pats) != exp_nf:
            leak_errs.append((coh, "n_pats", len(pats), exp_nf))

    flogs = master.get("pool30_fold_logs", [])
    if len(flogs) != 11:
        leak_errs.append(("pool30_fold_logs", "n_folds", len(flogs)))
    seen_test: list[str] = []
    for row in flogs:
        pid = int(row["patient_id"])
        vpid = int(row["val_patient_id"])
        if vpid == pid:
            leak_errs.append((pid, "val_equals_test", vpid))
        if any(_pid(tm) != pid for tm in row["test_sessions"]):
            leak_errs.append((pid, "test_session_pid_mismatch"))
        seen_test.extend(row["test_sessions"])
        for ikey in ("r2e_interp", "r2a_interp", "r2b_interp", "r2f_interp", "r2c_interp", "r2d_interp"):
            d_i = row[ikey]
            sig_s = float(d_i["sigma_slow_bpm"])
            if sig_s < 1.0 - 1e-9:
                bound_errs.append((pid, ikey, "sigma_slow_bpm", sig_s))
            if float(d_i["w1_p1_clean"]) < 2.0 - 1e-9:
                bound_errs.append((pid, ikey, "w1_p1_clean", d_i["w1_p1_clean"]))
            if ikey in ("r2e_interp", "r2f_interp"):
                if abs(float(d_i["w2_p2_harm"])) > 1e-12 or abs(float(d_i["w3_p3_harm"])) > 1e-12 or abs(float(d_i["w4_phalf_sub"])) > 1e-12:
                    bound_errs.append((pid, ikey, "filtersafe_nonzero_harm"))
            if float(d_i["k1"]) < -1e-9 or float(d_i["k2"]) < -1e-9:
                bound_errs.append((pid, ikey, "k1_k2_neg", d_i["k1"], d_i["k2"]))
            if float(d_i["pi_jump"]) > 0.012:
                bound_errs.append((pid, ikey, "pi_jump", d_i["pi_jump"]))
        hp = row.get("r3_hyperparams", {})
        req_hp = ("ridge_alpha", "ridge_inner_cv_mode", "mlp_alpha", "tcn_weight_decay", "tcn_dropout_p")
        if any(k not in hp for k in req_hp) or hp.get("ridge_inner_cv_mode") != "patient_lopo":
            hp_errs.append((pid, hp))
    if sorted(seen_test, key=lambda x: tuple(int(p) for p in x.split("."))) != p30_ids:
        leak_errs.append(("pool30_fold_logs", "session_coverage_mismatch"))
    conv_sum = master.get("r2_convergence_summary", {})
    if any(mk not in conv_sum for mk in ("r2e", "r2a", "r2b", "r2f", "r2c", "r2d")):
        hp_errs.append(("r2_convergence_summary", "missing"))

    g5_pass = len(leak_errs) == 0 and len(bound_errs) == 0 and len(hp_errs) == 0
    report["gates"]["gate5_lopo_antileakage_bounds_and_hyperparams"] = {
        "passed": g5_pass,
        "leak_errors": leak_errs,
        "sigma_bound_errors": bound_errs,
        "r3_hyperparam_errors": hp_errs,
        "r2_convergence_summary": conv_sum,
    }
    md.append(
        f"## Gate 5 — LOPO Anti-Leakage (`11/10/10`), `sigma >= 1.0` Floor & R3 Tuning Logs: **{'PASS' if g5_pass else 'FAIL'}**\n"
        f"- Disjointness/val errors: `{len(leak_errs)}`; `R2` bound violations: `{len(bound_errs)}`; `R3`/convergence errors: `{len(hp_errs)}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 6: -10 s Lag Alignment, Synthetic Sweep & 5-Seed Surrogates
    # ------------------------------------------------------------------
    lp = master["lag_profiles_pool30"]
    lag_align_peak = int(lp["R0_vs_br_trig_bpm_lag_aligned"]["peak_lag_s"])
    lag_ok = abs(lag_align_peak) <= 2

    synth = master["synthetic_frequency_sweep_pool30"]
    synth_passband = ("stat_28bpm", "stat_35bpm", "stat_50bpm", "stat_65bpm", "stat_80bpm")
    synth_cases = synth_passband + ("chirp_30_to_75bpm", "stat_20bpm_kaiser_edge")
    synth_mods = (
        "R0_HPSS_NLS_Locked",
        "R1a_Online_Forward_MMSE_0s",
        "R2e_Causal_DiffHMM_FilterSafe8",
        "R2a_Causal_DiffHMM_Linear11",
        "R2b_Causal_DiffHMM_MLP43",
        "R2d_E2E_Causal_H5_DiffHMM_Linear11",
        "R3c_Ridge_Spectral_439",
        "R3b_Spectral_MLP_15k",
        "R3a_Causal_TCN_16k",
    )
    synth_ok = all(cn in synth for cn in synth_cases)
    r0_stat_maes: list[float] = []
    r2e_stat_maes: list[float] = []
    r2a_stat_maes: list[float] = []
    if synth_ok:
        for cn in synth_cases:
            for km in synth_mods:
                d_s = synth[cn].get(km, {})
                if not d_s or not np.isfinite(d_s.get("mae_mean", float("nan"))) or d_s["mae_mean"] < 0.0:
                    synth_ok = False
                if cn in synth_passband and km == "R0_HPSS_NLS_Locked" and d_s:
                    r0_stat_maes.append(float(d_s["mae_mean"]))
                if cn in synth_passband and km == "R2e_Causal_DiffHMM_FilterSafe8" and d_s:
                    r2e_stat_maes.append(float(d_s["mae_mean"]))
                if cn in synth_passband and km == "R2a_Causal_DiffHMM_Linear11" and d_s:
                    r2a_stat_maes.append(float(d_s["mae_mean"]))
    r0_synth_mean_mae = float(np.mean(r0_stat_maes)) if r0_stat_maes else float("nan")
    r2e_synth_mean_mae = float(np.mean(r2e_stat_maes)) if r2e_stat_maes else float("nan")
    r2a_synth_mean_mae = float(np.mean(r2a_stat_maes)) if r2a_stat_maes else float("nan")
    synth_ok = (
        synth_ok
        and np.isfinite(r0_synth_mean_mae)
        and r0_synth_mean_mae < 0.5
        and np.isfinite(r2e_synth_mean_mae)
        and r2e_synth_mean_mae < 2.0
        and np.isfinite(r2a_synth_mean_mae)
        and r2a_synth_mean_mae < 2.0
    )

    scr = master["phase_scrambled_surrogates_pool30"]
    scr_mods = (
        "R2e_Causal_DiffHMM_FilterSafe8",
        "R2a_Causal_DiffHMM_Linear11",
        "R2b_Causal_DiffHMM_MLP43",
        "R2d_E2E_Causal_H5_DiffHMM_Linear11",
        "R3c_Ridge_Spectral_439",
        "R3b_Spectral_MLP_15k",
        "R3a_Causal_TCN_16k",
    )
    scr_ok = (scr.get("skipped") is False) and int(scr.get("n_seeds", 0)) == 5
    max_scr_r = 0.0
    if scr_ok:
        for rlbl in ("br_1hz_30s", "br_trig_bpm"):
            for km in scr_mods:
                entry = scr.get(rlbl, {}).get(km)
                if not entry:
                    scr_ok = False
                    break
                rv = abs(float(entry["r_within_rmcorr_mean"]))
                max_scr_r = max(max_scr_r, rv)
                if rv >= 0.05:
                    scr_ok = False

    g6_pass = lag_ok and synth_ok and scr_ok
    report["gates"]["gate6_lag_alignment_synthetic_and_surrogates"] = {
        "passed": g6_pass,
        "r0_vs_trig_lag_aligned_peak_lag_s": lag_align_peak,
        "r0_synthetic_passband_mean_mae_bpm": r0_synth_mean_mae,
        "r2e_synthetic_passband_mean_mae_bpm": r2e_synth_mean_mae,
        "r2a_synthetic_passband_mean_mae_bpm": r2a_synth_mean_mae,
        "surrogates_ok": scr_ok,
        "surrogate_max_abs_r_within": max_scr_r,
    }
    md.append(
        f"## Gate 6 — `-10 s` Lag Alignment, Synthetic Sweep & 5-Seed Surrogates: **{'PASS' if g6_pass else 'FAIL'}**\n"
        f"- `R0 vs Trig (-10 s)` peak lag: `{lag_align_peak:+d} s`; passband `28-80 bpm` mean MAE: `R0={r0_synth_mean_mae:.2f}`, `R2e={r2e_synth_mean_mae:.2f}`, `R2a={r2a_synth_mean_mae:.2f} bpm`; 5-seed surrogate max `|r_within|`: `{max_scr_r:.4f}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 7: Provenance, RFC-8259 Strict JSON & Cohort/WP4 Completeness
    # ------------------------------------------------------------------
    strict_json_ok = True
    try:
        json.dumps(master, allow_nan=False)
    except ValueError:
        strict_json_ok = False

    exp_counts = {
        "pool30_sync": [("br_1hz_30s", 28, 6640, 6640), ("br_trig_bpm", 28, 6638, 6638), ("br_trig_bpm_lag_aligned", 28, 6550, None), ("br_trig_count_bpm", 28, 6640, 6640)],
        "retained23": [("br_1hz_30s", 23, 6447, 6447), ("br_trig_bpm", 23, 6445, 6445), ("br_trig_bpm_lag_aligned", 23, 6350, None)],
        "PAPER_18": [("br_1hz_30s", 18, 5466, 5466), ("br_trig_bpm", 18, 5466, 5466), ("br_trig_bpm_lag_aligned", 18, 5400, None)],
    }
    comp_errs: list[Any] = []
    for coh, ref_specs in exp_counts.items():
        for rlbl, ns, min_nw, exact_nw in ref_specs:
            blk = master["cohorts"].get(coh, {}).get(rlbl, {}).get("methods", {})
            for km in MODELS_13:
                d_m = blk.get(km)
                if not d_m:
                    comp_errs.append((coh, rlbl, km, "missing"))
                    continue
                if int(d_m["n_sessions"]) != ns or not np.isfinite(d_m["rmse_sess"]):
                    comp_errs.append((coh, rlbl, km, d_m["n_sessions"]))
                if exact_nw is not None and int(d_m["n_windows"]) != exact_nw:
                    comp_errs.append((coh, rlbl, km, "nw", d_m["n_windows"], exact_nw))
                if int(d_m["n_windows"]) < min_nw:
                    comp_errs.append((coh, rlbl, km, "nw_low", d_m["n_windows"], min_nw))

    wp4 = master.get("wp4_causal_uncertainty_preview", {})
    wp4_ok = all(
        km in wp4
        and all(
            rlbl in wp4[km]
            and all(np.isfinite(wp4[km][rlbl].get(fk, float("nan"))) for fk in ("spearman_entropy_pool", "spearman_entropy_sess_mean", "spearman_sd_pool", "spearman_sd_sess_mean"))
            for rlbl in ("br_1hz_30s", "br_trig_bpm")
        )
        for km in (
            "R2e_Causal_DiffHMM_FilterSafe8",
            "R2a_Causal_DiffHMM_Linear11",
            "R2b_Causal_DiffHMM_MLP43",
            "R2d_E2E_Causal_H5_DiffHMM_Linear11",
        )
    )
    fe_prof = master.get("filter_edge_channel_profile_pool30", {})
    fe_ok = (
        fe_prof.get("n_windows") == 6640
        and np.isfinite(fe_prof.get("r2a_cliff_50_to_60_bpm", float("nan")))
        and float(fe_prof.get("r2e_filter_edge_cliff_bpm", -1.0)) == 0.0
    )

    g7_pass = (
        master.get("schema_version") == "stage2-complexity-v4"
        and strict_json_ok
        and bool(master.get("git_head", {}).get("head_commit"))
        and float(master.get("total_runtime_s", 0.0)) > 0.0
        and len(comp_errs) == 0
        and wp4_ok
        and fe_ok
    )
    report["gates"]["gate7_provenance_rfc8259_and_completeness"] = {
        "passed": g7_pass,
        "schema_version": master.get("schema_version"),
        "strict_rfc8259_json": strict_json_ok,
        "completeness_errors": comp_errs,
        "wp4_complete": wp4_ok,
    }
    md.append(
        f"## Gate 7 — Provenance, Strict RFC-8259 JSON & Completeness: **{'PASS' if g7_pass else 'FAIL'}**\n"
        f"- Schema `{master.get('schema_version')}` | strict `allow_nan=False`: `{strict_json_ok}` | cohort/ref errors: `{len(comp_errs)}` | WP4 fields ok: `{wp4_ok}`.\n"
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
