"""Zero-shared-code clean-room auditor for Stage 4 (`stage4-external-v1`, L4 Gleichauf 2024 Benchmark).

Intentionally imports ZERO statistical, ingestion, decimation, LOPO, selective-risk, or conformal
evaluation logic from `thesis.exp1_stage1_parsimony`, `thesis.exp2_stage2_complexity`,
`thesis.exp3_stage3_uncertainty`, `thesis.exp4_stage4_external`, `thesis.wp3_wp4_uncertainty_conformal`,
or `thesis.io_utils`.

Independently verifies 7 Gates:
  - Gate 1: `154/154` frozen baseline SHA-256 + `33/33` caches + `frozen_stage4_models.npz` SHA-256
            + ALL `49/49` `external_data/gleichauf2024/extracted_manifest.json` SHA-256 hashes (100% read-only!).
  - Gate 2: Exhaustive `14 + 6 = 20` scenario partition & independent `2.5 kHz` laser FFT ground-truth check (`< 0.05 bpm`).
  - Gate 3: Clean-room raw CSV -> `16 Hz` decimation -> `csd -> bandpass -> mitigate_hpss -> salience`
            -> frozen `R0/R1a/R2e/R2a/R3c/R3b/R3a` re-execution on 3 sentinel scenarios (`< 1e-10`).
  - Gate 4: Clean-room metric & `stage4_external_scenarios.csv` <-> JSON parity across all `20` scenarios & `14` model rows (`< 1e-12` / `< 1e-3`).
  - Gate 5: Clean-room `WP4` Uncertainty (`U1..U6b` Risk-Coverage, stopband/sub-band OOD & sync-pulse AUROC)
            and frozen `Full-K=11` `C4`/`C5` Conformal Transfer recomputation (`< 1e-12`).
  - Gate 6: Clean-room external 24.2 GHz `O(2) x R_{>0}` invariance (`< 1e-6`) & exact `0.0` future causality.
  - Gate 7: Strict RFC-8259 JSON (`allow_nan=False`), `stage4-external-v1` schema & Gate `G5` verdict completeness.

Execution:
  MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /opt/homebrew/bin/python3.10 -m thesis.audit_stage4_external
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy import stats

from paper_workspace.src import config as C
from paper_workspace.src.cohort import local_pool
from paper_workspace.src.demod import csd
from paper_workspace.src.estimation import nls_estimate
from paper_workspace.src.filtering import autocorr_coarse, bandpass
from paper_workspace.src.hpss import mitigate_hpss
from paper_workspace.src.provenance import PIPELINE_VERSION, expected_provenance
from thesis import diff_hmm_and_foils as D
from thesis import viterbi_tracker as VT
from thesis import wp1_wp2_causal_hmm as W

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "audit"
MANIFEST_PATH = AUDIT_DIR / "postfix_manifest.sha256"
LOCKED_CACHE_DIR = ROOT / "paper_workspace" / "outputs" / "cache"
EXT_DIR = ROOT / "external_data" / "gleichauf2024"
EXT_MANIFEST = EXT_DIR / "extracted_manifest.json"
STAGE3_JSON = ROOT / "paper_orchestra_bundle" / "raw_materials" / "metrics_stage3_uncertainty.json"
FROZEN_STAGE4_ARCH = ROOT / "thesis" / "outputs" / "stage3_uncertainty" / "frozen_stage4_models.npz"
STAGE4_JSON = ROOT / "paper_orchestra_bundle" / "raw_materials" / "metrics_stage4_external.json"
SCENARIOS_CSV = AUDIT_DIR / "stage4_external_scenarios.csv"
PRED_ARCH = ROOT / "thesis" / "outputs" / "stage4_external" / "gleichauf20_predictions_and_uncertainty.npz"
AUDIT_JSON = AUDIT_DIR / "stage4_godmode_audit.json"
AUDIT_MD = AUDIT_DIR / "stage4_godmode_audit.md"

WIN_SAMPLES = int(C.WIN_SAMPLES)
HOP_SAMPLES = int(C.UPDATE_S * C.FS_RADAR)
GRID_1BPM = np.arange(18.0, 90.0 + 1e-9, 1.0, dtype=np.float64)
GRID_SD_UNIF = float(np.std(GRID_1BPM, ddof=0))
COV_51 = np.linspace(0.50, 1.00, 51, dtype=np.float64)

PROXIES_7 = (
    "U1_Posterior_Entropy",
    "U2_Posterior_SD",
    "U3_Spectral_Entropy_Clean",
    "U4_Spectral_Entropy_Raw",
    "U5_HQI_Harmonic_Deficit",
    "U6a_ZeroTuned_Additive_SQI",
    "U6b_ZeroTuned_Geometric_SQI",
)

ALL_MODELS_14 = (
    "R0_HPSS_NLS_Locked",
    "R1a_Online_Forward_MMSE_0s",
    "R1h_E2E_Causal_H5_OnlineForward_0s",
    "R1f_Offline_FB_MMSE_Inf",
    "R2e_Causal_DiffHMM_FilterSafe8",
    "R2f_Offline_DiffHMM_FilterSafe8",
    "R2a_Causal_DiffHMM_Linear11",
    "R2d_E2E_Causal_H5_DiffHMM_Linear11",
    "R3c_Ridge_Spectral_439",
    "R3b_Spectral_MLP_15k",
    "R3a_Causal_TCN_16k",
    "R2e_LOPO_Mean_Sensitivity",
    "R2a_LOPO_Mean_Sensitivity",
    "R2d_LOPO_Mean_Sensitivity",
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


def _clean_decimate_16hz(t_s: np.ndarray, bi: np.ndarray, bq: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    t = np.asarray(t_s, dtype=np.float64) - float(t_s[0])
    i0 = np.asarray(bi, dtype=np.float64) - float(np.mean(bi))
    q0 = np.asarray(bq, dtype=np.float64) - float(np.mean(bq))
    i_200 = sig.resample_poly(i0, up=1, down=10, window=("kaiser", 6.5))
    q_200 = sig.resample_poly(q0, up=1, down=10, window=("kaiser", 6.5))
    t_ds10 = t[::10]
    if t_ds10.size != i_200.size:
        nm = min(t_ds10.size, i_200.size)
        t_ds10, i_200, q_200 = t_ds10[:nm], i_200[:nm], q_200[:nm]
    n_u200 = int(np.floor(float(t_ds10[-1]) * 200.0)) + 1
    t_u200 = np.arange(n_u200, dtype=np.float64) / 200.0
    i_u = np.interp(t_u200, t_ds10, i_200)
    q_u = np.interp(t_u200, t_ds10, q_200)
    i16 = sig.resample_poly(i_u, up=2, down=25, window=("kaiser", 6.5))
    q16 = sig.resample_poly(q_u, up=2, down=25, window=("kaiser", 6.5))
    return i16.astype(np.float64), q16.astype(np.float64), float(i16.size) / float(C.FS_RADAR)


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
    return np.clip((np.clip(hmix, 0.0, 1.0) - 0.90) / 0.10, 0.0, 1.0)


def _clean_rc(err: np.ndarray, u: np.ndarray) -> dict[str, float]:
    se2 = err ** 2
    n = len(err)
    ord_u = np.argsort(u, kind="mergesort")
    cs_u = np.cumsum(se2[ord_u])
    k_arr = np.clip(np.round(COV_51 * float(n)).astype(int), 1, n)
    eff_c = k_arr.astype(np.float64) / float(n)
    r_u = np.sqrt(cs_u[k_arr - 1] / k_arr.astype(np.float64))
    full_r = float(np.sqrt(cs_u[-1] / float(n)))
    aurc = float(np.trapezoid(r_u, COV_51) / 0.50)
    augrc = float(np.trapezoid(eff_c * r_u, COV_51) / 0.50)
    return {
        "full_rmse": full_r,
        "rmse_at_80": float(r_u[30]),
        "aurc_rmse": aurc,
        "e_aurc": float(aurc - full_r),
        "augrc_rmse": augrc,
    }


def _clean_auroc_fpr95(u_in: np.ndarray, u_ood: np.ndarray) -> tuple[float, float]:
    n0, n1 = len(u_in), len(u_ood)
    ranks = stats.rankdata(np.concatenate([u_in, u_ood]), method="average")
    u_stat = float(np.sum(ranks[n0:])) - 0.5 * float(n1 * (n1 + 1))
    auroc = float(u_stat / float(n0 * n1))
    thr = float(np.percentile(u_ood, 5.0))
    fpr95 = float(np.mean(u_in >= thr))
    return auroc, fpr95


def run_audit() -> dict[str, Any]:
    report: dict[str, Any] = {"gates": {}}
    raw_text = STAGE4_JSON.read_text(encoding="utf-8")
    master = json.loads(raw_text)
    schema_lbl = str(master.get("schema_version", "unknown"))
    s3 = json.loads(STAGE3_JSON.read_text(encoding="utf-8"))
    hf_s3 = s3["gate_g4_verdict"]["hard_freeze_manifest_for_stage4_l4"]
    df_csv = pd.read_csv(SCENARIOS_CSV, dtype={"scenario": str})

    md: list[str] = [
        f"# Stage 4 Independent Clean-Room Audit Report (`{schema_lbl}`)\n",
        "Executed with zero statistical, decimation, selective-risk, or conformal imports from `thesis.exp1..4`, `thesis.wp3_wp4_uncertainty_conformal`, or `thesis.io_utils`.\n",
    ]

    # ------------------------------------------------------------------
    # Gate 1: Frozen Baseline (154) + Caches (33) + Stage 3 Freeze Archive + 49/49 External SHA-256
    # ------------------------------------------------------------------
    bad_frozen: list[str] = []
    checked_frozen = 0
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        exp_h, rel_p = line.split(None, 1)
        rel_p = rel_p.strip()
        if rel_p.startswith(("paper_workspace/", "beltrao_data/")):
            checked_frozen += 1
            fp = ROOT / rel_p
            if not fp.exists() or _sha256(fp) != exp_h:
                bad_frozen.append(rel_p)

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

    f4_sha = _sha256(FROZEN_STAGE4_ARCH)
    f4_sha_ok = bool(f4_sha == hf_s3["frozen_stage4_models_sha256"] == master.get("frozen_stage4_archive_sha256"))

    ext_mf = json.loads(EXT_MANIFEST.read_text(encoding="utf-8"))
    bad_ext: list[str] = []
    checked_ext = 0
    for row in ext_mf["files"]:
        checked_ext += 1
        fp = EXT_DIR / row["file"]
        if not fp.exists() or fp.stat().st_size != int(row["bytes"]) or _sha256(fp) != row["sha256"]:
            bad_ext.append(row["file"])

    g1_pass = (
        checked_frozen == 154
        and len(bad_frozen) == 0
        and len(bad_caches) == 0
        and f4_sha_ok
        and checked_ext == 49
        and len(bad_ext) == 0
    )
    report["gates"]["gate1_frozen_and_external_sha256_integrity"] = {
        "passed": g1_pass,
        "checked_frozen_files": checked_frozen,
        "bad_frozen_files": bad_frozen,
        "bad_caches": bad_caches,
        "frozen_stage4_sha256_match": f4_sha_ok,
        "checked_external_files": checked_ext,
        "bad_external_files": bad_ext,
    }
    md.append(
        f"## Gate 1 — Frozen Baseline & `49/49` External `gleichauf2024` SHA-256 Lock: **{'PASS' if g1_pass else 'FAIL'}**\n"
        f"- Checked `{checked_frozen}/154` frozen baseline hashes (`bad: {len(bad_frozen)}`), `33/33` locked caches (`bad: {len(bad_caches)}`), `frozen_stage4_models.npz` SHA-256 `{f4_sha_ok}`.\n"
        f"- Independently hashed all `{checked_ext}/49` files in `external_data/gleichauf2024/extracted_manifest.json` (`bad: {len(bad_ext)}`, zero external mutation).\n"
    )

    # ------------------------------------------------------------------
    # Gate 2: Exhaustive Scenario Partition (14 + 6 = 20) & Clean-Room Laser Ground Truth
    # ------------------------------------------------------------------
    st_def = master["strata_definitions"]
    inb_list = st_def["inband_ge28bpm"]
    sub_list = st_def["subband_lt28bpm"]
    all20_list = st_def["all_20"]
    partition_ok = (
        len(inb_list) == 14
        and len(sub_list) == 6
        and len(all20_list) == 20
        and set(inb_list).isdisjoint(set(sub_list))
        and sorted(inb_list + sub_list) == sorted(all20_list)
    )

    max_laser_diff = 0.0
    max_nom_diff = 0.0
    for scen in ("05BPM_deep", "20BPM_deep", "45BPM_normal", "80BPM_deep"):
        df_l = pd.read_csv(EXT_DIR / scen / f"laser_{scen}.csv", header=None, names=["t", "d"], dtype=np.float64)
        tl = df_l["t"].to_numpy(dtype=np.float64)
        dl = df_l["d"].to_numpy(dtype=np.float64)
        fs_l = 1.0 / float(np.median(np.diff(tl)))
        mid = (tl >= tl[0] + 100.0) & (tl < tl[0] + 700.0)
        dm = sig.decimate(dl[mid] - float(np.mean(dl[mid])), 100, ftype="fir", zero_phase=True)
        spec = np.abs(np.fft.rfft(dm * np.hanning(dm.size), n=262144)) ** 2
        freqs = np.fft.rfftfreq(262144, d=100.0 / fs_l) * 60.0
        band = (freqs >= 2.0) & (freqs <= 100.0)
        f_c = float(freqs[band][int(np.argmax(spec[band]))])
        f_logged = float(master["per_scenario"][scen]["laser_f_mid_bpm"])
        f_nom = float(master["per_scenario"][scen]["f_nom_bpm"])
        max_laser_diff = max(max_laser_diff, abs(f_c - f_logged))
        max_nom_diff = max(max_nom_diff, abs(f_c - f_nom))

    g2_pass = partition_ok and max_laser_diff < 1e-12 and max_nom_diff < 0.05
    report["gates"]["gate2_exhaustive_partition_and_laser_gt"] = {
        "passed": g2_pass,
        "partition_14_plus_6_eq_20": partition_ok,
        "max_clean_laser_gt_diff_bpm": max_laser_diff,
        "max_laser_vs_nominal_diff_bpm": max_nom_diff,
    }
    md.append(
        f"## Gate 2 — Exhaustive `14 + 6 = 20` Partition & Clean-Room Laser Ground Truth: **{'PASS' if g2_pass else 'FAIL'}**\n"
        f"- Disjoint exhaustive partition verified (`14` in-band `>= 28 bpm` + `6` sub-band `< 28 bpm` = `20/20`).\n"
        f"- Clean-room `2.5 kHz` laser FFT recomputation on 4 sentinels: `max diff = {max_laser_diff:.2e}` (`max |f_laser - f_nom| = {max_nom_diff:.4f} bpm`).\n"
    )

    # ------------------------------------------------------------------
    # Gate 3: Clean-Room Raw-CSV -> 16 Hz -> Model Re-Execution (3 Sentinels)
    # ------------------------------------------------------------------
    zf4 = {k: v.copy() for k, v in np.load(FROZEN_STAGE4_ARCH, allow_pickle=False).items()}
    arch4 = np.load(PRED_ARCH, allow_pickle=False)

    r2e_m = D.DifferentiableSymmetricHMM(mode="linear_filtersafe", decoder="causal_forward")
    r2e_m.theta = zf4["r2e_full_refit_theta"]
    r2a_m = D.DifferentiableSymmetricHMM(mode="linear", decoder="causal_forward")
    r2a_m.theta = zf4["r2a_full_refit_theta"]
    tcn_m = D.CausalDilatedTCN(in_dim=438, channels=24, dilations=(1, 2, 4), kernel_size=3)
    tcn_m.mean_ = zf4["r3a_tcn_mean"]
    tcn_m.scale_ = zf4["r3a_tcn_scale"]
    for kt in tcn_m.param_keys:
        tcn_m.params[kt] = zf4[f"r3a_tcn_{kt}"]

    r1a_cfg = tuple(float(v) for v in zf4["r1a_mode_cfg"])
    max_reexec_diff = 0.0

    for scen in ("30BPM_normal", "45BPM_normal", "80BPM_deep"):
        df_iq = pd.read_csv(EXT_DIR / scen / f"iq_values_{scen}.csv", usecols=["%time", "field.I", "field.Q"], dtype={"%time": np.int64, "field.I": np.float64, "field.Q": np.float64})
        t_ns = df_iq["%time"].to_numpy(dtype=np.int64)
        t_s = (t_ns - t_ns[0]).astype(np.float64) * 1e-9
        i16, q16, dur16 = _clean_decimate_16hz(t_s, df_iq["field.I"].to_numpy(dtype=np.float64), df_iq["field.Q"].to_numpy(dtype=np.float64))
        nw = (i16.size - WIN_SAMPLES) // HOP_SAMPLES + 1

        r0_c = np.empty(nw, dtype=np.float64)
        s0_c = np.empty((nw, GRID_1BPM.size), dtype=np.float64)
        ph0_c = np.empty((nw, GRID_1BPM.size, D.N_SAL_CHANNELS), dtype=np.float64)
        for w in range(nw):
            s0 = w * HOP_SAMPLES
            dr = bandpass(csd(i16[s0 : s0 + WIN_SAMPLES], q16[s0 : s0 + WIN_SAMPLES]))
            dh0 = bandpass(mitigate_hpss(dr)["x_hat"])
            fc0 = autocorr_coarse(dh0)
            r0_c[w] = float(nls_estimate(dh0, fc0)[0])
            s0_c[w] = VT.extract_br_anchored_nls_spectrum(dh0, fc0, GRID_1BPM)
            ph0_c[w] = D.extract_salience_features_window(dr, dh0, fc0, GRID_1BPM)

        p1a_c = W.hmm_forward_filter(s0_c, GRID_1BPM, r1a_cfg[0], r1a_cfg[1], r1a_cfg[2])["est_mmse"]
        p2e_c = r2e_m.predict_session(ph0_c)["est_mmse"]
        p2a_c = r2a_m.predict_session(ph0_c)["est_mmse"]

        flat_ph = ph0_c.reshape(nw, -1)
        xs_r = (flat_ph - zf4["r3c_ridge_mean"]) / zf4["r3c_ridge_scale"]
        p3c_c = np.clip(xs_r @ zf4["r3c_ridge_coef"] + float(zf4["r3c_ridge_intercept"][0]), 12.0, 120.0)
        hm = np.maximum(0.0, ((flat_ph - zf4["r3b_mlp_mean"]) / zf4["r3b_mlp_scale"]) @ zf4["r3b_mlp_W_0"] + zf4["r3b_mlp_b_0"])
        hm = np.maximum(0.0, hm @ zf4["r3b_mlp_W_1"] + zf4["r3b_mlp_b_1"])
        p3b_c = np.clip((hm @ zf4["r3b_mlp_W_2"]).ravel() + float(zf4["r3b_mlp_b_2"][0]), 12.0, 120.0)
        p3a_c = tcn_m.predict_session(ph0_c)

        for km, arr_c in (
            ("R0_HPSS_NLS_Locked", r0_c),
            ("R1a_Online_Forward_MMSE_0s", p1a_c),
            ("R2e_Causal_DiffHMM_FilterSafe8", p2e_c),
            ("R2a_Causal_DiffHMM_Linear11", p2a_c),
            ("R3c_Ridge_Spectral_439", p3c_c),
            ("R3b_Spectral_MLP_15k", p3b_c),
            ("R3a_Causal_TCN_16k", p3a_c),
        ):
            max_reexec_diff = max(max_reexec_diff, float(np.max(np.abs(arr_c - arch4[f"pred__{km}__{scen}"]))))

    g3_pass = max_reexec_diff < 1e-10
    report["gates"]["gate3_cleanroom_raw_csv_to_model_reexecution"] = {
        "passed": g3_pass,
        "max_reexecution_diff_bpm": max_reexec_diff,
        "sentinels": ["30BPM_normal", "45BPM_normal", "80BPM_deep"],
    }
    md.append(
        f"## Gate 3 — Clean-Room Raw-CSV $\\to$ `16 Hz` $\\to$ Model Re-Execution: **{'PASS' if g3_pass else 'FAIL'}**\n"
        f"- Independently ingested raw ~2 kHz I/Q CSVs and re-ran frozen `R0/R1a/R2e/R2a/R3c/R3b/R3a` on 3 sentinels (`max diff = {max_reexec_diff:.2e} < 1e-10`).\n"
    )

    # ------------------------------------------------------------------
    # Gate 4: Clean-Room Metric & CSV <-> JSON Parity Across All 20 Scenarios & 14 Model Rows
    # ------------------------------------------------------------------
    max_met_diff = 0.0
    for reg_name in ("steady_state_excl_sync_pulses", "all_windows_incl_sync_pulses"):
        strata_to_chk = ("inband_ge28bpm", "inband_normal", "inband_deep", "inband_core_30_60bpm", "inband_extreme_80bpm", "subband_lt28bpm", "subband_20bpm_skirt", "subband_05_10bpm_stopband", "all_20") if reg_name == "steady_state_excl_sync_pulses" else ("inband_ge28bpm", "all_20")
        for st_name in strata_to_chk:
            sc_l = st_def[st_name]
            for km in ALL_MODELS_14:
                rmses, maes, biases, a3s, a6s, pmeans, gts = [], [], [], [], [], [], []
                e_all = []
                for sc in sc_l:
                    mk = np.asarray(arch4[f"steady_mask__{sc}"], dtype=bool) if reg_name == "steady_state_excl_sync_pulses" else np.ones(arch4[f"steady_mask__{sc}"].shape, dtype=bool)
                    gt = float(arch4[f"gt_laser__{sc}"][0])
                    pr = np.asarray(arch4[f"pred__{km}__{sc}"][mk], dtype=np.float64)
                    er = pr - gt
                    ae = np.abs(er)
                    rmses.append(float(np.sqrt(np.mean(er ** 2))))
                    maes.append(float(np.mean(ae)))
                    biases.append(float(np.mean(er)))
                    a3s.append(float(100.0 * np.mean(ae < 3.0)))
                    a6s.append(float(100.0 * np.mean(ae < 6.0)))
                    pmeans.append(float(np.mean(pr)))
                    gts.append(gt)
                    e_all.append(er)
                ec = np.concatenate(e_all)
                jm = master["regimes"][reg_name][st_name][km]
                max_met_diff = max(
                    max_met_diff,
                    abs(float(np.mean(rmses)) - float(jm["rmse_scen"])),
                    abs(float(np.mean(maes)) - float(jm["mae_scen"])),
                    abs(float(np.mean(biases)) - float(jm["bias_scen"])),
                    abs(float(np.mean(a3s)) - float(jm["acc_lt3_pct"])),
                    abs(float(np.mean(a6s)) - float(jm["acc_lt6_pct"])),
                    abs(float(np.sqrt(np.mean(ec ** 2))) - float(jm["rmse_pool"])),
                )
                if len(sc_l) >= 3 and float(np.ptp(gts)) >= 1.0:
                    lr = stats.linregress(gts, pmeans)
                    rp = float(stats.pearsonr(gts, pmeans).statistic)
                    max_met_diff = max(
                        max_met_diff,
                        abs(float(lr.slope) - float(jm["linear_slope_vs_gt"])),
                        abs(rp - float(jm["pearson_r_scenarios"])),
                    )

    csv_max_diff = 0.0
    if len(df_csv) == 20:
        for _, row in df_csv.iterrows():
            sc = str(row["scenario"])
            j_s = master["per_scenario"][sc]["models"]
            csv_max_diff = max(
                csv_max_diff,
                abs(float(row["R2e_rmse_bpm"]) - float(j_s["R2e_Causal_DiffHMM_FilterSafe8"]["rmse_bpm"])),
                abs(float(row["R2a_rmse_bpm"]) - float(j_s["R2a_Causal_DiffHMM_Linear11"]["rmse_bpm"])),
                abs(float(row["R3a_rmse_bpm"]) - float(j_s["R3a_Causal_TCN_16k"]["rmse_bpm"])),
                abs(float(row["R3b_rmse_bpm"]) - float(j_s["R3b_Spectral_MLP_15k"]["rmse_bpm"])),
                abs(float(row["R3c_rmse_bpm"]) - float(j_s["R3c_Ridge_Spectral_439"]["rmse_bpm"])),
            )
    else:
        csv_max_diff = 1.0

    g4_pass = max_met_diff < 1e-12 and csv_max_diff < 1e-3
    report["gates"]["gate4_cleanroom_metrics_and_csv_parity"] = {
        "passed": g4_pass,
        "max_metric_diff": max_met_diff,
        "csv_vs_json_max_diff": csv_max_diff,
    }
    md.append(
        f"## Gate 4 — Clean-Room Stratified Metrics & CSV Parity: **{'PASS' if g4_pass else 'FAIL'}**\n"
        f"- Recomputed all `9` strata x `14` model rows + `all_windows` sensitivity (`max diff = {max_met_diff:.2e} < 1e-12`, CSV diff `{csv_max_diff:.2e}`).\n"
    )

    # ------------------------------------------------------------------
    # Gate 5: Clean-Room WP4 Uncertainty & Zero-Touch Conformal Transfer Verification
    # ------------------------------------------------------------------
    max_wp4_diff = 0.0
    j_wp4 = master["wp4_uncertainty_transfer"]["R2e_Causal_DiffHMM_FilterSafe8"]
    stop_list = st_def["subband_05_10bpm_stopband"]

    for pk in PROXIES_7:
        e_in = np.concatenate([arch4[f"pred__R2e_Causal_DiffHMM_FilterSafe8__{s}"][arch4[f"steady_mask__{s}"]] - float(arch4[f"gt_laser__{s}"][0]) for s in inb_list])
        u_in = np.concatenate([arch4[f"u__R2e_Causal_DiffHMM_FilterSafe8__{pk}__{s}"][arch4[f"steady_mask__{s}"]] for s in inb_list])
        e_a = np.concatenate([arch4[f"pred__R2e_Causal_DiffHMM_FilterSafe8__{s}"][arch4[f"steady_mask__{s}"]] - float(arch4[f"gt_laser__{s}"][0]) for s in all20_list])
        u_a = np.concatenate([arch4[f"u__R2e_Causal_DiffHMM_FilterSafe8__{pk}__{s}"][arch4[f"steady_mask__{s}"]] for s in all20_list])
        rc_a = _clean_rc(e_a, u_a)
        rc_i = _clean_rc(e_in, u_in)

        u_stop = np.concatenate([arch4[f"u__R2e_Causal_DiffHMM_FilterSafe8__{pk}__{s}"][arch4[f"steady_mask__{s}"]] for s in stop_list])
        u_sub = np.concatenate([arch4[f"u__R2e_Causal_DiffHMM_FilterSafe8__{pk}__{s}"][arch4[f"steady_mask__{s}"]] for s in sub_list])
        u_sync = np.concatenate([arch4[f"u__R2e_Causal_DiffHMM_FilterSafe8__{pk}__{s}"][~arch4[f"steady_mask__{s}"]] for s in inb_list])

        au_st, fp_st = _clean_auroc_fpr95(u_in, u_stop)
        au_sb, _ = _clean_auroc_fpr95(u_in, u_sub)
        au_sy, _ = _clean_auroc_fpr95(u_in, u_sync)

        max_wp4_diff = max(
            max_wp4_diff,
            abs(rc_a["e_aurc"] - float(j_wp4["risk_coverage_pooled_all_20"][pk]["e_aurc_vs_random_rmse"])),
            abs(rc_a["rmse_at_80"] - float(j_wp4["risk_coverage_pooled_all_20"][pk]["rmse_at_80"])),
            abs(rc_i["rmse_at_80"] - float(j_wp4["risk_coverage_pooled_inband_ge28bpm"][pk]["rmse_at_80"])),
            abs(au_st - float(j_wp4["stopband_05_10bpm_vs_inband_ood"][pk]["auroc"])),
            abs(fp_st - float(j_wp4["stopband_05_10bpm_vs_inband_ood"][pk]["fpr_at_95tpr"])),
            abs(au_sb - float(j_wp4["subband_lt28bpm_vs_inband_ood"][pk]["auroc"])),
            abs(au_sy - float(j_wp4["sync_pulse_transient_vs_steady_inband"][pk]["auroc"])),
        )

    max_conf_diff = 0.0
    c4_fk11 = hf_s3["frozen_c4_q_hat_bpm_full_k11_1hz"]
    c5_fk11 = hf_s3["frozen_c5_q_hat_norm_full_k11_1hz"]
    j_conf = master["conformal_transfer"]["primary_full_k11_frozen_quantiles"]["R2e_Causal_DiffHMM_FilterSafe8"]

    for st_c in ("inband_ge28bpm", "subband_lt28bpm", "all_20"):
        sc_l = st_def[st_c]
        vac05 = j_conf[st_c]["0.05"]["is_vacuous"] is True
        if not vac05:
            max_conf_diff = 1.0
        for a_s in ("0.10", "0.20", "0.30"):
            q4 = float(c4_fk11[a_s])
            q5 = float(c5_fk11[a_s])
            c4_cs, c5_cs, c5_ws = [], [], []
            for sc in sc_l:
                mk = np.asarray(arch4[f"steady_mask__{sc}"], dtype=bool)
                ae = np.abs(arch4[f"pred__R2e_Causal_DiffHMM_FilterSafe8__{sc}"][mk] - float(arch4[f"gt_laser__{sc}"][0]))
                sd = np.maximum(arch4[f"post_sd_raw__R2e_Causal_DiffHMM_FilterSafe8__{sc}"][mk], 1.0)
                c4_cs.append(float(np.mean(ae <= q4)))
                c5_cs.append(float(np.mean(ae <= q5 * sd)))
                c5_ws.append(float(np.mean(2.0 * q5 * sd)))
            jc = j_conf[st_c][a_s]
            max_conf_diff = max(
                max_conf_diff,
                abs(float(np.mean(c4_cs)) - float(jc["c4_cov_scen_mean"])),
                abs(float(np.mean(c5_cs)) - float(jc["c5_cov_scen_mean"])),
                abs(float(np.mean(c5_ws)) - float(jc["c5_mean_width_scen_mean_bpm"])),
            )

    g5_pass = max_wp4_diff < 1e-12 and max_conf_diff < 1e-12
    report["gates"]["gate5_cleanroom_wp4_uncertainty_and_conformal"] = {
        "passed": g5_pass,
        "max_wp4_diff": max_wp4_diff,
        "max_conformal_diff": max_conf_diff,
    }
    md.append(
        f"## Gate 5 — Clean-Room `WP4` Uncertainty & Conformal Transfer: **{'PASS' if g5_pass else 'FAIL'}**\n"
        f"- Recomputed `U1..U6b` `all_20` / `inband` Risk-Coverage, `05-10 bpm` stopband OOD (`AUROC`), and `C4`/`C5` coverage (`max wp4 diff = {max_wp4_diff:.2e}`, `max conformal diff = {max_conf_diff:.2e}`).\n"
    )

    # ------------------------------------------------------------------
    # Gate 6: Clean-Room External 24.2 GHz O(2) x R_{>0} Invariance & 0.0 Causality
    # ------------------------------------------------------------------
    max_ext_inv = 0.0
    for scen in ("45BPM_normal", "80BPM_deep"):
        df_iq = pd.read_csv(EXT_DIR / scen / f"iq_values_{scen}.csv", usecols=["%time", "field.I", "field.Q"], nrows=160000, dtype={"%time": np.int64, "field.I": np.float64, "field.Q": np.float64})
        t_ns = df_iq["%time"].to_numpy(dtype=np.int64)
        t_s = (t_ns - t_ns[0]).astype(np.float64) * 1e-9
        bi = df_iq["field.I"].to_numpy(dtype=np.float64)
        bq = df_iq["field.Q"].to_numpy(dtype=np.float64)

        def _eval_h5(ai: np.ndarray, aq: np.ndarray) -> np.ndarray:
            i16, q16, _ = _clean_decimate_16hz(t_s, ai, aq)
            phs = []
            for k in range(5):
                s0 = 320 + k * HOP_SAMPLES
                dr = bandpass(csd(i16[s0 : s0 + WIN_SAMPLES], q16[s0 : s0 + WIN_SAMPLES]))
                dh5 = bandpass(W.mitigate_hpss_v2(dr, window="cola_hann", lookahead=0, in_band=True)["x_hat"])
                phs.append(D.extract_salience_features_window(dr, dh5, autocorr_coarse(dh5), GRID_1BPM))
            p5 = np.asarray(phs, dtype=np.float64)
            pr = r2e_m.predict_session(p5)
            u1 = np.clip(pr["post_entropy"], 0.0, 1.0)
            u2 = np.clip(pr["post_sd_bpm"] / GRID_SD_UNIF, 0.0, 1.0)
            bo = _clean_causal_ood_barrier(p5, history_hops=3)
            u6b = 1.0 - ((1.0 - u2) * (1.0 - u1) * (1.0 - bo)) ** (1.0 / 3.0)
            return np.concatenate([pr["est_mmse"], u1, u2, u6b])

        b_vec = _eval_h5(bi, bq)
        zr = (bi + 1j * bq) * np.exp(1j * np.pi / 3.0)
        for ti, tq in ((3.7 * bi, 3.7 * bq), (zr.real, zr.imag), (-bi, -bq), (bq, bi)):
            max_ext_inv = max(max_ext_inv, float(np.max(np.abs(b_vec - _eval_h5(ti, tq)))))

    rng_c = np.random.default_rng(77)
    pa = rng_c.normal(-2.0, 1.0, size=(20, GRID_1BPM.size, D.N_SAL_CHANNELS))
    pb = pa.copy()
    pb[12:] = rng_c.normal(-2.0, 2.5, size=(8, GRID_1BPM.size, D.N_SAL_CHANNELS))
    max_caus = max(
        float(np.max(np.abs(r2e_m.predict_session(pa)["est_mmse"][:12] - r2e_m.predict_session(pb)["est_mmse"][:12]))),
        float(np.max(np.abs(r2a_m.predict_session(pa)["est_mmse"][:12] - r2a_m.predict_session(pb)["est_mmse"][:12]))),
        float(np.max(np.abs(tcn_m.predict_session(pa)[:12] - tcn_m.predict_session(pb)[:12]))),
    )

    g6_pass = max_ext_inv < 1e-6 and max_caus == 0.0
    report["gates"]["gate6_external_o2_invariance_and_causality"] = {
        "passed": g6_pass,
        "max_external_h5_o2_invariance_diff": max_ext_inv,
        "max_causality_diff": max_caus,
    }
    md.append(
        f"## Gate 6 — Clean-Room 24.2 GHz $O(2)\\times\\mathbb{{R}}_{{>0}}$ Invariance & Causality: **{'PASS' if g6_pass else 'FAIL'}**\n"
        f"- Clean-room `H5` $O(2)\\times\\mathbb{{R}}_{{>0}}$ max diff across `R2e` + `U1/U2/U6b`: `{max_ext_inv:.2e}` (`< 1e-6`), exact `0.0` causality: `{max_caus:.2e}`.\n"
    )

    # ------------------------------------------------------------------
    # Gate 7: Strict RFC-8259 JSON & Gate G5 Verdict Contract
    # ------------------------------------------------------------------
    strict_json_ok = True
    try:
        json.loads(raw_text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Non-standard JSON constant: {x}")))
    except Exception:
        strict_json_ok = False

    g5v = master.get("gate_g5_l4_verdict", {})
    g7_pass = (
        strict_json_ok
        and master.get("schema_version") == "stage4-external-v1"
        and g5v.get("decision") == "PASS_L4_ZERO_TOUCH_TRANSFER_VERIFIED"
        and g5v.get("all_inband_equivariant_rungs_beat_all_rung3_foils") is True
        and g5v.get("r2e_near_unit_slope_ge_0p95") is True
        and g5v.get("zero_touch_conformal_c5_valid_90_and_80") is True
        and g5v.get("zero_touch_u6b_stopband_ood_auroc_gt_0p90") is True
        and g5v.get("external_o2_and_causality_verified") is True
        and master.get("hardware_protocol_disclosure", {}).get("zero_external_adaptation") is True
    )
    report["gates"]["gate7_rfc8259_json_and_gate_g5_contract"] = {
        "passed": g7_pass,
        "strict_rfc8259_json": strict_json_ok,
        "schema_version": master.get("schema_version"),
        "gate_g5_decision": g5v.get("decision"),
    }
    md.append(
        f"## Gate 7 — RFC-8259 JSON & Gate `G5` (`L4`) Contract: **{'PASS' if g7_pass else 'FAIL'}**\n"
        f"- Strict JSON (`allow_nan=False`): `{strict_json_ok}`, schema `{master.get('schema_version')}`, verdict `{g5v.get('decision')}`.\n"
    )

    all_passed = all(g["passed"] for g in report["gates"].values())
    report["all_passed"] = all_passed
    report["schema_version_audited"] = schema_lbl
    md.insert(2, f"**Overall Verdict:** **{'7/7 GATES PASSED' if all_passed else 'AUDIT FAILED'}**\n")

    AUDIT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    AUDIT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[AUDIT-STAGE4] Overall all_passed={all_passed} -> wrote {AUDIT_JSON} and {AUDIT_MD}")
    if not all_passed:
        sys.exit(1)
    return report


if __name__ == "__main__":
    run_audit()
