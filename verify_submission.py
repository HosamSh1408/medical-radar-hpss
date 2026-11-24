#!/usr/bin/env python3
"""One-command integrity, test-suite, and headline-metric verifier for examiners.

Usage:
  python3 verify_submission.py                  # Full fresh-clone verification + pytest (35/35)
  python3 verify_submission.py --run-v1-baseline  # Also run Python 3.9.6 locked v1 checks (19/19)
  python3 verify_submission.py --skip-pytest    # Manifest + metrics summary only (< 1 s)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_FROZEN_ARCHIVE_SHA = (
    "af33a83e157f7d2efb0f47e50c9544ee19bacb7da88ab0e0235ccead029cbcc4"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_manifest(manifest_path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        entries.append((digest.strip(), rel.strip()))
    return entries


def check_freeze_manifest() -> tuple[int, int]:
    entries = parse_manifest(ROOT / "audit" / "thesis_freeze_manifest.sha256")
    bad: list[str] = []
    for expected, rel in entries:
        fp = ROOT / rel
        if not fp.is_file() or sha256_file(fp) != expected:
            bad.append(rel)
    if bad:
        raise RuntimeError(f"Freeze manifest mismatch ({len(bad)} bad): {bad[:5]}")
    return len(entries), len(entries)


def check_baseline_subset() -> tuple[int, int, int]:
    """Verify the 154-file paper_workspace/ + beltrao_data/ subset of postfix_manifest.sha256."""
    entries = parse_manifest(ROOT / "audit" / "postfix_manifest.sha256")
    subset = [
        (d, r) for d, r in entries if r.startswith(("paper_workspace/", "beltrao_data/"))
    ]
    checked = 0
    unbundled_raw = 0
    bad: list[str] = []
    for expected, rel in subset:
        fp = ROOT / rel
        if fp.is_file():
            checked += 1
            if sha256_file(fp) != expected:
                bad.append(rel)
        elif rel.startswith(("beltrao_data/radar/", "beltrao_data/reference/")):
            unbundled_raw += 1
        else:
            bad.append(rel)
    if bad:
        raise RuntimeError(f"Baseline manifest mismatch ({len(bad)} bad): {bad[:5]}")
    return checked, len(subset), unbundled_raw


def run_pytest_suite() -> int:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": "/tmp/mpl",
            "OMP_NUM_THREADS": "1",
        }
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--rootdir=.", "-v"],
        cwd=ROOT,
        env=env,
        check=False,
    )
    return int(proc.returncode)


def run_v1_baseline() -> int:
    py39 = Path("/usr/bin/python3")
    exe = str(py39) if py39.exists() else sys.executable
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    clt = Path("/Library/Developer/CommandLineTools")
    if clt.exists():
        env.setdefault("DEVELOPER_DIR", str(clt))
    proc = subprocess.run(
        [exe, "-m", "paper_workspace.src.test_pipeline"],
        cwd=ROOT,
        env=env,
        check=False,
    )
    return int(proc.returncode)


def print_headline_metrics() -> None:
    raw_dir = ROOT / "paper_orchestra_bundle" / "raw_materials"
    s2 = json.loads((raw_dir / "metrics_stage2_complexity.json").read_text(encoding="utf-8"))
    s3 = json.loads((raw_dir / "metrics_stage3_uncertainty.json").read_text(encoding="utf-8"))
    s4 = json.loads((raw_dir / "metrics_stage4_external.json").read_text(encoding="utf-8"))

    print("\n" + "=" * 78)
    print("VERIFIED HEADLINE METRICS (loaded directly from frozen metrics_stage*.json)")
    print("=" * 78)

    print("\n1. Stage 2 NICU LOPO Complexity Ladder (pool30_sync: K=11 infants, S=28 sessions)")
    m_1hz = s2["cohorts"]["pool30_sync"]["br_1hz_30s"]["methods"]
    m_trg = s2["cohorts"]["pool30_sync"]["br_trig_bpm_lag_aligned"]["methods"]
    p_1hz = s2["cohorts"]["pool30_sync"]["br_1hz_30s"]["paired_vs_R0"]
    p_trg = s2["cohorts"]["pool30_sync"]["br_trig_bpm_lag_aligned"]["paired_vs_R0"]
    ladder = [
        ("R0  HPSS+NLS (0p)", "R0_HPSS_NLS_Locked"),
        ("R1a Fwd MMSE (3p)", "R1a_Online_Forward_MMSE_0s"),
        ("R1f Offline FB (3p)", "R1f_Offline_FB_MMSE_Inf"),
        ("R2e FilterSafe-8 (8p)", "R2e_Causal_DiffHMM_FilterSafe8"),
        ("R3c Ridge (439p)", "R3c_Ridge_Spectral_439"),
        ("R3b MLP (14.6k)", "R3b_Spectral_MLP_15k"),
        ("R3a TCN (15.8k)", "R3a_Causal_TCN_16k"),
    ]
    print(f"  {'Model':<24} {'1Hz RMSE':>9} {'p_Holm':>8}   {'Trig-10s RMSE':>13} {'p_Holm':>8}")
    for label, key in ladder:
        ph1 = f"{p_1hz[key]['p_wilcoxon_patient_holm']:.4f}" if key in p_1hz else "baseline"
        pht = f"{p_trg[key]['p_wilcoxon_patient_holm']:.4f}" if key in p_trg else "baseline"
        print(
            f"  {label:<24} {m_1hz[key]['rmse_sess']:9.4f} {ph1:>8}   "
            f"{m_trg[key]['rmse_sess']:13.4f} {pht:>8}"
        )

    print("\n2. Stage 3 Selective UQ, Apnea-Pause Safety & Finite-K Conformal (R2e, 1 Hz)")
    sp = s3["selective_prediction_by_cohort"]["pool30_sync"]["br_1hz_30s"][
        "R2e_Causal_DiffHMM_FilterSafe8"
    ]
    ap = s3["apnea_pause_safety_by_cohort"]["pool30_sync"]["R2e_Causal_DiffHMM_FilterSafe8"]
    ood = s3["ood_abstention_audit_pool30"]["models"]["R2e_Causal_DiffHMM_FilterSafe8"]
    for label, key in [
        ("U2 Posterior SD (0p)", "U2_Posterior_SD"),
        ("U6b Heuristic Barrier", "U6b_ZeroTuned_Geometric_SQI"),
        ("U6c 5p Regularized SQI", "U6c_LOPO_Calibrated_SQI"),
    ]:
        eaurc = sp[key]["session_macro"]["e_aurc_vs_random_rmse_sess"]
        ppat = sp[key]["session_macro"]["p_wilcoxon_patient"]
        rr80 = ap[key]["pooled_threshold_rejection"]["0.80"]["rejection_risk_ratio"]
        auroc = ood[key]["wgn_200_seed20261011"]["auroc"]
        print(
            f"  {label:<24} E-AURC={eaurc:+.4f} (raw p_pat={ppat:.4f}) | "
            f"Pause RR@80%={rr80:.4f}x | WGN AUROC={auroc:.4f}"
        )
    c95 = s3["finite_k_conformal_by_cohort"]["pool30_sync"]["br_1hz_30s"][
        "R2e_Causal_DiffHMM_FilterSafe8"
    ]["by_alpha"]["0.05"]
    c80 = s3["finite_k_conformal_by_cohort"]["pool30_sync"]["br_1hz_30s"][
        "R2e_Causal_DiffHMM_FilterSafe8"
    ]["by_alpha"]["0.20"]["regimes"]["C5_LocallyAdaptive_BayesConformal_Hierarchical_LOPO"]
    print(
        f"  Conformal K_cal=10: 95% vacuous={c95['patient_exchangeable_is_vacuous']} "
        f"(min K_cal={c95['min_cal_patients_required_for_finite_interval']}); "
        f"80% C5 cov={c80['empirical_coverage_patient_mean']:.4f}, "
        f"mean width={c80['mean_interval_width_bpm']:.2f} bpm"
    )

    print("\n3. Stage 4 Zero-Touch External Hardware Transfer (gleichauf2024, 24.2 GHz, L4)")
    inb = s4["regimes"]["steady_state_excl_sync_pulses"]["inband_ge28bpm"]
    skt = s4["regimes"]["steady_state_excl_sync_pulses"]["subband_20bpm_skirt"]
    for label, key in [
        ("R2e FilterSafe-8 (8p)", "R2e_Causal_DiffHMM_FilterSafe8"),
        ("R3c Ridge (439p)", "R3c_Ridge_Spectral_439"),
        ("R3b MLP (14.6k)", "R3b_Spectral_MLP_15k"),
        ("R3a TCN (15.8k)", "R3a_Causal_TCN_16k"),
    ]:
        slp = inb[key]["linear_slope_vs_gt"]
        print(
            f"  {label:<24} in-band={inb[key]['rmse_scen']:7.4f} bpm (slope={slp:.4f}) | "
            f"20-bpm skirt={skt[key]['rmse_scen']:7.4f} bpm"
        )
    u6b_stop = s4["wp4_uncertainty_transfer"]["R2e_Causal_DiffHMM_FilterSafe8"][
        "stopband_05_10bpm_vs_inband_ood"
    ]["U6b_ZeroTuned_Geometric_SQI"]["auroc"]
    c5_ext = s4["conformal_transfer"]["primary_full_k11_frozen_quantiles"][
        "R2e_Causal_DiffHMM_FilterSafe8"
    ]["inband_ge28bpm"]["0.20"]
    print(
        f"  Zero-touch U6b stopband AUROC={u6b_stop:.4f} | "
        f"C5 80% in-band cov={c5_ext['c5_cov_scen_mean']:.4f}, "
        f"width={c5_ext['c5_mean_width_scen_mean_bpm']:.2f} bpm"
    )
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Master's Thesis submission integrity.")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip running pytest.")
    parser.add_argument(
        "--run-v1-baseline",
        action="store_true",
        help="Also run locked v1 paper_workspace.src.test_pipeline (19/19).",
    )
    args = parser.parse_args()

    print("Checking cryptographic manifests and frozen Stage 4 model archive...")
    ok_freeze, tot_freeze = check_freeze_manifest()
    print(f"  [PASS] thesis_freeze_manifest.sha256: {ok_freeze}/{tot_freeze} OK")

    ok_base, tot_base, unbundled = check_baseline_subset()
    if unbundled == 0:
        print(f"  [PASS] postfix_manifest baseline subset: {ok_base}/{tot_base} OK")
    else:
        print(
            f"  [PASS] postfix_manifest tracked baseline: {ok_base}/{tot_base} OK "
            f"({unbundled} optional raw Zenodo CSV/EDF files not bundled in git)"
        )

    arch = ROOT / "thesis" / "outputs" / "stage3_uncertainty" / "frozen_stage4_models.npz"
    if not arch.is_file() or sha256_file(arch) != EXPECTED_FROZEN_ARCHIVE_SHA:
        raise RuntimeError(f"Frozen Stage 4 archive missing or hash mismatch: {arch}")
    print(f"  [PASS] frozen_stage4_models.npz SHA-256: {EXPECTED_FROZEN_ARCHIVE_SHA[:16]}... OK")

    if not args.skip_pytest:
        print("\nRunning Stage 0-4 pytest suite (expected 35/35)...")
        rc = run_pytest_suite()
        if rc != 0:
            raise SystemExit(rc)

    if args.run_v1_baseline:
        print("\nRunning locked v1 baseline suite (expected 19/19)...")
        rc = run_v1_baseline()
        if rc != 0:
            raise SystemExit(rc)

    print_headline_metrics()
    print("\nALL SUBMISSION VERIFICATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
