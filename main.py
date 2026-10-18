#!/usr/bin/env python3
"""Unified CLI and Runner for Medical Radar Reproduction & HPSS Extension.

Academic Project: Contactless 24 GHz CW Radar Breathing Monitoring in the NICU.
1. Reproduction of Beltrão et al. (Sci Rep 2022 12:5150).
2. HPSS (Harmonic-Percussive Source Separation) Extension.

Usage:
  python3 main.py --demo         # Run session 7.3 A/B demo (NMF vs HPSS)
  python3 main.py --test         # Run complete test suite (unit + regression)
  python3 main.py --batch        # Run full cohort batch (cached, provenance-locked)
  python3 main.py --tables       # Print paper scenario tables (Fig. 6 & Fig. 7)
  python3 main.py --multiverse   # Build cohort ledger & 4-variant multiverse CSVs
  python3 main.py --plots        # Generate all publication & comparative figures
  python3 main.py --all          # Run tests, demo, tables, multiverse, and plots
"""
import argparse
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_cmd(cmd_list, description):
    print(f"\n{'='*70}\n>> {description}\n{'='*70}")
    res = subprocess.run([PYTHON] + cmd_list, cwd=ROOT)
    if res.returncode != 0:
        print(f"Error: {description} failed with return code {res.returncode}")
        return False
    return True


def run_tests():
    return run_cmd(["-m", "paper_workspace.src.test_pipeline"], "Running Pipeline Test Suite")


def run_demo():
    return run_cmd(["-m", "paper_workspace.src.compare_ab"], "Running Anchor Session 7.3 A/B Demo")


def run_batch(force=False):
    args = ["-m", "paper_workspace.src.run_batch"]
    if force:
        args.append("--force")
    return run_cmd(args, "Processing Cohort Batch")


def run_tables():
    return run_cmd(["-m", "paper_workspace.src.result_tables"], "Generating Benchmark Tables (Fig. 6 & Fig. 7)")


def run_multiverse():
    ok1 = run_cmd(["-m", "paper_workspace.src.build_metadata"], "Building Cohort SHA-256 Evidence Ledger")
    ok2 = run_cmd(["-m", "paper_workspace.src.reconstruction_multiverse"], "Evaluating 4-Variant Cohort Multiverse")
    return ok1 and ok2


def run_plots():
    ok1 = run_cmd(["-m", "paper_workspace.src.plot_figs"], "Generating Beltrão Reproduction Figures")
    ok2 = run_cmd(["-m", "paper_workspace.src.plot_nmf_vs_hpss"], "Generating NMF vs HPSS Comparative Visuals")
    return ok1 and ok2


def main():
    parser = argparse.ArgumentParser(
        description="Medical Radar Reproduction & HPSS Extension CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--test", action="store_true", help="Run automated unit test suite")
    parser.add_argument("--demo", action="store_true", help="Run session 7.3 A/B comparison demo")
    parser.add_argument("--batch", action="store_true", help="Run batch processing across cohort")
    parser.add_argument("--force", action="store_true", help="Force recompute batch cache")
    parser.add_argument("--tables", action="store_true", help="Display scenario benchmark tables")
    parser.add_argument("--multiverse", action="store_true", help="Build cohort ledger & 4-variant multiverse CSVs")
    parser.add_argument("--plots", action="store_true", help="Generate all publication-grade figures")
    parser.add_argument("--all", action="store_true", help="Execute complete pipeline (test, demo, tables, multiverse, plots)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.all:
        print("\n🚀 Executing Complete Pipeline End-to-End...")
        if not run_tests():
            sys.exit(1)
        ok = run_demo()
        ok = run_batch(force=args.force) and ok
        ok = run_tables() and ok
        ok = run_multiverse() and ok
        ok = run_plots() and ok
        if not ok:
            print("\nPipeline failed; no success message emitted.")
            sys.exit(1)
        print("\n🎉 Complete pipeline execution finished successfully!")
        return

    if args.test:
        if not run_tests():
            sys.exit(1)
    if args.demo:
        if not run_demo():
            sys.exit(1)
    if args.batch:
        if not run_batch(force=args.force):
            sys.exit(1)
    if args.tables:
        if not run_tables():
            sys.exit(1)
    if args.multiverse:
        if not run_multiverse():
            sys.exit(1)
    if args.plots:
        if not run_plots():
            sys.exit(1)


if __name__ == "__main__":
    main()

