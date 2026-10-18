# Contactless Neonatal Radar Respiration Monitoring: Beltrão Reimplementation & HPSS Extension

[![Pipeline Version](https://img.shields.io/badge/pipeline-main--v3--audit--fix--2-blue)](#reproducibility--execution-guide)
[![Tests](https://img.shields.io/badge/tests-15%2F15%20passing-brightgreen)](#reproducibility--execution-guide)
[![Provenance Caches](https://img.shields.io/badge/caches-33%2F33%20SHA256--verified-success)](#reproducibility--execution-guide)

An audited, cryptographically provenance-locked academic codebase for **contactless neonatal breathing rate estimation** using $24\text{ GHz}$ continuous-wave (CW) interferometer radar in the Neonatal Intensive Care Unit (NICU).

1. **Goal 1 — Constrained Reimplementation of Beltrão et al. (2022):** Reconstructs the four-stage reference pipeline (*Complex Signal Demodulation $\to$ STFT + NMF/RBM Mitigation $\to$ Bandpass Filtering $\to$ Harmonic Nonlinear Least Squares [NLS]*) from [Beltrão et al., *Scientific Reports* 12, 5150 (2022)](https://doi.org/10.1038/s41598-022-08860-x).
2. **Goal 2 — Deterministic HPSS Extension:** Replaces iterative Non-Negative Matrix Factorization (`NMF`, $\mathcal{O}(I \cdot K \cdot F \cdot T)$) with closed-form, zero-frequency-centered **Harmonic–Percussive Source Separation (`HPSS`, $\mathcal{O}(F \cdot T)$)** while keeping all upstream demodulation, pre/post bandpass filtering, and downstream harmonic NLS estimation 100% identical.

> **Complete Engineering & Audit History:** All historical iterations, experiments, mathematical root-cause analyses (resolving the *"Raw NLS Paradox"*), cohort multiverse evaluations, and lessons learned are documented in [**`RESEARCH_AND_ENGINEERING_LOG.md`**](file:///Users/hosamshibly/Downloads/medical_radar/RESEARCH_AND_ENGINEERING_LOG.md).

---

## Key Empirical Results

### 1. Paper-Inspired 18-Session Benchmark (`PAPER_18`, $N = 5,466$ Valid 30-s Windows)
Per-measurement scenario means reproduced via `python3 main.py --tables`:

| Scenario | $N_{\text{meas}}$ | $N_{\text{win}}$ | `DFT` (No Mit.) | `NLS` (No Mit.) | `NLS + NMF` (Baseline) | `HPSS + NLS` (Proposed) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ALL** | 18 | 5,466 | 10.11 bpm (74.8% Acc@10) | 7.49 bpm (82.4% Acc@10) | **6.86 bpm (85.4% Acc@10)** | **6.80 bpm (85.8% Acc@10)** |
| **Supine** | 11 | 3,120 | 11.54 bpm (69.8% Acc@10) | 8.35 bpm (78.1% Acc@10) | **7.63 bpm (81.3% Acc@10)** | **7.55 bpm (81.5% Acc@10)** |
| **Prone** | 7 | 2,346 | 7.85 bpm (82.6% Acc@10) | 6.12 bpm (89.0% Acc@10) | **5.65 bpm (91.7% Acc@10)** | **5.62 bpm (92.4% Acc@10)** |
| **Single** | 14 | 4,002 | 9.79 bpm (76.5% Acc@10) | 7.54 bpm (82.4% Acc@10) | **6.89 bpm (85.2% Acc@10)** | **6.76 bpm (85.9% Acc@10)** |
| **Twins** | 4 | 1,464 | 11.23 bpm (68.7% Acc@10) | 7.31 bpm (82.3% Acc@10) | **6.77 bpm (85.8% Acc@10)** | **6.94 bpm (85.1% Acc@10)** |

### 2. Four-Variant Cohort Multiverse (`HPSS - NMF`)
Reproduced via `python3 main.py --multiverse` ([`reconstruction_multiverse.csv`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/outputs/audits/reconstruction_multiverse.csv)):

| Predefined Cohort Variant | Evaluable Sessions | Valid Windows | `NLS+NMF` RMSE | `HPSS+NLS` RMSE | $\Delta\text{RMSE}$ (bpm) | 95% Session Bootstrap CI | $\Delta\text{Acc@10}$ (pp) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `all_paired_local` | 31 / 33 | 7,105 | 8.349 | 8.361 | **+0.012** | $[-0.456, +0.372]$ | $-0.35\text{ pp}$ |
| `pool30_no_zero_overlap` | 28 / 30 | 6,642 | 8.181 | 8.085 | **−0.096** | $[-0.574, +0.287]$ | $-0.28\text{ pp}$ |
| `retained23_named_exclusions` | 23 / 23 | 6,447 | 7.796 | 7.821 | **+0.025** | $[-0.202, +0.276]$ | $+0.19\text{ pp}$ |
| `current_reference_validity_ranked_18` | 18 / 18 | 5,466 | 6.863 | 6.798 | **−0.065** | $[-0.249, +0.099]$ | $+0.39\text{ pp}$ |

* **Separation-Stage Speedup:** HPSS's fixed-cost median filtering is **~6–7× faster** than iterative NMF per 30-s window, measured in the same process on the same windows ([`reconstruction_multiverse.csv`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/outputs/audits/reconstruction_multiverse.csv) records per-window means; in the `main-v3-audit-fix-2` rebuild NMF $\approx 17.8$ ms vs. HPSS $\approx 2.8$ ms, $\approx 6.4\times$, for `all_paired_local`). Absolute runtimes are strongly environment/load dependent, so only the ratio is reported; a controlled benchmark is pending (WS8). The batch builder is single-process/sequential and HPSS is 100% deterministic.

---

## Repository Structure

```text
medical_radar/
├── README.md                          # Submission overview & reproduction guide
├── RESEARCH_AND_ENGINEERING_LOG.md    # Complete history of versions, solved bugs, audits & lessons learned
├── main.py                            # Unified command-line interface (CLI)
├── beltrao_data/                      # Raw NICU radar CSVs (16 Hz), reference EDFs (256 Hz Resp / 1 Hz RespRate), & paper PDF
└── paper_workspace/
    ├── metadata/
    │   ├── analysis_plan_v1.md        # Pre-specified evaluation protocol & estimands
    │   ├── claim_gate_status.json     # Machine-readable scientific claim boundaries (L0–L5)
    │   ├── cohort_ledger.csv          # 33-session SHA-256 input & exclusion evidence ledger
    │   └── unresolved_protocol_items.yaml
    ├── outputs/
    │   ├── cache/                     # 33 SHA-256 provenance-locked .npz caches (main-v3-audit-fix-2)
    │   ├── audits/                    # Multiverse & per-session CSV tables
    │   ├── fig4_7.3_60s.png           # Beltrão Fig. 4 spectrogram reproduction
    │   ├── fig5_7.3.png               # Beltrão Fig. 5 25-min tracking reproduction
    │   ├── bland_altman_7.3.png       # Beltrão Fig. 5b Bland-Altman reproduction
    │   ├── fig6_scenario_comparison.png # Beltrão Fig. 6 4-way scenario bar chart
    │   ├── nmf_vs_hpss_spectrogram.png  # Extension Fig. 1: NMF vs HPSS spectrogram & waveform
    │   ├── nmf_vs_hpss_tracking.png     # Extension Fig. 2: Session 7.3 tracking & Bland-Altman
    │   ├── nmf_vs_hpss_cohort_benchmark.png # Extension Fig. 3: Cohort RMSE, Acc@10 & Runtime
    │   ├── poster.html                # Self-contained printable HTML scientific poster
    │   └── professor_audit.html       # Self-contained interactive professor briefing
    └── src/
        ├── config.py                  # Physical radar constants & locked algorithm parameters
        ├── protocols.py               # Clinical posture, twin, and incubator lid metadata
        ├── cohort.py                  # Cohort definitions (PAPER_18, pool30, retained23, local_pool)
        ├── data_loader.py             # Raw I/Q CSV & EDF reference loader
        ├── demod.py                   # Complex Signal Demodulation (CSD)
        ├── filtering.py               # 64-tap Kaiser FIR bandpass (0.3–3.0 Hz, β=6.5) & coarse autocorrelation
        ├── estimation.py              # Zero-padded DFT & Harmonic Nonlinear Least Squares (NLS)
        ├── rbm_nmf.py                 # Baseline NMF + RBM detection + Wiener ratio mask + env_normalize
        ├── hpss.py                    # Proposed fftshift-centered HPSS + soft Wiener mask + env_normalize
        ├── lags.py                    # Metadata-anchored cross-correlation lag search
        ├── evaluation.py              # Saturation-masked synchronization, valid-window gating, & metrics
        ├── provenance.py              # Cryptographic SHA-256 cache provenance validator
        ├── run_measurement.py         # Single-window & full-session 4-way runner
        ├── run_all.py                 # Session orchestrator & scenario aggregator
        ├── run_batch.py               # Sequential single-process batch cache builder
        ├── compare_ab.py              # Anchor session 7.3 4-way A/B demo
        ├── result_tables.py           # Fig. 6, Fig. 7, & pooled scenario tables
        ├── build_metadata.py          # Cohort SHA-256 ledger generator
        ├── reconstruction_multiverse.py # 4-cohort multiverse & bootstrap CI evaluator
        ├── plot_figs.py               # Beltrão reproduction figure generator
        ├── plot_nmf_vs_hpss.py        # NMF vs HPSS comparative figure generator
        ├── test_pipeline.py           # 15-check unit, mathematical, provenance & golden-value regression test suite
        ├── requirements.txt           # Core Python dependencies
        └── requirements-lock.txt      # Pinned reproducible environment (Python 3.9.6)
```

---

## Reproducibility & Execution Guide

### 1. Environment Setup
```bash
pip install -r paper_workspace/src/requirements.txt
# Or install exact pinned versions:
pip install -r paper_workspace/src/requirements-lock.txt
```

### 2. Reproduce All Verification Checks, Tables, Multiverse CSVs, and Figures
All tasks are orchestrated through [`main.py`](file:///Users/hosamshibly/Downloads/medical_radar/main.py):

```bash
# 1. Run the 15-check unit, mathematical, provenance & golden-value regression test suite
python3 main.py --test

# 2. Run the 4-way A/B comparison on anchor session 7.3
python3 main.py --demo

# 3. Print Fig. 6 (scenario mean), Fig. 7 (minimal movement), and pooled tables
python3 main.py --tables

# 4. Rebuild the 33-session SHA-256 cohort ledger and 4-variant multiverse CSVs
python3 main.py --multiverse

# 5. Regenerate all publication and comparative figures in paper_workspace/outputs/
python3 main.py --plots

# 6. Run the complete verification, table, multiverse, and figure pipeline end-to-end
python3 main.py --all
```

To recompute all `.npz` caches from scratch (rather than using the verified SHA-256 locked caches):
```bash
# Rebuild all 33 paired-session caches (needed for the four-variant multiverse):
python3 -m paper_workspace.src.run_batch --all-paired --force

# Rebuild only the 18-session benchmark cohort (PAPER_18):
python3 main.py --batch --force
```
