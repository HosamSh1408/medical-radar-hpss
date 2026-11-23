# Chapter 2 Dense Input: Baseline Audit & EXP-0 Failure Forensics

Strict symbol adherence: [notation_glossary.tex](file:///Users/hosamshibly/Downloads/medical_radar/paper_orchestra_bundle/raw_materials/notation_glossary.tex). Claim constraints (Chapter 2 / WP0 baseline scope): `L0-L3` internal validation only (`L2` exact Beltrao cohort reproduction blocked; `L5` clinical bedside utility claims prohibited; `L4` external zero-touch transfer is evaluated separately in Stage 4 / Ch. 7 on `gleichauf2024`).

## 1. Cohort Multiverse & Lag-Leakage Decomposition (`EXP-0` Locked Outputs)
- **Primary Confirmatory Cohort (`pool30_sync`, $S=28$ synced sessions, $11$ patients, $|\mathcal{W}|=6640$):** Under cached `v1` lags, `HPSS+NLS` obtains $\rmseSess = 8.085\,\mathrm{bpm}$ ($\constLoso = 8.124$, $\oracleSd = 4.610$, $\rWithinSessMean = 0.447$, ANCOVA $\rWithin = 0.437$).
- **Branch-Neutral Synchronization Impact (`branch_neutral_raw_nls_lag`):** Fixing synchronization lags without candidate leakage reduces `pool30_sync` $\rmseSess$ from $8.085$ to $7.817\,\mathrm{bpm}$ (full ``all_paired`` cohort: $8.361 \to 8.119\,\mathrm{bpm}$), isolating Session `12.1` as a pure NMF lag-pull artifact (`18.63` -> `3.47` bpm).
- **Sensitivity Cohorts (`neutral`):** `retained23` $\rmseSess = 7.973\,\mathrm{bpm}$; `PAPER_18` $\rmseSess = 7.135\,\mathrm{bpm}$ ($\rmsePool = 7.363\,\mathrm{bpm}$).

## 2. True Physical Error Modes Persisting Under Neutral Synchronization
- **Quadrature Imbalance & Collinearity (`WP1` Target):** Session `11.1` (empirical I/Q SD ratio $\rho_{IQ} \equiv \max(\sigma_I/\sigma_Q, \sigma_Q/\sigma_I) = 5.86$, $|r_{IQ}|=0.863$, clipping `5.06%`) persists at $\mathrm{RMSE}_s = 12.65\,\mathrm{bpm}$ across all lags.
- **Mid-Session Hardware / UCD Dropouts (`WP1` / `WP4` Target):** Session `9.2` exhibits mid-session `UCD Status != ChildPresent` (`9.2` $\mathrm{RMSE}_s = 18.75\,\mathrm{bpm}$).
- **High-Rate Reference Verification:** `1024 Hz Resp.-Trigger` reproduces `1 Hz RespRate` across $7101$ windows (bias $+2.21\,\mathrm{bpm}$, MAE $5.18\,\mathrm{bpm}$), and `1024 Hz R-Trigger` matches `1 Hz HeartRate` (`SignalIQ >= 50`, MAE $0.94\,\mathrm{bpm}$).

## 3. Top Observational Covariates Explaining Excess `pool30_sync` Error
- `pct_win_hr_collision_band` (Gross error near HR or HR/2): Spearman $\rho_{\mathrm{excess}} = +0.696$ ($p=0.0000$), $\rho_r = -0.073$.
- `minmove_frac` (NMF minimal-movement window frac): Spearman $\rho_{\mathrm{excess}} = -0.563$ ($p=0.0018$), $\rho_r = +0.512$.
- `pct_win_near_half_hr_10bpm` (Windows within 10 bpm of HR/2): Spearman $\rho_{\mathrm{excess}} = +0.541$ ($p=0.0030$), $\rho_r = +0.082$.
- `pct_win_br_double_octave` (Harmonic double-octave ratio [1.8,2.2]): Spearman $\rho_{\mathrm{excess}} = +0.431$ ($p=0.0220$), $\rho_r = -0.132$.
- `ucd_nonchild_frac_session` (UCD != ChildPresent fraction): Spearman $\rho_{\mathrm{excess}} = +0.335$ ($p=0.0810$), $\rho_r = -0.112$.
