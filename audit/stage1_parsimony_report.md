# EXP-1 Stage 1.1 Parsimony & Causality Benchmark Report (`stage1-parsimony-v3`)

- **Git HEAD:** `681de7e` (`thesis/main`, Python `3.10.19`)
- **Locked Baseline Parity (`max |F0 - est_hpss|` across 31 sessions):** `0.00e+00 bpm`
- **Total Wall Time:** `122.2 s`
- **New Stage 1 Operators O(2) × R+>0 Invariance (< 1e-6 bpm):** `PASS` (clean max `1.42e-14`, clipped max `2.13e-14`; frozen `h0` Nyquist-swap continuous-MMSE diag `2.10e-04 bpm`)
- **Exact Frame Causality Diffs:** `hpss_filter_causal(L=0)` `0.00e+00`, `mitigate_hpss_v2(env_norm='causal')` `mask` `0.00e+00` / `mag_filt` `0.00e+00` (vs `env_norm='block'` intra-block median coupling diag `89.25`), HMM Forward `0.00e+00`, HMM Lag-4 `0.00e+00`

## 1. Dual-Reference Discrepancy Summary (`1024 Hz Resp.-Trigger` minus `1 Hz RespRate`)

| Cohort | $S$ | $N_{1\mathrm{Hz}}$ | $N_{\mathrm{trig}}$ | $>5\text{ s}$ Pauses | Mean Diff (sess / pool) | MAE (pool) | $r_{\mathrm{within}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pool30_sync` | 28 | 6640 | 6638 | 1581 (23.8%) | `+1.65` / `+2.15` | 5.03 | 0.606 |
| `retained23` | 23 | 6447 | 6445 | 1558 (24.2%) | `+1.64` / `+2.13` | 5.01 | 0.604 |
| `PAPER_18` | 18 | 5466 | 5466 | 952 (17.4%) | `+1.39` / `+1.69` | 4.65 | 0.614 |
| `all_paired` | 31 | 7103 | 7101 | 1584 (22.3%) | `+1.69` / `+2.21` | 5.18 | 0.596 |

### 1b. Monitor Group-Delay & Pause-Exclusion Diagnostic (`pool30_sync`, `+lag` = first signal trails)

| Pair | $r_{\mathrm{within}}$ (`0 s`) | Peak Lag (s) | Peak $r_{\mathrm{within}}$ | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| `br_1hz_30s` vs `br_trig_bpm` | 0.606 | `+10 s` | 0.706 | VitaGuard `1 Hz` trend trails physical breath triggers |
| `R0_NLS` vs `br_1hz_30s` / `br_trig_bpm` | 0.423 / 0.343 | `+0 s` / `+10 s` | 0.423 / 0.402 | `branch_neutral_raw_nls_lag` aligns `R0` with `1 Hz` (`+10 s` behind `Trig`) |
| `R1a (0 s)` vs `br_1hz_30s` / `br_trig_bpm` | 0.454 / 0.307 | `+6 s` / `+16 s` | 0.471 / 0.438 | Causal Forward-HMM filter delay stacks with VitaGuard lag |
| `R1f (∞ FB)` vs `br_1hz_30s` / `br_trig_bpm` | 0.454 / 0.391 | `+4 s` / `+14 s` | 0.458 / 0.457 | Symmetric offline smoother (flat peak ~0–4 s vs 1 Hz; avoids causal filter phase lag) |
- **Pause-Exclusion Bias (`>5 s` gaps dropped in `br_trig_bpm` vs whole-window `br_trig_count_bpm = 60*N/30s`):** Overall `interval - 1Hz = +2.15 bpm` (`sess +1.65`) vs `count - 1Hz = -7.08 bpm` (`sess -6.83`); on `>5 s` pause windows ($n=1579$): `interval - 1Hz = +5.39` vs `count - 1Hz = -13.52 bpm`; on regular windows ($n=5059$): `interval - 1Hz = +1.14` vs `count - 1Hz = -5.07 bpm`.

## 2. WP1 Front-End Gate G1 (`pool30_sync` Dual-Reference & Sentinels)

- **Gate G1 Verdict:** **`RETAIN_F0_CSD_LOCKED`**

| Method | $\mathrm{RMSE}_{\mathrm{sess}}$ (1 Hz) | $\Delta_{1\mathrm{Hz}}$ [95% CI] | $p_{\mathrm{pat,Holm}}$ | $\mathrm{RMSE}_{\mathrm{sess}}$ (Trig) | $\Delta_{\mathrm{Trig}}$ | 2.3 | 11.1 | 7.3 | 4.1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `F0_CSD_Locked` | 7.817 | anchor | — | 8.968 | anchor | 11.73 | 12.65 | 4.81 | 4.85 |
| `F1a_ClipChanInterp_CSD` | 7.832 | +0.014 [-0.01, +0.05] | 1.0000 | 8.963 | -0.004 | 11.77 | 12.58 | 4.84 | 4.85 |
| `F1b_ClipJointInterp_CSD` | 7.831 | +0.014 [-0.02, +0.05] | 1.0000 | 8.966 | -0.002 | 11.79 | 12.57 | 4.87 | 4.85 |
| `F2a_SymWhitened_lam005` | 7.804 | -0.014 [-0.49, +0.53] | 1.0000 | 8.949 | -0.018 | 14.31 | 13.21 | 4.25 | 5.21 |
| `F2b_SymWhitened_lam010` | 7.833 | +0.015 [-0.42, +0.50] | 1.0000 | 8.976 | +0.008 | 13.95 | 13.03 | 4.24 | 5.09 |
| `F2c_SymWhitened_lam025` | 7.689 | -0.128 [-0.49, +0.29] | 1.0000 | 8.804 | -0.163 | 14.21 | 12.78 | 4.03 | 5.02 |
| `F3_ClipChan_SymWhitened_lam010` | 7.812 | -0.006 [-0.42, +0.48] | 1.0000 | 8.935 | -0.033 | 13.71 | 12.98 | 4.21 | 5.04 |

## 3. WP2 Causal Radar-Native HPSS Gate G2 (`pool30_sync` Dual-Reference & Speedup)

- **Gate G2 Verdict:** **`RETAIN_H0_LOCKED`** (Hypothesis `H1b` falsified cleanly; `H5` & `H6` fail ±0.50 bpm TOST on both references; `H0` retained as primary and `H5` carried as strict 0 s STFT lookahead ablation)

| Method | STFT Lookahead | ms/win | Speedup | $\mathrm{RMSE}_{\mathrm{sess}}$ (1 Hz) | $\Delta_{1\mathrm{Hz}}$ | $p_{\mathrm{TOST,sess / pat}}$ (1 Hz) | $\mathrm{RMSE}_{\mathrm{sess}}$ (Trig) | $p_{\mathrm{TOST,sess / pat}}$ (Trig) | $r_{\mathrm{within}}$ (1 Hz / Trig) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `H0_Locked_Rect_Full_L3` | 3 s | 3.47 | 1.00x | 7.817 | anchor | — | 8.968 | — | 0.423 / 0.343 |
| `H1_COLA_Hann_Full_L3` | 3 s | 3.69 | 0.94x | 7.999 | +0.182 | 0.0009 / 0.0002 | 9.113 | 0.0022 / 0.0015 | 0.404 / 0.335 |
| `H2_COLA_Hann_InBand_L3` | 3 s | 2.38 | 1.46x | 8.068 | +0.250 | 0.0081 / 0.0423 | 9.146 | 0.0058 / 0.0040 | 0.400 / 0.334 |
| `H3_Causal_COLA_InBand_L2` | 2 s | 2.37 | 1.46x | 8.121 | +0.303 | 0.0557 / 0.0993 | 9.074 | 0.0010 / 0.0023 | 0.399 / 0.337 |
| `H4_Causal_COLA_InBand_L1` | 1 s | 2.35 | 1.47x | 8.176 | +0.358 | 0.2283 / 0.2267 | 9.235 | 0.1775 / 0.1328 | 0.392 / 0.328 |
| `H5_Causal_COLA_InBand_L0` | 0 s | 2.35 | 1.47x | 8.267 | +0.449 | 0.4139 / 0.5190 | 9.398 | 0.4116 / 0.4305 | 0.384 / 0.315 |
| `H6_Causal_Rect_Full_L0` | 0 s | 3.41 | 1.02x | 8.079 | +0.262 | 0.0532 / 0.0267 | 9.303 | 0.2308 / 0.1861 | 0.410 / 0.322 |

## 4. Rung 1 Causality Frontier & Unpinned LOPO (`pool30_sync`, guard `1.00 × anchor_r`, `σ >= 1.0`, trained ONLY on `1 Hz`)

- **Constant Baselines (`1 Hz` / `1024 Hz Trig`):** `const_loso` = `8.096` / `10.298 bpm`; `const_lopo_median` = `8.300` / `10.369 bpm`.

| Method | Lookahead (Block / STFT) | $\mathrm{RMSE}_{\mathrm{sess}}$ (1 Hz) | Acc@10 (1 Hz) | $r_{\mathrm{within}}$ (1 Hz) | $\mathrm{RMSE}_{\mathrm{sess}}$ (Trig) | Acc@10 (Trig) | $r_{\mathrm{within}}$ (Trig) | TOST vs R1g (1 Hz) | Lower $\sigma$ Pin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `R0_HPSS_NLS_Locked` | 0 s / 3 s | 7.817 | 79.4% | 0.423 | 8.968 | 76.2% | 0.343 | — | — |
| `R1a_Online_Forward_MMSE_0s` | 0 s / 3 s | 6.990 | 82.6% | 0.454 | 8.624 | 76.8% | 0.307 | 0.4846 | 0/11 |
| `R1b_Online_Forward_MAP_0s` | 0 s / 3 s | 7.458 | 80.0% | 0.389 | 8.987 | 75.3% | 0.255 | 0.9937 | 1/11 |
| `R1c_FixedLag_MMSE_4s` | 4 s / 3 s | 6.893 | 83.0% | 0.473 | 8.402 | 77.8% | 0.337 | 0.2768 | 0/11 |
| `R1d_FixedLag_MMSE_8s` | 8 s / 3 s | 6.787 | 83.4% | 0.479 | 8.246 | 78.3% | 0.349 | 0.0698 | 0/11 |
| `R1e_FixedLag_MMSE_16s` | 16 s / 3 s | 6.759 | 83.8% | 0.484 | 8.083 | 79.6% | 0.384 | 0.0829 | 0/11 |
| `R1f_Offline_FB_MMSE_Inf` | ∞ / 3 s | 6.395 | 86.2% | 0.454 | 7.789 | 80.9% | 0.391 | 0.0000 | 11/11 |
| `R1g_Offline_Viterbi_Unpinned` | ∞ / 3 s | 6.498 | 85.2% | 0.439 | 7.876 | 80.7% | 0.379 | ref | 11/11 |
| `R1h_E2E_Causal_H5_OnlineForward_0s` | 0 s / 0 s | 7.136 | 82.6% | 0.396 | 8.864 | 76.5% | 0.263 | 0.7738 | 1/11 |
| `R1i_E2E_Causal_H6_OnlineForward_0s` | 0 s / 0 s | 7.119 | 82.6% | 0.392 | 8.917 | 76.2% | 0.258 | 0.7677 | 1/11 |

### 4a. Paired Patient-Level Comparisons vs `const_loso` (`N = 11` Infants, `pool30_sync`)

| Method | $\Delta_{1\mathrm{Hz}}$ vs `const_loso` | Wins (Pat) | $p_{\mathrm{pat}}$ / $p_{\mathrm{Holm}}$ (1 Hz) | $\Delta_{\mathrm{Trig}}$ vs `const_loso` | Wins (Pat) | $p_{\mathrm{pat}}$ / $p_{\mathrm{Holm}}$ (Trig) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `R0_HPSS_NLS_Locked` | -0.279 [-2.91, +2.13] | 5/11 | 0.8311 / 1.0000 | -1.330 [-3.95, +0.86] | 6/11 | 0.2402 / 0.3047 |
| `R1a_Online_Forward_MMSE_0s` | -1.105 [-3.55, +1.19] | 7/11 | 0.4131 / 1.0000 | -1.673 [-3.94, +0.31] | 9/11 | 0.0420 / 0.2100 |
| `R1f_Offline_FB_MMSE_Inf` | -1.701 [-4.11, +0.36] | 7/11 | 0.1748 / 1.0000 | -2.508 [-4.83, -0.61] | 10/11 | 0.0186 / 0.1670 |
| `R1g_Offline_Viterbi_Unpinned` | -1.598 [-3.95, +0.42] | 7/11 | 0.1748 / 1.0000 | -2.421 [-4.69, -0.56] | 10/11 | 0.0137 / 0.1367 |
| `R1h_E2E_Causal_H5_OnlineForward_0s` | -0.959 [-3.19, +1.12] | 7/11 | 0.4131 / 1.0000 | -1.434 [-3.45, +0.33] | 9/11 | 0.0674 / 0.2695 |
- *Note:* On `1 Hz`, `const_loso` patient-level significance is bounded by low-variability/high-radar-error infants (`pat 9`: `const = 6.17` vs `R1a = 12.47 bpm`), whereas on physical `Trig` `R1f`/`R1g` win `10/11` infants (`p_pat < 0.02`).

### 4b. 5-Seed Phase-Scrambled Radar Anti-Inertia Surrogate (`pool30_sync`)

| Method | Real RMSE (1 Hz) | Scrambled RMSE (1 Hz) | Real $r_{\mathrm{within}}$ | Scrambled $r_{\mathrm{within}}$ | Real RMSE (Trig) | Scrambled RMSE (Trig) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `R0_HPSS_NLS_Locked` | 7.817 | 9.435±0.113 | 0.423 | -0.006 | 8.968 | 11.667±0.214 |
| `R1a_Online_Forward_MMSE_0s` | 6.990 | 8.513±0.099 | 0.454 | 0.006 | 8.624 | 10.885±0.223 |
| `R1f_Offline_FB_MMSE_Inf` | 6.395 | 8.171±0.174 | 0.454 | -0.010 | 7.789 | 10.708±0.225 |
| `R1g_Offline_Viterbi_Unpinned` | 6.498 | 8.246±0.197 | 0.439 | -0.016 | 7.876 | 10.798±0.223 |
| `R1h_E2E_Causal_H5_OnlineForward_0s` | 7.136 | 8.212±0.142 | 0.396 | 0.015 | 8.864 | 10.610±0.160 |

### 4c. Shifted-Grid Midpoint-Shrinkage Control (`[18, 90]` vs `[18, 120]` bpm across 5 sessions)

| Decoder Regime | RMSE `[18, 90]` | Bias `[18, 90]` | RMSE `[18, 120]` | Bias `[18, 120]` | Shift (`Δbias`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `memoryless_mmse_naive_eps001` | 7.405 | +4.50 | 9.087 | +7.32 | `+2.82` |
| `fwd_mmse_legacy_b05_e005_sig8` | 7.522 | +4.73 | 7.754 | +4.99 | `+0.26` |
| `fwd_mmse_equivariant_b1_e1e3_sig2` | 7.868 | +4.38 | 7.868 | +4.38 | `+0.00` |
| `fb_mmse_equivariant_b1_e1e3_sig1` | 7.562 | +4.04 | 7.562 | +4.04 | `-0.00` |
| `viterbi_b1_e1e3_sig1` | 7.458 | +3.92 | 7.458 | +3.92 | `+0.00` |

### 4d. LOPO Guard & Floor Sensitivity Audit Across Cohorts (`Stage 1.1` vs `Legacy v2` vs `Fixed σ=2.0`)

| Cohort | Regime | `R1a (0 s)` 1Hz / Trig | `R1c (4 s)` 1Hz / Trig | `R1f (∞ FB)` 1Hz / Trig | `R1g (∞ Vit)` 1Hz / Trig |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `pool30_sync` | Stage 1.1 (1.00×r, σ≥1) | 6.990 / 8.624 | 6.893 / 8.402 | 6.395 / 7.789 | 6.498 / 7.876 |
| `pool30_sync` | Legacy v2 (0.95×r, σ≥0.5) | 6.990 / 8.624 | 7.190 / 8.676 | 6.514 / 7.989 | 6.586 / 7.883 |
| `pool30_sync` | Fixed σ=2.0, ε=1e-3, β=1 | 6.884 / 8.582 | 6.821 / 8.348 | 6.620 / 7.937 | 6.864 / 8.091 |
| `retained23` | Stage 1.1 (1.00×r, σ≥1) | 7.349 / 8.808 | 7.218 / 8.589 | 6.560 / 7.826 | 6.696 / 7.959 |
| `retained23` | Legacy v2 (0.95×r, σ≥0.5) | 7.707 / 9.121 | 7.579 / 8.922 | 6.795 / 7.903 | 7.029 / 8.080 |
| `retained23` | Fixed σ=2.0, ε=1e-3, β=1 | 7.197 / 8.700 | 7.099 / 8.499 | 6.844 / 8.056 | 7.075 / 8.238 |
| `PAPER_18` | Stage 1.1 (1.00×r, σ≥1) | 6.437 / 7.549 | 6.327 / 7.317 | 5.953 / 6.837 | 6.096 / 6.963 |
| `PAPER_18` | Legacy v2 (0.95×r, σ≥0.5) | 6.885 / 8.018 | 6.745 / 7.785 | 6.269 / 6.885 | 6.335 / 7.043 |
| `PAPER_18` | Fixed σ=2.0, ε=1e-3, β=1 | 6.448 / 7.738 | 6.338 / 7.485 | 6.120 / 6.995 | 6.386 / 7.268 |

## 5. WP4 Causal Posterior Uncertainty Preview & Quiet-Apnea Safety

- **`R1a_Online_Forward_MMSE_0s`:** Spearman($H_t$, $|e|$) = `0.256` pool / `0.107` within-sess (1 Hz), `0.242` pool / `0.120` within-sess (Trig); Spearman($\mathrm{SD}$, $|e|$) = `0.279` / `0.105` (1 Hz); quiet-apnea >5 s mean $H_t$ = `0.518` vs regular `0.498` ($n=1581/5059$).
- **`R1h_E2E_Causal_H5_OnlineForward_0s`:** Spearman($H_t$, $|e|$) = `0.157` pool / `0.091` within-sess (1 Hz), `0.131` pool / `0.099` within-sess (Trig); Spearman($\mathrm{SD}$, $|e|$) = `0.188` / `0.092` (1 Hz); quiet-apnea >5 s mean $H_t$ = `0.480` vs regular `0.451` ($n=1581/5059$).
