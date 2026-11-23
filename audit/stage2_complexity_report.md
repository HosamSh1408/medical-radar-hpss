# EXP-2 Stage 2 Complexity Ladder & Shortcut-Learning Audit (`stage2-complexity-v4`)

- **Git HEAD:** `681de7e` (`thesis/main`, Python `3.10.19`)
- **Total Wall Time:** `1527.3 s` | **Stage 1.1 Anchor Parity (`R0/R1a/R1f/R1g`):** `max diff = 0.00e+00`
- **R2 O(2)xR+ Invariance on H5 (< 1e-6 bpm):** `PASS` (clean max `1.42e-14`, clipped max `2.13e-14`; frozen H0 swap diag `6.76e-04`)
- **Strict 0.0 Causality Diffs:** `R2e=0.00e+00`, `R2a=0.00e+00`, `R2b=0.00e+00`, `R3a_TCN=0.00e+00`
- **Exact Operator +10 bpm Equivariance (`[-60, 180]`):** `R2e=1.42e-14`, `R2a=7.11e-15`, `R2b=0.00e+00 bpm`
- **Calibrated `[18, 90]` +10 bpm Shift Err (`36->46`):** `R2e=1.200e-07`, `R2a=3.179e-05` vs `R3c=3.01`, `R3b=5.66`, `R3a=8.86 bpm`
- **24 GHz Waveform `+10 bpm` Shift (`36->46`):** `R2e=9.987`, `R2a=10.001` vs `R3c=6.523`, `R3b=15.043`, `R3a=5.465 bpm`
- **Shifted-Grid `[18, 90]` vs `[18, 120]` Bias Shift:** `R2e=0.0000`, `R2a=0.0188`, `R2b=0.0163 bpm`
- **R2 L-BFGS-B Convergence (`maxiter=200`):** `R2e=11/11 (med 15)`, `R2a=11/11`, `R2b=8/11 (med 178)`, `R2f=11/11`, `R2c=11/11`

## 1. 4-Rung Complexity Ladder (`pool30_sync`, trained ONLY on `1 Hz RespRate`, 4-way blind test)

| Rung | Method | Params | Lookahead (Block / STFT) | $\mathrm{RMSE}_{\mathrm{sess}}$ (1 Hz) | Rank (1 Hz) | $r_{\mathrm{within}}$ (1 Hz) | $\mathrm{SD}/\mathrm{SD}_{1\mathrm{Hz}}$ | $\mathrm{SD}/\mathrm{SD}_{\mathrm{T10}}$ | $\mathrm{RMSE}_{\mathrm{sess}}$ (Trig 0s) | $\mathrm{RMSE}_{\mathrm{sess}}$ (Trig -10s) | Rank (-10s) | $r_{\mathrm{within}}$ (-10s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| R0 | `R0_HPSS_NLS_Locked` | 0 | 0 s / 3 s | 7.817 | #13 | 0.423 | 1.66 | 0.99 | 8.968 | 8.727 | #13 | 0.402 |
| R1 | `R1a_Online_Forward_MMSE_0s` | 3 | 0 s / 3 s | 6.990 | #10 | 0.454 | 1.36 | 0.82 | 8.624 | 8.197 | #6 | 0.408 |
| R1 | `R1f_Offline_FB_MMSE_Inf` | 3 | inf / 3 s | 6.395 | #3 | 0.454 | 1.16 | 0.70 | 7.789 | 7.597 | #1 | 0.430 |
| R1 | `R1g_Offline_Viterbi_Unpinned` | 2 | inf / 3 s | 6.498 | #4 | 0.439 | 1.20 | 0.73 | 7.876 | 7.715 | #2 | 0.408 |
| R2 | `R2e_Causal_DiffHMM_FilterSafe8` | 8 | 0 s / 3 s | 6.849 | #7 | 0.465 | 1.36 | 0.82 | 8.473 | 8.063 | #4 | 0.441 |
| R2 | `R2a_Causal_DiffHMM_Linear11` | 11 | 0 s / 3 s | 6.870 | #8 | 0.496 | 1.40 | 0.84 | 8.771 | 8.350 | #8 | 0.455 |
| R2 | `R2b_Causal_DiffHMM_MLP43` | 43 | 0 s / 3 s | 6.991 | #11 | 0.477 | 1.36 | 0.82 | 8.975 | 8.502 | #10 | 0.424 |
| R2 | `R2f_Offline_DiffHMM_FilterSafe8` | 8 | inf / 3 s | 6.847 | #6 | 0.462 | 1.29 | 0.79 | 8.114 | 7.879 | #3 | 0.436 |
| R2 | `R2c_Offline_DiffHMM_Linear11` | 11 | inf / 3 s | 6.941 | #9 | 0.444 | 1.32 | 0.79 | 8.487 | 8.197 | #7 | 0.427 |
| R2 | `R2d_E2E_Causal_H5_DiffHMM_Linear11` | 11 | 0 s / 0 s | 7.115 | #12 | 0.437 | 1.50 | 0.87 | 9.047 | 8.578 | #12 | 0.409 |
| R3 | `R3c_Ridge_Spectral_439` | 439 | 0 s / 3 s | 5.798 | #1 | 0.490 | 0.88 | 0.55 | 8.255 | 8.102 | #5 | 0.425 |
| R3 | `R3b_Spectral_MLP_15k` | 14,593 | 0 s / 3 s | 6.584 | #5 | 0.481 | 1.35 | 0.82 | 8.791 | 8.515 | #11 | 0.412 |
| R3 | `R3a_Causal_TCN_16k` | 15,817 | 0 s / 3 s | 6.075 | #2 | 0.466 | 0.84 | 0.52 | 8.586 | 8.498 | #9 | 0.404 |

## 2. Interpretable Rung 2 Learned Parameters (`8` & `11` params mean ± SD across 11 LOPO folds)

| Parameter | `R2e_Causal_DiffHMM_FilterSafe8` | `R2a_Causal_DiffHMM_Linear11` | `R2d_E2E_Causal_H5_DiffHMM_Linear11` |
| :--- | :---: | :---: | :---: |
| `w1_p1_clean` | `+2.000 ± 0.000` | `+2.000 ± 0.000` | `+2.000 ± 0.000` |
| `w2_p2_harm` | `+0.000 ± 0.000` | `+0.760 ± 0.401` | `+0.422 ± 0.288` |
| `w3_p3_harm` | `+0.000 ± 0.000` | `+1.046 ± 0.107` | `+0.803 ± 0.212` |
| `w4_phalf_sub` | `+0.000 ± 0.000` | `-0.476 ± 0.060` | `-0.359 ± 0.227` |
| `w5_anchor` | `+3.925 ± 0.133` | `+2.351 ± 0.391` | `+1.772 ± 0.369` |
| `w6_p1_raw` | `-0.500 ± 0.000` | `-0.500 ± 0.000` | `-0.500 ± 0.000` |
| `sigma_slow_bpm` | `+1.081 ± 0.143` | `+2.153 ± 0.621` | `+2.338 ± 1.900` |
| `sigma_jump_bpm` | `+5.000 ± 0.000` | `+12.723 ± 2.932` | `+12.918 ± 1.930` |
| `pi_jump` | `+0.001 ± 0.000` | `+0.010 ± 0.002` | `+0.007 ± 0.005` |
| `k1` | `+0.000 ± 0.000` | `+0.000 ± 0.000` | `+0.000 ± 0.000` |
| `k2` | `+0.012 ± 0.026` | `+0.051 ± 0.043` | `+0.026 ± 0.048` |

## 3. Out-of-Range 24 GHz Passband Synthetic Frequency Sweep (`28–80 bpm`, zero synthetic tuning)

| Method | `28 bpm` MAE | `35 bpm` MAE | `50 bpm` MAE | `65 bpm` MAE | `80 bpm` MAE | Chirp `30->75` RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `R0_HPSS_NLS_Locked` | 0.04 | 0.05 | 0.05 | 0.09 | 0.03 | 2.01 |
| `R1a_Online_Forward_MMSE_0s` | 0.07 | 0.03 | 0.04 | 0.05 | 0.06 | 1.47 |
| `R2e_Causal_DiffHMM_FilterSafe8` | 0.04 | 0.01 | 0.03 | 0.01 | 0.01 | 2.37 |
| `R2a_Causal_DiffHMM_Linear11` | 0.00 | 0.00 | 0.00 | 0.01 | 0.04 | 2.06 |
| `R2b_Causal_DiffHMM_MLP43` | 0.00 | 0.00 | 0.00 | 0.01 | 0.05 | 2.27 |
| `R2d_E2E_Causal_H5_DiffHMM_Linear11` | 1.97 | 0.01 | 0.00 | 0.02 | 0.05 | 1.36 |
| `R3c_Ridge_Spectral_439` | 7.51 | 5.27 | 2.01 | 3.04 | 11.18 | 2.50 |
| `R3b_Spectral_MLP_15k` | 6.64 | 2.88 | 5.43 | 4.81 | 4.28 | 4.15 |
| `R3a_Causal_TCN_16k` | 12.45 | 6.06 | 2.80 | 5.81 | 11.14 | 4.99 |
- *Kaiser Low-Edge Roll-off Diagnostic (`20 bpm = 0.333 Hz`, 64-tap `BAND_LOW_HZ=0.30 Hz` skirt attenuates `1f` below `2f=40 bpm`):* `R0=20.06`, `R2e=20.00`, `R2a=17.48`, `R3a=24.51 bpm` MAE.

## 4. 5-Seed Phase-Scrambled Radar Anti-Inertia Surrogates (`pool30_sync`)

| Method | Real RMSE (1 Hz) | Scrambled RMSE (1 Hz) | Real $r_{\mathrm{within}}$ | Scrambled $r_{\mathrm{within}}$ |
| :--- | :---: | :---: | :---: | :---: |
| `R2e_Causal_DiffHMM_FilterSafe8` | 6.849 | 8.361±0.105 | 0.465 | -0.006 |
| `R2a_Causal_DiffHMM_Linear11` | 6.870 | 9.083±0.106 | 0.496 | -0.009 |
| `R2b_Causal_DiffHMM_MLP43` | 6.991 | 9.067±0.104 | 0.477 | -0.011 |
| `R2d_E2E_Causal_H5_DiffHMM_Linear11` | 7.115 | 8.633±0.115 | 0.437 | 0.009 |
| `R3c_Ridge_Spectral_439` | 5.798 | 7.826±0.122 | 0.490 | -0.018 |
| `R3b_Spectral_MLP_15k` | 6.584 | 8.969±0.207 | 0.481 | -0.018 |
| `R3a_Causal_TCN_16k` | 6.075 | 7.838±0.108 | 0.466 | -0.021 |
