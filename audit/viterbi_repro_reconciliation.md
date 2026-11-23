# Viterbi Reproduction & Baseline Reconciliation Audit

**Pipeline Anchor:** `main-v3-audit-fix-2` (`33/33` SHA-256 locked baseline caches verified)  
**Schema:** `viterbi-repro-v2` | **Git Head:** `681de7e` (`thesis/main`) | **Python:** `3.10.19`

## 1. Locked Baseline Parity Check (Fresh Python 3.10 vs Frozen Python 3.9 Caches)
- Max window deviation across all 33 sessions — `DFT`: `0.00e+00` bpm, `Raw NLS`: `0.00e+00` bpm, `NMF+NLS`: `0.00e+00` bpm, `HPSS+NLS`: `0.00e+00` bpm.
- `PAPER_18` (`cached_nmf_lag`) locked baseline reproduction: `NLS+NMF` = `6.863` bpm (`85.4%`), `HPSS+NLS` = `6.798` bpm (`85.8%`).

## 2. Primary Confirmatory Cohort (`pool30_sync`, Branch-Neutral Synchronization)
- Constant baselines: `const_loso` = `8.096` bpm, `const_lopo_median` = `8.300` bpm, `const_lopo_mean` = `8.258` bpm, `const_global_oracle` = `7.775` bpm, `session_mean_oracle` = `4.576` bpm.

| Method | S | Windows | RMSE_sess | RMSE_pool | MAE_sess | Acc@10 | r_within (rmcorr) | r_sess_mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A1_DFT_Locked` | 28 | 6640 | 11.878 | 11.459 | 9.353 | 65.1% | 0.353 | 0.365 |
| `A2_RawNLS_Locked` | 28 | 6640 | 8.172 | 8.021 | 6.597 | 77.1% | 0.466 | 0.472 |
| `A3_Beltrao_NMF_NLS` | 28 | 6640 | 8.074 | 8.146 | 6.397 | 79.1% | 0.377 | 0.335 |
| `A4_HPSS_NLS` | 28 | 6640 | 7.817 | 7.970 | 6.237 | 79.4% | 0.423 | 0.397 |
| `B1_Clean_WelchPeak_Raw` | 28 | 6640 | 12.273 | 11.928 | 9.810 | 62.0% | 0.355 | 0.324 |
| `B2_Clean_WelchPeak_NMF` | 28 | 6640 | 12.624 | 11.772 | 10.030 | 62.4% | 0.289 | 0.274 |
| `B3_Clean_WelchPeak_HPSS` | 28 | 6640 | 12.238 | 11.645 | 9.657 | 63.4% | 0.316 | 0.302 |
| `B4_HarmNLS_FullGrid_NMF` | 28 | 6640 | 11.810 | 11.397 | 9.332 | 63.5% | 0.233 | 0.179 |
| `B5_HarmNLS_FullGrid_HPSS` | 28 | 6640 | 12.225 | 11.534 | 9.682 | 61.8% | 0.263 | 0.223 |
| `B5a_Anchored_HarmNLS_Peak_HPSS` | 28 | 6640 | 7.701 | 7.887 | 6.148 | 79.3% | 0.430 | 0.407 |
| `B6_Diag_Legacy_AD_NaiveFFT` | 28 | 6640 | 12.921 | 13.428 | 10.249 | 58.7% | 0.152 | 0.183 |
| `B7_Diag_Legacy_AD_RawNMF` | 28 | 6640 | 12.564 | 13.064 | 10.463 | 54.5% | 0.059 | 0.126 |
| `C1a_1D_Vit_NMF` | 28 | 6640 | 11.314 | 10.474 | 9.322 | 65.7% | 0.318 | 0.282 |
| `C1b_1D_Vit_HPSS` | 28 | 6640 | 10.109 | 10.219 | 8.093 | 70.7% | 0.361 | 0.366 |
| `C1m_1D_Vit_HPSS_MatchedW05` | 28 | 6640 | 9.741 | 9.873 | 7.849 | 71.6% | 0.363 | 0.369 |
| `C2a_1D_HQI_NMF` | 28 | 6640 | 11.196 | 10.414 | 9.215 | 66.6% | 0.313 | 0.264 |
| `C2b_1D_HQI_HPSS` | 28 | 6640 | 10.077 | 10.160 | 8.207 | 70.4% | 0.369 | 0.364 |
| `C2m_1D_HQI_HPSS_MatchedW05` | 28 | 6640 | 9.655 | 9.924 | 7.839 | 72.0% | 0.360 | 0.383 |
| `C3a_2D_JointVit_NMF` | 28 | 6640 | 10.746 | 10.206 | 8.953 | 68.0% | 0.304 | 0.281 |
| `C3b_2D_JointVit_HPSS` | 28 | 6640 | 9.648 | 9.923 | 7.807 | 72.3% | 0.359 | 0.381 |
| `C3c_2D_JointVit_HPSS_HPSSHR` | 28 | 6640 | 9.661 | 10.058 | 7.789 | 71.8% | 0.358 | 0.385 |
| `C3d_2D_JointVit_NMF_ADHR_Diag` | 28 | 6640 | 11.083 | 10.310 | 9.285 | 65.7% | 0.310 | 0.310 |
| `C4a_2D_HarmNLS_Vit_NMF` | 28 | 6640 | 12.398 | 13.576 | 10.492 | 59.5% | 0.174 | 0.109 |
| `C4b_2D_HarmNLS_Vit_HPSS` | 28 | 6640 | 13.030 | 14.344 | 11.119 | 55.9% | 0.159 | 0.071 |
| `C4c_1D_Anchored_HarmNLS_Vit_HPSS` | 28 | 6640 | 7.182 | 7.513 | 5.786 | 81.6% | 0.452 | 0.496 |
| `C4d_2D_Anchored_HarmNLS_Vit_HPSS` | 28 | 6640 | 7.036 | 7.306 | 5.673 | 82.0% | 0.471 | 0.502 |
| `C5a_2D_JointVit_TransSt_NMF` | 28 | 6640 | 9.161 | 9.010 | 7.301 | 75.0% | 0.376 | 0.338 |
| `C5b_2D_JointVit_TransSt_HPSS` | 28 | 6640 | 8.431 | 8.582 | 6.709 | 76.9% | 0.426 | 0.404 |
| `C6a_LOPO_Cal_Vit_HPSS` | 28 | 6640 | 10.143 | 10.212 | 8.214 | 70.5% | 0.353 | 0.340 |
| `C6b_LOPO_Cal_HarmNLS_HPSS` | 28 | 6640 | 10.014 | 9.525 | 8.093 | 68.8% | 0.418 | 0.334 |
| `C6c_LOPO_Cal_HarmNLS_NMF` | 28 | 6640 | 9.441 | 9.048 | 7.711 | 70.2% | 0.396 | 0.354 |
| `C6d_LOPO_Wide_AnchoredNLS_HPSS` | 28 | 6640 | 6.877 | 7.205 | 5.504 | 83.7% | 0.456 | 0.499 |
| `C6e_LOPO_Wide_AnchoredNLS_NMF` | 28 | 6640 | 6.826 | 7.129 | 5.491 | 83.7% | 0.446 | 0.401 |

## 3. Attached Draft Bridge on `retained23` (Branch-Neutral Synchronization, 6,447 Valid 2-s Windows)
- Attached draft Table 1 claimed (`0.5 Hz` ref interpolation / `1-s` grid = `12397` windows, dynamic per-method lag): `Naive FFT = 17.47 bpm (50.6% Acc10)`, `No Joint 2D Viterbi = 17.40 bpm (50.7% Acc10)`, `Proposed = 9.34 bpm (77.7% Acc10)`, `Prone = 7.76 bpm (84.8% Acc10)`.

| Method | RMSE_sess | RMSE_pool | Acc@10_sess | r_within | BA Bias [LoA] (Session) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `B6_Diag_Legacy_AD_NaiveFFT` | 12.897 | 13.444 | 60.7% | 0.142 | -3.25 [-16.59, +10.09] |
| `B7_Diag_Legacy_AD_RawNMF` | 12.952 | 13.123 | 53.4% | 0.048 | -3.75 [-19.19, +11.69] |
| `C3d_2D_JointVit_NMF_ADHR_Diag` | 10.173 | 10.192 | 69.4% | 0.303 | +5.81 [-6.98, +18.60] |
| `B1_Clean_WelchPeak_Raw` | 12.028 | 11.877 | 65.1% | 0.357 | +5.58 [-7.16, +18.32] |
| `B2_Clean_WelchPeak_NMF` | 11.904 | 11.673 | 65.8% | 0.289 | +6.51 [-5.68, +18.69] |
| `B3_Clean_WelchPeak_HPSS` | 11.818 | 11.575 | 66.3% | 0.314 | +6.38 [-5.92, +18.68] |
| `B5a_Anchored_HarmNLS_Peak_HPSS` | 7.912 | 7.893 | 78.7% | 0.429 | +2.69 [-5.83, +11.21] |
| `A1_DFT_Locked` | 11.289 | 11.348 | 67.6% | 0.352 | +5.21 [-6.94, +17.35] |
| `A2_RawNLS_Locked` | 8.273 | 8.025 | 77.1% | 0.466 | +1.30 [-8.43, +11.03] |
| `A3_Beltrao_NMF_NLS` | 8.083 | 8.133 | 78.9% | 0.377 | +2.76 [-5.35, +10.86] |
| `A4_HPSS_NLS` | 7.973 | 7.968 | 78.9% | 0.421 | +2.72 [-5.73, +11.16] |
| `C1b_1D_Vit_HPSS` | 10.318 | 10.223 | 69.8% | 0.352 | +5.68 [-6.91, +18.27] |
| `C1m_1D_Vit_HPSS_MatchedW05` | 9.916 | 9.872 | 70.7% | 0.354 | +5.39 [-6.85, +17.63] |
| `C2b_1D_HQI_HPSS` | 10.068 | 10.106 | 70.6% | 0.361 | +5.31 [-7.78, +18.41] |
| `C3b_2D_JointVit_HPSS` | 9.811 | 9.918 | 71.9% | 0.350 | +5.22 [-7.42, +17.85] |
| `C4b_2D_HarmNLS_Vit_HPSS` | 12.537 | 14.273 | 59.9% | 0.173 | -6.00 [-26.48, +14.49] |
| `C4c_1D_Anchored_HarmNLS_Vit_HPSS` | 7.406 | 7.528 | 80.6% | 0.450 | +2.81 [-5.80, +11.42] |
| `C4d_2D_Anchored_HarmNLS_Vit_HPSS` | 7.244 | 7.317 | 80.8% | 0.470 | +2.70 [-5.79, +11.20] |
| `C6a_LOPO_Cal_Vit_HPSS` | 10.105 | 10.132 | 71.0% | 0.360 | +5.31 [-7.83, +18.45] |
| `C6b_LOPO_Cal_HarmNLS_HPSS` | 9.491 | 9.375 | 71.9% | 0.423 | +0.06 [-9.58, +9.70] |
| `C6c_LOPO_Cal_HarmNLS_NMF` | 8.974 | 8.875 | 72.9% | 0.401 | +0.56 [-7.92, +9.05] |
| `C6d_LOPO_Wide_AnchoredNLS_HPSS` | 7.251 | 7.310 | 81.3% | 0.454 | +2.64 [-5.98, +11.25] |
| `C6e_LOPO_Wide_AnchoredNLS_NMF` | 7.046 | 7.141 | 82.8% | 0.447 | +2.49 [-5.97, +10.94] |

## 4. Secondary HR Validation Against EDF 1024 Hz R-Trigger (`SignalIQ >= 50`)
| HR Estimator | Windows | MAE_sess (bpm) | RMSE_sess (bpm) | <=10 bpm (sess) | r_within | Gate G2 (>=50%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `hr_peak_csd_vel` | 6618 | 20.09 | 24.50 | 35.4% | 0.067 | False |
| `hr_peak_hpss_vel` | 6618 | 17.95 | 22.26 | 39.8% | 0.101 | False |
| `hr_peak_dacm_vel` | 6618 | 20.04 | 24.15 | 27.9% | -0.032 | False |
| `hr_peak_br_notched_vel` | 6618 | 20.11 | 24.60 | 35.3% | 0.067 | False |
| `hr_peak_csd_vel_n480` | 6618 | 20.13 | 24.59 | 35.1% | 0.076 | False |
| `hr_peak_ad_diag` | 6618 | 20.57 | 24.52 | 26.8% | -0.026 | False |
| `hr_vit_c3b` | 6618 | 19.65 | 23.35 | 35.8% | 0.096 | False |
| `hr_vit_c4b` | 6618 | 19.62 | 23.30 | 35.9% | 0.099 | False |
| `hr_vit_c4d` | 6618 | 19.45 | 23.13 | 36.6% | 0.105 | False |

## 5. LOPO Hyperparameter Selection & Pinning Diagnostics (`pool30_sync`, `branch_neutral_raw_nls_lag`)
| LOPO Family | Grid | Folds | Fallback Folds | Upper-Pinned Folds | Any-Boundary Folds |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `C6a_LOPO_Cal_Vit_HPSS` | narrow | 11 | 10 | 0 | 2 |
| `C6b_LOPO_Cal_HarmNLS_HPSS` | narrow | 11 | 0 | 11 | 11 |
| `C6c_LOPO_Cal_HarmNLS_NMF` | narrow | 11 | 0 | 11 | 11 |
| `C6d_LOPO_Wide_AnchoredNLS_HPSS` | wide | 11 | 0 | 0 | 11 |
| `C6e_LOPO_Wide_AnchoredNLS_NMF` | wide | 11 | 0 | 0 | 11 |

## 6. Physics & Multi-Seed Surrogate Stress-Tests (`branch_neutral_raw_nls_lag`)
### Welch 30-s (`nperseg=480`) vs 16-s (`nperseg=256`) on `pool30_sync`
| Variant | RMSE_sess | RMSE_pool | Acc@10_sess | r_within |
| :--- | :---: | :---: | :---: | :---: |
| `B3_Clean_WelchPeak_HPSS_n256` | 12.238 | 11.645 | 63.4% | 0.316 |
| `B3_Clean_WelchPeak_HPSS_n480` | 12.437 | 11.780 | 63.5% | 0.298 |
| `C3b_2D_JointVit_HPSS_n256` | 9.648 | 9.923 | 72.3% | 0.359 |
| `C3b_2D_JointVit_HPSS_br_n480` | 9.569 | 9.924 | 72.1% | 0.359 |
| `C3b_2D_JointVit_HPSS_both_n480` | 9.515 | 9.887 | 72.1% | 0.361 |

### Legacy AD + NMF Drift & Init Sensitivity
| Cohort | Variant | RMSE_sess | RMSE_pool | Acc@10_sess | r_within |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `retained23` | `B7_raw_nndsvda_baseline` | 12.952 | 13.123 | 53.4% | 0.048 |
| `retained23` | `B7_sens_raw_nndsvd` | 12.986 | 13.567 | 60.6% | 0.138 |
| `retained23` | `B7_sens_detrend_bp_nndsvd` | 10.290 | 10.950 | 68.5% | 0.190 |
| `pool30_sync` | `B7_raw_nndsvda_baseline` | 12.564 | 13.064 | 54.5% | 0.059 |
| `pool30_sync` | `B7_sens_raw_nndsvd` | 13.050 | 13.557 | 58.4% | 0.146 |
| `pool30_sync` | `B7_sens_detrend_bp_nndsvd` | 10.384 | 10.940 | 66.7% | 0.193 |
