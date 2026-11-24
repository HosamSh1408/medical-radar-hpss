# Chapter 3 Dense Outline: Baseline Forensic Audit (`EXP-0`) and Stage 0 Viterbi Reproduction

**Notation Contract:** Strictly [`notation_glossary.tex`](notation_glossary.tex).  
**Supersession Note:** Under the locked 7-chapter monograph architecture, this chapter incorporates and supersedes earlier preliminary draft notes in frozen [`idea_dense_ch2_wp0.md`](idea_dense_ch2_wp0.md) (which remains unmodified under the `5e458c1` SHA-256 lock).  
**Authoritative Data Sources:** `metrics_wp0_forensics.json`, `metrics_viterbi_repro.json`, `forensics_attribution.csv`, `paper_workspace/metadata/claim_gate_status.json`, `paper_workspace/metadata/unresolved_protocol_items.yaml`.  
**Frozen Figures (7 PDFs):** `exp0_session_error_decomposition.pdf`, `exp0_physics_error_drivers.pdf`, `exp0_edf256_reference_agreement.pdf`, `viterbi_repro_forensic_bridge.pdf`, `viterbi_repro_ablation_ladder.pdf`, `viterbi_repro_bland_altman.pdf`, `viterbi_repro_tracking_examples.pdf`.

---

## 3.1 Baseline Provenance, Unresolved Protocol Items & Multiverse Audit
- Verify `33/33` locked `v1` baseline caches (`main-v3-audit-fix-2`, `154/154` baseline SHA-256 files, `19/19` tests passing) under `/usr/bin/python3` (`3.9.6`).
- Explicitly document why `L2` (exact `beltrao2022contactless` cohort IDs `N=18`, `~4964` denominator vs. active `5466`, `NMF` sklearn solver iterations, and exact synchronization rule) is structurally blocked per `unresolved_protocol_items.yaml` and `claim_gate_status.json`, motivating the pre-registered four-cohort multiverse (`all_paired` $S=31, |\mathcal{W}|=7103$; `pool30_sync` confirmatory $S=28, |\mathcal{W}|=6640, K=11$; `retained23` $S=23, |\mathcal{W}|=6447$; `PAPER_18` $S=18, |\mathcal{W}|=5466$).
- Disambiguate legacy `PAPER_18` HPSS $\rmseSess = 6.798\,\mathrm{bpm}$ vs. $\rmsePool = 7.217\,\mathrm{bpm}$ and $\rWithinSessMean = 0.465$ vs. ANCOVA $\rWithin$ under cached `v1` lags (`notation_glossary.tex` Table 2).

## 3.2 `EXP-0` Lag-Leakage Forensics & Physical Error Attribution
- **Synchronization Leakage Decomposition (`cached_nmf_lag` vs. `branch_neutral_raw_nls_lag` vs. `pure_metadata_clock_lag`):**
  - On `pool30_sync`, eliminating candidate NMF/HPSS branch-specific lag selection via neutral `branch_neutral_raw_nls_lag` reduces `HPSS+NLS` $\rmseSess$ from `8.0854` ($S=28, |\mathcal{W}|=6642$, `const_loso = 8.1244`, `oracle_sd = 4.6099`) to **`7.8173 bpm`** ($S=28, |\mathcal{W}|=6640$, `NMF = 8.0736`, `RawNLS = 8.1719`, `rmse_pool_hpss = 7.9697`, `mae_sess_hpss = 6.2366`, `const_loso = 8.0958`, `oracle_sd = 4.5763`), while `all_paired` moves `8.3607 -> 8.1185 bpm` and `retained23` (`neutral`) sits at `7.9735 bpm`. Pure clock alignment (`pure_metadata_clock_lag`, $S=27, |\mathcal{W}|=6426$) suffers $\rmseSess = 10.4283\,\mathrm{bpm}$ due to coarse EDF clock drift.
  - Isolate **Session `12.1`** as a pure NMF lag-pull artifact (`18.63 -> 3.47 bpm` upon neutral sync), versus persistent physical failure modes: **Session `11.1`** (quadrature imbalance $\rho_{IQ} \equiv \max(\sigma_I/\sigma_Q, \sigma_Q/\sigma_I) = 5.86$, $|r_{IQ}| = 0.863$, clipping `5.06%`, persistent $\mathrm{RMSE}_{11.1} = 12.65\,\mathrm{bpm}$) and **Session `9.2`** (`UCD Status != ChildPresent` mid-session recalibration/boot outage, $\mathrm{RMSE}_{9.2} = 18.75\,\mathrm{bpm}$).
- **High-Rate `256 Hz` / `1024 Hz` Reference Audit (`exp0_edf256_reference_agreement.pdf`):**
  - Across $7101$ valid windows (`1024 Hz Resp.-Trigger` vs. `1 Hz RespRate`), physical breath-trigger rate disagrees with smoothed `1 Hz` by bias `+2.2143 bpm`, MAE `5.1750 bpm`, and exhibits `1584` (`22.3%` across 31 sessions; `1581/6640 = 23.8%` on `pool30_sync`) quiet-apnea/pause windows with inter-breath gap `> 5.0 s`. `1024 Hz R-Trigger` matches `1 Hz HeartRate` within MAE `0.9353 bpm` (`n=7081`, `SignalIQ >= 50`).
- **Spearman Session Excess-Error Drivers (`exp0_physics_error_drivers.pdf`, `exp0_session_error_decomposition.pdf`):**
  - `pct_win_hr_collision_band` ($\rho_{\mathrm{excess}} = +0.696, p < 10^{-4}$; raw neutral $\rho = +0.790$);
  - `minmove_frac` ($\rho_{\mathrm{excess}} = -0.563, p = 0.0018$, $\rho_r = +0.512$);
  - `pct_win_near_half_hr_10bpm` ($\rho_{\mathrm{excess}} = +0.541, p = 0.0030$);
  - `pct_win_br_double_octave` ($\rho_{\mathrm{excess}} = +0.431, p = 0.0220$);
  - `ucd_nonchild_frac_session` ($\rho_{\mathrm{excess}} = +0.335, p = 0.0810$).

## 3.3 Stage 0 Structured Viterbi Reproduction & Non-Equivariant Trap Analysis (`metrics_viterbi_repro.json`)
- **Full `A–C` Ablation Ladder (`pool30_sync`, neutral lag, $1\,\mathrm{Hz}$ ref, `const_loso = 8.0958 bpm`; Figures `viterbi_repro_forensic_bridge.pdf`, `viterbi_repro_ablation_ladder.pdf`):**
  - `A1_DFT_Locked`: `11.8778 bpm` (`+4.061` vs `A4_HPSS_NLS`, $p_{\mathrm{pat}} = 0.0010$, $r_{\mathrm{within}} = 0.3526$); `A2_RawNLS_Locked`: `8.1719`; `A3_Beltrao_NMF_NLS`: `8.0736`; `A4_HPSS_NLS`: `7.8173` ($r_{\mathrm{within}} = 0.4229$); `B5a_Anchored_HarmNLS_Peak_HPSS`: `7.7009`.
  - **Discovery of the Unanchored Viterbi Failure (`C1–C3`):** Naive 1D Viterbi (`C1b_1D_Vit_HPSS`: `10.1090 bpm`, `+2.292` worse than `A4`), HQI Viterbi (`C2b`: `10.0766`), and 2D Joint BR/HR Viterbi (`C3b`: `9.6481`, `C5b` `8.4314`) **degrade** below `A4_HPSS_NLS`! Why? Replacing Beltrão's autocorr-anchored harmonic NLS (`B5a`) with unanchored raw spectral log-power allows double/half-rate cardiac and Bessel harmonic peaks to hijack trajectories.
  - **Anchored Viterbi + Wide LOPO Transition Recovery (`C4–C6`):** Restoring harmonic NLS emission (`C4c_1D_Anchored`: `7.1817 bpm`, $\Delta = -0.6357$, $p_{\mathrm{pat,Holm}} = 0.0205$; `C4d_2D_Anchored`: `7.0357`, $\Delta = -0.7816$, $p_{\mathrm{pat,Holm}} = 0.0205$) and LOPO-fitted wide state transitions recovers full gains (`C6d_LOPO_Wide_AnchoredNLS_HPSS`: `6.8765 bpm`, $\Delta = -0.9408$, $p_{\mathrm{pat,Holm}} = 0.0205$, $r_{\mathrm{within}} = 0.4557$; `C6e_NMF`: `6.8257`).
  - **Mandatory Negative Controls & Limitations Resolved in Ch. 4 (`viterbi_repro_bland_altman.pdf`, `viterbi_repro_tracking_examples.pdf`):** Expose the `23–36%` synthetic phase-scrambling inertial trap and note that `C6d`/`C6e` offline Viterbi uses **non-causal $\infty$-lookahead** (`Stage 0`), motivating strict causal `0s` forward-MMSE filtering, shifted-grid midpoint audits, and 5-seed phase-scrambling anti-inertia guards in **Chapter 4**.
