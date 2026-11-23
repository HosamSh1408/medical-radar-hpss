# Analysis Plan v3 — Parsimony, Causal Inductive Bias & Selective Uncertainty for Contactless NICU Radar Breathing Monitoring

**Degree & Program:** Master of Science (M.Sc.) in Artificial Intelligence, Brandenburg University of Technology (BTU) Cottbus–Senftenberg (Module 13600, 30 ECTS, 24-week statutory writing window).  
**Status:** Pre-registered AI/ML & Signal-Processing Thesis Protocol prior to Stage 1 (`WP1 + WP2 + Rung 1`) execution.  
**Frozen Provenance Anchors (Read-Only):**  
- Baseline freeze: `v1.0-student-baseline` (`paper_workspace/src/`, `pipeline_version = main-v3-audit-fix-2`, `33/33` SHA-256 locked `.npz` caches).  
- Stage 0 Viterbi reproduction & audit freeze: `viterbi-repro-v2` ([`metrics_viterbi_repro.json`](file:///Users/hosamshibly/Downloads/medical_radar/paper_orchestra_bundle/raw_materials/metrics_viterbi_repro.json), `7/7` clean-room gates in [`audit/viterbi_godmode_audit.md`](file:///Users/hosamshibly/Downloads/medical_radar/audit/viterbi_godmode_audit.md)).  
- Protocol predecessors & governance: [`analysis_plan_v2.md`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/metadata/analysis_plan_v2.md), [`claim_gate_status.json`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/metadata/claim_gate_status.json), [`unresolved_protocol_items.yaml`](file:///Users/hosamshibly/Downloads/medical_radar/paper_workspace/metadata/unresolved_protocol_items.yaml), [`notation_glossary.tex`](file:///Users/hosamshibly/Downloads/medical_radar/paper_orchestra_bundle/raw_materials/notation_glossary.tex).  
- **Non-Mutation Contract:** Zero files inside `paper_workspace/`, `beltrao_data/`, or `thesis/outputs/viterbi_repro/` may be edited under this protocol.

---

## 1. Core Thesis Identity: The Parsimony & Inductive-Bias Thesis

Rather than stacking over-parameterized black-box neural networks onto a small clinical cohort ($K_{\mathrm{total}} = 11$ infants in `pool30_sync`, $S = 28$ sessions, `6,640` overlapping 30-s windows $\approx 443$ non-overlapping 30-s blocks), this thesis investigates **where algorithmic complexity hits the small-$N$ clinical and reference-noise ceiling** on 24 GHz Continuous-Wave (CW) quadrature Doppler radar in the NICU.

Through a strictly controlled **4-Rung Complexity Ladder** evaluated simultaneously against bedside monitor trends (`1 Hz RespRate`), physical breath-trigger intervals (`1024 Hz Resp.-Trigger`), and a zero-touch external mechanical thorax simulator (`gleichauf2024`), this protocol quantifies the exact trade-offs between physical group invariance ($O(2) \times \mathbb{R}_{>0}$ on the I/Q plane and frequency-shift equivariance), strict bedside causality ($0\text{ s}$ future lookahead vs. fixed-lag vs. offline smoothing), genuine learnability vs. **monitor shortcut learning**, and **trustworthy label-free selective abstention**.

### Research Questions & Falsifiable Hypotheses

1. **RQ1 (Front-End Symmetry vs. Hardware Telemetry & Causal HPSS):** Do symmetry-preserving covariance whitening / ADC rail interpolation (`WP1`) and strictly causal COLA-Hann in-band HPSS (`WP2`) improve internal LOPO accuracy or reduce latency without breaking $O(2) \times \mathbb{R}_{>0}$ I/Q group invariance?
   - **H1a (`WP1`):** Tested leanly at Gate `G1`; adopted into `Rung 1+` ONLY if it improves `\rmseSess` on **both** `1 Hz RespRate` and `1024 Hz Resp.-Trigger` across `pool30_sync` without degrading clean sentinel recordings (`7.3`, `4.1`) by $> 0.50\text{ bpm}$ (negative result retained and documented otherwise).
   - **H1b (`WP2`):** Causal COLA-Hann in-band HPSS ($0\text{ s}$ STFT lookahead) achieves TOST statistical equivalence ($\pm 0.50\text{ bpm}$) to non-causal `v1` rectangular HPSS on both references while maintaining $< 10^{-6}\text{ bpm}$ group invariance and exact $0.0$ future-frame causality.
2. **RQ2 (Causality vs. Continuity in 1D State-Space Inference):** Does strictly causal online Bayesian forward filtering ($0\text{ s}$ lookahead) recover the majority of offline 1D Anchored Viterbi gains over memoryless NLS (`7.817 -> 6.877 bpm` on `pool30_sync` `1 Hz`), and does unpinning the transition step $\sigma_{\mathrm{BR}}$ downward to $0.5\text{ bpm/hop}$ eliminate the boundary-pinning observed at $\sigma_{\mathrm{BR}} = 2.0\text{ bpm/hop}$ in `viterbi-repro-v2`?
   - **H2:** Removing future lookahead incurs a bounded causal cost, quantified continuously across `lookahead` $\in \{0\text{ s}, 4\text{ s}, 8\text{ s}, 16\text{ s}, \infty\}$ under LOPO and TOST equivalence ($\pm 0.50\text{ bpm}$) against offline Viterbi.
3. **RQ3 (Inductive Bias & Shortcut Learning Along the Complexity Ladder):** Under $K=11$ infant LOPO, does an ultra-compact ($< 50$ params), $O(2) \times \mathbb{R}_{>0}$-invariant and frequency-shift-equivariant **Differentiable 1D-HMM** (`Rung 2`) match or improve generalization across **both** clinical references and untouched `gleichauf2024`, whereas an unconstrained 1D-CNN/TCN (`Rung 3`, $\sim 10^4$ params) trained freely on `1 Hz RespRate` suffers rank inversion on `1024 Hz Resp.-Trigger` and `gleichauf2024` due to monitor-bias shortcut learning?
   - **H3:** Structural frequency-shift equivariance (zero learned additive frequency offset) prevents overfitting VitaGuard's empirical monitor offset, whereas unconstrained ERM on `1 Hz RespRate` exploits that offset and fails external `L4` transfer.
4. **RQ4 (Trustworthy Clinical Abstention vs. Finite-$K$ Conformal Limits):** Does intrinsic causal HPSS + Forward-HMM posterior uncertainty enable safe label-free selective prediction (lower `AUGRC`/`E-AURC`, $\mathrm{AUROC} \ge 0.95$ on OOD/WGN while **preserving** real $>5\text{ s}$ quiet-apnea pauses rather than mistaking pauses for dropouts), and why are $95\%$ distribution-free patient-exchangeable conformal intervals mathematically vacuous at $K=10$ calibration infants?
   - **H4:** Causal Forward-HMM posterior entropy $H_t$ and predictive SD selectively reject tracking ambiguity (`E-AURC < 0`) without disproportionate false rejection during quiet breathing pauses, while finite-$K$ hierarchical conformal calibration yields non-vacuous coverage only for $\alpha \ge \frac{1}{K+1} \approx 0.091$ (demonstrated at $\alpha \ge 0.20$).

---

## 2. The 4-Rung Complexity Ladder (Pre-Registered Scope)

| Rung | Model Family | Learned Params (approx.) | Equivariances / Constraints | Causality | Thesis Role |
| :---: | :--- | :---: | :--- | :---: | :--- |
| **Rung 0** | Locked & Refined DSP (`CSD -> HPSS -> Autocorr-NLS`) | `0` | Exact $O(2) \times \mathbb{R}_{>0}$, freq-equivariant | `0–3 s` (STFT) | Memoryless physics anchor |
| **Rung 1** | Parametric Causal / Fixed-Lag / Offline 1D-HMM | `2–3` ($\sigma_{\mathrm{BR}}, \varepsilon, \beta$) | Exact $O(2) \times \mathbb{R}_{>0}$, freq-shift-equivariant | `0 s` / `4–16 s` / `∞` | Core probabilistic baseline & causality frontier |
| **Rung 2** | Differentiable Symmetric 1D-HMM (Stage 2) | `< 50` | Exact $O(2) \times \mathbb{R}_{>0}$, strict zero-offset freq equivariance | `0 s` (Forward) | Structured interpretable ML candidate |
| **Rung 3** | Unconstrained 1D-CNN / TCN (Stage 2) | `~10^4` | Unconstrained ERM on `br_1hz_30s` | `0 s` (causal conv) | **Shortcut-learning foil** exposing small-$N$ & monitor-bias overfitting |

*Note on Stage 0 Exclusion:* Following `viterbi-repro-v2` (where all 9 radar cardiac extractors failed Gate `G2` at `26.8%–39.8%` `Acc@10_HR < 50%`, and matched 2D vs. 1D BR Viterbi showed negligible separation), 2D `(BR, HR)` radar decoding is formally **closed at Stage 0**; BR harmonic disambiguation proceeds strictly via 1D autocorrelation-anchored harmonic NLS + HQI awareness.

---

## 3. Cohort Hierarchy, Synchronization & Claim-Gate Boundaries

### 3.1 Cohort Tiers (Unchanged from `analysis_plan_v2.md`)
- **Primary Confirmatory Cohort (`pool30_sync`):** $S = 28$ sessions, $K = 11$ infants (`Patients 2..12`), `6,640` valid windows under `branch_neutral_raw_nls_lag`. Primary baseline reference anchors: `\constLoso = 8.096 bpm`, `A3_Beltrao_NMF_NLS = 8.074 bpm`, `A4_HPSS_NLS = 7.817 bpm`, `C6d_LOPO_Wide_AnchoredNLS_HPSS = 6.877 bpm`.
- **Sensitivity Cohorts:** `retained23` ($S=23$, `10` infants, `6,447` windows), `PAPER_18` ($S=18$, `10` infants, `5,466` windows; historical benchmark only), `all_paired` ($S=31$, `12` infants; isolates zero-overlap recordings `1.1, 2.1, 2.2`). Preserved non-evaluable ledger: `1.3, 5.1` (zero valid non-saturated windows) and `1.2, 3.2, 10.3` (unpaired).

### 3.2 Synchronization Regime
- **Primary (`branch_neutral_raw_nls_lag`):** One shared metadata-anchored lag $\hat{\tau}_s$ computed once from upstream raw NLS (`audit/phase2_ws3_session_lags.csv`), strictly invariant across all rungs so no estimator benefits from lag-pull leakage.
- **Sensitivity (`pure_metadata_clock_lag`):** Carried for final confirmatory thesis tables (`Stage 2+`).

### 3.3 Claim-Gate Boundaries (`claim_gate_status.json`)
- **`L0–L3+` (Permitted here):** Internal LOPO comparisons across `pool30_sync`, `retained23`, `PAPER_18`, `all_paired`.
- **`L4` (Stage 4 only, post-`G4` freeze):** Zero-touch generalization on [`external_data/gleichauf2024/`](file:///Users/hosamshibly/Downloads/medical_radar/external_data/gleichauf2024/README.md) (`20` mechanical thorax simulator scenarios, `05–80 bpm`, `{normal, deep}`, 24.2 GHz CW radar + 2.5 kHz laser micrometer reference).
- **`L2` & `L5` (Prohibited):** Never claim exact Beltrão et al. (2022) cohort/denominator reproduction (`L2`) or clinical/bedside patient-outcome utility (`L5`).

---

## 4. Dual-Reference Anti-Shortcut Contract (`Stage 1–4`)

Empirical data auditing revealed a systematic discrepancy between the VitaGuard 1 Hz monitor trend (`br_1hz_30s`) and the physical breath-trigger timestamps (`br_trig_bpm`, `1024 Hz Resp.-Trigger`). Therefore:
1. **Training / LOPO Selection Anchor:** All Stage 1–2 LOPO selection and training uses **only** `br_1hz_30s` (with patient-disjoint training folds and the locked anti-inertia constraint $\rWithin^{\mathrm{train}} \ge 0.95\,\rWithin^{\mathrm{anchor}}$) to preserve strict continuity with `analysis_plan_v2.md`.
2. **Mandatory Blind Physical Crosscheck (`br_trig_bpm`):** Every model evaluated on `br_1hz_30s` is **simultaneously scored without re-tuning** on `br_trig_bpm` over identical saturation-gated windows (`valid_1hz & isfinite(br_trig_bpm)`). A candidate whose improvement on `br_1hz_30s` disappears or reverses on `br_trig_bpm` is flagged as exploiting monitor-trend shortcuts rather than recovering true cardiopulmonary periodicity.
3. **Zero-Touch External Bench (`gleichauf2024`):** Final arbitrating test for `L4` zero-offset physical transfer after Gate `G4` freeze.

---

## 5. Uncertainty Quantification, Selective Prediction & Finite-$K$ Conformal Rules

1. **Primary Bedside Abstention Paradigm (Label-Free Selective Prediction, `WP4`):**
   - Evaluates intrinsic HPSS + Causal Forward-HMM epistemic/ambiguity proxies (normalized posterior Shannon entropy $H_t \in [0, 1]$, posterior SD, harmonic sharpness, OOD spectral entropy) via **Risk–Coverage curves** ($c_{\mathrm{cov}} \in [0.5, 1.0]$), **`AURC`**, **`E-AURC`**, and **`AUGRC`** (Traub et al., NeurIPS 2024).
   - **Mandatory Safety Stratification (`1024 Hz` Pause Audit):** Stratified reporting on windows containing verified respiratory pauses (`resp_max_gap_s > 5.0` s) vs. regular breathing windows to ensure selective abstention does not covertly silence true central apnea alarms.
   - **Online ACI Explicitly Excluded:** Adaptive Conformal Inference (Gibbs & Candès 2024) requires sequential test-time observation of true labels $y_t$ to update $\alpha_t$. In non-contact radar NICU deployment (precisely motivated by absent adhesive thoracic ECG/IP electrodes), live $y_t$ is unavailable; thus Online ACI is physically non-deployable and excluded from candidate methods.
2. **Secondary Finite-$K$ Conformal Audit:**
   - Demonstrates empirically the inflation/under-coverage gap between naive window-level split conformal prediction (violating within-session $93.3\%$ overlap dependence) and patient-level hierarchical conformal prediction (Dunn et al., JASA 2023; Angelopoulos et al., ICLR 2024).
   - Explicitly registers the finite-sample theorem for $K = 10$ training infants per LOPO fold: exact patient-exchangeable distribution-free coverage requires $\alpha \ge \frac{1}{K+1} = \frac{1}{11} \approx 0.091$, rendering nominal $\alpha = 0.05$ ($95\%$) intervals mathematically vacuous ($[-\infty, +\infty]$). Valid finite-$K$ conformal risk/coverage auditing is therefore pre-registered for $\alpha \in \{0.10, 0.20, 0.30\}$.

---

## 6. Stage 1 Decision Gates (`G1`, `G2`, `G3`) & Metrics Lock

All evaluations follow [`notation_glossary.tex`](file:///Users/hosamshibly/Downloads/medical_radar/paper_orchestra_bundle/raw_materials/notation_glossary.tex): primary `\rmseSess`, plus `\rmsePool`, `\maeSess`, `\accTau{3}`, `\accTau{6}`, `\accTau{10}` (strict `<`), `\rWithin` (Bland–Altman 1995 ANCOVA `rmcorr`), `\rWithinSessMean`, session-level Bland–Altman & `ICC(1,1)`, 90% paired TOST (`TOST_MARGIN = 0.50 bpm`), and patient-clustered paired Wilcoxon signed-rank tests ($N = 11$, Holm–Bonferroni corrected within family).

- **Gate `G1` (`WP1` Front-End Ablation):** Variants `F1a_ClipChanInterp_CSD`, `F1b_ClipJointInterp_CSD`, `F2a..c_SymWhitened_{0.05,0.10,0.25}`, and `F3_ClipChan_SymWhitened` are compared against `F0_CSD_Locked`. Under the parsimony principle, adopt a variant ONLY if: (i) `\rmseSess` improves over `F0` on **both** `br_1hz_30s` and `br_trig_bpm` on `pool30_sync`, (ii) improvement on `br_1hz_30s` is statistically supported (`ci95_hi < 0.0` on the 95% patient-clustered bootstrap or `p_wilcoxon_patient_holm < 0.05`), (iii) neither clean sentinel `7.3` nor `4.1` degrades by $> 0.50\text{ bpm}$, and (iv) $< 10^{-6}\text{ bpm}$ $O(2) \times \mathbb{R}_{>0}$ group invariance holds. Otherwise **RETAIN `F0_CSD_Locked`** as a scientifically informative negative result.
- **Gate `G2` (`WP2` Causal HPSS Ablation):** Variants `H1..H6` (`COLA-Hann` vs `Rect`, full-band `M_H` vs in-band `M_{H,\mathrm{band}}`, lookahead $L \in \{0, 1, 2, 3\}\text{ s}$) are evaluated against locked `H0_Locked_Rect_Full_L3`. Adopt $0\text{ s}$-lookahead causal HPSS (`H5` or `H6`) if it achieves TOST equivalence ($\pm 0.50\text{ bpm}$) or superiority on both references, exact $0.0$ causality verification, $< 10^{-6}\text{ bpm}$ group invariance, and $< 10\text{ ms/window}$ runtime.
- **Gate `G3` (`Rung 1` Causal vs. Offline HMM & Unpinned Grid):** Evaluates `Online Forward MMSE/MAP (0 s)`, `Fixed-Lag MMSE (4 s, 8 s, 16 s)`, `Offline FB MMSE`, and `Offline Viterbi` over the unpinned downward transition grid $\sigma_{\mathrm{BR}} \in \{0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0\}$ and strictly frequency-shift-equivariant likelihood grid $\varepsilon \in \{10^{-4}, 10^{-3}\}$, $\beta \in \{1.0, 2.0, 4.0\}$ (excluding tempered $\beta < 1$ / elevated $\varepsilon$ regimes that leak $[18, 90]\,\mathrm{bpm}$ midpoint shrinkage; verified via shifted-grid `[18, 90]` vs `[18, 120]` control `< 0.05 bpm`) on both references (`H0` and end-to-end causal `H5`/`H6` features), including 5-seed phase-scrambled anti-inertia surrogates, boundary-pinning audit, and `WP4` causal uncertainty correlation preview (`post_entropy` & `post_sd_bpm` on normal vs. $>5\text{ s}$ pause windows).

---

## 7. Stage 1.1 Forensic Audit Addendum (Pre-Stage-2 Lock, `stage1-parsimony-v3`)

Following an adversarial clean-room audit of the initial Stage 1 execution (`stage1-parsimony-v2`), four methodological hardenings are locked below **before** executing Stage 2 (`Rung 2` vs `Rung 3`), while preserving Sections 1–6 verbatim as pre-registered:

1. **Two-Axis Causality Contract (Block Lookahead vs. Intra-Window STFT Lookahead):**
   - Stage 1 explicitly separates **Block-level future-sample lookahead past window end $t$** (`0 s` for `R0`, `R1a`, `R1b`, `R1h`, `R1i`; `4 s, 8 s, 16 s` for `R1c, R1d, R1e`; $\infty$ for `R1f, R1g`) from **Intra-window STFT-median lookahead** (`3 s` for `H0/H1/H2`, `0 s` for `H5/H6`). This resolves the ambiguity where `R0` and `R1a` shared identical `H0` block inputs yet displayed conflicting scalar lookahead labels (`3 s` vs. `0 s`).
   - Inside `mitigate_hpss_v2`, default `env_norm="block"` (`env_normalize_scale_inv`) and the outer 64-tap zero-phase Kaiser `bandpass()` (`filtfilt`) operate strictly inside `[t - 30 s, t]` (zero access past $t$, i.e., block-causal). True **streaming STFT-frame causality** requires `env_norm="causal"` (`env_normalize_causal`, exact `0.0` future-frame diff on `mag_filt`) together with causal filtering and accounting for the `48`-sample (`3 s`) overlap-add synthesis support of `istft_complex_windowed`.
2. **Non-Degrading LOPO Guard (`1.00 × anchor_r`) & Physiological Step Floor (`σ_min = 1.0 bpm/hop`):**
   - Forensic inspection of `stage1-parsimony-v2` revealed that permitting a $5\%$ training correlation drop ($\rWithin^{\mathrm{train}} \ge 0.95\,\rWithin^{\mathrm{anchor}}$) alongside ultra-stiff $\sigma_{\mathrm{BR}} = 0.5\,\mathrm{bpm}/2\text{ s}$ (`0.25 bpm/s`) created a single-infant leverage cliff when high-variability `pat=11` (`11.1–11.3`) was held out: $\sigma_{\mathrm{BR}} = 0.5$ slipped past the guard on that fold, breaking monotonicity (`R1c > R1a` on `pool30_sync`), crashing `R1a` worse than memoryless `R0` on `br_trig_bpm` in `retained23` and `PAPER_18`, and pinning `R1f` at `9/11` folds while degrading held-out `r_within`.
   - Accordingly, Stage 1.1 locks the primary Rung 1 LOPO selection rule to **non-degrading correlation** ($\rWithin^{\mathrm{train}} \ge 1.00\,\rWithin^{\mathrm{anchor}}$) and physiological **$\sigma_{\mathrm{BR}} \in \{1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0\}\,\mathrm{bpm/hop}$** ($\ge 0.5\,\mathrm{bpm/s}$). Both `(guard=0.95, σ_min=0.5)` and fixed $\sigma_{\mathrm{BR}} = 2.0, \varepsilon = 10^{-3}, \beta = 1.0$ (zero fold-switching causality frontier) are exported side-by-side in `metrics_stage1_parsimony.json` for full transparency. `R1g_Offline_Viterbi_Unpinned` retains fixed $\beta = 1.0$ to preserve exact `0.00 bpm` path parity with Stage 0 `decode_viterbi_1d` (up to transition edge normalization).
3. **Dual-Reference Group-Delay (`+10 s` Monitor Lag) & Pause-Exclusion Diagnostic:**
   - Cross-correlation (`rmcorr` over $[-16\text{ s}, +16\text{ s}]$ lags on `pool30_sync`) shows that VitaGuard `br_1hz_30s` trails physical breath triggers `br_trig_bpm` by **`+10 s`** ($\rWithin$ rises from `0.606` at $0\text{ s}$ to `0.706` at $+10\text{ s}$). Under `branch_neutral_raw_nls_lag`, causal `R1a` (`+6 s` filter group delay) stacks with the baked-in monitor lag (`+16 s` peak lag against `br_trig_bpm`), suppressing zero-lag `r_within` on `Trig` purely via phase delay rather than amplitude inaccuracy.
   - Furthermore, `interval_rate_in_window` computes `median(60 / gaps)` after discarding inter-breath gaps $> 5\text{ s}$ (`< 12 bpm`), causing `br_trig_bpm` to report active-breathing rates during the `23.8%` of `pool30_sync` windows containing pauses. Stage 1.1 adds explicit lag cross-correlation profiles and whole-window breath-count rate `br_trig_count_bpm = 60 * N_events / 30 s` so Stage 2 cleanly separates genuine ERM offset overfitting from monitor lag and pause definitions.
4. **Gate `G2` Patient-Clustered TOST & Discrete-FFT `band_low_hz = 0.25 Hz` Lock:**
   - Gate `G2` is adjudicated on **both** session-level `tost_sess` and patient-clustered `tost_patient` ($N = 11$) across both `br_1hz_30s` and `br_trig_bpm`, with an explicit `gate_g2_verdict.decision` (`RETAIN_H0_LOCKED` if equivalence fails on either reference, falsifying Hypothesis `H1b` cleanly).
   - Inside `hpss_filter_causal(..., in_band=True)`, `band_low_hz = 0.25 Hz` (STFT bin 4 at $15.0\,\mathrm{bpm}$, since bin 5 is $0.3125\,\mathrm{Hz} = 18.75\,\mathrm{bpm}$ at $f_s=16\,\mathrm{Hz}, N_{\mathrm{FFT}}=256$) is explicitly documented as the discrete-FFT guard preserving full support at nominal `\bandLow = 0.30 Hz` ($18.0\,\mathrm{bpm}$).

---

## 8. Stage 2 Pre-Execution Lock (`stage2-complexity-v2`)

Prior to executing `exp2_stage2_complexity.py`, six hardening contracts are locked so Stage 2 compares Rungs 0–3 cleanly and matches Stage 1.1 provenance:

1. **Exact Stage 1.1 Rung 0–1 Parity (`guard_mult = 1.00`, `sigma_min = 1.0`):**
   - `R0_HPSS_NLS_Locked`, `R1a_Online_Forward_MMSE_0s`, `R1f_Offline_FB_MMSE_Inf`, and `R1g_Offline_Viterbi_Unpinned` inside `exp2_stage2_complexity.py` MUST use `1.00 * base_r` and `E1.SIGMA_GRID` (`sigma_br >= 1.0`) and match `metrics_stage1_parsimony.json` (`stage1-parsimony-v3`) within `< 1e-12` on `\rmseSess`, `\rmsePool`, and `\rWithin` across `pool30_sync`, `retained23`, and `PAPER_18`.
2. **Fair Physiological Step Floor on `Rung 2` (`sigma_slow >= 1.0 bpm/hop`):**
   - `DifferentiableSymmetricHMM` bounds `log_sig_slow` at `[log(1.0), log(6.0)]` (`>= 0.5 bpm/s`), matching Rung 1's Stage 1.1 floor so no differentiable HMM can gain unfair inertia over Rung 1 via sub-physiological smoothing.
3. **Steel-Manned Patient-Disjoint Inner-Validation Hyperparameter Selection for `Rung 3` Foils:**
   - `R3c_Ridge_Spectral_439`: Exact SVD-accelerated inner Leave-One-Patient-Out CV over `alphas = np.logspace(-1, 5, 13)` on valid equal-session training weights (eliminating overlapping-window GCV leakage).
   - `R3b_Spectral_MLP_15k`: Inner-validation selection over `alpha in (0.1, 1.0, 10.0, 100.0)` minimizing equal-session RMSE on held-out inner patient `va3_h0`.
   - `R3a_Causal_TCN_16k`: Inner-validation selection over `(weight_decay, dropout_p) in [(1e-3, 0.0), (1e-2, 0.0), (1e-2, 0.15), (5e-2, 0.25)]` with inverted hidden dropout during training only (`dropout_p = 0.0` at validation and inference).
4. **Two-Axis Causality Metadata (`block_lookahead_s` vs. `stft_lookahead_s`):**
   - `MODEL_META` replaces scalar `lookahead_s` with `block_lookahead_s` (`0.0` for `R0`, `R1a`, `R2a`, `R2b`, `R2d`, `R3a`, `R3b`, `R3c`; `None`/`"inf"` for `R1f`, `R1g`, `R2c`), `block_lookahead_label` (`"0 s"` / `"inf"`), and `stft_lookahead_s` (`3.0` for `H0`-backed models; `0.0` for `R2d` on `H5`).
5. **Four-Way Blind Reference Evaluation & Disentanglement (`pool30_sync`):**
   - Every model (trained solely on `br_1hz_30s`) is evaluated blindly across: (i) `br_1hz_30s`, (ii) `br_trig_bpm` (`0 s`), (iii) `br_trig_bpm_lag_aligned` (`lag_s = neutral_raw_nls_lag_s + 10`, removing VitaGuard's empirical `+10 s` monitor delay), and (iv) `br_trig_count_bpm` (`60 * N_events / 30 s`), alongside `pred_sd_ratio_1hz`, `[-16, +16] s` `rmcorr` lag profiles, and zero-touch `20–80 bpm` + chirp synthetic sweeps.
6. **Operator vs. Clinical-Grid Shift Equivariance:**
   - Pure shift equivariance (`< 1e-9 bpm` for `+10 bpm` rigid shift) is verified on wide interior padded grid `[-60, 180] bpm` (excluding edge clamping), complemented by the clinical `[18, 90] bpm` shift contrast between equivariant Rung 2 models and unconstrained Rung 3 foils plus the `[18, 90]` vs `[18, 120]` shifted-grid test (`< 0.05 bpm`).

### 8.1 Stage 2.1 Forensic Addendum — Closing the Tempered-Posterior Backdoor & Passband Calibration (`stage2-complexity-v3`)

During clean-room verification of `stage2-complexity-v2`, Gate 4 and Gate 6 exposed two critical findings that are locked below prior to final Stage 2 certification:

1. **Closing the Tempered-Posterior Backdoor in `DifferentiableSymmetricHMM`:**
   - Under permissive bounds (`w1 >= 0.5`, unconstrained negative `k1, k2`, `pi_jump <= 0.50`), `L-BFGS-B` drove `w1 -> 0.501`, `(k1, k2) -> (-0.280, -0.111)` (reducing 5-tap kernel DC gain `1 + 2k1 + 2k2` to `0.218`), `sigma_slow -> 6.0`, and `pi_jump -> 0.162`. The resulting effective emission log-temperature `~0.22` collapsed posterior sharpness (`H_t = 0.909`, `SD = 14.0 bpm`) so that uniform background floor `log(eps)` induced `0.4888 bpm` shifted-grid `[18, 90]` vs `[18, 120]` drift (`R2a`; `0.1970 bpm` `R2b`) and `0.366 bpm` trained operator non-equivariance on `[-60, 180]`.
   - Stage 2.1 locks an **untempered equivariant parameter box** (`beta_eff >= 1.0`): `w1 in [2.0, 5.0]` (prior anchored at `2.0`), `w2, w3 in [0.0, 3.0]`, `w4 in [-0.5, 0.5]`, `w5 in [0.0, 5.0]`, `w6 in [-0.5, 2.0]`, non-negative smoothing kernel taps `k1, k2 in [0.0, 0.5]`, bounded MLP residual head `W2 in [-0.10, 0.10]` for `R2b`, `sigma_slow in [1.0, 6.0]`, `sigma_jump in [5.0, 15.0]`, and `logit_pi in [-7.0, -4.5]` (`pi_jump <= 0.011`).
2. **Passband Calibration of the Synthetic CW Frequency Sweep (`28–80 bpm`):**
   - Double-pass 64-tap Kaiser `bandpass()` (`BAND_LOW_HZ = 0.30 Hz`) attenuates `20–25 bpm` (`0.333–0.417 Hz`) on the transition skirt below the Jacobi–Anger second harmonic (`40–50 bpm`), causing even locked zero-parameter `R0_HPSS_NLS_Locked` to incur `20.06 bpm` MAE at `20 bpm` while achieving `0.00–0.09 bpm` MAE across `28–80 bpm`. Stage 2.1 therefore evaluates passband out-of-range frequency transfer over `28, 35, 50, 65, 80 bpm` + `chirp_30_to_75bpm` while preserving `stat_20bpm_kaiser_edge` transparently as a front-end FIR skirt diagnostic.

### 8.2 Stage 2.2 Forensic Addendum — Closing the Bandpass Filter-Edge Backdoor, Shift-Probe Calibration & Physical SD Normalization (`stage2-complexity-v4`)

Forensic audit of `stage2-complexity-v3` revealed four Stage 2.2 remediations locked below:

1. **Closing the Bandpass Filter-Edge Backdoor (`R2e` / `R2f` 8-Parameter Filter-Safe Ablation):**
   - Although `R2a` (`11` params) is mathematically zero-offset shift-equivariant on abstract $\Phi$, fixed locked front-end filtering at `[0.30, 3.00] Hz` (`18–180 bpm`) clips `ch2 = log(P_clean(3f) + eps)` for candidates `f >= 60 bpm` (`3f >= 3.0 Hz`) and `ch3 = log(P_clean(0.5f) + eps)` for `f < 36 bpm` (`0.5f < 0.30 Hz`). Exploiting that fixed stopband wall (`w3 = +1.047, w4 = -0.474`), `R2a` created a `-7.78` log-likelihood cliff across `20 -> 90 bpm` (`-3.43` across `50 -> 60 bpm`) that penalized clinical tachypnea (`> 60 bpm`).
   - Stage 2.2 adds strictly passband-safe `1f`-only 8-parameter ablations `R2e_Causal_DiffHMM_FilterSafe8` (causal forward) and `R2f_Offline_DiffHMM_FilterSafe8` (offline FB) restricted exclusively to `FILTERSAFE_CH_IDX = (0, 4, 5)` (`P_clean(1f)`, `A_coarse(f)`, `P_raw(1f)`), whose support lies inside `[0.30, 3.00] Hz` across the entire `[18, 90] bpm` grid.
2. **Shift-Probe `A_coarse` Floor Alignment & End-to-End Waveform Shift Probe:**
   - The Stage 2.1 clinical `[18, 90]` salience probe (`p_b -> p_s`) omitted `np.maximum(0.05, gauss)` on `ch4`, causing `-444.6 sigma` standardization outliers that saturated `R3b` against `12 bpm`. Stage 2.2 aligns `ch4` (`C.NLS_RADIUS_BPM`, floor `0.05`) in `_make_clinical_shift_salience_pair` and adds an end-to-end 24 GHz radar waveform `+10 bpm` shift probe (`probe_shift_36bpm` $\to$ `probe_shift_46bpm`).
3. **Full L-BFGS-B Convergence (`maxiter = 200`):**
   - Raise `L-BFGS-B` `maxiter` from `80` to `200` across all Rung 2 models (`R2a..R2f`) so 43-param `R2b` and offline `R2c` converge cleanly, logging `nit` and `success` per fold.
4. **Cross-Cohort `-10 s` Lag-Aligned `Trig` & Physical Dynamic-Range Ratio:**
   - Export `br_trig_bpm_lag_aligned` (`lag_s = neutral_raw_nls_lag_s + 10`) on all three cohorts (`pool30_sync`, `retained23`, `PAPER_18`) and report physical dynamic-range retention $\overline{\mathrm{SD}}(\hat{f}_b) / \overline{\mathrm{SD}}(f^{\mathrm{ref}}_{\mathrm{Trig},-10\mathrm{s}})$ alongside the `1 Hz` SD ratio.

---

## 9. Stage 3 Pre-Execution Lock (`stage3-uncertainty-v1`)

Prior to executing `exp3_stage3_uncertainty.py`, six contracts are locked for `WP3/WP4` Bayesian/Spectral Selective Prediction, Quiet-Apnea Pause Safety, Out-of-Distribution (OOD) Abstention, Finite-$K$ Conformal Calibration, and Gate `G4` Hard Freeze:

1. **Exact Stage 1.1 & Stage 2.2 Full-Coverage Parity (`c_cov = 1.00`, `< 1e-12`):**
   - At full retention ($c_{\mathrm{cov}} = 1.00$), Equal-Session Macro-Averaged Selective Risk $\mathrm{RMSE}_{\mathrm{sess}}(1.00)$ and Window-Pooled Selective Risk $\mathrm{RMSE}_{\mathrm{pool}}(1.00)$ MUST match the frozen Stage 1.1 (`stage1-parsimony-v3`) and Stage 2.2 (`stage2-complexity-v4`) predictions within `< 1e-12` across `pool30_sync`, `retained23`, and `PAPER_18`.
2. **8-Proxy Label-Free Uncertainty & Signal Quality Index Family ($\mathrm{SQI}_{s,w} = 1 - u_{s,w} \in [0, 1]$):**
   - `U1_Posterior_Entropy` ($H_t \in [0, 1]$): Normalized Causal Forward-HMM Shannon entropy $-\frac{1}{\ln N_B}\sum_{q=1}^{N_B} p_t(q)\ln p_t(q)$.
   - `U2_Posterior_SD` ($\tilde{\sigma}_{\mathrm{post},t} = \mathrm{clip}(\sigma_{\mathrm{post},t} / \sigma_{\mathrm{unif}}, 0, 1)$): Causal Forward-HMM posterior standard deviation normalized by uniform `[18, 90] bpm` grid standard deviation $\sigma_{\mathrm{unif}} = \sqrt{\frac{1}{N_B}\sum_{q=1}^{N_B}(f_q - \bar{f})^2} \approx 21.0713\,\mathrm{bpm}$.
   - `U3_Spectral_Entropy_Clean` ($H_{\mathrm{spec},t} \in [0, 1]$): Memoryless normalized Shannon entropy of post-HPSS folded fundamental PSD $\tilde{P}_{\mathrm{clean}}(f_q)$ over `[18, 90] bpm`.
   - `U4_Spectral_Entropy_Raw` ($H_{\mathrm{raw},t} \in [0, 1]$): Memoryless normalized Shannon entropy of pre-HPSS raw folded PSD $\tilde{P}_{\mathrm{raw}}(f_q)$ over `[18, 90] bpm`.
   - `U5_HQI_Harmonic_Deficit` ($1 - \mathrm{HQI}_t \in [0, 1]$): Complement of the Harmonic Quality Index (`\hqiScore`), measuring fundamental ($\pm 2\,\mathrm{bpm}$) plus second-harmonic peak energy concentration relative to passband and subharmonic ($0.5f$) background.
   - `U6a_ZeroTuned_Additive_SQI` ($u_{\mathrm{add},t} = 0.45\,\tilde{\sigma}_{\mathrm{post},t} + 0.35\,H_t + 0.20\,b_{\mathrm{OOD},t}$): Zero-parameter additive Bayesian–OOD composite (`0` learned abstention weights), where $b_{\mathrm{OOD},t} = \mathrm{clip}((H_{\mathrm{mix},t} - 0.90)/0.10, 0, 1)$ is the strictly causal 4-window ($[t-6\,\mathrm{s}, t]$, `0.0` future lookahead) temporal mixture spectral entropy tail barrier separating real clinical windows ($95\text{th percentile} \le 0.913$, including quiet-apnea pauses) from OOD white noise ($H_{\mathrm{mix},t} \ge 0.945$).
   - `U6b_ZeroTuned_Geometric_SQI` ($\mathrm{SQI}_{\mathrm{geom},t} = [(1 - \tilde{\sigma}_{\mathrm{post},t})(1 - H_t)(1 - b_{\mathrm{OOD},t})]^{1/3}$, $u_{\mathrm{geom},t} = 1 - \mathrm{SQI}_{\mathrm{geom},t}$): Zero-parameter equal-exponent geometric-mean confidence composite (`0` learned abstention weights; logical-AND OOD spectral tail veto coupled with Causal Forward-HMM posterior concentration, preventing unthresholded single-window $H_{\mathrm{spec},t}$ from leaking pause over-rejection into in-distribution tracking).
   - `U6c_LOPO_Calibrated_SQI`: Patient-disjoint LOPO Ridge-regularized non-negative least squares (`Z-NNLS`, $\lambda = 100$) trained strictly on the $K-1$ training patients' `br_1hz_30s` absolute errors $|e_{s,w}|$ over standardized $(H_t, \tilde{\sigma}_{\mathrm{post},t}, H_{\mathrm{spec},t}, H_{\mathrm{raw},t}, 1 - \mathrm{HQI}_t)$, testing whether learned SQI weights improve over zero-tuned `U6a`/`U6b` under small-$N$ LOPO (and demonstrating how unconstrained linear weighting of single-window $H_{\mathrm{spec},t}$ trades pause neutrality for pooled variance reduction).
3. **Dual-Aggregation Risk–Coverage (`AURC`, `E-AURC`, `AUGRC`) & Clustered Inference ($c_{\mathrm{cov}} \in [0.50, 1.00]$):**
   - Evaluates both **Equal-Session Macro-Averaged Selective Risk** $\mathrm{RMSE}_{\mathrm{sess}}(c_{\mathrm{cov}}) = \frac{1}{S}\sum_{s=1}^S \mathrm{RMSE}_s(c_{\mathrm{cov}})$ (each session retaining its lowest-uncertainty $k_s(c_{\mathrm{cov}}) = \max(1, \mathrm{round}(c_{\mathrm{cov}} |\mathcal{W}_s|))$ windows via deterministic stable merge-sort) and **Window-Pooled Selective Risk** ($\mathrm{RMSE}_{\mathrm{pool}}(c_{\mathrm{cov}})$, $\mathrm{MAE}_{\mathrm{pool}}(c_{\mathrm{cov}})$, and $>10\,\mathrm{bpm}$ gross-error tail %) over the 51-point grid $c_{\mathrm{cov}} \in [0.50, 1.00]$ (`step = 0.01`, plus full $[0.10, 1.00]$ curve).
   - Reports exact trapezoidal integration (`/ 0.50`) for $\mathrm{AURC}_{[0.5, 1.0]}$, $\mathrm{E\text{-}AURC}_{\mathrm{vs\_rand}} = \mathrm{AURC}_{[0.5, 1.0]} - R(1.00)$ (`< 0` denotes positive selective gain over random abstention), $\mathrm{E\text{-}AURC}_{\mathrm{vs\_oracle}} = \mathrm{AURC}_{[0.5, 1.0]} - \mathrm{AURC}_{\mathrm{oracle},[0.5, 1.0]} \ge 0$, oracle-normalized efficiency $\eta_{\mathrm{sel}}$, and Traub et al. (NeurIPS 2024) Generalized Risk $\mathrm{AUGRC}_{[0.5, 1.0]}$ & $\mathrm{E\text{-}AUGRC}_{\mathrm{vs\_rand}}$ where $GR(c) = \frac{k(c)}{N} R(c)$, accompanied by 10,000-resample patient-clustered bootstrap 95% CIs (`seed = 20261011`) and session/patient paired Wilcoxon signed-rank tests with Holm–Bonferroni correction.
4. **Quiet-Apnea Respiratory Pause Safety Stratification (`1024 Hz Resp.-Trigger`, `resp_max_gap_s > 5.0 s`):**
   - Audits uncertainty distributions and empirical rejection rates at $c_{\mathrm{cov}} \in \{0.90, 0.80, 0.70\}$ across verified respiratory pause windows (`resp_max_gap_s > 5.0 s`, $N = 1,581$ on `pool30_sync`, `23.8%` of valid windows) vs. regular breathing windows ($N = 5,059$), testing whether causal Forward-HMM state-space tracking (`U1`, `U2`, `U6a`, `U6b`) eliminates the disproportionate pause over-rejection (`risk_ratio <= 1.15`) exhibited by memoryless spectral entropy (`U3`, `U4`).
5. **Out-of-Distribution (OOD), WGN, Surrogate & Firmware-Dropout Abstention Audit (`Gate G4`):**
   - Evaluates OOD rejection `AUROC` and `FPR@95%TPR` against valid `pool30_sync` windows across: (i) Complex White Gaussian Noise (`N = 200` windows $\times$ `480` samples, fixed `seed = 20261011`), (ii) 5-seed Phase-Scrambled Radar Surrogates (`33,200` windows; documenting Parseval global-PSD preservation vs. local spectral & causal temporal Jensen–Shannon entropy), and (iii) `UCD Status != ChildPresent` windows (`ucd_nonchild_w > 0` and `ucd_nonchild_w >= 0.5`), explicitly separating strictly $O(2) \times \mathbb{R}_{>0}$-invariant waveform-only proxies from hardware-telemetry-gated abstention.
6. **5-Regime Finite-$K$ Conformal Prediction Audit (`C1..C5` across $\alpha \in \{0.05, 0.10, 0.20, 0.30\}$):**
   - Empirically and mathematically contrasts:
     - `C1_Naive_RandomWindowSplit_50_50`: Random 50/50 window split ignoring $93.3\%$ window overlap and patient clustering.
     - `C2_Naive_LOPO_PooledWindow`: Patient-disjoint LOPO split using the pooled training-window quantile $\lceil (N_{\mathrm{cal}}+1)(1-\alpha) \rceil / N_{\mathrm{cal}}$ (ignoring infant clustering).
     - `C3_SessionCluster_Hierarchical_LOPO`: Session-level hierarchical conformal over $S_{\mathrm{cal}} \in [25, 27]$ training sessions per LOPO fold.
     - `C4_PatientExchangeable_Hierarchical_LOPO`: Exact patient-exchangeable hierarchical conformal (Dunn et al., JASA 2023) over $K_{\mathrm{cal}} = 10$ calibration infants per `pool30_sync` LOPO fold (`K_cal = 9` on `retained23`/`PAPER_18`, `K_cal = 11` on `all_paired`), proving vacuity ($[-\infty, +\infty]$) at $\alpha = 0.05 < \frac{1}{K_{\mathrm{cal}}+1}$ ($K_{\mathrm{cal}} \ge 19$ required for $95\%$ coverage) and valid $\ge 1-\alpha$ patient coverage across $\alpha \in \{0.10, 0.20, 0.30\}$.
     - `C5_LocallyAdaptive_BayesConformal_Hierarchical_LOPO`: Patient-exchangeable hierarchical conformal on Forward-HMM posterior-normalized residuals $|\hat{f}_{b,t} - f^{\mathrm{ref}}_t| / \max(\sigma_{\mathrm{post},t}, 1.0\,\mathrm{bpm})$, yielding time-varying heteroscedastic intervals $\hat{f}_{b,t} \pm \hat{q}_{1-\alpha}^{\mathrm{adapt}}\max(\sigma_{\mathrm{post},t}, 1.0\,\mathrm{bpm})$.
7. **Gate `G4` Hard-Freeze Manifest & Full-Cohort (`K=11` / `pool30_sync`, $S=28$) Model Archive (`frozen_stage4_models.npz`):**
   - Upon passing Gate `G4`, hard-freeze all internal `WP1–WP4` architecture elections, lookahead lags (`H0_Locked_Rect_Full_L3`: `48 samples = 3.0 s` at `16 Hz`; `H5_Causal_COLA_InBand_L0`: `0 samples = 0.0 s` at `16 Hz`), conformal calibration quantiles (`C4` and `C5` across $\alpha \in \{0.10, 0.20, 0.30\}$), and both 11-fold LOPO mean and full-cohort (`K=11`, $S=28$ non-empty `pool30_sync` sessions) refit weights for `R1a`, `R1h`, `R1f`, `R2e`, `R2f`, `R2a`, `R2d`, **`R3c_Ridge_Spectral_439`**, **`R3b_Spectral_MLP_15k`**, and **`R3a_Causal_TCN_16k`** into `thesis/outputs/stage3_uncertainty/frozen_stage4_models.npz` (with SHA-256 hash recorded in `hard_freeze_manifest_for_stage4_l4`), pre-registering the primary in-band (`28–80 bpm`) vs. out-of-passband FIR-skirt (`< 28 bpm`) evaluation strata before touching `external_data/gleichauf2024/`.

---

## 10. Stage 4 Pre-Execution Lock (`stage4-external-v1`, Zero-Touch External `L4` Benchmark)

Prior to reading or processing any waveform arrays in [`external_data/gleichauf2024/`](file:///Users/hosamshibly/Downloads/medical_radar/external_data/gleichauf2024/README.md), three pre-flight contracts are locked for the strict one-shot, zero-touch `L4` benchmark:

1. **Primary vs. Sensitivity Frozen Parameterization:**
   - **Primary Stage 4 Point Estimate:** **`Full-Cohort K=11` refit** (`r2e_full_refit_theta`, `r2f_full_refit_theta`, `r2a_full_refit_theta`, `r2d_full_refit_theta`—keys inside `.npz` omit the `frozen_` prefix—matching full-cohort `R3c` on all 11 patients, `R3a`/`R3b` trained on 10 patients with `val_pat_full=12` held out for early stopping, and full-`K=11` `C4`/`C5` conformal quantiles) loaded verbatim from SHA-256-locked `thesis/outputs/stage3_uncertainty/frozen_stage4_models.npz` (`af33a83e157f7d2efb0f47e50c9544ee19bacb7da88ab0e0235ccead029cbcc4`).
   - **Secondary Sensitivity Check:** **`LOPO-Mean`** (`*_lopo_mean_theta` for `R2e`, `R2a`, `R2d`, and 11-fold LOPO-mean `C4`/`C5` conformal quantiles), evaluated and reported side-by-side without post-hoc selection.
2. **Mutually Exclusive & Exhaustive Frequency Strata (`14 + 6 = 20`, 100% of Scenarios) + Exact Locked Bandpass Specification:**
   - **Locked Bandpass Specification (`paper_workspace/src/filtering.py`):** Double-pass zero-phase (`filtfilt`) **64-tap Kaiser-windowed FIR bandpass filter (`KAISER_BETA = 6.5`, nominal cutoffs `[0.30, 3.00] Hz` = `18–180 bpm`)**, whose transition skirt attenuates low frequencies (`0.300 Hz` / `18 bpm`: `-12.07 dB` magnitude; `0.333 Hz` / `20 bpm`: `-10.01 dB`) before entering the flat passband region (`>= 28 bpm`, `-4.12 dB` at `28 bpm`, `-3.17 dB` at `30 bpm`, `-0.13 dB` at `45 bpm`), feeding the `[18.0, 90.0] bpm` (`0.30–1.50 Hz`) HMM grid (superseding the shorthand `"0.40–1.50 Hz 4th-order Butterworth"` prose string in Stage 3's unmutated JSON metadata).
   - **Primary In-Band Stratum (`f_gt >= 28.0 bpm`, $N = 14$):** All `14` passband scenarios `{30, 40, 45, 50, 55, 60, 80} BPM × {normal, deep}`, with pre-registered sub-stratifications: `normal` (`7`) vs. `deep` (`7`), and core `30–60 bpm` (`12`) vs. extreme tachypnea `80 bpm` (`2`).
   - **Secondary Sub-Band / FIR-Skirt Stratum (`f_gt < 28.0 bpm`, $N = 6$):** All `6` sub-band scenarios `{05, 10, 20} BPM × {normal, deep}`, with sub-stratifications: `20 bpm` Kaiser transition skirt (`2`, within `[18, 90] bpm` grid but on `-10 dB` FIR skirt) vs. `05–10 bpm` stopband / below-grid (`4`, physically below `[18, 90] bpm` support; diagnostic OOD/stopband test).
   - **Combined (`all_20`, $N = 20$):** Disjoint, exhaustive union (`14 + 6 = 20`, 100% of scenarios).
3. **Deterministic Format Alignment & Zero External Adaptation:**
   - Hardware difference logged transparently: InnoSenT iSYS-4001 `24.2 GHz` CW radar ($\lambda = 12.39\,\mathrm{mm}$, raw ADC `~2000.02 Hz`) vs. NICU Beltrão `24.0 GHz` ($\lambda = 12.50\,\mathrm{mm}$, `16 Hz`); **zero** downstream parameter, threshold, filter, window, or scaler adjustment is permitted on `external_data/gleichauf2024/`.
   - Deterministic zero-mean two-stage anti-aliasing decimation: `(I - mean(I), Q - mean(Q))` $\to$ `scipy.signal.resample_poly(..., up=1, down=10, window=("kaiser", 6.5))` to `~200 Hz` $\to$ linear interpolation onto exact uniform `200.000 Hz` grid over `[0, t_end]` $\to$ `scipy.signal.resample_poly(..., up=2, down=25, window=("kaiser", 6.5))` to exact `16.000 Hz` matching `C.FS_RADAR = 16 Hz`, preserving exact $O(2) \times \mathbb{R}_{>0}$ I/Q group invariance (`< 1e-12`).
   - Ground truth: Independent `~2.5 kHz` Micro-Epsilon optoCONTROL 2520 laser micrometer steady-state rate `f_laser_mid` (Hann-windowed zero-padded FFT over steady-state `[t0+100s, t0+700s]`; both `f_laser_mid` and nominal `f_nom` logged and compared).
   - Window regimes: **Primary `steady_state_excl_sync_pulses`** (`w_start >= 10.0 s` and `w_end <= T - 10.0 s`, deterministically excluding the simulator's start/end synchronization steps at `t ≈ 4–6 s` and `t ≈ T - 2 s`) and **Sensitivity `all_windows_incl_sync_pulses`** (all overlapping `30 s / 2 s` windows across `[0, T]`).



