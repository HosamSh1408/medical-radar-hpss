# AUDIT_PLAN.md — Medical-Radar "Colossal Mega Audit" — Master Plan

Status: proposed. This document is the master plan for a full adversarial audit of the
`medical-radar-hpss` repository. It is grounded in an initial reconnaissance pass that
read every source module, the MATLAB ground-truth mapping logic, the paper's Methods
(extracted text), the metadata ledgers, and verified key claims empirically (test suite,
EDF/CSV schema inspection, filter measurement, window-count recomputation).

Scope of audit target:

- `README.md` and `RESEARCH_AND_ENGINEERING_LOG.md` — the *claims*.
- `paper_workspace/src/` — the *implementation*.
- `paper_workspace/outputs/` and `paper_workspace/metadata/` — the *artifacts*.
- `main.py` — the *orchestration*.

---

## 0. Mandate & doubt-first philosophy

Treat the repository as **guilty until proven correct**. Every number in the documentation,
every line in `paper_workspace/src/`, and every binary in `outputs/` must independently
survive falsification. The repo self-describes as *"provenance-locked," "audited,"* and
*"cryptographically verified"* — those are exactly the claims this audit attacks hardest,
because reconnaissance already found them partly false (see §2).

**Definition of "bulletproof" (acceptance bar):** every documented claim is

1. traceable to a paper equation / MATLAB reference / artifact,
2. reproducible bit-for-bit by an *independent* re-implementation from the ground-truth
   sources, and
3. robust to adversarial negative controls, and
4. not over-described relative to what the code actually does.

---

## 1. Ground-truth references (the audit anchors)

| Anchor | Extracted fact | Relevance |
|---|---|---|
| Paper Methods | CSD adopted; **bandpass Kaiser β=6.5, 0.3–3.0 Hz (18–180 bpm)**; STFT **rectangular 3 s, 2 s overlap, zero-pad 256**; NMF **K=11**; harmonic NLS **Lk=2**, search **±5 bpm** around a time-domain coarse estimate; estimator = **summation of harmonics over the PSD** via FFT + grid search | The code must be measured against *this*, not the repo's prose |
| Paper Results | **18 valid measurements, ~4,964 valid estimates**, overall RMSE 6.38 bpm | The repo reports 5,466 windows / 6.86 bpm — a different quantity |
| `readRadarRaw.m` | CSV cols **11,12** (`I value`, `Q value`), `readmatrix` | Verify loader equivalence |
| `readReferenceData.m` | `record{:,2}` = Resp, `record{:,9}` = RespRate | Verify EDF channel indexing |
| EDF (empirically) | **Resp = 256 Hz**, RespRate = 1 Hz, 11 signals, clipped at **±3.0** | Contradicts repo docs (§2) |
| Radar CSV (empirically) | 24,000 rows @16 Hz = 1500 s; columns `I value`, `Q value` | Length differs from reference (1500 vs 1526 s) |

---

## 2. Confirmed defects from reconnaissance (Category A — red flags)

These are **already-verified** doc↔code↔artifact contradictions. They seed the audit and
prove it is not speculative.

| # | Finding | Evidence | Severity |
|---|---|---|---|
| A1 | **Filter misdescribed.** Docs say "5th-order zero-phase Butterworth [0.33,1.50] Hz". Code is a **64-tap Kaiser FIR, 0.3–3.0 Hz** (which *matches the paper*; the docs do not). | `filtering.py:7-11`, `config.py:51-54`; `README.md:75`; log lines 95,143,166 | High (integrity) |
| A2 | **Reference sample rate wrong in docs.** "25 Hz impedance pneumography"; actual EDF `Resp` is **256 Hz**. | EDF inspection; `README.md:50`; log line 249 | High |
| A3 | **Harmonic order wrong in docs.** Log says "K=3 harmonics"; code/paper use **Lk=2**. | `config.py:57`; log line 167 | Medium |
| A4 | **Multiverse window counts don't match artifact.** README/log: pool30 **6,857**, retained23 **5,976**. Actual CSV & recompute: **6,642** and **6,447**. | `reconstruction_multiverse.csv`; recomputed from caches; `README.md:35-36` | High (reproducibility) |
| A5 | **"Parallel 8-worker batch" is fiction.** `run_batch.py` is a sequential for-loop; no multiprocessing. Speedup headline ("6.92×, 8-worker") is unsupported by the code. | `run_batch.py:17-48`; `README.md:39,84`; log line 175 | High |
| A6 | **Provenance is incomplete.** `_CORE_FILES` hashes 14 modules but **omits `protocols.py`** (scenario labels, twin/supine strata) — changing it silently alters tables/figures without invalidating caches; the badge "33/33 verified" overstates coverage. | `provenance.py:23-28` | High |
| A7 | **Dead header-timestamp path.** `getStartDatetime()` does not exist in pyedflib 0.1.42; the call is swallowed by `except`, so all metadata lags silently come from filename parsing only. | `lags.py:120-131`; verified AttributeError | Medium |
| A8 | **Saturation gate misdescribed.** Log says "≤5% per 30-s window"; code requires **exactly 0** saturated seconds. | `evaluation.py:101`; log line 172 | Medium |
| A9 | **`env_normalize` misdescribed.** Log claims "0.2×median(peaks) gating"; code is a hard per-frame gain clip `[0.5,2.0]`. | `rbm_nmf.py:79-86`; log line 101 | Medium |
| A10 | **Dead code in the "canonical" path:** `demod.ad`, `lags.joint_dynamic_lag`, `lags.hybrid_lag`, `evaluation.ref_saturation_mask`, `evaluation.ref_valid_from_raw`, `estimation.dual_center`, `rbm_nmf.detect_rbm_hoyer`/`hoyer_sparsity`, and the `NMF_PRESET='attached_locked'` branch; the log references a `valid_mask` function that does not exist. | grep results; log line 171 | Low–Medium |
| A11 | **`stft_complex` comment lies** ("tail: pad last partial frame") — no tail frame is added. | `rbm_nmf.py:23` | Low |
| A12 | **Timing drift.** README 5.46/37.79 ms; artifact means 5.51/37.86 ms. | CSV vs README | Low |

---

## 3. Audit workstreams

Each workstream lists the attack, the method of falsification, and the pass gate.

### WS1 — Algorithm fidelity vs. the paper (equation-by-equation)

- **CSD** (`demod.csd`): confirm per-window DC removal is the paper's CSD. Derive the
  paper's `x(t)=I+jQ` model; test on synthetic I/Q with known gain/phase imbalance;
  quantify sensitivity to DC drift, amplitude imbalance, phase offset.
- **Autocorr coarse estimator** (`filtering.autocorr_coarse`): paper says peak detection on
  `r(t)` → mean inter-peak time. Code uses `find_peaks(distance=fs/3, prominence=0.1)` then
  **mean of all valid gaps** and `1/mean`. Falsify: (i) mean-of-gaps is not a robust period
  estimator; (ii) `prominence=0.1` is fixed; (iii) lag-0 is included. Negative control:
  synthetic multi-harmonic + noise vs. paper's procedure and vs. a robust estimator.
- **PSD fold** (`estimation.psd_fft`): code folds `|X(+f)|²+|X(-f)|²`. For a **complex** CSD
  signal this mixes non-conjugate frequencies; the paper's model is complex one-sided.
  Falsify on synthetic analytic vs non-analytic complex signals; show whether the fold
  biases the NLS score.
- **NLS** (`estimation.nls_estimate`): reproduce Eq.14–17 exactly from the paper
  (Vandermonde/Zk, Lk=2, grid `{0, 2π/N, …}`); compare to the PSD-sum + `np.interp` +
  0.1 bpm grid. Quantify estimator bias/variance vs. the paper implementation.
- **NMF** (`rbm_nmf`): K=11 ✓, but is `nndsvda` + Frobenius + `alpha_H=0` the paper's?
  Falsify the RBM-detection rule (`alpha=3.0`, top-3>0.6, mean over full H) as a faithful
  reading of "adaptive amplitude threshold + sparsity". Reproduce Fig. 4c–g on session 7.3,
  window @60 s, and diff visually / by PSNR.
- **Gate:** independent re-implementation of CSD→STFT→NMF→iSTFT→bandpass→autocorr→NLS from
  the paper text by a second author track, matching within tight numerical tolerance.

### WS2 — HPSS correctness & the fairness of the A/B

- Prove/edit the median-filter geometry (`fftshift` axis, `(1,L_harm)` vs `(L_perc,1)`,
  `mode='nearest'`, `L_harm=7` vs the docstring's 5) and the soft-mask algebra.
- **Fairness audit:** confirm upstream (CSD, pre-bandpass, STFT) and downstream (iSTFT,
  post-bandpass, autocorr, NLS, sync) are byte-identical between branches, or document
  every difference.
- **Asymmetry bug:** `minmove` (Fig. 7 subset) is defined **only by NMF** (`n_removed<=2`,
  `run_all.py:37`); HPSS is scored on NMF-chosen windows. Falsify with an HPSS-intrinsic
  movement proxy.

### WS3 — Evaluation, synchronization & leakage (highest scientific risk)

- **Lag is tuned on test data:** `sync_arrays(est, …)` maximizes correlation of **NMF**
  estimates against the reference (`run_all.py:30-32`), then that lag scores *all* methods.
  This is a hyperparameter fit to the target → **optimistically biased accuracy, biased
  toward NMF**. Falsify: freeze lag from metadata only; report degenerate-lag sensitivity;
  quantify the inflation.
- Whole-window vs center-sample gating; the exact 0-saturation rule (A8); window-averaging
  of the reference; NaN handling; boundary windows.
- **Metrics:** `scores` uses `err<t` strict-`<`; verify against the paper's accuracy
  definition; verify RMSE/MAE/bias denominators; Bland-Altman SD convention (paper vs
  `np.std` ddof).

### WS4 — Statistical rigor & claim calibration

- **"Equivalence" is an overclaim.** Log §1 asserts "Statistical Equivalence"; the code only
  shows CIs spanning zero (failing to reject is not equivalence). Require a pre-registered
  **TOST/margin** or downgrade to "no detected difference."
- Bootstrap: `BOOTSTRAP_SEED + n` makes the seed cohort-dependent; percentile CI without
  BCa; 2,000 resamples; session- and patient-clustered variants. Audit RNG determinism,
  resampling unit, and multiplicity across 4 cohorts × 5 scenarios × 2 metrics (no
  correction).
- **Outcome-dependent cohort:** `PAPER_18` is explicitly "reference-validity-ranked" (itself
  correlated with error) yet is the **headline** table. Run sensitivity removing it; confirm
  conclusions are unchanged.
- Denominator: 5,466 vs the paper's ~4,964; quantify what construction yields each; keep the
  "exact reproduction" gate **blocked**.

### WS5 — Provenance & reproducibility infrastructure

- Fix/extend `_CORE_FILES` (A6); include `protocols.py`, and define which consumers
  invalidate what.
- Verify 33/33 caches genuinely pass `cache_is_current` **on a clean checkout and a second
  Python** (3.9.6 pinned); document brittleness (any whitespace → mass invalidation).
- Confirm `main.py --all` / `--batch --force` rebuild reproduces the committed CSVs
  bit-for-bit (hash the outputs).
- Audit `requirements.txt` vs `requirements-lock.txt`; enforce a lock; CI that fails on stale
  caches.

### WS6 — Data ingestion & ID mapping

- Verify `skiprows=2` + named columns ≡ `readmatrix` cols 11/12 across **all** CSVs; check
  for variant headers, NaNs, mixed dtypes, timestamp discontinuities, dropped samples.
- Verify EDF channel resolution **by label** (not the fragile index fallback 1/8); confirm
  every EDF has Resp@256 / RespRate@1; audit the `10.3`/`1.2`/`3.2`/`10.4` mapping and the
  zero-overlap claims (`1.1, 2.1, 2.2`) with duration math.
- Validate `protocols.MEASURE_INFO` against `annotations.pdf` line-by-line; quantify the
  `7.2` "assumed" label's effect.

### WS7 — Cohort construction & multiverse integrity

- Recompute all four cohorts, valid windows, and evaluable counts from caches; reconcile A4.
- Verify `retained23 ⊂ pool30` and exclusion bookkeeping; audit `retained_26` (superseded,
  still in code) for spurious scoring warnings (2.2).
- Check `cohort_ledger.csv` hashes match the working tree; regenerate and diff.

### WS8 — Performance claims

- Replace sequential timing with a **controlled benchmark**: warmup, N repeats, report
  mean±SD and percentiles; separate solver time from mask/iSTFT; state single-thread
  explicitly.
- Reconcile with A5 ("8-worker") and A12; put the benchmark in CI with a reproducible
  harness.

### WS9 — Numerical & engineering hardening

- Remove dead code (A10) or quarantine it behind explicit flags; make docstrings match the
  code (A9, A11).
- Determinism tests (NMF seed, filters, FFT), dtype/finite guards, input validation, error
  paths (`filtfilt` edge cases on short windows), device/OS invariance, memory.
- Add property-based / mutation tests; raise test assertions from smoke-level (e.g.
  `test_real_window` only checks `18 <= f <= 180`) to metric regression fixtures with golden
  values.

### WS10 — Figures/artifacts ↔ claims traceability

- Every figure number must be machine-regenerated and diffed against the committed PNG's
  embedded data; verify axis limits, N labels, and RMSE values printed in legends equal the
  caches.
- Check for hard-coded values (the log's Phase 1 admits earlier hard-coded runtime bars);
  prove none remain.

---

## 4. Cross-cutting adversarial methodology

1. **Negative controls:** shuffle the reference; time-reverse; replace the signal with white
   noise / a pure 1 Hz tone; confirm metrics degrade to chance and do not accidentally "pass."
2. **Semi-synthetic physics benchmark** (the log's own roadmap): inject real RBM bursts onto
   simulated neonatal chest motion over SAR ∈ [−20, +10] dB; validate both branches against
   known ground truth — the only way to measure true error where the reference sensor
   saturates.
3. **Invariance tests:** resample offset, window phase, amplitude scale, channel swap (I↔Q),
   sign flips; results must be invariant or the sensitivity documented.
4. **Independent re-implementation** (WS1/WS2) in a separate module or clean repo, compared
   numerically.
5. **Doc-truth table:** every sentence in README/log mapped to a code line or artifact,
   machine-checked where possible.

---

## 5. Prioritization

| Priority | Items |
|---|---|
| **P0 (blocks trust)** | WS3 leakage/lag bias; WS4 "equivalence" overclaim; A4 count mismatch; A5 fake parallelism; A6 provenance gap; WS1 NLS/PSD fidelity |
| **P1 (scientific)** | WS2 fairness/minmove asymmetry; WS7 cohort reconciliation; WS4 multiplicity; WS6 mapping |
| **P2 (engineering)** | WS9 hardening/tests; WS8 benchmark; A1, A2, A3, A7, A8, A9 doc-truth fixes |
| **P3 (polish)** | A10 dead code, A11 comment, A12 drift, traceability automation |

---

## 6. Phased execution (proposed)

- **Phase 0 (48 h):** freeze repo + hash the tree; build the doc-truth table (A1–A12
  confirmed/fixed); write the reproducible benchmark harness; add golden-value regression
  tests. *No claim changes yet.*
- **Phase 1 (1 wk):** WS1 (paper-exact re-implementation) + WS2 (fairness) + independent
  NLS/PSD reproduction.
- **Phase 2 (1 wk):** WS3 (remove lag leakage, re-score all artifacts), WS6, WS7; regenerate
  tables/figures; reconcile all numbers.
- **Phase 3 (3–4 d):** WS4 (TOST or downgrade claims), WS5 provenance fix + clean-env
  verification, WS8 benchmark.
- **Phase 4 (3–4 d):** adversarial controls + semi-synthetic SAR sweep; WS10 traceability.
- **Phase 5:** rewrite README/log to the audited truth; regenerate everything under a new
  `PIPELINE_VERSION`; final frozen artifact set.

---

## 7. Deliverables / definition of done

1. `AUDIT_FINDINGS.md` — every finding with `file:line`, evidence, severity, status.
2. `DOC_TRUTH_TABLE.csv` — each documentation claim → code/artifact → verified/false.
3. A **clean-room re-implementation** with a numeric diff vs. the repo.
4. Regenerated, hash-locked tables/figures + a CI that fails on provenance/regression drift.
5. A recalibrated claims section (no "equivalence"/"reproduction" without a margin/denominator
   match) and an explicit "cannot establish" list (L2 / L4 / L5).

---

## 8. Explicitly out of scope (cannot be fixed by auditing)

Exact paper reproduction (cohort IDs / denominator unknown), external generalization, and
clinical readiness — the repo already flags these; the audit must ensure they stay
**blocked**, not quietly implied.

---

## 9. Bottom line

This is a well-built pipeline whose *code mostly matches the paper*, but whose
**documentation, claims, provenance coverage, and evaluation protocol contain concrete,
already-verified errors** (A1–A12). The most serious are test-data-tuned synchronization
(WS3) and an "equivalence" overclaim (WS4). The plan attacks those first, then hardens
everything else behind adversarial controls and an independent re-implementation.
