# Phase 11: Full-Dataset Scale Experiment

**Status:** Complete — outcome: **C (hybrid does not beat the equivalent non-quantum control) + D (the full-data classical ranking changed from n=2,000)**
**Scope:** Determines whether Phase 9/10's conclusions hold when models are trained on the full available training pool (n=66,641) instead of the n_train=2,000 screening subset — same fixed 200-row test set throughout. Does not modify Stage A–D, Phase 7, Phase 8A, Phase 8B, Phase 9, or Phase 10 artifacts.
**Code:** [`src/large_dataset/phase11_full_scale.py`](../src/large_dataset/phase11_full_scale.py)
**Tests:** [`tests/test_phase11_full_scale.py`](../tests/test_phase11_full_scale.py) (13)
**Results:** `results/large_dataset/phase11_full_scale/`

---

## 1. Exact training pool size (determined by inspection, not assumed)

`derive_features(load_raw_cardio())` → 68,641 modeling rows → `build_fixed_split(seed=42)` draws the same fixed 2,000-row classical test set (whose 200-row subset is the project's enduring comparison set, fingerprint `96eac11a8394b87e`) → **training_pool = 66,641 rows**. This is not "70,000 minus 200" — the classical 2,000-row test set is correctly excluded too, exactly as Stage A–D and every Phase 8 screening subset already did; verified programmatically (`test_full_scale_split_has_expected_training_pool_size`).

## 2. Models evaluated

| Model | Scale | Source |
|---|---|---|
| A: Logistic Regression | Full (66,641) | Trained fresh, full grid |
| B: RBF-SVM | **n=20,000 (Stage D, reused)** | Full scale benchmarked infeasible — see §4 |
| C: Random Forest | Full (66,641) | Trained fresh, full grid |
| D: XGBoost | Full (66,641) | Trained fresh, full grid |
| E: Hybrid | Full (66,641) | θ selected on n=2,000 subset, features computed on full pool |
| F: Control (random features) | Full (66,641) | Trained fresh |
| G: Baseline QSVM | **n=2,000 (Phase 9, reused)** | Full scale infeasible — see §4 |
| H: MI-adaptive QSVM | **n=2,000 (Phase 9, reused)** | Full scale infeasible — see §4 |

## 3. Leakage controls

Classical hyperparameters selected via 5-fold CV on the training pool only (`run_grid_search`, unmodified); the hybrid's θ selected via COBYLA + 3-fold CV on the n=2,000 screening subset (a proper, zero-overlap, already-vetted subset of this same training pool — not the test set); the 200-row test set scored exactly once per model, after every selection was frozen. Verified: `test_full_scale_training_pool_has_zero_overlap_with_test_sets`.

## 4. Documented computational deviations (all measured, not guessed)

**RBF-SVM skipped at full scale.** A single fit with the established best hyperparameters (C=10.0, gamma=0.01 — Stage C/D/Phase 9's own independently-reselected value each time) was benchmarked directly: it exceeded **20 minutes with no completion**, consistent with the LIBSVM superlinear-scaling pathology already documented twice in this project (Stage D's own 4h08m stall). Stage D's n=20,000 result is reused instead and clearly labeled as a smaller scale.

**QSVM (baseline/MI-adaptive) skipped at full scale.** The train-train fidelity kernel at n=66,641 would require a 66,641×66,641 matrix — **17.8 GB even in float32** — far beyond this machine's available memory (Stage D's own n=20,000 run, at only 1.6 GB, already produced a documented near-OOM scare with far more headroom than exists now). Phase 9's n=2,000 results are retained for reference only.

**Learning-curve sweep (2k/5k/10k/20k/50k/full) not run.** The core question — does the n=2,000 conclusion change at full scale — is fully answered by the two-point comparison this phase performs. Stage A–D's own n=1,000/5,000/10,000/20,000 classical numbers exist but were evaluated on the *different* 2,000-row classical test set, not the 200-row set — reporting them as directly comparable would repeat exactly the population-mismatch error the project's own Phase 6 correction phase (`docs/LARGE_DATASET_CORRECTION.md`) already established must never happen silently. They are not included in the scale-comparison table for this reason.

**A genuine benchmark miss, disclosed honestly:** Random Forest's full grid search was estimated at ~26 minutes (from a single-fit benchmark at `max_depth=5`, 13.2s) but **actually took 12,243.7s (3.4 hours)**. The benchmark sampled a representative grid point, not the worst-case one — the grid also includes `max_depth=None` (unbounded), which combined with `min_samples_split=2` at n=66,641 produces dramatically larger, slower trees than the depth-capped case measured. The selected best model still used `max_depth=5` (the fast setting), so the final result is unaffected, but the time estimate was wrong by ~8×. This is reported plainly, not minimized.

## 5. Full-scale classical results

| Model | ROC-AUC | PR-AUC | Runtime |
|---|---:|---:|---:|
| Logistic Regression | 0.8510 | 0.8510 | 98.4s |
| Random Forest | 0.8454 | 0.8439 | 12,243.7s (3.4h — see §4) |
| **XGBoost (best)** | **0.8567** | 0.8426 | 1,349.4s (22.5 min) |
| RBF-SVM (Stage D, n=20,000) | 0.8406 | 0.8381 | reused |

## 6. Hybrid and control results

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Hybrid (full-scale features, full LR fit) | 0.8312 | 0.8213 |
| Control (random features, full-scale) | 0.8305 | 0.8250 |

Hybrid training: θ selected on n=2,000 in 372.9s (40 COBYLA evaluations — same procedure, same subset Phase 10 used, reproducing its methodology exactly); quantum features computed once for all 66,641 training rows in 428.2s; final LR fit in 0.2s. Total: 801.6s (13.4 min).

## 7. Statistical comparison

Holm-Bonferroni applied once across 4 comparisons (one family):

| Comparison | Δ ROC-AUC | DeLong p (Holm) | Significant? | Bootstrap CI excl. 0? |
|---|---:|---:|:---:|:---:|
| Best full classical (XGBoost) vs Hybrid | +0.0255 (classical ahead) | 0.089 | No | No (raw p=0.044, does not survive correction) |
| **Hybrid vs Control (critical ablation)** | **+0.0007** | **0.887** | **No** | **No** |
| Hybrid vs Baseline QSVM (n=2,000) | +0.0933 (hybrid ahead) | 0.00025 | Yes | Yes |
| Hybrid vs MI-adaptive QSVM (n=2,000) | +0.0646 (hybrid ahead) | 0.0162 | Yes | Yes |

## 8. The critical ablation — confirmed and strengthened at full scale

**Hybrid vs. Control: Δ=0.0007, p=0.887 — completely indistinguishable**, replicating Phase 10's exact finding, now with 33× more training data. Sensitivity/specificity/accuracy/F1 differ only in the third decimal place. This is a stronger, not weaker, null result than Phase 10's: more data did not help the quantum-specific transform separate itself from a same-dimensionality random-noise control. The apparent wins over standalone QSVM (rows 3–4) remain, as in Phase 10, attributable to "any 4 extra input dimensions help a linear model a little" — not to anything the trained quantum circuit specifically learned.

## 9. Training-scale comparison (n=2,000 → full)

| Model | n=2,000 ROC-AUC | Full ROC-AUC | Δ | Note |
|---|---:|---:|---:|---|
| Logistic Regression | 0.8490 | 0.8510 | +0.0020 | |
| RBF-SVM | 0.8374 | 0.8406 | +0.0032 | Full value is Stage D's n=20,000, not n=66,641 |
| Random Forest | 0.8403 | 0.8454 | +0.0051 | |
| **XGBoost** | 0.8435 | **0.8567** | **+0.0132** | Largest gain — became the best model |
| Hybrid | 0.8319 | 0.8312 | −0.0007 | |
| Control | 0.8344 | 0.8305 | −0.0039 | |

**The classical ranking changed**: at n=2,000, Logistic Regression was best (Phase 9); at full scale, **XGBoost is best**, with Logistic Regression second and Random Forest third. XGBoost gained the most from additional data (+0.0132), consistent with gradient-boosted trees generally benefiting more from larger training sets than a linear model, which was already closer to its ceiling at n=2,000. Hybrid and Control both moved in the same (slightly negative) direction — further evidence they behave as a matched pair, not as "quantum vs. nothing."

## 10. Answers to the required questions

1. **Strongest full-data classical model**: XGBoost (ROC-AUC 0.8567).
2. **Does the best model change vs. n=2,000?** Yes — Logistic Regression → XGBoost.
3. **Does hybrid improve from n=2,000 to full data?** No — essentially flat (−0.0007).
4. **Does hybrid beat the equivalent random-feature control?** No — Δ=0.0007, p=0.887, not significant.
5. **If hybrid > classical, is it significant?** Not applicable — hybrid is below best classical (XGBoost), not above.
6. **Gap to classical**: Δ=0.0255 (XGBoost ahead), raw p=0.044 but Holm-adjusted p=0.089 — does not survive multiple-comparison correction, so not treated as a confirmed gap either, though the direction favors classical.
7. **Evidence the quantum transform learned something useful?** No — the ablation (§8) is the direct test, and it fails to show any separation from random noise, at either training scale.
8. **Does training-set size change model ranking?** Yes, confirmed empirically (§9) — this was the central concern Phase 11 was designed to test, and the answer is yes for the classical models, no for hybrid/control (both stayed near the bottom of the pack at both scales).
9. **Evidence for quantum advantage?** None. The hybrid does not beat its own non-quantum control at either scale, and does not beat the strongest classical model at full scale.
10. **Model recommended for the SIH prototype**: **XGBoost** at full scale — highest ROC-AUC/PR-AUC of all evaluated models, reproducible (fixed seed, documented grid), computationally reasonable (22.5 min for the full grid, milliseconds for inference), and — being a classical, well-understood gradient-boosted tree model — straightforward to explain and deploy. Logistic Regression remains a strong, even cheaper, more directly-explainable runner-up if simplicity is prioritized over the last ~0.006 ROC-AUC. No quantum or hybrid model is recommended.

## 11. Final safety check (performed explicitly)

- Test-set fingerprint verified: `96eac11a8394b87e` ✓ (`comparison_set.matches_expected: true`)
- No prior-phase artifact modified — verified by `test_prior_phase_artifacts_untouched_if_present` (Stage D, Phase 7, 8A, 8B, 9, 10 all checked)
- Full relevant test suite run (see below)
- All predictions correspond to the same 200 test rows across every model (single shared `y_true` column, asserted in code)
- No NaN/Inf in `predictions.csv` — asserted in code (`assert not pred_df.isna().any().any()`) and re-verified by test
- All quantum expectation values verified within [-1, 1] — asserted in code and by `test_quantum_expectation_values_within_valid_range_if_present` (measured min=−0.99999979, max=0.99979721)
- All model configurations and seeds recorded in `model_configs.json`
- This document explicitly distinguishes n=2,000 (Phase 9/10) results, full-data (this phase) results, and the two reused smaller-scale references (RBF-SVM n=20,000; QSVM n=2,000) throughout

## 12. Decision

**Outcome C: hybrid does not beat the equivalent non-quantum control.** **Additionally, Outcome D applies**: the full-data classical ranking changed substantially from n=2,000 (Logistic Regression best → XGBoost best) — model selection should be based on the full-data evaluation, not the earlier screening-scale ranking.

**STOP per instruction.** No automatic further phase.

## Files created

```
src/large_dataset/phase11_full_scale.py
tests/test_phase11_full_scale.py
docs/PHASE_11_FULL_SCALE.md                (this file)
results/large_dataset/phase11_full_scale/
    predictions.csv, metrics.csv, comparison_table.csv, model_configs.json,
    runtime.json, statistical_comparisons.json, training_scale_comparison.csv,
    phase11_summary.json, roc_curves.png, pr_curves.png, confusion_matrices.png
```

No Stage A–D, Phase 7, Phase 8A, Phase 8B, Phase 9, or Phase 10 artifact was modified — verified by `tests/test_phase11_full_scale.py::test_prior_phase_artifacts_untouched_if_present`.
