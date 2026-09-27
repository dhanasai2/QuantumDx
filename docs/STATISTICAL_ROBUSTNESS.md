# Phase 6A: Statistical Robustness Analysis

**Status:** Complete
**Scope:** Resolves an apparent disagreement between bootstrap and McNemar significance testing on the corrected classical-vs-QSVM comparison ([`LARGE_DATASET_CORRECTION.md`](LARGE_DATASET_CORRECTION.md)). Adds DeLong's test, applies Holm-Bonferroni correction across the declared family of primary comparisons, and produces bootstrap confidence intervals for all metrics at all stages. Does **not** touch the dataset, training data, Stage A/B/C result files, the Phase 3-5 methodology, or any model/hyperparameter/threshold decision.
**Code:** [`src/large_dataset/statistical_robustness.py`](../src/large_dataset/statistical_robustness.py)
**Tests:** [`tests/test_statistical_robustness.py`](../tests/test_statistical_robustness.py)
**Results:** `results/large_dataset/statistical_robustness/`

---

## 1. Objective

Stage A/B/C's corrected comparison (200-row identical test set, `docs/LARGE_DATASET_CORRECTION.md`) reported a paired bootstrap significant classical-over-QSVM ROC-AUC gap at every stage, alongside a McNemar's test that was significant at Stage A but not at Stage B or C for several models. This is a real disagreement between two different statistical tests, not an inconsistency in the data, but it was not yet explained or quantified rigorously. Phase 6A exists to:

1. Add DeLong's test — the standard, purpose-built paired test for comparing two ROC-AUCs on the same sample — as an independent check on the bootstrap result.
2. Apply one multiple-comparison correction across the full declared family of primary hypotheses (12 comparisons: 4 classical models × 3 stages), not per-stage.
3. Quantify uncertainty (95% CIs) for every reported metric, not just ROC-AUC.
4. Explain, explicitly, why DeLong/bootstrap and McNemar can legitimately disagree.
5. Verify the observed A→C gap-reduction trend numerically, without claiming it is a fitted or predictive trend.

No retraining occurred. All predictions used here were already persisted by the correction phase.

## 2. Data and test-set provenance

All comparisons in this document use the same 200-row held-out set established by the correction phase:

- Fingerprint (SHA-256 of sorted row ids): `96eac11a8394b87e`
- n = 200, positive rate = 0.495
- Verified identical (same ids, same `y_true`, in the same order) across all three stages' `predictions.csv` files — checked programmatically by `verify_cross_stage_consistency()` before any statistic was computed.
- Decision threshold fixed at 0.5 (unchanged from Phase 3).
- Zero overlap with any stage's training data (verified in the correction phase and unchanged here).

## 3. Experimental setup

For each of the 3 stages (A: n_train=1,000; B: n_train=5,000; C: n_train=10,000) and each of the 4 classical models (Logistic Regression, RBF-SVM, Random Forest, XGBoost), QSVM's predicted probabilities and labels on the 200-row set are compared against the classical model's predicted probabilities and labels on the **same 200 rows** — giving 4 × 3 = **12 primary paired comparisons**. Each comparison is evaluated three ways:

- **DeLong's test** on ROC-AUC (new in this phase)
- **Paired bootstrap** on ROC-AUC and PR-AUC (reused, unmodified, from the correction phase: `paired_bootstrap_delta`)
- **McNemar's exact test** on the 0.5-threshold labels (reused, unmodified: `mcnemar_exact`)

## 4. Why paired tests are required

Every model's predictions in this analysis come from the identical 200 observations — QSVM and each classical model score the same cases. An **independent-samples** AUC test (e.g., Hanley-McNeil) assumes the two AUCs come from unrelated samples and would overstate the variance of the difference, because it ignores the positive correlation induced by scoring the same cases twice. DeLong's method and the paired bootstrap both explicitly model this correlation: DeLong via the covariance between the two classifiers' structural components on shared cases; the bootstrap by resampling the **same** row indices for both classifiers on every resample. McNemar is inherently paired (it operates on a 2×2 table of the two classifiers' agreement per case). Using paired tests throughout is what makes DeLong, bootstrap, and McNemar comparable to each other on this dataset.

## 5. Bootstrap methodology

Unchanged from the correction phase: 2,000 resamples, seed 42, percentile method. For each resample, the same set of resampled row indices is applied to both classifiers' probability arrays before computing the delta metric (`paired_bootstrap_delta`), so the paired structure is preserved on every resample. The two-sided bootstrap p-value is computed as twice the smaller one-tailed proportion of resampled deltas on the opposite side of zero from the observed delta, capped at 1.0. This procedure was reused exactly, not re-derived, to keep this phase's bootstrap results traceable to the same code already tested in `tests/test_corrected_comparison.py`.

## 6. DeLong methodology

DeLong's test has no maintained implementation among the project's existing dependencies (scikit-learn does not ship one; no third-party DeLong package is installed). It is implemented directly from the standard structural-components method (DeLong, DeLong & Clarke-Pearson, 1988), in `delong_test()`:

1. For each classifier, split its predicted probabilities by true label into positive-case scores and negative-case scores.
2. Compute the structural components `V10[i] = mean_j psi(pos[i], neg[j])` (per positive case) and `V01[j] = mean_i psi(pos[i], neg[j])` (per negative case), where `psi(x,y) = 1` if `x>y`, `0.5` if `x==y`, `0` if `x<y`. `mean(V10) == mean(V01) == AUC` — asserted in code as a self-check on every call.
3. Stack the two classifiers' `V10` vectors and compute their 2×2 sample covariance (`ddof=1`); do the same for `V01`.
4. The variance of the AUC difference is `Var(ΔAUC) = S10[0,0]/n1 + S01[0,0]/n0 + S10[1,1]/n1 + S01[1,1]/n0 - 2·(S10[0,1]/n1 + S01[0,1]/n0)`.
5. A two-sided z-test and a normal-approximation 95% CI follow directly from `ΔAUC` and its standard error.

At n=200 the O(n1·n0) dense computation is exact and fast (no rank-based approximation needed). The implementation was validated against `sklearn.metrics.roc_auc_score` (marginal AUCs must match exactly) and exercised on synthetic data with known separation properties (zero delta / zero p for identical scores, antisymmetric delta and identical p when the two inputs are swapped, positive delta with a CI excluding zero for a clearly-better classifier) — see `tests/test_statistical_robustness.py`.

## 7. McNemar methodology

Reused, unmodified: an exact binomial test (`scipy.stats.binomtest`) on the discordant pairs of a 2×2 agreement table between two classifiers' 0.5-threshold predictions, conditional on the true label being correctly or incorrectly predicted by each. McNemar tests only whether the two classifiers **disagree asymmetrically** at the classification threshold — it does not use the predicted probabilities at all, and is blind to any ranking information beyond the single 0.5 cut.

## 8. Multiple-comparison correction

Twelve primary hypotheses are declared as **one family**: "classical model *m* has a different ROC-AUC than QSVM at stage *s*," for all 4 models × 3 stages. Per the task instruction, Holm-Bonferroni step-down correction is applied **once across all 12 p-values**, not separately within each stage's 3-comparison subset — correcting per-stage would understate the true multiplicity of 12 simultaneous tests. This was applied twice, independently: once to the 12 DeLong p-values (primary) and once to the 12 bootstrap p-values (secondary, as a cross-check), each as its own 12-comparison family.

## 9. Results

### 9.1 The 12 paired AUC comparisons

| Stage | n_train | Classical model | AUC (classical) | AUC (QSVM) | Δ (DeLong) | DeLong 95% CI | DeLong p | Holm p | Sig.(Holm) | Bootstrap p | McNemar p |
|---|---:|---|---:|---:|---:|---|---:|---:|:---:|---:|---:|
| A | 1,000 | Logistic Regression | 0.8453 | 0.7391 | +0.1062 | [0.0521, 0.1603] | 1.20e-04 | 1.44e-03 | Yes | 0.000 | 0.0113 |
| A | 1,000 | RBF-SVM | 0.8350 | 0.7391 | +0.0960 | [0.0423, 0.1496] | 4.59e-04 | 3.40e-03 | Yes | 0.000 | 0.0576 |
| A | 1,000 | Random Forest | 0.8357 | 0.7391 | +0.0966 | [0.0453, 0.1479] | 2.22e-04 | 1.99e-03 | Yes | 0.000 | 0.0019 |
| A | 1,000 | XGBoost | 0.8418 | 0.7391 | +0.1027 | [0.0500, 0.1555] | 1.36e-04 | 1.45e-03 | Yes | 0.000 | 0.0076 |
| B | 5,000 | Logistic Regression | 0.8503 | 0.7715 | +0.0788 | [0.0350, 0.1226] | 4.25e-04 | 3.40e-03 | Yes | 0.001 | 0.5572 |
| B | 5,000 | RBF-SVM | 0.8381 | 0.7715 | +0.0666 | [0.0267, 0.1065] | 1.07e-03 | 5.35e-03 | Yes | 0.000 | 0.5413 |
| B | 5,000 | Random Forest | 0.8383 | 0.7715 | +0.0668 | [0.0291, 0.1045] | 5.14e-04 | 3.40e-03 | Yes | 0.000 | 0.0347 |
| B | 5,000 | XGBoost | 0.8477 | 0.7715 | +0.0762 | [0.0371, 0.1153] | 1.32e-04 | 1.45e-03 | Yes | 0.000 | 0.0784 |
| C | 10,000 | Logistic Regression | 0.8504 | 0.8013 | +0.0491 | [0.0118, 0.0864] | 9.84e-03 | 3.94e-02 | Yes | 0.013 | 0.6900 |
| C | 10,000 | RBF-SVM | 0.8415 | 0.8013 | +0.0403 | [0.0044, 0.0761] | 2.78e-02 | 3.94e-02 | Yes | 0.022 | 0.4244 |
| C | 10,000 | Random Forest | 0.8471 | 0.8013 | +0.0458 | [0.0097, 0.0819] | 1.29e-02 | 3.94e-02 | Yes | 0.013 | 0.2478 |
| C | 10,000 | XGBoost | 0.8516 | 0.8013 | +0.0503 | [0.0115, 0.0891] | 1.10e-02 | 3.94e-02 | Yes | 0.012 | 0.0931 |

Full precision values: `results/large_dataset/statistical_robustness/auc_comparisons.csv`.

### 9.2 Bootstrap results

Bootstrap ROC-AUC deltas and CIs agree with DeLong to within bootstrap resolution at every one of the 12 comparisons (deltas match to 4+ decimal places; every bootstrap CI excludes zero exactly where every DeLong CI excludes zero). PR-AUC bootstrap deltas are also positive (classical > QSVM) at all 12 comparisons. Full results: `results/large_dataset/statistical_robustness/bootstrap_results.csv` (24 rows: 12 comparisons × {roc_auc, pr_auc}).

### 9.3 DeLong results

Reported inline in 9.1; the raw structural-components output (`auc_a`, `auc_b`, `var_delta`, `z`, `p_value` per comparison) is in `results/large_dataset/statistical_robustness/delong_results.csv`.

### 9.4 McNemar results

| Stage | Classical model | Both correct | Classical-only correct | QSVM-only correct | Both wrong | n discordant | p-value |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Logistic Regression | 130 | 26 | 10 | 34 | 36 | 0.0113 |
| A | RBF-SVM | 129 | 23 | 11 | 37 | 34 | 0.0576 |
| A | Random Forest | 132 | 27 | 8 | 33 | 35 | 0.0019 |
| A | XGBoost | 130 | 27 | 10 | 33 | 37 | 0.0076 |
| B | Logistic Regression | 137 | 15 | 11 | 37 | 26 | 0.5572 |
| B | RBF-SVM | 138 | 14 | 10 | 38 | 24 | 0.5413 |
| B | Random Forest | 142 | 17 | 6 | 35 | 23 | 0.0347 |
| B | XGBoost | 142 | 15 | 6 | 37 | 21 | 0.0784 |
| C | Logistic Regression | 139 | 14 | 11 | 36 | 25 | 0.6900 |
| C | RBF-SVM | 140 | 15 | 10 | 35 | 25 | 0.4244 |
| C | Random Forest | 140 | 17 | 10 | 33 | 27 | 0.2478 |
| C | XGBoost | 143 | 16 | 7 | 34 | 23 | 0.0931 |

Full results: `results/large_dataset/statistical_robustness/mcnemar_results.csv`.

### 9.5 Holm-corrected results

Applying Holm-Bonferroni once across the 12-comparison family:

- **DeLong family: 12 / 12 comparisons remain significant** at α=0.05 after correction (largest adjusted p = 0.0394, at Stage C).
- **Bootstrap family: 12 / 12 comparisons remain significant** at α=0.05 after correction (largest adjusted p = 0.048, at Stage C).

Both correction runs are recorded in full in `results/large_dataset/statistical_robustness/auc_comparisons.csv` (columns `delong_p_holm`, `delong_significant_holm`, `bootstrap_p_holm`, `bootstrap_significant_holm`).

### 9.6 Metric confidence intervals

Bootstrap 95% CIs (2,000 resamples, seed 42) were computed for all 7 metrics (ROC-AUC, PR-AUC, sensitivity, specificity, accuracy, precision, F1), all 5 models (4 classical + QSVM), all 3 stages — 15 rows, 25 columns, in `results/large_dataset/statistical_robustness/metric_confidence_intervals.csv`. ROC-AUC row is representative of the pattern seen across all 7 metrics:

| Stage | Model | ROC-AUC | 95% CI |
|---|---|---:|---|
| A | Logistic Regression | 0.8453 | [0.7897, 0.8952] |
| A | QSVM | 0.7391 | [0.6685, 0.8081] |
| B | Logistic Regression | 0.8503 | [0.7946, 0.8989] |
| B | QSVM | 0.7715 | [0.6991, 0.8361] |
| C | XGBoost | 0.8516 | [0.7960, 0.9033] |
| C | QSVM | 0.8013 | [0.7360, 0.8628] |

At every stage, QSVM's CI and the best classical model's CI are both wide (n=200) but the point estimates sit on either side with the gap comparable to the DeLong/bootstrap deltas above — consistent with, not independent evidence beyond, the paired tests in 9.1-9.2.

## 10. Bootstrap vs. McNemar interpretation

The apparent "disagreement" is genuine and expected — the two tests answer **different questions**:

- **DeLong / bootstrap ROC-AUC** measure whether a classifier ranks positive cases above negative cases more often than another classifier, using the full continuous probability output across all possible thresholds. This is a global, threshold-free measure of ranking quality.
- **McNemar** measures whether two classifiers **disagree** in their final binary decision at one fixed threshold (0.5) more often in one direction than the other. It discards all information beyond "which side of 0.5" each prediction falls on, and it only has power to detect an effect when the classifiers' discordant-pair counts are themselves imbalanced (in this data, "classical-only correct" vs. "QSVM-only correct").

A classical model can rank cases better overall across the whole probability range (driving a significant DeLong/bootstrap result) while agreeing with QSVM on **most** individual 0.5-threshold decisions (limiting McNemar's power) — especially as QSVM's overall separation improves from Stage A to Stage C, which mechanically increases decision-level agreement with the classical models even while a smaller but still real ranking gap persists. This is exactly the pattern observed: McNemar is significant for most models at Stage A (n_discordant ≈ 34-37, out of 200) but loses significance for most models by Stage B/C as the classifiers' binary decisions converge (n_discordant ≈ 21-27), even though DeLong and bootstrap remain significant throughout because the underlying continuous ranking gap, while narrowing, has not closed.

This is not a bug, a leakage problem, or evidence that one test is "wrong" — DeLong and McNemar are testing different null hypotheses on the same data, and both results are simultaneously correct answers to their respective questions.

## 11. Effect-size analysis

Using the best classical model at each stage (by `auc_classical`, per the correction phase's convention):

| Stage | n_train | Best classical model | ΔAUC (classical − QSVM) | Gap-reduction vs. Stage A |
|---|---:|---|---:|---:|
| A | 1,000 | Logistic Regression | 0.10621 | — |
| B | 5,000 | Logistic Regression | 0.07881 | 25.80% |
| C | 10,000 | XGBoost | 0.05031 | **52.64%** |

The claimed ≈52.6% A→C gap reduction is **verified**: `100 × (0.10621 − 0.05031) / 0.10621 = 52.6365%`, computed directly from the same DeLong deltas reported in 9.1 (`results/large_dataset/statistical_robustness/gap_analysis.csv`). The intermediate A→B (25.80%) and B→C (36.17%) reductions are also verified from the same source values. These are three measured points connected by percentage arithmetic — not a fitted curve, and not a claim about what happens at any n_train beyond 10,000.

## 12. Limitations

- **n=200 fixes the achievable precision of every test here.** All CIs (DeLong, bootstrap, and per-metric) are wide relative to the point estimates; none of this analysis can distinguish a "true" gap of, say, 0.03 from 0.08 with much confidence at any single stage.
- **Three stages is three points, not a trend line.** The 52.64% gap-reduction figure describes what was measured between three specific (n_train, gap) pairs; it is not a rate constant and must not be extrapolated to predict the gap at any larger n_train.
- **No independent replication.** Every number in this document derives from one fixed 200-row sample and one fixed train/test partition (seed 42). A different sample would give different point estimates, though the paired design controls for sample-to-sample variation between the compared classifiers on that same sample.
- **McNemar's reduced power at n=200 is a property of the test, not a defect being explained away** — it genuinely has less ability to detect small, consistent ranking advantages than DeLong/bootstrap when discordant pairs are few and roughly balanced.
- **This analysis does not evaluate generalization** beyond the Kaggle cardiovascular dataset, clinical validity, or performance on hardware.

## 13. Final scientific conclusion

At all three measured stages (n_train = 1,000 / 5,000 / 10,000), every one of the 4 classical baselines shows a **statistically significant ROC-AUC advantage over QSVM** on this identical 200-row test set, confirmed independently by DeLong's test and the paired bootstrap (12/12 comparisons significant after Holm-Bonferroni correction across the full family, by both methods). This gap **narrows** across the three measured points (≈0.106 → 0.079 → 0.050, a 52.6% reduction from Stage A to Stage C), a direction consistent with — but not proof of — QSVM improving faster than the classical baselines as training size grows; it remains three observed points, not an established trend. McNemar's test, which answers the different question of discrete 0.5-threshold decision agreement, is not significant for most models by Stage B/C — this is an expected, explainable consequence of decision-level agreement converging while a real ranking-level gap persists, not a contradiction of the DeLong/bootstrap finding. No evidence of quantum advantage exists in this analysis; the evidence at every stage points the other way.

Answers to the seven required interrogation questions:

1. **Does the classical-vs-QSVM gap remain statistically significant after Holm-Bonferroni correction across all 12 comparisons?** Yes — 12/12 by DeLong, 12/12 by bootstrap.
2. **Is the result consistent across all four classical baselines, or driven by one model?** Consistent — all four classical models beat QSVM significantly at all three stages; the best-performing classical model changes from Logistic Regression (A, B) to XGBoost (C) by a small margin, but no classical model loses to QSVM at any stage.
3. **Do DeLong and bootstrap agree with each other?** Yes, closely — deltas and CI-exclusion-of-zero decisions match at all 12 comparisons.
4. **What is the pattern of McNemar's disagreement with DeLong/bootstrap?** McNemar is significant (matches DeLong's conclusion) for 3/4 models at Stage A, but drops to non-significant for most models at Stage B and Stage C, even as DeLong/bootstrap stay significant throughout.
5. **Is this disagreement expected and explainable?** Yes — see Section 10: they test different hypotheses (continuous ranking vs. discrete threshold agreement), and decision-level agreement mechanically increases as QSVM's overall ranking quality improves, independent of whether the ranking gap has closed.
6. **Does n=200 materially limit precision?** Yes — all CIs are wide; this is a stated limitation (Section 12), not something the analysis can correct without a larger held-out set.
7. **Is there evidence of leakage or implementation problems?** No — fingerprint, id, and label identity were verified programmatically across all three stages before any statistic was computed; DeLong's marginal AUCs were verified against `sklearn.roc_auc_score`; all four statistical methods were validated against synthetic data with known ground-truth properties (see `tests/test_statistical_robustness.py`).

## 14. Recommendation for the next phase

The evidence at all three measured stages is consistent and statistically robust: classical models currently outperform QSVM on this dataset and feature representation, with a narrowing but still real and significant gap. Of the permissible next steps, the evidence supports **proceeding to Stage D (n_train=20,000 or larger, pending resource/time budget) specifically to determine whether the observed narrowing trend continues, plateaus, or reverses** — this is squarely a "measure, don't extrapolate" question that three points cannot answer. It does not support claiming quantum advantage, changing the QSVM configuration to chase a more favorable result, or halting large-dataset work, since the trend itself (not just the current gap) is the open empirical question. This recommendation is evidence-based, not outcome-motivated: if Stage D shows the gap stabilizing or widening again, that is an equally valid and reportable result.

**Per the governing instruction for this phase: STOP here.** No Stage D run, VQC work, hardware validation, API, dashboard, or deployment work should begin without explicit direction to proceed.

## Files created

```
src/large_dataset/statistical_robustness.py
tests/test_statistical_robustness.py
docs/STATISTICAL_ROBUSTNESS.md                      (this file)
results/large_dataset/statistical_robustness/
    statistical_summary.json
    auc_comparisons.csv
    delong_results.csv
    bootstrap_results.csv
    mcnemar_results.csv
    metric_confidence_intervals.csv
    gap_analysis.csv
    forest_plot_delta_auc.png
    auc_across_stages.png
```

No historical or Stage A/B/C corrected-comparison file was modified. All predictions used were already persisted by the correction phase — no retraining occurred in this phase.
