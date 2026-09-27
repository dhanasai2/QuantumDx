# Methodology Correction: Fair Classical-vs-QSVM Comparison

**Status:** Post-Stage-C correction phase — complete
**Scope:** Corrects the evaluation population mismatch identified after Stage C. Does not change training data, preprocessing, feature engineering, PCA, the quantum feature map/kernel, QSVM configuration, classical hyperparameter grids, or the decision threshold.
**Code:** `src/large_dataset/corrected_comparison.py`
**Results:** `results/large_dataset/corrected_comparison/{stage_1000,stage_5000,stage_10000}/`, `abc_summary.json`

---

## Problem identified

Stages A/B/C evaluated classical models on a fixed 2,000-row test set but evaluated QSVM on a fixed 200-row subset of that same 2,000-row set. The reported "classical-vs-QSVM gap" therefore compared metrics computed on two different populations.

## Why this matters

ROC-AUC (and every other threshold-dependent or ranking metric) is a property of the specific evaluation sample, not a fixed property of the model. A ROC-AUC computed over 200 observations is not the same quantity as one computed over 2,000, even when the smaller set is a strict subset of the larger one — different observations, different class balance realization, different noise. The previously reported Stage C gap of **0.0027** (QSVM 0.8013 vs. XGBoost 0.8040) was not a like-for-like comparison and should not have been read as evidence of a near-tie.

## Correction

Every model at every stage was **re-evaluated on the identical, pre-existing 200-row test set** (fingerprint `96eac11a8394b87e`, recovered deterministically via `seed=42` — never re-sampled). No persisted model artifacts existed from Stages A/B/C, so each classical model's training + CV hyperparameter search was reproduced exactly (same grids, same folds, same seed) on the same stage training subset, frozen, and only then scored on the 200 rows. The QSVM's training was likewise reproduced with its existing configuration (same feature map, qubits, kernel, C-grid) and scored on the same 200 rows. The 200-row set was never used for fitting, feature selection, PCA fitting, hyperparameter selection, model selection, or threshold selection — verified by `tests/test_corrected_comparison.py` (zero overlap with every training subset, deterministic recovery, fixed 0.5 threshold).

## What remains unchanged

Training data and stage subsets, test-set membership, preprocessing pipeline, feature groups, PCA (4 components), quantum feature map (`zz_feature_map`, reps=2, linear entanglement, 4 qubits), the fidelity kernel definition, classical hyperparameter grids, the QSVM C-grid `{0.01, 0.1, 1, 10, 100}`, and the fixed 0.5 decision threshold (Phase 3 methodology, not selected from test data).

---

## Corrected results

**Stage A (n=1,000):**

| Model | ROC-AUC | PR-AUC | Sens | Spec | Acc | Prec | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8453 | 0.8432 | 0.7475 | 0.8119 | 0.7800 | 0.7957 | 0.7708 |
| RBF-SVM | 0.8350 | 0.8101 | 0.7374 | 0.7822 | 0.7600 | 0.7684 | 0.7526 |
| Random Forest | 0.8357 | 0.8137 | 0.7576 | 0.8317 | 0.7950 | 0.8152 | 0.7853 |
| XGBoost | 0.8418 | 0.8027 | 0.7677 | 0.8020 | 0.7850 | 0.7917 | 0.7795 |
| **QSVM** | **0.7391** | 0.7192 | 0.7172 | 0.6832 | 0.7000 | 0.6893 | 0.7030 |

**Stage B (n=5,000):**

| Model | ROC-AUC | PR-AUC | Sens | Spec | Acc | Prec | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8503 | 0.8483 | 0.7273 | 0.7921 | 0.7600 | 0.7742 | 0.7500 |
| RBF-SVM | 0.8381 | 0.8300 | 0.7374 | 0.7822 | 0.7600 | 0.7684 | 0.7526 |
| Random Forest | 0.8383 | 0.8079 | 0.7374 | 0.8515 | 0.7950 | 0.8295 | 0.7807 |
| XGBoost | 0.8477 | 0.8085 | 0.7475 | 0.8218 | 0.7850 | 0.8043 | 0.7749 |
| **QSVM** | **0.7715** | 0.7364 | 0.7374 | 0.7426 | 0.7400 | 0.7374 | 0.7374 |

**Stage C (n=10,000):**

| Model | ROC-AUC | PR-AUC | Sens | Spec | Acc | Prec | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8504 | 0.8485 | 0.7273 | 0.8020 | 0.7650 | 0.7826 | 0.7539 |
| RBF-SVM | 0.8415 | 0.8301 | 0.7576 | 0.7921 | 0.7750 | 0.7812 | 0.7692 |
| Random Forest | 0.8471 | 0.8429 | 0.7475 | 0.8218 | 0.7850 | 0.8043 | 0.7749 |
| XGBoost | 0.8516 | 0.8294 | 0.7677 | 0.8218 | 0.7950 | 0.8085 | 0.7876 |
| **QSVM** | **0.8013** | 0.8002 | 0.7475 | 0.7525 | 0.7500 | 0.7475 | 0.7475 |

All rows in each table use the **identical 200 test observations**, threshold 0.5, same seed.

## Direct classical-vs-QSVM comparison (best classical model per stage)

| Stage | n | QSVM ROC-AUC | Best classical | Best ROC-AUC | Δ ROC-AUC | Bootstrap 95% CI | Boot. p | McNemar p |
|---|---:|---:|---|---:|---:|---|---:|---:|
| A | 1,000 | 0.7391 | Logistic Regression | 0.8453 | +0.1062 | [+0.055, +0.162] | 0.000 | 0.011 |
| B | 5,000 | 0.7715 | Logistic Regression | 0.8503 | +0.0788 | [+0.036, +0.126] | 0.001 | 0.557 |
| C | 10,000 | 0.8013 | XGBoost | 0.8516 | +0.0503 | [+0.012, +0.089] | 0.012 | 0.093 |

**At all three stages, every classical model's ROC-AUC bootstrap 95% CI vs. QSVM excludes zero** (paired bootstrap, 2000 resamples, seed 42) — full per-model results in each stage's `corrected_comparison.json`. The gap narrows from A to C but remains bootstrap-significant throughout on this metric.

**McNemar's test (paired 0.5-threshold label agreement) does not reach significance at Stage C** (p=0.093 for the best classical model) and is inconsistent across stages (significant at A, not at B or C for the specific best-classical-model comparison shown; see per-model results for the full picture). This is a genuine, reportable disagreement between the two tests, not an error: bootstrap ROC-AUC compares continuous ranking quality across the full probability range, while McNemar compares only the discrete decisions at one fixed threshold — a model can rank observations more accurately overall while agreeing with another model on most binary calls, especially at n=200 where McNemar has limited power to detect anything but large discordance.

**No correction for multiple comparisons (e.g., Bonferroni) was applied** across the 4 models × 2 tests × 3 stages = 24 comparisons reported here. This is a stated limitation, not an oversight.

## Previously reported (uncorrected) gaps — for direct contrast

| Stage | Previously reported gap (different test populations) | Corrected gap (identical 200 rows) |
|---|---:|---:|
| A | ~0.056 | **+0.106** |
| B | ~0.030 | **+0.079** |
| C | ~0.003 (misleadingly close) | **+0.050** |

The correction **increases** the reported gap at every stage relative to the previous, non-like-for-like figures. The Stage C "near-tie" was an artifact of comparing 2,000 classical observations against 200 QSVM observations, not a property of the models.

## What this correction establishes

A fairer, directly comparable evaluation of the existing Stage A/B/C model configurations on identical observations. At all three measured points, classical models show a bootstrap-significant ROC-AUC advantage over QSVM on this dataset, this feature representation, and this feature map. The gap's direction of change across A→B→C (narrowing) is consistent with — not proof of — QSVM improving faster than classical as training size increases; three points remain three points.

## What this correction does NOT establish

Quantum advantage or superiority (the evidence points the other way at every stage). Generalization to any cardiovascular population beyond this one Kaggle dataset. Clinical validity. Statistical significance under McNemar's test, which does not agree with the bootstrap ROC-AUC result. Anything about whether the gap would continue narrowing, plateau, or reverse beyond n=10,000 — that requires Stage D, which was explicitly out of scope for this correction.

## Files created

```
src/large_dataset/corrected_comparison.py
tests/test_corrected_comparison.py
docs/LARGE_DATASET_CORRECTION.md          (this file)
results/large_dataset/corrected_comparison/
    stage_1000/  {corrected_comparison.json, metrics.csv, predictions.csv,
                  roc_curves.png, pr_curves.png, confusion_matrices.png}
    stage_5000/  (same structure)
    stage_10000/ (same structure)
    abc_summary.json
```

No historical file was modified: `results/large_dataset/stage_comparison.json` (the original, uncorrected Stage A/B/C summary) and every Phase 3-5 result file are unchanged — verified by file timestamp and by `tests/test_corrected_comparison.py::test_phase5_historical_result_is_unchanged`.
