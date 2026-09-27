# Phase 9: Classical Ceiling Benchmark & Fair Quantum-vs-Classical Comparison

**Status:** Complete — outcome: **C (classical model remains clearly stronger, statistically supported)**
**Scope:** Evidence-gathering, not metric-maximization. Establishes how strong classical models are on this dataset/split, whether the QSVM variants are competitive, and which model should become the SIH prototype's candidate. Does not modify Stage A–D, Phase 7, Phase 8A, or Phase 8B artifacts.
**Code:** [`src/large_dataset/phase9_classical_benchmark.py`](../src/large_dataset/phase9_classical_benchmark.py)
**Tests:** [`tests/test_phase9_classical_benchmark.py`](../tests/test_phase9_classical_benchmark.py) (9)
**Results:** `results/large_dataset/phase9_classical_benchmark/`

---

## 1. Method

Reused, unmodified: the identical n_train=2,000/seed=42 screening split and fixed 200-row test set (fingerprint `96eac11a8394b87e`) every Phase 8 experiment used (`build_screening_split()`); the existing 4-model classical registry (Logistic Regression, RBF-SVM, Random Forest, XGBoost) with its full, unreduced hyperparameter grids (`get_available_models`, `run_grid_search` — Phase 3's per-fold-refit CV methodology, unchanged); and Phase 8A's already-persisted baseline/MI-adaptive/label-aware QSVM predictions (`results/large_dataset/phase8a_label_aware/predictions.csv`) — **no quantum model was retrained**. Every classical model was scored on the SAME 200-row test set the quantum models used (not the usual 2,000-row classical test set), so all 7 models in this report share identical held-out observations.

## 2. Computational budget

Estimated and dry-run tested before the real run (n=150 smoke test: ~103s total). Real run at n=2,000: **415s (~7 min) total** for all 4 classical models combined — well within a controlled budget, no uncontrolled sweep.

## 3. Comparison table

| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity | Accuracy | F1 | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.8490** | 0.8469 | 0.7475 | 0.8119 | 0.78 | 0.7708 | 5.6 |
| RBF-SVM | 0.8374 | 0.8246 | 0.7475 | 0.7921 | 0.77 | 0.7629 | 73.0 |
| Random Forest | 0.8403 | 0.8173 | 0.7172 | 0.8416 | 0.78 | 0.7634 | 323.2 |
| XGBoost | 0.8435 | 0.8070 | 0.7778 | 0.8218 | 0.80 | 0.7938 | 36.0 |
| Baseline QSVM | 0.7379 | 0.6920 | 0.7172 | 0.6931 | 0.705 | 0.7065 | reused from Phase 8A |
| MI-adaptive QSVM | 0.7666 | 0.7015 | 0.7374 | 0.7228 | 0.73 | 0.7300 | reused from Phase 8A |
| Label-aware QSVM | 0.7475 | 0.7297 | 0.6667 | 0.6931 | 0.68 | 0.6735 | reused from Phase 8A |

## 4. Best classical model

**Logistic Regression**, ROC-AUC=0.8490, PR-AUC=0.8469 — notably, the simplest classical model here outperforms RBF-SVM, Random Forest, and XGBoost on this dataset/split at n=2,000, and trains in 5.6s. All four classical models cluster tightly (0.8374–0.8490), well above every quantum variant.

## 5. Quantum comparison

Best quantum candidate among baseline/MI-adaptive: **MI-adaptive QSVM** (0.7666). All three paired comparisons (Holm-Bonferroni applied once across this 3-comparison family):

| Comparison | Δ ROC-AUC | DeLong p (raw) | DeLong p (Holm) | Significant? | Bootstrap CI excludes 0? | McNemar p |
|---|---:|---:|---:|:---:|:---:|---:|
| Best classical (LR) vs Baseline QSVM | +0.1111 | 1.03e-05 | 3.09e-05 | **Yes** | Yes | 0.0081 |
| Best classical (LR) vs MI-adaptive QSVM | +0.0824 | 0.00101 | 0.00201 | **Yes** | Yes | 0.0987 |
| MI-adaptive vs Baseline QSVM | +0.0287 | 0.235 | 0.235 | No | No | 0.424 |

The third row reproduces exactly the non-significant MI-adaptive-vs-baseline result already established in Phase 8A/7 — an internal consistency check confirming this phase's statistical pipeline agrees with the prior ones.

## 6. Decision (Step 8 framework)

**Outcome C: classical model remains clearly stronger, statistically supported.** Best classical (Logistic Regression) beats best quantum candidate (MI-adaptive QSVM) by Δ=+0.0824 ROC-AUC, Holm-adjusted DeLong p=0.0020, bootstrap CI [0.033, 0.135] excludes zero. This is not close, not borderline, and not a single-test artifact — DeLong and bootstrap agree closely, and even McNemar (a lower-power test at n=200) is close to significance (p=0.099).

## 7. Which model should be taken forward

**Logistic Regression** is the clear candidate for the SIH prototype at this stage: highest ROC-AUC/PR-AUC of all 7 models, by far the cheapest to train (5.6s vs. 36–323s for the other classical models, and vs. the multi-minute-to-hour costs of every quantum variant measured across Phases 6–8), and the simplest to explain/deploy/maintain. Nothing in this evidence supports choosing a QSVM variant over it for a production-facing prototype today.

## 8. Scientifically justified next experiment

None of Phase 9's evidence supports investing further in the currently-tested quantum directions:
- **MI-adaptive QSVM** already failed a 5-seed reproducibility check (Phase 7) and now also loses clearly to classical (this phase) — no further quantum-kernel tuning of this specific mechanism is justified.
- **Label-aware QSVM** already failed screening (Phase 8A) — not scaled, consistent with that result.
- **VQC** already lost clearly and significantly to baseline QSVM itself (Phase 8B), which is now also clearly beaten by classical — a fortiori, this VQC configuration is not competitive with classical either.

If a credible quantum research direction remains, it is not "tune the existing mechanisms harder" — it is a materially different approach (e.g., a genuinely different, better-justified feature-map family, or a hybrid architecture) that would need its own separate, explicit justification before implementation. This document does not recommend any of the potential next directions listed in the task brief (kernel alignment, multiple kernels, noise robustness, etc.) be started automatically — the evidence gathered here is a reason to pause and reconsider the overall quantum direction, not to pick the next tuning knob.

## 9. What should explicitly NOT be pursued

- Scaling the MI-adaptive or label-aware QSVM hypotheses further (both already falsified at earlier stages).
- Continuing or extending the tested VQC architecture.
- Building a separate QNN implementation on the assumption VQC "just needed more training" (Phase 8B's own limitation notwithstanding — that would be a new, separately-justified experiment).
- Any hardware, API, dashboard, or deployment work built around a quantum model, given classical models are currently both stronger and far cheaper.
- Declaring "quantum advantage" from any result in Phases 6–9 — none is statistically supported, and this phase's evidence points the other way.

## Files created

```
src/large_dataset/phase9_classical_benchmark.py
tests/test_phase9_classical_benchmark.py
docs/PHASE_9_CLASSICAL_BENCHMARK.md                (this file)
results/large_dataset/phase9_classical_benchmark/
    predictions.csv, metrics.csv, comparison_table.csv, model_configs.json,
    runtime.json, statistics.json, phase9_summary.json,
    roc_curves.png, pr_curves.png, confusion_matrices.png
```

No Stage A–D, Phase 7, Phase 8A, or Phase 8B artifact was modified — verified by `tests/test_phase9_classical_benchmark.py::test_prior_phase_artifacts_untouched_if_present`.
