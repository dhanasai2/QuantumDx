# Adaptive Feature-Map QSVM Experiment

**Status:** Complete — outcome: **B (improvement observed, evidence inconclusive)**
**Scope:** Tests whether a training-data-informed entangling structure improves the QSVM's fidelity kernel over the fixed `zz_feature_map("linear")` baseline, on the IDENTICAL Stage D training subset (n=20,000) and 200-row test set (fingerprint `96eac11a8394b87e`). Does not change the dataset, PCA, qubit count, kernel definition, SVM formulation, or evaluation protocol.
**Code:** [`src/quantum/adaptive_feature_map.py`](../src/quantum/adaptive_feature_map.py), [`src/large_dataset/adaptive_qsvm_experiment.py`](../src/large_dataset/adaptive_qsvm_experiment.py)
**Tests:** [`tests/test_adaptive_feature_map.py`](../tests/test_adaptive_feature_map.py) (12), [`tests/test_adaptive_qsvm_experiment.py`](../tests/test_adaptive_qsvm_experiment.py) (7)
**Results:** `results/large_dataset/adaptive_qsvm/`

---

## 1. Motivation

Stage D found the fixed-kernel QSVM plateauing against classical baselines at n=20,000 after narrowing the gap from A to C (`docs/STAGE_D.md`). One candidate explanation the project had not yet tested: the baseline's entangling structure (a fixed "linear chain" — qubit 0↔1↔2↔3, independent of the data) may not reflect which of the 4 PCA-derived features actually have meaningful pairwise dependence. If the feature map's structure is generic rather than informed by the data's own relationships, the fidelity kernel may be leaving representational capacity on the table.

## 2. Hypothesis

"Can a data-informed/adaptive quantum feature map improve the QSVM's representational ability compared with the existing fixed ZZFeatureMap, while preserving the same preprocessing, evaluation protocol, and leakage controls?" Tested via one principled, deterministic adaptive mechanism — not an architecture search.

## 3. Baseline architecture

Unchanged from Stage D: 4 qubits, `zz_feature_map`, reps=2, entanglement="linear" (ZZ terms on adjacent pairs (0,1),(1,2),(2,3)), PCA-4, fidelity kernel, `SVC(kernel="precomputed")`, C selected by 5-fold CV over the same C-grid. **The baseline QSVM was not retrained for this experiment** — its predictions on the 200-row test set are reused directly from `results/large_dataset/stage_d/predictions.csv` (verified against the same test fingerprint before use), since retraining an identical model would only reproduce Stage D's own already-persisted result.

## 4. Adaptive feature-map design

Structurally identical to the baseline in every respect except **which** 3 qubit pairs receive the ZZ interaction term: built via `pauli_feature_map(paulis=("Z","ZZ"))` (verified structurally identical to `zz_feature_map` for the "linear" case — same gate counts: 22 single-qubit `u` gates, 12 `cx` gates, for reps=2) with a **custom entanglement dict** replacing the fixed linear-chain pair list with pairs selected by mutual information.

**Term budget is deliberately unchanged**: 4 Z terms (one per qubit) + exactly 3 ZZ terms (matching the linear chain's edge count), same `reps=2`. This isolates *which pairs* are entangled from *how many* — a change in pair count would confound structure with raw expressivity/depth.

## 5. Why the adaptation is data-informed

Pairwise **mutual information** (not Pearson correlation) between the 4 PCA-derived, range-normalized training features, estimated via scikit-learn's k-NN based `mutual_info_regression` (`random_state=42`), symmetrized by averaging both estimation directions. **Correlation was deliberately rejected**: the 4 features are PCA components fit on this same training data, and PCA components are linearly uncorrelated by construction over their own fitting data — a correlation-based rule would be near-degenerate here. Mutual information captures general (including nonlinear) statistical dependence and is not zeroed out by PCA's linear decorrelation, making it a non-trivial, principled signal for "how much would an explicit two-qubit interaction between features i and j capture that independent single-qubit encoding would miss."

The top-3 pairs by mutual information (ties broken deterministically by qubit index) become the adaptive map's ZZ pairs.

## 6. How training-only information is used

Every function in `adaptive_feature_map.py` takes only a training feature array — none accepts a test array, a label array, or a performance metric (verified structurally in `tests/test_adaptive_feature_map.py::test_no_function_in_module_accepts_test_data`, by inspecting each function's signature for forbidden parameter names). The mutual-information matrix and selected pairs are therefore incapable of depending on the 200-row test set, not merely disciplined not to. The map was derived once, from the Stage D training subset's PCA output, before the 200-row set was touched at all.

## 7. Exact implementation

`build_adaptive_feature_map_from_training_data(X_train_quantum, n_qubits=4, reps=2, random_state=42)`:
1. `compute_pairwise_mutual_information` → symmetric 4×4 MI matrix.
2. `select_adaptive_entanglement_pairs` → top-3 pairs, deterministic tie-break.
3. `build_adaptive_feature_map` → `pauli_feature_map` with `entanglement={1: [[0],[1],[2],[3]], 2: [selected pairs]}`, `paulis=("Z","ZZ")`.

Measured mutual-information matrix (this run, n_train=20,000):

| | q0 | q1 | q2 | q3 |
|---|---:|---:|---:|---:|
| q0 | — | 0.2108 | 0.0719 | 0.0926 |
| q1 | 0.2108 | — | **0.7811** | 0.0463 |
| q2 | 0.0719 | **0.7811** | — | 0.0432 |
| q3 | 0.0926 | 0.0463 | 0.0432 | — |

**Selected pairs: (1,2), (0,1), (0,3)** — replacing the baseline linear chain's (2,3) edge with (0,3). The (1,2) pair's mutual information (0.7811) is dramatically larger than any other pair, a genuine, non-trivial structural signal (not noise-level differences among near-equal candidates).

## 8. Experimental protocol

Baseline: reused from Stage D (no retraining). Adaptive: preprocessing/PCA refit on the exact Stage D n=20,000 training subset (deterministically reproduces Stage D's own X_train_quantum/X_test_quantum), adaptive map derived from that training array only, statevectors computed, memory-safe blockwise float32 kernel assembled (same method Stage D used), C selected by the identical 5-fold CV C-grid search Stage D used — **no hyperparameter beyond C was tuned**; pair count (3) was fixed by design (§4), not swept, to keep the comparison to one isolated variable. The 200-row test set was scored exactly once, after C selection, for both models.

## 9. Results

| Model | ROC-AUC | 95% CI | PR-AUC | Sens | Spec | Acc | F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| Baseline QSVM | 0.7995 | [0.7327, 0.8603] | 0.8000 | 0.7576 | 0.7624 | 0.760 | 0.7576 |
| **Adaptive QSVM** | **0.8193** | [0.7580, 0.8777] | 0.8239 | 0.7273 | 0.8020 | 0.765 | 0.7539 |

Adaptive QSVM improved ROC-AUC by **+0.0198** and PR-AUC by **+0.0239**, driven mainly by higher specificity (0.802 vs 0.762) at a small sensitivity cost (0.727 vs 0.758) — 4 fewer false positives, 3 more false negatives (confusion matrices: `results/large_dataset/adaptive_qsvm/confusion_matrices.png`).

## 10. Statistical comparison (Adaptive − Baseline, paired, same 200 rows)

| Test | Δ | 95% CI | p-value | Significant? |
|---|---:|---|---:|:---:|
| DeLong (ROC-AUC) | +0.0198 | [−0.0086, +0.0482] | 0.172 | **No** |
| Paired bootstrap (ROC-AUC) | +0.0198 | [−0.0073, +0.0492] | 0.185 | **No** |
| Paired bootstrap (PR-AUC) | +0.0239 | [−0.0223, +0.0731] | 0.359 | **No** |
| McNemar (0.5-threshold labels) | — | — | 1.000 | **No** (n_discordant=15, 8 vs 7) |

**Every CI includes zero. No test reaches significance.** The point-estimate improvement is real and consistent in direction across ROC-AUC and PR-AUC, but at n=200 the paired evidence cannot distinguish it from noise. This is reported as-is — the sign is positive, but "improvement" is not statistically established.

## 11. Computational cost

Adaptive-map derivation (mutual information + pair selection): 1.4s — negligible. Statevectors: 20.0s (train) + 0.2s (test). Kernel assembly: 6.8s (train-train) + 0.08s (test-train), memory-safe blockwise float32, 1,600MB, RSS peaked at 1,849MB (comparable to Stage D's own QSVM phase, no new memory risk introduced). CV C-grid search: 1,919.5s (the dominant cost, same grid/folds as Stage D). Total: ~32.4 minutes — comparable to Stage D's own QSVM phase (~30.1 min), as expected since the circuit's gate budget is identical to baseline.

## 12. Limitations

- n=200 bounds statistical power severely; a +0.02 ROC-AUC point-estimate difference is well within this test set's noise floor for all three paired tests.
- Pair count (3) was fixed, not swept — the experiment does not know whether a different number of adaptive pairs would perform better or worse; that would be a different, larger experiment.
- Mutual information was estimated with a single k-NN estimator configuration (scikit-learn defaults, `random_state=42`); a different MI estimator could plausibly select different pairs, untested here.
- Single run at n=20,000, no repeated-seed replication (mirrors Stage D's own stated limitation).
- Even taking the point estimate at face value, the gap to the best classical baseline (XGBoost, 0.8613) would only narrow from 0.0618 to ~0.0420 (~32%, descriptive only, not statistically established) — a meaningfully smaller gap than baseline QSVM's, but still a substantial, unclosed gap.

## 13. Interpretation

The adaptive mechanism identified one genuinely strong, non-trivial pairwise relationship (q1↔q2, MI=0.78, far above the other 5 pairs) and used it in place of the baseline's arbitrary linear-chain edge. The resulting QSVM scored higher on both primary metrics on this test set, with a plausible mechanistic story (specificity gain at the entangled pair's expense of some sensitivity). But this is a single point estimate on 200 observations with all three paired significance tests returning null results — the evidence supports "worth investigating further," not "established improvement."

## 14. Was the hypothesis supported?

**Outcome B: the adaptive feature map shows improvement, but the evidence is inconclusive.** The direction is consistent (positive on both ROC-AUC and PR-AUC) and the underlying mutual-information signal driving the pair selection is not noise-level (0.78 vs. next-highest 0.21), but no paired statistical test reaches significance at n=200. This should not be read as "adaptive feature maps work" or "adaptive feature maps don't work" — it is a single, honestly-inconclusive data point.

## Recommendation for next step

Per Section 13's explicit instruction, VQC/QNN is **not** recommended as an automatic next step from this result — an inconclusive adaptive-kernel result is not evidence that a trainable variational circuit would help, and no concrete hypothesis for why a VQC would overcome QSVM's specific plateau has been established. If this adaptive-map direction is pursued further, the most defensible next step is evaluating on a **larger or repeated test set** (to gain the statistical power this 200-row set cannot provide) before concluding anything about the mechanism, not scaling training size further or introducing a new quantum model family.

**STOP per instruction.** No VQC/QNN, hardware, API, dashboard, or deployment work has been started.

## Files created

```
src/quantum/adaptive_feature_map.py
src/large_dataset/adaptive_qsvm_experiment.py
tests/test_adaptive_feature_map.py
tests/test_adaptive_qsvm_experiment.py
docs/ADAPTIVE_QSVM.md                          (this file)
results/large_dataset/adaptive_qsvm/
    predictions.csv, metrics.csv, adaptive_map_config.json, kernel_diagnostics.json,
    runtime.json, statistics.json, adaptive_qsvm_summary.json,
    roc_curves.png, pr_curves.png, confusion_matrices.png
```

`results/large_dataset/stage_d/predictions.csv` (the baseline QSVM's source) was read, never modified — verified by `tests/test_adaptive_qsvm_experiment.py::test_stage_d_baseline_metrics_unchanged_by_this_experiment`.
