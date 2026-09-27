# Phase 10: Hybrid Quantum-Classical Feature Layer + Logistic Regression Head

**Status:** Complete — outcome: **C (hybrid does not improve over classical Logistic Regression)**, with a critical control finding: **the hybrid's apparent edge over standalone QSVM is not attributable to anything quantum-specific**
**Scope:** Tests whether a trainable quantum FEATURE layer adds predictive value to a classical Logistic Regression head — not a quantum-vs-classical contest. Does not modify Stage A–D, Phase 7, Phase 8A, Phase 8B, or Phase 9 artifacts.
**Code:** [`src/quantum/hybrid_quantum_features.py`](../src/quantum/hybrid_quantum_features.py), [`src/large_dataset/phase10_hybrid_qml.py`](../src/large_dataset/phase10_hybrid_qml.py)
**Tests:** [`tests/test_hybrid_quantum_features.py`](../tests/test_hybrid_quantum_features.py) (11), [`tests/test_phase10_hybrid_qml.py`](../tests/test_phase10_hybrid_qml.py) (9)
**Results:** `results/large_dataset/phase10_hybrid_qml/`

---

## 1. Why Phase 8B's VQC was stopped

Phase 8B's standalone VQC used a trainable quantum circuit as the classifier itself (via shot-based `SamplerQNN`), and lost clearly and significantly to the baseline QSVM (ROC-AUC 0.6072 vs 0.7379, DeLong p=0.00083). That result said the circuit was a poor *classifier* under that specific, cheap training budget — it said nothing about whether a quantum circuit could be useful as a *feature transform* feeding a different, stronger classifier.

## 2. Why Phase 9 established Logistic Regression as the classical reference

Phase 9's classical ceiling benchmark found Logistic Regression the strongest of 4 classical models (ROC-AUC 0.8490), beating every QSVM variant significantly, with no evidence of quantum advantage anywhere. It is therefore the correct baseline this phase's hybrid must be measured against, not a weaker classical model.

## 3. Why a hybrid architecture is a scientifically justified new experiment

Neither prior result rules out a genuinely different architecture: a small quantum circuit producing *features* (expectation values) that are concatenated with the classical PCA representation and fed to Logistic Regression. This isolates a different, previously-untested question — "does the quantum transform add information the classical features don't already have?" — rather than repeating "can a quantum circuit alone classify well?" (already answered no, cheaply, in Phase 8B) or "does the existing QSVM kernel beat classical?" (already answered no, in Phases 7–9).

## 4. Exact hybrid architecture

```
Classical PCA-4 features (x)
    -> RY(x_i) encoding on qubit i (re-uploaded)
    -> RY(theta_i) trainable rotation on qubit i
    -> linear CNOT entangling
    -> exact <Z_i> expectation value per qubit (4 quantum features)
Concatenate: [4 classical PCA features, 4 quantum expectation features]
    -> sklearn LogisticRegression  <-- the ONLY component that predicts
Disease probability
```

**4 trainable quantum parameters** (`n_qubits=4`, `reps=1`). **Exact, no-shot-noise** computation: `EstimatorQNN` backed by `StatevectorEstimator(default_precision=0.0)`. A real, initially-silent bug was caught and fixed here during development: `EstimatorQNN` has its **own** `default_precision` parameter (default 0.015625) that overrides the estimator's own setting on every internal call — passing `default_precision=0.0` to the estimator alone was not enough; it had to also be passed to `EstimatorQNN` itself. Caught by a unit test asserting expectation values stay within the mathematically-required [-1, 1] bound (an early run produced -1.02, outside the valid range) — fixed and verified against a direct `Statevector.expectation_value()` computation before any real experiment ran.

**Training procedure** (why this is not "another VQC"): for a candidate quantum-weight vector θ, quantum features are computed for all training samples (exact, batched), concatenated with the classical PCA features, and a Logistic Regression is fit and scored via 3-fold cross-validation — on training data only. COBYLA searches θ to maximize this CV ROC-AUC. The winning θ's quantum features are then concatenated with the full training set's classical features, and a final Logistic Regression is fit once. Removing the quantum layer entirely leaves a complete, working classical pipeline (Model A) — the circuit is structurally a feature transform, never the decision function.

## 5. Data/preprocessing protocol

Identical to every Phase 8/9 experiment: n_train=2,000, seed=42 screening subset (`build_screening_split()`, reused unmodified), preprocessing/PCA-4 fit train-only (`process_stage`), fixed 200-row test set (fingerprint `96eac11a8394b87e`).

## 6. Leakage controls

The quantum layer's weights are selected via train-only CV (§4); the 200-row test set is transformed using already-fitted training preprocessing and scored exactly once, after all training/selection is frozen — verified structurally (`tests/test_hybrid_quantum_features.py::test_fit_accepts_no_test_data_parameter`, `test_predict_proba_accepts_no_label_parameter`).

## 7. Computational cost

Benchmarked before the full run (per instruction): a full-batch (n=2,000) forward pass + inner-CV LR fit costs ~8.0s per COBYLA function evaluation (measured directly, maxiter=6 → 48.06s). At the chosen `optimizer_maxiter=40`, estimated ~5.3 minutes; **measured 285.0s (4.75 min)** — matched the estimate. Total experiment (including Models A/B/D) well under 6 minutes.

## 8. Results (n_train=2,000, identical 200-row test set)

| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **A: Classical Logistic Regression** (reused) | **0.8490** | 0.8469 | 0.7475 | 0.8119 | 0.78 | 0.7708 |
| D: Control (PCA-4 + 4 fixed random features) | 0.8344 | 0.8280 | 0.7374 | 0.8119 | 0.775 | 0.7644 |
| **C: Hybrid (PCA-4 + trained quantum features)** | **0.8319** | 0.8248 | 0.7374 | 0.8119 | 0.775 | 0.7644 |
| B: Baseline QSVM (reused) | 0.7379 | 0.6920 | 0.7172 | 0.6931 | 0.705 | 0.7065 |
| B: MI-adaptive QSVM (reused) | 0.7666 | 0.7015 | 0.7374 | 0.7228 | 0.73 | 0.7300 |

## 9. Statistical comparison

Holm-Bonferroni applied once across all 4 comparisons (one family):

| Comparison | Δ ROC-AUC | DeLong p (Holm) | Significant? | Bootstrap CI excl. 0? |
|---|---:|---:|:---:|:---:|
| **Hybrid vs Classical LR (primary)** | −0.0171 | 0.191 | No | No |
| Hybrid vs Baseline QSVM | +0.0940 | 0.00017 | **Yes** | Yes |
| Hybrid vs MI-adaptive QSVM | +0.0653 | 0.0184 | **Yes** | Yes |
| **Hybrid vs Control (ablation)** | −0.0025 | 0.734 | No | No |

## 10. Ablation results — the critical finding

**The hybrid model and the non-quantum control (PCA-4 + 4 fixed random Gaussian features) are statistically indistinguishable** (Δ=−0.0025, p=0.734, McNemar p=1.0, identical sensitivity/specificity/accuracy/F1 to 4 decimal places). This means: the hybrid's significant improvement over standalone QSVM (§9, rows 2–3) is **not evidence that the trained quantum transform is doing anything useful** — a Logistic Regression fed 4 completely untrained random numbers performs equivalently. The correct reading of "hybrid beats QSVM" is "adding 4 more input dimensions to a linear model helps a little, regardless of what those dimensions contain" — not "the quantum feature layer learned something valuable." This ablation is the single most important result in this phase; without it, "hybrid significantly beats QSVM" would have been a genuinely misleading, unsupported quantum-advantage-adjacent claim.

## 11. Explainability

Final Logistic Regression coefficients (8 total: 4 classical PCA + 4 quantum) are persisted in `explainability.json`. Each `quantum_expZ_qubit{i}` feature is defined exactly as: the exact expectation value of the Pauli-Z observable on qubit i, after that qubit's PCA-encoded input passes through the trained ansatz and entangling layer — a value in [-1, 1]. **No claim is made about any clinical or physical meaning of these features beyond this literal definition** — given the ablation result (§10), such a claim would be unsupported.

## 12. Limitations

- The ablation (§10) is itself strong evidence against reading anything quantum-specific into the hybrid's numbers — this is disclosed as the headline finding, not buried.
- Single seed, single architecture (4 qubits, reps=1, 4 trainable params) — no sweep was performed, per instruction.
- The outer optimization (COBYLA over 4 parameters, 40 evaluations) is modest; a larger budget might find a different θ, though the control result suggests this would not change the substantive conclusion (any 4 extra numbers perform similarly here).
- n=2,000 training / n=200 test bound precision throughout.

## 13. Decision

**Outcome C: hybrid does not improve over the classical Logistic Regression model.** Δ ROC-AUC = −0.0171 (hybrid is numerically slightly worse, not better), Holm-adjusted p=0.191, bootstrap CI includes zero — not significant either way, but the point estimate is negative, so this is not even "improvement, but not significant" (Outcome B) — it is a clean non-improvement.

## 14. Does the hybrid direction deserve further investigation?

Not in its current form. The evidence gathered here — no improvement over classical LR, and no distinguishable difference from a non-quantum random-feature control — gives no basis for continuing this specific hybrid mechanism. Any future hybrid attempt would need to first clear the same ablation bar this one did not: showing the quantum-specific transform outperforms an equivalent-dimensionality non-quantum control, which this experiment shows it currently does not.

**STOP per instruction.** No automatic progression to a further phase; this result is for review.

## Files created

```
src/quantum/hybrid_quantum_features.py
src/large_dataset/phase10_hybrid_qml.py
tests/test_hybrid_quantum_features.py
tests/test_phase10_hybrid_qml.py
docs/PHASE_10_HYBRID_QML.md                (this file)
results/large_dataset/phase10_hybrid_qml/
    predictions.csv, metrics.csv, comparison_table.csv, hybrid_config.json,
    runtime.json, statistics.json, explainability.json,
    quantum_feature_outputs.json, phase10_summary.json,
    roc_curves.png, pr_curves.png, confusion_matrices.png
```

No Stage A–D, Phase 7, Phase 8A, Phase 8B, or Phase 9 artifact was modified — verified by `tests/test_phase10_hybrid_qml.py::test_prior_phase_artifacts_untouched_if_present`.
