# Phase 8B: VQC (Variational Quantum Classifier) Screening

**Status:** Complete — outcome: **C (VQC clearly worse than baseline QSVM, statistically supported) — stop this specific VQC direction**
**Scope:** A cheap (n_train=2,000) screening test of whether a small trainable VQC provides enough evidence of improvement over the baseline QSVM to justify further investigation. Does not modify Stage A–D, Phase 7, or Phase 8A artifacts.
**Code:** [`src/quantum/vqc_classifier.py`](../src/quantum/vqc_classifier.py), [`src/large_dataset/phase8b_vqc_screening.py`](../src/large_dataset/phase8b_vqc_screening.py)
**Tests:** [`tests/test_vqc_classifier.py`](../tests/test_vqc_classifier.py) (10), [`tests/test_phase8b_vqc_screening.py`](../tests/test_phase8b_vqc_screening.py) (7)
**Results:** `results/large_dataset/phase8b_vqc/`

---

## 1. Objective

Determine whether a fundamentally different quantum learning mechanism — a trainable variational circuit, rather than a fixed fidelity kernel — provides screening-level evidence of improvement over the baseline QSVM, before any further investment.

## 2. Hypothesis

"Does a small trainable Variational Quantum Classifier provide enough evidence of improvement over the existing baseline QSVM to justify further investigation?" A screening question, not a final performance claim.

## 3. Existing baseline

The baseline QSVM at n_train=2,000, seed=42, on the identical fixed 200-row test set — **reused directly from Phase 8A** (`results/large_dataset/phase8a_label_aware/predictions.csv`, `baseline_proba`/`baseline_pred` columns), verified against the same test fingerprint before use. Not retrained.

## 4. VQC architecture

**Dependency note**: this repository had no `qiskit-machine-learning` / `qiskit-algorithms` / `qiskit-aer` before this phase (verified before writing any code — only bare `qiskit==2.2.3`, used for exact statevector simulation throughout the existing QSVM pipeline). Per explicit instruction, `qiskit-machine-learning==0.9.1` and `qiskit-algorithms==0.4.0` were installed (`--no-deps`, no existing pin to conflict with — no `requirements.txt` exists) and verified compatible with the already-installed qiskit via both an import check and a functional fit/predict smoke test, before use.

- **Feature map**: this project's own `build_feature_map("zz_feature_map", reps=1, entanglement="linear")` — the SAME angle-encoding circuit family the baseline QSVM's fidelity kernel already uses (reps trimmed from 2 to 1 to keep total depth shallow alongside the ansatz). Reusing the project's own encoding means any VQC-vs-QSVM difference reflects "trainable circuit vs. fixed kernel" specifically, not a different encoding.
- **Ansatz**: qiskit's `real_amplitudes` (RY rotations + linear CNOT entanglement — a standard, hardware-efficient ansatz), reps=2 → **12 trainable parameters** (4 qubits × 3 layers of RY, since `real_amplitudes(reps=2)` has `n*(reps+1)` rotation angles).
- **Optimizer**: COBYLA (gradient-free, standard for small variational-circuit problems), `qiskit_machine_learning.optimizers.COBYLA`.
- **Readout**: `qiskit_machine_learning.algorithms.VQC`'s default `SamplerQNN` (parity interpretation of measurement outcomes → class probabilities).

**Disclosed methodological difference from the QSVM baseline**: `VQC`'s `SamplerQNN` reads probabilities from **finite-shot measurement counts** (`StatevectorSampler` has no exact/analytic mode — verified: `shots=None` raises a `TypeError`). This means VQC training and inference carry genuine shot noise, unlike every other quantum computation in this project (the fidelity kernel is computed via exact `Statevector` amplitudes, zero shot noise). A fixed seed (`StatevectorSampler(seed=42)`) keeps the run reproducible, but this is not the same exactness guarantee the QSVM enjoys — disclosed here, not hidden.

## 5. Dataset and sampling

n_train=2,000, seed=42 — the identical screening subset Phase 8A used (`build_screening_split()`, reused unmodified), verified zero-overlap with the fixed 200-row test set (fingerprint `96eac11a8394b87e`).

## 6. Preprocessing

Identical to Phase 8A: `process_stage` (Phase 2's `SharedFeaturePipeline` + PCA-4), fit on the n=2,000 training subset only.

## 7. Leakage controls

`VQCClassifier`/`VQC.fit()` accepts only `(X_train, y_train)`; `predict_proba()` accepts only a feature array. Preprocessing/PCA is fit once, train-only, before any VQC code runs. The 200-row test set is transformed using the already-fitted training preprocessing and scored exactly once, after training completes — verified structurally (`tests/test_vqc_classifier.py::test_build_vqc_accepts_no_data_at_all`) and by the orchestrator's single `predict_proba` call on the test set.

## 8. Experimental protocol — computational cost estimation (required before running)

Before the full run, cost was benchmarked explicitly (per Section 11's instruction), not assumed:

| Config tried | Result |
|---|---|
| shots=4096, maxiter=5 (14 evals, n=2,000) | **399.0s** → extrapolated maxiter=100 run: 1.5–4+ hours. **Flagged as unexpectedly expensive; stopped.** |
| Profiling | Cost dominated by fixed per-sample circuit-dispatch overhead (~2.6ms/sample), not shot count — shots=256 only dropped cost to 92.9s/14 evals (6.6s/eval), not proportionally |
| shots=512, maxiter=20 (confirmed: maxiter caps n_function_evals 1:1) | 180.0s → 9.0s/eval |
| **Final config: shots=512, maxiter=50** | **Estimated ~7.5 min; measured 437.4s (7.3 min) — matched the estimate** |

This is a genuinely screening-scale optimization budget (~4× the 12 trainable parameters worth of function evaluations), not a fully-converged VQC — disclosed as a limitation (§12), not hidden.

## 9. Results (n_train=2,000, identical 200-row test set)

| Model | ROC-AUC | PR-AUC | Sens | Spec | Acc | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline QSVM | 0.7379 | 0.6920 | 0.7172 | 0.6931 | 0.705 | 0.7065 |
| **VQC** | **0.6072** | 0.6137 | 0.5051 | 0.6832 | 0.595 | 0.5525 |

VQC underperforms on every primary and secondary metric.

## 10. Statistical comparison (VQC − Baseline QSVM)

| Test | Δ ROC-AUC | p-value / CI | Significant? |
|---|---:|---|:---:|
| DeLong | −0.1307 | p=0.00083 | **Yes** |
| Paired bootstrap | −0.1307 | CI excludes zero | **Yes** |

Both tests agree: the gap is large and clears significance even at this screening scale (n=200 test set) — this is not a borderline or manufactured result.

## 11. Runtime/memory

VQC training: 437.4s (7.3 min), matching the pre-run estimate closely. Inference: 0.8s for 200 test samples. Memory: peak ~353MB process RSS — trivial, no concern at this scale.

## 12. Limitations

- **The VQC was lightly trained** (50 COBYLA function evaluations for 12 parameters) — a deliberate, disclosed choice to keep this screening cheap (Section 11's explicit stop-if-expensive instruction), not because more training was tried and failed. A more heavily-optimized VQC (larger shot budget, more iterations, a gradient-based method with parameter-shift gradients) could plausibly perform differently; this screening does not rule that out.
- Finite-shot (512) measurement noise adds variance to both training and inference that the exact-statevector QSVM baseline does not have.
- Single seed, single configuration — no architecture or hyperparameter sweep was performed, per explicit instruction not to tune until something wins.
- n=2,000 training / n=200 test both bound precision, though here the effect is large enough that this is not the limiting factor for THIS conclusion.

## 13. Outcome

**C — VQC is clearly worse than baseline QSVM (statistically supported).** Δ ROC-AUC = −0.1307, DeLong p=0.00083, bootstrap CI excludes zero. This is not a borderline call decided by convention; the gap is large and consistent across both statistical tests and every secondary metric.

## 14. Recommendation for next phase

**Stop this specific VQC screening direction** — this particular architecture/training-budget combination does not show competitive performance at this scale. This does **not** establish that no VQC could ever work here: the training budget was deliberately minimal (§12), and a materially different, better-resourced attempt (more optimizer iterations, exact expectation-based readout via `EstimatorQNN` instead of shot-based `SamplerQNN`, a different ansatz) might behave differently — but that would be a new, separately-justified experiment, not an automatic next step from this result. Per instruction, **no Phase 8C, no multi-seed robustness run, no scaling, no separate QNN implementation, no hardware/API/dashboard/deployment work** should begin until this result has been reviewed.

**STOP per instruction.**

## Files created

```
src/quantum/vqc_classifier.py
src/large_dataset/phase8b_vqc_screening.py
tests/test_vqc_classifier.py
tests/test_phase8b_vqc_screening.py
docs/PHASE_8B_VQC_SCREENING.md                (this file)
results/large_dataset/phase8b_vqc/
    predictions.csv, metrics.csv, vqc_config.json, runtime.json,
    statistics.json, phase8b_summary.json,
    roc_curves.png, pr_curves.png, confusion_matrices.png
```

No Stage A–D, Phase 7, or Phase 8A artifact was modified — verified by `tests/test_phase8b_vqc_screening.py::test_phase8a_artifacts_untouched_if_present`.
