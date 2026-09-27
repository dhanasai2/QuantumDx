# Phase 12: Trainable Quantum Representation vs. Matched Non-Quantum Control

**Status:** Complete — outcome: **B (no measurable quantum contribution demonstrated)**
**Scope:** Tests a genuinely new hypothesis — can a materially different, richer trainable quantum representation add predictive information beyond an equivalent-capacity, matched non-quantum representation? Does not modify Stage A–D, Phase 7, Phase 8A, Phase 8B, Phase 9, Phase 10, or Phase 11 artifacts.
**Code:** [`src/quantum/quantum_latent_representation.py`](../src/quantum/quantum_latent_representation.py), [`src/quantum/matched_classical_control.py`](../src/quantum/matched_classical_control.py), [`src/large_dataset/phase12_quantum_representation.py`](../src/large_dataset/phase12_quantum_representation.py)
**Tests:** [`tests/test_phase12_quantum_representation.py`](../tests/test_phase12_quantum_representation.py) (19)
**Results:** `results/large_dataset/phase12_quantum_representation/`

---

## 1. Why Phase 12 exists

Phases 7–11 tested and closed several distinct quantum directions (unsupervised MI-adaptive and label-aware QSVM feature maps, a standalone VQC, and Phase 10's shallow single-layer hybrid) — none demonstrated a measurable, reproducible quantum contribution, at either n=2,000 or full scale (n=66,641). Rather than keep tuning any of those closed directions, Phase 12 asks a new question with a materially different circuit.

## 2–4. What Phases 9, 10, and 11 established

**Phase 9**: Logistic Regression was the strongest model at n=2,000 (ROC-AUC 0.8490), beating every QSVM variant significantly. **Phase 10**: a shallow (1-layer, K=4, fixed-CNOT) quantum feature layer feeding Logistic Regression was statistically indistinguishable from a random-Gaussian-feature control (Δ=−0.0025, p=0.734) — the quantum transform demonstrated no useful learned information. **Phase 11**: at full scale (n=66,641), this null result was *confirmed and strengthened* (Δ=0.0007, p=0.887), and the classical ranking itself changed (XGBoost overtook Logistic Regression) — establishing XGBoost as the reference classical model and reinforcing that "beats standalone QSVM" is not valid evidence of quantum contribution.

## 5. Why the previous hybrid architecture was rejected as the basis for further tuning

Phase 10/11's architecture (1 re-uploading layer, RY-only trainable rotation, fixed CNOT, single-qubit Z readout only, K=4) had already been tested at two training scales with a consistent null ablation result. Continuing to tune that same architecture would not test a new hypothesis — it would re-litigate an already-answered question. Per the explicit stop condition ("STOP if the proposed architecture is effectively identical to Phase 10"), Phase 12 uses a structurally different circuit (§6).

## 6. What is fundamentally different about Phase 12

| | Phase 10 | Phase 12 |
|---|---|---|
| Re-uploading layers | 1 | 2 |
| Trainable single-qubit gates | RY only | RY (per layer) |
| Entanglement | FIXED CNOT (linear chain) | **TRAINABLE RZZ** (circular connectivity, one more edge) |
| Observables | 4× `<Z_i>` only | 4× `<Z_i>` **and** 4× `<Z_i Z_j>` two-qubit correlators |
| Trainable parameters | 4 | 16 |
| Output dimension K | 4 | 8 |

The two-qubit ZZ correlators are, by construction, information a linear (PCA) representation or a single-qubit-only quantum readout cannot express — this is the concrete, falsifiable thing Phase 12 tests that Phase 10 did not.

**Matched control redesign**: Phase 10/11 used untrained random Gaussian noise as the control. Phase 12 uses classical **Random Fourier Features** (`cos(W·x + b + φ)`, Rahimi & Recht 2007) — a genuinely capable, well-established classical nonlinear feature-map family, structurally analogous to the quantum circuit (a Pauli-Z expectation value from an RY rotation *is* a cosine of that angle). W and b are fixed per seed; φ (K=8 phase parameters) is trained via the identical outer procedure the quantum block uses. This directly satisfies the requirement that the control "must not be intentionally weak."

## 7. Exact architecture

```
PCA-4 features (x)
  for layer in {1, 2}:
    RY(x_i) on qubit i           (data re-uploading, every layer)
    RY(theta_i) on qubit i       (TRAINABLE)
    RZZ(theta_ij) circular pairs (0,1),(1,2),(2,3),(3,0)   (TRAINABLE entanglement)
  measure: <Z_0>,<Z_1>,<Z_2>,<Z_3>, <Z_0 Z_1>,<Z_1 Z_2>,<Z_2 Z_3>,<Z_3 Z_0>   (K=8, EXACT, no shots)
Concatenate: [PCA-4, K=8 quantum] -> {XGBoost (Phase 11's frozen best params), Logistic Regression}
```

4 qubits throughout (same budget as every prior quantum phase); exact statevector-based expectation values via `EstimatorQNN` + `StatevectorEstimator(default_precision=0.0)` — **the Phase 10 `default_precision` bug fix is preserved and re-verified** (`EstimatorQNN(..., default_precision=0.0)` explicitly, not just the estimator; regression-tested by `test_estimator_qnn_default_precision_is_exact` and a direct agreement check against `Statevector.expectation_value()`).

## 8. Data split and training protocol

Reused, unmodified: `build_full_scale_split` (Phase 11's own function — training pool n=66,641, fixed 200-row test set, fingerprint `96eac11a8394b87e`) and `build_screening_split` (Phase 8A's n=2,000 subset, used here only for representation-weight selection). Representation weights (quantum θ, control φ) are selected via COBYLA + train-only 3-fold CV (scored with a fast Logistic Regression proxy) on the n=2,000 subset, independently across **5 seeds (42, 123, 2024, 7, 99)**. The seed=42 weights are then applied, in one forward pass, to the full 66,641-row training pool; the downstream classifiers (XGBoost with Phase 11's frozen hyperparameters, and Logistic Regression) are fit once on the full data; the 200-row test set is scored exactly once, after every selection is frozen.

## 9. Leakage controls

`select_representation_weights` has no test-data parameter (verified: `test_select_representation_weights_accepts_no_test_data_parameter`); PCA/preprocessing fit train-only (`process_stage`); the control's random projection (W, b) depends only on a seed, never on labels or the test set; the 200-row test set is transformed using already-fitted training preprocessing and scored once. No post-hoc re-tuning against test performance occurred anywhere in this module.

## 10. Computational cost — two honest surprises, disclosed plainly

Benchmarked before the full run: a single full-batch (n=2,000) forward pass cost 15.47s (≈2.3× Phase 10's circuit, consistent with its greater depth/observable count). Estimated per-seed COBYLA cost (maxiter=40) ≈ 620s, total dev experiment ≈ 52 min.

**Actual dev experiment: 10,350.9s (2.9 hours)** — roughly double the estimate. Per-seed quantum optimization times: 1035s, 1363s, 1056s, **5888s**, 986s. The **seed=7 outlier (5888s, ~16× the median)** is a genuine, disclosed finding: COBYLA's wall-clock cost varies substantially with its trust-region trajectory from different initial points, even though the *resulting* CV score (0.7834) was consistent with every other seed. This mirrors Phase 11's Random Forest runtime miss — reported honestly rather than minimized. The final full-scale quantum forward pass took 964.3s (16.1 min), close to the ~515s extrapolation's order of magnitude (somewhat higher, plausibly reflecting the same per-run variance).

## 11. Results (identical 200-row test set)

| Representation | Classifier | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| PCA-4 alone (reference) | XGBoost | 0.8396 | 0.8322 |
| PCA-4 alone (reference) | Logistic Regression | 0.8300 | 0.8242 |
| **Quantum (PCA-4 + K=8 quantum)** | **XGBoost** | **0.8369** | 0.8303 |
| Quantum (PCA-4 + K=8 quantum) | Logistic Regression | 0.8320 | 0.8155 |
| **Control (PCA-4 + K=8 RFF)** | **XGBoost** | **0.8433** | 0.8367 |
| Control (PCA-4 + K=8 RFF) | Logistic Regression | 0.8380 | 0.8312 |
| Model A: Best classical (Phase 11 XGBoost, full features, reused) | XGBoost | 0.8567 | 0.8426 |

Notably, the quantum representation's XGBoost result (0.8369) is numerically *below* using PCA-4 alone with no added representation at all (0.8396) — adding the trained quantum features did not even clear the bar of "no representation added."

## 12. Statistical comparisons

Holm-Bonferroni applied once across 4 comparisons (one family):

| Comparison | Δ ROC-AUC | DeLong p (Holm) | Significant? | Bootstrap CI excl. 0? |
|---|---:|---:|:---:|:---:|
| **Quantum (XGB) vs Control (XGB) — primary** | **−0.0064** | 0.388 | No | No |
| Quantum (LR) vs Control (LR) | −0.0060 | 0.691 | No | No |
| Quantum (XGB) vs Best Classical | −0.0198 | 0.083 | No (raw p=0.021 does not survive correction) | No |
| Quantum (XGB) vs PCA-4 alone | −0.0027 | 0.691 | No | No |

Every comparison's point estimate favors the non-quantum alternative (control, best classical, or plain PCA-4); none reaches significance in either direction.

## 13. Reproducibility (5 seeds, not averaged away)

| Seed | Quantum dev CV ROC-AUC | Control dev CV ROC-AUC |
|---:|---:|---:|
| 42 | 0.78791 | 0.78626 |
| 123 | 0.78605 | 0.78073 |
| 2024 | 0.78770 | 0.78420 |
| 7 | 0.78345 | 0.78452 |
| 99 | 0.78411 | 0.78376 |
| **Mean ± std** | **0.7858 ± 0.0018** | **0.7839 ± 0.0018** |

The quantum representation's CV performance is **highly stable across seeds** (std=0.0018, ≈0.2% relative) — this is not an unstable, seed-sensitive result; it is a consistently-reproduced null finding.

## 14. Runtime

Dev experiment: 10,350.9s (2.9h, dominated by the seed=7 outlier, §10). Full-scale preprocessing/PCA: 4.5s. Full-scale quantum forward pass: 964.3s (16.1 min, one-time). Classifier fits: 0.1–3.8s each (trivial).

## 15. Limitations

- COBYLA's wall-clock cost is unpredictable seed-to-seed (§10) even though outcomes are stable — a practical, not scientific, limitation worth flagging for anyone planning to repeat this class of experiment.
- Only one circuit architecture (2 layers, circular RZZ entanglement, Z+ZZ observables) was tested — the null result applies to this specific, materially-different-from-Phase-10 design, not to every conceivable richer quantum representation.
- The control (RFF) is one reasonable, principled choice of "matched non-quantum representation" — a different classical nonlinear family might behave differently, though RFF's structural analogy to the quantum circuit's own cosine-based readout makes it a strong, deliberately fair choice.
- n=200 test set bounds precision throughout, though here the effect sizes are all small and centered near zero, not merely imprecisely estimated.

## 16. Was quantum contribution demonstrated?

**No.** The quantum representation does not beat its matched non-quantum control (§12, primary comparison), does not beat the strongest classical baseline, and does not even beat using no added representation at all (plain PCA-4). This is a clean, highly reproducible (§13) null result for this specific architecture.

## 17. Was quantum advantage demonstrated?

**No — and the evidence bar for even considering that question (Success Criteria Case 3: quantum beats both control AND best classical) was not reached**, so no further confirmation analysis (additional seeds beyond the 5 already run, noise sensitivity, independent validation) is warranted or was performed.

## 18. Final recommendation

**Outcome B: no measurable quantum contribution.** Do not continue tuning this Phase 12 architecture. The classical XGBoost-on-full-features result from Phase 11 (ROC-AUC 0.8567) remains the strongest, most defensible candidate for the SIH prototype. Any further quantum investigation would need a new, concretely-motivated hypothesis distinct from both Phase 10's shallow single-layer design and Phase 12's richer multi-layer, trainable-entanglement, multi-observable design — both have now been tested, at multiple training scales and (for Phase 12) multiple seeds, with consistent null results.

**STOP per instruction.** No automatic further phase.

## Files created

```
src/quantum/quantum_latent_representation.py
src/quantum/matched_classical_control.py
src/large_dataset/phase12_quantum_representation.py
tests/test_phase12_quantum_representation.py
docs/PHASE_12_QUANTUM_REPRESENTATION.md                (this file)
results/large_dataset/phase12_quantum_representation/
    predictions.csv, metrics.csv, seed_by_seed_results.csv, model_configs.json,
    runtime.json, statistical_comparisons.json, reproducibility_summary.json,
    explainability.json, phase12_summary.json,
    roc_curves.png, pr_curves.png, confusion_matrices.png
```

No Stage A–D, Phase 7, Phase 8A, Phase 8B, Phase 9, Phase 10, or Phase 11 artifact was modified — verified by `tests/test_phase12_quantum_representation.py::test_prior_phase_artifacts_untouched_if_present`.
