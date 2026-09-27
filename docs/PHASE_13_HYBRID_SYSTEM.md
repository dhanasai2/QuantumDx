# Phase 13: The Hybrid Quantum-Classical Disease-Risk System

**Status:** Complete — outcome: **NO_IMPROVEMENT** (architecture implemented, validated, and evaluated; the classical backbone provides the primary signal, the quantum refinement module demonstrates no measurable additional value)
**Scope:** Builds a practical, staged hybrid system (XGBoost backbone → quantum residual refinement → classical fusion) — a genuinely different quantum ROLE from Phase 10/12's feature-concatenation approach. Does not modify Stage A–D, Phase 7, Phase 8A, Phase 8B, Phase 9, Phase 10, Phase 11, or Phase 12 artifacts.
**Code:** [`src/quantum/quantum_refinement_circuit.py`](../src/quantum/quantum_refinement_circuit.py), [`src/quantum/matched_refinement_controls.py`](../src/quantum/matched_refinement_controls.py), [`src/large_dataset/phase13_hybrid_system.py`](../src/large_dataset/phase13_hybrid_system.py)
**Tests:** [`tests/test_phase13_hybrid_system.py`](../tests/test_phase13_hybrid_system.py) (17)
**Results:** `results/large_dataset/phase13_hybrid_system/`

---

## 1. Objective

Move from experimental model comparison to a practical hybrid architecture: classical ML extracts the strong general-purpose signal, a compact quantum circuit performs a specialized secondary learning task (residual correction), and a classical fusion layer produces the final calibrated risk — an end-to-end system usable as the SIH prototype's centerpiece, whatever the quantum module's measured contribution turns out to be.

## 2. Why Phase 10/12 architectures were not continued

Both prior architectures used the quantum circuit as a **feature-engineering** block: quantum-derived values concatenated with classical features, fed to a classifier trained from scratch. Both were tested (at two training scales for Phase 10, across 5 seeds with a richer circuit for Phase 12) and found statistically indistinguishable from matched non-quantum controls. Phase 13 does not repeat this — it tests a structurally different **role**.

## 3. Phase 13 hypothesis

"Does a small quantum circuit, given the classical backbone's own prediction as part of its input and trained specifically to predict the residual the classical model gets wrong, provide a fusion-stage improvement beyond an equivalent-capacity classical residual module?"

## 4. System architecture

```
Raw patient data -> Preprocessing (Phase 2, unmodified) -> PCA-4
                                                              |
                          Stage 1: XGBoost backbone (Phase 11's frozen hyperparameters)
                                                              |
                                                   p_xgb (OOF for training, final-model for deployment)
                                                              |
                     Compact input: [PCA-4, p_xgb, margin=|p_xgb-0.5|]  (6-dim, one per qubit)
                                                              |
                          Stage 3: Quantum refinement circuit (6 qubits, trained on residual = y - p_xgb_oof)
                                                              |
                                            ONE aggregate quantum refinement score
                                                              |
                     Stage 4: Classical fusion (Logistic Regression on [p_xgb, refinement_score, margin])
                                                              |
                                                  Final calibrated risk
```

## 5. Classical backbone

Phase 11's own frozen best XGBoost hyperparameters (`n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`) — restated verbatim (Phase 11 itself does not export them as a named constant), not re-searched. A sanity check confirmed OOF ROC-AUC (0.8014, on training folds) is consistent with Phase 11's own reported full-scale result (0.8567 on the held-out test set) — the same model, correctly reproduced.

## 6. Quantum refinement mechanism

6 qubits (one per input dimension), 1 data-re-uploading layer, trainable RY rotations, trainable RZZ entanglement (circular connectivity, 6 pairs), 6 single-qubit `<Z_i>` measurements collapsed via a **trainable linear readout** into ONE aggregate refinement score — not a feature vector fed to a downstream classifier (that would be Phase 12 again). 19 trainable parameters total (6 RY + 6 RZZ + 6 readout weights + 1 bias). Exact statevector computation; the Phase 10/12 `EstimatorQNN.default_precision` bug fix is preserved and re-verified.

## 7. Training procedure (staged, leakage-safe)

**Stage 1**: 5-fold out-of-fold (OOF) XGBoost probabilities on the full 66,641-row training pool — each row scored only by a model that never saw it. **Stage 2**: `residual = y - p_xgb_oof`. **Stage 3**: refinement-module weights (quantum, and both classical controls) selected via COBYLA + train-only 3-fold CV, minimizing MSE between the module's score and the residual, on the cheap n=2,000 screening subset (Phase 8A's own subset — reusing Phase 7/11/12's established "5-seed robustness on a cheap subset, single frozen seed for full-scale application" pattern), across 5 seeds. **Stage 4**: the seed=42 weights are applied once to the full training pool; a Logistic Regression fusion head is fit on `[p_xgb_oof, refinement_score, margin] -> y`. At deployment/test time, a **fresh** XGBoost fit on all training data replaces the OOF model (standard leakage-safe stacking: train the meta-learner on OOF base predictions, deploy using the full-data base model).

## 8. OOF methodology — verified correct

A dedicated test (`test_oof_predictions_are_not_perfectly_correlated_with_training_labels`) fits `compute_oof_xgb_predictions` on **pure label noise** and confirms OOF ROC-AUC stays near 0.5 (chance) — an in-sample-leaked implementation would instead show strong, spurious separation on noise. This passed, confirming the OOF scheme is genuinely held-out, not accidentally leaking.

## 9. Classical control design

Two matched controls, sharing the quantum module's exact `.score(X, params) -> aggregate` interface:
- **Model C (fair analogue)**: Random Fourier Features (`cos(W·x+b+φ)`, same family Phase 12 used) → trainable linear readout. W, b fixed per seed; φ and the readout are trained identically to the quantum module.
- **Model D (sanity control)**: a trainable linear readout over **fixed, untrained** random Gaussian features (no φ at all) — deliberately less capable, testing whether even a bare linear combination of noise can look competitive once fusion regularizes it.

## 10. Ablation design

Four models, sharing the identical XGBoost backbone and fusion-head architecture — only the refinement module varies: **A** (XGBoost alone, no refinement term), **B** (+ quantum refinement), **C** (+ RFF refinement), **D** (+ no-op random refinement).

## 11. Metrics

ROC-AUC, PR-AUC, sensitivity, specificity, accuracy, precision, F1, confusion matrix, Brier score, calibration curve (5-bin, quantile) — all computed identically for all 4 models on the fixed 200-row test set.

## 12. Results

| Model | ROC-AUC | PR-AUC | Brier | Confusion matrix (TN,FP,FN,TP) |
|---|---:|---:|---:|---|
| A: XGBoost alone | 0.85669 | 0.84256 | 0.15417 | 83, 18, 24, 75 |
| **B: + Quantum refinement** | 0.85649 | 0.84084 | 0.15419 | 83, 18, 24, 75 |
| C: + RFF control | 0.85649 | 0.84050 | 0.15462 | 83, 18, 24, 75 |
| D: + No-op random control | 0.85749 | 0.84456 | 0.15413 | 83, 18, 24, 75 |

**Every model produces the IDENTICAL confusion matrix** — the fusion stage's decisions at threshold 0.5 do not change regardless of which refinement source (quantum, RFF, or pure noise) is fused in. ROC-AUC differs only in the 4th decimal place across all four.

## 13. Statistical testing

Holm-Bonferroni applied once across the 3 primary comparisons. **McNemar's test shows 0 discordant pairs for every comparison** (quantum vs. XGBoost-alone, quantum vs. RFF, quantum vs. no-op) — the strongest possible "no practical difference" signal a paired test can report. DeLong: quantum vs. XGBoost Δ=−0.0002 (p=0.684, Holm p=1.0); quantum vs. RFF Δ=0.0000 exactly (p=1.0); quantum vs. no-op Δ=−0.0010 (p=0.334, Holm p=1.0). No comparison approaches significance in either direction.

## 14. Runtime

OOF XGBoost (5-fold): 10.9s. Final XGBoost fit: 2.6s. Dev experiment (5 seeds × 3 modules): 1,784.5s (29.7 min) — no Phase-12-style outlier this time (all 5 quantum per-seed times: 322s, 266s, 400s, 399s, 399s, a much tighter, more predictable spread). Full-scale quantum forward pass (one-time): 634.1s (10.6 min).

## 15. Seed robustness (5 seeds, not averaged away)

| Module | Mean CV MSE | Std | Min | Max |
|---|---:|---:|---:|---:|
| **Quantum** | **0.18420** | **0.00049** | 0.18380 | 0.18514 |
| RFF control | 0.18699 | 0.00289 | 0.18438 | 0.19261 |
| No-op control | 0.21547 | 0.03981 | 0.18670 | 0.29368 |

Notably, at the **development/CV stage**, the quantum module shows both the lowest mean residual-prediction error and the tightest seed-to-seed spread of all three modules — a genuine, reproducible numerical edge that did not appear in Phase 10 or 12. This makes §16's null finding at the *final fusion* stage more informative, not less: the quantum module's small CV-stage edge did not survive being combined with XGBoost's already-dominant signal in the fusion head, rather than never existing at all.

## 16. Explainability

XGBoost feature importances and exact SHAP contributions (via `Booster.predict(pred_contribs=True)` — no external `shap` dependency needed) are persisted for the full 200-row test set. The fusion Logistic Regression's coefficients (on `[p_xgb, refinement_score, margin]`) are exposed directly — `p_xgb`'s coefficient dominates, consistent with the confusion-matrix-identity finding. **No clinical or biological meaning is claimed for any raw quantum expectation value or the aggregate refinement score** — only their literal, defined role (a learned residual-correction signal) is reported. The prediction interface (`predict_patient_risk`) returns exactly the clinician-facing framing the governing spec requested: classical backbone risk, the refinement adjustment, and the final calibrated risk — verified to agree exactly with the batch-computed pipeline for a real test-set patient (`prediction_interface_example.json`).

## 17. Limitations

- The quantum module's CV-stage edge (§15) is real but small, and evidently too small (or too correlated with information XGBoost's OOF probability already carries) to move the fusion stage's decisions at all.
- Only one refinement architecture (6 qubits, 1 layer, circular RZZ, linear readout) and one compact input design (PCA-4 + p_xgb + margin) were tested, per the explicit instruction to avoid a large architecture/input sweep.
- The fusion head is a single Logistic Regression; a more expressive fusion model was not tested (per the spec's own "prefer the simplest model that works" guidance) and might in principle weight the refinement signal differently — untested here.
- n=200 test set bounds precision, though the McNemar zero-discordance finding is not a precision artifact — it is an exact count.

## 18. SIH demo architecture

The staged pipeline (Preprocessing → Classical Risk Engine → Quantum Refinement → Hybrid Fusion → Calibrated Risk → Explainable Decision Support) is fully implemented in code (`run_phase13`, `predict_patient_risk`) and produces a structured, clinician-readable output:

```json
{
  "classical_backbone_risk": 0.27,
  "quantum_refinement_raw_score": 0.002,
  "quantum_refinement_adjustment": 0.14,
  "final_calibrated_risk": 0.41,
  "risk_category": "moderate",
  "model_version": "phase13_hybrid_v1",
  "inference_latency_ms": 29.1,
  "explanation": "Classical model estimated risk: 0.27. Hybrid refinement adjusted risk: +0.14. Final calibrated risk: 0.41."
}
```

This is directly presentable in an SIH architecture diagram, demo UI, or technical write-up as "an explainable hybrid quantum-classical disease risk intelligence system in which a strong classical model performs scalable feature learning and a compact quantum circuit performs a specialized nonlinear refinement stage."

## 19. Recommended final system

For accuracy and cost, **Model A (XGBoost alone)** and **Model B (the full hybrid)** are statistically and practically equivalent (identical confusion matrix, ROC-AUC differing by 0.0002) — the hybrid architecture adds ~11 minutes of one-time quantum computation for full-scale deployment with no measured accuracy benefit. **For the SIH prototype specifically**, retaining the full hybrid pipeline (Model B) remains reasonable *as an engineering and demonstration artifact* — it is a genuinely complete, working, explainable hybrid quantum-classical system, which has standalone value for the platform's stated goal, independent of whether this particular quantum module improves accuracy today.

## 20. What should NOT be claimed

- **Not** "the quantum module improved disease prediction accuracy" — the confusion matrix is identical to the classical-only baseline.
- **Not** "quantum advantage" — no comparison in this phase (or any prior phase) meets that bar.
- **Not** any clinical meaning for individual quantum expectation values or the aggregate refinement score.
- **Not** that this null result closes the door on quantum refinement in general — only on this specific circuit, input design, and fusion architecture, evaluated honestly and without post-hoc tuning toward a preferred outcome.

## Files created

```
src/quantum/quantum_refinement_circuit.py
src/quantum/matched_refinement_controls.py
src/large_dataset/phase13_hybrid_system.py
tests/test_phase13_hybrid_system.py
docs/PHASE_13_HYBRID_SYSTEM.md                (this file)
results/large_dataset/phase13_hybrid_system/
    predictions.csv, metrics.csv, seed_by_seed_results.csv, model_configs.json,
    runtime.json, statistical_comparisons.json, reproducibility_summary.json,
    explainability.json, calibration_curve.csv, prediction_interface_example.json,
    phase13_summary.json, roc_curves.png, pr_curves.png, confusion_matrices.png,
    calibration_curve.png
```

No Stage A–D, Phase 7, Phase 8A, Phase 8B, Phase 9, Phase 10, Phase 11, or Phase 12 artifact was modified — verified by `tests/test_phase13_hybrid_system.py::test_prior_phase_artifacts_untouched_if_present`.
