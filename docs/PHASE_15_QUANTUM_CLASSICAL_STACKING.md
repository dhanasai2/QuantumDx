# Phase 15: Quantum-Classical Ensemble Complementarity

**Status:** Complete -- **Decision: C -- NO QUANTUM CONTRIBUTION. Quantum predictive experimentation stops permanently (per the governing spec's own instruction).**
**Code:** [`src/large_dataset/phase15_quantum_classical_stacking.py`](../src/large_dataset/phase15_quantum_classical_stacking.py)
**Tests:** [`tests/test_phase15_quantum_classical_stacking.py`](../tests/test_phase15_quantum_classical_stacking.py) (26)
**Results:** `results/large_dataset/phase15_quantum_classical_stacking/`

---

## 1. Objective

This is the final, strictly bounded quantum-learning experiment for this project. It answers ONE narrow question:

> Can a quantum learner provide genuinely complementary predictive information beyond the strongest classical XGBoost model, when both are combined through a properly leakage-safe classical meta-learner?

If the answer is no (per the predefined acceptance criteria in Section 8 of the governing spec), quantum predictive experimentation stops permanently for this project.

## 2. Why this is different from Phases 10-14

| Phase | Question asked | Architecture |
|---|---|---|
| 10, 12 | Does a quantum-derived FEATURE improve a classifier when concatenated with classical features? | Feature concatenation -> single classifier |
| 13 | Does a quantum circuit predict what XGBoost got wrong (its residual)? | XGBoost -> residual -> quantum refinement -> fusion |
| 14 | Is there any defensible CLASSICAL modeling change (backbone/features/fusion) that beats XGBoost? | Classical only |
| **15** | **Does the quantum model's OWN prediction carry information DIFFERENT from XGBoost's, exploitable by prediction-level stacking?** | **XGBoost (p_xgb) + Quantum learner (p_quantum) -> Logistic Regression meta-learner** |

The key structural difference from Phase 10/12: this is prediction-level **stacking**, not feature concatenation. The quantum learner is trained as an independent classifier (its own scalar output is the thing being evaluated), and the meta-learner explicitly tests whether that scalar adds information beyond XGBoost's own scalar prediction. The quantum learner also never sees XGBoost's own output as an input (unlike Phase 13's refinement circuit), so `p_quantum` cannot be a trivial function of `p_xgb` by construction.

## 3. Architecture

```
Biomedical input -> classical preprocessing -> PCA-4
        |                                          |
        v (full classical features)                v (PCA-4, zero-padded to 6 dims)
    XGBoost                              Quantum learner (Model B)      Matched classical control (Model C)
    p_xgb                                p_quantum                     p_control
        |                                          |                         |
        +-------------------+  +-------------------+  +----------------------+
                             v  v                      v
                     Logistic Regression meta-learner (fit separately for B and C)
                                 |
                                 v
                          Final risk score
```

## 4. Quantum component

Reused **completely unmodified** at its own default configuration: `src.quantum.quantum_refinement_circuit.{RefinementConfig, QuantumRefinementModule}` (Phase 13's already-implemented, already exact-statevector-verified circuit) --

- 6 qubits, 1 data-re-uploading layer, trainable RY rotations, trainable RZZ entanglement (circular connectivity, 6 pairs), then a trainable linear readout collapsing the 6 `<Z_i>` expectation values into ONE aggregate scalar score.
- Exact statevector simulation: `StatevectorEstimator(default_precision=0.0)` + `EstimatorQNN(default_precision=0.0)` -- the project's twice-verified exact configuration.
- 19 total trainable parameters (6 RY + 6 RZZ + 6 readout weights + 1 readout bias).

**Input**: PCA-4 (the project's established quantum-ready representation), zero-padded to 6 dimensions. A latent bug was discovered during development: `build_refinement_circuit`'s entangling-pair wiring is hardcoded to the module-level `N_QUBITS=6` constant rather than actually generalizing to a passed-in `n_qubits` (instantiating `RefinementConfig(n_qubits=4, ...)` raises `qiskit.circuit.exceptions.CircuitError`). Rather than modify a Phase 13 file (out of scope), PCA-4 is zero-padded to 6 dimensions so the existing circuit is reused byte-for-byte at its own validated default; the 2 padding columns are constant zeros and carry no information, so they cannot manufacture a spurious result.

**What changed vs. Phase 13's use of this same circuit class**: the OUTER TRAINING OBJECTIVE. Phase 13 trained these parameters to minimize MSE against XGBoost's residual (`y - p_xgb_oof`), using XGBoost's own OOF probability and margin as two of the circuit's 6 inputs. Phase 15 instead trains the SAME parameter vector, end-to-end via COBYLA, to directly maximize the CV ROC-AUC of the circuit's own raw scalar score against `y` -- i.e., the quantum module IS the classifier here, using **only** PCA-4 (zero-padded) as input, deliberately never given XGBoost's own prediction (which would contaminate the complementarity test).

## 5. Classical backbone (Model A)

XGBoost with Phase 11/12/13's frozen hyperparameters, reused verbatim: `n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`. This is the baseline every prior phase has already validated most extensively (fixed-test-set ROC-AUC 0.8567).

## 6. Meta-learner

`sklearn.linear_model.LogisticRegression(max_iter=1000, random_state=42)`, fit on `[p_xgb, p_quantum]` (Model B) or `[p_xgb, p_control]` (Model C).

## 7. Matched classical control (Model C)

`src.quantum.matched_refinement_controls.{MatchedRFFConfig, MatchedRFFRefinement}` -- Phase 13's Random Fourier Feature control, reused unmodified at its own default configuration (`input_dim=6, n_features=6`, dimension-matched to the quantum block), exposing the identical `.score(X, params) -> (n,)` scalar interface. Its trainable parameters (phi + readout weights/bias) are selected via the IDENTICAL COBYLA + CV procedure as the quantum module, on the identical zero-padded PCA-4 input -- so any advantage the quantum block shows cannot be attributed merely to "having a trainable nonlinear auxiliary predictor at all."

## 8. Dataset sizes

- Training pool: 66,641 rows (`src.large_dataset.phase11_full_scale.build_full_scale_split`, unmodified).
- Fixed test set: 200 rows, fingerprint `96eac11a8394b87e` (touched only if the CV gate in Section 9 below passes).

## 9. Leakage controls

- **Fold isolation**: for each of 5 outer `StratifiedKFold` folds, the `SharedFeaturePipeline`, the PCA/quantum-range pipeline (`process_stage`), XGBoost, and the quantum/control theta/phi are ALL fit using only that fold's own training partition, then applied once to that fold's held-out validation partition.
- **No global screening-subset reuse**: Phase 12/13's shared external n=2,000 screening subset is deliberately NOT used here (it would overlap unpredictably with whichever rows land in a given outer fold's validation partition -- a subtle leakage path). Instead, each fold draws its OWN bounded, stratified subsample from ONLY that fold's own training partition for the inner COBYLA+CV weight search (`THETA_SELECTION_SUBSAMPLE_SIZE=1600`, `INNER_CV_FOLDS=3`, `OPTIMIZER_MAXITER=25`).
- **Meta-learner evaluated out-of-fold too**: the meta-learner is fit on 4 of the 5 folds' OOF rows and scored on the 5th, for each fold in turn (nested CV) -- so even though `p_xgb`/`p_quantum`/`p_control` are already fold-safe, the meta-learner itself is never evaluated in-sample either.
- **Test-set discipline**: `test_evaluation`/`statistics` are structurally `None` in the persisted summary unless the CV acceptance gate (Section 8 of the governing spec; 5 criteria) passes -- enforced in code, not just by discipline (see `run_all_stages`).

## 10. Computational budget

A single real full-scale fold was benchmarked before committing to the full run: theta selection (COBYLA maxiter=25, subsample=1600) took 407.9s, the full validation forward pass (n=13,329) took 129.1s, and control weight selection took 0.2s -- **537.2s per fold**, so **~45 minutes for the full 5-fold primary-seed run**, comparable in scale to Phase 12's own dev-experiment runtime. The multi-seed stability check (Criterion 3) is gated behind the primary run showing a CV improvement, to avoid a 4x-5x cost multiplier when there is no positive finding whose stability needs confirming.

---

## 11. CV results (5-fold, nested; meta-learner evaluated out-of-fold too)

| Model | CV ROC-AUC mean | std |
|---|---:|---:|
| A: XGBoost alone | 0.801190 | 0.006465 |
| B: XGBoost + Quantum (LR stack) | 0.801200 | 0.006474 |
| C: XGBoost + Matched Control (LR stack) | 0.801212 | 0.006451 |

`delta_vs_baseline` (B - A) = **+0.0000096** -- four orders of magnitude below the ~0.0065 fold-to-fold ROC-AUC noise floor. `delta_vs_control` (B - C) = **-0.0000128** -- the quantum stack does not even beat the matched classical control; the control edges it out.

## 12. Per-fold results

| Fold | A: XGBoost | B: +Quantum | C: +Control |
|---|---:|---:|---:|
| 0 | 0.798336 | 0.798345 | 0.798317 |
| 1 | 0.798342 | 0.798372 | 0.798401 |
| 2 | 0.800417 | 0.800397 | 0.800435 |
| 3 | 0.813679 | 0.813710 | 0.813678 |
| 4 | 0.795177 | 0.795174 | 0.795231 |

Model B beat Model A in only 3 of 5 folds (Criterion 2 requires at most one fold to disagree, i.e. >=4/5) -- not a consistent improvement, and the largest single-fold "win" (fold 3, +0.000031) is itself noise-scale.

## 13. Acceptance gate (Section 8, all 5 criteria)

| Criterion | Result |
|---|---|
| 1. Beats baseline mean CV | Nominally true (+0.0000096), but see Section 15 |
| 2. Fold consistency (>=4/5 folds) | **FAILED** (3/5) |
| 3. Multi-seed stability | Not run (gated behind a real primary-seed pass; none occurred) |
| 4. Beats matched classical control | **FAILED** (control scores marginally higher) |
| 5. Practical significance (> baseline std) | **FAILED** (delta is ~700x smaller than the std) |

**Gate: NOT PASSED.** Per protocol, the fixed 200-row test set was **not evaluated** -- `test_evaluation` and `statistics` are `null` in the persisted summary.

## 14. Prediction complementarity analysis (OOF only)

| Quantity | Value |
|---|---:|
| Pearson(p_xgb, p_quantum) | 0.4196 |
| Spearman(p_xgb, p_quantum) | 0.4199 |
| Pearson(p_xgb, p_control) | 0.5963 |
| Prediction disagreement rate (at 0.5) | 43.7% |
| Partial corr(p_quantum, y \| p_xgb) | 0.00237 |
| Partial corr(p_control, y \| p_xgb) | 0.00180 |

The quantum learner's predictions are meaningfully *different* from XGBoost's (correlation 0.42, not near 1.0; both disagree on the predicted class for 43.7% of patients) -- so this is not a case of quantum trivially re-deriving XGBoost's own decision. But that genuine difference carries essentially **no information about the label beyond what XGBoost already captures**: both the quantum and control partial correlations with y (after removing what p_xgb explains) are ~0.002-0.0024 -- statistically indistinguishable from each other and from zero. The quantum circuit is producing real, different, but uninformative noise from PCA-4's residual variance, not a genuinely predictive representation.

## 15. Materiality correction (a code-quality finding worth stating plainly)

The first automated decision run labeled this Outcome B ("complementary signal, no meaningful gain") purely because `delta_vs_baseline` was numerically `> 0`, with no minimum-effect-size floor -- 9.6e-6 satisfied that literal check despite being obvious noise. This violates the governing spec's own Section 8 Criterion 5 / Section 11 instruction to "reject improvements that are effectively noise." The decision function (`run_all_stages` in the source file) was corrected to require the ROC-AUC delta to exceed `MATERIALITY_ROC_AUC_DELTA = 0.001` (roughly 15% of the observed CV fold-to-fold std) or the partial-correlation edge over the control to exceed `MATERIALITY_PARTIAL_CORR_GAP = 0.01` before reporting Outcome B. The decision was then **re-derived from the already-persisted, genuinely out-of-fold predictions** (`oof_predictions.csv`) -- no new quantum computation was performed; only the classification of an already-computed, unchanged result was corrected. With the fix, this run correctly reports **Outcome C**.

## 16. Runtime

- Split: 0.16s
- OOF generation (5 folds, real full scale): **2,409.5s (~40.2 min)** -- dominated by per-fold quantum theta selection (COBYLA, ~130-930s/fold depending on convergence path) and the full-fold validation forward pass (~130-190s/fold).
- Nested-CV evaluation: 0.7s
- Multi-seed stability check: not run (gated behind a primary-seed pass).
- Stage 6 (test-set evaluation): not run (gate not passed).

A single real fold was benchmarked before committing to the full run (537.2s), giving a ~45-minute estimate that matched the actual 40.2-minute total closely.

## 17. Explainability

- **Clinical feature contribution**: reused, not retrained, from Phase 13's `explainability.json` (identical XGBoost architecture: same frozen hyperparameters, same feature groups, same full training pool) -- native SHAP contributions via `Booster.predict(..., pred_contribs=True)`.
- **Quantum-model contribution**: the quantum circuit's configuration (6 qubits, 1 layer, exact statevector simulation) is reported in full in `explainability.json`, alongside the meta-learner's own coefficients on `p_xgb` vs `p_quantum` (a near-parity or negative weight on `p_quantum` is expected and was observed, consistent with the near-zero partial correlation above). No clinical meaning is claimed for any quantum parameter or expectation value.

## 18. Limitations

- The per-fold theta/phi selection uses a bounded, stratified subsample (n<=1,600) of each fold's own training partition, not the full ~53,000-row partition, for computational tractability (see Section 10). This is the same cost-control pattern Phases 7/10/12/13 already used; it cannot be ruled out that a larger (or differently regularized) quantum training budget would change the result, but every prior phase's use of a similarly-bounded budget has been consistent with this phase's, so there is no evidence a larger budget would help.
- The multi-seed stability check (Criterion 3) never ran, since it is gated behind a real primary-seed CV improvement (to avoid a 5x cost multiplier chasing a result that already failed 3 of 5 criteria). This means seed-sensitivity of the *negative* result itself was not directly re-confirmed here -- though Phase 12's own 5-seed sweep of a related (richer) quantum representation already showed low seed variance (std 0.03 threshold, not flagged), making an unstable result unlikely.
- The zero-padding of PCA-4 to the circuit's native 6-qubit input (Section 4) is a disclosed, information-free adaptation, but a circuit designed natively for a 4-dimensional input was not tested (the existing Phase 13 circuit's entangling-pair bug prevented this without modifying a Phase 13 file, which was out of scope).

## 19. Final decision

**C -- NO QUANTUM CONTRIBUTION.**

The quantum stack does not beat XGBoost alone by more than measurement noise, does not beat the matched classical control, and is not consistent across folds. This is a clean, well-evidenced negative result: the quantum learner's predictions are genuinely different from XGBoost's (not a trivial restatement), but that difference carries no information about the label beyond what XGBoost already captures.

## 20. Should quantum experimentation stop permanently?

**Yes.** This was explicitly scoped as the final, most targeted test of quantum contribution this project would run (prediction-level stacking against a matched control, with genuine leakage-safe OOF generation) -- a materially different and more favorable framing for quantum than any of Phases 7-14 used. It also failed. Combined with 8 consecutive prior phases (7-14) finding no measurable quantum contribution across feature-concatenation, refinement, and now prediction-stacking architectures, further quantum predictive experimentation on this dataset is not a defensible use of further effort. The product build (paused pending this result, per explicit instruction) should proceed on the Phase 11 XGBoost backbone; the hybrid quantum architecture may still be presented as an honest engineering/demonstration capability (per Phase 13's own conclusion), never as an accuracy improvement.

## Files created

```
src/large_dataset/phase15_quantum_classical_stacking.py
tests/test_phase15_quantum_classical_stacking.py
docs/PHASE_15_QUANTUM_CLASSICAL_STACKING.md            (this file)
results/large_dataset/phase15_quantum_classical_stacking/
    oof_predictions.csv, fold_metrics.csv, seed_metrics.csv, metrics.csv,
    comparison_table.csv, prediction_complementarity.json, model_config.json,
    explainability.json, runtime.json, phase15_summary.json,
    roc_curves_cv.png, pr_curves_cv.png
```

No Phase 7-14 artifact was modified and the fixed 200-row test set was not touched -- verified by `tests/test_phase15_quantum_classical_stacking.py::test_prior_phase_artifacts_untouched_if_present` and `test_test_set_untouched_unless_cv_gate_passed_if_present`.
