# Full-Scale QSVM Experiment — Phase 5

**Status:** Phase 5 — Quantum Simulator Experiments / QSVM Benchmark
**Builds on:** `docs/DATASET.md` (Phase 1), `docs/PREPROCESSING.md` (Phase 2), `docs/CLASSICAL_BASELINE.md` (Phase 3), `docs/QUANTUM_PIPELINE.md` (Phase 4)
**Feeds:** Phase 6 (scoped IBM hardware validation)
**Code:** `src/quantum/qsvm.py`, `src/quantum/qsvm_experiment.py`
**Results:** `results/quantum/phase5/` (kernels, CV grid, final metrics, comparison table, figures)

This document reports the actual numbers from the real Phase 5 run. It does **not** claim the QSVM outperforms, matches, or is inferior to the classical baselines for any reason beyond what was directly measured — see Section 10.

---

## 1. Why a Precomputed-Kernel SVM

A QSVM in this project is **not** a new kind of neural network — it is an ordinary support vector machine, `sklearn.svm.SVC(kernel='precomputed')`, fed a **quantum-computed** kernel matrix instead of a classically-computed one (e.g. RBF). This is the standard construction for a fidelity-based quantum kernel method, and it was chosen (back in Phase 4's design, `docs/QUANTUM_PIPELINE.md` §6.2) specifically because:

- **Zero trainable quantum parameters.** The feature map is a fixed circuit template; nothing about it is fit to data. The *only* thing that is optimized is the classical SVM's regularization parameter `C` — exactly the same object Phase 3 tuned for the classical RBF-SVM.
- **Minimal, attributable difference from the classical baseline.** Phase 3's RBF-SVM and Phase 5's QSVM differ in exactly one place — the kernel function — with the surrounding machinery (leakage-safe CV, `GridSearchCV`, the multi-metric scoring dict, the single-touch test evaluation) **identical and reused, not reimplemented**.
- **`SVC(kernel='precomputed')` reuses sklearn's own well-tested SVM implementation.** Nothing about the optimization, the support-vector selection, or the decision function is quantum-specific — only the *number* `K(x_i, x_j)` fed into it is.

---

## 2. How the Fidelity Kernel Works (recap, unchanged from Phase 4)

```
K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2,      |phi(x)> = U_phi(x)|0000>
```

`x_i`, `x_j` are 4-dimensional quantum-ready feature vectors (PCA-reduced, range-normalized to `[0, π]` — Phase 2). `U_phi` is the `zz_feature_map` circuit (4 qubits, reps=2, linear entanglement — Phase 4's default, unchanged here). The kernel value is the squared overlap between the two resulting quantum states — a similarity measure, exactly analogous to what an RBF kernel provides classically, just computed via quantum state fidelity instead of a Gaussian of Euclidean distance.

**Nothing about the kernel definition changed in Phase 5.** `src/quantum/qsvm.py` and `qsvm_experiment.py` call Phase 4's `compute_statevectors()` and `kernel_matrix_from_statevectors()` directly — there is no second, parallel kernel implementation.

---

## 3. The Full 242/61 Experiment

| Step | What happens | Reused from |
|---|---|---|
| 1 | Load the locked `QuantumDataset` (242 train, 61 test, `split_id=e471025b07519a64`) | Phase 4's `load_quantum_dataset()`, itself calling Phase 2's `run_preprocessing_pipeline()` — **no new split created** |
| 2 | Build the `zz_feature_map` (4 qubits, reps=2, linear) | Phase 4's `build_feature_map()`, unchanged defaults |
| 3 | Compute statevectors for all 242 train rows, once each | Phase 4's `compute_statevectors()` |
| 4 | Compute statevectors for all 61 test rows, once each | Phase 4's `compute_statevectors()` |
| 5 | Assemble `K_train_train` (242×242, symmetric) from the train statevectors | Phase 4's `kernel_matrix_from_statevectors()` |
| 6 | Assemble `K_test_train` (61×242, asymmetric) from the test and train statevectors — **reusing the same train statevectors from step 3**, not recomputed | Phase 4's `kernel_matrix_from_statevectors()` |
| 7 | Cache both matrices to disk, keyed by feature-map config + split_id | New in Phase 5 (`compute_or_load_kernels()`) |
| 8 | Leakage-safe `C` tuning via `GridSearchCV` on `K_train_train`/`y_train` only | New wrapper (`run_qsvm_grid_search()`) around Phase 3's `CV_SCORING` / `MODEL_SELECTION_METRIC` |
| 9 | One evaluation on `K_test_train`/`y_test` | Phase 3's `evaluate_on_test()`, **called unmodified** |
| 10 | Descriptive comparison against Phase 3's already-saved Experiment 3 CSV | New (`build_descriptive_comparison()`), a pure file read — Phase 3 is not re-run |

**Confirmed shapes (Phase 5 Section C):** `K_train_train.shape == (242, 242)`, `K_test_train.shape == (61, 242)` — asserted in code (`qsvm_experiment.py`) and verified by `tests/test_quantum.py::test_full_scale_train_kernel_shape` / `test_full_scale_test_kernel_shape`.

---

## 4. How Leakage Is Prevented

### 4.1 The test set

Identical discipline to every prior phase: `run_qsvm_grid_search()`'s signature has **no test-kernel parameter at all** (verified by `test_run_qsvm_grid_search_has_no_test_kernel_parameter`, inspecting the signature directly — the same structural-guard pattern Phase 3 used for `run_grid_search`). `K_test_train` is passed to exactly one function, `evaluate_on_test()`, exactly once, after `search` (the completed `GridSearchCV`) already exists.

**Behavioral confirmation, not just structural:** `test_qsvm_cv_scores_unaffected_by_downstream_test_kernel_use` fits a model against `K_test_train` (simulating an out-of-order mistake) and then re-runs the tuning — the CV scores and selected `C` are bit-identical to a run where `K_test_train` was never touched at all, because tuning genuinely has no code path to it.

### 4.2 Why slicing the ONE precomputed 242×242 kernel per CV fold is leakage-safe (not just convenient)

This is the one genuinely new methodological question Phase 5 had to answer, and it is answered here explicitly rather than assumed.

`K(x_i, x_j)` is a **fixed function of two feature vectors** — the feature map has no fitted or learned state (Section 1). Unlike Phase 3's classical preprocessing (an imputer's median, a scaler's mean, PCA's axes — all of which genuinely depend on *which rows are present* and therefore must be refit inside every CV fold), a kernel *value* does not change depending on which other rows happen to be in the matrix. Slicing `K_train_train[fold_idx][:, fold_idx]` for a given fold is therefore **mathematically identical** to computing that fold's kernel from scratch using only its own rows — not an approximation, and not a leakage shortcut.

**This was verified empirically, not just argued:** `tests/test_quantum.py::test_sklearn_precomputed_cv_slicing_matches_manual_fold_slicing` compares `sklearn.model_selection.cross_val_score` on the full precomputed kernel against a hand-written loop that manually slices `K[np.ix_(train_idx, train_idx)]` / `K[np.ix_(val_idx, train_idx)]` per fold and fits/scores directly — **the scores match exactly**. `sklearn`'s native `kernel='precomputed'` handling inside `GridSearchCV`/`StratifiedKFold` performs precisely this two-axis slicing internally.

### 4.3 A disclosed, bounded simplification (read this before citing the CV numbers)

`X_train_quantum` (the 4-dimensional PCA representation every kernel value is computed from) was produced by **one** PCA fit on all 242 training rows — the single, top-level fit established by the Phase 2/4 data contract, which Phase 5 was explicitly instructed to reuse unchanged. This differs from Phase 3's classical CV protocol, which refit the **entire** preprocessing chain (imputer, encoder, scaler, selector, **and PCA**) fresh inside every fold.

Practically: a Phase 5 CV fold's "validation" rows had their PCA coordinates computed by a fit that also saw those same rows (along with the rest of the 242) — a mild, inherited form of leakage **upstream of and outside** anything Phase 5 itself does. It is disclosed here precisely because it is real, not because it is expected to matter much: PCA axes estimated from 242 rows versus a ~194-row fold-training subset are not expected to differ meaningfully at this sample size, and (per Section 4.2) the kernel computation itself adds no further leakage on top of it. **A fully fold-isolated quantum CV protocol — refitting PCA and recomputing a fold-specific kernel per fold — is a legitimate methodological upgrade for a future phase, not attempted here**, because Phase 5's brief was explicit about reusing the single locked representation, and because Section 4.2 already establishes that the *kernel* step itself is exact.

---

## 5. Hyperparameter Selection Methodology

| Setting | Value | Source |
|---|---|---|
| Parameter tuned | `C` only (the kernel has no other hyperparameters — the feature map is fixed) | — |
| Grid | `{0.01, 0.1, 1, 10, 100}` | The example grid given in the Phase 5 brief; a superset of Phase 3's RBF-SVM grid (`{0.1, 1, 10, 100}`), widened on the small side since a fidelity kernel's value scale differs from RBF's |
| CV method | `StratifiedKFold`, 5 folds, shuffled | `src.classical.train_baselines.CV_FOLDS` / `CV_SHUFFLE` (Phase 3's constants, reused) |
| Seed | 42 | `PreprocessingConfig.random_seed`, reused |
| Scoring | The full 7-metric dict (`roc_auc`, `average_precision`, `accuracy`, `recall`, `precision`, `specificity`, `f1`) | `src.classical.tuning.CV_SCORING` — **imported directly, not redefined** |
| Selection criterion | Mean CV ROC-AUC | `src.classical.tuning.MODEL_SELECTION_METRIC` — **imported directly**, the identical criterion and identical reasoning Phase 3 used (`docs/CLASSICAL_BASELINE.md` §9) |

### 5.1 The full CV grid table (actual run)

| C | CV ROC-AUC (mean ± std) | CV Accuracy | CV Sensitivity | CV Specificity | CV Precision | CV F1 | Rank (by ROC-AUC) |
|---|---|---|---|---|---|---|---|
| **0.01** | **0.6945 ± 0.0281** | 0.5413 | **0.0000** | **1.0000** | 0.0000 | 0.0000 | **1 (selected)** |
| 0.1 | 0.6945 ± 0.0285 | 0.5413 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 2 |
| 1.0 | 0.6942 ± 0.0306 | 0.6403 | 0.5755 | 0.6943 | 0.6147 | 0.5929 | 3 |
| 10.0 | 0.6629 ± 0.0737 | 0.6403 | 0.6032 | 0.6709 | 0.6109 | 0.6056 | 4 |
| 100.0 | 0.6461 ± 0.0949 | 0.6406 | 0.5858 | 0.6869 | 0.6068 | 0.5931 | 5 |

**A finding that must be reported precisely, not smoothed over:** the selected `C=0.01` has the *highest* CV ROC-AUC (barely — 0.6945 vs 0.6945 vs 0.6942 for C=0.01/0.1/1.0, a difference far smaller than the fold-to-fold standard deviation of ~0.03) but **CV sensitivity = 0.0 and CV specificity = 1.0 at that C** — the model predicts the majority class ("no disease") for every single validation patient, in every fold, at the default 0.5 threshold. ROC-AUC is threshold-independent and can still register moderately above chance (0.69) purely from how the model *ranks* patients by decision score, even while the *0.5-threshold prediction* has collapsed to one class. This is precisely the general principle already documented in `docs/CLASSICAL_BASELINE.md` §2 (a headline metric does not guarantee threshold-level usefulness) — Phase 5 is a concrete, quantum-specific instance of it, not a new failure mode invented here.

**This was not "fixed" by picking a different C.** The selection criterion (ROC-AUC, `refit="roc_auc"`) was fixed *before* this result was observed, inherited unchanged from Phase 3 for methodological consistency (Phase 5 Section F: "keep the experimental protocol consistent where appropriate"). Switching criteria after seeing an unflattering result would itself be a form of outcome-chasing. The finding is reported as-is; what to do about it (a different selection criterion, a wider C grid, a different feature map) is explicitly a Phase 6 question (Section 11).

---

## 6. Final Evaluation Methodology

After `GridSearchCV` selected `C=0.01`, sklearn **automatically refits** the winning model on the **entire** `K_train_train`/`y_train` (all 242 rows, no held-out fold) — this is sklearn's own `refit=` behavior, not reimplemented, and it is the same "freeze hyperparameters, refit on the full training portion" step Phase 3 relies on for its classical models.

`evaluate_on_test()` (Phase 3's function, called unmodified) is then invoked **exactly once**, with `K_test_train` as its only new input, producing `predict_proba(K_test_train)[:, 1]` and the full metric suite at the default 0.5 threshold, plus a percentile bootstrap 95% CI (1000 resamples of the 61 test predictions) for the headline metrics.

### 6.1 Final locked-test result (touched exactly once)

| Metric | Value | 95% Bootstrap CI |
|---|---|---|
| Accuracy | 0.6721 | [0.557, 0.787] |
| **Sensitivity** | **0.7500** | [0.594, 0.909] |
| **Specificity** | **0.6061** | [0.437, 0.774] |
| Precision | 0.6176 | — |
| F1 | 0.6774 | — |
| **ROC-AUC** | **0.7798** | [0.661, 0.888] |
| PR-AUC | 0.6839 | [0.527, 0.875] |

**Confusion matrix:** TN=20, FP=13, FN=7, TP=21 (out of 61 test patients).

**A second finding worth stating plainly:** the final, full-242-row-refit model does **not** show the degenerate all-negative behavior the CV folds showed at `C=0.01` — sensitivity is a substantive 0.75 on the locked test set. This is a real, observed discrepancy between the CV-averaged behavior at `C=0.01` and the behavior of the model actually deployed (refit on all 242 rows rather than the ~194-row CV-fold subsets). A plausible explanation is that a modestly larger training set shifts which points become support vectors and where the decision boundary sits relative to the 0.5 threshold — but this is **stated as a plausible read of the data, not confirmed**, per the instruction to avoid manufacturing explanations. It is flagged as a concrete Phase 6 investigation item (Section 11).

---

## 7. Computational Cost

Per Phase 5 Section E, cost was **measured, not assumed**, before running the full experiment.

| Step | Measured time (real 242/61 run) |
|---|---|
| 242 training-row statevectors | 0.449 s |
| 61 test-row statevectors | 0.085 s |
| `K_train_train` assembly (242×242, 29,161 pairs, symmetric) | 0.091 s |
| `K_test_train` assembly (61×242, 14,762 pairs) | 0.036 s |
| **Total kernel computation** | **0.662 s** |
| `GridSearchCV` (5 C values × 5 folds = 25 fits on precomputed sub-kernels) | ≈ 0.4 s |
| **Total experiment wall-clock (cold cache)** | **≈ 1.1 s** |
| Total wall-clock (warm cache) | ≈ 0.5 s (CV + eval only; kernel load is near-instant) |

**Efficiency choice, explicit:** Phase 4's `compute_kernel_matrix_cached()` was **deliberately not used** for this experiment, because calling it once for `K_train_train` and once for `K_test_train` would recompute the 242 training-row statevectors a second time inside the second call. Instead, `compute_or_load_kernels()` calls `compute_statevectors()` exactly once for the 242 training rows and once for the 61 test rows, and both kernel matrices are assembled from those same cached statevectors — zero redundant circuit evaluations.

**Restartability, and why fine-grained checkpointing was not built:** the measured total cost (≈0.66 s) is far below the threshold where mid-computation checkpointing would earn its complexity. `compute_or_load_kernels()` instead caches the two **final** matrices to disk, keyed by feature-map configuration and `split_id`; a restarted run checks for both cache files and, if present, skips kernel computation entirely (`test_compute_or_load_kernels_restartable_cache_hit`). This is a deliberately proportionate level of restartability for a sub-second computation, not a limitation glossed over — see Section 9 for the honest statement of what this would NOT be sufficient for at a larger scale.

**No premature reformulation.** Per instruction, the existing statevector-based fidelity kernel (Phase 4) was reused as-is; the only optimization applied was eliminating the one redundant recomputation described above — a minimal, reproducible change, not a new kernel implementation.

---

## 8. Comparison Protocol with Phase 3

**This is a descriptive comparison only. No statistical significance test (McNemar, DeLong, or otherwise) is performed in Phase 5** — per explicit instruction, that is deferred to a later phase.

| Model | Feature space | Accuracy | Sensitivity | Specificity | Precision | F1 | ROC-AUC | PR-AUC | Train (s) | Inference (ms/rec) |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | quantum_ready | 0.8689 | 0.8571 | 0.8788 | 0.8571 | 0.8571 | 0.9210 | 0.9033 | 0.079 | 0.918 |
| RBF-Kernel SVM | quantum_ready | 0.8689 | 0.8929 | 0.8485 | 0.8333 | 0.8621 | 0.9242 | 0.9090 | 0.114 | 0.488 |
| Random Forest | quantum_ready | 0.8689 | 0.8571 | 0.8788 | 0.8571 | 0.8571 | 0.9167 | 0.9046 | 0.274 | 0.455 |
| XGBoost | quantum_ready | 0.8361 | 0.8571 | 0.8182 | 0.8000 | 0.8276 | 0.9275 | 0.8963 | 0.126 | 0.248 |
| **QSVM (fidelity kernel)** | quantum_ready | **0.6721** | **0.7500** | **0.6061** | 0.6176 | 0.6774 | **0.7798** | 0.6839 | 0.003 | 0.005 |

**All five rows come from models trained and evaluated on the identical 4-dimensional, PCA-reduced, `[0, π]`-normalized representation, on the identical locked 242/61 split** — this is exactly the fair, apples-to-apples comparison Phase 3's Experiment 3 was built to enable (`docs/CLASSICAL_BASELINE.md` §15).

**On every metric measured, the classical models outperform the QSVM on this dataset, at this qubit count, with this feature map, on this test set.** This is stated as a direct numeric fact, not softened — and Section 10 states precisely what it does and does not establish.

**One honest, non-performance observation:** the QSVM's training and inference times are far lower than the classical models' — but this reflects only that `SVC(kernel='precomputed')` skips the classical models' internal feature transformation and hyperparameter-search overhead at *evaluation* time, and specifically **does not include the ≈0.66 s kernel-construction cost**, which is a real, unavoidable, and much larger cost than any classical model's fit time. Reporting the bare SVM fit time without this context would be misleading; it is disclosed here explicitly.

---

## 9. Reproducibility

Recorded in every `results/quantum/phase5/run_record.json`:

| Item | Value |
|---|---|
| Random seed | 42 |
| Split ID | `e471025b07519a64` (asserted in code — the experiment refuses to proceed on an unrecognized split) |
| Python | 3.11.9 |
| scikit-learn | 1.8.0 |
| Qiskit | 2.2.3 |
| NumPy | 1.26.4 |
| Quantum config | `{backend_name: statevector, feature_map_name: zz_feature_map, reps: 2, entanglement: linear, ...}` |
| CV configuration | 5-fold `StratifiedKFold`, shuffled, seed 42, `C_grid=[0.01, 0.1, 1, 10, 100]`, selection metric `roc_auc` |
| Kernel cache identifiers | `K_train_train_zz_feature_map_q4_r2_linear_e471025b07519a64.npy`, `K_test_train_zz_feature_map_q4_r2_linear_e471025b07519a64.npy` |
| Timing | Full breakdown per computation stage (Section 7) |

**Verified, not assumed:** `tests/test_quantum.py::test_qsvm_grid_search_reproducible_with_same_seed` and `test_kernel_computation_reproducible_at_full_scale` confirm bit-identical results across repeated runs; re-running `python -m src.quantum.qsvm_experiment` a second time produced a cache hit and identical output (verified during development — see the deliverables report).

---

## 10. What This Result Establishes — and What It Does Not

**Establishes:**
- The full quantum data → feature map → fidelity kernel → precomputed-kernel SVM → locked-test-evaluation pipeline **works correctly end to end** at full scale (242/61), with all Phase 4 sanity-check properties (correct shapes, symmetry, diagonal ≈ 1, valid range, determinism) holding at scale.
- Leakage-safe `C`-tuning on a precomputed quantum kernel is both theoretically sound (Section 4.2) and empirically verified against manual fold-by-fold recomputation.
- On **this** dataset (UCI Heart Disease, Cleveland, n=303), at **this** qubit count (4), with **this** feature map (`zz_feature_map`, reps=2, linear), on **this** locked test split, the classical baselines from Phase 3 score higher on every reported metric than this QSVM configuration.

**Does NOT establish:**
- That quantum kernels are inferior to classical kernels in general, or on cardiovascular risk data in general, or even on this dataset under a different feature map/qubit count/encoding.
- That the ~5-point gap between the CV-tied top-3 `C` values (0.01/0.1/1.0) and their wildly different sensitivity behavior is fully understood — it is documented, not explained away (Section 5.1).
- That the CV-vs-final-test sensitivity discrepancy (Section 6.1) has a confirmed cause.
- Anything statistically about whether the classical-vs-QSVM gap is significant — no significance test was run (by design, Section 8).
- Anything about real quantum hardware — this entire experiment ran on the exact `Statevector` simulator established in Phase 4; no noise, no hardware, no Phase 6 work is incorporated here.

---

## 11. Limitations

| # | Limitation | Consequence |
|---|---|---|
| 1 | Single feature-map configuration | `zz_feature_map`/reps=2/linear was not swept. The observed gap could narrow, widen, or reverse under a different configuration — untested here by design (Phase 4/5 scope). |
| 2 | PCA fit once, not per-fold (Section 4.3) | A disclosed, bounded methodological simplification inherited from the Phase 2/4 data contract; not expected to materially affect the result at this sample size, but not proven not to. |
| 3 | `C` grid is small (5 values) | Per instruction, proportionate to the dataset size — but the top-3 values were statistically tied on CV ROC-AUC while behaving very differently at the 0.5 threshold (Section 5.1), suggesting the grid may be too coarse to resolve this region well. |
| 4 | No statistical significance test | The classical-vs-QSVM gap has not been tested for significance (by design — deferred). It should not be read as "proven better," only "numerically higher on this run." |
| 5 | Exact simulation only | No noise model, no hardware. Real-device results (Phase 6, deliberately scoped small per `MVP_SPEC.md` §8.5) may differ substantially. |
| 6 | CV-vs-final-test sensitivity discrepancy unexplained | Section 6.1's observation is reported, not diagnosed. |
| 7 | Small test set (n=61) | Bootstrap CIs are wide (e.g. ROC-AUC CI spans 0.66–0.89) — the 0.78 point estimate should be read with that width in mind. |

---

## 12. Traceability

| Claim | Verified by |
|---|---|
| `K_train_train` shape (242,242), `K_test_train` shape (61,242) | `test_full_scale_train_kernel_shape`, `test_full_scale_test_kernel_shape` |
| `K_train_train` symmetric | `test_full_scale_train_kernel_symmetric` |
| Diagonal ≈ 1 | `test_full_scale_train_kernel_diagonal_near_one` |
| Values in valid fidelity range | `test_full_scale_kernel_values_within_valid_range` |
| Correct `SVC(kernel='precomputed')` usage | `test_build_qsvm_uses_precomputed_kernel`, `test_qsvm_rejects_non_square_matrix_as_train_kernel` |
| No train/test leakage (structural + behavioral) | `test_run_qsvm_grid_search_has_no_test_kernel_parameter`, `test_qsvm_cv_scores_unaffected_by_downstream_test_kernel_use` |
| Precomputed-kernel CV slicing is exact | `test_sklearn_precomputed_cv_slicing_matches_manual_fold_slicing` |
| Deterministic behavior | `test_qsvm_grid_search_reproducible_with_same_seed`, `test_kernel_computation_reproducible_at_full_scale` |
| Metric calculation | `test_qsvm_final_metrics_within_valid_ranges` |
| Restartable caching | `test_compute_or_load_kernels_restartable_cache_hit`, `test_compute_or_load_kernels_cache_miss_on_different_config` |
| Result serialization/loading | `test_run_qsvm_experiment_end_to_end_and_artifacts_load_back` |
| Configuration validation | `test_default_c_grid_is_the_brief_example_grid`, `test_run_qsvm_grid_search_rejects_mismatched_kernel_and_label_length` |
| Phase 1–4 unaffected | Full existing suite (141 tests) re-run green; raw data checksum unchanged |

This document should be treated as **derived from, and kept in sync with**, `results/quantum/phase5/`. If a number here ever disagrees with a fresh run of `python -m src.quantum.qsvm_experiment`, the generated artifacts are authoritative and this document should be corrected.
