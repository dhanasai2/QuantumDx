# Quantum Data Pipeline & Fidelity Kernel — Phase 4

**Status:** Phase 4 — Quantum Environment + Quantum Data Pipeline (infrastructure only)
**Builds on:** `docs/DATASET.md` (Phase 1), `docs/PREPROCESSING.md` (Phase 2), `docs/CLASSICAL_BASELINE.md` (Phase 3)
**Feeds:** Phase 5 (quantum simulator experiments / QSVM benchmark), Phase 6 (scoped IBM hardware validation)
**Code:** `src/quantum/config.py`, `data_contract.py`, `feature_maps.py`, `backends.py`, `kernel.py`, `sanity_experiment.py`
**Results:** `results/quantum/sanity/` (JSON report + kernel matrices)

This document explains the quantum infrastructure built in Phase 4, why each design choice was made, and reports the actual sanity-experiment numbers from the real run. **Nothing in this phase is a benchmark, a tuned model, or a performance claim** — see Section 9.

---

## 1. What Phase 4 Is (and Is Not)

Phase 4 builds the **infrastructure** that turns an already-preprocessed patient record into a quantum kernel value — deterministically, reproducibly, and without touching the locked test set in any way that matters. It does **not**:

- train or tune a QSVM (that is Phase 5),
- report any accuracy, sensitivity, or specificity for a quantum model (there is no quantum *model* yet — a kernel matrix is not a classifier),
- execute anything on real IBM hardware (that is Phase 6, and is deliberately scoped small — see `MVP_SPEC.md` §8.5),
- compare quantum against classical in any performance sense.

What it **does** produce, verified and tested: a deterministic mapping from the Phase 2 quantum-ready feature vector to a quantum state, a feature-map circuit with documented structure, a backend abstraction, a correctly-shaped and correctly-behaved fidelity kernel, and a small sanity experiment proving the whole chain runs end to end.

---

## 2. The Quantum Data Pipeline

```
Raw clinical data (data/raw/processed.cleveland.data)
      |
      v
Phase 2 preprocessing (impute, encode, scale, select)      <- UNCHANGED, reused
      |
      v
Leakage-safe, stratified train/test split (242 / 61, seed 42)   <- UNCHANGED, reused
      |
      v
Phase 2 quantum-ready branch: PCA (4 components) + range-normalize to [0, pi]  <- UNCHANGED, reused
      |
      v
QuantumDataset  (Section 3: X_train_quantum, X_test_quantum, y_train, y_test)
      |
      v
Quantum feature map  (Section 5: zz_feature_map, 4 qubits, reps=2, linear entanglement)
      |
      v
Fidelity kernel  (Section 6: K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2)
      |
      v
QSVM-ready kernel matrices: K_train_train, K_test_train, K_test_test
      |
      v
[Phase 5: fit a classical SVC(kernel='precomputed') on these matrices -- NOT done in Phase 4]
```

Everything above the "Quantum feature map" line is **Phase 2/3 code, called unmodified**. Phase 4 adds exactly two new stages: embedding a feature vector into a quantum state, and computing fidelities between those states.

---

## 3. The Quantum Input Data Contract

`src/quantum/data_contract.py::load_quantum_dataset()` is the **only** way anything in `src/quantum/` obtains data. It calls `src.preprocessing.pipeline.run_preprocessing_pipeline()` directly — the same function Phase 3 used for its Experiment 3 (the quantum-ready classical benchmark) — and returns:

| Field | Meaning |
|---|---|
| `X_train_quantum` | (242, 4) — PCA-reduced, range-normalized to `[0, π]` |
| `X_test_quantum` | (61, 4) — same transform, fit on train only, applied to test |
| `y_train`, `y_test` | Binary early-risk labels, identical to every prior phase |
| `n_qubits` | 4 — derived from `PreprocessingConfig.pca_n_components`, never hardcoded |
| `split_id` | `e471025b07519a64` — **identical** to the split_id already reported in `docs/PREPROCESSING.md` and `docs/CLASSICAL_BASELINE.md` |
| `preprocess_hash` | A SHA-256 fingerprint of the preprocessing config, for a later phase's fairness checks (mirrors Phase 2/3's `split_id` pattern) |

**No new split is created.** `test_quantum_dataset_reuses_phase2_split_id` and `test_load_quantum_dataset_does_not_create_a_new_split` (`tests/test_quantum.py`) verify this directly — the split_id above is reproduced, not re-derived, and calling `load_quantum_dataset()` twice returns bit-identical arrays.

**No information from the test set is used to fit anything in Phase 4**, for a structural reason as much as a procedural one: the quantum feature map is a **fixed, parameter-free circuit template** (Section 5) — there is nothing in it to fit. Every function in `src/quantum/kernel.py` accepts feature arrays only; none of them accept a label argument at all (verified by `test_kernel_functions_accept_no_label_parameter`, which inspects the function signatures directly, the same structural-guard pattern used in Phase 3 for `run_grid_search`).

---

## 4. Quantum Feature Scaling / Encoding

```
x_classical  (13 raw clinical features)
     |  Phase 2: impute + encode + scale + select   (fit on train only)
     v
x_selected   (10 features)
     |  Phase 2: PCA (4 components), fit on train only
     v
x_pca        (4 real numbers, unbounded range)
     |  Phase 2: MinMaxScaler(feature_range=(0, pi), clip=True), fit on train only
     v
x_quantum ∈ [0, π]^4        <-- this is what Phase 4 receives, unchanged
     |  Phase 4: angle encoding via the feature map's Parameter binding
     v
|phi(x_quantum)>  =  U_phi(x_quantum) |0000>          <-- a 4-qubit quantum state
```

**This entire chain up to `x_quantum` is Phase 2 code, already documented in `docs/PREPROCESSING.md` §8 and already leakage-tested there.** Phase 4 adds only the final arrow: binding the 4 real numbers in `x_quantum` to the feature map's 4 symbolic `Parameter` objects (`qc.assign_parameters(x_quantum)`), which is a **deterministic, side-effect-free, non-fitted** operation — there is no statistic to estimate, so there is no train/test asymmetry to get wrong at this step. The only reason this still deserves its own diagram is to make explicit that **no ad-hoc, per-sample renormalization happens anywhere** — the range guarantee `x_quantum ∈ [0, π]` comes entirely from Phase 2's already-fitted `MinMaxScaler(clip=True)`, and Phase 4 trusts it rather than re-deriving or re-checking it per sample (though `compute_statevectors()` does defensively check that the *column count* matches the feature map's qubit count).

---

## 5. The Quantum Feature Map

### 5.1 Choice: angle encoding via `zz_feature_map`

| Property | Value | Configurable via |
|---|---|---|
| Feature map | `zz_feature_map` (default) | `QuantumConfig.feature_map_name` — also supports `z_feature_map`, `pauli_feature_map` |
| Qubits | 4 | Derived from `PreprocessingConfig.pca_n_components` — **not** a separate quantum-side setting, so it can never silently disagree with Phase 2 |
| Repetitions | 2 | `QuantumConfig.reps` |
| Entanglement | `linear` | `QuantumConfig.entanglement` — also supports `circular`, `full` |
| Encoding mechanism | **Angle encoding** — one classical feature per qubit, entering as a single-qubit `RZ`-family rotation angle, repeated with entangling `RZZ`-family interactions between qubit pairs | — |
| Trainable parameters | **Zero.** The feature map is a fixed template; only the 4 *data* values are bound in. There is nothing here for a classical optimizer to learn — that is a structural difference from a VQC's ansatz (a later, separate model type), not an oversight. | — |

### 5.2 Why this feature map, at this qubit count, was selected

- **One qubit per feature (angle encoding), not amplitude encoding.** With 4 selected quantum-ready features, angle encoding needs exactly 4 qubits and keeps circuit depth linear in `reps`; amplitude encoding could pack 4 features into 2 qubits but requires a much deeper state-preparation routine for essentially no benefit at this scale. Angle encoding is also the encoding `MinMaxScaler(feature_range=(0, π))` was built for (Phase 2 §8) — the two were designed together.
- **`zz_feature_map` over `z_feature_map`.** `z_feature_map` applies only single-qubit rotations — no entangling gates at all, meaning the resulting kernel is essentially a product of single-feature kernels and cannot capture *feature interactions*. `zz_feature_map` adds `RZZ`-style entangling terms between qubit pairs (pattern set by `entanglement`), giving the kernel access to pairwise feature correlations, which is the entire point of using a quantum kernel instead of a classical product kernel. `pauli_feature_map` generalizes further (configurable Pauli strings) and is available for a later sweep, but `zz_feature_map` is the standard, most-published starting point and was chosen as the default for exactly that reason — reproducible and comparable to existing literature.
- **`reps=2`, not more.** Circuit depth (Section 5.3) grows with `reps`; 2 repetitions is the conventional default in the literature this project is benchmarked against and keeps the transpiled circuit shallow enough to remain plausible on near-term hardware (Phase 6). `QuantumConfig` warns (does not silently allow unchecked) above `reps=2` via `check_reps_budget()`.
- **`entanglement="linear"`, not `"full"`.** Linear entanglement (each qubit coupled only to its neighbor) uses the fewest two-qubit gates of the three options, which matters twice over: it keeps the *simulated* circuit cheap, and it is the entanglement pattern most compatible with a real device's limited qubit connectivity (relevant ahead of Phase 6, even though Phase 4 does not execute on hardware). `"full"` and `"circular"` are implemented and tested (`test_kernel_matrix_is_symmetric_when_expected` uses `"full"`) but are not the default.
- **This choice was deliberately NOT optimized.** Per the phase instructions, no aggressive circuit optimization or feature-map search was attempted — `zz_feature_map`/reps=2/linear is the standard, defensible starting point, chosen for correctness and reproducibility, not tuned for performance. Adaptive or learned feature maps are explicitly out of scope here (and, per `PRD.md` §6.1 / `MVP_SPEC.md`, out of scope for the MVP generally — a future research extension, not this phase's deliverable).

### 5.3 Measured circuit properties (not asserted — computed from the actual built circuit)

From the real sanity-experiment run (`results/quantum/sanity/sanity_report.json`):

| Property | Value |
|---|---|
| Feature map | `zz_feature_map` |
| Qubits | 4 |
| Repetitions | 2 |
| Entanglement | `linear` |
| Free parameters | 4 (one per qubit — matches feature count exactly) |
| Logical circuit depth (decomposed) | 19 |
| Gate counts (decomposed) | `{'u': 22, 'cx': 12}` |

`describe_feature_map()` decomposes the circuit before measuring depth and gate counts specifically so "depth" means the same thing a transpiler-facing hardware report (Phase 6) will also mean — not the depth of a single opaque high-level instruction.

---

## 6. The Fidelity Kernel

### 6.1 Definition

```
K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2,      |phi(x)> = U_phi(x) |0000>
```

This is the standard quantum kernel used by a fidelity-based QSVM: the squared overlap between two quantum states, each produced by encoding one patient's 4-dimensional quantum-ready feature vector through the same fixed feature map.

### 6.2 Why a fidelity kernel (and not, e.g., training a variational circuit) at this stage

A fidelity kernel requires **zero trainable quantum parameters** — the feature map is fixed, and the only thing that varies is the classical *data* bound into it. This makes it the cleanest possible first quantum experiment: any later difference measured between a QSVM (fidelity kernel + classical `SVC`) and the classical RBF-SVM baseline (`docs/CLASSICAL_BASELINE.md`) is attributable to the **kernel**, and to nothing else — no optimizer noise, no initialization variance, no training instability. A variational model (VQC/QNN) introduces all of those confounds and is deliberately a *separate*, later comparison, not conflated with this one.

### 6.3 Implementation

`src/quantum/kernel.py` computes the kernel in two stages, for efficiency and clarity:

1. **`compute_statevectors(X, feature_map, backend)`** — embeds every row of `X` through the feature map **once** (a statevector depends only on its own row, not on which other row it will be compared against). This is `O(n)` circuit evaluations, not `O(n²)`.
2. **`kernel_matrix_from_statevectors(...)`** — assembles the full matrix from cheap vector inner products of the already-computed statevectors. When `symmetric=True` (i.e. `X_a is X_b`, as for `K_train_train`), only the upper triangle is computed and mirrored — but **the diagonal is still computed explicitly**, never hardcoded to 1.0, because a measured deviation from exactly 1.0 is itself a useful floating-point / correctness signal (Section 7).

Three matrices are produced, matching the QSVM training/inference contract that Phase 5 will need:

| Matrix | Shape (sanity run) | Used for (Phase 5) |
|---|---|---|
| `K_train_train` | (20, 20) in the sanity run; (242, 242) at full scale | Fitting `SVC(kernel='precomputed')` |
| `K_test_train` | (10, 20) in the sanity run; (61, 242) at full scale | Predicting on the locked test set |
| `K_test_test` | (10, 10) in the sanity run; (61, 61) at full scale | Diagnostics only (not required to fit or predict) |

### 6.4 Verified properties (Section 6 checklist, all confirmed on the real sanity run)

| Check | Requirement | Actual result |
|---|---|---|
| Correct dimensions | `K_train_train` square at `n_train`; `K_test_train` is `n_test × n_train`; `K_test_test` square at `n_test` | (20,20), (10,20), (10,10) — all correct |
| Symmetric where expected | `K_train_train`, `K_test_test` symmetric; `K_test_train` not expected to be (and is not square) | `is_symmetric: true` for both square matrices |
| Diagonal ≈ 1 for ideal simulation | `K(x,x) = 1` up to floating-point error | Diagonal mean `0.999999999999997`; max deviation from 1.0 = **3.55e-15** (`K_train_train`), **3.33e-15** (`K_test_test`) — machine-precision only |
| Values within valid fidelity range | Every entry in `[0, 1]` | Observed range `[3.6e-05, 1.0]` across all three matrices; `within_valid_fidelity_range: true` |
| Deterministic under fixed configuration | Recomputing produces identical output | `K_train_train` recomputed fresh: **bit-identical**, max absolute difference = **0.0** |
| No test information leaks into kernel construction | Computing `K_test_train`/`K_test_test` must not alter `K_train_train`, and no label array is ever passed to any kernel function | `K_train_train` recomputed *after* computing both test-involving matrices: **identical**; kernel function signatures inspected directly — none accept a label parameter |

All six checks pass, and are enforced by `tests/test_quantum.py` (not just demonstrated once in the sanity script).

---

## 7. Backend: Statevector Simulation (and why hardware is postponed)

### 7.1 The abstraction

```
QuantumBackend (Protocol: compute_statevector(), capabilities())
      |
      +-- StatevectorBackend     IMPLEMENTED   -- Phase 4's only active backend
      |
      +-- NoisySimulatorBackend  interface only -- raises NotImplementedError; a Phase 5 decision
      |
      +-- IBMHardwareBackend     interface only -- raises NotImplementedError; scoped to Phase 6
```

`kernel.py` and `sanity_experiment.py` depend only on the `QuantumBackend` Protocol (`compute_statevector`, `capabilities`) — never on a concrete class, and never on `QuantumConfig.backend_name` directly. `get_backend(name)` is the **single** place backend selection happens. Swapping backends is therefore a one-line config change (`QuantumConfig(backend_name=...)`), not a rewrite of any pipeline code — verified by `test_backend_selection_is_purely_config_driven`.

### 7.2 Why `StatevectorBackend` needs no `qiskit-aer`

`StatevectorBackend` computes exact quantum states via `qiskit.quantum_info.Statevector.from_instruction()` — part of **core Qiskit**, already installed (`qiskit==2.2.3`). This was a deliberate, tested decision, not an oversight:

- **The risk that was avoided:** installing `qiskit-aer` (or `qiskit-machine-learning`, which depends on it) in this environment pulls in `numpy>=2.0` — a **major** version jump from the `numpy==1.26.4` that Phases 1–3's already-tested pipeline (`pandas`, `scikit-learn`, `xgboost`) currently runs on (confirmed via `pip install --dry-run` during development). A numpy major-version upgrade is exactly the kind of change that could silently alter floating-point behavior or break a dependency somewhere in the existing, validated Phase 1–3 test suite.
- **Why it was unnecessary for Phase 4:** at 4–6 qubits, a statevector has 16–64 complex amplitudes — trivially small regardless of simulator backend. Aer's advantages (GPU acceleration, large-qubit-count batching, shot-based sampling, device noise models) simply do not apply at this scale for *exact* simulation.
- **Verification, not assumption:** after building the quantum package, the full Phase 1–3 test suite (68 tests) was re-run and confirmed green, and `numpy.__version__` was confirmed unchanged at `1.26.4` — see Section 10.
- **This is why "exact simulator/statevector backend" (the Phase 4 instruction) and "no qiskit-aer" are not in tension** — `Statevector` *is* an exact statevector simulator; Aer is one *particular*, GPU-capable implementation of that idea, not the only one, and not the one this phase needed.

### 7.3 Why IBM hardware is intentionally postponed

Per explicit instruction, Phase 4 does not execute anything on real IBM hardware, and does not consume IBM Quantum runtime. `qiskit-ibm-runtime` (0.44.0) **is** already installed in this environment, but `IBMHardwareBackend` is a placeholder that raises `NotImplementedError` — the credentials, job submission, and quota-aware scoped experiment design are Phase 6 work (see `MVP_SPEC.md` §8.5 for the already-documented quota arithmetic: a full kernel is ~43,900 circuit evaluations, ≈3–4 hours of QPU time, against a free-tier allowance on the order of 10 minutes/month — so Phase 6 will run a small, fixed subsample validation experiment, not the full benchmark, exactly as already planned).

### 7.4 Why a noisy simulator is also postponed

`NoisySimulatorBackend` is a placeholder for the same reason `qiskit-aer` was not installed (Section 7.2) — a device noise model is normally built on Aer's noise-model machinery. Whether Phase 5 needs a noise-robustness check badly enough to accept that dependency (with a fresh compatibility check at that time) is a decision left to Phase 5, not assumed here.

---

## 8. Configuration and Reproducibility

Every quantum-side experimental choice lives in `src/quantum/config.py::QuantumConfig` (immutable, validated, `frozen=True` — the same pattern as `PreprocessingConfig`), with a matching `configs/quantum.yaml`:

| Field | Default | Meaning |
|---|---|---|
| `backend_name` | `statevector` | Which `QuantumBackend` to use |
| `feature_map_name` | `zz_feature_map` | Which feature map to build |
| `reps` | 2 | Feature-map repetitions (validated: 1–4, warns above 2) |
| `entanglement` | `linear` | `linear` / `circular` / `full` |
| `paulis` | `("Z", "ZZ")` | Only used by `pauli_feature_map` |
| `random_seed` | 42 | Matches `PreprocessingConfig.random_seed` by convention |
| `max_train_samples` | 300 | O(N²) kernel-cost guard for Phase 5 |
| `kernel_cache_dir` | `results/quantum/cache/kernels` | Disk cache location (Section 6.3) |

**`n_qubits` is deliberately absent from `QuantumConfig`.** It is derived at run time from `QuantumDataset.n_qubits` (itself derived from `PreprocessingConfig.pca_n_components`), so the quantum pipeline can never be configured to silently disagree with what Phase 2 actually produced.

**Reproducibility, verified (not assumed):**

| Property | How verified |
|---|---|
| Same config → identical kernel matrices | `test_kernel_computation_is_bitwise_reproducible`; sanity run: max diff on recompute = `0.0` |
| Same config → identical `QuantumDataset` | `test_load_quantum_dataset_does_not_create_a_new_split` |
| Invalid config caught before any computation | 6 dedicated tests, e.g. `test_quantum_config_rejects_invalid_backend`, `test_check_qubit_budget_raises_above_hard_ceiling` |
| Software versions recorded | Every sanity report includes `python`, `platform`, `numpy`, `qiskit` versions |

Sanity-run environment: Python 3.11.9, Windows-10-10.0.26200, numpy 1.26.4, qiskit 2.2.3.

---

## 9. What This Document Does NOT Claim

- **No quantum model exists yet.** A kernel matrix is not a classifier. Phase 5 fits `SVC(kernel='precomputed')` on top of these matrices — that is where a "QSVM" first exists.
- **No performance claim of any kind.** The sanity experiment reports kernel value statistics (min/max/mean/diagonal deviation), not accuracy, sensitivity, or specificity. Section 11's report explicitly labels itself `"status": "sanity_check_only_not_a_benchmark"`.
- **No quantum-vs-classical comparison.** That requires a trained QSVM (Phase 5) evaluated on the same locked test set the classical baselines already used (`docs/CLASSICAL_BASELINE.md`), with the same statistical rigor (CV, bootstrap CIs, paired significance tests) — none of which exists yet.
- **No claim that quantum will outperform classical.** Nothing here should be read as evidence either way.

---

## 10. Reproducibility of the Environment Itself

| Check | Result |
|---|---|
| New dependencies installed in Phase 4 | **None.** `qiskit==2.2.3` was already present; no new package was installed. |
| `numpy` version before and after Phase 4 | `1.26.4` → `1.26.4` (unchanged) |
| Phase 1–3 test suite after Phase 4's code was added | 68/68 passed (`tests/test_dataset.py`, `tests/test_preprocessing.py`, `tests/test_classical.py`) |
| `data/raw/processed.cleveland.data` SHA-256 | `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8` — unchanged from Phase 1 |

---

## 11. Sanity Experiment — Actual Results

**This is not a benchmark.** It is a small, fixed subsample (20 train, 10 test rows drawn from the *locked* Phase 2 split, seed 42) used to prove the full chain works.

| Item | Value |
|---|---|
| Samples (train / test) | 20 / 10 (drawn from the full locked 242 / 61 split) |
| Quantum features | 4 |
| Qubits | 4 |
| Feature map | `zz_feature_map`, reps=2, entanglement=linear |
| Backend | `statevector` (exact) |
| `K_train_train` shape | (20, 20) |
| `K_test_train` shape | (10, 20) |
| `K_test_test` shape | (10, 10) |
| Min kernel value (across all 3 matrices) | 3.61e-05 |
| Max kernel value | 1.0 (diagonal) |
| Mean kernel value | 0.120 (`K_train_train`) / 0.082 (`K_test_train`) / 0.202 (`K_test_test`) |
| Diagonal deviation from 1.0 | 3.55e-15 (`K_train_train`), 3.33e-15 (`K_test_test`) — machine precision |
| Reproducibility check | Bit-identical on recompute (max diff = 0.0) |
| Split ID | `e471025b07519a64` — same as Phase 2/3 |

Raw report: `results/quantum/sanity/sanity_report.json`. Raw matrices: `results/quantum/sanity/K_*.npy`.

**This report is not, and must not be read as, model performance.** No label was used anywhere in its computation.

---

## 12. Limitations

| # | Limitation | Consequence |
|---|---|---|
| 1 | Sanity experiment uses only 20 train / 10 test rows | Proves correctness, not scale. The full kernel (242 train rows) is ~29,000 pairwise evaluations for `K_train_train` alone — feasible with the current `Statevector`-based approach at 4 qubits, but not yet measured; Phase 5's job. |
| 2 | No noise model | All kernel values reported here are from *exact* simulation. Real hardware (Phase 6) or a noisy simulator (if added in Phase 5) will show different, noisier values — Section 6.4's "diagonal ≈ 1" result is specific to exact simulation and will not hold on a noisy backend. |
| 3 | Feature map not optimized or swept | `zz_feature_map`/reps=2/linear is a defensible default, not a validated best choice. A configuration sweep (qubits fixed at 4, but reps/entanglement/feature-map-type varied) is future work, not attempted here to avoid conflating "infrastructure works" with "this configuration is good." |
| 4 | `qiskit==2.2.3`'s class-based feature maps (`ZZFeatureMap`, etc.) are deprecated | This implementation uses the function-based replacements (`zz_feature_map`, etc.) throughout — documented in `src/quantum/feature_maps.py`'s module docstring so a future Qiskit upgrade's changelog can be checked against this specific note. |
| 5 | Kernel computation is currently pure-Python-loop based (`kernel_matrix_from_statevectors`) | Adequate for the sanity scale (tens of rows); Phase 5 should re-time this at the full 242-row scale before assuming it is fast enough, and consider batching or `qiskit-aer` at that point if not (Section 7.2 explains why that trade-off was deferred, not ruled out). |

---

## 13. Traceability

| Claim in this document | Verified by |
|---|---|
| Quantum dataset dimensions match Phase 2 config | `test_quantum_dataset_dimensions_match_preprocessing_config` |
| Same split as Phase 2/3, no new split created | `test_quantum_dataset_reuses_phase2_split_id`, `test_load_quantum_dataset_does_not_create_a_new_split` |
| Feature map construction correct for all 3 types | `test_build_feature_map_all_types`, `test_build_feature_map_qubit_count_matches_request` |
| Invalid configuration rejected | `test_quantum_config_rejects_*` (6 tests), `test_build_feature_map_invalid_*` (2 tests) |
| Kernel matrix shapes correct | `test_kernel_matrix_shape_symmetric`, `test_kernel_matrix_shape_asymmetric` |
| Kernel symmetry | `test_kernel_matrix_is_symmetric_when_expected` |
| Kernel diagonal ≈ 1 | `test_kernel_diagonal_approximately_one` |
| Kernel values in valid range | `test_kernel_values_within_valid_fidelity_range` |
| Reproducibility | `test_kernel_computation_is_bitwise_reproducible`, `test_kernel_cache_hit_on_second_call` |
| Backend abstraction is config-driven | `test_backend_selection_is_purely_config_driven` |
| No label leakage into kernel functions | `test_kernel_functions_accept_no_label_parameter` |
| Sanity experiment runs end to end | `test_sanity_experiment_runs_end_to_end`, `test_sanity_experiment_reports_split_id_matching_phase2` |
| Phase 1–3 unaffected by Phase 4 | Full existing suite (68 tests) re-run green; raw data checksum unchanged |

This document should be treated as **derived from, and kept in sync with**, `results/quantum/sanity/sanity_report.json`. If a number here ever disagrees with a fresh run of `python -m src.quantum.sanity_experiment`, the generated report is authoritative and this document should be corrected.
