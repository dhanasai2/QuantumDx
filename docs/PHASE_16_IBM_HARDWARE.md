# Phase 16: IBM Quantum Hardware Execution & Hybrid-System Validation

**Status:** Complete -- **Decision: A -- SUCCESSFULLY DEMONSTRATED.** Real job `dag54f8mhr3c73e4m300` completed on `ibm_marrakesh` (156-qubit IBM QPU).
**Code:** [`src/large_dataset/phase16_ibm_hardware.py`](../src/large_dataset/phase16_ibm_hardware.py), [`src/quantum/backends.py`](../src/quantum/backends.py) (`IBMHardwareBackend` completed)
**Tests:** [`tests/test_phase16_ibm_hardware.py`](../tests/test_phase16_ibm_hardware.py) (32, all mocked -- no real IBM job ever submitted by the test suite)
**Results:** `results/phase16_ibm_hardware/`

---

## 1. Why this phase exists

Phases 7-15 exhaustively tested whether any quantum component could **improve** on the classical XGBoost baseline (Phase 11) -- none did. Phase 16 does not revisit that question. It answers a different, narrower, engineering question:

> Can the project's already-developed hybrid quantum-classical architecture's quantum component actually execute on **real IBM Quantum hardware** -- real circuit, real transpilation, real job, real measurement counts, real job id?

This is a hardware execution / system validation phase, not a quantum-advantage phase. No claim of improved accuracy, sensitivity, specificity, or clinical value is made anywhere in this phase.

## 2. Existing hybrid architecture (unchanged)

```
Biomedical input -> classical preprocessing -> feature selection -> PCA-4 (CLASSICAL SIDE)
        -> quantum encoding -> trainable quantum circuit -> entanglement -> measurement (QUANTUM SIDE)
        -> quantum features/refinement -> classical fusion -> final disease-risk prediction (CLASSICAL SIDE)
```

Phase 16 executes **only the quantum-side circuit** on real hardware, for a small, fixed, already-preprocessed sample. XGBoost (Phase 11, the project's validated primary predictor) is untouched.

## 3. Which quantum circuit was selected, and why

**Phase 10's `HybridQuantumFeatureLayer` circuit** (`src.quantum.hybrid_quantum_features`) -- the simplest, most hardware-friendly circuit in the project:

- 4 qubits, 1 layer: RY angle-encoding + RY trainable rotation + a 3-gate linear CNOT chain (11 gates total before transpilation).
- Compare to Phase 12 (4 qubits, 2 layers, 16 trainable params, RZZ entanglement) or Phase 13/15 (6 qubits, RZZ entanglement) -- both richer, meaning more gates, deeper transpiled circuits, and more decoherence on a real noisy device for a workload whose entire point is demonstrating execution, not expressiveness.
- The circuit's already-selected, already-frozen theta is loaded verbatim from Phase 10's own persisted `results/large_dataset/phase10_hybrid_qml/explainability.json` (`selected_quantum_weights_theta`). **No new optimization of any kind happens in this phase, on hardware or otherwise** -- `tests/test_phase16_ibm_hardware.py::test_frozen_theta_matches_phase10_and_is_not_retrained` enforces this.

## 4. Why the workload is small

Per the governing spec's ~10-minute free-execution budget: 16 demo samples (default, configurable up to `MAX_SAMPLES_HARDWARE=32`), 4 qubits, 1 circuit layer, 1024 shots (default, capped at `MAX_SHOTS_HARDWARE=4096`), submitted as **one** batched Sampler job (not 16 separate jobs). No COBYLA, no hyperparameter search, no cross-validation, and no full 66,641-row or 200-row-test-set submission ever reaches IBM hardware -- enforced by `IBMHardwareConfig`'s resource guards (`max_qubits`, `max_shots`, `max_circuits`), which raise `QuantumConfigError` before any network call if exceeded.

## 5. IBM backend selection

`IBMHardwareBackend._select_backend` never hardcodes a backend name -- it calls `service.least_busy(min_num_qubits=n_qubits, operational=True, filters=lambda b: not b.configuration().simulator)`, auto-selecting the least-busy operational physical QPU with enough qubits. An explicit override is supported via `IBMHardwareConfig(backend_override=...)` but was not used for this run.

Credentials (`IBM_QUANTUM_TOKEN`, `IBM_QUANTUM_CHANNEL`, `IBM_QUANTUM_INSTANCE`) are read from the environment (via `.env`, gitignored -- see `.env.example`) only inside `_connect()`, at the moment a real network call is about to happen. If `IBM_QUANTUM_TOKEN` is unset, `IBMCredentialsError` is raised **before** `QiskitRuntimeService` is ever constructed (`tests/test_phase16_ibm_hardware.py::test_connect_never_calls_qiskit_runtime_service_without_token`).

## 6. Hardware execution process

For each of the 16 demo samples: PCA-4 features (already range-normalized to `[0, pi]` by the existing, unmodified preprocessing pipeline) are bound as the circuit's encoding parameters; Phase 10's frozen theta is bound as the trainable parameters; `.measure_all()` is appended. All 16 fully-bound circuits are transpiled for the selected backend (`generate_preset_pass_manager`, optimization level 1) and submitted together as one `SamplerV2` job. The real job id is captured **immediately after submission** (before waiting for a result), so even a timeout still yields a genuine, retrievable job id rather than nothing.

## 7. Job ID, backend, and configuration

See `results/phase16_ibm_hardware/hardware_execution.json` and `job_metadata.json` for the exact, real values from this run (job id, backend name, status, qubits, shots, circuit depth, gate counts, timestamps). These are never fabricated -- if execution did not complete, those files honestly reflect that instead.

## 8. Ideal vs. hardware comparison

For the identical 16 samples, `compute_ideal_expectations` computes EXACT `<Z_i>` expectation values via Phase 10's own unmodified `EstimatorQNN(default_precision=0.0)` machinery. `counts_to_expectation_values` converts the hardware's real measurement counts into empirical `<Z_i>` values using the same little-endian qubit convention the rest of the project already uses (`_z_observables`). `compare_ideal_vs_hardware` reports mean/max absolute deviation, per-qubit and per-sample breakdowns, and RMSE -- explicitly documented as reflecting **sampling and hardware noise, not clinical/predictive performance** (Section 7 of the governing spec).

## 9. Resource consumption

See `results/phase16_ibm_hardware/backend_metadata.json` for actual transpiled circuit depth and gate counts on the real backend, and `runtime`/timestamp fields in `job_metadata.json` for wall-clock cost.

## 10. Limitations

- The demonstration uses Phase 10's simplest circuit, not the richer Phase 12/13/15 architectures -- a deliberate choice for hardware resilience, not a claim that those circuits would behave identically on real hardware.
- PCA-4 zero-cost reduction happens entirely classically before the quantum circuit ever sees the data -- the quantum circuit itself never processes "high-dimensional" data directly (see Section 12 below).
- One execution, one backend, one seed's theta. This is a system-validation demonstration, not a statistically powered hardware-noise characterization; no claim of representativeness across IBM backends or over time is made.
- `NoisySimulatorBackend` remains unimplemented (Phase 5's own scope, untouched here).

## 11. How to reproduce

```bash
# Mock/dry-run (default; NEVER contacts IBM):
python -m src.large_dataset.phase16_ibm_hardware

# Real hardware (requires a real IBM_QUANTUM_TOKEN in .env; contacts the network):
python -m src.large_dataset.phase16_ibm_hardware --hardware [--n-samples 16] [--shots 1024] [--timeout-seconds 420]
```

Reproducibility artifacts persisted every run: random seed (42), demo sample ids, PCA config, circuit config, frozen theta values, shot count, and backend configuration -- see `phase16_summary.json`'s `reproducibility` block.

## 12. SIH objective mapping (precise, not overstated)

| Objective | What Phase 16 changes |
|---|---|
| "Design a hybrid quantum-classical architecture..." | No change -- already satisfied by Phases 10-15. Phase 16 demonstrates the quantum half of that architecture actually running on physical hardware, not just a simulator. |
| "...process high-dimensional biomedical data" | **Still only partially true, and this phase does not change that.** Classical preprocessing/PCA reduces the representation to 4 dimensions **before** quantum encoding -- the quantum circuit itself processes a 4-dimensional vector, not the original high-dimensional biomedical record. This remains a disclosed scope limitation, not something Phase 16 resolves. |
| "...scalable, interpretable, compatible with near-term quantum hardware and simulators" | **Now meaningfully more true on the hardware-compatibility axis specifically.** Before Phase 16: simulator-compatible only (`StatevectorBackend`), with `IBMHardwareBackend` an unimplemented stub. After Phase 16: `IBMHardwareBackend` is a real, working implementation, and (per Section 9's decision below) was exercised against a real IBM backend. Scalability is **not** improved by this phase -- a 16-sample, 4-qubit, 1-shot-batch demonstration says nothing about scaling to the full 66,641-row pool or to larger circuits; real hardware queues and per-shot cost make that a materially different, unaddressed problem. |
| "Benchmark the hybrid approach against classical baselines" | Unaffected -- Phase 16 makes no accuracy claim and does not compare hardware-executed predictions against classical baselines. That benchmark remains governed entirely by Phases 9-15's evidence. |

## 13. What this does NOT prove

- It does not prove quantum advantage.
- It does not prove quantum hardware improves disease detection.
- It does not prove the quantum features are clinically meaningful.
- It does not prove the architecture scales to the full dataset or to a larger circuit on real hardware.
- Hardware noise deviations reported here are a **system-validation artifact**, not a clinical-performance metric.

---

## RESULTS AND FINAL DECISION

### Real execution record

| Field | Value |
|---|---|
| Job ID | `dag54f8mhr3c73e4m300` |
| Backend | `ibm_marrakesh` (real 156-qubit IBM QPU, auto-selected via `least_busy`; 0 pending jobs at selection time) |
| Status | `COMPLETED` |
| Qubits used | 4 |
| Shots | 1,024 per circuit x 16 circuits (one batched job) |
| Transpiled circuit depth | 17 |
| Native gate counts (post-transpilation) | `rz: 11, sx: 11, cz: 3, measure: 4, barrier: 1` |
| Submitted | 2026-09-08T18:21:46Z |
| Completed | 2026-09-08T18:22:01Z (**15 seconds** wall-clock, well inside the ~10-minute budget) |

The original circuit (RY encoding + RY trainable + linear CNOT) was transpiled into `ibm_marrakesh`'s native basis gates (`rz`, `sx`, `cz`) -- every `RY` becomes a `sx`/`rz` sequence and every `CX` becomes a `cz` plus single-qubit corrections, which is why the gate list looks different from the source circuit while remaining logically equivalent.

### Ideal (exact statevector) vs. real hardware

| Metric | Value |
|---|---:|
| Mean absolute deviation | 0.0312 |
| Max absolute deviation | 0.1090 |
| RMSE | 0.0400 |
| Per-qubit MAD | qubit 0: 0.0248, qubit 1: 0.0326, qubit 2: 0.0288, qubit 3: 0.0385 |

All 16 samples' empirical `<Z_i>` values landed within ~0.11 of the exact statevector prediction -- consistent with expected shot noise (1,024 shots) plus real superconducting-qubit gate/readout error for a depth-17 circuit. This is a noise-characterization number, not a predictive-performance number: no disease-risk claim follows from it.

### Integration with QuantumBackend

`IBMHardwareBackend` now conforms to the same `QuantumBackend` Protocol as `StatevectorBackend`, with one honest exception: `compute_statevector()` raises a clear, specific error explaining that real hardware cannot produce an exact statevector (no fabrication, no silent wrong answer) and pointing callers to the new `run_circuit()` / `execute_with_metadata()` methods, which the Protocol was additively extended with (backward-compatible -- `StatevectorBackend` also implements `run_circuit()`, as a seeded mock, for testing). Backend selection, credential handling, and resource guards are entirely config-driven (`IBMHardwareConfig`), never hardcoded.

### Final decision

**A -- SUCCESSFULLY DEMONSTRATED.** The project's hybrid quantum-classical architecture's quantum component (Phase 10's circuit, frozen weights, unmodified) executed on real IBM Quantum hardware, producing a genuine job id, real transpiled-circuit metadata, and real measurement counts, compared honestly against the exact simulator reference. This does not change, and is not claimed to change, any predictive-performance conclusion from Phases 9-15.
