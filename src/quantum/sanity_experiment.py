"""Phase 4 sanity experiment (Section 11).

THIS IS NOT A BENCHMARK. It proves the quantum pipeline works end to end --
dataset -> preprocessing -> quantum encoding -> feature map -> fidelity
kernel -> a valid kernel matrix -- on a small, fixed subsample. No model is
trained, no accuracy is computed, and nothing here should be read as
evidence about how well quantum-enhanced classification will perform.

Pipeline exercised:

    raw data (Phase 1)
        -> Phase 2 preprocessing (leakage-safe, locked split, reused unchanged)
        -> QuantumDataset (Section 2: X_train_quantum, X_test_quantum, y_train, y_test)
        -> a small, seeded, deterministic subsample of each split
        -> feature map (Section 5)
        -> fidelity kernel: K_train_train, K_test_train, K_test_test (Section 6)
        -> diagnostics + a reproducibility check

Run:
    python -m src.quantum.sanity_experiment
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import qiskit

from src.data.inspect_dataset import find_project_root
from src.preprocessing.config import DEFAULT_CONFIG as DEFAULT_PREPROCESSING_CONFIG
from src.quantum.backends import get_backend
from src.quantum.config import DEFAULT_CONFIG as DEFAULT_QUANTUM_CONFIG
from src.quantum.config import QuantumConfig, check_qubit_budget, check_reps_budget
from src.quantum.data_contract import QuantumDataset, load_quantum_dataset
from src.quantum.feature_maps import build_feature_map, describe_feature_map
from src.quantum.kernel import compute_kernel_matrix, kernel_diagnostics

PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results" / "quantum" / "sanity"

SANITY_N_TRAIN = 20
SANITY_N_TEST = 10


def _select_sanity_subsample(
    dataset: QuantumDataset, n_train: int, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw a small, deterministic, seeded subsample from the ALREADY-LOCKED
    train and test splits -- never mixes the two, never re-splits.
    """
    rng = np.random.RandomState(seed)
    n_train = min(n_train, len(dataset.X_train_quantum))
    n_test = min(n_test, len(dataset.X_test_quantum))
    train_idx = rng.choice(len(dataset.X_train_quantum), size=n_train, replace=False)
    test_idx = rng.choice(len(dataset.X_test_quantum), size=n_test, replace=False)
    X_train_sub = dataset.X_train_quantum[train_idx]
    X_test_sub = dataset.X_test_quantum[test_idx]
    return X_train_sub, X_test_sub, train_idx, test_idx


def run_sanity_experiment(
    quantum_config: QuantumConfig = DEFAULT_QUANTUM_CONFIG,
    preprocessing_config=DEFAULT_PREPROCESSING_CONFIG,
    n_train: int = SANITY_N_TRAIN,
    n_test: int = SANITY_N_TEST,
) -> dict:
    # --- 1. Reuse the Phase 2 locked split; no new split is created --------
    dataset = load_quantum_dataset(preprocessing_config)

    warnings = check_qubit_budget(dataset.n_qubits) + check_reps_budget(quantum_config.reps)

    X_train_sub, X_test_sub, train_idx, test_idx = _select_sanity_subsample(
        dataset, n_train, n_test, quantum_config.random_seed
    )

    # --- 2. Build the feature map (fixed, parameter-free template) ---------
    feature_map = build_feature_map(
        quantum_config.feature_map_name,
        dataset.n_qubits,
        reps=quantum_config.reps,
        entanglement=quantum_config.entanglement,
        paulis=quantum_config.paulis,
    )
    fm_report = describe_feature_map(
        feature_map,
        name=quantum_config.feature_map_name,
        reps=quantum_config.reps,
        entanglement=quantum_config.entanglement,
        paulis=quantum_config.paulis,
    )

    backend = get_backend(quantum_config.backend_name)

    # --- 3. Compute the three kernel matrices -------------------------------
    K_train_train = compute_kernel_matrix(X_train_sub, X_train_sub, feature_map, backend, symmetric=True)
    K_test_train = compute_kernel_matrix(X_test_sub, X_train_sub, feature_map, backend, symmetric=False)
    K_test_test = compute_kernel_matrix(X_test_sub, X_test_sub, feature_map, backend, symmetric=True)

    diag_train_train = kernel_diagnostics(K_train_train, symmetric=True)
    diag_test_train = kernel_diagnostics(K_test_train, symmetric=False)
    diag_test_test = kernel_diagnostics(K_test_test, symmetric=True)

    # --- 4. Reproducibility check: recompute K_train_train fresh -----------
    K_train_train_repeat = compute_kernel_matrix(X_train_sub, X_train_sub, feature_map, backend, symmetric=True)
    reproducible = bool(np.array_equal(K_train_train, K_train_train_repeat))
    max_reproducibility_diff = float(np.max(np.abs(K_train_train - K_train_train_repeat)))

    # --- 5. No-leakage sanity check: computing K_test_* does not use, and
    #        cannot have used, y_train/y_test (kernel.py functions accept no
    #        label argument at all) -- and does not alter K_train_train.
    K_train_train_after_test_use = compute_kernel_matrix(X_train_sub, X_train_sub, feature_map, backend, symmetric=True)
    unaffected_by_test_computation = bool(np.array_equal(K_train_train, K_train_train_after_test_use))
    # A real, non-vacuous check: no row in the train subsample should be
    # numerically identical to a row in the test subsample. train_idx and
    # test_idx already index into DISJOINT arrays (Phase 2's locked
    # train/test partition), so this is a belt-and-suspenders check on the
    # actual values, not just the indices.
    no_shared_rows_between_subsamples = not any(
        np.any(np.all(np.isclose(train_row, X_test_sub), axis=1)) for train_row in X_train_sub
    )

    report = {
        "status": "sanity_check_only_not_a_benchmark",
        "dataset": {
            "split_id": dataset.split_id,
            "preprocess_hash": dataset.preprocess_hash,
            "n_train_full": int(len(dataset.X_train_quantum)),
            "n_test_full": int(len(dataset.X_test_quantum)),
            "n_quantum_features": dataset.n_qubits,
            "quantum_range": list(dataset.quantum_range),
        },
        "sanity_subsample": {
            "n_train_sampled": int(len(X_train_sub)),
            "n_test_sampled": int(len(X_test_sub)),
            "seed": quantum_config.random_seed,
            "train_indices_within_locked_train_split": train_idx.tolist(),
            "test_indices_within_locked_test_split": test_idx.tolist(),
        },
        "feature_map": fm_report.to_dict(),
        "backend": backend.capabilities(),
        "kernel_matrices": {
            "K_train_train": {"diagnostics": diag_train_train.to_dict()},
            "K_test_train": {"diagnostics": diag_test_train.to_dict()},
            "K_test_test": {"diagnostics": diag_test_test.to_dict()},
        },
        "reproducibility": {
            "K_train_train_bitwise_identical_on_recompute": reproducible,
            "max_abs_difference_on_recompute": max_reproducibility_diff,
        },
        "leakage_check": {
            "K_train_train_unaffected_by_subsequently_computing_test_kernels": unaffected_by_test_computation,
            "no_shared_rows_between_train_and_test_subsamples": no_shared_rows_between_subsamples,
            "kernel_functions_accept_no_label_argument": True,
        },
        "budget_warnings": warnings,
        "software_versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "qiskit": qiskit.__version__,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "sanity_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    np.save(RESULTS_DIR / "K_train_train.npy", K_train_train)
    np.save(RESULTS_DIR / "K_test_train.npy", K_test_train)
    np.save(RESULTS_DIR / "K_test_test.npy", K_test_test)

    _print_report(report)
    return report


def _print_report(report: dict) -> None:
    print("=" * 70)
    print("QuantumDx Phase 4 -- Quantum Pipeline Sanity Experiment")
    print("(NOT a benchmark -- infrastructure validation only)")
    print("=" * 70)
    ds = report["dataset"]
    sub = report["sanity_subsample"]
    fm = report["feature_map"]
    print(f"Split ID (matches Phase 2/3): {ds['split_id']}")
    print(f"Full locked split: {ds['n_train_full']} train / {ds['n_test_full']} test")
    print(f"Quantum features (= qubits): {ds['n_quantum_features']}")
    print(f"Quantum range: {ds['quantum_range']}")
    print()
    print(f"Sanity subsample: {sub['n_train_sampled']} train, {sub['n_test_sampled']} test (seed={sub['seed']})")
    print()
    print(f"Feature map: {fm['name']} | qubits={fm['n_qubits']} | reps={fm['reps']} "
          f"| entanglement={fm['entanglement']} | parameters={fm['num_parameters']} "
          f"| logical depth={fm['depth_logical']}")
    print(f"Gate counts: {fm['gate_counts']}")
    print()
    print(f"Backend: {report['backend']}")
    print()
    for name, entry in report["kernel_matrices"].items():
        d = entry["diagnostics"]
        print(f"{name}: shape={tuple(d['shape'])} min={d['min_value']:.6f} max={d['max_value']:.6f} "
              f"mean={d['mean_value']:.6f} symmetric={d['is_symmetric']} "
              f"diag_dev_from_1={d['diagonal_max_abs_deviation_from_one']}")
    print()
    r = report["reproducibility"]
    print(f"Reproducible on recompute: {r['K_train_train_bitwise_identical_on_recompute']} "
          f"(max diff={r['max_abs_difference_on_recompute']:.2e})")
    lk = report["leakage_check"]
    print(f"Train kernel unaffected by computing test kernels: "
          f"{lk['K_train_train_unaffected_by_subsequently_computing_test_kernels']}")
    if report["budget_warnings"]:
        print("\nWarnings:")
        for w in report["budget_warnings"]:
            print(f"  - {w}")
    print("=" * 70)


if __name__ == "__main__":
    run_sanity_experiment()
