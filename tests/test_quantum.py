"""Tests for src.quantum (Phase 4: quantum data pipeline, feature maps,
backend abstraction, fidelity kernel; Phase 5: full-scale QSVM experiment).

Phase 4 tests use the exact StatevectorBackend at small qubit counts (2-4)
and tiny sample counts -- fast, fully deterministic, no live IBM hardware
anywhere in this file. Phase 5 tests additionally exercise the REAL,
full-scale (242/61) locked dataset directly, because the full kernel
computation measured under one second (see docs/QUANTUM_QSVM.md Section 7)
-- there is no need to synthesize a smaller stand-in for speed.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest
from qiskit.circuit import QuantumCircuit

from src.quantum.backends import (
    IBMHardwareBackend,
    NoisySimulatorBackend,
    QuantumBackend,
    StatevectorBackend,
    get_backend,
)
from src.quantum.config import (
    MAX_QUBITS,
    QuantumConfig,
    QuantumConfigError,
    check_qubit_budget,
    check_reps_budget,
    check_sample_budget,
)
from src.quantum.data_contract import load_quantum_dataset
from src.quantum.feature_maps import build_feature_map, describe_feature_map
from src.quantum.kernel import (
    compute_kernel_matrix,
    compute_kernel_matrix_cached,
    compute_statevectors,
    kernel_diagnostics,
    kernel_matrix_from_statevectors,
    kernel_matrix_from_statevectors_blockwise,
    kernel_matrix_from_statevectors_vectorized,
)
from src.quantum.qsvm import DEFAULT_C_GRID, build_qsvm, run_qsvm_grid_search
from src.quantum.qsvm_experiment import compute_or_load_kernels, run_qsvm_experiment
from src.quantum.sanity_experiment import run_sanity_experiment
from src.preprocessing.config import DEFAULT_CONFIG as DEFAULT_PREPROCESSING_CONFIG

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def quantum_dataset():
    return load_quantum_dataset(DEFAULT_PREPROCESSING_CONFIG)


@pytest.fixture
def backend() -> StatevectorBackend:
    return StatevectorBackend()


# --------------------------------------------------------------------------
# 1 & 2. Quantum feature transformation / correct output dimensions
# --------------------------------------------------------------------------


def test_quantum_dataset_dimensions_match_preprocessing_config(quantum_dataset) -> None:
    assert quantum_dataset.X_train_quantum.shape[1] == DEFAULT_PREPROCESSING_CONFIG.pca_n_components
    assert quantum_dataset.X_test_quantum.shape[1] == DEFAULT_PREPROCESSING_CONFIG.pca_n_components
    assert quantum_dataset.n_qubits == DEFAULT_PREPROCESSING_CONFIG.pca_n_components


def test_quantum_dataset_row_counts_match_locked_split(quantum_dataset) -> None:
    assert len(quantum_dataset.X_train_quantum) == len(quantum_dataset.y_train)
    assert len(quantum_dataset.X_test_quantum) == len(quantum_dataset.y_test)
    assert len(quantum_dataset.X_train_quantum) + len(quantum_dataset.X_test_quantum) == 303


def test_quantum_dataset_values_within_declared_range(quantum_dataset) -> None:
    low, high = quantum_dataset.quantum_range
    assert quantum_dataset.X_train_quantum.min() >= low - 1e-9
    assert quantum_dataset.X_train_quantum.max() <= high + 1e-9
    assert quantum_dataset.X_test_quantum.min() >= low - 1e-9
    assert quantum_dataset.X_test_quantum.max() <= high + 1e-9


def test_quantum_dataset_labels_are_binary(quantum_dataset) -> None:
    assert set(np.unique(quantum_dataset.y_train).tolist()).issubset({0, 1})
    assert set(np.unique(quantum_dataset.y_test).tolist()).issubset({0, 1})


# --------------------------------------------------------------------------
# 3. Train/test separation (reuses the LOCKED Phase 2 split; no new split)
# --------------------------------------------------------------------------


def test_quantum_dataset_reuses_phase2_split_id(quantum_dataset) -> None:
    # This is the split_id already reported and used throughout Phase 2/3
    # (docs/PREPROCESSING.md, docs/CLASSICAL_BASELINE.md) -- reproduced
    # here, not re-derived, proving no new split was created.
    assert quantum_dataset.split_id == "e471025b07519a64"


def test_load_quantum_dataset_does_not_create_a_new_split() -> None:
    ds_a = load_quantum_dataset(DEFAULT_PREPROCESSING_CONFIG)
    ds_b = load_quantum_dataset(DEFAULT_PREPROCESSING_CONFIG)
    assert ds_a.split_id == ds_b.split_id
    np.testing.assert_array_equal(ds_a.X_train_quantum, ds_b.X_train_quantum)
    np.testing.assert_array_equal(ds_a.X_test_quantum, ds_b.X_test_quantum)
    np.testing.assert_array_equal(ds_a.y_train, ds_b.y_train)
    np.testing.assert_array_equal(ds_a.y_test, ds_b.y_test)


def test_train_and_test_quantum_arrays_share_no_rows(quantum_dataset) -> None:
    for train_row in quantum_dataset.X_train_quantum[:30]:  # a sample, full O(n^2) check is unnecessary
        assert not np.any(np.all(np.isclose(train_row, quantum_dataset.X_test_quantum), axis=1))


# --------------------------------------------------------------------------
# 4. Deterministic preprocessing (reused from Phase 2 -- reconfirmed here
#    in the context of the quantum contract specifically)
# --------------------------------------------------------------------------


def test_quantum_dataset_deterministic_given_same_config() -> None:
    ds_a = load_quantum_dataset(DEFAULT_PREPROCESSING_CONFIG)
    ds_b = load_quantum_dataset(DEFAULT_PREPROCESSING_CONFIG)
    assert ds_a.preprocess_hash == ds_b.preprocess_hash


def test_quantum_dataset_hash_changes_with_different_config() -> None:
    ds_full = load_quantum_dataset(DEFAULT_PREPROCESSING_CONFIG.with_overrides(feature_set="full"))
    ds_screening = load_quantum_dataset(DEFAULT_PREPROCESSING_CONFIG.with_overrides(feature_set="screening"))
    assert ds_full.preprocess_hash != ds_screening.preprocess_hash


# --------------------------------------------------------------------------
# 5, 6 & 7. Feature-map construction / number of qubits / circuit construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["zz_feature_map", "z_feature_map", "pauli_feature_map"])
def test_build_feature_map_all_types(name: str) -> None:
    qc = build_feature_map(name, n_qubits=3, reps=2, entanglement="linear")
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 3
    assert qc.num_parameters == 3  # one free parameter per qubit/feature


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 6])
def test_build_feature_map_qubit_count_matches_request(n_qubits: int) -> None:
    qc = build_feature_map("zz_feature_map", n_qubits=n_qubits, reps=1, entanglement="linear")
    assert qc.num_qubits == n_qubits
    assert qc.num_parameters == n_qubits


def test_build_feature_map_invalid_name_raises() -> None:
    with pytest.raises(QuantumConfigError):
        build_feature_map("not_a_real_feature_map", n_qubits=4)


def test_build_feature_map_invalid_entanglement_raises() -> None:
    with pytest.raises(QuantumConfigError):
        build_feature_map("zz_feature_map", n_qubits=4, entanglement="not_a_pattern")


def test_feature_map_binds_to_a_concrete_data_row() -> None:
    qc = build_feature_map("zz_feature_map", n_qubits=4, reps=2, entanglement="linear")
    row = np.array([0.1, 1.0, 2.0, 3.0])
    bound = qc.assign_parameters(row)
    assert bound.num_parameters == 0  # fully bound, ready to simulate


def test_describe_feature_map_report_fields() -> None:
    qc = build_feature_map("zz_feature_map", n_qubits=4, reps=2, entanglement="linear")
    report = describe_feature_map(qc, name="zz_feature_map", reps=2, entanglement="linear")
    assert report.n_qubits == 4
    assert report.num_parameters == 4
    assert report.depth_logical > 0
    assert sum(report.gate_counts.values()) > 0


def test_more_reps_increases_or_maintains_depth() -> None:
    shallow = build_feature_map("zz_feature_map", n_qubits=4, reps=1, entanglement="linear")
    deep = build_feature_map("zz_feature_map", n_qubits=4, reps=3, entanglement="linear")
    assert deep.decompose().depth() > shallow.decompose().depth()


# --------------------------------------------------------------------------
# 8, 9, 10 & 11. Kernel matrix shape / symmetry / diagonal / value range
# --------------------------------------------------------------------------


def test_kernel_matrix_shape_symmetric(backend) -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=2, reps=1, entanglement="linear")
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    K = compute_kernel_matrix(X, X, fm, backend, symmetric=True)
    assert K.shape == (3, 3)


def test_kernel_matrix_shape_asymmetric(backend) -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=2, reps=1, entanglement="linear")
    X_a = np.array([[0.1, 0.2], [0.3, 0.4]])
    X_b = np.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])
    K = compute_kernel_matrix(X_a, X_b, fm, backend, symmetric=False)
    assert K.shape == (2, 3)


def test_kernel_matrix_is_symmetric_when_expected(backend) -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=3, reps=2, entanglement="full")
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(5, 3))
    K = compute_kernel_matrix(X, X, fm, backend, symmetric=True)
    np.testing.assert_allclose(K, K.T, atol=1e-10)


def test_kernel_diagonal_approximately_one(backend) -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=4, reps=2, entanglement="linear")
    rng = np.random.RandomState(1)
    X = rng.uniform(0, np.pi, size=(6, 4))
    K = compute_kernel_matrix(X, X, fm, backend, symmetric=True)
    diag = np.diag(K)
    np.testing.assert_allclose(diag, 1.0, atol=1e-8)


def test_kernel_values_within_valid_fidelity_range(backend) -> None:
    fm = build_feature_map("pauli_feature_map", n_qubits=3, reps=2, entanglement="linear", paulis=("Z", "ZZ"))
    rng = np.random.RandomState(2)
    X_a = rng.uniform(0, np.pi, size=(8, 3))
    X_b = rng.uniform(0, np.pi, size=(6, 3))
    K = compute_kernel_matrix(X_a, X_b, fm, backend, symmetric=False)
    assert K.min() >= -1e-9
    assert K.max() <= 1.0 + 1e-9


def test_kernel_diagnostics_reports_correct_summary(backend) -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=3, reps=1, entanglement="linear")
    rng = np.random.RandomState(3)
    X = rng.uniform(0, np.pi, size=(5, 3))
    K = compute_kernel_matrix(X, X, fm, backend, symmetric=True)
    diag = kernel_diagnostics(K, symmetric=True)
    assert diag.shape == (5, 5)
    assert diag.is_square is True
    assert diag.is_symmetric is True
    assert diag.diagonal_max_abs_deviation_from_one < 1e-8
    assert diag.within_valid_fidelity_range is True


def test_kernel_matrix_from_statevectors_matches_direct_computation(backend) -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=2, reps=1, entanglement="linear")
    X = np.array([[0.1, 0.9], [1.5, 2.5], [3.0, 0.5]])
    svs = compute_statevectors(X, fm, backend)
    K_from_sv = kernel_matrix_from_statevectors(svs, svs, symmetric=True)
    K_direct = compute_kernel_matrix(X, X, fm, backend, symmetric=True)
    np.testing.assert_allclose(K_from_sv, K_direct, atol=1e-12)


def test_vectorized_kernel_matches_reference_symmetric(backend) -> None:
    """The Phase 6 vectorized kernel assembly (kernel_matrix_from_statevectors_
    vectorized) must compute the identical mathematical quantity as the
    Phase 4/5 reference (kernel_matrix_from_statevectors), for a symmetric
    (train-train-style) matrix -- verified here on a synthetic example, and
    on the real Stage A (n=1000) data during development (max abs diff
    ~1.2e-15, see docs/LARGE_DATASET.md).
    """
    fm = build_feature_map("zz_feature_map", n_qubits=3, reps=2, entanglement="linear")
    rng = np.random.RandomState(7)
    X = rng.uniform(0, np.pi, size=(25, 3))
    svs = compute_statevectors(X, fm, backend)

    K_reference = kernel_matrix_from_statevectors(svs, svs, symmetric=True)
    K_vectorized = kernel_matrix_from_statevectors_vectorized(svs, svs, symmetric=True)

    np.testing.assert_allclose(K_reference, K_vectorized, atol=1e-10, rtol=1e-8)
    assert np.allclose(K_vectorized, K_vectorized.T, atol=1e-10)  # still symmetric
    assert np.max(np.abs(np.diag(K_vectorized) - 1.0)) < 1e-10  # still ~1 on the diagonal


def test_vectorized_kernel_matches_reference_asymmetric(backend) -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=3, reps=2, entanglement="linear")
    rng = np.random.RandomState(11)
    X_train = rng.uniform(0, np.pi, size=(20, 3))
    X_test = rng.uniform(0, np.pi, size=(8, 3))
    train_svs = compute_statevectors(X_train, fm, backend)
    test_svs = compute_statevectors(X_test, fm, backend)

    K_reference = kernel_matrix_from_statevectors(test_svs, train_svs, symmetric=False)
    K_vectorized = kernel_matrix_from_statevectors_vectorized(test_svs, train_svs, symmetric=False)

    assert K_reference.shape == K_vectorized.shape == (8, 20)
    np.testing.assert_allclose(K_reference, K_vectorized, atol=1e-10, rtol=1e-8)


def test_vectorized_kernel_rejects_mismatched_symmetric_lengths(backend) -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=2, reps=1, entanglement="linear")
    svs_a = compute_statevectors(np.zeros((3, 2)), fm, backend)
    svs_b = compute_statevectors(np.zeros((5, 2)), fm, backend)
    with pytest.raises(ValueError):
        kernel_matrix_from_statevectors_vectorized(svs_a, svs_b, symmetric=True)


def test_blockwise_kernel_matches_reference_symmetric_various_block_sizes(backend) -> None:
    """The Stage D memory-safe blockwise kernel assembly
    (kernel_matrix_from_statevectors_blockwise) must compute the identical
    quantity as the Phase 4/5 reference, regardless of block_size -- a
    block_size that does not evenly divide n, one larger than n (single
    block, degenerating to the whole-matrix case), and block_size=1 (the
    most fragmented case) are all exercised.
    """
    fm = build_feature_map("zz_feature_map", n_qubits=3, reps=2, entanglement="linear")
    rng = np.random.RandomState(13)
    X = rng.uniform(0, np.pi, size=(23, 3))
    svs = compute_statevectors(X, fm, backend)
    K_reference = kernel_matrix_from_statevectors(svs, svs, symmetric=True)

    for block_size in [1, 4, 7, 1000]:
        K_block = kernel_matrix_from_statevectors_blockwise(svs, svs, symmetric=True, block_size=block_size)
        np.testing.assert_allclose(K_reference, K_block, atol=1e-10, rtol=1e-8)


def test_blockwise_kernel_matches_reference_asymmetric(backend) -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=3, reps=2, entanglement="linear")
    rng = np.random.RandomState(17)
    X_train = rng.uniform(0, np.pi, size=(19, 3))
    X_test = rng.uniform(0, np.pi, size=(6, 3))
    train_svs = compute_statevectors(X_train, fm, backend)
    test_svs = compute_statevectors(X_test, fm, backend)

    K_reference = kernel_matrix_from_statevectors(test_svs, train_svs, symmetric=False)
    K_block = kernel_matrix_from_statevectors_blockwise(test_svs, train_svs, symmetric=False, block_size=2)

    assert K_reference.shape == K_block.shape == (6, 19)
    np.testing.assert_allclose(K_reference, K_block, atol=1e-10, rtol=1e-8)


def test_blockwise_kernel_matches_vectorized_kernel(backend) -> None:
    """Cross-check against the already-verified whole-matrix vectorized
    implementation, not just the loop reference -- both optimizations must
    agree with each other, not merely each independently with the loop."""
    fm = build_feature_map("zz_feature_map", n_qubits=4, reps=2, entanglement="linear")
    rng = np.random.RandomState(19)
    X = rng.uniform(0, np.pi, size=(30, 4))
    svs = compute_statevectors(X, fm, backend)

    K_vectorized = kernel_matrix_from_statevectors_vectorized(svs, svs, symmetric=True)
    K_block = kernel_matrix_from_statevectors_blockwise(svs, svs, symmetric=True, block_size=9)
    np.testing.assert_allclose(K_vectorized, K_block, atol=1e-12, rtol=1e-10)


def test_blockwise_kernel_float32_matches_float64_within_tolerance(backend) -> None:
    """Stage D's memory optimization stores the kernel as float32 (halving
    resident memory for the (20000, 20000) matrix). Fidelity values lie in
    [0, 1]; float32 must reproduce float64 to well within SVM-relevant
    precision (checked here at 1e-6, far tighter than anything that could
    change a GridSearchCV C-selection or a predicted probability)."""
    fm = build_feature_map("zz_feature_map", n_qubits=3, reps=2, entanglement="linear")
    rng = np.random.RandomState(23)
    X = rng.uniform(0, np.pi, size=(15, 3))
    svs = compute_statevectors(X, fm, backend)

    K_64 = kernel_matrix_from_statevectors_blockwise(svs, svs, symmetric=True, block_size=4, dtype=np.float64)
    K_32 = kernel_matrix_from_statevectors_blockwise(svs, svs, symmetric=True, block_size=4, dtype=np.float32)

    assert K_32.dtype == np.float32
    np.testing.assert_allclose(K_64, K_32.astype(np.float64), atol=1e-6)


def test_blockwise_kernel_rejects_mismatched_symmetric_lengths(backend) -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=2, reps=1, entanglement="linear")
    svs_a = compute_statevectors(np.zeros((3, 2)), fm, backend)
    svs_b = compute_statevectors(np.zeros((5, 2)), fm, backend)
    with pytest.raises(ValueError):
        kernel_matrix_from_statevectors_blockwise(svs_a, svs_b, symmetric=True)


def test_blockwise_kernel_rejects_invalid_block_size(backend) -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=2, reps=1, entanglement="linear")
    svs = compute_statevectors(np.zeros((3, 2)), fm, backend)
    with pytest.raises(ValueError):
        kernel_matrix_from_statevectors_blockwise(svs, svs, symmetric=True, block_size=0)


def test_reference_kernel_implementation_is_unmodified_by_vectorized_addition(backend) -> None:
    """Guards against the vectorized function ever being merged INTO the
    reference implementation -- Phase 4/5 code must keep calling the
    original, untouched loop-based function."""
    fm = build_feature_map("zz_feature_map", n_qubits=2, reps=1, entanglement="linear")
    X = np.array([[0.1, 0.9], [1.5, 2.5]])
    svs = compute_statevectors(X, fm, backend)
    K = kernel_matrix_from_statevectors(svs, svs, symmetric=True)
    assert K.shape == (2, 2)
    assert abs(K[0, 0] - 1.0) < 1e-9
    assert abs(K[1, 1] - 1.0) < 1e-9


def test_compute_statevectors_rejects_wrong_qubit_count(backend) -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=4, reps=1, entanglement="linear")
    X_wrong = np.zeros((3, 2))  # feature_map expects 4 columns, not 2
    with pytest.raises(ValueError):
        compute_statevectors(X_wrong, fm, backend)


# --------------------------------------------------------------------------
# 12. Repeated execution reproducibility
# --------------------------------------------------------------------------


def test_kernel_computation_is_bitwise_reproducible(backend) -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=4, reps=2, entanglement="linear")
    rng = np.random.RandomState(4)
    X = rng.uniform(0, np.pi, size=(10, 4))
    K1 = compute_kernel_matrix(X, X, fm, backend, symmetric=True)
    K2 = compute_kernel_matrix(X, X, fm, backend, symmetric=True)
    np.testing.assert_array_equal(K1, K2)


def test_kernel_cache_hit_on_second_call(tmp_path, backend) -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=2, reps=1, entanglement="linear")
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    fm_config = {"name": "zz_feature_map", "reps": 1, "entanglement": "linear"}

    K1, hit1 = compute_kernel_matrix_cached(X, X, fm, fm_config, backend, tmp_path, symmetric=True)
    K2, hit2 = compute_kernel_matrix_cached(X, X, fm, fm_config, backend, tmp_path, symmetric=True)

    assert hit1 is False
    assert hit2 is True
    np.testing.assert_array_equal(K1, K2)


def test_kernel_cache_miss_on_different_config(tmp_path, backend) -> None:
    fm1 = build_feature_map("zz_feature_map", n_qubits=2, reps=1, entanglement="linear")
    fm2 = build_feature_map("zz_feature_map", n_qubits=2, reps=2, entanglement="linear")
    X = np.array([[0.1, 0.2], [0.3, 0.4]])

    _, hit1 = compute_kernel_matrix_cached(
        X, X, fm1, {"reps": 1}, backend, tmp_path, symmetric=True
    )
    _, hit2 = compute_kernel_matrix_cached(
        X, X, fm2, {"reps": 2}, backend, tmp_path, symmetric=True
    )
    assert hit1 is False
    assert hit2 is False  # different config -> different cache key -> no false hit


# --------------------------------------------------------------------------
# 13. Backend configuration
# --------------------------------------------------------------------------


def test_get_backend_statevector_returns_implemented_backend() -> None:
    backend = get_backend("statevector")
    assert isinstance(backend, StatevectorBackend)
    assert backend.capabilities()["implemented"] is True


def test_get_backend_noisy_simulator_is_still_an_unimplemented_placeholder() -> None:
    """NoisySimulatorBackend remains Phase 5's deferred scope, untouched by
    Phase 16 (which only completed IBMHardwareBackend)."""
    noisy = get_backend("noisy_simulator")
    assert isinstance(noisy, NoisySimulatorBackend)
    assert noisy.capabilities()["implemented"] is False


def test_get_backend_ibm_hardware_is_now_implemented() -> None:
    """As of Phase 16, IBMHardwareBackend is a real, working implementation
    (see src.large_dataset.phase16_ibm_hardware) -- no longer the Phase 4/6
    placeholder this test used to assert. `implemented=True` describes the
    CODE path, independent of whether credentials happen to be configured
    in this environment (see capabilities()['credentials_configured'])."""
    ibm = get_backend("ibm_hardware")
    assert isinstance(ibm, IBMHardwareBackend)
    assert ibm.capabilities()["implemented"] is True


def test_placeholder_backends_raise_not_implemented_not_silently_pass() -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=2, reps=1, entanglement="linear")
    bound = fm.assign_parameters([0.1, 0.2])
    with pytest.raises(NotImplementedError):
        NoisySimulatorBackend().compute_statevector(bound)
    with pytest.raises(NotImplementedError):
        IBMHardwareBackend().compute_statevector(bound)


def test_get_backend_invalid_name_raises() -> None:
    with pytest.raises(QuantumConfigError):
        get_backend("not_a_real_backend")


def test_backend_selection_is_purely_config_driven() -> None:
    """Changing QuantumConfig.backend_name changes which backend get_backend()
    returns, without any other code path branching on the name -- this is
    the property that lets a later phase add a real backend without
    touching kernel.py or sanity_experiment.py.
    """
    cfg_sim = QuantumConfig(backend_name="statevector")
    cfg_ibm = QuantumConfig(backend_name="ibm_hardware")
    assert type(get_backend(cfg_sim.backend_name)) is StatevectorBackend
    assert type(get_backend(cfg_ibm.backend_name)) is IBMHardwareBackend


def test_statevector_backend_rejects_unbound_circuit(backend) -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=2, reps=1, entanglement="linear")
    with pytest.raises(QuantumConfigError):
        backend.compute_statevector(fm)  # still has free parameters


# --------------------------------------------------------------------------
# 14. Failure handling for invalid quantum configuration
# --------------------------------------------------------------------------


def test_quantum_config_rejects_invalid_backend() -> None:
    with pytest.raises(QuantumConfigError):
        QuantumConfig(backend_name="quantum_magic")  # type: ignore[arg-type]


def test_quantum_config_rejects_invalid_feature_map() -> None:
    with pytest.raises(QuantumConfigError):
        QuantumConfig(feature_map_name="not_real")  # type: ignore[arg-type]


def test_quantum_config_rejects_invalid_entanglement() -> None:
    with pytest.raises(QuantumConfigError):
        QuantumConfig(entanglement="diagonal")  # type: ignore[arg-type]


def test_quantum_config_rejects_reps_out_of_range() -> None:
    with pytest.raises(QuantumConfigError):
        QuantumConfig(reps=0)
    with pytest.raises(QuantumConfigError):
        QuantumConfig(reps=100)


def test_quantum_config_rejects_empty_paulis() -> None:
    with pytest.raises(QuantumConfigError):
        QuantumConfig(paulis=())


def test_check_qubit_budget_raises_above_hard_ceiling() -> None:
    with pytest.raises(QuantumConfigError):
        check_qubit_budget(MAX_QUBITS + 1)


def test_check_qubit_budget_warns_but_does_not_raise_in_soft_zone() -> None:
    warnings = check_qubit_budget(9)  # above WARN_QUBITS_ABOVE=8, below MAX_QUBITS=10
    assert len(warnings) == 1


def test_check_qubit_budget_silent_within_budget() -> None:
    assert check_qubit_budget(4) == []


def test_check_reps_budget_warns_above_threshold() -> None:
    assert check_reps_budget(3) != []
    assert check_reps_budget(2) == []


def test_check_sample_budget_warns_above_max_train_samples() -> None:
    cfg = QuantumConfig(max_train_samples=50)
    assert check_sample_budget(100, cfg) != []
    assert check_sample_budget(10, cfg) == []


def test_config_with_overrides_does_not_mutate_original() -> None:
    original = QuantumConfig.default()
    modified = original.with_overrides(reps=3)
    assert original.reps == 2
    assert modified.reps == 3


def test_config_from_yaml_round_trip(tmp_path) -> None:
    yaml_path = tmp_path / "q.yaml"
    yaml_path.write_text("backend_name: statevector\nreps: 3\nentanglement: circular\n", encoding="utf-8")
    cfg = QuantumConfig.from_yaml(yaml_path)
    assert cfg.reps == 3
    assert cfg.entanglement == "circular"
    assert cfg.feature_map_name == QuantumConfig.default().feature_map_name  # falls back to default


# --------------------------------------------------------------------------
# Structural leakage guards (kernel functions never see labels)
# --------------------------------------------------------------------------


def test_kernel_functions_accept_no_label_parameter() -> None:
    for func in (compute_statevectors, compute_kernel_matrix, kernel_matrix_from_statevectors):
        params = [p.lower() for p in inspect.signature(func).parameters]
        assert not any(p in ("y", "y_train", "y_test", "labels") for p in params), (
            f"{func.__name__} must not accept a label parameter; found {params}"
        )


# --------------------------------------------------------------------------
# Sanity experiment (small, fast end-to-end run)
# --------------------------------------------------------------------------


def test_sanity_experiment_runs_end_to_end(tmp_path, monkeypatch) -> None:
    import src.quantum.sanity_experiment as sanity_module

    monkeypatch.setattr(sanity_module, "RESULTS_DIR", tmp_path)
    report = run_sanity_experiment(n_train=6, n_test=4)

    assert report["sanity_subsample"]["n_train_sampled"] == 6
    assert report["sanity_subsample"]["n_test_sampled"] == 4
    assert report["kernel_matrices"]["K_train_train"]["diagnostics"]["shape"] == [6, 6]
    assert report["kernel_matrices"]["K_test_train"]["diagnostics"]["shape"] == [4, 6]
    assert report["kernel_matrices"]["K_test_test"]["diagnostics"]["shape"] == [4, 4]
    assert report["reproducibility"]["K_train_train_bitwise_identical_on_recompute"] is True
    assert report["leakage_check"]["no_shared_rows_between_train_and_test_subsamples"] is True
    assert (tmp_path / "sanity_report.json").is_file()


def test_sanity_experiment_reports_split_id_matching_phase2(tmp_path, monkeypatch) -> None:
    import src.quantum.sanity_experiment as sanity_module

    monkeypatch.setattr(sanity_module, "RESULTS_DIR", tmp_path)
    report = run_sanity_experiment(n_train=5, n_test=3)
    assert report["dataset"]["split_id"] == "e471025b07519a64"


# ==========================================================================
# Phase 5: full-scale QSVM experiment
# ==========================================================================

FULL_TRAIN_N = 242
FULL_TEST_N = 61


@pytest.fixture(scope="module")
def full_kernels(quantum_dataset):
    """The REAL, full-scale (242/61) kernel matrices, computed once and
    shared across this module's tests -- computing them takes well under a
    second (see docs/QUANTUM_QSVM.md Section 7), so there is no need to
    synthesize a smaller stand-in.
    """
    fm = build_feature_map("zz_feature_map", n_qubits=quantum_dataset.n_qubits, reps=2, entanglement="linear")
    backend = get_backend("statevector")
    train_svs = compute_statevectors(quantum_dataset.X_train_quantum, fm, backend)
    test_svs = compute_statevectors(quantum_dataset.X_test_quantum, fm, backend)
    K_train_train = kernel_matrix_from_statevectors(train_svs, train_svs, symmetric=True)
    K_test_train = kernel_matrix_from_statevectors(test_svs, train_svs, symmetric=False)
    return K_train_train, K_test_train


# --------------------------------------------------------------------------
# Correct train/test kernel shapes (full scale)
# --------------------------------------------------------------------------


def test_full_scale_train_kernel_shape(full_kernels) -> None:
    K_train_train, _ = full_kernels
    assert K_train_train.shape == (FULL_TRAIN_N, FULL_TRAIN_N)


def test_full_scale_test_kernel_shape(full_kernels) -> None:
    _, K_test_train = full_kernels
    assert K_test_train.shape == (FULL_TEST_N, FULL_TRAIN_N)


def test_full_scale_train_kernel_symmetric(full_kernels) -> None:
    K_train_train, _ = full_kernels
    np.testing.assert_allclose(K_train_train, K_train_train.T, atol=1e-10)


def test_full_scale_train_kernel_diagonal_near_one(full_kernels) -> None:
    K_train_train, _ = full_kernels
    diag = np.diag(K_train_train)
    np.testing.assert_allclose(diag, 1.0, atol=1e-8)


def test_full_scale_kernel_values_within_valid_range(full_kernels) -> None:
    K_train_train, K_test_train = full_kernels
    for K in (K_train_train, K_test_train):
        assert K.min() >= -1e-9
        assert K.max() <= 1.0 + 1e-9


def test_full_scale_diagnostics_report_agrees(full_kernels) -> None:
    K_train_train, K_test_train = full_kernels
    diag_tt = kernel_diagnostics(K_train_train, symmetric=True)
    diag_te = kernel_diagnostics(K_test_train, symmetric=False)
    assert diag_tt.shape == (242, 242)
    assert diag_te.shape == (61, 242)
    assert diag_tt.is_symmetric is True
    assert diag_tt.diagonal_max_abs_deviation_from_one < 1e-6
    assert diag_tt.within_valid_fidelity_range is True
    assert diag_te.within_valid_fidelity_range is True


# --------------------------------------------------------------------------
# compute_or_load_kernels: efficiency (no redundant statevector recompute)
# and restartable disk caching
# --------------------------------------------------------------------------


def test_compute_or_load_kernels_correct_shapes(tmp_path, quantum_dataset) -> None:
    cfg = QuantumConfig.default()
    K_tt, K_te, timing, cache_hit = compute_or_load_kernels(quantum_dataset, cfg, tmp_path)
    assert K_tt.shape == (FULL_TRAIN_N, FULL_TRAIN_N)
    assert K_te.shape == (FULL_TEST_N, FULL_TRAIN_N)
    assert cache_hit is False
    assert timing["n_train_statevectors"] == FULL_TRAIN_N
    assert timing["n_test_statevectors"] == FULL_TEST_N


def test_compute_or_load_kernels_restartable_cache_hit(tmp_path, quantum_dataset) -> None:
    cfg = QuantumConfig.default()
    K_tt_first, K_te_first, _, hit_first = compute_or_load_kernels(quantum_dataset, cfg, tmp_path)
    K_tt_second, K_te_second, timing_second, hit_second = compute_or_load_kernels(quantum_dataset, cfg, tmp_path)

    assert hit_first is False
    assert hit_second is True
    assert timing_second == {"cache_hit": True}
    np.testing.assert_array_equal(K_tt_first, K_tt_second)
    np.testing.assert_array_equal(K_te_first, K_te_second)


def test_compute_or_load_kernels_cache_miss_on_different_config(tmp_path, quantum_dataset) -> None:
    cfg_a = QuantumConfig.default()
    cfg_b = QuantumConfig.default().with_overrides(reps=3)
    _, _, _, hit_a = compute_or_load_kernels(quantum_dataset, cfg_a, tmp_path)
    _, _, _, hit_b = compute_or_load_kernels(quantum_dataset, cfg_b, tmp_path)
    assert hit_a is False
    assert hit_b is False  # different reps -> different cache tag -> no false hit


# --------------------------------------------------------------------------
# QSVM: correct SVC precomputed-kernel input format
# --------------------------------------------------------------------------


def test_build_qsvm_uses_precomputed_kernel() -> None:
    svc = build_qsvm(random_seed=42)
    assert svc.kernel == "precomputed"
    assert svc.probability is True


def test_build_qsvm_with_fixed_c() -> None:
    svc = build_qsvm(random_seed=42, C=10.0)
    assert svc.C == 10.0


def test_qsvm_rejects_non_square_matrix_as_train_kernel(full_kernels) -> None:
    _, K_test_train = full_kernels  # (61, 242) -- not a valid TRAINING kernel (must be square)
    svc = build_qsvm(random_seed=42, C=1.0)
    y_wrong = np.zeros(61, dtype=int)
    y_wrong[:30] = 1
    with pytest.raises(Exception):
        svc.fit(K_test_train, y_wrong)  # sklearn itself rejects a non-square "Gram" matrix


# --------------------------------------------------------------------------
# No train/test leakage in model-selection logic (structural + behavioral)
# --------------------------------------------------------------------------


def test_run_qsvm_grid_search_has_no_test_kernel_parameter() -> None:
    sig = inspect.signature(run_qsvm_grid_search)
    param_names = [p.lower() for p in sig.parameters]
    assert not any("test" in name for name in param_names), (
        f"run_qsvm_grid_search must not accept any test-kernel parameter; found: {param_names}"
    )


def test_qsvm_cv_scores_unaffected_by_downstream_test_kernel_use(full_kernels, quantum_dataset) -> None:
    """Mirrors the Phase 3 CV-leakage test: computing/using K_test_train
    afterward must not change the CV scores already produced from
    K_train_train alone.
    """
    K_train_train, K_test_train = full_kernels

    search_a = run_qsvm_grid_search(K_train_train, quantum_dataset.y_train, C_grid=[0.1, 1.0], cv_folds=3, random_seed=42)
    _ = build_qsvm(42).fit(K_train_train, quantum_dataset.y_train).predict_proba(K_test_train)  # use test kernel
    search_b = run_qsvm_grid_search(K_train_train, quantum_dataset.y_train, C_grid=[0.1, 1.0], cv_folds=3, random_seed=42)

    np.testing.assert_array_equal(search_a.cv_results_["mean_test_roc_auc"], search_b.cv_results_["mean_test_roc_auc"])
    assert search_a.best_params_ == search_b.best_params_


def test_sklearn_precomputed_cv_slicing_matches_manual_fold_slicing(full_kernels, quantum_dataset) -> None:
    """The empirical verification underlying the design (see
    src.quantum.qsvm module docstring): sklearn's native handling of
    kernel='precomputed' inside cross-validation performs the exact same
    two-axis fold slicing as a manual per-fold recomputation would.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.svm import SVC

    K_train_train, _ = full_kernels
    y = quantum_dataset.y_train
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    auto_scores = cross_val_score(SVC(kernel="precomputed", C=1.0), K_train_train, y, cv=cv, scoring="accuracy")

    manual_scores = []
    for train_idx, val_idx in cv.split(K_train_train, y):
        K_fold_train = K_train_train[np.ix_(train_idx, train_idx)]
        K_fold_val = K_train_train[np.ix_(val_idx, train_idx)]
        m = SVC(kernel="precomputed", C=1.0)
        m.fit(K_fold_train, y[train_idx])
        manual_scores.append(m.score(K_fold_val, y[val_idx]))

    np.testing.assert_allclose(auto_scores, manual_scores)


# --------------------------------------------------------------------------
# Deterministic behavior
# --------------------------------------------------------------------------


def test_qsvm_grid_search_reproducible_with_same_seed(full_kernels, quantum_dataset) -> None:
    K_train_train, _ = full_kernels
    search_a = run_qsvm_grid_search(K_train_train, quantum_dataset.y_train, C_grid=DEFAULT_C_GRID, random_seed=42)
    search_b = run_qsvm_grid_search(K_train_train, quantum_dataset.y_train, C_grid=DEFAULT_C_GRID, random_seed=42)
    assert search_a.best_params_ == search_b.best_params_
    np.testing.assert_allclose(search_a.best_score_, search_b.best_score_)
    proba_a = search_a.best_estimator_.predict_proba(K_train_train)[:, 1]
    proba_b = search_b.best_estimator_.predict_proba(K_train_train)[:, 1]
    np.testing.assert_allclose(proba_a, proba_b)


def test_kernel_computation_reproducible_at_full_scale(quantum_dataset) -> None:
    fm = build_feature_map("zz_feature_map", n_qubits=quantum_dataset.n_qubits, reps=2, entanglement="linear")
    backend = get_backend("statevector")
    svs_a = compute_statevectors(quantum_dataset.X_train_quantum, fm, backend)
    svs_b = compute_statevectors(quantum_dataset.X_train_quantum, fm, backend)
    K_a = kernel_matrix_from_statevectors(svs_a, svs_a, symmetric=True)
    K_b = kernel_matrix_from_statevectors(svs_b, svs_b, symmetric=True)
    np.testing.assert_array_equal(K_a, K_b)


# --------------------------------------------------------------------------
# Metric calculation (via the reused Phase 3 evaluate_on_test)
# --------------------------------------------------------------------------


def test_qsvm_final_metrics_within_valid_ranges(full_kernels, quantum_dataset) -> None:
    from src.classical.evaluation import evaluate_on_test

    K_train_train, K_test_train = full_kernels
    search = run_qsvm_grid_search(K_train_train, quantum_dataset.y_train, C_grid=[0.1, 1.0, 10.0], random_seed=42)
    result = evaluate_on_test(
        search, "test_phase5", "qsvm", "QSVM", "quantum_ready", K_test_train, quantum_dataset.y_test,
        n_train=242, random_seed=42,
    )
    tm = result.test_metrics
    for key in ("accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc", "pr_auc"):
        assert 0.0 <= tm[key] <= 1.0, f"{key}={tm[key]} out of [0,1]"
    assert tm["tn"] + tm["fp"] + tm["fn"] + tm["tp"] == FULL_TEST_N


# --------------------------------------------------------------------------
# Configuration validation (Phase 5-specific)
# --------------------------------------------------------------------------


def test_default_c_grid_is_the_brief_example_grid() -> None:
    assert DEFAULT_C_GRID == [0.01, 0.1, 1.0, 10.0, 100.0]


def test_run_qsvm_grid_search_rejects_mismatched_kernel_and_label_length(full_kernels) -> None:
    K_train_train, _ = full_kernels
    y_wrong_length = np.zeros(10, dtype=int)
    with pytest.raises(Exception):
        run_qsvm_grid_search(K_train_train, y_wrong_length, C_grid=[1.0], random_seed=42)


# --------------------------------------------------------------------------
# Result serialization / loading
# --------------------------------------------------------------------------


def test_run_qsvm_experiment_end_to_end_and_artifacts_load_back(tmp_path, monkeypatch) -> None:
    import src.quantum.qsvm_experiment as exp_module

    results_dir = tmp_path / "phase5"
    kernel_cache_dir = results_dir / "kernels"
    figures_dir = results_dir / "figures"
    monkeypatch.setattr(exp_module, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(exp_module, "KERNEL_CACHE_DIR", kernel_cache_dir)
    monkeypatch.setattr(exp_module, "FIGURES_DIR", figures_dir)

    outcome = run_qsvm_experiment()

    assert outcome["dataset"].split_id == "e471025b07519a64"
    assert outcome["search"].best_estimator_ is not None

    # Every declared artifact exists and is loadable in its declared format.
    run_record_path = results_dir / "run_record.json"
    final_metrics_path = results_dir / "final_metrics.json"
    comparison_path = results_dir / "comparison_with_phase3.csv"
    cv_grid_path = results_dir / "cv_grid_results.csv"

    assert run_record_path.is_file()
    assert final_metrics_path.is_file()
    assert comparison_path.is_file()
    assert cv_grid_path.is_file()

    with open(run_record_path, encoding="utf-8") as fh:
        run_record = json.load(fh)
    assert run_record["split_id"] == "e471025b07519a64"
    assert run_record["n_train"] == FULL_TRAIN_N
    assert run_record["n_test"] == FULL_TEST_N
    assert run_record["status"] == "descriptive_comparison_only_no_significance_testing"

    with open(final_metrics_path, encoding="utf-8") as fh:
        final_metrics = json.load(fh)
    assert "roc_auc" in final_metrics["test_metrics"]

    comparison_df = pd.read_csv(comparison_path)
    assert "QSVM (fidelity kernel)" in comparison_df["Model"].values
    # Phase 3's four classical models must also be present (pure file read, not re-run)
    assert len(comparison_df) == 5

    assert (figures_dir / "qsvm_roc.png").is_file()
    assert (figures_dir / "qsvm_confusion.png").is_file()
    assert (figures_dir / "qsvm_cv_vs_C.png").is_file()
    assert (kernel_cache_dir).is_dir()
    assert len(list(kernel_cache_dir.glob("*.npy"))) == 2
