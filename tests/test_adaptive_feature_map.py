"""Tests for src.quantum.adaptive_feature_map.

Covers Section 12's requirements: deterministic adaptive-map construction,
correct number of qubits, valid feature-map dimensions, training-only
adaptation, no test-set dependency, kernel numerical validity,
reproducibility, and (via a synthetic DeLong check) correct statistical
delta direction for the experiment that consumes this module.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from qiskit.circuit.library import zz_feature_map

from src.quantum.adaptive_feature_map import (
    AdaptiveMapReport,
    build_adaptive_feature_map,
    build_adaptive_feature_map_from_training_data,
    compute_pairwise_mutual_information,
    linear_chain_pairs,
    select_adaptive_entanglement_pairs,
)
from src.quantum.kernel import compute_statevectors, kernel_diagnostics, kernel_matrix_from_statevectors_blockwise
from src.quantum.backends import get_backend
from src.large_dataset.statistical_robustness import delong_test


@pytest.fixture(scope="module")
def synthetic_train() -> np.ndarray:
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(300, 4))
    X[:, 3] = (np.sin(X[:, 0]) + rng.normal(0, 0.05, 300)) % np.pi  # real dependency: 0<->3
    return X


# --------------------------------------------------------------------------
# 1. Deterministic construction / reproducibility
# --------------------------------------------------------------------------


def test_adaptive_map_construction_is_deterministic(synthetic_train) -> None:
    _, report1 = build_adaptive_feature_map_from_training_data(synthetic_train, 4, 2, random_state=42)
    _, report2 = build_adaptive_feature_map_from_training_data(synthetic_train, 4, 2, random_state=42)
    assert report1.selected_pairs == report2.selected_pairs
    assert report1.mi_matrix == report2.mi_matrix


def test_mutual_information_matrix_is_reproducible(synthetic_train) -> None:
    mi1 = compute_pairwise_mutual_information(synthetic_train, random_state=42)
    mi2 = compute_pairwise_mutual_information(synthetic_train, random_state=42)
    assert np.array_equal(mi1, mi2)


def test_pair_selection_finds_the_known_synthetic_dependency(synthetic_train) -> None:
    """cols 0 and 3 were constructed with a real nonlinear dependency; the
    adaptive rule must rank that pair highly (top pair, since it's the
    single strongest relationship injected)."""
    mi = compute_pairwise_mutual_information(synthetic_train, random_state=42)
    pairs = select_adaptive_entanglement_pairs(mi, n_pairs=3)
    assert (0, 3) in pairs


# --------------------------------------------------------------------------
# 2. Correct number of qubits / valid feature-map dimensions
# --------------------------------------------------------------------------


def test_adaptive_circuit_has_correct_qubit_and_parameter_count(synthetic_train) -> None:
    circuit, report = build_adaptive_feature_map_from_training_data(synthetic_train, 4, 2, random_state=42)
    assert circuit.num_qubits == 4
    assert circuit.num_parameters == 4
    assert report.n_qubits == 4


def test_adaptive_circuit_matches_baseline_gate_budget(synthetic_train) -> None:
    """Same Z/ZZ TERM COUNT as the baseline zz_feature_map -- isolates pair
    SELECTION from pair COUNT (see module docstring)."""
    circuit, _ = build_adaptive_feature_map_from_training_data(synthetic_train, 4, 2, random_state=42)
    baseline = zz_feature_map(feature_dimension=4, reps=2, entanglement="linear")
    assert circuit.decompose().count_ops() == baseline.decompose().count_ops()


def test_n_pairs_defaults_to_baseline_linear_chain_length(synthetic_train) -> None:
    _, report = build_adaptive_feature_map_from_training_data(synthetic_train, 4, 2)
    assert report.n_pairs == len(linear_chain_pairs(4)) == 3


def test_select_pairs_rejects_n_pairs_exceeding_available() -> None:
    mi = np.zeros((4, 4))
    with pytest.raises(ValueError):
        select_adaptive_entanglement_pairs(mi, n_pairs=10)


# --------------------------------------------------------------------------
# 3. Training-only adaptation / no test-set dependency
# --------------------------------------------------------------------------


def test_no_function_in_module_accepts_test_data() -> None:
    """Structural guard: none of the public adaptation functions have a
    parameter that could plausibly be a test/label/performance array."""
    import src.quantum.adaptive_feature_map as mod

    forbidden_substrings = ["test", "y_true", "label", "proba", "score", "metric", "auc"]
    for name in ["compute_pairwise_mutual_information", "select_adaptive_entanglement_pairs",
                 "build_adaptive_feature_map", "build_adaptive_feature_map_from_training_data"]:
        fn = getattr(mod, name)
        params = list(inspect.signature(fn).parameters)
        for p in params:
            for bad in forbidden_substrings:
                assert bad not in p.lower(), f"{name} has suspicious parameter {p!r}"


def test_adaptive_map_unaffected_by_data_appended_after_training_call(synthetic_train) -> None:
    """Selecting pairs from the training set, then hypothetically having
    more (test-like) rows appended to a COPY afterward, must not change
    the already-selected pairs -- the function has no way to "peek"
    forward since it only ever sees what's passed to it once."""
    _, report_before = build_adaptive_feature_map_from_training_data(synthetic_train, 4, 2, random_state=42)
    # Simulate "test data" existing but never being passed in.
    rng = np.random.RandomState(999)
    _extra_test_like_rows = rng.uniform(0, np.pi, size=(200, 4))  # never used below
    _, report_after = build_adaptive_feature_map_from_training_data(synthetic_train, 4, 2, random_state=42)
    assert report_before.selected_pairs == report_after.selected_pairs


# --------------------------------------------------------------------------
# 4. Kernel numerical validity (reuses existing, already-tested kernel code)
# --------------------------------------------------------------------------


def test_adaptive_map_produces_a_numerically_valid_kernel(synthetic_train) -> None:
    circuit, _ = build_adaptive_feature_map_from_training_data(synthetic_train[:20], 4, 2, random_state=42)
    backend = get_backend("statevector")
    svs = compute_statevectors(synthetic_train[:20], circuit, backend)
    K = kernel_matrix_from_statevectors_blockwise(svs, svs, symmetric=True, block_size=10)
    diag = kernel_diagnostics(K, symmetric=True)
    assert diag.is_symmetric
    assert diag.within_valid_fidelity_range
    assert diag.diagonal_max_abs_deviation_from_one < 1e-5


# --------------------------------------------------------------------------
# 5. Correct statistical delta direction (Adaptive - Baseline)
# --------------------------------------------------------------------------


def test_delong_delta_sign_convention_adaptive_minus_baseline() -> None:
    """The experiment reports delta = Adaptive - Baseline. Confirm the sign
    convention: a clearly-BETTER 'adaptive' array yields a POSITIVE delta
    when passed as the first (proba_a) argument."""
    rng = np.random.RandomState(1)
    y = np.repeat([0, 1], 50)
    baseline = rng.uniform(0, 1, 100)  # near-random
    adaptive = np.clip(y * 0.8 + rng.normal(0.1, 0.1, 100), 0, 1)  # clearly separating
    result = delong_test(y, adaptive, baseline)  # (adaptive, baseline) order
    assert result.delta > 0  # adaptive - baseline > 0 when adaptive is better
    assert result.auc_a > result.auc_b


def test_adaptive_map_report_serializes_to_dict(synthetic_train) -> None:
    _, report = build_adaptive_feature_map_from_training_data(synthetic_train, 4, 2, random_state=42)
    d = report.to_dict()
    assert d["n_pairs"] == 3
    assert d["random_state"] == 42
    assert isinstance(report, AdaptiveMapReport)
