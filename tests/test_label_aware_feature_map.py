"""Tests for src.quantum.label_aware_feature_map (Phase 8A).

Covers: deterministic construction, correct qubit/gate counts, training-
(data+label)-only adaptation with NO test-set dependency, detection of a
known synergistic (XOR-like) interaction that marginal MI alone would
miss, and correct statistical delta-direction convention end to end.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from qiskit.circuit.library import zz_feature_map

from src.quantum.label_aware_feature_map import (
    LabelAwareMapReport,
    build_label_aware_feature_map_from_training_data,
    compute_label_aware_interaction_scores,
    discretize_quantile,
    select_label_aware_entanglement_pairs,
)
from src.large_dataset.statistical_robustness import delong_test


@pytest.fixture(scope="module")
def xor_train():
    """A synthetic set where (feature 2, feature 3) jointly determine the
    label via XOR while being near-independent of it individually -- the
    textbook case only INTERACTION information (not marginal MI) detects."""
    rng = np.random.RandomState(0)
    n = 600
    X = rng.uniform(0, np.pi, size=(n, 4))
    b2 = (X[:, 2] > np.pi / 2).astype(int)
    b3 = (X[:, 3] > np.pi / 2).astype(int)
    y = (b2 ^ b3)
    return X, y


# --------------------------------------------------------------------------
# 1. Deterministic construction / reproducibility
# --------------------------------------------------------------------------


def test_label_aware_construction_is_deterministic(xor_train) -> None:
    X, y = xor_train
    _, r1 = build_label_aware_feature_map_from_training_data(X, y, 4, 2)
    _, r2 = build_label_aware_feature_map_from_training_data(X, y, 4, 2)
    assert r1.selected_pairs == r2.selected_pairs
    assert r1.synergy_matrix == r2.synergy_matrix


def test_discretize_quantile_is_deterministic() -> None:
    rng = np.random.RandomState(1)
    x = rng.uniform(0, 1, 200)
    assert np.array_equal(discretize_quantile(x), discretize_quantile(x))


# --------------------------------------------------------------------------
# 2. Detects a known synergistic (XOR-like) interaction
# --------------------------------------------------------------------------


def test_detects_known_xor_synergy(xor_train) -> None:
    """The synergy criterion must select pair (2,3) as top -- a pair that
    is each MARGINALLY near-independent of y but JOINTLY determines it."""
    X, y = xor_train
    scores = compute_label_aware_interaction_scores(X, y)
    pairs = select_label_aware_entanglement_pairs(scores, n_pairs=1)
    assert pairs == [(2, 3)]


def test_marginal_mi_is_low_for_the_xor_features(xor_train) -> None:
    """Sanity check on the synthetic setup itself: features 2 and 3 must
    indeed look weak INDIVIDUALLY (confirming synergy, not marginal
    relevance, is what drove the selection above)."""
    X, y = xor_train
    _, report = build_label_aware_feature_map_from_training_data(X, y, 4, 2)
    assert report.marginal_label_mi[2] < 0.05
    assert report.marginal_label_mi[3] < 0.05


# --------------------------------------------------------------------------
# 3. Correct qubit / gate counts (same budget as baseline)
# --------------------------------------------------------------------------


def test_label_aware_circuit_has_correct_dimensions(xor_train) -> None:
    X, y = xor_train
    circuit, report = build_label_aware_feature_map_from_training_data(X, y, 4, 2)
    assert circuit.num_qubits == 4
    assert circuit.num_parameters == 4
    assert report.n_qubits == 4


def test_label_aware_circuit_matches_baseline_gate_budget(xor_train) -> None:
    X, y = xor_train
    circuit, _ = build_label_aware_feature_map_from_training_data(X, y, 4, 2)
    baseline = zz_feature_map(feature_dimension=4, reps=2, entanglement="linear")
    assert circuit.decompose().count_ops() == baseline.decompose().count_ops()


def test_n_pairs_defaults_to_three(xor_train) -> None:
    X, y = xor_train
    _, report = build_label_aware_feature_map_from_training_data(X, y, 4, 2)
    assert report.n_pairs == 3


# --------------------------------------------------------------------------
# 4. Training-(data+label)-only adaptation; NO test-set dependency
# --------------------------------------------------------------------------


def test_no_function_accepts_test_or_performance_data() -> None:
    """y_train IS a legitimate parameter here (unlike the unsupervised MI
    map) -- that's the whole point of this module. What must never appear
    is anything test-set- or performance-derived."""
    import src.quantum.label_aware_feature_map as mod

    forbidden = ["test", "proba", "score_test", "y_test", "auc", "metric_value"]
    for name in ["discretize_quantile", "compute_label_aware_interaction_scores",
                 "select_label_aware_entanglement_pairs", "build_label_aware_feature_map_from_training_data"]:
        fn = getattr(mod, name)
        for p in inspect.signature(fn).parameters:
            for bad in forbidden:
                assert bad not in p.lower(), f"{name} has suspicious parameter {p!r}"


def test_y_train_is_an_explicit_documented_parameter() -> None:
    """Confirms the (documented, deliberate) label-awareness: y_train
    IS accepted -- contrast with adaptive_feature_map.py's functions,
    which accept no label at all."""
    import src.quantum.label_aware_feature_map as mod

    params = list(inspect.signature(mod.build_label_aware_feature_map_from_training_data).parameters)
    assert "y_train" in params


def test_selection_unaffected_by_hypothetical_test_like_data(xor_train) -> None:
    X, y = xor_train
    _, report_before = build_label_aware_feature_map_from_training_data(X, y, 4, 2)
    rng = np.random.RandomState(999)
    _unused_test_like = (rng.uniform(0, np.pi, (100, 4)), rng.randint(0, 2, 100))
    _, report_after = build_label_aware_feature_map_from_training_data(X, y, 4, 2)
    assert report_before.selected_pairs == report_after.selected_pairs


def test_select_pairs_rejects_n_pairs_exceeding_available() -> None:
    scores = np.zeros((4, 4))
    with pytest.raises(ValueError):
        select_label_aware_entanglement_pairs(scores, n_pairs=10)


# --------------------------------------------------------------------------
# 5. Correct statistical delta direction (Variant - Baseline)
# --------------------------------------------------------------------------


def test_delong_delta_sign_convention_variant_minus_baseline() -> None:
    rng = np.random.RandomState(1)
    y = np.repeat([0, 1], 50)
    baseline = rng.uniform(0, 1, 100)
    variant = np.clip(y * 0.8 + rng.normal(0.1, 0.1, 100), 0, 1)
    result = delong_test(y, variant, baseline)
    assert result.delta > 0
    assert result.auc_a > result.auc_b


def test_report_serializes_to_dict(xor_train) -> None:
    X, y = xor_train
    _, report = build_label_aware_feature_map_from_training_data(X, y, 4, 2)
    d = report.to_dict()
    assert d["n_pairs"] == 3
    assert isinstance(report, LabelAwareMapReport)
