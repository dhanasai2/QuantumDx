"""Tests for src.quantum.hybrid_quantum_features (Phase 10).

Fast by design: tiny synthetic data, tiny optimizer budget (maxiter=2-3).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.quantum.hybrid_quantum_features import (
    HybridConfig,
    HybridQuantumFeatureLayer,
    build_hybrid_circuit,
)


@pytest.fixture
def tiny_data():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(40, 4))
    y = rng.randint(0, 2, 40)
    return X, y


# --------------------------------------------------------------------------
# 1. Quantum feature layer shape
# --------------------------------------------------------------------------


def test_circuit_has_correct_qubit_count() -> None:
    qc, x_params, theta_params = build_hybrid_circuit(n_qubits=4, reps=1)
    assert qc.num_qubits == 4
    assert len(x_params) == 4
    assert len(theta_params) == 4


def test_quantum_features_have_correct_shape(tiny_data) -> None:
    X, y = tiny_data
    cfg = HybridConfig(optimizer_maxiter=2, seed=42)
    layer = HybridQuantumFeatureLayer(cfg)
    theta = np.zeros(cfg.n_trainable_params())
    features = layer.quantum_features(X, theta)
    assert features.shape == (len(X), 4)


def test_quantum_features_are_bounded_expectation_values(tiny_data) -> None:
    """<Z> expectation values must lie in [-1, 1]."""
    X, y = tiny_data
    cfg = HybridConfig(seed=42)
    layer = HybridQuantumFeatureLayer(cfg)
    theta = np.random.RandomState(1).uniform(-1, 1, cfg.n_trainable_params())
    features = layer.quantum_features(X, theta)
    assert np.all(features >= -1.0 - 1e-9) and np.all(features <= 1.0 + 1e-9)


# --------------------------------------------------------------------------
# 2. Deterministic behavior under fixed seed
# --------------------------------------------------------------------------


def test_same_seed_gives_identical_fit(tiny_data) -> None:
    X, y = tiny_data
    cfg = HybridConfig(optimizer_maxiter=3, seed=42)
    layer1 = HybridQuantumFeatureLayer(cfg)
    layer1.fit(X, y)
    layer2 = HybridQuantumFeatureLayer(cfg)
    layer2.fit(X, y)
    assert np.allclose(layer1.fit_result_.theta, layer2.fit_result_.theta)
    assert np.allclose(layer1.predict_proba(X), layer2.predict_proba(X))


def test_quantum_features_deterministic_for_fixed_theta(tiny_data) -> None:
    X, y = tiny_data
    cfg = HybridConfig(seed=42)
    layer = HybridQuantumFeatureLayer(cfg)
    theta = np.array([0.1, 0.2, 0.3, 0.4])
    f1 = layer.quantum_features(X, theta)
    f2 = layer.quantum_features(X, theta)
    assert np.array_equal(f1, f2)


# --------------------------------------------------------------------------
# 3. Preprocessing / leakage constraints (structural)
# --------------------------------------------------------------------------


def test_fit_accepts_no_test_data_parameter() -> None:
    params = list(inspect.signature(HybridQuantumFeatureLayer.fit).parameters)
    for p in params:
        assert "test" not in p.lower()


def test_predict_proba_accepts_no_label_parameter() -> None:
    params = list(inspect.signature(HybridQuantumFeatureLayer.predict_proba).parameters)
    for p in params:
        assert p == "self" or "y" not in p.lower()


# --------------------------------------------------------------------------
# 4. Hybrid feature concatenation
# --------------------------------------------------------------------------


def test_concat_stacks_classical_and_quantum_features(tiny_data) -> None:
    X, y = tiny_data
    cfg = HybridConfig(seed=42)
    layer = HybridQuantumFeatureLayer(cfg)
    theta = np.zeros(cfg.n_trainable_params())
    Z = layer._concat(X, theta)
    assert Z.shape == (len(X), 4 + 4)
    assert np.array_equal(Z[:, :4], X)


# --------------------------------------------------------------------------
# 5. Prediction output shape
# --------------------------------------------------------------------------


def test_predict_proba_output_shape(tiny_data) -> None:
    X, y = tiny_data
    cfg = HybridConfig(optimizer_maxiter=2, seed=42)
    layer = HybridQuantumFeatureLayer(cfg)
    layer.fit(X, y)
    proba = layer.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_raises_before_fit() -> None:
    layer = HybridQuantumFeatureLayer()
    with pytest.raises(RuntimeError):
        layer.predict_proba(np.zeros((5, 4)))


# --------------------------------------------------------------------------
# Feature report / explainability
# --------------------------------------------------------------------------


def test_feature_report_names_and_length() -> None:
    cfg = HybridConfig(n_qubits=4)
    layer = HybridQuantumFeatureLayer(cfg)
    report = layer.feature_report()
    assert len(report.feature_names) == 8
    assert report.feature_names[:4] == ["pca_0", "pca_1", "pca_2", "pca_3"]
    assert all(n.startswith("quantum_expZ_qubit") for n in report.feature_names[4:])
