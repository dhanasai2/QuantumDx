"""Tests for src.quantum.vqc_classifier (Phase 8B).

Fast by design: uses tiny synthetic data and a tiny optimizer budget
(maxiter=2-3) throughout -- these tests exercise the CODE PATH, not
convergence quality (that's what the real n=2,000 screening run measures).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.quantum.vqc_classifier import VQCConfig, build_vqc, build_vqc_circuits, n_trainable_params


@pytest.fixture
def tiny_data():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(30, 4))
    y = rng.randint(0, 2, 30)
    return X, y


# --------------------------------------------------------------------------
# 1. Circuit construction / qubit count / trainable parameters
# --------------------------------------------------------------------------


def test_circuits_have_correct_qubit_count() -> None:
    cfg = VQCConfig(n_qubits=4, feature_map_reps=1, ansatz_reps=2)
    fm, ansatz = build_vqc_circuits(cfg)
    assert fm.num_qubits == 4
    assert ansatz.num_qubits == 4


def test_feature_map_has_one_parameter_per_qubit() -> None:
    cfg = VQCConfig(n_qubits=4, feature_map_reps=1)
    fm, _ = build_vqc_circuits(cfg)
    assert fm.num_parameters == 4  # one input feature per qubit, angle encoding


def test_n_trainable_params_matches_ansatz() -> None:
    cfg = VQCConfig(n_qubits=4, ansatz_reps=2)
    _, ansatz = build_vqc_circuits(cfg)
    assert n_trainable_params(cfg) == ansatz.num_parameters


def test_ansatz_reps_changes_parameter_count() -> None:
    cfg1 = VQCConfig(n_qubits=4, ansatz_reps=1)
    cfg2 = VQCConfig(n_qubits=4, ansatz_reps=2)
    assert n_trainable_params(cfg2) > n_trainable_params(cfg1)


# --------------------------------------------------------------------------
# 2. Forward/prediction behavior -- binary output
# --------------------------------------------------------------------------


def test_predict_proba_returns_valid_probabilities(tiny_data) -> None:
    X, y = tiny_data
    cfg = VQCConfig(optimizer_maxiter=2, shots=128, seed=42)
    vqc = build_vqc(cfg)
    vqc.fit(X, y)
    proba = vqc.predict_proba(X)
    assert proba.shape[0] == len(X)
    assert np.all(proba >= -1e-9) and np.all(proba <= 1 + 1e-9)
    # two-column (P(0), P(1)) output should sum to ~1 per row
    if proba.ndim == 2 and proba.shape[1] == 2:
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_returns_binary_labels(tiny_data) -> None:
    X, y = tiny_data
    cfg = VQCConfig(optimizer_maxiter=2, shots=128, seed=42)
    vqc = build_vqc(cfg)
    vqc.fit(X, y)
    preds = vqc.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


# --------------------------------------------------------------------------
# 3. Deterministic seed behavior
# --------------------------------------------------------------------------


def test_same_seed_gives_identical_fitted_model(tiny_data) -> None:
    X, y = tiny_data
    cfg = VQCConfig(optimizer_maxiter=2, shots=128, seed=42)
    vqc1 = build_vqc(cfg)
    vqc1.fit(X, y)
    vqc2 = build_vqc(cfg)
    vqc2.fit(X, y)
    assert np.allclose(vqc1.predict_proba(X), vqc2.predict_proba(X))


def test_different_seed_gives_different_initial_point() -> None:
    cfg1 = VQCConfig(seed=1)
    cfg2 = VQCConfig(seed=2)
    vqc1 = build_vqc(cfg1)
    vqc2 = build_vqc(cfg2)
    assert not np.allclose(vqc1.initial_point, vqc2.initial_point)


# --------------------------------------------------------------------------
# 4. Training-only preprocessing / no test-data leakage (structural)
# --------------------------------------------------------------------------


def test_build_vqc_accepts_no_data_at_all() -> None:
    """build_vqc/build_vqc_circuits take only a config -- structurally
    incapable of seeing training OR test data at construction time."""
    for fn in (build_vqc, build_vqc_circuits):
        params = list(inspect.signature(fn).parameters)
        for p in params:
            assert "x" not in p.lower() or p.lower() == "config"
            assert "test" not in p.lower()
            assert "y" not in p.lower()


def test_config_to_dict_is_json_serializable() -> None:
    import json

    cfg = VQCConfig()
    json.dumps(cfg.to_dict())  # must not raise
