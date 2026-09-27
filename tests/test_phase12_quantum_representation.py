"""Tests for src.large_dataset.phase12_quantum_representation and the
underlying src.quantum.quantum_latent_representation /
matched_classical_control modules.

Covers: circuit/representation shapes, bounded expectation values
(regression guard for the Phase 10 EstimatorQNN default_precision bug),
train-only leakage invariants, matched-representation-agnostic training,
decision-rule correctness, and output/artifact schema (if the real
experiment has run).
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from src.large_dataset import phase12_quantum_representation as exp
from src.quantum.matched_classical_control import MatchedControlConfig, MatchedControlLayer
from src.quantum.quantum_latent_representation import (
    OBSERVABLE_NAMES,
    QuantumRepresentationConfig,
    QuantumRepresentationLayer,
    build_quantum_representation_circuit,
)

EXPECTED_FINGERPRINT = "96eac11a8394b87e"


# --------------------------------------------------------------------------
# 1. Quantum representation shape / architecture distinctness from Phase 10
# --------------------------------------------------------------------------


def test_circuit_has_correct_qubit_and_layer_structure() -> None:
    qc, x_params, theta_params = build_quantum_representation_circuit(n_qubits=4, n_layers=2)
    assert qc.num_qubits == 4
    assert len(x_params) == 4
    assert len(theta_params) == 16  # (4 RY + 4 RZZ) * 2 layers


def test_output_dimension_is_eight_richer_than_phase10() -> None:
    """Phase 10's hybrid had K=4 (single-qubit Z only); Phase 12 must be
    materially richer (K=8: 4 single-qubit + 4 two-qubit correlators)."""
    cfg = QuantumRepresentationConfig()
    assert cfg.output_dim() == 8
    assert len(OBSERVABLE_NAMES) == 8
    assert any("ZZ" in n for n in OBSERVABLE_NAMES)  # two-qubit correlators present


def test_uses_trainable_entanglement_not_fixed_cnot() -> None:
    """Phase 10 used a FIXED cx (CNOT) entangling gate; Phase 12 must use
    a TRAINABLE two-qubit gate (rzz) instead. Checked at the UNDECOMPOSED
    circuit level -- rzz decomposes into cx+rz internally, so checking
    after .decompose() would (incorrectly) show cx regardless."""
    qc, _, _ = build_quantum_representation_circuit()
    ops = qc.count_ops()
    assert "rzz" in ops
    assert "cx" not in ops
    assert ops["rzz"] == 8  # 4 entangling pairs x 2 layers, each a TRAINABLE angle


# --------------------------------------------------------------------------
# 2. Bounded expectation values (regression guard for Phase 10's caught bug)
# --------------------------------------------------------------------------


def test_quantum_features_bounded_in_valid_range() -> None:
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(30, 4))
    cfg = QuantumRepresentationConfig(seed=42)
    layer = QuantumRepresentationLayer(cfg)
    theta = rng.uniform(-1, 1, cfg.n_trainable_params())
    out = layer.transform(X, theta)
    assert out.shape == (30, 8)
    assert np.all(out >= -1.0 - 1e-6) and np.all(out <= 1.0 + 1e-6)


def test_estimator_qnn_default_precision_is_exact() -> None:
    """Direct regression test for the Phase 10 bug: EstimatorQNN's OWN
    default_precision must be explicitly set to 0.0, not just the
    underlying StatevectorEstimator's."""
    layer = QuantumRepresentationLayer(QuantumRepresentationConfig(seed=42))
    assert layer._qnn.default_precision == 0.0


def test_quantum_features_match_direct_statevector_computation() -> None:
    """End-to-end check that EstimatorQNN's output agrees with an
    independent, direct Statevector.expectation_value() computation --
    the exact check that caught Phase 10's precision bug."""
    from qiskit.quantum_info import Statevector
    from src.quantum.quantum_latent_representation import _observables

    cfg = QuantumRepresentationConfig(seed=42)
    layer = QuantumRepresentationLayer(cfg)
    rng = np.random.RandomState(1)
    x_row = rng.uniform(0, np.pi, 4)
    theta = rng.uniform(-1, 1, cfg.n_trainable_params())

    qnn_out = layer.transform(x_row.reshape(1, -1), theta)[0]

    bindings = dict(zip(layer._x_params, x_row))
    bindings.update(dict(zip(layer._theta_params, theta)))
    bound = layer.circuit.assign_parameters(bindings)
    sv = Statevector.from_instruction(bound)
    exact = np.array([np.real(sv.expectation_value(o)) for o in _observables(cfg.n_qubits)])

    assert np.allclose(qnn_out, exact, atol=1e-6)


# --------------------------------------------------------------------------
# 3. Matched control: same interface, deterministic, bounded
# --------------------------------------------------------------------------


def test_control_layer_output_shape_matches_quantum() -> None:
    cfg = MatchedControlConfig(input_dim=4, output_dim=8, seed=42)
    layer = MatchedControlLayer(cfg)
    X = np.random.RandomState(0).uniform(0, np.pi, size=(20, 4))
    phi = np.zeros(cfg.n_trainable_params())
    out = layer.transform(X, phi)
    assert out.shape == (20, 8)


def test_control_layer_output_is_bounded_cosine_range() -> None:
    cfg = MatchedControlConfig(seed=42)
    layer = MatchedControlLayer(cfg)
    X = np.random.RandomState(0).uniform(-10, 10, size=(50, 4))
    phi = np.random.RandomState(1).uniform(-1, 1, cfg.n_trainable_params())
    out = layer.transform(X, phi)
    assert np.all(out >= -1.0 - 1e-9) and np.all(out <= 1.0 + 1e-9)


def test_control_layer_deterministic_for_fixed_seed_and_weights() -> None:
    cfg = MatchedControlConfig(seed=42)
    layer1 = MatchedControlLayer(cfg)
    layer2 = MatchedControlLayer(cfg)
    X = np.random.RandomState(0).uniform(0, np.pi, size=(10, 4))
    phi = np.zeros(cfg.n_trainable_params())
    assert np.array_equal(layer1.transform(X, phi), layer2.transform(X, phi))


# --------------------------------------------------------------------------
# 4. select_representation_weights: representation-agnostic, train-only
# --------------------------------------------------------------------------


def test_select_representation_weights_accepts_no_test_data_parameter() -> None:
    params = list(inspect.signature(exp.select_representation_weights).parameters)
    for p in params:
        assert "test" not in p.lower()


def test_select_representation_weights_works_with_quantum_layer() -> None:
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(40, 4))
    y = rng.randint(0, 2, 40)
    layer = QuantumRepresentationLayer(QuantumRepresentationConfig(seed=42))
    result = exp.select_representation_weights(
        layer.transform, layer.config.n_trainable_params(), X, y, seed=42, maxiter=2,
    )
    assert "weights" in result and "best_cv_score" in result
    assert 0.0 <= result["best_cv_score"] <= 1.0


def test_select_representation_weights_works_with_control_layer() -> None:
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(40, 4))
    y = rng.randint(0, 2, 40)
    layer = MatchedControlLayer(MatchedControlConfig(seed=42))
    result = exp.select_representation_weights(
        layer.transform, layer.config.n_trainable_params(), X, y, seed=42, maxiter=5,
    )
    assert "weights" in result and "best_cv_score" in result


def test_select_representation_weights_is_deterministic() -> None:
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(40, 4))
    y = rng.randint(0, 2, 40)
    layer = MatchedControlLayer(MatchedControlConfig(seed=42))
    r1 = exp.select_representation_weights(layer.transform, layer.config.n_trainable_params(), X, y, seed=42, maxiter=5)
    r2 = exp.select_representation_weights(layer.transform, layer.config.n_trainable_params(), X, y, seed=42, maxiter=5)
    assert np.allclose(r1["weights"], r2["weights"])


def test_maxiter_none_resolves_to_module_constant(monkeypatch) -> None:
    """Regression test: OPTIMIZER_MAXITER must be resolved at CALL time
    (monkeypatch-friendly), not bound as a stale function-definition-time
    default -- this is what makes cheap dry runs of Phase 12 possible."""
    monkeypatch.setattr(exp, "OPTIMIZER_MAXITER", 2)
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(30, 4))
    y = rng.randint(0, 2, 30)
    layer = MatchedControlLayer(MatchedControlConfig(seed=42))
    result = exp.select_representation_weights(layer.transform, layer.config.n_trainable_params(), X, y, seed=42)
    assert result["n_function_evaluations"] <= 10  # bounded by the tiny patched maxiter (+ COBYLA's own floor)


# --------------------------------------------------------------------------
# 5. Real artifacts, if present
# --------------------------------------------------------------------------


def test_predictions_csv_schema_if_present() -> None:
    path = exp.RESULTS_ROOT / "predictions.csv"
    if not path.is_file():
        pytest.skip("Phase 12 has not been run yet")
    df = pd.read_csv(path)
    assert len(df) == 200
    assert not df.isna().any().any()
    proba_cols = [c for c in df.columns if c.endswith("_proba")]
    for c in proba_cols:
        assert df[c].between(0.0, 1.0).all()


def test_summary_reports_expected_fingerprint_and_decision_if_present() -> None:
    path = exp.RESULTS_ROOT / "phase12_summary.json"
    if not path.is_file():
        pytest.skip("Phase 12 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["comparison_set"]["fingerprint"] == EXPECTED_FINGERPRINT
    assert summary["comparison_set"]["matches_expected"] is True
    assert summary["n_train_full"] == 66641
    assert summary["decision"]["outcome"] in {"B", "B2", "C", "D"}


def test_seed_by_seed_results_has_all_five_seeds_if_present() -> None:
    path = exp.RESULTS_ROOT / "seed_by_seed_results.csv"
    if not path.is_file():
        pytest.skip("Phase 12 has not been run yet")
    df = pd.read_csv(path)
    assert set(df["seed"].unique()) == {42, 123, 2024, 7, 99}
    assert set(df["representation"].unique()) == {"quantum", "control"}
    assert len(df) == 10  # 5 seeds x 2 representations


def test_reproducibility_summary_reports_std_not_just_mean_if_present() -> None:
    path = exp.RESULTS_ROOT / "reproducibility_summary.json"
    if not path.is_file():
        pytest.skip("Phase 12 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert "std" in summary["quantum"] and "std" in summary["control"]
    assert len(summary["quantum"]["per_seed_cv_roc_auc"]) == 5


def test_prior_phase_artifacts_untouched_if_present() -> None:
    from src.data.inspect_dataset import find_project_root

    root = find_project_root() / "results" / "large_dataset"
    for path, expected_len in [
        (root / "stage_d" / "predictions.csv", 200),
        (root / "phase8a_label_aware" / "predictions.csv", 200),
        (root / "phase8b_vqc" / "predictions.csv", 200),
        (root / "phase9_classical_benchmark" / "predictions.csv", 200),
        (root / "phase10_hybrid_qml" / "predictions.csv", 200),
        (root / "phase11_full_scale" / "predictions.csv", 200),
    ]:
        if path.is_file():
            assert len(pd.read_csv(path)) == expected_len
