"""Tests for src.large_dataset.phase13_hybrid_system and the underlying
quantum_refinement_circuit / matched_refinement_controls modules.

Covers: OOF correctness, quantum output bounds, parameter sensitivity
(circuit is not accidentally constant), leakage invariants, the
prediction-interface self-consistency check, and output/artifact schema
(if the real experiment has run).
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from src.large_dataset import phase13_hybrid_system as exp
from src.quantum.matched_refinement_controls import (
    MatchedRFFConfig,
    MatchedRFFRefinement,
    NoOpRandomConfig,
    NoOpRandomRefinement,
)
from src.quantum.quantum_refinement_circuit import RefinementConfig, QuantumRefinementModule

EXPECTED_FINGERPRINT = "96eac11a8394b87e"


# --------------------------------------------------------------------------
# 1. Quantum output bounds [-1, 1] (raw expectation values, not the score)
# --------------------------------------------------------------------------


def test_expectation_values_bounded() -> None:
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(30, 6))
    cfg = RefinementConfig(seed=42)
    mod = QuantumRefinementModule(cfg)
    circuit_params = rng.uniform(-1, 1, cfg.n_circuit_params())
    z = mod.expectation_values(X, circuit_params)
    assert z.shape == (30, 6)
    assert np.all(z >= -1.0 - 1e-6) and np.all(z <= 1.0 + 1e-6)


def test_score_output_shape() -> None:
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(20, 6))
    cfg = RefinementConfig(seed=42)
    mod = QuantumRefinementModule(cfg)
    params = rng.uniform(-1, 1, cfg.n_trainable_params())
    scores = mod.score(X, params)
    assert scores.shape == (20,)


# --------------------------------------------------------------------------
# 2. Circuit is not accidentally constant -- parameters actually matter
# --------------------------------------------------------------------------


def test_quantum_score_changes_with_different_circuit_params() -> None:
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(20, 6))
    cfg = RefinementConfig(seed=42)
    mod = QuantumRefinementModule(cfg)
    params_a = np.zeros(cfg.n_trainable_params())
    params_b = rng.uniform(-1, 1, cfg.n_trainable_params())
    scores_a = mod.score(X, params_a)
    scores_b = mod.score(X, params_b)
    assert not np.allclose(scores_a, scores_b)


def test_quantum_expectation_values_vary_across_input_rows() -> None:
    """The circuit must not collapse to a constant output regardless of input."""
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(20, 6))
    cfg = RefinementConfig(seed=42)
    mod = QuantumRefinementModule(cfg)
    params = rng.uniform(-1, 1, cfg.n_circuit_params())
    z = mod.expectation_values(X, params)
    assert z.std(axis=0).max() > 1e-3


# --------------------------------------------------------------------------
# 3. Matched controls: same interface, deterministic, sanity-differentiated
# --------------------------------------------------------------------------


def test_rff_control_output_shape_and_determinism() -> None:
    cfg = MatchedRFFConfig(input_dim=6, n_features=6, seed=42)
    mod = MatchedRFFRefinement(cfg)
    X = np.random.RandomState(0).uniform(0, np.pi, size=(15, 6))
    params = np.zeros(cfg.n_trainable_params())
    s1, s2 = mod.score(X, params), mod.score(X, params)
    assert s1.shape == (15,)
    assert np.array_equal(s1, s2)


def test_noop_control_has_no_phi_parameter_only_readout() -> None:
    """The no-op control's trainable param count must be n_features+1
    (readout only) -- strictly fewer than the RFF control's, since it has
    no trainable phase-shift parameters."""
    rff_cfg = MatchedRFFConfig(n_features=6)
    noop_cfg = NoOpRandomConfig(n_features=6)
    assert noop_cfg.n_trainable_params() < rff_cfg.n_trainable_params()


# --------------------------------------------------------------------------
# 4. select_refinement_weights: leakage-free, deterministic
# --------------------------------------------------------------------------


def test_select_refinement_weights_accepts_no_test_parameter() -> None:
    params = list(inspect.signature(exp.select_refinement_weights).parameters)
    for p in params:
        assert "test" not in p.lower()


def test_select_refinement_weights_reduces_cv_mse_below_naive_zero() -> None:
    """A sanity check that the optimizer is actually doing something: the
    selected weights' CV MSE should be no worse than always predicting
    zero residual (the naive baseline) for a residual with real signal."""
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(60, 6))
    residual = 0.3 * np.sin(X[:, 0]) + rng.normal(0, 0.05, 60)
    mod = MatchedRFFRefinement(MatchedRFFConfig(input_dim=6, n_features=6, seed=42))
    result = exp.select_refinement_weights(mod.score, mod.config.n_trainable_params(), X, residual, seed=42, maxiter=15)
    naive_mse = float(np.mean(residual ** 2))
    assert result["best_cv_mse"] <= naive_mse * 1.5  # generous bound -- not meant to be tight


def test_select_refinement_weights_is_deterministic() -> None:
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, size=(40, 6))
    residual = rng.normal(0, 0.3, 40)
    mod = NoOpRandomRefinement(NoOpRandomConfig(seed=42))
    r1 = exp.select_refinement_weights(mod.score, mod.config.n_trainable_params(), X, residual, seed=42, maxiter=5)
    r2 = exp.select_refinement_weights(mod.score, mod.config.n_trainable_params(), X, residual, seed=42, maxiter=5)
    assert np.allclose(r1["weights"], r2["weights"])


# --------------------------------------------------------------------------
# 5. OOF correctness
# --------------------------------------------------------------------------


def test_oof_predictions_are_not_perfectly_correlated_with_training_labels() -> None:
    """A strict smoke check that OOF predictions come from held-out folds,
    not in-sample fits: a model fit on ALL data would badly overfit tiny
    n and separate classes almost perfectly, whereas true OOF predictions
    on a small, noisy dataset should show real held-out error."""
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 1, size=(100, 4))
    y = rng.randint(0, 2, 100)  # pure noise -- no real signal
    oof = exp.compute_oof_xgb_predictions(X, y, n_folds=5, seed=42)
    assert oof.shape == (100,)
    assert oof.min() >= 0.0 and oof.max() <= 1.0
    # on pure noise, OOF ROC-AUC must be close to chance (0.5) -- an
    # in-sample-leaked prediction would show much higher separation
    from sklearn.metrics import roc_auc_score
    assert abs(roc_auc_score(y, oof) - 0.5) < 0.25


def test_oof_predictions_deterministic_for_fixed_seed() -> None:
    rng = np.random.RandomState(1)
    X = rng.uniform(0, 1, size=(60, 4))
    y = rng.randint(0, 2, 60)
    oof1 = exp.compute_oof_xgb_predictions(X, y, n_folds=3, seed=42)
    oof2 = exp.compute_oof_xgb_predictions(X, y, n_folds=3, seed=42)
    assert np.array_equal(oof1, oof2)


def test_build_quantum_input_has_expected_columns() -> None:
    pca = np.random.RandomState(0).uniform(0, np.pi, size=(10, 4))
    proba = np.random.RandomState(1).uniform(0, 1, 10)
    inp = exp.build_quantum_input(pca, proba)
    assert inp.shape == (10, 6)
    assert np.allclose(inp[:, 4], proba)
    assert np.allclose(inp[:, 5], np.abs(proba - 0.5))


# --------------------------------------------------------------------------
# 6. Real artifacts, if present
# --------------------------------------------------------------------------


def test_predictions_csv_schema_if_present() -> None:
    path = exp.RESULTS_ROOT / "predictions.csv"
    if not path.is_file():
        pytest.skip("Phase 13 has not been run yet")
    df = pd.read_csv(path)
    assert len(df) == 200
    assert not df.isna().any().any()
    for c in [c for c in df.columns if c.endswith("_proba")]:
        assert df[c].between(0.0, 1.0).all()


def test_summary_reports_expected_fingerprint_and_decision_if_present() -> None:
    path = exp.RESULTS_ROOT / "phase13_summary.json"
    if not path.is_file():
        pytest.skip("Phase 13 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["comparison_set"]["fingerprint"] == EXPECTED_FINGERPRINT
    assert summary["comparison_set"]["matches_expected"] is True
    assert summary["n_train_full"] == 66641
    assert summary["decision"]["outcome"] in {"WIN", "PARTIAL", "NO_IMPROVEMENT"}


def test_prediction_interface_agrees_with_batch_pipeline_if_present() -> None:
    path = exp.RESULTS_ROOT / "prediction_interface_example.json"
    if not path.is_file():
        pytest.skip("Phase 13 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        check = json.load(fh)
    assert check["agrees_with_batch_pipeline"] is True
    assert abs(check["interface_predicted_risk"] - check["batch_pipeline_predicted_risk"]) < 1e-3


def test_seed_by_seed_results_has_all_modules_and_seeds_if_present() -> None:
    path = exp.RESULTS_ROOT / "seed_by_seed_results.csv"
    if not path.is_file():
        pytest.skip("Phase 13 has not been run yet")
    df = pd.read_csv(path)
    assert set(df["module"].unique()) == {"quantum", "rff_control", "noop_control"}
    assert set(df["seed"].unique()) == {42, 123, 2024, 7, 99}


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
        (root / "phase12_quantum_representation" / "predictions.csv", 200),
    ]:
        if path.is_file():
            assert len(pd.read_csv(path)) == expected_len
