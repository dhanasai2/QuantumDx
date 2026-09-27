"""Tests for src.large_dataset.phase15_quantum_classical_stacking.

Fast, synthetic-data tests cover the logic (fold isolation, leakage
structure, gate arithmetic, probability validity, prediction-interface
contracts, reproducibility). Slow, "if present" tests check the real
persisted run's artifacts/schema against the critical protocol
invariants (test set untouched unless the CV gate passed, etc.) --
mirroring tests/test_phase14_predictive_experiments.py's own convention.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.large_dataset import phase15_quantum_classical_stacking as p15

EXPECTED_FINGERPRINT = "96eac11a8394b87e"


# --------------------------------------------------------------------------
# 1. Padding helper (quantum/control input construction)
# --------------------------------------------------------------------------


def test_pad_to_quantum_input_dim_pads_with_zeros() -> None:
    X = np.array([[1.0, 2.0, 3.0, 4.0]])
    out = p15._pad_to_quantum_input_dim(X, target_dim=6)
    assert out.shape == (1, 6)
    assert np.array_equal(out[:, :4], X)
    assert np.array_equal(out[:, 4:], np.zeros((1, 2)))


def test_pad_to_quantum_input_dim_truncates_if_already_wide_enough() -> None:
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    out = p15._pad_to_quantum_input_dim(X, target_dim=6)
    assert out.shape == (1, 6)


def test_quantum_and_control_configs_are_dimension_matched() -> None:
    """Section 6's requirement: the control must be matched in role/dimensionality."""
    from src.quantum.matched_refinement_controls import MatchedRFFConfig
    from src.quantum.quantum_refinement_circuit import RefinementConfig

    q_cfg = RefinementConfig(seed=42)
    c_cfg = MatchedRFFConfig(seed=42)
    assert q_cfg.n_qubits == p15.QUANTUM_INPUT_DIM
    assert c_cfg.input_dim == p15.QUANTUM_INPUT_DIM
    assert c_cfg.n_features == p15.QUANTUM_INPUT_DIM


# --------------------------------------------------------------------------
# 2. Sigmoid / valid-probability guarantees
# --------------------------------------------------------------------------


def test_sigmoid_always_in_open_unit_interval() -> None:
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0, np.nan]).astype(float)
    x = x[~np.isnan(x)]
    out = p15._sigmoid(x)
    assert np.all(out > 0.0) and np.all(out < 1.0)


def test_quantum_and_control_scores_after_sigmoid_are_valid_probabilities() -> None:
    from src.quantum.matched_refinement_controls import MatchedRFFConfig, MatchedRFFRefinement
    from src.quantum.quantum_refinement_circuit import QuantumRefinementModule, RefinementConfig

    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, size=(20, p15.QUANTUM_INPUT_DIM))
    q_module = QuantumRefinementModule(RefinementConfig(seed=42))
    params = rng.uniform(-0.1, 0.1, size=q_module.config.n_trainable_params())
    p_quantum = p15._sigmoid(q_module.score(X, params))
    assert p_quantum.shape == (20,)
    assert np.all((p_quantum > 0) & (p_quantum < 1))

    c_module = MatchedRFFRefinement(MatchedRFFConfig(seed=42))
    c_params = rng.uniform(-0.1, 0.1, size=c_module.config.n_trainable_params())
    p_control = p15._sigmoid(c_module.score(X, c_params))
    assert np.all((p_control > 0) & (p_control < 1))


# --------------------------------------------------------------------------
# 3. Weight selection: no test/val parameter, deterministic, fold-isolated
# --------------------------------------------------------------------------


def test_select_scalar_classifier_weights_has_no_test_parameter() -> None:
    for p in inspect.signature(p15.select_scalar_classifier_weights).parameters:
        assert "test" not in p.lower() and "val" not in p.lower()


def test_select_scalar_classifier_weights_deterministic_for_fixed_seed() -> None:
    from src.quantum.matched_refinement_controls import MatchedRFFConfig, MatchedRFFRefinement

    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, size=(200, p15.QUANTUM_INPUT_DIM))
    y = rng.randint(0, 2, 200)
    module = MatchedRFFRefinement(MatchedRFFConfig(seed=7))
    r1 = p15.select_scalar_classifier_weights(module, module.config.n_trainable_params(), X, y, seed=7, maxiter=5, subsample_size=100)
    module2 = MatchedRFFRefinement(MatchedRFFConfig(seed=7))
    r2 = p15.select_scalar_classifier_weights(module2, module2.config.n_trainable_params(), X, y, seed=7, maxiter=5, subsample_size=100)
    assert np.allclose(r1["weights"], r2["weights"])
    assert r1["best_cv_auc"] == r2["best_cv_auc"]


def test_stratified_subsample_indices_stay_within_bounds_and_are_unique() -> None:
    rng = np.random.RandomState(1)
    y = np.array([0] * 800 + [1] * 200)
    idx = p15._stratified_subsample_indices(y, 100, rng)
    assert len(idx) <= 100
    assert len(set(idx.tolist())) == len(idx)
    assert idx.min() >= 0 and idx.max() < len(y)
    # roughly stratified (not exact due to rounding)
    pos_frac = np.mean(y[idx] == 1)
    assert 0.1 < pos_frac < 0.3


# --------------------------------------------------------------------------
# 4. OOF generation: genuine fold isolation on a small synthetic dataset
# --------------------------------------------------------------------------


def _make_synthetic_pool(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """A tiny synthetic pool matching the cardio schema well enough for
    process_stage to run (real feature columns + a target column)."""
    from src.large_dataset.schema import TARGET_COLUMN, get_cardio_feature_groups

    rng = np.random.RandomState(seed)
    fg = get_cardio_feature_groups()
    data = {"id": np.arange(n)}
    for col in fg.all_columns:
        if col in ("gender_male", "smoke", "alco", "active"):
            data[col] = rng.randint(0, 2, n)
        elif col == "cholesterol" or col == "gluc":
            data[col] = rng.randint(1, 4, n)
        elif col == "age_years":
            data[col] = rng.randint(30, 70, n)
        elif col == "height":
            data[col] = rng.uniform(150, 190, n)
        elif col == "weight":
            data[col] = rng.uniform(50, 100, n)
        elif col in ("ap_hi",):
            data[col] = rng.uniform(100, 160, n)
        elif col in ("ap_lo",):
            data[col] = rng.uniform(60, 100, n)
        else:
            data[col] = rng.uniform(0, 1, n)
    data[TARGET_COLUMN] = rng.randint(0, 2, n)
    return pd.DataFrame(data)


def test_oof_predictions_partition_every_row_exactly_once() -> None:
    from src.large_dataset.schema import get_cardio_feature_groups
    from src.preprocessing.config import PreprocessingConfig

    pool = _make_synthetic_pool(n=150)
    fg = get_cardio_feature_groups()
    pcfg = PreprocessingConfig()
    result = p15.generate_oof_predictions(pool, fg, pcfg, seed=42, n_folds=3)

    assert len(result.oof_df) == len(pool)
    assert set(result.oof_df["fold"].unique()) == {0, 1, 2}
    # every id appears exactly once (a genuine partition, not overlapping folds)
    assert result.oof_df["id"].nunique() == len(pool)
    assert result.oof_df[["p_xgb", "p_quantum", "p_control"]].notna().all().all()
    assert result.oof_df[["p_xgb", "p_quantum", "p_control"]].apply(lambda s: s.between(0, 1).all()).all()


def test_oof_generation_calls_process_stage_exactly_once_per_fold(monkeypatch) -> None:
    from src.large_dataset.schema import get_cardio_feature_groups
    from src.preprocessing.config import PreprocessingConfig

    pool = _make_synthetic_pool(n=120)
    fg = get_cardio_feature_groups()
    pcfg = PreprocessingConfig()

    call_count = {"n": 0}
    real_process_stage = p15.process_stage

    def counting_process_stage(*args, **kwargs):
        call_count["n"] += 1
        return real_process_stage(*args, **kwargs)

    monkeypatch.setattr(p15, "process_stage", counting_process_stage)
    p15.generate_oof_predictions(pool, fg, pcfg, seed=42, n_folds=3)
    assert call_count["n"] == 3  # exactly one fit-transform per outer fold, never on the pooled/global data


# --------------------------------------------------------------------------
# 5. Nested-CV meta-learner evaluation: only sees OOF columns
# --------------------------------------------------------------------------


def test_evaluate_stacks_via_nested_cv_uses_only_oof_columns() -> None:
    rng = np.random.RandomState(0)
    n = 100
    oof_df = pd.DataFrame({
        "id": np.arange(n), "y": rng.randint(0, 2, n), "fold": rng.randint(0, 5, n),
        "p_xgb": rng.uniform(0, 1, n), "p_quantum": rng.uniform(0, 1, n), "p_control": rng.uniform(0, 1, n),
    })
    result = p15.evaluate_stacks_via_nested_cv(oof_df)
    assert set(result["summary"].keys()) == {"xgboost_alone", "quantum_stack", "control_stack"}
    for name, s in result["summary"].items():
        assert "roc_auc_mean" in s and "roc_auc_std" in s and "per_fold_roc_auc" in s
        assert len(s["per_fold_roc_auc"]) == oof_df["fold"].nunique()


def test_nested_cv_meta_learner_never_fit_on_its_own_validation_fold() -> None:
    """A meta-learner fit on ALL folds (including its own validation fold)
    would very likely NOT match one fit on the other folds only -- use
    this as an indirect leakage probe: refitting with a shuffled fold
    column should change per-fold scores (proving the fold split matters,
    i.e. it genuinely governs what the meta-learner sees)."""
    rng = np.random.RandomState(3)
    n = 200
    y = rng.randint(0, 2, n)
    # Deliberately noisy/imperfect (not a clean separator) so that different
    # fold partitions can actually produce different per-fold ROC-AUC values
    # -- a cleanly separable p_xgb (ROC-AUC=1.0 regardless of split) would
    # make this probe vacuous.
    oof_df = pd.DataFrame({
        "id": np.arange(n), "y": y, "fold": np.repeat(np.arange(5), n // 5),
        "p_xgb": y * 0.15 + rng.uniform(0, 1, n), "p_quantum": rng.uniform(0, 1, n), "p_control": rng.uniform(0, 1, n),
    })
    result_a = p15.evaluate_stacks_via_nested_cv(oof_df)
    shuffled = oof_df.copy()
    shuffled["fold"] = rng.permutation(shuffled["fold"].to_numpy())
    result_b = p15.evaluate_stacks_via_nested_cv(shuffled)
    # Different fold partitions should not be guaranteed identical per-fold scores.
    assert result_a["summary"]["xgboost_alone"]["per_fold_roc_auc"] != result_b["summary"]["xgboost_alone"]["per_fold_roc_auc"]


# --------------------------------------------------------------------------
# 6. Acceptance-gate arithmetic (Section 8's 5 criteria)
# --------------------------------------------------------------------------


def _fake_cv_eval(a_mean, a_std, a_folds, b_mean, b_folds, c_mean) -> dict:
    return {"summary": {
        "xgboost_alone": {"roc_auc_mean": a_mean, "roc_auc_std": a_std, "per_fold_roc_auc": a_folds},
        "quantum_stack": {"roc_auc_mean": b_mean, "roc_auc_std": 0.01, "per_fold_roc_auc": b_folds},
        "control_stack": {"roc_auc_mean": c_mean, "roc_auc_std": 0.01, "per_fold_roc_auc": [c_mean] * len(a_folds)},
    }}


def test_gate_fails_when_quantum_does_not_beat_baseline() -> None:
    cv = _fake_cv_eval(0.85, 0.01, [0.85] * 5, 0.84, [0.84] * 5, 0.83)
    gate = p15.check_acceptance_gate(cv)
    assert gate["gate_passed"] is False
    assert gate["criterion_1_beats_baseline_mean_cv"] is False


def test_gate_fails_when_quantum_does_not_beat_matched_control() -> None:
    cv = _fake_cv_eval(0.85, 0.005, [0.85] * 5, 0.86, [0.86] * 5, 0.90)
    gate = p15.check_acceptance_gate(cv)
    assert gate["criterion_4_beats_matched_control"] is False
    assert gate["gate_passed"] is False


def test_gate_fails_on_a_single_lucky_fold() -> None:
    a_folds = [0.80, 0.80, 0.80, 0.80, 0.80]
    b_folds = [0.95, 0.79, 0.79, 0.79, 0.79]  # only fold 0 improved -- a lucky fold, not consistent
    cv = _fake_cv_eval(0.80, 0.001, a_folds, np.mean(b_folds), b_folds, 0.70)
    gate = p15.check_acceptance_gate(cv)
    assert gate["criterion_2_fold_consistency"]["passed"] is False
    assert gate["gate_passed"] is False


def test_gate_passes_only_when_all_five_criteria_hold() -> None:
    a_folds = [0.80, 0.81, 0.79, 0.80, 0.80]
    b_folds = [0.90, 0.91, 0.89, 0.90, 0.90]
    cv = _fake_cv_eval(float(np.mean(a_folds)), float(np.std(a_folds)), a_folds, float(np.mean(b_folds)), b_folds, 0.82)
    gate = p15.check_acceptance_gate(cv)
    assert gate["gate_passed"] is True
    assert all(gate[k] is True or (isinstance(gate[k], dict) and gate[k]["passed"]) for k in
               ("criterion_1_beats_baseline_mean_cv", "criterion_2_fold_consistency", "criterion_4_beats_matched_control", "criterion_5_practical_significance"))


# --------------------------------------------------------------------------
# 7. Prediction complementarity diagnostics
# --------------------------------------------------------------------------


def test_complementarity_analysis_schema_and_bounds() -> None:
    rng = np.random.RandomState(0)
    n = 300
    y = rng.randint(0, 2, n)
    oof_df = pd.DataFrame({"p_xgb": rng.uniform(0, 1, n), "p_quantum": rng.uniform(0, 1, n), "p_control": rng.uniform(0, 1, n), "y": y})
    result = p15.prediction_complementarity_analysis(oof_df)
    for key in ("pearson_corr_p_xgb_vs_p_quantum", "spearman_corr_p_xgb_vs_p_quantum", "prediction_disagreement_rate_at_0.5",
                "partial_corr_p_quantum_vs_y_given_p_xgb", "partial_corr_p_control_vs_y_given_p_xgb", "interpretation"):
        assert key in result
    assert -1.0 <= result["pearson_corr_p_xgb_vs_p_quantum"] <= 1.0
    assert 0.0 <= result["prediction_disagreement_rate_at_0.5"] <= 1.0


def test_complementarity_detects_perfect_correlation() -> None:
    rng = np.random.RandomState(0)
    n = 100
    p_xgb = rng.uniform(0, 1, n)
    oof_df = pd.DataFrame({"p_xgb": p_xgb, "p_quantum": p_xgb.copy(), "p_control": rng.uniform(0, 1, n), "y": rng.randint(0, 2, n)})
    result = p15.prediction_complementarity_analysis(oof_df)
    assert result["pearson_corr_p_xgb_vs_p_quantum"] == pytest.approx(1.0, abs=1e-9)
    assert result["prediction_disagreement_rate_at_0.5"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# 8. Prediction interface + batch consistency (fakes -- no real pipeline needed)
# --------------------------------------------------------------------------


class _FakeSharedPipeline:
    def transform(self, X):
        return X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)


class _FakeQuantumPipeline:
    def transform(self, X):
        return np.asarray(X)[:, :4]


class _FakeXGB:
    def predict_proba(self, X):
        X = np.asarray(X)
        p1 = 1 / (1 + np.exp(-X.sum(axis=1)))
        return np.column_stack([1 - p1, p1])


class _FakeQuantumModule:
    def score(self, X, params):
        return np.asarray(X).sum(axis=1) * 0.1


class _FakeMetaLearner:
    def predict_proba(self, X):
        X = np.asarray(X)
        p1 = 1 / (1 + np.exp(-(0.7 * X[:, 0] + 0.3 * X[:, 1])))
        return np.column_stack([1 - p1, p1])


class _FakeFeatureGroups:
    all_columns = ["f0", "f1", "f2", "f3", "f4", "f5"]


def _fake_deployed() -> dict:
    return {
        "feature_groups": _FakeFeatureGroups(), "shared_pipeline": _FakeSharedPipeline(),
        "quantum_pipeline": _FakeQuantumPipeline(), "xgb_final": _FakeXGB(),
        "quantum_module": _FakeQuantumModule(), "quantum_weights": np.zeros(1), "meta_learner": _FakeMetaLearner(),
    }


def test_predict_patient_risk_schema_and_bounds() -> None:
    deployed = _fake_deployed()
    row = pd.Series({c: 0.1 * i for i, c in enumerate(_FakeFeatureGroups.all_columns)})
    result = p15.predict_patient_risk(row, deployed)
    for key in ("classical_backbone_risk", "quantum_learner_risk", "final_stacked_risk", "risk_category", "model_version", "inference_latency_ms"):
        assert key in result
    assert 0.0 <= result["classical_backbone_risk"] <= 1.0
    assert 0.0 <= result["quantum_learner_risk"] <= 1.0
    assert 0.0 <= result["final_stacked_risk"] <= 1.0
    assert result["risk_category"] in ("low", "moderate", "high")


def test_predict_batch_matches_predict_patient_risk_for_the_same_row() -> None:
    deployed = _fake_deployed()
    row_dict = {c: 0.1 * i for i, c in enumerate(_FakeFeatureGroups.all_columns)}
    row = pd.Series(row_dict)
    single = p15.predict_patient_risk(row, deployed)

    df = pd.DataFrame([row_dict])
    batch = p15.predict_batch(df, deployed)
    # predict_patient_risk rounds to 4 decimals for display; predict_batch
    # does not -- compare with a tolerance that accommodates that rounding.
    assert batch.loc[0, "classical_backbone_risk"] == pytest.approx(single["classical_backbone_risk"], abs=1e-3)
    assert batch.loc[0, "quantum_learner_risk"] == pytest.approx(single["quantum_learner_risk"], abs=1e-3)
    assert batch.loc[0, "final_stacked_risk"] == pytest.approx(single["final_stacked_risk"], abs=1e-3)


# --------------------------------------------------------------------------
# 9. Real artifacts, if present -- the critical protocol checks
# --------------------------------------------------------------------------


def test_test_set_untouched_unless_cv_gate_passed_if_present() -> None:
    path = p15.RESULTS_ROOT / "phase15_summary.json"
    if not path.is_file():
        pytest.skip("Phase 15 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    gate = summary["acceptance_gate"]
    stability = summary.get("stability_check")
    if not gate["gate_passed"] or (stability is not None and not stability["consistent_across_seeds"]):
        assert summary["test_evaluation"] is None
        assert summary["statistics"] is None
        assert summary["decision"] != "A_QUANTUM_COMPLEMENTARITY_DEMONSTRATED"
    else:
        assert summary["test_evaluation"] is not None


def test_test_predictions_file_exists_iff_test_evaluation_ran_if_present() -> None:
    summary_path = p15.RESULTS_ROOT / "phase15_summary.json"
    if not summary_path.is_file():
        pytest.skip("Phase 15 has not been run yet")
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    test_pred_path = p15.RESULTS_ROOT / "test_predictions.csv"
    assert test_pred_path.is_file() == (summary["test_evaluation"] is not None)


def test_comparison_set_fingerprint_verified_if_present() -> None:
    path = p15.RESULTS_ROOT / "phase15_summary.json"
    if not path.is_file():
        pytest.skip("Phase 15 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["comparison_set"]["fingerprint"] == EXPECTED_FINGERPRINT
    assert summary["comparison_set"]["matches_expected"] is True


def test_oof_predictions_csv_schema_if_present() -> None:
    path = p15.RESULTS_ROOT / "oof_predictions.csv"
    if not path.is_file():
        pytest.skip("Phase 15 has not been run yet")
    df = pd.read_csv(path)
    for col in ("id", "y", "fold", "p_xgb", "p_quantum", "p_control"):
        assert col in df.columns
    assert df["id"].nunique() == len(df)
    assert df[["p_xgb", "p_quantum", "p_control"]].apply(lambda s: s.between(0, 1).all()).all()


def test_decision_is_one_of_the_three_defined_outcomes_if_present() -> None:
    path = p15.RESULTS_ROOT / "phase15_summary.json"
    if not path.is_file():
        pytest.skip("Phase 15 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["decision"] in (
        "A_QUANTUM_COMPLEMENTARITY_DEMONSTRATED",
        "B_COMPLEMENTARY_SIGNAL_BUT_NO_MEANINGFUL_GAIN",
        "C_NO_QUANTUM_CONTRIBUTION",
    )


def test_prior_phase_artifacts_untouched_if_present() -> None:
    from src.data.inspect_dataset import find_project_root

    root = find_project_root() / "results" / "large_dataset"
    for path, expected_len in [
        (root / "phase9_classical_benchmark" / "predictions.csv", 200),
        (root / "phase11_full_scale" / "predictions.csv", 200),
        (root / "phase12_quantum_representation" / "predictions.csv", 200),
        (root / "phase13_hybrid_system" / "predictions.csv", 200),
    ]:
        if path.is_file():
            assert len(pd.read_csv(path)) == expected_len
