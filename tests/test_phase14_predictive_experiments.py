"""Tests for src.large_dataset.phase14_predictive_experiments.

Covers: engineered-feature correctness, CV-scoring correctness, the
critical "test set untouched unless CV supports it" protocol, and
output/artifact schema (if the real experiment has run).
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from src.large_dataset import phase14_predictive_experiments as exp

EXPECTED_FINGERPRINT = "96eac11a8394b87e"


# --------------------------------------------------------------------------
# 1. Engineered features are clinically standard and correctly computed
# --------------------------------------------------------------------------


def test_engineered_features_are_computed_correctly() -> None:
    df = pd.DataFrame({"weight": [80.0], "height": [160.0], "ap_hi": [140.0], "ap_lo": [90.0],
                        "age_years": [50.0], "cholesterol": [2]})
    out = exp.add_engineered_features(df)
    assert out["bmi"].iloc[0] == pytest.approx(80.0 / (1.6 ** 2))
    assert out["pulse_pressure"].iloc[0] == pytest.approx(50.0)
    assert out["mean_arterial_pressure"].iloc[0] == pytest.approx(90.0 + 50.0 / 3)
    assert out["age_x_cholesterol"].iloc[0] == pytest.approx(100.0)


def test_engineered_features_preserve_original_columns() -> None:
    df = pd.DataFrame({"weight": [70.0], "height": [170.0], "ap_hi": [120.0], "ap_lo": [80.0],
                        "age_years": [40.0], "cholesterol": [1], "gluc": [1]})
    out = exp.add_engineered_features(df)
    for col in df.columns:
        assert col in out.columns
    assert not out.equals(df)  # new columns were actually added


# --------------------------------------------------------------------------
# 2. CV scoring: training-data-only, deterministic, no test parameter
# --------------------------------------------------------------------------


def test_cv_scores_accepts_no_test_parameter() -> None:
    for fn in (exp._cv_scores, exp.stage2_backbone_comparison, exp.stage3_feature_engineering):
        for p in inspect.signature(fn).parameters:
            assert "test" not in p.lower()


def test_cv_scores_deterministic_for_fixed_seed() -> None:
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 1, size=(100, 4))
    y = rng.randint(0, 2, 100)
    r1 = exp._cv_scores(exp._build_xgb, X, y, seed=42, folds=3)
    r2 = exp._cv_scores(exp._build_xgb, X, y, seed=42, folds=3)
    assert r1 == r2


def test_cv_scores_reports_mean_and_std_for_every_metric() -> None:
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 1, size=(100, 4))
    y = rng.randint(0, 2, 100)
    scores = exp._cv_scores(exp._build_hgb, X, y, seed=42, folds=3)
    for metric in ("roc_auc", "pr_auc", "sensitivity", "specificity", "accuracy", "f1"):
        assert f"{metric}_mean" in scores
        assert f"{metric}_std" in scores
        assert 0.0 <= scores[f"{metric}_mean"] <= 1.0


# --------------------------------------------------------------------------
# 3. Diagnostic analysis (read-only, uses Phase 11's existing predictions)
# --------------------------------------------------------------------------


def test_diagnostic_analysis_does_not_modify_phase11_file_if_present() -> None:
    if not exp.PHASE11_PREDICTIONS_PATH.is_file():
        pytest.skip("Phase 11 has not been run yet")
    before = exp.PHASE11_PREDICTIONS_PATH.read_bytes()
    exp.stage1_diagnostic_analysis()
    after = exp.PHASE11_PREDICTIONS_PATH.read_bytes()
    assert before == after


def test_diagnostic_analysis_rejects_wrong_fingerprint(tmp_path, monkeypatch) -> None:
    fake = pd.DataFrame({"id": [1, 2, 3], "y_true": [0, 1, 0], "xgboost_proba": [0.2, 0.7, 0.3]})
    path = tmp_path / "predictions.csv"
    fake.to_csv(path, index=False)
    monkeypatch.setattr(exp, "PHASE11_PREDICTIONS_PATH", path)
    with pytest.raises(ValueError, match="fingerprint"):
        exp.stage1_diagnostic_analysis()


def test_diagnostic_analysis_schema_if_present() -> None:
    if not exp.PHASE11_PREDICTIONS_PATH.is_file():
        pytest.skip("Phase 11 has not been run yet")
    diagnosis = exp.stage1_diagnostic_analysis()
    assert set(diagnosis["confidence_by_outcome"].keys()) == {"TP", "TN", "FP", "FN"}
    assert "threshold_analysis" in diagnosis
    assert "interpretation" in diagnosis


# --------------------------------------------------------------------------
# 4. Real artifacts, if present -- the critical protocol check
# --------------------------------------------------------------------------


def test_test_set_only_touched_when_cv_supports_a_candidate_if_present() -> None:
    """The single most important invariant this phase must satisfy: if CV
    evidence did not show a consistent, seed-stable improvement, the test
    set must NOT have been evaluated at all."""
    path = exp.RESULTS_ROOT / "phase14_experiments_summary.json"
    if not path.is_file():
        pytest.skip("Phase 14 experiments have not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    if not summary["consistent_improvement_in_cv"]:
        assert summary["test_evaluation"] is None
        assert summary["statistics"] is None
        assert summary["decision"] == "B_NO_DEFENSIBLE_IMPROVEMENT_FOUND"
    else:
        assert summary["test_evaluation"] is not None


def test_comparison_set_fingerprint_verified_if_present() -> None:
    path = exp.RESULTS_ROOT / "phase14_experiments_summary.json"
    if not path.is_file():
        pytest.skip("Phase 14 experiments have not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["comparison_set"]["fingerprint"] == EXPECTED_FINGERPRINT
    assert summary["comparison_set"]["matches_expected"] is True


def test_cv_comparison_table_schema_if_present() -> None:
    path = exp.RESULTS_ROOT / "cv_comparison_table.csv"
    if not path.is_file():
        pytest.skip("Phase 14 experiments have not been run yet")
    df = pd.read_csv(path)
    assert "candidate" in df.columns
    assert "roc_auc_mean" in df.columns
    assert "roc_auc_std" in df.columns
    assert len(df) >= 4  # at least the backbone + feature-engineering candidates


def test_prior_phase_artifacts_untouched_if_present() -> None:
    from src.data.inspect_dataset import find_project_root

    root = find_project_root() / "results" / "large_dataset"
    for path, expected_len in [
        (root / "phase9_classical_benchmark" / "predictions.csv", 200),
        (root / "phase11_full_scale" / "predictions.csv", 200),
        (root / "phase13_hybrid_system" / "predictions.csv", 200),
    ]:
        if path.is_file():
            assert len(pd.read_csv(path)) == expected_len
