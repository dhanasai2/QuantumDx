"""Tests for src.large_dataset.adaptive_qsvm_experiment.

Covers the orchestrator-level pieces of Section 12 not already exercised
in tests/test_adaptive_feature_map.py: baseline reuse (never retrained),
schema validity of persisted artifacts (if the real experiment has run),
and the delta-direction convention end to end.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.inspect_dataset import find_project_root
from src.large_dataset import adaptive_qsvm_experiment as exp

EXPECTED_FINGERPRINT = "96eac11a8394b87e"


def test_load_baseline_predictions_requires_stage_d_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "STAGE_D_PREDICTIONS_PATH", tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        exp.load_baseline_predictions()


def test_load_baseline_predictions_rejects_wrong_fingerprint(tmp_path, monkeypatch) -> None:
    fake = pd.DataFrame({
        "id": [1, 2, 3], "y_true": [0, 1, 0],
        "qsvm_proba": [0.2, 0.7, 0.3], "qsvm_pred": [0, 1, 0],
    })
    path = tmp_path / "predictions.csv"
    fake.to_csv(path, index=False)
    monkeypatch.setattr(exp, "STAGE_D_PREDICTIONS_PATH", path)
    with pytest.raises(ValueError, match="fingerprint"):
        exp.load_baseline_predictions()


def test_load_baseline_predictions_matches_stage_d_if_present() -> None:
    if not exp.STAGE_D_PREDICTIONS_PATH.is_file():
        pytest.skip("Stage D has not been run yet")
    baseline = exp.load_baseline_predictions()
    assert len(baseline.ids) == 200
    assert baseline.model_key == "baseline_qsvm"
    df = pd.read_csv(exp.STAGE_D_PREDICTIONS_PATH)
    assert np.array_equal(baseline.y_proba, df["qsvm_proba"].to_numpy())
    assert np.array_equal(baseline.y_pred, df["qsvm_pred"].to_numpy())


# --------------------------------------------------------------------------
# Real artifacts, if present
# --------------------------------------------------------------------------


def test_predictions_csv_schema_if_present() -> None:
    path = exp.RESULTS_ROOT / "predictions.csv"
    if not path.is_file():
        pytest.skip("Adaptive QSVM experiment has not been run yet")
    df = pd.read_csv(path)
    assert len(df) == 200
    for c in ["baseline_qsvm_proba", "adaptive_qsvm_proba"]:
        assert df[c].between(0.0, 1.0).all()
    assert df["id"].duplicated().sum() == 0


def test_summary_json_reports_expected_fingerprint_if_present() -> None:
    path = exp.RESULTS_ROOT / "adaptive_qsvm_summary.json"
    if not path.is_file():
        pytest.skip("Adaptive QSVM experiment has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["comparison_set"]["fingerprint"] == EXPECTED_FINGERPRINT
    assert summary["comparison_set"]["matches_expected"] is True
    assert summary["n_train"] == 20000
    assert summary["adaptive_map"]["n_pairs"] == 3


def test_statistics_delta_definition_and_sign_if_present() -> None:
    path = exp.RESULTS_ROOT / "statistics.json"
    if not path.is_file():
        pytest.skip("Adaptive QSVM experiment has not been run yet")
    with open(path, encoding="utf-8") as fh:
        stats = json.load(fh)
    assert "Adaptive QSVM - Baseline QSVM" in stats["delta_definition"]
    metrics_path = exp.RESULTS_ROOT / "metrics.csv"
    metrics = pd.read_csv(metrics_path).set_index("model_key")
    expected_delta = metrics.loc["adaptive_qsvm", "roc_auc"] - metrics.loc["baseline_qsvm", "roc_auc"]
    assert stats["delong"]["delta"] == pytest.approx(expected_delta, abs=1e-9)


def test_stage_d_baseline_metrics_unchanged_by_this_experiment() -> None:
    """This experiment must never rewrite Stage D's own predictions.csv."""
    if not exp.STAGE_D_PREDICTIONS_PATH.is_file():
        pytest.skip("Stage D has not been run yet")
    df = pd.read_csv(exp.STAGE_D_PREDICTIONS_PATH)
    assert len(df) == 200
    assert "qsvm_proba" in df.columns  # still present, unmodified schema
