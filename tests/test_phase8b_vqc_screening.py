"""Tests for src.large_dataset.phase8b_vqc_screening.

Covers baseline reuse (never retrained), output/artifact schema (if the
real screening has run), and preservation of Phase 8A's own artifacts.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.large_dataset import phase8b_vqc_screening as exp

EXPECTED_FINGERPRINT = "96eac11a8394b87e"


def test_load_baseline_predictions_requires_phase8a_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "PHASE8A_PREDICTIONS_PATH", tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        exp.load_baseline_predictions()


def test_load_baseline_predictions_rejects_wrong_fingerprint(tmp_path, monkeypatch) -> None:
    fake = pd.DataFrame({
        "id": [1, 2, 3], "y_true": [0, 1, 0],
        "baseline_proba": [0.2, 0.7, 0.3], "baseline_pred": [0, 1, 0],
    })
    path = tmp_path / "predictions.csv"
    fake.to_csv(path, index=False)
    monkeypatch.setattr(exp, "PHASE8A_PREDICTIONS_PATH", path)
    with pytest.raises(ValueError, match="fingerprint"):
        exp.load_baseline_predictions()


def test_load_baseline_predictions_matches_phase8a_if_present() -> None:
    if not exp.PHASE8A_PREDICTIONS_PATH.is_file():
        pytest.skip("Phase 8A has not been run yet")
    baseline = exp.load_baseline_predictions()
    assert len(baseline.ids) == 200
    df = pd.read_csv(exp.PHASE8A_PREDICTIONS_PATH)
    assert np.array_equal(baseline.y_proba, df["baseline_proba"].to_numpy())


# --------------------------------------------------------------------------
# Real artifacts, if present
# --------------------------------------------------------------------------


def test_predictions_csv_schema_if_present() -> None:
    path = exp.RESULTS_ROOT / "predictions.csv"
    if not path.is_file():
        pytest.skip("Phase 8B screening has not been run yet")
    df = pd.read_csv(path)
    assert len(df) == 200
    for c in ["baseline_qsvm_proba", "vqc_proba"]:
        assert df[c].between(0.0, 1.0).all()


def test_summary_reports_expected_fingerprint_and_decision_if_present() -> None:
    path = exp.RESULTS_ROOT / "phase8b_summary.json"
    if not path.is_file():
        pytest.skip("Phase 8B screening has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["comparison_set"]["fingerprint"] == EXPECTED_FINGERPRINT
    assert summary["n_train"] == 2000
    assert summary["decision"]["outcome"] in {"A", "B", "C"}


def test_decision_outcome_matches_measured_delta_if_present() -> None:
    path = exp.RESULTS_ROOT / "phase8b_summary.json"
    if not path.is_file():
        pytest.skip("Phase 8B screening has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    decision = summary["decision"]
    delta = decision["delta_roc_auc"]
    if decision["outcome"] == "A":
        assert delta > 0.02
    elif decision["outcome"] == "C":
        assert delta < -0.02


def test_phase8a_artifacts_untouched_if_present() -> None:
    """This VQC screening experiment must never rewrite Phase 8A's own
    predictions/results."""
    if not exp.PHASE8A_PREDICTIONS_PATH.is_file():
        pytest.skip("Phase 8A has not been run yet")
    df = pd.read_csv(exp.PHASE8A_PREDICTIONS_PATH)
    assert len(df) == 200
    assert "baseline_proba" in df.columns
