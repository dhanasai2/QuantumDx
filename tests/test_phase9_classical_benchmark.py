"""Tests for src.large_dataset.phase9_classical_benchmark.

Covers quantum-prediction reuse (never retrains QSVM/VQC), decision-rule
correctness, and output/artifact schema (if the real benchmark has run).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.large_dataset import phase9_classical_benchmark as exp

EXPECTED_FINGERPRINT = "96eac11a8394b87e"


def test_load_phase8a_predictions_requires_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "PHASE8A_PREDICTIONS_PATH", tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        exp.load_phase8a_quantum_predictions()


def test_load_phase8a_predictions_rejects_wrong_fingerprint(tmp_path, monkeypatch) -> None:
    fake = pd.DataFrame({"id": [1, 2, 3], "y_true": [0, 1, 0]})
    path = tmp_path / "predictions.csv"
    fake.to_csv(path, index=False)
    monkeypatch.setattr(exp, "PHASE8A_PREDICTIONS_PATH", path)
    with pytest.raises(ValueError, match="fingerprint"):
        exp.load_phase8a_quantum_predictions()


def test_load_phase8a_predictions_matches_source_if_present() -> None:
    if not exp.PHASE8A_PREDICTIONS_PATH.is_file():
        pytest.skip("Phase 8A has not been run yet")
    df = exp.load_phase8a_quantum_predictions()
    assert len(df) == 200
    for col in ["baseline_proba", "mi_adaptive_proba", "label_aware_proba"]:
        assert col in df.columns


# --------------------------------------------------------------------------
# Real artifacts, if present
# --------------------------------------------------------------------------


def test_predictions_csv_has_all_seven_models_if_present() -> None:
    path = exp.RESULTS_ROOT / "predictions.csv"
    if not path.is_file():
        pytest.skip("Phase 9 has not been run yet")
    df = pd.read_csv(path)
    assert len(df) == 200
    expected_models = ["logistic_regression", "rbf_svm", "random_forest", "xgboost",
                        "baseline", "mi_adaptive", "label_aware"]
    for key in expected_models:
        assert f"{key}_proba" in df.columns
        assert df[f"{key}_proba"].between(0.0, 1.0).all()


def test_comparison_table_has_expected_columns_if_present() -> None:
    path = exp.RESULTS_ROOT / "comparison_table.csv"
    if not path.is_file():
        pytest.skip("Phase 9 has not been run yet")
    df = pd.read_csv(path)
    expected = ["Model", "ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "Accuracy", "F1", "Runtime (s)"]
    assert list(df.columns) == expected
    assert len(df) == 7


def test_classical_runtime_reflects_full_grid_search_not_just_refit_if_present() -> None:
    """Regression test for a real bug caught during development: the
    comparison table's Runtime column must be the full grid-search wall
    time, not ModelResult.fit_time_seconds (which only times the final
    refit and would be 1-2 orders of magnitude too small)."""
    summary_path = exp.RESULTS_ROOT / "phase9_summary.json"
    table_path = exp.RESULTS_ROOT / "comparison_table.csv"
    if not (summary_path.is_file() and table_path.is_file()):
        pytest.skip("Phase 9 has not been run yet")
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    table = pd.read_csv(table_path).set_index("Model")
    per_model_seconds = summary["runtime"]["classical_per_model_seconds"]
    name_map = {"logistic_regression": "Logistic Regression", "xgboost": "XGBoost"}
    for key, display in name_map.items():
        assert table.loc[display, "Runtime (s)"] == pytest.approx(per_model_seconds[key], rel=1e-6)


def test_decision_rule_matches_measured_deltas_if_present() -> None:
    path = exp.RESULTS_ROOT / "phase9_summary.json"
    if not path.is_file():
        pytest.skip("Phase 9 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    decision = summary["decision"]
    delta = decision["delta_roc_auc_quantum_minus_classical"]
    significant = decision["delong_p_holm"] < 0.05 and not decision["bootstrap_ci_includes_zero"]
    if decision["outcome"] == "A":
        assert delta > 0 and significant
    elif decision["outcome"] == "B":
        assert delta > 0 and not significant
    elif decision["outcome"] == "C":
        assert delta <= 0 and significant


def test_holm_correction_applied_across_all_three_comparisons_if_present() -> None:
    path = exp.RESULTS_ROOT / "statistics.json"
    if not path.is_file():
        pytest.skip("Phase 9 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        stats = json.load(fh)
    assert len(stats) == 3
    for comp in stats.values():
        assert "delong_p_holm" in comp
        assert comp["delong_p_holm"] >= comp["delong"]["p_value"] - 1e-12  # Holm never decreases p


def test_prior_phase_artifacts_untouched_if_present() -> None:
    from src.data.inspect_dataset import find_project_root

    root = find_project_root() / "results" / "large_dataset"
    for path, expected_len in [
        (root / "stage_d" / "predictions.csv", 200),
        (root / "adaptive_robustness" / "per_seed_results.csv", 5),
        (root / "phase8a_label_aware" / "predictions.csv", 200),
        (root / "phase8b_vqc" / "predictions.csv", 200),
    ]:
        if path.is_file():
            assert len(pd.read_csv(path)) == expected_len
