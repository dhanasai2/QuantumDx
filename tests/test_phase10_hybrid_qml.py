"""Tests for src.large_dataset.phase10_hybrid_qml.

Covers reference-model reuse (never retrains LR/QSVM), decision-rule
correctness, the critical control-vs-hybrid diagnostic, and output/
artifact schema (if the real experiment has run).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.large_dataset import phase10_hybrid_qml as exp

EXPECTED_FINGERPRINT = "96eac11a8394b87e"


def test_load_phase9_references_requires_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "PHASE9_PREDICTIONS_PATH", tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        exp.load_phase9_references()


def test_load_phase9_references_rejects_wrong_fingerprint(tmp_path, monkeypatch) -> None:
    fake = pd.DataFrame({"id": [1, 2, 3], "y_true": [0, 1, 0]})
    path = tmp_path / "predictions.csv"
    fake.to_csv(path, index=False)
    monkeypatch.setattr(exp, "PHASE9_PREDICTIONS_PATH", path)
    with pytest.raises(ValueError, match="fingerprint"):
        exp.load_phase9_references()


def test_load_phase9_references_matches_source_if_present() -> None:
    if not exp.PHASE9_PREDICTIONS_PATH.is_file():
        pytest.skip("Phase 9 has not been run yet")
    df = exp.load_phase9_references()
    assert len(df) == 200
    for col in ["logistic_regression_proba", "baseline_proba", "mi_adaptive_proba"]:
        assert col in df.columns


# --------------------------------------------------------------------------
# Real artifacts, if present
# --------------------------------------------------------------------------


def test_predictions_csv_has_all_five_models_if_present() -> None:
    path = exp.RESULTS_ROOT / "predictions.csv"
    if not path.is_file():
        pytest.skip("Phase 10 has not been run yet")
    df = pd.read_csv(path)
    assert len(df) == 200
    for key in ["classical_lr", "baseline_qsvm", "mi_adaptive_qsvm", "control_random_features", "hybrid"]:
        assert f"{key}_proba" in df.columns
        assert df[f"{key}_proba"].between(0.0, 1.0).all()


def test_decision_rule_matches_measured_delta_if_present() -> None:
    path = exp.RESULTS_ROOT / "phase10_summary.json"
    if not path.is_file():
        pytest.skip("Phase 10 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    decision = summary["decision"]
    delta = decision["delta_roc_auc_hybrid_minus_classical_lr"]
    significant = decision["delong_p_holm"] < 0.05 and not decision["bootstrap_ci_includes_zero"]
    if decision["outcome"] == "A":
        assert delta > 0 and significant
    elif decision["outcome"] == "B":
        assert delta > 0 and not significant
    elif decision["outcome"] == "C":
        assert not (delta > 0 and significant)


def test_hybrid_vs_control_comparison_present_if_run() -> None:
    """The non-quantum ablation control is the critical diagnostic for
    this phase -- it must always be computed, not merely optional."""
    path = exp.RESULTS_ROOT / "statistics.json"
    if not path.is_file():
        pytest.skip("Phase 10 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        stats = json.load(fh)
    assert "hybrid_vs_control" in stats
    assert "delong_p_holm" in stats["hybrid_vs_control"]


def test_explainability_exposes_both_classical_and_quantum_coefficients_if_present() -> None:
    path = exp.RESULTS_ROOT / "explainability.json"
    if not path.is_file():
        pytest.skip("Phase 10 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        expl = json.load(fh)
    coefs = expl["logistic_regression_coefficients"]
    assert any(k.startswith("pca_") for k in coefs)
    assert any(k.startswith("quantum_expZ_qubit") for k in coefs)
    assert len(coefs) == 8


def test_holm_correction_applied_across_four_comparisons_if_present() -> None:
    path = exp.RESULTS_ROOT / "statistics.json"
    if not path.is_file():
        pytest.skip("Phase 10 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        stats = json.load(fh)
    assert len(stats) == 4
    for comp in stats.values():
        assert comp["delong_p_holm"] >= comp["delong"]["p_value"] - 1e-12


def test_prior_phase_artifacts_untouched_if_present() -> None:
    from src.data.inspect_dataset import find_project_root

    root = find_project_root() / "results" / "large_dataset"
    for path, expected_len in [
        (root / "stage_d" / "predictions.csv", 200),
        (root / "adaptive_robustness" / "per_seed_results.csv", 5),
        (root / "phase8a_label_aware" / "predictions.csv", 200),
        (root / "phase8b_vqc" / "predictions.csv", 200),
        (root / "phase9_classical_benchmark" / "predictions.csv", 200),
    ]:
        if path.is_file():
            assert len(pd.read_csv(path)) == expected_len
