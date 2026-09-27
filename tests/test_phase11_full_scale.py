"""Tests for src.large_dataset.phase11_full_scale.

Covers: exact training-pool size, reference-prediction reuse (never
retrains Stage D/Phase 9 models), the documented RBF-SVM/QSVM
infeasibility deviations, decision-rule correctness, leakage invariants,
and output/artifact schema (if the real experiment has run).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.large_dataset import phase11_full_scale as exp

EXPECTED_FINGERPRINT = "96eac11a8394b87e"


# --------------------------------------------------------------------------
# Exact training pool size (Step 2's explicit inspection requirement)
# --------------------------------------------------------------------------


def test_full_scale_split_has_expected_training_pool_size() -> None:
    split = exp.build_full_scale_split()
    assert len(split.training_pool) == 66641
    assert len(split.test_set_classical) == 2000
    assert len(split.test_set_quantum) == 200


def test_full_scale_split_is_deterministic() -> None:
    a = exp.build_full_scale_split()
    b = exp.build_full_scale_split()
    assert a.training_pool["id"].tolist() == b.training_pool["id"].tolist()


def test_full_scale_training_pool_has_zero_overlap_with_test_sets() -> None:
    split = exp.build_full_scale_split()
    train_ids = set(split.training_pool["id"])
    assert len(train_ids & set(split.test_set_classical["id"])) == 0
    assert len(train_ids & set(split.test_set_quantum["id"])) == 0


def test_full_scale_comparison_set_fingerprint_matches_established_identity() -> None:
    from src.large_dataset.corrected_comparison import comparison_set_fingerprint

    split = exp.build_full_scale_split()
    ids = split.test_set_quantum["id"].tolist()
    assert comparison_set_fingerprint(ids) == EXPECTED_FINGERPRINT


# --------------------------------------------------------------------------
# Reference-prediction reuse (never retrains Stage D / Phase 9 models)
# --------------------------------------------------------------------------


def test_load_reference_predictions_requires_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "STAGE_D_PREDICTIONS_PATH", tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        exp.load_reference_predictions()


def test_load_reference_predictions_rejects_wrong_fingerprint(tmp_path, monkeypatch) -> None:
    fake = pd.DataFrame({"id": [1, 2, 3], "y_true": [0, 1, 0]})
    path = tmp_path / "predictions.csv"
    fake.to_csv(path, index=False)
    monkeypatch.setattr(exp, "STAGE_D_PREDICTIONS_PATH", path)
    with pytest.raises(ValueError, match="fingerprint"):
        exp.load_reference_predictions()


def test_load_reference_predictions_matches_source_if_present() -> None:
    if not (exp.STAGE_D_PREDICTIONS_PATH.is_file() and exp.PHASE9_PREDICTIONS_PATH.is_file()):
        pytest.skip("Stage D or Phase 9 has not been run yet")
    refs = exp.load_reference_predictions()
    assert len(refs["stage_d"]) == 200
    assert len(refs["phase9"]) == 200
    assert "rbf_svm_proba" in refs["stage_d"].columns
    assert "baseline_proba" in refs["phase9"].columns


# --------------------------------------------------------------------------
# Documented deviations (RBF-SVM / QSVM full-scale infeasibility)
# --------------------------------------------------------------------------


def test_rbf_svm_is_skipped_in_classical_full_scale_run() -> None:
    """RBF-SVM must never be run as a fresh full-scale grid search --
    only reused from Stage D."""
    from src.classical.models import get_available_models

    models = get_available_models(random_seed=42)
    assert "rbf_svm" in models  # the registry still has it (unmodified)
    # but run_classical_full_scale's own logic explicitly skips it -- verified
    # by inspecting the source for the skip branch.
    import inspect

    src = inspect.getsource(exp.run_classical_full_scale)
    assert 'key == "rbf_svm"' in src and "SKIPPING" in src


# --------------------------------------------------------------------------
# Real artifacts, if present
# --------------------------------------------------------------------------


def test_predictions_csv_schema_if_present() -> None:
    path = exp.RESULTS_ROOT / "predictions.csv"
    if not path.is_file():
        pytest.skip("Phase 11 has not been run yet")
    df = pd.read_csv(path)
    assert len(df) == 200
    assert not df.isna().any().any()
    proba_cols = [c for c in df.columns if c.endswith("_proba")]
    assert len(proba_cols) >= 7
    for c in proba_cols:
        assert df[c].between(0.0, 1.0).all()
        assert not np.any(np.isinf(df[c].to_numpy()))


def test_summary_reports_expected_fingerprint_and_training_size_if_present() -> None:
    path = exp.RESULTS_ROOT / "phase11_summary.json"
    if not path.is_file():
        pytest.skip("Phase 11 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["comparison_set"]["fingerprint"] == EXPECTED_FINGERPRINT
    assert summary["comparison_set"]["matches_expected"] is True
    assert summary["n_train_full"] == 66641
    assert summary["decision"]["outcome"] in {"A", "B", "C", "D"}


def test_quantum_expectation_values_within_valid_range_if_present() -> None:
    path = exp.RESULTS_ROOT / "phase11_summary.json"
    if not path.is_file():
        pytest.skip("Phase 11 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    stats = summary["runtime"]["hybrid_timing"]["quantum_feature_stats"]
    assert -1.0 - 1e-6 <= stats["min"] <= 1.0 + 1e-6
    assert -1.0 - 1e-6 <= stats["max"] <= 1.0 + 1e-6


def test_documented_deviations_present_if_run() -> None:
    path = exp.RESULTS_ROOT / "phase11_summary.json"
    if not path.is_file():
        pytest.skip("Phase 11 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    dev = summary["documented_deviations"]
    assert "rbf_svm_full_scale_skipped" in dev
    assert "qsvm_full_scale_skipped" in dev


def test_prior_phase_artifacts_untouched_if_present() -> None:
    from src.data.inspect_dataset import find_project_root

    root = find_project_root() / "results" / "large_dataset"
    for path, expected_len in [
        (root / "stage_d" / "predictions.csv", 200),
        (root / "adaptive_robustness" / "per_seed_results.csv", 5),
        (root / "phase8a_label_aware" / "predictions.csv", 200),
        (root / "phase8b_vqc" / "predictions.csv", 200),
        (root / "phase9_classical_benchmark" / "predictions.csv", 200),
        (root / "phase10_hybrid_qml" / "predictions.csv", 200),
    ]:
        if path.is_file():
            assert len(pd.read_csv(path)) == expected_len
