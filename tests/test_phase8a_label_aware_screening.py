"""Tests for src.large_dataset.phase8a_label_aware_screening.

Schema/consistency checks on the real screening artifacts (if present),
plus a structural guard that the screening split never overlaps the fixed
test sets.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.large_dataset import phase8a_label_aware_screening as exp

EXPECTED_FINGERPRINT = "96eac11a8394b87e"


def test_screening_split_has_expected_size_and_no_test_overlap() -> None:
    screen = exp.build_screening_split()
    assert len(screen.train_df) == exp.SCREEN_N_TRAIN
    train_ids = set(screen.train_df["id"])
    assert len(train_ids & set(screen.split.test_set_classical["id"])) == 0
    assert len(train_ids & set(screen.split.test_set_quantum["id"])) == 0


def test_screening_split_is_deterministic() -> None:
    a = exp.build_screening_split()
    b = exp.build_screening_split()
    assert a.train_df["id"].tolist() == b.train_df["id"].tolist()


def test_predictions_csv_schema_if_present() -> None:
    path = exp.RESULTS_ROOT / "predictions.csv"
    if not path.is_file():
        pytest.skip("Phase 8A screening has not been run yet")
    df = pd.read_csv(path)
    assert len(df) == 200
    for c in ["baseline_proba", "mi_adaptive_proba", "label_aware_proba"]:
        assert df[c].between(0.0, 1.0).all()
    assert df["id"].duplicated().sum() == 0


def test_summary_reports_expected_fingerprint_and_n_train_if_present() -> None:
    path = exp.RESULTS_ROOT / "phase8a_summary.json"
    if not path.is_file():
        pytest.skip("Phase 8A screening has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["n_train"] == 2000
    assert summary["comparison_set"]["fingerprint"] == EXPECTED_FINGERPRINT
    assert summary["comparison_set"]["matches_expected"] is True


def test_both_variants_use_same_pair_budget_as_baseline_if_present() -> None:
    path = exp.RESULTS_ROOT / "phase8a_summary.json"
    if not path.is_file():
        pytest.skip("Phase 8A screening has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["mi_adaptive_map"]["n_pairs"] == 3
    assert summary["label_aware_map"]["n_pairs"] == 3


def test_delta_sign_matches_metrics_csv_if_present() -> None:
    stats_path = exp.RESULTS_ROOT / "statistics.json"
    metrics_path = exp.RESULTS_ROOT / "metrics.csv"
    if not (stats_path.is_file() and metrics_path.is_file()):
        pytest.skip("Phase 8A screening has not been run yet")
    with open(stats_path, encoding="utf-8") as fh:
        stats = json.load(fh)
    metrics = pd.read_csv(metrics_path).set_index("model_key")
    for key in ["mi_adaptive", "label_aware"]:
        expected = metrics.loc[key, "roc_auc"] - metrics.loc["baseline", "roc_auc"]
        actual = stats["comparisons_vs_baseline"][key]["delong"]["delta"]
        assert actual == pytest.approx(expected, abs=1e-9)


def test_stage_and_phase7_artifacts_untouched_if_present() -> None:
    """This screening experiment must never modify Stage D or Phase 7 outputs."""
    from src.data.inspect_dataset import find_project_root

    root = find_project_root() / "results" / "large_dataset"
    stage_d = root / "stage_d" / "predictions.csv"
    phase7 = root / "adaptive_robustness" / "per_seed_results.csv"
    if stage_d.is_file():
        assert len(pd.read_csv(stage_d)) == 200
    if phase7.is_file():
        assert len(pd.read_csv(phase7)) == 5
