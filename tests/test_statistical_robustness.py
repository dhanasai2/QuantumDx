"""Tests for Phase 6A (src.large_dataset.statistical_robustness).

Covers Task 12's requirements: paired bootstrap correctness (via reuse of
already-tested corrected_comparison functions, plus new DeLong-specific
checks), test-set identity, no accidental test-set replacement, metric
calculation, multiple-comparison correction, and result schema validity.

Fast by design: DeLong is exercised on synthetic data with known
properties; the persisted-artifact tests skip gracefully if Phase 6A has
not been run yet (mirroring the existing test_saved_predictions_are_
aligned_by_id_if_present pattern in test_corrected_comparison.py).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.data.inspect_dataset import find_project_root
from src.large_dataset.statistical_robustness import (
    ALPHA,
    ALL_MODELS,
    CLASSICAL_MODELS,
    EXPECTED_FINGERPRINT,
    STAGES,
    delong_test,
    holm_bonferroni,
    load_stage_predictions,
    verify_cross_stage_consistency,
)

RESULTS_DIR = find_project_root() / "results" / "large_dataset" / "statistical_robustness"


# --------------------------------------------------------------------------
# DeLong correctness
# --------------------------------------------------------------------------


def test_delong_auc_matches_sklearn() -> None:
    rng = np.random.RandomState(0)
    y = np.repeat([0, 1], 100)
    proba_a = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 200), 0, 1)
    proba_b = np.clip(y * 0.2 + rng.normal(0.4, 0.2, 200), 0, 1)
    result = delong_test(y, proba_a, proba_b)
    assert result.auc_a == pytest.approx(roc_auc_score(y, proba_a), abs=1e-9)
    assert result.auc_b == pytest.approx(roc_auc_score(y, proba_b), abs=1e-9)


def test_delong_identical_scores_gives_zero_delta_and_p_one() -> None:
    rng = np.random.RandomState(1)
    y = np.repeat([0, 1], 50)
    proba = np.clip(y * 0.5 + rng.normal(0.25, 0.2, 100), 0, 1)
    result = delong_test(y, proba, proba)
    assert result.delta == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0, abs=1e-9)
    assert result.ci_low <= 0.0 <= result.ci_high


def test_delong_detects_a_clearly_better_model() -> None:
    rng = np.random.RandomState(2)
    y = np.repeat([0, 1], 100)
    strong = np.clip(y * 0.8 + rng.normal(0.1, 0.1, 200), 0, 1)
    weak = rng.uniform(0, 1, 200)
    result = delong_test(y, strong, weak)
    assert result.delta > 0
    assert result.p_value < 0.01
    assert result.ci_low > 0  # CI excludes zero


def test_delong_delta_is_antisymmetric() -> None:
    rng = np.random.RandomState(3)
    y = np.repeat([0, 1], 60)
    a = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 120), 0, 1)
    b = np.clip(y * 0.3 + rng.normal(0.35, 0.2, 120), 0, 1)
    ab = delong_test(y, a, b)
    ba = delong_test(y, b, a)
    assert ab.delta == pytest.approx(-ba.delta, abs=1e-9)
    assert ab.p_value == pytest.approx(ba.p_value, abs=1e-9)


def test_delong_requires_both_classes_present() -> None:
    y = np.ones(10)
    with pytest.raises(ValueError):
        delong_test(y, np.random.rand(10), np.random.rand(10))


# --------------------------------------------------------------------------
# Holm-Bonferroni correctness
# --------------------------------------------------------------------------


def test_holm_bonferroni_matches_hand_worked_example() -> None:
    """Classic textbook example: p = [0.01, 0.02, 0.03, 0.04], alpha=0.05.
    Step-down multipliers are [4,3,2,1] applied to sorted p in order, with
    running maximum enforced for monotonicity.
    """
    p = [0.01, 0.02, 0.03, 0.04]
    result = holm_bonferroni(p, alpha=0.05)
    adjusted = [r["adjusted_p"] for r in result]
    # sorted p * [4,3,2,1] = [0.04, 0.06, 0.06, 0.04] -> running max -> [0.04, 0.06, 0.06, 0.06]
    assert adjusted == pytest.approx([0.04, 0.06, 0.06, 0.06], abs=1e-9)
    assert [r["significant"] for r in result] == [True, False, False, False]


def test_holm_bonferroni_preserves_input_order() -> None:
    p = [0.04, 0.01, 0.03, 0.02]  # deliberately unsorted
    result = holm_bonferroni(p, alpha=0.05)
    # index 1 (p=0.01) must correspond to the smallest raw p in the output order
    assert result[1]["raw_p"] == 0.01
    assert result[0]["raw_p"] == 0.04


def test_holm_bonferroni_adjusted_p_never_exceeds_one() -> None:
    p = [0.5, 0.6, 0.7, 0.9]
    result = holm_bonferroni(p)
    assert all(r["adjusted_p"] <= 1.0 for r in result)


def test_holm_bonferroni_all_zero_p_are_all_significant() -> None:
    result = holm_bonferroni([0.0] * 5)
    assert all(r["significant"] for r in result)


# --------------------------------------------------------------------------
# Test-set identity (Task 9) -- exercised on persisted artifacts if present
# --------------------------------------------------------------------------


def _skip_if_missing():
    for n in STAGES.values():
        path = find_project_root() / "results" / "large_dataset" / "corrected_comparison" / f"stage_{n}" / "predictions.csv"
        if not path.is_file():
            pytest.skip(f"corrected comparison predictions for stage {n} not present")


def test_all_stage_predictions_share_the_expected_fingerprint() -> None:
    _skip_if_missing()
    for n in STAGES.values():
        df = load_stage_predictions(n)
        assert len(df) == 200


def test_cross_stage_consistency_passes_on_real_artifacts() -> None:
    _skip_if_missing()
    dfs = {n: load_stage_predictions(n) for n in STAGES.values()}
    verify_cross_stage_consistency(dfs)  # must not raise


def test_cross_stage_consistency_detects_id_mismatch() -> None:
    """A synthetic, deliberately-corrupted second stage must be rejected."""
    base = pd.DataFrame({"id": [1, 2, 3], "y_true": [0, 1, 0]})
    tampered = pd.DataFrame({"id": [1, 2, 4], "y_true": [0, 1, 0]})  # id 4 != 3
    with pytest.raises(ValueError, match="ids differ"):
        verify_cross_stage_consistency({1000: base, 5000: tampered})


def test_cross_stage_consistency_detects_label_mismatch() -> None:
    base = pd.DataFrame({"id": [1, 2, 3], "y_true": [0, 1, 0]})
    tampered = pd.DataFrame({"id": [1, 2, 3], "y_true": [0, 1, 1]})  # label flipped
    with pytest.raises(ValueError, match="y_true differs"):
        verify_cross_stage_consistency({1000: base, 5000: tampered})


def test_cross_stage_consistency_detects_duplicate_ids() -> None:
    dup = pd.DataFrame({"id": [1, 1, 2], "y_true": [0, 1, 0]})
    with pytest.raises(ValueError, match="duplicate ids"):
        verify_cross_stage_consistency({1000: dup})


def test_cross_stage_consistency_detects_out_of_range_proba() -> None:
    bad = pd.DataFrame({"id": [1, 2, 3], "y_true": [0, 1, 0], "qsvm_proba": [0.5, 1.2, 0.1]})
    with pytest.raises(ValueError, match="out-of-range"):
        verify_cross_stage_consistency({1000: bad})


def test_load_stage_predictions_rejects_wrong_fingerprint(tmp_path, monkeypatch) -> None:
    """A comparison set that does NOT match the established 200-row identity
    must be rejected, not silently used."""
    import src.large_dataset.statistical_robustness as mod

    fake_dir = tmp_path / "stage_1000"
    fake_dir.mkdir()
    fake_df = pd.DataFrame({"id": [999, 998, 997], "y_true": [0, 1, 0]})
    fake_df.to_csv(fake_dir / "predictions.csv", index=False)

    monkeypatch.setattr(mod, "CORRECTED_ROOT", tmp_path)
    with pytest.raises(ValueError, match="fingerprint"):
        mod.load_stage_predictions(1000)


# --------------------------------------------------------------------------
# Result schema validity (exercised on real artifacts if present)
# --------------------------------------------------------------------------


def test_auc_comparisons_csv_has_12_rows_and_expected_columns() -> None:
    path = RESULTS_DIR / "auc_comparisons.csv"
    if not path.is_file():
        pytest.skip("Phase 6A has not been run yet")
    df = pd.read_csv(path)
    assert len(df) == 12
    assert set(df["stage"].unique()) == {"A", "B", "C"}
    assert set(df["classical_model"].unique()) == set(CLASSICAL_MODELS)
    for col in ["delong_delta", "delong_p", "delong_p_holm", "bootstrap_delta", "mcnemar_p"]:
        assert col in df.columns
        assert df[col].notna().all()


def test_metric_confidence_intervals_csv_has_15_rows() -> None:
    path = RESULTS_DIR / "metric_confidence_intervals.csv"
    if not path.is_file():
        pytest.skip("Phase 6A has not been run yet")
    df = pd.read_csv(path)
    assert len(df) == 15  # 5 models x 3 stages
    assert set(df["model"].unique()) == set(ALL_MODELS)
    for metric in ["roc_auc", "pr_auc", "sensitivity", "specificity", "accuracy", "precision", "f1"]:
        assert (df[f"{metric}_ci_low"] <= df[metric]).all()
        assert (df[metric] <= df[f"{metric}_ci_high"]).all()


def test_statistical_summary_json_reports_expected_fingerprint() -> None:
    path = RESULTS_DIR / "statistical_summary.json"
    if not path.is_file():
        pytest.skip("Phase 6A has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["data_integrity"]["fingerprint"] == EXPECTED_FINGERPRINT
    assert summary["data_integrity"]["fingerprint_matches_expected"] is True
    assert summary["data_integrity"]["ids_identical_across_stages"] is True
    assert summary["n_primary_comparisons"] == 12
    assert 0 <= summary["significant_after_holm_delong"] <= 12


def test_gap_reduction_matches_expected_approximate_value() -> None:
    """Sanity-check the specific A-to-C gap-reduction figure against the
    independently-stated expectation (~52.6%), without hardcoding it as the
    source of truth -- the CSV/JSON outputs remain authoritative."""
    path = RESULTS_DIR / "statistical_summary.json"
    if not path.is_file():
        pytest.skip("Phase 6A has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["gap_analysis"]["A_to_C_pct"] == pytest.approx(52.6, abs=1.0)


def test_historical_phase5_and_uncorrected_stage_files_still_untouched() -> None:
    root = find_project_root() / "results"
    phase5 = root / "quantum" / "phase5" / "final_metrics.json"
    uncorrected = root / "large_dataset" / "stage_comparison.json"
    if phase5.is_file():
        with open(phase5, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["test_metrics"]["roc_auc"] == pytest.approx(0.7798, abs=5e-4)
    if uncorrected.is_file():
        with open(uncorrected, encoding="utf-8") as fh:
            json.load(fh)  # must still be valid, unmodified JSON
