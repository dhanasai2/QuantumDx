"""Tests for the post-Stage-C methodology correction
(src.large_dataset.corrected_comparison).

Covers the Section 26 requirements: exact test-set identity, no training
overlap, metric calculation, paired comparison calculation, bootstrap
reproducibility, prediction alignment by id, and confirmation that the
historical Phase 3-5 result files are untouched.

These tests are deliberately FAST -- they never run a full stage training
(which takes minutes to an hour). Statistical functions are exercised on
synthetic data with known properties; test-set identity is exercised on the
real (cheap) sampling code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.inspect_dataset import find_project_root
from src.large_dataset.cleaning import compute_cleaning_flags, split_modeling_subset
from src.large_dataset.corrected_comparison import (
    BOOTSTRAP_SEED,
    DECISION_THRESHOLD,
    bootstrap_metric_ci,
    mcnemar_exact,
    paired_bootstrap_delta,
    comparison_set_fingerprint,
)
from src.large_dataset.sampling import build_fixed_split, nested_stratified_stage_samples
from src.large_dataset.schema import TARGET_COLUMN, derive_features, load_raw_cardio

#: The identity of the 200-row comparison set, established when the
#: correction phase began. If this ever changes, the comparison set was
#: re-sampled -- which is exactly what must never happen.
EXPECTED_COMPARISON_FINGERPRINT = "96eac11a8394b87e"


@pytest.fixture(scope="module")
def modeling_pool():
    df = derive_features(load_raw_cardio())
    flags = compute_cleaning_flags(df)
    modeling_df, _ = split_modeling_subset(df, flags)
    return modeling_df


@pytest.fixture(scope="module")
def fixed_split(modeling_pool):
    return build_fixed_split(modeling_pool, TARGET_COLUMN, seed=42)


# --------------------------------------------------------------------------
# 1. Exact test-set identity
# --------------------------------------------------------------------------


def test_comparison_set_has_expected_size(fixed_split) -> None:
    assert len(fixed_split.test_set_quantum) == 200


def test_comparison_set_fingerprint_is_stable(fixed_split) -> None:
    """The comparison set must be the SAME 200 observations every run."""
    ids = fixed_split.test_set_quantum["id"].tolist()
    assert comparison_set_fingerprint(ids) == EXPECTED_COMPARISON_FINGERPRINT


def test_comparison_set_is_deterministic_across_calls(modeling_pool) -> None:
    a = build_fixed_split(modeling_pool, TARGET_COLUMN, seed=42)
    b = build_fixed_split(modeling_pool, TARGET_COLUMN, seed=42)
    assert a.test_set_quantum["id"].tolist() == b.test_set_quantum["id"].tolist()


def test_comparison_set_is_strict_subset_of_classical_test_set(fixed_split) -> None:
    q_ids = set(fixed_split.test_set_quantum["id"])
    c_ids = set(fixed_split.test_set_classical["id"])
    assert q_ids.issubset(c_ids)
    assert len(q_ids) < len(c_ids)


# --------------------------------------------------------------------------
# 2. No overlap with training data
# --------------------------------------------------------------------------


def test_comparison_set_does_not_overlap_training_pool(fixed_split) -> None:
    q_ids = set(fixed_split.test_set_quantum["id"])
    pool_ids = set(fixed_split.training_pool["id"])
    assert len(q_ids & pool_ids) == 0


@pytest.mark.parametrize("stage_size", [1000, 5000])
def test_comparison_set_does_not_overlap_stage_training_subsets(fixed_split, stage_size) -> None:
    stages = nested_stratified_stage_samples(
        fixed_split.training_pool, TARGET_COLUMN, [stage_size], seed=42
    )
    q_ids = set(fixed_split.test_set_quantum["id"])
    assert len(q_ids & set(stages[stage_size]["id"])) == 0


# --------------------------------------------------------------------------
# 3. Metric calculation
# --------------------------------------------------------------------------


def test_threshold_is_the_phase3_fixed_half() -> None:
    """Threshold must remain the established 0.5 -- never selected from test data."""
    assert DECISION_THRESHOLD == 0.5


def test_bootstrap_metric_ci_brackets_point_estimate() -> None:
    rng = np.random.RandomState(0)
    y_true = np.repeat([0, 1], 100)
    y_proba = np.clip(y_true * 0.4 + rng.normal(0.3, 0.2, size=200), 0, 1)
    res = bootstrap_metric_ci(y_true, y_proba, "roc_auc", n_resamples=300, seed=1)
    assert res["ci_low"] <= res["point_estimate"] <= res["ci_high"]
    assert 0.0 <= res["ci_low"] <= 1.0
    assert 0.0 <= res["ci_high"] <= 1.0


# --------------------------------------------------------------------------
# 4 & 5. Paired comparison + bootstrap reproducibility
# --------------------------------------------------------------------------


def test_paired_bootstrap_is_reproducible_with_same_seed() -> None:
    rng = np.random.RandomState(3)
    y_true = np.repeat([0, 1], 100)
    pa = np.clip(y_true * 0.5 + rng.normal(0.25, 0.2, 200), 0, 1)
    pb = np.clip(y_true * 0.2 + rng.normal(0.4, 0.3, 200), 0, 1)

    r1 = paired_bootstrap_delta(y_true, pa, pb, "roc_auc", n_resamples=300, seed=BOOTSTRAP_SEED)
    r2 = paired_bootstrap_delta(y_true, pa, pb, "roc_auc", n_resamples=300, seed=BOOTSTRAP_SEED)
    assert r1 == r2


def test_paired_bootstrap_identical_models_gives_zero_delta() -> None:
    """A model compared against itself must have exactly zero observed delta
    and a CI containing zero."""
    rng = np.random.RandomState(5)
    y_true = np.repeat([0, 1], 60)
    p = np.clip(y_true * 0.5 + rng.normal(0.25, 0.2, 120), 0, 1)
    res = paired_bootstrap_delta(y_true, p, p, "roc_auc", n_resamples=200, seed=7)
    assert res["observed_delta"] == pytest.approx(0.0, abs=1e-12)
    assert res["ci_includes_zero"]


def test_paired_bootstrap_detects_a_clearly_better_model() -> None:
    """A strongly separating model vs a near-random one should produce a CI
    that excludes zero."""
    rng = np.random.RandomState(9)
    y_true = np.repeat([0, 1], 150)
    strong = np.clip(y_true * 0.8 + rng.normal(0.1, 0.1, 300), 0, 1)
    weak = rng.uniform(0, 1, 300)
    res = paired_bootstrap_delta(y_true, strong, weak, "roc_auc", n_resamples=400, seed=11)
    assert res["observed_delta"] > 0
    assert not res["ci_includes_zero"]


# --------------------------------------------------------------------------
# McNemar
# --------------------------------------------------------------------------


def test_mcnemar_identical_predictions_gives_p_one() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0])
    pred = np.array([0, 1, 1, 1, 0, 0])
    res = mcnemar_exact(y_true, pred, pred)
    assert res["n_discordant"] == 0
    assert res["p_value"] == 1.0


def test_mcnemar_contingency_counts_are_correct() -> None:
    y_true = np.array([1, 1, 1, 1])
    a = np.array([1, 1, 0, 0])  # correct, correct, wrong, wrong
    b = np.array([1, 0, 1, 0])  # correct, wrong, correct, wrong
    res = mcnemar_exact(y_true, a, b)
    c = res["contingency"]
    assert c["both_correct"] == 1
    assert c["a_correct_b_wrong"] == 1
    assert c["a_wrong_b_correct"] == 1
    assert c["both_wrong"] == 1
    assert res["n_discordant"] == 2


# --------------------------------------------------------------------------
# 6. Prediction alignment by id
# --------------------------------------------------------------------------


def test_saved_predictions_are_aligned_by_id_if_present() -> None:
    """If a corrected-comparison run has been executed, every model's
    predictions must be stored against the same id column, in the same
    order, with a single shared y_true."""
    import pandas as pd

    pred_path = (
        find_project_root() / "results" / "large_dataset" / "corrected_comparison"
        / "stage_1000" / "predictions.csv"
    )
    if not pred_path.is_file():
        pytest.skip("corrected comparison for stage 1000 has not been run yet")

    df = pd.read_csv(pred_path)
    assert "id" in df.columns and "y_true" in df.columns
    assert len(df) == 200
    assert df["id"].duplicated().sum() == 0

    proba_cols = [c for c in df.columns if c.endswith("_proba")]
    pred_cols = [c for c in df.columns if c.endswith("_pred")]
    assert len(proba_cols) == 5  # 4 classical + QSVM
    assert len(pred_cols) == 5
    for c in proba_cols:
        assert df[c].between(0.0, 1.0).all()
    # Binary predictions must be consistent with the fixed 0.5 threshold
    for c in proba_cols:
        model = c[: -len("_proba")]
        expected = (df[c] >= DECISION_THRESHOLD).astype(int)
        assert (df[f"{model}_pred"] == expected).all()


# --------------------------------------------------------------------------
# 7. Historical results unchanged
# --------------------------------------------------------------------------


def test_phase5_historical_result_is_unchanged() -> None:
    """The Cleveland Phase 5 benchmark must not be modified by any
    large-dataset work."""
    path = find_project_root() / "results" / "quantum" / "phase5" / "final_metrics.json"
    if not path.is_file():
        pytest.skip("Phase 5 results not present")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["test_metrics"]["roc_auc"] == pytest.approx(0.7798, abs=5e-4)
    assert data["test_metrics"]["tn"] == 20
    assert data["test_metrics"]["fp"] == 13
    assert data["test_metrics"]["fn"] == 7
    assert data["test_metrics"]["tp"] == 21


def test_corrected_comparison_writes_to_separate_location() -> None:
    """Corrected results must never overwrite the historical stage files."""
    root = find_project_root() / "results" / "large_dataset"
    corrected = root / "corrected_comparison"
    historical = root / "stage_comparison.json"
    if corrected.exists():
        assert corrected.is_dir()
        assert historical.resolve() not in [p.resolve() for p in corrected.rglob("*")]
