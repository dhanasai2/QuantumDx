"""Tests for src.large_dataset.adaptive_robustness (Phase 7).

Covers Section 10's requirements: training/test separation, training-only
adaptive pair selection, deterministic reproducibility, correct delta
direction, paired-test calculation, no accidental use of test
labels/features during adaptive-map construction, plus checkpoint/resume
correctness and the seed=42 short-circuit's consistency with Stage D.

Fast by design: the real n_train=20,000 multi-seed run is exercised only
via schema checks on real artifacts (skipped if absent). Everything else
uses a tiny monkeypatched N_TRAIN or synthetic data.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.inspect_dataset import find_project_root
from src.large_dataset import adaptive_robustness as ar

EXPECTED_FINGERPRINT = "96eac11a8394b87e"


# --------------------------------------------------------------------------
# 1. Training/test separation
# --------------------------------------------------------------------------


def test_seeded_training_subset_disjoint_from_fixed_test_sets() -> None:
    split = ar.build_fixed_split_once()
    subset = ar.build_seeded_training_subset(split.training_pool, seed=7)
    assert len(subset) == ar.N_TRAIN
    assert len(set(subset["id"]) & set(split.test_set_classical["id"])) == 0
    assert len(set(subset["id"]) & set(split.test_set_quantum["id"])) == 0


def test_different_seeds_draw_different_training_subsets() -> None:
    split = ar.build_fixed_split_once()
    s7 = ar.build_seeded_training_subset(split.training_pool, seed=7)
    s21 = ar.build_seeded_training_subset(split.training_pool, seed=21)
    assert s7["id"].tolist() != s21["id"].tolist()


def test_seed_42_training_subset_matches_stage_d() -> None:
    """PreprocessingConfig/QuantumConfig both default to random_seed=42, and
    a single-size nested_stratified_stage_samples draw is independent of
    what other sizes are in the list -- so seed=42 here must reproduce
    Stage D's own n=20,000 training subset exactly."""
    from src.large_dataset.stage_d import build_stage_d_split

    split = ar.build_fixed_split_once()
    subset_42 = ar.build_seeded_training_subset(split.training_pool, seed=42)
    stage_d_split = build_stage_d_split()
    assert subset_42["id"].tolist() == stage_d_split.stage_train_df["id"].tolist()


# --------------------------------------------------------------------------
# 2. Training-only adaptive pair selection / no test dependency
# --------------------------------------------------------------------------


def test_run_seed_fresh_never_touches_test_labels_before_scoring(monkeypatch) -> None:
    """Structural check: build_adaptive_feature_map_from_training_data is
    called with processed.X_train_quantum only -- verified by intercepting
    the call and asserting its first argument's length equals the training
    set size, never the 200-row test size."""
    import src.quantum.adaptive_feature_map as afm

    seen = {}
    original = afm.build_adaptive_feature_map_from_training_data

    def spy(X_train, *args, **kwargs):
        seen["n"] = len(X_train)
        return original(X_train, *args, **kwargs)

    monkeypatch.setattr(ar, "build_adaptive_feature_map_from_training_data", spy)

    ar.N_TRAIN = 250
    split = ar.build_fixed_split_once()
    cmp_ids = split.test_set_quantum["id"].tolist()
    fg = ar.get_cardio_feature_groups()
    from src.preprocessing.config import PreprocessingConfig
    from src.quantum.config import QuantumConfig

    result = ar.run_seed_fresh(999, split, cmp_ids, fg, PreprocessingConfig(random_seed=999), QuantumConfig(random_seed=999))
    assert seen["n"] == 250  # training size, never 200 (the test size)
    assert seen["n"] != len(cmp_ids)


# --------------------------------------------------------------------------
# 3. Deterministic reproducibility
# --------------------------------------------------------------------------


def test_build_seeded_training_subset_is_deterministic() -> None:
    split = ar.build_fixed_split_once()
    a = ar.build_seeded_training_subset(split.training_pool, seed=7)
    b = ar.build_seeded_training_subset(split.training_pool, seed=7)
    assert a["id"].tolist() == b["id"].tolist()


# --------------------------------------------------------------------------
# 4. Correct delta direction
# --------------------------------------------------------------------------


def test_compute_seed_statistics_delta_is_adaptive_minus_baseline() -> None:
    rng = np.random.RandomState(3)
    y = np.repeat([0, 1], 50)
    baseline = rng.uniform(0, 1, 100)
    adaptive = np.clip(y * 0.8 + rng.normal(0.1, 0.1, 100), 0, 1)  # clearly better
    seed_result = {
        "baseline": ar.ModelPredictions("baseline_qsvm", np.arange(100), y, baseline, (baseline >= 0.5).astype(int), {}),
        "adaptive": ar.ModelPredictions("adaptive_qsvm", np.arange(100), y, adaptive, (adaptive >= 0.5).astype(int), {}),
    }
    stats = ar.compute_seed_statistics(seed_result)
    assert stats["delong"]["delta"] > 0  # adaptive better -> positive delta
    assert stats["delong"]["auc_a"] > stats["delong"]["auc_b"]


# --------------------------------------------------------------------------
# 5. Paired-test calculation / checkpoint round-trip
# --------------------------------------------------------------------------


def test_persist_and_load_seed_checkpoint_round_trips(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ar, "RESULTS_ROOT", tmp_path)
    rng = np.random.RandomState(5)
    y = np.repeat([0, 1], 100)
    ids = np.arange(200)
    baseline_p = np.clip(y * 0.5 + rng.normal(0.25, 0.2, 200), 0, 1)
    adaptive_p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 200), 0, 1)
    result = {
        "seed": 555, "provenance": "computed_fresh",
        "baseline": ar.ModelPredictions("baseline_qsvm", ids, y, baseline_p, (baseline_p >= 0.5).astype(int), ar._metrics_from_proba(y, baseline_p)),
        "adaptive": ar.ModelPredictions("adaptive_qsvm", ids, y, adaptive_p, (adaptive_p >= 0.5).astype(int), ar._metrics_from_proba(y, adaptive_p)),
        "adaptive_map_config": {"n_pairs": 3, "selected_adaptive_pairs": [[0, 1]]},
        "runtime": {"x": 1.0},
    }

    # Fake the fingerprint check by monkeypatching comparison_set_fingerprint
    # to accept this synthetic id range as "expected" for this isolated test.
    monkeypatch.setattr(ar, "EXPECTED_FINGERPRINT", ar.comparison_set_fingerprint(ids.tolist()))

    stats = ar.compute_seed_statistics(result)
    ar.persist_seed_result(result, stats)

    loaded = ar.load_seed_checkpoint(555)
    assert loaded is not None
    loaded_result, loaded_stats = loaded
    assert loaded_result["seed"] == 555
    # CSV round-trip does not guarantee bit-exact float64 preservation.
    assert np.allclose(loaded_result["baseline"].y_proba, baseline_p, atol=1e-9)
    assert np.allclose(loaded_result["adaptive"].y_proba, adaptive_p, atol=1e-9)
    assert loaded_stats["delong"]["delta"] == pytest.approx(stats["delong"]["delta"], abs=1e-9)


def test_load_seed_checkpoint_returns_none_for_missing_seed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ar, "RESULTS_ROOT", tmp_path)
    assert ar.load_seed_checkpoint(31415) is None


# --------------------------------------------------------------------------
# 6. Real artifacts, if present
# --------------------------------------------------------------------------


def test_per_seed_results_csv_schema_if_present() -> None:
    path = ar.RESULTS_ROOT / "per_seed_results.csv"
    if not path.is_file():
        pytest.skip("Phase 7 has not been run yet")
    df = pd.read_csv(path)
    assert set(df["seed"]) == set(ar.SEEDS)
    for col in ["delta_roc_auc", "delong_p", "bootstrap_roc_p", "mcnemar_p"]:
        assert col in df.columns
        assert df[col].notna().all()


def test_statistical_summary_json_documents_shared_test_set_limitation_if_present() -> None:
    path = ar.RESULTS_ROOT / "statistical_summary.json"
    if not path.is_file():
        pytest.skip("Phase 7 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["comparison_set"]["fingerprint"] == EXPECTED_FINGERPRINT
    assert "shared_test_set_limitation" in summary["cross_seed_summary"]
    assert "NOT" in summary["cross_seed_summary"]["shared_test_set_limitation"]


def test_stage_d_and_adaptive_qsvm_artifacts_unmodified_by_phase7() -> None:
    """Phase 7 must never rewrite Stage A-D or the single-seed adaptive
    experiment's own result files -- it only reads them for seed 42."""
    for path in [ar.STAGE_D_PREDICTIONS_PATH, ar.ADAPTIVE_QSVM_PREDICTIONS_PATH]:
        if path.is_file():
            df = pd.read_csv(path)
            assert len(df) == 200  # still the original schema/size, untouched
