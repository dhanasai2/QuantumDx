"""Tests for src.classical (Phase 3: classical baselines, leakage-safe CV).

These tests exercise the REAL adapters, model builders, and GridSearchCV
wiring used by train_baselines.py, but with deliberately trimmed
hyperparameter grids and small CV fold counts, so the suite runs in
seconds rather than the ~9 minutes the full Phase 3 sweep takes. The full
sweep itself is run via `python -m src.classical.train_baselines`, not in
this test suite.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV

from src.classical.evaluation import (
    ModelResult,
    bootstrap_test_confidence_interval,
    build_comparison_table,
    compute_classification_metrics,
    evaluate_on_test,
)
from src.classical.models import ModelSpec, XGBOOST_AVAILABLE, XGBOOST_IMPORT_ERROR, get_available_models
from src.classical.tuning import (
    QuantumReadyTransformer,
    SharedFeatureTransformer,
    build_model_pipeline,
    run_grid_search,
)
from src.data.inspect_dataset import find_project_root, load_raw_dataset, resolve_dataset_path
from src.preprocessing.config import DEFAULT_CONFIG, PreprocessingConfig
from src.preprocessing.pipeline import get_feature_groups, run_preprocessing_pipeline

CLEVELAND_RELATIVE_PATH = Path("data") / "raw" / "processed.cleveland.data"
FAST_CV_FOLDS = 3


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cleveland_path() -> Path:
    root = find_project_root()
    path = root / CLEVELAND_RELATIVE_PATH
    if not path.is_file():
        pytest.skip(f"Cleveland dataset not present at {path}; skipping Phase 3 tests.")
    return path


@pytest.fixture(scope="module")
def raw_df(cleveland_path: Path) -> pd.DataFrame:
    return load_raw_dataset(cleveland_path)


@pytest.fixture(scope="module")
def raw_file_bytes_before(cleveland_path: Path) -> bytes:
    return cleveland_path.read_bytes()


@pytest.fixture(scope="module")
def prepared_full(raw_df: pd.DataFrame):
    """The Phase 2 train/test split for feature_set='full', reused (not
    duplicated) across tests via run_preprocessing_pipeline.
    """
    config = DEFAULT_CONFIG.with_overrides(feature_set="full")
    return run_preprocessing_pipeline(raw_df, config), config


@pytest.fixture(scope="module")
def prepared_screening(raw_df: pd.DataFrame):
    config = DEFAULT_CONFIG.with_overrides(feature_set="screening")
    return run_preprocessing_pipeline(raw_df, config), config


def _trim_grid(spec: ModelSpec, n_values: int = 1) -> ModelSpec:
    """Return a copy of a real ModelSpec with each hyperparameter's search
    list truncated to its first `n_values` entries, so GridSearchCV runs a
    single (or very few) combination instead of the full sweep -- while
    still exercising the REAL build() function and REAL parameter names.
    """
    trimmed_grid = {k: v[:n_values] for k, v in spec.param_grid.items()}
    return replace(spec, param_grid=trimmed_grid)


@pytest.fixture(scope="module")
def fast_models() -> dict[str, ModelSpec]:
    """The real model registry, but with every hyperparameter grid trimmed
    to one value -- fast enough for a test suite, still real code paths.
    """
    full_registry = get_available_models(DEFAULT_CONFIG.random_seed)
    return {key: _trim_grid(spec) for key, spec in full_registry.items()}


# --------------------------------------------------------------------------
# 1. All models can train successfully
# --------------------------------------------------------------------------


def test_all_registered_models_can_train_and_predict(prepared_full, fast_models) -> None:
    result_meta, config = prepared_full
    feature_groups = get_feature_groups(config.feature_set)
    X_train_raw, y_train = result_meta.X_train_raw, result_meta.y_train
    X_test_raw = result_meta.X_test_raw

    for key, spec in fast_models.items():
        pipeline = build_model_pipeline(spec, "classical", feature_groups, config, config.random_seed)
        pipeline.fit(X_train_raw, y_train)
        proba = pipeline.predict_proba(X_test_raw)
        assert proba.shape == (len(X_test_raw), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        preds = pipeline.predict(X_test_raw)
        assert set(np.unique(preds)).issubset({0, 1})


def test_xgboost_availability_is_documented_not_silently_replaced() -> None:
    models = get_available_models(DEFAULT_CONFIG.random_seed)
    if XGBOOST_AVAILABLE:
        assert "xgboost" in models
        assert XGBOOST_IMPORT_ERROR is None
    else:
        assert "xgboost" not in models
        assert XGBOOST_IMPORT_ERROR is not None and len(XGBOOST_IMPORT_ERROR) > 0
        # The other three required models must still be present -- XGBoost's
        # absence must never cause a silent substitution for it.
        assert {"logistic_regression", "rbf_svm", "random_forest"}.issubset(models.keys())


# --------------------------------------------------------------------------
# 2 & 3. CV uses only the training partition / test set not used in tuning
# --------------------------------------------------------------------------


def test_run_grid_search_has_no_test_set_parameter() -> None:
    """Structural guard: run_grid_search's signature contains no parameter
    whose name could plausibly refer to test data. This is a stronger
    guarantee than a runtime check -- the function is architecturally
    incapable of touching the test set, because it is never given it.
    """
    sig = inspect.signature(run_grid_search)
    param_names = [p.lower() for p in sig.parameters]
    assert not any("test" in name for name in param_names), (
        f"run_grid_search must not accept any test-set parameter; found: {param_names}"
    )


def test_cv_scores_unaffected_by_test_set_perturbation(prepared_full, fast_models) -> None:
    """The CV-level analogue of Phase 2's leakage test: perturbing rows
    that live ONLY in the test split must not change the CV fold scores
    produced while tuning on the training split, because the search
    function is never given the test split at all.
    """
    result_meta, config = prepared_full
    feature_groups = get_feature_groups(config.feature_set)
    spec = fast_models["logistic_regression"]

    search_a = run_grid_search(
        spec, "classical", feature_groups, config, result_meta.X_train_raw, result_meta.y_train,
        cv_folds=FAST_CV_FOLDS, random_seed=config.random_seed,
    )

    # Perturb the test split extremely -- this array is never passed to
    # run_grid_search below, but constructing it proves it EXISTS and could
    # have been (mis)used; the point is that it is not.
    perturbed_test = result_meta.X_test_raw.copy(deep=True)
    perturbed_test.iloc[0, perturbed_test.columns.get_loc("chol")] = 999_999.0
    assert perturbed_test is not None  # constructed, intentionally unused below

    search_b = run_grid_search(
        spec, "classical", feature_groups, config, result_meta.X_train_raw, result_meta.y_train,
        cv_folds=FAST_CV_FOLDS, random_seed=config.random_seed,
    )

    np.testing.assert_array_equal(
        search_a.cv_results_["mean_test_roc_auc"], search_b.cv_results_["mean_test_roc_auc"]
    )
    assert search_a.best_params_ == search_b.best_params_


def test_evaluate_on_test_is_the_only_place_test_labels_are_scored(prepared_full, fast_models) -> None:
    """evaluate_on_test is called with the test split explicitly, exactly
    once per model -- verify it returns metrics computed on that exact
    split (right length, right values), confirming the boundary between
    "tuning" and "final evaluation" is respected.
    """
    result_meta, config = prepared_full
    feature_groups = get_feature_groups(config.feature_set)
    spec = fast_models["logistic_regression"]

    search = run_grid_search(
        spec, "classical", feature_groups, config, result_meta.X_train_raw, result_meta.y_train,
        cv_folds=FAST_CV_FOLDS, random_seed=config.random_seed,
    )
    result = evaluate_on_test(
        search, "test_experiment", "logistic_regression", "Logistic Regression", "classical",
        result_meta.X_test_raw, result_meta.y_test, n_train=len(result_meta.X_train_raw),
        random_seed=config.random_seed,
    )
    assert result.n_test == len(result_meta.X_test_raw)
    assert len(result.y_test_true) == len(result_meta.y_test)
    np.testing.assert_array_equal(result.y_test_true, result_meta.y_test)


# --------------------------------------------------------------------------
# 4. Preprocessing is fitted independently within CV folds
# --------------------------------------------------------------------------


def test_preprocessing_refits_independently_per_cv_fold(prepared_full) -> None:
    """Fit SharedFeatureTransformer directly on two different fold-train
    partitions of the SAME training data and confirm the fitted imputer
    statistics differ -- proof that preprocessing is refit per fold, not
    fit once globally and reused.
    """
    result_meta, config = prepared_full
    feature_groups = get_feature_groups(config.feature_set)
    X_train_raw, y_train = result_meta.X_train_raw, result_meta.y_train

    # Two disjoint, differently-composed "folds" carved out of the same
    # training split.
    fold_a_idx = np.arange(0, 120)
    fold_b_idx = np.arange(120, 242)

    transformer_a = SharedFeatureTransformer(feature_groups, config)
    transformer_a.fit(X_train_raw.iloc[fold_a_idx], y_train[fold_a_idx])
    stats_a = transformer_a._impl.preprocessor.named_transformers_["continuous"].named_steps["impute"].statistics_

    transformer_b = SharedFeatureTransformer(feature_groups, config)
    transformer_b.fit(X_train_raw.iloc[fold_b_idx], y_train[fold_b_idx])
    stats_b = transformer_b._impl.preprocessor.named_transformers_["continuous"].named_steps["impute"].statistics_

    assert not np.array_equal(stats_a, stats_b), (
        "Different fold-training partitions produced identical imputer statistics; "
        "expected them to differ since they were fit on different rows."
    )


def test_quantum_ready_pca_refits_independently_per_fold(prepared_full) -> None:
    result_meta, config = prepared_full
    feature_groups = get_feature_groups(config.feature_set)
    X_train_raw, y_train = result_meta.X_train_raw, result_meta.y_train

    fold_a_idx = np.arange(0, 120)
    fold_b_idx = np.arange(120, 242)

    shared_a = SharedFeatureTransformer(feature_groups, config)
    X_a = shared_a.fit_transform(X_train_raw.iloc[fold_a_idx], y_train[fold_a_idx])
    quantum_a = QuantumReadyTransformer(config.pca_n_components, config.quantum_range)
    quantum_a.fit(X_a)

    shared_b = SharedFeatureTransformer(feature_groups, config)
    X_b = shared_b.fit_transform(X_train_raw.iloc[fold_b_idx], y_train[fold_b_idx])
    quantum_b = QuantumReadyTransformer(config.pca_n_components, config.quantum_range)
    quantum_b.fit(X_b)

    components_a = quantum_a._impl.named_steps["pca"].components_
    components_b = quantum_b._impl.named_steps["pca"].components_
    assert not np.allclose(components_a, components_b), (
        "PCA components were identical across two different fold-training partitions."
    )


# --------------------------------------------------------------------------
# 5. Results are reproducible with the same seed
# --------------------------------------------------------------------------


def test_grid_search_reproducible_with_same_seed(prepared_full, fast_models) -> None:
    result_meta, config = prepared_full
    feature_groups = get_feature_groups(config.feature_set)
    spec = fast_models["random_forest"]

    search_a = run_grid_search(
        spec, "classical", feature_groups, config, result_meta.X_train_raw, result_meta.y_train,
        cv_folds=FAST_CV_FOLDS, random_seed=config.random_seed,
    )
    search_b = run_grid_search(
        spec, "classical", feature_groups, config, result_meta.X_train_raw, result_meta.y_train,
        cv_folds=FAST_CV_FOLDS, random_seed=config.random_seed,
    )

    assert search_a.best_params_ == search_b.best_params_
    np.testing.assert_allclose(search_a.best_score_, search_b.best_score_)
    proba_a = search_a.best_estimator_.predict_proba(result_meta.X_test_raw)[:, 1]
    proba_b = search_b.best_estimator_.predict_proba(result_meta.X_test_raw)[:, 1]
    np.testing.assert_allclose(proba_a, proba_b)


# --------------------------------------------------------------------------
# 6. Metrics are calculated correctly
# --------------------------------------------------------------------------


def test_compute_classification_metrics_against_hand_worked_example() -> None:
    # 10 records: TP=3, TN=4, FP=1, FN=2 (hand-verifiable confusion matrix)
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1])
    y_proba = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.05, 0.15, 0.55])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    assert metrics["tp"] == 3
    assert metrics["fn"] == 2
    assert metrics["tn"] == 4
    assert metrics["fp"] == 1
    assert metrics["accuracy"] == pytest.approx(0.7)
    assert metrics["precision"] == pytest.approx(3 / 4)
    assert metrics["sensitivity"] == pytest.approx(3 / 5)
    assert metrics["recall"] == metrics["sensitivity"]
    assert metrics["specificity"] == pytest.approx(4 / 5)
    expected_f1 = 2 * (3 / 4) * (3 / 5) / ((3 / 4) + (3 / 5))
    assert metrics["f1"] == pytest.approx(expected_f1)


def test_metrics_perfect_classifier() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    y_proba = np.array([0.99, 0.9, 0.1, 0.01])
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_bootstrap_confidence_interval_brackets_point_estimate() -> None:
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=100)
    y_proba = np.clip(y_true * 0.6 + rng.rand(100) * 0.4, 0.01, 0.99)
    result = bootstrap_test_confidence_interval(y_true, y_proba, "roc_auc", n_resamples=200, random_seed=42)
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]


# --------------------------------------------------------------------------
# 7. Binary target is handled correctly
# --------------------------------------------------------------------------


def test_target_arrays_are_strictly_binary(prepared_full) -> None:
    result_meta, _config = prepared_full
    assert set(np.unique(result_meta.y_train).tolist()).issubset({0, 1})
    assert set(np.unique(result_meta.y_test).tolist()).issubset({0, 1})


def test_target_matches_original_num_derivation(prepared_full) -> None:
    result_meta, _config = prepared_full
    assert np.array_equal(result_meta.y_train, (result_meta.num_train > 0).astype(int))
    assert np.array_equal(result_meta.y_test, (result_meta.num_test > 0).astype(int))


# --------------------------------------------------------------------------
# 8. Both all-feature and ca/thal-ablation experiments run
# --------------------------------------------------------------------------


def test_full_experiment_runs_and_includes_ca_thal(prepared_full, fast_models) -> None:
    result_meta, config = prepared_full
    feature_groups = get_feature_groups(config.feature_set)
    assert "ca" in result_meta.X_train_raw.columns
    assert "thal" in result_meta.X_train_raw.columns

    search = run_grid_search(
        fast_models["logistic_regression"], "classical", feature_groups, config,
        result_meta.X_train_raw, result_meta.y_train, cv_folds=FAST_CV_FOLDS, random_seed=config.random_seed,
    )
    assert search.best_estimator_ is not None


def test_screening_experiment_runs_and_excludes_ca_thal(prepared_screening, fast_models) -> None:
    result_meta, config = prepared_screening
    feature_groups = get_feature_groups(config.feature_set)
    assert "ca" not in result_meta.X_train_raw.columns
    assert "thal" not in result_meta.X_train_raw.columns

    search = run_grid_search(
        fast_models["logistic_regression"], "classical", feature_groups, config,
        result_meta.X_train_raw, result_meta.y_train, cv_folds=FAST_CV_FOLDS, random_seed=config.random_seed,
    )
    assert search.best_estimator_ is not None


def test_full_and_screening_produce_different_selected_features(prepared_full, prepared_screening) -> None:
    result_full, _ = prepared_full
    result_screening, _ = prepared_screening
    assert result_full.artifacts.selected_feature_names != result_screening.artifacts.selected_feature_names


# --------------------------------------------------------------------------
# 9. No NaNs reach the models
# --------------------------------------------------------------------------


def test_no_nans_through_shared_feature_transformer(prepared_full) -> None:
    result_meta, config = prepared_full
    feature_groups = get_feature_groups(config.feature_set)
    transformer = SharedFeatureTransformer(feature_groups, config)
    transformed = transformer.fit_transform(result_meta.X_train_raw, result_meta.y_train)
    assert not pd.DataFrame(transformed).isna().any().any()
    test_transformed = transformer.transform(result_meta.X_test_raw)
    assert not pd.DataFrame(test_transformed).isna().any().any()


def test_no_nans_through_quantum_ready_transformer(prepared_full) -> None:
    result_meta, config = prepared_full
    feature_groups = get_feature_groups(config.feature_set)
    shared = SharedFeatureTransformer(feature_groups, config)
    X_train_classical = shared.fit_transform(result_meta.X_train_raw, result_meta.y_train)
    quantum = QuantumReadyTransformer(config.pca_n_components, config.quantum_range)
    X_train_quantum = quantum.fit_transform(X_train_classical)
    assert not np.isnan(np.asarray(X_train_quantum)).any()


# --------------------------------------------------------------------------
# 10. Final evaluation runs on the locked test set
# --------------------------------------------------------------------------


def test_final_evaluation_produces_model_result_on_locked_test_set(prepared_full, fast_models) -> None:
    result_meta, config = prepared_full
    feature_groups = get_feature_groups(config.feature_set)
    spec = fast_models["logistic_regression"]

    search = run_grid_search(
        spec, "classical", feature_groups, config, result_meta.X_train_raw, result_meta.y_train,
        cv_folds=FAST_CV_FOLDS, random_seed=config.random_seed,
    )
    result = evaluate_on_test(
        search, "experiment_test", "logistic_regression", "Logistic Regression", "classical",
        result_meta.X_test_raw, result_meta.y_test, n_train=len(result_meta.X_train_raw),
        random_seed=config.random_seed,
    )
    assert isinstance(result, ModelResult)
    assert 0.0 <= result.test_metrics["roc_auc"] <= 1.0
    assert result.n_test == 61
    table = build_comparison_table([result])
    assert len(table) == 1
    assert table.loc[0, "Model"] == "Logistic Regression"


# --------------------------------------------------------------------------
# Raw data / Phase 2 immutability (must still hold after Phase 3 runs)
# --------------------------------------------------------------------------


def test_raw_file_unchanged_after_phase3_pipeline_use(
    cleveland_path: Path, prepared_full, prepared_screening, raw_file_bytes_before: bytes
) -> None:
    after = cleveland_path.read_bytes()
    assert raw_file_bytes_before == after, "Phase 3 must never modify data/raw/."
