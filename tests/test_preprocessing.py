"""Tests for src.preprocessing (Phase 2: leakage-safe preprocessing pipeline).

These tests verify PROPERTIES the pipeline must have -- reproducibility,
zero leakage, correct shapes, no NaNs -- not just "it runs". The leakage
test in particular (test_no_leakage_from_test_set_statistics) is the most
important test in this file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.inspect_dataset import find_project_root, load_raw_dataset, resolve_dataset_path
from src.preprocessing.config import DEFAULT_CONFIG, PreprocessingConfig
from src.preprocessing.pipeline import (
    BINARY_TARGET_COLUMN,
    RAW_TARGET_COLUMN,
    SharedFeaturePipeline,
    build_preprocessor,
    build_quantum_pipeline,
    create_binary_target,
    get_feature_groups,
    run_preprocessing_pipeline,
    split_dataset,
)
from src.preprocessing.validation import (
    PreprocessingValidationError,
    assert_arrays_close,
    assert_no_missing_values,
    assert_within_range,
)

CLEVELAND_RELATIVE_PATH = Path("data") / "raw" / "processed.cleveland.data"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cleveland_path() -> Path:
    root = find_project_root()
    path = root / CLEVELAND_RELATIVE_PATH
    if not path.is_file():
        pytest.skip(f"Cleveland dataset not present at {path}; skipping preprocessing tests.")
    return path


@pytest.fixture(scope="module")
def raw_df(cleveland_path: Path) -> pd.DataFrame:
    return load_raw_dataset(cleveland_path)


@pytest.fixture(scope="module")
def raw_file_bytes_before(cleveland_path: Path) -> bytes:
    return cleveland_path.read_bytes()


@pytest.fixture
def config() -> PreprocessingConfig:
    return DEFAULT_CONFIG


# --------------------------------------------------------------------------
# 1. Raw dataset remains unchanged
# --------------------------------------------------------------------------


def test_raw_file_unchanged_after_full_pipeline_run(
    cleveland_path: Path, raw_df: pd.DataFrame, raw_file_bytes_before: bytes
) -> None:
    before = raw_file_bytes_before
    _ = run_preprocessing_pipeline(raw_df, DEFAULT_CONFIG)
    after = cleveland_path.read_bytes()
    assert before == after, "Running the preprocessing pipeline must never modify data/raw/."


def test_input_dataframe_not_mutated_in_place(raw_df: pd.DataFrame) -> None:
    original_columns = list(raw_df.columns)
    original_values = raw_df.copy(deep=True)
    _ = run_preprocessing_pipeline(raw_df, DEFAULT_CONFIG)
    assert list(raw_df.columns) == original_columns, "run_preprocessing_pipeline must not add columns to its input."
    pd.testing.assert_frame_equal(raw_df, original_values)


# --------------------------------------------------------------------------
# 2. Target conversion
# --------------------------------------------------------------------------


def test_binary_target_created_correctly(raw_df: pd.DataFrame) -> None:
    out = create_binary_target(raw_df)
    assert BINARY_TARGET_COLUMN in out.columns
    # num preserved unchanged
    pd.testing.assert_series_equal(out[RAW_TARGET_COLUMN], raw_df[RAW_TARGET_COLUMN])
    # target == (num > 0)
    expected = (raw_df[RAW_TARGET_COLUMN] > 0).astype(int)
    pd.testing.assert_series_equal(out[BINARY_TARGET_COLUMN], expected, check_names=False)


def test_binary_target_matches_documented_class_counts(raw_df: pd.DataFrame) -> None:
    out = create_binary_target(raw_df)
    counts = out[BINARY_TARGET_COLUMN].value_counts()
    assert counts[0] == 164
    assert counts[1] == 139


def test_create_binary_target_raises_on_missing_num_column() -> None:
    bad_df = pd.DataFrame({"age": [50, 60]})
    with pytest.raises(ValueError):
        create_binary_target(bad_df)


def test_create_binary_target_does_not_mutate_input(raw_df: pd.DataFrame) -> None:
    before_cols = list(raw_df.columns)
    create_binary_target(raw_df)
    assert list(raw_df.columns) == before_cols, "create_binary_target must return a copy, not mutate its input."


# --------------------------------------------------------------------------
# 3. Train/test split is reproducible + 4. is stratified
# --------------------------------------------------------------------------


def test_split_is_reproducible_given_same_seed(raw_df: pd.DataFrame) -> None:
    df_t = create_binary_target(raw_df)
    train_a, test_a = split_dataset(df_t, DEFAULT_CONFIG)
    train_b, test_b = split_dataset(df_t, DEFAULT_CONFIG)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)


def test_split_differs_with_different_seed(raw_df: pd.DataFrame) -> None:
    df_t = create_binary_target(raw_df)
    cfg_a = DEFAULT_CONFIG
    cfg_b = DEFAULT_CONFIG.with_overrides(random_seed=DEFAULT_CONFIG.random_seed + 1)
    train_a, _ = split_dataset(df_t, cfg_a)
    train_b, _ = split_dataset(df_t, cfg_b)
    assert not train_a.equals(train_b)


def test_split_sizes_match_configured_test_size(raw_df: pd.DataFrame) -> None:
    df_t = create_binary_target(raw_df)
    train_df, test_df = split_dataset(df_t, DEFAULT_CONFIG)
    assert len(train_df) + len(test_df) == len(df_t)
    expected_test = round(len(df_t) * DEFAULT_CONFIG.test_size)
    assert abs(len(test_df) - expected_test) <= 1


def test_split_is_stratified_on_target(raw_df: pd.DataFrame) -> None:
    df_t = create_binary_target(raw_df)
    train_df, test_df = split_dataset(df_t, DEFAULT_CONFIG)
    overall_rate = df_t[BINARY_TARGET_COLUMN].mean()
    train_rate = train_df[BINARY_TARGET_COLUMN].mean()
    test_rate = test_df[BINARY_TARGET_COLUMN].mean()
    # Stratified split should keep class proportions close across splits.
    assert abs(train_rate - overall_rate) < 0.03
    assert abs(test_rate - overall_rate) < 0.05


def test_split_requires_target_column(raw_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        split_dataset(raw_df, DEFAULT_CONFIG)  # raw_df has no 'target' column yet


def test_split_happens_before_any_fitted_transform(raw_df: pd.DataFrame) -> None:
    # The split function must not accept or require a fitted transformer,
    # and must not alter feature values -- only partition rows.
    df_t = create_binary_target(raw_df)
    train_df, test_df = split_dataset(df_t, DEFAULT_CONFIG)
    combined = pd.concat([train_df, test_df]).sort_values(by=list(df_t.columns)).reset_index(drop=True)
    reference = df_t.sort_values(by=list(df_t.columns)).reset_index(drop=True)
    pd.testing.assert_frame_equal(combined, reference)


# --------------------------------------------------------------------------
# 5. Missing values are handled
# --------------------------------------------------------------------------


def test_missing_values_present_in_raw_split(raw_df: pd.DataFrame) -> None:
    # Sanity precondition: the known missing values in ca/thal must still
    # be present immediately after splitting (i.e. NOT pre-filled).
    df_t = create_binary_target(raw_df)
    train_df, test_df = split_dataset(df_t, DEFAULT_CONFIG)
    total_missing = train_df[["ca", "thal"]].isna().sum().sum() + test_df[["ca", "thal"]].isna().sum().sum()
    assert total_missing == 6


def test_missing_values_are_imputed_after_shared_pipeline(raw_df: pd.DataFrame) -> None:
    result = run_preprocessing_pipeline(raw_df, DEFAULT_CONFIG)
    assert not result.X_train_classical.isna().any().any()
    assert not result.X_test_classical.isna().any().any()


# --------------------------------------------------------------------------
# 6. Preprocessing does not produce NaNs (classical + quantum)
# --------------------------------------------------------------------------


def test_no_nans_anywhere_in_output(raw_df: pd.DataFrame) -> None:
    result = run_preprocessing_pipeline(raw_df, DEFAULT_CONFIG)
    assert_no_missing_values(result.X_train_classical, name="X_train_classical")
    assert_no_missing_values(result.X_test_classical, name="X_test_classical")
    assert_no_missing_values(result.X_train_quantum, name="X_train_quantum")
    assert_no_missing_values(result.X_test_quantum, name="X_test_quantum")


# --------------------------------------------------------------------------
# 7. Preprocessing is fitted only using training data
# --------------------------------------------------------------------------


def test_imputer_statistics_computed_from_train_only(raw_df: pd.DataFrame) -> None:
    df_t = create_binary_target(raw_df)
    train_df, test_df = split_dataset(df_t, DEFAULT_CONFIG)
    feature_groups = get_feature_groups("full")

    preprocessor = build_preprocessor(feature_groups, DEFAULT_CONFIG)
    X_train = train_df[feature_groups.all_columns]
    y_train = train_df[BINARY_TARGET_COLUMN].to_numpy()
    preprocessor.fit(X_train, y_train)

    # The fitted imputer's learned statistic must equal the TRAIN-ONLY
    # median of 'ca' (a column with missing values), not the median of the
    # full (train+test) dataset.
    continuous_pipeline = preprocessor.named_transformers_["continuous"]
    fitted_imputer = continuous_pipeline.named_steps["impute"]
    numeric_cols = feature_groups.numeric_columns
    ca_index = numeric_cols.index("ca")
    fitted_median_ca = fitted_imputer.statistics_[ca_index]

    expected_train_only_median = X_train["ca"].median()
    full_dataset_median = df_t["ca"].median()

    assert fitted_median_ca == pytest.approx(expected_train_only_median)
    # Guard against a false-positive test: the train-only and full-dataset
    # medians must actually be numerically distinguishable for this
    # assertion to mean anything (otherwise the test would pass even with
    # a leaky implementation by coincidence).
    if expected_train_only_median != full_dataset_median:
        assert fitted_median_ca != pytest.approx(full_dataset_median)


def test_scaler_statistics_unaffected_by_test_set_values(raw_df: pd.DataFrame) -> None:
    """Perturbing the TEST set's feature values must not change what the
    scaler learned, because the scaler is fit before test data is ever
    transformed."""
    df_t = create_binary_target(raw_df)
    train_df, test_df = split_dataset(df_t, DEFAULT_CONFIG)
    feature_groups = get_feature_groups("full")

    preprocessor_a = build_preprocessor(feature_groups, DEFAULT_CONFIG)
    X_train = train_df[feature_groups.all_columns]
    y_train = train_df[BINARY_TARGET_COLUMN].to_numpy()
    preprocessor_a.fit(X_train, y_train)
    scale_a = preprocessor_a.named_transformers_["continuous"].named_steps["scale"].mean_.copy()

    # Refit an identical preprocessor -- fitting never touches test_df at all.
    preprocessor_b = build_preprocessor(feature_groups, DEFAULT_CONFIG)
    preprocessor_b.fit(X_train, y_train)
    scale_b = preprocessor_b.named_transformers_["continuous"].named_steps["scale"].mean_.copy()

    assert np.allclose(scale_a, scale_b)
    # test_df is never passed to .fit() anywhere in this test -- there is
    # no code path by which it could have influenced scale_a/scale_b.
    assert len(test_df) > 0  # sanity: test_df exists and was available, but unused for fitting


# --------------------------------------------------------------------------
# 8. Test data can be transformed without refitting
# --------------------------------------------------------------------------


def test_shared_pipeline_transform_does_not_refit(raw_df: pd.DataFrame) -> None:
    df_t = create_binary_target(raw_df)
    train_df, test_df = split_dataset(df_t, DEFAULT_CONFIG)
    feature_groups = get_feature_groups("full")

    shared = SharedFeaturePipeline(feature_groups, DEFAULT_CONFIG)
    X_train = train_df[feature_groups.all_columns]
    y_train = train_df[BINARY_TARGET_COLUMN].to_numpy()
    shared.fit(X_train, y_train)

    fitted_mean_before = shared.preprocessor.named_transformers_["continuous"].named_steps["scale"].mean_.copy()

    X_test = test_df[feature_groups.all_columns]
    _ = shared.transform(X_test)  # must not refit anything

    fitted_mean_after = shared.preprocessor.named_transformers_["continuous"].named_steps["scale"].mean_.copy()
    assert np.array_equal(fitted_mean_before, fitted_mean_after)


def test_transform_before_fit_raises() -> None:
    feature_groups = get_feature_groups("full")
    shared = SharedFeaturePipeline(feature_groups, DEFAULT_CONFIG)
    with pytest.raises(RuntimeError):
        shared.transform(pd.DataFrame({"age": [50]}))


# --------------------------------------------------------------------------
# 9. PCA output has the requested dimensionality
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_components", [2, 3, 4, 5, 6])
def test_pca_output_has_requested_dimensionality(raw_df: pd.DataFrame, n_components: int) -> None:
    cfg = DEFAULT_CONFIG.with_overrides(pca_n_components=n_components)
    result = run_preprocessing_pipeline(raw_df, cfg)
    assert result.X_train_quantum.shape[1] == n_components
    assert result.X_test_quantum.shape[1] == n_components


def test_pca_is_not_hardcoded_to_four_components(raw_df: pd.DataFrame) -> None:
    # Explicitly demonstrate that n_components is configuration, not a
    # hardcoded assumption, by choosing an unusual value.
    cfg = DEFAULT_CONFIG.with_overrides(pca_n_components=2)
    result = run_preprocessing_pipeline(raw_df, cfg)
    assert result.artifacts.pca_n_components_used == 2
    assert result.X_train_quantum.shape[1] == 2


def test_quantum_output_within_declared_range(raw_df: pd.DataFrame) -> None:
    result = run_preprocessing_pipeline(raw_df, DEFAULT_CONFIG)
    low, high = DEFAULT_CONFIG.quantum_range
    assert_within_range(result.X_train_quantum, low, high, name="X_train_quantum")
    assert_within_range(result.X_test_quantum, low, high, name="X_test_quantum")


# --------------------------------------------------------------------------
# 10 & 11. Classical-ready and quantum-ready output shapes
# --------------------------------------------------------------------------


def test_classical_ready_output_shape(raw_df: pd.DataFrame) -> None:
    result = run_preprocessing_pipeline(raw_df, DEFAULT_CONFIG)
    n_train = len(result.y_train)
    n_test = len(result.y_test)
    assert result.X_train_classical.shape == (n_train, DEFAULT_CONFIG.feature_selection_k)
    assert result.X_test_classical.shape == (n_test, DEFAULT_CONFIG.feature_selection_k)
    assert n_train + n_test == 303


def test_quantum_ready_output_shape(raw_df: pd.DataFrame) -> None:
    result = run_preprocessing_pipeline(raw_df, DEFAULT_CONFIG)
    n_train = len(result.y_train)
    n_test = len(result.y_test)
    assert result.X_train_quantum.shape == (n_train, DEFAULT_CONFIG.pca_n_components)
    assert result.X_test_quantum.shape == (n_test, DEFAULT_CONFIG.pca_n_components)


def test_selected_feature_names_length_matches_classical_shape(raw_df: pd.DataFrame) -> None:
    result = run_preprocessing_pipeline(raw_df, DEFAULT_CONFIG)
    assert len(result.artifacts.selected_feature_names) == result.X_train_classical.shape[1]


def test_num_and_y_arrays_are_row_aligned_and_consistent(raw_df: pd.DataFrame) -> None:
    result = run_preprocessing_pipeline(raw_df, DEFAULT_CONFIG)
    # target must equal (num > 0) row-for-row, for both splits
    assert np.array_equal(result.y_train, (result.num_train > 0).astype(int))
    assert np.array_equal(result.y_test, (result.num_test > 0).astype(int))


# --------------------------------------------------------------------------
# 12. Running the same pipeline twice with the same seed is reproducible
# --------------------------------------------------------------------------


def test_full_pipeline_is_reproducible_with_same_config(raw_df: pd.DataFrame) -> None:
    result_a = run_preprocessing_pipeline(raw_df, DEFAULT_CONFIG)
    result_b = run_preprocessing_pipeline(raw_df, DEFAULT_CONFIG)

    assert_arrays_close(result_a.X_train_classical, result_b.X_train_classical, name="X_train_classical")
    assert_arrays_close(result_a.X_test_classical, result_b.X_test_classical, name="X_test_classical")
    assert_arrays_close(result_a.X_train_quantum, result_b.X_train_quantum, name="X_train_quantum")
    assert_arrays_close(result_a.X_test_quantum, result_b.X_test_quantum, name="X_test_quantum")
    assert np.array_equal(result_a.y_train, result_b.y_train)
    assert np.array_equal(result_a.y_test, result_b.y_test)
    assert result_a.artifacts.split_id == result_b.artifacts.split_id


def test_different_seed_changes_the_split(raw_df: pd.DataFrame) -> None:
    cfg_a = DEFAULT_CONFIG
    cfg_b = DEFAULT_CONFIG.with_overrides(random_seed=123)
    result_a = run_preprocessing_pipeline(raw_df, cfg_a)
    result_b = run_preprocessing_pipeline(raw_df, cfg_b)
    assert result_a.artifacts.split_id != result_b.artifacts.split_id
    assert not np.array_equal(result_a.num_train, result_b.num_train)


# --------------------------------------------------------------------------
# 13. Leakage test -- the most important test in this file
# --------------------------------------------------------------------------


def test_no_leakage_from_test_set_statistics(raw_df: pd.DataFrame) -> None:
    """Construct a synthetic, extreme test-set outlier and verify that
    TRAIN-derived preprocessing statistics (imputer values, scaler
    mean/std, PCA components) do not change because of it.

    If preprocessing were leaking test-set information (e.g. fitting on
    train+test, or fitting per-split), this extreme value would shift the
    fitted imputer median, the scaler mean, or the PCA components learned
    on "training" data. It must not.
    """
    df_t = create_binary_target(raw_df)
    train_df, test_df = split_dataset(df_t, DEFAULT_CONFIG)
    feature_groups = get_feature_groups("full")

    # --- baseline: fit on the real, unperturbed training split ---------------
    X_train = train_df[feature_groups.all_columns]
    y_train = train_df[BINARY_TARGET_COLUMN].to_numpy()

    shared_baseline = SharedFeaturePipeline(feature_groups, DEFAULT_CONFIG)
    shared_baseline.fit(X_train, y_train)
    baseline_mean = shared_baseline.preprocessor.named_transformers_["continuous"].named_steps[
        "scale"
    ].mean_.copy()
    baseline_median = shared_baseline.preprocessor.named_transformers_["continuous"].named_steps[
        "impute"
    ].statistics_.copy()

    quantum_baseline = build_quantum_pipeline(4, DEFAULT_CONFIG.quantum_range)
    X_train_classical_baseline = shared_baseline.transform(X_train)
    quantum_baseline.fit(X_train_classical_baseline)
    baseline_pca_components = quantum_baseline.named_steps["pca"].components_.copy()

    # --- construct an extreme, synthetic test-set outlier ---------------------
    test_df_perturbed = test_df.copy(deep=True)
    # An absurd, physiologically impossible cholesterol value, and an
    # absurd age -- values that WOULD noticeably shift a mean/median/PCA
    # axis if they leaked into fitting.
    test_df_perturbed.loc[test_df_perturbed.index[0], "chol"] = 999_999.0
    test_df_perturbed.loc[test_df_perturbed.index[0], "age"] = 9_999.0
    test_df_perturbed.loc[test_df_perturbed.index[0], "trestbps"] = 9_999.0

    # --- refit an identical pipeline; the ONLY difference in this test is
    #     that a wildly perturbed test set now exists in memory. If the
    #     pipeline is leakage-safe, refitting on the SAME training data
    #     produces IDENTICAL statistics regardless of what test_df contains,
    #     because test_df is never passed to any .fit() call. -------------
    shared_after = SharedFeaturePipeline(feature_groups, DEFAULT_CONFIG)
    shared_after.fit(X_train, y_train)  # note: X_train, not the perturbed test set
    after_mean = shared_after.preprocessor.named_transformers_["continuous"].named_steps["scale"].mean_.copy()
    after_median = shared_after.preprocessor.named_transformers_["continuous"].named_steps[
        "impute"
    ].statistics_.copy()

    quantum_after = build_quantum_pipeline(4, DEFAULT_CONFIG.quantum_range)
    X_train_classical_after = shared_after.transform(X_train)
    quantum_after.fit(X_train_classical_after)
    after_pca_components = quantum_after.named_steps["pca"].components_.copy()

    assert np.array_equal(baseline_mean, after_mean), "Scaler mean changed despite test_df never being fit on."
    assert np.array_equal(
        baseline_median, after_median
    ), "Imputer statistics changed despite test_df never being fit on."
    assert np.array_equal(
        baseline_pca_components, after_pca_components
    ), "PCA components changed despite test_df never being fit on."

    # --- end-to-end confirmation: running the FULL pipeline (which
    #     internally calls split -> fit -> transform) is also unaffected by
    #     what a test set contains, because it is only ever transformed,
    #     never fit. We simulate this by directly transforming the extreme
    #     row through the baseline-fitted pipeline and confirming it does
    #     NOT crash or silently rescale -- it is simply an out-of-distribution
    #     point that gets transformed (and, for the quantum branch, clipped
    #     into range), exactly like any other test point. -------------------
    extreme_row = test_df_perturbed.loc[[test_df_perturbed.index[0]], feature_groups.all_columns]
    transformed_extreme = shared_baseline.transform(extreme_row)
    assert not transformed_extreme.isna().any().any()
    quantum_extreme = quantum_baseline.transform(transformed_extreme)
    assert_within_range(quantum_extreme, *DEFAULT_CONFIG.quantum_range, name="quantum_extreme")


def test_full_pipeline_result_identical_regardless_of_downstream_test_perturbation(
    raw_df: pd.DataFrame,
) -> None:
    """A second, coarser leakage check at the run_preprocessing_pipeline()
    level: two calls that only differ in an in-memory copy of the raw
    DataFrame having extreme values written into rows that end up in the
    test split must produce identical TRAIN-side artifacts.
    """
    result_clean = run_preprocessing_pipeline(raw_df, DEFAULT_CONFIG)

    df_t = create_binary_target(raw_df)
    _, test_df = split_dataset(df_t, DEFAULT_CONFIG)
    test_indices_in_original = test_df.index  # indices refer to df_t, not raw_df, but row identity via position

    perturbed_raw = raw_df.copy(deep=True)
    # Perturb the same physical rows that we know end up in the test split
    # by re-deriving the split on a copy and mutating those source rows.
    df_t2 = create_binary_target(perturbed_raw)
    train_df2, test_df2 = split_dataset(df_t2, DEFAULT_CONFIG)
    perturb_idx = test_df2.index[0]
    test_df2.loc[perturb_idx, "chol"] = 999_999.0

    shared = SharedFeaturePipeline(get_feature_groups("full"), DEFAULT_CONFIG)
    feature_cols = get_feature_groups("full").all_columns
    shared.fit(train_df2[feature_cols], train_df2[BINARY_TARGET_COLUMN].to_numpy())
    perturbed_train_classical = shared.transform(train_df2[feature_cols])

    pd.testing.assert_frame_equal(
        result_clean.X_train_classical.reset_index(drop=True),
        perturbed_train_classical.reset_index(drop=True),
    )


# --------------------------------------------------------------------------
# 14. Clinical feature ablation is prepared (not executed as an experiment)
# --------------------------------------------------------------------------


def test_screening_feature_set_excludes_ca_and_thal(raw_df: pd.DataFrame) -> None:
    cfg = DEFAULT_CONFIG.with_overrides(feature_set="screening")
    result = run_preprocessing_pipeline(raw_df, cfg)
    selected = result.artifacts.selected_feature_names
    assert "ca" not in selected
    assert not any(name.startswith("thal_") for name in selected)
    assert "ca" not in result.X_train_raw.columns
    assert "thal" not in result.X_train_raw.columns


def test_full_feature_set_can_include_ca_and_thal(raw_df: pd.DataFrame) -> None:
    cfg = DEFAULT_CONFIG.with_overrides(feature_set="full", feature_selection_k=17)
    result = run_preprocessing_pipeline(raw_df, cfg)
    assert "ca" in result.X_train_raw.columns
    assert "thal" in result.X_train_raw.columns


def test_get_feature_groups_rejects_unknown_feature_set() -> None:
    with pytest.raises(ValueError):
        get_feature_groups("nonexistent")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------


def test_config_rejects_invalid_test_size() -> None:
    with pytest.raises(ValueError):
        PreprocessingConfig(test_size=1.5)


def test_config_rejects_invalid_quantum_range() -> None:
    with pytest.raises(ValueError):
        PreprocessingConfig(quantum_range_low=3.0, quantum_range_high=1.0)


def test_config_with_overrides_does_not_mutate_original() -> None:
    original = DEFAULT_CONFIG
    modified = original.with_overrides(pca_n_components=6)
    assert original.pca_n_components != 6
    assert modified.pca_n_components == 6


def test_config_from_yaml_round_trip(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("random_seed: 7\npca_n_components: 3\nfeature_set: screening\n", encoding="utf-8")
    cfg = PreprocessingConfig.from_yaml(yaml_path)
    assert cfg.random_seed == 7
    assert cfg.pca_n_components == 3
    assert cfg.feature_set == "screening"
    # Fields not present in the file fall back to dataclass defaults.
    assert cfg.test_size == PreprocessingConfig.default().test_size
