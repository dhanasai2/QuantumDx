"""Tests for src.data.inspect_dataset.

These are structural / data-quality tests for the raw UCI Heart Disease
(Cleveland) dataset. They do NOT test any preprocessing, modeling, or
quantum code -- none of that exists yet (Phase 1 scope only).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.inspect_dataset import (
    COLUMN_NAMES,
    DatasetValidationError,
    TARGET_COLUMN,
    build_report,
    detect_domain_violations,
    detect_zero_sentinels,
    find_project_root,
    inspect_dataset,
    load_raw_dataset,
    resolve_dataset_path,
    validate_structure,
)

CLEVELAND_RELATIVE_PATH = Path("data") / "raw" / "processed.cleveland.data"


@pytest.fixture(scope="module")
def project_root() -> Path:
    return find_project_root()


@pytest.fixture(scope="module")
def cleveland_path(project_root: Path) -> Path:
    path = project_root / CLEVELAND_RELATIVE_PATH
    if not path.is_file():
        pytest.skip(f"Cleveland dataset not present at {path}; skipping dataset tests.")
    return path


@pytest.fixture(scope="module")
def cleveland_df(cleveland_path: Path) -> pd.DataFrame:
    return load_raw_dataset(cleveland_path)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_dataset_loads_successfully(cleveland_df: pd.DataFrame) -> None:
    assert cleveland_df is not None
    assert not cleveland_df.empty


def test_load_missing_file_raises_dataset_validation_error(tmp_path: Path) -> None:
    missing_file = tmp_path / "does_not_exist.data"
    with pytest.raises(DatasetValidationError):
        load_raw_dataset(missing_file)


def test_question_mark_is_parsed_as_missing(cleveland_df: pd.DataFrame) -> None:
    # The raw file is known to contain '?' sentinels; they must become NaN,
    # never the literal string '?'.
    assert not (cleveland_df.select_dtypes(include="object").astype(str) == "?").any().any()


# --------------------------------------------------------------------------
# Expected shape / columns
# --------------------------------------------------------------------------


def test_expected_number_of_columns(cleveland_df: pd.DataFrame) -> None:
    assert cleveland_df.shape[1] == 14


def test_expected_column_names_and_order(cleveland_df: pd.DataFrame) -> None:
    assert list(cleveland_df.columns) == COLUMN_NAMES


def test_expected_number_of_rows(cleveland_df: pd.DataFrame) -> None:
    # The Cleveland subset is documented (heart-disease.names) as 303 instances.
    assert cleveland_df.shape[0] == 303


# --------------------------------------------------------------------------
# Target existence
# --------------------------------------------------------------------------


def test_target_column_exists(cleveland_df: pd.DataFrame) -> None:
    assert TARGET_COLUMN in cleveland_df.columns


def test_target_column_has_no_missing_values(cleveland_df: pd.DataFrame) -> None:
    assert cleveland_df[TARGET_COLUMN].isna().sum() == 0


def test_target_column_has_at_least_two_classes(cleveland_df: pd.DataFrame) -> None:
    assert cleveland_df[TARGET_COLUMN].nunique() >= 2


def test_target_raw_values_within_documented_range(cleveland_df: pd.DataFrame) -> None:
    # heart-disease.names documents num as integer-valued 0 (no presence) to 4.
    assert cleveland_df[TARGET_COLUMN].min() >= 0
    assert cleveland_df[TARGET_COLUMN].max() <= 4


# --------------------------------------------------------------------------
# Missing-value detection
# --------------------------------------------------------------------------


def test_missing_values_detected_in_ca_and_thal(cleveland_df: pd.DataFrame) -> None:
    # Documented in PRD.md / MVP_SPEC.md and confirmed against the raw file:
    # 4 missing 'ca' values and 2 missing 'thal' values, encoded as '?'.
    assert cleveland_df["ca"].isna().sum() == 4
    assert cleveland_df["thal"].isna().sum() == 2


def test_no_missing_values_outside_ca_and_thal(cleveland_df: pd.DataFrame) -> None:
    missing_counts = cleveland_df.isna().sum()
    columns_with_missing = set(missing_counts[missing_counts > 0].index)
    assert columns_with_missing == {"ca", "thal"}


def test_zero_sentinel_detection_runs_without_error(cleveland_df: pd.DataFrame) -> None:
    result = detect_zero_sentinels(cleveland_df)
    assert isinstance(result, dict)
    assert "chol" in result and "trestbps" in result


# --------------------------------------------------------------------------
# Structural validation
# --------------------------------------------------------------------------


def test_validate_structure_accepts_valid_dataframe(cleveland_df: pd.DataFrame) -> None:
    # Should not raise.
    validate_structure(cleveland_df, source="cleveland_fixture")


def test_validate_structure_rejects_wrong_column_count() -> None:
    bad_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(DatasetValidationError):
        validate_structure(bad_df, source="bad_fixture")


def test_validate_structure_rejects_missing_target_column() -> None:
    cols = [c for c in COLUMN_NAMES if c != TARGET_COLUMN]
    bad_df = pd.DataFrame({c: [0, 1] for c in cols})
    with pytest.raises(DatasetValidationError):
        validate_structure(bad_df, source="no_target_fixture")


def test_validate_structure_rejects_empty_dataframe() -> None:
    empty_df = pd.DataFrame(columns=COLUMN_NAMES)
    with pytest.raises(DatasetValidationError):
        validate_structure(empty_df, source="empty_fixture")


def test_validate_structure_rejects_single_class_target() -> None:
    data = {c: [0.0] * 5 for c in COLUMN_NAMES}
    data[TARGET_COLUMN] = [0, 0, 0, 0, 0]  # only one class present
    single_class_df = pd.DataFrame(data)
    with pytest.raises(DatasetValidationError):
        validate_structure(single_class_df, source="single_class_fixture")


def test_validate_structure_rejects_non_numeric_junk() -> None:
    data = {c: [0.0, 1.0, 2.0] for c in COLUMN_NAMES}
    data["age"] = [0.0, "not-a-number", 2.0]
    data[TARGET_COLUMN] = [0, 1, 0]
    junk_df = pd.DataFrame(data)
    with pytest.raises(DatasetValidationError):
        validate_structure(junk_df, source="junk_fixture")


# --------------------------------------------------------------------------
# Domain checks
# --------------------------------------------------------------------------


def test_no_categorical_domain_violations_in_cleveland(cleveland_df: pd.DataFrame) -> None:
    assert detect_domain_violations(cleveland_df) == {}


# --------------------------------------------------------------------------
# Class distribution
# --------------------------------------------------------------------------


def test_binary_target_distribution_matches_documented_counts(cleveland_df: pd.DataFrame) -> None:
    # 164 records with num == 0, 139 with num > 0 (55+36+35+13), per
    # heart-disease.names Section 10 (Cleveland row) and DATASET.md.
    binary = (cleveland_df[TARGET_COLUMN] > 0).astype(int).value_counts()
    assert binary[0] == 164
    assert binary[1] == 139


def test_raw_target_distribution_matches_documented_counts(cleveland_df: pd.DataFrame) -> None:
    raw_counts = cleveland_df[TARGET_COLUMN].value_counts().sort_index()
    assert raw_counts.tolist() == [164, 55, 36, 35, 13]


# --------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------


def test_no_duplicate_rows_in_cleveland(cleveland_df: pd.DataFrame) -> None:
    assert cleveland_df.duplicated().sum() == 0


# --------------------------------------------------------------------------
# Raw file immutability (the single most important invariant of Phase 1)
# --------------------------------------------------------------------------


def test_raw_file_is_not_modified_by_inspection(cleveland_path: Path) -> None:
    before = cleveland_path.read_bytes()
    load_raw_dataset(cleveland_path)
    build_report(load_raw_dataset(cleveland_path), source_path=cleveland_path)
    after = cleveland_path.read_bytes()
    assert before == after, "Inspecting the dataset must never modify data/raw/."


# --------------------------------------------------------------------------
# End-to-end orchestration
# --------------------------------------------------------------------------


def test_inspect_dataset_end_to_end_default_path() -> None:
    report = inspect_dataset(None)
    assert report.n_rows == 303
    assert report.n_columns == 14


def test_inspect_dataset_raises_meaningful_error_for_missing_file() -> None:
    with pytest.raises(DatasetValidationError):
        inspect_dataset("data/raw/this_file_does_not_exist.data")


def test_resolve_dataset_path_is_absolute_and_within_project(project_root: Path) -> None:
    resolved = resolve_dataset_path(None)
    assert resolved.is_absolute()
    assert str(resolved).startswith(str(project_root))
