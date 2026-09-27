"""Dataset inspection and structural validation for the UCI Heart Disease data.

This module is a READ-ONLY inspection tool. It loads a raw, unmodified UCI
Heart Disease "processed" file (e.g. ``data/raw/processed.cleveland.data``),
checks that its structure matches what QuantumDx expects, and prints a
data-quality report: shape, dtypes, missing values, duplicates, per-feature
statistics, categorical value domains, and target distribution.

It deliberately does NOT do any of the following (that is Phase 2 work):
    - impute or drop missing values
    - scale, encode, or otherwise transform features
    - perform feature selection or dimensionality reduction
    - binarize / persist a modified target column
    - write anything back to ``data/raw/``

Column order and semantics are taken from ``data/raw/heart-disease.names``
(UCI Heart Disease, Cleveland Clinic Foundation subset) and cross-checked
against PRD.md Section 19 / MVP_SPEC.md Section 5.

Usage (from the project root):
    python -m src.data.inspect_dataset
    python -m src.data.inspect_dataset --path data/raw/processed.hungarian.data
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Schema constants (source of truth: data/raw/heart-disease.names, section 7)
# --------------------------------------------------------------------------

#: The 14 columns present in every "processed.*.data" file, in file order.
#: There is no header row in these files.
COLUMN_NAMES: list[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

EXPECTED_N_COLUMNS: int = len(COLUMN_NAMES)

#: The raw multi-level diagnosis column. QuantumDx's binary target
#: (target = 1 if num > 0 else 0) is derived from this in the preprocessing
#: phase, NOT here.
TARGET_COLUMN: str = "num"

#: How missing values are encoded in the raw UCI files.
RAW_MISSING_TOKEN: str = "?"

#: Continuous numeric features.
NUMERIC_COLUMNS: list[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]

#: Discrete numeric / ordinal-count feature (0-3 vessels). Numeric dtype,
#: but a small closed set of values.
DISCRETE_NUMERIC_COLUMNS: list[str] = ["ca"]

#: Binary categorical features (already 0/1 coded in the source data).
BINARY_CATEGORICAL_COLUMNS: list[str] = ["sex", "fbs", "exang"]

#: Nominal categorical features (unordered levels).
NOMINAL_CATEGORICAL_COLUMNS: list[str] = ["cp", "restecg", "thal"]

#: Ordinal categorical features (ordered levels).
ORDINAL_CATEGORICAL_COLUMNS: list[str] = ["slope"]

ALL_CATEGORICAL_COLUMNS: list[str] = (
    BINARY_CATEGORICAL_COLUMNS + NOMINAL_CATEGORICAL_COLUMNS + ORDINAL_CATEGORICAL_COLUMNS
)

#: Known valid value sets per categorical column, per heart-disease.names.
#: Used only to FLAG unexpected values in the report -- never to filter data.
KNOWN_VALUE_DOMAINS: dict[str, set[float]] = {
    "sex": {0.0, 1.0},
    "cp": {1.0, 2.0, 3.0, 4.0},
    "fbs": {0.0, 1.0},
    "restecg": {0.0, 1.0, 2.0},
    "exang": {0.0, 1.0},
    "slope": {1.0, 2.0, 3.0},
    "ca": {0.0, 1.0, 2.0, 3.0},
    "thal": {3.0, 6.0, 7.0},
}

#: Loose physiological plausibility bounds for continuous features, used
#: only to flag rows for review. Not enforced / not a filter.
PHYSIOLOGICAL_RANGES: dict[str, tuple[float, float]] = {
    "age": (18.0, 110.0),
    "trestbps": (60.0, 260.0),
    "chol": (80.0, 700.0),
    "thalach": (50.0, 230.0),
    "oldpeak": (0.0, 8.0),
}

#: Columns where a value of exactly 0 is clinically implausible and is a
#: known sentinel-for-missing pattern in some UCI Heart Disease sites
#: (e.g. all 123 Switzerland records have chol == 0). Flagged, not altered.
ZERO_IS_SENTINEL_COLUMNS: list[str] = ["chol", "trestbps"]

#: Default dataset used for MVP Phase 1 development.
DEFAULT_RELATIVE_PATH = Path("data") / "raw" / "processed.cleveland.data"


class DatasetValidationError(Exception):
    """Raised when the loaded file does not match the expected UCI schema."""


# --------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------


def find_project_root(start: Path | None = None) -> Path:
    """Locate the QuantumDx project root regardless of the current working directory.

    Walks upward from ``start`` (default: this file's directory) looking for a
    directory that contains both a ``data`` folder and ``PRD.md``, which
    uniquely identifies the QuantumDx repository root.

    Args:
        start: Directory to begin searching from.

    Returns:
        The resolved project root path.

    Raises:
        FileNotFoundError: If no matching ancestor directory is found.
    """
    current = (start or Path(__file__).resolve().parent).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "PRD.md").is_file() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the QuantumDx project root (a directory containing "
        "both 'PRD.md' and a 'data/' folder) starting from "
        f"'{current}'."
    )


def resolve_dataset_path(path: str | Path | None) -> Path:
    """Resolve a dataset path relative to the project root if it is not absolute.

    Args:
        path: A user-supplied path, or None to use the default Cleveland file.

    Returns:
        An absolute, resolved Path to the dataset file.
    """
    root = find_project_root()
    if path is None:
        return (root / DEFAULT_RELATIVE_PATH).resolve()
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load a raw UCI "processed.*.data" file, unmodified except for column naming.

    The only transformation applied is:
        - assigning the 14 known column names (the file has no header row)
        - recognising the literal string "?" as a missing-value marker so it
          becomes NaN instead of an unparseable string

    No imputation, scaling, encoding, filtering, or target binarization is
    performed here.

    Args:
        path: Path to a "processed.*.data" CSV file.

    Returns:
        A DataFrame with 14 named columns, numeric dtypes where parseable,
        and NaN wherever the source file had "?".

    Raises:
        DatasetValidationError: If the file is missing, empty, or does not
            parse into the expected shape.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise DatasetValidationError(f"Dataset file not found: '{resolved}'.")

    try:
        df = pd.read_csv(
            resolved,
            header=None,
            names=COLUMN_NAMES,
            na_values=[RAW_MISSING_TOKEN],
            sep=",",
        )
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
        raise DatasetValidationError(
            f"Failed to parse '{resolved}' as a {EXPECTED_N_COLUMNS}-column "
            f"comma-separated UCI Heart Disease file: {exc}"
        ) from exc

    return df


# --------------------------------------------------------------------------
# Structural validation
# --------------------------------------------------------------------------


def validate_structure(df: pd.DataFrame, source: str = "<dataframe>") -> None:
    """Validate that a loaded dataset matches the expected UCI Heart Disease schema.

    This is a structural check only (shape, columns, parseability, target
    presence). It does not check data quality (that is what the rest of the
    inspection report is for) and it never mutates ``df``.

    Args:
        df: The loaded dataset.
        source: A label for the data source, used in error messages.

    Raises:
        DatasetValidationError: On any structural mismatch, with a message
            naming exactly what is wrong.
    """
    if df.empty:
        raise DatasetValidationError(f"'{source}' loaded but contains zero rows.")

    if list(df.columns) != COLUMN_NAMES:
        raise DatasetValidationError(
            f"'{source}' has unexpected columns.\n"
            f"  expected: {COLUMN_NAMES}\n"
            f"  found:    {list(df.columns)}"
        )

    if df.shape[1] != EXPECTED_N_COLUMNS:
        raise DatasetValidationError(
            f"'{source}' has {df.shape[1]} columns; expected {EXPECTED_N_COLUMNS}."
        )

    if TARGET_COLUMN not in df.columns:
        raise DatasetValidationError(
            f"'{source}' is missing the target column '{TARGET_COLUMN}'."
        )

    if df[TARGET_COLUMN].isna().any():
        n_missing = int(df[TARGET_COLUMN].isna().sum())
        raise DatasetValidationError(
            f"'{source}' has {n_missing} row(s) with a missing target "
            f"('{TARGET_COLUMN}'). Every row must have a known diagnosis label."
        )

    non_target_cols = [c for c in df.columns if c != TARGET_COLUMN]
    unparseable: list[str] = []
    for col in non_target_cols:
        coerced = pd.to_numeric(df[col], errors="coerce")
        # A value is "unparseable" if to_numeric produced a NaN where the
        # original was not already NaN (i.e. it was junk, not a declared
        # missing marker).
        bad_mask = coerced.isna() & df[col].notna()
        if bad_mask.any():
            unparseable.append(col)
    if unparseable:
        raise DatasetValidationError(
            f"'{source}' has non-numeric, non-'{RAW_MISSING_TOKEN}' values in "
            f"column(s): {unparseable}. Every feature must be numeric or the "
            f"declared missing-value token."
        )

    if df[TARGET_COLUMN].nunique() < 2:
        raise DatasetValidationError(
            f"'{source}' target column '{TARGET_COLUMN}' has fewer than 2 "
            f"distinct values; cannot support a classification task."
        )


# --------------------------------------------------------------------------
# Report data structures
# --------------------------------------------------------------------------


@dataclass
class DatasetQualityReport:
    """Aggregates every inspection result for one dataset file."""

    source_path: Path
    n_rows: int
    n_columns: int
    dtypes: pd.Series
    missing_counts: pd.Series
    missing_percent: pd.Series
    n_duplicate_rows: int
    duplicate_row_indices: list[int]
    numeric_summary: pd.DataFrame
    categorical_value_counts: dict[str, pd.Series]
    domain_violations: dict[str, list[float]]
    zero_sentinel_counts: dict[str, int]
    out_of_range_counts: dict[str, int]
    zero_variance_columns: list[str]
    raw_target_distribution: pd.Series
    binary_target_preview: pd.Series
    warnings: list[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def compute_missing_report(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (missing_counts, missing_percent) per column, columns with 0 included."""
    counts = df.isna().sum()
    percent = (counts / len(df) * 100).round(2)
    return counts, percent


def compute_duplicate_report(df: pd.DataFrame) -> tuple[int, list[int]]:
    """Return (count, index_list) of exact duplicate rows (all columns equal)."""
    dup_mask = df.duplicated(keep="first")
    return int(dup_mask.sum()), df.index[dup_mask].tolist()


def compute_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return min/max/mean/median/std/n-unique for numeric + discrete-numeric columns."""
    cols = NUMERIC_COLUMNS + DISCRETE_NUMERIC_COLUMNS
    summary = df[cols].agg(["min", "max", "mean", "median", "std"]).T
    summary["n_unique"] = df[cols].nunique()
    return summary.round(3)


def compute_categorical_value_counts(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return value_counts() for every categorical column (NaN excluded, reported separately)."""
    return {col: df[col].value_counts(dropna=True).sort_index() for col in ALL_CATEGORICAL_COLUMNS}


def detect_domain_violations(df: pd.DataFrame) -> dict[str, list[float]]:
    """Flag categorical values outside the known UCI value domain for that column.

    Returns a dict mapping column -> sorted list of unexpected values found.
    Columns with no violations are omitted.
    """
    violations: dict[str, list[float]] = {}
    for col, domain in KNOWN_VALUE_DOMAINS.items():
        observed = set(df[col].dropna().unique().tolist())
        unexpected = sorted(observed - domain)
        if unexpected:
            violations[col] = unexpected
    return violations


def detect_zero_sentinels(df: pd.DataFrame) -> dict[str, int]:
    """Count clinically-implausible zero values in columns known to use 0-as-missing."""
    return {col: int((df[col] == 0).sum()) for col in ZERO_IS_SENTINEL_COLUMNS if col in df.columns}


def detect_out_of_range(df: pd.DataFrame) -> dict[str, int]:
    """Count rows outside the loose physiological plausibility bounds, per column."""
    result: dict[str, int] = {}
    for col, (low, high) in PHYSIOLOGICAL_RANGES.items():
        out = ((df[col] < low) | (df[col] > high)) & df[col].notna()
        result[col] = int(out.sum())
    return result


def detect_zero_variance_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns (excluding target) with a single distinct non-null value."""
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    return [c for c in feature_cols if df[c].dropna().nunique() <= 1]


def compute_target_distribution(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (raw 0-4 distribution, binary at-risk-preview distribution).

    The binary series is an INSPECTION PREVIEW ONLY of the
    ``target = 1 if num > 0 else 0`` rule defined in PRD.md Section 19.2 /
    MVP_SPEC.md Section 5.2. The actual binarization for modeling happens in
    the preprocessing pipeline (Phase 2), not here.
    """
    raw = df[TARGET_COLUMN].value_counts().sort_index()
    binary = (df[TARGET_COLUMN] > 0).astype(int).value_counts().sort_index()
    binary.index = binary.index.map({0: "0 (no disease)", 1: "1 (at risk)"})
    return raw, binary


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def build_report(df: pd.DataFrame, source_path: Path) -> DatasetQualityReport:
    """Run every inspection check and assemble a single DatasetQualityReport."""
    missing_counts, missing_percent = compute_missing_report(df)
    n_dup, dup_idx = compute_duplicate_report(df)
    domain_violations = detect_domain_violations(df)
    zero_sentinels = detect_zero_sentinels(df)
    out_of_range = detect_out_of_range(df)
    zero_variance = detect_zero_variance_columns(df)
    raw_target, binary_target = compute_target_distribution(df)

    warnings: list[str] = []
    if n_dup:
        warnings.append(f"{n_dup} exact duplicate row(s) found.")
    if domain_violations:
        warnings.append(f"Unexpected categorical values found: {domain_violations}.")
    nonzero_sentinels = {k: v for k, v in zero_sentinels.items() if v > 0}
    if nonzero_sentinels:
        warnings.append(
            f"Clinically-implausible zero values (likely missing-value sentinels): "
            f"{nonzero_sentinels}."
        )
    nonzero_out_of_range = {k: v for k, v in out_of_range.items() if v > 0}
    if nonzero_out_of_range:
        warnings.append(f"Values outside physiological plausibility bounds: {nonzero_out_of_range}.")
    if zero_variance:
        warnings.append(f"Zero-variance feature column(s): {zero_variance}.")
    minority_share = binary_target.min() / binary_target.sum()
    if minority_share < 0.20:
        warnings.append(
            f"Minority class share is {minority_share:.1%}, below the 20% imbalance threshold."
        )

    return DatasetQualityReport(
        source_path=source_path,
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        dtypes=df.dtypes,
        missing_counts=missing_counts,
        missing_percent=missing_percent,
        n_duplicate_rows=n_dup,
        duplicate_row_indices=dup_idx,
        numeric_summary=compute_numeric_summary(df),
        categorical_value_counts=compute_categorical_value_counts(df),
        domain_violations=domain_violations,
        zero_sentinel_counts=zero_sentinels,
        out_of_range_counts=out_of_range,
        zero_variance_columns=zero_variance,
        raw_target_distribution=raw_target,
        binary_target_preview=binary_target,
        warnings=warnings,
    )


def inspect_dataset(path: str | Path | None = None) -> DatasetQualityReport:
    """Load, validate, and inspect a UCI Heart Disease dataset file end to end.

    Args:
        path: Path to a "processed.*.data" file. Defaults to
            ``data/raw/processed.cleveland.data`` when None.

    Returns:
        A populated DatasetQualityReport.

    Raises:
        DatasetValidationError: If the file cannot be loaded or does not
            match the expected structure.
    """
    resolved = resolve_dataset_path(path)
    df = load_raw_dataset(resolved)
    validate_structure(df, source=str(resolved))
    return build_report(df, source_path=resolved)


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------


def _section(title: str) -> str:
    bar = "=" * len(title)
    return f"\n{title}\n{bar}"


def format_report(report: DatasetQualityReport) -> str:
    """Render a DatasetQualityReport as a human-readable text report."""
    lines: list[str] = []
    lines.append(_section("QuantumDx Dataset Inspection Report"))
    lines.append(f"Source file       : {report.source_path}")
    lines.append(f"Rows x Columns    : {report.n_rows} x {report.n_columns}")

    lines.append(_section("Column Types (as loaded)"))
    lines.append(report.dtypes.to_string())

    lines.append(_section("Missing Values (per column)"))
    missing_df = pd.DataFrame(
        {"missing_count": report.missing_counts, "missing_percent": report.missing_percent}
    )
    nonzero_missing = missing_df[missing_df["missing_count"] > 0]
    if nonzero_missing.empty:
        lines.append("No missing values detected in any column.")
    else:
        lines.append(nonzero_missing.to_string())
    lines.append(f"\n(Missing values are encoded as '{RAW_MISSING_TOKEN}' in the raw file.)")

    lines.append(_section("Duplicate Rows"))
    lines.append(f"Exact duplicate rows: {report.n_duplicate_rows}")
    if report.duplicate_row_indices:
        lines.append(f"Duplicate row indices: {report.duplicate_row_indices}")

    lines.append(_section("Numeric Feature Summary"))
    lines.append(report.numeric_summary.to_string())

    lines.append(_section("Categorical Feature Value Counts"))
    for col, counts in report.categorical_value_counts.items():
        lines.append(f"\n{col}:")
        lines.append(counts.to_string())

    lines.append(_section("Domain / Quality Flags"))
    if report.domain_violations:
        lines.append(f"Unexpected categorical values: {report.domain_violations}")
    else:
        lines.append("No categorical values outside the known UCI value domains.")
    nonzero_sentinels = {k: v for k, v in report.zero_sentinel_counts.items() if v > 0}
    lines.append(
        f"Clinically-implausible zero values (chol/trestbps): "
        f"{nonzero_sentinels if nonzero_sentinels else 'none found'}"
    )
    nonzero_range = {k: v for k, v in report.out_of_range_counts.items() if v > 0}
    lines.append(
        f"Values outside physiological plausibility bounds: "
        f"{nonzero_range if nonzero_range else 'none found'}"
    )
    lines.append(
        f"Zero-variance feature columns: "
        f"{report.zero_variance_columns if report.zero_variance_columns else 'none found'}"
    )

    lines.append(_section("Target Distribution"))
    lines.append("Raw 'num' (0=no disease, 1-4=increasing severity):")
    lines.append(report.raw_target_distribution.to_string())
    lines.append("\nBinary preview only (target = 1 if num > 0 else 0):")
    lines.append(report.binary_target_preview.to_string())
    total = report.binary_target_preview.sum()
    minority = report.binary_target_preview.min()
    lines.append(f"\nMinority class share: {minority / total:.1%}")
    lines.append(
        "(This binarization is a PREVIEW for reporting only. The modeling "
        "target is derived in the Phase 2 preprocessing pipeline, not here.)"
    )

    lines.append(_section("Warnings Summary"))
    if report.warnings:
        for w in report.warnings:
            lines.append(f"  - {w}")
    else:
        lines.append("No data-quality warnings raised.")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 = success, 1 = failure)."""
    parser = argparse.ArgumentParser(
        description="Inspect and structurally validate a UCI Heart Disease dataset file."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help=(
            "Path to a processed.*.data file, relative to the project root "
            "or absolute. Defaults to data/raw/processed.cleveland.data."
        ),
    )
    args = parser.parse_args(argv)

    try:
        report = inspect_dataset(args.path)
    except DatasetValidationError as exc:
        print(f"ERROR: dataset validation failed.\n  {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
