"""Formal, reproducible audit of cardio_train.csv (Phase 6, Section 4).

Run as:
    python -m src.large_dataset.audit

Produces results/large_dataset/audit/audit_report.json (machine-readable)
and prints a human-readable summary. This script is READ-ONLY with respect
to data/external/cardio_train.csv -- it never writes to that file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.inspect_dataset import find_project_root
from src.large_dataset.cleaning import compute_cleaning_flags, target_rate_by_flag
from src.large_dataset.schema import (
    NON_FEATURE_COLUMNS,
    RAW_TARGET_COLUMN,
    derive_features,
    load_raw_cardio,
    resolve_cardio_path,
)

RESULTS_DIR = find_project_root() / "results" / "large_dataset" / "audit"


def run_audit() -> dict:
    path = resolve_cardio_path()
    df = load_raw_cardio(path)
    df = derive_features(df)

    n_total, n_cols = df.shape

    missing = df.isna().sum()
    missing_report = {c: int(v) for c, v in missing.items() if v > 0}

    dup_full = int(df.drop(columns=["id", "target"]).duplicated().sum())
    dup_id = int(df["id"].duplicated().sum())

    target_counts = df[RAW_TARGET_COLUMN].value_counts().sort_index()
    target_report = {
        "counts": {int(k): int(v) for k, v in target_counts.items()},
        "positive_rate": float(df[RAW_TARGET_COLUMN].mean()),
    }

    numeric_cols = ["age_years", "height", "weight", "ap_hi", "ap_lo"]
    numeric_summary = df[numeric_cols].describe().T.to_dict(orient="index")

    categorical_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
    categorical_summary = {
        c: {int(k): int(v) for k, v in df[c].value_counts().sort_index().items()}
        for c in categorical_cols
    }

    zero_variance = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and df[c].nunique(dropna=True) <= 1]

    flags = compute_cleaning_flags(df)
    cleaning_summary = flags.summary(n_total)
    target_by_flag = target_rate_by_flag(df, flags, RAW_TARGET_COLUMN)

    report = {
        "source_file": str(path),
        "shape": {"rows": n_total, "columns": n_cols},
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": missing_report,
        "duplicate_rows_excluding_id": dup_full,
        "duplicate_id_values": dup_id,
        "target": target_report,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "zero_variance_columns": zero_variance,
        "cleaning_flags": cleaning_summary,
        "target_rate_by_cleaning_flag": target_by_flag,
        "known_dataset_specific_notes": [
            "age is stored in DAYS, not years (age_years derived: age/365.25).",
            "gender in {1,2}: documented (assumed, not independently verifiable) "
            "convention is 1=women, 2=men; gender_male derived accordingly.",
            "cholesterol and gluc are ORDINAL 3-level categories (1/2/3), "
            "NOT the same measurement as Cleveland's continuous mg/dL chol.",
            "This dataset has no analog to Cleveland's cp/restecg/thal/slope/ca/"
            "exang/oldpeak/fbs/trestbps -- it is an independent feature space, "
            "not a reformatting of Cleveland.",
        ],
    }
    return report


def print_report(report: dict) -> None:
    print("=" * 70)
    print("Large Dataset Audit -- cardio_train.csv")
    print("=" * 70)
    print(f"Source: {report['source_file']}")
    print(f"Shape: {report['shape']['rows']} rows x {report['shape']['columns']} columns")
    print(f"Missing values: {report['missing_values'] or 'none'}")
    print(f"Duplicate rows (excl. id): {report['duplicate_rows_excluding_id']}")
    print(f"Duplicate id values: {report['duplicate_id_values']}")
    print(f"Target positive rate: {report['target']['positive_rate']:.4f} "
          f"(counts={report['target']['counts']})")
    print(f"Zero-variance columns: {report['zero_variance_columns'] or 'none'}")
    print()
    print("Cleaning flags:")
    for k, v in report["cleaning_flags"].items():
        print(f"  {k}: {v}")
    print()
    print("Target rate by cleaning flag (the finding that must not be buried):")
    for k, v in report["target_rate_by_cleaning_flag"].items():
        print(f"  {k}: {v}")
    print("=" * 70)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = run_audit()
    print_report(report)
    out_path = RESULTS_DIR / "audit_report.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\nFull report written to: {out_path}")


if __name__ == "__main__":
    main()
