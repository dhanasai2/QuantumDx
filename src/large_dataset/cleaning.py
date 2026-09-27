"""Documented, non-destructive data-quality flagging for cardio_train.csv.

Per the phase instructions: "Do not remove suspicious values silently.
Every cleaning decision must be documented." This module NEVER deletes a
row in place -- it adds a boolean flag column and returns both the full
(flagged) frame and the modeling subset as separate, explicit outputs, so
the exclusion decision is always visible and reversible.

Thresholds are stated explicitly below, each with its clinical rationale,
so a reviewer can disagree with a specific threshold without having to
reverse-engineer it from code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Physiologically-motivated bounds. These are not statistical outlier
# bounds (e.g. IQR-based) -- they are clinical plausibility bounds, chosen
# to catch obvious data-entry errors (e.g. a decimal point typo turning
# 160 into 1600) without discarding genuine extreme-but-real measurements.
AP_HI_MIN, AP_HI_MAX = 60, 250   # systolic BP; <60 or >250 mmHg is not a
                                  # plausible reading for a living outpatient
AP_LO_MIN, AP_LO_MAX = 40, 200   # diastolic BP; same reasoning
HEIGHT_MIN, HEIGHT_MAX = 100, 220  # cm; bounds an adult population
WEIGHT_MIN = 30                   # kg; below this is implausible for an
                                   # adult in this age range (29-65 years)


@dataclass(frozen=True)
class CleaningFlags:
    """One boolean Series per implausibility check, plus their union.

    Kept as a dataclass (rather than just adding columns to the DataFrame)
    so each individual flag remains inspectable on its own -- e.g. to
    answer "how many rows are flagged for ap_lo specifically" without
    reconstructing the flag from bounds again.
    """

    ap_hi_implausible: pd.Series
    ap_lo_implausible: pd.Series
    ap_hi_below_ap_lo: pd.Series
    height_implausible: pd.Series
    weight_implausible: pd.Series

    @property
    def any_flag(self) -> pd.Series:
        return (
            self.ap_hi_implausible
            | self.ap_lo_implausible
            | self.ap_hi_below_ap_lo
            | self.height_implausible
            | self.weight_implausible
        )

    def summary(self, n_total: int) -> dict:
        any_flag = self.any_flag
        return {
            "n_total": n_total,
            "ap_hi_implausible": int(self.ap_hi_implausible.sum()),
            "ap_lo_implausible": int(self.ap_lo_implausible.sum()),
            "ap_hi_below_ap_lo": int(self.ap_hi_below_ap_lo.sum()),
            "height_implausible": int(self.height_implausible.sum()),
            "weight_implausible": int(self.weight_implausible.sum()),
            "any_flag_count": int(any_flag.sum()),
            "any_flag_percent": float(any_flag.mean() * 100),
            "clean_count": int((~any_flag).sum()),
            "clean_percent": float((~any_flag).mean() * 100),
        }


def compute_cleaning_flags(df: pd.DataFrame) -> CleaningFlags:
    """Compute every implausibility flag. Never mutates `df`."""
    ap_hi_implausible = (df["ap_hi"] <= 0) | (df["ap_hi"] < AP_HI_MIN) | (df["ap_hi"] > AP_HI_MAX)
    ap_lo_implausible = (df["ap_lo"] <= 0) | (df["ap_lo"] < AP_LO_MIN) | (df["ap_lo"] > AP_LO_MAX)
    ap_hi_below_ap_lo = df["ap_hi"] < df["ap_lo"]
    height_implausible = (df["height"] < HEIGHT_MIN) | (df["height"] > HEIGHT_MAX)
    weight_implausible = df["weight"] < WEIGHT_MIN

    return CleaningFlags(
        ap_hi_implausible=ap_hi_implausible,
        ap_lo_implausible=ap_lo_implausible,
        ap_hi_below_ap_lo=ap_hi_below_ap_lo,
        height_implausible=height_implausible,
        weight_implausible=weight_implausible,
    )


def target_rate_by_flag(df: pd.DataFrame, flags: CleaningFlags, target_column: str) -> dict:
    """The finding that MUST be surfaced, not buried: flagged rows have a
    substantially different target rate than clean rows (measured: 75.1%
    vs 49.5% in the full 70,000-row audit). Excluding flagged rows is
    therefore not a neutral action on class balance.
    """
    any_flag = flags.any_flag
    return {
        "target_rate_flagged": float(df.loc[any_flag, target_column].mean()) if any_flag.any() else None,
        "target_rate_clean": float(df.loc[~any_flag, target_column].mean()) if (~any_flag).any() else None,
        "n_flagged": int(any_flag.sum()),
        "n_clean": int((~any_flag).sum()),
    }


def split_modeling_subset(
    df: pd.DataFrame, flags: CleaningFlags
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (modeling_df, excluded_df) -- both are views into the SAME
    rows as the input, just partitioned. Nothing is deleted; the excluded
    partition is returned so it remains inspectable, not discarded.

    DOCUMENTED DECISION (see docs/LARGE_DATASET.md for full reasoning):
    rows with any implausibility flag are excluded from the modeling
    dataset used for training/evaluation in this phase. This is the
    DEFAULT applied here. The alternative of swap-correcting
    ap_hi/ap_lo when ap_hi < ap_lo (a plausible column-transposition
    explanation for that specific flag) was considered and NOT applied,
    because it assumes a specific mechanism for the error rather than
    only describing that the value is invalid -- that assumption is not
    verifiable from the file alone.
    """
    any_flag = flags.any_flag
    modeling_df = df.loc[~any_flag].reset_index(drop=True)
    excluded_df = df.loc[any_flag].reset_index(drop=True)
    return modeling_df, excluded_df
