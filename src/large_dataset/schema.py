"""Dataset-specific schema for the large cardiovascular dataset (cardio_train.csv).

This is intentionally a SEPARATE, INDEPENDENT schema from Cleveland's
(src.data.inspect_dataset / src.preprocessing.pipeline.get_feature_groups).
No column here is claimed to be the "same feature" as a Cleveland column --
see docs/LARGE_DATASET.md for the explicit statement of what is and is not
comparable across the two datasets.

Source: the well-known Kaggle "Cardiovascular Disease dataset" (70,000
records), semicolon-delimited, columns:
    id;age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio

Column meanings (per the file itself and its public documentation):
    age         -- age in DAYS (not years!) -- confirmed empirically:
                   range 10798-23713 days = 29.6-64.9 years, a plausible
                   adult age range, which is how this was identified as
                   days rather than a data error.
    gender      -- {1, 2}. This dataset's own public documentation states
                   1=women, 2=men. This mapping is NOT independently
                   verifiable from the file alone (no ground truth to
                   check it against) -- it is used here as a documented
                   ASSUMPTION, not a verified fact, and is flagged as such
                   in docs/LARGE_DATASET.md.
    height      -- cm
    weight      -- kg
    ap_hi       -- systolic blood pressure (mmHg)
    ap_lo       -- diastolic blood pressure (mmHg)
    cholesterol -- ORDINAL, 1=normal, 2=above normal, 3=well above normal
                   (NOT the same measurement as Cleveland's continuous
                   mg/dL `chol` -- do not conflate them)
    gluc        -- ORDINAL, same 3-level convention as cholesterol
    smoke       -- binary
    alco        -- binary (alcohol intake)
    active      -- binary (physical activity)
    cardio      -- TARGET, binary, already 0/1 -- no derivation needed
                   (unlike Cleveland's num -> target step)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.inspect_dataset import find_project_root
from src.preprocessing.pipeline import FeatureGroups

RAW_TARGET_COLUMN = "cardio"
#: Renamed to match src.preprocessing.pipeline.split_dataset's expected
#: column name (BINARY_TARGET_COLUMN = "target"), so that generic Phase 2
#: utility is reusable unchanged. This is a NAME alignment only -- cardio
#: and target hold identical values, no transformation occurs.
TARGET_COLUMN = "target"

DEFAULT_RELATIVE_PATH = Path("data") / "external" / "cardio_train.csv"

#: Columns dropped before any feature-group assignment. `id` is a row
#: identifier with no predictive meaning (and using it would just memorize
#: row order); dropping it is not a "cleaning" decision, it is excluding a
#: non-feature.
NON_FEATURE_COLUMNS = ["id"]


class SchemaError(ValueError):
    """Raised when the loaded file does not match the expected cardio_train.csv schema."""


def resolve_cardio_path(path: str | Path | None = None) -> Path:
    if path is not None:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (find_project_root() / candidate).resolve()
    return (find_project_root() / DEFAULT_RELATIVE_PATH).resolve()


def load_raw_cardio(path: str | Path | None = None) -> pd.DataFrame:
    """Load cardio_train.csv exactly as-is (semicolon-delimited), no cleaning.

    Raises:
        SchemaError: if the file does not have the expected 13 columns.
    """
    resolved = resolve_cardio_path(path)
    if not resolved.is_file():
        raise SchemaError(f"Dataset file not found: '{resolved}'.")

    df = pd.read_csv(resolved, sep=";")
    expected_cols = {
        "id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active", "cardio",
    }
    if set(df.columns) != expected_cols:
        raise SchemaError(
            f"'{resolved}' does not match the expected cardio_train.csv schema.\n"
            f"  expected columns: {sorted(expected_cols)}\n"
            f"  found columns:    {sorted(df.columns)}"
        )
    return df


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add documented, non-destructive derived columns. Returns a NEW DataFrame.

    - age_years: age (days) / 365.25 -- purely a unit conversion for
      interpretability; the original `age` (days) column is preserved.
    - gender_male: 1 if gender==2 else 0, per the documented (assumed, not
      independently verified) convention 1=women, 2=men. The original
      `gender` column is preserved so this assumption is auditable.
    - target: a renamed COPY of `cardio` (see TARGET_COLUMN docstring above).
      `cardio` itself is preserved unchanged.
    """
    out = df.copy(deep=True)
    out["age_years"] = out["age"] / 365.25
    out["gender_male"] = (out["gender"] == 2).astype(int)
    out[TARGET_COLUMN] = out[RAW_TARGET_COLUMN].astype(int)
    return out


def get_cardio_feature_groups() -> FeatureGroups:
    """The semantic feature grouping for cardio_train.csv, in the same
    FeatureGroups shape Phase 2 uses for Cleveland -- but built independently
    for this dataset's own columns. Reuses the generic FeatureGroups
    dataclass and downstream SharedFeaturePipeline machinery unchanged.

    Grouping rationale:
    - continuous: age_years, height, weight, ap_hi, ap_lo -- genuinely
      continuous physiological measurements, median-imputed (robust to the
      implausible-value tail; see cleaning.py) then scaled.
    - ordinal_categorical: cholesterol, gluc -- ordered 3-level clinical
      categories; kept as a single ordered numeric column (not one-hot),
      consistent with how Phase 2 treated Cleveland's `slope`.
    - binary_categorical: gender_male, smoke, alco, active -- already
      0/1-coded (gender_male derived above); imputed only, never scaled.
    - discrete_numeric / nominal_categorical: none -- this dataset has no
      analog to Cleveland's vessel-count (`ca`) or unordered categories
      (`cp`, `restecg`, `thal`); both groups are legitimately empty.
    """
    return FeatureGroups(
        continuous=["age_years", "height", "weight", "ap_hi", "ap_lo"],
        discrete_numeric=[],
        ordinal_categorical=["cholesterol", "gluc"],
        binary_categorical=["gender_male", "smoke", "alco", "active"],
        nominal_categorical=[],
    )
