"""Central configuration for the Phase 2 preprocessing and feature pipeline.

Every value that could otherwise be hardcoded inside pipeline.py lives here:
the random seed, the train/test split ratio, imputation strategies, the
scaler choice, feature-selection method and count, PCA component count, the
quantum encoding range, and which feature set (full vs screening) to use.

Nothing in this module fits, transforms, or looks at any data. It only
describes *how* the pipeline in pipeline.py should be built.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import yaml

FeatureSet = Literal["full", "screening"]
ScalerName = Literal["standard", "minmax", "robust"]
SelectionMethod = Literal["mutual_info", "f_classif"]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Immutable configuration for one run of the preprocessing pipeline.

    Frozen so that a config object, once built, cannot be silently mutated
    partway through a pipeline run -- if you need a variant, use
    ``replace()`` (re-exported below) to get a new object.
    """

    # --- reproducibility ---------------------------------------------------
    random_seed: int = 42

    # --- train/test split ---------------------------------------------------
    test_size: float = 0.20
    stratify: bool = True

    # --- missing-value imputation --------------------------------------------
    # median is robust to outliers and appropriate for skewed continuous
    # clinical measurements (e.g. cholesterol); most_frequent is the
    # standard default for categorical/ordinal columns with very few
    # missing values (see docs/PREPROCESSING.md for the full justification).
    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"

    # --- scaling --------------------------------------------------------------
    scaler: ScalerName = "standard"

    # --- feature selection ------------------------------------------------
    # mutual_info_classif captures non-linear dependence between a feature
    # and the binary target, which suits a small clinical dataset where
    # relationships (e.g. oldpeak vs risk) are not guaranteed to be linear.
    feature_selection_method: SelectionMethod = "mutual_info"
    feature_selection_k: int = 10

    # --- dimensionality reduction (quantum-ready branch) ---------------------
    # NOT assumed optimal -- deliberately configurable. See
    # docs/PREPROCESSING.md Section 8 for the reasoning and the explained-
    # variance trade-off across n_components in {2, ..., 6}.
    pca_n_components: int = 4
    quantum_range_low: float = 0.0
    quantum_range_high: float = math.pi

    # --- clinical feature ablation (prepared, not executed, in Phase 2) ------
    # "full"      -> all 13 predictors, including ca and thal
    # "screening" -> 11 predictors, excluding ca and thal (see
    #                docs/PREPROCESSING.md Section on ca/thal)
    feature_set: FeatureSet = "full"

    def __post_init__(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1); got {self.test_size}.")
        if self.feature_selection_k < 1:
            raise ValueError(f"feature_selection_k must be >= 1; got {self.feature_selection_k}.")
        if self.pca_n_components < 1:
            raise ValueError(f"pca_n_components must be >= 1; got {self.pca_n_components}.")
        if self.quantum_range_low >= self.quantum_range_high:
            raise ValueError(
                f"quantum_range_low ({self.quantum_range_low}) must be < "
                f"quantum_range_high ({self.quantum_range_high})."
            )
        if self.feature_set not in ("full", "screening"):
            raise ValueError(f"feature_set must be 'full' or 'screening'; got {self.feature_set!r}.")

    @property
    def quantum_range(self) -> tuple[float, float]:
        return (self.quantum_range_low, self.quantum_range_high)

    def with_overrides(self, **kwargs: object) -> "PreprocessingConfig":
        """Return a new config with the given fields replaced.

        Example:
            screening_cfg = config.with_overrides(feature_set="screening")
        """
        return replace(self, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def default(cls) -> "PreprocessingConfig":
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PreprocessingConfig":
        """Load a config from a YAML file, falling back to defaults for any
        field not present in the file.
        """
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(**raw)

    def to_dict(self) -> dict:
        return {
            "random_seed": self.random_seed,
            "test_size": self.test_size,
            "stratify": self.stratify,
            "numeric_impute_strategy": self.numeric_impute_strategy,
            "categorical_impute_strategy": self.categorical_impute_strategy,
            "scaler": self.scaler,
            "feature_selection_method": self.feature_selection_method,
            "feature_selection_k": self.feature_selection_k,
            "pca_n_components": self.pca_n_components,
            "quantum_range_low": self.quantum_range_low,
            "quantum_range_high": self.quantum_range_high,
            "feature_set": self.feature_set,
        }


#: A ready-to-use default configuration. Import this directly for the common
#: case; use PreprocessingConfig(...) or DEFAULT_CONFIG.with_overrides(...)
#: to customize.
DEFAULT_CONFIG = PreprocessingConfig.default()
