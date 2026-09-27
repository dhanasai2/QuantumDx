"""Leakage-safe hyperparameter tuning for Phase 3.

The core idea: wrap the EXISTING Phase 2 preprocessing components
(SharedFeaturePipeline, build_quantum_pipeline) in thin sklearn-compatible
adapters, place them as the first step(s) of an sklearn Pipeline together
with a classifier, and hand the whole thing to GridSearchCV.

GridSearchCV's own, well-tested cross-validation machinery then guarantees
exactly the leakage-safe fold discipline this phase requires: for every
fold and every hyperparameter combination, it clones the pipeline fresh and
calls .fit() on the fold's TRAINING rows only -- which re-fits the
preprocessing (imputer, scaler, encoder, selector, and PCA where used)
strictly within that fold -- then scores on the fold's VALIDATION rows via
.transform() only. No preprocessing statistic is ever computed from a
validation or test row.

No preprocessing LOGIC is reimplemented here. SharedFeatureTransformer and
QuantumReadyTransformer delegate entirely to
src.preprocessing.pipeline.SharedFeaturePipeline and
src.preprocessing.pipeline.build_quantum_pipeline respectively -- they
exist only to satisfy sklearn's clone()/get_params() contract, which the
Phase 2 classes do not implement (Phase 2 was not written to be dropped
into GridSearchCV, and Phase 2 code is not modified here to add that).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.classical.models import ModelSpec
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.pipeline import FeatureGroups, SharedFeaturePipeline, build_quantum_pipeline

FeatureSpace = Literal["classical", "quantum_ready"]


# --------------------------------------------------------------------------
# sklearn-compatible adapters around Phase 2 (delegation only, no new logic)
# --------------------------------------------------------------------------


class SharedFeatureTransformer(BaseEstimator, TransformerMixin):
    """sklearn adapter around src.preprocessing.pipeline.SharedFeaturePipeline.

    Delegates fit/transform entirely to the Phase 2 implementation. Exists
    only so the Phase 2 impute+encode+scale+select stage can live inside a
    GridSearchCV pipeline, which requires get_params()/set_params()
    (sklearn's clone() contract) and refits automatically on every fold.
    """

    def __init__(self, feature_groups: FeatureGroups, config: PreprocessingConfig):
        self.feature_groups = feature_groups
        self.config = config

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "SharedFeatureTransformer":
        self._impl = SharedFeaturePipeline(self.feature_groups, self.config)
        self._impl.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._impl.transform(X)

    def get_feature_names_out(self, input_features=None) -> list[str]:  # noqa: ANN001
        return self._impl.selected_feature_names


class QuantumReadyTransformer(BaseEstimator, TransformerMixin):
    """sklearn adapter around src.preprocessing.pipeline.build_quantum_pipeline.

    Delegates fit/transform entirely to the Phase 2 implementation (PCA +
    range-normalize to [0, pi]). Continues directly from whatever
    SharedFeatureTransformer produced -- no re-imputation, re-encoding, or
    re-scaling happens here, matching the "no duplicated preprocessing
    logic" design established in Phase 2.
    """

    def __init__(self, n_components: int, quantum_range: tuple[float, float]):
        self.n_components = n_components
        self.quantum_range = quantum_range

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "QuantumReadyTransformer":
        self._impl = build_quantum_pipeline(self.n_components, self.quantum_range)
        self._impl.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self._impl.transform(X)


def build_model_pipeline(
    model_spec: ModelSpec,
    feature_space: FeatureSpace,
    feature_groups: FeatureGroups,
    config: PreprocessingConfig,
    random_seed: int,
) -> Pipeline:
    """Build the UNFITTED sklearn Pipeline for one model on one feature space.

    feature_space="classical":     shared (impute+encode+scale+select) -> model
    feature_space="quantum_ready": shared -> PCA+range-normalize         -> model

    This is the object handed to GridSearchCV; it is never fit here.
    """
    steps: list[tuple[str, BaseEstimator]] = [
        ("shared", SharedFeatureTransformer(feature_groups, config))
    ]
    if feature_space == "quantum_ready":
        steps.append(("quantum", QuantumReadyTransformer(config.pca_n_components, config.quantum_range)))
    elif feature_space != "classical":
        raise ValueError(f"Unknown feature_space: {feature_space!r} (expected 'classical' or 'quantum_ready').")

    steps.append(("model", model_spec.build(random_seed)))
    return Pipeline(steps)


# --------------------------------------------------------------------------
# Multi-metric CV scoring (threshold=0.5 for label-based metrics; ROC-AUC /
# PR-AUC are threshold-independent, computed from predicted probabilities)
# --------------------------------------------------------------------------


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Specificity = recall of the NEGATIVE class = TN / (TN + FP)."""
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


#: The full metric suite computed on every CV fold. Label-based metrics
#: (accuracy, recall, precision, specificity, f1) implicitly use the
#: default 0.5 probability threshold via each estimator's .predict();
#: roc_auc and average_precision use predicted probabilities directly and
#: are threshold-independent.
#:
#: precision/f1 use explicit zero_division=0 (rather than sklearn's default
#: 'warn') because a genuine hyperparameter sweep is EXPECTED to include
#: some combinations (e.g. very strong regularization) that predict only
#: the majority class on a given fold, making precision undefined for the
#: positive class. 0.0 is the correct score to record for that
#: combination; this only silences the per-fold warning spam, it does not
#: change which value is reported.
CV_SCORING: dict[str, object] = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",  # PR-AUC
    "accuracy": "accuracy",
    "recall": "recall",  # sensitivity
    "precision": make_scorer(precision_score, zero_division=0),
    "specificity": make_scorer(specificity_score),
    "f1": make_scorer(f1_score, zero_division=0),
}

#: The model-selection criterion used to pick the best hyperparameters
#: (via GridSearchCV's `refit=`) and, later, the best model across the
#: comparison table. Chosen deliberately over accuracy: ROC-AUC is
#: threshold-independent and reflects ranking quality across the full
#: operating range, appropriate for a benchmark that does not yet commit to
#: one clinical decision threshold (see docs/CLASSICAL_BASELINE.md).
MODEL_SELECTION_METRIC = "roc_auc"


def run_grid_search(
    model_spec: ModelSpec,
    feature_space: FeatureSpace,
    feature_groups: FeatureGroups,
    config: PreprocessingConfig,
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    *,
    cv_folds: int = 5,
    cv_shuffle: bool = True,
    random_seed: int,
) -> GridSearchCV:
    """Run leakage-safe hyperparameter search for one model on the TRAINING split only.

    IMPORTANT (structural leakage guard): this function's signature has no
    test-set parameter at all -- there is no X_test/y_test argument for it
    to accidentally use. It can only ever see X_train_raw/y_train, which is
    itself the TRAINING portion of the top-level split produced in Phase 2
    (see src.preprocessing.pipeline.run_preprocessing_pipeline /
    split_dataset). Verified by
    tests/test_classical.py::test_run_grid_search_has_no_test_set_parameter.

    Internally, StratifiedKFold(cv_folds) further splits X_train_raw into
    fold-train/fold-validation partitions; GridSearchCV clones and refits
    the full pipeline (preprocessing included) on each fold-train partition
    and scores on that fold's held-out validation partition. After the
    search completes, GridSearchCV refits the pipeline with the best
    hyperparameters on the ENTIRE X_train_raw/y_train (still no test data)
    -- this is exactly the "freeze hyperparameters, refit on the full
    training portion" step required before final test evaluation, and it
    is sklearn's own behavior (refit=MODEL_SELECTION_METRIC), not something
    reimplemented here.

    Returns:
        A FITTED GridSearchCV. `.best_estimator_` is the refit-on-full-train
        pipeline ready for a single, final evaluation on the locked test set.
    """
    pipeline = build_model_pipeline(model_spec, feature_space, feature_groups, config, random_seed)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=cv_shuffle, random_state=random_seed if cv_shuffle else None)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=model_spec.param_grid,
        scoring=CV_SCORING,
        refit=MODEL_SELECTION_METRIC,
        cv=cv,
        n_jobs=1,  # deterministic, and the dataset is small enough that this is fast regardless
        return_train_score=False,
    )
    search.fit(X_train_raw, y_train)
    return search
