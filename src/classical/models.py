"""Classical model registry for Phase 3.

Defines the four required classical baselines -- Logistic Regression,
RBF-kernel SVM, Random Forest, XGBoost -- as a small, uniform registry of
(unfitted estimator builder, hyperparameter grid, rationale). Nothing here
fits a model or touches data; this module only describes what models exist
and what is reasonable to search over.

If XGBoost cannot be imported in the current environment, that fact is
recorded explicitly (XGBOOST_AVAILABLE / XGBOOST_IMPORT_ERROR) and XGBoost
is excluded from the registry with a clear, logged reason -- it is never
silently swapped for another model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
    XGBOOST_IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - exercised only when xgboost is absent
    XGBClassifier = None  # type: ignore[assignment,misc]
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = str(exc)


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to build, tune, and later re-weight one classical model."""

    key: str
    display_name: str
    build: Callable[[int], BaseEstimator]  # (random_seed) -> unfitted estimator
    param_grid: dict[str, list]
    param_grid_rationale: str
    #: How this model supports class re-weighting, for the class-imbalance
    #: side-check in evaluation.py. "class_weight" (sklearn's standard
    #: keyword) or "scale_pos_weight" (XGBoost's convention) or None.
    class_weight_kind: str | None


# --------------------------------------------------------------------------
# 1. Logistic Regression
# --------------------------------------------------------------------------

# C controls inverse regularization strength (smaller C = stronger
# regularization); the grid spans heavily-regularized to near-unregularized
# fits on standardized (mean 0, std 1) features.
#
# NOTE on penalty choice: an l1-vs-l2 comparison was considered, but the
# installed sklearn version (1.8) deprecated the 'penalty' keyword on
# LogisticRegression in favor of a unified 'l1_ratio' API, and combining
# penalty='l1' with solver='liblinear' under that transition emits an
# "Inconsistent values: penalty=l1 with l1_ratio=0.0" warning that suggests
# the l1 request may not be honored as expected. Rather than depend on
# ambiguous cross-version behavior, this implementation deliberately tunes
# only C with the standard L2 penalty via the 'lbfgs' solver (sklearn's
# current default, well-tested, no deprecation edge cases). This is
# documented here, not silently substituted.
_LOGREG_PARAM_GRID = {
    "model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
}
_LOGREG_RATIONALE = (
    "C in {0.001..100} spans strong to weak L2 regularization on "
    "standardized features (StandardScaler output has unit variance, so "
    "this range covers heavily-regularized to near-unregularized fits "
    "without needing a wider search). An l1-vs-l2 penalty comparison was "
    "considered but dropped due to an sklearn 1.8 API transition making "
    "penalty='l1' behavior ambiguous with solver='liblinear' in this "
    "environment (see code comment); L2 via 'lbfgs' is used instead as the "
    "unambiguous, well-supported choice."
)


def _build_logreg(random_seed: int) -> LogisticRegression:
    return LogisticRegression(solver="lbfgs", max_iter=5000, random_state=random_seed)


# --------------------------------------------------------------------------
# 2. RBF-kernel SVM
# --------------------------------------------------------------------------

# C trades off margin width against training error, as for logistic
# regression. gamma controls the RBF kernel's locality; 'scale' (the
# sklearn default, 1/(n_features * X.var())) is included as a
# data-driven baseline alongside a small manual sweep, because the right
# gamma order of magnitude is hard to guess a priori on a standardized,
# low-dimensional (10-17 feature) clinical dataset.
_SVM_PARAM_GRID = {
    "model__C": [0.1, 1.0, 10.0, 100.0],
    "model__gamma": ["scale", 0.01, 0.1, 1.0],
}
_SVM_RATIONALE = (
    "C in {0.1..100} mirrors the logistic regression sweep. gamma includes "
    "sklearn's data-driven 'scale' default plus a manual sweep over three "
    "orders of magnitude, since the dataset is small and standardized "
    "(mean 0, std 1), making gamma values much larger than ~1 unlikely to "
    "be useful and values much smaller likely to under-fit."
)


def _build_svm(random_seed: int) -> SVC:
    return SVC(kernel="rbf", probability=True, random_state=random_seed)


# --------------------------------------------------------------------------
# 3. Random Forest
# --------------------------------------------------------------------------

# A modest grid appropriate for ~242 training rows: too many trees or too
# much depth on this little data mainly costs time, not accuracy, so the
# grid favors a few well-spaced values over an exhaustive sweep.
_RF_PARAM_GRID = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [3, 5, None],
    "model__min_samples_split": [2, 5],
    "model__max_features": ["sqrt", "log2"],
}
_RF_RATIONALE = (
    "n_estimators in {100, 300} -- forest performance typically plateaus "
    "well before 300 trees on a dataset this small, so this brackets "
    "'enough' vs 'more than enough'. max_depth in {3, 5, None} spans "
    "shallow (regularized) to unrestricted trees, appropriate given only "
    "~242 training rows where deep unrestricted trees risk overfitting. "
    "min_samples_split and max_features are standard regularization knobs "
    "included at their most common alternative values."
)


def _build_rf(random_seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=random_seed, n_jobs=1)


# --------------------------------------------------------------------------
# 4. XGBoost
# --------------------------------------------------------------------------

_XGB_PARAM_GRID = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [2, 3, 4],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}
_XGB_RATIONALE = (
    "max_depth is kept shallow (2-4) because deep boosted trees overfit "
    "quickly on ~242 rows. learning_rate in {0.05, 0.1} pairs a slower, "
    "more regularized rate with the common default. subsample and "
    "colsample_bytree in {0.8, 1.0} test whether row/column subsampling "
    "(a regularizer) helps on this small dataset, compared against using "
    "all data each round."
)


def _build_xgb(random_seed: int):  # -> XGBClassifier, typed loosely since import is conditional
    return XGBClassifier(
        random_state=random_seed,
        eval_metric="logloss",
        n_jobs=1,
        tree_method="hist",
    )


def get_available_models(random_seed: int) -> dict[str, ModelSpec]:
    """Return the registry of classical models to train in Phase 3.

    XGBoost is included only if it imported successfully; if not, it is
    omitted here and the reason is exposed via XGBOOST_AVAILABLE /
    XGBOOST_IMPORT_ERROR for the caller to log and document explicitly.
    """
    registry: dict[str, ModelSpec] = {
        "logistic_regression": ModelSpec(
            key="logistic_regression",
            display_name="Logistic Regression",
            build=_build_logreg,
            param_grid=_LOGREG_PARAM_GRID,
            param_grid_rationale=_LOGREG_RATIONALE,
            class_weight_kind="class_weight",
        ),
        "rbf_svm": ModelSpec(
            key="rbf_svm",
            display_name="RBF-Kernel SVM",
            build=_build_svm,
            param_grid=_SVM_PARAM_GRID,
            param_grid_rationale=_SVM_RATIONALE,
            class_weight_kind="class_weight",
        ),
        "random_forest": ModelSpec(
            key="random_forest",
            display_name="Random Forest",
            build=_build_rf,
            param_grid=_RF_PARAM_GRID,
            param_grid_rationale=_RF_RATIONALE,
            class_weight_kind="class_weight",
        ),
    }

    if XGBOOST_AVAILABLE:
        registry["xgboost"] = ModelSpec(
            key="xgboost",
            display_name="XGBoost",
            build=_build_xgb,
            param_grid=_XGB_PARAM_GRID,
            param_grid_rationale=_XGB_RATIONALE,
            class_weight_kind="scale_pos_weight",
        )

    return registry
