"""QSVM: a classical SVC(kernel='precomputed') fit on the Phase 4 fidelity kernel (Phase 5).

This module introduces NO new preprocessing, scoring, or metric logic. It
reuses, directly and unmodified:

    - src.classical.tuning.CV_SCORING / MODEL_SELECTION_METRIC
      (Phase 3's multi-metric scoring dict and model-selection criterion --
      these are generic sklearn scorers, not Pipeline-specific, so they
      apply to a bare SVC(kernel='precomputed') exactly as they did to
      Phase 3's classical Pipelines)
    - src.classical.evaluation.evaluate_on_test (Phase 3's single-touch
      final-test-evaluation function -- generic over any fitted
      GridSearchCV, regardless of what estimator it wraps)

The only genuinely new logic here is: (a) what parameter grid to search
(C only -- the kernel itself has no hyperparameters, since the feature map
is fixed), and (b) that the "X" GridSearchCV sees is a precomputed Gram
matrix rather than a feature matrix.

WHY SLICING THE PRECOMPUTED KERNEL PER FOLD IS LEAKAGE-SAFE (not just
convenient): K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2 is a FIXED function of
two feature vectors -- the feature map has zero trainable/fitted
parameters (see docs/QUANTUM_PIPELINE.md Section 5.1). Unlike Phase 3's
classical preprocessing (imputer medians, scaler means, PCA axes), which
generically DO depend on which rows are present and therefore must be
refit per fold, a kernel VALUE K(x_i, x_j) does not change depending on
which other rows happen to be in the matrix. Slicing K_train_train[fold_idx]
[:, fold_idx] is therefore mathematically IDENTICAL to recomputing the
kernel from scratch using only that fold's rows -- not an approximation.
This was verified empirically against manual fold-by-fold recomputation
during development (see docs/QUANTUM_QSVM.md Section 4 for the numbers).
scikit-learn's GridSearchCV/StratifiedKFold perform exactly this slicing
natively for any `kernel='precomputed'` estimator.
"""

from __future__ import annotations

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from src.classical.tuning import CV_SCORING, MODEL_SELECTION_METRIC

#: A small, defensible grid -- explicitly suggested by the Phase 5 brief and
#: a superset of Phase 3's RBF-SVM C-grid ({0.1, 1, 10, 100}), widened by
#: one order of magnitude on the small side since a precomputed fidelity
#: kernel's value scale differs from an RBF kernel's and an untested
#: assumption that the same range is optimal is worth checking cheaply.
DEFAULT_C_GRID: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0]


def build_qsvm(random_seed: int, C: float | None = None) -> SVC:
    """Build an (optionally C-fixed) precomputed-kernel SVC.

    probability=True mirrors Phase 3's RBF-SVM configuration exactly, so
    predict_proba is available for ROC-AUC/PR-AUC on decision scores
    rather than hard labels (Phase 5 Section G).
    """
    kwargs = {"kernel": "precomputed", "probability": True, "random_state": random_seed}
    if C is not None:
        kwargs["C"] = C
    return SVC(**kwargs)


def run_qsvm_grid_search(
    K_train_train,
    y_train,
    *,
    C_grid: list[float] | None = None,
    cv_folds: int = 5,
    cv_shuffle: bool = True,
    random_seed: int,
) -> GridSearchCV:
    """Leakage-safe C-tuning for the QSVM, using ONLY the training kernel.

    IMPORTANT (structural leakage guard, mirroring
    src.classical.tuning.run_grid_search): this function's signature has
    no test-kernel parameter at all -- there is no K_test_train/y_test
    argument for it to accidentally use. It can only ever see
    K_train_train/y_train.

    K_train_train must be the FULL (n_train, n_train) Gram matrix.
    GridSearchCV's native precomputed-kernel handling slices
    K[fold_train_idx][:, fold_train_idx] for fitting and
    K[fold_val_idx][:, fold_train_idx] for scoring, per fold -- verified
    to exactly match manual fold-by-fold recomputation (see module
    docstring). After the search completes, GridSearchCV refits the
    winning C on the ENTIRE K_train_train/y_train (still no test data) --
    the same "freeze hyperparameters, refit on full training data" step
    Phase 3 relies on, and it is sklearn's own behavior here too, not
    reimplemented.

    Returns:
        A FITTED GridSearchCV. `.best_estimator_` is ready for a single,
        final evaluation via K_test_train.
    """
    svc = build_qsvm(random_seed)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=cv_shuffle, random_state=random_seed if cv_shuffle else None)

    search = GridSearchCV(
        estimator=svc,
        param_grid={"C": C_grid or DEFAULT_C_GRID},
        scoring=CV_SCORING,
        refit=MODEL_SELECTION_METRIC,
        cv=cv,
        n_jobs=1,
        return_train_score=False,
    )
    search.fit(K_train_train, y_train)
    return search
