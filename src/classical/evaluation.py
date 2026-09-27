"""Metrics, confidence intervals, comparison tables, and plots for Phase 3.

This module computes things; it does not fit anything and it never
receives an unfitted model. Two distinct kinds of "confidence" are
produced, and they are kept clearly separate throughout (see
docs/CLASSICAL_BASELINE.md):

    1. CV variability  -- mean/std/CI across the k cross-validation folds
       computed DURING model selection, entirely on the training split.
    2. Final test CI    -- a bootstrap confidence interval computed from the
       single, locked test-set evaluation, done exactly once per model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: never opens a window, always just writes files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from src.classical.models import ModelSpec
from src.classical.tuning import CV_SCORING, FeatureSpace, build_model_pipeline, specificity_score
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.pipeline import FeatureGroups

# --------------------------------------------------------------------------
# Point-in-time metrics (used for the final, single test-set evaluation)
# --------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Compute the full Phase 3 metric suite at threshold=0.5 for label
    metrics, plus threshold-independent ROC-AUC / PR-AUC.

    Sensitivity and Specificity are reported under BOTH their clinical name
    and their ML-standard synonym, because this is a medical early-risk
    detection problem where the distinction matters (see
    docs/CLASSICAL_BASELINE.md):
        Sensitivity (= Recall): correctly identifies patients WITH disease/risk.
        Specificity: correctly identifies patients WITHOUT disease/risk.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall of the positive class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # recall of the negative class
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "recall": sensitivity,  # explicit synonym, see docstring
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# --------------------------------------------------------------------------
# CV variability (mean, std, t-based CI) -- from GridSearchCV.cv_results_
# --------------------------------------------------------------------------

#: Metrics reported from the CV stage (must match the keys used in
#: src.classical.tuning.CV_SCORING).
CV_METRIC_KEYS = ["roc_auc", "average_precision", "accuracy", "recall", "precision", "specificity", "f1"]


def cv_fold_scores(search: GridSearchCV, metric: str) -> np.ndarray:
    """Extract the per-fold validation scores for `metric` at the best
    hyperparameter combination found by `search`.
    """
    best_index = search.best_index_
    n_splits = search.n_splits_
    return np.array(
        [search.cv_results_[f"split{i}_test_{metric}"][best_index] for i in range(n_splits)]
    )


def cv_confidence_interval(fold_scores: np.ndarray, confidence: float = 0.95) -> dict[str, float]:
    """Mean/std/CI across CV folds using a t-distribution (appropriate for
    the small number of fold scores, e.g. k=5, rather than assuming
    normality with a z-interval or bootstrapping too few numbers).
    """
    mean = float(np.mean(fold_scores))
    std = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
    n = len(fold_scores)
    if n > 1 and std > 0:
        t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin = t_crit * std / np.sqrt(n)
    else:
        margin = 0.0
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
        "n_folds": n,
    }


def extract_cv_summary(search: GridSearchCV) -> dict[str, dict[str, float]]:
    """Return {metric_name: {mean, std, ci_low, ci_high, n_folds}} for
    every metric in CV_METRIC_KEYS, at the best hyperparameter combination.
    """
    summary: dict[str, dict[str, float]] = {}
    for metric in CV_METRIC_KEYS:
        fold_scores = cv_fold_scores(search, metric)
        summary[metric] = cv_confidence_interval(fold_scores)
    return summary


# --------------------------------------------------------------------------
# Final, single-evaluation test-set bootstrap CI
# --------------------------------------------------------------------------


def bootstrap_test_confidence_interval(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_seed: int,
) -> dict[str, float]:
    """Percentile bootstrap CI for one metric on the LOCKED test set.

    The test set is only ever read here (never fit), and this function is
    called exactly once per model per experiment, after model selection is
    fully complete -- see train_baselines.py.
    """
    rng = np.random.RandomState(random_seed)
    n = len(y_true)
    y_pred = (y_proba >= 0.5).astype(int)

    def _score(idx: np.ndarray) -> float:
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            return np.nan  # a resample with only one class can't score AUC-type metrics
        if metric == "roc_auc":
            return roc_auc_score(yt, ypr)
        if metric == "pr_auc":
            return average_precision_score(yt, ypr)
        if metric == "sensitivity":
            return compute_classification_metrics(yt, yp, ypr)["sensitivity"]
        if metric == "specificity":
            return compute_classification_metrics(yt, yp, ypr)["specificity"]
        if metric == "accuracy":
            return compute_classification_metrics(yt, yp, ypr)["accuracy"]
        raise ValueError(f"Unsupported bootstrap metric: {metric!r}")

    point_estimate = _score(np.arange(n))
    resampled = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        val = _score(idx)
        if not np.isnan(val):
            resampled.append(val)
    resampled_arr = np.array(resampled)
    alpha = (1 - confidence) / 2
    ci_low = float(np.quantile(resampled_arr, alpha))
    ci_high = float(np.quantile(resampled_arr, 1 - alpha))
    return {"point_estimate": float(point_estimate), "ci_low": ci_low, "ci_high": ci_high, "n_resamples": len(resampled_arr)}


# --------------------------------------------------------------------------
# Per-model result container + comparison table
# --------------------------------------------------------------------------


@dataclass
class ModelResult:
    """Everything about one (experiment, model) pair, ready for reporting."""

    experiment: str
    model_key: str
    display_name: str
    feature_space: str
    best_params: dict
    cv_summary: dict[str, dict[str, float]]
    test_metrics: dict[str, float]
    test_bootstrap: dict[str, dict[str, float]]
    fit_time_seconds: float
    inference_time_ms_per_record: float
    n_train: int
    n_test: int
    y_test_true: np.ndarray = field(repr=False)
    y_test_proba: np.ndarray = field(repr=False)
    explainability: dict = field(default_factory=dict, repr=False)


def evaluate_on_test(
    search: GridSearchCV,
    experiment: str,
    model_key: str,
    display_name: str,
    feature_space: str,
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    *,
    n_train: int,
    random_seed: int,
) -> ModelResult:
    """The ONLY function in Phase 3 that touches the locked test set.

    Called exactly once per (experiment, model) pair, after
    run_grid_search() has already completed model selection using the
    training split alone. `search.best_estimator_` was refit on the full
    training split by GridSearchCV itself (see tuning.run_grid_search
    docstring); this function only calls .predict_proba() on it, never
    .fit().
    """
    t0 = time.perf_counter()
    y_proba = search.best_estimator_.predict_proba(X_test_raw)[:, 1]
    inference_seconds = time.perf_counter() - t0
    y_pred = (y_proba >= 0.5).astype(int)

    test_metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    test_bootstrap = {
        m: bootstrap_test_confidence_interval(y_test, y_proba, m, random_seed=random_seed)
        for m in ("roc_auc", "pr_auc", "sensitivity", "specificity", "accuracy")
    }

    # mean fold fit time from CV as the "training time" figure -- reflects
    # what fitting this model (with preprocessing) actually costs, without
    # conflating it with the (much larger) cost of the grid search itself.
    fit_time = float(search.cv_results_["mean_fit_time"][search.best_index_])

    return ModelResult(
        experiment=experiment,
        model_key=model_key,
        display_name=display_name,
        feature_space=feature_space,
        best_params={k.replace("model__", ""): v for k, v in search.best_params_.items()},
        cv_summary=extract_cv_summary(search),
        test_metrics=test_metrics,
        test_bootstrap=test_bootstrap,
        fit_time_seconds=fit_time,
        inference_time_ms_per_record=(inference_seconds / len(X_test_raw)) * 1000,
        n_train=n_train,
        n_test=len(X_test_raw),
        y_test_true=np.asarray(y_test),
        y_test_proba=y_proba,
    )


def build_comparison_table(results: list[ModelResult]) -> pd.DataFrame:
    """The unified comparison table: Model, Accuracy, Sensitivity,
    Specificity, Precision, F1, ROC-AUC, PR-AUC, Training time, Inference
    time -- one row per model, for the FINAL LOCKED TEST SET.
    """
    rows = []
    for r in results:
        tm = r.test_metrics
        rows.append(
            {
                "Model": r.display_name,
                "Feature space": r.feature_space,
                "Accuracy": round(tm["accuracy"], 4),
                "Sensitivity": round(tm["sensitivity"], 4),
                "Specificity": round(tm["specificity"], 4),
                "Precision": round(tm["precision"], 4),
                "F1": round(tm["f1"], 4),
                "ROC-AUC": round(tm["roc_auc"], 4),
                "PR-AUC": round(tm["pr_auc"], 4),
                "Training time (s)": round(r.fit_time_seconds, 4),
                "Inference time (ms/record)": round(r.inference_time_ms_per_record, 4),
            }
        )
    return pd.DataFrame(rows)


def build_cv_summary_table(results: list[ModelResult]) -> pd.DataFrame:
    """CV mean +/- std table (training-split-only), the number that should
    be trusted more than any single test-set score on a dataset this small.
    """
    rows = []
    for r in results:
        row = {"Model": r.display_name, "Feature space": r.feature_space}
        for metric in ("roc_auc", "average_precision", "accuracy", "recall", "specificity", "precision", "f1"):
            s = r.cv_summary[metric]
            row[f"{metric}_mean"] = round(s["mean"], 4)
            row[f"{metric}_std"] = round(s["std"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_roc_curves(results: list[ModelResult], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(r.y_test_true, r.y_test_proba)
        auc = r.test_metrics["roc_auc"]
        ax.plot(fpr, tpr, label=f"{r.display_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_curves(results: list[ModelResult], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for r in results:
        precision, recall, _ = precision_recall_curve(r.y_test_true, r.y_test_proba)
        ap = r.test_metrics["pr_auc"]
        ax.plot(recall, precision, label=f"{r.display_name} (AP={ap:.3f})")
    positive_rate = float(np.mean(results[0].y_test_true)) if results else 0.0
    ax.axhline(positive_rate, linestyle="--", color="grey", label=f"Chance (positive rate={positive_rate:.2f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results: list[ModelResult], out_path: Path, title: str) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = np.array([[r.test_metrics["tn"], r.test_metrics["fp"]], [r.test_metrics["fn"], r.test_metrics["tp"]]])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No disease", "At risk"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(r.display_name, fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_comparison(results: list[ModelResult], out_path: Path, title: str) -> None:
    metrics = ["accuracy", "sensitivity", "specificity", "roc_auc", "pr_auc"]
    labels = ["Accuracy", "Sensitivity", "Specificity", "ROC-AUC", "PR-AUC"]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(results), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(results):
        values = [r.test_metrics[m] for m in metrics]
        ax.bar(x + i * width, values, width, label=r.display_name)
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (locked test set)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calibration(results: list[ModelResult], out_path: Path, title: str, n_bins: int = 5) -> None:
    """Optional calibration plot: mean predicted probability vs observed
    frequency, binned by predicted probability. With only 61 test records,
    bins are necessarily coarse -- this is exploratory, not a precise
    calibration assessment.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for r in results:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(r.y_test_proba, bins) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
        mean_pred, obs_freq = [], []
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() == 0:
                continue
            mean_pred.append(r.y_test_proba[mask].mean())
            obs_freq.append(r.y_test_true[mask].mean())
        ax.plot(mean_pred, obs_freq, marker="o", label=r.display_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title + "\n(n=61 test records -- exploratory only, not a precise assessment)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Class-imbalance side-check
# --------------------------------------------------------------------------


def compare_class_weighting(
    model_spec: ModelSpec,
    feature_space: FeatureSpace,
    feature_groups: FeatureGroups,
    config: PreprocessingConfig,
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    best_params: dict,
    *,
    cv_folds: int,
    random_seed: int,
) -> dict[str, dict[str, dict[str, float]]] | None:
    """Cross-validated comparison of unweighted vs class-weighted training,
    at the ALREADY-TUNED hyperparameters, on the training split only.

    The dataset is near-balanced (54.1% / 45.9%), so per the phase
    instructions this is evaluated explicitly rather than assumed
    unnecessary -- if it is not applicable to a model (no
    class_weight/scale_pos_weight support), this returns None.
    """
    if model_spec.class_weight_kind is None:
        return None

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    fixed_params = {f"model__{k}": v for k, v in best_params.items()}

    def _cv_summary(pipe) -> dict[str, dict[str, float]]:
        cv_out = cross_validate(pipe, X_train_raw, y_train, cv=cv, scoring=CV_SCORING, n_jobs=1)
        return {
            metric: {
                "mean": float(np.mean(cv_out[f"test_{metric}"])),
                "std": float(np.std(cv_out[f"test_{metric}"], ddof=1)),
            }
            for metric in CV_METRIC_KEYS
        }

    pipe_unweighted = build_model_pipeline(model_spec, feature_space, feature_groups, config, random_seed)
    pipe_unweighted.set_params(**fixed_params)
    unweighted_summary = _cv_summary(pipe_unweighted)

    pipe_weighted = build_model_pipeline(model_spec, feature_space, feature_groups, config, random_seed)
    weighted_params = dict(fixed_params)
    if model_spec.class_weight_kind == "class_weight":
        weighted_params["model__class_weight"] = "balanced"
    elif model_spec.class_weight_kind == "scale_pos_weight":
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        weighted_params["model__scale_pos_weight"] = n_neg / n_pos if n_pos > 0 else 1.0
    pipe_weighted.set_params(**weighted_params)
    weighted_summary = _cv_summary(pipe_weighted)

    return {"unweighted": unweighted_summary, "weighted": weighted_summary}
