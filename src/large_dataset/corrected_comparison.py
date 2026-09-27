"""Methodology correction: evaluate classical models and QSVM on the IDENTICAL
200-row held-out test set (post-Stage-C correction phase).

THE PROBLEM THIS FIXES
----------------------
Stages A/B/C evaluated classical models on the fixed 2,000-row test set but
evaluated QSVM on the fixed 200-row subset. The reported "classical-vs-QSVM
gap" therefore compared metrics computed on different populations. The
200-row set is a strict subset of the 2,000-row set, but that does not make
the metrics interchangeable -- a ROC-AUC over 200 observations is not the
same quantity as one over 2,000.

WHAT THIS MODULE CHANGES
------------------------
The EVALUATION POPULATION only. Nothing else:
    - training subsets: unchanged (nested, seed 42)
    - test-set membership: unchanged (the existing 200 rows, recovered
      deterministically -- NOT re-sampled)
    - preprocessing / feature groups / PCA / scaling: unchanged
    - quantum feature map, qubits, reps, entanglement, kernel: unchanged
    - classical hyperparameter grids and CV/model-selection procedure:
      unchanged (run_grid_search is reused verbatim)
    - QSVM C-grid and selection: unchanged (run_qsvm_grid_search reused)
    - decision threshold: unchanged (fixed 0.5, established in Phase 3)

MODEL PROVENANCE
----------------
No Stage A/B/C model artifacts or predictions were persisted (verified by
inspection). This module therefore takes the "reproduce" path: it re-runs
the SAME training + CV model-selection procedure on the SAME stage training
subset with the SAME seed, freezes the resulting estimator, and only then
evaluates it. The 200-row set is never used for fitting, hyperparameter
selection, or threshold selection -- it is touched exactly once per model,
for prediction.

Predictions are keyed by the dataset's own `id` column so that paired
statistical comparisons are aligned observation-by-observation rather than
by positional assumption.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.classical.evaluation import compute_classification_metrics
from src.classical.models import get_available_models
from src.classical.tuning import run_grid_search
from src.data.inspect_dataset import find_project_root
from src.large_dataset.cleaning import compute_cleaning_flags, split_modeling_subset
from src.large_dataset.pipeline import process_stage
from src.large_dataset.sampling import build_fixed_split, nested_stratified_stage_samples
from src.large_dataset.schema import (
    TARGET_COLUMN,
    derive_features,
    get_cardio_feature_groups,
    load_raw_cardio,
)
from src.preprocessing.config import PreprocessingConfig
from src.quantum.backends import get_backend
from src.quantum.config import QuantumConfig
from src.quantum.feature_maps import build_feature_map
from src.quantum.kernel import (
    compute_statevectors,
    kernel_diagnostics,
    kernel_matrix_from_statevectors_vectorized,
)
from src.quantum.qsvm import run_qsvm_grid_search

CV_FOLDS = 5
CV_SHUFFLE = True
DECISION_THRESHOLD = 0.5  # Phase 3 methodology; NOT selected from test data
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42

RESULTS_ROOT = find_project_root() / "results" / "large_dataset" / "corrected_comparison"


# --------------------------------------------------------------------------
# Test-set identity
# --------------------------------------------------------------------------


def comparison_set_fingerprint(ids: list[int]) -> str:
    """Stable identity hash of the comparison set, so any future run can
    prove it used the same 200 observations."""
    import hashlib

    return hashlib.sha256(str(sorted(ids)).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Per-model prediction on the identical comparison set
# --------------------------------------------------------------------------


@dataclass
class ModelPredictions:
    """Predictions for one model on the identical comparison set, keyed by id."""

    model_key: str
    display_name: str
    ids: np.ndarray
    y_true: np.ndarray
    y_proba: np.ndarray
    y_pred: np.ndarray
    metrics: dict
    best_params: dict
    provenance: str  # "reproduced_training" | "loaded_artifact"
    train_seconds: float


def _metrics_from_proba(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    return compute_classification_metrics(y_true, y_pred, y_proba)


def evaluate_classical_on_comparison_set(
    stage_train_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    feature_groups,
    preprocessing_config: PreprocessingConfig,
) -> list[ModelPredictions]:
    """Reproduce Stage training for all 4 classical models, freeze, then
    predict on the comparison set.

    run_grid_search is reused verbatim -- same grids, same CV, same seed --
    and receives ONLY the stage training data. The comparison set is passed
    to predict_proba after the search has completed.
    """
    feature_cols = feature_groups.all_columns
    X_train_raw = stage_train_df[feature_cols]
    y_train = stage_train_df[TARGET_COLUMN].to_numpy()

    X_cmp_raw = comparison_df[feature_cols]
    y_cmp = comparison_df[TARGET_COLUMN].to_numpy()
    cmp_ids = comparison_df["id"].to_numpy()

    models = get_available_models(preprocessing_config.random_seed)
    out: list[ModelPredictions] = []

    for key, spec in models.items():
        t0 = time.perf_counter()
        search = run_grid_search(
            spec, "classical", feature_groups, preprocessing_config,
            X_train_raw, y_train, cv_folds=CV_FOLDS, cv_shuffle=CV_SHUFFLE,
            random_seed=preprocessing_config.random_seed,
        )
        train_seconds = time.perf_counter() - t0

        # Model is now FROZEN. Only prediction happens below.
        y_proba = search.best_estimator_.predict_proba(X_cmp_raw)[:, 1]
        y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

        out.append(
            ModelPredictions(
                model_key=key,
                display_name=spec.display_name,
                ids=cmp_ids,
                y_true=y_cmp,
                y_proba=y_proba,
                y_pred=y_pred,
                metrics=_metrics_from_proba(y_cmp, y_proba),
                best_params={k.replace("model__", ""): v for k, v in search.best_params_.items()},
                provenance="reproduced_training",
                train_seconds=train_seconds,
            )
        )
    return out


def evaluate_qsvm_on_comparison_set(
    stage_train_df: pd.DataFrame,
    test_classical_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    feature_groups,
    preprocessing_config: PreprocessingConfig,
    quantum_config: QuantumConfig,
) -> tuple[ModelPredictions, dict, dict]:
    """Reproduce the Stage QSVM exactly (same feature map, qubits, kernel,
    C-grid, seed) and predict on the comparison set.

    Note: process_stage already uses the 200-row set as its quantum test
    set, so the QSVM's evaluation population was ALREADY the comparison
    set in Stages A/B/C -- it is the classical side that changes here.
    This function re-derives the QSVM predictions so they can be paired
    observation-by-observation with the classical ones.
    """
    processed = process_stage(
        stage_train_df, test_classical_df, comparison_df,
        TARGET_COLUMN, feature_groups, preprocessing_config,
    )

    feature_map = build_feature_map(
        quantum_config.feature_map_name, processed.n_qubits,
        reps=quantum_config.reps, entanglement=quantum_config.entanglement,
        paulis=quantum_config.paulis,
    )
    backend = get_backend(quantum_config.backend_name)

    t0 = time.perf_counter()
    train_svs = compute_statevectors(processed.X_train_quantum, feature_map, backend)
    test_svs = compute_statevectors(processed.X_test_quantum, feature_map, backend)
    K_train_train = kernel_matrix_from_statevectors_vectorized(train_svs, train_svs, symmetric=True)
    K_test_train = kernel_matrix_from_statevectors_vectorized(test_svs, train_svs, symmetric=False)
    kernel_seconds = time.perf_counter() - t0

    diag = kernel_diagnostics(K_train_train, symmetric=True)

    t1 = time.perf_counter()
    search = run_qsvm_grid_search(
        K_train_train, processed.y_train,
        cv_folds=CV_FOLDS, cv_shuffle=CV_SHUFFLE,
        random_seed=preprocessing_config.random_seed,
    )
    tuning_seconds = time.perf_counter() - t1

    # FROZEN. Predict only.
    y_proba = search.best_estimator_.predict_proba(K_test_train)[:, 1]
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    y_cmp = processed.y_test_quantum
    cmp_ids = comparison_df["id"].to_numpy()

    preds = ModelPredictions(
        model_key="qsvm",
        display_name="QSVM (fidelity kernel)",
        ids=cmp_ids,
        y_true=y_cmp,
        y_proba=y_proba,
        y_pred=y_pred,
        metrics=_metrics_from_proba(y_cmp, y_proba),
        best_params={k.replace("model__", ""): v for k, v in search.best_params_.items()},
        provenance="reproduced_training",
        train_seconds=kernel_seconds + tuning_seconds,
    )
    timing = {
        "kernel_seconds": kernel_seconds,
        "cv_tuning_seconds": tuning_seconds,
        "n_train": int(len(processed.X_train_quantum)),
        "n_comparison": int(len(processed.X_test_quantum)),
    }
    return preds, timing, diag.to_dict()


# --------------------------------------------------------------------------
# Paired statistics on the identical observations
# --------------------------------------------------------------------------


def paired_bootstrap_delta(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    metric: str = "roc_auc",
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired bootstrap of (metric_a - metric_b) on the SAME observations.

    Paired is the correct design here: every model is evaluated on the
    identical rows, so resampling the same indices for both models removes
    the between-sample variance that an unpaired comparison would carry.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    def _score(yt, pa):
        if len(np.unique(yt)) < 2:
            return np.nan
        return roc_auc_score(yt, pa) if metric == "roc_auc" else average_precision_score(yt, pa)

    observed = _score(y_true, proba_a) - _score(y_true, proba_b)

    deltas = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        d = _score(yt, proba_a[idx]) - _score(yt, proba_b[idx])
        if not np.isnan(d):
            deltas.append(d)

    deltas_arr = np.array(deltas)
    ci_low, ci_high = np.quantile(deltas_arr, [0.025, 0.975])
    # Two-sided bootstrap p-value: proportion of resamples on the opposite
    # side of zero from the observed effect, doubled.
    if observed >= 0:
        p = 2.0 * float(np.mean(deltas_arr <= 0))
    else:
        p = 2.0 * float(np.mean(deltas_arr >= 0))
    p = min(1.0, p)

    return {
        "metric": metric,
        "observed_delta": float(observed),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci_includes_zero": bool(ci_low <= 0.0 <= ci_high),
        "bootstrap_p_value": p,
        "n_resamples_used": int(len(deltas_arr)),
        "seed": seed,
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "roc_auc",
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Percentile bootstrap CI for a single model's metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    fn = roc_auc_score if metric == "roc_auc" else average_precision_score
    point = float(fn(y_true, y_proba))

    vals = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(fn(y_true[idx], y_proba[idx]))
    arr = np.array(vals)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return {
        "metric": metric,
        "point_estimate": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_resamples_used": int(len(arr)),
        "seed": seed,
    }


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test on paired binary predictions (same observations).

    Contingency counts the DISCORDANT cells only:
        b = A correct, B wrong
        c = A wrong,   B correct
    Under H0 (no difference), b ~ Binomial(b + c, 0.5). The exact binomial
    test is used rather than the chi-square approximation because with
    n=200 the discordant counts can be small.

    Note: this tests the 0.5-threshold LABEL decisions, not ROC-AUC.
    """
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    both_correct = int(np.sum(a_correct & b_correct))
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))

    n_discordant = a_only + b_only
    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = float(stats.binomtest(a_only, n_discordant, 0.5).pvalue)

    return {
        "contingency": {
            "both_correct": both_correct,
            "a_correct_b_wrong": a_only,
            "a_wrong_b_correct": b_only,
            "both_wrong": both_wrong,
        },
        "n_discordant": n_discordant,
        "test": "exact binomial McNemar",
        "p_value": p_value,
    }


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_curves(all_preds: list[ModelPredictions], out_dir: Path, stage_size: int, n_cmp: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    for p in all_preds:
        fpr, tpr, _ = roc_curve(p.y_true, p.y_proba)
        ax.plot(fpr, tpr, label=f"{p.display_name} (AUC={p.metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC -- Stage n={stage_size:,}\nIdentical {n_cmp}-row held-out test set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(7, 6))
    for p in all_preds:
        prec, rec, _ = precision_recall_curve(p.y_true, p.y_proba)
        ax.plot(rec, prec, label=f"{p.display_name} (AP={p.metrics['pr_auc']:.4f})")
    base = float(np.mean(all_preds[0].y_true))
    ax.axhline(base, ls="--", color="grey", label=f"Chance ({base:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall -- Stage n={stage_size:,}\nIdentical {n_cmp}-row held-out test set")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=150)
    plt.close(fig)

    # Confusion matrices
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, axes = plt.subplots(1, len(all_preds), figsize=(4 * len(all_preds), 4))
    for ax, p in zip(np.atleast_1d(axes), all_preds):
        m = p.metrics
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ConfusionMatrixDisplay(cm, display_labels=["No CVD", "CVD"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(p.display_name, fontsize=9)
    fig.suptitle(f"Confusion matrices -- identical {n_cmp}-row test set (threshold=0.5)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Stage orchestrator
# --------------------------------------------------------------------------


def run_corrected_stage(stage_size: int, *, make_plots: bool = True) -> dict:
    """Run the full corrected comparison for one stage size."""
    out_dir = RESULTS_ROOT / f"stage_{stage_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = derive_features(load_raw_cardio())
    flags = compute_cleaning_flags(df)
    modeling_df, _ = split_modeling_subset(df, flags)
    split = build_fixed_split(modeling_df, TARGET_COLUMN, seed=42)
    stages = nested_stratified_stage_samples(
        split.training_pool, TARGET_COLUMN, [stage_size], seed=42
    )
    stage_train_df = stages[stage_size]

    comparison_df = split.test_set_quantum  # the EXISTING 200-row set, recovered
    cmp_ids = comparison_df["id"].tolist()
    fingerprint = comparison_set_fingerprint(cmp_ids)

    # Leakage guards -- assert, do not assume
    assert len(set(cmp_ids) & set(stage_train_df["id"])) == 0, "comparison set overlaps training data"
    assert set(cmp_ids).issubset(set(split.test_set_classical["id"])), "comparison set not a subset of the classical test set"

    fg = get_cardio_feature_groups()
    pcfg = PreprocessingConfig()
    qcfg = QuantumConfig()

    classical_preds = evaluate_classical_on_comparison_set(stage_train_df, comparison_df, fg, pcfg)
    qsvm_preds, qsvm_timing, qsvm_diag = evaluate_qsvm_on_comparison_set(
        stage_train_df, split.test_set_classical, comparison_df, fg, pcfg, qcfg
    )

    all_preds = classical_preds + [qsvm_preds]

    # Alignment guard: every model must have predicted the same ids in the same order
    for p in all_preds:
        assert list(p.ids) == cmp_ids, f"{p.model_key} predictions are not aligned to the comparison ids"
        assert np.array_equal(p.y_true, qsvm_preds.y_true), f"{p.model_key} y_true differs from QSVM's"

    # Per-model bootstrap CIs
    single_cis = {
        p.model_key: {
            "roc_auc": bootstrap_metric_ci(p.y_true, p.y_proba, "roc_auc"),
            "pr_auc": bootstrap_metric_ci(p.y_true, p.y_proba, "pr_auc"),
        }
        for p in all_preds
    }

    # Paired classical-vs-QSVM comparisons
    paired = {}
    for p in classical_preds:
        paired[p.model_key] = {
            "vs": "qsvm",
            "roc_auc": paired_bootstrap_delta(p.y_true, p.y_proba, qsvm_preds.y_proba, "roc_auc"),
            "pr_auc": paired_bootstrap_delta(p.y_true, p.y_proba, qsvm_preds.y_proba, "pr_auc"),
            "mcnemar": mcnemar_exact(p.y_true, p.y_pred, qsvm_preds.y_pred),
            "direct_differences": {
                m: float(p.metrics[m] - qsvm_preds.metrics[m])
                for m in ("roc_auc", "pr_auc", "sensitivity", "specificity", "accuracy", "f1")
            },
        }

    # Persist predictions keyed by id
    pred_df = pd.DataFrame({"id": cmp_ids, "y_true": qsvm_preds.y_true})
    for p in all_preds:
        pred_df[f"{p.model_key}_proba"] = p.y_proba
        pred_df[f"{p.model_key}_pred"] = p.y_pred
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    metrics_rows = [
        {
            "Model": p.display_name,
            "model_key": p.model_key,
            **{k: p.metrics[k] for k in
               ("roc_auc", "pr_auc", "sensitivity", "specificity", "accuracy", "precision", "f1",
                "tn", "fp", "fn", "tp")},
            "best_params": json.dumps(p.best_params),
            "provenance": p.provenance,
            "train_seconds": round(p.train_seconds, 3),
        }
        for p in all_preds
    ]
    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False)

    report = {
        "stage_size": stage_size,
        "comparison_set": {
            "n": len(cmp_ids),
            "fingerprint_sha256_16": fingerprint,
            "ids": cmp_ids,
            "positive_rate": float(np.mean(qsvm_preds.y_true)),
            "source": "src.large_dataset.sampling.build_fixed_split(...).test_set_quantum, seed=42",
            "recovered_not_resampled": True,
            "overlap_with_training": 0,
            "is_subset_of_2000_row_classical_test_set": True,
        },
        "methodology": {
            "model_provenance": "reproduced_training (no artifacts were persisted in Stages A/B/C)",
            "threshold": DECISION_THRESHOLD,
            "threshold_source": "Phase 3 fixed 0.5; not selected from test data",
            "cv_folds": CV_FOLDS,
            "seed": 42,
            "unchanged": [
                "training subsets", "test-set membership", "preprocessing", "feature groups",
                "PCA components", "quantum feature map", "qubits", "reps", "entanglement",
                "kernel definition", "classical hyperparameter grids", "QSVM C-grid",
            ],
        },
        "metrics": {p.model_key: p.metrics for p in all_preds},
        "best_params": {p.model_key: p.best_params for p in all_preds},
        "bootstrap_ci": single_cis,
        "paired_classical_vs_qsvm": paired,
        "qsvm_timing": qsvm_timing,
        "qsvm_kernel_diagnostics": qsvm_diag,
        "bootstrap_config": {"n_resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
    }
    with open(out_dir / "corrected_comparison.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)

    if make_plots:
        plot_curves(all_preds, out_dir, stage_size, len(cmp_ids))

    return report


def print_stage_report(report: dict) -> None:
    n = report["comparison_set"]["n"]
    print("=" * 78)
    print(f"CORRECTED COMPARISON -- Stage n={report['stage_size']:,} "
          f"| identical {n}-row test set (fp={report['comparison_set']['fingerprint_sha256_16']})")
    print("=" * 78)
    hdr = f"{'Model':<24}{'ROC-AUC':>9}{'PR-AUC':>9}{'Sens':>8}{'Spec':>8}{'Acc':>8}{'Prec':>8}{'F1':>8}"
    print(hdr)
    print("-" * len(hdr))
    for key, m in report["metrics"].items():
        name = "QSVM (fidelity kernel)" if key == "qsvm" else key.replace("_", " ").title()
        print(f"{name:<24}{m['roc_auc']:>9.4f}{m['pr_auc']:>9.4f}{m['sensitivity']:>8.4f}"
              f"{m['specificity']:>8.4f}{m['accuracy']:>8.4f}{m['precision']:>8.4f}{m['f1']:>8.4f}")
    print()
    print("Paired comparisons vs QSVM (same 200 observations):")
    for key, p in report["paired_classical_vs_qsvm"].items():
        r = p["roc_auc"]
        mc = p["mcnemar"]
        print(f"  {key:<22} dROC-AUC={r['observed_delta']:+.4f} "
              f"95%CI=[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] "
              f"{'includes 0' if r['ci_includes_zero'] else 'EXCLUDES 0'} "
              f"boot_p={r['bootstrap_p_value']:.3f} | McNemar p={mc['p_value']:.3f}")
    print("=" * 78)


if __name__ == "__main__":
    import sys

    stage = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    rep = run_corrected_stage(stage)
    print_stage_report(rep)
