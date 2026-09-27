"""Adaptive-feature-map QSVM experiment (post-Stage-D).

Tests: does a training-data-informed entangling structure (see
src.quantum.adaptive_feature_map) improve the QSVM's fidelity kernel over
the fixed zz_feature_map("linear") baseline, on the IDENTICAL Stage D
n_train=20,000 training subset and IDENTICAL 200-row test set?

WHAT IS REUSED, UNMODIFIED:
    - src.large_dataset.stage_d.build_stage_d_split (the exact same n=20,000
      training subset and fixed test sets Stage D used -- no new sampling)
    - src.large_dataset.pipeline.process_stage (Phase 2 preprocessing/PCA,
      fit fresh on the Stage D training subset -- deterministically
      reproduces the exact X_train_quantum/X_test_quantum Stage D itself
      used, since same data + same seed)
    - src.quantum.kernel.compute_statevectors / kernel_matrix_from_statevectors_blockwise
      / kernel_diagnostics (the SAME memory-safe kernel path Stage D used)
    - src.quantum.qsvm.run_qsvm_grid_search (the SAME C-grid, 5-fold CV,
      seed -- only the feature map differs; C is still the only tuned
      hyperparameter, exactly as at Stage D)
    - src.large_dataset.corrected_comparison.{comparison_set_fingerprint,
      DECISION_THRESHOLD, paired_bootstrap_delta, bootstrap_metric_ci,
      mcnemar_exact} and src.large_dataset.statistical_robustness.delong_test
      (the SAME paired-statistics code every prior phase used)

WHAT IS NEW:
    - src.quantum.adaptive_feature_map (the adaptive map itself)
    - The BASELINE QSVM is NOT retrained here. Its predictions on the
      200-row set are already fully persisted at
      results/large_dataset/stage_d/predictions.csv (qsvm_proba/qsvm_pred),
      verified against the same test fingerprint before use. Retraining an
      identical baseline would waste ~30 minutes of QSVM compute for a
      result that must, by construction, already exist.

TEST-SET DISCIPLINE: the adaptive map is derived from
processed.X_train_quantum ONLY (see adaptive_feature_map.py's own
docstring for the structural guarantee). The 200-row test set is touched
exactly once, after the adaptive QSVM's hyperparameter (C) has already
been selected by training-only CV -- identical discipline to every prior
phase.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import precision_recall_curve, roc_curve

from src.classical.evaluation import compute_classification_metrics
from src.data.inspect_dataset import find_project_root
from src.large_dataset.corrected_comparison import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DECISION_THRESHOLD,
    bootstrap_metric_ci,
    comparison_set_fingerprint,
    mcnemar_exact,
    paired_bootstrap_delta,
)
from src.large_dataset.pipeline import process_stage
from src.large_dataset.schema import TARGET_COLUMN, get_cardio_feature_groups
from src.large_dataset.stage_d import build_stage_d_split
from src.large_dataset.statistical_robustness import delong_test
from src.preprocessing.config import PreprocessingConfig
from src.quantum.adaptive_feature_map import build_adaptive_feature_map_from_training_data
from src.quantum.backends import get_backend
from src.quantum.config import QuantumConfig
from src.quantum.kernel import compute_statevectors, kernel_diagnostics, kernel_matrix_from_statevectors_blockwise
from src.quantum.qsvm import run_qsvm_grid_search

CV_FOLDS = 5
CV_SHUFFLE = True
EXPECTED_FINGERPRINT = "96eac11a8394b87e"

RESULTS_ROOT = find_project_root() / "results" / "large_dataset" / "adaptive_qsvm"
STAGE_D_PREDICTIONS_PATH = find_project_root() / "results" / "large_dataset" / "stage_d" / "predictions.csv"

KERNEL_BLOCK_SIZE = 2000
KERNEL_DTYPE = np.float32


@dataclass
class ModelPredictions:
    model_key: str
    display_name: str
    ids: np.ndarray
    y_true: np.ndarray
    y_proba: np.ndarray
    y_pred: np.ndarray
    metrics: dict


def _metrics_from_proba(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    return compute_classification_metrics(y_true, y_pred, y_proba)


# --------------------------------------------------------------------------
# Baseline: reuse Stage D's already-persisted predictions
# --------------------------------------------------------------------------


def load_baseline_predictions() -> ModelPredictions:
    if not STAGE_D_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Expected Stage D predictions at {STAGE_D_PREDICTIONS_PATH}; the baseline QSVM "
            f"for this experiment is REUSED from Stage D, never retrained, per this module's "
            f"own docstring. Run Stage D first if this file is missing."
        )
    df = pd.read_csv(STAGE_D_PREDICTIONS_PATH)
    fp = comparison_set_fingerprint(df["id"].tolist())
    if fp != EXPECTED_FINGERPRINT:
        raise ValueError(f"Stage D predictions.csv fingerprint {fp} != expected {EXPECTED_FINGERPRINT}")
    y_true = df["y_true"].to_numpy()
    y_proba = df["qsvm_proba"].to_numpy()
    y_pred = df["qsvm_pred"].to_numpy()
    return ModelPredictions(
        model_key="baseline_qsvm", display_name="Baseline QSVM (zz_feature_map, linear)",
        ids=df["id"].to_numpy(), y_true=y_true, y_proba=y_proba, y_pred=y_pred,
        metrics=_metrics_from_proba(y_true, y_proba),
    )


# --------------------------------------------------------------------------
# Adaptive QSVM
# --------------------------------------------------------------------------


def run_adaptive_qsvm(*, make_plots: bool = True) -> dict:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()
    runtime: dict[str, float] = {}

    print("[adaptive_qsvm] loading baseline predictions from Stage D (no retraining) ...", flush=True)
    baseline = load_baseline_predictions()

    print("[adaptive_qsvm] rebuilding Stage D's exact n=20,000 split ...", flush=True)
    t0 = time.perf_counter()
    stage_d = build_stage_d_split()
    runtime["split_seconds"] = time.perf_counter() - t0

    cmp_ids = stage_d.split.test_set_quantum["id"].tolist()
    fingerprint = comparison_set_fingerprint(cmp_ids)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"Comparison set fingerprint {fingerprint} != expected {EXPECTED_FINGERPRINT}")
    if list(baseline.ids) != cmp_ids:
        raise ValueError("Baseline predictions' ids are not aligned to the current comparison set.")

    fg = get_cardio_feature_groups()
    pcfg = PreprocessingConfig()
    qcfg = QuantumConfig()

    print("[adaptive_qsvm] fitting preprocessing/PCA on the Stage D training subset (train-only fit) ...", flush=True)
    t1 = time.perf_counter()
    processed = process_stage(
        stage_d.stage_train_df, stage_d.split.test_set_classical, stage_d.split.test_set_quantum,
        TARGET_COLUMN, fg, pcfg,
    )
    runtime["preprocessing_and_pca_seconds"] = time.perf_counter() - t1

    print("[adaptive_qsvm] deriving adaptive entangling structure from TRAINING DATA ONLY "
          f"(n_train={len(processed.X_train_quantum)}, n_qubits={processed.n_qubits}) ...", flush=True)
    t2 = time.perf_counter()
    feature_map, adaptive_report = build_adaptive_feature_map_from_training_data(
        processed.X_train_quantum, processed.n_qubits, qcfg.reps, random_state=qcfg.random_seed,
    )
    runtime["adaptive_map_derivation_seconds"] = time.perf_counter() - t2
    print(f"[adaptive_qsvm] selected pairs: {adaptive_report.selected_pairs} "
          f"(baseline linear-chain pairs: {adaptive_report.baseline_pairs})", flush=True)

    backend = get_backend(qcfg.backend_name)

    t3 = time.perf_counter()
    train_svs = compute_statevectors(processed.X_train_quantum, feature_map, backend)
    t4 = time.perf_counter()
    print(f"[adaptive_qsvm] train statevectors done in {t4 - t3:.1f}s", flush=True)
    test_svs = compute_statevectors(processed.X_test_quantum, feature_map, backend)
    t5 = time.perf_counter()
    print(f"[adaptive_qsvm] test statevectors done in {t5 - t4:.1f}s", flush=True)

    rss_before_kernel_mb = proc.memory_info().rss / 1e6
    print(f"[adaptive_qsvm] assembling K_train_train ({len(train_svs)}x{len(train_svs)}, "
          f"dtype={KERNEL_DTYPE.__name__}, block_size={KERNEL_BLOCK_SIZE}); "
          f"RSS before = {rss_before_kernel_mb:.0f} MB ...", flush=True)
    K_train_train = kernel_matrix_from_statevectors_blockwise(
        train_svs, train_svs, symmetric=True, block_size=KERNEL_BLOCK_SIZE, dtype=KERNEL_DTYPE,
    )
    t6 = time.perf_counter()
    rss_after_train_kernel_mb = proc.memory_info().rss / 1e6
    print(f"[adaptive_qsvm] K_train_train done in {t6 - t5:.1f}s; RSS after = {rss_after_train_kernel_mb:.0f} MB", flush=True)
    K_test_train = kernel_matrix_from_statevectors_blockwise(
        test_svs, train_svs, symmetric=False, block_size=KERNEL_BLOCK_SIZE, dtype=KERNEL_DTYPE,
    )
    t7 = time.perf_counter()
    print(f"[adaptive_qsvm] K_test_train done in {t7 - t6:.1f}s", flush=True)

    diag = kernel_diagnostics(K_train_train, symmetric=True)
    print(f"[adaptive_qsvm] kernel diagnostics: {diag.to_dict()}", flush=True)

    print("[adaptive_qsvm] starting CV C-grid search (same grid/folds/seed as Stage D baseline) ...", flush=True)
    t8 = time.perf_counter()
    search = run_qsvm_grid_search(
        K_train_train, processed.y_train, cv_folds=CV_FOLDS, cv_shuffle=CV_SHUFFLE,
        random_seed=pcfg.random_seed,
    )
    cv_seconds = time.perf_counter() - t8
    rss_after_cv_mb = proc.memory_info().rss / 1e6
    print(f"[adaptive_qsvm] CV done in {cv_seconds:.1f}s, best_C={search.best_params_}, "
          f"RSS after CV = {rss_after_cv_mb:.0f} MB", flush=True)

    t9 = time.perf_counter()
    y_proba = search.best_estimator_.predict_proba(K_test_train)[:, 1]
    inference_seconds = time.perf_counter() - t9
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    adaptive = ModelPredictions(
        model_key="adaptive_qsvm", display_name="Adaptive QSVM (mutual-information entanglement)",
        ids=np.array(cmp_ids), y_true=processed.y_test_quantum, y_proba=y_proba, y_pred=y_pred,
        metrics=_metrics_from_proba(processed.y_test_quantum, y_proba),
    )
    assert np.array_equal(adaptive.y_true, baseline.y_true), "adaptive/baseline y_true mismatch"
    assert list(adaptive.ids) == list(baseline.ids), "adaptive/baseline id mismatch"

    kernel_timing = {
        "n_train": int(len(processed.X_train_quantum)),
        "n_test": int(len(processed.X_test_quantum)),
        "train_statevectors_seconds": t4 - t3,
        "test_statevectors_seconds": t5 - t4,
        "K_train_train_assembly_seconds": t6 - t5,
        "K_test_train_assembly_seconds": t7 - t6,
        "cv_tuning_seconds": cv_seconds,
        "final_inference_seconds": inference_seconds,
        "total_qsvm_wall_seconds": (t7 - t3) + cv_seconds + inference_seconds,
        "K_train_train_shape": list(K_train_train.shape),
        "K_train_train_dtype": str(K_train_train.dtype),
        "K_train_train_memory_mb": K_train_train.nbytes / 1e6,
        "kernel_block_size": KERNEL_BLOCK_SIZE,
        "process_rss_mb": {
            "before_kernel_assembly": rss_before_kernel_mb,
            "after_train_train_kernel": rss_after_train_kernel_mb,
            "after_cv_tuning": rss_after_cv_mb,
        },
    }
    runtime["qsvm_total_seconds"] = kernel_timing["total_qsvm_wall_seconds"]
    runtime["kernel_timing"] = kernel_timing

    # ---- paired statistics: delta = adaptive - baseline ----
    print("[adaptive_qsvm] computing paired statistics (DeLong, bootstrap, McNemar) ...", flush=True)
    dl = delong_test(baseline.y_true, adaptive.y_proba, baseline.y_proba)
    bs_roc = paired_bootstrap_delta(baseline.y_true, adaptive.y_proba, baseline.y_proba, "roc_auc",
                                     n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    bs_pr = paired_bootstrap_delta(baseline.y_true, adaptive.y_proba, baseline.y_proba, "pr_auc",
                                    n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    mc = mcnemar_exact(baseline.y_true, adaptive.y_pred, baseline.y_pred)

    single_cis = {}
    for m in (baseline, adaptive):
        single_cis[m.model_key] = {
            "roc_auc": bootstrap_metric_ci(m.y_true, m.y_proba, "roc_auc", n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED),
            "pr_auc": bootstrap_metric_ci(m.y_true, m.y_proba, "pr_auc", n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED),
        }

    statistics = {
        "delong": dl.to_dict(),
        "bootstrap_roc_auc": bs_roc,
        "bootstrap_pr_auc": bs_pr,
        "mcnemar": mc,
        "single_model_cis": single_cis,
        "delta_definition": "Adaptive QSVM - Baseline QSVM (positive = adaptive better)",
    }

    # ---- persist ----
    pred_df = pd.DataFrame({
        "id": cmp_ids, "y_true": baseline.y_true,
        "baseline_qsvm_proba": baseline.y_proba, "baseline_qsvm_pred": baseline.y_pred,
        "adaptive_qsvm_proba": adaptive.y_proba, "adaptive_qsvm_pred": adaptive.y_pred,
    })
    pred_df.to_csv(RESULTS_ROOT / "predictions.csv", index=False)

    metrics_df = pd.DataFrame([
        {"Model": m.display_name, "model_key": m.model_key, **m.metrics}
        for m in (baseline, adaptive)
    ])
    metrics_df.to_csv(RESULTS_ROOT / "metrics.csv", index=False)

    with open(RESULTS_ROOT / "adaptive_map_config.json", "w", encoding="utf-8") as fh:
        json.dump(adaptive_report.to_dict(), fh, indent=2, default=float)

    with open(RESULTS_ROOT / "kernel_diagnostics.json", "w", encoding="utf-8") as fh:
        json.dump({"timing": kernel_timing, "diagnostics": diag.to_dict()}, fh, indent=2, default=float)

    with open(RESULTS_ROOT / "runtime.json", "w", encoding="utf-8") as fh:
        json.dump(runtime, fh, indent=2, default=float)

    with open(RESULTS_ROOT / "statistics.json", "w", encoding="utf-8") as fh:
        json.dump(statistics, fh, indent=2, default=float)

    summary = {
        "comparison_set": {"n": len(cmp_ids), "fingerprint": fingerprint, "matches_expected": fingerprint == EXPECTED_FINGERPRINT,
                            "positive_rate": float(np.mean(baseline.y_true))},
        "n_train": int(len(processed.X_train_quantum)),
        "adaptive_map": adaptive_report.to_dict(),
        "baseline_metrics": baseline.metrics,
        "adaptive_metrics": adaptive.metrics,
        "best_C": {k: v for k, v in search.best_params_.items()},
        "statistics": statistics,
        "runtime": runtime,
        "kernel_diagnostics": diag.to_dict(),
    }
    with open(RESULTS_ROOT / "adaptive_qsvm_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    if make_plots:
        _plot_curves([baseline, adaptive], RESULTS_ROOT)

    return summary


def _plot_curves(preds: list[ModelPredictions], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    n_cmp = len(preds[0].y_true)

    fig, ax = plt.subplots(figsize=(7, 6))
    for p in preds:
        fpr, tpr, _ = roc_curve(p.y_true, p.y_proba)
        ax.plot(fpr, tpr, label=f"{p.display_name} (AUC={p.metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC -- Adaptive vs Baseline QSVM\nIdentical {n_cmp}-row held-out test set, n_train=20,000")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for p in preds:
        prec, rec, _ = precision_recall_curve(p.y_true, p.y_proba)
        ax.plot(rec, prec, label=f"{p.display_name} (AP={p.metrics['pr_auc']:.4f})")
    base = float(np.mean(preds[0].y_true))
    ax.axhline(base, ls="--", color="grey", label=f"Chance ({base:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall -- Adaptive vs Baseline QSVM\nIdentical {n_cmp}-row held-out test set")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(preds), figsize=(4 * len(preds), 4))
    for ax, p in zip(np.atleast_1d(axes), preds):
        m = p.metrics
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ConfusionMatrixDisplay(cm, display_labels=["No CVD", "CVD"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(p.display_name, fontsize=9)
    fig.suptitle(f"Confusion matrices -- Adaptive vs Baseline QSVM, identical {n_cmp}-row test set (threshold=0.5)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    result = run_adaptive_qsvm()
    print(json.dumps({k: v for k, v in result.items() if k != "statistics"}, indent=2, default=float))
