"""Phase 8A: label-aware adaptive feature-map QSVM SCREENING experiment
(n_train=2,000 only -- see docs/PHASE_8A_LABEL_AWARE_SCREENING.md).

HYPOTHESIS: feature-pair interactions that are informative about the class
label (measured via training-only interaction information / synergy, see
src.quantum.label_aware_feature_map) may produce a more useful QSVM
fidelity kernel than interactions selected solely from feature-feature
mutual information (the Phase 6/7 MI-adaptive map, which was found NOT
reproducible across seeds at n=20,000 -- docs/PHASE_7_ADAPTIVE_ROBUSTNESS.md).

WHY n_train=2,000, NOT 20,000: this is a screening experiment. Per the
governing instruction, a new mechanism is checked cheaply first; scaling
only happens after an explicit, separate decision. n_train=2,000 was
never used by any prior stage (A=1,000, B=5,000, C=10,000, D=20,000) --
it is a genuinely new, smaller checkpoint, deliberately chosen to make
this screen fast (a single representative n=20,000 QSVM CV pass cost
~1,700-1,950s in Phase 7/Stage D; at n=2,000 the same grid is expected,
and measured below, to cost roughly two orders of magnitude less).

WHAT IS REUSED, UNMODIFIED:
    - src.large_dataset.schema / cleaning / sampling (the exact same
      cleaned modeling pool, fixed test sets, and nested-sampling function
      every prior stage used -- n_train=2,000 is just a new size argument
      to the SAME nested_stratified_stage_samples call)
    - src.large_dataset.pipeline.process_stage (Phase 2 preprocessing/PCA,
      fit ONCE on this screen's n=2,000 training subset, shared across all
      3 feature maps since none of them change what feeds the encoding)
    - src.quantum.feature_maps.build_feature_map (baseline zz_feature_map)
    - src.quantum.adaptive_feature_map (the Phase 6/7 MI-adaptive map,
      re-derived at n=2,000 for a fair, same-training-size 3-way
      comparison -- not reused from the n=20,000 result, which used a
      different training subset entirely)
    - src.quantum.label_aware_feature_map (the new mechanism)
    - src.quantum.kernel.{compute_statevectors, kernel_matrix_from_statevectors_blockwise,
      kernel_diagnostics} (the same memory-safe kernel path Stage D/Phase 7 used)
    - src.quantum.qsvm.run_qsvm_grid_search (same C-grid, 5-fold CV, seed)
    - src.large_dataset.corrected_comparison / statistical_robustness
      (the same paired-statistics functions every prior phase used)

TEST-SET DISCIPLINE: all three feature maps are derived from n_train=2,000
TRAINING data (and, for the label-aware map, training labels) only. The
200-row test set is touched exactly once per model, after each model's own
training-only CV has already selected C.
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
from src.large_dataset.cleaning import compute_cleaning_flags, split_modeling_subset
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
from src.large_dataset.sampling import build_fixed_split, nested_stratified_stage_samples
from src.large_dataset.schema import TARGET_COLUMN, derive_features, get_cardio_feature_groups, load_raw_cardio
from src.large_dataset.statistical_robustness import delong_test, holm_bonferroni
from src.preprocessing.config import PreprocessingConfig
from src.quantum.adaptive_feature_map import build_adaptive_feature_map_from_training_data
from src.quantum.backends import get_backend
from src.quantum.config import QuantumConfig
from src.quantum.feature_maps import build_feature_map
from src.quantum.kernel import compute_statevectors, kernel_diagnostics, kernel_matrix_from_statevectors_blockwise
from src.quantum.label_aware_feature_map import build_label_aware_feature_map_from_training_data
from src.quantum.qsvm import run_qsvm_grid_search

SCREEN_N_TRAIN = 2000
CV_FOLDS = 5
CV_SHUFFLE = True
EXPECTED_FINGERPRINT = "96eac11a8394b87e"
SAMPLING_SEED = 42

RESULTS_ROOT = find_project_root() / "results" / "large_dataset" / "phase8a_label_aware"
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


@dataclass
class ScreeningSplit:
    split: object  # LargeDatasetSplit
    train_df: pd.DataFrame


def build_screening_split() -> ScreeningSplit:
    df = derive_features(load_raw_cardio())
    flags = compute_cleaning_flags(df)
    modeling_df, _ = split_modeling_subset(df, flags)
    split = build_fixed_split(modeling_df, TARGET_COLUMN, seed=SAMPLING_SEED)
    stages = nested_stratified_stage_samples(split.training_pool, TARGET_COLUMN, [SCREEN_N_TRAIN], seed=SAMPLING_SEED)
    train_df = stages[SCREEN_N_TRAIN]

    train_ids = set(train_df["id"])
    overlap_c = train_ids & set(split.test_set_classical["id"])
    overlap_q = train_ids & set(split.test_set_quantum["id"])
    if overlap_c or overlap_q:
        raise ValueError(f"Screening training set overlaps a fixed test set: {len(overlap_c)} / {len(overlap_q)} rows.")

    return ScreeningSplit(split=split, train_df=train_df)


def _run_one_feature_map(
    model_key: str,
    display_name: str,
    circuit,
    processed,
    cmp_ids: list,
    proc: psutil.Process,
    quantum_config: QuantumConfig,
    preprocessing_config: PreprocessingConfig,
) -> tuple[ModelPredictions, dict]:
    backend = get_backend(quantum_config.backend_name)
    rss0 = proc.memory_info().rss / 1e6

    t0 = time.perf_counter()
    train_svs = compute_statevectors(processed.X_train_quantum, circuit, backend)
    t1 = time.perf_counter()
    test_svs = compute_statevectors(processed.X_test_quantum, circuit, backend)
    t2 = time.perf_counter()

    K_train_train = kernel_matrix_from_statevectors_blockwise(
        train_svs, train_svs, symmetric=True, block_size=KERNEL_BLOCK_SIZE, dtype=KERNEL_DTYPE,
    )
    t3 = time.perf_counter()
    K_test_train = kernel_matrix_from_statevectors_blockwise(
        test_svs, train_svs, symmetric=False, block_size=KERNEL_BLOCK_SIZE, dtype=KERNEL_DTYPE,
    )
    t4 = time.perf_counter()
    diag = kernel_diagnostics(K_train_train, symmetric=True)
    rss1 = proc.memory_info().rss / 1e6

    t5 = time.perf_counter()
    search = run_qsvm_grid_search(
        K_train_train, processed.y_train, cv_folds=CV_FOLDS, cv_shuffle=CV_SHUFFLE,
        random_seed=preprocessing_config.random_seed,
    )
    cv_seconds = time.perf_counter() - t5
    rss2 = proc.memory_info().rss / 1e6

    t6 = time.perf_counter()
    y_proba = search.best_estimator_.predict_proba(K_test_train)[:, 1]
    inference_seconds = time.perf_counter() - t6
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    preds = ModelPredictions(
        model_key=model_key, display_name=display_name, ids=np.array(cmp_ids),
        y_true=processed.y_test_quantum, y_proba=y_proba, y_pred=y_pred,
        metrics=_metrics_from_proba(processed.y_test_quantum, y_proba),
    )
    timing = {
        "train_statevectors_seconds": t1 - t0, "test_statevectors_seconds": t2 - t1,
        "K_train_train_assembly_seconds": t3 - t2, "K_test_train_assembly_seconds": t4 - t3,
        "cv_tuning_seconds": cv_seconds, "final_inference_seconds": inference_seconds,
        "total_wall_seconds": (t4 - t0) + cv_seconds + inference_seconds,
        "best_C": search.best_params_, "kernel_diagnostics": diag.to_dict(),
        "rss_mb": {"before": rss0, "after_kernel": rss1, "after_cv": rss2},
        "K_train_train_memory_mb": K_train_train.nbytes / 1e6,
    }
    return preds, timing


def run_phase8a_screening(*, make_plots: bool = True) -> dict:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()
    runtime: dict = {}

    print(f"[phase8a] building n_train={SCREEN_N_TRAIN} screening split ...", flush=True)
    t0 = time.perf_counter()
    screen = build_screening_split()
    runtime["split_seconds"] = time.perf_counter() - t0

    cmp_ids = screen.split.test_set_quantum["id"].tolist()
    fingerprint = comparison_set_fingerprint(cmp_ids)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"Comparison set fingerprint {fingerprint} != expected {EXPECTED_FINGERPRINT}")

    fg = get_cardio_feature_groups()
    pcfg = PreprocessingConfig()
    qcfg = QuantumConfig()

    print("[phase8a] fitting preprocessing/PCA on the screening training subset (train-only) ...", flush=True)
    t1 = time.perf_counter()
    processed = process_stage(
        screen.train_df, screen.split.test_set_classical, screen.split.test_set_quantum,
        TARGET_COLUMN, fg, pcfg,
    )
    runtime["preprocessing_and_pca_seconds"] = time.perf_counter() - t1

    # ---- 1. Baseline (zz_feature_map, linear) ----
    print("[phase8a] deriving baseline zz_feature_map circuit ...", flush=True)
    baseline_circuit = build_feature_map(
        qcfg.feature_map_name, processed.n_qubits, reps=qcfg.reps, entanglement=qcfg.entanglement,
    )
    print("[phase8a] running BASELINE QSVM ...", flush=True)
    t2 = time.perf_counter()
    baseline_preds, baseline_timing = _run_one_feature_map(
        "baseline", "Baseline QSVM (zz_feature_map, linear)", baseline_circuit, processed, cmp_ids, proc, qcfg, pcfg,
    )
    runtime["baseline_seconds"] = time.perf_counter() - t2
    print(f"[phase8a] baseline done in {runtime['baseline_seconds']:.1f}s, roc_auc={baseline_preds.metrics['roc_auc']:.4f}", flush=True)

    # ---- 2. MI-adaptive (Phase 6/7 mechanism, re-derived at n=2,000) ----
    print("[phase8a] deriving MI-adaptive circuit from TRAINING DATA ONLY (n=2,000) ...", flush=True)
    mi_circuit, mi_report = build_adaptive_feature_map_from_training_data(
        processed.X_train_quantum, processed.n_qubits, qcfg.reps, random_state=qcfg.random_seed,
    )
    print(f"[phase8a] MI-adaptive selected pairs: {mi_report.selected_pairs}", flush=True)
    print("[phase8a] running MI-ADAPTIVE QSVM ...", flush=True)
    t3 = time.perf_counter()
    mi_preds, mi_timing = _run_one_feature_map(
        "mi_adaptive", "MI-adaptive QSVM", mi_circuit, processed, cmp_ids, proc, qcfg, pcfg,
    )
    runtime["mi_adaptive_seconds"] = time.perf_counter() - t3
    print(f"[phase8a] MI-adaptive done in {runtime['mi_adaptive_seconds']:.1f}s, roc_auc={mi_preds.metrics['roc_auc']:.4f}", flush=True)

    # ---- 3. Label-aware (new mechanism) ----
    print("[phase8a] deriving label-aware circuit from TRAINING DATA + TRAINING LABELS ONLY (n=2,000) ...", flush=True)
    la_circuit, la_report = build_label_aware_feature_map_from_training_data(
        processed.X_train_quantum, processed.y_train, processed.n_qubits, qcfg.reps,
    )
    print(f"[phase8a] label-aware selected pairs: {la_report.selected_pairs}", flush=True)
    print("[phase8a] running LABEL-AWARE QSVM ...", flush=True)
    t4 = time.perf_counter()
    la_preds, la_timing = _run_one_feature_map(
        "label_aware", "Label-aware QSVM", la_circuit, processed, cmp_ids, proc, qcfg, pcfg,
    )
    runtime["label_aware_seconds"] = time.perf_counter() - t4
    print(f"[phase8a] label-aware done in {runtime['label_aware_seconds']:.1f}s, roc_auc={la_preds.metrics['roc_auc']:.4f}", flush=True)

    all_preds = [baseline_preds, mi_preds, la_preds]
    for p in all_preds:
        assert list(p.ids) == cmp_ids
        assert np.array_equal(p.y_true, baseline_preds.y_true)

    # ---- paired statistics: each adaptive variant vs baseline ----
    print("[phase8a] computing paired statistics vs baseline ...", flush=True)
    comparisons = {}
    for p in (mi_preds, la_preds):
        dl = delong_test(baseline_preds.y_true, p.y_proba, baseline_preds.y_proba)
        bs_roc = paired_bootstrap_delta(baseline_preds.y_true, p.y_proba, baseline_preds.y_proba, "roc_auc",
                                         n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
        bs_pr = paired_bootstrap_delta(baseline_preds.y_true, p.y_proba, baseline_preds.y_proba, "pr_auc",
                                        n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
        mc = mcnemar_exact(baseline_preds.y_true, p.y_pred, baseline_preds.y_pred)
        comparisons[p.model_key] = {"delong": dl.to_dict(), "bootstrap_roc_auc": bs_roc, "bootstrap_pr_auc": bs_pr, "mcnemar": mc}

    holm = holm_bonferroni([comparisons["mi_adaptive"]["delong"]["p_value"], comparisons["label_aware"]["delong"]["p_value"]])
    for key, h in zip(["mi_adaptive", "label_aware"], holm):
        comparisons[key]["delong_p_holm"] = h["adjusted_p"]
        comparisons[key]["delong_significant_holm"] = h["significant"]

    single_cis = {
        p.model_key: {
            "roc_auc": bootstrap_metric_ci(p.y_true, p.y_proba, "roc_auc", n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED),
            "pr_auc": bootstrap_metric_ci(p.y_true, p.y_proba, "pr_auc", n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED),
        }
        for p in all_preds
    }

    # ---- persist ----
    pred_df = pd.DataFrame({"id": cmp_ids, "y_true": baseline_preds.y_true})
    for p in all_preds:
        pred_df[f"{p.model_key}_proba"] = p.y_proba
        pred_df[f"{p.model_key}_pred"] = p.y_pred
    pred_df.to_csv(RESULTS_ROOT / "predictions.csv", index=False)

    metrics_df = pd.DataFrame([{"Model": p.display_name, "model_key": p.model_key, **p.metrics} for p in all_preds])
    metrics_df.to_csv(RESULTS_ROOT / "metrics.csv", index=False)

    with open(RESULTS_ROOT / "mi_adaptive_map_config.json", "w", encoding="utf-8") as fh:
        json.dump(mi_report.to_dict(), fh, indent=2, default=float)
    with open(RESULTS_ROOT / "label_aware_map_config.json", "w", encoding="utf-8") as fh:
        json.dump(la_report.to_dict(), fh, indent=2, default=float)

    runtime["per_model_timing"] = {"baseline": baseline_timing, "mi_adaptive": mi_timing, "label_aware": la_timing}
    with open(RESULTS_ROOT / "runtime.json", "w", encoding="utf-8") as fh:
        json.dump(runtime, fh, indent=2, default=float)

    statistics = {"comparisons_vs_baseline": comparisons, "single_model_cis": single_cis,
                  "delta_definition": "Variant - Baseline (positive = variant better)"}
    with open(RESULTS_ROOT / "statistics.json", "w", encoding="utf-8") as fh:
        json.dump(statistics, fh, indent=2, default=float)

    summary = {
        "n_train": SCREEN_N_TRAIN,
        "comparison_set": {"n": len(cmp_ids), "fingerprint": fingerprint, "matches_expected": fingerprint == EXPECTED_FINGERPRINT},
        "metrics": {p.model_key: p.metrics for p in all_preds},
        "mi_adaptive_map": mi_report.to_dict(),
        "label_aware_map": la_report.to_dict(),
        "statistics": statistics,
        "runtime": runtime,
    }
    with open(RESULTS_ROOT / "phase8a_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    if make_plots:
        _plot_curves(all_preds, RESULTS_ROOT)

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
    ax.set_title(f"ROC -- Phase 8A screening (n_train={SCREEN_N_TRAIN:,})\nIdentical {n_cmp}-row held-out test set")
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
    ax.set_title(f"Precision-Recall -- Phase 8A screening (n_train={SCREEN_N_TRAIN:,})")
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
    fig.suptitle(f"Confusion matrices -- Phase 8A screening, n_train={SCREEN_N_TRAIN:,}, identical {n_cmp}-row test set")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    result = run_phase8a_screening()
    print(json.dumps({k: v for k, v in result.items() if k != "statistics"}, indent=2, default=float))
