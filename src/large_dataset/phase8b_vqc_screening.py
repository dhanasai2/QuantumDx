"""Phase 8B: VQC (Variational Quantum Classifier) SCREENING experiment
(n_train=2,000 only -- see docs/PHASE_8B_VQC_SCREENING.md).

EXPERIMENTAL QUESTION: does a small trainable VQC provide enough evidence
of improvement over the existing baseline QSVM to justify further
investigation? A screening question, not a final performance claim.

WHAT IS REUSED, UNMODIFIED:
    - src.large_dataset.phase8a_label_aware_screening.build_screening_split
      (the IDENTICAL n_train=2,000, seed=42 training subset and fixed
      200-row test set Phase 8A used -- no new sampling code)
    - src.large_dataset.pipeline.process_stage (Phase 2 preprocessing/PCA,
      fit ONCE on this screen's n=2,000 training subset)
    - src.quantum.vqc_classifier (the new VQC, built for this phase)
    - src.large_dataset.corrected_comparison / statistical_robustness
      (the same paired-statistics functions every prior phase used)

WHAT IS REUSED FROM PHASE 8A, NOT RETRAINED: the baseline QSVM's
predictions on the 200-row test set, at this EXACT n_train=2,000/seed=42
setup, are already fully persisted at
results/large_dataset/phase8a_label_aware/predictions.csv
(baseline_proba/baseline_pred columns) -- verified against the same test
fingerprint before use. Retraining an identical baseline QSVM would waste
compute reproducing an already-known result.

TEST-SET DISCIPLINE: preprocessing/PCA is fit on the n=2,000 training
subset only; the VQC is trained on that same training subset only,
via VQCClassifier.fit(X_train, y_train) [sic -- see vqc_classifier.py],
which has no test-data parameter. The 200-row test set is scored exactly
once, after training completes.
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
from src.large_dataset.phase8a_label_aware_screening import SCREEN_N_TRAIN, build_screening_split
from src.large_dataset.pipeline import process_stage
from src.large_dataset.schema import TARGET_COLUMN, get_cardio_feature_groups
from src.large_dataset.statistical_robustness import delong_test
from src.preprocessing.config import PreprocessingConfig
from src.quantum.vqc_classifier import VQCConfig, build_vqc, n_trainable_params

EXPECTED_FINGERPRINT = "96eac11a8394b87e"
PHASE8A_PREDICTIONS_PATH = find_project_root() / "results" / "large_dataset" / "phase8a_label_aware" / "predictions.csv"
RESULTS_ROOT = find_project_root() / "results" / "large_dataset" / "phase8b_vqc"


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


def load_baseline_predictions() -> ModelPredictions:
    """Reuses Phase 8A's baseline QSVM predictions -- SAME n_train=2,000,
    seed=42, and 200-row test set this phase itself uses. Never retrains."""
    if not PHASE8A_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Expected Phase 8A predictions at {PHASE8A_PREDICTIONS_PATH}; the baseline QSVM "
            f"for Phase 8B is REUSED from Phase 8A, never retrained. Run Phase 8A first."
        )
    df = pd.read_csv(PHASE8A_PREDICTIONS_PATH)
    fp = comparison_set_fingerprint(df["id"].tolist())
    if fp != EXPECTED_FINGERPRINT:
        raise ValueError(f"Phase 8A predictions.csv fingerprint {fp} != expected {EXPECTED_FINGERPRINT}")
    y_true = df["y_true"].to_numpy()
    y_proba = df["baseline_proba"].to_numpy()
    y_pred = df["baseline_pred"].to_numpy()
    return ModelPredictions(
        model_key="baseline_qsvm", display_name="Baseline QSVM (n_train=2,000)",
        ids=df["id"].to_numpy(), y_true=y_true, y_proba=y_proba, y_pred=y_pred,
        metrics=_metrics_from_proba(y_true, y_proba),
    )


def run_phase8b_screening(*, vqc_config: VQCConfig | None = None, make_plots: bool = True) -> dict:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()
    runtime: dict = {}
    vqc_config = vqc_config or VQCConfig()

    print("[phase8b] loading baseline QSVM predictions from Phase 8A (no retraining) ...", flush=True)
    baseline = load_baseline_predictions()

    print(f"[phase8b] rebuilding Phase 8A's exact n_train={SCREEN_N_TRAIN} screening split ...", flush=True)
    t0 = time.perf_counter()
    screen = build_screening_split()
    runtime["split_seconds"] = time.perf_counter() - t0

    cmp_ids = screen.split.test_set_quantum["id"].tolist()
    fingerprint = comparison_set_fingerprint(cmp_ids)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"Comparison set fingerprint {fingerprint} != expected {EXPECTED_FINGERPRINT}")
    if list(baseline.ids) != cmp_ids:
        raise ValueError("Baseline predictions' ids are not aligned to the current comparison set.")

    fg = get_cardio_feature_groups()
    pcfg = PreprocessingConfig()

    print("[phase8b] fitting preprocessing/PCA on the screening training subset (train-only) ...", flush=True)
    t1 = time.perf_counter()
    processed = process_stage(
        screen.train_df, screen.split.test_set_classical, screen.split.test_set_quantum,
        TARGET_COLUMN, fg, pcfg,
    )
    runtime["preprocessing_and_pca_seconds"] = time.perf_counter() - t1

    print(f"[phase8b] building VQC (n_qubits={vqc_config.n_qubits}, ansatz_reps={vqc_config.ansatz_reps}, "
          f"shots={vqc_config.shots}, optimizer_maxiter={vqc_config.optimizer_maxiter}, "
          f"n_trainable_params={n_trainable_params(vqc_config)}) ...", flush=True)
    vqc = build_vqc(vqc_config)

    rss0 = proc.memory_info().rss / 1e6
    print(f"[phase8b] training VQC on n_train={len(processed.X_train_quantum)} (TRAIN DATA ONLY) ...", flush=True)
    t2 = time.perf_counter()
    vqc.fit(processed.X_train_quantum, processed.y_train)
    train_seconds = time.perf_counter() - t2
    rss1 = proc.memory_info().rss / 1e6
    print(f"[phase8b] VQC training done in {train_seconds:.1f}s ({train_seconds/60:.1f} min)", flush=True)

    t3 = time.perf_counter()
    proba_raw = vqc.predict_proba(processed.X_test_quantum)
    inference_seconds = time.perf_counter() - t3
    # qiskit-machine-learning's VQC.predict_proba returns (n, 2) columns
    # [P(class 0), P(class 1)] -- take the P(y=1) column.
    y_proba = proba_raw[:, 1] if proba_raw.ndim == 2 else proba_raw
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    vqc_preds = ModelPredictions(
        model_key="vqc", display_name="VQC (variational, trainable)",
        ids=np.array(cmp_ids), y_true=processed.y_test_quantum, y_proba=y_proba, y_pred=y_pred,
        metrics=_metrics_from_proba(processed.y_test_quantum, y_proba),
    )
    assert list(vqc_preds.ids) == list(baseline.ids)
    assert np.array_equal(vqc_preds.y_true, baseline.y_true)

    runtime["vqc_training_seconds"] = train_seconds
    runtime["vqc_inference_seconds"] = inference_seconds
    runtime["process_rss_mb"] = {"before_training": rss0, "after_training": rss1}

    print("[phase8b] computing paired statistics (VQC vs baseline QSVM) ...", flush=True)
    dl = delong_test(baseline.y_true, vqc_preds.y_proba, baseline.y_proba)
    bs_roc = paired_bootstrap_delta(baseline.y_true, vqc_preds.y_proba, baseline.y_proba, "roc_auc",
                                     n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    bs_pr = paired_bootstrap_delta(baseline.y_true, vqc_preds.y_proba, baseline.y_proba, "pr_auc",
                                    n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    mc = mcnemar_exact(baseline.y_true, vqc_preds.y_pred, baseline.y_pred)

    single_cis = {
        p.model_key: {
            "roc_auc": bootstrap_metric_ci(p.y_true, p.y_proba, "roc_auc", n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED),
            "pr_auc": bootstrap_metric_ci(p.y_true, p.y_proba, "pr_auc", n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED),
        }
        for p in (baseline, vqc_preds)
    }

    statistics = {
        "delong": dl.to_dict(), "bootstrap_roc_auc": bs_roc, "bootstrap_pr_auc": bs_pr, "mcnemar": mc,
        "single_model_cis": single_cis,
        "delta_definition": "VQC - Baseline QSVM (positive = VQC better)",
    }

    # ---- decision rule ----
    delta_roc = dl.delta
    significant = dl.p_value < 0.05 and not bs_roc["ci_includes_zero"]
    if delta_roc > 0.02 and significant:
        outcome = "A"
        outcome_text = "VQC shows a meaningful, statistically-supported improvement over baseline QSVM."
    elif abs(delta_roc) <= 0.02 or not significant:
        outcome = "B"
        outcome_text = "VQC is approximately equal to baseline QSVM; evidence is inconclusive."
    else:
        outcome = "C"
        outcome_text = "VQC is clearly worse than baseline QSVM."
    if delta_roc < -0.02 and significant:
        outcome = "C"
        outcome_text = "VQC is clearly worse than baseline QSVM (statistically supported)."

    decision = {
        "outcome": outcome, "outcome_text": outcome_text,
        "delta_roc_auc": delta_roc, "delong_p_value": dl.p_value,
        "bootstrap_ci_includes_zero": bs_roc["ci_includes_zero"],
        "rule": "A: delta>0.02 AND significant. C: delta<-0.02 AND significant. B: otherwise (inconclusive).",
    }

    # ---- persist ----
    pred_df = pd.DataFrame({
        "id": cmp_ids, "y_true": baseline.y_true,
        "baseline_qsvm_proba": baseline.y_proba, "baseline_qsvm_pred": baseline.y_pred,
        "vqc_proba": vqc_preds.y_proba, "vqc_pred": vqc_preds.y_pred,
    })
    pred_df.to_csv(RESULTS_ROOT / "predictions.csv", index=False)

    metrics_df = pd.DataFrame([{"Model": p.display_name, "model_key": p.model_key, **p.metrics} for p in (baseline, vqc_preds)])
    metrics_df.to_csv(RESULTS_ROOT / "metrics.csv", index=False)

    with open(RESULTS_ROOT / "vqc_config.json", "w", encoding="utf-8") as fh:
        json.dump(vqc_config.to_dict(), fh, indent=2, default=float)

    with open(RESULTS_ROOT / "runtime.json", "w", encoding="utf-8") as fh:
        json.dump(runtime, fh, indent=2, default=float)

    with open(RESULTS_ROOT / "statistics.json", "w", encoding="utf-8") as fh:
        json.dump(statistics, fh, indent=2, default=float)

    summary = {
        "n_train": SCREEN_N_TRAIN,
        "comparison_set": {"n": len(cmp_ids), "fingerprint": fingerprint, "matches_expected": fingerprint == EXPECTED_FINGERPRINT},
        "vqc_config": vqc_config.to_dict(),
        "n_trainable_params": n_trainable_params(vqc_config),
        "baseline_metrics": baseline.metrics,
        "vqc_metrics": vqc_preds.metrics,
        "statistics": statistics,
        "runtime": runtime,
        "decision": decision,
    }
    with open(RESULTS_ROOT / "phase8b_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"[phase8b] DECISION: outcome {outcome} -- {outcome_text}", flush=True)

    if make_plots:
        _plot_curves([baseline, vqc_preds], RESULTS_ROOT)

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
    ax.set_title(f"ROC -- Phase 8B VQC screening (n_train={SCREEN_N_TRAIN:,})\nIdentical {n_cmp}-row held-out test set")
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
    ax.set_title(f"Precision-Recall -- Phase 8B VQC screening (n_train={SCREEN_N_TRAIN:,})")
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
    fig.suptitle(f"Confusion matrices -- Phase 8B VQC screening, n_train={SCREEN_N_TRAIN:,}, identical {n_cmp}-row test set")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    result = run_phase8b_screening()
    print(json.dumps({k: v for k, v in result.items() if k != "statistics"}, indent=2, default=float))
