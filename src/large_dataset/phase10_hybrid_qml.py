"""Phase 10: hybrid quantum-classical feature layer + Logistic Regression
head (see docs/PHASE_10_HYBRID_QML.md).

HYPOTHESIS: does a trainable quantum feature transformation provide
measurable predictive value when added to a strong classical Logistic
Regression model? NOT a quantum-vs-classical experiment -- the quantum
circuit is a feature layer feeding a classical prediction head.

WHY THIS EXPERIMENT NOW: Phase 8B's standalone VQC was stopped (clearly,
significantly worse than baseline QSVM). Phase 9 established Logistic
Regression as the strongest model overall and found no evidence of
quantum advantage in any tested QSVM variant. Neither result rules out a
genuinely different architecture -- a quantum REPRESENTATION layer
feeding a classical head, rather than a quantum circuit acting as the
classifier itself -- which is what this phase tests, as a new,
separately-justified experiment (not a continuation of either failed
direction).

MODELS COMPARED (none retrained except the new hybrid):
    A. Classical reference: Phase 9's Logistic Regression, REUSED from
       results/large_dataset/phase9_classical_benchmark/predictions.csv
       (logistic_regression_proba column) -- not retrained.
    B. Quantum reference: Phase 9's baseline QSVM and MI-adaptive QSVM,
       REUSED from the same file (baseline_proba / mi_adaptive_proba) --
       not retrained.
    C. Proposed hybrid: src.quantum.hybrid_quantum_features.HybridQuantumFeatureLayer,
       trained fresh (the only new training in this phase) on the
       IDENTICAL n_train=2,000/seed=42 screening subset.
    D. Non-quantum control (ablation): Logistic Regression on PCA-4 +
       4 FIXED random Gaussian features (untrained, same seed) -- isolates
       whether any hybrid improvement is due to the TRAINED quantum
       transform specifically, or merely from having 4 extra input
       dimensions regardless of their content.

WHAT IS REUSED, UNMODIFIED: build_screening_split (Phase 8A), process_stage
(Phase 2 preprocessing/PCA), and the paired-statistics functions every
prior phase used (corrected_comparison.py / statistical_robustness.py).

TEST-SET DISCIPLINE: the hybrid layer's weights are selected via
train-only cross-validation (see hybrid_quantum_features.py); the 200-row
test set is scored exactly once per model, after all training/selection
is frozen.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve

from src.classical.evaluation import compute_classification_metrics
from src.data.inspect_dataset import find_project_root
from src.large_dataset.corrected_comparison import (
    DECISION_THRESHOLD,
    comparison_set_fingerprint,
    mcnemar_exact,
    paired_bootstrap_delta,
)
from src.large_dataset.phase8a_label_aware_screening import SCREEN_N_TRAIN, build_screening_split
from src.large_dataset.pipeline import process_stage
from src.large_dataset.schema import TARGET_COLUMN, get_cardio_feature_groups
from src.large_dataset.statistical_robustness import delong_test, holm_bonferroni
from src.preprocessing.config import PreprocessingConfig
from src.quantum.hybrid_quantum_features import HybridConfig, HybridQuantumFeatureLayer

EXPECTED_FINGERPRINT = "96eac11a8394b87e"
PHASE9_PREDICTIONS_PATH = find_project_root() / "results" / "large_dataset" / "phase9_classical_benchmark" / "predictions.csv"
RESULTS_ROOT = find_project_root() / "results" / "large_dataset" / "phase10_hybrid_qml"


def _metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    return compute_classification_metrics(y_true, y_pred, y_proba)


def load_phase9_references() -> pd.DataFrame:
    """Reuses Phase 9's Logistic Regression and QSVM predictions -- never
    retrains either."""
    if not PHASE9_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Expected Phase 9 predictions at {PHASE9_PREDICTIONS_PATH}; Phase 10 reuses "
            f"the classical/quantum reference models from there. Run Phase 9 first."
        )
    df = pd.read_csv(PHASE9_PREDICTIONS_PATH)
    fp = comparison_set_fingerprint(df["id"].tolist())
    if fp != EXPECTED_FINGERPRINT:
        raise ValueError(f"Phase 9 predictions.csv fingerprint {fp} != expected {EXPECTED_FINGERPRINT}")
    return df


def run_phase10(*, hybrid_config: HybridConfig | None = None, make_plots: bool = True) -> dict:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    runtime: dict = {}
    hybrid_config = hybrid_config or HybridConfig()

    print("[phase10] loading Phase 9 reference predictions (LR + QSVM, no retraining) ...", flush=True)
    ref_df = load_phase9_references()

    print(f"[phase10] rebuilding the exact n_train={SCREEN_N_TRAIN} screening split ...", flush=True)
    t0 = time.perf_counter()
    screen = build_screening_split()
    runtime["split_seconds"] = time.perf_counter() - t0

    cmp_ids = screen.split.test_set_quantum["id"].tolist()
    fingerprint = comparison_set_fingerprint(cmp_ids)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"Comparison set fingerprint {fingerprint} != expected {EXPECTED_FINGERPRINT}")
    if ref_df["id"].tolist() != cmp_ids:
        raise ValueError("Phase 9 predictions' ids are not aligned to the current comparison set.")

    fg = get_cardio_feature_groups()
    pcfg = PreprocessingConfig()

    print("[phase10] fitting preprocessing/PCA on the screening training subset (train-only) ...", flush=True)
    t1 = time.perf_counter()
    processed = process_stage(
        screen.train_df, screen.split.test_set_classical, screen.split.test_set_quantum,
        TARGET_COLUMN, fg, pcfg,
    )
    runtime["preprocessing_and_pca_seconds"] = time.perf_counter() - t1
    X_train, y_train = processed.X_train_quantum, processed.y_train
    X_test, y_test = processed.X_test_quantum, processed.y_test_quantum

    y_true = ref_df["y_true"].to_numpy()
    assert np.array_equal(y_test, y_true), "test labels mismatch between this phase and Phase 9's reference file"

    all_models: dict[str, dict] = {}

    # ---- Model A: classical reference (reused) ----
    all_models["classical_lr"] = {
        "display_name": "Model A: Classical Logistic Regression (Phase 9, reused)",
        "y_proba": ref_df["logistic_regression_proba"].to_numpy(),
    }
    # ---- Model B: quantum reference(s) (reused) ----
    all_models["baseline_qsvm"] = {
        "display_name": "Model B: Baseline QSVM (Phase 9, reused)",
        "y_proba": ref_df["baseline_proba"].to_numpy(),
    }
    all_models["mi_adaptive_qsvm"] = {
        "display_name": "Model B: MI-adaptive QSVM (Phase 9, reused)",
        "y_proba": ref_df["mi_adaptive_proba"].to_numpy(),
    }

    # ---- Model D: non-quantum control (ablation) -- LR + 4 fixed random features ----
    print("[phase10] fitting Model D (non-quantum control: PCA-4 + 4 fixed random features) ...", flush=True)
    rng = np.random.RandomState(hybrid_config.seed)
    random_train = rng.normal(size=(len(X_train), hybrid_config.n_qubits))
    random_test = np.random.RandomState(hybrid_config.seed + 1).normal(size=(len(X_test), hybrid_config.n_qubits))
    lr_control = LogisticRegression(C=hybrid_config.lr_C, max_iter=1000, random_state=hybrid_config.seed)
    lr_control.fit(np.hstack([X_train, random_train]), y_train)
    control_proba = lr_control.predict_proba(np.hstack([X_test, random_test]))[:, 1]
    all_models["control_random_features"] = {
        "display_name": "Model D: Control (PCA-4 + 4 fixed random features)",
        "y_proba": control_proba,
    }

    # ---- Model C: the proposed hybrid (the only NEW training this phase) ----
    print(f"[phase10] training Model C (hybrid quantum feature layer + LR head), "
          f"n_train={len(X_train)}, n_trainable_params={hybrid_config.n_trainable_params()}, "
          f"optimizer_maxiter={hybrid_config.optimizer_maxiter} ...", flush=True)
    t2 = time.perf_counter()
    hybrid = HybridQuantumFeatureLayer(hybrid_config)
    hybrid.fit(X_train, y_train)
    hybrid_train_seconds = time.perf_counter() - t2
    runtime["hybrid_training_seconds"] = hybrid_train_seconds
    print(f"[phase10] hybrid training done in {hybrid_train_seconds:.1f}s "
          f"({hybrid.fit_result_.n_function_evaluations} evals, final train CV ROC-AUC={hybrid.fit_result_.final_cv_score:.4f})", flush=True)

    t3 = time.perf_counter()
    hybrid_proba = hybrid.predict_proba(X_test)[:, 1]
    runtime["hybrid_inference_seconds"] = time.perf_counter() - t3
    all_models["hybrid"] = {
        "display_name": "Model C: Hybrid (PCA-4 + trained quantum features -> LR)",
        "y_proba": hybrid_proba,
    }

    for m in all_models.values():
        m["y_true"] = y_true
        m["y_pred"] = (m["y_proba"] >= DECISION_THRESHOLD).astype(int)
        m["metrics"] = _metrics(y_true, m["y_proba"])

    # ---- statistics: hybrid vs classical LR, hybrid vs each quantum reference ----
    print("[phase10] computing paired statistics ...", flush=True)

    def _pair(key_a: str, key_b: str) -> dict:
        pa, pb = all_models[key_a]["y_proba"], all_models[key_b]["y_proba"]
        preda, predb = all_models[key_a]["y_pred"], all_models[key_b]["y_pred"]
        dl = delong_test(y_true, pa, pb)
        bs = paired_bootstrap_delta(y_true, pa, pb, "roc_auc", n_resamples=2000, seed=42)
        bs_pr = paired_bootstrap_delta(y_true, pa, pb, "pr_auc", n_resamples=2000, seed=42)
        mc = mcnemar_exact(y_true, preda, predb)
        return {"a": key_a, "b": key_b, "delong": dl.to_dict(), "bootstrap_roc_auc": bs, "bootstrap_pr_auc": bs_pr, "mcnemar": mc}

    comparisons = {
        "hybrid_vs_classical_lr": _pair("hybrid", "classical_lr"),
        "hybrid_vs_baseline_qsvm": _pair("hybrid", "baseline_qsvm"),
        "hybrid_vs_mi_adaptive_qsvm": _pair("hybrid", "mi_adaptive_qsvm"),
        "hybrid_vs_control": _pair("hybrid", "control_random_features"),
    }
    raw_p = [c["delong"]["p_value"] for c in comparisons.values()]
    holm = holm_bonferroni(raw_p)
    for (name, c), h in zip(comparisons.items(), holm):
        c["delong_p_holm"] = h["adjusted_p"]
        c["delong_significant_holm"] = h["significant"]

    # ---- decision rule (Success Criteria A/B/C/D) ----
    primary = comparisons["hybrid_vs_classical_lr"]
    delta = primary["delong"]["delta"]
    significant = primary["delong_significant_holm"] and not primary["bootstrap_roc_auc"]["ci_includes_zero"]
    if delta > 0 and significant:
        outcome, outcome_text = "A", "Hybrid provides a statistically supported improvement over classical Logistic Regression."
    elif delta > 0 and not significant:
        outcome, outcome_text = "B", "Hybrid shows numerical improvement but statistical evidence is insufficient."
    else:
        outcome, outcome_text = "C", "Hybrid does not improve over the classical model."

    decision = {
        "outcome": outcome, "outcome_text": outcome_text,
        "delta_roc_auc_hybrid_minus_classical_lr": delta,
        "delong_p_holm": primary["delong_p_holm"],
        "bootstrap_ci_includes_zero": primary["bootstrap_roc_auc"]["ci_includes_zero"],
    }

    # ---- explainability: final LR coefficients ----
    feature_report = hybrid.feature_report()
    coefficients = dict(zip(feature_report.feature_names, hybrid.lr_.coef_[0].tolist()))
    explainability = {
        "feature_report": feature_report.to_dict(),
        "logistic_regression_coefficients": coefficients,
        "intercept": float(hybrid.lr_.intercept_[0]),
        "selected_quantum_weights_theta": hybrid.fit_result_.theta.tolist(),
    }

    # ---- persist ----
    pred_df = pd.DataFrame({"id": cmp_ids, "y_true": y_true})
    for key, m in all_models.items():
        pred_df[f"{key}_proba"] = m["y_proba"]
        pred_df[f"{key}_pred"] = m["y_pred"]
    pred_df.to_csv(RESULTS_ROOT / "predictions.csv", index=False)

    metrics_rows = [{"Model": m["display_name"], "model_key": key, **m["metrics"]} for key, m in all_models.items()]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULTS_ROOT / "metrics.csv", index=False)

    comparison_table = metrics_df[["Model", "roc_auc", "pr_auc", "sensitivity", "specificity", "accuracy", "f1"]].copy()
    comparison_table.columns = ["Model", "ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "Accuracy", "F1"]
    comparison_table.to_csv(RESULTS_ROOT / "comparison_table.csv", index=False)

    with open(RESULTS_ROOT / "hybrid_config.json", "w", encoding="utf-8") as fh:
        json.dump(hybrid_config.to_dict(), fh, indent=2, default=float)
    with open(RESULTS_ROOT / "runtime.json", "w", encoding="utf-8") as fh:
        json.dump(runtime, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "statistics.json", "w", encoding="utf-8") as fh:
        json.dump(comparisons, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "explainability.json", "w", encoding="utf-8") as fh:
        json.dump(explainability, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "quantum_feature_outputs.json", "w", encoding="utf-8") as fh:
        json.dump({
            "test_set_quantum_features": hybrid.quantum_features(X_test, hybrid.fit_result_.theta).tolist(),
            "feature_names": [n for n in feature_report.feature_names if n.startswith("quantum_")],
        }, fh, indent=2, default=float)

    summary = {
        "n_train": SCREEN_N_TRAIN,
        "comparison_set": {"n": len(cmp_ids), "fingerprint": fingerprint, "matches_expected": fingerprint == EXPECTED_FINGERPRINT},
        "hybrid_config": hybrid_config.to_dict(),
        "metrics": {key: m["metrics"] for key, m in all_models.items()},
        "statistics": comparisons,
        "runtime": runtime,
        "decision": decision,
        "explainability": explainability,
    }
    with open(RESULTS_ROOT / "phase10_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"[phase10] DECISION: outcome {outcome} -- {outcome_text}", flush=True)

    if make_plots:
        _plot_curves(all_models, RESULTS_ROOT)

    return summary


def _plot_curves(all_models: dict, out_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    items = list(all_models.items())
    n_cmp = len(items[0][1]["y_true"])

    fig, ax = plt.subplots(figsize=(8, 7))
    for key, m in items:
        fpr, tpr, _ = roc_curve(m["y_true"], m["y_proba"])
        ax.plot(fpr, tpr, label=f"{m['display_name']} (AUC={m['metrics']['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC -- Phase 10 hybrid QML\nIdentical {n_cmp}-row held-out test set")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for key, m in items:
        prec, rec, _ = precision_recall_curve(m["y_true"], m["y_proba"])
        ax.plot(rec, prec, label=f"{m['display_name']} (AP={m['metrics']['pr_auc']:.4f})")
    base = float(np.mean(items[0][1]["y_true"]))
    ax.axhline(base, ls="--", color="grey", label=f"Chance ({base:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall -- Phase 10 hybrid QML")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=150)
    plt.close(fig)

    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5))
    for ax, (key, m) in zip(np.atleast_1d(axes), items):
        mm = m["metrics"]
        cm = np.array([[mm["tn"], mm["fp"]], [mm["fn"], mm["tp"]]])
        ConfusionMatrixDisplay(cm, display_labels=["No CVD", "CVD"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(m["display_name"], fontsize=7)
    fig.suptitle(f"Confusion matrices -- Phase 10, identical {n_cmp}-row test set")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    result = run_phase10()
    print(json.dumps({k: v for k, v in result.items() if k not in ("statistics", "explainability")}, indent=2, default=float))
