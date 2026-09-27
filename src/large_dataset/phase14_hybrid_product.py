"""Phase 14: builds and persists the PRODUCTION artifacts for the hybrid
quantum-classical disease-risk PRODUCT (see docs/PHASE_14_HYBRID_PRODUCT.md).

THIS IS AN ENGINEERING PHASE, NOT A NEW SCIENCE PHASE. Phases 7-13 already
established, with extensive controls and multi-seed robustness checks,
what this project's quantum components do and do not contribute. Phase 14
does not re-litigate those questions -- it REUSES their already-validated
outputs to build a real, working, deployable three-model product:

    Model A (Classical):     Phase 11's XGBoost backbone, refit on the
                              full training pool (cheap, ~3s) with Phase
                              11's own frozen best hyperparameters.
    Model B (Quantum-primary): Phase 8B's VQC architecture/result, REUSED
                              for comparison/analytics -- not re-trained
                              (Phase 8B already established, cheaply and
                              conclusively, that it underperforms; serving
                              it live would add cost for no product value).
    Model C (Hybrid):        Phase 12's quantum embedding circuit (4
                              qubits, 2 layers, trainable RY + trainable
                              RZZ, 8 observables) using its ALREADY-SELECTED,
                              ALREADY-VALIDATED theta (results/large_dataset/
                              phase12_quantum_representation/explainability.json
                              -> quantum_theta_primary_seed) -- re-optimizing
                              theta would cost ~3 hours for zero product
                              benefit, since Phase 12 already ran a 5-seed
                              robustness check and selected this exact
                              value. Only the ONE-TIME forward pass over the
                              full training pool (~16 min, matching Phase
                              12's own measured cost) and a fresh classical
                              head fit are new work here.

ARTIFACT LAYOUT (models/phase14/):
    preprocessing/  the fitted SharedFeaturePipeline + quantum (PCA+range) pipeline
    classical/      Model A's XGBoost
    quantum/        the quantum embedding circuit CONFIG + frozen theta (the
                    circuit itself is rebuilt from config at load time, not
                    pickled -- Qiskit circuits are not reliably picklable
                    across versions; the config + theta fully determine it)
    fusion/         Model C's classical head (XGBoost on [PCA-4, embedding])

The inference-time data flow (also documented in docs/PHASE_14_HYBRID_PRODUCT.md):
    raw patient row -> SharedFeaturePipeline.transform -> quantum_pipeline.transform (PCA-4)
        -> Model A: XGBoost(classical features) -> classical probability
        -> Model C: [PCA-4, QuantumEmbeddingLayer(PCA-4; frozen theta)] -> XGBoost -> hybrid probability
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from xgboost import XGBClassifier

from src.classical.evaluation import compute_classification_metrics
from src.data.inspect_dataset import find_project_root
from src.large_dataset.corrected_comparison import DECISION_THRESHOLD, comparison_set_fingerprint
from src.large_dataset.phase11_full_scale import build_full_scale_split
from src.large_dataset.pipeline import process_stage
from src.large_dataset.schema import TARGET_COLUMN, get_cardio_feature_groups
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.pipeline import build_quantum_pipeline
from src.quantum.quantum_latent_representation import OBSERVABLE_NAMES, QuantumRepresentationConfig, QuantumRepresentationLayer

EXPECTED_FINGERPRINT = "96eac11a8394b87e"
ARTIFACTS_ROOT = find_project_root() / "models" / "phase14"
RESULTS_ROOT = find_project_root() / "results" / "large_dataset" / "phase14_hybrid_product"

PHASE12_EXPLAINABILITY_PATH = find_project_root() / "results" / "large_dataset" / "phase12_quantum_representation" / "explainability.json"
PHASE9_METRICS_PATH = find_project_root() / "results" / "large_dataset" / "phase9_classical_benchmark" / "metrics.csv"
PHASE8B_SUMMARY_PATH = find_project_root() / "results" / "large_dataset" / "phase8b_vqc" / "phase8b_summary.json"

XGB_FROZEN_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)

MODEL_VERSION = "phase14_hybrid_v1"


def _metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    return compute_classification_metrics(y_true, y_pred, y_proba)


def load_frozen_quantum_theta() -> np.ndarray:
    """Reuses Phase 12's already-selected, already-validated quantum
    embedding weights -- NOT re-optimized here (see module docstring)."""
    if not PHASE12_EXPLAINABILITY_PATH.is_file():
        raise FileNotFoundError(
            f"Expected Phase 12 explainability.json at {PHASE12_EXPLAINABILITY_PATH}; "
            f"Phase 14 reuses its already-selected quantum theta. Run Phase 12 first."
        )
    with open(PHASE12_EXPLAINABILITY_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    return np.array(data["quantum_theta_primary_seed"])


def build_and_persist_artifacts(*, make_plots: bool = True) -> dict:
    """Runs Stage 1 (train-only) and produces every artifact the FastAPI
    inference service needs, plus the evaluation report for the model
    card / analytics dashboard."""
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    for sub in ("preprocessing", "classical", "quantum", "fusion"):
        (ARTIFACTS_ROOT / sub).mkdir(exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    runtime: dict = {}

    print("[phase14] rebuilding the full fixed split (Phase 11's own function, unmodified) ...", flush=True)
    t0 = time.perf_counter()
    split = build_full_scale_split()
    runtime["split_seconds"] = time.perf_counter() - t0
    cmp_ids = split.test_set_quantum["id"].tolist()
    fingerprint = comparison_set_fingerprint(cmp_ids)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"Comparison set fingerprint {fingerprint} != expected {EXPECTED_FINGERPRINT}")

    fg = get_cardio_feature_groups()
    pcfg = PreprocessingConfig()

    print("[phase14] fitting preprocessing/PCA on the FULL training pool (train-only) ...", flush=True)
    t1 = time.perf_counter()
    processed = process_stage(split.training_pool, split.test_set_classical, split.test_set_quantum, TARGET_COLUMN, fg, pcfg)
    runtime["preprocessing_seconds"] = time.perf_counter() - t1

    X_train_classical = processed.X_train_classical.to_numpy() if hasattr(processed.X_train_classical, "to_numpy") else processed.X_train_classical
    X_test_classical = processed.X_test_quantum_classical_space
    X_test_classical = X_test_classical.to_numpy() if hasattr(X_test_classical, "to_numpy") else X_test_classical
    X_train_pca, y_train = processed.X_train_quantum, processed.y_train
    X_test_pca, y_test = processed.X_test_quantum, processed.y_test_quantum

    # Standalone quantum_pipeline (PCA+range), refit identically to what
    # process_stage fit internally -- needed as a persistable, reusable
    # object for the inference service (LargeDatasetProcessed does not
    # expose the fitted pipeline object itself).
    quantum_pipeline = build_quantum_pipeline(pcfg.pca_n_components, pcfg.quantum_range)
    quantum_pipeline.fit(processed.X_train_classical)

    # ---- Model A: classical backbone (Phase 11's XGBoost, refit fresh) ----
    print("[phase14] ===== Model A: classical backbone (XGBoost, full training pool) =====", flush=True)
    t2 = time.perf_counter()
    xgb_classical = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1, tree_method="hist", **XGB_FROZEN_PARAMS)
    xgb_classical.fit(X_train_classical, y_train)
    runtime["classical_fit_seconds"] = time.perf_counter() - t2
    proba_a_test = xgb_classical.predict_proba(X_test_classical)[:, 1]

    # ---- Model C: hybrid (Phase 12's quantum embedding, frozen theta + fresh classical head) ----
    print("[phase14] ===== Model C: hybrid (Phase 12 quantum embedding, frozen theta) =====", flush=True)
    theta = load_frozen_quantum_theta()
    q_config = QuantumRepresentationConfig(seed=42)
    q_layer = QuantumRepresentationLayer(q_config)

    print(f"[phase14] computing quantum embedding for the FULL training pool "
          f"(n={len(X_train_pca)}, ONE forward pass at Phase 12's frozen theta) ...", flush=True)
    t3 = time.perf_counter()
    q_train_embedding = q_layer.transform(X_train_pca, theta)
    runtime["quantum_full_batch_forward_seconds"] = time.perf_counter() - t3
    assert np.all(q_train_embedding >= -1.0 - 1e-6) and np.all(q_train_embedding <= 1.0 + 1e-6), \
        "quantum embedding values out of the physically valid [-1, 1] range"
    q_test_embedding = q_layer.transform(X_test_pca, theta)

    Z_train = np.hstack([X_train_pca, q_train_embedding])
    Z_test = np.hstack([X_test_pca, q_test_embedding])

    t4 = time.perf_counter()
    xgb_hybrid = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1, tree_method="hist", **XGB_FROZEN_PARAMS)
    xgb_hybrid.fit(Z_train, y_train)
    runtime["fusion_fit_seconds"] = time.perf_counter() - t4
    proba_c_test = xgb_hybrid.predict_proba(Z_test)[:, 1]

    # ---- Model B: quantum-primary (Phase 8B VQC, reused for reporting only) ----
    model_b_metrics = None
    if PHASE8B_SUMMARY_PATH.is_file():
        with open(PHASE8B_SUMMARY_PATH, encoding="utf-8") as fh:
            phase8b_summary = json.load(fh)
        model_b_metrics = phase8b_summary.get("vqc_metrics")

    metrics_a = _metrics(y_test, proba_a_test)
    metrics_c = _metrics(y_test, proba_c_test)

    print(f"[phase14] Model A (classical) test ROC-AUC={metrics_a['roc_auc']:.4f}; "
          f"Model C (hybrid) test ROC-AUC={metrics_c['roc_auc']:.4f}"
          + (f"; Model B (VQC, reused) test ROC-AUC={model_b_metrics['roc_auc']:.4f}" if model_b_metrics else ""), flush=True)

    # ---- persist artifacts for the inference service ----
    joblib.dump(processed.shared_pipeline, ARTIFACTS_ROOT / "preprocessing" / "shared_pipeline.joblib")
    joblib.dump(quantum_pipeline, ARTIFACTS_ROOT / "preprocessing" / "quantum_pipeline.joblib")
    joblib.dump(xgb_classical, ARTIFACTS_ROOT / "classical" / "xgboost_model.joblib")
    joblib.dump(xgb_hybrid, ARTIFACTS_ROOT / "fusion" / "xgboost_fusion_model.joblib")
    np.save(ARTIFACTS_ROOT / "quantum" / "theta.npy", theta)
    with open(ARTIFACTS_ROOT / "quantum" / "circuit_config.json", "w", encoding="utf-8") as fh:
        json.dump(q_config.to_dict(), fh, indent=2, default=float)
    with open(ARTIFACTS_ROOT / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump({
            "model_version": MODEL_VERSION, "feature_columns": fg.all_columns,
            "n_train_full": len(split.training_pool), "comparison_set_fingerprint": fingerprint,
            "decision_threshold": DECISION_THRESHOLD,
        }, fh, indent=2, default=float)

    # ---- persist evaluation report ----
    pred_df = pd.DataFrame({"id": cmp_ids, "y_true": y_test, "classical_proba": proba_a_test, "hybrid_proba": proba_c_test})
    pred_df.to_csv(RESULTS_ROOT / "predictions.csv", index=False)

    metrics_rows = [
        {"Model": "A: Classical (XGBoost)", "model_key": "classical", **metrics_a},
        {"Model": "C: Hybrid (Quantum embedding + XGBoost)", "model_key": "hybrid", **metrics_c},
    ]
    if model_b_metrics:
        metrics_rows.append({"Model": "B: Quantum-primary (VQC, reused from Phase 8B, n=2,000)", "model_key": "vqc", **model_b_metrics})
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULTS_ROOT / "metrics.csv", index=False)

    feature_importances_a = dict(zip(fg.all_columns, xgb_classical.feature_importances_.tolist()))
    feature_importances_c = dict(zip(fg.all_columns + OBSERVABLE_NAMES, xgb_hybrid.feature_importances_.tolist()))

    model_card = {
        "model_version": MODEL_VERSION,
        "architecture": {
            "model_a_classical": "XGBoost (Phase 11 frozen hyperparameters)",
            "model_b_quantum_primary": "VQC (Phase 8B architecture, reused result -- not served live)",
            "model_c_hybrid": "PCA-4 -> quantum embedding (4 qubits, 2 layers, Phase 12 architecture, frozen theta) "
                               "concatenated with PCA-4 -> XGBoost",
        },
        "training_dataset_size": len(split.training_pool),
        "test_dataset_size": len(cmp_ids),
        "feature_dimensions": {"raw_classical_features": len(fg.all_columns), "pca_dimensions": q_config.n_qubits,
                                "quantum_embedding_dimensions": q_config.output_dim()},
        "n_qubits": q_config.n_qubits, "circuit_depth_layers": q_config.n_layers,
        "backend": "statevector (exact, no shot noise) -- src.quantum.backends.StatevectorBackend",
        "future_backend_path": "src.quantum.backends.IBMHardwareBackend (interface already defined, not yet wired up)",
        "training_method": "Classical: 5-fold CV grid search (Phase 3/11 methodology). "
                            "Quantum embedding weights: COBYLA + train-only CV on a cheap subset (Phase 12), reused verbatim here.",
        "decision_threshold": DECISION_THRESHOLD,
        "test_fingerprint": fingerprint,
        "evaluation_metrics": {"classical": metrics_a, "hybrid": metrics_c, "quantum_primary_reused": model_b_metrics},
        "known_limitations": [
            "Phases 9-13 found no statistically significant improvement from any tested quantum component over the classical XGBoost backbone on this dataset.",
            "The quantum embedding's weights were selected on a 2,000-row subset, not the full training pool, for computational tractability.",
            "The quantum circuit runs on an exact statevector simulator; real QPU execution (shot noise, hardware error) is not yet evaluated.",
            "This is a decision-support research prototype, not a validated clinical diagnostic tool.",
        ],
    }
    with open(RESULTS_ROOT / "model_card.json", "w", encoding="utf-8") as fh:
        json.dump(model_card, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "explainability.json", "w", encoding="utf-8") as fh:
        json.dump({"classical_feature_importances": feature_importances_a, "hybrid_feature_importances": feature_importances_c,
                    "quantum_observable_names": OBSERVABLE_NAMES}, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "runtime.json", "w", encoding="utf-8") as fh:
        json.dump(runtime, fh, indent=2, default=float)

    if make_plots:
        _plot_curves(y_test, {"Model A: Classical": proba_a_test, "Model C: Hybrid": proba_c_test}, RESULTS_ROOT)

    return {"metrics": {"classical": metrics_a, "hybrid": metrics_c, "quantum_primary_reused": model_b_metrics},
            "runtime": runtime, "model_card": model_card}


def _plot_curves(y_true: np.ndarray, probas: dict[str, np.ndarray], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score_safe(y_true, p):.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC -- Phase 14 product models"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "roc_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax.plot(rec, prec, label=label)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR -- Phase 14"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "pr_curves.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, len(probas), figsize=(5 * len(probas), 4.5))
    for ax, (label, p) in zip(np.atleast_1d(axes), probas.items()):
        m = _metrics(y_true, p)
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ConfusionMatrixDisplay(cm, display_labels=["No CVD", "CVD"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(label, fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrices.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, p in probas.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=5, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration -- Phase 14"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "calibration_curve.png", dpi=150); plt.close(fig)


def roc_auc_score_safe(y_true, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, p)


if __name__ == "__main__":
    result = build_and_persist_artifacts()
    print(json.dumps(result["metrics"], indent=2, default=float))
