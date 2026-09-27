"""Phase 14 inference service: loads persisted artifacts ONCE and runs
the REAL trained hybrid pipeline for a single patient or a batch.

This module is imported by both:
    - the FastAPI backend (backend/app/services/inference_service.py),
      for live API requests, and
    - src.large_dataset.phase14_hybrid_product's own self-check,

so the API and the training script's own validation always exercise
IDENTICAL inference code -- no drift between "what the training script
verified" and "what the API actually serves."

DATA FLOW (also see docs/PHASE_14_HYBRID_PRODUCT.md):
    raw patient dict
        -> SharedFeaturePipeline.transform            (classical preprocessing)
        -> quantum_pipeline.transform                 (PCA-4, range-normalized)
        -> Model A: XGBoost(classical features)        -> classical probability
        -> Model C: [PCA-4, QuantumEmbeddingLayer(PCA-4; frozen theta)]
                    -> XGBoost                          -> hybrid probability

No clinical meaning is assigned to any individual quantum expectation
value anywhere in this module.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.classical.evaluation import compute_classification_metrics
from src.quantum.quantum_latent_representation import OBSERVABLE_NAMES, QuantumRepresentationConfig, QuantumRepresentationLayer

DECISION_THRESHOLD = 0.5


@dataclass
class LoadedArtifacts:
    manifest: dict
    shared_pipeline: object
    quantum_pipeline: object
    xgb_classical: object
    xgb_hybrid: object
    quantum_theta: np.ndarray
    quantum_config: QuantumRepresentationConfig
    quantum_layer: QuantumRepresentationLayer
    feature_columns: list[str]


def load_artifacts(artifacts_root: Path) -> LoadedArtifacts:
    """Loads every persisted Phase 14 artifact ONCE. The FastAPI app calls
    this exactly once at startup (see backend/app/main.py) -- inference
    requests reuse the same in-memory objects, never reloading or
    retraining per request."""
    with open(artifacts_root / "manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(artifacts_root / "quantum" / "circuit_config.json", encoding="utf-8") as fh:
        qcfg_dict = json.load(fh)
    quantum_config = QuantumRepresentationConfig(n_qubits=qcfg_dict["n_qubits"], n_layers=qcfg_dict["n_layers"], seed=qcfg_dict["seed"])
    quantum_theta = np.load(artifacts_root / "quantum" / "theta.npy")

    return LoadedArtifacts(
        manifest=manifest,
        shared_pipeline=joblib.load(artifacts_root / "preprocessing" / "shared_pipeline.joblib"),
        quantum_pipeline=joblib.load(artifacts_root / "preprocessing" / "quantum_pipeline.joblib"),
        xgb_classical=joblib.load(artifacts_root / "classical" / "xgboost_model.joblib"),
        xgb_hybrid=joblib.load(artifacts_root / "fusion" / "xgboost_fusion_model.joblib"),
        quantum_theta=quantum_theta,
        quantum_config=quantum_config,
        quantum_layer=QuantumRepresentationLayer(quantum_config),
        feature_columns=manifest["feature_columns"],
    )


def _risk_category(p: float) -> str:
    return "HIGH" if p >= 0.66 else ("MEDIUM" if p >= 0.33 else "LOW")


def predict_patient_risk(patient_data: dict, artifacts: LoadedArtifacts) -> dict:
    """Runs the REAL trained pipeline on one patient's raw feature dict.
    Returns the structured response the API/frontend consume.
    """
    t0 = time.perf_counter()
    X_raw = pd.DataFrame([{col: patient_data[col] for col in artifacts.feature_columns}])
    X_classical = artifacts.shared_pipeline.transform(X_raw)
    X_pca = np.asarray(artifacts.quantum_pipeline.transform(X_classical))

    X_classical_arr = X_classical.to_numpy() if hasattr(X_classical, "to_numpy") else X_classical
    p_classical = float(artifacts.xgb_classical.predict_proba(X_classical_arr)[:, 1][0])

    embedding = artifacts.quantum_layer.transform(X_pca, artifacts.quantum_theta)
    Z = np.hstack([X_pca, embedding])
    p_hybrid = float(artifacts.xgb_hybrid.predict_proba(Z)[:, 1][0])

    latency_ms = (time.perf_counter() - t0) * 1000
    # Confidence: distance of the final probability from the decision
    # threshold, rescaled to [0,1] -- a simple, literal, non-clinical measure.
    confidence = float(min(1.0, abs(p_hybrid - DECISION_THRESHOLD) / DECISION_THRESHOLD))

    return {
        "prediction": {
            "probability": round(p_hybrid, 4),
            "class": "CVD" if p_hybrid >= DECISION_THRESHOLD else "No CVD",
            "risk_level": _risk_category(p_hybrid),
            "threshold": DECISION_THRESHOLD,
            "confidence": round(confidence, 4),
        },
        "classical": {
            "model": "XGBoost",
            "probability": round(p_classical, 4),
            "top_features": _top_feature_contributions(artifacts, X_classical_arr),
        },
        "quantum": {
            "qubits": artifacts.quantum_config.n_qubits,
            "circuit_depth": artifacts.quantum_config.n_layers,
            "embedding": [round(float(v), 4) for v in embedding[0]],
            "observables": OBSERVABLE_NAMES,
            "backend": "statevector_simulator",
        },
        "decision_support": {
            "explanation": (
                f"Classical model estimated risk: {p_classical:.2f}. "
                f"Hybrid (classical + quantum) calibrated risk: {p_hybrid:.2f}."
            ),
        },
        "model_info": {
            "version": artifacts.manifest["model_version"],
            "backend": "statevector_simulator",
            "inference_latency_ms": round(latency_ms, 3),
        },
    }


def _top_feature_contributions(artifacts: LoadedArtifacts, X_classical_arr: np.ndarray, top_k: int = 5) -> list[dict]:
    """Exact SHAP contributions for this one prediction, via the XGBoost
    Booster's native pred_contribs (no external `shap` dependency)."""
    from xgboost import DMatrix

    booster = artifacts.xgb_classical.get_booster()
    contribs = booster.predict(DMatrix(X_classical_arr), pred_contribs=True)[0][:-1]  # drop bias term
    order = np.argsort(-np.abs(contribs))[:top_k]
    return [{"feature": artifacts.feature_columns[i], "contribution": round(float(contribs[i]), 4)} for i in order]


def predict_batch(records: list[dict], artifacts: LoadedArtifacts) -> list[dict]:
    return [predict_patient_risk(r, artifacts) for r in records]


def validate_dataset(df: pd.DataFrame, artifacts: LoadedArtifacts) -> dict:
    """Reports the same data-quality summary the frontend's "Data Quality"
    step displays -- never used to alter model behavior, purely descriptive."""
    missing = df.isna().sum()
    report = {
        "n_records": int(len(df)),
        "n_features_expected": len(artifacts.feature_columns),
        "missing_required_columns": [c for c in artifacts.feature_columns if c not in df.columns],
        "missing_value_counts": {c: int(missing.get(c, 0)) for c in artifacts.feature_columns if c in df.columns},
        "dtypes": {c: str(df[c].dtype) for c in df.columns if c in artifacts.feature_columns},
    }
    report["valid"] = len(report["missing_required_columns"]) == 0
    return report
