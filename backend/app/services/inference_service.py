"""Thin service layer: holds the ONE loaded LoadedArtifacts instance for
the process's lifetime (loaded once at app startup, see app/main.py) and
delegates to src.quantum.phase14.inference for all real ML/quantum work.

No model is ever retrained or reloaded per-request -- this module exists
specifically to make that guarantee visible and testable in one place.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from src.quantum.phase14.inference import LoadedArtifacts, load_artifacts, predict_batch, predict_patient_risk, validate_dataset

ARTIFACTS_ROOT = _REPO_ROOT / "models" / "phase14"

_artifacts: LoadedArtifacts | None = None


def startup_load_artifacts() -> None:
    global _artifacts
    if not (ARTIFACTS_ROOT / "manifest.json").is_file():
        raise RuntimeError(
            f"Phase 14 artifacts not found at {ARTIFACTS_ROOT}. Run "
            f"`python -m src.large_dataset.phase14_hybrid_product` first to train and persist them."
        )
    _artifacts = load_artifacts(ARTIFACTS_ROOT)


def is_ready() -> bool:
    return _artifacts is not None


def get_artifacts() -> LoadedArtifacts:
    if _artifacts is None:
        raise RuntimeError("Artifacts not loaded -- startup_load_artifacts() must run first.")
    return _artifacts


def predict_one(patient_dict: dict) -> dict:
    return predict_patient_risk(patient_dict, get_artifacts())


def predict_many(records: list[dict]) -> list[dict]:
    return predict_batch(records, get_artifacts())


def validate_uploaded_dataset(df: pd.DataFrame) -> dict:
    return validate_dataset(df, get_artifacts())


def model_info() -> dict:
    a = get_artifacts()
    return {
        "model_version": a.manifest["model_version"],
        "n_train_full": a.manifest["n_train_full"],
        "test_fingerprint": a.manifest["comparison_set_fingerprint"],
        "decision_threshold": a.manifest["decision_threshold"],
        "feature_columns": a.feature_columns,
        "quantum": a.quantum_config.to_dict(),
    }


def quantum_circuit_info() -> dict:
    a = get_artifacts()
    cfg = a.quantum_config.to_dict()
    cfg["theta"] = a.quantum_theta.tolist()
    cfg["backend"] = "statevector_simulator"
    cfg["future_backend"] = "src.quantum.backends.IBMHardwareBackend (interface defined, not yet wired up)"
    return cfg
