"""The quantum input data contract (Phase 4, Section 2).

Defines exactly what enters the quantum pipeline: X_train_quantum,
X_test_quantum, y_train, y_test, sharing the IDENTICAL patient/sample split
already established by Phase 2 (src.preprocessing.pipeline). No new split
is created here, and nothing in this module fits anything -- it only reads
the output of Phase 2's already-tested, leakage-safe pipeline.

Contract:
    - X_train_quantum / X_test_quantum are the Phase 2 "quantum-ready"
      arrays: PCA-reduced (n_qubits = PreprocessingConfig.pca_n_components,
      currently 4) and range-normalized to [0, pi] -- see
      docs/PREPROCESSING.md Section 8. They are produced by ONE fit on the
      training split only (via run_preprocessing_pipeline), exactly as
      Phase 3's Experiment 3 (the quantum-ready classical benchmark) used.
    - y_train / y_test are the same binary early-risk labels used
      throughout Phases 2 and 3.
    - split_id matches Phase 2/3's reported split_id
      ("e471025b07519a64" for the default config), so this dataset is
      provably the SAME split, not a new one.
    - The test arrays are read-only data: nothing in src/quantum ever
      calls .fit() on anything derived from them (there is nothing to fit
      -- feature maps are fixed, parameter-free circuit templates; see
      docs/QUANTUM_PIPELINE.md).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.inspect_dataset import load_raw_dataset, resolve_dataset_path
from src.preprocessing.config import DEFAULT_CONFIG as DEFAULT_PREPROCESSING_CONFIG
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.pipeline import run_preprocessing_pipeline


@dataclass(frozen=True)
class QuantumDataset:
    """The locked, leakage-safe quantum input contract for one experiment."""

    X_train_quantum: np.ndarray
    X_test_quantum: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    n_qubits: int
    quantum_range: tuple[float, float]
    split_id: str
    preprocess_hash: str
    preprocessing_config: PreprocessingConfig


def _compute_preprocess_hash(config: PreprocessingConfig) -> str:
    """A short, deterministic fingerprint of the preprocessing config, in
    the same spirit as split_id (compute_split_id in Phase 2) -- lets a
    later phase assert two QuantumDataset / classical benchmark runs used
    an identical preprocessing configuration before comparing them.
    """
    canonical = json.dumps(config.to_dict(), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_quantum_dataset(
    preprocessing_config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG,
    raw_df: pd.DataFrame | None = None,
) -> QuantumDataset:
    """Produce the quantum data contract by reusing Phase 2's pipeline.

    This calls src.preprocessing.pipeline.run_preprocessing_pipeline()
    directly -- the SAME function Phase 3 used for its Experiment 3
    (quantum-ready classical benchmark) -- so the split, the PCA fit, and
    the range-normalization are identical to what has already been
    validated and reported in docs/PREPROCESSING.md and
    docs/CLASSICAL_BASELINE.md. No new split is created.

    Args:
        preprocessing_config: Governs the split, feature set, and PCA
            dimensionality. Defaults to the project-wide default (feature
            set "full", pca_n_components=4, seed=42).
        raw_df: Pre-loaded raw DataFrame, for tests that want to avoid
            re-reading the file. Loaded fresh via Phase 1 if omitted.

    Returns:
        A QuantumDataset. X_train_quantum / X_test_quantum are already
        PCA-reduced and range-normalized to preprocessing_config.quantum_range
        (default [0, pi]) -- ready to bind directly to a quantum feature
        map's parameters.
    """
    if raw_df is None:
        raw_df = load_raw_dataset(resolve_dataset_path(None))

    result = run_preprocessing_pipeline(raw_df, preprocessing_config)

    return QuantumDataset(
        X_train_quantum=result.X_train_quantum,
        X_test_quantum=result.X_test_quantum,
        y_train=result.y_train,
        y_test=result.y_test,
        n_qubits=result.X_train_quantum.shape[1],
        quantum_range=preprocessing_config.quantum_range,
        split_id=result.artifacts.split_id,
        preprocess_hash=_compute_preprocess_hash(preprocessing_config),
        preprocessing_config=preprocessing_config,
    )
