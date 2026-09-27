"""Per-stage classical + QSVM experiment runner (Phase 6).

Reuses, unmodified:
    - src.classical.models.get_available_models         (4-model registry)
    - src.classical.tuning.run_grid_search               (per-fold-refit CV,
      exactly Phase 3's methodology -- classical preprocessing is refit
      inside every CV fold on the STAGE's raw training subsample)
    - src.classical.evaluation.evaluate_on_test / ModelResult
    - src.quantum.feature_maps.build_feature_map
    - src.quantum.backends.get_backend
    - src.quantum.kernel.compute_statevectors / kernel_matrix_from_statevectors
    - src.quantum.qsvm.run_qsvm_grid_search

METHODOLOGY NOTE (documented, matching Phase 5's own disclosed choice):
the classical branch refits its ENTIRE preprocessing chain inside every CV
fold (via run_grid_search's Pipeline), exactly as Phase 3 did. The QSVM
branch's quantum-ready representation is fit ONCE per stage (via
large_dataset.pipeline.process_stage) and then CV-tuned by slicing the one
resulting precomputed kernel -- this is the identical, already-justified
simplification Phase 5 used and documented (docs/QUANTUM_QSVM.md Section
4.3): a kernel value K(x_i,x_j) is a fixed function with no fitted state,
so slicing is mathematically exact for the KERNEL step; only the upstream
PCA fit is shared across folds, a disclosed, bounded simplification, not a
leakage bug.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.classical.evaluation import ModelResult, evaluate_on_test
from src.classical.models import get_available_models
from src.classical.tuning import run_grid_search
from src.large_dataset.pipeline import LargeDatasetProcessed, process_stage
from src.large_dataset.sampling import LargeDatasetSplit
from src.large_dataset.schema import TARGET_COLUMN
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.pipeline import FeatureGroups
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


@dataclass
class StageResult:
    stage_size: int
    classical_results: list[ModelResult]
    qsvm_result: ModelResult | None
    kernel_timing: dict = field(default_factory=dict)
    kernel_diagnostics_train: dict | None = None
    processed: LargeDatasetProcessed | None = None


def run_stage_classical(
    stage_size: int,
    stage_train_df: pd.DataFrame,
    split: LargeDatasetSplit,
    feature_groups: FeatureGroups,
    preprocessing_config: PreprocessingConfig,
) -> list[ModelResult]:
    """Run all 4 classical models for one stage. Preprocessing is refit
    inside every CV fold via run_grid_search -- the RAW (unprocessed)
    stage training subsample is passed in, never a pre-fit representation.
    """
    feature_cols = feature_groups.all_columns
    X_train_raw = stage_train_df[feature_cols]
    y_train = stage_train_df[TARGET_COLUMN].to_numpy()
    X_test_raw = split.test_set_classical[feature_cols]
    y_test = split.test_set_classical[TARGET_COLUMN].to_numpy()

    models = get_available_models(preprocessing_config.random_seed)
    results: list[ModelResult] = []
    for key, spec in models.items():
        search = run_grid_search(
            spec, "classical", feature_groups, preprocessing_config,
            X_train_raw, y_train, cv_folds=CV_FOLDS, cv_shuffle=CV_SHUFFLE,
            random_seed=preprocessing_config.random_seed,
        )
        result = evaluate_on_test(
            search, f"large_dataset_stage_{stage_size}", key, spec.display_name,
            "classical", X_test_raw, y_test, n_train=len(X_train_raw),
            random_seed=preprocessing_config.random_seed,
        )
        results.append(result)
    return results


def run_stage_qsvm(
    stage_size: int,
    stage_train_df: pd.DataFrame,
    split: LargeDatasetSplit,
    feature_groups: FeatureGroups,
    preprocessing_config: PreprocessingConfig,
    quantum_config: QuantumConfig,
    *,
    C_grid: list[float] | None = None,
) -> tuple[ModelResult, dict, dict, LargeDatasetProcessed]:
    """Run the QSVM for one stage. Returns (result, kernel_timing,
    kernel_diagnostics_dict, processed_data) -- the caller is expected to
    inspect kernel_timing BEFORE requesting a larger stage (Section 7/12:
    "measure first").
    """
    processed = process_stage(
        stage_train_df, split.test_set_classical, split.test_set_quantum,
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
    t1 = time.perf_counter()
    test_svs = compute_statevectors(processed.X_test_quantum, feature_map, backend)
    t2 = time.perf_counter()
    # Uses the vectorized kernel assembly (implementation/performance
    # optimization only -- verified numerically equivalent to the Phase 4/5
    # reference loop-based implementation to ~1e-15 absolute / ~1e-13
    # relative difference on Stage A data; see docs/LARGE_DATASET.md,
    # "Kernel Optimization" section). The reference implementation
    # (kernel_matrix_from_statevectors) is untouched and remains what
    # Phase 4/5's own code paths use.
    K_train_train = kernel_matrix_from_statevectors_vectorized(train_svs, train_svs, symmetric=True)
    t3 = time.perf_counter()
    K_test_train = kernel_matrix_from_statevectors_vectorized(test_svs, train_svs, symmetric=False)
    t4 = time.perf_counter()

    diag = kernel_diagnostics(K_train_train, symmetric=True)

    kernel_timing = {
        "n_train": len(processed.X_train_quantum),
        "n_test": len(processed.X_test_quantum),
        "train_statevectors_seconds": t1 - t0,
        "test_statevectors_seconds": t2 - t1,
        "K_train_train_assembly_seconds": t3 - t2,
        "K_test_train_assembly_seconds": t4 - t3,
        "total_kernel_seconds": t4 - t0,
        "K_train_train_memory_mb": K_train_train.nbytes / 1e6,
        "K_train_train_pairs": len(processed.X_train_quantum) * (len(processed.X_train_quantum) - 1) // 2,
    }

    t5 = time.perf_counter()
    search = run_qsvm_grid_search(
        K_train_train, processed.y_train, C_grid=C_grid,
        cv_folds=CV_FOLDS, cv_shuffle=CV_SHUFFLE,
        random_seed=preprocessing_config.random_seed,
    )
    t6 = time.perf_counter()
    kernel_timing["cv_tuning_seconds"] = t6 - t5

    result = evaluate_on_test(
        search, f"large_dataset_stage_{stage_size}", "qsvm", "QSVM (fidelity kernel)",
        "quantum_ready", K_test_train, processed.y_test_quantum,
        n_train=len(processed.X_train_quantum), random_seed=preprocessing_config.random_seed,
    )

    return result, kernel_timing, diag.to_dict(), processed
