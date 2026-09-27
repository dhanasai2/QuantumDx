"""Leakage-safe preprocessing orchestration for one stage of the large-dataset
experiment (Phase 6). Mirrors src.preprocessing.pipeline.run_preprocessing_pipeline's
discipline exactly, but built for the cardio schema and the fixed-test-set /
growing-training-set design (see sampling.py):

    stage training subsample  --fit_transform-->  SharedFeaturePipeline
                                                          |
    FIXED classical test set  ------transform---------- >|--> classical-ready
    FIXED quantum test set    ------transform---------- >|
                                                          v
                                              build_quantum_pipeline
                                                          |
                              (fit on stage TRAIN classical-ready output only)
                                                          v
                                            quantum-ready (train / both test sets)

The SharedFeaturePipeline and build_quantum_pipeline objects are Phase 2's
own, completely unmodified -- only the FeatureGroups (schema.py) and the
data feeding them are new.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.pipeline import FeatureGroups, SharedFeaturePipeline, build_quantum_pipeline
from src.preprocessing.validation import assert_no_missing_values, assert_within_range


@dataclass
class LargeDatasetProcessed:
    """Everything one stage's classical and QSVM experiments need."""

    stage_size: int
    X_train_classical: pd.DataFrame
    X_test_classical: pd.DataFrame          # fixed 2000-row test set, transformed
    X_test_quantum_classical_space: pd.DataFrame  # fixed 200-row set, classical-ready (pre-PCA)
    X_train_quantum: np.ndarray
    X_test_quantum: np.ndarray              # fixed 200-row test set, PCA + range-normalized
    y_train: np.ndarray
    y_test_classical: np.ndarray
    y_test_quantum: np.ndarray
    n_qubits: int
    selected_feature_names: list[str]
    shared_pipeline: SharedFeaturePipeline
    pca_explained_variance_ratio: np.ndarray


def process_stage(
    stage_train_df: pd.DataFrame,
    test_classical_df: pd.DataFrame,
    test_quantum_df: pd.DataFrame,
    target_column: str,
    feature_groups: FeatureGroups,
    config: PreprocessingConfig,
) -> LargeDatasetProcessed:
    """Fit preprocessing on `stage_train_df` ONLY; transform (never fit) both
    fixed test sets. This is called once per stage -- a fresh
    SharedFeaturePipeline and quantum pipeline are fit each time, exactly
    as Phase 3's per-fold CV refits, so a larger stage's preprocessing
    never sees a smaller stage's statistics or vice versa.
    """
    feature_cols = feature_groups.all_columns

    X_train_raw = stage_train_df[feature_cols]
    y_train = stage_train_df[target_column].to_numpy()
    X_test_c_raw = test_classical_df[feature_cols]
    y_test_c = test_classical_df[target_column].to_numpy()
    X_test_q_raw = test_quantum_df[feature_cols]
    y_test_q = test_quantum_df[target_column].to_numpy()

    shared = SharedFeaturePipeline(feature_groups, config)
    X_train_classical = shared.fit_transform(X_train_raw, y_train)
    X_test_classical = shared.transform(X_test_c_raw)
    X_test_quantum_classical_space = shared.transform(X_test_q_raw)

    assert_no_missing_values(X_train_classical, name="X_train_classical")
    assert_no_missing_values(X_test_classical, name="X_test_classical")
    assert_no_missing_values(X_test_quantum_classical_space, name="X_test_quantum_classical_space")

    n_components = min(config.pca_n_components, X_train_classical.shape[1], X_train_classical.shape[0])
    quantum_pipeline = build_quantum_pipeline(n_components, config.quantum_range)
    X_train_quantum = quantum_pipeline.fit_transform(X_train_classical)
    X_test_quantum = quantum_pipeline.transform(X_test_quantum_classical_space)

    assert_within_range(X_train_quantum, *config.quantum_range, name="X_train_quantum")
    assert_within_range(X_test_quantum, *config.quantum_range, name="X_test_quantum")

    return LargeDatasetProcessed(
        stage_size=len(stage_train_df),
        X_train_classical=X_train_classical,
        X_test_classical=X_test_classical,
        X_test_quantum_classical_space=X_test_quantum_classical_space,
        X_train_quantum=np.asarray(X_train_quantum),
        X_test_quantum=np.asarray(X_test_quantum),
        y_train=y_train,
        y_test_classical=y_test_c,
        y_test_quantum=y_test_q,
        n_qubits=n_components,
        selected_feature_names=shared.selected_feature_names,
        shared_pipeline=shared,
        pca_explained_variance_ratio=quantum_pipeline.named_steps["pca"].explained_variance_ratio_,
    )
