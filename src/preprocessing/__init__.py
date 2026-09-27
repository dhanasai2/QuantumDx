"""Leakage-safe preprocessing and feature pipeline (QuantumDx Phase 2).

Public entry points:
    - config.PreprocessingConfig / config.DEFAULT_CONFIG
    - pipeline.run_preprocessing_pipeline(df, config) -> PreprocessedData
"""

from src.preprocessing.config import DEFAULT_CONFIG, PreprocessingConfig
from src.preprocessing.pipeline import (
    PreprocessedData,
    PreprocessingArtifacts,
    create_binary_target,
    run_preprocessing_pipeline,
    split_dataset,
)

__all__ = [
    "DEFAULT_CONFIG",
    "PreprocessingConfig",
    "PreprocessedData",
    "PreprocessingArtifacts",
    "create_binary_target",
    "run_preprocessing_pipeline",
    "split_dataset",
]
