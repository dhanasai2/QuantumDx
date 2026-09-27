"""Phase 14: the hybrid quantum-classical PRODUCT's serving layer.

This package contains the INFERENCE-time code only -- loading persisted
artifacts and running the real trained pipeline. Training/artifact
production lives in src.large_dataset.phase14_hybrid_product (mirroring
every other phase's single-file training-script convention in this
project); this package is what the FastAPI backend imports.
"""
