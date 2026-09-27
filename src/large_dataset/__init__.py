"""Large-scale cardiovascular dataset experiment (post-Phase-5).

This package is entirely ADDITIVE. It does not modify, import destructively
from, or invalidate anything in src/data, src/preprocessing, src/classical,
or src/quantum's existing Cleveland-based behavior. It REUSES the generic
(non-Cleveland-specific) building blocks from those packages:

    - src.preprocessing.pipeline.FeatureGroups, SharedFeaturePipeline,
      build_quantum_pipeline   (schema-agnostic: take column-name lists,
      not hardcoded columns)
    - src.classical.tuning.build_model_pipeline / CV_SCORING /
      MODEL_SELECTION_METRIC   (generic over any FeatureGroups + config)
    - src.classical.evaluation.evaluate_on_test / ModelResult /
      build_comparison_table
    - src.quantum.feature_maps / backends / kernel / qsvm

The only genuinely NEW code here is: the schema for the new dataset
(schema.py), its audit (audit.py), documented cleaning decisions
(cleaning.py), reproducible nested subsampling (sampling.py), and the
per-stage orchestrator (experiment.py).

The Cleveland dataset (Phases 1-5) remains the historical, unmodified
control benchmark. Nothing in this package touches data/raw/ or the Phase
2 locked split (split_id=e471025b07519a64).
"""
