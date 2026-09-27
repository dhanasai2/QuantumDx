"""Quantum data pipeline, feature maps, backends, and fidelity kernel (QuantumDx Phase 4).

This is the ONLY package in the codebase that imports Qiskit -- see
docs/QUANTUM_PIPELINE.md for the isolation rationale.

Public entry points:
    - config.QuantumConfig / config.DEFAULT_CONFIG
    - data_contract.load_quantum_dataset(preprocessing_config) -> QuantumDataset
    - feature_maps.build_feature_map(...) -> QuantumCircuit
    - backends.get_backend(name) -> QuantumBackend
    - kernel.compute_kernel_matrix(...) / compute_kernel_matrix_cached(...)

Run the Phase 4 sanity experiment:
    python -m src.quantum.sanity_experiment
"""
