"""Research Service: exposes research roadmap, objective status mapping, and phase summaries."""

from __future__ import annotations


def get_sih_objectives_status() -> list[dict]:
    return [
        {
            "id": "O1",
            "objective": "Design a hybrid quantum-classical machine learning architecture suitable for early disease detection.",
            "status": "IMPLEMENTED",
            "badge": "success",
            "details": "A modular end-to-end pipeline with shared ProcessedDataset fairness contract, fit-on-train-only leakage firewall, classical branch and quantum branch."
        },
        {
            "id": "O2",
            "objective": "Develop quantum-enhanced classification/regression models that can process high-dimensional biomedical data.",
            "status": "PARTIALLY SATISFIED",
            "badge": "warning",
            "details": "High-dimensional biomedical features (13-30 features) are mapped via PCA/SelectKBest to n <= 8 qubits before entering quantum encoding. The quantum circuit processes a reduced representation."
        },
        {
            "id": "O3",
            "objective": "Improve detection accuracy, sensitivity, and specificity compared with classical machine learning baselines.",
            "status": "NOT ACHIEVED (HONEST FINDING)",
            "badge": "danger",
            "details": "Across 16 research phases, adaptive QSVM and VQC matched tuned classical baselines (RBF-SVM, XGBoost) but did not demonstrate a statistically significant predictive advantage (p > 0.05)."
        },
        {
            "id": "O4",
            "objective": "Ensure the platform is scalable, interpretable, and compatible with near-term quantum hardware and simulators.",
            "status": "IMPLEMENTED",
            "badge": "success",
            "details": "Surrogate TreeSHAP explainability implemented; statevector sim and noisy sim supported; real IBM Quantum QPU execution confirmed on ibm_marrakesh (Job ID dag54f8mhr3c73e4m300)."
        },
        {
            "id": "O5",
            "objective": "Incorporate data pre-processing, feature selection, and model explainability modules.",
            "status": "IMPLEMENTED",
            "badge": "success",
            "details": "Schema validation, sentinel mapping, median/mode imputation, StandardScaler, mutual information selection, and TreeSHAP feature attribution."
        },
        {
            "id": "O6",
            "objective": "Benchmark the hybrid approach against classical models in terms of accuracy, computational efficiency, and generalization performance.",
            "status": "EXTENSIVELY IMPLEMENTED",
            "badge": "success",
            "details": "Extensive benchmarking across 16 phases on multiple cohorts, noise sweeps, class imbalance sweeps, and 5-fold outer cross-validation with DeLong significance tests."
        }
    ]


def get_research_phases_summary() -> list[dict]:
    return [
        {
            "phase": 1,
            "title": "Data Preprocessing & Schema Validation",
            "summary": "Implemented schema validation, sentinel conversion, missingness checks, and patient-stratified hold-out split."
        },
        {
            "phase": 2,
            "title": "Classical Baseline Implementation",
            "summary": "Trained Optuna-tuned Logistic Regression, RBF-SVM, and XGBoost baselines."
        },
        {
            "phase": 3,
            "title": "Quantum Simulator Setup",
            "summary": "Configured Qiskit Aer statevector and shot-sampling execution backends with gradient parameter-shift rules."
        },
        {
            "phase": 4,
            "title": "Quantum Feature Map Design",
            "summary": "Evaluated angle encoding, Z-FeatureMap, and Pauli-ZZ feature maps on target qubit budgets."
        },
        {
            "phase": 5,
            "title": "QSVM & VQC Model Suite",
            "summary": "Constructed FidelityQuantumKernel QSVM and data-reuploading Variational Quantum Classifier (VQC)."
        },
        {
            "phase": 6,
            "title": "Cross-Validation & Metric Harness",
            "summary": "Established 5-fold outer cross-validation, PR-AUC headline scoring, and bootstrap confidence intervals."
        },
        {
            "phase": 7,
            "title": "Adaptive Robustness & Class Imbalance",
            "summary": "Evaluated class imbalance ratio sweeps (1:1 to 1:20) and class-weight balancing inside training folds."
        },
        {
            "phase": 8,
            "title": "Label-Aware Feature Maps & VQC Screening",
            "summary": "Tested supervised feature map modifications to optimize state separation for rare positive diseases."
        },
        {
            "phase": 9,
            "title": "Classical Baseline Saturation",
            "summary": "Benchmarked classical RBF-SVM and XGBoost on benchmark tabular datasets (WDBC ROC-AUC = 0.998)."
        },
        {
            "phase": 10,
            "title": "Hybrid QML Feature Layer Integration",
            "summary": "Concatenated 4 quantum-extracted expectation features with classical tabular features."
        },
        {
            "phase": 11,
            "title": "Full-Scale Classical XGBoost Benchmark",
            "summary": "Validated XGBoost performance ceiling on N=66,641 multi-center health cohort (Test ROC-AUC = 0.8567)."
        },
        {
            "phase": 12,
            "title": "Quantum Representation Expressivity",
            "summary": "Probed multi-qubit RZZ entangling circuits vs classical PCA representations (ROC-AUC delta = +0.0001)."
        },
        {
            "phase": 13,
            "title": "Hybrid Residual Quantum Refinement",
            "summary": "Attempted quantum circuit training on XGBoost residual error scores."
        },
        {
            "phase": 14,
            "title": "Predictive Improvement Experiments",
            "summary": "Tested HistGradientBoosting, feature fusion redesigns, and multi-seed stability (No defensible gain over XGBoost)."
        },
        {
            "phase": 15,
            "title": "Quantum-Classical Prediction Stacking",
            "summary": "Performed leakage-safe out-of-fold stacking. Final CV ROC-AUC: XGBoost=0.80119, Q-Stack=0.80120, RFF Control=0.80121. Decision C: No Quantum Advantage."
        },
        {
            "phase": 16,
            "title": "IBM Quantum Hardware Execution",
            "summary": "Executed 4-qubit circuit on IBM QPU (ibm_marrakesh, Job ID dag54f8mhr3c73e4m300). Confirmed hardware portability and depth-17 transpilation with MAD = 0.0312."
        }
    ]
