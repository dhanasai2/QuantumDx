"""Phase 5 orchestrator: the first full-scale QSVM experiment.

Reuses, without modification or duplication:
    - Phase 1/2 (src.data, src.preprocessing): raw loading, the LOCKED
      242/61 split (split_id e471025b07519a64), the quantum-ready
      (PCA + range-normalized) feature representation.
    - Phase 4 (src.quantum.data_contract/feature_maps/backends/kernel):
      the quantum data contract, feature map construction, the
      StatevectorBackend, and the fidelity-kernel primitives.
    - Phase 3 (src.classical.tuning/evaluation): the multi-metric CV
      scoring dict, the ROC-AUC model-selection criterion, and the
      single-touch final-test-evaluation function (evaluate_on_test),
      applied here to a precomputed-kernel SVC instead of a classical
      Pipeline -- both are just fitted sklearn estimators to that code.

Pipeline exercised:

    QuantumDataset (locked split, reused unchanged)
        -> feature map (Phase 4 default: zz_feature_map, 4 qubits, reps=2, linear)
        -> statevectors for all 242 train + 61 test rows, computed ONCE each
        -> K_train_train (242,242), K_test_train (61,242)
            -- assembled from the SAME cached statevectors; no row is
               embedded twice
        -> disk cache (restartable: skips recomputation if cache files
           from an identical configuration already exist)
        -> leakage-safe C-tuning via GridSearchCV on K_train_train only
        -> ONE evaluation on the locked test set via K_test_train
        -> a descriptive (non-statistical) comparison against Phase 3's
           already-saved Experiment 3 classical results

Run:
    python -m src.quantum.qsvm_experiment
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
import scipy
import sklearn

from src.classical.evaluation import (
    ModelResult,
    build_comparison_table,
    build_cv_summary_table,
    evaluate_on_test,
)
from src.classical.tuning import CV_SCORING, MODEL_SELECTION_METRIC
from src.data.inspect_dataset import find_project_root
from src.preprocessing.config import DEFAULT_CONFIG as DEFAULT_PREPROCESSING_CONFIG
from src.quantum.backends import get_backend
from src.quantum.config import DEFAULT_CONFIG as DEFAULT_QUANTUM_CONFIG
from src.quantum.config import QuantumConfig
from src.quantum.data_contract import load_quantum_dataset
from src.quantum.feature_maps import build_feature_map, describe_feature_map
from src.quantum.kernel import compute_statevectors, kernel_diagnostics, kernel_matrix_from_statevectors
from src.quantum.qsvm import DEFAULT_C_GRID, run_qsvm_grid_search

PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results" / "quantum" / "phase5"
KERNEL_CACHE_DIR = RESULTS_DIR / "kernels"
FIGURES_DIR = RESULTS_DIR / "figures"
PHASE3_EXP3_DIR = PROJECT_ROOT / "results" / "classical" / "experiment3_full_quantum_ready"

CV_FOLDS = 5
CV_SHUFFLE = True

EXPERIMENT_NAME = "phase5_qsvm_full_scale"


# --------------------------------------------------------------------------
# Kernel computation, with a simple, honest restartable disk cache
# --------------------------------------------------------------------------


def _kernel_config_id(feature_map_name: str, n_qubits: int, reps: int, entanglement: str, split_id: str) -> str:
    """A short, human-readable (not hashed) cache tag -- deliberately
    readable rather than opaque, since Phase 5's full kernel computation
    costs well under a second (see docs/QUANTUM_QSVM.md Section 7); the
    cache exists for convenience and restartability, not because
    recomputation is expensive.
    """
    return f"{feature_map_name}_q{n_qubits}_r{reps}_{entanglement}_{split_id}"


def compute_or_load_kernels(
    dataset, quantum_config: QuantumConfig, cache_dir: Path
) -> tuple[np.ndarray, np.ndarray, dict, bool]:
    """Compute K_train_train (242,242) and K_test_train (61,242), or load
    them from disk if a matching cache already exists.

    Efficiency (Phase 5 Section E): train and test statevectors are each
    computed EXACTLY ONCE via compute_statevectors(), then reused to
    assemble BOTH matrices -- Phase 4's compute_kernel_matrix_cached() is
    deliberately NOT used here because calling it once per matrix would
    recompute the 242 training statevectors a second time for
    K_test_train. See docs/QUANTUM_QSVM.md Section 7 for the measured
    cost this avoids (small, but avoidable, so avoided).

    Restartability: if BOTH cache files for this exact configuration
    already exist, they are loaded and returned immediately -- an
    interrupted run does not repeat the (sub-second, but non-zero)
    kernel computation.

    Returns:
        (K_train_train, K_test_train, timing_dict, cache_hit)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = _kernel_config_id(
        quantum_config.feature_map_name, dataset.n_qubits, quantum_config.reps,
        quantum_config.entanglement, dataset.split_id,
    )
    train_path = cache_dir / f"K_train_train_{tag}.npy"
    test_path = cache_dir / f"K_test_train_{tag}.npy"

    if train_path.is_file() and test_path.is_file():
        K_train_train = np.load(train_path)
        K_test_train = np.load(test_path)
        return K_train_train, K_test_train, {"cache_hit": True}, True

    feature_map = build_feature_map(
        quantum_config.feature_map_name, dataset.n_qubits,
        reps=quantum_config.reps, entanglement=quantum_config.entanglement, paulis=quantum_config.paulis,
    )
    backend = get_backend(quantum_config.backend_name)

    t0 = time.perf_counter()
    train_statevectors = compute_statevectors(dataset.X_train_quantum, feature_map, backend)
    t1 = time.perf_counter()
    test_statevectors = compute_statevectors(dataset.X_test_quantum, feature_map, backend)
    t2 = time.perf_counter()

    K_train_train = kernel_matrix_from_statevectors(train_statevectors, train_statevectors, symmetric=True)
    t3 = time.perf_counter()
    K_test_train = kernel_matrix_from_statevectors(test_statevectors, train_statevectors, symmetric=False)
    t4 = time.perf_counter()

    np.save(train_path, K_train_train)
    np.save(test_path, K_test_train)

    timing = {
        "cache_hit": False,
        "train_statevectors_seconds": t1 - t0,
        "test_statevectors_seconds": t2 - t1,
        "K_train_train_assembly_seconds": t3 - t2,
        "K_test_train_assembly_seconds": t4 - t3,
        "total_seconds": t4 - t0,
        "n_train_statevectors": len(train_statevectors),
        "n_test_statevectors": len(test_statevectors),
    }
    return K_train_train, K_test_train, timing, False


# --------------------------------------------------------------------------
# Descriptive comparison against Phase 3's already-saved Experiment 3
# --------------------------------------------------------------------------


def load_phase3_experiment3_comparison_row() -> pd.DataFrame:
    """Read Phase 3's already-saved Experiment 3 (quantum-ready classical
    benchmark) comparison table -- Phase 3 is NOT re-run. This is a pure
    file read.
    """
    path = PHASE3_EXP3_DIR / "comparison_table.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Phase 3 Experiment 3 results not found at {path}. "
            f"Phase 5 compares against them descriptively and requires them to exist; "
            f"run `python -m src.classical.train_baselines` (Phase 3) first if missing."
        )
    return pd.read_csv(path)


def build_descriptive_comparison(qsvm_result: ModelResult) -> pd.DataFrame:
    """A side-by-side, DESCRIPTIVE table only -- no significance test, no
    "winner" declared. Phase 5 Section 8 (LOCKED) explicitly excludes
    statistical significance testing.
    """
    classical_rows = load_phase3_experiment3_comparison_row()
    qsvm_row = build_comparison_table([qsvm_result])
    qsvm_row["Model"] = "QSVM (fidelity kernel)"
    combined = pd.concat([classical_rows, qsvm_row], ignore_index=True)
    return combined


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_qsvm_roc_pr(qsvm_result: ModelResult, out_dir: Path) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, ax = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(qsvm_result.y_test_true, qsvm_result.y_test_proba)
    ax.plot(fpr, tpr, label=f"QSVM (AUC={qsvm_result.test_metrics['roc_auc']:.3f})", color="purple")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("QSVM ROC Curve -- Phase 5 (locked test set, n=61)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "qsvm_roc.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    precision, recall, _ = precision_recall_curve(qsvm_result.y_test_true, qsvm_result.y_test_proba)
    ax.plot(recall, precision, label=f"QSVM (AP={qsvm_result.test_metrics['pr_auc']:.3f})", color="purple")
    positive_rate = float(np.mean(qsvm_result.y_test_true))
    ax.axhline(positive_rate, linestyle="--", color="grey", label=f"Chance (positive rate={positive_rate:.2f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("QSVM Precision-Recall Curve -- Phase 5 (locked test set, n=61)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "qsvm_pr.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(qsvm_result: ModelResult, out_dir: Path) -> None:
    from sklearn.metrics import ConfusionMatrixDisplay

    tm = qsvm_result.test_metrics
    cm = np.array([[tm["tn"], tm["fp"]], [tm["fn"], tm["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No disease", "At risk"])
    disp.plot(ax=ax, colorbar=False, cmap="Purples")
    ax.set_title("QSVM Confusion Matrix (locked test set)")
    fig.tight_layout()
    fig.savefig(out_dir / "qsvm_confusion.png", dpi=150)
    plt.close(fig)


def plot_c_grid_cv_curve(search, out_dir: Path) -> None:
    results = search.cv_results_
    C_values = [p["C"] for p in results["params"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        C_values, results["mean_test_roc_auc"], yerr=results["std_test_roc_auc"],
        marker="o", capsize=4, label="CV ROC-AUC (selection metric)",
    )
    ax.plot(C_values, results["mean_test_recall"], marker="s", linestyle="--", label="CV Sensitivity")
    ax.plot(C_values, results["mean_test_specificity"], marker="^", linestyle="--", label="CV Specificity")
    ax.set_xscale("log")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("QSVM: CV score vs C (training kernel only)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "qsvm_cv_vs_C.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Reproducibility record
# --------------------------------------------------------------------------


def _safe_relpath(path: Path) -> str:
    """Path relative to PROJECT_ROOT when possible, else the absolute path.

    Falls back gracefully when the results/cache directories have been
    redirected outside the project root (e.g. by a test using tmp_path) --
    the run record should still be writable and informative, not raise.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _software_versions() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "qiskit": qiskit.__version__,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def run_qsvm_experiment(
    quantum_config: QuantumConfig = DEFAULT_QUANTUM_CONFIG,
    preprocessing_config=DEFAULT_PREPROCESSING_CONFIG,
    C_grid: list[float] | None = None,
) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    KERNEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("QuantumDx Phase 5 -- Full-Scale QSVM Experiment")
    print("=" * 78)

    # --- 1. Reuse the Phase 2/4 locked split; no new split is created -----
    dataset = load_quantum_dataset(preprocessing_config)
    print(f"Loaded locked quantum dataset: split_id={dataset.split_id} "
          f"({len(dataset.X_train_quantum)} train / {len(dataset.X_test_quantum)} test, "
          f"{dataset.n_qubits} qubits)")
    assert dataset.split_id == "e471025b07519a64", (
        f"Unexpected split_id {dataset.split_id!r} -- expected the Phase 2/3/4 locked split. "
        f"Refusing to proceed with an unrecognized split."
    )

    feature_map = build_feature_map(
        quantum_config.feature_map_name, dataset.n_qubits,
        reps=quantum_config.reps, entanglement=quantum_config.entanglement, paulis=quantum_config.paulis,
    )
    fm_report = describe_feature_map(
        feature_map, name=quantum_config.feature_map_name, reps=quantum_config.reps,
        entanglement=quantum_config.entanglement, paulis=quantum_config.paulis,
    )
    print(f"Feature map: {fm_report.name} | qubits={fm_report.n_qubits} | reps={fm_report.reps} "
          f"| entanglement={fm_report.entanglement} | depth={fm_report.depth_logical}")

    # --- 2 & 3. Full train-train and test-train kernels (cached, restartable) ---
    print("\nComputing (or loading cached) kernel matrices...")
    K_train_train, K_test_train, timing, cache_hit = compute_or_load_kernels(
        dataset, quantum_config, KERNEL_CACHE_DIR
    )
    print(f"K_train_train shape: {K_train_train.shape}  |  K_test_train shape: {K_test_train.shape}")
    print(f"Cache hit: {cache_hit}" + (f"  |  total compute time: {timing['total_seconds']:.3f}s" if not cache_hit else ""))

    cache_tag = _kernel_config_id(
        quantum_config.feature_map_name, dataset.n_qubits, quantum_config.reps,
        quantum_config.entanglement, dataset.split_id,
    )
    train_cache_path = KERNEL_CACHE_DIR / f"K_train_train_{cache_tag}.npy"
    test_cache_path = KERNEL_CACHE_DIR / f"K_test_train_{cache_tag}.npy"

    assert K_train_train.shape == (242, 242), f"K_train_train has unexpected shape {K_train_train.shape}"
    assert K_test_train.shape == (61, 242), f"K_test_train has unexpected shape {K_test_train.shape}"

    diag_train_train = kernel_diagnostics(K_train_train, symmetric=True)
    diag_test_train = kernel_diagnostics(K_test_train, symmetric=False)
    print(f"K_train_train diagnostics: symmetric={diag_train_train.is_symmetric}, "
          f"diag_dev_from_1={diag_train_train.diagonal_max_abs_deviation_from_one:.2e}, "
          f"range=[{diag_train_train.min_value:.6f}, {diag_train_train.max_value:.6f}]")
    print(f"K_test_train diagnostics: shape={diag_test_train.shape} (correctly non-square), "
          f"range=[{diag_test_train.min_value:.6f}, {diag_test_train.max_value:.6f}]")
    assert diag_train_train.is_symmetric, "K_train_train must be symmetric"
    assert diag_train_train.diagonal_max_abs_deviation_from_one < 1e-6, "K_train_train diagonal must be ~1"
    assert diag_train_train.within_valid_fidelity_range, "K_train_train values must lie in [0, 1]"
    assert diag_test_train.within_valid_fidelity_range, "K_test_train values must lie in [0, 1]"

    # --- 4 & 5. Leakage-safe C-tuning, training kernel ONLY ------------------
    print(f"\nRunning leakage-safe C-tuning (grid={C_grid or DEFAULT_C_GRID}, "
          f"{CV_FOLDS}-fold stratified CV, seed={preprocessing_config.random_seed})...")
    t_cv0 = time.perf_counter()
    search = run_qsvm_grid_search(
        K_train_train, dataset.y_train, C_grid=C_grid, cv_folds=CV_FOLDS, cv_shuffle=CV_SHUFFLE,
        random_seed=preprocessing_config.random_seed,
    )
    t_cv1 = time.perf_counter()
    print(f"CV tuning complete in {t_cv1 - t_cv0:.2f}s. Selected C={search.best_params_['C']} "
          f"(CV {MODEL_SELECTION_METRIC}={search.best_score_:.4f})")

    # --- 6. ONE evaluation on the locked test set ---------------------------
    qsvm_result = evaluate_on_test(
        search, EXPERIMENT_NAME, "qsvm", "QSVM (fidelity kernel)", "quantum_ready",
        K_test_train, dataset.y_test, n_train=len(dataset.X_train_quantum),
        random_seed=preprocessing_config.random_seed,
    )
    print(f"\nLocked test-set evaluation (touched exactly once):")
    tm = qsvm_result.test_metrics
    print(f"  Accuracy={tm['accuracy']:.4f}  Sensitivity={tm['sensitivity']:.4f}  "
          f"Specificity={tm['specificity']:.4f}  ROC-AUC={tm['roc_auc']:.4f}  PR-AUC={tm['pr_auc']:.4f}")
    print(f"  Confusion matrix: TN={tm['tn']} FP={tm['fp']} FN={tm['fn']} TP={tm['tp']}")

    # --- 7. Descriptive (non-statistical) comparison with Phase 3 -----------
    comparison_table = build_descriptive_comparison(qsvm_result)
    cv_table = build_cv_summary_table([qsvm_result])

    # --- plots ---------------------------------------------------------------
    plot_qsvm_roc_pr(qsvm_result, FIGURES_DIR)
    plot_confusion_matrix(qsvm_result, FIGURES_DIR)
    plot_c_grid_cv_curve(search, FIGURES_DIR)

    # --- save artifacts --------------------------------------------------------
    comparison_table.to_csv(RESULTS_DIR / "comparison_with_phase3.csv", index=False)
    cv_table.to_csv(RESULTS_DIR / "cv_summary.csv", index=False)

    cv_grid_table = pd.DataFrame(search.cv_results_)
    cv_grid_table.to_csv(RESULTS_DIR / "cv_grid_results.csv", index=False)

    final_metrics = {
        "test_metrics": tm,
        "test_bootstrap_95ci": qsvm_result.test_bootstrap,
        "cv_summary": qsvm_result.cv_summary,
        "best_params": qsvm_result.best_params,
        "fit_time_seconds": qsvm_result.fit_time_seconds,
        "inference_time_ms_per_record": qsvm_result.inference_time_ms_per_record,
    }
    with open(RESULTS_DIR / "final_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(final_metrics, fh, indent=2, default=float)

    run_record = {
        "experiment_name": EXPERIMENT_NAME,
        "split_id": dataset.split_id,
        "n_train": len(dataset.X_train_quantum),
        "n_test": len(dataset.X_test_quantum),
        "random_seed": preprocessing_config.random_seed,
        "quantum_config": quantum_config.to_dict(),
        "preprocessing_config": preprocessing_config.to_dict(),
        "feature_map_report": fm_report.to_dict(),
        "backend": get_backend(quantum_config.backend_name).capabilities(),
        "cv_configuration": {"folds": CV_FOLDS, "shuffle": CV_SHUFFLE, "C_grid": C_grid or DEFAULT_C_GRID,
                              "selection_metric": MODEL_SELECTION_METRIC, "scoring_metrics": list(CV_SCORING.keys())},
        "kernel_diagnostics": {
            "K_train_train": diag_train_train.to_dict(),
            "K_test_train": diag_test_train.to_dict(),
        },
        "kernel_computation_timing": timing,
        "kernel_cache_files": {"train": _safe_relpath(train_cache_path), "test": _safe_relpath(test_cache_path)},
        "software_versions": _software_versions(),
        "status": "descriptive_comparison_only_no_significance_testing",
    }
    with open(RESULTS_DIR / "run_record.json", "w", encoding="utf-8") as fh:
        json.dump(run_record, fh, indent=2, default=str)

    print("\n" + "=" * 78)
    print("Descriptive comparison with Phase 3 (no significance test):")
    print(comparison_table.to_string(index=False))
    print("=" * 78)

    return {
        "dataset": dataset,
        "search": search,
        "qsvm_result": qsvm_result,
        "comparison_table": comparison_table,
        "run_record": run_record,
    }


if __name__ == "__main__":
    run_qsvm_experiment()
