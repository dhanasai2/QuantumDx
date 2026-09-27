"""Stage D (n_train=20,000): the fourth measured point in the QSVM-vs-classical
scaling experiment, on the IDENTICAL 200-row test set established in Stages
A/B/C (fingerprint 96eac11a8394b87e) -- see docs/STAGE_D.md.

WHAT IS REUSED, UNMODIFIED:
    - src.large_dataset.sampling.build_fixed_split / nested_stratified_stage_samples
      / verify_nesting (the exact same fixed test sets and nesting logic
      Stage A/B/C used -- no new sampling primitive was needed; see
      docs/STAGE_D.md, "Nesting verification")
    - src.large_dataset.pipeline.process_stage (Phase 2's preprocessing,
      completely unmodified)
    - src.classical.tuning.run_grid_search, src.classical.models.get_available_models
      (the SAME 4 models, SAME CV methodology. Logistic Regression, Random
      Forest, and XGBoost use their SAME, unreduced hyperparameter grids.
      RBF-SVM's grid IS narrowed -- see the "COMPUTATIONAL BUDGET DEVIATION"
      block below and docs/STAGE_D.md, "Computational budget", for the
      measured justification)
    - src.classical.evaluation.evaluate_on_test / compute_classification_metrics
    - src.quantum.feature_maps.build_feature_map, src.quantum.backends.get_backend,
      src.quantum.kernel.compute_statevectors / kernel_diagnostics,
      src.quantum.qsvm.run_qsvm_grid_search (identical quantum methodology:
      4 qubits, zz_feature_map, reps=2, linear entanglement, PCA-4, C-grid)
    - src.large_dataset.corrected_comparison.{comparison_set_fingerprint,
      DECISION_THRESHOLD, paired_bootstrap_delta, bootstrap_metric_ci,
      mcnemar_exact} (the exact same paired-statistics code Phase 6A used)

WHAT IS NEW (and why):
    - The classical evaluation here fits each model's GridSearchCV ONCE and
      scores it on BOTH the 2000-row classical test set (for continuity
      with Stage A/B/C's own reporting convention) AND the 200-row PRIMARY
      comparison set, rather than calling run_stage_classical() and
      evaluate_classical_on_comparison_set() separately -- the latter would
      each independently re-run the (expensive, multi-hour at n=20,000)
      grid search, doubling classical compute for no scientific reason.
    - The QSVM train-train kernel is assembled via the new memory-safe
      kernel_matrix_from_statevectors_blockwise (float32, block_size=2000)
      instead of the whole-matrix kernel_matrix_from_statevectors_vectorized
      Stage A/B/C used -- REQUIRED at n=20,000 because the whole-matrix
      approach transiently needs ~10+ GB (a full complex (n,n) intermediate
      plus multiple float64 copies) on a machine measured to have ~4.6 GB
      available. This is a memory optimization only, verified numerically
      equivalent (see tests/test_quantum.py); it does not change the kernel
      definition, feature map, or any modeling decision. See
      docs/STAGE_D.md, "Kernel memory diagnostics" for the measured numbers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import precision_recall_curve, roc_curve

from dataclasses import replace as _dc_replace

from src.classical.evaluation import compute_classification_metrics, evaluate_on_test
from src.classical.models import get_available_models
from src.classical.tuning import run_grid_search
from src.data.inspect_dataset import find_project_root
from src.large_dataset.cleaning import compute_cleaning_flags, split_modeling_subset
from src.large_dataset.corrected_comparison import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DECISION_THRESHOLD,
    bootstrap_metric_ci,
    comparison_set_fingerprint,
    mcnemar_exact,
    paired_bootstrap_delta,
)
from src.large_dataset.pipeline import process_stage
from src.large_dataset.sampling import build_fixed_split, nested_stratified_stage_samples, verify_nesting
from src.large_dataset.schema import TARGET_COLUMN, derive_features, get_cardio_feature_groups, load_raw_cardio
from src.preprocessing.config import PreprocessingConfig
from src.quantum.backends import get_backend
from src.quantum.config import QuantumConfig
from src.quantum.feature_maps import build_feature_map
from src.quantum.kernel import compute_statevectors, kernel_diagnostics, kernel_matrix_from_statevectors_blockwise
from src.quantum.qsvm import run_qsvm_grid_search

CV_FOLDS = 5
CV_SHUFFLE = True
STAGE_D_SIZE = 20000
PRIOR_STAGE_SIZES = [1000, 5000, 10000]  # A, B, C -- for the explicit nesting check

# --------------------------------------------------------------------------
# COMPUTATIONAL BUDGET DEVIATION (Section 8 of the Stage D spec) -- documented
# here, not silently applied. See docs/STAGE_D.md, "Computational budget".
#
# WHAT: RBF-SVM's hyperparameter grid is narrowed from the established 4x4=16
# (C, gamma) combinations (used unmodified at Stages A/B/C) to the SINGLE
# combination already selected as best by that same full grid search at
# Stage C (n_train=10,000, a nested subset of Stage D's training data):
# C=10.0, gamma=0.01 (results/large_dataset/corrected_comparison/stage_10000/
# corrected_comparison.json -> best_params.rbf_svm). Logistic Regression,
# Random Forest, and XGBoost retain their FULL, un-reduced grids -- only
# RBF-SVM is affected.
#
# WHY: measured, not assumed. Before running the full experiment, a single
# representative SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)
# fit was benchmarked in isolation on one real 16,000-row CV fold (80% of
# Stage D's 20,000-row training set, matching GridSearchCV's actual internal
# fold size): 68.8s with sklearn's default cache_size, 126.3s with a 15x
# larger cache (ruling out kernel-cache starvation as the cause -- a larger
# cache made it SLOWER here, so cache_size was left at its default).
# Extrapolating that single-fit cost across the full grid (16 combinations x
# 5 outer CV folds = 80 fits, each ALSO performing its own internal 5-fold
# Platt probability calibration per scikit-learn's probability=True) implies
# a total RBF-SVM budget of several hours at minimum, and likely much more
# for the grid's more expensive corners (e.g. C=100) -- consistent with the
# actual first attempt at this experiment, which was still on RBF-SVM's grid
# search after 4h08m of continuous CPU-bound execution with no completion in
# sight, and was stopped for exactly this reason.
#
# This is the MINIMUM modification that makes Stage D tractable: it reuses,
# rather than re-searches, a hyperparameter choice the project's own
# unmodified methodology already produced. It is NOT scientifically
# equivalent to a fresh, unreduced search at n=20,000 -- report Stage D's
# RBF-SVM runtime and result with that caveat, and do not compare RBF-SVM's
# Stage D wall time against Stage A/B/C's as if the search budgets were the
# same (they are not: 1 combination here vs. 16 there).
_RBF_SVM_FROZEN_PARAMS = {"model__C": [10.0], "model__gamma": [0.01]}
_RBF_SVM_FROZEN_PARAMS_SOURCE = (
    "results/large_dataset/corrected_comparison/stage_10000/corrected_comparison.json "
    "-> best_params.rbf_svm (Stage C's own full, unmodified 4x4 grid search result)"
)

EXPECTED_FINGERPRINT = "96eac11a8394b87e"

RESULTS_ROOT = find_project_root() / "results" / "large_dataset" / "stage_d"

# Memory-safe kernel assembly config -- see module docstring.
KERNEL_BLOCK_SIZE = 2000
KERNEL_DTYPE = np.float32


@dataclass
class ModelPredictions:
    """Predictions for one model on the identical 200-row comparison set --
    same shape as corrected_comparison.ModelPredictions, duplicated here
    (not imported) only because that dataclass is private to a module whose
    orchestration function this one does not call (see module docstring on
    avoiding double-training)."""

    model_key: str
    display_name: str
    ids: np.ndarray
    y_true: np.ndarray
    y_proba: np.ndarray
    y_pred: np.ndarray
    metrics: dict
    best_params: dict
    provenance: str
    train_seconds: float


def _metrics_from_proba(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    return compute_classification_metrics(y_true, y_pred, y_proba)


# --------------------------------------------------------------------------
# Split + nesting
# --------------------------------------------------------------------------


@dataclass
class StageDSplit:
    split: object  # LargeDatasetSplit
    stage_train_df: pd.DataFrame
    nesting: dict  # {size: bool} from verify_nesting
    prior_stage_ids: dict  # {size: set[id]} for A, B, C -- for the report


def build_stage_d_split() -> StageDSplit:
    """Build the fixed split (unchanged) and the Stage D (n=20,000) training
    subset, verifying explicitly that it nests the prior A/B/C subsets
    rather than assuming it.
    """
    df = derive_features(load_raw_cardio())
    flags = compute_cleaning_flags(df)
    modeling_df, _ = split_modeling_subset(df, flags)
    split = build_fixed_split(modeling_df, TARGET_COLUMN, seed=42)

    all_sizes = PRIOR_STAGE_SIZES + [STAGE_D_SIZE]
    stages = nested_stratified_stage_samples(split.training_pool, TARGET_COLUMN, all_sizes, seed=42)
    nesting = verify_nesting(stages, id_column="id")

    stage_d_ids = set(stages[STAGE_D_SIZE]["id"])
    overlap_classical_test = stage_d_ids & set(split.test_set_classical["id"])
    overlap_quantum_test = stage_d_ids & set(split.test_set_quantum["id"])
    if overlap_classical_test or overlap_quantum_test:
        raise ValueError(
            f"Stage D training set overlaps a fixed test set: "
            f"{len(overlap_classical_test)} rows vs classical test, "
            f"{len(overlap_quantum_test)} rows vs quantum test."
        )
    if not all(nesting.values()):
        raise ValueError(f"Stage nesting verification failed: {nesting}")

    return StageDSplit(
        split=split,
        stage_train_df=stages[STAGE_D_SIZE],
        nesting=nesting,
        prior_stage_ids={n: set(stages[n]["id"]) for n in PRIOR_STAGE_SIZES},
    )


# --------------------------------------------------------------------------
# Classical: fit ONCE per model, score on BOTH test sets
# --------------------------------------------------------------------------


def run_stage_d_classical(stage_d: StageDSplit, feature_groups, preprocessing_config: PreprocessingConfig):
    """Returns (results_2000: list[ModelResult], preds_200: list[ModelPredictions],
    search_seconds: dict[str, float])."""
    split = stage_d.split
    feature_cols = feature_groups.all_columns
    X_train_raw = stage_d.stage_train_df[feature_cols]
    y_train = stage_d.stage_train_df[TARGET_COLUMN].to_numpy()

    X_test_2000_raw = split.test_set_classical[feature_cols]
    y_test_2000 = split.test_set_classical[TARGET_COLUMN].to_numpy()

    X_test_200_raw = split.test_set_quantum[feature_cols]
    y_test_200 = split.test_set_quantum[TARGET_COLUMN].to_numpy()
    ids_200 = split.test_set_quantum["id"].to_numpy()

    models = get_available_models(preprocessing_config.random_seed)
    results_2000 = []
    preds_200 = []
    search_seconds = {}

    for key, spec in models.items():
        if key == "rbf_svm":
            spec = _dc_replace(spec, param_grid=_RBF_SVM_FROZEN_PARAMS)
            print(f"[stage_d] classical: rbf_svm grid NARROWED to 1 combination "
                  f"{_RBF_SVM_FROZEN_PARAMS} (source: {_RBF_SVM_FROZEN_PARAMS_SOURCE}); "
                  f"see docs/STAGE_D.md 'Computational budget' for why.", flush=True)
        n_combos = 1
        for v in spec.param_grid.values():
            n_combos *= len(v)
        print(f"[stage_d] classical: starting {key} grid search "
              f"({n_combos} combinations x {CV_FOLDS} folds, n_train={len(X_train_raw)}) ...", flush=True)
        t0 = time.perf_counter()
        search = run_grid_search(
            spec, "classical", feature_groups, preprocessing_config,
            X_train_raw, y_train, cv_folds=CV_FOLDS, cv_shuffle=CV_SHUFFLE,
            random_seed=preprocessing_config.random_seed,
        )
        search_seconds[key] = time.perf_counter() - t0
        print(f"[stage_d] classical: finished {key} in {search_seconds[key]:.1f}s "
              f"(best_params={search.best_params_})", flush=True)

        result_2000 = evaluate_on_test(
            search, f"large_dataset_stage_{STAGE_D_SIZE}", key, spec.display_name,
            "classical", X_test_2000_raw, y_test_2000, n_train=len(X_train_raw),
            random_seed=preprocessing_config.random_seed,
        )
        results_2000.append(result_2000)

        # PRIMARY evaluation: identical 200-row set. Model already frozen by
        # run_grid_search's own refit-on-full-train; only .predict_proba() below.
        y_proba_200 = search.best_estimator_.predict_proba(X_test_200_raw)[:, 1]
        y_pred_200 = (y_proba_200 >= DECISION_THRESHOLD).astype(int)
        preds_200.append(
            ModelPredictions(
                model_key=key, display_name=spec.display_name, ids=ids_200, y_true=y_test_200,
                y_proba=y_proba_200, y_pred=y_pred_200,
                metrics=_metrics_from_proba(y_test_200, y_proba_200),
                best_params={k.replace("model__", ""): v for k, v in search.best_params_.items()},
                provenance="reproduced_training_stage_d", train_seconds=search_seconds[key],
            )
        )

    return results_2000, preds_200, search_seconds


# --------------------------------------------------------------------------
# QSVM: memory-safe kernel assembly
# --------------------------------------------------------------------------


def run_stage_d_qsvm(stage_d: StageDSplit, feature_groups, preprocessing_config: PreprocessingConfig, quantum_config: QuantumConfig):
    """Returns (preds: ModelPredictions, kernel_timing: dict, kernel_diag: dict, processed)."""
    proc = psutil.Process()
    split = stage_d.split

    t0 = time.perf_counter()
    processed = process_stage(
        stage_d.stage_train_df, split.test_set_classical, split.test_set_quantum,
        TARGET_COLUMN, feature_groups, preprocessing_config,
    )
    preprocessing_seconds = time.perf_counter() - t0

    feature_map = build_feature_map(
        quantum_config.feature_map_name, processed.n_qubits,
        reps=quantum_config.reps, entanglement=quantum_config.entanglement,
        paulis=quantum_config.paulis,
    )
    backend = get_backend(quantum_config.backend_name)

    print(f"[stage_d] qsvm: preprocessing done in {preprocessing_seconds:.1f}s, "
          f"n_train={len(processed.X_train_quantum)}, n_qubits={processed.n_qubits}", flush=True)

    t1 = time.perf_counter()
    train_svs = compute_statevectors(processed.X_train_quantum, feature_map, backend)
    t2 = time.perf_counter()
    print(f"[stage_d] qsvm: train statevectors done in {t2 - t1:.1f}s", flush=True)
    test_svs = compute_statevectors(processed.X_test_quantum, feature_map, backend)
    t3 = time.perf_counter()
    print(f"[stage_d] qsvm: test statevectors done in {t3 - t2:.1f}s", flush=True)

    rss_before_kernel_mb = proc.memory_info().rss / 1e6
    print(f"[stage_d] qsvm: assembling K_train_train ({len(train_svs)}x{len(train_svs)}, "
          f"dtype={KERNEL_DTYPE.__name__}, block_size={KERNEL_BLOCK_SIZE}); "
          f"RSS before = {rss_before_kernel_mb:.0f} MB ...", flush=True)
    K_train_train = kernel_matrix_from_statevectors_blockwise(
        train_svs, train_svs, symmetric=True, block_size=KERNEL_BLOCK_SIZE, dtype=KERNEL_DTYPE,
    )
    t4 = time.perf_counter()
    rss_after_train_kernel_mb = proc.memory_info().rss / 1e6
    print(f"[stage_d] qsvm: K_train_train done in {t4 - t3:.1f}s; RSS after = {rss_after_train_kernel_mb:.0f} MB", flush=True)
    K_test_train = kernel_matrix_from_statevectors_blockwise(
        test_svs, train_svs, symmetric=False, block_size=KERNEL_BLOCK_SIZE, dtype=KERNEL_DTYPE,
    )
    t5 = time.perf_counter()
    print(f"[stage_d] qsvm: K_test_train done in {t5 - t4:.1f}s", flush=True)

    diag = kernel_diagnostics(K_train_train, symmetric=True)
    print(f"[stage_d] qsvm: kernel diagnostics: {diag.to_dict()}", flush=True)

    print("[stage_d] qsvm: starting CV C-grid search ...", flush=True)
    t6 = time.perf_counter()
    search = run_qsvm_grid_search(
        K_train_train, processed.y_train,
        cv_folds=CV_FOLDS, cv_shuffle=CV_SHUFFLE, random_seed=preprocessing_config.random_seed,
    )
    cv_seconds = time.perf_counter() - t6
    rss_after_cv_mb = proc.memory_info().rss / 1e6
    print(f"[stage_d] qsvm: CV done in {cv_seconds:.1f}s, best_C={search.best_params_}, "
          f"RSS after CV = {rss_after_cv_mb:.0f} MB", flush=True)

    t7 = time.perf_counter()
    y_proba = search.best_estimator_.predict_proba(K_test_train)[:, 1]
    inference_seconds = time.perf_counter() - t7
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    preds = ModelPredictions(
        model_key="qsvm", display_name="QSVM (fidelity kernel)",
        ids=split.test_set_quantum["id"].to_numpy(), y_true=processed.y_test_quantum,
        y_proba=y_proba, y_pred=y_pred,
        metrics=_metrics_from_proba(processed.y_test_quantum, y_proba),
        best_params={k.replace("model__", ""): v for k, v in search.best_params_.items()},
        provenance="reproduced_training_stage_d",
        train_seconds=(t5 - t1) + cv_seconds,
    )

    kernel_timing = {
        "n_train": int(len(processed.X_train_quantum)),
        "n_test": int(len(processed.X_test_quantum)),
        "preprocessing_and_pca_seconds": preprocessing_seconds,
        "train_statevectors_seconds": t2 - t1,
        "test_statevectors_seconds": t3 - t2,
        "K_train_train_assembly_seconds": t4 - t3,
        "K_test_train_assembly_seconds": t5 - t4,
        "total_kernel_seconds": t5 - t1,
        "cv_tuning_seconds": cv_seconds,
        "final_inference_seconds": inference_seconds,
        "qsvm_total_wall_seconds": (t5 - t1) + cv_seconds + inference_seconds,
        "K_train_train_shape": list(K_train_train.shape),
        "K_train_train_dtype": str(K_train_train.dtype),
        "K_train_train_memory_mb": K_train_train.nbytes / 1e6,
        "kernel_block_size": KERNEL_BLOCK_SIZE,
        "memory_optimization": "blockwise assembly + float32 storage (see kernel.py docstring); "
                                "verified numerically equivalent to the float64 reference before use "
                                "(tests/test_quantum.py::test_blockwise_kernel_*)",
        "process_rss_mb": {
            "before_kernel_assembly": rss_before_kernel_mb,
            "after_train_train_kernel": rss_after_train_kernel_mb,
            "after_cv_tuning": rss_after_cv_mb,
        },
    }
    return preds, kernel_timing, diag.to_dict(), processed


# --------------------------------------------------------------------------
# Statistics: reuse Phase 6A's exact functions, on Stage D's 200-row predictions
# --------------------------------------------------------------------------


def run_stage_d_statistics(classical_preds_200: list[ModelPredictions], qsvm_preds: ModelPredictions) -> dict:
    """Bootstrap CIs for all 7 metrics per model, plus paired DeLong/bootstrap/
    McNemar of each classical model vs QSVM -- same methodology and seed as
    Phase 6A's compute_all_metric_cis / run_statistical_robustness_analysis,
    reused here as separate calls (not the shared orchestrator itself, which
    is keyed to the fixed A/B/C stage dict) so Stage D's own outputs never
    touch or overwrite Phase 6A's results directory.
    """
    label_metrics = ["sensitivity", "specificity", "accuracy", "precision", "f1"]
    single_cis: dict[str, dict] = {}
    for p in classical_preds_200 + [qsvm_preds]:
        cis = {
            "roc_auc": bootstrap_metric_ci(p.y_true, p.y_proba, "roc_auc", n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED),
            "pr_auc": bootstrap_metric_ci(p.y_true, p.y_proba, "pr_auc", n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED),
        }
        rng = np.random.RandomState(BOOTSTRAP_SEED)
        n = len(p.y_true)
        samples = {m: [] for m in label_metrics}
        for _ in range(BOOTSTRAP_RESAMPLES):
            idx = rng.randint(0, n, size=n)
            yt, pr = p.y_true[idx], p.y_proba[idx]
            if len(np.unique(yt)) < 2:
                continue
            m = compute_classification_metrics(yt, (pr >= DECISION_THRESHOLD).astype(int), pr)
            for key in label_metrics:
                samples[key].append(m[key])
        for key, vals in samples.items():
            arr = np.array(vals)
            lo, hi = np.quantile(arr, [0.025, 0.975]) if len(arr) else (np.nan, np.nan)
            cis[key] = {
                "point_estimate": float(p.metrics[key]), "ci_low": float(lo), "ci_high": float(hi),
                "n_resamples_used": int(len(arr)), "seed": BOOTSTRAP_SEED,
            }
        single_cis[p.model_key] = cis

    paired = {}
    for p in classical_preds_200:
        bs_roc = paired_bootstrap_delta(p.y_true, p.y_proba, qsvm_preds.y_proba, "roc_auc", n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
        bs_pr = paired_bootstrap_delta(p.y_true, p.y_proba, qsvm_preds.y_proba, "pr_auc", n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
        mc = mcnemar_exact(p.y_true, p.y_pred, qsvm_preds.y_pred)
        dl = _delong(p.y_true, p.y_proba, qsvm_preds.y_proba)
        paired[p.model_key] = {"bootstrap_roc_auc": bs_roc, "bootstrap_pr_auc": bs_pr, "mcnemar": mc, "delong": dl}

    return {"single_model_cis": single_cis, "paired_vs_qsvm": paired}


def _delong(y_true, proba_a, proba_b) -> dict:
    """Local import wrapper -- reuses statistical_robustness.delong_test
    (Phase 6A's exact, already-tested implementation) unmodified."""
    from src.large_dataset.statistical_robustness import delong_test

    return delong_test(y_true, proba_a, proba_b).to_dict()


# --------------------------------------------------------------------------
# Extend Phase 6A's 12-comparison Holm-Bonferroni family to 16 (Task 15)
# --------------------------------------------------------------------------

PHASE_6A_ROOT = find_project_root() / "results" / "large_dataset" / "statistical_robustness"
STAGE_D_STATS_ROOT = PHASE_6A_ROOT / "stage_d_extension"

_AUC_COLUMNS = [
    "stage", "n_train", "classical_model", "auc_classical", "auc_qsvm",
    "delong_delta", "delong_ci_low", "delong_ci_high", "delong_p",
    "bootstrap_delta", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_p",
    "mcnemar_p", "mcnemar_n_discordant",
]


def _stage_d_auc_rows(classical_preds_200: list[ModelPredictions], stats_result: dict) -> list[dict]:
    rows = []
    for p in classical_preds_200:
        paired = stats_result["paired_vs_qsvm"][p.model_key]
        dl, bs, mc = paired["delong"], paired["bootstrap_roc_auc"], paired["mcnemar"]
        rows.append({
            "stage": "D", "n_train": STAGE_D_SIZE, "classical_model": p.model_key,
            "auc_classical": dl["auc_a"], "auc_qsvm": dl["auc_b"],
            "delong_delta": dl["delta"], "delong_ci_low": dl["ci_low"], "delong_ci_high": dl["ci_high"],
            "delong_p": dl["p_value"],
            "bootstrap_delta": bs["observed_delta"], "bootstrap_ci_low": bs["ci_low"],
            "bootstrap_ci_high": bs["ci_high"], "bootstrap_p": bs["bootstrap_p_value"],
            "mcnemar_p": mc["p_value"], "mcnemar_n_discordant": mc["n_discordant"],
        })
    return rows


def extend_holm_family_to_16(classical_preds_200: list[ModelPredictions], stats_result: dict) -> pd.DataFrame:
    """Combine Phase 6A's 12 existing DeLong/bootstrap comparisons (READ
    ONLY -- the source file is never written to) with Stage D's 4 new ones
    into a single 16-comparison family, and apply Holm-Bonferroni ONCE
    across all 16 p-values (per Task 15: do not correct Stage D alone).

    Reuses statistical_robustness.holm_bonferroni verbatim, the identical
    function Phase 6A used for its own 12-comparison correction.
    """
    from src.large_dataset.statistical_robustness import holm_bonferroni

    phase6a_path = PHASE_6A_ROOT / "auc_comparisons.csv"
    if not phase6a_path.is_file():
        raise FileNotFoundError(
            f"Expected Phase 6A's 12-comparison results at {phase6a_path}; "
            f"Stage D's 16-comparison family extends, and requires, that file."
        )
    phase6a_df = pd.read_csv(phase6a_path)[_AUC_COLUMNS]
    stage_d_df = pd.DataFrame(_stage_d_auc_rows(classical_preds_200, stats_result))[_AUC_COLUMNS]

    combined = pd.concat([phase6a_df, stage_d_df], ignore_index=True)
    assert len(combined) == 16, f"expected 16 comparisons (12 Phase 6A + 4 Stage D), got {len(combined)}"

    holm_delong = holm_bonferroni(combined["delong_p"].tolist())
    holm_bootstrap = holm_bonferroni(combined["bootstrap_p"].tolist())
    for i, (hd, hb) in enumerate(zip(holm_delong, holm_bootstrap)):
        combined.loc[i, "delong_p_holm"] = hd["adjusted_p"]
        combined.loc[i, "delong_significant_holm"] = hd["significant"]
        combined.loc[i, "bootstrap_p_holm"] = hb["adjusted_p"]
        combined.loc[i, "bootstrap_significant_holm"] = hb["significant"]

    return combined


# --------------------------------------------------------------------------
# Plots (same conventions as corrected_comparison.plot_curves)
# --------------------------------------------------------------------------


def plot_stage_d_curves(all_preds: list[ModelPredictions], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    n_cmp = len(all_preds[0].y_true)

    fig, ax = plt.subplots(figsize=(7, 6))
    for p in all_preds:
        fpr, tpr, _ = roc_curve(p.y_true, p.y_proba)
        ax.plot(fpr, tpr, label=f"{p.display_name} (AUC={p.metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC -- Stage n={STAGE_D_SIZE:,}\nIdentical {n_cmp}-row held-out test set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for p in all_preds:
        prec, rec, _ = precision_recall_curve(p.y_true, p.y_proba)
        ax.plot(rec, prec, label=f"{p.display_name} (AP={p.metrics['pr_auc']:.4f})")
    base = float(np.mean(all_preds[0].y_true))
    ax.axhline(base, ls="--", color="grey", label=f"Chance ({base:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall -- Stage n={STAGE_D_SIZE:,}\nIdentical {n_cmp}-row held-out test set")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(all_preds), figsize=(4 * len(all_preds), 4))
    for ax, p in zip(np.atleast_1d(axes), all_preds):
        m = p.metrics
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ConfusionMatrixDisplay(cm, display_labels=["No CVD", "CVD"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(p.display_name, fontsize=9)
    fig.suptitle(f"Confusion matrices -- Stage n={STAGE_D_SIZE:,}, identical {n_cmp}-row test set (threshold=0.5)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


def run_stage_d(*, make_plots: bool = True) -> dict:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    runtime: dict[str, float] = {}

    t0 = time.perf_counter()
    stage_d = build_stage_d_split()
    runtime["data_loading_and_split_seconds"] = time.perf_counter() - t0

    fg = get_cardio_feature_groups()
    pcfg = PreprocessingConfig()
    qcfg = QuantumConfig()

    cmp_ids = stage_d.split.test_set_quantum["id"].tolist()
    fingerprint = comparison_set_fingerprint(cmp_ids)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"Stage D comparison set fingerprint {fingerprint} != expected {EXPECTED_FINGERPRINT}")

    print("[stage_d] ===== PHASE: classical (4 models; LR/RF/XGB full established grids, "
          "RBF-SVM narrowed to Stage C's best combo -- see docs/STAGE_D.md; n_train=%d) ====="
          % len(stage_d.stage_train_df), flush=True)
    t1 = time.perf_counter()
    results_2000, classical_preds_200, classical_search_seconds = run_stage_d_classical(stage_d, fg, pcfg)
    runtime["classical_total_seconds"] = time.perf_counter() - t1
    runtime["classical_per_model_seconds"] = classical_search_seconds
    print(f"[stage_d] ===== classical phase done in {runtime['classical_total_seconds']:.1f}s ({runtime['classical_total_seconds']/60:.1f} min) =====", flush=True)

    # Checkpoint: persist classical results NOW, before starting the
    # memory-sensitive QSVM step, so a crash there does not lose a
    # multi-hour classical run.
    checkpoint = {
        "classical_search_seconds": classical_search_seconds,
        "classical_metrics_200row": {p.model_key: p.metrics for p in classical_preds_200},
        "classical_metrics_2000row": {r.model_key: r.test_metrics for r in results_2000},
        "classical_best_params": {p.model_key: p.best_params for p in classical_preds_200},
    }
    with open(RESULTS_ROOT / "_classical_checkpoint.json", "w", encoding="utf-8") as fh:
        json.dump(checkpoint, fh, indent=2, default=float)
    pd.DataFrame({"id": cmp_ids, "y_true": classical_preds_200[0].y_true, **{
        f"{p.model_key}_{field}": getattr(p, field) for p in classical_preds_200 for field in ("y_proba", "y_pred")
    }}).to_csv(RESULTS_ROOT / "_classical_checkpoint_predictions.csv", index=False)
    print("[stage_d] classical checkpoint written to _classical_checkpoint.json / _classical_checkpoint_predictions.csv", flush=True)

    print("[stage_d] ===== PHASE: QSVM (memory-safe blockwise kernel, n_train=%d) =====" % len(stage_d.stage_train_df), flush=True)
    t2 = time.perf_counter()
    qsvm_preds, kernel_timing, kernel_diag, processed = run_stage_d_qsvm(stage_d, fg, pcfg, qcfg)
    runtime["qsvm_total_seconds"] = time.perf_counter() - t2
    runtime["kernel_timing"] = kernel_timing
    print(f"[stage_d] ===== QSVM phase done in {runtime['qsvm_total_seconds']:.1f}s ({runtime['qsvm_total_seconds']/60:.1f} min) =====", flush=True)

    all_preds_200 = classical_preds_200 + [qsvm_preds]
    for p in all_preds_200:
        assert list(p.ids) == cmp_ids, f"{p.model_key} predictions not aligned to the comparison ids"
        assert np.array_equal(p.y_true, qsvm_preds.y_true), f"{p.model_key} y_true differs from QSVM's"

    stats_result = run_stage_d_statistics(classical_preds_200, qsvm_preds)

    combined_16 = extend_holm_family_to_16(classical_preds_200, stats_result)
    STAGE_D_STATS_ROOT.mkdir(parents=True, exist_ok=True)
    combined_16.to_csv(STAGE_D_STATS_ROOT / "auc_comparisons_16.csv", index=False)

    # ---- persist ----
    pred_df = pd.DataFrame({"id": cmp_ids, "y_true": qsvm_preds.y_true})
    for p in all_preds_200:
        pred_df[f"{p.model_key}_proba"] = p.y_proba
        pred_df[f"{p.model_key}_pred"] = p.y_pred
    pred_df.to_csv(RESULTS_ROOT / "predictions.csv", index=False)

    metrics_rows = [
        {
            "Model": p.display_name, "model_key": p.model_key,
            **{k: p.metrics[k] for k in ("roc_auc", "pr_auc", "sensitivity", "specificity", "accuracy", "precision", "f1", "tn", "fp", "fn", "tp")},
            "best_params": json.dumps(p.best_params), "provenance": p.provenance,
            "train_seconds": round(p.train_seconds, 3),
        }
        for p in all_preds_200
    ]
    pd.DataFrame(metrics_rows).to_csv(RESULTS_ROOT / "metrics.csv", index=False)

    with open(RESULTS_ROOT / "kernel_diagnostics.json", "w", encoding="utf-8") as fh:
        json.dump({"timing": kernel_timing, "diagnostics": kernel_diag}, fh, indent=2, default=float)

    with open(RESULTS_ROOT / "runtime.json", "w", encoding="utf-8") as fh:
        json.dump(runtime, fh, indent=2, default=float)

    summary = {
        "stage_size": STAGE_D_SIZE,
        "comparison_set": {
            "n": len(cmp_ids), "fingerprint": fingerprint, "matches_expected": fingerprint == EXPECTED_FINGERPRINT,
            "positive_rate": float(np.mean(qsvm_preds.y_true)),
        },
        "nesting_verified": stage_d.nesting,
        "computational_budget_deviations": {
            "rbf_svm_grid_narrowed": {
                "applied": True,
                "original_grid": {"C": [0.1, 1.0, 10.0, 100.0], "gamma": ["scale", 0.01, 0.1, 1.0]},
                "narrowed_to": {k.replace("model__", ""): v for k, v in _RBF_SVM_FROZEN_PARAMS.items()},
                "source_of_frozen_params": _RBF_SVM_FROZEN_PARAMS_SOURCE,
                "reason": "Full 16-combination grid at n_train=20,000 measured (isolated single-fit "
                          "benchmark, one representative combo on one real 16,000-row fold) at 68.8s "
                          "per fit; extrapolated across 80 grid/fold fits plus each fit's internal "
                          "5-fold Platt probability calibration implies a multi-hour-to-multi-day "
                          "budget. A first attempt at the full grid was stopped after 4h08m of "
                          "continuous CPU-bound execution with no completion in sight. LR, RF, and "
                          "XGB retain their full, unmodified grids -- only RBF-SVM was narrowed.",
                "caveat": "RBF-SVM's Stage D runtime/result is NOT under the same search budget as "
                          "Stage A/B/C (1 combination here vs. 16 there); do not compare wall time "
                          "or treat this as a fresh hyperparameter search at n=20,000.",
            },
        },
        "metrics_200row": {p.model_key: p.metrics for p in all_preds_200},
        "metrics_2000row_classical": {r.model_key: r.test_metrics for r in results_2000},
        "best_params": {p.model_key: p.best_params for p in all_preds_200},
        "statistics": stats_result,
        "runtime": runtime,
        "kernel_diagnostics": kernel_diag,
        "holm_16_comparison_family": {
            "n_comparisons": len(combined_16),
            "significant_after_holm_delong": int(combined_16["delong_significant_holm"].sum()),
            "significant_after_holm_bootstrap": int(combined_16["bootstrap_significant_holm"].sum()),
            "csv_path": str(STAGE_D_STATS_ROOT / "auc_comparisons_16.csv"),
        },
    }
    with open(RESULTS_ROOT / "stage_d_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    if make_plots:
        plot_stage_d_curves(all_preds_200, RESULTS_ROOT)

    return summary


if __name__ == "__main__":
    result = run_stage_d()
    print(json.dumps({k: v for k, v in result.items() if k != "statistics"}, indent=2, default=float))
