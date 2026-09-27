"""Phase 3 orchestrator: leakage-safe classical baseline training and evaluation.

Reuses, without modification or duplication:
    - Phase 1 (src.data.inspect_dataset): raw data loading
    - Phase 2 (src.preprocessing): target creation, train/test split,
      SharedFeaturePipeline (impute+encode+scale+select), the PCA /
      quantum-ready branch, and PreprocessingConfig

Runs three experiments:
    1. "full" feature set, classical (selected-feature) space      -- the
       primary classical benchmark.
    2. "screening" feature set (ca, thal excluded), classical space -- the
       clinical-leakage ablation.
    3. "full" feature set, quantum-ready (PCA-reduced) space         -- the
       benchmark a future quantum model must be compared against fairly,
       since it will see the same PCA-reduced representation.

For every (experiment, model) pair: hyperparameters are tuned via
leakage-safe GridSearchCV on the training split ONLY (see
src.classical.tuning.run_grid_search), then the resulting best_estimator_
(already refit by sklearn on the FULL training split) is evaluated EXACTLY
ONCE on the locked test split (src.classical.evaluation.evaluate_on_test).

Run as:
    python -m src.classical.train_baselines
"""

from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from src.classical.evaluation import (
    ModelResult,
    build_comparison_table,
    build_cv_summary_table,
    compare_class_weighting,
    plot_calibration,
    plot_confusion_matrices,
    plot_metric_comparison,
    plot_pr_curves,
    plot_roc_curves,
)
from src.classical.models import XGBOOST_AVAILABLE, XGBOOST_IMPORT_ERROR, get_available_models
from src.classical.tuning import run_grid_search
from src.data.inspect_dataset import find_project_root, load_raw_dataset, resolve_dataset_path
from src.preprocessing.config import DEFAULT_CONFIG, PreprocessingConfig
from src.preprocessing.pipeline import FeatureGroups, get_feature_groups, run_preprocessing_pipeline

CV_FOLDS = 5
CV_SHUFFLE = True

PROJECT_ROOT = find_project_root()
RESULTS_DIR = PROJECT_ROOT / "results" / "classical"
FIGURES_DIR = RESULTS_DIR / "figures"

#: (experiment_name, config, feature_space) for the three required experiments.
EXPERIMENTS: list[tuple[str, PreprocessingConfig, str]] = [
    ("experiment1_full_classical", DEFAULT_CONFIG.with_overrides(feature_set="full"), "classical"),
    ("experiment2_screening_classical", DEFAULT_CONFIG.with_overrides(feature_set="screening"), "classical"),
    ("experiment3_full_quantum_ready", DEFAULT_CONFIG.with_overrides(feature_set="full"), "quantum_ready"),
]


# --------------------------------------------------------------------------
# Explainability metadata (prepared now, consumed by a later phase)
# --------------------------------------------------------------------------


def extract_explainability_metadata(fitted_pipeline) -> dict:
    """Save what a later explainability phase will need: selected feature
    names, and model coefficients / feature importances where available.
    No SHAP or any new dependency is used -- this reads attributes sklearn
    estimators already expose after fitting.
    """
    shared_step = fitted_pipeline.named_steps["shared"]
    selected_feature_names = list(shared_step.get_feature_names_out())

    info: dict = {"selected_feature_names": selected_feature_names}

    if "quantum" in fitted_pipeline.named_steps:
        n_components = fitted_pipeline.named_steps["quantum"].n_components
        model_input_names = [f"PC{i + 1}" for i in range(n_components)]
        info["model_input_feature_names"] = model_input_names
        info["note"] = (
            "Model was trained on PCA-reduced components, not the raw selected "
            "features listed above -- coefficients/importances below refer to "
            "principal components (PC1..PCn), which are linear combinations of "
            "selected_feature_names. A component-to-feature loading map is a "
            "later-phase (explainability) concern, not computed here."
        )
    else:
        model_input_names = selected_feature_names

    model = fitted_pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        info["coefficients"] = dict(zip(model_input_names, coef.tolist()))
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_).ravel()
        info["feature_importances"] = dict(zip(model_input_names, importances.tolist()))

    return info


# --------------------------------------------------------------------------
# One experiment = all available models, tuned + evaluated
# --------------------------------------------------------------------------


def run_experiment(
    df: pd.DataFrame, experiment_name: str, config: PreprocessingConfig, feature_space: str
) -> tuple[list[ModelResult], dict]:
    # Reused directly from Phase 2 -- gives us the leakage-safe train/test
    # split and raw feature frames. Its own fitted preprocessing artifacts
    # (X_train_classical / X_train_quantum) are NOT used for modeling here;
    # each model's own GridSearchCV pipeline refits preprocessing itself,
    # per-fold, from X_train_raw (see src.classical.tuning).
    result_meta = run_preprocessing_pipeline(df, config)
    X_train_raw, y_train = result_meta.X_train_raw, result_meta.y_train
    X_test_raw, y_test = result_meta.X_test_raw, result_meta.y_test

    feature_groups: FeatureGroups = get_feature_groups(config.feature_set)
    models = get_available_models(config.random_seed)

    results: list[ModelResult] = []
    for key, spec in models.items():
        print(f"  [{experiment_name}] tuning {spec.display_name} ...", flush=True)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            # Two EXPECTED, benign warning sources during a genuine
            # hyperparameter sweep on a small dataset:
            #  (a) a rare category level absent from a fold's training
            #      partition -> OneHotEncoder(handle_unknown='ignore')
            #      correctly encodes it as all-zero (Phase 2 design, see
            #      docs/PREPROCESSING.md Section 5).
            #  (b) sklearn convergence warnings at extreme regularization.
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=Warning, module="sklearn.linear_model")
            search = run_grid_search(
                spec,
                feature_space,
                feature_groups,
                config,
                X_train_raw,
                y_train,
                cv_folds=CV_FOLDS,
                cv_shuffle=CV_SHUFFLE,
                random_seed=config.random_seed,
            )
        result = evaluate_on_test_safe(
            search, experiment_name, key, spec.display_name, feature_space, X_test_raw, y_test,
            n_train=len(X_train_raw), random_seed=config.random_seed,
        )
        result.explainability = extract_explainability_metadata(search.best_estimator_)
        results.append(result)
        elapsed = time.perf_counter() - t0
        print(
            f"    done in {elapsed:.1f}s | best CV ROC-AUC={result.cv_summary['roc_auc']['mean']:.4f} "
            f"| test ROC-AUC={result.test_metrics['roc_auc']:.4f}",
            flush=True,
        )

    return results, {"result_meta": result_meta, "feature_groups": feature_groups}


def evaluate_on_test_safe(*args, **kwargs) -> ModelResult:
    """Thin wrapper so the import stays local to this orchestrator and the
    "only touches test data once, after tuning" property is visible at the
    call site (see src.classical.evaluation.evaluate_on_test docstring).
    """
    from src.classical.evaluation import evaluate_on_test

    return evaluate_on_test(*args, **kwargs)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _software_versions() -> dict:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "matplotlib": __import__("matplotlib").__version__,
    }
    if XGBOOST_AVAILABLE:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    else:
        versions["xgboost"] = f"UNAVAILABLE: {XGBOOST_IMPORT_ERROR}"
    return versions


def save_experiment_outputs(experiment_name: str, results: list[ModelResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp_dir = RESULTS_DIR / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    comparison_table = build_comparison_table(results)
    cv_table = build_cv_summary_table(results)
    comparison_table.to_csv(exp_dir / "comparison_table.csv", index=False)
    cv_table.to_csv(exp_dir / "cv_summary.csv", index=False)

    best_params_record = {r.model_key: r.best_params for r in results}
    with open(exp_dir / "best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best_params_record, fh, indent=2)

    explain_record = {r.model_key: r.explainability for r in results}
    with open(exp_dir / "explainability_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(explain_record, fh, indent=2)

    plot_roc_curves(results, FIGURES_DIR / f"{experiment_name}_roc.png", f"ROC Curves — {experiment_name}")
    plot_pr_curves(results, FIGURES_DIR / f"{experiment_name}_pr.png", f"Precision-Recall Curves — {experiment_name}")
    plot_confusion_matrices(
        results, FIGURES_DIR / f"{experiment_name}_confusion.png", f"Confusion Matrices — {experiment_name}"
    )
    plot_metric_comparison(
        results, FIGURES_DIR / f"{experiment_name}_metrics.png", f"Metric Comparison — {experiment_name}"
    )
    plot_calibration(
        results, FIGURES_DIR / f"{experiment_name}_calibration.png", f"Calibration — {experiment_name}"
    )

    return comparison_table, cv_table


def print_report(
    all_results: dict[str, list[ModelResult]],
    class_weight_results: dict,
) -> None:
    print("\n" + "=" * 78)
    print("QuantumDx Phase 3 -- Classical Baseline Results")
    print("=" * 78)

    for experiment_name, results in all_results.items():
        print(f"\n--- {experiment_name} ---")
        cv_table = build_cv_summary_table(results)
        print("\nCross-validation (training split only), mean +/- std over "
              f"{CV_FOLDS} stratified folds:")
        print(cv_table.to_string(index=False))

        print("\nBest hyperparameters:")
        for r in results:
            print(f"  {r.display_name:20s}: {r.best_params}")

        print("\nFinal locked test-set evaluation (touched exactly once per model):")
        comparison_table = build_comparison_table(results)
        print(comparison_table.to_string(index=False))

        print("\nBootstrap 95% CI on the locked test set (1000 resamples):")
        for r in results:
            b = r.test_bootstrap
            print(
                f"  {r.display_name:20s} "
                f"ROC-AUC={b['roc_auc']['point_estimate']:.3f} "
                f"[{b['roc_auc']['ci_low']:.3f}, {b['roc_auc']['ci_high']:.3f}]  "
                f"Sensitivity={b['sensitivity']['point_estimate']:.3f} "
                f"[{b['sensitivity']['ci_low']:.3f}, {b['sensitivity']['ci_high']:.3f}]  "
                f"Specificity={b['specificity']['point_estimate']:.3f} "
                f"[{b['specificity']['ci_low']:.3f}, {b['specificity']['ci_high']:.3f}]"
            )

    print("\n--- Class-weighting side-check (Experiment 1, tuned hyperparameters) ---")
    for model_key, comparison in class_weight_results.items():
        if comparison is None:
            print(f"  {model_key}: class weighting not applicable to this model.")
            continue
        uw, w = comparison["unweighted"], comparison["weighted"]
        print(
            f"  {model_key:20s} "
            f"unweighted ROC-AUC={uw['roc_auc']['mean']:.4f}+/-{uw['roc_auc']['std']:.4f}  "
            f"weighted ROC-AUC={w['roc_auc']['mean']:.4f}+/-{w['roc_auc']['std']:.4f}  |  "
            f"unweighted sensitivity={uw['recall']['mean']:.4f}  weighted sensitivity={w['recall']['mean']:.4f}  |  "
            f"unweighted specificity={uw['specificity']['mean']:.4f}  weighted specificity={w['specificity']['mean']:.4f}"
        )

    if not XGBOOST_AVAILABLE:
        print(f"\nWARNING: XGBoost unavailable in this environment ({XGBOOST_IMPORT_ERROR}); "
              f"only Logistic Regression, RBF-SVM, and Random Forest were run.")

    print("\nLimitations: see docs/CLASSICAL_BASELINE.md Section 'Limitations of the 303-sample dataset'.")
    print("=" * 78)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"XGBoost available: {XGBOOST_AVAILABLE}" + (f" ({XGBOOST_IMPORT_ERROR})" if not XGBOOST_AVAILABLE else ""))

    df = load_raw_dataset(resolve_dataset_path(None))
    print(f"Loaded raw dataset: {df.shape}")

    all_results: dict[str, list[ModelResult]] = {}
    all_meta: dict[str, dict] = {}

    for experiment_name, config, feature_space in EXPERIMENTS:
        print(f"\n=== Running {experiment_name} "
              f"(feature_set={config.feature_set}, feature_space={feature_space}) ===")
        results, meta = run_experiment(df, experiment_name, config, feature_space)
        all_results[experiment_name] = results
        all_meta[experiment_name] = meta
        save_experiment_outputs(experiment_name, results)

    # --- class-imbalance side-check, on Experiment 1's tuned hyperparameters ---
    exp1_name, exp1_config, exp1_feature_space = EXPERIMENTS[0]
    exp1_meta = all_meta[exp1_name]["result_meta"]
    exp1_feature_groups = all_meta[exp1_name]["feature_groups"]
    models = get_available_models(exp1_config.random_seed)
    class_weight_results = {}
    for r in all_results[exp1_name]:
        spec = models[r.model_key]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            class_weight_results[r.model_key] = compare_class_weighting(
                spec,
                exp1_feature_space,
                exp1_feature_groups,
                exp1_config,
                exp1_meta.X_train_raw,
                exp1_meta.y_train,
                r.best_params,
                cv_folds=CV_FOLDS,
                random_seed=exp1_config.random_seed,
            )
    with open(RESULTS_DIR / "class_weighting_comparison.json", "w", encoding="utf-8") as fh:
        json.dump(class_weight_results, fh, indent=2)

    # --- top-level run record (reproducibility) ---
    run_record = {
        "cv_folds": CV_FOLDS,
        "cv_shuffle": CV_SHUFFLE,
        "random_seed": DEFAULT_CONFIG.random_seed,
        "experiments": {
            name: {
                "feature_set": cfg.feature_set,
                "feature_space": fs,
                "split_id": all_meta[name]["result_meta"].artifacts.split_id,
                "n_train": len(all_meta[name]["result_meta"].X_train_raw),
                "n_test": len(all_meta[name]["result_meta"].X_test_raw),
                "preprocessing_config": cfg.to_dict(),
                "best_params": {r.model_key: r.best_params for r in all_results[name]},
            }
            for name, cfg, fs in EXPERIMENTS
        },
        "software_versions": _software_versions(),
        "xgboost_available": XGBOOST_AVAILABLE,
        "xgboost_import_error": XGBOOST_IMPORT_ERROR,
    }
    with open(RESULTS_DIR / "run_record.json", "w", encoding="utf-8") as fh:
        json.dump(run_record, fh, indent=2, default=str)

    print_report(all_results, class_weight_results)

    return {"results": all_results, "meta": all_meta, "class_weighting": class_weight_results, "run_record": run_record}


if __name__ == "__main__":
    main()
