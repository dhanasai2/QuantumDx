"""Phase 11: full-dataset scale experiment (see docs/PHASE_11_FULL_SCALE.md).

QUESTION: do the conclusions from Phases 9 and 10 change when models are
trained on the FULL available training pool (n=66,641) rather than the
n_train=2,000 screening subset -- same fixed 200-row test set throughout?

EXACT TRAINING POOL SIZE (determined by inspection before writing this,
per instruction -- NOT assumed): derive_features(load_raw_cardio()) ->
68,641 modeling rows -> build_fixed_split(seed=42) draws the SAME fixed
2,000-row classical test set (whose 200-row quantum subset is the
project's one enduring comparison set) -> leaves training_pool = 66,641
rows. This is the SAME training_pool Stage A/B/C/D and Phase 8's
screening subset were themselves nested inside -- no new sampling
primitive was needed; "full training" here means "the entire pool", not
"70,000 minus 200" (the classical 2,000-row test set is correctly
excluded too, exactly as every prior phase already did).

WHAT IS REUSED, UNMODIFIED:
    - src.large_dataset.sampling.build_fixed_split (the exact same fixed
      test sets, fingerprint 96eac11a8394b87e, every phase has used)
    - src.large_dataset.pipeline.process_stage (Phase 2 preprocessing/PCA)
    - src.classical.models.get_available_models / src.classical.tuning.run_grid_search
      / src.classical.evaluation.evaluate_on_test (Phase 3's unmodified
      methodology, scored on the 200-row set as Phase 9 already did)
    - src.quantum.hybrid_quantum_features.HybridQuantumFeatureLayer / HybridConfig
      (Phase 10's architecture, byte-for-byte unmodified)
    - src.large_dataset.corrected_comparison / statistical_robustness
      (the same paired-statistics functions every prior phase used)

DOCUMENTED, MEASURED DEVIATIONS (both benchmarked before this module was
finalized -- neither is guessed):

    1. RBF-SVM at n=66,641 is NOT run. A single fit with the established
       best hyperparameters (C=10.0, gamma=0.01 -- Stage C/D/Phase 9's own
       selection) was benchmarked directly and exceeded 20 minutes with
       no completion, consistent with the LIBSVM superlinear-scaling
       pathology already documented twice in this project (Stage D's own
       4h08m stall at n=20,000's full grid). RBF-SVM's Stage D result
       (n=20,000, already evaluated on the IDENTICAL 200-row test set) is
       reused instead, clearly labeled as a smaller scale, not full-data.

    2. QSVM (baseline / MI-adaptive) is NOT run at full scale. The
       train-train fidelity kernel at n=66,641 would require a
       66,641 x 66,641 matrix -- 17.75 GB even in float32 -- far beyond
       this machine's available memory (Stage D's own n=20,000 QSVM run,
       at only 1.6 GB, already produced a documented near-OOM memory
       scare with far more headroom than is available now). Phase 9's
       n=2,000 QSVM results are retained for reference and clearly
       labeled as a different, smaller scale -- not silently compared as
       if equivalent.

    3. The optional learning-curve sweep (2k/5k/10k/20k/50k/full) was NOT
       run: it would require repeating the RF/XGBoost grids (already the
       dominant cost here) at 5 additional scales, on top of an already
       large, multi-model full-scale run. The core question this phase
       exists to answer -- does the n=2,000 conclusion change at full
       scale -- is fully answered by the two-point (n=2,000 vs n=66,641)
       comparison the task itself also templates; Stage A-D's own
       n=1,000/5,000/10,000/20,000 classical numbers exist but were
       evaluated on the DIFFERENT 2,000-row classical test set (not the
       200-row set), so they are reported as informal historical context
       only, explicitly labeled as not directly comparable -- exactly the
       population-mismatch trap the project's own Phase 6 correction
       phase (docs/LARGE_DATASET_CORRECTION.md) already established must
       never be silently repeated.

TEST-SET DISCIPLINE: classical hyperparameters are selected via 5-fold CV
on the full training pool ONLY (run_grid_search never sees the test set);
the hybrid's quantum weights theta are selected via COBYLA + 3-fold CV on
the SAME n_train=2,000 screening subset Phase 10 used (a proper,
already-vetted, zero-overlap-with-test subset of this exact training
pool) -- documented explicitly below, not on the full 66,641 rows,
because a full-batch COBYLA search there was benchmarked (implicitly, via
the single-forward-pass measurement below) to cost far more than
justified for parameter SELECTION specifically. The 200-row test set is
scored exactly once per model, after every selection is frozen.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve

from src.classical.evaluation import compute_classification_metrics, evaluate_on_test
from src.classical.models import get_available_models
from src.classical.tuning import run_grid_search
from src.data.inspect_dataset import find_project_root
from src.large_dataset.cleaning import compute_cleaning_flags, split_modeling_subset
from src.large_dataset.corrected_comparison import (
    DECISION_THRESHOLD,
    comparison_set_fingerprint,
    mcnemar_exact,
    paired_bootstrap_delta,
)
from src.large_dataset.phase8a_label_aware_screening import build_screening_split
from src.large_dataset.pipeline import process_stage
from src.large_dataset.sampling import build_fixed_split
from src.large_dataset.schema import TARGET_COLUMN, derive_features, get_cardio_feature_groups, load_raw_cardio
from src.large_dataset.statistical_robustness import delong_test, holm_bonferroni
from src.preprocessing.config import PreprocessingConfig
from src.quantum.hybrid_quantum_features import HybridConfig, HybridQuantumFeatureLayer

EXPECTED_FINGERPRINT = "96eac11a8394b87e"
SAMPLING_SEED = 42
CV_FOLDS = 5
CV_SHUFFLE = True

RESULTS_ROOT = find_project_root() / "results" / "large_dataset" / "phase11_full_scale"
STAGE_D_PREDICTIONS_PATH = find_project_root() / "results" / "large_dataset" / "stage_d" / "predictions.csv"
PHASE9_PREDICTIONS_PATH = find_project_root() / "results" / "large_dataset" / "phase9_classical_benchmark" / "predictions.csv"
PHASE10_METRICS_PATH = find_project_root() / "results" / "large_dataset" / "phase10_hybrid_qml" / "metrics.csv"

#: RBF-SVM's established best hyperparameters (Stage C/D, Phase 9 --
#: reselected independently at 3 different training sizes and landed on
#: the SAME values each time). Reused here, NOT re-searched -- see module
#: docstring, deviation 1.
RBF_SVM_FROZEN_PARAMS = {"C": 10.0, "gamma": 0.01}


def _metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    return compute_classification_metrics(y_true, y_pred, y_proba)


def build_full_scale_split():
    df = derive_features(load_raw_cardio())
    flags = compute_cleaning_flags(df)
    modeling_df, _ = split_modeling_subset(df, flags)
    return build_fixed_split(modeling_df, TARGET_COLUMN, seed=SAMPLING_SEED)


def load_reference_predictions() -> dict[str, pd.DataFrame]:
    """Loads Stage D (for RBF-SVM, n=20,000) and Phase 9 (for QSVM, n=2,000)
    reference predictions -- both verified against the same test
    fingerprint. Neither is retrained."""
    refs = {}
    for name, path in [("stage_d", STAGE_D_PREDICTIONS_PATH), ("phase9", PHASE9_PREDICTIONS_PATH)]:
        if not path.is_file():
            raise FileNotFoundError(f"Expected {name} predictions at {path}; required for Phase 11 reuse.")
        df = pd.read_csv(path)
        fp = comparison_set_fingerprint(df["id"].tolist())
        if fp != EXPECTED_FINGERPRINT:
            raise ValueError(f"{name} predictions.csv fingerprint {fp} != expected {EXPECTED_FINGERPRINT}")
        refs[name] = df
    return refs


def run_classical_full_scale(split, feature_groups, preprocessing_config) -> tuple[dict, dict]:
    """Runs LR, RF, XGBoost full grids at n=66,641 (all benchmarked
    feasible beforehand: single-fit costs of 130s/13.2s/3.2s respectively
    extrapolate to well-bounded full-grid totals). RBF-SVM is NOT run
    here -- see module docstring, deviation 1."""
    feature_cols = feature_groups.all_columns
    X_train_raw = split.training_pool[feature_cols]
    y_train = split.training_pool[TARGET_COLUMN].to_numpy()
    X_test_raw = split.test_set_quantum[feature_cols]
    y_test = split.test_set_quantum[TARGET_COLUMN].to_numpy()

    models = get_available_models(preprocessing_config.random_seed)
    results = {}
    search_seconds = {}
    for key, spec in models.items():
        if key == "rbf_svm":
            print("[phase11] classical: SKIPPING rbf_svm full-scale grid search "
                  "(benchmarked infeasible -- see docs/PHASE_11_FULL_SCALE.md)", flush=True)
            continue
        n_combos = 1
        for v in spec.param_grid.values():
            n_combos *= len(v)
        print(f"[phase11] classical: starting {key} grid search ({n_combos} combinations x {CV_FOLDS} folds, "
              f"n_train={len(X_train_raw)}) ...", flush=True)
        t0 = time.perf_counter()
        search = run_grid_search(
            spec, "classical", feature_groups, preprocessing_config,
            X_train_raw, y_train, cv_folds=CV_FOLDS, cv_shuffle=CV_SHUFFLE,
            random_seed=preprocessing_config.random_seed,
        )
        search_seconds[key] = time.perf_counter() - t0
        result = evaluate_on_test(
            search, "phase11_full_scale", key, spec.display_name,
            "classical", X_test_raw, y_test, n_train=len(X_train_raw),
            random_seed=preprocessing_config.random_seed,
        )
        results[key] = result
        print(f"[phase11] classical: finished {key} in {search_seconds[key]:.1f}s "
              f"(roc_auc={result.test_metrics['roc_auc']:.4f}, best_params={result.best_params})", flush=True)
    return results, search_seconds


def run_hybrid_full_scale(split, processed, hybrid_config: HybridConfig) -> tuple[HybridQuantumFeatureLayer, dict]:
    """theta selected on the n_train=2,000 screening subset (documented
    deviation, module docstring), then quantum features computed ONCE for
    the full training pool and a fresh LogisticRegression fit on the full
    concatenated [classical, quantum] feature matrix."""
    print("[phase11] selecting hybrid quantum weights theta via COBYLA on the "
          "n_train=2,000 screening subset (train-only; SAME subset Phase 10 used) ...", flush=True)
    screen = build_screening_split()  # n=2,000, seed=42, zero overlap with test -- verified in Phase 8A's own tests
    screen_processed = process_stage(
        screen.train_df, split.test_set_classical, split.test_set_quantum,
        TARGET_COLUMN, get_cardio_feature_groups(), PreprocessingConfig(),
    )
    theta_selector = HybridQuantumFeatureLayer(hybrid_config)
    t0 = time.perf_counter()
    theta_selector.fit(screen_processed.X_train_quantum, screen_processed.y_train)
    theta_selection_seconds = time.perf_counter() - t0
    theta = theta_selector.fit_result_.theta
    print(f"[phase11] theta selected in {theta_selection_seconds:.1f}s "
          f"({theta_selector.fit_result_.n_function_evaluations} evals, theta={theta})", flush=True)

    print(f"[phase11] computing quantum features for the FULL training pool (n={len(processed.X_train_quantum)}) "
          f"at the selected theta (ONE forward pass, no further optimization) ...", flush=True)
    t1 = time.perf_counter()
    train_quantum_features = theta_selector.quantum_features(processed.X_train_quantum, theta)
    full_batch_forward_seconds = time.perf_counter() - t1
    assert np.all(train_quantum_features >= -1.0 - 1e-6) and np.all(train_quantum_features <= 1.0 + 1e-6), \
        "quantum expectation values out of the physically valid [-1, 1] range"

    t2 = time.perf_counter()
    Z_train = np.hstack([processed.X_train_quantum, train_quantum_features])
    final_lr = LogisticRegression(C=hybrid_config.lr_C, max_iter=1000, random_state=hybrid_config.seed)
    final_lr.fit(Z_train, processed.y_train)
    final_fit_seconds = time.perf_counter() - t2

    # Wire the fitted pieces into a HybridQuantumFeatureLayer-shaped object for reuse of predict_proba's logic.
    theta_selector.fit_result_.theta = theta  # unchanged, kept for clarity
    theta_selector.lr_ = final_lr

    timing = {
        "theta_selection_subset_n": len(screen.train_df),
        "theta_selection_seconds": theta_selection_seconds,
        "theta_selection_n_evals": theta_selector.fit_result_.n_function_evaluations,
        "full_batch_quantum_forward_seconds": full_batch_forward_seconds,
        "final_lr_fit_seconds": final_fit_seconds,
        "full_train_n": len(processed.X_train_quantum),
        "quantum_feature_stats": {
            "min": float(train_quantum_features.min()), "max": float(train_quantum_features.max()),
            "mean": train_quantum_features.mean(axis=0).tolist(), "std": train_quantum_features.std(axis=0).tolist(),
        },
    }
    return theta_selector, timing


def run_phase11(*, hybrid_config: HybridConfig | None = None, make_plots: bool = True) -> dict:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    runtime: dict = {}
    hybrid_config = hybrid_config or HybridConfig()

    print("[phase11] loading Stage D (RBF-SVM, n=20,000) and Phase 9 (QSVM, n=2,000) references ...", flush=True)
    refs = load_reference_predictions()

    print("[phase11] rebuilding the full fixed split (training_pool + fixed test sets) ...", flush=True)
    t0 = time.perf_counter()
    split = build_full_scale_split()
    runtime["split_seconds"] = time.perf_counter() - t0
    n_train_full = len(split.training_pool)
    print(f"[phase11] full training pool size: {n_train_full}", flush=True)

    cmp_ids = split.test_set_quantum["id"].tolist()
    fingerprint = comparison_set_fingerprint(cmp_ids)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"Comparison set fingerprint {fingerprint} != expected {EXPECTED_FINGERPRINT}")
    for name, df in refs.items():
        if df["id"].tolist() != cmp_ids:
            raise ValueError(f"{name} reference predictions are not aligned to the current comparison set.")

    fg = get_cardio_feature_groups()
    pcfg = PreprocessingConfig()

    print(f"[phase11] ===== classical full-scale benchmark (n_train={n_train_full}) =====", flush=True)
    t1 = time.perf_counter()
    classical_results, classical_search_seconds = run_classical_full_scale(split, fg, pcfg)
    runtime["classical_total_seconds"] = time.perf_counter() - t1
    runtime["classical_per_model_seconds"] = classical_search_seconds

    print("[phase11] fitting preprocessing/PCA on the FULL training pool (train-only) ...", flush=True)
    t2 = time.perf_counter()
    processed = process_stage(
        split.training_pool, split.test_set_classical, split.test_set_quantum,
        TARGET_COLUMN, fg, pcfg,
    )
    runtime["preprocessing_and_pca_seconds"] = time.perf_counter() - t2
    y_true = processed.y_test_quantum

    all_models: dict[str, dict] = {}
    for key, r in classical_results.items():
        all_models[key] = {
            "display_name": f"{r.display_name} (full n={n_train_full})", "kind": "classical_full",
            "y_proba": r.y_test_proba, "fit_time_seconds": classical_search_seconds[key],
        }
    all_models["rbf_svm_n20000"] = {
        "display_name": "RBF-SVM (Stage D, n=20,000 -- reused, full-scale infeasible)", "kind": "classical_reused",
        "y_proba": refs["stage_d"]["rbf_svm_proba"].to_numpy(), "fit_time_seconds": None,
    }
    all_models["baseline_qsvm_n2000"] = {
        "display_name": "Baseline QSVM (Phase 9, n=2,000 -- reused, full-scale infeasible)", "kind": "quantum_reused",
        "y_proba": refs["phase9"]["baseline_proba"].to_numpy(), "fit_time_seconds": None,
    }
    all_models["mi_adaptive_qsvm_n2000"] = {
        "display_name": "MI-adaptive QSVM (Phase 9, n=2,000 -- reused, full-scale infeasible)", "kind": "quantum_reused",
        "y_proba": refs["phase9"]["mi_adaptive_proba"].to_numpy(), "fit_time_seconds": None,
    }

    # ---- control (Model F): full-scale, non-quantum ----
    print(f"[phase11] fitting Model F (non-quantum control: PCA-4 + 4 fixed random features, n={n_train_full}) ...", flush=True)
    rng = np.random.RandomState(hybrid_config.seed)
    random_train = rng.normal(size=(len(processed.X_train_quantum), hybrid_config.n_qubits))
    random_test = np.random.RandomState(hybrid_config.seed + 1).normal(size=(len(processed.X_test_quantum), hybrid_config.n_qubits))
    t3 = time.perf_counter()
    lr_control = LogisticRegression(C=hybrid_config.lr_C, max_iter=1000, random_state=hybrid_config.seed)
    lr_control.fit(np.hstack([processed.X_train_quantum, random_train]), processed.y_train)
    runtime["control_fit_seconds"] = time.perf_counter() - t3
    control_proba = lr_control.predict_proba(np.hstack([processed.X_test_quantum, random_test]))[:, 1]
    all_models["control_random_features"] = {
        "display_name": f"Model F: Control (PCA-4 + 4 fixed random features, full n={n_train_full})",
        "kind": "control_full", "y_proba": control_proba, "fit_time_seconds": runtime["control_fit_seconds"],
    }

    # ---- hybrid (Model E): full-scale ----
    print(f"[phase11] ===== training Model E: hybrid at full scale (n_train={n_train_full}) =====", flush=True)
    t4 = time.perf_counter()
    hybrid, hybrid_timing = run_hybrid_full_scale(split, processed, hybrid_config)
    runtime["hybrid_total_seconds"] = time.perf_counter() - t4
    runtime["hybrid_timing"] = hybrid_timing
    hybrid_proba = hybrid.predict_proba(processed.X_test_quantum)[:, 1]
    all_models["hybrid"] = {
        "display_name": f"Model E: Hybrid (PCA-4 + trained quantum features -> LR, full n={n_train_full})",
        "kind": "hybrid_full", "y_proba": hybrid_proba, "fit_time_seconds": runtime["hybrid_total_seconds"],
    }
    print(f"[phase11] hybrid full-scale done in {runtime['hybrid_total_seconds']:.1f}s "
          f"(roc_auc={_metrics(y_true, hybrid_proba)['roc_auc']:.4f})", flush=True)

    for m in all_models.values():
        m["y_true"] = y_true
        m["y_pred"] = (m["y_proba"] >= DECISION_THRESHOLD).astype(int)
        m["metrics"] = _metrics(y_true, m["y_proba"])
        assert not np.any(np.isnan(m["y_proba"])) and not np.any(np.isinf(m["y_proba"]))

    # ---- statistics ----
    print("[phase11] computing paired statistical comparisons ...", flush=True)
    best_classical_full_key = max(
        (k for k, v in all_models.items() if v["kind"] == "classical_full"),
        key=lambda k: all_models[k]["metrics"]["roc_auc"],
    )

    def _pair(key_a: str, key_b: str) -> dict:
        pa, pb = all_models[key_a]["y_proba"], all_models[key_b]["y_proba"]
        preda, predb = all_models[key_a]["y_pred"], all_models[key_b]["y_pred"]
        dl = delong_test(y_true, pa, pb)
        bs = paired_bootstrap_delta(y_true, pa, pb, "roc_auc", n_resamples=2000, seed=42)
        bs_pr = paired_bootstrap_delta(y_true, pa, pb, "pr_auc", n_resamples=2000, seed=42)
        mc = mcnemar_exact(y_true, preda, predb)
        return {"a": key_a, "b": key_b, "delong": dl.to_dict(), "bootstrap_roc_auc": bs, "bootstrap_pr_auc": bs_pr, "mcnemar": mc}

    comparisons = {
        "best_full_classical_vs_hybrid": _pair(best_classical_full_key, "hybrid"),
        "hybrid_vs_control": _pair("hybrid", "control_random_features"),
        "hybrid_vs_baseline_qsvm_n2000": _pair("hybrid", "baseline_qsvm_n2000"),
        "hybrid_vs_mi_adaptive_qsvm_n2000": _pair("hybrid", "mi_adaptive_qsvm_n2000"),
    }
    raw_p = [c["delong"]["p_value"] for c in comparisons.values()]
    holm = holm_bonferroni(raw_p)
    for (name, c), h in zip(comparisons.items(), holm):
        c["delong_p_holm"] = h["adjusted_p"]
        c["delong_significant_holm"] = h["significant"]

    # ---- decision framework (A/B/C/D) ----
    hybrid_vs_control = comparisons["hybrid_vs_control"]
    hybrid_vs_classical = comparisons["best_full_classical_vs_hybrid"]
    beats_control = hybrid_vs_control["delong"]["delta"] > 0 and hybrid_vs_control["delong_significant_holm"] \
        and not hybrid_vs_control["bootstrap_roc_auc"]["ci_includes_zero"]
    # delta stored as (a-b) = (best_classical - hybrid); hybrid beats classical iff delta < 0
    hybrid_minus_classical = -hybrid_vs_classical["delong"]["delta"]
    beats_classical = hybrid_minus_classical > 0 and hybrid_vs_classical["delong_significant_holm"] \
        and not hybrid_vs_classical["bootstrap_roc_auc"]["ci_includes_zero"]

    n2000_ranking = ["classical_lr", "xgboost", "random_forest", "rbf_svm"]  # Phase 9's descending order
    full_ranking = sorted(
        (k for k, v in all_models.items() if v["kind"] == "classical_full"),
        key=lambda k: -all_models[k]["metrics"]["roc_auc"],
    )
    ranking_changed = full_ranking[0] != "classical_lr"

    if beats_control and beats_classical:
        outcome, outcome_text = "A", "Hybrid significantly beats the control AND the strongest classical model -- evidence of useful quantum contribution."
    elif beats_control and not beats_classical:
        outcome, outcome_text = "B", "Hybrid beats the control but not the strongest classical model -- scientifically interesting, no demonstrated practical advantage."
    elif not beats_control:
        outcome, outcome_text = "C", "Hybrid does not beat the equivalent non-quantum control -- no demonstrated useful value from the quantum transform."
    else:
        outcome, outcome_text = "C", "Hybrid does not beat the equivalent non-quantum control."
    if ranking_changed:
        outcome_text += f" ADDITIONALLY (Outcome D): the full-data classical ranking changed (best classical at n=2,000 was Logistic Regression; at full scale it is {all_models[full_ranking[0]]['display_name']})."

    decision = {
        "outcome": outcome, "outcome_text": outcome_text,
        "beats_control": bool(beats_control), "beats_best_full_classical": bool(beats_classical),
        "best_full_classical_model": best_classical_full_key, "ranking_changed_from_n2000": bool(ranking_changed),
        "full_classical_ranking_by_roc_auc": full_ranking,
    }

    # ---- training-scale comparison table ----
    phase9_metrics = pd.read_csv(PHASE9_PREDICTIONS_PATH.parent / "metrics.csv").set_index("model_key")
    phase10_metrics = pd.read_csv(PHASE10_METRICS_PATH).set_index("model_key") if PHASE10_METRICS_PATH.is_file() else None
    scale_rows = []
    pairs = [
        ("logistic_regression", "Logistic Regression", "classical_lr" if "classical_lr" in phase9_metrics.index else "logistic_regression"),
        ("rbf_svm", "RBF-SVM", "rbf_svm"),
        ("random_forest", "Random Forest", "random_forest"),
        ("xgboost", "XGBoost", "xgboost"),
    ]
    for p9_key, label, p11_key in pairs:
        n2000_row = phase9_metrics.loc[p9_key] if p9_key in phase9_metrics.index else None
        if p11_key in all_models:
            full_row = all_models[p11_key]["metrics"]
        elif p11_key == "rbf_svm":
            full_row = all_models["rbf_svm_n20000"]["metrics"]
        else:
            full_row = None
        if n2000_row is not None and full_row is not None:
            scale_rows.append({
                "Model": label, "n2000_roc_auc": float(n2000_row["roc_auc"]), "full_roc_auc": full_row["roc_auc"],
                "delta_roc_auc": full_row["roc_auc"] - float(n2000_row["roc_auc"]),
                "n2000_pr_auc": float(n2000_row["pr_auc"]), "full_pr_auc": full_row["pr_auc"],
                "delta_pr_auc": full_row["pr_auc"] - float(n2000_row["pr_auc"]),
                "n2000_sensitivity": float(n2000_row["sensitivity"]), "full_sensitivity": full_row["sensitivity"],
                "n2000_specificity": float(n2000_row["specificity"]), "full_specificity": full_row["specificity"],
                "n2000_accuracy": float(n2000_row["accuracy"]), "full_accuracy": full_row["accuracy"],
                "n2000_f1": float(n2000_row["f1"]), "full_f1": full_row["f1"],
                "note": "RBF-SVM full-scale value is Stage D's n=20,000 result, not n=66,641 (infeasible; see docs)." if label == "RBF-SVM" else "",
            })
    if phase10_metrics is not None and "hybrid" in phase10_metrics.index:
        scale_rows.append({
            "Model": "Hybrid", "n2000_roc_auc": float(phase10_metrics.loc["hybrid", "roc_auc"]),
            "full_roc_auc": all_models["hybrid"]["metrics"]["roc_auc"],
            "delta_roc_auc": all_models["hybrid"]["metrics"]["roc_auc"] - float(phase10_metrics.loc["hybrid", "roc_auc"]),
            "n2000_pr_auc": float(phase10_metrics.loc["hybrid", "pr_auc"]), "full_pr_auc": all_models["hybrid"]["metrics"]["pr_auc"],
            "delta_pr_auc": all_models["hybrid"]["metrics"]["pr_auc"] - float(phase10_metrics.loc["hybrid", "pr_auc"]),
            "n2000_sensitivity": float(phase10_metrics.loc["hybrid", "sensitivity"]), "full_sensitivity": all_models["hybrid"]["metrics"]["sensitivity"],
            "n2000_specificity": float(phase10_metrics.loc["hybrid", "specificity"]), "full_specificity": all_models["hybrid"]["metrics"]["specificity"],
            "n2000_accuracy": float(phase10_metrics.loc["hybrid", "accuracy"]), "full_accuracy": all_models["hybrid"]["metrics"]["accuracy"],
            "n2000_f1": float(phase10_metrics.loc["hybrid", "f1"]), "full_f1": all_models["hybrid"]["metrics"]["f1"],
            "note": "",
        })
        scale_rows.append({
            "Model": "Control", "n2000_roc_auc": float(phase10_metrics.loc["control_random_features", "roc_auc"]),
            "full_roc_auc": all_models["control_random_features"]["metrics"]["roc_auc"],
            "delta_roc_auc": all_models["control_random_features"]["metrics"]["roc_auc"] - float(phase10_metrics.loc["control_random_features", "roc_auc"]),
            "n2000_pr_auc": float(phase10_metrics.loc["control_random_features", "pr_auc"]), "full_pr_auc": all_models["control_random_features"]["metrics"]["pr_auc"],
            "delta_pr_auc": all_models["control_random_features"]["metrics"]["pr_auc"] - float(phase10_metrics.loc["control_random_features", "pr_auc"]),
            "n2000_sensitivity": float(phase10_metrics.loc["control_random_features", "sensitivity"]), "full_sensitivity": all_models["control_random_features"]["metrics"]["sensitivity"],
            "n2000_specificity": float(phase10_metrics.loc["control_random_features", "specificity"]), "full_specificity": all_models["control_random_features"]["metrics"]["specificity"],
            "n2000_accuracy": float(phase10_metrics.loc["control_random_features", "accuracy"]), "full_accuracy": all_models["control_random_features"]["metrics"]["accuracy"],
            "n2000_f1": float(phase10_metrics.loc["control_random_features", "f1"]), "full_f1": all_models["control_random_features"]["metrics"]["f1"],
            "note": "",
        })
    scale_df = pd.DataFrame(scale_rows)

    # ---- persist ----
    pred_df = pd.DataFrame({"id": cmp_ids, "y_true": y_true})
    for key, m in all_models.items():
        pred_df[f"{key}_proba"] = m["y_proba"]
        pred_df[f"{key}_pred"] = m["y_pred"]
    pred_df.to_csv(RESULTS_ROOT / "predictions.csv", index=False)
    assert not pred_df.isna().any().any(), "NaN found in Phase 11 predictions.csv"

    metrics_rows = [{"Model": m["display_name"], "model_key": key, "kind": m["kind"],
                      **{k: m["metrics"][k] for k in ("roc_auc", "pr_auc", "sensitivity", "specificity", "accuracy", "precision", "f1", "tn", "fp", "fn", "tp")},
                      "runtime_seconds": m["fit_time_seconds"]} for key, m in all_models.items()]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULTS_ROOT / "metrics.csv", index=False)

    comparison_table = metrics_df[["Model", "roc_auc", "pr_auc", "sensitivity", "specificity", "accuracy", "f1", "runtime_seconds"]].copy()
    comparison_table.columns = ["Model", "ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "Accuracy", "F1", "Runtime (s)"]
    comparison_table.to_csv(RESULTS_ROOT / "comparison_table.csv", index=False)
    scale_df.to_csv(RESULTS_ROOT / "training_scale_comparison.csv", index=False)

    model_configs = {
        "rbf_svm_frozen_params": RBF_SVM_FROZEN_PARAMS,
        "hybrid_config": hybrid_config.to_dict(),
        "classical_best_params": {key: r.best_params for key, r in classical_results.items()},
        "seed": SAMPLING_SEED,
    }
    with open(RESULTS_ROOT / "model_configs.json", "w", encoding="utf-8") as fh:
        json.dump(model_configs, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "runtime.json", "w", encoding="utf-8") as fh:
        json.dump(runtime, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "statistical_comparisons.json", "w", encoding="utf-8") as fh:
        json.dump(comparisons, fh, indent=2, default=float)

    feature_report = hybrid.feature_report()
    explainability = {
        "feature_report": feature_report.to_dict(),
        "logistic_regression_coefficients": dict(zip(feature_report.feature_names, hybrid.lr_.coef_[0].tolist())),
        "intercept": float(hybrid.lr_.intercept_[0]),
        "selected_quantum_weights_theta": hybrid.fit_result_.theta.tolist(),
    }

    summary = {
        "n_train_full": n_train_full,
        "comparison_set": {"n": len(cmp_ids), "fingerprint": fingerprint, "matches_expected": fingerprint == EXPECTED_FINGERPRINT},
        "metrics": {key: m["metrics"] for key, m in all_models.items()},
        "statistics": comparisons,
        "runtime": runtime,
        "decision": decision,
        "explainability": explainability,
        "documented_deviations": {
            "rbf_svm_full_scale_skipped": "Single fit with frozen best params (C=10.0, gamma=0.01) exceeded 20 minutes with no completion; Stage D's n=20,000 result reused instead.",
            "qsvm_full_scale_skipped": f"Train-train kernel at n={n_train_full} would require {n_train_full**2 * 4 / 1e9:.1f} GB (float32); infeasible on this machine. Phase 9's n=2,000 results reused for reference.",
            "learning_curve_sweep_skipped": "Not run; the core n=2,000-vs-full-scale question is answered by this experiment's two-point comparison. See module docstring.",
        },
    }
    with open(RESULTS_ROOT / "phase11_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"[phase11] DECISION: outcome {outcome} -- {outcome_text}", flush=True)

    if make_plots:
        _plot_curves(all_models, RESULTS_ROOT)

    return summary


def _plot_curves(all_models: dict, out_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    items = list(all_models.items())
    n_cmp = len(items[0][1]["y_true"])

    fig, ax = plt.subplots(figsize=(9, 8))
    for key, m in items:
        fpr, tpr, _ = roc_curve(m["y_true"], m["y_proba"])
        ax.plot(fpr, tpr, label=f"{m['display_name']} (AUC={m['metrics']['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC -- Phase 11 full-scale experiment\nIdentical {n_cmp}-row held-out test set")
    ax.legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    for key, m in items:
        prec, rec, _ = precision_recall_curve(m["y_true"], m["y_proba"])
        ax.plot(rec, prec, label=f"{m['display_name']} (AP={m['metrics']['pr_auc']:.4f})")
    base = float(np.mean(items[0][1]["y_true"]))
    ax.axhline(base, ls="--", color="grey", label=f"Chance ({base:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall -- Phase 11 full-scale experiment")
    ax.legend(loc="lower left", fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=150)
    plt.close(fig)

    n = len(items)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 7))
    for ax, (key, m) in zip(axes.flat, items):
        mm = m["metrics"]
        cm = np.array([[mm["tn"], mm["fp"]], [mm["fn"], mm["tp"]]])
        ConfusionMatrixDisplay(cm, display_labels=["No CVD", "CVD"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(m["display_name"], fontsize=6)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"Confusion matrices -- Phase 11, identical {n_cmp}-row test set")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    result = run_phase11()
    print(json.dumps({k: v for k, v in result.items() if k not in ("statistics", "explainability")}, indent=2, default=float))
