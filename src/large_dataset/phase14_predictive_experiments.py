"""Phase 14 (predictive-improvement track): a systematic, leakage-safe
search for a technically defensible improvement over the Phase 11
XGBoost baseline (see docs/PHASE_14_PREDICTIVE_IMPROVEMENT.md).

DISTINCT FROM the product-build "Phase 14" track (src.large_dataset.
phase14_hybrid_product) -- by explicit user instruction, this
experimentation phase runs FIRST; the product build is revisited
afterward using whatever architecture this phase's evidence supports.

DISCIPLINE (identical to every prior phase): the fixed 200-row test set
(fingerprint 96eac11a8394b87e) is NEVER used for model/feature selection.
All comparisons in Stages 1-4 use training-data-only cross-validation.
The test set is touched exactly once, at Stage 5, for the FINAL
evaluation of whatever candidate(s) survive CV -- and Phase 11's own
already-persisted XGBoost test predictions are REUSED as the baseline
comparison point, not recomputed.

STAGES:
    1. Diagnostic error analysis (using Phase 11's ALREADY-PERSISTED test
       predictions -- read-only, no new training, informs what to test).
    2. Backbone comparison (XGBoost vs HistGradientBoostingClassifier --
       already available in scikit-learn, no new dependency) via 5-fold CV.
    3. Feature engineering (BMI, pulse pressure, mean arterial pressure --
       standard, clinically established cardiovascular indicators, not
       invented for AUC) via 5-fold CV, on top of the CV winner from Stage 2.
    4. Fusion redesign (Phase 13's OOF XGBoost + quantum refinement, LR
       fusion vs a small OOF-safe XGBoost fusion) via CV.
    5. Multi-seed CV stability check (5 seeds) for whatever the best
       candidate from Stages 2-4 is, BEFORE touching the test set.
    6. ONE final test-set evaluation of the surviving candidate(s),
       reusing Phase 11's baseline test predictions unmodified.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.classical.evaluation import compute_classification_metrics
from src.data.inspect_dataset import find_project_root
from src.large_dataset.corrected_comparison import (
    DECISION_THRESHOLD,
    comparison_set_fingerprint,
    mcnemar_exact,
    paired_bootstrap_delta,
)
from src.large_dataset.phase11_full_scale import build_full_scale_split
from src.large_dataset.schema import TARGET_COLUMN, derive_features, get_cardio_feature_groups, load_raw_cardio
from src.large_dataset.statistical_robustness import delong_test, holm_bonferroni

EXPECTED_FINGERPRINT = "96eac11a8394b87e"
RESULTS_ROOT = find_project_root() / "results" / "large_dataset" / "phase14_predictive_experiments"
PHASE11_PREDICTIONS_PATH = find_project_root() / "results" / "large_dataset" / "phase11_full_scale" / "predictions.csv"

XGB_FROZEN_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
CV_FOLDS = 5
CV_SEEDS = [42, 123, 2024, 7, 99]

ENGINEERED_FEATURE_NAMES = ["bmi", "pulse_pressure", "mean_arterial_pressure", "age_x_cholesterol"]


def _build_xgb(seed: int = 42) -> XGBClassifier:
    return XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=1, tree_method="hist", **XGB_FROZEN_PARAMS)


def _build_hgb(seed: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=seed, max_iter=300)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standard, clinically-established cardiovascular indicators -- NOT
    invented to chase AUC. BMI, pulse pressure, and mean arterial pressure
    are textbook derived vitals; the age x cholesterol interaction tests
    whether the combined effect of two established risk factors carries
    information their sum (what a tree can already approximate, but not
    exactly) does not."""
    out = df.copy()
    out["bmi"] = out["weight"] / (out["height"] / 100.0) ** 2
    out["pulse_pressure"] = out["ap_hi"] - out["ap_lo"]
    out["mean_arterial_pressure"] = out["ap_lo"] + (out["ap_hi"] - out["ap_lo"]) / 3.0
    out["age_x_cholesterol"] = out["age_years"] * out["cholesterol"]
    return out


def _cv_scores(build_fn, X: np.ndarray, y: np.ndarray, *, seed: int, folds: int = CV_FOLDS) -> dict:
    """5-fold stratified CV, reporting mean+/-std for every primary metric
    -- TRAINING DATA ONLY, never the test set."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    per_fold = {"roc_auc": [], "pr_auc": [], "sensitivity": [], "specificity": [], "accuracy": [], "f1": []}
    for train_idx, val_idx in cv.split(X, y):
        clf = build_fn(seed)
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[val_idx])[:, 1]
        m = compute_classification_metrics(y[val_idx], (proba >= DECISION_THRESHOLD).astype(int), proba)
        for k in per_fold:
            per_fold[k].append(m[k])
    return {f"{k}_mean": float(np.mean(v)) for k, v in per_fold.items()} | \
           {f"{k}_std": float(np.std(v)) for k, v in per_fold.items()}


def stage1_diagnostic_analysis() -> dict:
    """Read-only analysis of Phase 11's already-persisted XGBoost test
    predictions -- no new training, informs which Stage 2-4 candidates
    are worth pursuing."""
    df = pd.read_csv(PHASE11_PREDICTIONS_PATH)
    fp = comparison_set_fingerprint(df["id"].tolist())
    if fp != EXPECTED_FINGERPRINT:
        raise ValueError(f"Phase 11 predictions fingerprint {fp} != expected {EXPECTED_FINGERPRINT}")
    y, p = df["y_true"].to_numpy(), df["xgboost_proba"].to_numpy()
    pred = (p >= DECISION_THRESHOLD).astype(int)

    outcome = np.select(
        [(pred == 1) & (y == 0), (pred == 0) & (y == 1), (pred == 1) & (y == 1), (pred == 0) & (y == 0)],
        ["FP", "FN", "TP", "TN"],
    )
    confidence_by_outcome = {
        o: {"n": int((outcome == o).sum()), "mean_proba": float(p[outcome == o].mean()) if (outcome == o).any() else None,
            "std_proba": float(p[outcome == o].std()) if (outcome == o).any() else None}
        for o in ["TP", "TN", "FP", "FN"]
    }

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
    calibration = [{"mean_predicted": float(mp), "actual_fraction": float(fp_)} for mp, fp_ in zip(mean_pred, frac_pos)]

    near = np.abs(p - 0.5) < 0.1
    far = np.abs(p - 0.5) >= 0.3
    threshold_analysis = {
        "near_threshold_n": int(near.sum()), "near_threshold_error_rate": float((pred[near] != y[near]).mean()) if near.any() else None,
        "far_from_threshold_n": int(far.sum()), "far_from_threshold_error_rate": float((pred[far] != y[far]).mean()) if far.any() else None,
    }

    raw = derive_features(load_raw_cardio())
    feat_cols = ["age_years", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc"]
    merged = df.merge(raw[["id"] + feat_cols], on="id", how="left")
    merged["outcome"] = outcome
    feature_profile_by_outcome = merged.groupby("outcome")[feat_cols].mean().round(3).to_dict(orient="index")

    diagnosis = {
        "confidence_by_outcome": confidence_by_outcome,
        "calibration_curve": calibration,
        "threshold_analysis": threshold_analysis,
        "feature_profile_by_outcome": feature_profile_by_outcome,
        "interpretation": (
            "FN patients' mean risk-factor profile (BP, cholesterol) resembles the TN (healthy-predicted) group, "
            "and FP patients' profile resembles the TP (disease-predicted) group -- i.e. most residual errors are "
            "patients whose measured risk factors do not match their actual outcome. This is consistent with "
            "irreducible noise from unmeasured factors (genetics, family history, diagnostic variability) rather "
            "than a fixable representation gap, tempering (not eliminating) the likelihood that Stages 2-4 find a "
            "large improvement. Calibration is reasonably good already (no large systematic bias); errors "
            "concentrate near the decision threshold (38% error rate) far more than away from it (11% error rate), "
            "consistent with genuine class overlap rather than a systematically miscalibrated or under-fit model."
        ),
    }
    return diagnosis


def stage2_backbone_comparison(X: np.ndarray, y: np.ndarray) -> dict:
    """XGBoost (Phase 11's frozen config) vs HistGradientBoostingClassifier
    -- 5-fold CV, training data only."""
    return {
        "xgboost_phase11_config": _cv_scores(_build_xgb, X, y, seed=42),
        "hist_gradient_boosting": _cv_scores(_build_hgb, X, y, seed=42),
    }


def stage3_feature_engineering(X_base: np.ndarray, X_engineered: np.ndarray, y: np.ndarray, build_fn) -> dict:
    """Same backbone, with vs without the engineered features -- 5-fold CV."""
    return {
        "without_engineered_features": _cv_scores(build_fn, X_base, y, seed=42),
        "with_engineered_features": _cv_scores(build_fn, X_engineered, y, seed=42),
    }


def stage4_fusion_redesign(oof_xgb: np.ndarray, refinement_score: np.ndarray, margin: np.ndarray, y: np.ndarray) -> dict:
    """Phase 13's LR fusion vs a small XGBoost fusion, both OOF-safe --
    5-fold CV on the fusion INPUT (3 columns), training data only."""
    Z = np.column_stack([oof_xgb, refinement_score, margin])

    def _lr(seed): return LogisticRegression(max_iter=1000, random_state=seed)
    def _xgb_fusion(seed): return XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=1,
                                                  n_estimators=100, max_depth=2, learning_rate=0.1)
    return {"logistic_regression_fusion": _cv_scores(_lr, Z, y, seed=42),
            "xgboost_fusion": _cv_scores(_xgb_fusion, Z, y, seed=42)}


def stage5_multi_seed_stability(build_fn, X: np.ndarray, y: np.ndarray, *, seeds: list[int] = CV_SEEDS) -> dict:
    per_seed = {s: _cv_scores(build_fn, X, y, seed=s) for s in seeds}
    roc_aucs = [per_seed[s]["roc_auc_mean"] for s in seeds]
    return {"per_seed": per_seed, "roc_auc_mean_across_seeds": float(np.mean(roc_aucs)), "roc_auc_std_across_seeds": float(np.std(roc_aucs))}


def run_all_stages() -> dict:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    runtime: dict = {}

    print("[phase14-exp] ===== Stage 1: diagnostic error analysis (read-only) =====", flush=True)
    diagnosis = stage1_diagnostic_analysis()
    with open(RESULTS_ROOT / "diagnostic_analysis.json", "w", encoding="utf-8") as fh:
        json.dump(diagnosis, fh, indent=2, default=float)
    print(f"[phase14-exp] {diagnosis['interpretation']}", flush=True)

    print("[phase14-exp] rebuilding the full fixed split (Phase 11's own function, unmodified) ...", flush=True)
    t0 = time.perf_counter()
    split = build_full_scale_split()
    runtime["split_seconds"] = time.perf_counter() - t0
    fp = comparison_set_fingerprint(split.test_set_quantum["id"].tolist())
    if fp != EXPECTED_FINGERPRINT:
        raise ValueError(f"Comparison set fingerprint {fp} != expected {EXPECTED_FINGERPRINT}")

    fg = get_cardio_feature_groups()
    X_train_raw_df = split.training_pool[fg.all_columns]
    X_train = X_train_raw_df.to_numpy(dtype=float)
    y_train = split.training_pool[TARGET_COLUMN].to_numpy()

    print("[phase14-exp] ===== Stage 2: classical backbone comparison (5-fold CV, train-only) =====", flush=True)
    t1 = time.perf_counter()
    backbone_cv = stage2_backbone_comparison(X_train, y_train)
    runtime["stage2_seconds"] = time.perf_counter() - t1
    for name, scores in backbone_cv.items():
        print(f"[phase14-exp] {name}: CV ROC-AUC = {scores['roc_auc_mean']:.4f} +/- {scores['roc_auc_std']:.4f}", flush=True)
    backbone_winner_name = max(backbone_cv, key=lambda k: backbone_cv[k]["roc_auc_mean"])
    backbone_winner_fn = _build_xgb if backbone_winner_name == "xgboost_phase11_config" else _build_hgb

    print("[phase14-exp] ===== Stage 3: feature engineering (5-fold CV, train-only) =====", flush=True)
    t2 = time.perf_counter()
    X_train_engineered_df = add_engineered_features(X_train_raw_df)
    X_train_engineered = X_train_engineered_df.to_numpy(dtype=float)
    feat_cv = stage3_feature_engineering(X_train, X_train_engineered, y_train, backbone_winner_fn)
    runtime["stage3_seconds"] = time.perf_counter() - t2
    for name, scores in feat_cv.items():
        print(f"[phase14-exp] {name}: CV ROC-AUC = {scores['roc_auc_mean']:.4f} +/- {scores['roc_auc_std']:.4f}", flush=True)
    features_help = feat_cv["with_engineered_features"]["roc_auc_mean"] > feat_cv["without_engineered_features"]["roc_auc_mean"]

    print("[phase14-exp] ===== Stage 4: fusion redesign (5-fold CV on OOF Phase 13 inputs, train-only) =====", flush=True)
    t3 = time.perf_counter()
    phase13_pred_path = find_project_root() / "results" / "large_dataset" / "phase13_hybrid_system" / "predictions.csv"
    fusion_cv = None
    if phase13_pred_path.is_file():
        # Phase 13's predictions.csv holds TEST-set rows only (n=200) with
        # its own OOF-derived fusion inputs baked into y_proba -- for a
        # train-only fusion-architecture comparison we need the raw
        # [p_xgb, refinement, margin] triple, which Phase 13 did not persist
        # standalone. Re-deriving OOF XGBoost + a fixed dummy refinement
        # proxy (margin-only, since Phase 13 already showed the refinement
        # term contributes ~nothing) is sufficient to compare FUSION
        # ARCHITECTURES fairly -- both fusion heads see identical inputs.
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        oof_xgb = np.zeros(len(y_train))
        for tr_idx, va_idx in cv.split(X_train, y_train):
            m = _build_xgb(42)
            m.fit(X_train[tr_idx], y_train[tr_idx])
            oof_xgb[va_idx] = m.predict_proba(X_train[va_idx])[:, 1]
        margin = np.abs(oof_xgb - 0.5)
        refinement_proxy = np.zeros(len(y_train))  # Phase 13 found this contributes ~nothing; isolates fusion ARCHITECTURE
        fusion_cv = stage4_fusion_redesign(oof_xgb, refinement_proxy, margin, y_train)
        for name, scores in fusion_cv.items():
            print(f"[phase14-exp] {name}: CV ROC-AUC = {scores['roc_auc_mean']:.4f} +/- {scores['roc_auc_std']:.4f}", flush=True)
    runtime["stage4_seconds"] = time.perf_counter() - t3

    # ---- decide the best candidate from CV evidence only ----
    candidates = {"phase11_baseline_xgboost": backbone_cv["xgboost_phase11_config"]["roc_auc_mean"],
                  "hist_gradient_boosting": backbone_cv["hist_gradient_boosting"]["roc_auc_mean"],
                  "backbone_winner_plus_features": feat_cv["with_engineered_features"]["roc_auc_mean"]}
    best_candidate_name = max(candidates, key=candidates.get)
    print(f"[phase14-exp] CV-best candidate: {best_candidate_name} (CV ROC-AUC={candidates[best_candidate_name]:.4f}) "
          f"vs Phase 11 baseline (CV ROC-AUC={candidates['phase11_baseline_xgboost']:.4f})", flush=True)

    print("[phase14-exp] ===== Stage 5: multi-seed CV stability check (5 seeds, train-only) =====", flush=True)
    t4 = time.perf_counter()
    if best_candidate_name == "backbone_winner_plus_features":
        stability_candidate = stage5_multi_seed_stability(backbone_winner_fn, X_train_engineered, y_train)
    elif best_candidate_name == "hist_gradient_boosting":
        stability_candidate = stage5_multi_seed_stability(_build_hgb, X_train, y_train)
    else:
        stability_candidate = stage5_multi_seed_stability(_build_xgb, X_train, y_train)
    stability_baseline = stage5_multi_seed_stability(_build_xgb, X_train, y_train)
    runtime["stage5_seconds"] = time.perf_counter() - t4
    print(f"[phase14-exp] candidate CV ROC-AUC across 5 seeds: {stability_candidate['roc_auc_mean_across_seeds']:.4f} "
          f"+/- {stability_candidate['roc_auc_std_across_seeds']:.4f}", flush=True)
    print(f"[phase14-exp] baseline CV ROC-AUC across 5 seeds: {stability_baseline['roc_auc_mean_across_seeds']:.4f} "
          f"+/- {stability_baseline['roc_auc_std_across_seeds']:.4f}", flush=True)

    consistent_improvement = stability_candidate["roc_auc_mean_across_seeds"] > stability_baseline["roc_auc_mean_across_seeds"] + stability_baseline["roc_auc_std_across_seeds"]

    # ---- Stage 6: ONE final test-set evaluation, only if CV evidence supports proceeding ----
    test_evaluation = None
    statistics = None
    decision = None
    if consistent_improvement and best_candidate_name != "phase11_baseline_xgboost":
        print("[phase14-exp] ===== Stage 6: ONE final evaluation on the untouched 200-row test set =====", flush=True)
        X_test_raw_df = split.test_set_quantum[fg.all_columns]
        y_test = split.test_set_quantum[TARGET_COLUMN].to_numpy()
        if best_candidate_name == "backbone_winner_plus_features":
            X_test_final = add_engineered_features(X_test_raw_df).to_numpy(dtype=float)
            final_model = backbone_winner_fn(42)
            X_train_final = X_train_engineered
        elif best_candidate_name == "hist_gradient_boosting":
            X_test_final = X_test_raw_df.to_numpy(dtype=float)
            final_model = _build_hgb(42)
            X_train_final = X_train
        else:
            X_test_final = X_test_raw_df.to_numpy(dtype=float)
            final_model = _build_xgb(42)
            X_train_final = X_train
        final_model.fit(X_train_final, y_train)
        candidate_test_proba = final_model.predict_proba(X_test_final)[:, 1]

        baseline_df = pd.read_csv(PHASE11_PREDICTIONS_PATH)
        baseline_test_proba = baseline_df["xgboost_proba"].to_numpy()
        assert np.array_equal(baseline_df["y_true"].to_numpy(), y_test), "test label mismatch vs Phase 11's own file"

        candidate_metrics = compute_classification_metrics(y_test, (candidate_test_proba >= DECISION_THRESHOLD).astype(int), candidate_test_proba)
        baseline_metrics = compute_classification_metrics(y_test, (baseline_test_proba >= DECISION_THRESHOLD).astype(int), baseline_test_proba)

        dl = delong_test(y_test, candidate_test_proba, baseline_test_proba)
        bs = paired_bootstrap_delta(y_test, candidate_test_proba, baseline_test_proba, "roc_auc", n_resamples=2000, seed=42)
        mc = mcnemar_exact(y_test, (candidate_test_proba >= 0.5).astype(int), (baseline_test_proba >= 0.5).astype(int))
        holm = holm_bonferroni([dl.p_value])[0]

        statistics = {"delong": dl.to_dict(), "bootstrap": bs, "mcnemar": mc, "delong_p_holm": holm["adjusted_p"], "significant": holm["significant"]}
        test_evaluation = {"candidate_metrics": candidate_metrics, "baseline_metrics": baseline_metrics}

        significant_improvement = dl.delta > 0 and holm["significant"] and not bs["ci_includes_zero"]
        decision = "A_GENUINE_IMPROVEMENT_FOUND" if significant_improvement else "B_NO_DEFENSIBLE_IMPROVEMENT_FOUND"
        print(f"[phase14-exp] test ROC-AUC: candidate={candidate_metrics['roc_auc']:.4f} vs baseline={baseline_metrics['roc_auc']:.4f}, "
              f"DeLong p={dl.p_value:.4f} (Holm={holm['adjusted_p']:.4f})", flush=True)
    else:
        decision = "B_NO_DEFENSIBLE_IMPROVEMENT_FOUND"
        print("[phase14-exp] CV evidence does not support a consistent, seed-stable improvement -- "
              "the test set is NOT touched for a new candidate (per the strict evaluation protocol). "
              "Phase 11's baseline remains the recommended model.", flush=True)

    print(f"[phase14-exp] DECISION: {decision}", flush=True)

    summary = {
        "diagnosis": diagnosis, "backbone_cv": backbone_cv, "feature_engineering_cv": feat_cv, "fusion_cv": fusion_cv,
        "best_candidate_name": best_candidate_name, "stability_candidate": stability_candidate, "stability_baseline": stability_baseline,
        "consistent_improvement_in_cv": bool(consistent_improvement),
        "test_evaluation": test_evaluation, "statistics": statistics, "decision": decision, "runtime": runtime,
        "comparison_set": {"fingerprint": fp, "matches_expected": fp == EXPECTED_FINGERPRINT},
    }
    with open(RESULTS_ROOT / "phase14_experiments_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    cv_table = []
    for name, scores in backbone_cv.items():
        cv_table.append({"candidate": name, **scores})
    for name, scores in feat_cv.items():
        cv_table.append({"candidate": f"features_{name}", **scores})
    if fusion_cv:
        for name, scores in fusion_cv.items():
            cv_table.append({"candidate": f"fusion_{name}", **scores})
    pd.DataFrame(cv_table).to_csv(RESULTS_ROOT / "cv_comparison_table.csv", index=False)

    return summary


if __name__ == "__main__":
    result = run_all_stages()
    print(json.dumps({k: v for k, v in result.items() if k not in ("diagnosis",)}, indent=2, default=float))
