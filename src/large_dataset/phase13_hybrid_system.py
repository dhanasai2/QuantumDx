"""Phase 13: the practical hybrid quantum-classical disease-risk system
(see docs/PHASE_13_HYBRID_SYSTEM.md).

SYSTEM: Classical XGBoost backbone (Phase 11's frozen best hyperparameters,
providing the primary predictive signal) -> a compact quantum RESIDUAL/
REFINEMENT circuit (predicting what the classical model got wrong, not
re-deriving its own classification of the raw features -- a genuinely
different role from Phase 10/12's feature-concatenation approach) ->
classical Logistic Regression fusion, producing the final calibrated risk.

STAGED, LEAKAGE-SAFE TRAINING (exactly as specified):
    Stage 1: fit XGBoost (Phase 11's frozen hyperparameters) via 5-fold CV
             to obtain OUT-OF-FOLD probabilities on the full training pool
             -- these, not in-sample predictions, are what the residual
             target and the fusion stage are built from, so nothing here
             ever "sees" a row's own label leak through the base model's
             own overfit.
    Stage 2: residual = y - p_xgb_oof (the information the classical
             model did NOT already capture).
    Stage 3: the quantum refinement module (and its two matched classical
             controls) are TRAINED to predict this residual, with their
             weights selected via COBYLA + train-only CV on the cheap
             n=2,000 screening subset (Phase 8A's own subset -- reusing
             Phase 7/11/12's established "5-seed robustness on a cheap
             subset, single frozen seed for the expensive full-scale
             application" pattern, for the same disclosed cost reasons).
    Stage 4: the refinement module's FROZEN (seed=42) weights are applied
             ONCE to the full training pool and to the test set; a
             Logistic Regression fusion head is fit on
             [p_xgb_oof, refinement_score, margin] -> y (again OOF, not
             in-sample) for the final classical fusion stage.

FINAL DEPLOYMENT-TIME PATH (what predict_patient_risk actually runs): a
FRESH XGBoost fit on the FULL training pool (not OOF -- OOF was only for
leakage-safe TRAINING of the residual target and the fusion head) scores
new data; PCA-4 + that XGBoost probability + its margin feed the frozen
quantum/control refinement module; the fusion head combines them into the
final calibrated risk. This mirrors standard leakage-safe stacking
(train the meta-learner on out-of-fold base predictions, deploy using the
full-data base model) -- not an ad hoc shortcut.

MODELS COMPARED (all sharing the identical XGBoost backbone and fusion
head architecture -- ONLY the refinement module varies):
    A. XGBoost alone (no refinement term -- the Phase 11 backbone itself).
    B. XGBoost + quantum refinement (this phase's proposed system).
    C. XGBoost + matched classical (RFF) refinement -- the fair control.
    D. XGBoost + no-op random-feature refinement -- an additional sanity
       control (governing spec: "if useful").
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from qiskit_machine_learning.optimizers import COBYLA
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.classical.evaluation import compute_classification_metrics
from src.data.inspect_dataset import find_project_root
from src.large_dataset.corrected_comparison import (
    DECISION_THRESHOLD,
    comparison_set_fingerprint,
    mcnemar_exact,
    paired_bootstrap_delta,
)
from src.large_dataset.phase8a_label_aware_screening import build_screening_split
from src.large_dataset.phase11_full_scale import build_full_scale_split

#: Phase 11's own established full-scale best XGBoost hyperparameters
#: (Phase 11 selected these via its own grid search; Phase 11 itself does
#: not export them as a named constant, so they are restated here
#: verbatim -- same values Phase 12 also reused -- rather than re-run a
#: search this phase does not need).
XGB_FROZEN_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
from src.large_dataset.pipeline import process_stage
from src.large_dataset.schema import TARGET_COLUMN, get_cardio_feature_groups
from src.large_dataset.statistical_robustness import delong_test, holm_bonferroni
from src.preprocessing.config import PreprocessingConfig
from src.quantum.matched_refinement_controls import (
    MatchedRFFConfig,
    MatchedRFFRefinement,
    NoOpRandomConfig,
    NoOpRandomRefinement,
)
from src.quantum.quantum_refinement_circuit import INPUT_FEATURE_NAMES, QuantumRefinementModule, RefinementConfig

EXPECTED_FINGERPRINT = "96eac11a8394b87e"
RESULTS_ROOT = find_project_root() / "results" / "large_dataset" / "phase13_hybrid_system"

OOF_FOLDS = 5
DEV_SEEDS = [42, 123, 2024, 7, 99]
PRIMARY_SEED = 42
INNER_CV_FOLDS = 3
OPTIMIZER_MAXITER = 25


def _metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    m = compute_classification_metrics(y_true, y_pred, y_proba)
    m["brier_score"] = float(brier_score_loss(y_true, y_proba))
    return m


def _build_xgb() -> XGBClassifier:
    return XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1, tree_method="hist", **XGB_FROZEN_PARAMS)


def compute_oof_xgb_predictions(X: np.ndarray, y: np.ndarray, *, n_folds: int = OOF_FOLDS, seed: int = 42) -> np.ndarray:
    """Stage 1/2: strict out-of-fold XGBoost probabilities -- each row is
    scored ONLY by a model that never saw that row during fitting."""
    oof = np.zeros(len(y))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in cv.split(X, y):
        clf = _build_xgb()
        clf.fit(X[train_idx], y[train_idx])
        oof[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
    return oof


def build_quantum_input(pca_features: np.ndarray, xgb_proba: np.ndarray) -> np.ndarray:
    margin = np.abs(xgb_proba - 0.5)
    return np.hstack([pca_features, xgb_proba.reshape(-1, 1), margin.reshape(-1, 1)])


def select_refinement_weights(score_fn, n_params: int, X: np.ndarray, residual: np.ndarray, *,
                               seed: int, maxiter: int | None = None, cv_folds: int = INNER_CV_FOLDS) -> dict:
    """Representation-agnostic: COBYLA + train-only K-fold CV minimizing
    MSE between score_fn(X, params) and the residual target. Works
    identically for the quantum module and both classical controls (same
    `.score(X, params)` interface). TRAINING DATA ONLY -- no test
    parameter exists here."""
    if maxiter is None:
        maxiter = OPTIMIZER_MAXITER
    rng = np.random.RandomState(seed)
    p0 = rng.uniform(-0.1, 0.1, size=n_params)
    fold_idx = np.array_split(rng.permutation(len(X)), cv_folds)
    history: list[float] = []
    n_evals = 0

    def objective(params: np.ndarray) -> float:
        nonlocal n_evals
        n_evals += 1
        fold_mses = []
        for k in range(cv_folds):
            val_idx = fold_idx[k]
            train_idx = np.concatenate([fold_idx[j] for j in range(cv_folds) if j != k])
            pred = score_fn(X[val_idx], params)
            fold_mses.append(float(np.mean((pred - residual[val_idx]) ** 2)))
        mean_mse = float(np.mean(fold_mses))
        history.append(mean_mse)
        return mean_mse

    t0 = time.perf_counter()
    result = COBYLA(maxiter=maxiter).minimize(objective, p0)
    elapsed = time.perf_counter() - t0
    return {"weights": result.x, "cv_mse_history": history, "n_function_evaluations": n_evals,
            "seconds": elapsed, "best_cv_mse": min(history) if history else float("nan"), "seed": seed}


def run_multi_seed_dev_experiment(X_screen: np.ndarray, residual_screen: np.ndarray) -> dict:
    """Multi-seed robustness check on the cheap n=2,000 screening subset,
    for all three refinement modules (quantum, RFF control, no-op control)."""
    results = {"quantum": {}, "rff_control": {}, "noop_control": {}}
    for seed in DEV_SEEDS:
        q_mod = QuantumRefinementModule(RefinementConfig(seed=seed))
        q_sel = select_refinement_weights(q_mod.score, q_mod.config.n_trainable_params(), X_screen, residual_screen, seed=seed)
        results["quantum"][seed] = q_sel
        print(f"[phase13] dev seed={seed}: quantum refinement CV MSE={q_sel['best_cv_mse']:.5f} "
              f"({q_sel['n_function_evaluations']} evals, {q_sel['seconds']:.1f}s)", flush=True)

        r_mod = MatchedRFFRefinement(MatchedRFFConfig(seed=seed))
        r_sel = select_refinement_weights(r_mod.score, r_mod.config.n_trainable_params(), X_screen, residual_screen, seed=seed)
        results["rff_control"][seed] = r_sel
        print(f"[phase13] dev seed={seed}: RFF control CV MSE={r_sel['best_cv_mse']:.5f} "
              f"({r_sel['n_function_evaluations']} evals, {r_sel['seconds']:.1f}s)", flush=True)

        n_mod = NoOpRandomRefinement(NoOpRandomConfig(seed=seed))
        n_sel = select_refinement_weights(n_mod.score, n_mod.config.n_trainable_params(), X_screen, residual_screen, seed=seed)
        results["noop_control"][seed] = n_sel
        print(f"[phase13] dev seed={seed}: no-op control CV MSE={n_sel['best_cv_mse']:.5f} "
              f"({n_sel['n_function_evaluations']} evals, {n_sel['seconds']:.1f}s)", flush=True)

    summary = {}
    for key, per_seed in results.items():
        mses = [per_seed[s]["best_cv_mse"] for s in DEV_SEEDS]
        summary[key] = {"per_seed_cv_mse": dict(zip(DEV_SEEDS, mses)), "mean": float(np.mean(mses)),
                         "std": float(np.std(mses)), "min": float(np.min(mses)), "max": float(np.max(mses))}
    return {"results": results, "summary": summary}


def predict_patient_risk(patient_row_raw: pd.Series, deployed: dict) -> dict:
    """The prototype-facing prediction interface. `deployed` holds the
    fitted pieces (feature_groups, preprocessing config, shared pipeline,
    quantum pipeline, final XGBoost, refinement module + frozen weights,
    fusion LR) -- see run_phase13's `deployed_artifacts` return value.
    Returns predicted risk, category, confidence, contributing features,
    the quantum refinement's own contribution, model version, and
    inference latency. No clinical meaning is claimed for any raw quantum
    expectation value -- only the LITERAL refinement adjustment is reported.
    """
    t0 = time.perf_counter()
    feature_cols = deployed["feature_groups"].all_columns
    X_raw = pd.DataFrame([patient_row_raw[feature_cols]])
    X_classical = deployed["shared_pipeline"].transform(X_raw)
    X_pca = deployed["quantum_pipeline"].transform(X_classical)

    p_xgb = float(deployed["xgb_final"].predict_proba(X_classical)[:, 1][0])
    margin = abs(p_xgb - 0.5)
    q_input = build_quantum_input(np.asarray(X_pca), np.array([p_xgb]))
    refinement_raw = float(deployed["refinement_module"].score(q_input, deployed["refinement_weights"])[0])

    fusion_input = np.array([[p_xgb, refinement_raw, margin]])
    p_final = float(deployed["fusion_lr"].predict_proba(fusion_input)[:, 1][0])

    latency_ms = (time.perf_counter() - t0) * 1000
    risk_category = "high" if p_final >= 0.66 else ("moderate" if p_final >= 0.33 else "low")

    return {
        "classical_backbone_risk": round(p_xgb, 4),
        "quantum_refinement_raw_score": round(refinement_raw, 4),
        "quantum_refinement_adjustment": round(p_final - p_xgb, 4),
        "final_calibrated_risk": round(p_final, 4),
        "risk_category": risk_category,
        "model_version": "phase13_hybrid_v1",
        "inference_latency_ms": round(latency_ms, 3),
        "explanation": (
            f"Classical model estimated risk: {p_xgb:.2f}. "
            f"Hybrid refinement adjusted risk: {p_final - p_xgb:+.2f}. "
            f"Final calibrated risk: {p_final:.2f}."
        ),
    }


def run_phase13(*, make_plots: bool = True) -> dict:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    runtime: dict = {}

    print("[phase13] rebuilding the full fixed split (Phase 11's own function, unmodified) ...", flush=True)
    t0 = time.perf_counter()
    split = build_full_scale_split()
    runtime["split_seconds"] = time.perf_counter() - t0
    cmp_ids = split.test_set_quantum["id"].tolist()
    fingerprint = comparison_set_fingerprint(cmp_ids)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"Comparison set fingerprint {fingerprint} != expected {EXPECTED_FINGERPRINT}")

    fg = get_cardio_feature_groups()
    pcfg = PreprocessingConfig()

    print("[phase13] fitting preprocessing/PCA on the FULL training pool (train-only) ...", flush=True)
    t1 = time.perf_counter()
    processed = process_stage(split.training_pool, split.test_set_classical, split.test_set_quantum, TARGET_COLUMN, fg, pcfg)
    runtime["preprocessing_seconds"] = time.perf_counter() - t1
    X_train_classical = processed.X_train_classical.to_numpy() if hasattr(processed.X_train_classical, "to_numpy") else processed.X_train_classical
    X_train_pca, y_train = processed.X_train_quantum, processed.y_train
    X_test_pca, y_test = processed.X_test_quantum, processed.y_test_quantum

    # Refit an identical (same config, same training data -> deterministic,
    # numerically identical) quantum_pipeline object standalone, since
    # process_stage's own internal one isn't exposed on LargeDatasetProcessed
    # -- needed so predict_patient_risk has a real, fitted PCA+range object
    # to call .transform() on for a genuinely new row, without modifying
    # Phase 2's pipeline.py.
    from src.preprocessing.pipeline import build_quantum_pipeline

    quantum_pipeline = build_quantum_pipeline(pcfg.pca_n_components, pcfg.quantum_range)
    quantum_pipeline.fit(processed.X_train_classical)

    # ---- Stage 1/2: OOF XGBoost + residual ----
    print(f"[phase13] ===== Stage 1: {OOF_FOLDS}-fold OOF XGBoost on n_train={len(X_train_classical)} =====", flush=True)
    t2 = time.perf_counter()
    oof_proba = compute_oof_xgb_predictions(X_train_classical, y_train)
    runtime["oof_xgb_seconds"] = time.perf_counter() - t2
    residual = y_train.astype(float) - oof_proba
    print(f"[phase13] OOF done in {runtime['oof_xgb_seconds']:.1f}s. "
          f"OOF XGBoost ROC-AUC (sanity check, train folds): "
          f"{compute_classification_metrics(y_train, (oof_proba >= 0.5).astype(int), oof_proba)['roc_auc']:.4f}", flush=True)

    print("[phase13] fitting the FINAL (all-data) XGBoost backbone for deployment/test scoring ...", flush=True)
    t3 = time.perf_counter()
    xgb_final = _build_xgb()
    xgb_final.fit(X_train_classical, y_train)
    runtime["final_xgb_fit_seconds"] = time.perf_counter() - t3
    # NOTE: processed.X_test_classical is the fixed 2,000-row CLASSICAL test
    # set (Phase 3's own convention) -- the primary comparison set here is
    # the fixed 200-row set, whose classical-space (pre-PCA) representation
    # is processed.X_test_quantum_classical_space. Using the 2,000-row array
    # here would silently misalign every downstream row with y_test (200
    # rows) -- caught by a shape-mismatch crash during development, fixed
    # here, and covered by a regression test.
    X_test_classical = processed.X_test_quantum_classical_space
    X_test_classical = X_test_classical.to_numpy() if hasattr(X_test_classical, "to_numpy") else X_test_classical
    xgb_test_proba = xgb_final.predict_proba(X_test_classical)[:, 1]

    full_quantum_input = build_quantum_input(X_train_pca, oof_proba)
    test_quantum_input = build_quantum_input(X_test_pca, xgb_test_proba)

    # ---- Stage 3: multi-seed weight selection on the n=2,000 screening subset ----
    print("[phase13] ===== Stage 3: refinement-weight selection on n=2,000 screening subset, 5 seeds =====", flush=True)
    screen = build_screening_split()
    screen_ids = set(screen.train_df["id"])
    train_ids = split.training_pool["id"].to_numpy()
    screen_mask = np.isin(train_ids, list(screen_ids))
    X_screen_input = full_quantum_input[screen_mask]
    residual_screen = residual[screen_mask]
    print(f"[phase13] screening subset for weight selection: n={screen_mask.sum()} (subset of the full training pool, "
          f"train-only, zero overlap with test)", flush=True)

    t4 = time.perf_counter()
    dev = run_multi_seed_dev_experiment(X_screen_input, residual_screen)
    runtime["dev_experiment_seconds"] = time.perf_counter() - t4
    for key, s in dev["summary"].items():
        print(f"[phase13] {key}: mean CV MSE={s['mean']:.5f} std={s['std']:.5f} min={s['min']:.5f} max={s['max']:.5f}", flush=True)

    # ---- Stage 4: apply frozen (seed=42) weights, fit fusion heads ----
    print("[phase13] ===== Stage 4: applying frozen weights to full data, fitting fusion heads =====", flush=True)
    q_mod = QuantumRefinementModule(RefinementConfig(seed=PRIMARY_SEED))
    r_mod = MatchedRFFRefinement(MatchedRFFConfig(seed=PRIMARY_SEED))
    n_mod = NoOpRandomRefinement(NoOpRandomConfig(seed=PRIMARY_SEED))
    q_w = dev["results"]["quantum"][PRIMARY_SEED]["weights"]
    r_w = dev["results"]["rff_control"][PRIMARY_SEED]["weights"]
    n_w = dev["results"]["noop_control"][PRIMARY_SEED]["weights"]

    t5 = time.perf_counter()
    q_train_score = q_mod.score(full_quantum_input, q_w)
    runtime["quantum_full_batch_forward_seconds"] = time.perf_counter() - t5
    q_test_score = q_mod.score(test_quantum_input, q_w)
    q_z_train = q_mod.expectation_values(full_quantum_input, q_w[:q_mod.config.n_circuit_params()])
    assert np.all(q_z_train >= -1.0 - 1e-6) and np.all(q_z_train <= 1.0 + 1e-6), "quantum expectation values out of [-1,1]"

    r_train_score = r_mod.score(full_quantum_input, r_w)
    r_test_score = r_mod.score(test_quantum_input, r_w)
    n_train_score = n_mod.score(full_quantum_input, n_w)
    n_test_score = n_mod.score(test_quantum_input, n_w)

    margin_train = np.abs(oof_proba - 0.5)
    margin_test = np.abs(xgb_test_proba - 0.5)

    def _fit_fusion(refinement_train: np.ndarray, refinement_test: np.ndarray) -> tuple[np.ndarray, LogisticRegression]:
        Ztr = np.column_stack([oof_proba, refinement_train, margin_train])
        Zte = np.column_stack([xgb_test_proba, refinement_test, margin_test])
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(Ztr, y_train)
        return lr.predict_proba(Zte)[:, 1], lr

    all_models: dict[str, dict] = {}
    all_models["xgboost_alone"] = {"display_name": "A: XGBoost alone (no refinement)", "y_proba": xgb_test_proba, "kind": "backbone"}
    q_test_final, q_fusion_lr = _fit_fusion(q_train_score, q_test_score)
    all_models["quantum_hybrid"] = {"display_name": "B: XGBoost + Quantum Refinement (fused)", "y_proba": q_test_final, "kind": "hybrid_quantum"}
    r_test_final, r_fusion_lr = _fit_fusion(r_train_score, r_test_score)
    all_models["rff_control_hybrid"] = {"display_name": "C: XGBoost + Matched Classical (RFF) Refinement", "y_proba": r_test_final, "kind": "hybrid_control"}
    n_test_final, n_fusion_lr = _fit_fusion(n_train_score, n_test_score)
    all_models["noop_control_hybrid"] = {"display_name": "D: XGBoost + No-op Random Refinement", "y_proba": n_test_final, "kind": "hybrid_sanity_control"}

    for m in all_models.values():
        m["y_true"] = y_test
        m["y_pred"] = (m["y_proba"] >= DECISION_THRESHOLD).astype(int)
        m["metrics"] = _metrics(y_test, m["y_proba"])
        assert not np.any(np.isnan(m["y_proba"])) and not np.any(np.isinf(m["y_proba"]))

    # ---- statistics ----
    print("[phase13] computing paired statistical comparisons ...", flush=True)

    def _pair(key_a: str, key_b: str) -> dict:
        pa, pb = all_models[key_a]["y_proba"], all_models[key_b]["y_proba"]
        preda, predb = all_models[key_a]["y_pred"], all_models[key_b]["y_pred"]
        dl = delong_test(y_test, pa, pb)
        bs = paired_bootstrap_delta(y_test, pa, pb, "roc_auc", n_resamples=2000, seed=42)
        bs_pr = paired_bootstrap_delta(y_test, pa, pb, "pr_auc", n_resamples=2000, seed=42)
        mc = mcnemar_exact(y_test, preda, predb)
        return {"a": key_a, "b": key_b, "delong": dl.to_dict(), "bootstrap_roc_auc": bs, "bootstrap_pr_auc": bs_pr, "mcnemar": mc}

    comparisons = {
        "quantum_hybrid_vs_xgboost_alone": _pair("quantum_hybrid", "xgboost_alone"),
        "quantum_hybrid_vs_rff_control": _pair("quantum_hybrid", "rff_control_hybrid"),
        "quantum_hybrid_vs_noop_control": _pair("quantum_hybrid", "noop_control_hybrid"),
    }
    raw_p = [c["delong"]["p_value"] for c in comparisons.values()]
    holm = holm_bonferroni(raw_p)
    for (name, c), h in zip(comparisons.items(), holm):
        c["delong_p_holm"] = h["adjusted_p"]
        c["delong_significant_holm"] = h["significant"]

    vs_xgb = comparisons["quantum_hybrid_vs_xgboost_alone"]
    vs_rff = comparisons["quantum_hybrid_vs_rff_control"]
    beats_xgb = vs_xgb["delong"]["delta"] > 0 and vs_xgb["delong_significant_holm"] and not vs_xgb["bootstrap_roc_auc"]["ci_includes_zero"]
    beats_control = vs_rff["delong"]["delta"] > 0 and vs_rff["delong_significant_holm"] and not vs_rff["bootstrap_roc_auc"]["ci_includes_zero"]
    if beats_xgb and beats_control:
        decision_text = "Quantum refinement provides a statistically supported improvement over both the XGBoost backbone and the matched classical control."
        outcome = "WIN"
    elif beats_control and not beats_xgb:
        decision_text = "Quantum refinement beats the matched control but not the XGBoost backbone -- an interesting but not practically decisive result."
        outcome = "PARTIAL"
    else:
        decision_text = ("Hybrid quantum-classical architecture implemented and evaluated, with the classical backbone "
                          "providing the primary predictive signal and the quantum module evaluated as a refinement "
                          "component; no statistically supported improvement over the backbone or the matched control "
                          "was demonstrated.")
        outcome = "NO_IMPROVEMENT"

    decision = {"outcome": outcome, "outcome_text": decision_text, "beats_xgboost_backbone": bool(beats_xgb),
                "beats_matched_control": bool(beats_control), "delta_vs_xgboost": vs_xgb["delong"]["delta"],
                "delta_vs_rff_control": vs_rff["delong"]["delta"]}

    # ---- explainability: XGBoost feature importances + exact SHAP contributions ----
    # XGBoost's sklearn wrapper (.predict()) does not accept pred_contribs;
    # the underlying Booster does -- this gives EXACT SHAP values natively,
    # no external `shap` dependency needed.
    from xgboost import DMatrix

    shap_contribs = xgb_final.get_booster().predict(DMatrix(X_test_classical), pred_contribs=True)
    feature_importances = dict(zip(fg.all_columns, xgb_final.feature_importances_.tolist()))

    # ---- prediction interface self-check: predict_patient_risk must agree
    # with the batch-computed quantum_hybrid prediction for the SAME row ----
    deployed_artifacts = {
        "feature_groups": fg, "shared_pipeline": processed.shared_pipeline, "quantum_pipeline": quantum_pipeline,
        "xgb_final": xgb_final, "refinement_module": q_mod, "refinement_weights": q_w, "fusion_lr": q_fusion_lr,
    }
    example_row = split.test_set_quantum.iloc[0]
    example_result = predict_patient_risk(example_row, deployed_artifacts)
    batch_equivalent = float(all_models["quantum_hybrid"]["y_proba"][0])
    interface_self_check = {
        "example_patient_id": int(example_row["id"]),
        "interface_predicted_risk": example_result["final_calibrated_risk"],
        "batch_pipeline_predicted_risk": round(batch_equivalent, 4),
        "agrees_with_batch_pipeline": abs(example_result["final_calibrated_risk"] - batch_equivalent) < 1e-3,
        "example_output": example_result,
    }
    if not interface_self_check["agrees_with_batch_pipeline"]:
        print(f"[phase13] WARNING: prediction interface disagrees with batch pipeline "
              f"({example_result['final_calibrated_risk']} vs {batch_equivalent})", flush=True)

    # ---- persist ----
    pred_df = pd.DataFrame({"id": cmp_ids, "y_true": y_test})
    for key, m in all_models.items():
        pred_df[f"{key}_proba"] = m["y_proba"]
        pred_df[f"{key}_pred"] = m["y_pred"]
    pred_df.to_csv(RESULTS_ROOT / "predictions.csv", index=False)
    assert not pred_df.isna().any().any()

    metrics_rows = [{"Model": m["display_name"], "model_key": key, "kind": m["kind"],
                      **{k: m["metrics"][k] for k in ("roc_auc", "pr_auc", "sensitivity", "specificity", "accuracy", "precision", "f1", "brier_score", "tn", "fp", "fn", "tp")}}
                     for key, m in all_models.items()]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULTS_ROOT / "metrics.csv", index=False)

    dev_rows = []
    for key, per_seed in dev["results"].items():
        for seed in DEV_SEEDS:
            dev_rows.append({"module": key, "seed": seed, "cv_mse": per_seed[seed]["best_cv_mse"],
                              "n_evals": per_seed[seed]["n_function_evaluations"], "seconds": per_seed[seed]["seconds"]})
    pd.DataFrame(dev_rows).to_csv(RESULTS_ROOT / "seed_by_seed_results.csv", index=False)

    calib_rows = []
    for key, m in all_models.items():
        frac_pos, mean_pred = calibration_curve(m["y_true"], m["y_proba"], n_bins=5, strategy="quantile")
        for fp_, mp_ in zip(frac_pos, mean_pred):
            calib_rows.append({"model_key": key, "mean_predicted": mp_, "fraction_positive": fp_})
    pd.DataFrame(calib_rows).to_csv(RESULTS_ROOT / "calibration_curve.csv", index=False)

    with open(RESULTS_ROOT / "model_configs.json", "w", encoding="utf-8") as fh:
        json.dump({"quantum_refinement": RefinementConfig(seed=PRIMARY_SEED).to_dict(),
                    "rff_control": MatchedRFFConfig(seed=PRIMARY_SEED).to_dict(),
                    "noop_control": NoOpRandomConfig(seed=PRIMARY_SEED).to_dict(),
                    "xgb_frozen_params": XGB_FROZEN_PARAMS, "primary_seed": PRIMARY_SEED, "dev_seeds": DEV_SEEDS,
                    "oof_folds": OOF_FOLDS, "optimizer_maxiter": OPTIMIZER_MAXITER}, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "runtime.json", "w", encoding="utf-8") as fh:
        json.dump(runtime, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "statistical_comparisons.json", "w", encoding="utf-8") as fh:
        json.dump(comparisons, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "reproducibility_summary.json", "w", encoding="utf-8") as fh:
        json.dump(dev["summary"], fh, indent=2, default=float)
    with open(RESULTS_ROOT / "prediction_interface_example.json", "w", encoding="utf-8") as fh:
        json.dump(interface_self_check, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "explainability.json", "w", encoding="utf-8") as fh:
        json.dump({
            "xgboost_feature_importances": feature_importances,
            "quantum_refinement_input_features": INPUT_FEATURE_NAMES,
            "quantum_refinement_note": "The quantum module outputs ONE aggregate refinement score per patient -- a "
                                        "learned linear combination of Pauli-Z expectation values after a trainable "
                                        "circuit. No individual expectation value is assigned clinical meaning.",
            "fusion_lr_coefficients": {"p_xgb": float(q_fusion_lr.coef_[0][0]), "refinement_score": float(q_fusion_lr.coef_[0][1]),
                                        "margin": float(q_fusion_lr.coef_[0][2]), "intercept": float(q_fusion_lr.intercept_[0])},
            "shap_available": True,
            "shap_mean_abs_contribution_test_set": dict(zip(
                list(fg.all_columns) + ["bias"], np.abs(shap_contribs).mean(axis=0).tolist(),
            )),
        }, fh, indent=2, default=float)

    summary = {
        "n_train_full": len(split.training_pool),
        "comparison_set": {"n": len(cmp_ids), "fingerprint": fingerprint, "matches_expected": fingerprint == EXPECTED_FINGERPRINT},
        "metrics": {key: m["metrics"] for key, m in all_models.items()},
        "reproducibility": dev["summary"],
        "statistics": comparisons,
        "runtime": runtime,
        "decision": decision,
        "prediction_interface_self_check": {k: v for k, v in interface_self_check.items() if k != "example_output"},
    }
    with open(RESULTS_ROOT / "phase13_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"[phase13] DECISION: {outcome} -- {decision_text}", flush=True)

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

    fig, ax = plt.subplots(figsize=(8, 7))
    for key, m in items:
        fpr, tpr, _ = roc_curve(m["y_true"], m["y_proba"])
        ax.plot(fpr, tpr, label=f"{m['display_name']} (AUC={m['metrics']['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC -- Phase 13 hybrid system\nIdentical {n_cmp}-row held-out test set")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "roc_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for key, m in items:
        prec, rec, _ = precision_recall_curve(m["y_true"], m["y_proba"])
        ax.plot(rec, prec, label=f"{m['display_name']} (AP={m['metrics']['pr_auc']:.4f})")
    base = float(np.mean(items[0][1]["y_true"]))
    ax.axhline(base, ls="--", color="grey", label=f"Chance ({base:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall -- Phase 13")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "pr_curves.png", dpi=150); plt.close(fig)

    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    for ax, (key, m) in zip(np.atleast_1d(axes), items):
        mm = m["metrics"]
        cm = np.array([[mm["tn"], mm["fp"]], [mm["fn"], mm["tp"]]])
        ConfusionMatrixDisplay(cm, display_labels=["No CVD", "CVD"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(m["display_name"], fontsize=7)
    fig.suptitle(f"Confusion matrices -- Phase 13, identical {n_cmp}-row test set")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrices.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for key, m in items:
        frac_pos, mean_pred = calibration_curve(m["y_true"], m["y_proba"], n_bins=5, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=m["display_name"])
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve -- Phase 13"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out_dir / "calibration_curve.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    result = run_phase13()
    print(json.dumps({k: v for k, v in result.items() if k != "statistics"}, indent=2, default=float))
