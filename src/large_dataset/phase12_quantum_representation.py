"""Phase 12: trainable quantum LATENT REPRESENTATION vs a matched
non-quantum control (see docs/PHASE_12_QUANTUM_REPRESENTATION.md).

RESEARCH QUESTION: "Can a trainable quantum representation contribute
useful predictive information when integrated into a classical ML
pipeline, beyond an equivalent non-quantum representation?" This is a NEW
hypothesis, not a repeat of Phase 10 (see
src.quantum.quantum_latent_representation's module docstring for the
concrete architectural differences) or Phase 7/8A/8B/9 (already-falsified
directions this phase does not revisit).

MODELS (per the governing spec):
    A. Best classical baseline: Phase 11's full-scale XGBoost on the
       original classical features -- REUSED, not retrained.
    B. Quantum hybrid: PCA-4 concatenated with the K=8 quantum latent
       representation -> {XGBoost, Logistic Regression} (identical
       classifier config used for B and C -- only the representation varies).
    C. Matched non-quantum control: PCA-4 concatenated with the K=8
       classical Random Fourier Feature representation -> the SAME
       {XGBoost, Logistic Regression} configs as B.
    (D. PCA-4 alone -> the same classifiers, as an internal reference
       point showing what the added representation is being compared
       against before even reaching the control.)

WHAT IS REUSED, UNMODIFIED:
    - src.large_dataset.phase11_full_scale.build_full_scale_split (the
      exact n=66,641 training pool + fixed 200-row test set)
    - src.large_dataset.phase8a_label_aware_screening.build_screening_split
      (the n=2,000 subset used for multi-seed representation-weight
      selection -- train-only, zero overlap with test, already vetted)
    - src.large_dataset.pipeline.process_stage (Phase 2 preprocessing/PCA)
    - src.quantum.quantum_latent_representation / matched_classical_control
      (this phase's new representation blocks)
    - src.large_dataset.corrected_comparison / statistical_robustness
      (the same paired-statistics functions every prior phase used)

REPRODUCIBILITY / MULTI-SEED PROTOCOL: representation weights (quantum
theta, control phi) are selected via COBYLA + train-only 3-fold CV
(scored with a fast Logistic Regression proxy, matching Phase 10/11's own
convention) on the n=2,000 screening subset, independently for FIVE seeds
(42, 123, 2024, 7, 99) -- mean/std/individual-seed CV scores are reported
for BOTH the quantum and control representations, without averaging away
instability. Exactly one seed (42, the project's established default
throughout every phase) is then used to compute the FINAL representation
weights that get applied, ONCE, to the full n=66,641 training pool and
scored, ONCE, on the fixed 200-row test set -- mirroring Phase 7's own
"5-seed robustness check at cheap scale, single seed for the expensive
final run" pattern, which keeps this phase's cost bounded without
sacrificing the reproducibility evidence the governing spec requires.

FAIRNESS RULE (explicit, and enforced structurally): the 200-row test set
is never touched by select_representation_weights (no test-data parameter
exists on it) and is scored exactly once, after every seed's weights are
already frozen -- no post-hoc re-tuning against test performance occurs
anywhere in this module.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from qiskit_machine_learning.optimizers import COBYLA
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
from src.large_dataset.pipeline import process_stage
from src.large_dataset.schema import TARGET_COLUMN, get_cardio_feature_groups
from src.large_dataset.statistical_robustness import delong_test, holm_bonferroni
from src.preprocessing.config import PreprocessingConfig
from src.quantum.matched_classical_control import MatchedControlConfig, MatchedControlLayer
from src.quantum.quantum_latent_representation import OBSERVABLE_NAMES, QuantumRepresentationConfig, QuantumRepresentationLayer

EXPECTED_FINGERPRINT = "96eac11a8394b87e"
RESULTS_ROOT = find_project_root() / "results" / "large_dataset" / "phase12_quantum_representation"
PHASE11_METRICS_PATH = find_project_root() / "results" / "large_dataset" / "phase11_full_scale" / "metrics.csv"
PHASE11_PREDICTIONS_PATH = find_project_root() / "results" / "large_dataset" / "phase11_full_scale" / "predictions.csv"

DEV_SEEDS = [42, 123, 2024, 7, 99]
PRIMARY_SEED = 42
INNER_CV_FOLDS = 3
OPTIMIZER_MAXITER = 40

#: Phase 11's own frozen best XGBoost hyperparameters, reused verbatim so
#: the ONLY thing that varies across representations B/C/D is the
#: representation itself -- "the downstream classifier must be identical."
XGB_FROZEN_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)


def _metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    return compute_classification_metrics(y_true, y_pred, y_proba)


def select_representation_weights(
    transform_fn, n_params: int, X_train: np.ndarray, y_train: np.ndarray, *,
    seed: int, maxiter: int | None = None, cv_folds: int = INNER_CV_FOLDS,
) -> dict:
    """Representation-agnostic outer training loop: for a candidate
    weight vector w, transform_fn(X_train, w) produces K latent features,
    concatenated with X_train itself, scored via train-only K-fold CV with
    a fast Logistic Regression proxy. COBYLA searches w to maximize mean
    CV ROC-AUC. TRAINING DATA ONLY -- no test-data parameter exists here.
    """
    if maxiter is None:
        maxiter = OPTIMIZER_MAXITER  # module-level, resolved at CALL time (monkeypatch-friendly for dry runs)
    rng = np.random.RandomState(seed)
    w0 = rng.uniform(-0.1, 0.1, size=n_params)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    history: list[float] = []
    n_evals = 0

    def objective(w: np.ndarray) -> float:
        nonlocal n_evals
        n_evals += 1
        Z = np.hstack([X_train, transform_fn(X_train, w)])
        lr = LogisticRegression(max_iter=1000, random_state=seed)
        scores = cross_val_score(lr, Z, y_train, cv=cv, scoring="roc_auc")
        mean_score = float(np.mean(scores))
        history.append(mean_score)
        return -mean_score

    t0 = time.perf_counter()
    result = COBYLA(maxiter=maxiter).minimize(objective, w0)
    elapsed = time.perf_counter() - t0

    return {
        "weights": result.x, "cv_score_history": history, "n_function_evaluations": n_evals,
        "seconds": elapsed, "best_cv_score": max(history) if history else float("nan"), "seed": seed,
    }


def run_multi_seed_dev_experiment(screen_processed) -> dict:
    """Runs select_representation_weights for BOTH the quantum and
    control representations, across all 5 DEV_SEEDS, on the n=2,000
    screening subset (train-only). Reports per-seed and aggregate CV
    scores -- no instability is averaged away."""
    X_train, y_train = screen_processed.X_train_quantum, screen_processed.y_train
    quantum_results, control_results = {}, {}

    for seed in DEV_SEEDS:
        print(f"[phase12] dev seed={seed}: selecting quantum representation weights ...", flush=True)
        q_layer = QuantumRepresentationLayer(QuantumRepresentationConfig(seed=seed))
        q_sel = select_representation_weights(
            q_layer.transform, q_layer.config.n_trainable_params(), X_train, y_train, seed=seed,
        )
        quantum_results[seed] = q_sel
        print(f"[phase12] dev seed={seed}: quantum CV ROC-AUC={q_sel['best_cv_score']:.4f} "
              f"({q_sel['n_function_evaluations']} evals, {q_sel['seconds']:.1f}s)", flush=True)

        c_layer = MatchedControlLayer(MatchedControlConfig(seed=seed))
        c_sel = select_representation_weights(
            c_layer.transform, c_layer.config.n_trainable_params(), X_train, y_train, seed=seed,
        )
        control_results[seed] = c_sel
        print(f"[phase12] dev seed={seed}: control CV ROC-AUC={c_sel['best_cv_score']:.4f} "
              f"({c_sel['n_function_evaluations']} evals, {c_sel['seconds']:.1f}s)", flush=True)

    q_scores = [quantum_results[s]["best_cv_score"] for s in DEV_SEEDS]
    c_scores = [control_results[s]["best_cv_score"] for s in DEV_SEEDS]
    summary = {
        "seeds": DEV_SEEDS,
        "quantum": {"per_seed_cv_roc_auc": dict(zip(DEV_SEEDS, q_scores)), "mean": float(np.mean(q_scores)), "std": float(np.std(q_scores))},
        "control": {"per_seed_cv_roc_auc": dict(zip(DEV_SEEDS, c_scores)), "mean": float(np.mean(c_scores)), "std": float(np.std(c_scores))},
    }
    return {"quantum_results": quantum_results, "control_results": control_results, "summary": summary}


def run_phase12(*, make_plots: bool = True) -> dict:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    runtime: dict = {}

    print("[phase12] loading Phase 11 reference (best classical: full-scale XGBoost) ...", flush=True)
    phase11_metrics = pd.read_csv(PHASE11_METRICS_PATH).set_index("model_key")
    phase11_preds = pd.read_csv(PHASE11_PREDICTIONS_PATH)
    fp = comparison_set_fingerprint(phase11_preds["id"].tolist())
    if fp != EXPECTED_FINGERPRINT:
        raise ValueError(f"Phase 11 predictions fingerprint {fp} != expected {EXPECTED_FINGERPRINT}")

    print("[phase12] rebuilding the full fixed split (Phase 11's own function, unmodified) ...", flush=True)
    t0 = time.perf_counter()
    split = build_full_scale_split()
    runtime["split_seconds"] = time.perf_counter() - t0
    cmp_ids = split.test_set_quantum["id"].tolist()
    fingerprint = comparison_set_fingerprint(cmp_ids)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"Comparison set fingerprint {fingerprint} != expected {EXPECTED_FINGERPRINT}")
    if phase11_preds["id"].tolist() != cmp_ids:
        raise ValueError("Phase 11 predictions are not aligned to the current comparison set.")

    fg = get_cardio_feature_groups()
    pcfg = PreprocessingConfig()

    # ---- Step 8-9: multi-seed development experiment on the cheap n=2,000 subset ----
    print("[phase12] ===== multi-seed development experiment (n_train=2,000, train-only CV) =====", flush=True)
    t1 = time.perf_counter()
    screen = build_screening_split()
    screen_processed = process_stage(
        screen.train_df, split.test_set_classical, split.test_set_quantum,
        TARGET_COLUMN, fg, pcfg,
    )
    dev = run_multi_seed_dev_experiment(screen_processed)
    runtime["dev_experiment_seconds"] = time.perf_counter() - t1
    print(f"[phase12] dev experiment done in {runtime['dev_experiment_seconds']:.1f}s. "
          f"Quantum CV ROC-AUC: mean={dev['summary']['quantum']['mean']:.4f} std={dev['summary']['quantum']['std']:.4f}; "
          f"Control CV ROC-AUC: mean={dev['summary']['control']['mean']:.4f} std={dev['summary']['control']['std']:.4f}", flush=True)

    # ---- Step 10: freeze weights from the PRIMARY seed; full-scale, ONE-TIME test evaluation ----
    print(f"[phase12] fitting preprocessing/PCA on the FULL training pool (train-only, n={len(split.training_pool)}) ...", flush=True)
    t2 = time.perf_counter()
    processed = process_stage(
        split.training_pool, split.test_set_classical, split.test_set_quantum,
        TARGET_COLUMN, fg, pcfg,
    )
    runtime["preprocessing_and_pca_seconds"] = time.perf_counter() - t2
    X_train, y_train = processed.X_train_quantum, processed.y_train
    X_test, y_test = processed.X_test_quantum, processed.y_test_quantum
    if phase11_preds["y_true"].tolist() != list(y_test):
        raise ValueError("Test labels mismatch between Phase 12 and Phase 11's reference file.")

    q_theta = dev["quantum_results"][PRIMARY_SEED]["weights"]
    c_phi = dev["control_results"][PRIMARY_SEED]["weights"]
    q_layer = QuantumRepresentationLayer(QuantumRepresentationConfig(seed=PRIMARY_SEED))
    c_layer = MatchedControlLayer(MatchedControlConfig(seed=PRIMARY_SEED))

    print(f"[phase12] computing quantum representation for the FULL training pool "
          f"(n={len(X_train)}, ONE forward pass at the seed={PRIMARY_SEED}-selected weights) ...", flush=True)
    t3 = time.perf_counter()
    q_train_features = q_layer.transform(X_train, q_theta)
    runtime["quantum_full_batch_forward_seconds"] = time.perf_counter() - t3
    assert np.all(q_train_features >= -1.0 - 1e-6) and np.all(q_train_features <= 1.0 + 1e-6)
    q_test_features = q_layer.transform(X_test, q_theta)
    assert np.all(q_test_features >= -1.0 - 1e-6) and np.all(q_test_features <= 1.0 + 1e-6)

    c_train_features = c_layer.transform(X_train, c_phi)
    c_test_features = c_layer.transform(X_test, c_phi)

    representations = {
        "pca_only": (X_train, X_test),
        "quantum": (np.hstack([X_train, q_train_features]), np.hstack([X_test, q_test_features])),
        "control": (np.hstack([X_train, c_train_features]), np.hstack([X_test, c_test_features])),
    }

    all_models: dict[str, dict] = {}
    print("[phase12] fitting downstream classifiers (identical config per classifier type, only representation varies) ...", flush=True)
    for repr_name, (Ztr, Zte) in representations.items():
        t4 = time.perf_counter()
        xgb = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1, tree_method="hist", **XGB_FROZEN_PARAMS)
        xgb.fit(Ztr, y_train)
        xgb_seconds = time.perf_counter() - t4
        xgb_proba = xgb.predict_proba(Zte)[:, 1]
        all_models[f"{repr_name}_xgboost"] = {
            "display_name": f"{repr_name} -> XGBoost", "y_proba": xgb_proba, "fit_seconds": xgb_seconds,
            "kind": f"repr_{repr_name}", "classifier": "xgboost", "fitted_model": xgb,
        }

        t5 = time.perf_counter()
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(Ztr, y_train)
        lr_seconds = time.perf_counter() - t5
        lr_proba = lr.predict_proba(Zte)[:, 1]
        all_models[f"{repr_name}_logreg"] = {
            "display_name": f"{repr_name} -> Logistic Regression", "y_proba": lr_proba, "fit_seconds": lr_seconds,
            "kind": f"repr_{repr_name}", "classifier": "logreg",
            "coefficients": dict(zip(
                [f"pca_{i}" for i in range(4)] + (OBSERVABLE_NAMES if repr_name == "quantum" else [f"control_{i}" for i in range(8)] if repr_name == "control" else []),
                lr.coef_[0].tolist(),
            )) if repr_name != "pca_only" else dict(zip([f"pca_{i}" for i in range(4)], lr.coef_[0].tolist())),
        }

    # ---- Model A: best classical baseline (Phase 11, reused) ----
    all_models["best_classical_xgboost_full_features"] = {
        "display_name": "Model A: Best classical (Phase 11 XGBoost, full features, reused)",
        "y_proba": phase11_preds["xgboost_proba"].to_numpy(), "fit_seconds": None,
        "kind": "reused_reference", "classifier": "xgboost",
    }

    for m in all_models.values():
        m["y_true"] = y_test
        m["y_pred"] = (m["y_proba"] >= DECISION_THRESHOLD).astype(int)
        m["metrics"] = _metrics(y_test, m["y_proba"])
        assert not np.any(np.isnan(m["y_proba"])) and not np.any(np.isinf(m["y_proba"]))

    # ---- Step 11: statistics ----
    print("[phase12] computing paired statistical comparisons ...", flush=True)

    def _pair(key_a: str, key_b: str) -> dict:
        pa, pb = all_models[key_a]["y_proba"], all_models[key_b]["y_proba"]
        preda, predb = all_models[key_a]["y_pred"], all_models[key_b]["y_pred"]
        dl = delong_test(y_test, pa, pb)
        bs = paired_bootstrap_delta(y_test, pa, pb, "roc_auc", n_resamples=2000, seed=42)
        bs_pr = paired_bootstrap_delta(y_test, pa, pb, "pr_auc", n_resamples=2000, seed=42)
        mc = mcnemar_exact(y_test, preda, predb)
        return {"a": key_a, "b": key_b, "delong": dl.to_dict(), "bootstrap_roc_auc": bs, "bootstrap_pr_auc": bs_pr, "mcnemar": mc}

    comparisons = {
        "quantum_xgb_vs_control_xgb": _pair("quantum_xgboost", "control_xgboost"),
        "quantum_logreg_vs_control_logreg": _pair("quantum_logreg", "control_logreg"),
        "quantum_xgb_vs_best_classical": _pair("quantum_xgboost", "best_classical_xgboost_full_features"),
        "quantum_xgb_vs_pca_only_xgb": _pair("quantum_xgboost", "pca_only_xgboost"),
    }
    raw_p = [c["delong"]["p_value"] for c in comparisons.values()]
    holm = holm_bonferroni(raw_p)
    for (name, c), h in zip(comparisons.items(), holm):
        c["delong_p_holm"] = h["adjusted_p"]
        c["delong_significant_holm"] = h["significant"]

    # ---- Step 15: decision (A/B/C/D per the governing spec's success criteria) ----
    primary = comparisons["quantum_xgb_vs_control_xgb"]
    delta_vs_control = primary["delong"]["delta"]
    beats_control = delta_vs_control > 0 and primary["delong_significant_holm"] and not primary["bootstrap_roc_auc"]["ci_includes_zero"]
    vs_classical = comparisons["quantum_xgb_vs_best_classical"]
    delta_vs_classical = vs_classical["delong"]["delta"]
    beats_classical = delta_vs_classical > 0 and vs_classical["delong_significant_holm"] and not vs_classical["bootstrap_roc_auc"]["ci_includes_zero"]
    seed_instability = dev["summary"]["quantum"]["std"] > 0.03  # descriptive threshold, reported either way

    if not beats_control:
        outcome = "B"
        outcome_text = "No measurable quantum contribution was demonstrated (quantum representation does not beat the matched non-quantum control)."
    elif beats_control and not beats_classical:
        outcome = "B2"
        outcome_text = "The quantum representation shows evidence of incremental predictive contribution over the matched non-quantum control, but no quantum advantage over the strongest classical baseline."
    else:
        outcome = "C"
        outcome_text = "Quantum beats BOTH the control and the strongest classical baseline -- requires further confirmation (multi-seed already run; see robustness) before any quantum-advantage language is used."

    decision = {
        "outcome": outcome, "outcome_text": outcome_text,
        "beats_control": bool(beats_control), "beats_best_classical": bool(beats_classical),
        "delta_vs_control": delta_vs_control, "delta_vs_best_classical": delta_vs_classical,
        "seed_instability_flagged": bool(seed_instability),
        "dev_seed_cv_std_quantum": dev["summary"]["quantum"]["std"],
    }

    # ---- persist ----
    pred_df = pd.DataFrame({"id": cmp_ids, "y_true": y_test})
    for key, m in all_models.items():
        pred_df[f"{key}_proba"] = m["y_proba"]
        pred_df[f"{key}_pred"] = m["y_pred"]
    pred_df.to_csv(RESULTS_ROOT / "predictions.csv", index=False)
    assert not pred_df.isna().any().any()

    metrics_rows = [{"Model": m["display_name"], "model_key": key, "kind": m["kind"], "classifier": m["classifier"],
                      **{k: m["metrics"][k] for k in ("roc_auc", "pr_auc", "sensitivity", "specificity", "accuracy", "precision", "f1", "tn", "fp", "fn", "tp")},
                      "fit_seconds": m["fit_seconds"]} for key, m in all_models.items()]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULTS_ROOT / "metrics.csv", index=False)

    dev_rows = []
    for seed in DEV_SEEDS:
        dev_rows.append({"seed": seed, "representation": "quantum", "cv_roc_auc": dev["quantum_results"][seed]["best_cv_score"],
                          "n_evals": dev["quantum_results"][seed]["n_function_evaluations"], "seconds": dev["quantum_results"][seed]["seconds"]})
        dev_rows.append({"seed": seed, "representation": "control", "cv_roc_auc": dev["control_results"][seed]["best_cv_score"],
                          "n_evals": dev["control_results"][seed]["n_function_evaluations"], "seconds": dev["control_results"][seed]["seconds"]})
    pd.DataFrame(dev_rows).to_csv(RESULTS_ROOT / "seed_by_seed_results.csv", index=False)

    with open(RESULTS_ROOT / "model_configs.json", "w", encoding="utf-8") as fh:
        json.dump({"quantum_representation": QuantumRepresentationConfig(seed=PRIMARY_SEED).to_dict(),
                    "matched_control": MatchedControlConfig(seed=PRIMARY_SEED).to_dict(),
                    "xgb_frozen_params": XGB_FROZEN_PARAMS, "primary_seed": PRIMARY_SEED, "dev_seeds": DEV_SEEDS}, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "runtime.json", "w", encoding="utf-8") as fh:
        json.dump(runtime, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "statistical_comparisons.json", "w", encoding="utf-8") as fh:
        json.dump(comparisons, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "reproducibility_summary.json", "w", encoding="utf-8") as fh:
        json.dump(dev["summary"], fh, indent=2, default=float)

    explainability = {
        "quantum_observables": OBSERVABLE_NAMES,
        "quantum_observable_definition": "Each quantum_expZ_qubit{i} is the exact expectation value of Pauli-Z on qubit i; "
                                          "each quantum_expZZ_qubit{i}_qubit{j} is the exact expectation value of the two-qubit "
                                          "Pauli ZZ correlator on qubits i,j, after the trainable circuit. No clinical meaning is "
                                          "claimed for any individual value beyond this literal mathematical definition.",
        "quantum_theta_primary_seed": q_theta.tolist(),
        "control_phi_primary_seed": c_phi.tolist(),
        "logistic_regression_coefficients": {k: v.get("coefficients") for k, v in all_models.items() if "coefficients" in v},
        "xgboost_feature_importances": {
            repr_name: dict(zip(
                [f"pca_{i}" for i in range(4)] + {"pca_only": [], "quantum": OBSERVABLE_NAMES, "control": [f"control_{i}" for i in range(8)]}[repr_name],
                all_models[f"{repr_name}_xgboost"]["fitted_model"].feature_importances_.tolist(),
            )) for repr_name in ["pca_only", "quantum", "control"]
        },
    }
    with open(RESULTS_ROOT / "explainability.json", "w", encoding="utf-8") as fh:
        json.dump(explainability, fh, indent=2, default=float)

    summary = {
        "n_train_full": len(split.training_pool),
        "comparison_set": {"n": len(cmp_ids), "fingerprint": fingerprint, "matches_expected": fingerprint == EXPECTED_FINGERPRINT},
        "metrics": {key: m["metrics"] for key, m in all_models.items()},
        "reproducibility": dev["summary"],
        "statistics": comparisons,
        "runtime": runtime,
        "decision": decision,
    }
    with open(RESULTS_ROOT / "phase12_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"[phase12] DECISION: outcome {outcome} -- {outcome_text}", flush=True)

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
    ax.set_title(f"ROC -- Phase 12 quantum representation\nIdentical {n_cmp}-row held-out test set")
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
    ax.set_title("Precision-Recall -- Phase 12 quantum representation")
    ax.legend(loc="lower left", fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=150)
    plt.close(fig)

    n = len(items)
    ncols = (n + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 7))
    for ax, (key, m) in zip(axes.flat, items):
        mm = m["metrics"]
        cm = np.array([[mm["tn"], mm["fp"]], [mm["fn"], mm["tp"]]])
        ConfusionMatrixDisplay(cm, display_labels=["No CVD", "CVD"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(m["display_name"], fontsize=6)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"Confusion matrices -- Phase 12, identical {n_cmp}-row test set")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    result = run_phase12()
    print(json.dumps({k: v for k, v in result.items() if k not in ("statistics",)}, indent=2, default=float))
