"""Phase 9: classical ceiling benchmark + fair quantum-vs-classical
comparison (see docs/PHASE_9_CLASSICAL_BENCHMARK.md).

PURPOSE: evidence-gathering, not a metric-maximization contest. Determines
(1) how strong the best classical model is on this dataset/split, (2)
whether the existing QSVM variants are competitive, (3) whether the
MI-adaptive QSVM has earned a more expensive follow-up experiment, and
(4) which model should be the SIH prototype's final candidate.

WHAT IS REUSED, UNMODIFIED:
    - src.large_dataset.phase8a_label_aware_screening.build_screening_split
      (the IDENTICAL n_train=2,000, seed=42 training subset and fixed
      200-row test set every Phase 8 experiment used)
    - src.classical.models.get_available_models (the SAME 4-model
      registry -- Logistic Regression, RBF-SVM, Random Forest, XGBoost --
      Stage A-D used, full unreduced grids: cheap at n=2,000)
    - src.classical.tuning.run_grid_search (Phase 3's per-fold-refit CV
      methodology, unchanged)
    - src.classical.evaluation.evaluate_on_test (the ONLY function that
      touches the locked test set; called here with the 200-row set
      instead of the usual 2,000-row classical set, so EVERY model in
      this phase -- classical and quantum -- is scored on the identical
      observations)
    - src.large_dataset.corrected_comparison / statistical_robustness
      (the same paired-statistics functions every prior phase used)

WHAT IS REUSED FROM PHASE 8A, NOT RETRAINED: baseline QSVM, MI-adaptive
QSVM, and label-aware QSVM predictions on the 200-row test set, at this
EXACT n_train=2,000/seed=42 setup, are already fully persisted at
results/large_dataset/phase8a_label_aware/predictions.csv -- verified
against the same test fingerprint before use. None are retrained here.

TEST-SET DISCIPLINE: each classical model's preprocessing and
hyperparameters are selected via 5-fold CV on the n=2,000 training subset
ONLY (run_grid_search never sees the 200-row test set); the test set is
scored exactly once per model, after selection is frozen.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.classical.evaluation import evaluate_on_test
from src.classical.models import get_available_models
from src.classical.tuning import run_grid_search
from src.data.inspect_dataset import find_project_root
from src.large_dataset.corrected_comparison import (
    DECISION_THRESHOLD,
    comparison_set_fingerprint,
    mcnemar_exact,
    paired_bootstrap_delta,
)
from src.large_dataset.phase8a_label_aware_screening import SCREEN_N_TRAIN, build_screening_split
from src.large_dataset.schema import TARGET_COLUMN, get_cardio_feature_groups
from src.large_dataset.statistical_robustness import delong_test, holm_bonferroni
from src.preprocessing.config import PreprocessingConfig

EXPECTED_FINGERPRINT = "96eac11a8394b87e"
PHASE8A_PREDICTIONS_PATH = find_project_root() / "results" / "large_dataset" / "phase8a_label_aware" / "predictions.csv"
RESULTS_ROOT = find_project_root() / "results" / "large_dataset" / "phase9_classical_benchmark"

CV_FOLDS = 5
CV_SHUFFLE = True


def load_phase8a_quantum_predictions() -> pd.DataFrame:
    """Reuses Phase 8A's baseline/MI-adaptive/label-aware QSVM predictions
    -- never retrains any quantum model in this phase."""
    if not PHASE8A_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Expected Phase 8A predictions at {PHASE8A_PREDICTIONS_PATH}; Phase 9 reuses "
            f"the quantum predictions from there rather than retraining. Run Phase 8A first."
        )
    df = pd.read_csv(PHASE8A_PREDICTIONS_PATH)
    fp = comparison_set_fingerprint(df["id"].tolist())
    if fp != EXPECTED_FINGERPRINT:
        raise ValueError(f"Phase 8A predictions.csv fingerprint {fp} != expected {EXPECTED_FINGERPRINT}")
    return df


def run_classical_benchmark(screen, feature_groups, preprocessing_config) -> tuple[list, dict]:
    """Runs the full (unreduced -- cheap at n=2,000) classical grid for
    all 4 models, scored on the 200-row quantum test set (not the usual
    2,000-row classical test set), so classical and quantum models in
    this phase share the IDENTICAL held-out observations."""
    feature_cols = feature_groups.all_columns
    X_train_raw = screen.train_df[feature_cols]
    y_train = screen.train_df[TARGET_COLUMN].to_numpy()
    X_test_raw = screen.split.test_set_quantum[feature_cols]
    y_test = screen.split.test_set_quantum[TARGET_COLUMN].to_numpy()

    models = get_available_models(preprocessing_config.random_seed)
    results = []
    search_seconds = {}
    for key, spec in models.items():
        n_combos = 1
        for v in spec.param_grid.values():
            n_combos *= len(v)
        print(f"[phase9] classical: starting {key} grid search ({n_combos} combinations x {CV_FOLDS} folds, "
              f"n_train={len(X_train_raw)}) ...", flush=True)
        t0 = time.perf_counter()
        search = run_grid_search(
            spec, "classical", feature_groups, preprocessing_config,
            X_train_raw, y_train, cv_folds=CV_FOLDS, cv_shuffle=CV_SHUFFLE,
            random_seed=preprocessing_config.random_seed,
        )
        search_seconds[key] = time.perf_counter() - t0
        result = evaluate_on_test(
            search, "phase9_classical_benchmark", key, spec.display_name,
            "classical", X_test_raw, y_test, n_train=len(X_train_raw),
            random_seed=preprocessing_config.random_seed,
        )
        results.append(result)
        print(f"[phase9] classical: finished {key} in {search_seconds[key]:.1f}s "
              f"(roc_auc={result.test_metrics['roc_auc']:.4f}, best_params={result.best_params})", flush=True)
    return results, search_seconds


def run_phase9(*, make_plots: bool = True) -> dict:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    runtime: dict = {}

    print("[phase9] loading Phase 8A quantum predictions (no retraining) ...", flush=True)
    q_df = load_phase8a_quantum_predictions()

    print(f"[phase9] rebuilding Phase 8A's exact n_train={SCREEN_N_TRAIN} screening split ...", flush=True)
    t0 = time.perf_counter()
    screen = build_screening_split()
    runtime["split_seconds"] = time.perf_counter() - t0

    cmp_ids = screen.split.test_set_quantum["id"].tolist()
    fingerprint = comparison_set_fingerprint(cmp_ids)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"Comparison set fingerprint {fingerprint} != expected {EXPECTED_FINGERPRINT}")
    if q_df["id"].tolist() != cmp_ids:
        raise ValueError("Phase 8A predictions' ids are not aligned to the current comparison set.")

    fg = get_cardio_feature_groups()
    pcfg = PreprocessingConfig()

    print("[phase9] ===== classical benchmark (4 models, full grids, n_train=%d) =====" % SCREEN_N_TRAIN, flush=True)
    t1 = time.perf_counter()
    classical_results, classical_search_seconds = run_classical_benchmark(screen, fg, pcfg)
    runtime["classical_total_seconds"] = time.perf_counter() - t1
    runtime["classical_per_model_seconds"] = classical_search_seconds

    # ---- assemble all models (classical + reused quantum) into one table ----
    all_models: dict[str, dict] = {}
    for r in classical_results:
        all_models[r.model_key] = {
            "display_name": r.display_name, "kind": "classical",
            "y_true": r.y_test_true, "y_proba": r.y_test_proba,
            "y_pred": (r.y_test_proba >= DECISION_THRESHOLD).astype(int),
            "metrics": r.test_metrics, "best_params": r.best_params,
            # NOTE: this is the FULL grid-search wall time (all combinations x
            # folds), not ModelResult.fit_time_seconds (which only times the
            # single final refit-on-full-data call and would understate the
            # actual cost of producing this model by ~1-2 orders of magnitude).
            "fit_time_seconds": classical_search_seconds[r.model_key],
        }
    for key, display in [("baseline", "Baseline QSVM"), ("mi_adaptive", "MI-adaptive QSVM"), ("label_aware", "Label-aware QSVM")]:
        proba_col, pred_col = f"{key}_proba", f"{key}_pred"
        y_true = q_df["y_true"].to_numpy()
        y_proba = q_df[proba_col].to_numpy()
        y_pred = q_df[pred_col].to_numpy()
        from src.classical.evaluation import compute_classification_metrics
        all_models[key] = {
            "display_name": display, "kind": "quantum",
            "y_true": y_true, "y_proba": y_proba, "y_pred": y_pred,
            "metrics": compute_classification_metrics(y_true, y_pred, y_proba),
            # Reused from Phase 8A, not retrained/retimed here -- see
            # docs/PHASE_9_CLASSICAL_BENCHMARK.md for Phase 8A's own
            # measured training time for each of these.
            "best_params": {}, "fit_time_seconds": None,
        }

    for m in all_models.values():
        assert list(cmp_ids) == cmp_ids  # sanity: same list object semantics
        assert np.array_equal(m["y_true"], all_models["baseline"]["y_true"])

    best_classical_key = max(
        (k for k, v in all_models.items() if v["kind"] == "classical"),
        key=lambda k: all_models[k]["metrics"]["roc_auc"],
    )
    best_quantum_key = max(
        ["baseline", "mi_adaptive"],
        key=lambda k: all_models[k]["metrics"]["roc_auc"],
    )
    print(f"[phase9] best classical = {best_classical_key} (roc_auc={all_models[best_classical_key]['metrics']['roc_auc']:.4f}); "
          f"best quantum (baseline/MI-adaptive only) = {best_quantum_key} (roc_auc={all_models[best_quantum_key]['metrics']['roc_auc']:.4f})",
          flush=True)

    # ---- Step 7: 3 primary paired comparisons, one Holm family ----
    print("[phase9] computing paired statistical comparisons ...", flush=True)
    y_true = all_models["baseline"]["y_true"]

    def _pair(key_a: str, key_b: str) -> dict:
        pa, pb = all_models[key_a]["y_proba"], all_models[key_b]["y_proba"]
        preda, predb = all_models[key_a]["y_pred"], all_models[key_b]["y_pred"]
        dl = delong_test(y_true, pa, pb)
        bs = paired_bootstrap_delta(y_true, pa, pb, "roc_auc", n_resamples=2000, seed=42)
        mc = mcnemar_exact(y_true, preda, predb)
        return {"a": key_a, "b": key_b, "delong": dl.to_dict(), "bootstrap": bs, "mcnemar": mc}

    comparisons = {
        "best_classical_vs_baseline_qsvm": _pair(best_classical_key, "baseline"),
        "best_classical_vs_mi_adaptive_qsvm": _pair(best_classical_key, "mi_adaptive"),
        "mi_adaptive_vs_baseline_qsvm": _pair("mi_adaptive", "baseline"),
    }
    raw_p = [c["delong"]["p_value"] for c in comparisons.values()]
    holm = holm_bonferroni(raw_p)
    for (name, c), h in zip(comparisons.items(), holm):
        c["delong_p_holm"] = h["adjusted_p"]
        c["delong_significant_holm"] = h["significant"]

    # ---- Step 8: quantum advantage decision rule ----
    best_q_vs_c = comparisons["best_classical_vs_baseline_qsvm"] if best_quantum_key == "baseline" \
        else comparisons["best_classical_vs_mi_adaptive_qsvm"]
    delta_q_minus_c = -best_q_vs_c["delong"]["delta"]  # delong stored as (a - b) = classical - quantum
    significant = best_q_vs_c["delong_significant_holm"] and not best_q_vs_c["bootstrap"]["ci_includes_zero"]
    if delta_q_minus_c > 0 and significant:
        decision_outcome = "A"
        decision_text = "Quantum model clearly beats the strongest classical model; improvement is statistically supported."
    elif delta_q_minus_c > 0 and not significant:
        decision_outcome = "B"
        decision_text = "Quantum model numerically beats classical, but evidence is not statistically convincing."
    elif delta_q_minus_c <= 0 and significant:
        decision_outcome = "C"
        decision_text = "Classical model remains clearly stronger (statistically supported)."
    else:
        decision_outcome = "D"
        decision_text = "Results are inconclusive."

    decision = {
        "outcome": decision_outcome, "outcome_text": decision_text,
        "best_classical_model": best_classical_key, "best_quantum_model": best_quantum_key,
        "delta_roc_auc_quantum_minus_classical": delta_q_minus_c,
        "delong_p_holm": best_q_vs_c["delong_p_holm"],
        "bootstrap_ci_includes_zero": best_q_vs_c["bootstrap"]["ci_includes_zero"],
    }

    # ---- persist ----
    pred_df = pd.DataFrame({"id": cmp_ids, "y_true": y_true})
    for key, m in all_models.items():
        pred_df[f"{key}_proba"] = m["y_proba"]
        pred_df[f"{key}_pred"] = m["y_pred"]
    pred_df.to_csv(RESULTS_ROOT / "predictions.csv", index=False)

    metrics_rows = []
    for key, m in all_models.items():
        row = {"Model": m["display_name"], "model_key": key, "kind": m["kind"],
               **{k: m["metrics"][k] for k in ("roc_auc", "pr_auc", "sensitivity", "specificity", "accuracy", "precision", "f1", "tn", "fp", "fn", "tp")},
               "runtime_seconds": m["fit_time_seconds"]}
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULTS_ROOT / "metrics.csv", index=False)

    comparison_table = metrics_df[["Model", "roc_auc", "pr_auc", "sensitivity", "specificity", "accuracy", "f1", "runtime_seconds"]].copy()
    comparison_table.columns = ["Model", "ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "Accuracy", "F1", "Runtime (s)"]
    comparison_table.to_csv(RESULTS_ROOT / "comparison_table.csv", index=False)

    model_configs = {key: {"best_params": m["best_params"], "kind": m["kind"]} for key, m in all_models.items()}
    with open(RESULTS_ROOT / "model_configs.json", "w", encoding="utf-8") as fh:
        json.dump(model_configs, fh, indent=2, default=float)

    with open(RESULTS_ROOT / "runtime.json", "w", encoding="utf-8") as fh:
        json.dump(runtime, fh, indent=2, default=float)

    with open(RESULTS_ROOT / "statistics.json", "w", encoding="utf-8") as fh:
        json.dump(comparisons, fh, indent=2, default=float)

    summary = {
        "n_train": SCREEN_N_TRAIN,
        "comparison_set": {"n": len(cmp_ids), "fingerprint": fingerprint, "matches_expected": fingerprint == EXPECTED_FINGERPRINT},
        "metrics": {key: m["metrics"] for key, m in all_models.items()},
        "best_classical_model": best_classical_key,
        "best_quantum_model": best_quantum_key,
        "statistics": comparisons,
        "runtime": runtime,
        "decision": decision,
    }
    with open(RESULTS_ROOT / "phase9_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"[phase9] DECISION: outcome {decision_outcome} -- {decision_text}", flush=True)

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
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC -- Phase 9 classical ceiling benchmark (n_train={items[0][1] and 2000:,})\nIdentical {n_cmp}-row held-out test set")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for key, m in items:
        prec, rec, _ = precision_recall_curve(m["y_true"], m["y_proba"])
        ax.plot(rec, prec, label=f"{m['display_name']} (AP={m['metrics']['pr_auc']:.4f})")
    base = float(np.mean(items[0][1]["y_true"]))
    ax.axhline(base, ls="--", color="grey", label=f"Chance ({base:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall -- Phase 9 classical ceiling benchmark")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=150)
    plt.close(fig)

    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5))
    for ax, (key, m) in zip(np.atleast_1d(axes), items):
        mm = m["metrics"]
        cm = np.array([[mm["tn"], mm["fp"]], [mm["fn"], mm["tp"]]])
        ConfusionMatrixDisplay(cm, display_labels=["No CVD", "CVD"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(m["display_name"], fontsize=8)
    fig.suptitle(f"Confusion matrices -- Phase 9, identical {n_cmp}-row test set")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    result = run_phase9()
    print(json.dumps({k: v for k, v in result.items() if k != "statistics"}, indent=2, default=float))
