"""Phase 7: multi-seed reproducibility check of the adaptive-feature-map
QSVM result (docs/ADAPTIVE_QSVM.md) found +0.0198 ROC-AUC over baseline,
statistically inconclusive at n=200, single seed. This module asks: does
that improvement hold across independently-drawn n_train=20,000 training
subsets?

WHAT IS REUSED, UNMODIFIED:
    - src.large_dataset.sampling.nested_stratified_stage_samples (the SAME
      stratified-draw primitive every prior stage used -- called with a
      single size and a NEW seed per trial; no new sampling code)
    - src.large_dataset.pipeline.process_stage (Phase 2 preprocessing/PCA,
      fit fresh per seed on that seed's own training subset)
    - src.quantum.feature_maps.build_feature_map (baseline zz_feature_map),
      src.quantum.adaptive_feature_map.build_adaptive_feature_map_from_training_data
      (adaptive map, training-only by construction -- see that module's
      own docstring for the structural guarantee)
    - src.quantum.kernel.compute_statevectors / kernel_matrix_from_statevectors_blockwise
      / kernel_diagnostics, src.quantum.qsvm.run_qsvm_grid_search (the SAME
      memory-safe kernel path and CV C-grid search every prior phase used)
    - src.large_dataset.statistical_robustness.delong_test,
      src.large_dataset.corrected_comparison.{paired_bootstrap_delta,
      mcnemar_exact, bootstrap_metric_ci, comparison_set_fingerprint,
      DECISION_THRESHOLD} (the SAME paired-statistics code)

SEED=42 SHORT-CIRCUIT: PreprocessingConfig() and QuantumConfig() both
default to random_seed=42, and nested_stratified_stage_samples(pool,
target, [20000], seed=42) is verified (test_adaptive_robustness.py) to
draw the IDENTICAL 20,000 rows Stage D itself used. Rather than recompute
an identical baseline+adaptive pair, seed=42's results are loaded directly
from results/large_dataset/stage_d/predictions.csv and
results/large_dataset/adaptive_qsvm/predictions.csv -- verified against
the shared test fingerprint before use. Seeds 7, 21, 84, 123 are computed
fresh.

SHARED TEST SET (Section 6): all 5 seeds vary only the TRAINING subset.
Every seed is scored on the SAME fixed 200-row test set (fingerprint
96eac11a8394b87e). This is a reproducibility/stability analysis across
training draws, NOT 5 independent test evaluations -- the cross-seed
summary and every downstream interpretation must not treat per-seed
results as independent samples of test performance.

CHECKPOINTING: each seed's full result is written to disk immediately
after that seed completes (results/large_dataset/adaptive_robustness/
seed_<n>/), so a crash partway through the ~3-4 hour multi-seed run loses
at most one seed's ~60 minutes, not the whole experiment. run_all_seeds()
skips any seed whose checkpoint already exists and is valid, making the
whole run resumable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from src.classical.evaluation import compute_classification_metrics
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
from src.large_dataset.sampling import build_fixed_split, nested_stratified_stage_samples
from src.large_dataset.schema import TARGET_COLUMN, derive_features, get_cardio_feature_groups, load_raw_cardio
from src.large_dataset.statistical_robustness import delong_test
from src.preprocessing.config import PreprocessingConfig
from src.quantum.adaptive_feature_map import build_adaptive_feature_map_from_training_data
from src.quantum.backends import get_backend
from src.quantum.config import QuantumConfig
from src.quantum.feature_maps import build_feature_map
from src.quantum.kernel import compute_statevectors, kernel_diagnostics, kernel_matrix_from_statevectors_blockwise
from src.quantum.qsvm import run_qsvm_grid_search

SEEDS = [7, 21, 42, 84, 123]
N_TRAIN = 20000
CV_FOLDS = 5
CV_SHUFFLE = True
EXPECTED_FINGERPRINT = "96eac11a8394b87e"
KERNEL_BLOCK_SIZE = 2000
KERNEL_DTYPE = np.float32

RESULTS_ROOT = find_project_root() / "results" / "large_dataset" / "adaptive_robustness"
STAGE_D_PREDICTIONS_PATH = find_project_root() / "results" / "large_dataset" / "stage_d" / "predictions.csv"
ADAPTIVE_QSVM_PREDICTIONS_PATH = find_project_root() / "results" / "large_dataset" / "adaptive_qsvm" / "predictions.csv"
ADAPTIVE_QSVM_CONFIG_PATH = find_project_root() / "results" / "large_dataset" / "adaptive_qsvm" / "adaptive_map_config.json"


@dataclass
class ModelPredictions:
    model_key: str
    ids: np.ndarray
    y_true: np.ndarray
    y_proba: np.ndarray
    y_pred: np.ndarray
    metrics: dict


def _metrics_from_proba(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    return compute_classification_metrics(y_true, y_pred, y_proba)


# --------------------------------------------------------------------------
# Fixed split (test sets never re-seeded) + per-seed training subset
# --------------------------------------------------------------------------


def build_fixed_split_once():
    df = derive_features(load_raw_cardio())
    flags = compute_cleaning_flags(df)
    modeling_df, _ = split_modeling_subset(df, flags)
    return build_fixed_split(modeling_df, TARGET_COLUMN, seed=42)


def build_seeded_training_subset(training_pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    """A fresh, stratified n_train=20,000 draw from the fixed training pool,
    independent of any other seed's draw (NOT nested -- each seed is an
    independent resample, unlike Stages A/B/C/D's nesting-by-design)."""
    result = nested_stratified_stage_samples(training_pool, TARGET_COLUMN, [N_TRAIN], seed=seed)
    return result[N_TRAIN]


# --------------------------------------------------------------------------
# Seed=42 short-circuit: reuse Stage D + adaptive_qsvm results
# --------------------------------------------------------------------------


def load_seed_42_from_existing_results() -> dict:
    if not (STAGE_D_PREDICTIONS_PATH.is_file() and ADAPTIVE_QSVM_PREDICTIONS_PATH.is_file()):
        raise FileNotFoundError(
            "Seed 42 is meant to reuse Stage D + adaptive_qsvm results (identical config), "
            f"but one or both are missing: {STAGE_D_PREDICTIONS_PATH}, {ADAPTIVE_QSVM_PREDICTIONS_PATH}"
        )
    stage_d_df = pd.read_csv(STAGE_D_PREDICTIONS_PATH)
    adaptive_df = pd.read_csv(ADAPTIVE_QSVM_PREDICTIONS_PATH)
    fp_a = comparison_set_fingerprint(stage_d_df["id"].tolist())
    fp_b = comparison_set_fingerprint(adaptive_df["id"].tolist())
    if fp_a != EXPECTED_FINGERPRINT or fp_b != EXPECTED_FINGERPRINT:
        raise ValueError(f"Fingerprint mismatch reusing seed 42 results: {fp_a}, {fp_b}")
    if stage_d_df["id"].tolist() != adaptive_df["id"].tolist():
        raise ValueError("Stage D and adaptive_qsvm predictions are not row-aligned by id.")

    baseline = ModelPredictions(
        model_key="baseline_qsvm", ids=stage_d_df["id"].to_numpy(), y_true=stage_d_df["y_true"].to_numpy(),
        y_proba=stage_d_df["qsvm_proba"].to_numpy(), y_pred=stage_d_df["qsvm_pred"].to_numpy(),
        metrics=_metrics_from_proba(stage_d_df["y_true"].to_numpy(), stage_d_df["qsvm_proba"].to_numpy()),
    )
    adaptive = ModelPredictions(
        model_key="adaptive_qsvm", ids=adaptive_df["id"].to_numpy(), y_true=adaptive_df["y_true"].to_numpy(),
        y_proba=adaptive_df["adaptive_qsvm_proba"].to_numpy(), y_pred=adaptive_df["adaptive_qsvm_pred"].to_numpy(),
        metrics=_metrics_from_proba(adaptive_df["y_true"].to_numpy(), adaptive_df["adaptive_qsvm_proba"].to_numpy()),
    )
    with open(ADAPTIVE_QSVM_CONFIG_PATH, encoding="utf-8") as fh:
        adaptive_map_config = json.load(fh)

    return {
        "seed": 42, "baseline": baseline, "adaptive": adaptive,
        "adaptive_map_config": adaptive_map_config,
        "provenance": "reused_from_stage_d_and_adaptive_qsvm", "runtime": {},
    }


# --------------------------------------------------------------------------
# Fresh per-seed computation (seeds 7, 21, 84, 123)
# --------------------------------------------------------------------------


def _run_one_feature_map(
    model_key: str, feature_map, processed, pcfg: PreprocessingConfig, backend, proc: psutil.Process,
) -> tuple[ModelPredictions, dict]:
    t0 = time.perf_counter()
    train_svs = compute_statevectors(processed.X_train_quantum, feature_map, backend)
    t1 = time.perf_counter()
    test_svs = compute_statevectors(processed.X_test_quantum, feature_map, backend)
    t2 = time.perf_counter()
    rss_before = proc.memory_info().rss / 1e6
    K_train_train = kernel_matrix_from_statevectors_blockwise(
        train_svs, train_svs, symmetric=True, block_size=KERNEL_BLOCK_SIZE, dtype=KERNEL_DTYPE,
    )
    t3 = time.perf_counter()
    rss_after_kernel = proc.memory_info().rss / 1e6
    K_test_train = kernel_matrix_from_statevectors_blockwise(
        test_svs, train_svs, symmetric=False, block_size=KERNEL_BLOCK_SIZE, dtype=KERNEL_DTYPE,
    )
    t4 = time.perf_counter()

    diag = kernel_diagnostics(K_train_train, symmetric=True)

    t5 = time.perf_counter()
    search = run_qsvm_grid_search(
        K_train_train, processed.y_train, cv_folds=CV_FOLDS, cv_shuffle=CV_SHUFFLE, random_seed=pcfg.random_seed,
    )
    cv_seconds = time.perf_counter() - t5
    rss_after_cv = proc.memory_info().rss / 1e6

    t6 = time.perf_counter()
    y_proba = search.best_estimator_.predict_proba(K_test_train)[:, 1]
    inference_seconds = time.perf_counter() - t6
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    preds = ModelPredictions(
        model_key=model_key, ids=np.arange(len(processed.y_test_quantum)),  # ids filled in by caller
        y_true=processed.y_test_quantum, y_proba=y_proba, y_pred=y_pred,
        metrics=_metrics_from_proba(processed.y_test_quantum, y_proba),
    )
    timing = {
        "train_statevectors_seconds": t1 - t0, "test_statevectors_seconds": t2 - t1,
        "K_train_train_assembly_seconds": t3 - t2, "K_test_train_assembly_seconds": t4 - t3,
        "cv_tuning_seconds": cv_seconds, "final_inference_seconds": inference_seconds,
        "total_seconds": (t4 - t0) + cv_seconds + inference_seconds,
        "best_params": {k: v for k, v in search.best_params_.items()},
        "kernel_diagnostics": diag.to_dict(),
        "process_rss_mb": {"before_kernel": rss_before, "after_kernel": rss_after_kernel, "after_cv": rss_after_cv},
    }
    return preds, timing


def run_seed_fresh(seed: int, fixed_split, cmp_ids: list, fg, pcfg: PreprocessingConfig, qcfg: QuantumConfig) -> dict:
    proc = psutil.Process()
    print(f"[adaptive_robustness] seed={seed}: building fresh n={N_TRAIN} training subset ...", flush=True)
    t0 = time.perf_counter()
    stage_train_df = build_seeded_training_subset(fixed_split.training_pool, seed)

    overlap_classical = set(stage_train_df["id"]) & set(fixed_split.test_set_classical["id"])
    overlap_quantum = set(stage_train_df["id"]) & set(fixed_split.test_set_quantum["id"])
    if overlap_classical or overlap_quantum:
        raise ValueError(f"seed={seed}: training subset overlaps a fixed test set.")
    split_seconds = time.perf_counter() - t0

    print(f"[adaptive_robustness] seed={seed}: fitting preprocessing/PCA (train-only) ...", flush=True)
    t1 = time.perf_counter()
    processed = process_stage(
        stage_train_df, fixed_split.test_set_classical, fixed_split.test_set_quantum,
        TARGET_COLUMN, fg, pcfg,
    )
    preprocessing_seconds = time.perf_counter() - t1

    print(f"[adaptive_robustness] seed={seed}: deriving adaptive map from TRAINING DATA ONLY ...", flush=True)
    t2 = time.perf_counter()
    adaptive_circuit, adaptive_report = build_adaptive_feature_map_from_training_data(
        processed.X_train_quantum, processed.n_qubits, qcfg.reps, random_state=qcfg.random_seed,
    )
    adaptive_derivation_seconds = time.perf_counter() - t2
    print(f"[adaptive_robustness] seed={seed}: selected pairs {adaptive_report.selected_pairs}", flush=True)

    baseline_circuit = build_feature_map(
        qcfg.feature_map_name, processed.n_qubits, reps=qcfg.reps, entanglement=qcfg.entanglement,
    )
    backend = get_backend(qcfg.backend_name)

    print(f"[adaptive_robustness] seed={seed}: running BASELINE QSVM ...", flush=True)
    baseline, baseline_timing = _run_one_feature_map("baseline_qsvm", baseline_circuit, processed, pcfg, backend, proc)
    baseline.ids = np.array(cmp_ids)
    print(f"[adaptive_robustness] seed={seed}: baseline done in {baseline_timing['total_seconds']:.1f}s, "
          f"roc_auc={baseline.metrics['roc_auc']:.4f}", flush=True)

    print(f"[adaptive_robustness] seed={seed}: running ADAPTIVE QSVM ...", flush=True)
    adaptive, adaptive_timing = _run_one_feature_map("adaptive_qsvm", adaptive_circuit, processed, pcfg, backend, proc)
    adaptive.ids = np.array(cmp_ids)
    print(f"[adaptive_robustness] seed={seed}: adaptive done in {adaptive_timing['total_seconds']:.1f}s, "
          f"roc_auc={adaptive.metrics['roc_auc']:.4f}", flush=True)

    assert np.array_equal(baseline.y_true, adaptive.y_true)
    assert list(baseline.ids) == cmp_ids and list(adaptive.ids) == cmp_ids

    return {
        "seed": seed, "baseline": baseline, "adaptive": adaptive,
        "adaptive_map_config": adaptive_report.to_dict(),
        "provenance": "computed_fresh",
        "runtime": {
            "split_seconds": split_seconds, "preprocessing_seconds": preprocessing_seconds,
            "adaptive_derivation_seconds": adaptive_derivation_seconds,
            "baseline": baseline_timing, "adaptive": adaptive_timing,
        },
    }


# --------------------------------------------------------------------------
# Per-seed paired statistics + persistence
# --------------------------------------------------------------------------


def compute_seed_statistics(seed_result: dict) -> dict:
    b, a = seed_result["baseline"], seed_result["adaptive"]
    dl = delong_test(b.y_true, a.y_proba, b.y_proba)
    bs_roc = paired_bootstrap_delta(b.y_true, a.y_proba, b.y_proba, "roc_auc", n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    bs_pr = paired_bootstrap_delta(b.y_true, a.y_proba, b.y_proba, "pr_auc", n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    mc = mcnemar_exact(b.y_true, a.y_pred, b.y_pred)
    return {"delong": dl.to_dict(), "bootstrap_roc_auc": bs_roc, "bootstrap_pr_auc": bs_pr, "mcnemar": mc}


def _seed_dir(seed: int) -> Path:
    return RESULTS_ROOT / f"seed_{seed}"


def persist_seed_result(seed_result: dict, stats: dict) -> None:
    d = _seed_dir(seed_result["seed"])
    d.mkdir(parents=True, exist_ok=True)
    b, a = seed_result["baseline"], seed_result["adaptive"]

    pd.DataFrame({
        "id": b.ids, "y_true": b.y_true,
        "baseline_qsvm_proba": b.y_proba, "baseline_qsvm_pred": b.y_pred,
        "adaptive_qsvm_proba": a.y_proba, "adaptive_qsvm_pred": a.y_pred,
    }).to_csv(d / "predictions.csv", index=False)

    with open(d / "seed_result.json", "w", encoding="utf-8") as fh:
        json.dump({
            "seed": seed_result["seed"], "provenance": seed_result["provenance"],
            "baseline_metrics": b.metrics, "adaptive_metrics": a.metrics,
            "adaptive_map_config": seed_result["adaptive_map_config"],
            "statistics": stats, "runtime": seed_result["runtime"],
        }, fh, indent=2, default=float)


def load_seed_checkpoint(seed: int) -> dict | None:
    d = _seed_dir(seed)
    result_path, pred_path = d / "seed_result.json", d / "predictions.csv"
    if not (result_path.is_file() and pred_path.is_file()):
        return None
    with open(result_path, encoding="utf-8") as fh:
        data = json.load(fh)
    df = pd.read_csv(pred_path)
    fp = comparison_set_fingerprint(df["id"].tolist())
    if fp != EXPECTED_FINGERPRINT:
        return None  # stale/invalid checkpoint -- recompute
    baseline = ModelPredictions(
        model_key="baseline_qsvm", ids=df["id"].to_numpy(), y_true=df["y_true"].to_numpy(),
        y_proba=df["baseline_qsvm_proba"].to_numpy(), y_pred=df["baseline_qsvm_pred"].to_numpy(),
        metrics=data["baseline_metrics"],
    )
    adaptive = ModelPredictions(
        model_key="adaptive_qsvm", ids=df["id"].to_numpy(), y_true=df["y_true"].to_numpy(),
        y_proba=df["adaptive_qsvm_proba"].to_numpy(), y_pred=df["adaptive_qsvm_pred"].to_numpy(),
        metrics=data["adaptive_metrics"],
    )
    return {
        "seed": seed, "baseline": baseline, "adaptive": adaptive,
        "adaptive_map_config": data["adaptive_map_config"], "provenance": data["provenance"] + "_checkpoint",
        "runtime": data["runtime"],
    }, data["statistics"]


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


def run_all_seeds(seeds: list[int] | None = None) -> dict:
    seeds = seeds if seeds is not None else SEEDS
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    fixed_split = build_fixed_split_once()
    cmp_ids = fixed_split.test_set_quantum["id"].tolist()
    fingerprint = comparison_set_fingerprint(cmp_ids)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"Fingerprint {fingerprint} != expected {EXPECTED_FINGERPRINT}")

    fg = get_cardio_feature_groups()

    seed_results, seed_stats = {}, {}
    for seed in seeds:
        checkpoint = load_seed_checkpoint(seed)
        if checkpoint is not None:
            print(f"[adaptive_robustness] seed={seed}: valid checkpoint found, skipping recompute.", flush=True)
            seed_results[seed], seed_stats[seed] = checkpoint
            continue

        if seed == 42:
            result = load_seed_42_from_existing_results()
        else:
            pcfg = PreprocessingConfig(random_seed=seed)
            qcfg = QuantumConfig(random_seed=seed)
            result = run_seed_fresh(seed, fixed_split, cmp_ids, fg, pcfg, qcfg)

        assert list(result["baseline"].ids) == cmp_ids
        assert list(result["adaptive"].ids) == cmp_ids

        stats = compute_seed_statistics(result)
        persist_seed_result(result, stats)
        seed_results[seed], seed_stats[seed] = result, stats
        print(f"[adaptive_robustness] seed={seed}: complete and checkpointed "
              f"(delta_roc_auc={stats['delong']['delta']:+.4f}, delong_p={stats['delong']['p_value']:.4f})", flush=True)

    return _summarize(seed_results, seed_stats, cmp_ids, fingerprint)


def _summarize(seed_results: dict, seed_stats: dict, cmp_ids: list, fingerprint: str) -> dict:
    rows = []
    for seed in sorted(seed_results):
        r, s = seed_results[seed], seed_stats[seed]
        rows.append({
            "seed": seed, "provenance": r["provenance"],
            "baseline_roc_auc": r["baseline"].metrics["roc_auc"], "adaptive_roc_auc": r["adaptive"].metrics["roc_auc"],
            "delta_roc_auc": s["delong"]["delta"],
            "baseline_pr_auc": r["baseline"].metrics["pr_auc"], "adaptive_pr_auc": r["adaptive"].metrics["pr_auc"],
            "delta_pr_auc": s["bootstrap_pr_auc"]["observed_delta"],
            "delong_p": s["delong"]["p_value"], "delong_ci_low": s["delong"]["ci_low"], "delong_ci_high": s["delong"]["ci_high"],
            "bootstrap_roc_p": s["bootstrap_roc_auc"]["bootstrap_p_value"],
            "bootstrap_pr_p": s["bootstrap_pr_auc"]["bootstrap_p_value"],
            "mcnemar_p": s["mcnemar"]["p_value"], "mcnemar_n_discordant": s["mcnemar"]["n_discordant"],
            "baseline_sensitivity": r["baseline"].metrics["sensitivity"], "baseline_specificity": r["baseline"].metrics["specificity"],
            "adaptive_sensitivity": r["adaptive"].metrics["sensitivity"], "adaptive_specificity": r["adaptive"].metrics["specificity"],
            "baseline_accuracy": r["baseline"].metrics["accuracy"], "adaptive_accuracy": r["adaptive"].metrics["accuracy"],
            "baseline_f1": r["baseline"].metrics["f1"], "adaptive_f1": r["adaptive"].metrics["f1"],
        })
    per_seed_df = pd.DataFrame(rows)

    pair_rows = []
    for seed in sorted(seed_results):
        cfg = seed_results[seed]["adaptive_map_config"]
        pair_rows.append({
            "seed": seed, "selected_pairs": json.dumps(cfg["selected_adaptive_pairs"]),
            "mi_matrix": json.dumps(cfg["mutual_information_matrix"]),
        })
    pair_df = pd.DataFrame(pair_rows)

    # Cross-seed pair-selection stability (descriptive only, Section 5).
    from collections import Counter
    pair_counter = Counter()
    for seed in seed_results:
        for pair in seed_results[seed]["adaptive_map_config"]["selected_adaptive_pairs"]:
            pair_counter[tuple(pair)] += 1

    delta_roc = per_seed_df["delta_roc_auc"].to_numpy()
    delta_pr = per_seed_df["delta_pr_auc"].to_numpy()

    def _dir_summary(deltas: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(deltas)), "median": float(np.median(deltas)),
            "std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "min": float(np.min(deltas)), "max": float(np.max(deltas)),
            "n_adaptive_better": int(np.sum(deltas > 0)), "n_adaptive_worse": int(np.sum(deltas < 0)),
            "n_seeds": len(deltas),
        }

    cross_seed_summary = {
        "roc_auc": _dir_summary(delta_roc),
        "pr_auc": _dir_summary(delta_pr),
        "n_seeds_delong_significant": int((per_seed_df["delong_p"] < 0.05).sum()),
        "n_seeds_bootstrap_roc_significant": int((per_seed_df["bootstrap_roc_p"] < 0.05).sum()),
        "n_seeds_delong_significant_after_holm": None,  # filled below
        "pair_selection_frequency": {str(k): v for k, v in pair_counter.items()},
        "shared_test_set_limitation": (
            "All seeds share the SAME fixed 200-row test set (fingerprint " + fingerprint + "). "
            "This is a training-subset reproducibility/stability analysis, NOT 5 independent test "
            "evaluations -- per-seed results are correlated through the shared test observations "
            "and must not be pooled as if drawn from independent test sets."
        ),
    }

    # Holm-Bonferroni across the 5 DeLong p-values (one family -- Phase 6A convention).
    from src.large_dataset.statistical_robustness import holm_bonferroni
    holm = holm_bonferroni(per_seed_df["delong_p"].tolist())
    for i, h in enumerate(holm):
        per_seed_df.loc[i, "delong_p_holm"] = h["adjusted_p"]
        per_seed_df.loc[i, "delong_significant_holm"] = h["significant"]
    cross_seed_summary["n_seeds_delong_significant_after_holm"] = int(per_seed_df["delong_significant_holm"].sum())

    # ---- persist ----
    per_seed_df.to_csv(RESULTS_ROOT / "per_seed_results.csv", index=False)
    pair_df.to_csv(RESULTS_ROOT / "adaptive_pair_selection.csv", index=False)

    delong_rows = [{"seed": s, **seed_stats[s]["delong"]} for s in sorted(seed_stats)]
    pd.DataFrame(delong_rows).to_csv(RESULTS_ROOT / "delong_results.csv", index=False)

    bootstrap_rows = []
    for s in sorted(seed_stats):
        bootstrap_rows.append({"seed": s, "metric": "roc_auc", **seed_stats[s]["bootstrap_roc_auc"]})
        bootstrap_rows.append({"seed": s, "metric": "pr_auc", **seed_stats[s]["bootstrap_pr_auc"]})
    pd.DataFrame(bootstrap_rows).to_csv(RESULTS_ROOT / "bootstrap_results.csv", index=False)

    mcnemar_rows = [{"seed": s, **seed_stats[s]["mcnemar"]["contingency"], "n_discordant": seed_stats[s]["mcnemar"]["n_discordant"],
                      "p_value": seed_stats[s]["mcnemar"]["p_value"]} for s in sorted(seed_stats)]
    pd.DataFrame(mcnemar_rows).to_csv(RESULTS_ROOT / "mcnemar_results.csv", index=False)

    cross_seed_df = pd.DataFrame([
        {"metric": "roc_auc", **cross_seed_summary["roc_auc"]},
        {"metric": "pr_auc", **cross_seed_summary["pr_auc"]},
    ])
    cross_seed_df.to_csv(RESULTS_ROOT / "cross_seed_summary.csv", index=False)

    summary = {
        "seeds": sorted(seed_results.keys()), "comparison_set": {"n": len(cmp_ids), "fingerprint": fingerprint},
        "cross_seed_summary": cross_seed_summary,
        "per_seed": per_seed_df.to_dict(orient="records"),
    }
    with open(RESULTS_ROOT / "statistical_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    _plot_cross_seed(per_seed_df, RESULTS_ROOT)

    return summary


def _plot_cross_seed(per_seed_df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in [(axes[0], "roc_auc", "ROC-AUC"), (axes[1], "pr_auc", "PR-AUC")]:
        x = np.arange(len(per_seed_df))
        ax.bar(x - 0.2, per_seed_df[f"baseline_{metric}"], width=0.4, label="Baseline", color="#888888")
        ax.bar(x + 0.2, per_seed_df[f"adaptive_{metric}"], width=0.4, label="Adaptive", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(per_seed_df["seed"])
        ax.set_xlabel("Training seed")
        ax.set_ylabel(label)
        ax.set_title(f"{label}: baseline vs adaptive across seeds\n(SAME shared 200-row test set)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cross_seed_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.bar(per_seed_df["seed"].astype(str), per_seed_df["delta_roc_auc"],
           color=["#1f77b4" if d > 0 else "#d62728" for d in per_seed_df["delta_roc_auc"]])
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Delta ROC-AUC (Adaptive - Baseline)")
    ax.set_title("Adaptive - Baseline ROC-AUC delta across seeds\n(shared 200-row test set, not independent samples)")
    fig.tight_layout()
    fig.savefig(out_dir / "delta_roc_auc_across_seeds.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    result = run_all_seeds()
    print(json.dumps(result["cross_seed_summary"], indent=2, default=float))
