"""Phase 6A: Statistical robustness analysis of the corrected classical-vs-QSVM
comparison (src.large_dataset.corrected_comparison).

Reuses, unmodified, from corrected_comparison.py:
    - paired_bootstrap_delta   (paired bootstrap of Delta-AUC / Delta-PR-AUC)
    - bootstrap_metric_ci      (per-model metric CIs)
    - mcnemar_exact            (paired 0.5-threshold label comparison)

The only genuinely new statistical method here is DeLong's test, which has
no maintained implementation in the project's existing dependencies
(scikit-learn does not ship one, and no third-party DeLong package is
installed -- verified before writing this). It is implemented directly
using the standard structural-components method (DeLong, DeLong & Clarke-
Pearson, 1988; the compact form is sometimes called "Sun & Xu's fast
DeLong"). At n=200 the dense O(n1*n0) computation is trivial, so no
approximation is needed.

WHY A PAIRED TEST IS REQUIRED (not an independent-samples test): every
model's predictions come from the SAME 200 observations. An independent-
samples AUC test (e.g. Hanley-McNeil) assumes the two AUCs were measured on
unrelated samples and would overstate the variance of the difference,
because it ignores the positive correlation induced by scoring the same
cases. DeLong's method and the paired bootstrap both explicitly account for
this correlation via the shared observations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

from src.data.inspect_dataset import find_project_root
from src.large_dataset.corrected_comparison import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DECISION_THRESHOLD,
    bootstrap_metric_ci,
    comparison_set_fingerprint,
    mcnemar_exact,
    paired_bootstrap_delta,
)
from src.classical.evaluation import compute_classification_metrics

RESULTS_ROOT = find_project_root() / "results" / "large_dataset" / "statistical_robustness"
CORRECTED_ROOT = find_project_root() / "results" / "large_dataset" / "corrected_comparison"

STAGES = {"A": 1000, "B": 5000, "C": 10000}
CLASSICAL_MODELS = ["logistic_regression", "rbf_svm", "random_forest", "xgboost"]
ALL_MODELS = CLASSICAL_MODELS + ["qsvm"]
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "rbf_svm": "RBF-SVM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "qsvm": "QSVM (fidelity kernel)",
}
EXPECTED_FINGERPRINT = "96eac11a8394b87e"
ALPHA = 0.05


# --------------------------------------------------------------------------
# DeLong's test (paired) -- structural components method
# --------------------------------------------------------------------------


def _midrank_psi_sums(pos_scores: np.ndarray, neg_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute V10 (per positive case) and V01 (per negative case) structural
    components for ONE classifier's scores, per DeLong et al. (1988).

    psi(x, y) = 1 if x > y, 0.5 if x == y, 0 if x < y.
    V10[i] = mean_j psi(pos[i], neg[j])   -- length n1
    V01[j] = mean_i psi(pos[i], neg[j])   -- length n0

    mean(V10) == mean(V01) == AUC (the Mann-Whitney U statistic form).
    """
    n1, n0 = len(pos_scores), len(neg_scores)
    # (n1, n0) comparison matrix, vectorized (dense is fine at n<=~10k here, n=200)
    diff = pos_scores[:, None] - neg_scores[None, :]
    psi = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))
    V10 = psi.mean(axis=1)  # shape (n1,)
    V01 = psi.mean(axis=0)  # shape (n0,)
    return V10, V01


@dataclass(frozen=True)
class DeLongResult:
    auc_a: float
    auc_b: float
    delta: float
    var_delta: float
    ci_low: float
    ci_high: float
    z: float
    p_value: float

    def to_dict(self) -> dict:
        return {
            "auc_a": self.auc_a, "auc_b": self.auc_b, "delta": self.delta,
            "var_delta": self.var_delta, "ci_low": self.ci_low, "ci_high": self.ci_high,
            "z": self.z, "p_value": self.p_value,
        }


def delong_test(y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray) -> DeLongResult:
    """Paired DeLong test for AUC_a - AUC_b on the SAME observations.

    Returns a 95% CI (normal-approximation, from the DeLong variance
    estimate) and a two-sided z-test p-value for H0: AUC_a == AUC_b.
    """
    y_true = np.asarray(y_true)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n1, n0 = int(pos_mask.sum()), int(neg_mask.sum())
    if n1 < 2 or n0 < 2:
        raise ValueError(f"DeLong requires >=2 cases per class; got n1={n1}, n0={n0}.")

    V10_a, V01_a = _midrank_psi_sums(proba_a[pos_mask], proba_a[neg_mask])
    V10_b, V01_b = _midrank_psi_sums(proba_b[pos_mask], proba_b[neg_mask])

    auc_a = float(V10_a.mean())
    auc_b = float(V10_b.mean())
    # Sanity: V01 mean must agree with V10 mean (same AUC via either marginal)
    assert abs(auc_a - float(V01_a.mean())) < 1e-9
    assert abs(auc_b - float(V01_b.mean())) < 1e-9

    V10 = np.vstack([V10_a, V10_b])  # (2, n1)
    V01 = np.vstack([V01_a, V01_b])  # (2, n0)

    S10 = np.cov(V10, ddof=1)  # (2,2)
    S01 = np.cov(V01, ddof=1)  # (2,2)
    cov_theta = S10 / n1 + S01 / n0  # covariance matrix of [auc_a, auc_b]

    var_delta = float(cov_theta[0, 0] + cov_theta[1, 1] - 2 * cov_theta[0, 1])
    delta = auc_a - auc_b

    if var_delta <= 0:
        # Degenerate (e.g. identical scores) -- delta is exactly 0 with no variance.
        se = 0.0
        z = 0.0
        p_value = 1.0
        ci_low = ci_high = 0.0
    else:
        se = float(np.sqrt(var_delta))
        z = delta / se
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
        ci_low = delta - 1.959963984540054 * se
        ci_high = delta + 1.959963984540054 * se

    return DeLongResult(
        auc_a=auc_a, auc_b=auc_b, delta=delta, var_delta=var_delta,
        ci_low=ci_low, ci_high=ci_high, z=z, p_value=p_value,
    )


# --------------------------------------------------------------------------
# Holm-Bonferroni correction
# --------------------------------------------------------------------------


def holm_bonferroni(p_values: list[float], alpha: float = ALPHA) -> list[dict]:
    """Holm-Bonferroni step-down correction across a family of p-values.

    Applied ONCE across the full family of 12 primary AUC comparisons (4
    classical models x 3 stages), not independently within each stage --
    the 12 comparisons are the declared family of primary hypotheses for
    this analysis (Task 4 instruction), so correcting only within a
    3-comparison subfamily per stage would understate the multiplicity.

    Returns a list (in the ORIGINAL input order) of
    {raw_p, adjusted_p, significant} dicts.
    """
    n = len(p_values)
    order = np.argsort(p_values)  # ascending
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):  # rank 0..n-1, i.e. step 1..n
        m_minus_k_plus_1 = n - rank
        adj = p_values[idx] * m_minus_k_plus_1
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return [
        {"raw_p": float(p_values[i]), "adjusted_p": float(adjusted[i]), "significant": bool(adjusted[i] < alpha)}
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# Data loading (reuses the already-persisted, verified predictions)
# --------------------------------------------------------------------------


def load_stage_predictions(stage_n: int) -> pd.DataFrame:
    path = CORRECTED_ROOT / f"stage_{stage_n}" / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected persisted predictions at {path}, but the file does not exist. "
            f"Per Task 1, predictions must already exist from the corrected-comparison "
            f"phase; retraining is not attempted here."
        )
    df = pd.read_csv(path)
    fp = comparison_set_fingerprint(df["id"].tolist())
    if fp != EXPECTED_FINGERPRINT:
        raise ValueError(
            f"predictions.csv for stage {stage_n} has fingerprint {fp}, expected "
            f"{EXPECTED_FINGERPRINT}. Refusing to proceed with a comparison set that "
            f"does not match the established 200-row identity."
        )
    return df


def verify_cross_stage_consistency(dfs: dict[int, pd.DataFrame]) -> None:
    """Task 9 data-integrity check: ids and y_true must be IDENTICAL across
    every stage's prediction file -- if they are not, the "same test set"
    premise of this entire analysis is violated.
    """
    ref_n = list(dfs.keys())[0]
    ref_ids = dfs[ref_n]["id"].tolist()
    ref_y = dfs[ref_n]["y_true"].tolist()
    for n, df in dfs.items():
        if df["id"].tolist() != ref_ids:
            raise ValueError(f"Stage {n} ids differ from stage {ref_n} -- test sets are not identical.")
        if df["y_true"].tolist() != ref_y:
            raise ValueError(f"Stage {n} y_true differs from stage {ref_n} -- labels are not identical.")
        if df["id"].duplicated().any():
            raise ValueError(f"Stage {n} has duplicate ids in the comparison set.")
        for c in df.columns:
            if c.endswith("_proba") and not df[c].between(0.0, 1.0).all():
                raise ValueError(f"Stage {n} column {c} has out-of-range probabilities.")


# --------------------------------------------------------------------------
# Per-model metric CIs (Task 6)
# --------------------------------------------------------------------------

METRIC_KEYS = ["roc_auc", "pr_auc", "sensitivity", "specificity", "accuracy", "precision", "f1"]


def compute_all_metric_cis(dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Bootstrap 95% CI for all 7 metrics, all 5 models, all 3 stages (105 rows).

    ROC-AUC and PR-AUC use the existing bootstrap_metric_ci (paired by
    construction -- same resampled indices applied to y_true/y_proba).
    The five 0.5-threshold label metrics (sensitivity, specificity,
    accuracy, precision, f1) are bootstrapped here with the identical
    resampling scheme for consistency, reusing compute_classification_metrics.
    """
    rows = []
    rng_master = np.random.RandomState(BOOTSTRAP_SEED)
    for stage_label, stage_n in STAGES.items():
        df = dfs[stage_n]
        y_true = df["y_true"].to_numpy()
        n = len(y_true)
        for model in ALL_MODELS:
            proba = df[f"{model}_proba"].to_numpy()

            roc_ci = bootstrap_metric_ci(y_true, proba, "roc_auc", n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
            pr_ci = bootstrap_metric_ci(y_true, proba, "pr_auc", n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)

            # Label-metric bootstrap: same resampling scheme (seeded RandomState,
            # regenerated per model/stage from the master seed for independence
            # across rows while remaining fully reproducible).
            rng = np.random.RandomState(BOOTSTRAP_SEED)
            label_samples = {m: [] for m in ["sensitivity", "specificity", "accuracy", "precision", "f1"]}
            point = compute_classification_metrics(y_true, (proba >= DECISION_THRESHOLD).astype(int), proba)
            for _ in range(BOOTSTRAP_RESAMPLES):
                idx = rng.randint(0, n, size=n)
                yt, pr = y_true[idx], proba[idx]
                if len(np.unique(yt)) < 2:
                    continue
                m = compute_classification_metrics(yt, (pr >= DECISION_THRESHOLD).astype(int), pr)
                for key in label_samples:
                    label_samples[key].append(m[key])

            row = {
                "stage": stage_label, "n_train": stage_n, "model": model,
                "display_name": DISPLAY_NAMES[model],
                "roc_auc": point["roc_auc"], "roc_auc_ci_low": roc_ci["ci_low"], "roc_auc_ci_high": roc_ci["ci_high"],
                "pr_auc": point["pr_auc"], "pr_auc_ci_low": pr_ci["ci_low"], "pr_auc_ci_high": pr_ci["ci_high"],
            }
            for key, samples in label_samples.items():
                arr = np.array(samples)
                lo, hi = np.quantile(arr, [0.025, 0.975]) if len(arr) else (np.nan, np.nan)
                row[key] = point[key]
                row[f"{key}_ci_low"] = float(lo)
                row[f"{key}_ci_high"] = float(hi)
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plots (Task 10)
# --------------------------------------------------------------------------


def plot_forest(auc_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    labels, deltas, los, his, colors = [], [], [], [], []
    for _, row in auc_df.iterrows():
        labels.append(f"{row['stage']} (n={row['n_train']:,}): {DISPLAY_NAMES[row['classical_model']]} vs QSVM")
        deltas.append(row["delong_delta"])
        los.append(row["delong_ci_low"])
        his.append(row["delong_ci_high"])
        colors.append("#1f77b4" if row["delong_ci_low"] > 0 else ("#d62728" if row["delong_ci_high"] < 0 else "#888888"))

    y_pos = np.arange(len(labels))[::-1]
    for y, d, lo, hi, c in zip(y_pos, deltas, los, his, colors):
        ax.plot([lo, hi], [y, y], color=c, lw=2)
        ax.plot(d, y, "o", color=c, markersize=6)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Delta ROC-AUC (Classical - QSVM), DeLong 95% CI")
    ax.set_title("Forest plot: paired DeLong Delta-AUC, identical 200-row test set\n"
                  "(blue: classical > QSVM, excludes 0; red: QSVM > classical, excludes 0; grey: CI includes 0)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_auc_across_stages(auc_df: pd.DataFrame, metric_ci_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    stage_order = ["A", "B", "C"]
    x = np.arange(len(stage_order))
    for model in ALL_MODELS:
        sub = metric_ci_df[metric_ci_df["model"] == model].set_index("stage").loc[stage_order]
        ax.errorbar(
            x, sub["roc_auc"],
            yerr=[sub["roc_auc"] - sub["roc_auc_ci_low"], sub["roc_auc_ci_high"] - sub["roc_auc"]],
            marker="o", capsize=4, label=DISPLAY_NAMES[model],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={STAGES[s]:,})" for s in stage_order])
    ax.set_ylabel("ROC-AUC (bootstrap 95% CI)")
    ax.set_title("ROC-AUC across 3 measured points -- identical 200-row test set\n"
                  "(3 points only: NOT a fitted trend line, no interpolation implied)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


def run_statistical_robustness_analysis() -> dict:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    dfs = {n: load_stage_predictions(n) for n in STAGES.values()}
    verify_cross_stage_consistency(dfs)

    # ---- Tasks 2, 3, 5: the 12 paired comparisons ----
    delong_rows, bootstrap_rows, mcnemar_rows, auc_rows = [], [], [], []
    for stage_label, stage_n in STAGES.items():
        df = dfs[stage_n]
        y_true = df["y_true"].to_numpy()
        qsvm_proba = df["qsvm_proba"].to_numpy()
        qsvm_pred = df["qsvm_pred"].to_numpy()

        for model in CLASSICAL_MODELS:
            proba = df[f"{model}_proba"].to_numpy()
            pred = df[f"{model}_pred"].to_numpy()

            dl = delong_test(y_true, proba, qsvm_proba)
            bs = paired_bootstrap_delta(y_true, proba, qsvm_proba, "roc_auc",
                                          n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
            bs_pr = paired_bootstrap_delta(y_true, proba, qsvm_proba, "pr_auc",
                                             n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
            mc = mcnemar_exact(y_true, pred, qsvm_pred)

            delong_rows.append({
                "stage": stage_label, "n_train": stage_n, "classical_model": model,
                **dl.to_dict(),
            })
            bootstrap_rows.append({
                "stage": stage_label, "n_train": stage_n, "classical_model": model,
                "metric": "roc_auc", **{k: v for k, v in bs.items() if k != "metric"},
            })
            bootstrap_rows.append({
                "stage": stage_label, "n_train": stage_n, "classical_model": model,
                "metric": "pr_auc", **{k: v for k, v in bs_pr.items() if k != "metric"},
            })
            mcnemar_rows.append({
                "stage": stage_label, "n_train": stage_n, "classical_model": model,
                **mc["contingency"], "n_discordant": mc["n_discordant"],
                "test": mc["test"], "p_value": mc["p_value"],
            })
            auc_rows.append({
                "stage": stage_label, "n_train": stage_n, "classical_model": model,
                "auc_classical": dl.auc_a, "auc_qsvm": dl.auc_b,
                "delong_delta": dl.delta, "delong_ci_low": dl.ci_low, "delong_ci_high": dl.ci_high,
                "delong_p": dl.p_value,
                "bootstrap_delta": bs["observed_delta"], "bootstrap_ci_low": bs["ci_low"],
                "bootstrap_ci_high": bs["ci_high"], "bootstrap_p": bs["bootstrap_p_value"],
                "mcnemar_p": mc["p_value"], "mcnemar_n_discordant": mc["n_discordant"],
            })

    delong_df = pd.DataFrame(delong_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    mcnemar_df = pd.DataFrame(mcnemar_rows)
    auc_df = pd.DataFrame(auc_rows)
    assert len(auc_df) == 12, f"expected 12 primary comparisons, got {len(auc_df)}"

    # ---- Task 4: Holm-Bonferroni across the 12 DeLong p-values (primary),
    #      and separately across the 12 bootstrap p-values (secondary check) ----
    holm_delong = holm_bonferroni(auc_df["delong_p"].tolist())
    holm_bootstrap = holm_bonferroni(auc_df["bootstrap_p"].tolist())
    for i, (hd, hb) in enumerate(zip(holm_delong, holm_bootstrap)):
        auc_df.loc[i, "delong_p_holm"] = hd["adjusted_p"]
        auc_df.loc[i, "delong_significant_holm"] = hd["significant"]
        auc_df.loc[i, "bootstrap_p_holm"] = hb["adjusted_p"]
        auc_df.loc[i, "bootstrap_significant_holm"] = hb["significant"]

    # ---- Task 6: metric CIs for all 5 models x 3 stages ----
    metric_ci_df = compute_all_metric_cis(dfs)

    # ---- Task 7: effect-size / gap analysis ----
    gap_rows = []
    for stage_label, stage_n in STAGES.items():
        sub = auc_df[auc_df["stage"] == stage_label]
        best_idx = sub["auc_classical"].idxmax()
        best_row = sub.loc[best_idx]
        gap_rows.append({
            "stage": stage_label, "n_train": stage_n,
            "qsvm_auc": best_row["auc_qsvm"],
            **{f"{m}_auc": auc_df[(auc_df.stage == stage_label) & (auc_df.classical_model == m)]["auc_classical"].iloc[0]
               for m in CLASSICAL_MODELS},
            "best_classical_model": best_row["classical_model"],
            "best_classical_auc": best_row["auc_classical"],
            "delta_auc": best_row["delong_delta"],
            "delong_ci_low": best_row["delong_ci_low"],
            "delong_ci_high": best_row["delong_ci_high"],
            "delong_p_holm": auc_df.loc[best_idx, "delong_p_holm"],
        })
    gap_df = pd.DataFrame(gap_rows)

    gap_A = float(gap_df.loc[gap_df.stage == "A", "delta_auc"].iloc[0])
    gap_B = float(gap_df.loc[gap_df.stage == "B", "delta_auc"].iloc[0])
    gap_C = float(gap_df.loc[gap_df.stage == "C", "delta_auc"].iloc[0])
    gap_reduction = {
        "A_to_B_pct": 100.0 * (gap_A - gap_B) / gap_A,
        "B_to_C_pct": 100.0 * (gap_B - gap_C) / gap_B,
        "A_to_C_pct": 100.0 * (gap_A - gap_C) / gap_A,
        "gap_A": gap_A, "gap_B": gap_B, "gap_C": gap_C,
    }

    # ---- Task 9: data integrity summary ----
    ref_ids = dfs[1000]["id"].tolist()
    integrity = {
        "fingerprint": comparison_set_fingerprint(ref_ids),
        "fingerprint_matches_expected": comparison_set_fingerprint(ref_ids) == EXPECTED_FINGERPRINT,
        "n": len(ref_ids),
        "positive_rate": float(dfs[1000]["y_true"].mean()),
        "ids_identical_across_stages": all(dfs[n]["id"].tolist() == ref_ids for n in STAGES.values()),
        "y_true_identical_across_stages": all(
            dfs[n]["y_true"].tolist() == dfs[1000]["y_true"].tolist() for n in STAGES.values()
        ),
        "threshold": DECISION_THRESHOLD,
    }

    # ---- persist outputs ----
    delong_df.to_csv(RESULTS_ROOT / "delong_results.csv", index=False)
    bootstrap_df.to_csv(RESULTS_ROOT / "bootstrap_results.csv", index=False)
    mcnemar_df.to_csv(RESULTS_ROOT / "mcnemar_results.csv", index=False)
    auc_df.to_csv(RESULTS_ROOT / "auc_comparisons.csv", index=False)
    metric_ci_df.to_csv(RESULTS_ROOT / "metric_confidence_intervals.csv", index=False)
    gap_df.to_csv(RESULTS_ROOT / "gap_analysis.csv", index=False)

    plot_forest(auc_df, RESULTS_ROOT / "forest_plot_delta_auc.png")
    plot_auc_across_stages(auc_df, metric_ci_df, RESULTS_ROOT / "auc_across_stages.png")

    summary = {
        "data_integrity": integrity,
        "n_primary_comparisons": len(auc_df),
        "holm_correction_family": "12 DeLong p-values (4 classical models x 3 stages), corrected once as a single family",
        "significant_after_holm_delong": int(auc_df["delong_significant_holm"].sum()),
        "significant_after_holm_bootstrap": int(auc_df["bootstrap_significant_holm"].sum()),
        "gap_analysis": gap_reduction,
        "alpha": ALPHA,
        "bootstrap_config": {"n_resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
    }
    with open(RESULTS_ROOT / "statistical_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    return {
        "auc_df": auc_df, "delong_df": delong_df, "bootstrap_df": bootstrap_df,
        "mcnemar_df": mcnemar_df, "metric_ci_df": metric_ci_df, "gap_df": gap_df,
        "summary": summary,
    }


def print_summary(result: dict) -> None:
    auc_df = result["auc_df"]
    print("=" * 100)
    print("PHASE 6A -- STATISTICAL ROBUSTNESS ANALYSIS")
    print("=" * 100)
    print(f"Data integrity: {result['summary']['data_integrity']}")
    print()
    cols = ["stage", "n_train", "classical_model", "auc_classical", "auc_qsvm",
            "delong_delta", "delong_ci_low", "delong_ci_high", "delong_p", "delong_p_holm",
            "delong_significant_holm", "mcnemar_p"]
    print(auc_df[cols].to_string(index=False))
    print()
    print("Gap analysis:")
    print(result["gap_df"][["stage", "n_train", "best_classical_model", "delta_auc"]].to_string(index=False))
    print()
    print("Observed gap reduction:", result["summary"]["gap_analysis"])
    print(f"Significant after Holm (DeLong, alpha={ALPHA}): "
          f"{result['summary']['significant_after_holm_delong']} / {len(auc_df)}")
    print(f"Significant after Holm (Bootstrap, alpha={ALPHA}): "
          f"{result['summary']['significant_after_holm_bootstrap']} / {len(auc_df)}")
    print("=" * 100)


if __name__ == "__main__":
    result = run_statistical_robustness_analysis()
    print_summary(result)
