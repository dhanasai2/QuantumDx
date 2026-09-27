"""Phase 15: quantum-classical ENSEMBLE COMPLEMENTARITY (see
docs/PHASE_15_QUANTUM_CLASSICAL_STACKING.md).

This is the final, strictly bounded quantum-learning experiment for this
project. Every prior quantum phase (7-13) asked "does a quantum
feature/refinement IMPROVE the classical model when merged into its
input?" and found no measurable contribution. This phase asks a
genuinely different, narrower question:

    Does a quantum learner's OWN prediction carry information that is
    DIFFERENT from XGBoost's prediction, such that a classical
    meta-learner can exploit BOTH via prediction-level stacking?

ARCHITECTURE (prediction-level stacking, not feature concatenation):

    Biomedical input -> classical preprocessing -> PCA-4
            |                                          |
            v (full classical features)                v (PCA-4)
        XGBoost                              Quantum learner (Model B)
        p_xgb                                Classical control (Model C)
            |                                          |
            +-------------------+  +-------------------+
                                 v  v
                     Logistic Regression meta-learner
                                 |
                                 v
                          Final risk score

WHAT IS REUSED, UNMODIFIED:
    - src.large_dataset.phase11_full_scale.build_full_scale_split (the
      n=66,641 training pool + fixed 200-row test set).
    - src.large_dataset.pipeline.process_stage (Phase 2 preprocessing/PCA),
      called ONCE PER OUTER FOLD with that fold's own train/validation
      partitions -- never with the fixed test set until Stage 6.
    - src.quantum.quantum_refinement_circuit.{RefinementConfig,
      QuantumRefinementModule} -- Phase 13's already-implemented, already
      exact-statevector-verified "circuit + trainable linear readout ->
      ONE scalar score" architecture, reused at its DEFAULT configuration
      (6 qubits) completely UNMODIFIED. Its input is PCA-4 zero-padded to
      6 dimensions (see QUANTUM_INPUT_DIM below for why) -- NOT Phase 13's
      own 6-dim input (4 PCA + XGBoost's OOF proba + margin): this phase
      deliberately never feeds the quantum learner XGBoost's own
      prediction, because doing so would contaminate the complementarity
      test (p_quantum would then be partly DERIVED from p_xgb rather than
      independent of it).
    - src.quantum.matched_refinement_controls.{MatchedRFFConfig,
      MatchedRFFRefinement} -- Phase 13's matched Random-Fourier-Feature
      control, which already exposes the IDENTICAL `.score(X, params) ->
      (n,)` scalar interface as the quantum module, reused at its default
      configuration (input_dim=6, n_features=6 -- dimension-matched to
      the quantum block) with the same zero-padded PCA-4 input.
    - src.large_dataset.corrected_comparison / statistical_robustness --
      the same paired-statistics functions (DeLong, bootstrap, McNemar,
      Holm-Bonferroni) every prior phase used, for the Stage 6 test-set
      comparison (only run if the CV gate passes).

WHAT IS NEW: the OUTER TRAINING OBJECTIVE for theta/phi. Phase 12/13
selected representation/refinement weights to maximize a DOWNSTREAM
classifier's CV performance (concatenated features) or to minimize
residual MSE. Phase 15 instead trains theta/phi (circuit params +
readout weights, ALL of `params`, end-to-end) via COBYLA to directly
maximize the CV ROC-AUC of the module's OWN raw scalar score against y --
i.e. the quantum/control module IS the classifier here, not a feature
generator for one. sigmoid(raw_score) is reported as p_quantum / p_control
(a monotonic squash into a valid probability; COBYLA's ROC-AUC objective
is threshold/scale invariant, so this does not change what is optimized).

LEAKAGE CONTROL (the critical new requirement this phase adds beyond
Phase 12/13's train-only discipline): genuine OUT-OF-FOLD predictions.
For each of 5 outer StratifiedKFold folds, EVERYTHING -- the
SharedFeaturePipeline, the PCA/quantum-range pipeline, XGBoost, AND the
quantum/control theta/phi -- is fit using ONLY that fold's own training
partition, then applied ONCE to that fold's held-out validation
partition. No stage anywhere sees a validation row's label (or feature
statistics derived from it) before predicting it.

COMPUTATIONAL BUDGET (Section 12 of the governing spec): running COBYLA
directly against a fold's full ~53,000-row training partition (as the
objective's CV split) would require an EstimatorQNN forward pass over
tens of thousands of rows per COBYLA function evaluation, repeated
~OPTIMIZER_MAXITER times per fold -- multi-hour per fold, unbounded. This
project's established cost-control pattern (Phase 7/10/12/13: "select
weights via COBYLA + train-only CV on a bounded, cheap subset; freeze
and apply once to the full pool") is reused here, adapted to preserve
STRICT fold isolation: instead of the single global n=2,000 screening
subset (Phase 8A's, which is a fixed subset of the full pool and would
overlap unpredictably with whichever rows land in each outer fold's
VALIDATION partition -- a subtle leakage path this phase cannot accept),
each fold draws its OWN bounded, stratified subsample from ONLY that
fold's own training partition for the inner COBYLA+CV weight search.
This keeps the compute bounded (comparable to a single Phase 12/13
dev-experiment run) while guaranteeing zero leakage from any fold's
validation rows into that fold's own theta/phi selection.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from qiskit_machine_learning.optimizers import COBYLA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
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
from src.large_dataset.phase11_full_scale import build_full_scale_split
from src.large_dataset.pipeline import process_stage
from src.large_dataset.schema import TARGET_COLUMN, get_cardio_feature_groups
from src.large_dataset.statistical_robustness import delong_test, holm_bonferroni
from src.preprocessing.config import PreprocessingConfig
from src.quantum.matched_refinement_controls import MatchedRFFConfig, MatchedRFFRefinement
from src.quantum.quantum_refinement_circuit import QuantumRefinementModule, RefinementConfig

EXPECTED_FINGERPRINT = "96eac11a8394b87e"
RESULTS_ROOT = find_project_root() / "results" / "large_dataset" / "phase15_quantum_classical_stacking"
PHASE13_EXPLAINABILITY_PATH = find_project_root() / "results" / "large_dataset" / "phase13_hybrid_system" / "explainability.json"

#: Phase 11/12/13's own established full-scale XGBoost hyperparameters --
#: reused verbatim so Model A (the baseline being challenged) is exactly
#: the model this whole project has already validated most extensively.
XGB_FROZEN_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)

OOF_FOLDS = 5
INNER_CV_FOLDS = 3
OPTIMIZER_MAXITER = 25
#: Bounded, per-fold, train-only subsample size for the inner COBYLA weight
#: search -- drawn ONLY from that fold's own training partition (see
#: module docstring, "Computational budget"). Matches this project's
#: established ~2,000-row screening-subset cost-control scale.
THETA_SELECTION_SUBSAMPLE_SIZE = 1600
PRIMARY_SEED = 42
STABILITY_SEEDS = [42, 123, 2024, 7, 99]
#: Section 8 Criterion 5 / Section 11 ("reject improvements that are
#: effectively noise"): the minimum ROC-AUC delta over the baseline, and
#: the minimum partial-correlation edge over the matched control, that a
#: numerically positive result must clear before being reported as
#: Outcome B ("complementary signal") rather than Outcome C ("no
#: contribution"). Chosen well below what would be needed to pass the
#: full gate, but well above pure floating-point/CV-fold noise (roughly
#: 15% of the ~0.0065 fold-to-fold ROC-AUC std observed across this
#: project's CV runs).
MATERIALITY_ROC_AUC_DELTA = 0.001
MATERIALITY_PARTIAL_CORR_GAP = 0.01
#: Quantum/control input: PCA-4 ONLY (never XGBoost's own prediction --
#: see module docstring on why that would contaminate the complementarity
#: test). QuantumRefinementModule/RefinementConfig's DEFAULT config (6
#: qubits) is reused completely UNMODIFIED (not even the qubit count is
#: overridden) -- discovered during development that
#: build_refinement_circuit's entangling-pair wiring is hardcoded to the
#: module-level N_QUBITS=6 constant rather than actually generalizing to
#: an arbitrary passed-in n_qubits (a latent bug in that Phase 13 file:
#: RefinementConfig(n_qubits=4, ...) raises CircuitError, since the
#: entangling loop indexes qubits 0-5 into a 4-qubit circuit). Rather than
#: touch a Phase 13 file (out of scope -- "do not modify Phases 7-14"),
#: PCA-4 is zero-padded to 6 dimensions so the EXISTING, already-verified
#: 6-qubit circuit can be reused byte-for-byte at its own default
#: configuration; the 2 padding columns are constant zeros, carrying no
#: information, so they cannot manufacture a spurious quantum advantage.
QUANTUM_INPUT_DIM = 6


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _pad_to_quantum_input_dim(X: np.ndarray, target_dim: int = QUANTUM_INPUT_DIM) -> np.ndarray:
    X = np.asarray(X)
    if X.shape[1] >= target_dim:
        return X[:, :target_dim]
    return np.hstack([X, np.zeros((X.shape[0], target_dim - X.shape[1]))])


def _metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    return compute_classification_metrics(y_true, y_pred, y_proba)


def _build_xgb() -> XGBClassifier:
    return XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1, tree_method="hist", **XGB_FROZEN_PARAMS)


def _stratified_subsample_indices(y: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    """Indices of a class-stratified subsample of size <= n, drawn from
    the array `y` itself (the caller is responsible for restricting `y`
    to whatever pool fold isolation requires -- this function has no
    concept of folds or a test set)."""
    n = min(n, len(y))
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    pos_frac = len(pos_idx) / len(y)
    n_pos = min(len(pos_idx), max(1, int(round(n * pos_frac))))
    n_neg = min(len(neg_idx), n - n_pos)
    n_pos = min(n_pos, n - n_neg)
    sel_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    sel_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    idx = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(idx)
    return idx


# --------------------------------------------------------------------------
# Quantum learner / matched classical control: end-to-end scalar-classifier
# weight selection (COBYLA + train-only inner CV, maximizing CV ROC-AUC of
# the module's OWN raw score -- see module docstring, "What is new").
# --------------------------------------------------------------------------


def select_scalar_classifier_weights(
    module, n_params: int, X: np.ndarray, y: np.ndarray, *,
    seed: int, maxiter: int = OPTIMIZER_MAXITER, cv_folds: int = INNER_CV_FOLDS,
    subsample_size: int = THETA_SELECTION_SUBSAMPLE_SIZE,
) -> dict:
    """Representation-agnostic (works identically for QuantumRefinementModule
    and MatchedRFFRefinement -- both expose `.score(X, params) -> (n,)`).
    `X`, `y` MUST already be restricted to a single fold's TRAINING
    partition by the caller -- no test/validation parameter exists here.
    """
    rng = np.random.RandomState(seed)
    sub_idx = _stratified_subsample_indices(y, subsample_size, rng)
    Xs, ys = X[sub_idx], y[sub_idx]
    fold_idx = np.array_split(rng.permutation(len(Xs)), cv_folds)
    history: list[float] = []
    n_evals = 0

    def objective(params: np.ndarray) -> float:
        nonlocal n_evals
        n_evals += 1
        fold_aucs = []
        for k in range(cv_folds):
            val_idx = fold_idx[k]
            if len(np.unique(ys[val_idx])) < 2:
                continue
            raw = module.score(Xs[val_idx], params)
            fold_aucs.append(roc_auc_score(ys[val_idx], raw))
        mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5
        history.append(mean_auc)
        return -mean_auc

    p0 = rng.uniform(-0.1, 0.1, size=n_params)
    t0 = time.perf_counter()
    result = COBYLA(maxiter=maxiter).minimize(objective, p0)
    elapsed = time.perf_counter() - t0
    return {
        "weights": result.x, "cv_auc_history": history, "n_function_evaluations": n_evals,
        "seconds": elapsed, "best_cv_auc": max(history) if history else float("nan"),
        "seed": seed, "n_subsample": len(Xs),
    }


# --------------------------------------------------------------------------
# Stage: genuine out-of-fold prediction generation
# --------------------------------------------------------------------------


@dataclass
class OOFResult:
    oof_df: pd.DataFrame           # id, y, fold, p_xgb, p_quantum, p_control
    fold_records: list[dict]       # per-fold timings + inner-CV diagnostics
    quantum_weights_by_fold: dict  # fold -> weights (for reporting/reuse)
    control_weights_by_fold: dict


def generate_oof_predictions(training_pool: pd.DataFrame, feature_groups, pcfg: PreprocessingConfig, *,
                              seed: int = PRIMARY_SEED, n_folds: int = OOF_FOLDS) -> OOFResult:
    y_full = training_pool[TARGET_COLUMN].to_numpy()
    ids_full = training_pool["id"].to_numpy()
    n = len(training_pool)
    oof_xgb = np.full(n, np.nan)
    oof_quantum = np.full(n, np.nan)
    oof_control = np.full(n, np.nan)
    fold_assignment = np.full(n, -1, dtype=int)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_records = []
    quantum_weights_by_fold: dict[int, np.ndarray] = {}
    control_weights_by_fold: dict[int, np.ndarray] = {}

    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(training_pool, y_full)):
        fold_assignment[val_idx] = fold_i
        fold_train_df = training_pool.iloc[tr_idx].reset_index(drop=True)
        fold_val_df = training_pool.iloc[val_idx].reset_index(drop=True)

        t_prep = time.perf_counter()
        processed = process_stage(fold_train_df, fold_val_df, fold_val_df, TARGET_COLUMN, feature_groups, pcfg)
        prep_seconds = time.perf_counter() - t_prep

        Xtr_classical = processed.X_train_classical.to_numpy() if hasattr(processed.X_train_classical, "to_numpy") else processed.X_train_classical
        Xval_classical = processed.X_test_classical.to_numpy() if hasattr(processed.X_test_classical, "to_numpy") else processed.X_test_classical
        ytr = processed.y_train
        Xtr_pca = _pad_to_quantum_input_dim(processed.X_train_quantum)
        Xval_pca = _pad_to_quantum_input_dim(processed.X_test_quantum)

        # ---- XGBoost (Model A's own base learner) ----
        t_xgb = time.perf_counter()
        clf = _build_xgb()
        clf.fit(Xtr_classical, ytr)
        p_xgb_val = clf.predict_proba(Xval_classical)[:, 1]
        xgb_seconds = time.perf_counter() - t_xgb
        oof_xgb[val_idx] = p_xgb_val

        # ---- Quantum learner: fold-train-only theta selection, applied once ----
        q_module = QuantumRefinementModule(RefinementConfig(seed=seed))
        q_sel = select_scalar_classifier_weights(q_module, q_module.config.n_trainable_params(), Xtr_pca, ytr, seed=seed)
        raw_q_val = q_module.score(Xval_pca, q_sel["weights"])
        p_quantum_val = _sigmoid(raw_q_val)
        oof_quantum[val_idx] = p_quantum_val
        quantum_weights_by_fold[fold_i] = q_sel["weights"]

        # ---- Matched classical control: identical procedure, identical dims ----
        c_module = MatchedRFFRefinement(MatchedRFFConfig(seed=seed))
        c_sel = select_scalar_classifier_weights(c_module, c_module.config.n_trainable_params(), Xtr_pca, ytr, seed=seed)
        raw_c_val = c_module.score(Xval_pca, c_sel["weights"])
        p_control_val = _sigmoid(raw_c_val)
        oof_control[val_idx] = p_control_val
        control_weights_by_fold[fold_i] = c_sel["weights"]

        yval = processed.y_test_classical
        fold_records.append({
            "fold": fold_i, "n_train": len(tr_idx), "n_val": len(val_idx),
            "preprocessing_seconds": prep_seconds, "xgb_fit_seconds": xgb_seconds,
            "xgb_val_roc_auc": float(roc_auc_score(yval, p_xgb_val)),
            "quantum_inner_cv_best_auc": q_sel["best_cv_auc"], "quantum_inner_seconds": q_sel["seconds"],
            "quantum_val_roc_auc": float(roc_auc_score(yval, p_quantum_val)),
            "control_inner_cv_best_auc": c_sel["best_cv_auc"], "control_inner_seconds": c_sel["seconds"],
            "control_val_roc_auc": float(roc_auc_score(yval, p_control_val)),
        })
        print(f"[phase15] fold {fold_i}: xgb_val_auc={fold_records[-1]['xgb_val_roc_auc']:.4f} "
              f"quantum_val_auc={fold_records[-1]['quantum_val_roc_auc']:.4f} "
              f"(inner_cv={q_sel['best_cv_auc']:.4f}, {q_sel['seconds']:.1f}s) "
              f"control_val_auc={fold_records[-1]['control_val_roc_auc']:.4f} "
              f"(inner_cv={c_sel['best_cv_auc']:.4f}, {c_sel['seconds']:.1f}s)", flush=True)

    assert not np.any(fold_assignment == -1)
    assert not np.any(np.isnan(oof_xgb)) and not np.any(np.isnan(oof_quantum)) and not np.any(np.isnan(oof_control))
    oof_df = pd.DataFrame({
        "id": ids_full, "y": y_full, "fold": fold_assignment,
        "p_xgb": oof_xgb, "p_quantum": oof_quantum, "p_control": oof_control,
    })
    return OOFResult(oof_df=oof_df, fold_records=fold_records,
                      quantum_weights_by_fold=quantum_weights_by_fold, control_weights_by_fold=control_weights_by_fold)


# --------------------------------------------------------------------------
# Section 7: prediction complementarity diagnostics (OOF only, read-only)
# --------------------------------------------------------------------------


def prediction_complementarity_analysis(oof_df: pd.DataFrame) -> dict:
    p_xgb, p_quantum, p_control, y = oof_df["p_xgb"].to_numpy(), oof_df["p_quantum"].to_numpy(), oof_df["p_control"].to_numpy(), oof_df["y"].to_numpy()
    from scipy.stats import pearsonr, spearmanr

    pearson_xq = pearsonr(p_xgb, p_quantum)
    spearman_xq = spearmanr(p_xgb, p_quantum)
    pearson_xc = pearsonr(p_xgb, p_control)
    residual = y.astype(float) - p_xgb
    pearson_resid_q = pearsonr(residual, p_quantum)
    pearson_resid_c = pearsonr(residual, p_control)
    pred_xgb_label = (p_xgb >= DECISION_THRESHOLD).astype(int)
    pred_quantum_label = (p_quantum >= DECISION_THRESHOLD).astype(int)
    disagreement_rate = float(np.mean(pred_xgb_label != pred_quantum_label))

    # Incremental-information check: does adding p_quantum to a model that
    # already has p_xgb change the RANKING it produces, beyond what noise
    # would? Measured here via the partial correlation of p_quantum with y,
    # controlling for p_xgb (residualize both against p_xgb, correlate the
    # residuals) -- descriptive only, NOT the acceptance test itself
    # (Section 8's CV comparison is the acceptance test).
    from numpy.polynomial import polynomial as P

    def _residualize(target: np.ndarray, against: np.ndarray) -> np.ndarray:
        A = np.vstack([against, np.ones_like(against)]).T
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        return target - A @ coef

    y_resid = _residualize(y.astype(float), p_xgb)
    q_resid = _residualize(p_quantum, p_xgb)
    c_resid = _residualize(p_control, p_xgb)
    partial_corr_quantum = float(pearsonr(y_resid, q_resid)[0]) if np.std(q_resid) > 1e-12 else 0.0
    partial_corr_control = float(pearsonr(y_resid, c_resid)[0]) if np.std(c_resid) > 1e-12 else 0.0

    return {
        "pearson_corr_p_xgb_vs_p_quantum": float(pearson_xq[0]), "pearson_p_value": float(pearson_xq[1]),
        "spearman_corr_p_xgb_vs_p_quantum": float(spearman_xq[0]), "spearman_p_value": float(spearman_xq[1]),
        "pearson_corr_p_xgb_vs_p_control": float(pearson_xc[0]),
        "prediction_disagreement_rate_at_0.5": disagreement_rate,
        "pearson_corr_residual_vs_p_quantum": float(pearson_resid_q[0]),
        "pearson_corr_residual_vs_p_control": float(pearson_resid_c[0]),
        "partial_corr_p_quantum_vs_y_given_p_xgb": partial_corr_quantum,
        "partial_corr_p_control_vs_y_given_p_xgb": partial_corr_control,
        "interpretation": (
            "High correlation between p_xgb and p_quantum/p_control would mean the auxiliary predictor "
            "is mostly re-deriving XGBoost's own decision, leaving little room for a meta-learner to gain "
            "anything. A partial correlation with y (after removing what p_xgb already explains) that is "
            "close to the control's own partial correlation would mean quantum is not doing anything a "
            "generic nonlinear auxiliary predictor could not also do. Correlation alone is NOT evidence of "
            "usefulness -- Section 8's CV comparison (whether the meta-learner can actually exploit it) is "
            "the acceptance test; this analysis is diagnostic context for interpreting that result."
        ),
    }


# --------------------------------------------------------------------------
# Section 6/8: nested-CV evaluation of Models A / B / C + acceptance gate
# --------------------------------------------------------------------------


def evaluate_stacks_via_nested_cv(oof_df: pd.DataFrame) -> dict:
    """Uses the SAME fold assignment the OOF predictions were generated
    with. For each fold, the meta-learner is fit ONLY on the OTHER folds'
    OOF rows and scored on this fold's OOF rows -- so the meta-learner
    itself is never evaluated in-sample, even though its inputs (p_xgb,
    p_quantum, p_control) are already fold-safe."""
    per_fold_rows = []
    pooled = {"xgboost_alone": [], "quantum_stack": [], "control_stack": []}
    pooled_y = []
    for k in sorted(oof_df["fold"].unique()):
        train_mask = oof_df["fold"] != k
        val_mask = oof_df["fold"] == k
        y_val = oof_df.loc[val_mask, "y"].to_numpy()
        y_train = oof_df.loc[train_mask, "y"].to_numpy()
        pooled_y.append(y_val)

        p_a = oof_df.loc[val_mask, "p_xgb"].to_numpy()

        lr_b = LogisticRegression(max_iter=1000, random_state=42)
        lr_b.fit(oof_df.loc[train_mask, ["p_xgb", "p_quantum"]].to_numpy(), y_train)
        p_b = lr_b.predict_proba(oof_df.loc[val_mask, ["p_xgb", "p_quantum"]].to_numpy())[:, 1]

        lr_c = LogisticRegression(max_iter=1000, random_state=42)
        lr_c.fit(oof_df.loc[train_mask, ["p_xgb", "p_control"]].to_numpy(), y_train)
        p_c = lr_c.predict_proba(oof_df.loc[val_mask, ["p_xgb", "p_control"]].to_numpy())[:, 1]

        for name, p in [("xgboost_alone", p_a), ("quantum_stack", p_b), ("control_stack", p_c)]:
            m = _metrics(y_val, p)
            per_fold_rows.append({"fold": int(k), "model": name, **m})
            pooled[name].append(p)

    fold_df = pd.DataFrame(per_fold_rows)
    summary = {}
    for name in pooled:
        sub = fold_df[fold_df["model"] == name]
        summary[name] = {f"{metric}_mean": float(sub[metric].mean()) for metric in ("roc_auc", "pr_auc", "sensitivity", "specificity", "accuracy", "f1")}
        summary[name].update({f"{metric}_std": float(sub[metric].std(ddof=0)) for metric in ("roc_auc", "pr_auc", "sensitivity", "specificity", "accuracy", "f1")})
        summary[name]["per_fold_roc_auc"] = sub["roc_auc"].tolist()

    y_pooled = np.concatenate(pooled_y)
    pooled_proba = {name: np.concatenate(vals) for name, vals in pooled.items()}
    return {"fold_df": fold_df, "summary": summary, "pooled_y": y_pooled, "pooled_proba": pooled_proba}


def check_acceptance_gate(cv_eval: dict) -> dict:
    s = cv_eval["summary"]
    a, b, c = s["xgboost_alone"], s["quantum_stack"], s["control_stack"]
    a_folds = np.array(a["per_fold_roc_auc"])
    b_folds = np.array(b["per_fold_roc_auc"])

    criterion_1_beats_baseline = b["roc_auc_mean"] > a["roc_auc_mean"]
    n_folds_won = int(np.sum(b_folds > a_folds))
    criterion_2_fold_consistency = n_folds_won >= (len(a_folds) - 1)  # at most one lucky/unlucky fold
    criterion_4_beats_control = b["roc_auc_mean"] > c["roc_auc_mean"]
    criterion_5_practical_significance = b["roc_auc_mean"] > (a["roc_auc_mean"] + a["roc_auc_std"])

    gate_passed = bool(criterion_1_beats_baseline and criterion_2_fold_consistency
                        and criterion_4_beats_control and criterion_5_practical_significance)

    return {
        "criterion_1_beats_baseline_mean_cv": bool(criterion_1_beats_baseline),
        "criterion_2_fold_consistency": {"passed": bool(criterion_2_fold_consistency), "n_folds_quantum_stack_won": n_folds_won, "n_folds_total": len(a_folds)},
        "criterion_4_beats_matched_control": bool(criterion_4_beats_control),
        "criterion_5_practical_significance": bool(criterion_5_practical_significance),
        "delta_vs_baseline": b["roc_auc_mean"] - a["roc_auc_mean"],
        "delta_vs_control": b["roc_auc_mean"] - c["roc_auc_mean"],
        "gate_passed": gate_passed,
    }


# --------------------------------------------------------------------------
# Stage 6: ONE-TIME test-set evaluation (only if the CV gate passed)
# --------------------------------------------------------------------------


def run_stage6_test_evaluation(split, feature_groups, pcfg: PreprocessingConfig, *, seed: int = PRIMARY_SEED) -> dict:
    """Fits the final models on the FULL 66,641-row training pool and
    scores the fixed 200-row test set EXACTLY ONCE. Only ever called if
    check_acceptance_gate(...)['gate_passed'] is True."""
    cmp_ids = split.test_set_quantum["id"].tolist()
    fingerprint = comparison_set_fingerprint(cmp_ids)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"Comparison set fingerprint {fingerprint} != expected {EXPECTED_FINGERPRINT}")

    processed = process_stage(split.training_pool, split.test_set_classical, split.test_set_quantum, TARGET_COLUMN, feature_groups, pcfg)
    Xtr_classical = processed.X_train_classical.to_numpy() if hasattr(processed.X_train_classical, "to_numpy") else processed.X_train_classical
    Xte_classical = processed.X_test_quantum_classical_space
    Xte_classical = Xte_classical.to_numpy() if hasattr(Xte_classical, "to_numpy") else Xte_classical
    ytr, yte = processed.y_train, processed.y_test_quantum
    Xtr_pca = _pad_to_quantum_input_dim(processed.X_train_quantum)
    Xte_pca = _pad_to_quantum_input_dim(processed.X_test_quantum)

    xgb_final = _build_xgb()
    xgb_final.fit(Xtr_classical, ytr)
    p_xgb_te = xgb_final.predict_proba(Xte_classical)[:, 1]

    q_module = QuantumRefinementModule(RefinementConfig(seed=seed))
    q_sel = select_scalar_classifier_weights(q_module, q_module.config.n_trainable_params(), Xtr_pca, ytr, seed=seed)
    p_quantum_te = _sigmoid(q_module.score(Xte_pca, q_sel["weights"]))

    c_module = MatchedRFFRefinement(MatchedRFFConfig(seed=seed))
    c_sel = select_scalar_classifier_weights(c_module, c_module.config.n_trainable_params(), Xtr_pca, ytr, seed=seed)
    p_control_te = _sigmoid(c_module.score(Xte_pca, c_sel["weights"]))

    # Meta-learners trained on the FULL OOF table generated earlier are
    # re-derived by the caller (run_all_stages) and passed in via closures
    # in practice; here we accept them as arguments-free by refitting on
    # the freshly generated OOF table is the caller's responsibility -- see
    # run_all_stages for the exact sequencing.
    return {
        "cmp_ids": cmp_ids, "fingerprint": fingerprint, "y_test": yte,
        "p_xgb_test": p_xgb_te, "p_quantum_test": p_quantum_te, "p_control_test": p_control_te,
        "quantum_weights": q_sel["weights"], "control_weights": c_sel["weights"],
        "xgb_final": xgb_final, "quantum_module": q_module, "control_module": c_module,
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run_all_stages(*, make_plots: bool = True, run_stability_check_if_gate_open: bool = True) -> dict:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    runtime: dict = {}

    print("[phase15] rebuilding the full fixed split (Phase 11's own function, unmodified) ...", flush=True)
    t0 = time.perf_counter()
    split = build_full_scale_split()
    runtime["split_seconds"] = time.perf_counter() - t0
    cmp_ids = split.test_set_quantum["id"].tolist()
    fingerprint = comparison_set_fingerprint(cmp_ids)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"Comparison set fingerprint {fingerprint} != expected {EXPECTED_FINGERPRINT}")
    print(f"[phase15] test set fingerprint verified ({fingerprint}); test set NOT touched beyond id fingerprinting.", flush=True)

    fg = get_cardio_feature_groups()
    pcfg = PreprocessingConfig()

    print(f"[phase15] ===== Stage: {OOF_FOLDS}-fold genuine OOF generation (XGBoost, quantum, control), seed={PRIMARY_SEED} =====", flush=True)
    t1 = time.perf_counter()
    oof_result = generate_oof_predictions(split.training_pool, fg, pcfg, seed=PRIMARY_SEED)
    runtime["oof_generation_seconds"] = time.perf_counter() - t1
    print(f"[phase15] OOF generation done in {runtime['oof_generation_seconds']:.1f}s", flush=True)

    print("[phase15] ===== Prediction complementarity diagnostics (OOF only) =====", flush=True)
    complementarity = prediction_complementarity_analysis(oof_result.oof_df)
    print(f"[phase15] pearson(p_xgb, p_quantum)={complementarity['pearson_corr_p_xgb_vs_p_quantum']:.4f}, "
          f"disagreement_rate={complementarity['prediction_disagreement_rate_at_0.5']:.4f}", flush=True)

    print("[phase15] ===== Nested-CV evaluation: Model A (XGBoost alone) vs B (+quantum) vs C (+control) =====", flush=True)
    t2 = time.perf_counter()
    cv_eval = evaluate_stacks_via_nested_cv(oof_result.oof_df)
    runtime["nested_cv_evaluation_seconds"] = time.perf_counter() - t2
    for name, s in cv_eval["summary"].items():
        print(f"[phase15] {name}: CV ROC-AUC mean={s['roc_auc_mean']:.4f} std={s['roc_auc_std']:.4f} per_fold={[round(x, 4) for x in s['per_fold_roc_auc']]}", flush=True)

    gate = check_acceptance_gate(cv_eval)
    print(f"[phase15] ACCEPTANCE GATE: {gate}", flush=True)

    # ---- diagnostic full-OOF meta-learner refit (reporting only, NOT used
    # for the acceptance decision -- see evaluate_stacks_via_nested_cv's
    # docstring for why the DECISION uses the nested-CV version instead) ----
    lr_full_b = LogisticRegression(max_iter=1000, random_state=42)
    lr_full_b.fit(oof_result.oof_df[["p_xgb", "p_quantum"]].to_numpy(), oof_result.oof_df["y"].to_numpy())
    lr_full_c = LogisticRegression(max_iter=1000, random_state=42)
    lr_full_c.fit(oof_result.oof_df[["p_xgb", "p_control"]].to_numpy(), oof_result.oof_df["y"].to_numpy())

    stability: dict | None = None
    if gate["gate_passed"] and run_stability_check_if_gate_open:
        print("[phase15] ===== Gate passed at primary seed -- running multi-seed stability check (Criterion 3) =====", flush=True)
        t3 = time.perf_counter()
        per_seed = {}
        for seed in STABILITY_SEEDS:
            if seed == PRIMARY_SEED:
                per_seed[seed] = {"cv_eval_summary": cv_eval["summary"], "gate": gate}
                continue
            oof_s = generate_oof_predictions(split.training_pool, fg, pcfg, seed=seed)
            cv_eval_s = evaluate_stacks_via_nested_cv(oof_s.oof_df)
            gate_s = check_acceptance_gate(cv_eval_s)
            per_seed[seed] = {"cv_eval_summary": cv_eval_s["summary"], "gate": gate_s}
            print(f"[phase15] stability seed={seed}: quantum_stack CV ROC-AUC mean={cv_eval_s['summary']['quantum_stack']['roc_auc_mean']:.4f}, gate_passed={gate_s['gate_passed']}", flush=True)
        runtime["stability_check_seconds"] = time.perf_counter() - t3
        n_seeds_gate_passed = sum(1 for v in per_seed.values() if v["gate"]["gate_passed"])
        stability = {"seeds": STABILITY_SEEDS, "per_seed": {str(k): v for k, v in per_seed.items()},
                     "n_seeds_gate_passed": n_seeds_gate_passed, "consistent_across_seeds": n_seeds_gate_passed == len(STABILITY_SEEDS)}
    elif not gate["gate_passed"]:
        print("[phase15] Gate NOT passed at primary seed -- skipping the expensive multi-seed stability sweep "
              "(no positive finding whose stability needs confirming; see docs for the cost-control rationale).", flush=True)

    # ---- Stage 6: ONE-TIME test-set evaluation, ONLY if the gate (and, if run, stability) passed ----
    test_evaluation = None
    statistics = None
    if gate["gate_passed"] and (stability is None or stability["consistent_across_seeds"]):
        print("[phase15] ===== Stage 6: gate passed -- ONE-TIME fixed 200-row test-set evaluation =====", flush=True)
        t4 = time.perf_counter()
        stage6 = run_stage6_test_evaluation(split, fg, pcfg, seed=PRIMARY_SEED)
        runtime["stage6_seconds"] = time.perf_counter() - t4

        yte = stage6["y_test"]
        p_a_te = stage6["p_xgb_test"]
        p_b_te = lr_full_b.predict_proba(np.column_stack([stage6["p_xgb_test"], stage6["p_quantum_test"]]))[:, 1]
        p_c_te = lr_full_c.predict_proba(np.column_stack([stage6["p_xgb_test"], stage6["p_control_test"]]))[:, 1]

        all_models = {
            "xgboost_alone": {"display_name": "A: XGBoost alone", "y_proba": p_a_te},
            "quantum_stack": {"display_name": "B: XGBoost + Quantum (LR stack)", "y_proba": p_b_te},
            "control_stack": {"display_name": "C: XGBoost + Matched Control (LR stack)", "y_proba": p_c_te},
        }
        for m in all_models.values():
            m["y_true"] = yte
            m["y_pred"] = (m["y_proba"] >= DECISION_THRESHOLD).astype(int)
            m["metrics"] = _metrics(yte, m["y_proba"])

        def _pair(key_a: str, key_b: str) -> dict:
            pa, pb = all_models[key_a]["y_proba"], all_models[key_b]["y_proba"]
            preda, predb = all_models[key_a]["y_pred"], all_models[key_b]["y_pred"]
            dl = delong_test(yte, pa, pb)
            bs = paired_bootstrap_delta(yte, pa, pb, "roc_auc", n_resamples=2000, seed=42)
            mc = mcnemar_exact(yte, preda, predb)
            return {"a": key_a, "b": key_b, "delong": dl.to_dict(), "bootstrap_roc_auc": bs, "mcnemar": mc}

        comparisons = {
            "quantum_stack_vs_xgboost_alone": _pair("quantum_stack", "xgboost_alone"),
            "quantum_stack_vs_control_stack": _pair("quantum_stack", "control_stack"),
        }
        raw_p = [c["delong"]["p_value"] for c in comparisons.values()]
        holm = holm_bonferroni(raw_p)
        for (name, c), h in zip(comparisons.items(), holm):
            c["delong_p_holm"] = h["adjusted_p"]
            c["delong_significant_holm"] = h["significant"]
        statistics = comparisons

        pred_df = pd.DataFrame({"id": stage6["cmp_ids"], "y_true": yte})
        for key, m in all_models.items():
            pred_df[f"{key}_proba"] = m["y_proba"]
            pred_df[f"{key}_pred"] = m["y_pred"]
        pred_df.to_csv(RESULTS_ROOT / "test_predictions.csv", index=False)

        test_evaluation = {"metrics": {k: m["metrics"] for k, m in all_models.items()},
                            "comparison_set": {"n": len(stage6["cmp_ids"]), "fingerprint": stage6["fingerprint"]}}
    else:
        print("[phase15] Test set NOT evaluated (CV gate did not pass, or stability check failed) -- "
              "per protocol, the fixed 200-row test set remains completely untouched by this phase.", flush=True)

    # ---- final decision (Section 13) ----
    b_summary = cv_eval["summary"]["quantum_stack"]
    a_summary = cv_eval["summary"]["xgboost_alone"]
    if test_evaluation is not None:
        te_delta = test_evaluation["metrics"]["quantum_stack"]["roc_auc"] - test_evaluation["metrics"]["xgboost_alone"]["roc_auc"]
        te_beats_control = test_evaluation["metrics"]["quantum_stack"]["roc_auc"] > test_evaluation["metrics"]["control_stack"]["roc_auc"]
        confirmed = te_delta > 0 and te_beats_control and statistics["quantum_stack_vs_xgboost_alone"]["delong_significant_holm"]
        decision = "A_QUANTUM_COMPLEMENTARITY_DEMONSTRATED" if confirmed else "C_NO_QUANTUM_CONTRIBUTION"
        decision_text = ("CV gate passed and the fixed test set confirms a meaningful, statistically supported improvement "
                          "from the quantum stack over both XGBoost alone and the matched classical control."
                          if confirmed else
                          "CV gate passed at the primary seed, but the one-time fixed test-set evaluation did NOT confirm "
                          "a meaningful, statistically supported improvement -- the CV signal did not generalize.")
    elif gate["gate_passed"] and stability is not None and not stability["consistent_across_seeds"]:
        decision = "C_NO_QUANTUM_CONTRIBUTION"
        decision_text = "CV gate passed at the primary seed but was NOT stable across the multi-seed check (Criterion 3 failed) -- treated as noise, not a genuine finding. Test set was not touched."
    elif (gate["delta_vs_baseline"] > MATERIALITY_ROC_AUC_DELTA
          or (complementarity["partial_corr_p_quantum_vs_y_given_p_xgb"] - complementarity["partial_corr_p_control_vs_y_given_p_xgb"]) > MATERIALITY_PARTIAL_CORR_GAP):
        # Section 8, Criterion 5 / Section 11: "reject improvements that are
        # effectively noise" -- criterion_1 alone (ANY positive delta, however
        # microscopic) is NOT sufficient for this branch. A delta an order of
        # magnitude (or more) below the observed CV fold-to-fold std, or a
        # partial-correlation edge over the control that is itself smaller
        # than the noise in that estimate, is indistinguishable from zero and
        # must be reported as Outcome C, not oversold as "complementary signal."
        decision = "B_COMPLEMENTARY_SIGNAL_BUT_NO_MEANINGFUL_GAIN"
        decision_text = ("The quantum stack showed a mean-CV-ROC-AUC edge over XGBoost alone, or a partial correlation with "
                          "the label (after removing what XGBoost already explains) exceeding the matched classical control's, "
                          "that is large enough to plausibly be more than measurement noise -- but it did not clear the full "
                          "acceptance gate (fold consistency, practical significance, and/or beating the matched control). "
                          "Retain the hybrid architecture as an experimental/engineering capability; XGBoost remains primary. "
                          "Test set was not touched.")
    else:
        decision = "C_NO_QUANTUM_CONTRIBUTION"
        decision_text = ("The quantum stack did not beat XGBoost alone in cross-validation, and/or did not beat the matched "
                          "classical control. No quantum predictive contribution was demonstrated. Test set was not touched.")

    print(f"[phase15] FINAL DECISION: {decision} -- {decision_text}", flush=True)

    # ---- explainability ----
    explainability = _build_explainability(cv_eval, lr_full_b, lr_full_c, oof_result, complementarity, test_evaluation)

    # ---- persist ----
    oof_result.oof_df.to_csv(RESULTS_ROOT / "oof_predictions.csv", index=False)
    cv_eval["fold_df"].to_csv(RESULTS_ROOT / "fold_metrics.csv", index=False)

    seed_rows = [{"seed": PRIMARY_SEED, "role": "primary", **{f"quantum_stack_{k}": v for k, v in cv_eval["summary"]["quantum_stack"].items() if k != "per_fold_roc_auc"}}]
    if stability is not None:
        for seed, v in stability["per_seed"].items():
            if int(seed) == PRIMARY_SEED:
                continue
            seed_rows.append({"seed": int(seed), "role": "stability_check", **{f"quantum_stack_{k}": val for k, val in v["cv_eval_summary"]["quantum_stack"].items() if k != "per_fold_roc_auc"}})
    pd.DataFrame(seed_rows).to_csv(RESULTS_ROOT / "seed_metrics.csv", index=False)

    comparison_rows = [{"model": name, **{k: v for k, v in s.items() if k != "per_fold_roc_auc"}} for name, s in cv_eval["summary"].items()]
    pd.DataFrame(comparison_rows).to_csv(RESULTS_ROOT / "comparison_table.csv", index=False)

    with open(RESULTS_ROOT / "prediction_complementarity.json", "w", encoding="utf-8") as fh:
        json.dump(complementarity, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "model_config.json", "w", encoding="utf-8") as fh:
        json.dump({
            "quantum_config": RefinementConfig(seed=PRIMARY_SEED).to_dict(),
            "control_config": MatchedRFFConfig(seed=PRIMARY_SEED).to_dict(),
            "xgb_frozen_params": XGB_FROZEN_PARAMS, "oof_folds": OOF_FOLDS, "inner_cv_folds": INNER_CV_FOLDS,
            "optimizer_maxiter": OPTIMIZER_MAXITER, "theta_selection_subsample_size": THETA_SELECTION_SUBSAMPLE_SIZE,
            "primary_seed": PRIMARY_SEED, "stability_seeds": STABILITY_SEEDS,
            "meta_learner": "LogisticRegression(max_iter=1000, random_state=42)",
        }, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "explainability.json", "w", encoding="utf-8") as fh:
        json.dump(explainability, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "runtime.json", "w", encoding="utf-8") as fh:
        json.dump(runtime, fh, indent=2, default=float)
    if statistics is not None:
        with open(RESULTS_ROOT / "statistics.json", "w", encoding="utf-8") as fh:
            json.dump(statistics, fh, indent=2, default=float)

    metrics_rows = [{"model": name, **{k: v for k, v in s.items() if k != "per_fold_roc_auc"}} for name, s in cv_eval["summary"].items()]
    if test_evaluation is not None:
        for name, m in test_evaluation["metrics"].items():
            metrics_rows.append({"model": f"{name}_TEST", **m})
    pd.DataFrame(metrics_rows).to_csv(RESULTS_ROOT / "metrics.csv", index=False)

    summary = {
        "n_train_full": len(split.training_pool),
        "comparison_set": {"n": len(cmp_ids), "fingerprint": fingerprint, "matches_expected": fingerprint == EXPECTED_FINGERPRINT},
        "cv_evaluation": {k: v for k, v in cv_eval["summary"].items()},
        "acceptance_gate": gate,
        "stability_check": stability,
        "prediction_complementarity": complementarity,
        "test_evaluation": test_evaluation,
        "statistics": statistics,
        "runtime": runtime,
        "decision": decision,
        "decision_text": decision_text,
        "quantum_experimentation_should_stop": decision != "A_QUANTUM_COMPLEMENTARITY_DEMONSTRATED",
    }
    with open(RESULTS_ROOT / "phase15_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    if make_plots:
        _plot_curves(cv_eval, test_evaluation, RESULTS_ROOT)

    return summary


def _build_explainability(cv_eval: dict, lr_full_b: LogisticRegression, lr_full_c: LogisticRegression,
                           oof_result: OOFResult, complementarity: dict, test_evaluation: dict | None) -> dict:
    xgb_explanation: dict
    if PHASE13_EXPLAINABILITY_PATH.is_file():
        with open(PHASE13_EXPLAINABILITY_PATH, encoding="utf-8") as fh:
            phase13_expl = json.load(fh)
        xgb_explanation = {
            "source": "reused from Phase 13's explainability.json -- IDENTICAL XGBoost architecture "
                       "(same frozen hyperparameters, same feature groups, same full training pool via "
                       "build_full_scale_split) already fit and persisted there; not retrained here.",
            "xgboost_feature_importances": phase13_expl.get("xgboost_feature_importances"),
            "shap_mean_abs_contribution_test_set": phase13_expl.get("shap_mean_abs_contribution_test_set"),
        }
    else:
        xgb_explanation = {"source": "Phase 13 explainability.json not found on disk; no XGBoost feature-level explanation available."}

    return {
        "clinical_feature_contribution": xgb_explanation,
        "quantum_model_contribution": {
            "note": "Technically accurate, non-clinical description of the quantum model's role -- no clinical "
                    "meaning is claimed for any quantum parameter or expectation value.",
            "quantum_circuit_config": RefinementConfig(seed=PRIMARY_SEED).to_dict(),
            "control_circuit_config": MatchedRFFConfig(seed=PRIMARY_SEED).to_dict(),
            "meta_learner_full_oof_refit_coefficients_quantum_stack": {
                "p_xgb": float(lr_full_b.coef_[0][0]), "p_quantum": float(lr_full_b.coef_[0][1]), "intercept": float(lr_full_b.intercept_[0]),
            },
            "meta_learner_full_oof_refit_coefficients_control_stack": {
                "p_xgb": float(lr_full_c.coef_[0][0]), "p_control": float(lr_full_c.coef_[0][1]), "intercept": float(lr_full_c.intercept_[0]),
            },
            "meta_learner_weight_interpretation": (
                "A near-zero or negative coefficient on p_quantum means the meta-learner did not find the quantum "
                "prediction useful once p_xgb was already available. This coefficient is descriptive only -- it is "
                "NOT the acceptance criterion (Section 8's nested-CV comparison is)."
            ),
            "prediction_complementarity_summary": complementarity,
        },
        "test_set_explanation_available": test_evaluation is not None,
    }


def _plot_curves(cv_eval: dict, test_evaluation: dict | None, out_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    y_pooled = cv_eval["pooled_y"]
    pooled_proba = cv_eval["pooled_proba"]
    display_names = {"xgboost_alone": "A: XGBoost alone", "quantum_stack": "B: XGBoost + Quantum (stack)", "control_stack": "C: XGBoost + Control (stack)"}

    fig, ax = plt.subplots(figsize=(8, 7))
    for key, proba in pooled_proba.items():
        fpr, tpr, _ = roc_curve(y_pooled, proba)
        auc = cv_eval["summary"][key]["roc_auc_mean"]
        ax.plot(fpr, tpr, label=f"{display_names[key]} (CV mean AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC -- Phase 15 nested-CV pooled OOF-of-meta-learner predictions")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "roc_curves_cv.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for key, proba in pooled_proba.items():
        prec, rec, _ = precision_recall_curve(y_pooled, proba)
        ax.plot(rec, prec, label=display_names[key])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall -- Phase 15 nested-CV")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "pr_curves_cv.png", dpi=150); plt.close(fig)

    if test_evaluation is not None:
        n = len(test_evaluation["metrics"])
        fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
        for ax, (key, m) in zip(np.atleast_1d(axes), test_evaluation["metrics"].items()):
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
            ConfusionMatrixDisplay(cm, display_labels=["No CVD", "CVD"]).plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(display_names.get(key, key), fontsize=7)
        fig.suptitle("Confusion matrices -- Phase 15, fixed 200-row test set")
        fig.tight_layout(); fig.savefig(out_dir / "confusion_matrices_test.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------
# Section 10 / 15: a genuine prediction interface (always available,
# built entirely from training-pool-only artifacts -- never touches the
# test set, regardless of the A/B/C decision, mirroring Phase 10-13's own
# convention of always shipping a working interface).
# --------------------------------------------------------------------------


def build_deployment_artifacts(training_pool: pd.DataFrame, feature_groups, pcfg: PreprocessingConfig, *, seed: int = PRIMARY_SEED) -> dict:
    processed = process_stage(training_pool, training_pool.iloc[:1], training_pool.iloc[:1], TARGET_COLUMN, feature_groups, pcfg)
    Xtr_classical = processed.X_train_classical.to_numpy() if hasattr(processed.X_train_classical, "to_numpy") else processed.X_train_classical
    ytr = processed.y_train
    Xtr_pca = _pad_to_quantum_input_dim(processed.X_train_quantum)

    xgb_final = _build_xgb()
    xgb_final.fit(Xtr_classical, ytr)
    p_xgb_train = xgb_final.predict_proba(Xtr_classical)[:, 1]

    q_module = QuantumRefinementModule(RefinementConfig(seed=seed))
    q_sel = select_scalar_classifier_weights(q_module, q_module.config.n_trainable_params(), Xtr_pca, ytr, seed=seed)
    p_quantum_train = _sigmoid(q_module.score(Xtr_pca, q_sel["weights"]))

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(np.column_stack([p_xgb_train, p_quantum_train]), ytr)

    from src.preprocessing.pipeline import build_quantum_pipeline

    quantum_pipeline = build_quantum_pipeline(pcfg.pca_n_components, pcfg.quantum_range)
    quantum_pipeline.fit(processed.X_train_classical)

    return {
        "feature_groups": feature_groups, "shared_pipeline": processed.shared_pipeline, "quantum_pipeline": quantum_pipeline,
        "xgb_final": xgb_final, "quantum_module": q_module, "quantum_weights": q_sel["weights"], "meta_learner": lr,
    }


def predict_patient_risk(patient_row_raw: pd.Series, deployed: dict) -> dict:
    t0 = time.perf_counter()
    feature_cols = deployed["feature_groups"].all_columns
    X_raw = pd.DataFrame([patient_row_raw[feature_cols]])
    X_classical = deployed["shared_pipeline"].transform(X_raw)
    X_pca = _pad_to_quantum_input_dim(np.asarray(deployed["quantum_pipeline"].transform(X_classical)))

    p_xgb = float(deployed["xgb_final"].predict_proba(X_classical)[:, 1][0])
    raw_q = deployed["quantum_module"].score(X_pca, deployed["quantum_weights"])
    p_quantum = float(_sigmoid(raw_q)[0])
    p_final = float(deployed["meta_learner"].predict_proba(np.array([[p_xgb, p_quantum]]))[:, 1][0])

    latency_ms = (time.perf_counter() - t0) * 1000
    risk_category = "high" if p_final >= 0.66 else ("moderate" if p_final >= 0.33 else "low")
    return {
        "classical_backbone_risk": round(p_xgb, 4), "quantum_learner_risk": round(p_quantum, 4),
        "final_stacked_risk": round(p_final, 4), "risk_category": risk_category,
        "model_version": "phase15_quantum_classical_stack_v1", "inference_latency_ms": round(latency_ms, 3),
    }


def predict_batch(df_raw: pd.DataFrame, deployed: dict) -> pd.DataFrame:
    feature_cols = deployed["feature_groups"].all_columns
    X_raw = df_raw[feature_cols]
    X_classical = deployed["shared_pipeline"].transform(X_raw)
    X_pca = _pad_to_quantum_input_dim(np.asarray(deployed["quantum_pipeline"].transform(X_classical)))
    p_xgb = deployed["xgb_final"].predict_proba(X_classical)[:, 1]
    p_quantum = _sigmoid(deployed["quantum_module"].score(X_pca, deployed["quantum_weights"]))
    p_final = deployed["meta_learner"].predict_proba(np.column_stack([p_xgb, p_quantum]))[:, 1]
    return pd.DataFrame({"classical_backbone_risk": p_xgb, "quantum_learner_risk": p_quantum, "final_stacked_risk": p_final})


if __name__ == "__main__":
    result = run_all_stages()
    print(json.dumps({k: v for k, v in result.items() if k not in ("statistics",)}, indent=2, default=float))
