"""Label-aware quantum feature map (Phase 8A screening experiment).

Tests a DIFFERENT hypothesis than the unsupervised MI-adaptive map
(src.quantum.adaptive_feature_map, Phase 6/7): that feature-pair
INTERACTIONS which are informative about the class label -- not merely
statistically dependent on each other -- make better candidates for a
QSVM's ZZ entangling terms.

WHY "INTERACTION INFORMATION" (SYNERGY), NOT JUST LABEL RELEVANCE:
Picking the pair of features that are each individually most predictive of
the label would not test anything new (a ranking metric like that could
just as well drive a classical feature-selection step, with no particular
relevance to why a JOINT two-qubit term should help). The criterion used
here is INTERACTION INFORMATION (synergy):

    Synergy(i, j) = I( (X_i, X_j) ; Y ) - I(X_i; Y) - I(X_j; Y)

the amount of label information the PAIR carries jointly, beyond what each
feature carries independently. A positive synergy means the pair is more
informative together than apart -- exactly the kind of pairwise structure
a joint (ZZ) interaction term can represent that two independent (Z-only)
terms cannot. This is a standard information-theoretic quantity, not a
new/ad hoc metric.

WHY DISCRETE (BINNED) MUTUAL INFORMATION, NOT THE k-NN CONTINUOUS
ESTIMATOR adaptive_feature_map.py USES: joint mutual information between
TWO continuous features and a label has no simple closed-form k-NN
estimator here (scikit-learn's mutual_info_classif is univariate-only,
one feature column against the label at a time). Discretizing each
feature into quantile bins (fit on TRAINING data only) makes the joint
distribution of a feature PAIR a simple discrete random variable, so both
I(X_i,X_j; Y) and the marginals I(X_i;Y), I(X_j;Y) can be computed exactly
via scikit-learn's discrete-discrete mutual_info_score -- no random_state,
no noise injection, fully deterministic given the data.

TRAINING-DATA-ONLY, BUT LABEL-AWARE (a real, documented difference from
adaptive_feature_map.py): every function here takes X_train AND y_train.
Unlike the unsupervised MI-adaptive map, this is BY DESIGN label-aware --
the hypothesis being tested is specifically about label-informed
structure. What it must never see, and structurally cannot (no such
parameter exists anywhere in this module), is the held-out TEST set or
any test-derived prediction/performance/metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from qiskit.circuit import QuantumCircuit
from sklearn.metrics import mutual_info_score

from src.quantum.adaptive_feature_map import build_adaptive_feature_map, linear_chain_pairs

DEFAULT_N_BINS = 4


def discretize_quantile(x: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> np.ndarray:
    """Quantile-bin a single continuous feature into integer bin labels,
    using ONLY the values passed in (the caller must pass TRAINING values
    only). Deterministic: no randomness anywhere in quantile binning.

    Falls back to fewer bins (via duplicates='drop') when a feature has
    too many repeated values at the requested quantile boundaries -- this
    can legitimately happen post-PCA/range-normalization and must not
    raise, just yield a coarser (still valid) discretization.
    """
    bins = pd.qcut(x, q=n_bins, labels=False, duplicates="drop")
    return np.asarray(bins, dtype=int)


def compute_label_aware_interaction_scores(
    X_train: np.ndarray, y_train: np.ndarray, *, n_bins: int = DEFAULT_N_BINS
) -> np.ndarray:
    """Symmetric (n_features, n_features) interaction-information (synergy)
    matrix: Synergy(i,j) = I((Xi,Xj);Y) - I(Xi;Y) - I(Xj;Y), estimated from
    TRAINING DATA AND TRAINING LABELS ONLY. Diagonal set to -inf.
    """
    n_features = X_train.shape[1]
    binned = np.column_stack([discretize_quantile(X_train[:, k], n_bins) for k in range(n_features)])

    marginal_mi = np.array([mutual_info_score(binned[:, k], y_train) for k in range(n_features)])

    scores = np.zeros((n_features, n_features))
    for i, j in combinations(range(n_features), 2):
        # Encode the joint (bin_i, bin_j) pair as a single discrete variable.
        joint = binned[:, i].astype(np.int64) * (int(binned[:, j].max()) + 1) + binned[:, j]
        joint_mi = mutual_info_score(joint, y_train)
        synergy = joint_mi - marginal_mi[i] - marginal_mi[j]
        scores[i, j] = scores[j, i] = synergy

    np.fill_diagonal(scores, -np.inf)
    return scores


def select_label_aware_entanglement_pairs(scores: np.ndarray, n_pairs: int) -> list[tuple[int, int]]:
    """Deterministically select the top-`n_pairs` qubit pairs by synergy
    score. Ties broken by (i, j) ascending -- identical convention to
    select_adaptive_entanglement_pairs, for consistency."""
    n_features = scores.shape[0]
    all_pairs = list(combinations(range(n_features), 2))
    if n_pairs > len(all_pairs):
        raise ValueError(f"n_pairs={n_pairs} exceeds the {len(all_pairs)} available pairs.")
    ranked = sorted(all_pairs, key=lambda p: (-scores[p[0], p[1]], p[0], p[1]))
    return ranked[:n_pairs]


@dataclass(frozen=True)
class LabelAwareMapReport:
    """Full, JSON-serializable record of how the label-aware map was
    derived -- the reproducibility artifact for docs/PHASE_8A_*.md."""

    n_qubits: int
    reps: int
    n_pairs: int
    n_bins: int
    marginal_label_mi: list[float]
    synergy_matrix: list[list[float]]
    baseline_pairs: list[tuple[int, int]]
    selected_pairs: list[tuple[int, int]]
    n_train_used: int

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "reps": self.reps,
            "n_pairs": self.n_pairs,
            "n_bins": self.n_bins,
            "marginal_label_mutual_information": self.marginal_label_mi,
            "synergy_matrix": self.synergy_matrix,
            "baseline_pairs_linear_chain": [list(p) for p in self.baseline_pairs],
            "selected_label_aware_pairs": [list(p) for p in self.selected_pairs],
            "n_train_used": self.n_train_used,
        }


def build_label_aware_feature_map_from_training_data(
    X_train_quantum: np.ndarray,
    y_train: np.ndarray,
    n_qubits: int,
    reps: int,
    *,
    n_pairs: int | None = None,
    n_bins: int = DEFAULT_N_BINS,
) -> tuple[QuantumCircuit, LabelAwareMapReport]:
    """Orchestrator: derive the label-aware entangling structure from
    TRAINING DATA + TRAINING LABELS ONLY, and build the circuit. No test
    array, test label, or performance metric is a parameter anywhere in
    this call graph.

    n_pairs defaults to len(linear_chain_pairs(n_qubits)) -- the SAME
    number of ZZ terms as the baseline (and as the MI-adaptive map), by
    design, to isolate SELECTION CRITERION as the only variable across
    the 3-way comparison (baseline / MI-adaptive / label-aware).
    """
    if n_pairs is None:
        n_pairs = len(linear_chain_pairs(n_qubits))

    scores = compute_label_aware_interaction_scores(X_train_quantum, y_train, n_bins=n_bins)
    marginal_mi = [
        float(mutual_info_score(discretize_quantile(X_train_quantum[:, k], n_bins), y_train))
        for k in range(n_qubits)
    ]
    selected_pairs = select_label_aware_entanglement_pairs(scores, n_pairs)
    circuit = build_adaptive_feature_map(n_qubits, reps, selected_pairs)

    report = LabelAwareMapReport(
        n_qubits=n_qubits,
        reps=reps,
        n_pairs=n_pairs,
        n_bins=n_bins,
        marginal_label_mi=marginal_mi,
        synergy_matrix=scores.tolist(),
        baseline_pairs=linear_chain_pairs(n_qubits),
        selected_pairs=selected_pairs,
        n_train_used=len(X_train_quantum),
    )
    return circuit, report
