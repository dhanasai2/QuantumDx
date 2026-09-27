"""Data-informed adaptive quantum feature map (post-Stage-D experiment).

Tests the hypothesis: does replacing the baseline zz_feature_map's FIXED
"linear chain" entangling structure with a structure chosen from TRAINING
DATA's own pairwise statistical dependence improve the QSVM's fidelity
kernel?

WHY MUTUAL INFORMATION, NOT CORRELATION: the 4 encoded features are PCA
components fit on the same training data. PCA components are, by
construction, LINEARLY uncorrelated (Pearson correlation ~0) over the data
they were fit on -- a correlation-based adaptive rule would therefore be
close to degenerate (near-random pair selection) here. Mutual information
captures general (including nonlinear) statistical dependence and is NOT
zeroed out by PCA's decorrelation, so it is a non-trivial, principled
signal for "how much does qubit i's feature tell you about qubit j's
feature" -- exactly the criterion relevant to whether an explicit ZZ
interaction term between qubits i and j is likely to encode information a
product-state (independent, Z-only) encoding would miss.

WHAT COUNTS AS "TRAINING DATA ONLY": every function in this module takes
only a training feature matrix (X_train_quantum, the same PCA + range-
normalized array kernel.py already consumes). None of them accept a test
array, a label array, or any prediction/performance metric -- verified by
signature inspection in tests/test_adaptive_feature_map.py. The mutual
information matrix and the selected pairs therefore cannot depend on the
200-row comparison set even in principle, not merely "by discipline."

WHAT STAYS FIXED (isolating ONE variable): the baseline zz_feature_map(4,
reps=2, entanglement="linear") has exactly 3 ZZ interaction terms (the
linear-chain edges (0,1),(1,2),(2,3)) plus 4 Z terms (one per qubit), for
reps repetitions. The adaptive map keeps EXACTLY the same term budget --
4 Z terms, 3 ZZ terms, same reps -- and only changes WHICH 3 pairs get the
ZZ term. This is a deliberate design choice (see docs/ADAPTIVE_QSVM.md,
"Why pair count is fixed, not swept") so the comparison isolates
entangling-structure SELECTION from entangling-structure DENSITY.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import pauli_feature_map
from sklearn.feature_selection import mutual_info_regression


def compute_pairwise_mutual_information(
    X_train: np.ndarray, *, random_state: int = 42
) -> np.ndarray:
    """Symmetric (n_features, n_features) mutual-information matrix,
    estimated from TRAINING DATA ONLY via scikit-learn's k-NN based
    estimator (mutual_info_regression, continuous-continuous).

    Diagonal is set to -inf (never a candidate "pair" with itself).
    Symmetrized by averaging both estimation directions (I(i -> j) and
    I(j -> i) can differ slightly for a k-NN estimator; the true mutual
    information is symmetric, so averaging is the standard correction).
    """
    n_features = X_train.shape[1]
    mi = np.zeros((n_features, n_features))
    for j in range(n_features):
        others = [i for i in range(n_features) if i != j]
        scores = mutual_info_regression(
            X_train[:, others], X_train[:, j], random_state=random_state
        )
        for idx, i in enumerate(others):
            mi[i, j] = scores[idx]
    mi_symmetric = (mi + mi.T) / 2.0
    np.fill_diagonal(mi_symmetric, -np.inf)
    return mi_symmetric


def select_adaptive_entanglement_pairs(
    mi_matrix: np.ndarray, n_pairs: int
) -> list[tuple[int, int]]:
    """Deterministically select the top-`n_pairs` qubit pairs by mutual
    information. Ties are broken by (i, j) ascending order (stable,
    reproducible -- no randomness in the selection itself).
    """
    n_features = mi_matrix.shape[0]
    all_pairs = list(combinations(range(n_features), 2))
    if n_pairs > len(all_pairs):
        raise ValueError(f"n_pairs={n_pairs} exceeds the {len(all_pairs)} available pairs.")
    # Sort by (-mi, i, j): highest MI first, deterministic tie-break.
    ranked = sorted(all_pairs, key=lambda p: (-mi_matrix[p[0], p[1]], p[0], p[1]))
    return ranked[:n_pairs]


def build_adaptive_feature_map(
    n_qubits: int,
    reps: int,
    entangling_pairs: list[tuple[int, int]],
    *,
    paulis: tuple[str, ...] = ("Z", "ZZ"),
) -> QuantumCircuit:
    """Build the adaptive feature-map circuit: same Z-term-on-every-qubit +
    ZZ-term structure as pauli_feature_map(paulis=("Z","ZZ")) (verified
    structurally identical to zz_feature_map for the "linear" case in
    development -- see docs/ADAPTIVE_QSVM.md), but with a CUSTOM ZZ-pair
    list instead of the fixed "linear" chain.
    """
    entanglement = {
        1: [[i] for i in range(n_qubits)],
        2: [list(pair) for pair in entangling_pairs],
    }
    return pauli_feature_map(
        feature_dimension=n_qubits, reps=reps, entanglement=entanglement, paulis=list(paulis)
    )


@dataclass(frozen=True)
class AdaptiveMapReport:
    """Full, JSON-serializable record of how the adaptive map was derived --
    the reproducibility artifact for docs/ADAPTIVE_QSVM.md."""

    n_qubits: int
    reps: int
    n_pairs: int
    mi_matrix: list[list[float]]
    baseline_pairs: list[tuple[int, int]]
    selected_pairs: list[tuple[int, int]]
    random_state: int
    n_train_used: int

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "reps": self.reps,
            "n_pairs": self.n_pairs,
            "mutual_information_matrix": self.mi_matrix,
            "baseline_pairs_linear_chain": [list(p) for p in self.baseline_pairs],
            "selected_adaptive_pairs": [list(p) for p in self.selected_pairs],
            "random_state": self.random_state,
            "n_train_used": self.n_train_used,
        }


def linear_chain_pairs(n_qubits: int) -> list[tuple[int, int]]:
    """The baseline zz_feature_map(entanglement='linear') pair structure,
    for direct comparison in the report (not used to build any circuit
    here -- Stage D's baseline circuit/predictions are reused as-is)."""
    return [(i, i + 1) for i in range(n_qubits - 1)]


def build_adaptive_feature_map_from_training_data(
    X_train_quantum: np.ndarray,
    n_qubits: int,
    reps: int,
    *,
    n_pairs: int | None = None,
    random_state: int = 42,
) -> tuple[QuantumCircuit, AdaptiveMapReport]:
    """Orchestrator: derive the adaptive entangling structure from TRAINING
    DATA ONLY (X_train_quantum -- no test array, no label array, no
    performance metric anywhere in this call graph) and build the circuit.

    n_pairs defaults to len(linear_chain_pairs(n_qubits)) -- i.e. the SAME
    number of ZZ terms as the baseline, by design (see module docstring).
    """
    if n_pairs is None:
        n_pairs = len(linear_chain_pairs(n_qubits))

    mi_matrix = compute_pairwise_mutual_information(X_train_quantum, random_state=random_state)
    selected_pairs = select_adaptive_entanglement_pairs(mi_matrix, n_pairs)
    circuit = build_adaptive_feature_map(n_qubits, reps, selected_pairs)

    report = AdaptiveMapReport(
        n_qubits=n_qubits,
        reps=reps,
        n_pairs=n_pairs,
        mi_matrix=mi_matrix.tolist(),
        baseline_pairs=linear_chain_pairs(n_qubits),
        selected_pairs=selected_pairs,
        random_state=random_state,
        n_train_used=len(X_train_quantum),
    )
    return circuit, report
