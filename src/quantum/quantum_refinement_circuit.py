"""Phase 13: a quantum RESIDUAL/REFINEMENT circuit -- a genuinely
different ROLE from every prior quantum phase's, not a variant of them.

WHY THIS IS NOT PHASE 10/12 AGAIN: Phase 10 and 12 both concatenated
quantum-derived features alongside classical features and asked a
classifier to learn from the combination directly (a "quantum feature
engineering" role). Phase 13's circuit instead receives a compact summary
that INCLUDES the classical backbone's own prediction (XGBoost's
out-of-fold probability + its margin from 0.5) and is trained to predict
the RESIDUAL the classical model got wrong (`y - p_xgb_oof`) -- a
"specialist correction" role, structurally analogous to residual/boosting
refinement stages in classical ensembling, but performed by a small
trainable quantum circuit instead of another tree ensemble.

ARCHITECTURE: 6 qubits (one per compact input dimension: 4 PCA features +
XGBoost OOF probability + margin), 1 data-re-uploading layer, trainable
RY rotations, trainable RZZ entanglement (circular connectivity, 6
pairs), then a TRAINABLE LINEAR READOUT collapsing the 6 single-qubit
<Z_i> expectation values into ONE aggregate quantum refinement score --
matching the governing spec's "one aggregate quantum refinement score"
option, and deliberately NOT a multi-dimensional feature vector fed to a
downstream classifier (that would just be Phase 12 again).

Kept deliberately small (per the explicit computational-budget
instruction): 1 layer, 6 qubits, 19 total trainable parameters (6 RY + 6
RZZ + 6 readout weights + 1 readout bias). Exact statevector computation
(EstimatorQNN + StatevectorEstimator(default_precision=0.0), the
Phase 10/12-established, twice-verified exact configuration -- the
`default_precision=0.0` fix is applied to BOTH the estimator and
EstimatorQNN itself, exactly as those phases require and test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

N_QUBITS = 6
N_LAYERS = 1
ENTANGLING_PAIRS = [(i, (i + 1) % N_QUBITS) for i in range(N_QUBITS)]  # circular, 6 pairs

INPUT_FEATURE_NAMES = ["pca_0", "pca_1", "pca_2", "pca_3", "xgb_oof_proba", "xgb_margin"]


def build_refinement_circuit(n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    x_params = ParameterVector("x", n_qubits)
    theta_params = ParameterVector("theta", (n_qubits + len(ENTANGLING_PAIRS)) * n_layers)
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(x_params[q], q)
        for q in range(n_qubits):
            qc.ry(theta_params[idx], q)
            idx += 1
        for (i, j) in ENTANGLING_PAIRS:
            qc.rzz(theta_params[idx], i, j)
            idx += 1
    return qc, x_params, theta_params


def _z_observables(n_qubits: int = N_QUBITS) -> list[SparsePauliOp]:
    obs = []
    for i in range(n_qubits):
        label = ["I"] * n_qubits
        label[n_qubits - 1 - i] = "Z"
        obs.append(SparsePauliOp("".join(label)))
    return obs


@dataclass(frozen=True)
class RefinementConfig:
    n_qubits: int = N_QUBITS
    n_layers: int = N_LAYERS
    seed: int = 42

    def n_circuit_params(self) -> int:
        return (self.n_qubits + len(ENTANGLING_PAIRS)) * self.n_layers

    def n_readout_params(self) -> int:
        return self.n_qubits + 1  # weights + bias

    def n_trainable_params(self) -> int:
        return self.n_circuit_params() + self.n_readout_params()

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits, "n_layers": self.n_layers, "seed": self.seed,
            "n_circuit_params": self.n_circuit_params(), "n_readout_params": self.n_readout_params(),
            "n_trainable_params": self.n_trainable_params(),
            "input_features": INPUT_FEATURE_NAMES,
            "encoding": "RY(x_i) angle encoding", "trainable_rotation": "RY(theta_i) per qubit",
            "trainable_entanglement": "RZZ(theta_ij), circular connectivity",
            "readout": "trainable linear combination of <Z_i> -> ONE aggregate quantum refinement score",
            "estimator": "StatevectorEstimator(default_precision=0.0) + EstimatorQNN(default_precision=0.0) -- EXACT, no shots",
        }


class QuantumRefinementModule:
    """`.score(X, params) -> (n_samples,)` -- ONE aggregate refinement
    score per sample, NOT a multi-dimensional feature vector (that
    distinction is the point: this is a residual predictor, not a
    feature-engineering block)."""

    def __init__(self, config: RefinementConfig | None = None) -> None:
        self.config = config or RefinementConfig()
        self.circuit, self._x_params, self._theta_params = build_refinement_circuit(
            self.config.n_qubits, self.config.n_layers,
        )
        self._observables = _z_observables(self.config.n_qubits)
        estimator = StatevectorEstimator(default_precision=0.0, seed=self.config.seed)
        self._qnn = EstimatorQNN(
            circuit=self.circuit, estimator=estimator, observables=self._observables,
            input_params=list(self._x_params), weight_params=list(self._theta_params),
            default_precision=0.0,  # see module docstring -- the Phase 10/12 EstimatorQNN bug fix
        )

    def expectation_values(self, X: np.ndarray, circuit_params: np.ndarray) -> np.ndarray:
        """The raw <Z_i> vector, shape (n_samples, n_qubits) -- exposed for
        explainability/debugging, bounded in [-1, 1]."""
        return np.asarray(self._qnn.forward(X, circuit_params))

    def score(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """params = [circuit_params (n_circuit_params), readout_weights (n_qubits), readout_bias (1)]."""
        n_circ = self.config.n_circuit_params()
        circuit_params = params[:n_circ]
        readout_w = params[n_circ:n_circ + self.config.n_qubits]
        readout_b = params[n_circ + self.config.n_qubits]
        z = self.expectation_values(X, circuit_params)
        return z @ readout_w + readout_b
