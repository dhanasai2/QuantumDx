"""Phase 12: a trainable quantum LATENT REPRESENTATION block -- materially
different from Phase 10's circuit, testing a genuinely different
hypothesis (see docs/PHASE_12_QUANTUM_REPRESENTATION.md, "What is
fundamentally different about Phase 12").

WHY THIS IS NOT PHASE 10 AGAIN (concrete, structural differences):

    Phase 10                          Phase 12
    --------                          --------
    1 data-re-uploading layer         2 data-re-uploading layers
    RY-only trainable rotation        RY trainable rotation per layer
    FIXED linear CNOT entanglement    TRAINABLE RZZ entanglement (circular
                                       connectivity: (0,1),(1,2),(2,3),(3,0)
                                       -- one MORE edge than Phase 10's
                                       linear chain, and the interaction
                                       STRENGTH is learned, not fixed)
    4 observables: <Z_i> only         8 observables: <Z_i> (4) AND
                                       <Z_i Z_j> two-qubit correlators (4,
                                       the same circular pairs) -- the
                                       correlators are information a
                                       linear (PCA) representation or a
                                       single-qubit-only readout cannot
                                       express, by construction
    4 trainable parameters            16 trainable parameters (8 RY + 8 RZZ)
    output dim K=4                    output dim K=8

STILL DELIBERATELY SMALL: 4 qubits (same budget as every prior quantum
phase), exact statevector-based expectation values (EstimatorQNN +
StatevectorEstimator(default_precision=0.0) -- the verified, Phase-10-
established exact configuration, reused byte-for-byte, INCLUDING the
default_precision=0.0 fix on EstimatorQNN itself, not just the estimator
-- see the bounded-expectation-value test in
tests/test_phase12_quantum_representation.py for the regression guard).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

N_QUBITS = 4
N_LAYERS = 2
#: Circular connectivity -- one more edge than Phase 10's linear chain
#: (0,1),(1,2),(2,3),(3,0) -- deliberately richer, still small.
ENTANGLING_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 0)]


def build_quantum_representation_circuit(
    n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS,
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Data re-uploading (n_layers layers) with TRAINABLE single-qubit RY
    rotations AND trainable two-qubit RZZ entangling gates (circular
    connectivity). Returns (circuit, x_params, theta_params).

    theta layout per layer: [n_qubits RY angles, len(ENTANGLING_PAIRS) RZZ angles].
    """
    x_params = ParameterVector("x", n_qubits)
    n_theta_per_layer = n_qubits + len(ENTANGLING_PAIRS)
    theta_params = ParameterVector("theta", n_theta_per_layer * n_layers)

    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(x_params[q], q)  # data re-uploading (encoding), every layer
        for q in range(n_qubits):
            qc.ry(theta_params[idx], q)  # trainable single-qubit rotation
            idx += 1
        for (i, j) in ENTANGLING_PAIRS:
            qc.rzz(theta_params[idx], i, j)  # trainable two-qubit entanglement
            idx += 1
    assert idx == len(theta_params)
    return qc, x_params, theta_params


def _observables(n_qubits: int = N_QUBITS) -> list[SparsePauliOp]:
    """4 single-qubit Z observables + 4 two-qubit ZZ correlators (the same
    circular pairs used for entanglement) -- 8 total, the quantum latent
    representation's dimensionality K."""
    obs = []
    for i in range(n_qubits):
        label = ["I"] * n_qubits
        label[n_qubits - 1 - i] = "Z"
        obs.append(SparsePauliOp("".join(label)))
    for (i, j) in ENTANGLING_PAIRS:
        label = ["I"] * n_qubits
        label[n_qubits - 1 - i] = "Z"
        label[n_qubits - 1 - j] = "Z"
        obs.append(SparsePauliOp("".join(label)))
    return obs


OBSERVABLE_NAMES = (
    [f"quantum_expZ_qubit{i}" for i in range(N_QUBITS)]
    + [f"quantum_expZZ_qubit{i}_qubit{j}" for (i, j) in ENTANGLING_PAIRS]
)


@dataclass(frozen=True)
class QuantumRepresentationConfig:
    n_qubits: int = N_QUBITS
    n_layers: int = N_LAYERS
    seed: int = 42

    def n_trainable_params(self) -> int:
        return (self.n_qubits + len(ENTANGLING_PAIRS)) * self.n_layers

    def output_dim(self) -> int:
        return self.n_qubits + len(ENTANGLING_PAIRS)

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits, "n_layers": self.n_layers, "seed": self.seed,
            "n_trainable_params": self.n_trainable_params(), "output_dim": self.output_dim(),
            "entangling_pairs": ENTANGLING_PAIRS,
            "observables": OBSERVABLE_NAMES,
            "encoding": "RY(x_i) angle encoding, re-uploaded every layer",
            "trainable_single_qubit": "RY(theta_i) per qubit per layer",
            "trainable_entanglement": "RZZ(theta_ij) per circular-adjacent pair per layer (TRAINABLE interaction strength)",
            "estimator": "StatevectorEstimator(default_precision=0.0) + EstimatorQNN(default_precision=0.0) -- EXACT, no shot noise",
        }


class QuantumRepresentationLayer:
    """The quantum latent-representation block. Structurally a FEATURE
    TRANSFORM (never a classifier) -- `.transform(X, theta)` returns the
    K=8-dimensional latent representation; the caller concatenates it with
    classical features and hands the result to a separate classifier."""

    def __init__(self, config: QuantumRepresentationConfig | None = None) -> None:
        self.config = config or QuantumRepresentationConfig()
        self.circuit, self._x_params, self._theta_params = build_quantum_representation_circuit(
            self.config.n_qubits, self.config.n_layers,
        )
        self._observables = _observables(self.config.n_qubits)
        estimator = StatevectorEstimator(default_precision=0.0, seed=self.config.seed)
        self._qnn = EstimatorQNN(
            circuit=self.circuit, estimator=estimator, observables=self._observables,
            input_params=list(self._x_params), weight_params=list(self._theta_params),
            # See Phase 10's documented bug: EstimatorQNN has its OWN
            # default_precision (0.015625) that silently overrides the
            # estimator's exact setting unless passed here too.
            default_precision=0.0,
        )

    def transform(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Exact expectation values for every sample in X, at weights
        theta. Shape (n_samples, output_dim())."""
        return np.asarray(self._qnn.forward(X, theta))
