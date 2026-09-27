"""Phase 12's matched non-quantum control: classical Random Fourier
Features (RFF), NOT untrained random noise (Phase 10/11's simpler
control). This phase's brief explicitly requires the control have
"comparable downstream classifier capacity" and "must not be
intentionally weak" -- RFF is a well-established (Rahimi & Recht 2007),
genuinely powerful classical technique for approximating nonlinear kernel
feature maps, and is structurally analogous to the quantum block: a
quantum RY-rotation's expectation value IS a cosine of the rotation
angle, so RFF's `cos(w.x + phi)` nonlinearity is not an arbitrary
comparison -- it's the same family of trigonometric nonlinear feature
map, built classically instead of on a quantum circuit.

WHAT IS FIXED vs TRAINABLE (matching the quantum block's own split):
    - W (the random projection directions) and b (baseline phase offset):
      FIXED per seed, drawn once, analogous to the quantum circuit's fixed
      STRUCTURE (which qubits, which gates) being decided before training.
    - phi (K additional phase-shift parameters, one per output dimension):
      TRAINABLE, selected via the IDENTICAL outer COBYLA + train-only CV
      procedure the quantum block uses (see
      src.large_dataset.phase12_quantum_representation.select_representation_weights)
      -- so "training effort" is matched, not just output dimensionality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchedControlConfig:
    input_dim: int = 4
    output_dim: int = 8
    seed: int = 42

    def n_trainable_params(self) -> int:
        return self.output_dim

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim, "output_dim": self.output_dim, "seed": self.seed,
            "n_trainable_params": self.n_trainable_params(),
            "family": "Random Fourier Features (Rahimi & Recht 2007): cos(W.x + b + phi)",
            "fixed": "W (random projection directions), b (baseline phase offsets) -- drawn once per seed",
            "trainable": "phi (K phase-shift parameters), selected via the SAME COBYLA+CV procedure as the quantum block",
        }


class MatchedControlLayer:
    """Same `.transform(X, phi) -> (n_samples, output_dim)` interface as
    QuantumRepresentationLayer -- the outer training procedure
    (select_representation_weights) is representation-agnostic and works
    with either."""

    def __init__(self, config: MatchedControlConfig | None = None) -> None:
        self.config = config or MatchedControlConfig()
        rng = np.random.RandomState(self.config.seed)
        self.W = rng.normal(size=(self.config.input_dim, self.config.output_dim))
        self.b = rng.uniform(0, 2 * np.pi, size=self.config.output_dim)

    def transform(self, X: np.ndarray, phi: np.ndarray) -> np.ndarray:
        projection = X @ self.W + self.b
        return np.cos(projection + phi)
