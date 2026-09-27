"""Phase 13's matched non-quantum refinement controls.

TWO controls, per the governing spec:

    MatchedRFFRefinement (Model C): the fair architectural analogue --
        classical Random Fourier Features (same family Phase 12 used,
        `cos(W.x + b + phi)`) reduced to ONE aggregate score via the SAME
        "K fixed random features -> trainable linear readout" structure
        the quantum module uses. W, b fixed per seed; phi (K phase
        shifts) AND the readout weights/bias are trainable, selected via
        the identical outer procedure.

    NoOpRandomRefinement (Model D): an additional sanity control -- a
        trainable linear readout over FIXED, UNTRAINED random Gaussian
        features (no phi at all). If even this can match the quantum
        module, the informative signal is coming from the readout's
        ability to shrink toward zero, not from any learned nonlinear
        transform -- an important sanity check the spec explicitly
        permits ("random/no-op refinement if useful as an additional
        sanity control").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchedRFFConfig:
    input_dim: int = 6
    n_features: int = 6
    seed: int = 42

    def n_trainable_params(self) -> int:
        return self.n_features + self.n_features + 1  # phi + readout_w + readout_b

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim, "n_features": self.n_features, "seed": self.seed,
            "n_trainable_params": self.n_trainable_params(),
            "family": "Random Fourier Features: cos(W.x + b + phi) -> trainable linear readout -> 1 aggregate score",
        }


class MatchedRFFRefinement:
    def __init__(self, config: MatchedRFFConfig | None = None) -> None:
        self.config = config or MatchedRFFConfig()
        rng = np.random.RandomState(self.config.seed)
        self.W = rng.normal(size=(self.config.input_dim, self.config.n_features))
        self.b = rng.uniform(0, 2 * np.pi, size=self.config.n_features)

    def score(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        k = self.config.n_features
        phi, readout_w, readout_b = params[:k], params[k:2 * k], params[2 * k]
        features = np.cos(X @ self.W + self.b + phi)
        return features @ readout_w + readout_b


@dataclass(frozen=True)
class NoOpRandomConfig:
    input_dim: int = 6
    n_features: int = 6
    seed: int = 42

    def n_trainable_params(self) -> int:
        return self.n_features + 1  # readout_w + readout_b only -- no phi

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim, "n_features": self.n_features, "seed": self.seed,
            "n_trainable_params": self.n_trainable_params(),
            "family": "FIXED untrained random Gaussian features -> trainable linear readout only -- sanity control",
        }


class NoOpRandomRefinement:
    def __init__(self, config: NoOpRandomConfig | None = None) -> None:
        self.config = config or NoOpRandomConfig()
        rng = np.random.RandomState(self.config.seed)
        self.random_features_fn_seed = self.config.seed
        self._W = rng.normal(size=(self.config.input_dim, self.config.n_features))

    def score(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        k = self.config.n_features
        readout_w, readout_b = params[:k], params[k]
        features = X @ self._W  # fixed, untrained linear random projection (no nonlinearity, no training on W)
        return features @ readout_w + readout_b
