"""Trainable quantum FEATURE layer for a hybrid quantum-classical
classifier (Phase 10). NOT a standalone quantum classifier.

ARCHITECTURE:

    classical PCA-4 features (x)
        -> [encoding: RY(x_i) on qubit i]           (fixed, per-sample)
        -> [trainable: RY(theta_i) on qubit i]       (shared across samples)
        -> [entangling: linear CNOT chain]
        -> exact expectation values <Z_i> for i in 0..3   (EXACT, no shots
           -- see "why EstimatorQNN, not SamplerQNN" below)
        = 4 quantum features, ONE PER QUBIT (interpretable 1:1 with which
          PCA component that qubit encodes -- see HybridFeatureReport)

    [classical PCA-4 features] concatenated with [4 quantum features]
        -> sklearn LogisticRegression (the ONLY component that produces a
           class prediction -- the quantum circuit never does)

This is deliberately NOT a repeat of Phase 8B's VQC: Phase 8B's VQC used
the RAW circuit output (via SamplerQNN's parity interpretation) AS the
classifier's decision function. Here, the circuit's expectation values are
merely EXTRA FEATURES handed to a real, separate classical Logistic
Regression, alongside the original classical features it would have used
anyway -- removing the quantum layer entirely still leaves a complete,
functioning classical pipeline (Model A from Phase 9).

WHY EstimatorQNN, NOT SamplerQNN (Phase 8B's choice): Phase 8B explicitly
disclosed that `VQC`'s SamplerQNN reads finite-shot measurement counts
(`StatevectorSampler` has no exact/analytic mode). `EstimatorQNN` backed by
`StatevectorEstimator(default_precision=0.0)` computes EXACT expectation
values (verified: default_precision=0.0 means analytic, not sampled) --
the preferred, shot-noise-free choice this phase explicitly asks for, and
consistent with the rest of the project's exact-statevector convention
(kernel.py's fidelity kernel has always been exact).

HOW THE TRAINABLE PARAMETERS ARE FIT (this is the part that makes this a
genuine "trainable quantum layer" rather than a fixed, arbitrary feature
transform, while staying strictly train-only): for a candidate weight
vector theta, the quantum features for ALL training samples are computed
(exact, batched via EstimatorQNN.forward), concatenated with the
classical PCA features, and a Logistic Regression is fit and scored via
K-fold cross-validation -- ON TRAINING DATA ONLY. A gradient-free
optimizer (COBYLA, matching Phase 8B's convention) searches theta to
maximize this CV score. Once theta is selected, the FINAL Logistic
Regression is refit on the FULL training set's concatenated features at
the winning theta -- identical in spirit to how every classical model in
this project selects hyperparameters via CV then refits on full training
data (src.classical.tuning.run_grid_search's own convention).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def build_hybrid_circuit(n_qubits: int, reps: int) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Encoding (RY(x_i), re-uploaded each rep) + trainable (RY(theta_i))
    + linear-CNOT entangling, repeated `reps` times. Structurally the same
    family Phase 8B's VQC used (angle encoding + real_amplitudes-style
    trainable rotations + linear entanglement) -- kept deliberately shallow.
    """
    x_params = ParameterVector("x", n_qubits)
    theta_params = ParameterVector("theta", n_qubits * reps)
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _rep in range(reps):
        for q in range(n_qubits):
            qc.ry(x_params[q], q)
        for q in range(n_qubits):
            qc.ry(theta_params[idx], q)
            idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc, x_params, theta_params


def _z_observables(n_qubits: int) -> list[SparsePauliOp]:
    """One SparsePauliOp per qubit: Z on qubit i, identity elsewhere.
    Qiskit's little-endian convention: rightmost character = qubit 0."""
    obs = []
    for i in range(n_qubits):
        label = ["I"] * n_qubits
        label[n_qubits - 1 - i] = "Z"
        obs.append(SparsePauliOp("".join(label)))
    return obs


@dataclass(frozen=True)
class HybridConfig:
    n_qubits: int = 4
    reps: int = 1
    optimizer_maxiter: int = 40
    inner_cv_folds: int = 3
    lr_C: float = 1.0
    seed: int = 42

    def n_trainable_params(self) -> int:
        return self.n_qubits * self.reps

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits, "reps": self.reps, "optimizer_maxiter": self.optimizer_maxiter,
            "inner_cv_folds": self.inner_cv_folds, "lr_C": self.lr_C, "seed": self.seed,
            "n_trainable_params": self.n_trainable_params(),
            "encoding": "RY(x_i) angle encoding, re-uploaded each rep",
            "ansatz": "RY(theta_i) trainable rotation + linear CNOT entanglement",
            "observables": [f"Z on qubit {i} (identity elsewhere), EXACT expectation value" for i in range(self.n_qubits)],
            "estimator": "StatevectorEstimator(default_precision=0.0) -- EXACT, no shot noise",
            "optimizer": "COBYLA (outer loop over quantum-layer weights, train-only CV objective)",
            "final_classifier": "sklearn.linear_model.LogisticRegression on [classical PCA features, quantum expectation features]",
        }


@dataclass
class HybridFitResult:
    theta: np.ndarray
    cv_score_history: list[float] = field(default_factory=list)
    n_function_evaluations: int = 0
    train_seconds: float = 0.0
    final_cv_score: float = 0.0


@dataclass(frozen=True)
class HybridFeatureReport:
    """Documents exactly what each quantum feature is -- required for the
    explainability section (Phase 10 spec). No claim beyond what is
    literally computed is made."""

    n_qubits: int
    feature_names: list[str]
    description: str

    def to_dict(self) -> dict:
        return {"n_qubits": self.n_qubits, "feature_names": self.feature_names, "description": self.description}


class HybridQuantumFeatureLayer:
    """The quantum FEATURE layer + classical LR head, trained end-to-end
    (in the CV-search sense described in the module docstring) on
    TRAINING DATA ONLY.

    `.fit(X_train, y_train)` -- no test-data parameter exists.
    `.predict_proba(X)` -- scores X only, no label parameter.
    """

    def __init__(self, config: HybridConfig | None = None) -> None:
        self.config = config or HybridConfig()
        self.circuit, self._x_params, self._theta_params = build_hybrid_circuit(
            self.config.n_qubits, self.config.reps
        )
        self._observables = _z_observables(self.config.n_qubits)
        estimator = StatevectorEstimator(default_precision=0.0, seed=self.config.seed)
        self._qnn = EstimatorQNN(
            circuit=self.circuit, estimator=estimator, observables=self._observables,
            input_params=list(self._x_params), weight_params=list(self._theta_params),
            # CRITICAL: EstimatorQNN has its OWN default_precision (0.015625
            # by default), which overrides the underlying estimator's own
            # setting on every .run() call it makes internally -- passing
            # default_precision=0.0 here too is required for exact (no
            # shot-noise-equivalent-precision) expectation values. Verified:
            # without this, EstimatorQNN output silently drifts from the
            # true Statevector.expectation_value() by up to several percent
            # (caught by a bounded-expectation-value unit test producing a
            # value of -1.02, outside [-1, 1] -- the real Statevector value
            # was -0.99). See tests/test_hybrid_quantum_features.py.
            default_precision=0.0,
        )
        self.fit_result_: HybridFitResult | None = None
        self.lr_: LogisticRegression | None = None

    def quantum_features(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Exact <Z_i> expectation values for every sample in X, at the
        given trainable weights theta. Shape (n_samples, n_qubits)."""
        return np.asarray(self._qnn.forward(X, theta))

    def _concat(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.hstack([X, self.quantum_features(X, theta)])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "HybridQuantumFeatureLayer":
        rng = np.random.RandomState(self.config.seed)
        theta0 = rng.uniform(-0.1, 0.1, size=self.config.n_trainable_params())
        cv = StratifiedKFold(n_splits=self.config.inner_cv_folds, shuffle=True, random_state=self.config.seed)

        history: list[float] = []
        n_evals = 0

        def outer_objective(theta: np.ndarray) -> float:
            nonlocal n_evals
            n_evals += 1
            Z = self._concat(X_train, theta)
            lr = LogisticRegression(C=self.config.lr_C, max_iter=1000, random_state=self.config.seed)
            scores = cross_val_score(lr, Z, y_train, cv=cv, scoring="roc_auc")
            mean_score = float(np.mean(scores))
            history.append(mean_score)
            return -mean_score  # COBYLA minimizes

        optimizer = COBYLA(maxiter=self.config.optimizer_maxiter)
        t0 = time.perf_counter()
        result = optimizer.minimize(outer_objective, theta0)
        train_seconds = time.perf_counter() - t0

        best_theta = result.x
        Z_full = self._concat(X_train, best_theta)
        self.lr_ = LogisticRegression(C=self.config.lr_C, max_iter=1000, random_state=self.config.seed)
        self.lr_.fit(Z_full, y_train)

        self.fit_result_ = HybridFitResult(
            theta=best_theta, cv_score_history=history, n_function_evaluations=n_evals,
            train_seconds=train_seconds, final_cv_score=max(history) if history else float("nan"),
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.fit_result_ is None or self.lr_ is None:
            raise RuntimeError("HybridQuantumFeatureLayer.fit() must be called before predict_proba().")
        Z = self._concat(X, self.fit_result_.theta)
        return self.lr_.predict_proba(Z)

    def feature_report(self, pca_feature_names: list[str] | None = None) -> HybridFeatureReport:
        pca_names = pca_feature_names or [f"pca_{i}" for i in range(self.config.n_qubits)]
        quantum_names = [f"quantum_expZ_qubit{i}" for i in range(self.config.n_qubits)]
        return HybridFeatureReport(
            n_qubits=self.config.n_qubits,
            feature_names=pca_names + quantum_names,
            description=(
                "Each quantum_expZ_qubit{i} feature is the EXACT expectation value of the "
                "Pauli-Z observable on qubit i, after that qubit's PCA-encoded input has passed "
                "through the trainable ansatz and linear entangling layer. It is not claimed to "
                "represent any specific clinical or physical quantity beyond this literal "
                "definition -- it is a learned, entangled nonlinear transform of the 4 PCA "
                "features, in [-1, 1]."
            ),
        )
