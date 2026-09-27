"""Variational Quantum Classifier (Phase 8B screening) -- built on
`qiskit-machine-learning` / `qiskit-algorithms` (installed for this phase;
the rest of the project's quantum pipeline -- kernel.py, backends.py --
uses bare `qiskit` only, since QSVM needs nothing more than exact
statevector fidelities. A trainable VQC is a genuinely different
computational pattern -- parameterized-circuit forward/backward passes,
an optimizer loop -- that qiskit-machine-learning's `VQC` class already
implements, tested and maintained; hand-rolling it here would duplicate
that machinery for no scientific benefit. Verified compatible with the
already-installed `qiskit==2.2.3` before use (functional smoke test: a
tiny VQC.fit()/.predict_proba() round-trip), not merely import-checked.

ARCHITECTURE:
    feature_map: THIS PROJECT's OWN build_feature_map("zz_feature_map",
                 reps=1, entanglement="linear") -- i.e. the IDENTICAL
                 angle-encoding circuit family the baseline QSVM's
                 fidelity kernel already uses, reps trimmed to 1 (vs. the
                 QSVM's reps=2) so the ansatz's own depth budget (below)
                 keeps the total circuit shallow. Reusing the project's
                 own encoding, rather than inventing a new one, means any
                 VQC-vs-QSVM difference reflects "trainable circuit vs.
                 fixed fidelity kernel" specifically, not "different data
                 encodings."
    ansatz:      qiskit's `real_amplitudes` (RY rotations + linear CNOT
                 entanglement -- a standard, hardware-efficient ansatz),
                 reps configurable (default 2, i.e. "1-2 trainable
                 layers" per the screening brief).
    optimizer:   COBYLA (qiskit_machine_learning.optimizers) -- gradient-
                 free, standard for small variational-circuit problems.

IMPORTANT, DISCLOSED METHODOLOGICAL DIFFERENCE FROM THE QSVM BASELINE:
qiskit-machine-learning's `VQC` is built on a `SamplerQNN`, which reads
out PROBABILITIES FROM FINITE-SHOT MEASUREMENT COUNTS (`StatevectorSampler`
has no exact/analytic mode -- verified: `shots=None` raises a TypeError).
This means VQC training and inference carry genuine shot noise, unlike
every other quantum computation in this project (the fidelity kernel is
computed via exact `Statevector` amplitudes, zero shot noise). This is
disclosed explicitly, not hidden: a fixed seed and a deliberately large
shot count (see VQCConfig.shots) keep the noise small and the run
reproducible, but it is not the same "exact simulation" guarantee the
QSVM enjoys. See docs/PHASE_8B_VQC_SCREENING.md, "VQC architecture".
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA

from src.quantum.feature_maps import build_feature_map


@dataclass(frozen=True)
class VQCConfig:
    """Defaults chosen from a measured cost benchmark (see
    docs/PHASE_8B_VQC_SCREENING.md, "Computational cost"), not guessed:
    a first attempt at shots=4096 cost ~28.5s per full-batch (n=2000)
    function evaluation -- a maxiter=100 run would have cost 1.5-4+ hours,
    an "unexpectedly expensive" result per this phase's own stop
    condition. Profiling showed the dominant cost is FIXED per-sample
    circuit-dispatch overhead (~2.6ms/sample, roughly shot-independent),
    not shot count itself -- so shots was reduced to 512 (still a
    reasonable, defensible shot count, not shot-starved) and
    `optimizer_maxiter` (confirmed to cap the number of function
    evaluations exactly, 1:1) was capped at 50, for a measured ~9.0s per
    evaluation -> ~7.5 minutes total training time. This is a genuinely
    screening-scale optimization budget (~4x the 12 trainable parameters
    worth of function evaluations), not a fully-converged VQC -- disclosed
    explicitly as a limitation, not hidden.
    """

    n_qubits: int = 4
    feature_map_reps: int = 1
    ansatz_reps: int = 2
    optimizer_maxiter: int = 50
    shots: int = 512
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits, "feature_map_reps": self.feature_map_reps,
            "ansatz_reps": self.ansatz_reps, "optimizer_maxiter": self.optimizer_maxiter,
            "shots": self.shots, "seed": self.seed,
            "feature_map": "zz_feature_map (project's own build_feature_map, linear entanglement)",
            "ansatz": "real_amplitudes (linear entanglement)",
            "optimizer": "COBYLA",
            "readout": "SamplerQNN default (parity interpretation), StatevectorSampler (finite-shot)",
        }


def build_vqc_circuits(config: VQCConfig) -> tuple[QuantumCircuit, QuantumCircuit]:
    """Returns (feature_map, ansatz) -- built separately so the caller/tests
    can inspect qubit counts and parameter counts before assembling the VQC."""
    feature_map = build_feature_map(
        "zz_feature_map", config.n_qubits, reps=config.feature_map_reps, entanglement="linear",
    )
    ansatz = real_amplitudes(num_qubits=config.n_qubits, reps=config.ansatz_reps, entanglement="linear")
    return feature_map, ansatz


def n_trainable_params(config: VQCConfig) -> int:
    """real_amplitudes(num_qubits=n, reps=r) has n*(r+1) trainable RY angles."""
    return config.n_qubits * (config.ansatz_reps + 1)


def build_vqc(config: VQCConfig | None = None) -> VQC:
    """Builds an UNFITTED VQC. All randomness (initial point, and the
    finite-shot sampler's own randomness) is seeded from config.seed, so
    two calls with the same config + same training data produce identical
    fitted models -- exact reproducibility despite the shot-based readout.
    """
    config = config or VQCConfig()
    feature_map, ansatz = build_vqc_circuits(config)
    sampler = StatevectorSampler(default_shots=config.shots, seed=config.seed)
    optimizer = COBYLA(maxiter=config.optimizer_maxiter)

    import numpy as np

    rng = np.random.RandomState(config.seed)
    initial_point = rng.uniform(-0.1, 0.1, size=ansatz.num_parameters)

    return VQC(
        feature_map=feature_map, ansatz=ansatz, optimizer=optimizer,
        sampler=sampler, initial_point=initial_point,
    )
