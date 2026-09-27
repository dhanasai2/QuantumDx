"""Central configuration for the Phase 4 quantum data/execution pipeline.

Mirrors the design of src.preprocessing.config.PreprocessingConfig: every
value that would otherwise be hardcoded in feature_maps.py / backends.py /
kernel.py lives here, as an immutable, validated dataclass, with a matching
configs/quantum.yaml for file-based overrides.

Nothing in this module builds a circuit, touches data, or imports Qiskit.
It only describes HOW the quantum pipeline should be configured.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import yaml

BackendName = Literal["statevector", "noisy_simulator", "ibm_hardware"]
FeatureMapName = Literal["zz_feature_map", "z_feature_map", "pauli_feature_map"]
Entanglement = Literal["linear", "circular", "full"]

#: Backends with a working Phase 4 implementation. The other entries in
#: BackendName are valid CONFIGURATION values (the abstraction accepts
#: them) but resolve to a NotImplementedError-raising placeholder -- see
#: src/quantum/backends.py and docs/QUANTUM_PIPELINE.md.
IMPLEMENTED_BACKENDS = {"statevector"}
ALL_BACKENDS = {"statevector", "noisy_simulator", "ibm_hardware"}
ALL_FEATURE_MAPS = {"zz_feature_map", "z_feature_map", "pauli_feature_map"}
ALL_ENTANGLEMENT = {"linear", "circular", "full"}

#: Hard qubit ceiling (state-vector simulation cost grows as 2^n; near-term
#: hardware is also small). Matches the budget already established in
#: MVP_SPEC.md Section 18.6 / PRD.md Section 18.6.
MAX_QUBITS = 10
WARN_QUBITS_ABOVE = 8
MAX_REPS = 4
WARN_REPS_ABOVE = 2


class QuantumConfigError(ValueError):
    """Raised when a QuantumConfig or a value derived from it is invalid."""


@dataclass(frozen=True)
class QuantumConfig:
    """Immutable configuration for the Phase 4 quantum pipeline.

    n_qubits is intentionally NOT fixed here -- it is derived at run time
    from the actual dimensionality of the Phase 2 quantum-ready feature
    vectors (PreprocessingConfig.pca_n_components, currently 4), so the
    quantum pipeline can never silently disagree with what Phase 2 already
    produced. See src/quantum/data_contract.py.
    """

    backend_name: BackendName = "statevector"
    feature_map_name: FeatureMapName = "zz_feature_map"
    reps: int = 2
    entanglement: Entanglement = "linear"
    #: Only used when feature_map_name == "pauli_feature_map"; ignored otherwise.
    paulis: tuple[str, ...] = ("Z", "ZZ")

    #: Reused from PreprocessingConfig.random_seed by convention at the call
    #: site (see data_contract.py) -- kept here too so a quantum-only config
    #: (e.g. a future noise model's seed) is still traceable on its own.
    random_seed: int = 42

    #: O(N^2) kernel-cost guard (Section 8.4 of MVP_SPEC.md) -- not
    #: exercised by Phase 4's small sanity experiment, but validated and
    #: tested now so Phase 5's full-scale kernel inherits the same guard.
    max_train_samples: int = 300

    kernel_cache_dir: str = "results/quantum/cache/kernels"

    def __post_init__(self) -> None:
        if self.backend_name not in ALL_BACKENDS:
            raise QuantumConfigError(
                f"backend_name must be one of {sorted(ALL_BACKENDS)}; got {self.backend_name!r}."
            )
        if self.feature_map_name not in ALL_FEATURE_MAPS:
            raise QuantumConfigError(
                f"feature_map_name must be one of {sorted(ALL_FEATURE_MAPS)}; got {self.feature_map_name!r}."
            )
        if self.entanglement not in ALL_ENTANGLEMENT:
            raise QuantumConfigError(
                f"entanglement must be one of {sorted(ALL_ENTANGLEMENT)}; got {self.entanglement!r}."
            )
        if not (1 <= self.reps <= MAX_REPS):
            raise QuantumConfigError(f"reps must be in [1, {MAX_REPS}]; got {self.reps}.")
        if self.max_train_samples < 1:
            raise QuantumConfigError(f"max_train_samples must be >= 1; got {self.max_train_samples}.")
        if len(self.paulis) == 0:
            raise QuantumConfigError("paulis must be a non-empty tuple when using pauli_feature_map.")

    def with_overrides(self, **kwargs: object) -> "QuantumConfig":
        return replace(self, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def default(cls) -> "QuantumConfig":
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QuantumConfig":
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if "paulis" in raw and raw["paulis"] is not None:
            raw["paulis"] = tuple(raw["paulis"])
        return cls(**raw)

    def to_dict(self) -> dict:
        return {
            "backend_name": self.backend_name,
            "feature_map_name": self.feature_map_name,
            "reps": self.reps,
            "entanglement": self.entanglement,
            "paulis": list(self.paulis),
            "random_seed": self.random_seed,
            "max_train_samples": self.max_train_samples,
            "kernel_cache_dir": self.kernel_cache_dir,
        }


DEFAULT_CONFIG = QuantumConfig.default()


def check_qubit_budget(n_qubits: int) -> list[str]:
    """Validate n_qubits against the project's quantum resource budget.

    Returns a list of human-readable warning strings (empty if none).
    Raises QuantumConfigError if the hard ceiling is exceeded.
    """
    warnings: list[str] = []
    if n_qubits < 1:
        raise QuantumConfigError(f"n_qubits must be >= 1; got {n_qubits}.")
    if n_qubits > MAX_QUBITS:
        raise QuantumConfigError(
            f"n_qubits={n_qubits} exceeds the hard budget of {MAX_QUBITS}. "
            f"Reduce PreprocessingConfig.pca_n_components or use a smaller feature subset."
        )
    if n_qubits > WARN_QUBITS_ABOVE:
        warnings.append(
            f"n_qubits={n_qubits} exceeds the recommended {WARN_QUBITS_ABOVE}-qubit threshold; "
            f"state-vector simulation cost grows as 2^n."
        )
    return warnings


def check_reps_budget(reps: int) -> list[str]:
    warnings: list[str] = []
    if reps > WARN_REPS_ABOVE:
        warnings.append(
            f"reps={reps} exceeds the recommended {WARN_REPS_ABOVE}; deeper feature maps cost more "
            f"circuit depth and, on real hardware, more decoherence."
        )
    return warnings


def check_sample_budget(n_samples: int, config: QuantumConfig) -> list[str]:
    """O(N^2) kernel-cost guard. Returns warnings; never raises (subsampling,
    not rejection, is the intended remedy -- left to the caller in Phase 5).
    """
    warnings: list[str] = []
    if n_samples > config.max_train_samples:
        warnings.append(
            f"n_samples={n_samples} exceeds max_train_samples={config.max_train_samples}; "
            f"kernel cost is O(n^2) circuit evaluations ({n_samples * (n_samples - 1) // 2} pairs "
            f"for a symmetric train/train kernel at this size). Consider subsampling."
        )
    return warnings
