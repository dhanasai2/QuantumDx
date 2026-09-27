"""Quantum feature map construction (Phase 4, Section 5).

Builds the fixed, parameter-free circuit template that embeds a classical
feature vector into a quantum state:

    |phi(x)> = U_phi(x) |0...0>

The map is a TEMPLATE (a QuantumCircuit with symbolic Parameters); binding
a concrete feature vector x to it (feature_map.assign_parameters(x)) is a
deterministic, side-effect-free operation done later, per-sample, in
kernel.py -- nothing here ever sees data.

Only this file and backends.py / kernel.py import Qiskit, per the
project's established isolation convention (see MVP_SPEC.md Section 15.2 /
17.4): if Qiskit's API moves again, the blast radius is confined to
src/quantum/.

Uses Qiskit's FUNCTION-based feature-map API (qiskit.circuit.library.
zz_feature_map / z_feature_map / pauli_feature_map), not the older
class-based ZZFeatureMap/ZFeatureMap/PauliFeatureMap. Verified during
development: in the installed qiskit==2.2.3, the class forms are
deprecated (removal planned for Qiskit 3.0) in favor of these functions,
which return a plain QuantumCircuit. This is a version-driven choice, not
a preference -- see docs/QUANTUM_PIPELINE.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import pauli_feature_map, z_feature_map, zz_feature_map

from src.quantum.config import ALL_ENTANGLEMENT, ALL_FEATURE_MAPS, QuantumConfigError


def build_feature_map(
    name: str,
    n_qubits: int,
    reps: int = 2,
    entanglement: str = "linear",
    paulis: tuple[str, ...] | list[str] | None = None,
) -> QuantumCircuit:
    """Build an UNBOUND feature-map circuit (a template with symbolic parameters).

    Args:
        name: One of "zz_feature_map", "z_feature_map", "pauli_feature_map".
        n_qubits: Number of qubits = number of classical features encoded
            (one feature per qubit, via angle encoding -- see
            docs/QUANTUM_PIPELINE.md Section on encoding).
        reps: Number of times the encoding (+ entangling, where applicable)
            layer is repeated.
        entanglement: "linear", "circular", or "full" -- ignored in effect
            by z_feature_map (which has no multi-qubit terms) but accepted
            uniformly since qiskit's function-based API takes the argument
            regardless.
        paulis: Only used when name == "pauli_feature_map"; the Pauli
            strings defining which terms appear (e.g. ("Z", "ZZ")).

    Returns:
        A QuantumCircuit with n_qubits qubits and n_qubits free Parameters
        (named x[0]..x[n_qubits-1]), not yet bound to any data.

    Raises:
        QuantumConfigError: On an unknown feature map name or entanglement
            pattern (checked here, in addition to QuantumConfig's own
            validation, so this function is safe to call directly).
    """
    if name not in ALL_FEATURE_MAPS:
        raise QuantumConfigError(f"Unknown feature_map_name: {name!r} (expected one of {sorted(ALL_FEATURE_MAPS)}).")
    if entanglement not in ALL_ENTANGLEMENT:
        raise QuantumConfigError(f"Unknown entanglement: {entanglement!r} (expected one of {sorted(ALL_ENTANGLEMENT)}).")
    if n_qubits < 1:
        raise QuantumConfigError(f"n_qubits must be >= 1; got {n_qubits}.")

    if name == "zz_feature_map":
        return zz_feature_map(feature_dimension=n_qubits, reps=reps, entanglement=entanglement)
    if name == "z_feature_map":
        return z_feature_map(feature_dimension=n_qubits, reps=reps, entanglement=entanglement)
    if name == "pauli_feature_map":
        pauli_list = list(paulis) if paulis else ["Z", "ZZ"]
        return pauli_feature_map(
            feature_dimension=n_qubits, reps=reps, entanglement=entanglement, paulis=pauli_list
        )
    raise QuantumConfigError(f"Unhandled feature_map_name: {name!r}.")  # pragma: no cover - guarded above


@dataclass(frozen=True)
class FeatureMapReport:
    """A snapshot of a built feature map's structural properties -- the
    "feature-map type / qubits / reps / depth" documentation Phase 4
    requires, computed from the circuit itself rather than asserted.
    """

    name: str
    n_qubits: int
    reps: int
    entanglement: str
    paulis: tuple[str, ...] | None
    num_parameters: int
    depth_logical: int
    gate_counts: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "reps": self.reps,
            "entanglement": self.entanglement,
            "paulis": list(self.paulis) if self.paulis else None,
            "num_parameters": self.num_parameters,
            "depth_logical": self.depth_logical,
            "gate_counts": self.gate_counts,
        }


def describe_feature_map(
    circuit: QuantumCircuit,
    *,
    name: str,
    reps: int,
    entanglement: str,
    paulis: tuple[str, ...] | None = None,
) -> FeatureMapReport:
    """Compute a FeatureMapReport by inspecting an already-built circuit.

    Decomposes the circuit before measuring depth/gate counts, since the
    feature-map functions return a circuit built from a single high-level
    instruction (not the underlying gate sequence) until decomposed --
    decomposing is what makes "circuit depth" mean the same thing a
    transpiler-facing report (a later phase) would also report.
    """
    decomposed = circuit.decompose()
    return FeatureMapReport(
        name=name,
        n_qubits=circuit.num_qubits,
        reps=reps,
        entanglement=entanglement,
        paulis=tuple(paulis) if paulis else None,
        num_parameters=circuit.num_parameters,
        depth_logical=decomposed.depth(),
        gate_counts=dict(decomposed.count_ops()),
    )
