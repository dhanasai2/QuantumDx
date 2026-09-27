"""Quantum backend abstraction (Phase 4, Section 8).

    QuantumBackend
          |
          +-- StatevectorBackend    (IMPLEMENTED -- Phase 4's primary backend)
          |
          +-- NoisySimulatorBackend (interface only -- Phase 5 decision, still not implemented)
          |
          +-- IBMHardwareBackend    (IMPLEMENTED -- Phase 16, real IBM Quantum execution)

The rest of the quantum pipeline (kernel.py, sanity_experiment.py) talks
only to the QuantumBackend Protocol, never to a concrete backend class
directly -- so switching backends is a configuration change
(QuantumConfig.backend_name), not a code change, exactly as required.

DEPENDENCY NOTE (deliberate, documented choice): StatevectorBackend uses
qiskit.quantum_info.Statevector for EXACT simulation. It does not depend on
qiskit-aer. This was a considered decision, not an oversight: installing
qiskit-aer (or qiskit-machine-learning) in this environment would have
pulled in numpy>=2.0, a major version upgrade from the numpy==1.26.4 that
Phases 1-3's tested pipeline (pandas, scikit-learn, xgboost) currently
runs on. Statevector-based exact simulation is fully sufficient for a
4-6 qubit feature map (state vectors of size 2^4..2^6 are trivial to hold
and multiply) and introduces zero new dependencies, hence zero risk to the
already-tested classical pipeline. See docs/QUANTUM_PIPELINE.md for the
full reasoning; this can be revisited in Phase 5 if shot-based sampling or
a device noise model becomes a scoped requirement.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Protocol

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.quantum.config import QuantumConfigError

try:
    from dotenv import load_dotenv  # module-level, so tests can monkeypatch this exact name
except ImportError:  # pragma: no cover -- python-dotenv is an existing project dependency
    def load_dotenv() -> bool:  # type: ignore[misc]
        return False


class QuantumBackend(Protocol):
    """The interface every backend (implemented or not-yet-implemented) satisfies."""

    name: str

    def compute_statevector(self, bound_circuit: QuantumCircuit) -> Statevector:
        """Return the exact or estimated statevector for an already-parameter-bound circuit."""
        ...

    def capabilities(self) -> dict:
        """Return a small dict describing what this backend can do (exact vs.
        sampled, implemented vs. not, shots, etc.) -- used for the
        hardware-readiness-style reporting a later phase will build on.
        """
        ...

    def run_circuit(self, bound_circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        """Return shot-based measurement COUNTS (bitstring -> count) for an
        already-parameter-bound circuit that ends in measurement.

        Added in Phase 16 (IBM hardware execution): unlike
        `compute_statevector`, this is something a REAL or noisy backend can
        actually satisfy (a physical QPU never returns an exact statevector,
        only measurement outcomes). Additive to the Protocol -- every
        backend implemented before Phase 16 (StatevectorBackend) also gets
        an implementation below, so nothing that already depends on
        `compute_statevector` changes behavior.
        """
        ...


class IBMCredentialsError(QuantumConfigError):
    """Raised when real IBM Quantum execution is requested but the required
    credentials (IBM_QUANTUM_TOKEN, at minimum) are not configured. Distinct
    from QuantumConfigError's other uses so callers can catch this specific,
    expected, non-fatal condition (see Phase 16's "blocked, not fabricated"
    handling) separately from a genuine configuration mistake.
    """


@dataclass(frozen=True)
class HardwareExecutionRecord:
    """Everything Phase 16 needs to report about ONE real (or mocked)
    hardware execution -- a superset of what the minimal `run_circuit`
    Protocol method returns, since only actual hardware/noisy backends have
    concepts like a job id or a physical backend name. Never constructed
    with fabricated values -- see IBMHardwareBackend.execute_with_metadata.
    """

    backend_name: str
    job_id: str | None
    status: str
    hardware_execution: bool
    num_qubits: int
    shots: int
    n_circuits: int
    circuit_depths: list[int]
    gate_counts: list[dict[str, int]]
    counts_list: list[dict[str, int]]
    submitted_at: str | None
    completed_at: str | None
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "backend_name": self.backend_name, "job_id": self.job_id, "status": self.status,
            "hardware_execution": self.hardware_execution, "num_qubits": self.num_qubits,
            "shots": self.shots, "n_circuits": self.n_circuits, "circuit_depths": self.circuit_depths,
            "gate_counts": self.gate_counts, "counts_list": self.counts_list,
            "submitted_at": self.submitted_at, "completed_at": self.completed_at,
            "error": self.error, "extra": self.extra,
        }


class StatevectorBackend:
    """Exact statevector simulation via qiskit.quantum_info.Statevector.

    Deterministic: the same bound circuit always produces the same
    statevector (no sampling noise, no RNG involved at all).
    """

    name = "statevector"

    def compute_statevector(self, bound_circuit: QuantumCircuit) -> Statevector:
        if bound_circuit.num_parameters != 0:
            raise QuantumConfigError(
                f"compute_statevector requires a fully parameter-bound circuit; "
                f"{bound_circuit.num_parameters} free parameter(s) remain."
            )
        return Statevector.from_instruction(bound_circuit)

    def capabilities(self) -> dict:
        return {
            "name": self.name,
            "implemented": True,
            "exact": True,
            "shots": None,
            "noise_model": None,
            "requires": "core qiskit only (no qiskit-aer)",
        }

    def run_circuit(self, bound_circuit: QuantumCircuit, shots: int, *, seed: int = 42) -> dict[str, int]:
        """A deterministic, seeded MOCK of shot-based execution: computes the
        EXACT probability distribution from the statevector (measurements
        stripped, since Statevector.from_instruction cannot evolve a
        non-unitary measure instruction), then draws `shots` samples from it
        with a seeded RNG. Used by Phase 16's tests as a stand-in for real
        hardware -- never used to represent an actual physical execution.
        """
        measurement_free = bound_circuit.remove_final_measurements(inplace=False)
        sv = self.compute_statevector(measurement_free)
        probs = sv.probabilities_dict()
        n_qubits = bound_circuit.num_qubits
        bitstrings = list(probs.keys())
        probabilities = [probs[b] for b in bitstrings]
        import numpy as np

        rng = np.random.RandomState(seed)
        draws = rng.multinomial(shots, probabilities)
        return {b.zfill(n_qubits): int(c) for b, c in zip(bitstrings, draws) if c > 0}


class NoisySimulatorBackend:
    """Interface placeholder for a device-noise-model simulator.

    NOT IMPLEMENTED in Phase 4 (by explicit instruction: "establish the
    interface required for it," not the backend itself). Implementing this
    would require qiskit-aer, which was deliberately not installed in this
    environment during Phase 4 -- see the module docstring. A future phase
    can implement this class without changing anything else in
    src/quantum, because kernel.py and sanity_experiment.py depend only on
    the QuantumBackend Protocol.
    """

    name = "noisy_simulator"

    def compute_statevector(self, bound_circuit: QuantumCircuit) -> Statevector:
        raise NotImplementedError(
            "NoisySimulatorBackend is an interface placeholder (Phase 4). "
            "Implementing it requires qiskit-aer, deliberately not installed "
            "yet -- see docs/QUANTUM_PIPELINE.md."
        )

    def capabilities(self) -> dict:
        return {
            "name": self.name,
            "implemented": False,
            "exact": False,
            "planned_phase": 5,
            "blocked_on": "qiskit-aer (not installed; see docs/QUANTUM_PIPELINE.md)",
        }


@dataclass(frozen=True)
class IBMHardwareConfig:
    """Resource guards + selection knobs for real IBM Quantum execution
    (Phase 16). Never holds a credential value itself -- those are read
    directly from the environment at connection time and never persisted.
    """

    max_qubits: int = 6
    max_shots: int = 4096
    max_circuits: int = 32
    backend_override: str | None = None  # explicit backend name; None = auto-select least-busy
    optimization_level: int = 1

    def __post_init__(self) -> None:
        if self.max_qubits < 1:
            raise QuantumConfigError(f"max_qubits must be >= 1; got {self.max_qubits}.")
        if self.max_shots < 1:
            raise QuantumConfigError(f"max_shots must be >= 1; got {self.max_shots}.")
        if self.max_circuits < 1:
            raise QuantumConfigError(f"max_circuits must be >= 1; got {self.max_circuits}.")


class IBMHardwareBackend:
    """Real IBM Quantum hardware execution (Phase 16), conforming to the
    QuantumBackend Protocol.

    `compute_statevector` is NOT satisfiable by a physical QPU (no exact
    statevector exists on real hardware) -- it raises a clear, specific
    error directing callers to `run_circuit` / `execute_with_metadata`
    instead, rather than silently returning nonsense.

    Credentials (IBM_QUANTUM_TOKEN, IBM_QUANTUM_CHANNEL, IBM_QUANTUM_INSTANCE)
    are read from the environment ONLY inside `_connect()`, at the moment a
    real network call is about to happen -- never hardcoded, never logged,
    never written to any results/ artifact. See .env.example.
    """

    name = "ibm_hardware"

    def __init__(self, config: IBMHardwareConfig | None = None) -> None:
        self.config = config or IBMHardwareConfig()

    def compute_statevector(self, bound_circuit: QuantumCircuit) -> Statevector:
        raise NotImplementedError(
            "IBMHardwareBackend cannot produce an exact statevector -- real hardware only "
            "returns shot-based measurement counts. Use run_circuit(...) or "
            "execute_with_metadata(...) instead, and compare against StatevectorBackend for "
            "the 'ideal' reference (see src.large_dataset.phase16_ibm_hardware)."
        )

    def capabilities(self) -> dict:
        try:
            import qiskit_ibm_runtime  # noqa: F401
            sdk_available = True
        except ImportError:
            sdk_available = False
        return {
            "name": self.name,
            "implemented": True,
            "exact": False,
            "shots": "configurable (see IBMHardwareConfig.max_shots)",
            "noise_model": "real device noise (whatever the selected physical backend exhibits)",
            "sdk_available": sdk_available,
            "credentials_configured": self._credentials_configured(),
            "max_qubits": self.config.max_qubits,
            "max_shots": self.config.max_shots,
            "max_circuits": self.config.max_circuits,
            "requires": "qiskit-ibm-runtime + IBM_QUANTUM_TOKEN (see .env.example)",
        }

    @staticmethod
    def _credentials_configured() -> bool:
        """Checks env-var PRESENCE only -- never validates the token against
        IBM's servers (that would itself be a network call). Loads a local
        .env file first if python-dotenv is available, mirroring how the
        rest of the project's scripts pick up .env (never committed; see
        .gitignore)."""
        load_dotenv()
        return bool(os.environ.get("IBM_QUANTUM_TOKEN"))

    def _connect(self):
        """Returns an authenticated QiskitRuntimeService, or raises
        IBMCredentialsError with an exact, actionable reason -- BEFORE any
        network call is attempted. This is the only place credentials are
        read."""
        load_dotenv()

        token = os.environ.get("IBM_QUANTUM_TOKEN")
        if not token:
            raise IBMCredentialsError(
                "IBM_QUANTUM_TOKEN is not set. Copy .env.example to .env and fill in a real "
                "IBM Quantum API token (https://quantum.cloud.ibm.com/ -> Account settings). "
                "Real hardware execution cannot proceed without it."
            )
        channel = os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform")
        instance = os.environ.get("IBM_QUANTUM_INSTANCE") or None

        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError as exc:
            raise IBMCredentialsError(
                "qiskit-ibm-runtime is not installed. Run `pip install qiskit-ibm-runtime`."
            ) from exc

        try:
            return QiskitRuntimeService(channel=channel, token=token, instance=instance)
        except Exception as exc:  # noqa: BLE001 -- surface the provider's own error verbatim
            raise IBMCredentialsError(f"IBM Quantum authentication failed: {exc}") from exc

    def _select_backend(self, service, min_qubits: int):
        """Never hard-codes a specific backend name -- auto-selects the
        least-busy OPERATIONAL, non-simulator backend with enough qubits,
        unless an explicit override is configured."""
        if self.config.backend_override:
            backend = service.backend(self.config.backend_override)
            if backend.num_qubits < min_qubits:
                raise QuantumConfigError(
                    f"Configured backend_override={self.config.backend_override!r} has "
                    f"{backend.num_qubits} qubits, fewer than the required {min_qubits}."
                )
            return backend
        try:
            return service.least_busy(
                min_num_qubits=min_qubits, operational=True,
                filters=lambda b: not b.configuration().simulator,
            )
        except Exception as exc:  # noqa: BLE001
            raise QuantumConfigError(
                f"No operational IBM Quantum backend with >= {min_qubits} qubits is currently "
                f"available: {exc}"
            ) from exc

    def execute_with_metadata(self, circuits: list[QuantumCircuit], shots: int, *,
                               timeout_seconds: float = 420.0) -> HardwareExecutionRecord:
        """The real, rich execution path -- validates resource guards,
        connects, selects a backend, transpiles, submits ONE Sampler job for
        the whole batch of circuits, and waits UP TO `timeout_seconds` for
        the result. The real job id is captured immediately after
        submission (before waiting), so even a timeout still returns a
        record with a genuine, retrievable job id and status
        "SUBMITTED_PENDING" -- never a fabricated result. Never fabricates a
        job id or result under any code path.
        """
        if len(circuits) > self.config.max_circuits:
            raise QuantumConfigError(
                f"{len(circuits)} circuits exceeds max_circuits={self.config.max_circuits} -- "
                f"refusing to submit an uncontrolled batch to real hardware."
            )
        if shots > self.config.max_shots:
            raise QuantumConfigError(f"shots={shots} exceeds max_shots={self.config.max_shots}.")
        n_qubits = circuits[0].num_qubits if circuits else 0
        if n_qubits > self.config.max_qubits:
            raise QuantumConfigError(f"num_qubits={n_qubits} exceeds max_qubits={self.config.max_qubits}.")
        for c in circuits:
            if c.num_qubits != n_qubits:
                raise QuantumConfigError("All circuits in one batch must share the same qubit count.")

        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2

        service = self._connect()
        backend = self._select_backend(service, n_qubits)
        pm = generate_preset_pass_manager(optimization_level=self.config.optimization_level, backend=backend)
        transpiled = [pm.run(c) for c in circuits]
        depths = [tc.depth() for tc in transpiled]
        gate_counts = [dict(tc.count_ops()) for tc in transpiled]

        sampler = SamplerV2(mode=backend)
        submitted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        job = sampler.run(transpiled, shots=shots)
        job_id = job.job_id()  # captured immediately -- real, retrievable even if we time out below

        from qiskit_ibm_runtime.exceptions import RuntimeJobTimeoutError

        try:
            result = job.result(timeout=timeout_seconds)
            completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            counts_list = [_extract_counts(result[i]) for i in range(len(transpiled))]
            status = "COMPLETED"
            error = None
        except RuntimeJobTimeoutError:
            completed_at = None
            counts_list = []
            status = "SUBMITTED_PENDING"
            error = (f"Job {job_id} did not complete within {timeout_seconds:.0f}s (queue/runtime still in "
                     f"progress on backend {backend.name}). It was NOT cancelled -- check its status later "
                     f"with this same job_id via QiskitRuntimeService.job(job_id).")
        except Exception as exc:  # noqa: BLE001 -- job submitted; report real (not fabricated) failure
            completed_at = None
            counts_list = []
            status = "FAILED"
            error = str(exc)

        return HardwareExecutionRecord(
            backend_name=backend.name, job_id=job_id, status=status, hardware_execution=(status == "COMPLETED"),
            num_qubits=n_qubits, shots=shots, n_circuits=len(circuits), circuit_depths=depths,
            gate_counts=gate_counts, counts_list=counts_list, submitted_at=submitted_at,
            completed_at=completed_at, error=error,
        )

    def run_circuit(self, bound_circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        record = self.execute_with_metadata([bound_circuit], shots)
        if record.status != "COMPLETED":
            raise QuantumConfigError(f"IBM hardware execution did not complete: {record.error}")
        return record.counts_list[0]


def check_job_status(job_id: str) -> dict:
    """Follow-up utility: look up a previously submitted real IBM job by its
    REAL job_id (e.g. one persisted with status SUBMITTED_PENDING) without
    resubmitting anything. Connects using the same credential convention as
    IBMHardwareBackend._connect()."""
    hw = IBMHardwareBackend()
    service = hw._connect()
    job = service.job(job_id)
    status = job.status()
    out = {"job_id": job_id, "status": status}
    if status == "DONE":
        try:
            result = job.result()
            out["counts_list"] = [_extract_counts(result[i]) for i in range(len(result))]
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)
    return out


def _extract_counts(pub_result) -> dict[str, int]:
    """SamplerV2 pub results expose per-classical-register data containers
    (e.g. `.data.meas` for a circuit built with `.measure_all()`, which
    names its register "meas"). Defensive against the exact attribute name
    varying by how the circuit was measured."""
    data_bin = pub_result.data
    for reg_name in ("meas", "c", "cr"):
        if hasattr(data_bin, reg_name):
            return dict(getattr(data_bin, reg_name).get_counts())
    # last resort: the first (and presumably only) classical register present
    reg_names = [f for f in dir(data_bin) if not f.startswith("_")]
    for reg_name in reg_names:
        attr = getattr(data_bin, reg_name)
        if hasattr(attr, "get_counts"):
            return dict(attr.get_counts())
    raise QuantumConfigError("Could not locate a classical register with get_counts() on the Sampler result.")


_BACKEND_REGISTRY: dict[str, type] = {
    "statevector": StatevectorBackend,
    "noisy_simulator": NoisySimulatorBackend,
    "ibm_hardware": IBMHardwareBackend,
}


def get_backend(name: str) -> QuantumBackend:
    """Resolve a backend name (from QuantumConfig.backend_name) to an instance.

    This is the ONLY place backend selection happens -- everything
    downstream (kernel.py, sanity_experiment.py) receives a QuantumBackend
    instance and never branches on the name itself, which is what makes
    backend selection a configuration concern rather than a code-path
    concern.
    """
    if name not in _BACKEND_REGISTRY:
        raise QuantumConfigError(
            f"Unknown backend_name: {name!r} (expected one of {sorted(_BACKEND_REGISTRY)})."
        )
    return _BACKEND_REGISTRY[name]()
