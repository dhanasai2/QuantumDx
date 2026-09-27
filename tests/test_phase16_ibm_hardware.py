"""Tests for src.large_dataset.phase16_ibm_hardware and the IBMHardwareBackend
completion in src.quantum.backends.

CRITICAL: these tests must NEVER submit a real IBM Quantum job. Every test
either uses StatevectorBackend's seeded mock `run_circuit`, synthetic
data, or monkeypatches the credential/connection path so no network call
can occur. Only the explicit `--hardware` CLI flag (never exercised here)
contacts IBM Quantum.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
from qiskit.circuit import QuantumCircuit

from src.large_dataset import phase16_ibm_hardware as p16
from src.quantum.backends import (
    HardwareExecutionRecord,
    IBMCredentialsError,
    IBMHardwareBackend,
    IBMHardwareConfig,
    QuantumBackend,
    StatevectorBackend,
)
from src.quantum.config import QuantumConfigError


# --------------------------------------------------------------------------
# 1. Backend abstraction compatibility (Protocol conformance)
# --------------------------------------------------------------------------


def test_ibm_hardware_backend_satisfies_quantum_backend_protocol_shape() -> None:
    hw = IBMHardwareBackend()
    assert hasattr(hw, "name")
    assert callable(hw.compute_statevector)
    assert callable(hw.capabilities)
    assert callable(hw.run_circuit)


def test_statevector_backend_also_implements_run_circuit() -> None:
    sb = StatevectorBackend()
    assert callable(sb.run_circuit)


def test_capabilities_report_credentials_configured_flag_without_network_call(monkeypatch) -> None:
    # .env in this repo now holds a real token -- must prevent _connect's own
    # load_dotenv() call from silently restoring it after delenv, or this
    # test would not actually exercise the "no credentials" path.
    monkeypatch.setattr("src.quantum.backends.load_dotenv", lambda: False)
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    hw = IBMHardwareBackend()
    caps = hw.capabilities()
    assert caps["credentials_configured"] is False
    assert caps["implemented"] is True  # the CODE is implemented, independent of credential availability


# --------------------------------------------------------------------------
# 2. Circuit construction: qubit count, parameter binding, bounded outputs
# --------------------------------------------------------------------------


def test_measurement_circuits_have_correct_qubit_count_and_are_fully_bound() -> None:
    theta = np.array([0.1, -0.2, 0.3, -0.4])
    X = np.array([[0.1, 0.2, 0.3, 0.4], [1.0, 1.5, 2.0, 2.5]])
    circuits = p16.build_measurement_circuits(X, theta)
    assert len(circuits) == 2
    for c in circuits:
        assert c.num_qubits == 4
        assert len(c.parameters) == 0  # fully bound -- nothing left to assign
        assert "measure" in [instr.operation.name for instr in c.data]


def test_ideal_expectation_values_are_bounded() -> None:
    theta = np.array([0.1, -0.2, 0.3, -0.4])
    X = np.array([[0.1, 0.2, 0.3, 0.4], [3.0, 1.5, 0.0, 2.9]])
    ideal = p16.compute_ideal_expectations(X, theta)
    assert ideal.shape == (2, 4)
    assert np.all(ideal >= -1.0 - 1e-9) and np.all(ideal <= 1.0 + 1e-9)


def test_counts_to_expectation_values_bounded_and_correct() -> None:
    # 2 qubits: bitstring "00" -> both +1, "11" -> both -1.
    counts = {"00": 600, "11": 400}
    exp = p16.counts_to_expectation_values(counts, n_qubits=2)
    assert exp.shape == (2,)
    assert np.all(exp >= -1.0) and np.all(exp <= 1.0)
    assert exp[0] == pytest.approx((600 - 400) / 1000)
    assert exp[1] == pytest.approx((600 - 400) / 1000)


def test_counts_to_expectation_values_all_one_outcome_is_exactly_bounded() -> None:
    exp = p16.counts_to_expectation_values({"1111": 1024}, n_qubits=4)
    assert np.all(exp == -1.0)


# --------------------------------------------------------------------------
# 3. Mock/fake backend behavior (StatevectorBackend.run_circuit)
# --------------------------------------------------------------------------


def test_statevector_backend_run_circuit_produces_valid_counts() -> None:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.7, 2)
    qc.measure_all()
    sb = StatevectorBackend()
    counts = sb.run_circuit(qc, shots=2000, seed=1)
    assert sum(counts.values()) == 2000
    for bitstring in counts:
        assert len(bitstring) == 3
        assert set(bitstring) <= {"0", "1"}


def test_statevector_backend_run_circuit_deterministic_for_fixed_seed() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    sb = StatevectorBackend()
    c1 = sb.run_circuit(qc, shots=500, seed=7)
    c2 = sb.run_circuit(qc, shots=500, seed=7)
    assert c1 == c2


# --------------------------------------------------------------------------
# 4. Credential absence handled safely (no network call attempted)
# --------------------------------------------------------------------------


def test_connect_raises_credentials_error_without_token(monkeypatch) -> None:
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    monkeypatch.setattr("src.quantum.backends.load_dotenv", lambda: None, raising=False)
    hw = IBMHardwareBackend()
    with pytest.raises(IBMCredentialsError, match="IBM_QUANTUM_TOKEN"):
        hw._connect()


def test_connect_never_calls_qiskit_runtime_service_without_token(monkeypatch) -> None:
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)

    def _boom(*a, **kw):
        raise AssertionError("QiskitRuntimeService must not be constructed without a token")

    import qiskit_ibm_runtime
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", _boom)
    hw = IBMHardwareBackend()
    with pytest.raises(IBMCredentialsError):
        hw._connect()


def test_execute_with_metadata_returns_blocked_record_when_run_hardware_demo_catches_it(monkeypatch) -> None:
    """run_hardware_demo's own try/except around IBMCredentialsError -- the
    path a real user hits if they run --hardware with no .env configured."""
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    monkeypatch.setattr("src.quantum.backends.load_dotenv", lambda: None, raising=False)
    summary = p16.run_hardware_demo(hardware=True, n_samples=4, shots=128, make_plots=False)
    assert summary["decision"] == "B_IMPLEMENTED_BUT_BLOCKED"
    assert summary["record"]["status"] == "BLOCKED_NO_CREDENTIALS"
    assert summary["record"]["job_id"] is None
    assert summary["record"]["hardware_execution"] is False


# --------------------------------------------------------------------------
# 5. Hardware configuration validation / resource guards
# --------------------------------------------------------------------------


def test_ibm_hardware_config_rejects_invalid_values() -> None:
    with pytest.raises(QuantumConfigError):
        IBMHardwareConfig(max_qubits=0)
    with pytest.raises(QuantumConfigError):
        IBMHardwareConfig(max_shots=0)
    with pytest.raises(QuantumConfigError):
        IBMHardwareConfig(max_circuits=0)


def test_execute_with_metadata_rejects_oversized_batch() -> None:
    hw = IBMHardwareBackend(IBMHardwareConfig(max_circuits=2))
    circuits = [QuantumCircuit(2) for _ in range(3)]
    with pytest.raises(QuantumConfigError, match="max_circuits"):
        hw.execute_with_metadata(circuits, shots=100)


def test_execute_with_metadata_rejects_excess_shots() -> None:
    hw = IBMHardwareBackend(IBMHardwareConfig(max_shots=100))
    with pytest.raises(QuantumConfigError, match="max_shots"):
        hw.execute_with_metadata([QuantumCircuit(2)], shots=200)


def test_execute_with_metadata_rejects_excess_qubits() -> None:
    hw = IBMHardwareBackend(IBMHardwareConfig(max_qubits=2))
    with pytest.raises(QuantumConfigError, match="max_qubits"):
        hw.execute_with_metadata([QuantumCircuit(3)], shots=100)


def test_run_hardware_demo_rejects_oversized_sample_request() -> None:
    with pytest.raises(QuantumConfigError, match="MAX_SAMPLES_HARDWARE"):
        p16.build_demo_dataset(n_samples=p16.MAX_SAMPLES_HARDWARE + 1)


def test_run_hardware_demo_rejects_excess_shots() -> None:
    with pytest.raises(QuantumConfigError, match="MAX_SHOTS_HARDWARE"):
        p16.run_hardware_demo(hardware=False, n_samples=4, shots=p16.MAX_SHOTS_HARDWARE + 1, make_plots=False)


# --------------------------------------------------------------------------
# 6. No credentials stored in the repository
# --------------------------------------------------------------------------


def test_ibm_hardware_config_dataclass_has_no_token_field() -> None:
    from dataclasses import fields
    field_names = {f.name for f in fields(IBMHardwareConfig)}
    assert not any("token" in n.lower() or "secret" in n.lower() or "password" in n.lower() for n in field_names)


def test_hardware_execution_record_never_carries_a_raw_token() -> None:
    from dataclasses import fields
    field_names = {f.name for f in fields(HardwareExecutionRecord)}
    assert not any("token" in n.lower() for n in field_names)


def test_env_file_is_gitignored() -> None:
    from src.data.inspect_dataset import find_project_root

    gitignore = (find_project_root() / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore


# --------------------------------------------------------------------------
# 7. Deterministic preprocessing / no test-set usage / demo dataset
# --------------------------------------------------------------------------


def test_build_demo_dataset_is_deterministic_for_fixed_seed() -> None:
    _, demo1 = p16.build_demo_dataset(n_samples=8, seed=42)
    _, demo2 = p16.build_demo_dataset(n_samples=8, seed=42)
    assert demo1["id"].tolist() == demo2["id"].tolist()


def test_build_demo_dataset_never_overlaps_fixed_test_set() -> None:
    split, demo = p16.build_demo_dataset(n_samples=8, seed=42)
    test_ids = set(split.test_set_quantum["id"].tolist())
    demo_ids = set(demo["id"].tolist())
    assert not (demo_ids & test_ids)


def test_build_demo_dataset_includes_both_outcome_classes() -> None:
    from src.large_dataset.schema import TARGET_COLUMN

    _, demo = p16.build_demo_dataset(n_samples=8, seed=42)
    assert set(demo[TARGET_COLUMN].unique()) == {0, 1}


# --------------------------------------------------------------------------
# 8. Job metadata schema
# --------------------------------------------------------------------------


def test_hardware_execution_record_to_dict_schema() -> None:
    record = HardwareExecutionRecord(
        backend_name="ibm_test", job_id="abc123", status="COMPLETED", hardware_execution=True,
        num_qubits=4, shots=1024, n_circuits=2, circuit_depths=[3, 3], gate_counts=[{"rz": 4}, {"rz": 4}],
        counts_list=[{"0000": 1024}, {"1111": 1024}], submitted_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:01:00Z",
    )
    d = record.to_dict()
    for key in ("backend_name", "job_id", "status", "hardware_execution", "num_qubits", "shots",
                "n_circuits", "circuit_depths", "gate_counts", "counts_list", "submitted_at", "completed_at", "error"):
        assert key in d


# --------------------------------------------------------------------------
# 9. Ideal vs hardware comparison calculations
# --------------------------------------------------------------------------


def test_compare_ideal_vs_hardware_correctness() -> None:
    ideal = np.array([[1.0, 0.0], [0.5, -0.5]])
    hardware = np.array([[0.9, 0.1], [0.4, -0.6]])
    result = p16.compare_ideal_vs_hardware(ideal, hardware)
    expected_mad = np.mean(np.abs(hardware - ideal))
    assert result["mean_absolute_deviation"] == pytest.approx(expected_mad)
    assert result["max_absolute_deviation"] == pytest.approx(0.1)
    assert len(result["per_qubit_mean_absolute_deviation"]) == 2
    assert len(result["per_sample_mean_absolute_deviation"]) == 2


def test_compare_ideal_vs_hardware_zero_deviation_when_identical() -> None:
    ideal = np.array([[0.3, -0.2, 0.7]])
    result = p16.compare_ideal_vs_hardware(ideal, ideal.copy())
    assert result["mean_absolute_deviation"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 10. End-to-end mock run (small, bounded -- the only "integration" test)
# --------------------------------------------------------------------------


def test_run_hardware_demo_mock_end_to_end_never_touches_ibm(monkeypatch) -> None:
    def _boom(*a, **kw):
        raise AssertionError("IBMHardwareBackend.execute_with_metadata must not be called when hardware=False")

    monkeypatch.setattr(IBMHardwareBackend, "execute_with_metadata", _boom)
    summary = p16.run_hardware_demo(hardware=False, n_samples=4, shots=256, make_plots=False)
    assert summary["decision"] == "A_MOCK_DEMONSTRATED_NO_REAL_HARDWARE"
    assert summary["record"]["hardware_execution"] is False
    assert summary["record"]["job_id"] is None
    assert summary["record"]["status"] == "MOCK_COMPLETED"


def test_run_hardware_demo_mock_persists_all_required_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(p16, "RESULTS_ROOT", p16.RESULTS_ROOT)  # no-op, uses the real results dir
    p16.run_hardware_demo(hardware=False, n_samples=4, shots=256, make_plots=False)
    for fname in ("demo_samples.csv", "circuit.qasm", "measurement_counts.json", "ideal_results.json",
                  "ideal_vs_hardware.json", "job_metadata.json", "backend_metadata.json", "phase16_summary.json",
                  "observables.csv", "hardware_execution.json"):
        assert (p16.RESULTS_ROOT / fname).is_file(), f"missing {fname}"


def test_frozen_theta_matches_phase10_and_is_not_retrained() -> None:
    theta = p16.load_frozen_theta()
    with open(p16.PHASE10_EXPLAINABILITY_PATH, encoding="utf-8") as fh:
        expl = json.load(fh)
    assert theta.tolist() == pytest.approx(expl["selected_quantum_weights_theta"])


# --------------------------------------------------------------------------
# 11. Real artifacts, if present
# --------------------------------------------------------------------------


def test_persisted_summary_decision_is_one_of_the_defined_outcomes_if_present() -> None:
    path = p16.RESULTS_ROOT / "phase16_summary.json"
    if not path.is_file():
        pytest.skip("Phase 16 has not been run yet")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["decision"] in (
        "A_SUCCESSFULLY_DEMONSTRATED", "A_MOCK_DEMONSTRATED_NO_REAL_HARDWARE",
        "B_IMPLEMENTED_BUT_BLOCKED", "C_FAILED",
    )


def test_prior_phase_artifacts_untouched_if_present() -> None:
    from src.data.inspect_dataset import find_project_root

    root = find_project_root() / "results" / "large_dataset"
    for path, expected_len in [
        (root / "phase9_classical_benchmark" / "predictions.csv", 200),
        (root / "phase11_full_scale" / "predictions.csv", 200),
        (root / "phase13_hybrid_system" / "predictions.csv", 200),
        (root / "phase15_quantum_classical_stacking" / "oof_predictions.csv", 66641),
    ]:
        if path.is_file():
            assert len(pd.read_csv(path)) == expected_len
