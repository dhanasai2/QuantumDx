"""Phase 16: IBM Quantum HARDWARE EXECUTION & hybrid-system validation
(see docs/PHASE_16_IBM_HARDWARE.md).

This is NOT another accuracy experiment. Phases 9-15 already established,
exhaustively, that no tested quantum component improves on the classical
XGBoost baseline (Phase 11) -- that finding is untouched and unrevisited
here. Phase 16 answers a completely different, narrower, ENGINEERING
question:

    Can the project's already-developed hybrid quantum-classical
    architecture's quantum component actually execute on REAL IBM Quantum
    hardware -- real circuit, real transpilation, real job, real
    measurement counts, real job id -- not just a statevector simulator?

WHICH CIRCUIT WAS SELECTED, AND WHY: Phase 10's `HybridQuantumFeatureLayer`
circuit (src.quantum.hybrid_quantum_features) -- the SIMPLEST, most
hardware-friendly circuit in the whole project: 4 qubits, 1 layer, RY
angle-encoding + RY trainable rotation + a 3-gate linear CNOT chain (11
total gates before transpilation). Phase 12/13/15's circuits are richer
(6-16 trainable parameters, RZZ entanglement, 1-2 layers) but that
richness means MORE gates -> deeper transpiled circuits -> more
decoherence on a real, noisy device for a workload whose entire point is
to demonstrate execution, not chase expressiveness. Phase 10's already-
selected, already-frozen theta (from its own persisted
`explainability.json`) is reused UNMODIFIED -- no new optimization of any
kind happens here, let alone on hardware (Section 9 of the governing spec
is explicit: no COBYLA, no hyperparameter search, no training on IBM
hardware).

WHAT RUNS ON HARDWARE: ONLY the fixed, already-trained quantum circuit,
for a small (default 16), deterministic sample of already-preprocessed
PCA-4 feature vectors, submitted as ONE batched Sampler job. XGBoost
(Phase 11's model) is not touched, not retrained, and not executed on
quantum hardware -- it remains the project's primary, validated predictor.

SAFETY: the default invocation of this module (no --hardware flag) NEVER
contacts IBM Quantum -- it runs the identical pipeline against
`StatevectorBackend`'s seeded mock `run_circuit`, useful for development
and for the test suite. Only `python -m
src.large_dataset.phase16_ibm_hardware --hardware` attempts a real
network call, and even then only after every resource guard (qubit count,
shot count, circuit count) has been checked.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import qasm2

from src.data.inspect_dataset import find_project_root
from src.large_dataset.corrected_comparison import comparison_set_fingerprint
from src.large_dataset.phase11_full_scale import build_full_scale_split
from src.large_dataset.pipeline import process_stage
from src.large_dataset.schema import TARGET_COLUMN, get_cardio_feature_groups
from src.preprocessing.config import PreprocessingConfig
from src.quantum.backends import (
    HardwareExecutionRecord,
    IBMCredentialsError,
    IBMHardwareBackend,
    IBMHardwareConfig,
    StatevectorBackend,
)
from src.quantum.config import QuantumConfigError
from src.quantum.hybrid_quantum_features import HybridConfig, HybridQuantumFeatureLayer, build_hybrid_circuit

EXPECTED_FINGERPRINT = "96eac11a8394b87e"
RESULTS_ROOT = find_project_root() / "results" / "phase16_ibm_hardware"
PHASE10_EXPLAINABILITY_PATH = find_project_root() / "results" / "large_dataset" / "phase10_hybrid_qml" / "explainability.json"

PRIMARY_SEED = 42
DEFAULT_N_SAMPLES = 16
MAX_SAMPLES_HARDWARE = 32
DEFAULT_SHOTS = 1024
MAX_SHOTS_HARDWARE = 4096
MAX_QUBITS_HARDWARE = 6
DEFAULT_TIMEOUT_SECONDS = 420.0


def load_frozen_theta() -> np.ndarray:
    """Phase 10's own already-selected quantum weights -- NOT re-optimized
    here. Raises if Phase 10 has not been run (this phase does not train
    anything as a fallback; it demonstrates hardware execution of an
    already-validated circuit)."""
    if not PHASE10_EXPLAINABILITY_PATH.is_file():
        raise FileNotFoundError(
            f"Phase 10 explainability artifact not found at {PHASE10_EXPLAINABILITY_PATH}. "
            f"Phase 16 reuses Phase 10's already-trained circuit weights and does not retrain them."
        )
    with open(PHASE10_EXPLAINABILITY_PATH, encoding="utf-8") as fh:
        expl = json.load(fh)
    theta = np.asarray(expl["selected_quantum_weights_theta"], dtype=float)
    expected_n = HybridConfig().n_trainable_params()
    if theta.shape != (expected_n,):
        raise ValueError(f"Frozen theta shape {theta.shape} != expected ({expected_n},).")
    return theta


def build_demo_dataset(n_samples: int = DEFAULT_N_SAMPLES, *, seed: int = PRIMARY_SEED):
    """A small, deterministic, class-balanced sample drawn ONLY from the
    66,641-row TRAINING POOL -- never the fixed 200-row test set. This is a
    hardware-execution demonstration, not a new benchmark: sample selection
    does not look at held-out performance in any way, just a seeded random
    draw stratified by the (already-known, non-held-out) training label so
    the tiny demo shows both outcome classes."""
    if n_samples > MAX_SAMPLES_HARDWARE:
        raise QuantumConfigError(f"n_samples={n_samples} exceeds MAX_SAMPLES_HARDWARE={MAX_SAMPLES_HARDWARE}.")

    split = build_full_scale_split()
    test_ids = set(split.test_set_quantum["id"].tolist())
    fp = comparison_set_fingerprint(split.test_set_quantum["id"].tolist())
    if fp != EXPECTED_FINGERPRINT:
        raise ValueError(f"Fixed test set fingerprint {fp} != expected {EXPECTED_FINGERPRINT} -- refusing to proceed.")

    pool = split.training_pool
    rng = np.random.RandomState(seed)
    y = pool[TARGET_COLUMN].to_numpy()
    n_per_class = n_samples // 2
    pos_idx = rng.choice(np.where(y == 1)[0], size=n_per_class, replace=False)
    neg_idx = rng.choice(np.where(y == 0)[0], size=n_samples - n_per_class, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)
    demo_df = pool.iloc[idx].reset_index(drop=True)

    demo_ids = set(demo_df["id"].tolist())
    if demo_ids & test_ids:
        raise ValueError("Demo sample ids overlap the fixed 200-row test set -- this must never happen.")

    return split, demo_df


def preprocess_demo_samples(split, demo_df: pd.DataFrame, feature_groups, pcfg: PreprocessingConfig) -> np.ndarray:
    """Fits preprocessing/PCA on the FULL training pool (identical
    convention to every prior phase), then transforms the demo rows through
    it. The demo rows are themselves part of the training pool the
    pipeline is fit on -- fine for a forward-pass demonstration (not an
    evaluation claim), and still never touches the fixed test set."""
    processed = process_stage(split.training_pool, demo_df, demo_df, TARGET_COLUMN, feature_groups, pcfg)
    return processed.X_test_quantum  # (n_samples, 4), range-normalized to [0, pi]


def compute_ideal_expectations(X_demo_quantum: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """EXACT expectation values via Phase 10's own, unmodified
    EstimatorQNN(default_precision=0.0) machinery -- the "ideal" reference
    every hardware run is compared against."""
    layer = HybridQuantumFeatureLayer(HybridConfig())
    return layer.quantum_features(X_demo_quantum, theta)


def build_measurement_circuits(X_demo_quantum: np.ndarray, theta: np.ndarray):
    """One fully-bound (both x and theta), measured circuit PER demo
    sample -- ready to transpile and submit as-is. Reuses
    build_hybrid_circuit unmodified; only appends .measure_all(), which
    Phase 10's own EstimatorQNN-based training path never needed (Estimator
    primitives never measure)."""
    cfg = HybridConfig()
    circuits = []
    for row in X_demo_quantum:
        qc, x_params, theta_params = build_hybrid_circuit(cfg.n_qubits, cfg.reps)
        bound = qc.assign_parameters({**dict(zip(x_params, row)), **dict(zip(theta_params, theta))})
        bound.measure_all()
        circuits.append(bound)
    return circuits


def counts_to_expectation_values(counts: dict[str, int], n_qubits: int) -> np.ndarray:
    """Empirical <Z_i> from measurement counts, matching this project's
    established little-endian convention (rightmost bitstring character =
    qubit 0, identical to _z_observables' SparsePauliOp labeling)."""
    total = sum(counts.values())
    exp = np.zeros(n_qubits)
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")[-n_qubits:]  # measure_all may prepend ancilla/other-register bits
        for i in range(n_qubits):
            bit = bits[n_qubits - 1 - i]
            exp[i] += count * (1 if bit == "0" else -1)
    return exp / total


def compare_ideal_vs_hardware(ideal: np.ndarray, hardware: np.ndarray) -> dict:
    """ideal, hardware: (n_samples, n_qubits) expectation-value arrays.
    Purely descriptive -- NOT interpreted as clinical performance (Section
    7 of the governing spec)."""
    deviation = hardware - ideal
    return {
        "mean_absolute_deviation": float(np.mean(np.abs(deviation))),
        "max_absolute_deviation": float(np.max(np.abs(deviation))),
        "per_qubit_mean_absolute_deviation": np.mean(np.abs(deviation), axis=0).tolist(),
        "per_sample_mean_absolute_deviation": np.mean(np.abs(deviation), axis=1).tolist(),
        "rmse": float(np.sqrt(np.mean(deviation ** 2))),
        "note": "Deviation between EXACT statevector expectation values and empirical (shot-based) expectation "
                "values from the executed circuit. This reflects sampling noise and, for real hardware, physical "
                "device noise -- it is NOT a measure of clinical/predictive performance.",
    }


def run_hardware_demo(*, hardware: bool = False, n_samples: int = DEFAULT_N_SAMPLES,
                       shots: int = DEFAULT_SHOTS, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                       make_plots: bool = True) -> dict:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    if shots > MAX_SHOTS_HARDWARE:
        raise QuantumConfigError(f"shots={shots} exceeds MAX_SHOTS_HARDWARE={MAX_SHOTS_HARDWARE}.")

    fg = get_cardio_feature_groups()
    pcfg = PreprocessingConfig()
    theta = load_frozen_theta()
    cfg = HybridConfig()

    split, demo_df = build_demo_dataset(n_samples, seed=PRIMARY_SEED)
    X_demo_quantum = preprocess_demo_samples(split, demo_df, fg, pcfg)
    ideal_expectations = compute_ideal_expectations(X_demo_quantum, theta)
    circuits = build_measurement_circuits(X_demo_quantum, theta)

    error: str | None = None
    if hardware:
        print(f"[phase16] Attempting REAL IBM Quantum execution: {len(circuits)} circuits, {shots} shots each ...", flush=True)
        backend_obj = IBMHardwareBackend(IBMHardwareConfig(max_qubits=MAX_QUBITS_HARDWARE, max_shots=MAX_SHOTS_HARDWARE,
                                                            max_circuits=MAX_SAMPLES_HARDWARE))
        try:
            record = backend_obj.execute_with_metadata(circuits, shots, timeout_seconds=timeout_seconds)
        except IBMCredentialsError as exc:
            record = HardwareExecutionRecord(
                backend_name="none", job_id=None, status="BLOCKED_NO_CREDENTIALS", hardware_execution=False,
                num_qubits=cfg.n_qubits, shots=shots, n_circuits=len(circuits), circuit_depths=[], gate_counts=[],
                counts_list=[], submitted_at=None, completed_at=None, error=str(exc),
            )
            error = str(exc)
        except QuantumConfigError as exc:
            record = HardwareExecutionRecord(
                backend_name="none", job_id=None, status="BLOCKED_NO_BACKEND_AVAILABLE", hardware_execution=False,
                num_qubits=cfg.n_qubits, shots=shots, n_circuits=len(circuits), circuit_depths=[], gate_counts=[],
                counts_list=[], submitted_at=None, completed_at=None, error=str(exc),
            )
            error = str(exc)
    else:
        print(f"[phase16] Mock/dry-run mode (no --hardware flag): using StatevectorBackend's seeded mock "
              f"run_circuit for {len(circuits)} circuits, {shots} shots each ...", flush=True)
        sim = StatevectorBackend()
        submitted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        counts_list = [sim.run_circuit(c, shots, seed=PRIMARY_SEED) for c in circuits]
        record = HardwareExecutionRecord(
            backend_name="statevector_mock", job_id=None, status="MOCK_COMPLETED", hardware_execution=False,
            num_qubits=cfg.n_qubits, shots=shots, n_circuits=len(circuits),
            circuit_depths=[c.depth() for c in circuits], gate_counts=[dict(c.count_ops()) for c in circuits],
            counts_list=counts_list, submitted_at=submitted_at,
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    if record.counts_list:
        empirical_expectations = np.array([counts_to_expectation_values(c, cfg.n_qubits) for c in record.counts_list])
        comparison = compare_ideal_vs_hardware(ideal_expectations, empirical_expectations)
    else:
        empirical_expectations = None
        comparison = {"note": "Comparison not possible -- hardware/mock execution did not produce measurement counts.",
                      "reason": record.error or record.status}

    if record.status in ("COMPLETED", "MOCK_COMPLETED"):
        decision = "A_SUCCESSFULLY_DEMONSTRATED" if record.hardware_execution else "A_MOCK_DEMONSTRATED_NO_REAL_HARDWARE"
    elif record.status in ("BLOCKED_NO_CREDENTIALS", "BLOCKED_NO_BACKEND_AVAILABLE", "SUBMITTED_PENDING"):
        decision = "B_IMPLEMENTED_BUT_BLOCKED"
    else:
        decision = "C_FAILED"

    # ---- persist ----
    demo_df[["id", TARGET_COLUMN]].assign(**{f"pca_{i}": X_demo_quantum[:, i] for i in range(cfg.n_qubits)}) \
        .to_csv(RESULTS_ROOT / "demo_samples.csv", index=False)

    with open(RESULTS_ROOT / "circuit.qasm", "w", encoding="utf-8") as fh:
        fh.write(qasm2.dumps(circuits[0]))

    obs_rows = []
    for i in range(len(circuits)):
        row = {"sample_index": i, "id": int(demo_df.iloc[i]["id"])}
        for q in range(cfg.n_qubits):
            row[f"ideal_expZ_qubit{q}"] = float(ideal_expectations[i, q])
            if empirical_expectations is not None:
                row[f"empirical_expZ_qubit{q}"] = float(empirical_expectations[i, q])
        obs_rows.append(row)
    pd.DataFrame(obs_rows).to_csv(RESULTS_ROOT / "observables.csv", index=False)

    with open(RESULTS_ROOT / "measurement_counts.json", "w", encoding="utf-8") as fh:
        json.dump({"counts_per_sample": record.counts_list}, fh, indent=2)
    with open(RESULTS_ROOT / "ideal_results.json", "w", encoding="utf-8") as fh:
        json.dump({"expectation_values": ideal_expectations.tolist(), "circuit_config": cfg.to_dict(),
                    "theta": theta.tolist(), "source": "Phase 10's frozen, already-trained theta -- not re-optimized"}, fh, indent=2)
    with open(RESULTS_ROOT / ("hardware_results.json" if hardware else "mock_results.json"), "w", encoding="utf-8") as fh:
        json.dump(record.to_dict(), fh, indent=2, default=float)
    with open(RESULTS_ROOT / "ideal_vs_hardware.json", "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, default=float)
    with open(RESULTS_ROOT / "job_metadata.json", "w", encoding="utf-8") as fh:
        json.dump({"job_id": record.job_id, "backend": record.backend_name, "status": record.status,
                    "submitted_at": record.submitted_at, "completed_at": record.completed_at,
                    "hardware_execution": record.hardware_execution}, fh, indent=2)
    with open(RESULTS_ROOT / "backend_metadata.json", "w", encoding="utf-8") as fh:
        json.dump({"circuit_depths": record.circuit_depths, "gate_counts": record.gate_counts,
                    "num_qubits": record.num_qubits, "shots": record.shots, "n_circuits": record.n_circuits}, fh, indent=2)

    hardware_execution_json = {
        "job_id": record.job_id, "backend": record.backend_name, "status": record.status,
        "num_qubits": record.num_qubits, "shots": record.shots, "circuit_depth": (record.circuit_depths[0] if record.circuit_depths else None),
        "gate_counts": (record.gate_counts[0] if record.gate_counts else None),
        "execution_timestamp": record.submitted_at, "hardware_execution": record.hardware_execution,
    }
    with open(RESULTS_ROOT / "hardware_execution.json", "w", encoding="utf-8") as fh:
        json.dump(hardware_execution_json, fh, indent=2)

    summary = {
        "decision": decision,
        "hardware_requested": hardware,
        "record": record.to_dict(),
        "n_samples": n_samples, "shots": shots, "circuit_config": cfg.to_dict(),
        "comparison_set": {"n_demo_samples": n_samples, "test_set_fingerprint": EXPECTED_FINGERPRINT,
                            "demo_samples_disjoint_from_test_set": True},
        "ideal_vs_hardware": comparison,
        "reproducibility": {"seed": PRIMARY_SEED, "theta": theta.tolist(), "demo_sample_ids": demo_df["id"].tolist(),
                             "pca_config": pcfg.to_dict() if hasattr(pcfg, "to_dict") else str(pcfg)},
        "error": error,
    }
    with open(RESULTS_ROOT / "phase16_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(f"[phase16] DECISION: {decision} (status={record.status}, job_id={record.job_id})", flush=True)

    if make_plots:
        _plot_results(circuits[0], ideal_expectations, empirical_expectations, RESULTS_ROOT)

    return summary


def _plot_results(example_circuit, ideal, empirical, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        fig = example_circuit.draw(output="mpl")
        fig.savefig(out_dir / "circuit_diagram.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 -- optional dependency (pylatexenc) may be missing
        (out_dir / "circuit_diagram.txt").write_text(str(example_circuit.draw(output="text")), encoding="utf-8")
        print(f"[phase16] matplotlib circuit diagram unavailable ({exc}); wrote circuit_diagram.txt instead.", flush=True)

    if empirical is not None:
        n_qubits = ideal.shape[1]
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(n_qubits)
        width = 0.35
        ax.bar(x - width / 2, ideal.mean(axis=0), width, label="Ideal (exact statevector)")
        ax.bar(x + width / 2, empirical.mean(axis=0), width, label="Executed (shot-based)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"qubit {i}" for i in range(n_qubits)])
        ax.set_ylabel("Mean <Z_i> across demo samples")
        ax.set_title("Ideal vs. executed expectation values -- Phase 16")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "ideal_vs_hardware_expectation_values.png", dpi=150)
        plt.close(fig)

        deviation = np.abs(empirical - ideal).flatten()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(deviation, bins=20)
        ax.set_xlabel("|deviation| (empirical - ideal expectation value)")
        ax.set_ylabel("count")
        ax.set_title("Hardware/mock noise-deviation distribution -- Phase 16")
        fig.tight_layout()
        fig.savefig(out_dir / "deviation_distribution.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 16: IBM Quantum hardware execution demo")
    parser.add_argument("--hardware", action="store_true", help="Attempt REAL IBM Quantum execution (contacts the network).")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    summary = run_hardware_demo(hardware=args.hardware, n_samples=args.n_samples, shots=args.shots,
                                 timeout_seconds=args.timeout_seconds)
    print(json.dumps({k: v for k, v in summary.items() if k != "reproducibility"}, indent=2, default=float))


if __name__ == "__main__":
    main()
