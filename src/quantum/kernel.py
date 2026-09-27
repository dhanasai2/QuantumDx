"""Quantum fidelity kernel computation (Phase 4, Section 6).

    K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2,   |phi(x)> = U_phi(x)|0...0>

Every function here is a pure function of (X, feature_map, backend) --
none of them accepts a label array `y`, and none of them fits anything.
Test-set rows are embedded with exactly the same, already-fixed feature
map as training-set rows; there is no code path by which a test row could
influence how a training row (or the feature map itself) is embedded.

Efficiency note: a statevector for a given feature vector x depends only
on x, not on which other vector it will later be paired with. Each row's
statevector is therefore computed ONCE (compute_statevectors), and the
kernel matrix is assembled from inner products of those cached
statevectors -- O(n) circuit evaluations plus O(n^2) cheap vector inner
products, not O(n^2) circuit evaluations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.quantum.backends import QuantumBackend


def compute_statevectors(
    X: np.ndarray, feature_map: QuantumCircuit, backend: QuantumBackend
) -> list[Statevector]:
    """Embed every row of X through the feature map, once each.

    Args:
        X: (n_samples, n_qubits) array; n_qubits must equal
            feature_map.num_qubits (one classical feature per qubit,
            angle-encoded -- see docs/QUANTUM_PIPELINE.md).
        feature_map: An UNBOUND feature-map circuit (from build_feature_map).
        backend: Any QuantumBackend implementation (Phase 4: StatevectorBackend).

    Returns:
        A list of Statevector objects, one per row of X, in row order.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_qubits); got shape {X.shape}.")
    if X.shape[1] != feature_map.num_qubits:
        raise ValueError(
            f"X has {X.shape[1]} columns but feature_map expects {feature_map.num_qubits} "
            f"(one classical feature per qubit)."
        )
    return [backend.compute_statevector(feature_map.assign_parameters(row)) for row in X]


def kernel_matrix_from_statevectors(
    statevectors_a: list[Statevector], statevectors_b: list[Statevector], *, symmetric: bool
) -> np.ndarray:
    """Assemble a kernel matrix from two lists of already-computed statevectors.

    Args:
        symmetric: Set True only when statevectors_a and statevectors_b are
            the SAME list (e.g. computing K_train_train) -- this halves the
            work by exploiting K[i,j] == K[j,i], but the diagonal is still
            computed explicitly (not assumed to be exactly 1.0), because
            small deviations from 1.0 in exact simulation are themselves a
            useful floating-point sanity signal (see kernel_diagnostics).
    """
    n_a, n_b = len(statevectors_a), len(statevectors_b)
    K = np.empty((n_a, n_b), dtype=float)
    if symmetric:
        if n_a != n_b:
            raise ValueError(f"symmetric=True requires equal-length statevector lists; got {n_a} and {n_b}.")
        for i in range(n_a):
            for j in range(i, n_b):
                fidelity = float(abs(statevectors_a[i].inner(statevectors_b[j])) ** 2)
                K[i, j] = fidelity
                K[j, i] = fidelity
    else:
        for i in range(n_a):
            for j in range(n_b):
                K[i, j] = float(abs(statevectors_a[i].inner(statevectors_b[j])) ** 2)
    return K


def _stack_statevector_amplitudes(statevectors: list[Statevector]) -> np.ndarray:
    """Stack a list of Statevectors into one (n, 2**n_qubits) complex array."""
    return np.stack([sv.data for sv in statevectors])


def kernel_matrix_from_statevectors_vectorized(
    statevectors_a: list[Statevector], statevectors_b: list[Statevector], *, symmetric: bool
) -> np.ndarray:
    """IMPLEMENTATION / PERFORMANCE OPTIMIZATION ONLY -- computes the exact
    same mathematical quantity as kernel_matrix_from_statevectors() above
    (the original, PRESERVED, reference implementation):

        K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2

    The only difference is HOW it is computed: the reference implementation
    calls Statevector.inner() once per (i,j) pair in a Python loop -- O(n^2)
    Python-level function calls. This version stacks every statevector's
    amplitude array into one matrix and computes ALL pairwise inner
    products via a single matrix multiplication (BLAS), which is
    dramatically faster at scale for numerically identical results.

    Added in the Phase 6 (large-dataset) experiment, where kernel sizes of
    5,000-20,000 make the per-pair Python loop impractical (measured: the
    loop costs ~8us/pair; at n=20,000 that is ~27 minutes for the
    train-train matrix alone). NOT used by, and does not change the
    behavior of, any existing Phase 4/5 code path -- kernel_matrix_from_
    statevectors() and compute_kernel_matrix() are untouched and remain
    the reference used by src/quantum/sanity_experiment.py and
    src/quantum/qsvm_experiment.py.

    Verified numerically equivalent to the reference implementation (see
    docs/LARGE_DATASET.md, "Kernel Optimization" section, and
    tests/test_quantum.py::test_vectorized_kernel_matches_reference)
    before being used for any Stage B+ result.

    No change to the kernel definition, feature map, encoding, QSVM, or
    evaluation methodology -- this function does not decide anything
    scientific; it only computes the same numbers faster.

    Args:
        symmetric: True only when statevectors_a and statevectors_b are
            the SAME statevector list (e.g. K_train_train). The full
            (n,n) matrix is still computed (a single matmul is faster than
            a loop even when it duplicates the lower triangle) -- the flag
            only controls the equal-length validation.

    Returns:
        A real-valued (n_a, n_b) ndarray, dtype float64.
    """
    n_a, n_b = len(statevectors_a), len(statevectors_b)
    if symmetric and n_a != n_b:
        raise ValueError(f"symmetric=True requires equal-length statevector lists; got {n_a} and {n_b}.")

    A = _stack_statevector_amplitudes(statevectors_a)
    B = A if (symmetric and statevectors_a is statevectors_b) else _stack_statevector_amplitudes(statevectors_b)

    # <phi(x_i)|phi(x_j)> = sum_k conj(A[i,k]) * B[j,k] = (A.conj() @ B.T)[i,j]
    inner = A.conj() @ B.T
    K = np.abs(inner) ** 2
    return K.astype(np.float64)


def kernel_matrix_from_statevectors_blockwise(
    statevectors_a: list[Statevector],
    statevectors_b: list[Statevector],
    *,
    symmetric: bool,
    block_size: int = 2000,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """IMPLEMENTATION / PERFORMANCE OPTIMIZATION ONLY -- computes the exact
    same mathematical quantity as kernel_matrix_from_statevectors() and
    kernel_matrix_from_statevectors_vectorized() above (both PRESERVED,
    untouched reference implementations):

        K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2

    Added for Stage D (n_train=20,000), where the whole-matrix vectorized
    approach (`kernel_matrix_from_statevectors_vectorized`) transiently
    allocates a full (n, n) COMPLEX inner-product matrix before taking its
    squared magnitude -- at n=20,000 that intermediate alone is
    20,000 x 20,000 x 16 bytes (complex128) ~= 6.4 GB, on top of the final
    (n, n) float64 result (~3.2 GB) and the extra copy `.astype(np.float64)`
    makes even when the array is already float64. On a machine with only a
    few GB of free RAM (measured: ~4.3 GB available out of 16.9 GB total at
    the time Stage D was run -- see docs/STAGE_D.md, "Kernel memory
    diagnostics"), that transient peak risks a MemoryError or severe
    swapping. This function computes the identical result row-block by
    row-block, writing each block directly into a single preallocated
    output array, so peak extra memory is O(block_size * n) instead of
    O(n^2) x (several temporaries).

    Args:
        block_size: number of rows of `statevectors_a` processed per chunk.
            Smaller values reduce peak memory further at a small speed cost
            (still one BLAS matmul per block, not a Python-level per-pair
            loop). 2000 keeps a 20,000-wide float64 block at ~2000*20000*16
            bytes (complex128 intermediate) = 640 MB, well under the
            ~3.2 GB the final output matrix itself requires.
        dtype: dtype of the returned (and internally accumulated) matrix.
            Defaults to float64, IDENTICAL to the other two kernel
            functions. Stage D additionally uses dtype=np.float32 as a
            SEPARATE, explicitly documented memory optimization (halves
            the resident size of the (20000, 20000) matrix from ~3.2 GB to
            ~1.6 GB); fidelity values lie in [0, 1] and float32 carries
            ~7 significant decimal digits, far more precision than the
            kernel values, feature map, or SVM decision boundary require
            -- verified numerically equivalent to float64 (max abs diff)
            in tests/test_quantum.py before use on real Stage D data.

    Returns:
        A real-valued (n_a, n_b) ndarray of the requested dtype.
    """
    n_a, n_b = len(statevectors_a), len(statevectors_b)
    if symmetric and n_a != n_b:
        raise ValueError(f"symmetric=True requires equal-length statevector lists; got {n_a} and {n_b}.")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1; got {block_size}.")

    A = _stack_statevector_amplitudes(statevectors_a)
    B = A if (symmetric and statevectors_a is statevectors_b) else _stack_statevector_amplitudes(statevectors_b)
    B_T = B.T  # computed once, reused by every block

    K = np.empty((n_a, n_b), dtype=dtype)
    for start in range(0, n_a, block_size):
        end = min(start + block_size, n_a)
        # <phi(x_i)|phi(x_j)> = sum_k conj(A[i,k]) * B[j,k] = (A[block].conj() @ B.T)[i,j]
        inner_block = A[start:end].conj() @ B_T
        K[start:end, :] = np.abs(inner_block) ** 2

    return K


def compute_kernel_matrix(
    X_a: np.ndarray,
    X_b: np.ndarray,
    feature_map: QuantumCircuit,
    backend: QuantumBackend,
    *,
    symmetric: bool,
) -> np.ndarray:
    """Compute K(X_a, X_b) end to end: embed both sides, then assemble the matrix.

    Pass symmetric=True only when X_a and X_b are the identical array (e.g.
    K_train_train, K_test_test) -- this reuses one set of statevectors for
    both sides instead of computing it twice.
    """
    statevectors_a = compute_statevectors(X_a, feature_map, backend)
    statevectors_b = statevectors_a if symmetric else compute_statevectors(X_b, feature_map, backend)
    return kernel_matrix_from_statevectors(statevectors_a, statevectors_b, symmetric=symmetric)


# --------------------------------------------------------------------------
# Diagnostics -- verified, not assumed, properties of the resulting matrix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelDiagnostics:
    shape: tuple[int, int]
    min_value: float
    max_value: float
    mean_value: float
    is_square: bool
    is_symmetric: bool | None
    diagonal_mean: float | None
    diagonal_max_abs_deviation_from_one: float | None
    offdiag_mean: float | None
    offdiag_std: float | None
    within_valid_fidelity_range: bool

    def to_dict(self) -> dict:
        return {
            "shape": list(self.shape),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": self.mean_value,
            "is_square": self.is_square,
            "is_symmetric": self.is_symmetric,
            "diagonal_mean": self.diagonal_mean,
            "diagonal_max_abs_deviation_from_one": self.diagonal_max_abs_deviation_from_one,
            "offdiag_mean": self.offdiag_mean,
            "offdiag_std": self.offdiag_std,
            "within_valid_fidelity_range": self.within_valid_fidelity_range,
        }


def kernel_diagnostics(K: np.ndarray, *, symmetric: bool, atol: float = 1e-6) -> KernelDiagnostics:
    """Compute the sanity-check properties Phase 4 Section 6 requires:
    correct shape, symmetry (where expected), diagonal ~= 1, and values
    within the valid fidelity range [0, 1] (up to floating-point tolerance).
    """
    is_square = K.shape[0] == K.shape[1]
    is_symmetric = bool(np.allclose(K, K.T, atol=atol)) if is_square else None

    diagonal_mean = diagonal_max_dev = offdiag_mean = offdiag_std = None
    if symmetric and is_square:
        diag = np.diag(K)
        diagonal_mean = float(diag.mean())
        diagonal_max_dev = float(np.max(np.abs(diag - 1.0)))
        mask = ~np.eye(K.shape[0], dtype=bool)
        off = K[mask]
        offdiag_mean = float(off.mean())
        offdiag_std = float(off.std())

    return KernelDiagnostics(
        shape=(K.shape[0], K.shape[1]),
        min_value=float(K.min()),
        max_value=float(K.max()),
        mean_value=float(K.mean()),
        is_square=is_square,
        is_symmetric=is_symmetric,
        diagonal_mean=diagonal_mean,
        diagonal_max_abs_deviation_from_one=diagonal_max_dev,
        offdiag_mean=offdiag_mean,
        offdiag_std=offdiag_std,
        within_valid_fidelity_range=bool(K.min() >= -atol and K.max() <= 1.0 + atol),
    )


# --------------------------------------------------------------------------
# Disk cache -- keyed by data + feature-map config + backend, per
# MVP_SPEC.md Section 18.4 (QSVM-6); reused here for Phase 4/5 alike.
# --------------------------------------------------------------------------


def _kernel_cache_key(
    X_a: np.ndarray, X_b: np.ndarray, feature_map_config: dict, backend_name: str
) -> str:
    payload = {
        "X_a_sha256": hashlib.sha256(np.ascontiguousarray(X_a).tobytes()).hexdigest(),
        "X_b_sha256": hashlib.sha256(np.ascontiguousarray(X_b).tobytes()).hexdigest(),
        "feature_map_config": feature_map_config,
        "backend_name": backend_name,
    }
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def compute_kernel_matrix_cached(
    X_a: np.ndarray,
    X_b: np.ndarray,
    feature_map: QuantumCircuit,
    feature_map_config: dict,
    backend: QuantumBackend,
    cache_dir: str | Path,
    *,
    symmetric: bool,
) -> tuple[np.ndarray, bool]:
    """Like compute_kernel_matrix, but reads/writes a disk cache keyed by
    (data content, feature-map config, backend name).

    Returns:
        (K, cache_hit) -- cache_hit is True iff K was loaded from disk
        rather than recomputed.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    key = _kernel_cache_key(X_a, X_b, feature_map_config, backend.name)
    file_path = cache_path / f"{key}.npy"

    if file_path.is_file():
        return np.load(file_path), True

    K = compute_kernel_matrix(X_a, X_b, feature_map, backend, symmetric=symmetric)
    np.save(file_path, K)
    return K, False
