"""Runtime assertion helpers for the preprocessing pipeline.

These are small, pure checks used in two places:
  1. Inside pipeline.py, as sanity gates after each pipeline stage (a
     transformer producing a NaN, wrong shape, or an out-of-range value is
     a bug, and should fail loudly and immediately, not surface three
     stages later as a mysterious model error).
  2. Inside tests/test_preprocessing.py, so tests read as assertions about
     *properties* the pipeline must have, not ad-hoc numpy calls repeated
     in every test function.

Nothing here fits or transforms data. These functions only inspect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PreprocessingValidationError(Exception):
    """Raised when a pipeline stage produces output that violates an
    invariant the rest of the pipeline (and Phase 3 onward) depends on.
    """


def _to_array(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    return X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)


def assert_no_missing_values(X: pd.DataFrame | np.ndarray, name: str = "array") -> None:
    """Raise if X contains any NaN. Used after imputation, selection, and PCA."""
    arr = _to_array(X)
    if arr.dtype.kind in ("U", "S", "O"):
        # Non-numeric dtype (shouldn't happen post-encoding) -- check via pandas isna.
        if pd.isna(arr).any():
            raise PreprocessingValidationError(f"'{name}' contains missing values.")
        return
    if np.isnan(arr.astype(float)).any():
        n_missing = int(np.isnan(arr.astype(float)).sum())
        raise PreprocessingValidationError(
            f"'{name}' contains {n_missing} missing value(s) after preprocessing; "
            f"every value must be imputed before this stage."
        )


def assert_no_infinite_values(X: pd.DataFrame | np.ndarray, name: str = "array") -> None:
    """Raise if X contains any +/-inf (can happen from a degenerate scaler)."""
    arr = _to_array(X).astype(float)
    if np.isinf(arr).any():
        raise PreprocessingValidationError(f"'{name}' contains infinite value(s).")


def assert_shape(
    X: pd.DataFrame | np.ndarray,
    *,
    expected_rows: int | None = None,
    expected_cols: int | None = None,
    name: str = "array",
) -> None:
    """Raise if X's shape does not match the expected row/column count.

    Either bound may be omitted to skip that check.
    """
    arr = _to_array(X)
    if arr.ndim != 2:
        raise PreprocessingValidationError(f"'{name}' must be 2-D; got shape {arr.shape}.")
    n_rows, n_cols = arr.shape
    if expected_rows is not None and n_rows != expected_rows:
        raise PreprocessingValidationError(
            f"'{name}' has {n_rows} rows; expected {expected_rows}."
        )
    if expected_cols is not None and n_cols != expected_cols:
        raise PreprocessingValidationError(
            f"'{name}' has {n_cols} columns; expected {expected_cols}."
        )


def assert_within_range(
    X: pd.DataFrame | np.ndarray, low: float, high: float, name: str = "array", atol: float = 1e-8
) -> None:
    """Raise if any value in X falls outside [low, high] (within a small tolerance).

    Used to verify the quantum-ready output actually lands inside the
    declared encoding range before it is ever handed to a quantum feature
    map in a later phase.
    """
    arr = _to_array(X).astype(float)
    if arr.min() < low - atol or arr.max() > high + atol:
        raise PreprocessingValidationError(
            f"'{name}' has values outside [{low}, {high}]: "
            f"observed range [{arr.min()}, {arr.max()}]."
        )


def assert_arrays_close(
    a: pd.DataFrame | np.ndarray,
    b: pd.DataFrame | np.ndarray,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    name: str = "arrays",
) -> None:
    """Raise if two arrays are not numerically equal (used for reproducibility tests)."""
    arr_a, arr_b = _to_array(a).astype(float), _to_array(b).astype(float)
    if arr_a.shape != arr_b.shape:
        raise PreprocessingValidationError(
            f"'{name}' shapes differ: {arr_a.shape} vs {arr_b.shape}."
        )
    if not np.allclose(arr_a, arr_b, rtol=rtol, atol=atol):
        max_diff = float(np.max(np.abs(arr_a - arr_b)))
        raise PreprocessingValidationError(
            f"'{name}' are not reproducible: max absolute difference {max_diff}."
        )


def assert_binary_labels(y: pd.Series | np.ndarray, name: str = "target") -> None:
    """Raise if y is not a strictly binary {0, 1} label array."""
    values = set(np.unique(_to_array(y)).tolist())
    if not values.issubset({0, 1}):
        raise PreprocessingValidationError(f"'{name}' must contain only 0/1; found {values}.")
