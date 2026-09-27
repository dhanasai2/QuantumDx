"""Reproducible sampling for the large-dataset experiment (Phase 6, Sections 7-9).

Design (documented here because it is a real methodological choice, not an
accident of code):

    cleaned modeling pool (68,641 rows, cardio_train.csv minus flagged rows)
        |
        +--> FIXED HELD-OUT TEST SET (classical), n=2,000, stratified, seed 42
        |       |
        |       +--> FIXED QSVM TEST SUBSET, n=200, stratified subsample of
        |             the 2,000-row test set, seed 42 -- kept small
        |             deliberately: quantum test-kernel cost is
        |             O(n_test * n_train), and this subset is IDENTICAL
        |             across every stage, so QSVM results remain
        |             comparable stage-to-stage.
        |
        +--> TRAINING POOL (the remaining ~66,641 rows)
                |
                +--> nested stratified stage subsamples, e.g.
                     1,000 subset-of 5,000 subset-of 10,000 subset-of 20,000
                     (drawn with the SAME seed so smaller stages are a
                     strict subset of larger ones, not independently
                     resampled -- this makes "did more data help" a clean
                     question, not confounded by which rows changed)

The held-out test set (both the 2,000-row classical version and the
200-row QSVM version) is drawn ONCE and reused, unchanged, across every
stage. It is never touched by any fitting step, never used for
hyperparameter selection, and is evaluated on only after a stage's model
selection is complete -- the same discipline as Phase 2/3/5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class LargeDatasetSplit:
    """The one-time, fixed partition reused by every stage."""

    training_pool: pd.DataFrame       # rows available for stage subsampling
    test_set_classical: pd.DataFrame  # fixed, n=2000
    test_set_quantum: pd.DataFrame    # fixed, n=200, subset of test_set_classical
    seed: int
    target_column: str


def build_fixed_split(
    modeling_df: pd.DataFrame,
    target_column: str,
    *,
    classical_test_size: int = 2000,
    quantum_test_size: int = 200,
    seed: int = 42,
) -> LargeDatasetSplit:
    """Create the ONE fixed test set (classical + quantum) and training pool.

    Called exactly once for the whole large-dataset experiment. Every
    stage draws its training subsample from `training_pool` only.
    """
    training_pool, test_set_classical = train_test_split(
        modeling_df,
        test_size=classical_test_size,
        stratify=modeling_df[target_column],
        random_state=seed,
    )
    # The QSVM test subset is a stratified subsample OF the classical test
    # set (not an independent draw), so every QSVM-evaluated point is also
    # one of the classical-evaluated points -- a strict subset, enabling a
    # direct, same-points comparison between QSVM and classical at those
    # 200 rows specifically, in addition to classical's own 2000-row read.
    _, test_set_quantum = train_test_split(
        test_set_classical,
        test_size=quantum_test_size,
        stratify=test_set_classical[target_column],
        random_state=seed,
    )

    return LargeDatasetSplit(
        training_pool=training_pool.reset_index(drop=True),
        test_set_classical=test_set_classical.reset_index(drop=True),
        test_set_quantum=test_set_quantum.reset_index(drop=True),
        seed=seed,
        target_column=target_column,
    )


def nested_stratified_stage_samples(
    training_pool: pd.DataFrame,
    target_column: str,
    stage_sizes: list[int],
    seed: int = 42,
) -> dict[int, pd.DataFrame]:
    """Draw nested stratified subsamples: the smallest stage is a subset of
    the next-largest, which is a subset of the next, etc. -- not
    independently resampled per stage.

    Args:
        stage_sizes: e.g. [1000, 5000, 10000, 20000]. Must all be <=
            len(training_pool); a ValueError is raised otherwise (this is a
            "measure and report" phase, not one that silently clips).

    Returns:
        {size: DataFrame} for every requested size.
    """
    sizes_sorted = sorted(set(stage_sizes), reverse=True)
    if sizes_sorted[0] > len(training_pool):
        raise ValueError(
            f"Requested stage size {sizes_sorted[0]} exceeds the training pool "
            f"size {len(training_pool)}."
        )

    result: dict[int, pd.DataFrame] = {}
    current_pool = training_pool
    for size in sizes_sorted:
        if size == len(current_pool):
            sampled = current_pool
        else:
            sampled, _ = train_test_split(
                current_pool,
                train_size=size,
                stratify=current_pool[target_column],
                random_state=seed,
            )
        result[size] = sampled.reset_index(drop=True)
        current_pool = sampled  # next (smaller) stage nests inside this one

    return result


def verify_nesting(stage_samples: dict[int, pd.DataFrame], id_column: str = "id") -> dict[int, bool]:
    """Sanity check: every smaller stage's ids must be a subset of every
    larger stage's ids. Returns {size: is_subset_of_next_larger}.
    """
    sizes_sorted = sorted(stage_samples.keys())
    result = {}
    for i, size in enumerate(sizes_sorted[:-1]):
        smaller_ids = set(stage_samples[size][id_column])
        larger_ids = set(stage_samples[sizes_sorted[i + 1]][id_column])
        result[size] = smaller_ids.issubset(larger_ids)
    result[sizes_sorted[-1]] = True  # the largest has nothing to be nested inside
    return result
