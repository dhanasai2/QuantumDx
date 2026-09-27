"""Leakage-safe preprocessing and feature pipeline for QuantumDx Phase 2.

This module turns the raw, validated UCI Heart Disease (Cleveland) DataFrame
(see src/data/inspect_dataset.py for Phase 1 loading/validation) into two
feature representations that share ONE upstream preprocessing+selection
stage, so there is no duplicated logic between the branches:

    raw DataFrame
        -> binary target creation (num -> target)
        -> stratified train/test split            <- split happens BEFORE any .fit()
        -> [TRAIN ONLY] impute + encode + scale    (ColumnTransformer)
        -> [TRAIN ONLY] feature selection          (SelectKBest)
        -> "classical-ready" features  ────────────┐
                                                    ├─ shared upstream, no duplication
        -> [TRAIN ONLY] PCA + range-normalize  ────┘
        -> "quantum-ready" features (bounded in [0, pi] by default)

Every fitted step (imputer, scaler, encoder, selector, PCA, range
normalizer) is fit exclusively on the training split and only ever
`.transform()`-ed on the test split. This is the single most important
property of this module and is what tests/test_preprocessing.py's leakage
test exists to prove.

No model is trained here. No quantum circuit is built here. This module
only produces the feature matrices Phase 3 (classical baselines) and a
later phase (QSVM/VQC) will consume.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from src.data.inspect_dataset import TARGET_COLUMN as RAW_TARGET_COLUMN
from src.preprocessing.config import DEFAULT_CONFIG, FeatureSet, PreprocessingConfig
from src.preprocessing.validation import (
    assert_binary_labels,
    assert_no_infinite_values,
    assert_no_missing_values,
    assert_shape,
    assert_within_range,
)

#: Name of the derived binary early-risk target column this module creates.
BINARY_TARGET_COLUMN = "target"


# --------------------------------------------------------------------------
# 1. Target definition
# --------------------------------------------------------------------------


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the binary early-risk target from the raw multi-level diagnosis.

    Rule (matches PRD.md Section 19.2 / MVP_SPEC.md Section 5.2, and the
    standard usage documented in data/raw/heart-disease.names):

        target = 0  if num == 0   (no significant coronary narrowing)
        target = 1  if num  > 0   (disease present, any severity)

    The original ``num`` column is preserved unchanged, alongside the new
    ``target`` column, for traceability -- so it remains possible to see
    exactly which severity levels were folded into the positive class.

    IMPORTANT: this is a binary CARDIOVASCULAR DISEASE RISK CLASSIFICATION
    task (early risk stratification), not a clinically validated diagnosis.
    The label reflects angiographic disease presence/absence in a 1988
    research cohort, not a prospective clinical outcome.

    Args:
        df: The raw, validated DataFrame (must contain a numeric ``num``
            column with no missing values -- see src/data/inspect_dataset.py).

    Returns:
        A NEW DataFrame (the input is never mutated) with an added
        ``target`` column. ``num`` is left untouched.

    Raises:
        ValueError: If ``num`` is missing or contains NaNs.
    """
    if RAW_TARGET_COLUMN not in df.columns:
        raise ValueError(f"Input DataFrame is missing the raw target column '{RAW_TARGET_COLUMN}'.")
    if df[RAW_TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Raw target column '{RAW_TARGET_COLUMN}' contains missing values; "
            f"cannot derive a binary target from an unknown diagnosis."
        )

    out = df.copy(deep=True)
    out[BINARY_TARGET_COLUMN] = (out[RAW_TARGET_COLUMN] > 0).astype(int)
    return out


# --------------------------------------------------------------------------
# 2. Train/test split (must happen before any fitted transformation)
# --------------------------------------------------------------------------


def split_dataset(
    df: pd.DataFrame, config: PreprocessingConfig = DEFAULT_CONFIG
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reproducible, stratified train/test split.

    This function performs NO imputation, encoding, scaling, selection, or
    reduction -- it only partitions rows. It must be called before any
    fitted transformer touches the data, because every downstream fitted
    step will be trained on the resulting ``train_df`` only.

    Args:
        df: A DataFrame that already contains the binary ``target`` column
            (i.e. the output of create_binary_target()).
        config: Supplies ``test_size``, ``random_seed``, and ``stratify``.

    Returns:
        (train_df, test_df), each with a freshly reset index.

    Raises:
        ValueError: If ``target`` is missing.
    """
    if BINARY_TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"'{BINARY_TARGET_COLUMN}' column not found; call create_binary_target() first."
        )

    stratify_col = df[BINARY_TARGET_COLUMN] if config.stratify else None
    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=stratify_col,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_split_id(config: PreprocessingConfig, n_rows: int) -> str:
    """A short, deterministic fingerprint of the split parameters + dataset size.

    Two runs with the same (n_rows, test_size, seed, stratify) will always
    produce the same split_id, and by construction (fixed seed, fixed
    sklearn split algorithm) the same actual row partition. This mirrors
    the ``split_id`` fairness-guard concept from MVP_SPEC.md Section 6.2,
    scaled down to what Phase 2 actually needs.
    """
    fingerprint = f"{n_rows}|{config.test_size}|{config.random_seed}|{config.stratify}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# 4. Feature types (semantic grouping, not "every float is continuous")
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureGroups:
    """Which raw columns belong to which semantic feature type.

    Grouping rationale (see docs/PREPROCESSING.md Section 4 for the full
    discussion):

    - continuous: genuinely continuous physiological measurements
      (age, trestbps, chol, thalach, oldpeak). Median-imputed, then scaled.

    - discrete_numeric (`ca`): an ordered COUNT (0-3 vessels). Treated as
      numeric rather than one-hot-encoded categorical, because one-hot
      encoding would discard the "more vessels = more severe" ordering
      that a plain numeric/scaled representation preserves. Grouped with
      `continuous` for imputation/scaling purposes but tracked separately
      here for documentation clarity.

    - ordinal_categorical (`slope`): an ordered 3-level clinical grading
      (upsloping < flat < downsloping is a documented clinical severity
      ordering). Kept as a single ordered numeric column (not one-hot
      encoded, to preserve the ordering) but imputed with the categorical
      (most_frequent) strategy since a mean/median of a 3-level code is not
      clinically meaningful.

    - binary_categorical (`sex`, `fbs`, `exang`): already 0/1-coded with no
      inherent scale; imputed only, never encoded or scaled.

    - nominal_categorical (`cp`, `restecg`, `thal`): unordered category
      codes (e.g. chest pain type 1-4 has no natural ranking). One-hot
      encoded after imputation.
    """

    continuous: list[str]
    discrete_numeric: list[str]
    ordinal_categorical: list[str]
    binary_categorical: list[str]
    nominal_categorical: list[str]

    @property
    def numeric_columns(self) -> list[str]:
        """Continuous + discrete-numeric columns (share the same impute+scale treatment)."""
        return list(self.continuous) + list(self.discrete_numeric)

    @property
    def all_columns(self) -> list[str]:
        """Every input feature column, in a stable, documented order."""
        return (
            list(self.continuous)
            + list(self.discrete_numeric)
            + list(self.ordinal_categorical)
            + list(self.binary_categorical)
            + list(self.nominal_categorical)
        )


def get_feature_groups(feature_set: FeatureSet = "full") -> FeatureGroups:
    """Return the semantic feature grouping for the requested feature set.

    feature_set="full":      all 13 predictors (includes ca, thal)
    feature_set="screening": 11 predictors, EXCLUDING ca and thal.

    The "screening" set exists to prepare (not yet run -- see
    docs/PREPROCESSING.md) the clinical-leakage ablation: `ca` (fluoroscopy
    vessel count) and `thal` (thallium stress test result) are both results
    of specialist cardiac investigations typically ordered once disease is
    already suspected, so a model relying heavily on them may overstate
    genuine *early*-detection performance.
    """
    continuous = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    discrete_numeric = ["ca"]
    ordinal_categorical = ["slope"]
    binary_categorical = ["sex", "fbs", "exang"]
    nominal_categorical = ["cp", "restecg", "thal"]

    if feature_set == "screening":
        discrete_numeric = [c for c in discrete_numeric if c != "ca"]
        nominal_categorical = [c for c in nominal_categorical if c != "thal"]
    elif feature_set != "full":
        raise ValueError(f"Unknown feature_set: {feature_set!r} (expected 'full' or 'screening').")

    return FeatureGroups(
        continuous=continuous,
        discrete_numeric=discrete_numeric,
        ordinal_categorical=ordinal_categorical,
        binary_categorical=binary_categorical,
        nominal_categorical=nominal_categorical,
    )


# --------------------------------------------------------------------------
# 5 & 6. Encoding + scaling (built as one ColumnTransformer)
# --------------------------------------------------------------------------


def _make_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError(f"Unknown scaler: {name!r} (expected 'standard', 'minmax', or 'robust').")


def build_preprocessor(feature_groups: FeatureGroups, config: PreprocessingConfig) -> ColumnTransformer:
    """Build the (unfitted) impute + encode + scale ColumnTransformer.

    Each feature group gets its own sub-pipeline so that imputation
    strategy and downstream treatment matches its semantic type (see
    FeatureGroups docstring). Output is configured as a pandas DataFrame
    (via set_output) so column names survive into feature selection and the
    final report -- essential for explainability in later phases.
    """
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    numeric_cols = feature_groups.numeric_columns
    if numeric_cols:
        transformers.append(
            (
                "continuous",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy=config.numeric_impute_strategy)),
                        ("scale", _make_scaler(config.scaler)),
                    ]
                ),
                numeric_cols,
            )
        )

    if feature_groups.ordinal_categorical:
        transformers.append(
            (
                "ordinal",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy=config.categorical_impute_strategy)),
                        ("scale", _make_scaler(config.scaler)),
                    ]
                ),
                feature_groups.ordinal_categorical,
            )
        )

    if feature_groups.binary_categorical:
        transformers.append(
            (
                "binary",
                Pipeline([("impute", SimpleImputer(strategy=config.categorical_impute_strategy))]),
                feature_groups.binary_categorical,
            )
        )

    if feature_groups.nominal_categorical:
        transformers.append(
            (
                "nominal",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy=config.categorical_impute_strategy)),
                        (
                            "ohe",
                            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                feature_groups.nominal_categorical,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers, sparse_threshold=0, verbose_feature_names_out=False
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


# --------------------------------------------------------------------------
# 7. Feature selection
# --------------------------------------------------------------------------


def build_selector(config: PreprocessingConfig, k: int) -> SelectKBest:
    """Build the (unfitted) SelectKBest feature selector.

    Method choice (see docs/PREPROCESSING.md Section 7 for full reasoning):
      - "mutual_info": mutual_info_classif captures non-linear dependence
        between each feature and the binary target -- appropriate here
        because there is no reason to assume every clinical risk factor
        relates to risk linearly. Seeded via config.random_seed because
        its internal nearest-neighbour estimator has a stochastic component.
      - "f_classif": ANOVA F-test, a simpler linear-dependence alternative,
        offered for comparison.

    ``k`` is passed in already clipped to the number of features actually
    available after encoding (see SharedFeaturePipeline.fit), so this
    function never triggers sklearn's "k > n_features" warning.
    """
    if config.feature_selection_method == "mutual_info":
        score_func = partial(mutual_info_classif, random_state=config.random_seed)
    elif config.feature_selection_method == "f_classif":
        score_func = f_classif
    else:
        raise ValueError(
            f"Unknown feature_selection_method: {config.feature_selection_method!r} "
            f"(expected 'mutual_info' or 'f_classif')."
        )
    return SelectKBest(score_func=score_func, k=k)


class SharedFeaturePipeline:
    """Impute + encode + scale + select, fit ONCE and reused by both branches.

    This is the single upstream stage that both the classical-ready and
    quantum-ready feature representations are built from (see module
    docstring diagram) -- it exists so that preprocessing logic is never
    duplicated between the two branches.

    Implemented as an explicit two-stage object, rather than a single bare
    sklearn Pipeline, because the feature-selection width
    (config.feature_selection_k) must be clipped against the ACTUAL
    post-encoding column count, which is only known after the
    ColumnTransformer has been fit (one-hot encoding expands nominal
    columns into a data-dependent number of dummy columns).
    """

    def __init__(self, feature_groups: FeatureGroups, config: PreprocessingConfig):
        self.feature_groups = feature_groups
        self.config = config
        self.preprocessor: ColumnTransformer = build_preprocessor(feature_groups, config)
        self.selector: SelectKBest | None = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SharedFeaturePipeline":
        """Fit the preprocessor and selector on TRAINING data only."""
        X_pre = self.preprocessor.fit_transform(X, y)
        k = min(self.config.feature_selection_k, X_pre.shape[1])
        self.selector = build_selector(self.config, k=k)
        self.selector.set_output(transform="pandas")
        self.selector.fit(X_pre, y)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the already-fitted preprocessor + selector. Never refits."""
        if not self._is_fitted or self.selector is None:
            raise RuntimeError("SharedFeaturePipeline.fit() must be called before transform().")
        X_pre = self.preprocessor.transform(X)
        return self.selector.transform(X_pre)

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    @property
    def selected_feature_names(self) -> list[str]:
        if not self._is_fitted or self.selector is None:
            raise RuntimeError("Not fitted yet.")
        return list(self.selector.get_feature_names_out())

    @property
    def n_features_after_encoding(self) -> int:
        """Column count after impute+encode+scale, BEFORE selection (for diagnostics)."""
        if not self._is_fitted:
            raise RuntimeError("Not fitted yet.")
        return len(self.preprocessor.get_feature_names_out())


# --------------------------------------------------------------------------
# 8. Dimensionality reduction -> quantum-ready representation
# --------------------------------------------------------------------------


def build_quantum_pipeline(n_components: int, quantum_range: tuple[float, float]) -> Pipeline:
    """Build the (unfitted) PCA + range-normalization pipeline.

    Continues directly from the classical-ready (selected) feature matrix
    -- it does NOT re-impute, re-encode, or re-scale, avoiding duplicated
    preprocessing logic.

    PCA is fit on training data only. The number of components is a
    configuration value, not a value chosen by inspecting test-set
    performance; see docs/PREPROCESSING.md Section 8 for the explained-
    variance trade-off across a small sweep (2-6 components) and why 4 is
    used only as an unvalidated starting default.

    The MinMaxScaler step maps the PCA output into ``quantum_range``
    (default [0, pi]) purely because that is the domain a future angle-
    encoding quantum feature map will require -- no quantum library is
    imported or used here.

    ``clip=True`` is deliberate: MinMaxScaler is fit on the TRAIN PCA
    output only, so it is expected and correct for a test-set point to
    project slightly outside the training-derived [min, max] range (that
    is exactly what "never fit on test data" implies -- the test set is
    allowed to be somewhat different from train). Clipping guarantees the
    quantum-ready output is ALWAYS bounded within ``quantum_range``, which
    a future angle-encoding feature map requires, without ever using test
    statistics to derive the bound itself.
    """
    return Pipeline(
        [
            ("pca", PCA(n_components=n_components)),
            ("range", MinMaxScaler(feature_range=quantum_range, clip=True)),
        ]
    )


# --------------------------------------------------------------------------
# 9-10. Orchestration: produce both outputs from one shared pipeline,
#        governed entirely by PreprocessingConfig.
# --------------------------------------------------------------------------


@dataclass
class PreprocessingArtifacts:
    """Everything needed to reproduce or reuse a fitted Phase 2 pipeline."""

    shared_pipeline: SharedFeaturePipeline
    quantum_pipeline: Pipeline
    feature_groups: FeatureGroups
    config: PreprocessingConfig
    selected_feature_names: list[str]
    n_features_after_encoding: int
    pca_explained_variance_ratio: np.ndarray
    pca_n_components_used: int
    split_id: str


@dataclass
class PreprocessedData:
    """The complete output of run_preprocessing_pipeline()."""

    # raw (untransformed) feature matrices, post-split, feature_set-filtered
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame

    # binary target (what models are trained/evaluated against)
    y_train: np.ndarray
    y_test: np.ndarray

    # original multi-level severity label, preserved for traceability
    num_train: np.ndarray
    num_test: np.ndarray

    # classical-ready: impute + encode + scale + select
    X_train_classical: pd.DataFrame
    X_test_classical: pd.DataFrame

    # quantum-ready: classical-ready -> PCA -> range-normalize to [0, pi]
    X_train_quantum: np.ndarray
    X_test_quantum: np.ndarray

    artifacts: PreprocessingArtifacts = field(repr=False)


def run_preprocessing_pipeline(
    df: pd.DataFrame, config: PreprocessingConfig = DEFAULT_CONFIG
) -> PreprocessedData:
    """Run the full Phase 2 pipeline end to end, exactly in this order:

        1. create the binary target (num -> target)
        2. stratified train/test split (BEFORE any fitting)
        3. select the configured feature set's raw columns (full/screening)
        4. fit impute+encode+scale+select on TRAIN ONLY -> classical-ready
        5. fit PCA+range-normalize on the TRAIN classical-ready output ONLY
           -> quantum-ready
        6. transform (never refit) the test split through both fitted stages

    Args:
        df: The raw, validated DataFrame (output of
            src.data.inspect_dataset.load_raw_dataset()). Not modified.
        config: Governs every choice in the pipeline (seed, split ratio,
            imputation, scaler, selection, PCA dimensionality, feature set).

    Returns:
        A PreprocessedData bundle with every array/DataFrame Phase 3 needs,
        plus the fitted artifacts for inspection or reuse at inference time.
    """
    df_with_target = create_binary_target(df)
    train_df, test_df = split_dataset(df_with_target, config)

    feature_groups = get_feature_groups(config.feature_set)
    feature_cols = feature_groups.all_columns

    X_train_raw = train_df[feature_cols].reset_index(drop=True)
    X_test_raw = test_df[feature_cols].reset_index(drop=True)
    y_train = train_df[BINARY_TARGET_COLUMN].to_numpy()
    y_test = test_df[BINARY_TARGET_COLUMN].to_numpy()
    num_train = train_df[RAW_TARGET_COLUMN].to_numpy()
    num_test = test_df[RAW_TARGET_COLUMN].to_numpy()

    assert_binary_labels(y_train, name="y_train")
    assert_binary_labels(y_test, name="y_test")

    # --- classical-ready branch (fit on train only) -------------------------
    shared = SharedFeaturePipeline(feature_groups, config)
    X_train_classical = shared.fit_transform(X_train_raw, y_train)
    X_test_classical = shared.transform(X_test_raw)

    assert_no_missing_values(X_train_classical, name="X_train_classical")
    assert_no_missing_values(X_test_classical, name="X_test_classical")
    assert_no_infinite_values(X_train_classical, name="X_train_classical")
    assert_no_infinite_values(X_test_classical, name="X_test_classical")
    assert_shape(X_test_classical, expected_cols=X_train_classical.shape[1], name="X_test_classical")

    # --- quantum-ready branch (continues from the classical output; fit on
    #     train only; does not re-impute/re-encode/re-scale) ----------------
    n_components = min(config.pca_n_components, X_train_classical.shape[1], X_train_classical.shape[0])
    quantum_pipeline = build_quantum_pipeline(n_components, config.quantum_range)
    X_train_quantum = quantum_pipeline.fit_transform(X_train_classical)
    X_test_quantum = quantum_pipeline.transform(X_test_classical)

    assert_no_missing_values(X_train_quantum, name="X_train_quantum")
    assert_no_missing_values(X_test_quantum, name="X_test_quantum")
    assert_within_range(X_train_quantum, *config.quantum_range, name="X_train_quantum")
    assert_within_range(X_test_quantum, *config.quantum_range, name="X_test_quantum")
    assert_shape(X_test_quantum, expected_cols=n_components, name="X_test_quantum")

    artifacts = PreprocessingArtifacts(
        shared_pipeline=shared,
        quantum_pipeline=quantum_pipeline,
        feature_groups=feature_groups,
        config=config,
        selected_feature_names=shared.selected_feature_names,
        n_features_after_encoding=shared.n_features_after_encoding,
        pca_explained_variance_ratio=quantum_pipeline.named_steps["pca"].explained_variance_ratio_,
        pca_n_components_used=n_components,
        split_id=compute_split_id(config, n_rows=len(df_with_target)),
    )

    return PreprocessedData(
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        y_train=y_train,
        y_test=y_test,
        num_train=num_train,
        num_test=num_test,
        X_train_classical=X_train_classical,
        X_test_classical=X_test_classical,
        X_train_quantum=np.asarray(X_train_quantum),
        X_test_quantum=np.asarray(X_test_quantum),
        artifacts=artifacts,
    )


# --------------------------------------------------------------------------
# Demonstration entry point
# --------------------------------------------------------------------------


def _print_demonstration(config: PreprocessingConfig = DEFAULT_CONFIG) -> PreprocessedData:
    """Run the pipeline once and print the Phase 2 verification checkpoint report.

    This performs no new computation beyond one call to
    run_preprocessing_pipeline() -- it exists purely to surface the numbers
    a reviewer needs to see (shapes, class balance, feature counts, NaN
    check) without requiring a notebook.
    """
    from src.data.inspect_dataset import load_raw_dataset, resolve_dataset_path

    df = load_raw_dataset(resolve_dataset_path(None))
    result = run_preprocessing_pipeline(df, config)

    def _class_counts(y: np.ndarray) -> str:
        n0 = int((y == 0).sum())
        n1 = int((y == 1).sum())
        total = n0 + n1
        return f"0={n0} ({n0/total:.1%})  1={n1} ({n1/total:.1%})"

    no_nans = (
        not result.X_train_classical.isna().any().any()
        and not result.X_test_classical.isna().any().any()
        and not np.isnan(result.X_train_quantum).any()
        and not np.isnan(result.X_test_quantum).any()
    )

    print("=" * 70)
    print("QuantumDx Phase 2 -- Preprocessing Pipeline Demonstration")
    print("=" * 70)
    print(f"Config: {config.to_dict()}")
    print()
    print(f"Original raw shape        : {df.shape}")
    print(f"Train shape (raw features): {result.X_train_raw.shape}")
    print(f"Test shape  (raw features): {result.X_test_raw.shape}")
    print()
    print(f"Class distribution (train): {_class_counts(result.y_train)}")
    print(f"Class distribution (test) : {_class_counts(result.y_test)}")
    print()
    print(f"Features after impute+encode+scale (pre-selection): {result.artifacts.n_features_after_encoding}")
    print(f"Number of selected features (feature selection)   : {len(result.artifacts.selected_feature_names)}")
    print(f"Selected feature names                             : {result.artifacts.selected_feature_names}")
    print(f"Number of PCA components (quantum-ready)           : {result.artifacts.pca_n_components_used}")
    print(f"PCA explained variance ratio (per component)       : {result.artifacts.pca_explained_variance_ratio}")
    print(f"PCA cumulative explained variance                  : {result.artifacts.pca_explained_variance_ratio.sum():.4f}")
    print()
    print(f"Classical-ready shape (train, test): {result.X_train_classical.shape}, {result.X_test_classical.shape}")
    print(f"Quantum-ready shape   (train, test): {result.X_train_quantum.shape}, {result.X_test_quantum.shape}")
    print(f"Quantum-ready value range          : [{result.X_train_quantum.min():.4f}, {result.X_train_quantum.max():.4f}]")
    print()
    print(f"No NaNs anywhere in any output: {no_nans}")
    print(f"Split ID (reproducibility fingerprint): {result.artifacts.split_id}")
    print("=" * 70)
    return result


if __name__ == "__main__":
    _print_demonstration()
