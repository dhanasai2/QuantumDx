# Preprocessing & Feature Pipeline — Phase 2

**Status:** Phase 2 — Leakage-Proof Preprocessing & Feature Pipeline
**Builds on:** `docs/DATASET.md` (Phase 1 dataset understanding)
**Feeds:** Phase 3 (classical baselines) and a later phase (QSVM/VQC) — both will consume the outputs described here without modification
**Code:** `src/preprocessing/config.py`, `src/preprocessing/pipeline.py`, `src/preprocessing/validation.py`

This document explains *what* the pipeline does and *why*, written for a student learning ML/QML rather than as a terse API reference. Every claim below is backed by a test in `tests/test_preprocessing.py`.

---

## 1. What "Phase 2" actually builds 

Phase 1 established that `data/raw/processed.cleveland.data` is a clean, well-understood, 303-row dataset with 6 missing cells and a near-balanced binary framing. Phase 2 turns that raw data into feature matrices a model can actually be trained on — **without ever letting the test set influence how those matrices are built.**

The pipeline produces **two** feature representations from **one** shared upstream stage:

```
RAW DATA (data/raw/processed.cleveland.data)
   |
   v
TARGET CREATION (num -> target, num preserved)
   |
   v
TRAIN / TEST SPLIT  <-- split happens here, BEFORE any fitting
   |
   v
PREPROCESSING (impute -> encode -> scale)      [fit on TRAIN only]
   |
   v
FEATURE SELECTION (SelectKBest)                 [fit on TRAIN only]
   |
   +----------------------+
   |                      |
   v                      v
CLASSICAL-READY      PCA + range-normalize      [fit on TRAIN only]
FEATURES                  |
                           v
                     QUANTUM-READY FEATURES
                     (bounded in [0, pi])
```

The classical-ready and quantum-ready branches share the **same** impute/encode/scale/select stage — the quantum branch simply continues from the classical branch's output through PCA and a range normalizer. This is a deliberate design choice (see Section 9) that guarantees the two branches can never silently diverge in how the data was cleaned.

---

## 2. Target Transformation

The raw target `num` is an integer 0–4 (angiographic disease severity). QuantumDx's task is **early risk detection**, which is a binary question, so:

```
target = 0   if num == 0     (no significant coronary narrowing)
target = 1   if num  > 0     (disease present, any severity)
```

`create_binary_target()` in `pipeline.py` implements exactly this rule and **preserves the original `num` column unchanged** alongside the new `target` column, so it always remains possible to see which severity levels were folded into the positive class.

**This is a binary cardiovascular disease *risk classification* task, not a clinically validated diagnosis.** The label reflects an angiographic finding in a 1988 research cohort at one hospital — a positive `target` means "this patient's diagnostic workup showed significant coronary narrowing," not "this patient will have a cardiac event." Nothing in this pipeline, and nothing that will be built on top of it, should be read as diagnostic output.

Verified counts (matching `docs/DATASET.md` §8.2): 164 negative, 139 positive, out of 303 total.

---

## 3. Train/Test Split Strategy

**The split happens immediately after target creation and before any other step.** This ordering is not a stylistic preference — it is the mechanism that makes every later "fit on train only" claim actually true, because the test rows are set aside before any statistic (a mean, a median, a PCA axis) is ever computed.

| Parameter | Value | Where it lives |
|---|---|---|
| Method | `sklearn.model_selection.train_test_split` | `pipeline.py::split_dataset()` |
| Test size | 20% (61 records) | `PreprocessingConfig.test_size`, default `0.20` |
| Stratification | On the binary `target` column | `PreprocessingConfig.stratify`, default `True` |
| Random seed | Fixed | `PreprocessingConfig.random_seed`, default `42` |

**Why stratified:** with a near-balanced but not perfectly 50/50 target (54.1% / 45.9%), an unstratified split can — by chance, especially with only 303 rows — produce a test set with a noticeably different class balance than the training set, which would make evaluation metrics harder to interpret. Stratification keeps the class proportion in both splits close to the overall proportion (verified in `test_split_is_stratified_on_target`: both splits stay within a few percentage points of the population rate; in the actual Cleveland run, train and test both land at exactly 54.1%/45.9%).

**Why a fixed seed:** so the exact same 242/61 partition is produced every time the pipeline runs with the same config, which is what makes every downstream number reproducible and comparable across experiments (see Section 10).

**The test set is never used for anything except a final transform.** No statistic derived from it is ever used to fit an imputer, scaler, encoder, selector, or PCA — see Section 9 for how this is enforced, and Section 13 (`tests/test_preprocessing.py::test_no_leakage_from_test_set_statistics`) for how it is proven.

---

## 4. Feature Types — Why Not Every `float64` Column Is "Continuous"

Every column in the raw file loads as `float64` (Phase 1), but treating all 13 predictors identically would throw away real structure. Each feature is assigned to a semantic group based on its actual clinical meaning (`src/preprocessing/pipeline.py::get_feature_groups()`):

| Group | Columns | Reasoning |
|---|---|---|
| **Continuous numerical** | `age`, `trestbps`, `chol`, `thalach`, `oldpeak` | Genuinely continuous physiological measurements with a meaningful magnitude and distance between values (e.g. cholesterol 250 vs 260 is a small, meaningful difference). Median-imputed (robust to outliers/skew), then scaled. |
| **Discrete numerical (ordinal count)** | `ca` | An ordered **count** of vessels (0–3) colored by fluoroscopy. Treated as numeric — **not** one-hot encoded — because one-hot encoding would discard the "more vessels involved = more severe" ordering a plain scaled numeric value preserves. Grouped with the continuous features for imputation/scaling purposes. |
| **Ordinal categorical** | `slope` | A 3-level clinical grading (upsloping / flat / downsloping) with a documented clinical severity ordering. Kept as a single ordered numeric column (not one-hot encoded, to preserve the order), but imputed with `most_frequent` rather than `median` — a median of a 3-level *code* is not a clinically meaningful operation the way a median of a count is. |
| **Binary categorical** | `sex`, `fbs`, `exang` | Already 0/1-coded with no inherent scale. Imputed only; never encoded (nothing to encode — they're already binary) and never scaled (scaling a 0/1 indicator changes nothing about the information it carries, and keeping it as literal 0/1 is more directly interpretable downstream). |
| **Nominal categorical** | `cp`, `restecg`, `thal` | Unordered category codes. Chest pain type `1` (typical angina) is not "less than" type `4` (asymptomatic) in any numeric sense — treating these as an ordered numeric scale would invent a relationship that does not exist. One-hot encoded after imputation. |

This grouping matches `docs/DATASET.md` §3.1 and is encoded as **constants**, not prose — `get_feature_groups()` is called by every stage of the pipeline, so the classification is enforced in code, not just documented.

---

## 5. Categorical Encoding

Only the **nominal categorical** group (`cp`, `restecg`, `thal`) is one-hot encoded, via `sklearn.preprocessing.OneHotEncoder(drop='first', handle_unknown='ignore')` inside a per-group sub-`Pipeline`.

| Choice | Why |
|---|---|
| `drop='first'` | Drops one dummy column per feature to avoid perfect multicollinearity (the classic "dummy variable trap") — with `k` levels, `k-1` dummy columns fully determine the category. |
| `handle_unknown='ignore'` | If a category level appears in the test set that was never seen during training (not observed in Cleveland, but a real risk with a future, larger dataset — see `docs/DATASET.md` §12), the encoder produces an all-zero row instead of crashing. |
| Fit location | Inside the shared `ColumnTransformer`, fit exclusively on `X_train` (see Section 9). |

Binary features (`sex`, `fbs`, `exang`) need no encoding — they are already a valid 0/1 representation. Ordinal features (`slope`) and the discrete-numeric feature (`ca`) are deliberately **not** one-hot encoded, for the ordering reasons given in Section 4.

**Fit → transform discipline (verified by test):** the encoder's learned category list comes from `X_train` only; `X_test` is only ever passed through `.transform()`.

---

## 6. Numerical Scaling

Continuous and discrete-numeric features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`) and the ordinal feature (`slope`) are scaled with `sklearn.preprocessing.StandardScaler` by default (`PreprocessingConfig.scaler`, also supports `"minmax"` and `"robust"`).

**Why standardization by default:** several downstream classical models planned for Phase 3 (logistic regression, RBF-SVM) are scale-sensitive — a feature measured in the hundreds (cholesterol, ~126–564) would otherwise dominate a feature measured in single digits (oldpeak, ~0–6.2) purely because of its units, not its actual predictive relevance. `StandardScaler` centers each feature to mean 0, std 1, removing that artifact.

**The scaler is fit exclusively on `X_train`.** Its learned `mean_` and `scale_` (or `min_`/`max_` for MinMax, or median/IQR for Robust) come only from training rows; `X_test` is transformed using those training-derived statistics, never used to compute them. This is verified directly in `test_scaler_statistics_unaffected_by_test_set_values` and the dedicated leakage test (Section 13).

Binary features are **not** scaled (Section 4) — scaling a 0/1 indicator does not change any model's ability to use it, and leaving it as literal 0/1 keeps it directly interpretable in later explainability work.

---

## 7. Feature Selection

**Method:** `sklearn.feature_selection.SelectKBest`, scoring by `mutual_info_classif` (configurable to `f_classif`), selecting the top `k` features from the impute+encode+scale output.

**Why mutual information:** on a small clinical dataset, there is no reason to assume every risk factor relates to disease risk *linearly* — `mutual_info_classif` captures general (including non-linear) statistical dependence between a feature and the binary target, rather than assuming a straight-line relationship the way a correlation-based filter would. `f_classif` (ANOVA F-test) is offered as a simpler, linear-dependence alternative for comparison.

**Why this method is appropriate for this dataset's size:** with only 242 training rows and at most 17 encoded features, an expensive wrapper method (e.g. recursive feature elimination with nested cross-validation) is unnecessary machinery for the payoff — a fast univariate filter is proportionate to the problem size and is trivially reproducible given a fixed random seed (`mutual_info_classif`'s internal nearest-neighbour estimator is seeded via `PreprocessingConfig.random_seed`).

**Number of selected features:** configurable via `PreprocessingConfig.feature_selection_k` (default **10**, out of 17 encoded columns in the full feature set / 14 in the screening set). `k` is automatically clipped to however many encoded columns actually exist (relevant because one-hot encoding produces a data-dependent number of dummy columns) — this is handled by `SharedFeaturePipeline.fit()`, which fits the encoder first, *then* builds the selector against the real resulting width, specifically so sklearn never has to silently clip an over-large `k` itself.

**Fit → transform discipline:** the selector is fit on the encoded **training** matrix and the training labels only; the same selected column subset is then applied to the test matrix via `.transform()`.

**Limitations (stated plainly):**
- Mutual information estimates are noisier with fewer samples; on 242 training rows, the ranking is a reasonable guide, not a precise measurement — small changes in `k` or the random seed can shift which features are selected near the boundary.
- Univariate selection evaluates each feature independently and can miss features that are only useful in combination with another (an interaction effect).
- `k=10` is a reasonable starting default for a 13-feature dataset, not a value tuned against held-out performance — see the "do not optimize on the test set" note below.

**Explicit discipline followed here:** feature selection was **not** tuned by looking at test-set accuracy. `k=10` was chosen as a proportionate default (roughly 60–70% of the available encoded columns) before any model existed to evaluate against. Choosing `k` by repeatedly checking test performance would itself be a leakage channel — a slower, statistical one, but a real one — and this pipeline avoids it by design (`k` is a config value, and Phase 2 does not run any evaluation loop against it at all).

---

## 8. Dimensionality Reduction (PCA) — the Quantum-Ready Branch

Because a future quantum model (QSVM/VQC, Phase 4+) will encode each feature onto a qubit, and near-term quantum hardware supports only a handful of usable qubits, the selected feature matrix (10 columns by default) must be compressed into a small vector — implemented here with `sklearn.decomposition.PCA`.

**Why PCA specifically:** it is the simplest, most standard, most explainable dimensionality-reduction method available, it is unsupervised (so it cannot use the target to "cheat" the reduction), and its output is a linear combination of the input features, which keeps a path open to explaining a reduced component back in terms of original clinical features later (via the loading matrix) — a property a non-linear reducer would not offer as directly.

**`n_components` is a configuration value, not an assumption.** `PreprocessingConfig.pca_n_components` defaults to **4**, but this default is explicitly *not* claimed to be optimal — `tests/test_preprocessing.py` runs the full pipeline at every value from 2 to 6 to prove the pipeline is genuinely parametric, not hardcoded. The right value for a real experiment should be chosen by inspecting explained variance (below) and, later, by comparing downstream model performance across a small sweep — not by picking 4 because it is a common qubit count in tutorials.

**Explained variance at the default (n=4, Cleveland, full feature set, seed 42):**

| Component | Explained variance ratio |
|---|---|
| PC1 | 33.5% |
| PC2 | 21.6% |
| PC3 | 16.7% |
| PC4 | 10.3% |
| **Cumulative** | **82.0%** |

So at 4 components, roughly 18% of the variance present in the selected 10-feature representation is discarded. This is real information loss, and it is reported here rather than hidden — a smaller `n_components` will discard more; a larger one, less (up to the ceiling of `min(k, n_samples)`).

**PCA is fit exclusively on the training split's classical-ready output** — never on test data, never on the full dataset. `X_test_classical` is only ever projected through the training-fitted PCA axes via `.transform()`.

**Range normalization to `[0, π]`:** after PCA, an `sklearn.preprocessing.MinMaxScaler(feature_range=(0, π), clip=True)` maps the reduced vector into the domain a future angle-encoding quantum feature map will require. **No quantum library is imported or used anywhere in this pipeline** — this step only prepares numbers to be *ready* for that later phase.

**Why `clip=True` matters (a subtlety worth understanding):** the scaler's min/max are learned from the *training* PCA output only. A test-set point can legitimately fall slightly outside that training-derived range — that is not a bug, it is the expected and correct consequence of never fitting anything on test data. `clip=True` guarantees the quantum-ready output is *always* bounded within `[0, π]` regardless, without ever using test-set values to determine where that bound sits. This was caught directly during development: the pipeline initially raised a range-validation error on the test set for exactly this reason, and the fix was to clip the output, not to loosen the "never fit on test data" rule.

---

## 9. Two Outputs, One Shared Pipeline — Avoiding Duplicated Logic

`SharedFeaturePipeline` (in `pipeline.py`) performs impute → encode → scale → select **once**, fit on the training split. Both outputs are then derived from that single fitted object:

- **Classical-ready features** = `SharedFeaturePipeline.transform(X)` directly.
- **Quantum-ready features** = `SharedFeaturePipeline.transform(X)`, then PCA + range-normalize on top.

There is exactly one place in the codebase that imputes, encodes, or scales anything. The quantum branch cannot silently drift from the classical branch's preprocessing, because it is not a separate pipeline — it is a continuation of the same one. This directly satisfies the "avoid duplicating preprocessing logic" requirement and is what will let Phase 3's classical models and a later phase's quantum models be compared fairly on the same underlying data treatment.

`SharedFeaturePipeline` is implemented as an explicit two-stage Python object rather than a single bare `sklearn.Pipeline`, for one concrete reason: the feature-selection width (`k`) must be clipped against the *actual* post-encoding column count, which is only known after the encoder has been fit (one-hot encoding expands nominal columns into a data-dependent number of dummy columns). Pre-computing `k` before fitting is not possible; fitting the encoder and selector as two explicit, ordered steps is.

---

## 10. Reproducibility

Every value that could otherwise be hardcoded in `pipeline.py` lives in **one place**: `src/preprocessing/config.py::PreprocessingConfig`, with a matching `configs/preprocessing.yaml` for file-based overrides.

| Config field | Governs | Default |
|---|---|---|
| `random_seed` | Train/test split, mutual-information estimator, (any future stochastic step) | 42 |
| `test_size` | Split ratio | 0.20 |
| `stratify` | Whether the split is stratified on `target` | True |
| `numeric_impute_strategy` | Continuous/discrete-numeric imputation | median |
| `categorical_impute_strategy` | Ordinal/binary/nominal imputation | most_frequent |
| `scaler` | standard / minmax / robust | standard |
| `feature_selection_method` | mutual_info / f_classif | mutual_info |
| `feature_selection_k` | Number of selected features | 10 |
| `pca_n_components` | Quantum-ready dimensionality | 4 |
| `quantum_range_low` / `_high` | Quantum encoding bounds | 0, π |
| `feature_set` | full / screening | full |

`PreprocessingConfig` is an **immutable** (`frozen=True`) dataclass — a variant configuration is created via `.with_overrides(...)`, which returns a new object rather than mutating a shared one, so two experiments run in the same process can never accidentally share (and silently corrupt) each other's configuration.

**What "reproducible" means here, precisely, and how it's tested:** running `run_preprocessing_pipeline(df, config)` twice with an identical `config` produces bit-identical `X_train_classical`, `X_test_classical`, `X_train_quantum`, and `X_test_quantum` arrays, and an identical `split_id` — verified in `test_full_pipeline_is_reproducible_with_same_config`. Changing `random_seed` changes the split (and therefore everything downstream) — verified in `test_different_seed_changes_the_split`.

`compute_split_id()` produces a short deterministic fingerprint of `(n_rows, test_size, random_seed, stratify)`, so two runs can be compared for "were these built from the same split" without re-comparing every row.

---

## 11. `ca` / `thal` — the Clinical Feature Consideration

Carried forward from `docs/DATASET.md` §6 (issue 2) and §10 (limitation 5): `ca` (number of vessels colored by fluoroscopy) and `thal` (thallium stress test result) are both results of **specialist cardiac investigations** typically ordered once disease is already suspected. `ca` in particular sits close to a direct read-out of the angiographic diagnosis. A model that leans heavily on these two features may look excellent while overstating what is achievable in a genuine **early**-detection setting, where such specialist results may not yet be available.

**What Phase 2 does about this — preparation only, per the phase scope:**

- `get_feature_groups(feature_set="screening")` returns the same 5 groups with `ca` and `thal` removed, requiring **no other code change** anywhere in the pipeline.
- `PreprocessingConfig.feature_set` is a first-class config field (`"full"` or `"screening"`); running the entire pipeline under the screening configuration is a one-line change (`config.with_overrides(feature_set="screening")`).
- This is verified end-to-end in `test_screening_feature_set_excludes_ca_and_thal`: under the screening config, the resulting classical-ready and quantum-ready features contain no trace of `ca` or `thal` (including their one-hot-derived dummy columns).
- `ca` and `thal` are **not** dropped from the primary (`"full"`) dataset or pipeline path — both experiments (full vs. screening) remain available side by side, exactly as the phase instructions specify.

**What Phase 2 deliberately does NOT do:** no model has been trained, so there is no "Experiment A vs. Experiment B" comparison to report yet. That comparison is Phase 3+ work, once classical baselines exist to run under both configurations.

---

## 12. Leakage Prevention — Summary of Every Mechanism

| Mechanism | Where | What it prevents |
|---|---|---|
| Split before any `.fit()` call | `run_preprocessing_pipeline()` calls `split_dataset()` immediately after `create_binary_target()`, before any transformer is constructed | Any fitted statistic ever seeing a test row |
| Every transformer fit on `X_train` only | `SharedFeaturePipeline.fit()`, `build_quantum_pipeline(...).fit()` | Imputation medians/modes, scaler mean/std, encoder categories, selector scores, and PCA axes all being computed from training data exclusively |
| `.transform()`-only path for test data | `SharedFeaturePipeline.transform()`, quantum pipeline `.transform()` | Test data ever being used to (re)fit anything |
| `k` clipped from the *fitted* encoder's real width | `SharedFeaturePipeline.fit()` | A hardcoded `k` guess ever needing to peek at data shape before fitting |
| Immutable config (`frozen=True`) | `PreprocessingConfig` | One experiment's config silently mutating and contaminating another's results |
| Explicit synthetic-outlier leakage test | `tests/test_preprocessing.py::test_no_leakage_from_test_set_statistics` | Regression — proves an extreme synthetic test-set value (cholesterol = 999,999) does not shift the training-fitted scaler mean, imputer median, or PCA axes, because the test set is never part of any `.fit()` call |
| Raw-file immutability test | `tests/test_preprocessing.py::test_raw_file_unchanged_after_full_pipeline_run` | The pipeline ever writing to `data/raw/` |

**The one-sentence guarantee this pipeline provides:** *nothing that is fit anywhere in this module has ever seen a row that ends up in the test split.*

---

## 13. Known Limitations (Phase 2 scope)

| # | Limitation | Consequence |
|---|---|---|
| 1 | Small training set (242 rows) | Feature-selection rankings and PCA axes are estimated from a small sample and may be somewhat unstable to seed changes — this is why the pipeline exposes the seed as config rather than hiding it, so this instability is measurable rather than invisible. |
| 2 | `k=10` and `n_components=4` are unvalidated defaults | Neither was chosen by comparing downstream model performance (Phase 2 trains no models). They are documented starting points for Phase 3+ to sweep, not conclusions. |
| 3 | PCA is linear | A linear reduction cannot capture non-linear structure in the selected features; a non-linear reducer (e.g. Kernel PCA, an autoencoder) is a possible future extension, not attempted here. |
| 4 | Explained variance loss at low `n_components` | At the default of 4 components, ~18% of the (already reduced-by-selection) variance is discarded — a real cost of preparing data for a qubit-limited quantum branch, stated plainly in Section 8 rather than glossed over. |
| 5 | One-hot encoding assumes Cleveland's observed category levels | `handle_unknown='ignore'` prevents a crash on an unseen level in a future dataset, but a category that appears only in test data still cannot be *learned from* — this is an inherent, unavoidable property of any encoder fit on training data only, not a defect specific to this implementation. |
| 6 | Mutual-information feature selection is univariate | It cannot detect a feature that is only useful in combination with another; this is a known trade-off for using a fast, simple, reproducible method appropriate to a 242-row dataset (Section 7). |
| 7 | No cross-validation yet | Phase 2 produces a single train/test split. Repeated cross-validation (planned for Phase 3+ per `MVP_SPEC.md` §9) will give a more trustworthy estimate of how sensitive these pipeline choices are to which 242 rows happened to land in training. |

---

## 14. Traceability

| Claim in this document | Verified by |
|---|---|
| Target derivation rule and class counts | `tests/test_preprocessing.py::test_binary_target_created_correctly`, `test_binary_target_matches_documented_class_counts` |
| Split is reproducible / stratified | `test_split_is_reproducible_given_same_seed`, `test_split_is_stratified_on_target` |
| Missing values handled without pre-filling | `test_missing_values_present_in_raw_split`, `test_missing_values_are_imputed_after_shared_pipeline` |
| No NaNs in any output | `test_no_nans_anywhere_in_output` |
| Fit-on-train-only (imputer, scaler) | `test_imputer_statistics_computed_from_train_only`, `test_scaler_statistics_unaffected_by_test_set_values` |
| Transform without refit | `test_shared_pipeline_transform_does_not_refit` |
| PCA dimensionality is configurable, not hardcoded | `test_pca_output_has_requested_dimensionality[2..6]`, `test_pca_is_not_hardcoded_to_four_components` |
| Classical-ready / quantum-ready shapes | `test_classical_ready_output_shape`, `test_quantum_ready_output_shape` |
| Full-pipeline reproducibility | `test_full_pipeline_is_reproducible_with_same_config` |
| **No leakage from test-set statistics** | `test_no_leakage_from_test_set_statistics`, `test_full_pipeline_result_identical_regardless_of_downstream_test_perturbation` |
| Screening ablation prepared correctly | `test_screening_feature_set_excludes_ca_and_thal`, `test_full_feature_set_can_include_ca_and_thal` |
| Raw data untouched | `test_raw_file_unchanged_after_full_pipeline_run` |

This document should be treated as **derived from, and kept in sync with**, `src/preprocessing/`. If the pipeline's actual behavior ever disagrees with a number or claim here, the code and its test suite are authoritative and this document should be corrected.
