# Dataset Documentation — UCI Heart Disease (Cleveland)

**Status:** Phase 1 — Dataset Understanding and Validation
**Scope:** This document describes the raw dataset only. No preprocessing, imputation, scaling, feature selection, or modeling has been performed. See `PRD.md` §19 and `MVP_SPEC.md` §5 for how this dataset will be *used* in later phases.

---

## 1. Dataset Source

| Field | Value |
|---|---|
| Repository | UCI Machine Learning Repository — Heart Disease |
| Original directory | `data/raw/` (files copied verbatim from the UCI source, unmodified) |
| File used for MVP development | `data/raw/processed.cleveland.data` |
| Documentation source | `data/raw/heart-disease.names` (verified against, see §7 below) |
| Collecting institution | Cleveland Clinic Foundation, plus 3 other sites also present in `data/raw/` (Hungarian Institute of Cardiology, Budapest; V.A. Medical Center, Long Beach; University Hospital, Zurich, Switzerland) |
| Principal investigator (Cleveland/VA data) | Robert Detrano, M.D., Ph.D. |
| Original collection date | 1988 |
| SHA-256 of `processed.cleveland.data` | `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8` (recorded here for integrity tracking; not yet wired into an automated checksum gate — that is Phase 2, per `MVP_SPEC.md` §17.7) |
| Total raw attributes collected | 76 per patient (only 14 are used — see §3) |
| Patient identifiers | Removed at source and replaced with dummy values by the original data donor before public release |

Four sibling files exist in `data/raw/` for the same 76-attribute schema (`cleveland.data`, `hungarian.data`, `switzerland.data`, `long-beach-va.data`), plus their 14-attribute "processed" counterparts (`processed.hungarian.data`, `processed.switzerland.data`, `processed.va.data`). **Only `processed.cleveland.data` is used for MVP development.** The other three processed files are reserved for the optional cross-site robustness check described in `MVP_SPEC.md` §12 (check #10) — they are not touched in Phase 1.

---

## 2. Dataset Structure

| Property | Value |
|---|---|
| Rows (patients) | **303** |
| Columns | **14** (13 input features + 1 target) |
| Header row | **None** — column names are not present in the file and must be assigned from `heart-disease.names` |
| Delimiter | Comma |
| Encoding | Plain ASCII text |
| Missing-value marker | The literal string `?` |
| Duplicate rows | **0** (verified — see §6) |

The file `heart-disease.names` documents that only 14 of the 76 collected attributes are used in the "processed" files, and states explicitly:

> *"Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0)."*

This is the origin of the binary framing QuantumDx adopts (§4).

---

## 3. Every Feature

Column order in the raw file, cross-checked against `heart-disease.names` §7 ("Attribute Information"):

| # | Column | Full name / meaning | Type | Observed domain | Notes |
|---|---|---|---|---|---|
| 1 | `age` | Age in years | Numerical (continuous) | 29–77, mean 54.4, 41 unique values | No missing values |
| 2 | `sex` | Biological sex (1 = male, 0 = female) | Categorical (binary) | {0.0, 1.0} — 97 female, 206 male | No missing values |
| 3 | `cp` | Chest pain type: 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic | Categorical (nominal, 4 levels) | {1,2,3,4} | No missing values |
| 4 | `trestbps` | Resting blood pressure on admission (mm Hg) | Numerical (continuous) | 94–200, mean 131.7, 50 unique values | No missing values; no zero values in Cleveland |
| 5 | `chol` | Serum cholesterol (mg/dL) | Numerical (continuous) | 126–564, mean 246.7, 152 unique values | No missing values; no zero values in Cleveland (see §5.2 — this is **not** true of the other three site files) |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dL (1 = true, 0 = false) | Categorical (binary) | {0.0, 1.0} — 258 false, 45 true | No missing values |
| 7 | `restecg` | Resting electrocardiographic result: 0 = normal, 1 = ST-T wave abnormality, 2 = probable/definite left ventricular hypertrophy | Categorical (nominal, 3 levels) | {0,1,2} — 151 / 4 / 148 | No missing values; level 1 is rare (4 records) |
| 8 | `thalach` | Maximum heart rate achieved during exercise test | Numerical (continuous) | 71–202, mean 149.6, 91 unique values | No missing values |
| 9 | `exang` | Exercise-induced angina (1 = yes, 0 = no) | Categorical (binary) | {0.0, 1.0} — 204 no, 99 yes | No missing values |
| 10 | `oldpeak` | ST depression induced by exercise relative to rest | Numerical (continuous) | 0.0–6.2, mean 1.04, 40 unique values | No missing values; no negative values |
| 11 | `slope` | Slope of the peak exercise ST segment: 1 = upsloping, 2 = flat, 3 = downsloping | Categorical (ordinal, 3 levels) | {1,2,3} — 142 / 140 / 21 | No missing values |
| 12 | `ca` | Number of major vessels (0–3) colored by fluoroscopy | Numerical (discrete, ordinal) | {0,1,2,3} | **4 missing values**, encoded as `?` |
| 13 | `thal` | Thallium stress test result: 3 = normal, 6 = fixed defect, 7 = reversible defect | Categorical (nominal, 3 levels) | {3,6,7} — 166 / 18 / 117 | **2 missing values**, encoded as `?` |
| 14 | `num` | **Target.** Angiographic disease diagnosis, 0 = <50% diameter narrowing (no significant disease), 1–4 = >50% narrowing with increasing severity/vessel involvement | Integer, ordinal (raw) | {0,1,2,3,4} | No missing values |

### 3.1 Numerical vs categorical grouping used going forward

| Group | Columns |
|---|---|
| **Continuous numerical** | `age`, `trestbps`, `chol`, `thalach`, `oldpeak` |
| **Discrete numerical (ordinal count)** | `ca` |
| **Binary categorical** | `sex`, `fbs`, `exang` |
| **Nominal categorical** | `cp`, `restecg`, `thal` |
| **Ordinal categorical** | `slope` |

This grouping matches `PRD.md` §19.1 and `MVP_SPEC.md` §5.3, and is encoded as constants (`NUMERIC_COLUMNS`, `BINARY_CATEGORICAL_COLUMNS`, etc.) in `src/data/inspect_dataset.py` so it is machine-checkable, not just documented prose.

---

## 4. Target Definition

The raw target column is `num`: an integer 0–4 describing angiographic disease severity (see §3, row 14).

**QuantumDx's early-detection task is binary, not 5-class.** The planned derivation rule (to be implemented in the Phase 2 preprocessing pipeline, **not yet implemented**) is:

```
target = 1 if num > 0 else 0
```

| target | Meaning | Count in Cleveland |
|---|---|---|
| `0` | No significant coronary narrowing | 164 |
| `1` | Disease present at some severity (at risk) | 139 |

This mirrors the exact framing `heart-disease.names` itself recommends ("distinguish presence ... from absence"), so it is not an invented simplification — it is the dataset's own documented standard usage.

**This document reports the binary distribution for informational purposes only.** No binary column has been created or persisted anywhere in the codebase yet; `inspect_dataset.py` computes it in memory purely to print the preview in its report.

---

## 5. Missing Values

### 5.1 Declared missing values (Cleveland)

| Column | Missing count | Missing % | Encoding |
|---|---|---|---|
| `ca` | 4 | 1.32% | `?` |
| `thal` | 2 | 0.66% | `?` |
| All other 12 columns | 0 | 0% | — |

Total: **6 missing cells out of 4,242** (303 rows × 14 columns) — **0.14%** of all cells. This is a very low missingness rate; imputation strategy is not a significant modeling risk on Cleveland specifically (see §8 for why this does not generalize).

### 5.2 A missing-value pattern that does NOT appear in Cleveland but exists elsewhere in `data/raw/`

`chol` (serum cholesterol) has a physiologically impossible value of exactly `0` in some of the other UCI site files, which is a known **undeclared** sentinel-for-missing pattern (distinct from the `?` marker):

| File | Rows | `chol == 0` count | `trestbps == 0` count |
|---|---|---|---|
| `processed.cleveland.data` (**this file**) | 303 | **0** | **0** |
| `processed.switzerland.data` | 123 | **123 (100%)** | 0 |
| `processed.hungarian.data` | 294 | 0 | 0 |
| `processed.va.data` | 200 | 49 (24.5%) | 1 |

This was verified directly against the raw files, not assumed. It is documented here because it is the concrete evidence behind a design decision already made in `MVP_SPEC.md` §5.4: sentinel codes are declared **per column, per dataset, in config** — never hardcoded — precisely because different UCI sites encode "missing" differently. `src/data/inspect_dataset.py` already implements a generic zero-sentinel check (`ZERO_IS_SENTINEL_COLUMNS`) that correctly flags this pattern when pointed at the Switzerland or VA files, and correctly reports "none found" for Cleveland.

### 5.3 No other missingness patterns detected

No column in Cleveland has non-numeric junk values, out-of-domain categorical codes, or values outside the loose physiological plausibility bounds checked by `inspect_dataset.py` (`PHYSIOLOGICAL_RANGES`). See §7 for the full validation result.

---

## 6. Data Quality Issues

| # | Issue | Severity | Detail |
|---|---|---|---|
| 1 | Missing `ca`/`thal` values | Low | 6 cells total, 0.14% of data (§5.1) |
| 2 | **`ca` and `thal` are near-label features** | **Notable — carried forward as a design decision, not just an observation** | Both are results of specialized cardiac investigations (fluoroscopy, thallium scan) typically ordered *because* disease is already suspected. `ca` in particular is close to a direct read-out of the angiographic diagnosis. `PRD.md` §5.6 (Risk 2) and `MVP_SPEC.md` §5.6 already commit to reporting two feature sets — full (13 features) and screening (11 features, excluding `ca`/`thal`) — specifically because of this dataset property. No action needed in Phase 1; flagged here as the origin of that later design decision. |
| 3 | Small sample size | Notable | 303 records total; a locked 20% test split (per the planned strategy) is only ~61 records, which is why the downstream evaluation plan (`MVP_SPEC.md` §9) relies on repeated cross-validation and bootstrap confidence intervals rather than a single train/test split. |
| 4 | Rare `restecg` level | Low | Level 1 (ST-T wave abnormality) has only 4 records out of 303; any model or explanation involving this level should be read cautiously. |
| 5 | Duplicate rows | None found | 0 exact duplicates (all 14 columns identical) — verified, see §6.1 below. |
| 6 | Out-of-domain categorical values | None found | Every categorical column's observed values are a subset of the domain documented in `heart-disease.names` (§7 below). |
| 7 | Physiologically implausible numeric values | None found | No `chol`/`trestbps` zero-sentinels, and no numeric value fell outside the loose plausibility bounds checked (`age` 18–110, `trestbps` 60–260, `chol` 80–700, `thalach` 50–230, `oldpeak` 0–8). |
| 8 | Zero-variance columns | None found | Every feature has more than one distinct value. |
| 9 | Original 76-attribute source data | Not used | The un-processed 76-attribute files (`cleveland.data`, etc.) contain many attributes documented as "not used" or "irrelevant" in `heart-disease.names` (e.g. `restckm`, `exerckm`, `earlobe`, `lvx1`–`lvx4`). QuantumDx uses only the pre-selected, clinically-vetted 14-attribute "processed" files, consistent with the dataset's own documented standard usage. |

### 6.1 Duplicate row check

Verified by exact row comparison across all 14 columns: **0 duplicate rows** in `processed.cleveland.data`.

---

## 7. Verification Against Official UCI Documentation

Every structural and semantic claim in this document was cross-checked against `data/raw/heart-disease.names` (the official UCI documentation shipped with the dataset), not assumed from prior knowledge. Specifically:

| Claim | Verified against `heart-disease.names` | Verified against the raw file |
|---|---|---|
| 14 used attributes and their names/order | §7 "Attribute Information" (line ~110) | Column count = 14 ✓ |
| Column semantics (units, value codings) | §7 "Complete attribute documentation" (lines 126–236) | Observed value domains match exactly (§3, §7 table below) ✓ |
| Missing-value marker is `?` | §9 "Missing Attribute Values: Several. Distinguished with value -9.0." — **see discrepancy note below** | `?` found at 6 cell locations; no `-9.0` sentinel found anywhere in `processed.cleveland.data` |
| Cleveland instance count = 303 | §5 "Number of Instances" | Row count = 303 ✓ |
| Cleveland class distribution 164/55/36/35/13 | §10 "Class Distribution" | Exact match ✓ |
| Target binarization convention (presence 1-4 vs absence 0) | §4 "Relevant Information" | Applied as the documented, not invented, standard usage |

**Discrepancy noted and resolved:** `heart-disease.names` §9 states missing values are "distinguished with value -9.0", but the actual `processed.cleveland.data` file uses the literal string `?`, not `-9.0`. This is a known inconsistency in the original UCI documentation — the `.names` file describes the *general* 76-attribute Heart Disease database family, while the 14-attribute *processed* files (prepared later, specifically for ML use) use `?`. `-9.0` does not appear anywhere in `processed.cleveland.data`. `src/data/inspect_dataset.py` uses `?` as the missing marker because that is what is empirically present in the file being loaded, and the inspection report would surface any `-9.0`-coded values as an unexplained numeric outlier if they existed (they do not).

### 7.1 Categorical domain cross-check (per column)

| Column | Documented domain (`heart-disease.names`) | Observed domain (`processed.cleveland.data`) | Match |
|---|---|---|---|
| `sex` | {0, 1} | {0.0, 1.0} | ✓ |
| `cp` | {1, 2, 3, 4} | {1.0, 2.0, 3.0, 4.0} | ✓ |
| `fbs` | {0, 1} | {0.0, 1.0} | ✓ |
| `restecg` | {0, 1, 2} | {0.0, 1.0, 2.0} | ✓ |
| `exang` | {0, 1} | {0.0, 1.0} | ✓ |
| `slope` | {1, 2, 3} | {1.0, 2.0, 3.0} | ✓ |
| `ca` | {0, 1, 2, 3} | {0.0, 1.0, 2.0, 3.0} | ✓ |
| `thal` | {3, 6, 7} | {3.0, 6.0, 7.0} | ✓ |

Zero domain violations in any categorical column. This cross-check is implemented as `detect_domain_violations()` in `src/data/inspect_dataset.py` and re-runs automatically every time the dataset is inspected — it is not a one-time manual check.

---

## 8. Class Distribution

### 8.1 Raw target (`num`, 5 classes)

| `num` | Meaning | Count | % |
|---|---|---|---|
| 0 | No significant disease | 164 | 54.1% |
| 1 | Mild | 55 | 18.2% |
| 2 | Moderate | 36 | 11.9% |
| 3 | Severe | 35 | 11.6% |
| 4 | Most severe | 13 | 4.3% |

### 8.2 Binary target preview (`num > 0`)

| target | Meaning | Count | % |
|---|---|---|---|
| 0 | No disease | 164 | 54.1% |
| 1 | At risk | 139 | 45.9% |

**Minority class share: 45.9%** — this is a near-balanced binary problem. Per the imbalance threshold already defined in `MVP_SPEC.md` §5.5 (warn below 20% minority share), Cleveland does **not** trigger an imbalance warning, and the MVP design already reflects this: class weighting is planned, not SMOTE (`PRD.md` §19.9 / FR-D9; `MVP_SPEC.md` §5.5).

This is confirmed empirically, not assumed: `src/data/inspect_dataset.py` computes this distribution and would raise the imbalance warning automatically on a dataset where it applied (verified — running it against `processed.switzerland.data`, minority share 6.5%, does trigger the warning; see §9 below).

---

## 9. Reusability Check — Other Site Files

Running `src/data/inspect_dataset.py --path data/raw/processed.switzerland.data` (for verification purposes only; Switzerland is not part of MVP development) confirms the inspection logic generalizes correctly and independently rediscovers exactly the issues predicted in `PRD.md`/`MVP_SPEC.md`:

- `chol == 0` in all 123 rows → flagged as both a zero-sentinel and an out-of-range value, and correctly identified as a **zero-variance column** (every value is 0)
- Minority class share 6.5% (8 "no disease" vs 115 "at risk") → imbalance warning triggered
- No categorical domain violations

This is included here as evidence that the schema/constants in `inspect_dataset.py` are genuinely dataset-driven and not overfit to Cleveland's specific (clean) characteristics.

---

## 10. Known Limitations

| # | Limitation | Consequence |
|---|---|---|
| 1 | **Small sample size** (303 records) | Wide confidence intervals on any held-out test metric; a single train/test split is not trustworthy on its own — repeated CV is required (already planned in `MVP_SPEC.md` §5.8) |
| 2 | **Single-site data** | Cleveland Clinic Foundation, one hospital, one country, data collected in 1988. Patient population, clinical practice, and instrumentation do not represent a modern or geographically diverse population |
| 3 | **Historical data (1988)** | Diagnostic criteria, measurement instruments, and typical patient risk-factor profiles have changed materially since collection |
| 4 | **Only 13 candidate features** | Not high-dimensional. Does not exercise the "high-dimensional, noisy, complex biomedical data" scenario named in SIH26139's background, and is not intended to — see §11 |
| 5 | **`ca`/`thal` clinical-leakage risk** | Both features come from tests ordered after disease is already suspected; a model relying heavily on them may overstate *early*-detection performance (see §6, issue 2) |
| 6 | **No temporal or repeated-measure structure** | One record per patient, cross-sectional; cannot support any time-to-event or trajectory analysis |
| 7 | **No external validation cohort in Phase 1** | Cleveland is used for both development and (later) evaluation; the other three UCI sites are reserved for an optional cross-site robustness check, not full external validation |
| 8 | **Undocumented value semantics for some derived design choices** | e.g. exact clinical thresholds used later for feature engineering (age bands, BP categories) are guideline-based approximations, not something this dataset specifies |

---

## 11. Why This Dataset Is Appropriate for Initial Development

- **It is the standard, citable benchmark** for cardiovascular risk classification research — results are comparable to a large existing literature, which matters for sanity-checking the classical baselines (`MVP_SPEC.md` §7.1) before any quantum component is added.
- **It is small enough to make the pipeline itself verifiable.** With only 303 rows and 14 columns, every stage of the leakage-safe pipeline (split → impute → encode → scale → select → reduce) can be inspected by hand, and errors are easy to spot — exactly what is needed while proving out `test_leakage.py` and the other Phase 1/2 correctness guarantees.
- **It is clean.** As shown in §6, Cleveland specifically has 0 duplicate rows, 0 domain violations, 0 zero-variance columns, and only 6 missing cells (0.14%). This lets Phase 1–2 focus on pipeline correctness rather than fighting data-quality problems, while the inspection tooling (§9) is still built generically enough to handle messier sources.
- **It is tractable for the quantum branch.** 13 features reduce comfortably to the 4–8 qubit range planned for QSVM/VQC (`MVP_SPEC.md` §8.4) without needing an aggressive, information-destroying reduction step.
- **It matches the dataset's own documented intended use** — binary presence/absence classification (§4) — so QuantumDx's early-detection framing is not a reinterpretation of the data, it is the standard one.

## 12. Why This Should NOT Be Treated as the Final Validation Dataset

This point is treated as a hard constraint throughout `PRD.md` (§1.5) and `MVP_SPEC.md` (§5.9), and is restated here with the concrete evidence from this inspection:

- **303 records is not enough to validate a clinical-grade model.** It is enough to validate a *pipeline*.
- **It is not high-dimensional.** SIH26139's background motivates QML using "high-dimensional, noisy, complex biomedical data (e.g. genomics, medical imaging...)". 13 tabular features is none of those things at meaningful scale — Cleveland is a development and benchmark dataset, not a demonstration of the problem SIH26139 is ultimately motivated by.
- **It is single-site and decades old** (§10, limitations 2–3), so any accuracy/sensitivity/specificity number obtained here says nothing about generalization to a modern, multi-site cardiovascular population.
- **The near-label feature risk (`ca`/`thal`, §6 issue 2) is specific to how this dataset happens to be constructed** and may not exist, or may exist differently, in a larger dataset — it must be re-examined, not assumed, when a new dataset is introduced.
- **The architecture, not the dataset, is what is designed to scale.** Every schema element used by `inspect_dataset.py` (column names, types, sentinel columns, target rule) is a constant module list today and is planned to move to a YAML config in Phase 2 (`MVP_SPEC.md` §5.9), specifically so that introducing a larger cardiovascular dataset later requires a new config file, not a rewrite of this code.

**Concrete commitment:** any performance claim produced later in this project must be reported alongside "n≈303, single site, 1988, Cleveland only" — exactly as `MVP_SPEC.md` §12.1 and Appendix C already require.

---

## 13. Traceability

| This document's claim | Verified by |
|---|---|
| Shape, columns, dtypes | `src/data/inspect_dataset.py::inspect_dataset()` (automated, re-runs every time) |
| Missing value counts | `compute_missing_report()` |
| Duplicate count | `compute_duplicate_report()` |
| Categorical domain match | `detect_domain_violations()` |
| Zero-sentinel pattern (chol/trestbps) | `detect_zero_sentinels()` |
| Class distribution | `compute_target_distribution()` |
| All of the above, tested | `tests/test_dataset.py` (27 tests, see test run output) |

This document should be treated as **derived from, and kept in sync with**, `src/data/inspect_dataset.py`. If the inspection script's output ever disagrees with a number in this file, the script's live output is authoritative and this document should be corrected.
