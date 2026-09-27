# MVP Specification and Implementation Plan

## QuantumDx — Hybrid Quantum ML Platform for Early Cardiovascular Disease Detection

| Field | Value |
|---|---|
| **Document** | MVP Specification & Implementation Plan v1.0 |
| **Companion document** | `PRD.md` (full product requirements) |
| **SIH Problem Statement** | SIH26139 — Hybrid Quantum Machine Learning Platform for Early Disease Detection |
| **Event** | Smart India Hackathon 2026 |
| **Disease (Phase 1)** | Cardiovascular disease |
| **Task** | Early cardiovascular disease risk detection — binary classification |
| **Development dataset** | UCI Heart Disease (Cleveland) |
| **Quantum framework** | **Qiskit** (single framework — justified in Section 15.2) |
| **Stack** | Python · scikit-learn · XGBoost · Qiskit/Aer · FastAPI · React (Vite) |
| **Date** | 2026-09-03 |
| **Status** | Build specification — ready to execute |

---

## 1. Purpose and Reading Guide

### 1.1 What this document is

The PRD says **what the product must do**. This document says **exactly what we build, in what order, by whom, and how we know it is finished**. It is written so a developer can open it, pick a phase, and start typing without asking product questions.

### 1.2 The governing constraint

> **The MVP must be the smallest COMPLETE system that satisfies SIH26139 — not the largest system we can build in the time available.**

Complete means every link in the chain works: dataset → validation → leakage-safe preprocessing → feature engineering → selection → reduction → classical baselines → quantum model → comparison → explainability → robustness → prediction → dashboard. Smallest means each link is implemented once, simply, with no speculative generality.

### 1.3 The three rules that resolve every scope argument

| Rule | Consequence |
|---|---|
| **R1 — Depth loses to completeness.** | A working end-to-end chain with one feature map beats a half-built chain with six feature maps. If a phase is running late, simplify that phase; never drop a later phase. |
| **R2 — Every stage must be demonstrable in the dashboard.** | If a feature cannot be shown to a judge in the demo, it is not MVP. |
| **R3 — Honesty is a feature.** | No claim of quantum advantage without a confidence interval and a significance test. A null result, measured properly, is a passing outcome. |

### 1.4 Document map

| Section | Contents |
|---|---|
| 2–4 | MVP definition, scope boundary, SIH26139 coverage |
| 5 | Dataset specification (target, features, missingness, class balance, leakage, preprocessing, split strategy) |
| 6–12 | Pipeline, classical models, quantum models, evaluation, quantum evaluation, explainability, robustness |
| 13–15 | Dashboard, backend API, technology stack |
| 16–17 | Project structure, engineering practices (git, config, tracking, reproducibility, versioning) |
| 18 | Development Phases 0–12, each with objective / inputs / outputs / files / libraries / tasks / tests / DoD |
| 19 | Final MVP acceptance test (PASS/FAIL checklist) |
| 20–25 | MVP vs Advanced table, weekly roadmap, team allocation, dependency map, Definition of Done, demo checklist |

---

## 2. MVP Definition

### 2.1 One-sentence definition

> **The QuantumDx MVP is a locally runnable web application in which a user loads a cardiovascular dataset, the system validates it, runs a leakage-safe pipeline to a quantum-ready feature vector, trains three classical baselines and one quantum model (QSVM) on exactly the same processed data, compares them with confidence intervals and a significance test, explains both, stress-tests both, and scores an individual patient — all reproducible from a stored run ID.**

### 2.2 The seven MVP capabilities

| # | Capability | Minimum implementation | Not in MVP |
|---|---|---|---|
| 1 | **Data handling** | Load bundled UCI CSV + upload a conforming CSV; schema-driven validation report | Multiple modalities, EHR connectors, database |
| 2 | **Leakage-safe pipeline** | Split first, fit-on-train-only chain, automated leakage test | Grouped/temporal splits, drift detection |
| 3 | **Feature work** | 4–6 clinical derived features (toggleable), 2 selection methods, PCA reduction to *n* qubits | Wrapper selection, autoencoders, feature synthesis |
| 4 | **Models** | LR + RBF-SVM + XGBoost/RF (each on full-*d* and reduced-*n*), **QSVM (mandatory)**, VQC (secondary) | Quantum ensembles, adaptive feature maps, stacking |
| 5 | **Evaluation** | 10 metrics + confusion matrix + ROC/PR + repeated stratified CV + bootstrap CI + McNemar/DeLong + timing + quantum resource report | Nested CV, Bayesian comparison, hardware runs |
| 6 | **Explainability + robustness** | SHAP (classical), permutation importance + KernelSHAP (quantum), PCA loading map; CV variance, noise sweep, subgroup slices, generalization gap | Counterfactuals, quantum-native attribution, adversarial robustness |
| 7 | **Product surface** | FastAPI backend + React dashboard (7 pages) + run registry + docs | Auth, multi-user, cloud deployment, mobile |

### 2.3 MVP quantum scope — precise

| Item | MVP decision |
|---|---|
| Mandatory quantum model | **QSVM** — quantum kernel (fidelity) + classical `SVC(kernel='precomputed')` |
| Secondary quantum model | **VQC** — built in Phase 5; **droppable** if Phase 4 overruns (see Section 2.4) |
| Feature maps | `ZZFeatureMap` (default), `ZFeatureMap`, `PauliFeatureMap` — **standard, published maps only** |
| Adaptive / optimized feature maps | **NOT in MVP.** Advanced/future (Section 20) |
| Qubits | 4–8, default **6**; hard cap 10 |
| Encoding | Angle encoding via the feature map, inputs scaled to `[0, π]` |
| Execution | Qiskit Aer — statevector (default, exact) and sampling (shots) modes |
| Noise model | Optional, single simple depolarizing/readout model — SHOULD, not MUST |
| Real hardware | **Scoped validation run on IBM Quantum, executed and reported.** A small kernel submatrix (≈735 circuits) computed on a real QPU and compared against the simulator. **Not** a full hardware benchmark — quota-infeasible (§8.5). Transpilation report and budgets retained. |

### 2.4 The one droppable MVP item (contingency rule)

If, at the end of Week 6, QSVM is not producing evaluated results, **VQC is cut** and its time is spent on evaluation, explainability and the dashboard. The MVP remains valid with QSVM alone, because SIH26139 requires "quantum-enhanced learning models (such as QSVM, QNN, or VQC)" — one implemented model satisfies the statement; a broken chain does not. This is the only pre-authorized cut. Every other stage is mandatory.

---

## 3. MVP Scope Boundary

### 3.1 IN scope — build these

```
✔ UCI Heart Disease loader + CSV upload + YAML schema config
✔ Validation report (schema, dtypes, sentinels, missingness, class balance, duplicates, ranges, correlation)
✔ Stratified 80/20 hold-out split created BEFORE any fitting; indices persisted
✔ Preprocessing: sentinel→NaN, median/mode imputation, one-hot + ordinal encoding, StandardScaler
✔ Quantum range normalizer → [0, π]
✔ Feature engineering: 5 clinical derived features, toggleable
✔ Feature selection: mutual information + L1-logistic (ranked table)
✔ Dimensionality reduction: PCA (default) + top-k passthrough, n = 4–8
✔ Classical: LogisticRegression, SVC(RBF), XGBoost or RandomForest — tuned, on full-d AND reduced-n
✔ Quantum: QSVM (mandatory), VQC (secondary)
✔ Evaluation: 10 metrics, confusion matrix, ROC/PR curves, repeated stratified 5-fold × 3, bootstrap CIs
✔ Statistics: McNemar (accuracy), DeLong or bootstrap (AUC), verdict string
✔ Quantum resource report: qubits, depth, params, circuits, shots, backend, exec time
✔ Explainability: SHAP (classical), permutation importance + KernelSHAP (quantum), PCA loading map
✔ Robustness: CV variance, 3-seed repeat, Gaussian noise sweep, subgroup slices (sex, age band), generalization gap
✔ Single-patient prediction + batch CSV scoring
✔ FastAPI backend (9 endpoints)
✔ React dashboard (7 pages)
✔ Run registry (JSON) + model artifacts + reproducibility
✔ Docs: README, ARCHITECTURE, METHODS, LIMITATIONS, DEMO_SCRIPT, API reference
```

### 3.2 OUT of scope — explicitly do not build

```
✘ Database (PostgreSQL/Mongo)          → JSON files + filesystem are sufficient
✘ Docker/Kubernetes/cloud deployment   → local `uvicorn` + `npm run dev` is the demo
✘ Authentication / user accounts        → single-user local tool
✘ Microservices                         → one backend process
✘ Message queues / Celery               → FastAPI BackgroundTasks is enough
✘ MLflow server                         → local JSON run registry
✘ DVC / data lake                       → SHA-256 checksums + versioned folders
✘ LLMs / chatbots / RAG                 → no requirement in SIH26139
✘ Blockchain                            → no requirement
✘ Mobile app                            → web only
✘ FULL benchmark on real hardware        → quota-infeasible; scaled validation run only (§8.5)
✘ Live QPU execution during the demo      → queue times are unbounded; hardware results are pre-computed and cached
✘ Adaptive/optimized feature maps        → advanced/future
✘ Imaging / genomics / ECG waveforms     → future modality
✘ Multi-class or multi-disease           → binary CVD risk only
✘ Error mitigation, tensor networks, GPU sim → future
✘ Nested CV, Bayesian model comparison   → repeated CV + bootstrap is sufficient rigour
```

### 3.3 Complexity budget

| Metric | Target ceiling | Why |
|---|---|---|
| Python modules in `src/` | ≈ 30 files | Keeps the codebase readable by a 5-person team |
| Core library LOC | ≈ 3,500–5,000 | Achievable in 10 weeks alongside coursework |
| Backend endpoints | 9 | One per dashboard need, no more |
| Frontend pages | 7 | Exactly the required pages |
| Frontend components | ≈ 20 | Tables, cards, charts, forms — nothing bespoke |
| Third-party runtime deps | ≈ 15 | Every dependency is a version-conflict risk |
| Config files | 3 (schema, run, device profile) | One concept per file |

---

## 4. SIH26139 Requirement → MVP Coverage

| SIH26139 requirement | MVP implementation | Where it is demonstrated | Status |
|---|---|---|---|
| Hybrid quantum-classical architecture | Shared `ProcessedDataset`; classical stages feed a quantum kernel/ansatz; classical optimizer/SVC closes the loop | Home page architecture diagram; Model Comparison page | MUST |
| Biomedical disease detection | Cardiovascular risk, UCI Heart Disease, binary early-risk target | Dataset Analysis + Patient Prediction pages | MUST |
| Classical preprocessing | Sentinel handling, imputation, encoding, scaling — fit on train only | Dataset Analysis page → preprocessing summary | MUST |
| Feature engineering | 5 clinical derived features, toggleable, with formulas shown | Dataset Analysis page → engineered feature table | MUST |
| Feature selection | Mutual information + L1-logistic, ranked table | Dataset Analysis page → ranked feature table | MUST |
| Quantum-enhanced ML | QSVM (fidelity quantum kernel) + VQC | Quantum Info page; Experiment Results | MUST |
| Prediction | Single-patient form + batch CSV, both branches | Patient Prediction page | MUST |
| Explainability | SHAP (classical), permutation + KernelSHAP (quantum), PCA loadings, stated limitations | Explainability page | MUST |
| Performance evaluation | 10 metrics + confusion matrix + curves + CV mean/std | Model Comparison page | MUST |
| Comparison with classical baselines | Same split, same preprocessing, Protocol A and B, McNemar/DeLong verdict | Model Comparison page | MUST |
| Accuracy / Sensitivity / Specificity | Reported with bootstrap CIs for every model | Model Comparison page | MUST |
| Computational efficiency | Train time, inference latency, circuits executed, transpiled depth, shots | Model Comparison + Quantum Info pages | MUST |
| Generalization | Repeated stratified CV, 3 seeds, generalization gap, subgroup slices, noise sweep | Experiment Results page | MUST |
| Scalability | Config-driven dataset swap; documented O(N²) quantum bottleneck + subsampling guard | Docs + Quantum Info page resource estimate | MUST |
| Quantum simulator compatibility | Aer statevector + sampling, seeded, resource-guarded | Quantum Info page | MUST |
| Near-term hardware compatibility | `QuantumBackend` abstraction + transpilation report against a device profile + qubit/depth budgets | Quantum Info page → hardware readiness checklist | MUST |
| Comprehensive documentation | 6 documents + inline API docs + reproducible demo script | `docs/` + `/docs` Swagger | MUST |

**Coverage claim:** every listed SIH26139 requirement maps to a MUST-HAVE MVP feature with a specific demo location. Nothing in the problem statement is deferred.

---

## 5. Dataset Specification — UCI Heart Disease

### 5.1 Source and versioning

| Item | Value |
|---|---|
| Dataset | UCI Machine Learning Repository — Heart Disease (ID 45) |
| Primary file | `processed.cleveland.data` — 303 records, 14 attributes |
| Secondary files (robustness only) | `processed.hungarian.data`, `processed.switzerland.data`, `processed.va.data` |
| Stored at | `data/raw/uci_heart/v1/processed.cleveland.data` |
| Integrity | SHA-256 recorded in `data/raw/CHECKSUMS.txt`; verified on every load |
| MVP scope | **Cleveland only** for training and headline results. Other sites are used only for the optional cross-site robustness check (Phase 6, SHOULD). |

**Rule:** the raw file is never edited. Every transformation happens in code so it is reproducible.

### 5.2 Exact target variable

| Property | Specification |
|---|---|
| Raw column | `num` — integer 0–4 (angiographic disease severity: 0 = <50% narrowing; 1–4 = >50% narrowing, increasing vessel involvement) |
| MVP target | `target` — binary |
| Derivation rule | `target = 1 if num > 0 else 0` |
| Positive class | `1` = **at risk / disease present** |
| Clinical meaning of positive | Presence of significant coronary narrowing — the outcome we want to detect early |
| Why binarize | SIH26139 asks for early *detection*; severity grading is a different (multi-class/ordinal) problem and is out of MVP scope |
| Configured in | `configs/schema_uci_heart.yaml → target: {column: num, rule: "gt", threshold: 0, positive_label: 1}` |

**Requirement:** the binarization rule lives in config, never hardcoded, so a future dataset with a different label encoding needs no code change.

### 5.3 Input features (13 predictors)

| # | Feature | Type (MVP treatment) | Range / levels | Clinical meaning |
|---|---|---|---|---|
| 1 | `age` | **Numerical** (continuous) | ~29–77 years | Age in years |
| 2 | `sex` | **Categorical** (binary) | 1 = male, 0 = female | Biological sex; also a subgroup slice variable |
| 3 | `cp` | **Categorical** (nominal, 4 levels) | 1–4 (typical angina, atypical angina, non-anginal pain, asymptomatic) | Chest-pain type |
| 4 | `trestbps` | **Numerical** (continuous) | ~94–200 mm Hg | Resting blood pressure on admission |
| 5 | `chol` | **Numerical** (continuous) | ~126–564 mg/dL | Serum cholesterol |
| 6 | `fbs` | **Categorical** (binary) | 1 = >120 mg/dL, 0 otherwise | Fasting blood sugar |
| 7 | `restecg` | **Categorical** (nominal, 3 levels) | 0, 1, 2 | Resting electrocardiographic result |
| 8 | `thalach` | **Numerical** (continuous) | ~71–202 bpm | Maximum heart rate achieved during exercise test |
| 9 | `exang` | **Categorical** (binary) | 1 = yes, 0 = no | Exercise-induced angina |
| 10 | `oldpeak` | **Numerical** (continuous) | ~0.0–6.2 | ST depression induced by exercise relative to rest |
| 11 | `slope` | **Categorical** (ordinal, 3 levels) | 1 = upsloping, 2 = flat, 3 = downsloping | Slope of peak-exercise ST segment |
| 12 | `ca` | **Numerical** (discrete ordinal 0–3) | 0–3, `?` = missing | Number of major vessels coloured by fluoroscopy |
| 13 | `thal` | **Categorical** (nominal, 3 levels) | 3 = normal, 6 = fixed defect, 7 = reversible defect, `?` = missing | Thallium perfusion scan result |

**Type-handling summary for the implementation:**

| Group | Features | Encoding in MVP |
|---|---|---|
| Continuous numeric | `age`, `trestbps`, `chol`, `thalach`, `oldpeak` | Median impute → StandardScaler |
| Discrete ordinal numeric | `ca` | Median impute → treated as numeric → scaled |
| Binary categorical | `sex`, `fbs`, `exang` | Mode impute → kept as 0/1 (no one-hot) |
| Nominal categorical | `cp`, `restecg`, `thal` | Mode impute → **one-hot** (`drop='first'`, `handle_unknown='ignore'`) |
| Ordinal categorical | `slope` | Mode impute → ordinal codes preserved → scaled |

Resulting processed dimensionality *d* ≈ **18–22 columns** before feature engineering (depends on one-hot expansion and the engineering toggle). This is the "full-*d*" space used by classical Protocol B.

### 5.4 Missing values

| Column | Missing marker | Count in Cleveland (verify at load) | MVP handling |
|---|---|---|---|
| `ca` | `?` | ~4 records | `?` → NaN → **median** imputation (fit on train) |
| `thal` | `?` | ~2 records | `?` → NaN → **mode** imputation (fit on train) |
| `chol` | `0` (clinically impossible) | 0 in Cleveland; **frequent in the Switzerland subset** | `0` declared as a sentinel in config → NaN → median imputation |
| `trestbps` | `0` | 0 in Cleveland; present in other sites | Same sentinel treatment |
| All others | — | 0 | — |

**Implementation requirements**

1. Sentinel codes are declared **per column in the YAML config**, never hardcoded:
   ```yaml
   sentinels:
     ca:       ["?"]
     thal:     ["?"]
     chol:     [0]
     trestbps: [0]
   ```
2. Sentinels are converted to NaN **before any statistic is computed** (otherwise a `chol=0` corrupts the median).
3. Imputers are fitted on the **training split only** and reused verbatim at inference.
4. The validator prints the **actual observed counts** — the numbers above are expectations to be verified, not constants to trust.
5. Because total missingness in Cleveland is ~6 cells out of ~3,900, imputation strategy is **not** a significant modelling decision here. It becomes one when a larger, messier dataset is introduced — which is exactly why it is configurable now.

### 5.5 Class distribution

| Class | Meaning | Approximate share (Cleveland) |
|---|---|---|
| `0` (`num = 0`) | No significant disease | ≈ 54% (~164 records) |
| `1` (`num > 0`) | Disease present / at risk | ≈ 46% (~139 records) |

**Consequences for the MVP:**

| Observation | Decision |
|---|---|
| The dataset is **nearly balanced** (≈ 54/46) | **No SMOTE in the MVP.** Resampling adds a leakage risk and a variance source for no benefit at this balance. Use `class_weight='balanced'` on LR and SVM, `scale_pos_weight` for XGBoost — cheap, fold-safe, no resampling. |
| Accuracy is not badly misleading here | But it is still **never reported alone**; sensitivity and specificity are the clinical metrics (PRD §23). |
| Test set = 61 records | A single split gives ±6% swings from a handful of records. **Repeated CV and bootstrap CIs are mandatory, not optional.** |
| Positive class is the minority | Threshold analysis at 0.5 plus a sensitivity-targeted threshold chosen on training data only. |

**Validator requirement:** the actual class counts are computed at load time and displayed; if the minority share falls below 20% (which can happen with a user-uploaded dataset), the UI raises an imbalance warning and enables the SMOTE option — which is then applied **inside training folds only**.

### 5.6 Potential leakage — the four risks and their controls

This subsection is the methodological core of the project. Judges who know ML will probe exactly here.

#### Risk 1 — Preprocessing leakage (statistical)

| Aspect | Detail |
|---|---|
| Mechanism | Fitting an imputer, scaler, selector or PCA on the full dataset lets test-set statistics influence the training representation |
| Effect | Optimistically biased metrics; the classic reason published accuracies do not reproduce |
| **Control** | Split **first**. Every fitted object is fitted on `X_train` only, then `transform`-ed onto test. Inside CV, the whole chain is refitted per fold via `sklearn.pipeline.Pipeline`. |
| **Verification** | `tests/test_leakage.py` — fit the chain on train-only vs on train+test and assert that the fitted parameters (imputer statistics, scaler mean/scale, PCA components, selected feature set) are **identical to the train-only fit** for the train-only path, and that the pipeline never calls `.fit` on test indices (asserted by a spy/mock). |

#### Risk 2 — Clinical / label-adjacent leakage (the important one)

| Aspect | Detail |
|---|---|
| Mechanism | `ca` (vessels coloured by **fluoroscopy**) and `thal` (**thallium perfusion scan**) are results of specialised cardiac investigations. In practice these tests are ordered *because* the patient is already strongly suspected of coronary disease, and `ca` in particular is close to a direct read-out of the angiographic label. |
| Effect | A model leaning on `ca`/`thal` may look excellent while being useless for **early** detection — the framing SIH26139 explicitly asks for |
| **Control (MVP, MUST)** | Run and report **two feature sets**: <br>• **Full set** — all 13 features (comparable to published literature) <br>• **Screening set** — 11 features, excluding `ca` and `thal` (deployable for early risk stratification) |
| **How it appears** | A `feature_set: full \| screening` switch in the run config; both results shown side by side in the Model Comparison page, with a one-line explanation of why they differ |
| **Why this matters for judging** | It converts a hidden weakness of every UCI Heart Disease project into an explicit, well-reasoned experimental design choice. It costs one config flag and two extra runs. |

#### Risk 3 — Target-derivation leakage

| Aspect | Detail |
|---|---|
| Mechanism | Accidentally leaving `num` (or any derivative of it) in the feature matrix |
| **Control** | The schema config lists features explicitly by name. The validator asserts `target_column ∉ feature_columns` and that no feature has |correlation| > 0.98 with the target. Failing this is a **blocking** error. |

#### Risk 4 — Duplicate / near-duplicate records

| Aspect | Detail |
|---|---|
| Mechanism | Identical rows landing in both train and test inflate scores. Relevant when the multi-site files are concatenated. |
| **Control** | Exact-duplicate detection at validation; duplicates removed **before** splitting, with the count reported. |

#### Risk 5 (noted, not present here) — Temporal / grouped leakage

The dataset is cross-sectional with one record per patient and no timestamps, so no temporal or grouped split is required. The `Splitter` interface nonetheless accepts an optional `group_column`, so a future dataset with repeated measures per patient needs no rewrite. **Not implemented in MVP; interface only.**

### 5.7 Appropriate preprocessing (the MVP recipe)

Executed strictly in this order:

| Step | Operation | Fitted on | Config key |
|---|---|---|---|
| 1 | Load raw, apply column names, coerce dtypes | — | `schema.columns` |
| 2 | Replace sentinel codes with NaN | — (rules from config) | `schema.sentinels` |
| 3 | Drop exact duplicate rows | — | `preprocess.drop_duplicates` |
| 4 | Derive binary `target` from `num` | — | `schema.target` |
| 5 | Drop `ca`, `thal` if `feature_set = screening` | — | `run.feature_set` |
| 6 | **Stratified train/test split (80/20, seed)** | — | `split.*` |
| 7 | Impute numeric (median) | **train** | `preprocess.impute_numeric` |
| 8 | Impute categorical (most frequent) | **train** | `preprocess.impute_categorical` |
| 9 | Feature engineering (5 derived features) | **train** (bin edges) | `features.engineering.enabled` |
| 10 | One-hot encode nominal; ordinal-code ordinals | **train** | `preprocess.encoding` |
| 11 | StandardScaler on numeric columns | **train** | `preprocess.scaler` |
| 12 | Feature selection (MI or L1) → top-*k* | **train** | `features.selection.*` |
| 13 | Dimensionality reduction (PCA) → *n* components | **train** | `features.reduction.*` |
| 14 | MinMax → `[0, π]` (quantum range normalizer) | **train** | `quantum.encoding_range` |

Outputs of this recipe:

- `X_train_full`, `X_test_full` — after step 12 (*d*-dimensional, classical Protocol B)
- `X_train_reduced`, `X_test_reduced` — after step 14 (*n*-dimensional, `[0, π]`, used by **both** the quantum branch and classical Protocol A)

**No step after 6 ever sees test data during `fit`.** This single sentence is the leakage guarantee, and `tests/test_leakage.py` exists to prove it.

### 5.8 Appropriate train/test strategy

| Layer | Strategy | Parameters | Purpose |
|---|---|---|---|
| **Outer hold-out** | Stratified train/test split | `test_size = 0.20`, `seed = 42`, stratified on `target` | The test set (~61 records) is **locked** and touched exactly once, at final evaluation |
| **Model selection** | Stratified k-fold CV **on the training set only** | `k = 5` | Hyperparameter search — never sees the test set |
| **Variance estimation** | **Repeated** stratified k-fold on the training set | `k = 5`, `repeats = 3` (15 fits) | Mean ± std per metric — this is the headline "how good is it really" number |
| **Seed robustness** | Whole experiment repeated | `seeds = [42, 43, 44]` | Detects split-luck; reported as across-seed variance |
| **Uncertainty on the test set** | Bootstrap resampling of test predictions | 1000 resamples, percentile CI | 95% CIs on accuracy, sensitivity, specificity, ROC-AUC |
| **Model comparison** | Paired tests on the same test set | McNemar (accuracy), DeLong or paired bootstrap (AUC) | The verdict string |

**Rules (enforced in code):**

1. **The test set is not used for anything except the final evaluation.** No threshold tuning, no feature selection, no early stopping, no model choice.
2. **Both branches share `split_id`.** `BenchmarkEngine.compare()` raises if `split_id` or `preprocessing_config_hash` differ.
3. **Sensitivity-targeted thresholds are chosen on training-fold predictions**, then applied unchanged to the test set.
4. **Report the CV number as the headline**, and the hold-out number as the confirmation. With 61 test records, CV mean ± std is the more trustworthy estimate, and the document must say so.

### 5.9 Designing now for a larger dataset later

| Design choice made in the MVP | What it buys us when a larger CVD dataset arrives |
|---|---|
| Schema in YAML, not code | New dataset = new config file |
| Sentinels declared per column | Different missing-value conventions handled by config |
| Target derivation as a rule | Different label encodings handled by config |
| `group_column` accepted by `Splitter` (unused now) | Repeated-measures data needs no rewrite |
| Reduction target *n* is a parameter | Feature count grows; qubit count stays inside budget |
| QSVM sample cap + subsampling with a loud warning | The O(N²) kernel cost fails safely instead of hanging |
| Metrics/benchmark code is model- and dataset-agnostic | Same evaluation harness, new numbers |

**Honest statement required in the docs:** the MVP is validated on ~300 records. Scalability is an *architectural* property demonstrated by config-driven design and a documented bottleneck analysis — **not** an empirically validated one. Empirical validation is Phase 2 of the project (post-SIH).
---

## 6. MVP Pipeline Specification

### 6.1 Pipeline diagram with data contracts

```
  configs/schema_uci_heart.yaml + configs/run_mvp.yaml
                    │
                    ▼
┌───────────────────────────────────┐
│ 1. DatasetLoader                  │  out: DataFrame(303×14), dataset_hash
│    src/data/loader.py             │
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│ 2. DataValidator                  │  out: ValidationReport (blocking? warnings?)
│    src/data/validator.py          │       ── BLOCKS the run if errors exist
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│ 3. TargetBuilder + Splitter       │  out: y (binary), SplitIndices(train_idx, test_idx, split_id)
│    src/data/splitter.py           │  ◄── SPLIT HAPPENS HERE, BEFORE ANY .fit()
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│ 4. Preprocessor  (fit on train)   │  out: X_train_pp, X_test_pp, artifacts{imputers, encoders, scaler}
│    src/preprocessing/pipeline.py  │
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│ 5. FeatureEngineer (fit on train) │  out: +5 derived columns, formulas recorded
│    src/features/engineering.py    │
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│ 6. FeatureSelector (fit on train) │  out: X_*_full (d≈12–20), ranking table
│    src/features/selection.py      │
└───────────────┬───────────────────┘
                ├──────────────────────────────► X_train_full / X_test_full  (Protocol B classical)
                ▼
┌───────────────────────────────────┐
│ 7. Reducer + RangeNormalizer      │  out: X_*_reduced (n=6), values ∈ [0, π], explained_variance
│    src/features/reduction.py      │
└───────────────┬───────────────────┘
                ▼
        ProcessedDataset  ◄── the fairness contract (see 6.2)
                │
    ┌───────────┴────────────┐
    ▼                        ▼
┌─────────────────┐   ┌──────────────────────────────┐
│ 8. CLASSICAL    │   │ 9. QUANTUM                   │
│  LR / RBF-SVM / │   │  FeatureMap → FidelityKernel │
│  XGB|RF         │   │   → QSVM (SVC precomputed)   │
│  on full-d AND  │   │  FeatureMap → Ansatz → VQC   │
│  reduced-n      │   │  via QuantumBackend (Aer)    │
│ src/classical/  │   │  src/quantum/                │
└────────┬────────┘   └───────────┬──────────────────┘
         └─────────────┬──────────┘
                       ▼
┌───────────────────────────────────┐
│ 10. BenchmarkEngine               │  out: metrics, CIs, McNemar/DeLong, verdicts, timings
│     src/evaluation/               │       ── REFUSES to compare mismatched split_id
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│ 11. Explainer                     │  out: SHAP (classical), permutation+KernelSHAP (quantum),
│     src/explainability/           │       PCA loading map
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│ 12. RobustnessSuite               │  out: CV table, seed variance, noise curve, subgroups, gen-gap
│     src/evaluation/robustness.py  │
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│ 13. RunRegistry → results/runs/   │  out: run_record.json + artifacts + plots
│     src/tracking/registry.py      │
└───────────────┬───────────────────┘
                ▼
      FastAPI backend  ──►  React dashboard (7 pages)
                ▼
      InferenceService (single patient / batch CSV)
      src/inference/predictor.py
```

### 6.2 `ProcessedDataset` — the fairness contract

```python
@dataclass(frozen=True)
class ProcessedDataset:
    # identity — used to prove a fair comparison
    split_id: str                 # hash of (dataset_hash, test_size, seed, stratify)
    preprocess_hash: str          # hash of the preprocessing+features config
    seed: int
    feature_set: str              # "full" | "screening"

    # the two feature spaces
    X_train_full: np.ndarray      # (n_train, d)  — classical Protocol B
    X_test_full:  np.ndarray
    X_train_reduced: np.ndarray   # (n_train, n)  — quantum AND classical Protocol A
    X_test_reduced:  np.ndarray   #                 values in [0, π]

    y_train: np.ndarray
    y_test:  np.ndarray

    feature_names_full: list[str]
    feature_names_reduced: list[str]   # e.g. ["PC1", ..., "PC6"]

    artifacts: dict               # fitted imputers, encoder, scaler, engineer, selector, reducer, normalizer
    provenance: dict              # dataset_id, dataset_hash, validation_report_ref, warnings_ack
```

**Three invariants, asserted in code:**

| # | Invariant | Enforced by |
|---|---|---|
| I1 | Every model in a run receives the *same* `ProcessedDataset` instance | `ExperimentRunner` constructs it once |
| I2 | `BenchmarkEngine.compare(a, b)` raises `UnfairComparisonError` unless `a.split_id == b.split_id and a.preprocess_hash == b.preprocess_hash` | `src/evaluation/benchmark.py` |
| I3 | All values in `X_*_reduced` lie in `[0, π]` | `assert` in `RangeNormalizer.transform` |

### 6.3 Two comparison protocols (both mandatory)

| Protocol | Classical input | Quantum input | Question answered |
|---|---|---|---|
| **A — like-for-like** | `X_*_reduced` (n=6) | `X_*_reduced` (n=6) | Does the quantum kernel extract more from the *same* compressed representation than the RBF kernel does? **This is the scientifically clean comparison.** |
| **B — practical** | `X_*_full` (d≈18) | `X_*_reduced` (n=6) | Does the hybrid pipeline compete with the best classical pipeline we can build? |

Reporting only A would flatter the quantum branch (it cripples the classical models); reporting only B would flatter the classical branch. **Both are reported, labelled, and explained in one sentence each on the Model Comparison page.**

### 6.4 Run configuration (`configs/run_mvp.yaml`)

```yaml
run_name: mvp_baseline
seed: 42
seeds_multi: [42, 43, 44]

dataset:
  schema: configs/schema_uci_heart.yaml
  path: data/raw/uci_heart/v1/processed.cleveland.data
  feature_set: full            # full | screening  (screening drops ca, thal)

split:
  test_size: 0.20
  stratify: true

preprocess:
  drop_duplicates: true
  impute_numeric: median       # median | mean | knn
  impute_categorical: most_frequent
  scaler: standard             # standard | minmax | robust
  class_weight: balanced

features:
  engineering: {enabled: true}
  selection:   {method: mutual_info, k: 12}   # mutual_info | l1_logistic | none
  reduction:   {method: pca, n_components: 6} # pca | topk
  encoding_range: [0, 3.14159265]

classical:
  models: [logistic_regression, rbf_svm, xgboost]
  search: {type: grid, cv: 5, scoring: roc_auc}

quantum:
  backend: aer_statevector      # aer_statevector | aer_sampling
  shots: null                   # null => exact; else e.g. 1024
  models:
    qsvm:
      feature_map: ZZFeatureMap  # ZZFeatureMap | ZFeatureMap | PauliFeatureMap
      reps: 2
      entanglement: linear       # linear | circular | full
      C_grid: [0.1, 1, 10]
      max_train_samples: 300
    vqc:
      enabled: true              # the one droppable MVP item
      ansatz: RealAmplitudes
      ansatz_reps: 2
      optimizer: COBYLA
      maxiter: 200
      restarts: 3

evaluation:
  cv: {k: 5, repeats: 3}
  bootstrap: 1000
  threshold_policy: [0.5, sensitivity_target_0.90]

robustness:
  noise_sigmas: [0.0, 0.01, 0.05, 0.10, 0.20]
  subgroups: [sex, age_band]

hardware_profile: configs/device_profile_generic27q.yaml
```

**Rule:** anything a developer might want to change during experiments lives here. If you find yourself editing Python to change an experiment, that value belongs in this file.

---

## 7. Classical Models Specification

### 7.1 The three baselines

| # | Model | sklearn / lib class | Role | Hyperparameter grid (MVP) |
|---|---|---|---|---|
| 1 | **Logistic Regression** | `sklearn.linear_model.LogisticRegression` | Interpretable linear baseline; the clinical standard | `C ∈ {0.01, 0.1, 1, 10}`, `penalty ∈ {l1, l2}`, `solver='liblinear'`, `class_weight='balanced'`, `max_iter=2000` |
| 2 | **RBF-SVM** | `sklearn.svm.SVC(kernel='rbf', probability=True)` | **The direct classical counterpart to QSVM** | `C ∈ {0.1, 1, 10, 100}`, `gamma ∈ {'scale', 0.01, 0.1, 1}`, `class_weight='balanced'` |
| 3 | **XGBoost** (fallback: RandomForest) | `xgboost.XGBClassifier` / `sklearn.ensemble.RandomForestClassifier` | Strong non-linear tabular baseline — the realistic performance ceiling | XGB: `n_estimators ∈ {100,300}`, `max_depth ∈ {2,3,4}`, `learning_rate ∈ {0.05,0.1}`, `subsample=0.8`, `scale_pos_weight` from train balance |

**Why RBF-SVM is structurally the most important baseline:** QSVM is `SVC(kernel='precomputed')` fed a quantum kernel. RBF-SVM is `SVC(kernel='rbf')` fed a classical kernel. Everything else — data, split, C-grid, class weights, metrics — is held constant. **Any measured difference between them is attributable to the kernel and nothing else.** This is the single cleanest scientific statement the project can make, and it is why CLF-2 is non-negotiable.

**XGBoost vs RandomForest decision rule:** use XGBoost if it installs cleanly on all team machines in Phase 0. If any member hits a build/wheel problem, the whole team switches to `RandomForestClassifier` — it is in scikit-learn, needs no extra dependency, and performs comparably on ~300 rows. Decide once, in Phase 0, and record it in `docs/DECISIONS.md`. Do not carry both.

### 7.2 Training requirements

| ID | Requirement |
|---|---|
| CL-1 | Each model is trained **twice**: on `X_*_full` (Protocol B) and on `X_*_reduced` (Protocol A). Six classical result rows total. |
| CL-2 | Hyperparameter search via `GridSearchCV(cv=StratifiedKFold(5), scoring='roc_auc')` on the **training set only** |
| CL-3 | The search runs inside a `sklearn.pipeline.Pipeline` so any remaining fitted step is refit per fold |
| CL-4 | All models expose `predict_proba`; SVC uses `probability=True` (Platt scaling, fitted internally on training folds) |
| CL-5 | Class imbalance handled by weights, **not** resampling (see §5.5) |
| CL-6 | `random_state=seed` on every stochastic component |
| CL-7 | Best params, CV score, fit time and inference latency recorded in the run record |
| CL-8 | Model persisted with `joblib` to `results/runs/<run_id>/models/<model_id>.joblib` |
| CL-9 | Every classical model implements the shared `Model` interface (§7.3) — identical to the quantum models |

### 7.3 The shared `Model` interface

```python
class Model(Protocol):
    model_id: str          # e.g. "rbf_svm__reduced_n"
    branch: str            # "classical" | "quantum"
    feature_space: str     # "full_d" | "reduced_n"

    def fit(self, X, y) -> "Model": ...
    def predict(self, X) -> np.ndarray: ...
    def predict_proba(self, X) -> np.ndarray: ...        # shape (n, 2)
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path) -> "Model": ...
    def describe(self) -> dict: ...                       # type, hyperparams, branch
    def resource_report(self) -> dict: ...                # classical: {fit_seconds, n_params}
                                                          # quantum: {qubits, depth, n_trainable_params,
                                                          #           circuits_executed, shots, backend,
                                                          #           quantum_seconds}
```

This one Protocol is what lets `BenchmarkEngine`, `Explainer`, `RobustnessSuite` and the dashboard treat both branches identically. **Write it in Phase 2 and never change it.**

---

## 8. Quantum Models Specification

Framework: **Qiskit** (single framework for the whole project — full justification in §15.2).

### 8.1 QSVM — mandatory first quantum model

| Aspect | MVP specification |
|---|---|
| Algorithm | Quantum kernel estimation + classical SVM |
| Kernel | `K(x_i, x_j) = \|⟨φ(x_i)\|φ(x_j)⟩\|²` — fidelity between feature-mapped states |
| Implementation | `qiskit_machine_learning.kernels.FidelityQuantumKernel` (or an explicit compute-uncompute circuit, if the API shifts) |
| Classifier | `sklearn.svm.SVC(kernel='precomputed', probability=True)` |
| Feature map | `ZZFeatureMap(feature_dimension=n, reps=2, entanglement='linear')` by default; `ZFeatureMap` and `PauliFeatureMap` selectable |
| Qubits | `n_qubits = n_components` (6 by default) — one qubit per reduced feature |
| Input | `X_*_reduced`, values in `[0, π]` |
| Trainable quantum parameters | **Zero.** The feature map is fixed; only the classical SVC (`C`, dual coefficients, support vectors) is learned. This must be stated explicitly in the docs and the Quantum Info page — it is a common misconception that QSVM "trains a circuit". |
| Hyperparameter search | `C ∈ {0.1, 1, 10}` via CV on the precomputed **training** kernel block |
| Sample cap | `max_train_samples = 300`; above this, subsample with a loud UI warning (kernel cost is O(N²)) |
| Caching | Kernel matrix cached at `results/cache/kernels/<hash>.npy`, key = SHA-256 of (reduced-data bytes, feature map, reps, entanglement, backend, shots) |

**Kernel computation cost (Cleveland, concrete):**

| Quantity | Value |
|---|---|
| Train samples (80% of 303) | 242 |
| Test samples | 61 |
| Train/train kernel evaluations (symmetric, unit diagonal) | 242 × 241 / 2 = **29,161** |
| Test/train kernel evaluations | 61 × 242 = **14,762** |
| Total circuit evaluations | ≈ **43,923** |
| Expected wall clock, 6 qubits, Aer statevector, batched | **≈ 1–8 minutes** on a laptop |

This is comfortably inside a demo budget — which is precisely why 6 qubits and ~300 samples were chosen.

**Implementation checklist:**

```
[ ] Build feature map from config
[ ] Wrap Aer backend behind QuantumBackend
[ ] Compute K_train (symmetric; exploit K[i,i]=1 and K[i,j]=K[j,i])
[ ] Compute K_test (test rows × train columns)
[ ] Cache both, keyed by config hash
[ ] Fit SVC(kernel='precomputed', probability=True) with C-grid CV
[ ] Emit resource_report(): qubits, transpiled depth, circuits, shots, backend, seconds
[ ] Emit diagnostics: mean/std of off-diagonal K entries (kernel concentration), eigenspectrum, rank
```

**Kernel concentration diagnostic (SHOULD, high value):** if the off-diagonal entries of `K_train` cluster tightly around a constant, the kernel has lost discriminative power and the SVM degenerates. Report `mean_offdiag` and `std_offdiag`; warn when `std_offdiag < 0.01`. Being able to say *"we detected kernel concentration at 8 qubits with reps=3, which is why we report 6 qubits"* is a far stronger jury answer than an unexplained mediocre accuracy.

### 8.2 VQC / QNN — secondary MVP model

| Aspect | MVP specification |
|---|---|
| Structure | `ZZFeatureMap(n, reps=2)` → `RealAmplitudes(n, reps=2)` → measurement → parity/expectation → probability |
| Implementation | `qiskit_machine_learning.algorithms.VQC` (or `NeuralNetworkClassifier` over an `EstimatorQNN`) |
| Optimizer | `COBYLA(maxiter=200)` default; `SPSA` when running in sampling mode |
| Trainable parameters | `RealAmplitudes(num_qubits=6, reps=2)` → `6 × (2+1)` = **18 parameters** (report the actual number from `ansatz.num_parameters`) |
| Initialization | Seeded uniform; 3 random restarts, best selected by **training-fold** score |
| Outputs | Loss curve per iteration; final parameters persisted as `.npy` + JSON config |
| Diagnostics | Loss curve plot; gradient/parameter-update variance as a barren-plateau smoke test |
| Time budget | Hard cap 15 minutes per fit; on timeout, stop early and record a partial result rather than hanging the run |

**Contingency (restated):** if QSVM is not producing evaluated results by end of Week 6, VQC is cut. See §2.4.

### 8.3 The `QuantumBackend` abstraction — how near-term hardware compatibility is proven

```python
class QuantumBackend(Protocol):
    name: str
    def run(self, circuits: list[QuantumCircuit], shots: int | None) -> Results: ...
    def capabilities(self) -> dict:   # {max_qubits, supports_shots, basis_gates, coupling_map|None, noise_model|None}
        ...
```

| Implementation | Status in MVP | Purpose |
|---|---|---|
| `AerStatevectorBackend` | **Built** | Exact, deterministic, fast — the default and the demo backend |
| `AerSamplingBackend` | **Built** | Shot-based path; proves the sampling code path works before hardware ever exists |
| `AerNoisyBackend` | SHOULD | One simple noise model (depolarizing + readout error) for the robustness page |
| `IBMRuntimeBackend` | **Built and executed (scoped)** | Real QPU access via `qiskit-ibm-runtime` `SamplerV2` in batch/job mode. Used for the hardware validation experiment in §8.5 — **never** for the headline benchmark, and **never live during the demo** |

**The hardware-compatibility argument the MVP makes** (this part holds even if QPU access fails on the day):

1. Circuits are transpiled against a realistic device profile (`configs/device_profile_generic27q.yaml`: basis gates, coupling map, 27 qubits).
2. A **transpilation report** is produced and shown: qubits required, transpiled depth, 2-qubit gate count, SWAP overhead.
3. Budgets are enforced: qubits ≤ profile size, transpiled depth ≤ configured limit, with a warning and concrete remedies when exceeded.
4. The shot-based execution path is exercised on the simulator, so switching to hardware is a **config change**, not a code change.

That is a complete, honest, demonstrable compatibility claim, and it costs roughly 150 lines of code. **Because we also have IBM Quantum cloud access, we go one step further and actually execute on a QPU — at a deliberately small scale (§8.5). Keep both: the transpilation argument is the fallback if hardware access fails during demo week, and it is what generalizes beyond IBM.**

### 8.4 Quantum resource budget (enforced in code)

| Parameter | Default | Hard cap | Enforcement |
|---|---|---|---|
| Qubits | 6 | 10 | `ConfigError` above cap; warning above 8 |
| Feature-map reps | 2 | 4 | Warning above 2 (depth) |
| Ansatz reps | 2 | 4 | Warning above 2 (barren plateaus) |
| Shots | `null` (exact) | 8192 | — |
| QSVM train samples | 300 | 500 | Subsample + prominent warning |
| Transpiled depth | — | warn > 200 | Hardware-readiness checklist |

**Pre-flight requirement:** before any quantum run starts, the system prints/displays an estimate — circuits to execute, estimated wall clock, memory, **and (for hardware backends) estimated QPU seconds against the remaining quota** — and requires confirmation above a threshold. No user should ever discover a 3-hour run, or a blown monthly quota, by waiting for it.

### 8.5 Real Hardware Validation on IBM Quantum

We have IBM Quantum cloud access, so the MVP **executes on real hardware**. This subsection defines exactly what we run there and — more importantly — what we deliberately do **not**.

#### 8.5.1 Why the full benchmark cannot run on hardware

| Quantity | Value |
|---|---|
| Full QSVM kernel (Cleveland, 242 train / 61 test) | ≈ **43,900 circuits** |
| Shots per circuit | 1,024 |
| Total shots | ≈ 45 million |
| Estimated QPU time at typical device repetition rates | **≈ 3–4 hours** |
| IBM Quantum **Open (free) plan** allowance | **on the order of 10 minutes per month** |
| Shortfall | **≈ 20–40× over a full monthly quota, for a single run** |

Now multiply that by what a credible benchmark actually requires: repeated 5-fold × 3 CV (15 refits), 3 seeds, and several feature-map configurations. A hardware benchmark is **two to three orders of magnitude outside the free tier**, and paid QPU time is billed per second — not a student budget.

**Conclusion (and it is a design decision, not a limitation we are apologising for): the headline benchmark stays on the simulator.** You cannot fairly compare two models when one of them is allotted ten noisy minutes a month. Hardware is used for *validation of the quantum path*, which is a different and cheaper question.

#### 8.5.2 What we DO run on hardware

**The hardware validation experiment.** Take a fixed subsample of the same `ProcessedDataset`, compute the **same kernel submatrix** three ways, and compare.

| Item | Value |
|---|---|
| Subsample | **30 training + 10 test** records, stratified, seeded, fixed across all three backends |
| Circuits | K_train `30×29/2 = 435` + K_test `10×30 = 300` = **735 circuits** |
| Shots | 1,024 (sweep 256/1024/4096 only if quota permits) |
| Estimated QPU time | **≈ 3–4 minutes** — fits the free monthly allowance with room for one retry |
| Backends compared | `aer_statevector` (exact) · `aer_noisy` (device noise model) · **`ibm_<device>` (real QPU)** |
| Qubits | 6 (unchanged) |
| Feature map | `ZZFeatureMap`, reps 2, linear (unchanged — one configuration only) |

#### 8.5.3 What we report from it

| Output | Description | Why it is worth the quota |
|---|---|---|
| **Three kernel heatmaps** | The same 30×30 matrix under exact / noisy-sim / hardware | Immediately legible; shows noise degrading kernel structure |
| **Element-wise error distribution** | Histogram and summary of \|K_hw − K_exact\|: mean absolute error, max, correlation | A single number quantifying the simulator-to-hardware gap |
| **Diagonal fidelity check** | `K(x,x)` should be exactly 1; on hardware it is not | The cleanest, most honest noise indicator available — and it needs no ground truth |
| **Kernel concentration on hardware vs simulator** | Off-diagonal mean/std for each backend | Shows whether device noise flattens the kernel |
| **Downstream accuracy on the subsample** | Fit `SVC(precomputed)` on each of the three kernels, evaluate on the same 10 test points | The end-to-end consequence — reported with the loud caveat that n=10 is far too small for an accuracy claim |
| **Job provenance** | Backend name, IBM job IDs, calibration timestamp, queue wait, QPU seconds consumed | Verifiable, and it is what makes the run a *result* rather than an anecdote |

#### 8.5.4 Implementation requirements

| ID | Requirement | Priority |
|---|---|---|
| HWV-1 | `IBMRuntimeBackend` implements the same `QuantumBackend` protocol — switching to hardware is a **config change only** | MUST |
| HWV-2 | Credentials read from an environment variable / local `.env` (git-ignored). **Never committed.** `.env.example` documents the required keys | MUST |
| HWV-3 | Backend selection, device name, shots, job IDs, calibration timestamp and QPU seconds recorded in the run record | MUST |
| HWV-4 | Pre-flight quota estimate displayed and confirmed before any hardware submission | MUST |
| HWV-5 | Batch/job submission with polling and a bounded timeout; a queued job never blocks the pipeline indefinitely | MUST |
| HWV-6 | Hardware results are **cached to disk on first retrieval** and loaded from cache thereafter — a hardware run is executed **once**, not per demo | MUST |
| HWV-7 | Every hardware result view states: device, date, shots, subsample size, and that it is a validation run, not a benchmark | MUST |
| HWV-8 | The pipeline runs to completion with hardware **unavailable** — hardware is strictly additive | MUST |
| HWV-9 | Transpilation happens against the **actual device** coupling map when a hardware backend is selected | MUST |
| HWV-10 | Error mitigation (readout mitigation, ZNE) | FUTURE — do not attempt in MVP |
| HWV-11 | Full CV or full-kernel hardware execution | FUTURE (needs a paid plan) |

#### 8.5.5 Things to verify on Day 1 of Phase 3 (do not assume)

IBM's platform, plan structure and quotas have changed repeatedly. **One person verifies all of the following in the first hour and records the answers in `docs/DECISIONS.md`:**

- [ ] Which IBM Quantum platform/account we are on, and the exact auth flow (API key, instance/CRN, `QiskitRuntimeService` save-account arguments)
- [ ] The **exact** current free-tier allowance (minutes/month, resets when) — our 735-circuit design assumes ~10 min/month; re-scale the subsample if it differs
- [ ] Whether our plan supports **Session** mode or only **batch/job** mode (batch is what our design needs; do not build around Sessions)
- [ ] Which devices we can access, their qubit counts, basis gates and coupling maps → feeds `configs/device_profile_*.yaml` (replace the generic 27q profile with the **real** device)
- [ ] Current queue times at the hours we plan to submit
- [ ] The pinned `qiskit-ibm-runtime` version and the primitive API (`SamplerV2` interface) that version exposes

#### 8.5.6 New risks introduced by using real hardware

| Risk | Impact | Mitigation |
|---|---|---|
| **Queue time is unbounded** (minutes to hours) | A live demo could stall indefinitely | **Hardware is never executed live.** Results are pre-computed in Week 9 and loaded from cache (HWV-6) |
| **Hardware results are not reproducible** — calibration drifts daily | Breaks the reproducibility guarantee (NFR-5) | Reproducibility is claimed for the **statevector** backend only. Hardware runs record device + calibration timestamp + job IDs, and the docs state plainly that they are not bit-reproducible |
| **Quota exhausted mid-project** | No hardware result at all | Budget the quota: one rehearsal run, one final run, one spare. Do not sweep configurations on hardware |
| **Credentials leaked to git** | Account compromise | `.env` git-ignored, `.env.example` committed, secret-scan in the honesty/security audit (Phase 11) |
| **Demo laptop is offline** (AC-43) | Hardware page appears broken | Cached hardware results are served from disk, so the page works offline. **Verify this explicitly in the offline rehearsal.** |
| **IBM API changes between now and the demo** | `src/quantum/backends_ibm.py` breaks | Exact version pins; the `QuantumBackend` abstraction confines the blast radius to one file; the simulator path is unaffected |
| **Data leaving the machine** | Privacy question from a judge | Only the **reduced, PCA-transformed, 6-dimensional feature vectors of a public benchmark dataset** are encoded into circuit parameters. No raw records, no identifiers. State this on the Quantum Info page |

#### 8.5.7 What this buys us with the judges

Executing on real hardware moves three claims from "designed for" to "demonstrated":

1. *"Compatible with near-term quantum hardware"* → **we ran on it**, here is the device and the job ID.
2. *"Simulator results may differ on hardware"* → **we measured the gap**, here is the error distribution.
3. *"We understand the constraints"* → **here is the quota arithmetic** showing why the benchmark stays on the simulator, which is a more sophisticated answer than an unexplained hardware number.

Point 3 is the one that separates a team that used hardware from a team that understands hardware.

---

## 9. Model Evaluation Specification

### 9.1 Metric suite — every metric the MVP must report

| Metric | Formula / source | Reported for |
|---|---|---|
| **Accuracy** | (TP+TN)/N | every model |
| **Precision (PPV)** | TP/(TP+FP) | every model |
| **Recall = Sensitivity** | TP/(TP+FN) | every model — **primary clinical metric** |
| **Specificity** | TN/(TN+FP) | every model |
| **NPV** | TN/(TN+FN) | every model |
| **F1-score** | 2·P·R/(P+R) | every model |
| **Balanced accuracy** | (Sens+Spec)/2 | every model |
| **MCC** | Matthews correlation | every model |
| **ROC-AUC** | `roc_auc_score` | every model |
| **PR-AUC** | `average_precision_score` | every model |
| **Confusion matrix** | TN, FP, FN, TP | every model |
| **Brier score** | mean((p−y)²) | every model (calibration) |
| **Training time** | wall clock, seconds | every model |
| **Inference time** | ms per record (mean over test set) | every model |

Note: *recall* and *sensitivity* are the same quantity. The MVP reports it under both names because clinicians read "sensitivity" and ML reviewers read "recall", and both appear in the SIH requirement list.

### 9.2 How each number is produced (this is the part that must not be sloppy)

| Reported figure | Source | Note |
|---|---|---|
| **Headline metric** | Repeated stratified 5-fold × 3 repeats on the **training set** → mean ± std | With 61 test records, this is the trustworthy estimate. The document and the UI must say this. |
| **Hold-out metric** | Single evaluation on the locked test set | Confirmation, reported with bootstrap 95% CI |
| **Uncertainty** | 1000-resample bootstrap of the test predictions, percentile CI | Accuracy, sensitivity, specificity, ROC-AUC |
| **Across-seed variance** | Whole experiment repeated for seeds 42/43/44 | Detects split luck |
| **Model-vs-model comparison** | Paired test on the same test predictions | McNemar (accuracy/errors), DeLong or paired bootstrap (AUC) |

### 9.3 The comparison verdict (anti-overclaim mechanism)

`BenchmarkEngine.compare(model_a, model_b, metric)` returns:

```json
{
  "a": "qsvm__reduced_n", "b": "rbf_svm__reduced_n",
  "protocol": "A",
  "metric": "roc_auc",
  "a_value": 0.881, "a_ci": [0.79, 0.95],
  "b_value": 0.874, "b_ci": [0.78, 0.94],
  "delta": 0.007, "delta_ci": [-0.06, 0.07],
  "test": "delong", "p_value": 0.71,
  "verdict": "no significant difference"
}
```

**`verdict` is one of exactly three strings**, chosen by the test result and never by the point estimate:

- `"significantly better"` (p < 0.05 and delta > 0)
- `"no significant difference"` (p ≥ 0.05)
- `"significantly worse"` (p < 0.05 and delta < 0)

The dashboard renders `verdict`; it has no code path that renders "quantum wins" from a raw delta. **This is a build requirement, not a style guideline** — it is what makes the honesty claim in §1.3 R3 structurally true.

### 9.4 Threshold policy

| Threshold | How chosen | Why |
|---|---|---|
| `0.5` | Fixed | Comparable, conventional |
| `sensitivity_target_0.90` | Smallest threshold on the **training-fold** ROC achieving ≥0.90 sensitivity, then applied unchanged to test | Clinically motivated: missing an at-risk patient costs more than a false alarm |

Both threshold rows are reported for every model. The test set is **never** used to choose a threshold.

### 9.5 Evaluation outputs (files written per run)

```
results/runs/<run_id>/
├── metrics.csv               # one row per model × protocol × threshold
├── metrics.json              # same, structured, with CIs
├── comparisons.json          # every quantum-vs-classical verdict
├── cv_results.csv            # per-fold, per-repeat metrics
├── seed_variance.csv
├── confusion_matrices.json
└── plots/
    ├── roc_overlay.png
    ├── pr_overlay.png
    ├── confusion_<model_id>.png
    └── calibration.png
```

---

## 10. Quantum Evaluation Specification

Every quantum model must emit a `resource_report()`. These fields are mandatory and appear on the Quantum Info page.

| Field | Meaning | QSVM (6 qubits, ZZ reps=2, Cleveland) | VQC (6 qubits, RealAmplitudes reps=2) |
|---|---|---|---|
| `n_qubits` | Width of the circuit | 6 | 6 |
| `feature_map` | Name + reps + entanglement | `ZZFeatureMap(reps=2, linear)` | `ZZFeatureMap(reps=2, linear)` |
| `ansatz` | Variational block, if any | — (none) | `RealAmplitudes(reps=2)` |
| `circuit_depth_logical` | Depth as constructed | reported | reported |
| `circuit_depth_transpiled` | Depth after transpiling to the device profile | reported | reported |
| `two_qubit_gate_count` | CX count after transpilation | reported | reported |
| `n_trainable_params` | **Trainable quantum parameters** | **0** (feature map is fixed; only classical SVC learns) | **18** = `ansatz.num_parameters` |
| `circuits_executed` | Total circuit evaluations | ≈ 43,923 | ≈ maxiter × restarts × batches |
| `shots` | Shots per circuit | `null` (exact) or e.g. 1024 | same |
| `backend` | Backend name | `aer_statevector` | `aer_statevector` |
| `noise_model` | Noise model used, or `none` | `none` (default) / `depolarizing+readout` (robustness run) | same |
| `quantum_seconds` | Wall clock spent in quantum execution | measured | measured |
| `total_seconds` | Full fit time including classical parts | measured | measured |
| `cache_hit` | Whether a cached kernel was reused | true/false | n/a |
| `device_name` | Real device, when a hardware backend was used | `null` (sim) / `ibm_<device>` | same |
| `job_ids` | IBM Runtime job IDs — provenance | `null` / list | same |
| `calibration_timestamp` | Device calibration time at execution | `null` / ISO-8601 | same |
| `queue_seconds` | Time spent queued before execution | `null` / measured | same |
| `qpu_seconds` | Billed QPU time consumed | `null` / measured | same |

**Mandatory honesty statements attached to every quantum result view:**

1. Backend-conditional, and it must match the run record: *"These results were produced on a classical simulator (`aer_statevector`)"* **or** *"These results were produced on IBM Quantum device `ibm_<device>` on `<date>`, calibration `<timestamp>`, job `<id>`."* The UI reads this from `resource_report()` — it is never hardcoded.
2. *"QSVM has zero trainable quantum parameters — the quantum component is a fixed feature map; the learning happens in the classical SVM."*
3. *"Results are conditional on n=6 qubits, this feature map, and a reduced 6-dimensional representation of the data."*
4. No claim of quantum advantage appears anywhere without a CI and a p-value (§9.3).

---

## 11. Explainability Specification

### 11.1 What the MVP explains, and how

| Branch | Scope | Method | Library | Output |
|---|---|---|---|---|
| Classical (LR) | Global | Coefficients + odds ratios | sklearn | Signed bar chart |
| Classical (XGB/RF) | Global | SHAP TreeExplainer | `shap` | Beeswarm + bar |
| Classical (any) | Local | SHAP values for one patient | `shap` | Waterfall plot |
| **Quantum (QSVM, VQC)** | Global | **Permutation importance** on the reduced features, measured on validation data | `sklearn.inspection.permutation_importance` | Ranked bar chart |
| **Quantum (QSVM, VQC)** | Local | **KernelSHAP** over the model's `predict_proba` | `shap.KernelExplainer` | Waterfall plot |
| Both | Bridge | **PCA loading map** — how each reduced component `PC1…PC6` is built from the original clinical features | numpy | Heatmap + top-3 contributors per component |
| Quantum (QSVM) | Local, optional | **Kernel-similarity explanation** — the training patients most similar to the query under the quantum kernel, with their labels | own code | "This patient resembles these 5 cases (4 at-risk)" |

### 11.2 The four questions the MVP must answer on-screen

The Explainability page is structured around exactly these, because the brief asks for them:

**1. Which input features contributed to this prediction?**
A ranked, signed attribution list for the selected patient — for a classical model directly in clinical feature terms, and for a quantum model in reduced-component terms **plus** a translation through the PCA loading map back to clinical features.

**2. How is the explanation generated?**
A short method note rendered next to every plot, e.g.: *"SHAP estimates each feature's contribution by averaging its marginal effect over feature orderings, using a background sample of 100 training records."* / *"Permutation importance measures the drop in ROC-AUC when a single feature's values are shuffled, averaged over 10 repeats."*

**3. What are the limitations?**
Rendered, not buried in docs:

| Method | Stated limitation |
|---|---|
| Permutation importance | Assumes feature independence; correlated features share/dilute credit |
| KernelSHAP | Approximate; depends on the background sample; expensive on a quantum model |
| SHAP on reduced components | Explains **components, not raw clinical variables** — the loading map is an interpretation aid, not an exact attribution |
| Surrogate models | Explain the surrogate, faithful only up to the reported fidelity |
| All | **Associational, not causal.** No intervention claim is implied. |

**4. How does explaining a quantum model differ from explaining a classical one?**
A short, permanent panel on the page:

| | Classical | Quantum |
|---|---|---|
| Internal structure | Inspectable (coefficients, tree splits) | Not directly interpretable — the state lives in a 2⁶-dimensional Hilbert space |
| Native attributions | Yes (coefficients, TreeSHAP) | **None** |
| Method available | Model-specific *and* model-agnostic | **Model-agnostic only** — anything that works off `predict_proba` |
| Cost | Milliseconds–seconds | Expensive: every perturbed sample requires new circuit evaluations |
| Feature space explained | Original clinical features | Reduced components → translated via PCA loadings |
| Practical consequence | Direct clinical narrative | An extra interpretive hop, which must be disclosed |

**This comparison panel is a deliverable, not commentary.** It is one of the clearest ways to show the jury we understand what we built.

### 11.3 Cost control

KernelSHAP on a quantum model calls `predict_proba` thousands of times, and each call may need circuit evaluations. MVP controls:

- Background sample capped at **100** training records (`shap.kmeans` summarization).
- `nsamples` capped at **200** per explanation.
- For QSVM, reuse the **cached kernel** where the query point is a test-set row.
- Explanations computed **on demand** for a selected patient, never precomputed for all patients.
- Hard timeout of 5 minutes per explanation with a clear "explanation unavailable within budget" message.

---

## 12. Robustness and Generalization Specification

| # | Check | Method | Output | Priority |
|---|---|---|---|---|
| 1 | **Cross-validation variance** | Repeated stratified 5-fold × 3 repeats on the training set | Mean ± std per metric per model | MUST |
| 2 | **Seed variance** | Full experiment at seeds 42, 43, 44 | Table of metric spread across seeds | MUST |
| 3 | **Noise robustness** | Add Gaussian noise σ ∈ {0, 0.01, 0.05, 0.10, 0.20} (in units of each feature's training std) to **test** features only; re-predict with the already-trained models | Metric-vs-σ degradation curve, all models on one axis | MUST |
| 4 | **Generalization gap** | Train metric − test metric per model | Column in the leaderboard; flag if gap > 0.10 on ROC-AUC | MUST |
| 5 | **Subgroup slices** | Sensitivity/specificity computed within `sex ∈ {0,1}` and age bands `{<45, 45–54, 55–64, ≥65}` | Subgroup table with disparity flags | MUST |
| 6 | **Learning curve** | Metric vs training fraction {0.2 … 1.0} (classical models only — too expensive for quantum) | Curve | SHOULD |
| 7 | **Missingness robustness** | Randomly mask 5–30% of test values, impute with training-fitted imputers | Degradation curve | SHOULD |
| 8 | **Quantum noise robustness** | Re-run QSVM under `AerNoisyBackend`; compare to ideal | One comparison row + delta | SHOULD |
| 9 | **Shot-noise sensitivity** | Shots ∈ {256, 1024, 4096} vs exact | Metric variance table | SHOULD |
| 10 | **Cross-site generalization** | Train on Cleveland, test on Hungarian subset (heavy missingness — expect degradation) | Honest external-ish check | SHOULD (high jury value) |
| 11 | **Feature-set ablation** | `full` (13 features) vs `screening` (no `ca`, `thal`) | Two result groups, side by side | **MUST** — this is the clinical-leakage control from §5.6 |

### 12.1 Interpreting robustness honestly

Two statements must appear in the UI and the docs:

1. *"Robustness here means stability under perturbation of this dataset. It is not external validation. The model has been evaluated on one cohort of ~300 patients."*
2. *"Performance is expected to drop on the screening feature set (without `ca` and `thal`). That drop is informative: it estimates how much of the model's apparent skill depends on features that are only available after specialist cardiac testing."*

### 12.2 Compute budget for robustness

| Check | Classical cost | Quantum cost | Mitigation |
|---|---|---|---|
| Repeated CV (15 fits) | seconds | **15 × kernel computation** — expensive | Cache kernels per fold; if over budget, run quantum CV at `repeats=1` (5 fits) and say so explicitly in the report |
| Noise sweep (5 levels) | seconds | 5 × test-kernel recomputation (train kernel unchanged) | Only the test/train block is recomputed — cheap |
| Seed variance (3 seeds) | minutes | 3 × full kernel computation | Run overnight; cache; this is a batch job, not a demo job |
| Subgroups | free (slicing existing predictions) | free | — |

**Rule:** all expensive robustness runs are executed **before demo day** and loaded from `results/runs/`. The live demo runs only a small, fast configuration.
---

## 13. Dashboard Specification

**Design rule: do not over-engineer the frontend.** The dashboard is a thin, professional viewer over the backend API. It contains **no ML logic, no computation, no state machine beyond the current run ID**. Every number it shows was computed in Python and stored in a run record.

### 13.1 The seven pages

#### Page 1 — Home / Overview

| Element | Content |
|---|---|
| Title block | Project name, SIH26139, team, one-line description |
| Architecture diagram | Static SVG/PNG of the hybrid pipeline (§6.1) |
| Pipeline status strip | 6 chips: Data ✓ · Preprocess ✓ · Classical ✓ · Quantum ✓ · Evaluation ✓ · Explainability ✓ |
| Headline cards | Best classical ROC-AUC, QSVM ROC-AUC, verdict string, dataset size, qubits used |
| **Disclaimer banner** | "Research prototype. Decision support only — not a diagnosis. Validated on UCI Heart Disease (n≈303) using a quantum simulator." |
| Action | "Load demo run" button → loads the cached run in ≤ 5 s |

#### Page 2 — Dataset Analysis

| Section | Content |
|---|---|
| Dataset summary | Rows, columns, source, SHA-256, feature set (`full`/`screening`) |
| Validation report | Table: check name, status (PASS/WARN/FAIL), detail |
| Missingness | Per-column count and %, with sentinel codes named (`ca: "?"`, `thal: "?"`, `chol: 0`) |
| Class distribution | Bar chart + counts + minority share |
| Numeric profile | min / max / mean / median / std / outliers per numeric feature |
| Categorical profile | Level frequencies |
| Correlation heatmap | PNG served by the backend |
| Engineered features | Name + formula + enabled flag |
| Ranked feature table | Feature, MI score, L1 coefficient, rank, selected ✓ |
| PCA panel | Scree plot, cumulative explained variance, chosen `n_components` |
| Preprocessing summary | What was imputed / encoded / scaled / dropped, and the **"fit on training data only"** statement |

#### Page 3 — Patient Prediction

| Element | Behaviour |
|---|---|
| Input form | 13 fields (or 11 in screening mode) with clinical labels, units, ranges, and sensible defaults. Client-side range validation. |
| "Load example patient" | Two buttons: a typical low-risk and a typical high-risk record |
| Model selector | Dropdown of trained models + a "Compare classical vs quantum" toggle |
| Result card | Risk probability (0–100%), risk band (Low <30% / Moderate 30–60% / High >60%), threshold used, predicted label |
| Side-by-side panel | Classical result ‖ Quantum result, plus an **agreement/disagreement flag** |
| Explanation strip | Top-5 contributing factors with direction (↑/↓ risk) |
| Batch mode | Upload CSV → scored CSV download |
| **Disclaimer** | Always visible on this page |

#### Page 4 — Model Comparison

| Element | Content |
|---|---|
| Protocol toggle | **Protocol A** (reduced-*n* vs reduced-*n*) ‖ **Protocol B** (reduced-*n* quantum vs full-*d* classical), each with a one-line explanation |
| Leaderboard | Model · Branch · Feature space · Accuracy (CI) · Sensitivity (CI) · Specificity (CI) · F1 · ROC-AUC (CI) · PR-AUC · MCC · Train time · Inference ms · Gen-gap — sortable |
| CV table | Mean ± std per metric from repeated stratified CV |
| ROC overlay | All models on one axis |
| PR overlay | All models on one axis |
| Confusion matrices | Grid, one per model |
| **Verdict panel** | For each quantum-vs-classical pair: metric, delta, delta CI, test name, p-value, and the verdict string rendered as a coloured badge |
| Efficiency table | Train seconds, inference ms, circuits executed, transpiled depth, shots |
| Feature-set ablation | `full` vs `screening` results side by side with the clinical-leakage explanation |
| Export | Download `metrics.csv` / `metrics.json` |

#### Page 5 — Explainability

| Element | Content |
|---|---|
| Model selector | Classical ‖ Quantum |
| Global importance | Bar chart (SHAP for classical, permutation importance for quantum) |
| Patient selector | Pick a test-set record (or reuse the one from Page 3) |
| Local explanation | SHAP waterfall (classical) / KernelSHAP waterfall (quantum) |
| PCA loading map | Heatmap `PC1…PC6 × clinical features` + top-3 contributors per component |
| Kernel-similarity panel | (QSVM) most similar training patients under the quantum kernel, with labels |
| **Method note** | How this explanation was generated |
| **Limitations note** | The table from §11.2 |
| **Classical vs quantum explainability panel** | The comparison table from §11.2 Q4 — permanent on this page |

#### Page 6 — Quantum Circuit / Model Information

| Element | Content |
|---|---|
| Configuration card | Feature map, reps, entanglement, qubits, ansatz (if VQC), optimizer, backend, shots, seed |
| Circuit diagram | PNG rendered by `circuit.draw('mpl')`, served by the backend |
| Resource report | The full table from §10 |
| Transpilation report | Target device profile, required qubits, transpiled depth, 2-qubit gate count, SWAP overhead |
| **Hardware readiness checklist** | Qubits ✓ · Depth ✓ · Basis gates ✓ · Connectivity ✓ · Shot-based path ✓ · Estimated jobs — each PASS/WARN |
| Diagnostics | Kernel concentration (mean/std off-diagonal), kernel eigenspectrum plot, VQC loss curve |
| **Real hardware panel** (§8.5) | Device name, execution date, calibration timestamp, IBM job IDs, shots, QPU seconds, queue time · three kernel heatmaps (exact ‖ noisy-sim ‖ hardware) · error histogram with mean/max \|ΔK\| · diagonal-fidelity deviation · subsample accuracy with the "n=10, not a performance claim" caveat |
| **Quota note** | The §8.5.1 arithmetic in two lines: why the headline benchmark runs on the simulator and hardware is used for validation |
| **Honesty note** | Backend-conditional (read from `resource_report()`): simulator vs named device and date. Plus: "QSVM has 0 trainable quantum parameters." |

#### Page 7 — Experiment Results

| Element | Content |
|---|---|
| Run list | run_id · date · dataset · feature set · qubits · feature map · seed · status · headline ROC-AUC |
| Run detail | Full config JSON (collapsible), all metrics, artifact links |
| Robustness section | CV table, seed-variance table, noise-degradation curve, subgroup table, generalization gaps |
| Reproducibility card | Seed, library versions, git commit, dataset SHA-256, split_id, preprocess_hash |
| Compare runs | Select two runs → side-by-side metric diff |
| Export | Download the run record JSON |

### 13.2 Frontend engineering constraints

| Constraint | Decision |
|---|---|
| Framework | **React + Vite** (JavaScript, not TypeScript — faster for a student team; add TS only if the team already knows it) |
| Routing | `react-router-dom`, 7 routes |
| Styling | One small CSS file + CSS variables, or Tailwind if a member already knows it. **No component library beyond one table/chart helper.** |
| Charts | **Recharts** for simple bars/lines/tables |
| Complex plots | **Rendered server-side as PNG by matplotlib** (ROC overlay, SHAP, confusion matrices, circuit diagram, heatmaps) and served via `GET /plots/{run_id}/{name}.png`. This removes ~60% of frontend work and guarantees the plots match the paper/report. |
| State | `useState` + a tiny `RunContext` holding `run_id`. **No Redux, no react-query, no state library.** |
| API calls | `fetch` in a single `src/api/client.js` |
| Components | ~20: `MetricTable`, `VerdictBadge`, `StatCard`, `PlotImage`, `PatientForm`, `RiskCard`, `FileUpload`, `RunSelector`, `Disclaimer`, ... |
| Build | `npm run dev` (port 5173) proxying `/api` → `http://localhost:8000` |
| Not building | Auth, dark mode, i18n, animations, responsive mobile layouts, offline PWA |

**Fallback (documented, pre-authorized):** if the React dashboard is not usable by end of Week 8, switch to **Streamlit** — the backend API and all computation are unchanged, and Streamlit can reproduce all 7 pages in ~500 lines. Losing the frontend must never cost us the MVP. Decide by the Week 8 checkpoint.

---

## 14. Backend API Specification

**One FastAPI process. No database. No auth. No queue.** State lives in `results/runs/` on disk.

### 14.1 Endpoints (exactly nine)

| # | Method | Path | Purpose | Response |
|---|---|---|---|---|
| 1 | `GET` | `/api/health` | Liveness + versions | `{status, version, qiskit_version, python_version}` |
| 2 | `GET` | `/api/dataset/summary?run_id=` | Validation report + profiles + class balance | `DatasetSummary` |
| 3 | `POST` | `/api/dataset/upload` | Upload a CSV, validate against schema | `{dataset_id, sha256, validation_report}` |
| 4 | `GET` | `/api/models` | List trained models with headline metrics | `[ModelInfo]` |
| 5 | `POST` | `/api/predict` | Score one patient | `PredictionResponse` |
| 6 | `POST` | `/api/predict/batch` | Score a CSV | CSV file response |
| 7 | `GET` | `/api/results/{run_id}` | Full metrics, comparisons, verdicts, efficiency | `RunResults` |
| 8 | `GET` | `/api/explain?run_id=&model_id=&record_index=` | Global + local explanation payload | `ExplanationResponse` |
| 9 | `GET` | `/api/quantum/info?run_id=` | Resource report, transpilation report, diagnostics | `QuantumInfo` |

Plus static plot serving: `GET /plots/{run_id}/{filename}.png`, and `GET /api/runs` returning the run list (folded into #7's router).

**Deliberately absent:** `POST /train`. Training is run from the CLI (`python -m src.cli run --config configs/run_mvp.yaml`), not from the browser. Training takes minutes, needs no UI, and adding async job management would cost a week for zero demo value. **The dashboard reads results; the CLI produces them.** This single decision removes background tasks, job polling, progress websockets, and cancellation from the MVP.

If the team wants live training in the demo, add one optional endpoint in Phase 10: `POST /api/train/demo` running a small pre-configured run via `BackgroundTasks`, polled by `GET /api/train/status`. **Optional, last.**

### 14.2 Key schemas (Pydantic)

```python
class PatientInput(BaseModel):
    age: int = Field(ge=18, le=110)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(ge=60, le=260)
    chol: float = Field(ge=80, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(ge=50, le=230)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=8)
    slope: int = Field(ge=1, le=3)
    ca: int | None = None      # optional; absent in screening mode
    thal: int | None = None

class PredictionResponse(BaseModel):
    model_id: str
    branch: str                       # "classical" | "quantum"
    probability: float
    risk_band: str                    # "Low" | "Moderate" | "High"
    predicted_label: int
    threshold_used: float
    top_factors: list[FactorContribution]
    run_id: str
    disclaimer: str                   # always populated, never empty
    agreement: str | None             # "agree" | "disagree" when comparing branches
```

### 14.3 Backend requirements

| ID | Requirement |
|---|---|
| BE-1 | Models and preprocessing artifacts are **loaded once at startup** (or lazily on first use) and cached in memory — never reloaded per request |
| BE-2 | A prediction applies the **stored** preprocessing artifacts; it never refits anything |
| BE-3 | Field-level validation errors return HTTP 422 with the offending field named |
| BE-4 | A consistent error envelope: `{error_code, message, details, run_id}` |
| BE-5 | CORS enabled for `http://localhost:5173` only |
| BE-6 | Auto-generated OpenAPI docs at `/docs` — this is the "API documentation" deliverable |
| BE-7 | Patient inputs are **not persisted** (privacy; §17.7) |
| BE-8 | Classical prediction ≤ 1 s; quantum prediction ≤ 10 s (cached kernel/model) |
| BE-9 | Startup fails loudly if the configured `run_id` artifacts are missing |

---

## 15. Technology Stack

### 15.1 The stack

| Layer | Choice | Version pin strategy | Why |
|---|---|---|---|
| Language | **Python 3.10 or 3.11** | Fixed for the whole team | 3.11 max — some quantum/ML wheels lag on 3.12+. **Do not mix versions across the team.** |
| Data | **pandas, NumPy** | pin minor | Standard |
| Classical ML | **scikit-learn** | pin minor | LR, SVC, pipelines, CV, metrics, permutation importance — one library covers most of the MVP |
| Boosting | **XGBoost** (fallback `RandomForestClassifier`) | pin minor | Strong tabular baseline; decide in Phase 0 |
| Quantum | **Qiskit + qiskit-aer + qiskit-machine-learning** | **exact pins** | See §15.2 |
| Quantum hardware | **qiskit-ibm-runtime** | **exact pin** | Real QPU access for the §8.5 validation run. Isolated in `src/quantum/backends_ibm.py`; the pipeline runs fine without it |
| Secrets | **python-dotenv** | pin minor | Loads the IBM API key from a git-ignored `.env` |
| Explainability | **shap** | pin minor | TreeExplainer + KernelExplainer covers both branches |
| Statistics | **SciPy** (+ own DeLong implementation or `statsmodels` McNemar) | pin minor | McNemar, bootstrap, tests |
| Plots (server) | **matplotlib** | pin minor | ROC/PR/SHAP/circuit/heatmap PNGs |
| Backend | **FastAPI + Uvicorn + Pydantic v2** | pin minor | Typed, auto-documented, tiny |
| Frontend | **React 18 + Vite + react-router-dom + Recharts** | `package-lock.json` committed | Lightweight, familiar |
| Config | **PyYAML** (+ Pydantic validation) | pin minor | Config-driven experiments |
| Persistence | **joblib, JSON, .npy** | — | No database |
| Testing | **pytest, pytest-cov** | pin minor | Unit, contract, leakage, reproducibility tests |
| Lint/format | **ruff + black** | pin minor | One command, no debate |
| VCS | **Git + GitHub** | — | Branch protection + PR review |

**Total runtime Python dependencies: ~15.** Every addition beyond this list needs a written justification in `docs/DECISIONS.md`.

### 15.2 Quantum framework decision: **Qiskit** (single framework)

**Decision: use Qiskit for the entire quantum branch. Do not also use PennyLane.**

| Criterion | Qiskit | PennyLane | Verdict |
|---|---|---|---|
| **QSVM support** | `FidelityQuantumKernel` + `QSVC` in `qiskit-machine-learning` — purpose-built, few lines | `qml.kernels` exists but more assembly required | **Qiskit** — QSVM is our mandatory model |
| **Standard feature maps** | `ZZFeatureMap`, `ZFeatureMap`, `PauliFeatureMap` built in, exactly the published constructions | Must construct manually | **Qiskit** |
| **Simulator** | Aer: statevector, sampling, and device noise models in one package | `default.qubit`, `lightning.qubit`; noise via plugins | **Qiskit** — noise models matter for our robustness section |
| **Near-term hardware compatibility evidence** | **Transpiler with coupling maps, basis gates, depth reports** — this is how we *prove* HW compatibility without a QPU | Weaker native transpilation story | **Qiskit — decisive** |
| **Path to real hardware** | IBM Quantum Runtime, free tier, same circuits | Requires a plugin per provider | **Qiskit** |
| **Variational training** | `VQC`, `EstimatorQNN`, COBYLA/SPSA — adequate | **Better**: autograd, PyTorch/JAX interfaces, parameter-shift gradients | PennyLane |
| **Docs/tutorials/community** | Largest; most SIH-adjacent material | Excellent but smaller | **Qiskit** |
| **Team learning cost** | One framework, one mental model | Two frameworks = two sets of bugs | **Qiskit** |

**Reasoning in one paragraph:** our mandatory model is QSVM, and Qiskit implements quantum kernels and the standard feature maps most directly. More importantly, SIH26139 explicitly requires *near-term quantum hardware compatibility* — and the only way to demonstrate that without QPU access is a transpilation report against a real device coupling map and basis-gate set, which is Qiskit's core strength. PennyLane's advantage (autograd-based variational training) applies only to VQC, which is our **droppable secondary** model. Adopting two frameworks to slightly improve an optional component would double the dependency surface, the API-churn risk, and the team's learning load. **One framework, chosen for the mandatory path.**

**Version discipline (critical — this is the #1 avoidable project killer):**

```
1. On Day 1 of Phase 3, one person resolves the environment and commits an EXACT pin set, e.g.
     qiskit==<resolved>
     qiskit-aer==<resolved>
     qiskit-machine-learning==<resolved>
2. Everyone installs from requirements.txt. Nobody upgrades.
3. Record the exact resolved versions in docs/DECISIONS.md and in every run record.
4. Isolate ALL Qiskit imports inside src/quantum/. No Qiskit import anywhere else in the codebase.
   If the API changes, exactly one directory breaks.
```

Qiskit's ML/kernel APIs have moved between releases. The abstraction in §8.3 plus this import-isolation rule is the insurance policy.

---

## 16. Project Structure

```
quantumdx/
├── README.md                        # what it is, install, run, demo in 10 lines
├── requirements.txt                 # exact pins
├── environment.yml                  # optional conda equivalent
├── pyproject.toml                   # ruff + black + pytest config
├── .gitignore                       # data/raw/*, results/*, node_modules, __pycache__, .env
├── .env.example                     # IBM_QUANTUM_API_KEY=, IBM_QUANTUM_INSTANCE=  (COMMITTED, values blank)
├── .env                             # real credentials — NEVER COMMITTED
│
├── configs/
│   ├── schema_uci_heart.yaml        # columns, types, sentinels, target rule, ranges
│   ├── run_mvp.yaml                 # the canonical MVP run config (§6.4)
│   ├── run_screening.yaml           # feature_set: screening (leakage ablation)
│   ├── sweep_quantum.yaml           # qubits × feature maps × reps (batch, pre-demo)
│   ├── run_hardware_validation.yaml # the §8.5 scoped QPU run (30 train / 10 test, 735 circuits)
│   └── device_profile_ibm.yaml      # REAL device: basis gates, coupling map, qubits, depth limit
│                                    # (replaces the generic 27q profile once the device is known)
│
├── data/
│   ├── raw/
│   │   ├── CHECKSUMS.txt            # SHA-256 per file — COMMITTED
│   │   └── uci_heart/v1/            # processed.cleveland.data (+ other sites) — GIT-IGNORED
│   ├── external/                    # user-uploaded CSVs at runtime — GIT-IGNORED
│   └── README.md                    # how to obtain the data + expected checksums
│
├── notebooks/                       # EXPLORATION ONLY — never imported by src/
│   ├── 01_eda_uci_heart.ipynb
│   ├── 02_quantum_kernel_sanity.ipynb
│   └── README.md                    # "nothing here is production code"
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # Pydantic config models + YAML loading + config hashing
│   ├── seeds.py                     # set_global_seeds(seed)
│   ├── cli.py                       # `python -m src.cli run|validate|report|predict`
│   ├── pipeline.py                  # ExperimentRunner — orchestrates everything
│   │
│   ├── data/
│   │   ├── loader.py                # DatasetLoader + SHA-256 verification
│   │   ├── validator.py             # DataValidator → ValidationReport
│   │   ├── splitter.py              # stratified split, split_id, persisted indices
│   │   └── schema.py                # Schema dataclasses
│   │
│   ├── preprocessing/
│   │   ├── sentinels.py             # config-driven sentinel → NaN
│   │   ├── pipeline.py              # build_preprocessor() → sklearn ColumnTransformer
│   │   └── range_normalizer.py      # → [0, π], with the range assertion
│   │
│   ├── features/
│   │   ├── engineering.py           # 5 clinical derived features
│   │   ├── selection.py             # mutual_info + l1_logistic, ranking table
│   │   └── reduction.py             # PCA / top-k + explained variance report
│   │
│   ├── classical/
│   │   ├── models.py                # LR, RBF-SVM, XGB/RF wrapped in the Model interface
│   │   └── training.py              # GridSearchCV, fit on both feature spaces
│   │
│   ├── quantum/                     # ◄── THE ONLY DIRECTORY THAT IMPORTS QISKIT
│   │   ├── backends.py              # QuantumBackend protocol + Aer statevector/sampling/noisy
│   │   ├── backends_ibm.py          # IBMRuntimeBackend — real QPU (SamplerV2), quota guard, job polling
│   │   ├── hardware_validation.py   # the §8.5 experiment: 3-backend kernel comparison + error stats
│   │   ├── feature_maps.py          # build_feature_map(name, n, reps, entanglement)
│   │   ├── kernel.py                # quantum kernel computation + disk cache + diagnostics
│   │   ├── qsvm.py                  # QSVM model (Model interface)
│   │   ├── vqc.py                   # VQC model (Model interface)
│   │   └── transpile_report.py      # device-profile transpilation + hardware checklist
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # the 14 metrics + confusion matrix
│   │   ├── statistics.py            # bootstrap CI, McNemar, DeLong
│   │   ├── benchmark.py             # BenchmarkEngine + UnfairComparisonError + verdicts
│   │   ├── robustness.py            # CV, seeds, noise, subgroups, gen-gap
│   │   └── plots.py                 # ROC/PR/confusion/calibration PNGs
│   │
│   ├── explainability/
│   │   ├── classical_explainer.py   # SHAP + coefficients
│   │   ├── quantum_explainer.py     # permutation importance + KernelSHAP + kernel similarity
│   │   └── loading_map.py           # PCA components → clinical features
│   │
│   ├── inference/
│   │   ├── artifacts.py             # load a run's models + preprocessing artifacts
│   │   └── predictor.py             # single-record and batch prediction
│   │
│   └── tracking/
│       ├── registry.py              # RunRegistry: create/update/get/list run records
│       └── model_store.py           # save/load models + manifest
│
├── backend/
│   ├── main.py                      # FastAPI app, CORS, startup artifact loading
│   ├── schemas.py                   # Pydantic request/response models
│   ├── deps.py                      # cached artifact/model providers
│   └── routers/
│       ├── dataset.py               # /api/dataset/*
│       ├── models.py                # /api/models
│       ├── predict.py               # /api/predict, /api/predict/batch
│       ├── results.py               # /api/results/{run_id}, /api/runs
│       ├── explain.py               # /api/explain
│       └── quantum.py               # /api/quantum/info
│
├── frontend/
│   ├── package.json
│   ├── vite.config.js               # /api proxy → localhost:8000
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx                  # router + layout + RunContext
│       ├── api/client.js            # every fetch call lives here
│       ├── components/              # ~20 small components
│       ├── pages/
│       │   ├── Home.jsx
│       │   ├── DatasetAnalysis.jsx
│       │   ├── PatientPrediction.jsx
│       │   ├── ModelComparison.jsx
│       │   ├── Explainability.jsx
│       │   ├── QuantumInfo.jsx
│       │   └── ExperimentResults.jsx
│       └── styles.css
│
├── tests/
│   ├── conftest.py                  # tiny synthetic fixture dataset
│   ├── test_loader.py
│   ├── test_validator.py
│   ├── test_leakage.py              # ◄── THE CRITICAL TEST
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_range_normalizer.py     # asserts [0, π]
│   ├── test_classical_models.py
│   ├── test_quantum_backend.py
│   ├── test_qsvm.py                 # 2 qubits, 20 samples — fast
│   ├── test_model_contract.py       # every model satisfies the Model Protocol
│   ├── test_metrics.py              # known confusion matrix → known metric values
│   ├── test_benchmark_fairness.py   # mismatched split_id raises
│   ├── test_reproducibility.py      # same seed → identical metrics
│   ├── test_inference.py
│   └── test_api.py                  # FastAPI TestClient
│
├── results/                         # GIT-IGNORED except .gitkeep
│   ├── runs/<run_id>/               # run_record.json, metrics.*, plots/, models/, artifacts/
│   ├── cache/kernels/               # <hash>.npy
│   └── registry/models.json         # index of trained models
│
├── docs/
│   ├── ARCHITECTURE.md              # the diagrams + module contracts
│   ├── METHODS.md                   # dataset, preprocessing, models, metrics, statistics
│   ├── LIMITATIONS.md               # threats to validity (mandatory)
│   ├── DECISIONS.md                 # dated decision log (XGB vs RF, versions, cuts)
│   ├── API.md                       # or a pointer to /docs
│   ├── DEMO_SCRIPT.md               # the 10-minute run sheet
│   └── SIH_TRACEABILITY.md          # requirement → feature → screen
│
└── scripts/
    ├── download_data.py             # fetch UCI files + verify checksums
    ├── run_all_experiments.sh       # produce every pre-demo result
    └── prepare_demo.sh              # build the cached demo run
```

**Structural rules (enforced in review):**

| Rule | Reason |
|---|---|
| `src/` never imports from `backend/`, `frontend/`, or `notebooks/` | The core library must be usable headless |
| `backend/` contains **no ML logic** — it only calls `src/` and serializes | Keeps the API thin and testable |
| **Only `src/quantum/` imports Qiskit** | One directory breaks on an API change |
| Notebooks are exploration only and are never imported | Notebooks rot; code in `src/` is tested |
| Every module under `src/` has a matching test file | Coverage by construction |
| No file exceeds ~400 lines | Forces decomposition |

---

## 17. Engineering Practices

### 17.1 Git workflow

| Aspect | Rule |
|---|---|
| Branches | `main` (always working, protected) + short-lived `feat/*`, `fix/*`, `docs/*` branches |
| Branch naming | `feat/qsvm-kernel`, `fix/scaler-leakage`, `docs/methods` |
| No direct pushes to `main` | Enforced by GitHub branch protection |
| Pull requests | Every change; **1 reviewer minimum**; CI (pytest + ruff) must be green |
| PR size | Target < 400 changed lines. A PR touching 15 files is a red flag |
| Commit style | Conventional commits: `feat(quantum): add fidelity kernel cache`, `fix(preprocess): fit imputer on train only` |
| Merge strategy | Squash-merge — one commit per feature on `main` |
| Tags | `v0.1-phase4`, `v0.9-integration`, **`v1.0-sih-demo`** (the frozen demo build) |
| Freeze | **Code freeze 5 days before the demo.** After the freeze: bug fixes and docs only, no new features |
| Large files | Never commit `data/raw/*` or `results/*`; `.gitignore` covers them; checksums are committed instead |

**Daily rhythm:** 15-minute stand-up (yesterday / today / blocked), all work on branches, merge to `main` at least every two days so integration problems surface early.

### 17.2 Configuration management

| Rule | Detail |
|---|---|
| Three config files only | `schema_*.yaml` (what the data is), `run_*.yaml` (what the experiment is), `device_profile_*.yaml` (what the target hardware is) |
| Validated on load | Pydantic models in `src/config.py`; an invalid config fails **before** any computation |
| **No magic numbers in code** | Test size, seed, `k`, `n_components`, qubits, reps, shots, thresholds, noise sigmas — all config |
| Config hashing | `preprocess_hash = sha256(canonical_json(preprocess + features sections))` — used by the fairness guard |
| Config copied into the run | `results/runs/<run_id>/config.yaml` — the run is self-describing |
| Overrides | CLI flags may override single values (`--seed 43`) and the override is recorded |

### 17.3 Experiment tracking

**No MLflow server.** A run is a directory:

```
results/runs/run_20260903_1412_a3f9/
├── run_record.json        # config + environment + metrics + comparisons + timings + status
├── config.yaml            # exact config used
├── validation_report.json
├── split_indices.npz      # train_idx, test_idx (+ split_id)
├── artifacts/             # imputer, encoder, scaler, engineer, selector, reducer, normalizer (joblib)
├── models/                # <model_id>.joblib / .npy + manifest.json
├── metrics.csv / .json
├── cv_results.csv
├── comparisons.json
├── robustness/            # noise curve, subgroups, seed variance
└── plots/                 # all PNGs
```

`run_record.json` must contain: `run_id`, timestamp, git commit, Python + library versions, dataset SHA-256, `split_id`, `preprocess_hash`, seed(s), every config section, per-model metrics with CIs, comparisons with verdicts, resource reports, timings, warnings, and `status`.

`RunRegistry.list()` scans this directory — that is the entire tracking system, and it is enough.

### 17.4 Reproducibility

| Guarantee | Mechanism |
|---|---|
| Same config + same seed → same metrics (statevector backend) | `tests/test_reproducibility.py` runs a small config twice and asserts metric equality |
| Environment reproducibility | Exact pins in `requirements.txt`; versions recorded per run |
| Data reproducibility | SHA-256 verified at load; mismatch = hard failure |
| Code reproducibility | Git commit hash stored in every run record |
| Split reproducibility | Split indices persisted; `split_id` derived from (dataset hash, test_size, seed, stratify) |
| Sampling-mode caveat | With `shots` set, results are stochastic; the docs state this and report across-seed variance instead of exact equality |
| **Hardware caveat** | **Hardware runs are NOT bit-reproducible** — device calibration drifts daily. The reproducibility guarantee is claimed for `aer_statevector` only. Hardware runs instead record device name, calibration timestamp, IBM job IDs, shots and QPU seconds, so the run is *auditable* even though it is not repeatable. Say this plainly in `docs/LIMITATIONS.md` rather than letting a judge find it |
| Secrets | `.env` is git-ignored and never enters a run record; `.env.example` documents the required keys with blank values |

### 17.5 Random seed strategy

```python
# src/seeds.py
def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    from qiskit_algorithms.utils import algorithm_globals   # import guarded/isolated
    algorithm_globals.random_seed = seed
```

| Rule | Detail |
|---|---|
| One seed governs the run | `config.seed`, default **42**, propagated to: split, CV splitters, LR/SVC/XGB `random_state`, PCA, SHAP sampling, Qiskit `algorithm_globals`, Aer `seed_simulator` and `seed_transpiler`, VQC initialization, bootstrap resampling, noise injection |
| Never call a global RNG implicitly | Pass `random_state=cfg.seed` explicitly everywhere |
| Multi-seed runs | `seeds_multi: [42, 43, 44]` for the variance report — three separate runs, three run IDs |
| Seed is recorded | In `run_record.json` and displayed on the Experiment Results page |
| Never tune on the seed | Choosing the seed that gives the best number is fabrication; the report shows all three seeds |

### 17.6 Model versioning

| Aspect | Rule |
|---|---|
| `model_id` | `{model_name}__{feature_space}` e.g. `qsvm__reduced_n`, `rbf_svm__full_d` |
| Global identity | `{run_id}/{model_id}` — a model is meaningless without its run |
| Storage | `results/runs/<run_id>/models/<model_id>.joblib` (+ `.npy` for VQC parameters) |
| Manifest | `models/manifest.json`: model_id, branch, feature space, hyperparameters, metrics, artifact paths, `preprocess_hash`, `split_id`, created_at, library versions |
| Immutability | Written once. Retraining creates a **new run**, never overwrites |
| Binding | Loading a model loads its run's preprocessing artifacts. A mismatched schema raises `ArtifactMismatchError` — never a silent mispredict |
| Registry index | `results/registry/models.json` — a flat list for the `/api/models` endpoint |
| Demo model | The frozen demo run is tagged in `docs/DEMO_SCRIPT.md` by run_id and backed up outside `results/` |

### 17.7 Dataset versioning

| Aspect | Rule |
|---|---|
| Layout | `data/raw/<dataset_name>/<version>/<files>` e.g. `data/raw/uci_heart/v1/processed.cleveland.data` |
| Integrity | `data/raw/CHECKSUMS.txt` (committed) holds SHA-256 per file; `DatasetLoader` verifies on every load and **fails hard** on mismatch |
| Acquisition | `scripts/download_data.py` fetches from UCI and verifies checksums — the data itself is **not** committed |
| Uploads | User CSVs land in `data/external/` with a generated `dataset_id` + hash; never overwrite `data/raw/` |
| Provenance | Dataset name, version and SHA-256 are recorded in every run record |
| A new dataset | New folder + new `schema_*.yaml`. **No code change.** This is the scalability claim in §5.9, made concrete |
| Privacy | No identifiers ingested; prediction inputs are not persisted (§14.3 BE-7); `data/` is git-ignored |
| No DVC | Overkill for a ~20 KB dataset; checksums + versioned folders give the same guarantee at zero cost |
---

## 18. MVP Development Phases

Thirteen phases, Phase 0 → Phase 12. Each has a fixed template: **Objective · Inputs · Outputs · Files/modules · Libraries · Implementation tasks · Testing requirements · Definition of Done.**

**Rule:** a phase is not "done" because the code exists. It is done when its Definition of Done checklist is fully ticked and merged to `main`.

---

### Phase 0 — Project Setup

| Field | Content |
|---|---|
| **Objective** | Every team member can clone the repo, install identical dependencies, run an empty test suite, and push a branch. Zero ambiguity about tooling for the rest of the project. |
| **Inputs** | GitHub repo, team machines, this document |
| **Outputs** | Working repo skeleton, pinned environment, green CI, decision log started |

**Files/modules to create**

```
README.md, requirements.txt, environment.yml, pyproject.toml, .gitignore, .github/workflows/ci.yml
src/__init__.py, src/config.py (stub), src/seeds.py
tests/conftest.py, tests/test_smoke.py
configs/ (empty), data/README.md, docs/DECISIONS.md
scripts/download_data.py
```

**Libraries** — `python 3.10/3.11`, `pytest`, `ruff`, `black`, `pyyaml`, `numpy`, `pandas`, `scikit-learn`

**Implementation tasks**

1. Create the GitHub repo; protect `main` (PR required, 1 approval, CI must pass).
2. Create the folder skeleton from §16 with `.gitkeep` files.
3. Write `.gitignore`: `data/raw/*`, `data/external/*`, `results/*`, `node_modules/`, `__pycache__/`, `*.ipynb_checkpoints`.
4. **Fix the Python version for the whole team** (3.10 or 3.11) and record it in `docs/DECISIONS.md`.
5. Install and freeze base deps → `requirements.txt` with exact pins.
6. **Decide XGBoost vs RandomForest** — everyone tries `pip install xgboost`; if any machine fails, the team uses RandomForest. Record the decision.
7. Configure `pyproject.toml` for ruff, black (line length 100), pytest.
8. Add CI workflow: install deps → `ruff check` → `black --check` → `pytest`.
9. Write `src/seeds.py::set_global_seeds`.
10. Write `scripts/download_data.py` (fetch UCI files, write `data/raw/uci_heart/v1/`, print SHA-256).
11. Write `README.md`: what it is, install, run, 5-line quickstart.
12. Everyone opens a trivial PR to verify the workflow end to end.

**Testing requirements** — `tests/test_smoke.py` imports `src` and asserts `set_global_seeds(42)` runs; CI green on every member's PR.

**Definition of Done**

- [ ] All members clone, install, and run `pytest` successfully on their own machine
- [ ] CI green on `main`
- [ ] `requirements.txt` has exact pins and is identical for everyone
- [ ] Python version and XGB/RF decision recorded in `docs/DECISIONS.md`
- [ ] `scripts/download_data.py` downloads the UCI data and prints checksums
- [ ] Branch protection active; everyone has merged one PR

---

### Phase 1 — Dataset + Preprocessing

| Field | Content |
|---|---|
| **Objective** | Turn the raw UCI file into a `ProcessedDataset` with a proven zero-leakage guarantee. **This is the most important phase in the project** — every later result depends on its correctness. |
| **Inputs** | `data/raw/uci_heart/v1/processed.cleveland.data`, `configs/schema_uci_heart.yaml` |
| **Outputs** | `ValidationReport`, persisted split indices, fitted preprocessing artifacts, `X_*_full`, `X_*_reduced` in `[0, π]`, ranked feature table, PCA report |

**Files/modules to create**

```
configs/schema_uci_heart.yaml, configs/run_mvp.yaml
src/config.py                       (full: Pydantic models, YAML load, config hashing)
src/data/schema.py, loader.py, validator.py, splitter.py
src/preprocessing/sentinels.py, pipeline.py, range_normalizer.py
src/features/engineering.py, selection.py, reduction.py
src/pipeline.py                     (ExperimentRunner — data stages only for now)
notebooks/01_eda_uci_heart.ipynb    (exploration, informs the config)
```

**Libraries** — pandas, NumPy, scikit-learn (`SimpleImputer`, `OneHotEncoder`, `StandardScaler`, `MinMaxScaler`, `ColumnTransformer`, `Pipeline`, `train_test_split`, `mutual_info_classif`, `SelectKBest`, `PCA`), PyYAML, Pydantic, joblib

**Implementation tasks**

1. EDA notebook: confirm row count, dtypes, **actual** `?` counts in `ca`/`thal`, **actual** class counts, value ranges, correlations.
2. Write `schema_uci_heart.yaml` from §5.3–5.4: column names, types, levels, sentinels, target rule, physiological ranges.
3. `src/config.py`: Pydantic models for schema + run config; `load_config()`; `config_hash()` over canonical JSON.
4. `loader.py`: read the headerless UCI file, apply column names, coerce dtypes, verify SHA-256 against `CHECKSUMS.txt`.
5. `validator.py` → `ValidationReport` with checks: required columns, dtypes, target present & binarizable, **sentinel detection**, missingness per column, class balance, duplicates, zero-variance columns, physiological ranges, correlation pairs, and the **target-in-features** guard (blocking).
6. `splitter.py`: derive binary target; **stratified 80/20 split**; compute `split_id`; persist `split_indices.npz`. Accept an unused `group_column` parameter for future use.
7. `sentinels.py`: config-driven replacement of sentinel values with NaN — **applied before any statistic**.
8. `preprocessing/pipeline.py`: `build_preprocessor(schema, cfg)` → `ColumnTransformer` with numeric (median impute → scaler) and categorical (mode impute → one-hot / ordinal) branches. `fit` on train only.
9. `features/engineering.py`: implement 5 derived features — `age_band`, `hr_reserve` (`thalach` vs `220 − age`), `bp_category`, `chol_category`, `ischemia_score` (from `exang`, `oldpeak`, `slope`). Bin edges fitted on train. Record formulas.
10. `features/selection.py`: `mutual_info_classif` and L1-logistic ranking → ranked table + top-*k* selection.
11. `features/reduction.py`: PCA to `n_components` (+ a `topk` passthrough option); return explained variance and loadings.
12. `range_normalizer.py`: MinMax to `[0, π]`, fitted on train, with `assert 0 <= X <= π` in `transform`.
13. `pipeline.py`: assemble stages 1–14 of §5.7 into `ProcessedDataset` with `split_id`, `preprocess_hash`, artifacts, provenance.
14. Add `feature_set: full|screening` handling (drop `ca`, `thal` before the split).

**Testing requirements**

| Test | Assertion |
|---|---|
| `test_loader.py` | 303 rows, 14 columns; checksum mismatch raises |
| `test_validator.py` | Injected `?` and `chol=0` are detected; missing target column is blocking; a feature correlated >0.98 with target is blocking |
| `test_preprocessing.py` | No NaNs after imputation; one-hot widths correct; unknown category at transform does not crash |
| **`test_leakage.py`** | **(a)** fitting the chain on train-only vs train+test yields **different** scaler means when test differs (proving the fit is data-dependent) and the pipeline's stored parameters equal the **train-only** statistics; **(b)** a spy asserts `.fit()` is never called with test indices; **(c)** PCA components and the selected feature set from the run equal those from a train-only refit |
| `test_features.py` | 5 engineered columns appear when enabled and vanish when disabled; bin edges come from train |
| `test_range_normalizer.py` | All outputs in `[0, π]`, including test rows outside the training range (clipped) |
| `test_splitter.py` | Stratification preserves class ratio ±2%; same seed → identical indices; `split_id` stable |

**Definition of Done**

- [ ] `python -m src.cli validate --config configs/run_mvp.yaml` prints a full validation report
- [ ] `ProcessedDataset` is produced with both feature spaces and all artifacts
- [ ] **`test_leakage.py` passes** and is wired into CI
- [ ] All `X_*_reduced` values verified within `[0, π]`
- [ ] Ranked feature table and PCA explained variance exported to `results/`
- [ ] `feature_set: screening` produces a valid dataset without `ca`/`thal`
- [ ] Actual class counts and missing-value counts documented in `docs/METHODS.md`

---

### Phase 2 — Classical ML

| Field | Content |
|---|---|
| **Objective** | Three tuned classical baselines producing evaluated results on **both** feature spaces, all behind the shared `Model` interface. This is the control arm — it must be strong, not a straw man. |
| **Inputs** | `ProcessedDataset` from Phase 1 |
| **Outputs** | 6 trained classical models (3 × 2 feature spaces), best hyperparameters, metrics, persisted artifacts |

**Files/modules to create**

```
src/classical/models.py       # Model-interface wrappers for LR, RBF-SVM, XGB/RF
src/classical/training.py     # GridSearchCV; train on full_d and reduced_n
src/evaluation/metrics.py     # the 14 metrics + confusion matrix
src/tracking/registry.py      # RunRegistry (create/update/get/list)
src/tracking/model_store.py   # save/load + manifest
src/cli.py                    # `run` command wired to classical training
```

**Libraries** — scikit-learn (`LogisticRegression`, `SVC`, `GridSearchCV`, `StratifiedKFold`, `metrics`), XGBoost or `RandomForestClassifier`, joblib

**Implementation tasks**

1. Define the `Model` Protocol (§7.3) in `src/classical/models.py` (or `src/models_base.py`) — **this interface is frozen once written**.
2. Wrap LR, `SVC(kernel='rbf', probability=True)`, XGB/RF with `model_id`, `branch`, `feature_space`, `describe()`, `resource_report()`.
3. `training.py`: for each model × each feature space → `GridSearchCV(cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring='roc_auc')` on the training set; record best params and CV score.
4. Measure fit wall-clock and mean inference latency per record.
5. `metrics.py`: accuracy, precision, recall/sensitivity, specificity, NPV, F1, balanced accuracy, MCC, ROC-AUC, PR-AUC, Brier, confusion matrix — pure functions over `(y_true, y_pred, y_proba)`.
6. `model_store.py`: joblib persistence + `manifest.json` with `preprocess_hash` and `split_id`.
7. `registry.py`: create `results/runs/<run_id>/`, write `run_record.json` incrementally, capture git commit + library versions.
8. `cli.py run`: config → data pipeline → classical training → metrics → run record.

**Testing requirements**

- `test_classical_models.py`: each model fits on a synthetic fixture, `predict_proba` shape `(n,2)` summing to 1, `save`/`load` round-trip yields identical predictions.
- `test_model_contract.py`: parametrized over every registered model — all `Model` Protocol methods exist and behave.
- `test_metrics.py`: a hand-built confusion matrix produces hand-computed metric values; a perfect classifier gives 1.0; degenerate cases (all one class) do not crash.
- `test_registry.py`: a run record round-trips; `list()` finds it.

**Definition of Done**

- [ ] `python -m src.cli run --config configs/run_mvp.yaml` trains 6 classical models end to end
- [ ] `metrics.csv` contains all 14 metrics for every model × feature space
- [ ] Best hyperparameters, fit time and inference latency recorded
- [ ] Models persist and reload with identical predictions
- [ ] A run record exists in `results/runs/<run_id>/` with git commit and versions
- [ ] Classical ROC-AUC is in a plausible range (~0.85–0.92 on the full feature set) — if it is 1.0, **stop and hunt for leakage**

---

### Phase 3 — Quantum Environment

| Field | Content |
|---|---|
| **Objective** | Qiskit installed and pinned across the team; a working `QuantumBackend` abstraction; feature maps built from config; a transpilation report against the **real IBM device**; and a verified IBM Quantum account that can execute a trivial circuit. **No ML yet** — this phase de-risks the environment. |
| **Inputs** | Pinned Python environment from Phase 0; IBM Quantum account |
| **Outputs** | `QuantumBackend` implementations (Aer + IBM), `build_feature_map()`, circuit diagrams, transpilation report, **real device profile**, verified QPU access |

**Files/modules to create**

```
configs/device_profile_generic27q.yaml
src/quantum/__init__.py
src/quantum/backends.py            # QuantumBackend protocol + Aer statevector/sampling (+noisy)
src/quantum/feature_maps.py        # build_feature_map(name, n_features, reps, entanglement)
src/quantum/transpile_report.py    # transpile against profile → depth, 2q count, SWAPs, checklist
notebooks/02_quantum_kernel_sanity.ipynb
```

**Libraries** — `qiskit`, `qiskit-aer`, `qiskit-machine-learning`, matplotlib (circuit drawing)

**Implementation tasks**

1. **One person resolves the Qiskit version set**, verifies imports of `ZZFeatureMap`, `FidelityQuantumKernel`, `AerSimulator`, `transpile`, and commits **exact pins**. Everyone reinstalls from `requirements.txt`.
2. Record the resolved versions in `docs/DECISIONS.md` and in the run record.
3. `feature_maps.py`: build `ZZFeatureMap` / `ZFeatureMap` / `PauliFeatureMap` from `(name, n_features, reps, entanglement)`; return the circuit plus `{depth, n_params, n_qubits}`.
4. `backends.py`: `QuantumBackend` Protocol + `AerStatevectorBackend`, `AerSamplingBackend` (shots), and optionally `AerNoisyBackend`. Seed `seed_simulator` / `seed_transpiler`. Expose `capabilities()`.
5. `device_profile_generic27q.yaml`: 27 qubits, basis gates (`['cx','id','rz','sx','x']`), a simple heavy-hex-like coupling map, `max_depth: 200`.
6. `transpile_report.py`: transpile the feature-map circuit against the profile → required qubits, transpiled depth, CX count, SWAP overhead → **hardware-readiness checklist** (PASS/WARN per row).
7. Circuit drawing helper → PNG into `results/runs/<run_id>/plots/circuit_<name>.png`.
8. Resource-budget guard: raise `ConfigError` above 10 qubits; warn above 8 qubits, reps > 2, transpiled depth > 200.
9. Sanity notebook: build a 4-qubit `ZZFeatureMap`, compute one fidelity by hand vs. the library, confirm `K(x,x) = 1`.
10. **Isolation check:** grep the repo — `import qiskit` must appear only under `src/quantum/`.
11. **IBM Quantum account setup (new — do this on Day 1, in the first hour):** create/verify the account, complete the §8.5.5 verification checklist, save credentials to a git-ignored `.env`, commit `.env.example`.
12. `backends_ibm.py::IBMRuntimeBackend`: same `QuantumBackend` protocol, `SamplerV2` in batch/job mode, job submission + polling with a bounded timeout, quota pre-flight estimate, and recording of device name, job IDs, calibration timestamp, queue seconds and QPU seconds.
13. **Hello-QPU test:** submit one trivial 2-qubit Bell circuit to a real device, retrieve counts, confirm the whole auth → submit → poll → retrieve loop works. Record QPU seconds consumed. **This is the de-risking milestone** — do it before writing any hardware kernel code.
14. Replace `device_profile_generic27q.yaml` with `device_profile_ibm.yaml` built from the **actual** device (`backend.configuration()`): qubit count, basis gates, coupling map.
15. Confirm `HWV-8`: with `.env` absent, the whole pipeline still runs on simulators without errors — hardware is strictly additive.

**Testing requirements**

- `test_quantum_backend.py`: statevector backend runs a 2-qubit circuit; two runs with the same seed give identical results; sampling backend returns counts summing to `shots`; `capabilities()` populated.
- `test_ibm_backend.py`: **mocked** — no QPU calls in CI. Asserts the protocol is satisfied, the quota pre-flight estimate is computed, a missing `.env` degrades gracefully (clear error, simulator path unaffected), and job metadata is recorded. The one real hardware call is a manual, logged smoke test, not a CI test.
- `test_feature_maps.py`: all three maps build at n=4; parameter count matches expectation; `reps` increases depth; invalid entanglement raises.
- `test_transpile_report.py`: report contains all required fields; a 12-qubit request against a 27-qubit profile passes; a 30-qubit request fails the checklist.
- `test_budget.py`: 11 qubits raises `ConfigError`.

**Definition of Done**

- [ ] Every team member can run a Qiskit circuit on Aer with the pinned versions
- [ ] `requirements.txt` contains exact Qiskit pins; `docs/DECISIONS.md` records them
- [ ] Feature maps build from config for n = 4, 6, 8
- [ ] A circuit diagram PNG is produced
- [ ] Transpilation report + hardware-readiness checklist generated against the **real** device profile
- [ ] Qubit/depth budgets enforced with clear errors
- [ ] `import qiskit` appears **only** inside `src/quantum/`
- [ ] **IBM Quantum account verified; a Bell circuit executed on a real device; counts retrieved**
- [ ] **§8.5.5 verification checklist answered and recorded in `docs/DECISIONS.md`** (plan, quota, session-vs-batch, device, queue times, runtime version)
- [ ] **`.env` git-ignored; `.env.example` committed; no credential anywhere in git history**
- [ ] **Pipeline verified to run end to end with `.env` absent** (hardware is additive, never required)

---

### Phase 4 — QSVM (mandatory quantum model)

| Field | Content |
|---|---|
| **Objective** | A working quantum-kernel SVM producing predictions and metrics on the same `ProcessedDataset` as the classical models. **This is the critical path of the whole project.** |
| **Inputs** | `X_*_reduced` (n=6, `[0, π]`), `QuantumBackend`, feature maps |
| **Outputs** | Cached kernel matrices, trained QSVM, predictions, probabilities, resource report, kernel diagnostics |

**Files/modules to create**

```
src/quantum/kernel.py     # kernel computation, symmetry exploitation, disk cache, diagnostics
src/quantum/qsvm.py       # QSVM implementing the Model interface
```

**Libraries** — `qiskit-machine-learning` (`FidelityQuantumKernel`), scikit-learn (`SVC(kernel='precomputed')`, `GridSearchCV`), NumPy, joblib

**Implementation tasks**

1. `kernel.py::compute_kernel(X_a, X_b, feature_map, backend, shots)`:
   - symmetric path when `X_a is X_b` — compute the upper triangle only, set the diagonal to 1;
   - batch circuit submission;
   - progress callback for the CLI.
2. Disk cache: key = SHA-256 of (data bytes, feature map name, reps, entanglement, backend name, shots) → `results/cache/kernels/<hash>.npy`. Log cache hit/miss.
3. `qsvm.py::QSVM(Model)`:
   - `fit(X_train, y_train)`: compute `K_train`, grid-search `C ∈ {0.1, 1, 10}` with `SVC(kernel='precomputed', probability=True)` via stratified CV on **training** kernel sub-blocks;
   - `predict/predict_proba(X)`: compute `K_test` (test × train), call the fitted SVC;
   - `save/load`: persist support vectors, dual coefficients, `C`, feature-map config, training reference data needed for the test kernel;
   - `resource_report()`: qubits, logical + transpiled depth, `n_trainable_params = 0`, circuits executed, shots, backend, `quantum_seconds`, `cache_hit`.
4. Sample cap: subsample above `max_train_samples` with a loud warning recorded in the run record.
5. **Kernel diagnostics**: mean/std of off-diagonal entries, eigenspectrum, effective rank → warn when `std_offdiag < 0.01` (kernel concentration).
6. Pre-flight estimate: circuits to run + estimated wall clock, printed before starting.
7. Wire QSVM into `ExperimentRunner` alongside the classical models.

**Testing requirements**

- `test_qsvm.py` (**fast**: 2 qubits, 20 synthetic samples, statevector): kernel is symmetric, diagonal ≈ 1, entries in `[0, 1]`; QSVM fits and predicts; `predict_proba` shape correct; save/load round-trips.
- `test_kernel_cache.py`: second computation with identical inputs is a cache hit and returns an identical matrix; changing `reps` misses the cache.
- `test_model_contract.py`: QSVM passes the same contract tests as the classical models.
- Manual: on a linearly separable 2-D toy problem, QSVM accuracy > 0.9 — a sanity check that the plumbing is correct.

**Definition of Done**

- [ ] QSVM trains on the real reduced UCI data (n=6) and produces test predictions
- [ ] Kernel matrix computed in ≤ 10 minutes and cached; a second run is a cache hit
- [ ] All 14 metrics computed for QSVM and written to `metrics.csv`
- [ ] `resource_report()` complete, including `n_trainable_params = 0`
- [ ] Kernel-concentration diagnostic reported
- [ ] Changing feature map / reps / entanglement demonstrably changes the kernel
- [ ] QSVM appears in the run record alongside the classical models with the same `split_id`

---

### Phase 5 — VQC / QNN (secondary; droppable)

| Field | Content |
|---|---|
| **Objective** | A trainable variational classifier, demonstrating the second family of quantum models named in SIH26139. |
| **Inputs** | `X_*_reduced`, feature maps, backend |
| **Outputs** | Trained ansatz parameters, loss curve, predictions, resource report |
| **Contingency** | **If Phase 4 is not complete by end of Week 6, cut this phase** (§2.4) and reallocate to Phases 6–9. |

**Files/modules to create** — `src/quantum/vqc.py`, `src/quantum/ansatz.py` (optional split)

**Libraries** — `qiskit-machine-learning` (`VQC` / `EstimatorQNN` + `NeuralNetworkClassifier`), `qiskit-algorithms` optimizers (COBYLA, SPSA), NumPy, matplotlib

**Implementation tasks**

1. Build `ZZFeatureMap(n, reps=2)` → `RealAmplitudes(n, reps=2)`; report `ansatz.num_parameters` (18 at n=6, reps=2).
2. `VQCModel(Model)` wrapping Qiskit's `VQC`: seeded initial point, `COBYLA(maxiter=200)`, callback capturing loss per iteration.
3. 3 random restarts; select the best by **training-fold** score; record restart-to-restart variance.
4. Loss-curve plot → `plots/vqc_loss.png`.
5. Barren-plateau smoke test: variance of parameter updates / finite-difference gradients at initialization; warn if below threshold.
6. Persist parameters (`.npy`) + config (JSON); `load` reconstructs and predicts identically.
7. Hard 15-minute timeout per fit → early stop with a partial result recorded, never a hang.
8. `resource_report()` with `n_trainable_params`, circuits executed, optimizer, iterations, `quantum_seconds`.

**Testing requirements**

- `test_vqc.py` (fast: 2 qubits, 30 samples, `maxiter=20`): fits, loss is recorded and decreases from its initial value, `predict_proba` valid, save/load round-trips.
- Contract test passes.
- Timeout path tested with an artificially tiny budget.

**Definition of Done**

- [ ] VQC trains on the real reduced data and produces test predictions
- [ ] Loss curve recorded and plotted
- [ ] `n_trainable_params` reported (actual value from the ansatz)
- [ ] 3 restarts run; variance reported
- [ ] Metrics written alongside QSVM and the classical models
- [ ] *(or)* the phase is formally cut, with the decision and date recorded in `docs/DECISIONS.md`

---

### Phase 6 — Evaluation, Benchmarking, Robustness

| Field | Content |
|---|---|
| **Objective** | Turn raw predictions into a **fair, statistically qualified comparison** with confidence intervals, significance tests, verdict strings, efficiency accounting, and robustness evidence. This phase is what separates the project from a notebook. |
| **Inputs** | All trained models + `ProcessedDataset` |
| **Outputs** | Leaderboard, CV table, CIs, McNemar/DeLong verdicts, ROC/PR/confusion plots, robustness reports |

**Files/modules to create**

```
src/evaluation/statistics.py   # bootstrap CI, McNemar, DeLong (or paired bootstrap AUC)
src/evaluation/benchmark.py    # BenchmarkEngine, UnfairComparisonError, verdicts, protocols A/B
src/evaluation/robustness.py   # repeated CV, seeds, noise sweep, subgroups, gen-gap, learning curve
src/evaluation/plots.py        # ROC/PR overlays, confusion matrices, calibration, degradation curves
```

**Libraries** — scikit-learn, SciPy, statsmodels (optional, for McNemar), NumPy, matplotlib

**Implementation tasks**

1. `statistics.py`: percentile bootstrap CI (1000 resamples, seeded); McNemar's test on paired errors; DeLong test for correlated AUCs (or a paired bootstrap of the AUC difference — simpler and acceptable, but say which was used).
2. `benchmark.py::BenchmarkEngine`:
   - `evaluate_all(models, dataset)` → metrics + CIs for every model;
   - **fairness guard**: raise `UnfairComparisonError` when `split_id` or `preprocess_hash` differ;
   - `compare(a, b, metric)` → the JSON object in §9.3 with a **verdict string chosen by the test, not the delta**;
   - Protocol A and Protocol B groupings.
3. Threshold policy: 0.5 and sensitivity-target-0.90 chosen on training folds only; both rows reported.
4. `robustness.py`:
   - repeated stratified 5-fold × 3 with the **full chain refit per fold**;
   - multi-seed driver (42/43/44);
   - noise sweep on test features (σ in units of train std) reusing already-trained models;
   - subgroup slices by `sex` and age band;
   - generalization gap (train − test) with a >0.10 flag;
   - learning curve (classical only).
5. `plots.py`: ROC overlay, PR overlay, per-model confusion matrix, calibration curve, noise-degradation curve, subgroup bars → PNGs into the run directory.
6. Efficiency table assembling `resource_report()` from every model.
7. Run the **feature-set ablation**: `run_mvp.yaml` vs `run_screening.yaml`, reported side by side.
8. Export `metrics.csv`, `metrics.json`, `comparisons.json`, `cv_results.csv`.
9. **Hardware validation experiment (§8.5)** — `src/quantum/hardware_validation.py`:
   - fix a seeded, stratified 30-train / 10-test subsample of the same `ProcessedDataset`;
   - compute the identical kernel submatrix on `aer_statevector`, `aer_noisy`, and the **real IBM device** (735 circuits, ~3–4 min QPU);
   - persist all three matrices plus job metadata to `results/runs/<run_id>/hardware/`;
   - compute the comparison statistics: mean/max \|K_hw − K_exact\|, correlation, diagonal deviation from 1.0, off-diagonal mean/std per backend;
   - fit `SVC(precomputed)` on each kernel and evaluate on the 10 test points, **with an explicit "n=10, not a performance claim" caveat attached to the numbers**;
   - render three heatmaps + an error histogram to `plots/`.
   - **Run this once, in Week 9. Cache the result. Never re-run for a demo.**

**Testing requirements**

- `test_statistics.py`: bootstrap CI of a known Bernoulli sample brackets the true value; McNemar on identical predictors gives p ≈ 1; AUC test on a clearly better model gives p < 0.05.
- **`test_benchmark_fairness.py`**: comparing models with different `split_id` raises `UnfairComparisonError`; same-split comparison succeeds.
- `test_verdict.py`: a large delta with p ≥ 0.05 still yields `"no significant difference"` — **the anti-overclaim guarantee, tested**.
- `test_robustness.py`: noise sweep returns one row per σ; subgroup slices sum to the full test set; CV returns `k × repeats` rows.

**Definition of Done**

- [ ] Leaderboard with all 14 metrics + bootstrap CIs for every model
- [ ] CV mean ± std table produced (5-fold × 3 repeats)
- [ ] McNemar and AUC tests run for every quantum-vs-classical pair, with verdict strings
- [ ] Protocol A and Protocol B both reported and labelled
- [ ] Efficiency table complete (time, circuits, depth, shots)
- [ ] Noise-degradation curve, subgroup table, generalization gap, seed variance produced
- [ ] Feature-set ablation (`full` vs `screening`) produced
- [ ] `test_benchmark_fairness.py` and `test_verdict.py` green in CI
- [ ] All plots written as PNGs to the run directory
- [ ] **Hardware validation run executed on a real IBM device; three kernel matrices + error statistics + job provenance persisted and cached**
- [ ] **Hardware results load from cache with networking disabled**

---

### Phase 7 — Explainability

| Field | Content |
|---|---|
| **Objective** | Explain classical **and** quantum predictions, globally and locally, with the method and its limitations stated on screen. |
| **Inputs** | Trained models, `ProcessedDataset`, PCA loadings |
| **Outputs** | Global importance, local attributions, PCA loading map, kernel-similarity panel, method/limitation text |

**Files/modules to create**

```
src/explainability/classical_explainer.py
src/explainability/quantum_explainer.py
src/explainability/loading_map.py
src/explainability/text.py            # method notes + limitation strings (single source of truth)
```

**Libraries** — `shap`, `sklearn.inspection.permutation_importance`, NumPy, matplotlib

**Implementation tasks**

1. Classical global: LR coefficients + odds ratios; SHAP `TreeExplainer` for XGB/RF; SHAP summary plot.
2. Classical local: SHAP waterfall for a selected record.
3. Quantum global: `permutation_importance` on `X_test_reduced` scored by ROC-AUC, 10 repeats, seeded.
4. Quantum local: `shap.KernelExplainer` over `model.predict_proba`, background summarized with `shap.kmeans(X_train_reduced, 100)`, `nsamples=200`, 5-minute timeout.
5. `loading_map.py`: PCA `components_` → heatmap + top-3 clinical contributors per component; translate component attributions back to approximate clinical-feature attributions (**explicitly labelled as an approximation**).
6. QSVM kernel-similarity panel: top-5 nearest training patients by quantum kernel value, with their labels.
7. `text.py`: one dictionary holding every method note and limitation string, consumed by both the API and the docs so the wording never diverges.
8. Serve all plots as PNGs; return attribution values as JSON for the frontend tables.

**Testing requirements**

- `test_classical_explainer.py`: SHAP values array shape matches `(n_features,)`; the sum of local SHAP values + base value ≈ the model output (within tolerance).
- `test_quantum_explainer.py`: permutation importance returns one score per reduced feature; KernelSHAP on a tiny model returns finite values; the timeout path returns a graceful message.
- `test_loading_map.py`: loading matrix shape `(n_components, n_features)`; rows have unit norm.
- `test_text.py`: every explanation method has a non-empty method note **and** a non-empty limitation string.

**Definition of Done**

- [ ] Global importance available for at least one classical and one quantum model
- [ ] Local explanation available for a selected patient in both branches
- [ ] PCA loading map rendered, with the approximation caveat
- [ ] Method note + limitation text attached to every explanation output
- [ ] The classical-vs-quantum explainability comparison table exists as data (not hardcoded in JSX)
- [ ] Quantum explanation completes within 5 minutes or degrades gracefully

---

### Phase 8 — Backend API

| Field | Content |
|---|---|
| **Objective** | Expose every stored result and the inference path over nine HTTP endpoints, with no ML logic in the API layer. |
| **Inputs** | Completed run directories in `results/runs/` |
| **Outputs** | Running FastAPI service with OpenAPI docs at `/docs` |

**Files/modules to create**

```
src/inference/artifacts.py, src/inference/predictor.py
backend/main.py, backend/schemas.py, backend/deps.py
backend/routers/dataset.py, models.py, predict.py, results.py, explain.py, quantum.py
```

**Libraries** — FastAPI, Uvicorn, Pydantic v2, python-multipart (uploads)

**Implementation tasks**

1. `artifacts.py`: load a run's preprocessing artifacts + models; raise `ArtifactMismatchError` on schema mismatch; cache in memory.
2. `predictor.py`: `predict_one(patient_dict, model_id)` — apply stored artifacts (**never refit**) → probability, label, risk band, threshold, top factors. `predict_batch(csv)`.
3. `schemas.py`: `PatientInput` with field ranges, `PredictionResponse` with a mandatory non-empty `disclaimer`.
4. Nine routers per §14.1; static plot mount for `/plots/{run_id}/...`.
5. `deps.py`: dependency-injected cached artifact provider; startup event loads the configured demo run and **fails loudly** if missing.
6. CORS for `http://localhost:5173`.
7. Error envelope `{error_code, message, details, run_id}` via an exception handler.
8. Never persist patient inputs; log only shapes and timings.

**Testing requirements**

- `test_api.py` with `TestClient`: `/api/health` 200; `/api/predict` returns a valid response with a non-empty disclaimer; an out-of-range `age` returns 422 naming the field; `/api/results/{bad_id}` returns a structured 404; `/api/quantum/info` returns the resource report; batch upload returns a CSV.
- `test_inference.py`: prediction for a training row through the API path matches the in-process model prediction (proves artifact reuse is correct).
- Latency check: classical prediction < 1 s.

**Definition of Done**

- [ ] `uvicorn backend.main:app --reload` starts and serves `/docs`
- [ ] All nine endpoints return valid responses against a real run
- [ ] Single-patient prediction works for both a classical and a quantum model
- [ ] Batch CSV scoring returns a downloadable file
- [ ] Validation errors are field-specific
- [ ] Disclaimer present in every prediction response
- [ ] `test_api.py` green in CI

---

### Phase 9 — Frontend Dashboard

| Field | Content |
|---|---|
| **Objective** | Seven pages that display everything the backend computed. **No logic, no computation, no cleverness.** |
| **Inputs** | Running backend API |
| **Outputs** | React app on `localhost:5173` |

**Files/modules to create** — `frontend/` per §16, 7 pages + ~20 components + `api/client.js`

**Libraries** — react, react-dom, react-router-dom, recharts, vite

**Implementation tasks**

1. Scaffold Vite + React; configure the `/api` proxy to port 8000.
2. `App.jsx`: layout, nav bar, 7 routes, `RunContext` holding `run_id`, permanent disclaimer footer.
3. `api/client.js`: one function per endpoint; all fetches live here.
4. Shared components: `StatCard`, `MetricTable`, `VerdictBadge`, `PlotImage`, `PatientForm`, `RiskCard`, `FileUpload`, `RunSelector`, `Disclaimer`, `MethodNote`, `LoadingSpinner`, `ErrorBanner`.
5. Build the 7 pages per §13.1, in this order (highest demo value first): **Model Comparison → Patient Prediction → Quantum Info → Dataset Analysis → Explainability → Experiment Results → Home**.
6. Render heavy plots as `<PlotImage>` (backend PNGs); use Recharts only for simple bars/lines.
7. Loading and error states on every page — never a blank screen.
8. One stylesheet with CSS variables; consistent spacing, one accent colour, readable tables.

**Testing requirements** — manual checklist per page (loads, no console errors, handles the API being down, handles an empty run); one outside tester completes the full workflow unaided; verify no numeric computation exists in JS (search for arithmetic on metrics).

**Definition of Done**

- [ ] All 7 pages render real data from the backend
- [ ] Patient prediction form submits and displays classical + quantum results side by side
- [ ] Model Comparison shows the leaderboard, plots, and verdict badges
- [ ] Quantum Info shows the circuit diagram, resource report, and hardware checklist
- [ ] Explainability shows global + local plots with method and limitation notes
- [ ] Disclaimer visible on Home, Model Comparison, and Patient Prediction
- [ ] No ML computation in frontend code
- [ ] *(or)* the Streamlit fallback is delivered with equivalent coverage

---

### Phase 10 — Integration

| Field | Content |
|---|---|
| **Objective** | One command produces every artifact the demo needs; the whole system runs from a clean clone. |
| **Inputs** | All previous phases |
| **Outputs** | Cached demo run, integration scripts, end-to-end verified flow |

**Files/modules to create** — `scripts/run_all_experiments.sh`, `scripts/prepare_demo.sh`, `docs/DEMO_SCRIPT.md`, `Makefile` (optional)

**Implementation tasks**

1. `run_all_experiments.sh`: full run (`full` feature set), screening run, 3-seed runs, noise sweep, quantum sweep (qubits × feature maps) — everything expensive, run once, before demo week.
2. `prepare_demo.sh`: select the canonical run, warm the kernel cache, generate every plot, verify the backend loads it in < 5 s.
3. Add the optional small live-training endpoint (`POST /api/train/demo`) **only if** Phases 1–9 are green.
4. Clean-clone rehearsal: fresh directory → install → download data → run → serve → open dashboard. Time it and fix every friction point.
5. Verify **offline operation**: disable networking and repeat.
6. Back up the demo run directory outside the repo (USB + cloud).
7. Write `docs/DEMO_SCRIPT.md`: the minute-by-minute run sheet.

**Testing requirements** — full end-to-end run from a clean clone on **two different machines**; offline run succeeds; cached demo loads in ≤ 5 s; every page populated.

**Definition of Done**

- [ ] `scripts/run_all_experiments.sh` reproduces every result
- [ ] `scripts/prepare_demo.sh` produces the cached demo run
- [ ] Clean-clone install-to-dashboard works on two machines
- [ ] Entire demo runs with networking disabled
- [ ] Demo run backed up in two places
- [ ] `docs/DEMO_SCRIPT.md` written and rehearsed once

---

### Phase 11 — Testing, Hardening, Documentation

| Field | Content |
|---|---|
| **Objective** | Prove correctness, handle failure gracefully, and write the documentation SIH26139 explicitly requires. |
| **Inputs** | Complete system |
| **Outputs** | Green test suite ≥70% coverage on `src/`, hardened error paths, six documents |

**Implementation tasks**

1. Fill test gaps to ≥ 70% coverage on `src/` (`pytest --cov=src`).
2. **Re-verify `test_leakage.py` and `test_reproducibility.py`** — these are the two tests a judge might ask to see run live.
3. Failure-injection pass: corrupt CSV, missing column, out-of-range field, missing run directory, backend down, over-budget qubit request. Each must produce an actionable message, never a stack trace in the UI.
4. Ensure a single model's failure does not abort the run (fail soft) while a bad config fails fast.
5. Write the documentation set:
   - `README.md` — install, run, demo in 10 lines
   - `docs/ARCHITECTURE.md` — diagrams, module contracts, `ProcessedDataset`
   - `docs/METHODS.md` — dataset spec, preprocessing recipe, models, metrics, statistics, seeds
   - **`docs/LIMITATIONS.md`** — n≈303, simulator not hardware, reduction information loss, multiple comparisons, `ca`/`thal` clinical leakage, explanation caveats, no clinical validation
   - `docs/SIH_TRACEABILITY.md` — requirement → feature → screen
   - `docs/DEMO_SCRIPT.md` — the run sheet
6. Docstrings on every public function; `ruff` and `black` clean.
7. **Honesty audit:** grep the entire repo, docs, and UI strings for "advantage", "outperform", "better", "superior" — every hit must be attached to a CI and a p-value or be removed.

**Testing requirements** — full suite green; coverage report ≥ 70% on `src/`; all failure-injection cases produce clean messages; docs reviewed by a member who did not write them.

**Definition of Done**

- [ ] `pytest` green; coverage ≥ 70% on `src/`
- [ ] Leakage and reproducibility tests pass and can be demonstrated live
- [ ] Every failure-injection case handled gracefully
- [ ] All six documents written and accurate
- [ ] Honesty audit complete — no unqualified superiority claim anywhere
- [ ] `ruff` and `black` clean; CI green

---

### Phase 12 — Final SIH Demo Preparation

| Field | Content |
|---|---|
| **Objective** | A rehearsed, robust, honest 10-minute demonstration that maps visibly onto SIH26139. |
| **Inputs** | Frozen system, cached demo run |
| **Outputs** | Rehearsed demo, slides, traceability page, Q&A preparation |

**Implementation tasks**

1. **Code freeze** (5 days out). Only bug fixes and docs after this point.
2. Tag `v1.0-sih-demo`.
3. Finalize the 10-minute run sheet (§25.2).
4. Rehearse **three times** end to end, including a deliberate failure recovery (kill the backend and restart mid-demo).
5. Prepare the offline fallback: screenshots and a screen recording of the full flow, on the laptop, in case of hardware failure.
6. Build the slide deck: problem → approach → architecture → results → honesty → limitations → future work.
7. Prepare the **Q&A brief** (§25.3) — the five questions a strong panel will ask.
8. Assign speaking roles: one driver, one narrator, one for quantum questions, one for ML/statistics questions.
9. Verify the demo laptop: charged, offline-capable, correct environment, backup laptop configured identically.

**Testing requirements** — three timed full rehearsals under 10 minutes; one rehearsal with networking off; one with a mid-demo backend restart; every team member can answer the Q&A brief.

**Definition of Done**

- [ ] Code frozen and tagged `v1.0-sih-demo`
- [ ] Three successful rehearsals completed within the time limit
- [ ] Offline fallback (recording + screenshots) prepared
- [ ] Slides finalized
- [ ] Q&A brief rehearsed by every member
- [ ] Two laptops configured identically
- [ ] The final acceptance checklist (§19) is fully PASS
---

## 19. Final MVP Acceptance Test

Run this checklist top to bottom on a **clean clone**, on the demo laptop, with networking disabled. Every line is PASS or FAIL. **The MVP is complete only when every MUST line is PASS.**

### 19.1 Core acceptance checklist (from the brief)

| # | Item | How to verify | Result |
|---|---|---|---|
| 1 | **Dataset loads correctly** | `python -m src.cli validate --config configs/run_mvp.yaml` → 303 rows, 14 columns, SHA-256 verified | [ ] PASS [ ] FAIL |
| 2 | **Data validation works** | Validation report lists `?` in `ca`/`thal`, class balance, duplicates, ranges; a deliberately corrupted CSV is rejected with a named reason | [ ] PASS [ ] FAIL |
| 3 | **Leakage-safe preprocessing works** | `pytest tests/test_leakage.py -v` → green; split precedes every `.fit()`; artifacts fitted on train only | [ ] PASS [ ] FAIL |
| 4 | **Feature selection works** | Ranked feature table produced (MI + L1) with scores; top-*k* subset applied | [ ] PASS [ ] FAIL |
| 5 | **Dimensionality reduction works** | PCA → exactly `n_components` columns; explained variance reported; all values within `[0, π]` | [ ] PASS [ ] FAIL |
| 6 | **Classical models train** | LR, RBF-SVM, XGB/RF each trained on full-*d* **and** reduced-*n* with tuned hyperparameters (6 rows) | [ ] PASS [ ] FAIL |
| 7 | **QSVM trains** | Quantum kernel computed, cached, SVC(precomputed) fitted, predictions produced | [ ] PASS [ ] FAIL |
| 8 | **Prediction works** | Single-patient form and batch CSV both return probability, band, threshold, factors | [ ] PASS [ ] FAIL |
| 9 | **Metrics are calculated** | Accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC, confusion matrix, train time, inference time — for every model | [ ] PASS [ ] FAIL |
| 10 | **Classical vs quantum comparison works** | Same `split_id` + `preprocess_hash`; Protocol A and B; deltas with CIs; McNemar/AUC p-values; verdict strings | [ ] PASS [ ] FAIL |
| 11 | **Explainability works** | Global + local explanations for a classical **and** a quantum model, with method and limitation notes | [ ] PASS [ ] FAIL |
| 12 | **Robustness/generalization evaluation works** | Repeated CV mean ± std, 3-seed variance, noise-degradation curve, subgroup table, generalization gap | [ ] PASS [ ] FAIL |
| 13 | **Backend works** | `uvicorn backend.main:app` serves all 9 endpoints; `/docs` renders | [ ] PASS [ ] FAIL |
| 14 | **Frontend works** | All 7 pages load real data with no console errors | [ ] PASS [ ] FAIL |
| 15 | **End-to-end prediction works** | Enter a patient in the browser → risk score + explanation returned in < 10 s | [ ] PASS [ ] FAIL |
| 16 | **Results are reproducible** | `pytest tests/test_reproducibility.py` green; re-running the demo config reproduces `metrics.json` exactly (statevector) | [ ] PASS [ ] FAIL |
| 17 | **Documentation exists** | README + ARCHITECTURE + METHODS + LIMITATIONS + SIH_TRACEABILITY + DEMO_SCRIPT all present and accurate | [ ] PASS [ ] FAIL |
| 18 | **SIH26139 requirements demonstrably satisfied** | Every row of the §4 coverage table can be shown on a named screen within the 10-minute demo | [ ] PASS [ ] FAIL |

### 19.2 Supplementary acceptance checks (project-specific, still MUST)

| # | Item | Verify | Result |
|---|---|---|---|
| 19 | Fairness guard active | Comparing models across different splits raises `UnfairComparisonError` (`pytest tests/test_benchmark_fairness.py`) | [ ] PASS [ ] FAIL |
| 20 | Anti-overclaim guard active | `pytest tests/test_verdict.py` — a large delta with p ≥ 0.05 still yields "no significant difference" | [ ] PASS [ ] FAIL |
| 21 | Quantum resource report complete | Qubits, logical + transpiled depth, trainable params, circuits, shots, backend, quantum seconds all present | [ ] PASS [ ] FAIL |
| 22 | Hardware-readiness checklist | Transpilation against the device profile produces a PASS/WARN checklist | [ ] PASS [ ] FAIL |
| 23 | Shot-based path works | Setting `shots: 1024` runs the same models with **no code change** | [ ] PASS [ ] FAIL |
| 23a | **Real hardware executed** | A kernel submatrix was computed on a named IBM Quantum device; job IDs, calibration timestamp and QPU seconds are in the run record | [ ] PASS [ ] FAIL |
| 23b | **Hardware vs simulator quantified** | Three kernel heatmaps + error statistics (mean/max \|ΔK\|, diagonal deviation) rendered on the Quantum Info page | [ ] PASS [ ] FAIL |
| 23c | **Hardware is additive, not required** | With `.env` removed, the full pipeline, backend and dashboard still work; only the hardware panel shows "not available" | [ ] PASS [ ] FAIL |
| 23d | **No credentials in git** | `git log -p \| grep -i "api_key\|token"` returns nothing; `.env` is git-ignored | [ ] PASS [ ] FAIL |
| 23e | **Hardware results cached** | Quantum Info hardware panel renders with networking disabled | [ ] PASS [ ] FAIL |
| 24 | Feature-set ablation reported | `full` vs `screening` (no `ca`/`thal`) results shown side by side | [ ] PASS [ ] FAIL |
| 25 | Qiskit isolation | `grep -r "import qiskit" src/ backend/` returns hits only under `src/quantum/` | [ ] PASS [ ] FAIL |
| 26 | Offline operation | Full demo runs with networking disabled | [ ] PASS [ ] FAIL |
| 27 | Cached demo speed | Demo run loads in the dashboard in ≤ 5 s | [ ] PASS [ ] FAIL |
| 28 | Disclaimer present | Visible on Home, Model Comparison, Patient Prediction, and in every API prediction response | [ ] PASS [ ] FAIL |
| 29 | Honesty audit | No unqualified "outperforms / better / advantage" anywhere in code, docs, or UI | [ ] PASS [ ] FAIL |
| 30 | Sanity guard | Classical ROC-AUC is plausible (~0.85–0.92), **not** 1.0 — a perfect score means leakage | [ ] PASS [ ] FAIL |

### 19.3 Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| Team Lead | | | |
| Quantum Lead | | | |
| ML Lead | | | |

**MVP status:** [ ] COMPLETE (all MUST rows PASS)  [ ] INCOMPLETE (list failing rows and owners)

---

## 20. MVP vs Advanced Feature Table

| Area | **MVP (build now)** | **Advanced / Future (do not build now)** |
|---|---|---|
| **Dataset** | UCI Heart Disease (Cleveland), CSV upload, YAML schema | Larger CVD cohort, multi-site harmonization, EHR/FHIR ingestion, imaging, genomics, ECG waveforms |
| **Target** | Binary `num > 0` | Multi-class severity, survival / time-to-event, multi-disease |
| **Validation** | Schema, sentinels, missingness, balance, duplicates, ranges, correlation | Great Expectations suite, automated data-drift monitoring |
| **Split** | Stratified 80/20 + repeated stratified CV | Grouped, temporal, site-based OOD splits; nested CV |
| **Imputation** | Median / most-frequent | KNN, iterative/MICE, missingness-indicator learning |
| **Imbalance** | Class weights | SMOTE variants, cost-sensitive thresholds, focal loss |
| **Feature engineering** | 5 clinical derived features | Automated feature synthesis, symbolic regression, domain-knowledge graphs |
| **Feature selection** | Mutual information + L1-logistic | RFE-CV, stability selection, Boruta, quantum-aware selection |
| **Reduction** | PCA + top-*k*, n = 4–8 | Kernel PCA, LDA, autoencoders, UMAP, learned quantum-aware embeddings |
| **Classical models** | LR, RBF-SVM, XGB/RF (tuned) | Stacking/ensembles, TabNet, AutoML, calibrated ensembles |
| **Quantum models** | **QSVM (mandatory)**, VQC (secondary) | Quantum kernel ridge, quantum boosting, QGAN augmentation, quantum ensembles |
| **Feature maps** | **Standard published maps** (ZZ / Z / Pauli), fixed | **Adaptive / optimized feature maps**, kernel-target alignment, feature-map architecture search, data re-uploading |
| **Encoding** | Angle encoding to `[0, π]` | Amplitude encoding, basis encoding, hybrid encodings |
| **Execution** | Aer statevector + sampling (+ one simple noise model) | Real QPU execution, error mitigation (ZNE, readout mitigation), tensor-network / GPU simulation |
| **Hardware** | Transpilation report + budgets **and** a scoped IBM Quantum validation run (735 circuits, 30×30 kernel, 3-backend comparison) | Full benchmark on hardware, repeated CV on hardware, multi-device comparison, error mitigation (ZNE, readout), cost-optimized batch scheduling, paid-plan execution |
| **Evaluation** | 14 metrics, bootstrap CIs, McNemar + AUC test, repeated CV, 3 seeds | Nested CV, Bayesian model comparison, multiplicity correction, decision-curve analysis |
| **Explainability** | SHAP (classical), permutation + KernelSHAP (quantum), PCA loading map, kernel similarity | Counterfactuals, anchors, concept-based explanations, quantum-native circuit attribution |
| **Robustness** | CV, seeds, noise, subgroups, gen-gap, feature-set ablation | Adversarial robustness, distribution-shift benchmarks, external cohort validation, fairness mitigation |
| **Scalability** | Config-driven design + documented O(N²) bottleneck + sample cap | Nyström / random-feature kernel approximation, distributed simulation, streaming ingestion |
| **Backend** | FastAPI, 9 endpoints, filesystem state | Async job queue, database, caching layer, horizontal scaling |
| **Frontend** | React, 7 pages, server-rendered plots | Interactive plots, real-time training view, report builder, theming |
| **Tracking** | JSON run registry | MLflow / Weights & Biases, experiment leaderboards, hyperparameter sweeps at scale |
| **Ops** | Local run, pinned requirements | Docker, CI/CD deployment, monitoring, model-drift alerts |
| **Security** | No PII, local-only, no persisted inputs | Auth, RBAC, TLS, encryption at rest, audit trails, HIPAA/DISHA alignment |
| **Clinical** | Research prototype with disclaimers | Prospective validation, ethics approval, regulatory pathway, clinician trial |

**How to use this table in the pitch:** the left column is what we built and can demonstrate; the right column is our stated roadmap. Presenting both signals scope discipline — the most common failure mode in hackathon projects is claiming the right column while demonstrating half the left one.

---

## 21. Week-by-Week Development Roadmap

Assumes a **10-week** runway with ~15–20 hours per member per week alongside coursework. Compression guidance follows in §21.2.

| Week | Phases | Primary deliverables | Checkpoint / gate |
|---|---|---|---|
| **W1** | Phase 0 + start Phase 1 | Repo, pinned env, CI green, data downloaded + checksummed, EDA notebook, `schema_uci_heart.yaml` drafted | Everyone can run `pytest`; XGB-vs-RF decided; Python version fixed |
| **W2** | Phase 1 | Loader, validator, splitter, preprocessing chain, engineering, selection, reduction, range normalizer → `ProcessedDataset` | **GATE 1: `test_leakage.py` green.** No later phase starts until this passes |
| **W3** | Phase 2 + start Phase 3 | 6 classical models trained and evaluated; metrics module; run registry; Qiskit environment pinned | Classical ROC-AUC plausible (not 1.0); Qiskit imports work for everyone |
| **W4** | Phase 3 + start Phase 4 | Backends, feature maps, transpilation report, **real device profile**, circuit diagrams; **IBM account verified + Bell circuit executed on a QPU**; kernel computation started | **GATE 2: a quantum circuit runs on Aer for every member, and one Bell circuit has run on a real IBM device** |
| **W5** | Phase 4 | Kernel computation + caching, QSVM fit/predict, resource report, diagnostics | QSVM produces predictions on real reduced data |
| **W6** | Phase 4 finish → Phase 5 + start Phase 6 | QSVM metrics in the run record; VQC started; benchmarking module begun | **GATE 3 (the big one): QSVM evaluated end to end.** If not met → **cut VQC** and move everyone to Phases 6–9 |
| **W7** | Phase 6 + Phase 7 | Statistics, benchmark engine + fairness guard + verdicts, robustness suite, plots; SHAP + quantum explainers | Leaderboard with CIs and p-values exists; explanations render |
| **W8** | Phase 8 + start Phase 9 | Inference service, 9 API endpoints, OpenAPI docs; frontend scaffold + 3 highest-value pages | **GATE 4: end-to-end prediction through HTTP works.** Decide React vs Streamlit fallback |
| **W9** | Phase 9 + Phase 10 | Remaining 4 pages; `run_all_experiments.sh`; **hardware validation run executed once on the real QPU and cached**; cached demo run; clean-clone rehearsal; offline check | Full workflow usable by an outside tester; **hardware results in the registry** |
| **W10** | Phase 11 + Phase 12 | Coverage ≥ 70%, failure hardening, 6 documents, honesty audit, code freeze, 3 rehearsals | **GATE 5: §19 acceptance checklist fully PASS** |

### 21.1 The four rules that keep the roadmap honest

1. **Gate 1 (Week 2) is absolute.** If leakage-safety is not proven, every downstream number is worthless. Do not proceed on hope.
2. **Gate 3 (Week 6) triggers the only pre-authorized cut.** QSVM evaluated → keep VQC. Not evaluated → cut VQC that day, no debate, record it in `docs/DECISIONS.md`.
3. **Expensive experiments run in Week 9, overnight, once.** Never during demo week.
4. **Week 10 adds no features.** Code freeze on day 5 of Week 10.

### 21.2 Compressed variants

| Runway | How to compress |
|---|---|
| **8 weeks** | Merge W3+W4 (classical and quantum-env in parallel across sub-teams); drop the learning curve, missingness-robustness, and noisy-simulator checks (all SHOULD); keep every MUST |
| **6 weeks** | Above, **plus**: cut VQC from the start (QSVM only); use the **Streamlit** dashboard instead of React from day one; reduce robustness to CV + noise + subgroups; single feature map (ZZ) with no sweep |
| **Never compress** | Leakage safety, the fairness guard, CIs + significance tests, explainability on both branches, the disclaimer, the documentation set |

---

## 22. Team Task Allocation

### 22.1 Default 5-person structure

| Member | Role | Owns (modules) | Owns (phases) | Backup for |
|---|---|---|---|---|
| **M1** | **Team Lead / Integration & Backend** | `src/config.py`, `src/pipeline.py`, `src/tracking/`, `src/inference/`, `backend/` | 0, 8, 10, 12 | M3 |
| **M2** | **Data & Preprocessing Engineer** | `src/data/`, `src/preprocessing/`, `src/features/`, `configs/schema_*` | 1 | M3 |
| **M3** | **Classical ML & Evaluation Engineer** | `src/classical/`, `src/evaluation/` (metrics, statistics, benchmark, robustness, plots) | 2, 6 | M2 |
| **M4** | **Quantum Engineer (Lead)** | `src/quantum/` — backends, feature maps, kernel, QSVM, transpile report | 3, 4 | M5 |
| **M5** | **Quantum #2 / Explainability / Frontend / Docs** | `src/quantum/vqc.py`, `src/explainability/`, `frontend/`, `docs/` | 5, 7, 9, 11 | M4 |

### 22.2 Six-person expansion (recommended if available)

Split M5's overloaded role:

| Member | Role | Owns |
|---|---|---|
| **M5** | **Quantum #2 / Explainability** | `src/quantum/vqc.py`, quantum diagnostics, `src/explainability/` |
| **M6** | **Frontend & Documentation** | `frontend/` (all 7 pages), `docs/`, demo assets, the traceability page |

This is the single highest-value addition, because Phase 9 (frontend) and Phase 11 (docs) both land in the final three weeks and otherwise collide.

### 22.3 Four-person contraction

| Member | Role |
|---|---|
| **M1** | Lead + Backend + Integration + Frontend (**Streamlit, not React** — mandatory at this size) |
| **M2** | Data + Preprocessing + Features |
| **M3** | Classical ML + Evaluation + Robustness |
| **M4** | Quantum (QSVM only — **cut VQC from the start**) + Explainability |

Docs are split: each member writes the section covering their own modules; M1 assembles.

### 22.4 Week-by-member allocation (5-person default)

| Week | M1 (Lead/Backend) | M2 (Data) | M3 (Classical/Eval) | M4 (Quantum) | M5 (Quantum2/XAI/FE/Docs) |
|---|---|---|---|---|---|
| W1 | Repo, CI, env, branch protection | EDA, schema YAML | XGB/RF install decision, metric plan | Qiskit install spike | Frontend spike, docs skeleton |
| W2 | `config.py`, config hashing, run registry | **Loader, validator, splitter, preprocessing** | Metrics module | Feature-map prototype | Assist Phase 1 tests |
| W3 | `pipeline.py` orchestration | Engineering, selection, reduction, normalizer | **Classical models + GridSearchCV** | Backends + device profile | Circuit drawing, docs |
| W4 | Model store, run record schema | Screening feature set, ablation config | Classical metrics + persistence | **Kernel computation + cache** | Transpile report |
| W5 | CLI polish, artifact loading | Support M3/M4 | Bootstrap CI + McNemar | **QSVM fit/predict/resource report** | VQC prototype |
| W6 | Inference service | Robustness data plumbing | **Benchmark engine + fairness guard** | QSVM diagnostics, sweeps | VQC training (or cut → help Eval) |
| W7 | API scaffolding | Subgroup slices, noise injection | **Robustness suite + plots** | Quantum CV with kernel caching | **SHAP + quantum explainers** |
| W8 | **9 endpoints + OpenAPI** | Batch prediction path | Verdict tests, threshold policy | Quantum sweep prep | **Frontend pages 4, 3, 6** |
| W9 | Integration scripts, clean-clone test | Data docs, METHODS.md | Run all experiments (overnight) | Cached demo kernel warm-up | **Frontend pages 2, 5, 7, 1** |
| W10 | Freeze, tag, rehearsals | LIMITATIONS.md | Coverage gaps, failure injection | Q&A prep (quantum) | Docs, slides, traceability page |

### 22.5 Collaboration rules

| Rule | Reason |
|---|---|
| **Interfaces before implementations.** `Model`, `QuantumBackend`, `ProcessedDataset` are agreed and merged in Week 2. | Lets five people work in parallel without blocking |
| Every member writes tests for their own modules | Nobody debugs someone else's untested code in Week 10 |
| Merge to `main` at least every 2 days | Integration pain surfaces early, not in Week 9 |
| 15-minute daily stand-up; blockers escalate same-day | A stuck quantum install can cost a week if hidden |
| M4 and M5 pair on the first kernel computation | The quantum path is the critical path — no single point of failure |
| Decisions go in `docs/DECISIONS.md` with a date | Prevents re-litigating settled choices at 2 a.m. |

---

## 23. Complete Technical Dependency Map

### 23.1 Module dependency graph (build order is topological)

```
config.py ─┬─────────────────────────────────────────────────────────┐
           │                                                         │
seeds.py ──┤                                                         │
           ▼                                                         │
    data/schema.py ──► data/loader.py ──► data/validator.py          │
                                              │                      │
                                              ▼                      │
                                     data/splitter.py                │
                                              │                      │
                                              ▼                      │
                              preprocessing/sentinels.py             │
                                              │                      │
                                              ▼                      │
                              preprocessing/pipeline.py              │
                                              │                      │
                                              ▼                      │
                              features/engineering.py                │
                                              │                      │
                                              ▼                      │
                              features/selection.py                  │
                                              │                      │
                                              ▼                      │
                              features/reduction.py                  │
                                              │                      │
                                              ▼                      │
                       preprocessing/range_normalizer.py             │
                                              │                      │
                                              ▼                      │
                              ══ ProcessedDataset ══                 │
                                    │           │                    │
                   ┌────────────────┘           └───────────────┐    │
                   ▼                                            ▼    │
        classical/models.py                          quantum/backends.py ◄─┘
                   │                                            │
                   ▼                                   quantum/feature_maps.py
        classical/training.py                                   │
                   │                             ┌──────────────┼──────────────┐
                   │                             ▼              ▼              ▼
                   │                   quantum/kernel.py  quantum/vqc.py  quantum/
                   │                             │                        transpile_report.py
                   │                             ▼
                   │                     quantum/qsvm.py
                   │                             │
                   └──────────────┬──────────────┘
                                  ▼
                       evaluation/metrics.py
                                  │
                                  ▼
                     evaluation/statistics.py
                                  │
                                  ▼
                     evaluation/benchmark.py ──► evaluation/robustness.py
                                  │                        │
                                  └────────┬───────────────┘
                                           ▼
                                  evaluation/plots.py
                                           │
                        ┌──────────────────┼──────────────────┐
                        ▼                  ▼                  ▼
          explainability/classical  explainability/quantum  tracking/registry.py
                        └──────────────────┬──────────────────┘
                                           ▼
                                  inference/artifacts.py
                                           │
                                           ▼
                                  inference/predictor.py
                                           │
                                           ▼
                                      backend/  (FastAPI)
                                           │
                                           ▼
                                     frontend/  (React)
```

**Critical path (longest chain to a demoable result):**
`config → loader → validator → splitter → preprocessing → features → reduction → normalizer → quantum/backends → feature_maps → kernel → qsvm → metrics → benchmark → registry → backend → frontend`

Everything on this path is a MUST. Anything off it (VQC, learning curves, noisy simulation, missingness robustness) is a candidate for cutting under time pressure.

### 23.2 External library dependency map

| Library | Used by | Purpose | Risk | Mitigation |
|---|---|---|---|---|
| **numpy** | everything | Arrays | Low | Pin |
| **pandas** | `data/`, `preprocessing/` | Tabular I/O | Low | Pin |
| **scikit-learn** | `preprocessing/`, `features/`, `classical/`, `evaluation/`, `explainability/` | Imputers, encoders, scalers, PCA, LR, SVC, CV, metrics, permutation importance | Low | Pin; the single most load-bearing dependency |
| **xgboost** *(or RandomForest)* | `classical/models.py` | Third baseline | **Medium** — wheel/build issues on some Windows setups | Decide in Phase 0; fall back to `RandomForestClassifier` (zero extra dependency) |
| **qiskit** | `quantum/` **only** | Circuits, feature maps, transpiler | **High** — API churn between releases | Exact pins; **import isolation to `src/quantum/`**; `QuantumBackend` abstraction |
| **qiskit-aer** | `quantum/backends.py` | Statevector / sampling / noise simulation | Medium | Exact pin |
| **qiskit-machine-learning** | `quantum/kernel.py`, `quantum/vqc.py` | `FidelityQuantumKernel`, `VQC` | **High** — the most volatile API surface | Exact pin; wrap behind our own `compute_kernel()` and `Model` classes so a breaking change touches two files |
| **shap** | `explainability/` | TreeExplainer + KernelExplainer | Medium — slow on quantum models | Background summarization, `nsamples` cap, timeout |
| **scipy** | `evaluation/statistics.py` | Distributions, tests | Low | Pin |
| **statsmodels** *(optional)* | `evaluation/statistics.py` | McNemar | Low | Or implement McNemar directly (~15 lines) and drop the dependency |
| **matplotlib** | `evaluation/plots.py`, `quantum/` | All server-side PNGs | Low | Use the `Agg` backend explicitly (headless-safe) |
| **pyyaml + pydantic** | `config.py`, `backend/schemas.py` | Config + API validation | Low | Pydantic **v2** — pin, and do not mix v1 syntax |
| **fastapi + uvicorn** | `backend/` | HTTP layer | Low | Pin |
| **joblib** | `tracking/`, `classical/` | Model persistence | Low | Pin |
| **pytest / pytest-cov / ruff / black** | `tests/`, CI | Quality | Low | Pin |
| **react / vite / react-router-dom / recharts** | `frontend/` | UI | Low | `package-lock.json` committed |

### 23.3 Phase dependency graph

```
P0 ──► P1 ──► P2 ──┐
        │          ├──► P6 ──► P7 ──► P8 ──► P9 ──► P10 ──► P11 ──► P12
        └──► P3 ──► P4 ──┤
                    P5 ──┘   (P5 optional — droppable at Gate 3)
```

| Dependency | Nature |
|---|---|
| P1 → everything | Hard. No `ProcessedDataset`, no models |
| P2 ⟂ P3 | **Independent** — run in parallel across sub-teams (this is where the 5-person split pays off) |
| P3 → P4 → P5 | Hard, sequential |
| P2 + P4 → P6 | Benchmarking needs at least one model from each branch |
| P6 → P7 | Explainability reuses trained models and the evaluation harness |
| P6/P7 → P8 → P9 | The API serves stored results; the UI reads the API |
| P8 + P9 → P10 → P11 → P12 | Integration, hardening, demo |

### 23.4 Single points of failure and their mitigations

| SPOF | Impact | Mitigation |
|---|---|---|
| Qiskit environment breaks | Quantum branch dead → project fails | Exact pins committed Week 3; every member installs the same set; a working env is archived (`pip freeze` + wheels cached) |
| Only M4 understands the quantum code | Illness/exams stall the critical path | M4 and M5 pair on kernel computation; `docs/ARCHITECTURE.md` documents `src/quantum/` in detail |
| Kernel computation too slow | Demo unusable | 6 qubits, ≤300 samples, symmetry exploitation, disk cache, pre-computed demo run |
| Leakage discovered late | All results invalid | Gate 1 in Week 2; leakage test in CI from Week 2 onward |
| Frontend not finished | No demo surface | Pre-authorized Streamlit fallback, decided at Gate 4 (Week 8) |
| Demo laptop fails | No demo | Second laptop configured identically + screen recording + screenshots |
| Live training overruns | Demo overruns | Cached run is the primary; the live run is small and optional |

---

## 24. Definition of Done

Four levels. A thing is done only when its level's checklist is fully satisfied.

### 24.1 Task-level DoD (per pull request)

- [ ] Code implements exactly the described task — no scope drift
- [ ] Type hints on public functions; docstrings explaining *why*, not *what*
- [ ] No magic numbers — every parameter comes from config
- [ ] Unit tests written and passing
- [ ] `ruff` and `black` clean
- [ ] No `print()` for diagnostics — use logging
- [ ] No Qiskit import outside `src/quantum/`
- [ ] PR < ~400 changed lines, reviewed and approved by one other member
- [ ] CI green

### 24.2 Module-level DoD

- [ ] Public interface matches the contract in §16 / §23.1
- [ ] Matching test file exists with meaningful assertions (not just "runs without error")
- [ ] Deterministic given a seed
- [ ] Errors are specific and actionable
- [ ] Artifacts (if any) serialize and deserialize losslessly
- [ ] Documented in `docs/ARCHITECTURE.md`

### 24.3 Phase-level DoD

- [ ] Every task in the phase's task list is complete
- [ ] The phase's Definition-of-Done checklist (§18) is fully ticked
- [ ] All new tests are in CI and green
- [ ] Merged to `main`
- [ ] Any decision made during the phase is recorded in `docs/DECISIONS.md`
- [ ] The next phase's owner has confirmed the handover works on their machine

### 24.4 MVP-level DoD

- [ ] **All 30 rows of §19 are PASS**
- [ ] The pipeline runs end to end from a clean clone, offline, on two machines
- [ ] Every SIH26139 requirement in §4 maps to a working, demonstrable feature
- [ ] Results are reproducible from `run_record.json` alone
- [ ] All six documents exist, are accurate, and were reviewed by someone who did not write them
- [ ] The honesty audit is complete — no unqualified superiority claim anywhere
- [ ] Three timed rehearsals completed within 10 minutes
- [ ] Code frozen and tagged `v1.0-sih-demo`
- [ ] Backup laptop and offline recording prepared

---

## 25. SIH Judging / Demo Checklist

### 25.1 Pre-demo checklist (the morning of)

| # | Item | Done |
|---|---|---|
| 1 | Both laptops charged, chargers packed | [ ] |
| 2 | Correct git tag checked out (`v1.0-sih-demo`) on both machines | [ ] |
| 3 | Python environment verified: `pytest` green on the demo laptop | [ ] |
| 4 | Backend starts: `uvicorn backend.main:app` → `/api/health` 200 | [ ] |
| 5 | Frontend starts: `npm run dev` → all 7 pages load | [ ] |
| 6 | Cached demo run loads in ≤ 5 s | [ ] |
| 7 | **Networking disabled — full flow still works** | [ ] |
| 8 | Example patients (low-risk, high-risk) pre-filled and verified | [ ] |
| 9 | Screen recording + screenshots on the desktop as fallback | [ ] |
| 10 | Slides open in a second window | [ ] |
| 11 | Speaking roles confirmed (driver, narrator, quantum Q&A, ML Q&A) | [ ] |
| 12 | Browser zoom set for projector legibility; notifications silenced | [ ] |
| 13 | `docs/SIH_TRACEABILITY.md` open in a tab for the compliance question | [ ] |

### 25.2 The 10-minute demo run sheet

| Time | Screen | What is said / shown | SIH26139 point being proved |
|---|---|---|---|
| 0:00–0:45 | Home | Problem, disease choice, dataset, and **the honesty framing**: "we built a fair measurement platform, not a quantum-advantage claim" | Scope, framing |
| 0:45–2:00 | Dataset Analysis | Load UCI; point at detected `?` sentinels in `ca`/`thal` and the class balance; show the ranked feature table and the PCA scree plot | Data ingestion, validation, feature engineering, selection, reduction |
| 2:00–3:00 | Dataset Analysis → terminal | Show the split-before-fit statement, then **run `pytest tests/test_leakage.py` live** | Leakage-safe preprocessing (the methodological differentiator) |
| 3:00–4:30 | Quantum Info | Feature map, entanglement, 6 qubits, circuit diagram, resource report, transpilation report against the **real device**; note "0 trainable quantum parameters in QSVM". Then the **hardware panel: device name, job ID, date, and the three kernel heatmaps with the error statistics** — followed by the two-line quota argument for why the benchmark stays on the simulator | Quantum ML, simulator compatibility, **near-term hardware compatibility — demonstrated, not just designed for** |
| 4:15–6:15 | Model Comparison | Protocol A vs B; leaderboard with accuracy/sensitivity/specificity and **confidence intervals**; ROC overlay; efficiency table; **read the verdict badge aloud, whatever it says** | Classical baselines, comparison, accuracy/sensitivity/specificity, computational efficiency, honesty |
| 6:15–7:15 | Explainability | Global importance for a classical and a quantum model; local explanation for one patient; PCA loading map; the classical-vs-quantum explainability panel and its limitations | Explainability |
| 7:15–8:15 | Experiment Results | CV mean ± std, 3-seed variance, noise-degradation curve, subgroup slices, **the `full` vs `screening` ablation** and why `ca`/`thal` matter clinically | Generalization, robustness, clinical rigour |
| 8:15–9:15 | Patient Prediction | Enter a high-risk patient; classical and quantum side by side; risk band; top factors; disclaimer | Prediction, end-to-end product |
| 9:15–10:00 | Home / slides | Scalability argument (config-driven, documented O(N²) bottleneck), MVP-vs-future table, limitations, one-line ask | Scalability, roadmap, documentation |

**Demo rules:**
- Never say "quantum outperforms" unless the verdict badge on screen says `significantly better`.
- If asked for a live quantum run, start the small cached-warm configuration — never a cold full kernel computation.
- If something breaks, switch to the recording and keep narrating. Recovery composure scores better than a perfect run.

### 25.3 Q&A brief — the five questions a strong panel will ask

| Question | Prepared answer |
|---|---|
| **"Did quantum beat classical?"** | "On this dataset, at 6 qubits, the difference is *[read the actual verdict]*. With 61 test records the confidence intervals overlap substantially, so we report no significant difference rather than a win. What we built is the apparatus that can answer this question fairly — and it currently says the honest answer is *X*." |
| **"How do you know there's no leakage?"** | "Three ways: the split happens before any `.fit()`; every transformer is fitted on training folds only and refitted inside each CV fold; and we have an automated test that asserts it — I can run it now." *(Then run it.)* |
| **"Isn't `ca`/`thal` basically the label?"** | "Yes, that's a real concern — both come from specialist cardiac tests usually ordered when disease is already suspected. That's why we report two feature sets: the full 13 features for literature comparability, and an 11-feature screening set without them, which is the deployable early-detection scenario. Here is the performance difference." |
| **"Did you run on real quantum hardware?"** | "Yes — on `ibm_<device>` on `<date>`, here is the job ID. But deliberately at small scale. The full kernel is ~43,900 circuits, about 3–4 hours of QPU time; the free plan gives roughly 10 minutes a month, and a credible benchmark needs repeated CV and multiple seeds on top of that. So we ran a 735-circuit validation: the same 30×30 kernel submatrix computed exactly, on a noisy simulator, and on the real device — and we report the divergence. The benchmark stays on the simulator because you can't fairly compare two models when one gets ten noisy minutes a month." |
| **"So how much did hardware noise hurt?"** | "Here is the measurement: mean and max \|ΔK\| against the exact kernel, and the diagonal deviation — `K(x,x)` should be exactly 1, and on hardware it isn't, which is the cleanest noise indicator available because it needs no ground truth. We also show whether the device flattens the kernel's off-diagonal spread. We do **not** draw an accuracy conclusion from the 10-point subsample, and the caveat is on screen next to the number." |
| **"Will this scale to a larger dataset?"** | "Architecturally yes — the dataset is a YAML config, and the reduction stage caps dimensionality at the qubit budget regardless of input width. Empirically, not yet validated: the quantum kernel is O(N²) in circuit evaluations, which we document, cap, and cache. Validating on a larger cardiovascular cohort is the first item on our post-SIH roadmap." |

### 25.4 What the judges should leave believing

1. The team built a **complete working system**, not a notebook.
2. The **methodology is rigorous** — leakage-safe, fair comparison, confidence intervals, significance tests.
3. The team is **honest about results and limitations**, which is rarer and more credible than a bold claim.
4. Every **SIH26139 requirement** maps to something they saw on screen.
5. The **architecture is genuinely extensible** to more data, more models, and real quantum hardware.

---

## Appendix — Quick Reference Card

| Thing | Value |
|---|---|
| Dataset | UCI Heart Disease, Cleveland, 303 × 14 |
| Target | `num > 0` → binary at-risk |
| Class balance | ≈ 54% / 46% (verify at load) |
| Missing | `ca` (~4), `thal` (~2) as `?`; `chol = 0` sentinel in other sites |
| Split | Stratified 80/20, seed 42; test locked |
| CV | Repeated stratified 5-fold × 3 |
| Seeds | 42, 43, 44 |
| Feature spaces | full-*d* (≈18–22) and reduced-*n* (6) |
| Encoding range | `[0, π]` |
| Qubits | 6 (range 4–8, hard cap 10) |
| Feature map | `ZZFeatureMap`, reps 2, linear entanglement |
| Quantum models | **QSVM (mandatory)**, VQC (droppable at Gate 3) |
| Classical models | LR, RBF-SVM, XGBoost/RF — each on both feature spaces |
| QSVM trainable quantum params | **0** |
| VQC trainable params | 18 (`RealAmplitudes`, n=6, reps=2) |
| Kernel evaluations | ≈ 43,900 (train+test) |
| Backend | `aer_statevector` (exact, headline), `aer_sampling` (shots), `aer_noisy`, **`ibm_<device>` (real QPU, validation only)** |
| Hardware validation | 30 train + 10 test → **735 circuits ≈ 3–4 min QPU**; 3-backend kernel comparison; run once in W9, cached |
| Why not benchmark on hardware | Full kernel ≈ 43,900 circuits ≈ 3–4 h QPU vs ~10 min/month free quota (§8.5.1) |
| Statistics | Bootstrap 1000, McNemar, DeLong/paired-bootstrap AUC |
| Protocols | A (reduced vs reduced), B (reduced-quantum vs full-classical) |
| API | FastAPI, 9 endpoints, no database |
| Frontend | React + Vite, 7 pages, server-rendered plots (Streamlit fallback) |
| Framework | **Qiskit only** |
| Runway | 10 weeks, 5 people, 5 gates |
| The one droppable item | VQC (at Gate 3, Week 6) |
| The never-droppable items | Leakage safety · fairness guard · CIs + p-values · explainability · disclaimers · docs |
