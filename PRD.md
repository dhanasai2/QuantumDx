# Product Requirements Document (PRD)

## QuantumDx — Hybrid Quantum Machine Learning Platform for Early Cardiovascular Disease Detection

| Field | Value |
|---|---|
| **Document title** | QuantumDx PRD v1.0 |
| **SIH Problem Statement ID** | SIH26139 |
| **Problem Statement Title** | Hybrid Quantum Machine Learning Platform for Early Disease Detection |
| **Event** | Smart India Hackathon 2026 |
| **Disease domain (Phase 1)** | Cardiovascular disease (CVD) |
| **Initial task** | Early cardiovascular disease risk detection (binary classification) |
| **Development / benchmark dataset** | UCI Heart Disease dataset |
| **Document status** | Baseline — approved for build |
| **Document owner** | Product / Team Lead |
| **Date** | 2026-09-03 |
| **Audience** | Developers, ML engineers, quantum engineers, evaluators, SIH jury |

### Priority Legend

Every requirement in this document carries exactly one priority label.

| Label | Meaning | Build expectation |
|---|---|---|
| **MUST HAVE** | Required for the SIH demonstration. The prototype is incomplete without it. | Built and demoed in MVP |
| **SHOULD HAVE** | Strongly desired; built if MVP is stable and time permits. | Built after MVP core is green |
| **NICE TO HAVE** | Adds polish or extra credit. Dropped without harm if time runs out. | Optional |
| **FUTURE** | Explicitly out of scope for SIH 2026. Documented to show architectural direction. | Not built now |

### Document Conventions

- Requirement IDs are stable and are referenced by the traceability matrix in Section 42.
- "Classical branch" and "quantum branch" always consume **the same processed feature matrix and the same train/test split**, unless a requirement explicitly says otherwise.
- The word "advantage" is never used as a claim in this document. It is used only as a **hypothesis to be tested**.

---

## 1. Executive Summary

### 1.1 What we are building

QuantumDx is a **hybrid quantum–classical machine learning platform for early cardiovascular disease risk detection**. It is a complete, runnable software system — not a notebook — that takes a tabular cardiovascular biomedical dataset from raw file to an explained, benchmarked risk prediction displayed on a dashboard.

The platform implements the full pipeline demanded by SIH26139: **data ingestion → validation → leakage-safe preprocessing → feature engineering → feature selection → dimensionality reduction to a quantum-compatible feature space → parallel training of classical baselines and hybrid quantum models → prediction → benchmarking → explainability → robustness/generalization evaluation → dashboard**.

### 1.2 Why hybrid, not pure quantum

Current quantum devices are noisy, small (tens of usable qubits), queue-limited, and cannot ingest a full biomedical record directly at scale. A **hybrid architecture** — classical computation for data handling, feature engineering, dimensionality reduction and optimization; quantum computation for the feature-map/kernel stage — is the only approach that is simultaneously (a) faithful to the problem statement, (b) executable today on simulators, and (c) portable to near-term hardware without a rewrite.

### 1.3 The honesty position (central to this product)

This platform **does not assume quantum models will beat classical models**. It is designed as a **fair, reproducible measurement instrument**. The deliverable value is:

1. A working hybrid QML platform that runs end to end.
2. A **methodologically fair benchmark** — identical splits, identical preprocessing, identical metrics, identical seeds, confidence intervals, and paired significance tests.
3. An honest, evidence-backed answer to the question *"on this cardiovascular data, at this feature dimensionality, does the quantum branch help, hurt, or tie?"*

A result showing **no quantum advantage**, if measured rigorously, is a **valid and successful outcome** of this product. Any accuracy, sensitivity or specificity improvement shown in a demo must be backed by a stored experiment run with its confidence interval visible on the dashboard.

### 1.4 What the evaluator will see in the demo

A user uploads a cardiovascular dataset; the platform validates and profiles it; runs the leakage-safe pipeline; trains three classical baselines and two quantum models on the same processed data; then displays a side-by-side comparison table (accuracy, sensitivity, specificity, ROC-AUC, training time), SHAP-based explanations for a selected patient, a robustness report under injected noise, and a final risk output — all reproducible from a stored run ID.

### 1.5 Scale statement (important)

Phase 1 uses the **UCI Heart Disease dataset** as the development and benchmark dataset. This is a small, tabular, well-understood dataset chosen deliberately because it makes the *pipeline* verifiable and the *benchmark* clean. **It is not representative of all high-dimensional biomedical data.** The architecture is explicitly built with a dataset-agnostic ingestion contract and a configurable dimensionality-reduction stage so that a **larger cardiovascular biomedical dataset can be introduced later** to validate scalability and generalization (Sections 24, 33, 39).

---

## 2. Problem Statement

### 2.1 Scope defined by SIH26139

SIH26139 asks for a **hybrid quantum machine learning platform for early disease detection** that integrates classical pre-processing and feature engineering with quantum-enhanced learning models (QSVM, QNN, or VQC), applied to biomedical datasets, supporting **data ingestion, hybrid model training, prediction, explainability, and performance evaluation against purely classical baselines**.

### 2.2 The stated objectives, restated as engineering targets

| # | SIH26139 Objective | Engineering target in QuantumDx |
|---|---|---|
| O1 | Design a hybrid quantum-classical ML architecture for early disease detection | A modular pipeline with an explicit classical branch and quantum branch sharing one processed-feature contract (Section 36) |
| O2 | Develop quantum-enhanced classification models that process high-dimensional biomedical data | QSVM (quantum kernel + classical SVM) and VQC/QNN, fed by a configurable reduction stage mapping *d* features → *n* qubits (Section 18) |
| O3 | Improve accuracy, sensitivity, specificity vs classical baselines | A fair benchmarking harness that **measures** these deltas with CIs and significance tests; improvement is a hypothesis, not an assumption (Section 23) |
| O4 | Scalable, interpretable, compatible with near-term quantum hardware and simulators | Simulator-first execution with a backend abstraction layer, qubit/depth budgets, transpilation reports (Sections 25, 26, 33) |
| O5 | Incorporate preprocessing, feature selection, explainability modules | Dedicated, independently testable modules with leakage-safe fit/transform discipline (Sections 19, 20, 21, 22) |
| O6 | Benchmark hybrid vs classical on accuracy, computational efficiency, generalization | Benchmark report capturing a metric table, a wall-clock/circuit-cost table, and a generalization-gap table (Sections 23, 24) |

### 2.3 The precise problem we solve

Clinicians and researchers cannot currently answer, with a single reproducible tool, the question: *"For this cardiovascular cohort, does a quantum-enhanced classifier detect early disease risk better than a well-tuned classical baseline — under identical data handling and identical evaluation?"*

Today that comparison is done ad hoc in scattered notebooks, with inconsistent preprocessing between branches, frequent data leakage, no confidence intervals, no explainability on the quantum side, and unfalsifiable "quantum advantage" claims. **QuantumDx exists to make that comparison rigorous, repeatable, and inspectable — and to run the resulting pipeline as a usable product.**

### 2.4 Constraints imposed by the problem statement

| Constraint | Consequence for the build |
|---|---|
| Must run on current simulators / near-term devices | Qubit count capped (4–10), circuit depth budgeted, shot counts configurable |
| Must be hybrid, not pure quantum | Classical stages own all heavy data manipulation |
| Must compare against purely classical baselines | Baselines must be *tuned*, not straw men |
| Must be interpretable | Explainability is a first-class module, required for both branches |
| Must be a functional platform | Deliverable is running software with API + dashboard + docs, not a research paper |
| Must handle noisy, complex biomedical data | Validation, imputation, robustness testing are mandatory modules |

---

## 3. Background and Motivation

### 3.1 Clinical motivation

Cardiovascular disease is the leading cause of death globally and in India. Its defining property for this project: **the window between "risk factors present" and "event occurs" is long, and intervention during that window is comparatively cheap and effective**, whereas intervention after the event is expensive and often too late. Early risk stratification therefore has an unusually high benefit-to-cost ratio — which is precisely why SIH26139 emphasizes *early* detection.

Two clinical properties shape our metric choices:

- **False negatives are more harmful than false positives.** A missed at-risk patient loses the intervention window; a false positive costs a follow-up test. Therefore **sensitivity (recall) is prioritized**, and specificity is reported alongside it rather than being silently traded away by accuracy-only reporting.
- **Class balance in real cohorts is uneven.** Accuracy alone is misleading. ROC-AUC, PR-AUC, and sensitivity/specificity at an explicitly chosen operating threshold are the honest metrics.

### 3.2 Technical motivation — where classical ML struggles

The problem statement names the classical limitation: high-dimensional, noisy, complex biomedical data. Concretely, in cardiovascular ML:

| Difficulty | Effect on classical models |
|---|---|
| Small labelled cohorts with many candidate features | Overfitting; unstable feature importance |
| Mixed data types (categorical, ordinal, continuous) | Requires careful encoding; naive encoding injects ordinal bias |
| Non-linear, interacting risk factors (age × cholesterol × exercise response) | Requires explicit interaction engineering, or high-capacity models that overfit small data |
| Missing and sentinel-coded values (a known issue in the non-Cleveland UCI sites) | Silent corruption if not validated |
| Site/cohort shift between hospitals | Models that do not generalize across sites |

### 3.3 Why quantum machine learning is a credible hypothesis here

A quantum feature map embeds a classical vector into a Hilbert space of dimension 2^n. The resulting quantum kernel can express similarity structures that are expensive to express classically. **For low-sample, non-linearly-separable problems — exactly the regime of a few hundred to a few thousand cardiovascular records — kernel methods are the appropriate classical tool, and quantum kernels are their natural quantum analogue.**

This is why **QSVM is our first quantum model**: it is the smallest honest step from the classical baseline (RBF-SVM) to a quantum model, changing exactly one component — the kernel — and holding everything else fixed. That makes the comparison interpretable and attributable.

### 3.4 Why we are not claiming advantage

We adopt the position stated in the literature: quantum kernels do not universally outperform classical kernels; performance depends on whether the data's structure matches the feature map's inductive bias. Known failure modes include:

| Failure mode | Symptom | Our response |
|---|---|---|
| **Kernel concentration** | Off-diagonal kernel values collapse toward a constant as qubits/depth grow; the SVM degenerates | Measure and report a kernel-concentration diagnostic per run (Section 18) |
| **Barren plateaus** | VQC gradients vanish; loss curve flat | Report gradient variance and loss curve; keep ansatz shallow (Section 18) |
| **Overfitting on small data** | Train ≫ test performance | Repeated CV + generalization-gap reporting (Section 24) |
| **Shot noise** | Non-deterministic kernel entries | Report shot count; run seed-repeats and report variance (Section 25) |

**Being able to detect and report these failure modes is a feature of this product, not an embarrassment.** Demonstrating that we measured them is stronger evidence of engineering maturity than an unverifiable accuracy claim.

### 3.5 Why hardware constraints force a hybrid design

| Reality of near-term quantum hardware | Design response |
|---|---|
| Small number of noisy usable qubits, limited connectivity | Reduce features to 4–10 dimensions before encoding |
| Short coherence times → shallow circuits only | Depth budget enforced; feature-map repetitions configurable and capped |
| Shot-based sampling → statistical noise in every value | Shot count is an experiment parameter; results reported with variance |
| Long queue times on real devices | Simulator is the default execution backend; hardware is opt-in and asynchronous |
| No efficient quantum loading of large datasets | Classical branch owns all data handling; only a small reduced vector crosses into the quantum branch |
| Cost per job on cloud QPUs | Kernel caching; batch submission; hardware runs are explicit, never implicit |

---

## 4. Product Vision

> **QuantumDx is the reference platform for honestly evaluating whether quantum machine learning helps in early cardiovascular disease detection — and for running the hybrid pipeline end to end when it does.**

### 4.1 Vision statement, expanded

We want a researcher- and clinician-facing tool in which the entire path from a raw cardiovascular CSV to a risk score is **visible, reproducible, leakage-free, explained, and benchmarked**. The quantum component is a swappable, measurable stage of that path — not a marketing label.

### 4.2 Vision pillars

| Pillar | What it means concretely |
|---|---|
| **Fairness by construction** | It is structurally impossible in this platform to compare a quantum model against a classical model on different splits or different preprocessing. The shared pipeline enforces it. |
| **Reproducibility** | Every result carries a run ID, config hash, seed, library versions, and backend descriptor. Re-running the ID reproduces the numbers. |
| **Interpretability on both branches** | A quantum prediction that cannot be explained is not clinically deliverable. Explainability applies to the quantum branch too. |
| **Hardware realism** | Simulator today, near-term device tomorrow, with no pipeline rewrite — only a backend swap. |
| **Scientific honesty** | The dashboard shows the confidence interval and the significance test next to any claimed improvement. |
| **Extensibility to bigger data** | The dataset is a plug-in, not a hardcoded assumption. |

### 4.3 What success looks like in one sentence

A judge, a clinician, and a developer can each sit in front of QuantumDx and get the answer they came for: *"it works end to end"*, *"the sensitivity is X with this confidence interval and here is why the model said so"*, and *"here is exactly where the quantum module plugs in"*.

---

## 5. Product Goals

| ID | Goal | Priority | Measurable definition of done |
|---|---|---|---|
| G1 | Deliver a runnable end-to-end hybrid QML platform for CVD risk detection | MUST HAVE | Raw CSV → risk output → dashboard, in one UI flow, with no manual notebook steps |
| G2 | Guarantee leakage-safe data handling | MUST HAVE | All fitted transformers fit on train folds only; automated leakage test passes |
| G3 | Train and evaluate at least 3 tuned classical baselines | MUST HAVE | LR, RBF-SVM, XGBoost/RF trained with hyperparameter search; metrics stored |
| G4 | Train and evaluate at least 2 hybrid quantum models | MUST HAVE | QSVM and VQC/QNN train to completion on simulator and produce predictions |
| G5 | Produce a fair, statistically qualified benchmark | MUST HAVE | Shared split, shared metrics, bootstrap CIs, McNemar/DeLong tests, stored report |
| G6 | Provide explainability for classical and quantum predictions | MUST HAVE | Global and local explanations for at least one classical and one quantum model |
| G7 | Provide robustness and generalization evidence | MUST HAVE | Repeated stratified CV, noise-injection robustness, subgroup slices |
| G8 | Provide a clear dashboard covering the full workflow | MUST HAVE | Upload, validate, configure, train, compare, explain, predict — all visible in the UI |
| G9 | Be simulator-executable and architecturally hardware-compatible | MUST HAVE | Backend abstraction with ≥1 simulator backend implemented; hardware adapter interface defined |
| G10 | Full experiment tracking and reproducibility | SHOULD HAVE | Every run reproducible from stored config + seed |
| G11 | Expose a documented REST API | SHOULD HAVE | OpenAPI spec; working train / predict / benchmark / explain endpoints |
| G12 | Do not overclaim | MUST HAVE | The UI cannot display an improvement claim without its uncertainty measure |
| G13 | Validate scalability on a larger cardiovascular dataset | FUTURE | Second dataset ingested through the same pipeline with config change only |
| G14 | Execute on real near-term quantum hardware | FUTURE | Same run config executed on a cloud QPU backend |

---

## 6. Non-Goals

These are deliberately **out of scope**. Listing them prevents scope creep and sets correct expectations for evaluators.

| ID | Non-goal | Rationale |
|---|---|---|
| NG1 | Not a clinical diagnostic device; no regulatory (CDSCO / FDA / CE) claim | Prototype; no clinical validation, no prospective trial |
| NG2 | Not a replacement for a physician's judgement | Output is a decision-support risk score with explanations |
| NG3 | Not a medical imaging platform (no CT/MRI/X-ray ingestion) in Phase 1 | The problem statement lists imaging as an *example*; we scope to tabular CVD data. Imaging is FUTURE |
| NG4 | Not a genomics pipeline in Phase 1 | Same reasoning as NG3; FUTURE |
| NG5 | Not a claim that quantum outperforms classical | Explicitly forbidden until experiments prove it |
| NG6 | Not a fault-tolerant quantum algorithm implementation (no Grover / HHL / QPE speedup claims) | Not executable on near-term hardware |
| NG7 | Not an EHR / HIS integration (no live HL7 or FHIR connection) | FUTURE |
| NG8 | Not a multi-tenant production SaaS with billing, SSO, org management | Prototype scope |
| NG9 | Not a real-time streaming inference system | Batch and single-record inference only |
| NG10 | Not a novel quantum algorithm research contribution in Phase 1 | Adaptive/optimized feature maps are FUTURE research, not the MVP novelty |
| NG11 | Not a mobile application | Web dashboard only |
| NG12 | Not a federated learning system | FUTURE |
| NG13 | No training on identifiable patient data | Only public/benchmark or de-identified data |
| NG14 | Not a time-to-event / survival modelling tool | Binary risk classification only in Phase 1 |

### 6.1 Special note on adaptive / optimized quantum feature maps

Adaptive feature-map search, kernel-target-alignment optimization, and architecture search over feature maps are **FUTURE research components (Section 39)**. They are **not** the MVP deliverable and **must not** be described as the project's proven novelty in the SIH pitch. The MVP uses **standard, published feature maps** — ZZFeatureMap, ZFeatureMap, PauliFeatureMap, and angle/amplitude encoding — with configurable entanglement patterns and repetitions.

---

## 7. Target Users

| ID | User type | Context of use | What they need from QuantumDx | Support priority |
|---|---|---|---|---|
| U1 | **QML researcher / student** | Testing whether quantum kernels help on CVD data | Fair benchmark harness, configurable feature maps, experiment tracking, exportable results | MUST HAVE |
| U2 | **Clinical data scientist / biomedical researcher** | Owns a cardiovascular cohort; evaluating new methods | Leakage-safe pipeline, familiar metrics, SHAP explanations, subgroup analysis, own-CSV upload | MUST HAVE |
| U3 | **SIH evaluator / jury member** | 10-minute evaluation window | A guided demo path showing every SIH26139 requirement working, with a traceability view | MUST HAVE |
| U4 | **Developer / contributor (our own team)** | Building and extending the platform | Modular code, clear stage contracts, tests, API, docs | MUST HAVE |
| U5 | **Clinician (cardiologist / physician)** | Reviewing risk output for a patient record | Single-patient risk score, plain-language explanation, contributing factors, confidence, disclaimer | SHOULD HAVE |
| U6 | **Hospital IT / compliance reviewer** | Assessing data handling | Privacy posture, audit log, no-PII guarantee, local execution | NICE TO HAVE |
| U7 | **Public health analyst** | Population-level screening prioritization | Batch scoring, cohort-level risk distribution | FUTURE |

---

## 8. User Personas

### Persona 1 — Ananya, QML Researcher (Primary)

| Attribute | Detail |
|---|---|
| Role | M.Tech student researching quantum kernel methods |
| Technical level | High (Python, Qiskit/PennyLane, scikit-learn) |
| Environment | Laptop plus free-tier cloud quantum access |
| Goal | Determine whether a ZZFeatureMap quantum kernel beats an RBF kernel on CVD data, and produce a defensible result |
| Frustrations | Unreproducible papers; notebooks where preprocessing differs between branches; no confidence intervals |
| Needs | Config-driven experiments, enforced identical splits, kernel-concentration diagnostics, exportable CSV/JSON results, run reproducibility |
| Success moment | She runs 20 configs overnight and the leaderboard tells her which feature-map / qubit-count combination is worth writing about — with p-values attached |

### Persona 2 — Dr. Rakesh, Clinical Data Scientist (Primary)

| Attribute | Detail |
|---|---|
| Role | Data scientist embedded in a cardiology department |
| Technical level | High ML, low quantum |
| Environment | Hospital workstation; offline execution preferred |
| Goal | Evaluate whether this method improves early-risk sensitivity on his cohort without leaking data |
| Frustrations | Tools that silently leak test data through scaling/imputation; black-box outputs he cannot defend to clinicians |
| Needs | Dataset validation report, explicit leakage guarantees, sensitivity/specificity at a chosen threshold, SHAP plots, subgroup slices, own-CSV ingestion |
| Success moment | He uploads his cohort, the validation report flags three sentinel-coded missing columns he did not know about, and he gets a benchmarked, explained model |

### Persona 3 — Prof. Meera, SIH Jury Evaluator (Primary)

| Attribute | Detail |
|---|---|
| Role | Academic evaluator, ~10 minutes per team |
| Technical level | High but broad; not a quantum specialist |
| Goal | Verify the team built what SIH26139 asked for, and that the claims are honest |
| Frustrations | Teams showing a notebook and claiming "quantum advantage" from one lucky split |
| Needs | A single guided demo, a visible mapping from the problem statement to features, live training on a simulator, honest reporting |
| Success moment | She opens the traceability page, clicks an objective, and lands on the working feature that satisfies it |

### Persona 4 — Karthik, Developer (Primary)

| Attribute | Detail |
|---|---|
| Role | Team member implementing the quantum branch |
| Technical level | High |
| Goal | Add a new quantum model without touching preprocessing code |
| Needs | Stable interfaces (`DataValidator`, `Preprocessor`, `FeatureSelector`, `Reducer`, `QuantumEncoder`, `Model`, `Evaluator`, `QuantumBackend`), documented contracts, unit tests, seeded determinism |
| Success moment | He implements a new VQC ansatz in one file, registers it, and it appears in the dashboard model list automatically |

### Persona 5 — Dr. Sunita, Cardiologist (Secondary)

| Attribute | Detail |
|---|---|
| Role | Practising cardiologist |
| Technical level | Low ML, zero quantum |
| Goal | Understand why a patient was flagged high risk |
| Needs | One screen: risk score, risk band, top contributing factors in plain language, model confidence, explicit "decision support, not diagnosis" disclaimer |
| Success moment | She sees "elevated risk driven by: exercise-induced angina, ST depression 2.3, age 61" and it matches her clinical intuition |

---

## 9. User Problems

| ID | Problem | Who feels it | Severity | How QuantumDx addresses it | Priority |
|---|---|---|---|---|---|
| P1 | Quantum vs classical comparisons are unfair (different splits/preprocessing) | U1, U3 | Critical | Single shared pipeline; both branches consume the same `ProcessedDataset` and split indices | MUST HAVE |
| P2 | Data leakage silently inflates reported scores | U1, U2 | Critical | Fit-on-train-only discipline enforced by a pipeline abstraction plus an automated leakage test | MUST HAVE |
| P3 | Biomedical data contains undeclared missing / sentinel values | U2 | High | Validation module reporting schema, ranges, sentinels and missingness before any training | MUST HAVE |
| P4 | Quantum models cannot ingest high-dimensional features | U1, U4 | Critical | Explicit, configurable dimensionality-reduction stage producing a quantum-ready vector | MUST HAVE |
| P5 | Quantum predictions are unexplainable, so clinically undeliverable | U2, U5 | High | Model-agnostic explainability (permutation importance, KernelSHAP over the decision function, surrogate model) | MUST HAVE |
| P6 | Accuracy-only reporting hides poor sensitivity | U2, U5 | High | Sensitivity, specificity, PPV, NPV and AUC always reported together; threshold selection explicit | MUST HAVE |
| P7 | Single-split results are noise, not evidence | U1, U3 | High | Repeated stratified CV, bootstrap CIs, paired significance tests | MUST HAVE |
| P8 | Quantum experiments are slow and get re-run wastefully | U1, U4 | Medium | Quantum kernel-matrix caching keyed by (data hash, feature-map config, backend, shots) | SHOULD HAVE |
| P9 | Results are not reproducible weeks later | U1, U4 | High | Run registry storing config, seed, versions and artifacts | SHOULD HAVE |
| P10 | Hard to tell whether the quantum model failed or the setup failed | U1, U4 | Medium | Diagnostics: kernel concentration, kernel eigenspectrum/rank, VQC loss curve, gradient variance | SHOULD HAVE |
| P11 | Switching from simulator to hardware requires rewriting everything | U1, U4 | Medium | Backend abstraction; per-backend transpilation; qubit/depth budget checks | MUST HAVE (interface) |
| P12 | Trained models are lost or unversioned | U4 | Medium | Model registry with artifacts and metadata | SHOULD HAVE |
| P13 | Non-technical users cannot operate CLIs or notebooks | U3, U5 | High | Dashboard covering the whole workflow | MUST HAVE |
| P14 | Cannot tell whether results generalize beyond this dataset | U2, U3 | High | Robustness suite plus an explicit "validated on UCI Heart Disease only" disclosure banner | MUST HAVE |

---

## 10. Proposed Solution

### 10.1 Solution in one paragraph

QuantumDx is a modular Python platform with a FastAPI service layer and a web dashboard. A **shared data pipeline** converts any conforming cardiovascular tabular dataset into a `ProcessedDataset` — train/test matrices, feature names, fitted transformer artifacts, split indices, and a quantum-ready reduced feature block. Two **branches** consume that identical object: a **classical branch** (Logistic Regression, RBF-SVM, XGBoost/Random Forest) and a **quantum branch** (QSVM using a quantum kernel; VQC/QNN using a variational ansatz), both executing through a **backend abstraction** that today points to a state-vector or shot-based simulator and tomorrow to a near-term QPU. A **benchmarking engine** evaluates all trained models with the same metric suite and produces confidence intervals and paired significance tests. An **explainability engine** produces global and local explanations for both branches. A **robustness engine** stresses the models with noise, missingness, subgroup slicing and repeated CV. Everything is recorded in an **experiment registry** and surfaced through a **dashboard**.

### 10.2 Key solution decisions and their justification

| Decision | Choice | Why | Rejected alternative |
|---|---|---|---|
| Comparison fairness mechanism | One `ProcessedDataset` shared by both branches | Makes unfair comparison structurally impossible | Separate scripts per branch (preprocessing diverges) |
| First quantum model | **QSVM (quantum kernel + classical SVC)** | Minimal, interpretable delta from the RBF-SVM baseline — only the kernel changes. Convex training, so no barren plateaus or optimizer variance confound the result | VQC first (adds optimizer noise; harder to attribute outcomes) |
| Second quantum model | **VQC / QNN** | Required by the problem statement; tests a trainable variational approach; exposes different failure modes | QGAN / quantum RL (out of scope) |
| Dimensionality reduction | Configurable: PCA, supervised selection (MI / ANOVA / RFE), optional autoencoder | Must map *d* features to *n* qubits; different reducers suit different regimes and must themselves be comparable | Fixed PCA only (an untested assumption) |
| Encoding | Angle encoding (default), amplitude encoding (option) | Angle encoding is shallow and hardware-friendly (n features → n qubits); amplitude is compact (2^n features → n qubits) but deeper | Amplitude-only (circuit depth explodes) |
| Feature map | Standard published maps (ZZ / Z / Pauli), configurable reps and entanglement | Reproducible, citable, no unverified novelty claim | A custom adaptive map as the MVP novelty (explicitly deferred to FUTURE) |
| Execution | Simulator-first with backend abstraction | Deterministic, fast, queue-free; hardware portability preserved | Hardware-first (unusable in a demo) |
| Explainability for quantum | Model-agnostic (permutation importance + KernelSHAP over the decision function + surrogate) | Quantum models expose no native feature attributions; model-agnostic methods work on any `predict_proba` | TreeSHAP (classical-only, inapplicable) |
| Dashboard technology | Streamlit for MVP | Fastest path to a full-workflow UI for a student team | React SPA in MVP (time cost too high; deferred) |

### 10.3 What makes this a *platform* and not a script

| Platform property | Implementation |
|---|---|
| Dataset-agnostic | Ingestion contract plus a schema config file; UCI is one config, not hardcoded logic |
| Model-agnostic | Model registry; adding a model means implementing an interface and registering it |
| Backend-agnostic | `QuantumBackend` interface with simulator and hardware adapters |
| Config-driven | Every run is a YAML/JSON config; no code edits needed to change an experiment |
| Observable | Structured logs, metrics, artifacts, run IDs, diagnostics |
| Reusable | API, CLI and UI all call the same core library |

---

## 11. Product Scope

### 11.1 In scope (Phase 1 — SIH 2026)

| Area | In scope |
|---|---|
| Disease | Cardiovascular disease risk (binary: at-risk / not-at-risk) |
| Data modality | Structured tabular clinical/biomedical features |
| Dataset | UCI Heart Disease (Cleveland primary; combined multi-site variant as a secondary robustness check) |
| Task | Binary classification with a calibrated risk probability |
| Classical models | Logistic Regression, RBF-SVM, XGBoost or Random Forest |
| Quantum models | QSVM (quantum kernel), VQC/QNN |
| Execution | Quantum simulators (state-vector and shot-based, optional noise model) |
| Deliverables | Core library + REST API + dashboard + benchmark report + explainability + documentation |

### 11.2 Out of scope (Phase 1)

Medical imaging, genomics, ECG waveform/time-series ingestion, multi-class disease typing, survival/time-to-event modelling, real QPU execution as a demo dependency, EHR integration, mobile app, multi-tenant deployment, federated learning, regulatory validation.

### 11.3 Scope boundary table

| Capability | Phase 1 | Phase 2 (post-SIH) | Rationale |
|---|---|---|---|
| Tabular CVD ingestion | Yes | Yes | Core |
| Larger CVD dataset (multi-site / national cohort) | Config-ready, not validated | Validated | Scalability proof needs time and data access |
| Imaging / genomics ingestion | No | Architectural hook only | A different pipeline entirely |
| Real quantum hardware run | Interface only | Yes | Queue time, cost, noise |
| Adaptive feature-map optimization | No | Research track | Not the MVP novelty |
| Multi-class / multi-disease | No | Yes | Requires new labels |
| Deployment hardening (auth, RBAC, TLS) | Minimal | Yes | Prototype scope |
---

## 12. Functional Requirements

Requirements are grouped by module. Every ID is referenced by Section 42 (traceability) and Section 43 (acceptance criteria).

### 12.1 Module A — Data Ingestion

| ID | Requirement | Priority | Acceptance signal |
|---|---|---|---|
| FR-A1 | Upload a tabular dataset via the dashboard (CSV) | MUST HAVE | File accepted, row/column count shown |
| FR-A2 | Load the bundled UCI Heart Disease dataset with one click ("Use sample dataset") | MUST HAVE | Dataset loads without any user file |
| FR-A3 | Accept a dataset schema config declaring feature names, types (numeric / categorical / ordinal), target column, positive class, and sentinel-missing codes | MUST HAVE | Schema file parsed; mismatches reported |
| FR-A4 | Support ingestion of a second, larger cardiovascular dataset through the same interface with config change only | SHOULD HAVE | A second config file runs end to end |
| FR-A5 | Support Parquet and Excel input | NICE TO HAVE | Files load correctly |
| FR-A6 | Ingest data via API upload endpoint | SHOULD HAVE | API returns a dataset ID |
| FR-A7 | Store the raw uploaded file immutably under a dataset ID with a content hash | SHOULD HAVE | Hash recorded in the run record |
| FR-A8 | Ingest imaging or genomic modalities | FUTURE | — |
| FR-A9 | Connect to a live EHR/FHIR source | FUTURE | — |

### 12.2 Module B — Data Validation and Inspection

| ID | Requirement | Priority | Acceptance signal |
|---|---|---|---|
| FR-B1 | Validate the dataset against the declared schema: required columns present, dtypes correct, target present and binary-derivable | MUST HAVE | Pass/fail report with per-column status |
| FR-B2 | Detect and report missing values, including sentinel codes (`?`, `-9`, `0` where clinically impossible such as cholesterol = 0) | MUST HAVE | Missingness table by column, counts and percentages |
| FR-B3 | Report class balance and warn when minority class share is below a configurable threshold | MUST HAVE | Class distribution chart and warning |
| FR-B4 | Produce a numeric profile: min, max, mean, median, std, quartiles, outlier count per numeric feature | MUST HAVE | Profile table shown in UI |
| FR-B5 | Produce a categorical profile: cardinality, level frequencies, unexpected levels | MUST HAVE | Profile table shown in UI |
| FR-B6 | Detect duplicate rows and constant (zero-variance) columns | MUST HAVE | Counts reported; user can drop them |
| FR-B7 | Flag clinically implausible values against configurable physiological ranges (e.g. resting BP, cholesterol, max heart rate) | SHOULD HAVE | Out-of-range rows listed |
| FR-B8 | Display a correlation heatmap and flag highly correlated feature pairs above a threshold | SHOULD HAVE | Heatmap plus flagged pair list |
| FR-B9 | Block progression to training when a **blocking** validation error exists (missing target, no usable features) | MUST HAVE | Train button disabled with a clear reason |
| FR-B10 | Allow the user to accept **non-blocking** warnings and proceed, with the decision recorded in the run record | MUST HAVE | Warnings acknowledged and logged |
| FR-B11 | Export the validation report as JSON/HTML | SHOULD HAVE | File downloads |

### 12.3 Module C — Splitting and Leakage Control

| ID | Requirement | Priority | Acceptance signal |
|---|---|---|---|
| FR-C1 | Perform a **stratified** train/test split **before** any fitted transformation | MUST HAVE | Split indices created first in the run log |
| FR-C2 | Make the split ratio and random seed user-configurable and record both | MUST HAVE | Values stored in run config |
| FR-C3 | Fit every transformer (imputer, scaler, encoder, selector, reducer) on training data only, then apply to test data | MUST HAVE | Automated leakage test passes |
| FR-C4 | Inside cross-validation, refit the entire transformation chain within each fold | MUST HAVE | Nested pipeline verified by test |
| FR-C5 | Persist split indices in the run record so any model can be re-evaluated on the exact same split | MUST HAVE | Indices retrievable by run ID |
| FR-C6 | Provide a held-out final validation set (three-way split) as an option | SHOULD HAVE | Configurable three-way split |
| FR-C7 | Support grouped splitting when a patient/site identifier exists, preventing the same subject appearing in train and test | SHOULD HAVE | Group-aware split available |
| FR-C8 | Provide a temporal / site-based out-of-distribution split | FUTURE | — |

### 12.4 Module D — Preprocessing

| ID | Requirement | Priority | Acceptance signal |
|---|---|---|---|
| FR-D1 | Impute missing numeric values (median default; mean and KNN as options) | MUST HAVE | No NaNs downstream |
| FR-D2 | Impute missing categorical values (most-frequent default; explicit "Missing" level as an option) | MUST HAVE | No NaNs downstream |
| FR-D3 | Convert declared sentinel codes to NaN before imputation | MUST HAVE | Sentinels no longer treated as valid values |
| FR-D4 | One-hot encode nominal categoricals; preserve ordinal encoding for ordered features | MUST HAVE | Encoded matrix shape correct |
| FR-D5 | Scale numeric features — StandardScaler default, MinMax and RobustScaler as options | MUST HAVE | Scaled statistics verified |
| FR-D6 | Provide a scaler mapping features into a bounded range suitable for angle encoding (default `[0, π]`, configurable) | MUST HAVE | Quantum-ready block within declared bounds |
| FR-D7 | Handle outliers via configurable winsorization/clipping | SHOULD HAVE | Clipped values reported |
| FR-D8 | Derive the binary target from a multi-level severity label using a configurable rule (e.g. UCI `num > 0` → positive) | MUST HAVE | Target distribution matches the rule |
| FR-D9 | Offer class-imbalance handling — class weights (default) and optional SMOTE applied **inside training folds only** | SHOULD HAVE | No resampling ever applied to test data |
| FR-D10 | Serialize all fitted preprocessing artifacts with the run so inference reuses them exactly | MUST HAVE | Inference loads and applies stored artifacts |

### 12.5 Module E — Feature Engineering

| ID | Requirement | Priority | Acceptance signal |
|---|---|---|---|
| FR-E1 | Provide clinically motivated derived features for CVD (see Section 20) behind an on/off toggle | MUST HAVE | Derived columns appear when enabled |
| FR-E2 | Support pairwise interaction terms for a user-selected subset of features | SHOULD HAVE | Interaction columns generated |
| FR-E3 | Support binning of continuous features into clinical bands (e.g. age bands, BP categories) | SHOULD HAVE | Binned columns generated |
| FR-E4 | Support monotone transforms (log, sqrt) for skewed features | NICE TO HAVE | Transformed columns generated |
| FR-E5 | Record every engineered feature with its formula in the run record | MUST HAVE | Feature provenance visible in UI |
| FR-E6 | Automated feature synthesis / search | FUTURE | — |

### 12.6 Module F — Feature Selection

| ID | Requirement | Priority | Acceptance signal |
|---|---|---|---|
| FR-F1 | Filter methods: variance threshold, ANOVA F-test, mutual information, chi-square | MUST HAVE | Ranked feature list produced |
| FR-F2 | Embedded methods: L1-logistic coefficients, tree-based importances | MUST HAVE | Ranked feature list produced |
| FR-F3 | Wrapper method: Recursive Feature Elimination with cross-validation | SHOULD HAVE | Selected subset produced |
| FR-F4 | Correlation-based redundancy pruning above a configurable threshold | SHOULD HAVE | Redundant features dropped, list shown |
| FR-F5 | User can force-include or force-exclude specific features (clinical override) | SHOULD HAVE | Overrides respected |
| FR-F6 | Fit selection on training data only, inside CV folds | MUST HAVE | Leakage test passes |
| FR-F7 | Display and export the ranked feature table with scores | MUST HAVE | Table visible and downloadable |
| FR-F8 | Stability selection across bootstrap resamples | NICE TO HAVE | Selection frequency per feature reported |

### 12.7 Module G — Dimensionality Reduction (Quantum-Ready Feature Vector)

| ID | Requirement | Priority | Acceptance signal |
|---|---|---|---|
| FR-G1 | Reduce the processed feature space to a configurable target dimension *n* matching the qubit budget | MUST HAVE | Output matrix has exactly *n* columns |
| FR-G2 | Support PCA with explained-variance reporting | MUST HAVE | Scree plot and cumulative variance shown |
| FR-G3 | Support supervised top-*k* selection as an alternative reducer | MUST HAVE | Selected feature names retained and shown |
| FR-G4 | Support Kernel PCA and LDA as additional reducers | SHOULD HAVE | Available in the reducer dropdown |
| FR-G5 | Support an autoencoder-based reducer | NICE TO HAVE | Trained encoder produces *n* dims |
| FR-G6 | Normalize the reduced vector into the encoding range required by the selected encoder | MUST HAVE | All values within declared bounds |
| FR-G7 | Allow *n* between 2 and 12, defaulting to 4–8, with a warning above the configured simulator budget | MUST HAVE | Warning shown when *n* is large |
| FR-G8 | Report information loss caused by reduction (explained variance and, optionally, classical accuracy before vs after reduction) | SHOULD HAVE | Loss summary shown in UI |
| FR-G9 | Persist the fitted reducer for inference | MUST HAVE | Reducer reloaded at prediction time |

### 12.8 Module H — Classical Model Training

| ID | Requirement | Priority | Acceptance signal |
|---|---|---|---|
| FR-H1 | Train Logistic Regression with regularization | MUST HAVE | Model trains; metrics stored |
| FR-H2 | Train RBF-kernel SVM with probability output enabled | MUST HAVE | Model trains; metrics stored |
| FR-H3 | Train XGBoost or Random Forest | MUST HAVE | Model trains; metrics stored |
| FR-H4 | Hyperparameter search (grid or randomized) with cross-validation on the training set only | MUST HAVE | Best params recorded per model |
| FR-H5 | Train classical models on **both** the full processed feature set and the reduced *n*-dimensional set, so a like-for-like comparison against quantum models exists | MUST HAVE | Two classical result groups reported |
| FR-H6 | Support probability calibration (Platt / isotonic) | SHOULD HAVE | Calibration curve available |
| FR-H7 | Train an additional linear-SVM and k-NN baseline | NICE TO HAVE | Extra rows in the leaderboard |
| FR-H8 | Persist trained classical models as artifacts | MUST HAVE | Model file written and reloadable |

### 12.9 Module I — Quantum Model Training

| ID | Requirement | Priority | Acceptance signal |
|---|---|---|---|
| FR-I1 | Train a **QSVM**: compute a quantum kernel matrix from a parameterized feature map and fit a classical SVC with `kernel='precomputed'` | MUST HAVE | Quantum kernel matrix computed; SVC trained; metrics stored |
| FR-I2 | Train a **VQC/QNN**: feature-map encoding plus a variational ansatz, optimized by a classical optimizer | MUST HAVE | Loss decreases; model predicts; metrics stored |
| FR-I3 | Feature map selectable: ZZFeatureMap, ZFeatureMap, PauliFeatureMap, plain angle encoding | MUST HAVE | Selection changes the executed circuit |
| FR-I4 | Configurable feature-map repetitions and entanglement pattern (linear, circular, full) | MUST HAVE | Config affects circuit depth, shown in UI |
| FR-I5 | Configurable ansatz (e.g. RealAmplitudes / EfficientSU2), layer count, and optimizer (COBYLA, SPSA, Adam) | MUST HAVE | Config respected; optimizer choice recorded |
| FR-I6 | Configurable qubit count derived from the reduced dimension | MUST HAVE | n_qubits = n_features for angle encoding |
| FR-I7 | Configurable shot count, including exact/state-vector mode | MUST HAVE | Shots recorded in the run record |
| FR-I8 | Cache computed kernel matrices keyed by (data hash, feature-map config, backend, shots) | SHOULD HAVE | Second identical run skips recomputation |
| FR-I9 | Report training progress: kernel-computation progress bar; VQC loss curve per iteration | SHOULD HAVE | Live progress visible in UI |
| FR-I10 | Emit quantum diagnostics: kernel concentration metric, kernel eigenspectrum/rank, VQC gradient variance | SHOULD HAVE | Diagnostics panel populated |
| FR-I11 | Render the quantum circuit diagram for the chosen configuration | SHOULD HAVE | Circuit image displayed |
| FR-I12 | Support a quantum-kernel variant of a different classical head (e.g. kernel ridge) | NICE TO HAVE | Extra model available |
| FR-I13 | Adaptive/optimized feature-map search (kernel-target alignment, architecture search) | FUTURE | — |
| FR-I14 | Persist trained quantum models (support vectors, kernel config, or trained ansatz parameters) | MUST HAVE | Model reloadable for inference |

### 12.10 Module J — Prediction / Inference

| ID | Requirement | Priority | Acceptance signal |
|---|---|---|---|
| FR-J1 | Predict for a single patient record entered through a form | MUST HAVE | Risk probability and class returned |
| FR-J2 | Predict for a batch CSV upload | MUST HAVE | Downloadable scored CSV |
| FR-J3 | Apply the exact stored preprocessing/selection/reduction artifacts at inference | MUST HAVE | Inference path uses saved artifacts, not refits |
| FR-J4 | Output a calibrated probability, a risk band (Low / Moderate / High), and the decision threshold used | MUST HAVE | All three shown |
| FR-J5 | Allow the user to select which trained model (classical or quantum) produces the prediction | MUST HAVE | Model selector present |
| FR-J6 | Show classical and quantum predictions side by side for the same record | MUST HAVE | Both shown with agreement/disagreement flag |
| FR-J7 | Reject records failing schema validation, with a specific error message | MUST HAVE | Clear validation error |
| FR-J8 | Display a "decision support, not a diagnosis" disclaimer on every prediction view | MUST HAVE | Disclaimer visible |
| FR-J9 | Expose an ensemble/consensus of classical and quantum predictions | NICE TO HAVE | Consensus output available |

### 12.11 Module K — Benchmarking and Evaluation

| ID | Requirement | Priority | Acceptance signal |
|---|---|---|---|
| FR-K1 | Compute accuracy, sensitivity (recall), specificity, precision (PPV), NPV, F1, ROC-AUC, PR-AUC, balanced accuracy, MCC | MUST HAVE | All metrics in the results table |
| FR-K2 | Produce a confusion matrix per model | MUST HAVE | Matrix rendered |
| FR-K3 | Produce ROC and precision–recall curves, overlaying all models on one axis | MUST HAVE | Overlay plots rendered |
| FR-K4 | Evaluate every model on the **identical** test split | MUST HAVE | Split ID identical across all result rows |
| FR-K5 | Report bootstrap confidence intervals for headline metrics | MUST HAVE | CI columns present |
| FR-K6 | Run a paired statistical test between a quantum model and a classical baseline (McNemar for accuracy; DeLong or bootstrap for AUC) | MUST HAVE | p-value shown with the comparison |
| FR-K7 | Report computational efficiency: training wall-clock time, inference latency, circuit count, circuit depth, total shots | MUST HAVE | Efficiency table present |
| FR-K8 | Produce a leaderboard sortable by any metric | MUST HAVE | Sortable table in UI |
| FR-K9 | Export the benchmark report (CSV, JSON, PDF/HTML) | SHOULD HAVE | Export downloads |
| FR-K10 | Threshold analysis: report metrics at the default 0.5 threshold and at a sensitivity-optimized threshold selected on the training data only | SHOULD HAVE | Both threshold rows present |
| FR-K11 | Suppress or annotate any "improvement" statement lacking an accompanying CI or p-value | MUST HAVE | UI never shows a bare win claim |
| FR-K12 | Calibration assessment (reliability curve, Brier score) | SHOULD HAVE | Calibration panel present |

### 12.12 Module L — Explainability

| ID | Requirement | Priority | Acceptance signal |
|---|---|---|---|
| FR-L1 | Global feature importance for classical models (coefficients, tree importance, SHAP) | MUST HAVE | Global importance plot |
| FR-L2 | Local per-patient explanation for classical models (SHAP values) | MUST HAVE | Per-record waterfall/force plot |
| FR-L3 | Global explanation for quantum models via permutation importance on the quantum decision function | MUST HAVE | Ranked importance plot for the quantum model |
| FR-L4 | Local explanation for quantum models via KernelSHAP or LIME over the quantum `predict_proba` | MUST HAVE | Per-record attribution for the quantum model |
| FR-L5 | Map reduced components back to original clinical features where the reducer is linear (PCA loadings) | SHOULD HAVE | Loading table linking components to features |
| FR-L6 | Surrogate model explanation: fit an interpretable model to the quantum model's predictions and report fidelity | SHOULD HAVE | Surrogate rules plus fidelity score |
| FR-L7 | Partial dependence / ICE plots for the top features | SHOULD HAVE | Plots rendered |
| FR-L8 | Plain-language explanation text for clinicians summarizing the top contributing factors | SHOULD HAVE | Readable sentence generated |
| FR-L9 | Explicitly state the limitations of each explanation method in the UI | MUST HAVE | Caveat text shown next to plots |
| FR-L10 | Counterfactual explanations ("what change would flip this prediction") | NICE TO HAVE | Counterfactual displayed |
| FR-L11 | Quantum-native interpretability (kernel-similarity attribution, circuit-level attribution) | FUTURE | — |

### 12.13 Module M — Robustness and Generalization

| ID | Requirement | Priority | Acceptance signal |
|---|---|---|---|
| FR-M1 | Repeated stratified k-fold cross-validation with mean ± std per metric | MUST HAVE | CV table present |
| FR-M2 | Multi-seed repetition and reporting of variance across seeds | MUST HAVE | Seed-variance table present |
| FR-M3 | Gaussian noise injection into test features at configurable levels; performance-vs-noise curve | MUST HAVE | Degradation curve plotted |
| FR-M4 | Simulated missing-value injection at configurable rates; performance-vs-missingness curve | SHOULD HAVE | Degradation curve plotted |
| FR-M5 | Subgroup evaluation by sex and age band, reporting per-subgroup sensitivity/specificity | MUST HAVE | Subgroup table present |
| FR-M6 | Learning curve: performance vs training-set fraction | SHOULD HAVE | Learning curve plotted |
| FR-M7 | Generalization-gap reporting (train metric minus test metric) per model | MUST HAVE | Gap column in leaderboard |
| FR-M8 | Evaluate under a simulated quantum noise model and compare against the ideal simulator | SHOULD HAVE | Noisy-vs-ideal comparison row |
| FR-M9 | External validation on a second, larger cardiovascular dataset | FUTURE | — |
| FR-M10 | Adversarial perturbation robustness | FUTURE | — |

### 12.14 Module N — Dashboard, Model Management, Tracking, API

Detailed in Sections 27, 28, 29 and 30 respectively. Summary of MUST-HAVE items: a guided multi-page dashboard covering the full workflow (FR-N1), a run registry with reproducible run IDs (FR-N2), a model registry with persisted artifacts (FR-N3), and a REST API exposing train / predict / benchmark / explain (FR-N4, SHOULD HAVE for full coverage; MUST HAVE for predict).

---

## 13. Non-Functional Requirements

| ID | Category | Requirement | Priority | Target / verification |
|---|---|---|---|---|
| NFR-1 | Performance | Full classical pipeline (validate → preprocess → train 3 baselines with hyperparameter search) on a UCI-sized dataset completes in ≤ 3 minutes on a standard laptop | MUST HAVE | Timed in demo rehearsal |
| NFR-2 | Performance | QSVM kernel computation for a ≤ 300-sample training set at 4–8 qubits completes in ≤ 10 minutes on a state-vector simulator | MUST HAVE | Timed; caching allowed |
| NFR-3 | Performance | Single-record inference (classical) returns in ≤ 1 second | MUST HAVE | Measured at API layer |
| NFR-4 | Performance | Single-record inference (quantum, cached kernel/trained ansatz) returns in ≤ 10 seconds | SHOULD HAVE | Measured at API layer |
| NFR-5 | Reproducibility | Re-running a stored run ID reproduces identical metrics under a state-vector backend, and metrics within reported variance under a shot-based backend | MUST HAVE | Automated reproducibility test |
| NFR-6 | Reproducibility | All randomness seeded and the seed persisted | MUST HAVE | Seed present in every run record |
| NFR-7 | Correctness | Zero data leakage across all fitted transformers | MUST HAVE | Automated leakage test in CI |
| NFR-8 | Modularity | Each pipeline stage is independently importable and unit-testable behind a documented interface | MUST HAVE | Interfaces documented; unit tests exist |
| NFR-9 | Extensibility | A new model can be added by implementing one interface and registering it — no changes to pipeline or UI code | MUST HAVE | Demonstrated by adding a model |
| NFR-10 | Portability | Runs on Windows, Linux and macOS with Python 3.10+ | MUST HAVE | Fresh-environment install test |
| NFR-11 | Portability | Runs fully offline once dependencies are installed (no internet dependency for the demo) | MUST HAVE | Demo executed with networking disabled |
| NFR-12 | Usability | A first-time user completes the full workflow without reading source code | MUST HAVE | Usability walkthrough with an outside tester |
| NFR-13 | Reliability | Any stage failure produces an actionable error message and never leaves a partially written run record | MUST HAVE | Failure-injection test |
| NFR-14 | Observability | Structured logs with run ID, stage name, duration and status | SHOULD HAVE | Log file inspected |
| NFR-15 | Maintainability | Type hints on public interfaces; linting and formatting enforced | SHOULD HAVE | Lint passes |
| NFR-16 | Test coverage | ≥ 70% line coverage on core library modules | SHOULD HAVE | Coverage report |
| NFR-17 | Documentation | README, architecture doc, API reference, and a reproducible demo script | MUST HAVE | Docs present and accurate |
| NFR-18 | Security/Privacy | No patient identifiers stored; uploads confined to a local workspace; no third-party data transmission | MUST HAVE | Code review; see Section 31 |
| NFR-19 | Scalability | Pipeline handles at least 50,000 rows and 100 features on the classical branch without code change | SHOULD HAVE | Synthetic-scale test |
| NFR-20 | Scalability | Quantum branch degrades gracefully via subsampling and caching rather than failing on large datasets | SHOULD HAVE | Documented strategy and guard rails |
| NFR-21 | Hardware compatibility | Circuits transpile successfully against a realistic near-term device coupling map and basis-gate set | MUST HAVE | Transpilation report generated |
| NFR-22 | Fairness of comparison | The benchmarking engine refuses to compare models trained on different splits or preprocessing configs | MUST HAVE | Guard raises an explicit error |
| NFR-23 | Accessibility | Dashboard readable at 1366×768; charts have text alternatives (data tables) | NICE TO HAVE | Manual check |
| NFR-24 | Deployment | Reproducible environment via `requirements.txt` / `environment.yml`; optional Dockerfile | SHOULD HAVE | Clean install works |

---

## 14. Detailed User Journey

### 14.1 Journey A — Researcher runs a full quantum-vs-classical benchmark (primary journey)

| Step | User action | System behaviour | Screen | Priority |
|---|---|---|---|---|
| 1 | Opens the dashboard | Shows the guided workflow with stages and the disclaimer banner | Home | MUST HAVE |
| 2 | Chooses "Use sample dataset (UCI Heart Disease)" or uploads a CSV | Loads data, assigns a dataset ID, computes a content hash | Data | MUST HAVE |
| 3 | Reviews the validation report | Shows schema check, missingness (including sentinel `?` in `ca`/`thal`), class balance, profiles, correlation heatmap, warnings | Validate | MUST HAVE |
| 4 | Acknowledges warnings and proceeds | Records the acknowledgement in the run record | Validate | MUST HAVE |
| 5 | Sets split ratio and seed | Creates the stratified split **before** any fitting; stores indices | Configure | MUST HAVE |
| 6 | Chooses preprocessing options (imputation, encoding, scaling) | Builds the transformer chain; fits on train only | Configure | MUST HAVE |
| 7 | Enables feature engineering and picks a feature-selection method | Produces the ranked feature table | Features | MUST HAVE |
| 8 | Selects the reducer and target dimension *n* (e.g. PCA → 6) | Produces the quantum-ready feature block; shows explained variance and encoding-range check | Features | MUST HAVE |
| 9 | Selects classical models and quantum models, feature map, entanglement, reps, ansatz, optimizer, shots | Validates the config against the qubit/depth budget; estimates runtime | Models | MUST HAVE |
| 10 | Clicks "Run experiment" | Trains classical models on both full and reduced features; computes the quantum kernel; trains QSVM; trains VQC; streams progress | Run | MUST HAVE |
| 11 | Reviews the leaderboard | Metrics with CIs, ROC/PR overlays, confusion matrices, efficiency table, generalization gap, p-values for quantum-vs-classical comparisons | Compare | MUST HAVE |
| 12 | Opens the diagnostics panel | Kernel concentration, kernel eigenspectrum, VQC loss curve, circuit diagram, transpilation report | Diagnostics | SHOULD HAVE |
| 13 | Opens explainability | Global importance for classical and quantum models; picks a patient and views local attributions; sees PCA loadings back to clinical features | Explain | MUST HAVE |
| 14 | Opens robustness | Repeated-CV table, noise-degradation curve, subgroup slices, seed variance | Robustness | MUST HAVE |
| 15 | Exports the report and notes the run ID | Writes CSV/JSON/HTML artifacts; run is reproducible from the ID | Compare | SHOULD HAVE |

### 14.2 Journey B — Clinician scores a single patient

| Step | User action | System behaviour | Priority |
|---|---|---|---|
| 1 | Opens the Predict page | Loads the available trained models with their headline metrics | MUST HAVE |
| 2 | Enters patient values in a clinical form (age, sex, chest-pain type, resting BP, cholesterol, max heart rate, exercise angina, ST depression, etc.) | Validates each field against physiological ranges | MUST HAVE |
| 3 | Selects a model (or "compare both branches") | Applies the stored preprocessing → selection → reduction artifacts | MUST HAVE |
| 4 | Submits | Returns calibrated probability, risk band, threshold used, and the classical/quantum agreement flag | MUST HAVE |
| 5 | Reads the explanation | Top contributing factors with direction of effect, in plain language, plus method caveats | SHOULD HAVE |
| 6 | Sees the disclaimer | "Decision support only — not a diagnosis. Validated on UCI Heart Disease data only." | MUST HAVE |

### 14.3 Journey C — Evaluator verifies SIH26139 compliance

| Step | User action | System behaviour | Priority |
|---|---|---|---|
| 1 | Opens the "SIH26139 Compliance" page | Displays the traceability matrix from Section 42 with live links | SHOULD HAVE |
| 2 | Clicks an objective | Navigates to the screen implementing it | SHOULD HAVE |
| 3 | Clicks "Run demo scenario" | Executes a pre-configured, cached end-to-end run in under 2 minutes | MUST HAVE |
| 4 | Inspects results | Sees honest reporting: CIs, p-values, and the "no advantage claimed without evidence" statement | MUST HAVE |

### 14.4 Journey D — Developer adds a new quantum model

| Step | Action | System behaviour | Priority |
|---|---|---|---|
| 1 | Implements the `Model` interface in a new file | — | MUST HAVE |
| 2 | Registers it in the model registry | The model appears in the dashboard selector and API schema | MUST HAVE |
| 3 | Runs the test suite | Contract tests validate fit/predict/predict_proba/persist behaviour | SHOULD HAVE |
| 4 | Runs a benchmark | New model appears in the leaderboard on the same split | MUST HAVE |

---

## 15. Complete Data Flow

### 15.1 End-to-end flow

```
                              ┌────────────────────────┐
                              │   RAW BIOMEDICAL DATA  │
                              │  (CSV upload / UCI     │
                              │   sample dataset)      │
                              └───────────┬────────────┘
                                          │
                              ┌───────────▼────────────┐
                              │   SCHEMA VALIDATION    │
                              │  dtypes, target, ranges│
                              │  sentinels, duplicates │
                              │  class balance, profile│
                              └───────────┬────────────┘
                                          │  (blocking errors stop here)
                              ┌───────────▼────────────┐
                              │  STRATIFIED SPLIT      │
                              │  train / test (+ seed) │
                              │  indices persisted     │
                              └───────────┬────────────┘
                                          │  ← split happens BEFORE any fitting
                              ┌───────────▼────────────┐
                              │ LEAKAGE-SAFE           │
                              │ PREPROCESSING          │
                              │ (fit on TRAIN only)    │
                              │ sentinel→NaN, impute,  │
                              │ encode, scale, clip    │
                              └───────────┬────────────┘
                                          │
                              ┌───────────▼────────────┐
                              │ FEATURE ENGINEERING    │
                              │ clinical derivations,  │
                              │ interactions, bins     │
                              └───────────┬────────────┘
                                          │
                              ┌───────────▼────────────┐
                              │ FEATURE SELECTION      │
                              │ filter / embedded /    │
                              │ wrapper + redundancy   │
                              └───────────┬────────────┘
                                          │
                        ┌─────────────────┴──────────────────┐
                        │                                    │
          ┌─────────────▼─────────────┐        ┌─────────────▼──────────────┐
          │ PROCESSED FEATURE MATRIX  │        │ DIMENSIONALITY REDUCTION   │
          │ (full d-dimensional)      │        │ PCA / top-k / KPCA → n dims│
          │                           │        │ + range normalization      │
          └─────────────┬─────────────┘        └─────────────┬──────────────┘
                        │                                    │
                        │                        ┌───────────▼────────────┐
                        │                        │ QUANTUM-READY FEATURE  │
                        │                        │ VECTOR  (n ≈ 4–8,      │
                        │                        │ values in [0, π])      │
                        │                        └───────────┬────────────┘
                        │                                    │
        ┌───────────────▼──────────────┐      ┌──────────────▼───────────────┐
        │      CLASSICAL BRANCH        │      │        QUANTUM BRANCH        │
        │  LR / RBF-SVM / XGB or RF    │      │  encoding → feature map →    │
        │  (trained on BOTH full-d and │      │  circuit → QSVM kernel  OR   │
        │   reduced-n features)        │      │  VQC/QNN ansatz + optimizer  │
        └───────────────┬──────────────┘      └──────────────┬───────────────┘
                        │                                    │
                        │  classical predictions             │  quantum predictions
                        └────────────────┬───────────────────┘
                                         │
                              ┌──────────▼───────────┐
                              │    BENCHMARKING      │
                              │ same split, same     │
                              │ metrics, CIs, tests  │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │   EXPLAINABILITY     │
                              │ global + local, both │
                              │ branches             │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │ ROBUSTNESS &         │
                              │ GENERALIZATION       │
                              │ CV, noise, subgroups │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │  FINAL RISK OUTPUT   │
                              │  probability, band,  │
                              │  threshold, caveats  │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │      DASHBOARD       │
                              └──────────────────────┘
```

### 15.2 Data artifacts produced at each stage

| Stage | Artifact | Format | Persisted |
|---|---|---|---|
| Ingestion | Raw dataset + content hash | CSV + JSON metadata | Yes |
| Validation | Validation report | JSON + HTML | Yes |
| Splitting | Train/test index arrays | NPY / JSON | Yes |
| Preprocessing | Fitted imputers, encoders, scalers | Pickle / joblib | Yes |
| Feature engineering | Engineered feature definitions | JSON | Yes |
| Feature selection | Ranked feature table, selected subset | CSV + JSON | Yes |
| Reduction | Fitted reducer + explained variance | joblib + JSON | Yes |
| Quantum-ready block | Reduced, range-normalized matrix | NPY | Yes |
| Classical training | Model artifacts + best params | joblib + JSON | Yes |
| Quantum training | Kernel matrix / trained parameters + circuit config | NPY + JSON | Yes |
| Benchmarking | Metrics table, curves, CIs, p-values | CSV + JSON + PNG | Yes |
| Explainability | SHAP values, importance tables, plots | NPY + CSV + PNG | Yes |
| Robustness | CV table, degradation curves, subgroup table | CSV + PNG | Yes |
| Run record | Full config, seeds, versions, timings, status | JSON | Yes |

### 15.3 The fairness invariant (non-negotiable)

> **Both branches must consume artifacts derived from the same `split_id` and the same `preprocessing_config_hash`.** The benchmarking engine validates this before producing a comparison and raises an explicit error otherwise (NFR-22). Two comparisons are always reported: **quantum (n dims) vs classical (n dims)** — the strict like-for-like comparison — and **quantum (n dims) vs classical (full d dims)** — the practical comparison against the best available classical model.

---

## 16. Complete ML Pipeline

### 16.1 Pipeline stages, inputs, outputs, and owners

| # | Stage | Input | Output | Fit on | Priority |
|---|---|---|---|---|---|
| 1 | Ingestion | Raw file + schema config | Typed DataFrame | — | MUST HAVE |
| 2 | Validation | Typed DataFrame | Validation report; pass/fail | — | MUST HAVE |
| 3 | Target derivation | Raw label column | Binary target | — | MUST HAVE |
| 4 | Stratified split | DataFrame + target | Train/test indices | — | MUST HAVE |
| 5 | Sentinel handling | Train/test frames | NaN-normalized frames | Train (rules from config) | MUST HAVE |
| 6 | Imputation | NaN-normalized frames | Complete frames | Train only | MUST HAVE |
| 7 | Encoding | Complete frames | Numeric matrix | Train only | MUST HAVE |
| 8 | Feature engineering | Numeric matrix | Extended matrix | Train only | MUST HAVE |
| 9 | Scaling | Extended matrix | Scaled matrix | Train only | MUST HAVE |
| 10 | Feature selection | Scaled matrix | Selected matrix + ranking | Train only | MUST HAVE |
| 11 | Dimensionality reduction | Selected matrix | n-dim matrix + variance report | Train only | MUST HAVE |
| 12 | Encoding-range normalization | n-dim matrix | Quantum-ready matrix in `[0, π]` | Train only | MUST HAVE |
| 13a | Classical training (full d) | Selected matrix | Trained classical models | Train only | MUST HAVE |
| 13b | Classical training (reduced n) | n-dim matrix | Trained classical models | Train only | MUST HAVE |
| 14 | Quantum kernel computation | Quantum-ready matrix | Kernel matrices (train/train, test/train) | Train geometry | MUST HAVE |
| 15 | QSVM training | Precomputed kernel | Trained SVC | Train only | MUST HAVE |
| 16 | VQC/QNN training | Quantum-ready matrix | Trained ansatz parameters | Train only | MUST HAVE |
| 17 | Prediction | Test / new data | Probabilities and labels | — | MUST HAVE |
| 18 | Benchmarking | All predictions | Metrics, CIs, tests, efficiency | — | MUST HAVE |
| 19 | Explainability | Models + data | Global and local attributions | — | MUST HAVE |
| 20 | Robustness | Models + perturbed data | Degradation and subgroup reports | — | MUST HAVE |
| 21 | Reporting | All artifacts | Dashboard views and exports | — | MUST HAVE |

### 16.2 Cross-validation discipline

```
For each repeat r in 1..R (seeds s_r):
  For each fold k in 1..K (stratified):
      train_idx, val_idx = fold split
      FIT   on train_idx:  sentinel rules → imputer → encoder → engineering →
                           scaler → selector → reducer → range-normalizer
      APPLY to val_idx (transform only)
      TRAIN classical models and quantum models on transformed train_idx
      EVALUATE on transformed val_idx
Aggregate: mean ± std per metric; report per-fold values
```

**Rule (MUST HAVE):** no transformer, selector or reducer is ever fitted on data that includes a validation or test fold. Any resampling (SMOTE) is applied strictly inside the training portion of a fold.

### 16.3 Two comparison protocols (both MUST HAVE)

| Protocol | Classical input | Quantum input | Question it answers |
|---|---|---|---|
| **Protocol A — strict like-for-like** | Reduced *n*-dim features | Same reduced *n*-dim features | Does the quantum kernel/model extract more from the *same* compressed representation? |
| **Protocol B — practical** | Full processed *d*-dim features | Reduced *n*-dim features | Does the hybrid pipeline as a whole compete with the best classical pipeline available? |

Reporting both prevents the two classic distortions: crippling the classical model by forcing it through reduction (which would inflate the quantum result), and comparing an unreduced classical model against a reduced quantum model without saying so.
---

## 17. Classical ML Requirements

The classical branch is not a formality. It is the **control arm of the experiment**, and a weak control invalidates every conclusion. Classical baselines must be genuinely tuned.

### 17.1 Required models

| ID | Model | Role | Key hyperparameters searched | Priority |
|---|---|---|---|---|
| CLF-1 | **Logistic Regression** | Interpretable linear baseline; the clinical standard | Penalty (L1/L2), C, solver, class_weight | MUST HAVE |
| CLF-2 | **RBF-kernel SVM** | The direct classical counterpart to QSVM — same algorithm, classical kernel | C, gamma, class_weight | MUST HAVE |
| CLF-3 | **XGBoost or Random Forest** | Strong non-linear tabular baseline; the realistic performance ceiling | n_estimators, max_depth, learning_rate, subsample, min_samples_leaf | MUST HAVE |
| CLF-4 | Linear SVM | Kernel-free reference point | C | NICE TO HAVE |
| CLF-5 | k-Nearest Neighbours | Distance-based reference | k, metric, weights | NICE TO HAVE |
| CLF-6 | Gradient-boosted ensemble stack | Upper-bound reference | — | FUTURE |

**CLF-2 is mandatory and structurally important:** QSVM differs from RBF-SVM only in the kernel. Any measured difference between them is attributable to the kernel and nothing else. This is the cleanest scientific statement the project can make.

### 17.2 Training requirements

| ID | Requirement | Priority |
|---|---|---|
| CML-1 | Hyperparameter search (grid or randomized) via stratified CV on the training set only | MUST HAVE |
| CML-2 | Class imbalance handled by `class_weight='balanced'` by default; SMOTE optional and fold-internal | MUST HAVE |
| CML-3 | Every classical model trained twice — once on the full processed feature set, once on the reduced *n*-dim set (Protocols A and B, Section 16.3) | MUST HAVE |
| CML-4 | Probability outputs required from all models (`predict_proba` or a calibrated decision function) | MUST HAVE |
| CML-5 | Probability calibration (Platt or isotonic) available as a configurable option | SHOULD HAVE |
| CML-6 | Selected hyperparameters, CV score and search space recorded in the run record | MUST HAVE |
| CML-7 | Deterministic given a seed | MUST HAVE |
| CML-8 | Trained model persisted as a reloadable artifact | MUST HAVE |
| CML-9 | Training wall-clock time and inference latency measured and stored | MUST HAVE |
| CML-10 | Classical models must implement the same `Model` interface as quantum models | MUST HAVE |

### 17.3 The `Model` interface (shared by both branches)

| Method | Contract |
|---|---|
| `fit(X, y) -> self` | Trains on the processed training matrix only |
| `predict(X) -> ndarray` | Returns binary labels |
| `predict_proba(X) -> ndarray` | Returns calibrated probabilities, shape (n, 2) |
| `save(path)` / `load(path)` | Round-trips the model without behaviour change |
| `describe() -> dict` | Returns model type, hyperparameters, and branch (`classical` / `quantum`) |
| `resource_report() -> dict` | Classical: fit time, params. Quantum: circuits executed, depth, qubits, shots |

This single interface is what makes the quantum branch pluggable and what allows benchmarking, explainability and the dashboard to treat both branches identically.

---

## 18. Quantum ML Requirements

### 18.1 Quantum models

| ID | Model | Description | Priority |
|---|---|---|---|
| QM-1 | **QSVM (Quantum Kernel SVM)** | Compute a quantum kernel matrix K(x_i, x_j) = \|⟨φ(x_i)\|φ(x_j)⟩\|² from a parameterized feature map; fit a classical SVC with `kernel='precomputed'` | MUST HAVE — **first quantum model** |
| QM-2 | **VQC / QNN** | Feature-map encoding followed by a trainable variational ansatz; parameters optimized by a classical optimizer against a classification loss | MUST HAVE — second quantum model |
| QM-3 | Quantum kernel with a different classical head (kernel ridge, kernel logistic regression) | NICE TO HAVE |
| QM-4 | Adaptive / optimized feature maps (kernel-target alignment, feature-map architecture search) | FUTURE — research track only |
| QM-5 | Quantum ensemble or quantum boosting | FUTURE |

**Rationale for QSVM first (restated for developers):** QSVM has convex training, so there is no optimizer noise, no barren-plateau risk and no initialization variance to confound the comparison. If QSVM differs from RBF-SVM, the kernel is the cause. VQC is added second because the problem statement requires a variational model and because it exercises a different part of the hybrid loop (classical optimizer driving quantum circuit evaluations).

### 18.2 Encoding requirements

| ID | Requirement | Detail | Priority |
|---|---|---|---|
| QE-1 | **Angle encoding** as the default | n features → n qubits, one rotation per feature; shallow and hardware-friendly | MUST HAVE |
| QE-2 | Features scaled into the encoder's valid range (default `[0, π]`) before encoding | Prevents rotation wrap-around ambiguity | MUST HAVE |
| QE-3 | **Amplitude encoding** as an option | 2^n features → n qubits; compact but deep state preparation; flagged in the UI as depth-expensive | SHOULD HAVE |
| QE-4 | Basis encoding for binary features | — | NICE TO HAVE |
| QE-5 | Encoding choice recorded in the run record and reflected in the rendered circuit | MUST HAVE |
| QE-6 | Data re-uploading (repeated encoding layers) | FUTURE |

### 18.3 Feature-map requirements

| ID | Requirement | Options | Priority |
|---|---|---|---|
| QF-1 | Selectable feature map | ZZFeatureMap (default), ZFeatureMap, PauliFeatureMap, plain angle encoding | MUST HAVE |
| QF-2 | Configurable repetitions (`reps`) | 1–4, default 2, with a depth warning above the budget | MUST HAVE |
| QF-3 | Configurable entanglement pattern | linear, circular, full | MUST HAVE |
| QF-4 | Circuit depth and gate counts computed and displayed before execution | — | MUST HAVE |
| QF-5 | Feature maps must be standard, published constructions in Phase 1 | No custom maps presented as proven novelty | MUST HAVE |
| QF-6 | Feature-map hyperparameters (including a global scaling factor on encoded angles) exposed for sweeps | — | SHOULD HAVE |

### 18.4 QSVM requirements

| ID | Requirement | Priority |
|---|---|---|
| QSVM-1 | Compute the symmetric train/train kernel matrix and the test/train kernel matrix | MUST HAVE |
| QSVM-2 | Exploit symmetry and the unit diagonal to halve fidelity evaluations | SHOULD HAVE |
| QSVM-3 | Fit `sklearn.svm.SVC(kernel='precomputed')`, searching C on the training folds only | MUST HAVE |
| QSVM-4 | Emit a **kernel-concentration diagnostic** — mean and standard deviation of off-diagonal kernel entries; warn when the spread collapses toward zero | SHOULD HAVE |
| QSVM-5 | Emit the kernel eigenspectrum and effective rank | SHOULD HAVE |
| QSVM-6 | Cache the kernel matrix keyed by (data hash, feature-map config, backend, shots) | SHOULD HAVE |
| QSVM-7 | Support subsampling of the training set with an explicit user warning when the sample count exceeds a configurable limit (kernel cost is O(N²) circuit evaluations) | SHOULD HAVE |
| QSVM-8 | Persist the kernel configuration and support vectors so inference reproduces training-time behaviour | MUST HAVE |
| QSVM-9 | Provide probability estimates (Platt scaling on the precomputed-kernel SVC) | MUST HAVE |

### 18.5 VQC / QNN requirements

| ID | Requirement | Priority |
|---|---|---|
| VQC-1 | Configurable ansatz (RealAmplitudes, EfficientSU2, or an equivalent hardware-efficient ansatz) with a configurable layer count | MUST HAVE |
| VQC-2 | Configurable classical optimizer (COBYLA default; SPSA for shot-based backends; Adam/gradient-based on simulators) | MUST HAVE |
| VQC-3 | Configurable maximum iterations and convergence tolerance | MUST HAVE |
| VQC-4 | Seeded parameter initialization; seed recorded | MUST HAVE |
| VQC-5 | Loss curve recorded per iteration and plotted | SHOULD HAVE |
| VQC-6 | Multiple random restarts with the best-by-validation model selected, and restart variance reported | SHOULD HAVE |
| VQC-7 | Gradient-variance diagnostic to detect barren plateaus; warn when variance falls below a threshold | SHOULD HAVE |
| VQC-8 | Measurement-to-probability mapping (parity or expectation value) documented and configurable | MUST HAVE |
| VQC-9 | Trained parameters persisted and reloadable | MUST HAVE |
| VQC-10 | Training-time budget with graceful early stop and a partial-result report | SHOULD HAVE |

### 18.6 Quantum resource budget (enforced)

| Parameter | Default | Hard limit (Phase 1) | Reason |
|---|---|---|---|
| Qubits | 6 | 12 | State-vector simulation cost grows as 2^n; near-term devices are small |
| Feature-map reps | 2 | 4 | Depth vs decoherence |
| Ansatz layers | 2 | 4 | Barren plateaus and depth |
| Shots | 1024 (exact mode available) | 8192 | Runtime |
| QSVM training samples | ≤ 300 | 500 (with warning) | O(N²) circuit evaluations |
| Circuit depth (transpiled) | — | Warn above 200 | Near-term hardware realism |

**Requirement (MUST HAVE):** the system estimates and displays the number of circuit evaluations and the estimated runtime **before** a run starts, and requires confirmation when the estimate exceeds a configurable threshold.

### 18.7 Explicit anti-overclaim requirements

| ID | Requirement | Priority |
|---|---|---|
| QH-1 | No UI text, report or export may state that a quantum model outperforms a classical model without an accompanying confidence interval and paired significance test | MUST HAVE |
| QH-2 | When the quantum result is statistically indistinguishable from the classical baseline, the system reports "no significant difference" explicitly rather than presenting the larger point estimate as a win | MUST HAVE |
| QH-3 | When the quantum result is worse, it is reported as-is; results are never filtered or hidden | MUST HAVE |
| QH-4 | Every result page states the dataset, sample size, qubit count, feature map, backend and shots that the result is conditional on | MUST HAVE |
| QH-5 | Documentation includes a "Limitations and Threats to Validity" section covering dataset size, simulator-vs-hardware gap, reduction information loss, and multiple-comparison risk across config sweeps | MUST HAVE |

---

## 19. Data Preprocessing Requirements

### 19.1 The development dataset (UCI Heart Disease) — reference schema

This is the Phase 1 configuration, expressed as a config file rather than code.

| Feature | Type | Clinical meaning | Known data issues | Preprocessing |
|---|---|---|---|---|
| `age` | numeric | Age in years | — | Scale; optional age bands |
| `sex` | binary categorical | 1 = male, 0 = female | — | Keep as binary; used for subgroup slices |
| `cp` | nominal categorical (4 levels) | Chest-pain type | — | One-hot encode |
| `trestbps` | numeric | Resting blood pressure (mm Hg) | Zeros / implausible values in some sites | Range check → sentinel → impute |
| `chol` | numeric | Serum cholesterol (mg/dL) | `0` used for missing in the Switzerland subset | Treat 0 as missing → impute |
| `fbs` | binary categorical | Fasting blood sugar > 120 mg/dL | Missing in some sites | Impute most-frequent |
| `restecg` | nominal categorical (3 levels) | Resting ECG result | — | One-hot encode |
| `thalach` | numeric | Maximum heart rate achieved | — | Scale |
| `exang` | binary categorical | Exercise-induced angina | — | Keep binary |
| `oldpeak` | numeric | ST depression induced by exercise | — | Scale; optionally clip |
| `slope` | ordinal (3 levels) | Slope of peak exercise ST segment | Missing in non-Cleveland sites | Ordinal encode; impute |
| `ca` | ordinal / numeric (0–3) | Number of major vessels coloured by fluoroscopy | `?` sentinel in Cleveland; largely missing in other sites | Sentinel → NaN → impute; flag high missingness |
| `thal` | nominal categorical | Thalassemia / perfusion defect | `?` sentinel; largely missing in other sites | Sentinel → NaN → impute; flag high missingness |
| `num` (target) | ordinal 0–4 | Severity of diagnosis | — | Binarize: `num > 0` → positive (at-risk) |

**Note (MUST HAVE in documentation):** the Cleveland subset has roughly 300 records; the combined multi-site version roughly 900 with substantially more missingness. This is the *development* dataset. The platform must not present it as a demonstration of scalability to high-dimensional biomedical data.

### 19.2 Preprocessing requirements

| ID | Requirement | Priority |
|---|---|---|
| PRE-1 | Sentinel codes declared per column in config, converted to NaN before any statistic is computed | MUST HAVE |
| PRE-2 | Clinically impossible values (e.g. cholesterol = 0, resting BP = 0) declared as sentinels in the CVD config | MUST HAVE |
| PRE-3 | Numeric imputation: median default; mean, KNN and iterative imputation as options | MUST HAVE |
| PRE-4 | Categorical imputation: most-frequent default; explicit `Missing` category as an option | MUST HAVE |
| PRE-5 | A binary missingness-indicator column may optionally be added for columns above a configurable missingness threshold (`ca`, `thal` qualify) | SHOULD HAVE |
| PRE-6 | Columns exceeding a configurable missingness limit (default 60%) are flagged and may be auto-dropped with the decision logged | SHOULD HAVE |
| PRE-7 | One-hot encoding for nominal features with unknown-category handling at inference | MUST HAVE |
| PRE-8 | Ordinal encoding preserving the declared order for ordinal features | MUST HAVE |
| PRE-9 | Scaling: StandardScaler default; MinMax and RobustScaler as options | MUST HAVE |
| PRE-10 | A dedicated quantum range-normalizer mapping the reduced vector into `[0, π]` (configurable) | MUST HAVE |
| PRE-11 | Outlier handling via configurable percentile winsorization | SHOULD HAVE |
| PRE-12 | Duplicate-row removal, with the count reported | MUST HAVE |
| PRE-13 | Zero-variance column removal | MUST HAVE |
| PRE-14 | Every transformer fitted on training data only, then applied to test/inference data | MUST HAVE |
| PRE-15 | The entire chain serialized as a single reloadable pipeline artifact | MUST HAVE |
| PRE-16 | A preprocessing summary displayed in the UI: what was imputed, encoded, scaled, dropped — and by how much | MUST HAVE |
| PRE-17 | Automated test asserting that test-set statistics never influence fitted parameters | MUST HAVE |

---

## 20. Feature Engineering Requirements

### 20.1 Clinically motivated derived features (CVD-specific)

These are toggleable and evaluated empirically — they are proposed, not assumed useful.

| ID | Derived feature | Definition | Clinical rationale | Priority |
|---|---|---|---|---|
| FE-1 | Age band | Categorized age (e.g. <40, 40–54, 55–64, ≥65) | Risk is non-linear in age | SHOULD HAVE |
| FE-2 | Heart-rate reserve proxy | `thalach` relative to an age-predicted maximum (e.g. 220 − age) | Chronotropic response is prognostic | SHOULD HAVE |
| FE-3 | BP category | Resting BP mapped to clinical hypertension stages | Guideline-aligned discretization | SHOULD HAVE |
| FE-4 | Cholesterol category | Desirable / borderline / high bands | Guideline-aligned discretization | SHOULD HAVE |
| FE-5 | Ischemia composite | Combination of `exang`, `oldpeak` and `slope` | These co-express exercise-induced ischemia | SHOULD HAVE |
| FE-6 | Age × cholesterol, age × max-HR interaction terms | Pairwise products | Captures known effect modification | SHOULD HAVE |
| FE-7 | Risk-factor count | Count of positive binary risk indicators | Simple, interpretable summary feature | NICE TO HAVE |
| FE-8 | Log/sqrt transform of skewed numerics | Monotone transform | Stabilizes variance | NICE TO HAVE |

### 20.2 Engineering process requirements

| ID | Requirement | Priority |
|---|---|---|
| FEP-1 | All engineering steps are configurable and can be turned off entirely (a no-engineering ablation must be runnable) | MUST HAVE |
| FEP-2 | Engineered features are computed identically for training, test and inference data, using parameters fitted on training data only (e.g. bin edges) | MUST HAVE |
| FEP-3 | Each engineered feature stores its formula and provenance in the run record | MUST HAVE |
| FEP-4 | The platform reports the measured impact of feature engineering by comparing runs with and without it | SHOULD HAVE |
| FEP-5 | Engineered features enter the same selection and reduction stages as raw features — no privileged path | MUST HAVE |
| FEP-6 | Automated feature synthesis or symbolic feature search | FUTURE |

---

## 21. Feature Selection Requirements

### 21.1 Why selection matters more here than in ordinary ML

The quantum branch has a hard dimensional ceiling (qubits). Selection decides **which clinical information survives to the quantum circuit**. A poor selector is indistinguishable from a poor quantum model in the final metric, so selection must be measurable, swappable and reported.

### 21.2 Requirements

| ID | Requirement | Method | Priority |
|---|---|---|---|
| FS-1 | Variance threshold filter | Removes near-constant features | MUST HAVE |
| FS-2 | Univariate statistical filters | ANOVA F-test, mutual information, chi-square | MUST HAVE |
| FS-3 | Embedded selection | L1-logistic coefficients; tree-based importances | MUST HAVE |
| FS-4 | Wrapper selection | Recursive Feature Elimination with CV | SHOULD HAVE |
| FS-5 | Redundancy pruning | Drop one of any pair correlated above a configurable threshold (default 0.9) | SHOULD HAVE |
| FS-6 | Clinical override | Force-include or force-exclude named features | SHOULD HAVE |
| FS-7 | Fold-internal fitting | Selection refitted inside every CV fold | MUST HAVE |
| FS-8 | Ranked output | Feature name, score, method, rank — displayed and exportable | MUST HAVE |
| FS-9 | Configurable *k* | Number of features retained, with a sweep option | MUST HAVE |
| FS-10 | Stability analysis | Selection frequency across bootstrap resamples | NICE TO HAVE |
| FS-11 | Comparability | The same selected feature set feeds both branches, so selection is never a confound between them | MUST HAVE |
| FS-12 | Quantum-aware selection (selecting features by their behaviour under a given feature map) | FUTURE |

---

## 22. Explainability Requirements

### 22.1 Explainability for the classical branch

| ID | Requirement | Method | Priority |
|---|---|---|---|
| XC-1 | Global feature importance | Logistic-regression coefficients with odds ratios; tree importances; SHAP summary plot | MUST HAVE |
| XC-2 | Local per-patient explanation | SHAP values with a waterfall or force plot | MUST HAVE |
| XC-3 | Partial dependence and ICE plots for top features | — | SHOULD HAVE |
| XC-4 | Direction of effect stated (increases / decreases risk) | — | MUST HAVE |
| XC-5 | Calibration and confidence shown next to the explanation | — | SHOULD HAVE |

### 22.2 Explainability for the quantum branch

Quantum models expose no native feature attributions, so all quantum explanations are **model-agnostic**, operating on the trained model's `predict_proba`.

| ID | Requirement | Method | Priority |
|---|---|---|---|
| XQ-1 | Global importance for quantum models | Permutation importance over the reduced features, measured on validation data | MUST HAVE |
| XQ-2 | Local explanation for quantum models | KernelSHAP or LIME over the quantum decision function | MUST HAVE |
| XQ-3 | Component-to-clinical-feature mapping | PCA loading matrix relating each reduced component to original clinical features | SHOULD HAVE |
| XQ-4 | Surrogate explanation | Fit a shallow decision tree / logistic model to the quantum model's predictions; report surrogate fidelity (R² or agreement rate) | SHOULD HAVE |
| XQ-5 | Kernel-similarity explanation for QSVM | Show the training patients most similar to the query point under the quantum kernel, with their labels ("this patient resembles these cases") | SHOULD HAVE |
| XQ-6 | Support-vector inspection | Number and distribution of support vectors | NICE TO HAVE |
| XQ-7 | Circuit-level / quantum-native attribution | — | FUTURE |

### 22.3 Explanation integrity requirements

| ID | Requirement | Priority |
|---|---|---|
| XI-1 | Every explanation view names the method used and its limitations (e.g. "permutation importance assumes feature independence"; "SHAP values here explain the reduced components, not raw clinical features directly") | MUST HAVE |
| XI-2 | When explanations are computed on reduced components, the UI must say so explicitly and offer the loading map | MUST HAVE |
| XI-3 | Explanations must be reproducible from the run ID and seed | SHOULD HAVE |
| XI-4 | Explanations must never be presented as causal claims | MUST HAVE |
| XI-5 | A clinician-facing plain-language summary is generated from the top-k attributions | SHOULD HAVE |
| XI-6 | Explanation computation must not refit or otherwise touch the test-set target values | MUST HAVE |

---

## 23. Benchmarking Requirements

### 23.1 Metric suite (all MUST HAVE unless noted)

| Metric | Definition | Why it is included |
|---|---|---|
| Accuracy | (TP+TN)/N | Familiar, but never reported alone |
| **Sensitivity / Recall** | TP/(TP+FN) | Primary clinical metric — missing an at-risk patient is the costliest error |
| **Specificity** | TN/(TN+FP) | Controls false alarms and unnecessary follow-up |
| Precision / PPV | TP/(TP+FP) | Clinical actionability |
| NPV | TN/(TN+FN) | Confidence in a negative screen |
| F1 | Harmonic mean of precision and recall | Balanced summary |
| Balanced accuracy | Mean of sensitivity and specificity | Imbalance-robust |
| ROC-AUC | Area under the ROC curve | Threshold-independent ranking quality |
| PR-AUC | Area under the precision–recall curve | More informative under imbalance |
| MCC | Matthews correlation coefficient | Robust single-number summary |
| Brier score | Mean squared error of probabilities | Calibration quality (SHOULD HAVE) |
| Training time | Wall clock, seconds | Computational efficiency (SIH objective O6) |
| Inference latency | ms per record | Deployability |
| Circuit evaluations | Count | Quantum cost accounting |
| Transpiled circuit depth | Integer | Hardware realism |

### 23.2 Fairness rules of the benchmark

| ID | Rule | Priority |
|---|---|---|
| BM-1 | All models are evaluated on the **identical test split** (same `split_id`) | MUST HAVE |
| BM-2 | All models consume data produced by the **same preprocessing config** (same config hash) | MUST HAVE |
| BM-3 | Hyperparameter search budgets are comparable between branches and disclosed in the report | MUST HAVE |
| BM-4 | The decision threshold policy is identical across models (default 0.5, plus an optional sensitivity-targeted threshold chosen on training data only, applied uniformly) | MUST HAVE |
| BM-5 | Both Protocol A (reduced-vs-reduced) and Protocol B (reduced-quantum vs full-classical) are reported | MUST HAVE |
| BM-6 | The benchmarking engine refuses to build a comparison table from runs with mismatched split or preprocessing hashes | MUST HAVE |
| BM-7 | Any preprocessing or resampling applied to one branch is applied to the other | MUST HAVE |

### 23.3 Statistical rigour

| ID | Requirement | Method | Priority |
|---|---|---|---|
| ST-1 | Confidence intervals on headline metrics | Bootstrap resampling of the test set (default 1000 resamples), percentile CI | MUST HAVE |
| ST-2 | Paired significance test on accuracy between two models on the same test set | McNemar's test | MUST HAVE |
| ST-3 | Paired significance test on AUC | DeLong test, or paired bootstrap of the AUC difference | MUST HAVE |
| ST-4 | Cross-validation variance | Mean ± std over repeated stratified k-fold | MUST HAVE |
| ST-5 | Effect size reported alongside p-values | Absolute metric difference with its CI | SHOULD HAVE |
| ST-6 | Multiple-comparison disclosure | When many configurations are swept, the report states the number of comparisons and warns about selection bias | SHOULD HAVE |
| ST-7 | Seed-variance reporting | Metrics across ≥3 seeds | MUST HAVE |

### 23.4 The benchmark report

| ID | Requirement | Priority |
|---|---|---|
| BR-1 | A leaderboard table: model, branch, feature space (full-d / reduced-n), accuracy, sensitivity, specificity, ROC-AUC (each with CI), training time | MUST HAVE |
| BR-2 | Confusion matrices for all models | MUST HAVE |
| BR-3 | Overlaid ROC and PR curves | MUST HAVE |
| BR-4 | A dedicated quantum-vs-classical comparison panel showing the metric delta, its CI, and the p-value, with an explicit verdict string: "significantly better" / "no significant difference" / "significantly worse" | MUST HAVE |
| BR-5 | A computational-efficiency table (time, circuits, depth, shots) | MUST HAVE |
| BR-6 | A generalization-gap table (train minus test per metric) | MUST HAVE |
| BR-7 | Export to CSV, JSON and HTML/PDF | SHOULD HAVE |
| BR-8 | The report header states dataset, sample size, split, seed, backend, shots, qubits and feature map | MUST HAVE |

---

## 24. Robustness and Generalization Requirements

| ID | Requirement | Method | Priority |
|---|---|---|---|
| RB-1 | Repeated stratified k-fold CV (default 5-fold × 3 repeats) | Full pipeline refitted per fold | MUST HAVE |
| RB-2 | Multi-seed evaluation (≥3 seeds) with variance reported | Seed sweep | MUST HAVE |
| RB-3 | Feature-noise robustness | Add Gaussian noise at σ ∈ {0.01, 0.05, 0.1, 0.2} of feature std to test data; plot metric vs σ for every model | MUST HAVE |
| RB-4 | Missingness robustness | Randomly mask 5–30% of test feature values; impute using training-fitted imputers; plot degradation | SHOULD HAVE |
| RB-5 | Subgroup evaluation | Report sensitivity/specificity by sex and by age band; flag disparities above a configurable gap | MUST HAVE |
| RB-6 | Learning curve | Metric vs training-set fraction (20%…100%) | SHOULD HAVE |
| RB-7 | Generalization gap | Train metric minus test metric per model; flag models above a configurable overfitting threshold | MUST HAVE |
| RB-8 | Class-balance sensitivity | Re-evaluate under artificially rebalanced test sets | NICE TO HAVE |
| RB-9 | Quantum-noise robustness | Re-run quantum models under a simulated device noise model (depolarizing/readout error) and compare against the ideal simulator | SHOULD HAVE |
| RB-10 | Shot-noise sensitivity | Sweep shots ∈ {256, 1024, 4096}; report metric variance | SHOULD HAVE |
| RB-11 | External dataset validation on a larger cardiovascular cohort | Same pipeline, new config | FUTURE |
| RB-12 | Cross-site / out-of-distribution validation (train Cleveland, test other UCI sites) | Site-based split | SHOULD HAVE |
| RB-13 | Adversarial perturbation robustness | — | FUTURE |

### 24.1 Generalization disclosure requirement

**RB-14 (MUST HAVE):** every results view and export carries a standing disclosure: *"Results are conditional on the UCI Heart Disease dataset (n ≈ 300–900), a reduced feature space of n dimensions, and simulator execution. Generalization to other cardiovascular cohorts and to real quantum hardware is unvalidated."*

---

## 25. Quantum Simulator Requirements

| ID | Requirement | Priority |
|---|---|---|
| SIM-1 | Support a **state-vector (exact) simulator** as the default backend for deterministic, reproducible results | MUST HAVE |
| SIM-2 | Support a **shot-based sampling simulator** with a configurable shot count | MUST HAVE |
| SIM-3 | Support an optional **noisy simulator** using a device noise model (depolarizing, thermal relaxation, readout error) | SHOULD HAVE |
| SIM-4 | Seeded simulation for reproducibility (`seed_simulator`, `seed_transpiler`) | MUST HAVE |
| SIM-5 | Batched circuit execution to amortize overhead when computing kernel matrices | SHOULD HAVE |
| SIM-6 | Parallel execution across CPU cores, configurable | SHOULD HAVE |
| SIM-7 | Report simulator resource usage: circuits executed, total shots, wall-clock time, peak memory estimate | MUST HAVE |
| SIM-8 | Guard rail preventing a simulation whose estimated memory or runtime exceeds configured limits, with a clear message and suggested remedies (fewer qubits, fewer samples, exact mode) | MUST HAVE |
| SIM-9 | GPU-accelerated simulation | NICE TO HAVE |
| SIM-10 | Tensor-network or matrix-product-state simulation for larger qubit counts | FUTURE |
| SIM-11 | Simulator selection exposed in the UI and recorded in the run record | MUST HAVE |
| SIM-12 | Deterministic behaviour verified by an automated test: two identical state-vector runs produce identical metrics | MUST HAVE |

---

## 26. Near-Term Quantum Hardware Compatibility

**Updated decision (supersedes the original Phase 1 position):** the team has IBM Quantum cloud access, so Phase 1 **does execute on real hardware — at a deliberately restricted scale.** A scoped validation run (≈735 circuits, a 30×30 kernel submatrix compared across exact simulator, noisy simulator and a real QPU) is executed once, cached, and reported.

Phase 1 does **not** run the headline benchmark on hardware. That is a quota decision, not a capability gap: the full QSVM kernel is ≈43,900 circuits ≈3–4 hours of QPU time, against a free-tier allowance on the order of ten minutes per month — and a credible benchmark multiplies that by repeated CV, multiple seeds and several configurations. **You cannot fairly compare two models when one of them is allotted ten noisy minutes a month.** The benchmark therefore stays on the simulator, and hardware answers a different, cheaper question: *how far does the real device diverge from the simulator we benchmarked on?*

The transpilation-based compatibility argument below is retained in full. It is the fallback if hardware access lapses during demo week, and it is what generalizes to devices other than IBM's. See `MVP_SPEC.md` §8.5 for the complete experiment design, quota arithmetic, and the risks that real hardware introduces.

### 26.1 Compatibility requirements

| ID | Requirement | Priority |
|---|---|---|
| HW-1 | A `QuantumBackend` abstraction with a uniform interface (`run(circuits, shots) -> results`), implemented by simulator adapters and extensible to hardware adapters | MUST HAVE |
| HW-2 | Backend selection is configuration, not code; switching backends changes no pipeline logic | MUST HAVE |
| HW-3 | Circuits built from a hardware-realistic basis-gate set and transpiled to a target coupling map | MUST HAVE |
| HW-4 | A **transpilation report** for a chosen target device profile: qubit count required, transpiled depth, two-qubit gate count, SWAP overhead | MUST HAVE |
| HW-5 | Qubit budget enforcement with a warning when the configuration exceeds a realistic near-term device size | MUST HAVE |
| HW-6 | Depth budget enforcement with a warning when transpiled depth exceeds a configurable coherence-realistic limit | MUST HAVE |
| HW-7 | Shot-based execution path exercised on the simulator, so hardware execution requires no new code path | MUST HAVE |
| HW-8 | Hardware-efficient ansatz options (nearest-neighbour entanglement) available | MUST HAVE |
| HW-9 | **Working IBM Quantum Runtime adapter** implementing `QuantumBackend`, with credentials loaded from a git-ignored `.env` and never committed | **MUST HAVE** (upgraded from stub) |
| HW-10 | Asynchronous job submission, polling with a bounded timeout, and result retrieval for queued devices | **MUST HAVE** (upgraded from documented-only) |
| HW-11 | **Scoped hardware validation run**: an identical kernel submatrix computed on exact simulator, noisy simulator and a real QPU, with element-wise divergence, diagonal-fidelity deviation, and kernel-concentration comparison reported | **MUST HAVE** |
| HW-12 | **Hardware provenance recorded**: device name, calibration timestamp, job IDs, shots, queue seconds, QPU seconds — stored in the run record and displayed | **MUST HAVE** |
| HW-13 | **Quota pre-flight**: estimated QPU seconds shown and confirmed before any hardware submission | **MUST HAVE** |
| HW-14 | **Hardware is strictly additive**: with credentials absent, the entire platform runs on simulators with no degradation beyond the hardware panel reporting "not available" | **MUST HAVE** |
| HW-15 | **Hardware results cached to disk** and served offline; the demo never queues a live QPU job | **MUST HAVE** |
| HW-16 | Reproducibility caveat stated: hardware runs are auditable (job IDs, calibration timestamp) but **not** bit-reproducible, because device calibration drifts | **MUST HAVE** |
| HW-17 | Error mitigation (readout-error mitigation, zero-noise extrapolation) | FUTURE |
| HW-18 | Full benchmark, repeated CV, or configuration sweeps executed on a cloud QPU | FUTURE (requires a paid plan) |
| HW-19 | Multi-device comparison; cost-optimized batch scheduling | FUTURE |
| HW-20 | Dynamic circuits, mid-circuit measurement, error correction | FUTURE |

### 26.2 Hardware-readiness checklist shown in the UI

| Check | Pass condition |
|---|---|
| Qubits required | ≤ target device qubit count |
| Transpiled depth | ≤ configured depth budget |
| Two-qubit gate count | Reported; flagged if unusually high |
| Basis gates | All gates in the target basis after transpilation |
| Connectivity | Circuit maps to the target coupling map without excessive SWAP overhead |
| Shot-based path | Model runs correctly in sampling mode (not just exact mode) |
| Estimated job count | Reported, with a cost/queue caveat |
---

## 27. Dashboard Requirements

### 27.1 Page map

| Page | Purpose | Key elements | Priority |
|---|---|---|---|
| **1. Home / Overview** | Orientation and demo entry point | Workflow diagram, "Run demo scenario" button, standing disclaimer, run history | MUST HAVE |
| **2. Data** | Ingest a dataset | Upload widget, "Use UCI sample" button, schema config selector, data preview table | MUST HAVE |
| **3. Validate** | Inspect data quality | Schema check results, missingness table, class balance chart, numeric/categorical profiles, correlation heatmap, warning acknowledgement | MUST HAVE |
| **4. Configure** | Set split and preprocessing | Split ratio, seed, imputation/encoding/scaling choices, imbalance strategy | MUST HAVE |
| **5. Features** | Engineering, selection, reduction | Toggle engineered features, choose selector and *k*, choose reducer and *n*, ranked feature table, PCA scree plot, encoding-range check | MUST HAVE |
| **6. Models** | Configure the experiment | Classical model checkboxes, quantum model checkboxes, feature map, reps, entanglement, ansatz, optimizer, backend, shots, resource estimate | MUST HAVE |
| **7. Run** | Execute and monitor | Stage-by-stage progress, kernel-computation progress, VQC loss curve, live log, cancel button | MUST HAVE |
| **8. Compare** | Benchmark results | Leaderboard with CIs, ROC/PR overlays, confusion matrices, efficiency table, generalization gap, quantum-vs-classical verdict panel, export buttons | MUST HAVE |
| **9. Explain** | Interpretability | Global importance (both branches), patient selector, local attributions, PCA loading map, method caveats, plain-language summary | MUST HAVE |
| **10. Robustness** | Stress-test evidence | Repeated-CV table, noise-degradation curves, subgroup table, seed variance, learning curve | MUST HAVE |
| **11. Predict** | Score a patient | Clinical input form, model selector, classical/quantum side-by-side output, risk band, explanation, disclaimer, batch CSV scoring | MUST HAVE |
| **12. Diagnostics** | Quantum health checks | Circuit diagram, transpilation report, kernel concentration, kernel eigenspectrum, gradient variance, hardware-readiness checklist | SHOULD HAVE |
| **13. Runs / Registry** | History and reproducibility | Run list with IDs, configs, metrics; re-run and compare-runs actions | SHOULD HAVE |
| **14. SIH26139 Compliance** | Evaluator view | Live traceability matrix linking each objective to the implementing screen | SHOULD HAVE |
| **15. Docs / About** | Documentation | Method descriptions, limitations, references, team info | SHOULD HAVE |

### 27.2 Dashboard behaviour requirements

| ID | Requirement | Priority |
|---|---|---|
| DB-1 | The workflow is guided: later stages stay disabled until prerequisites are satisfied, with the reason displayed | MUST HAVE |
| DB-2 | Long-running operations show progress and never block the UI without feedback | MUST HAVE |
| DB-3 | Every chart has an accompanying data table or CSV download | SHOULD HAVE |
| DB-4 | The standing disclaimer ("decision support only; validated on UCI Heart Disease only") appears on Home, Compare and Predict | MUST HAVE |
| DB-5 | Any quantum-vs-classical difference is displayed with its CI and p-value; the UI is incapable of rendering a bare win claim | MUST HAVE |
| DB-6 | The current run configuration is visible at all times (dataset, split, seed, backend, qubits, feature map) | MUST HAVE |
| DB-7 | Errors are shown as actionable messages, never raw stack traces (full traces available behind a "details" expander) | MUST HAVE |
| DB-8 | A cached demo run loads in under 5 seconds so the demo never depends on live training | MUST HAVE |
| DB-9 | Session state persists across page navigation within a session | MUST HAVE |
| DB-10 | Results exportable from every results page | SHOULD HAVE |
| DB-11 | Dark/light theme, responsive layout | NICE TO HAVE |
| DB-12 | Multi-user sessions with authentication | FUTURE |

---

## 28. Model Management

| ID | Requirement | Priority |
|---|---|---|
| MM-1 | A **model registry** mapping model IDs to implementations, so models are discovered rather than hardcoded in the UI | MUST HAVE |
| MM-2 | Every trained model persisted with its artifact plus metadata: model type, branch, hyperparameters, feature-space descriptor, split ID, preprocessing hash, metrics, timestamp | MUST HAVE |
| MM-3 | Model versioning: `{model_type}_{run_id}_{version}` naming, immutable once written | MUST HAVE |
| MM-4 | A trained model can be reloaded and used for inference without retraining | MUST HAVE |
| MM-5 | Models bind to their preprocessing artifacts; loading a model loads the exact transformation chain it was trained with | MUST HAVE |
| MM-6 | Registry listing in the UI with metrics, sortable and filterable by branch | SHOULD HAVE |
| MM-7 | Mark a model as "active" for the Predict page | SHOULD HAVE |
| MM-8 | Delete or archive a model | SHOULD HAVE |
| MM-9 | Model card auto-generated per model: intended use, data, metrics, limitations, subgroup performance | SHOULD HAVE |
| MM-10 | Quantum models persist their circuit configuration (feature map, reps, entanglement, qubits, backend, shots) alongside learned parameters | MUST HAVE |
| MM-11 | Loading a model with an incompatible schema or feature-space descriptor fails loudly with a clear message | MUST HAVE |
| MM-12 | Model promotion workflow (staging → production) with approvals | FUTURE |

---

## 29. Experiment Tracking

| ID | Requirement | Priority |
|---|---|---|
| ET-1 | Every execution creates a **run record** with a unique run ID | MUST HAVE |
| ET-2 | The run record stores: dataset ID and content hash, schema config, split config and indices, preprocessing config and hash, feature-engineering config, selection config and selected features, reduction config, model configs, backend and shots, seeds, library versions, environment info | MUST HAVE |
| ET-3 | The run record stores all resulting metrics, CIs, p-values, timings and artifact paths | MUST HAVE |
| ET-4 | A run is reproducible from its record alone | MUST HAVE |
| ET-5 | Run status tracked (`running` / `completed` / `failed`) with failure reason | MUST HAVE |
| ET-6 | Runs listed and comparable side by side in the UI | SHOULD HAVE |
| ET-7 | Parameter sweeps supported (grid over qubits, feature maps, reps, reducers), producing one run per configuration | SHOULD HAVE |
| ET-8 | Sweep results aggregated into a comparison view, with the multiple-comparison warning attached | SHOULD HAVE |
| ET-9 | Storage backend: local JSON/SQLite by default; MLflow integration optional | SHOULD HAVE |
| ET-10 | Artifacts organized under `runs/{run_id}/` with a documented layout | MUST HAVE |
| ET-11 | Runs exportable as a single archive for sharing | NICE TO HAVE |
| ET-12 | Git commit hash of the codebase captured per run | SHOULD HAVE |

### 29.1 Run record schema (illustrative)

```json
{
  "run_id": "run_20260903_141200_a3f9",
  "created_at": "2026-09-03T14:12:00Z",
  "code_version": "git:8c1d4ef",
  "dataset": {"id": "uci_heart_cleveland", "sha256": "…", "rows": 303, "cols": 14},
  "schema_config": "configs/schema_uci_heart.yaml",
  "split": {"type": "stratified", "test_size": 0.2, "seed": 42, "split_id": "split_7b21"},
  "preprocessing": {"config_hash": "pp_5d0c", "impute_num": "median", "impute_cat": "most_frequent",
                    "scaler": "standard", "sentinels": {"chol": [0], "ca": ["?"], "thal": ["?"]}},
  "feature_engineering": {"enabled": true, "derived": ["age_band", "hr_reserve", "ischemia_score"]},
  "feature_selection": {"method": "mutual_info", "k": 10, "selected": ["cp", "thalach", "oldpeak", "…"]},
  "reduction": {"method": "pca", "n_components": 6, "explained_variance": 0.87},
  "models": [
    {"id": "logreg", "branch": "classical", "space": "full_d", "params": {"C": 1.0, "penalty": "l2"}},
    {"id": "rbf_svm", "branch": "classical", "space": "reduced_n", "params": {"C": 10, "gamma": "scale"}},
    {"id": "qsvm", "branch": "quantum", "space": "reduced_n",
     "quantum": {"feature_map": "ZZFeatureMap", "reps": 2, "entanglement": "linear",
                 "qubits": 6, "backend": "statevector", "shots": null}}
  ],
  "metrics": {"qsvm": {"accuracy": 0.84, "accuracy_ci": [0.76, 0.91], "sensitivity": 0.86,
                       "specificity": 0.82, "roc_auc": 0.89, "roc_auc_ci": [0.82, 0.95],
                       "train_seconds": 412, "circuits_executed": 29040}},
  "comparisons": [{"a": "qsvm", "b": "rbf_svm", "metric": "accuracy",
                   "delta": 0.013, "delta_ci": [-0.05, 0.08],
                   "test": "mcnemar", "p_value": 0.62, "verdict": "no significant difference"}],
  "status": "completed",
  "artifacts_dir": "runs/run_20260903_141200_a3f9/"
}
```

---

## 30. API Requirements

The API exists so that the dashboard, the CLI and external consumers all use one code path.

### 30.1 Endpoints

| Method | Endpoint | Purpose | Priority |
|---|---|---|---|
| POST | `/datasets` | Upload a dataset; returns `dataset_id` and hash | SHOULD HAVE |
| GET | `/datasets/{id}/validate` | Return the validation report | SHOULD HAVE |
| GET | `/datasets/{id}/profile` | Return summary statistics | SHOULD HAVE |
| POST | `/runs` | Start an experiment from a run config; returns `run_id` | SHOULD HAVE |
| GET | `/runs/{run_id}` | Return status, config and metrics | SHOULD HAVE |
| GET | `/runs/{run_id}/benchmark` | Return the benchmark report | SHOULD HAVE |
| GET | `/runs/{run_id}/explain?model_id=&record_index=` | Return global or local explanations | SHOULD HAVE |
| GET | `/runs/{run_id}/robustness` | Return robustness results | SHOULD HAVE |
| GET | `/models` | List registered and trained models | SHOULD HAVE |
| POST | `/predict` | Score a single record with a chosen model | MUST HAVE |
| POST | `/predict/batch` | Score a CSV batch | SHOULD HAVE |
| POST | `/compare` | Compare two model IDs on the same split, returning delta, CI and p-value | SHOULD HAVE |
| GET | `/health` | Liveness and version info | SHOULD HAVE |
| GET | `/backends` | List available quantum backends and their limits | NICE TO HAVE |

### 30.2 API requirements

| ID | Requirement | Priority |
|---|---|---|
| API-1 | JSON request/response with typed schemas (Pydantic) | SHOULD HAVE |
| API-2 | Auto-generated OpenAPI/Swagger documentation | SHOULD HAVE |
| API-3 | Input validation with field-level error messages | MUST HAVE |
| API-4 | Long-running training executed asynchronously with a job ID and status polling | SHOULD HAVE |
| API-5 | Consistent error envelope: `{error_code, message, details, run_id}` | SHOULD HAVE |
| API-6 | Prediction responses include probability, label, risk band, threshold, model ID, run ID and a disclaimer field | MUST HAVE |
| API-7 | Request/response logging without persisting raw patient values by default | SHOULD HAVE |
| API-8 | API key authentication | NICE TO HAVE |
| API-9 | Rate limiting and quotas | FUTURE |
| API-10 | Versioned API path (`/v1/...`) | SHOULD HAVE |

---

## 31. Security and Privacy Considerations

This prototype is not a regulated medical device, but it handles health-shaped data and must behave responsibly.

| ID | Requirement | Priority |
|---|---|---|
| SEC-1 | No personally identifying fields are required, requested or stored; the schema config must not declare identifier columns as features | MUST HAVE |
| SEC-2 | Only public benchmark data or properly de-identified data is used for training in Phase 1 | MUST HAVE |
| SEC-3 | All processing is local by default; no dataset is transmitted to any third-party service | MUST HAVE |
| SEC-4 | If a cloud quantum backend is ever enabled, only the reduced, anonymized numerical feature vector is transmitted — never raw records or identifiers — and the UI must state this before enabling it | MUST HAVE |
| SEC-5 | Uploaded files are confined to a workspace directory; path traversal and arbitrary file writes are prevented | MUST HAVE |
| SEC-6 | Uploads are validated for type and size before parsing; no execution of uploaded content | MUST HAVE |
| SEC-7 | Prediction inputs are not persisted by default; persistence is opt-in and disclosed | SHOULD HAVE |
| SEC-8 | An audit log records who ran what and when (user-agnostic in single-user prototype mode) | SHOULD HAVE |
| SEC-9 | Secrets (e.g. quantum cloud API tokens) are read from environment variables or a local secrets file, never committed to the repository | MUST HAVE |
| SEC-10 | Dependencies pinned; a vulnerability scan is run before submission | SHOULD HAVE |
| SEC-11 | A clear medical disclaimer is shown wherever a risk output appears | MUST HAVE |
| SEC-12 | Bias and fairness reporting via subgroup metrics (sex, age band), with disparities surfaced rather than hidden | MUST HAVE |
| SEC-13 | Authentication, RBAC, TLS termination, encryption at rest | FUTURE |
| SEC-14 | HIPAA / DISHA / GDPR compliance programme | FUTURE |
| SEC-15 | Differential privacy or federated training | FUTURE |

---

## 32. Error Handling

### 32.1 Error taxonomy

| Class | Examples | System behaviour | Priority |
|---|---|---|---|
| **Input errors** | Wrong file type, missing target column, unparseable CSV, out-of-range field in the prediction form | Reject with a specific, field-level message; no partial state written | MUST HAVE |
| **Validation errors (blocking)** | No usable features, single-class target, all rows missing | Block progression; explain exactly what must be fixed | MUST HAVE |
| **Validation warnings (non-blocking)** | High missingness, class imbalance, implausible values, high correlation | Warn, allow the user to proceed, record the acknowledgement | MUST HAVE |
| **Configuration errors** | Qubits exceed budget, reduction dimension larger than the available feature count, incompatible encoder and feature map | Reject before execution with a suggested fix | MUST HAVE |
| **Resource errors** | Estimated simulation memory or runtime over limit | Refuse to start; suggest fewer qubits, fewer samples, exact mode, or subsampling | MUST HAVE |
| **Training failures** | Optimizer divergence, singular kernel, solver non-convergence | Fail that model only; other models continue; the run is marked partially completed | MUST HAVE |
| **Quantum backend errors** | Simulator crash, backend unavailable, job timeout | Retry with backoff (bounded), then fail gracefully with the reason | SHOULD HAVE |
| **Persistence errors** | Disk full, permission denied | Abort atomically; never leave a half-written run record | MUST HAVE |
| **Unexpected exceptions** | Any unhandled error | Catch at the stage boundary, log with the run ID, show a friendly message with a details expander | MUST HAVE |

### 32.2 Error-handling principles

| ID | Principle | Priority |
|---|---|---|
| EH-1 | Fail fast on configuration; fail soft during training (one model's failure must not destroy the run) | MUST HAVE |
| EH-2 | Every user-visible error names the stage, the cause, and at least one concrete remedy | MUST HAVE |
| EH-3 | Run records are written atomically; a failed run is marked `failed` with the stage and traceback stored | MUST HAVE |
| EH-4 | No silent fallbacks — if the system substitutes a default (e.g. reducing qubits), it says so prominently | MUST HAVE |
| EH-5 | Long operations are cancellable, and cancellation leaves consistent state | SHOULD HAVE |
| EH-6 | Warnings never disappear silently; they are collected into the run record and shown in the report | MUST HAVE |
| EH-7 | Quantum-specific failures are diagnosed, not generic: e.g. "kernel matrix is near-constant (concentration detected) — try fewer qubits or fewer feature-map repetitions" | SHOULD HAVE |

---

## 33. Scalability

### 33.1 Where the platform scales, and where it does not

| Dimension | Classical branch | Quantum branch (simulator) | Mitigation |
|---|---|---|---|
| Rows (N) | Scales to tens of thousands easily | QSVM kernel cost is **O(N²)** circuit evaluations — the binding constraint | Subsampling, caching, batching, Nyström-style approximation (FUTURE) |
| Features (d) | Scales to hundreds | Must be reduced to *n* ≤ 12 | Mandatory reduction stage |
| Qubits (n) | n/a | State-vector memory grows as 2^n | Budget enforcement, warnings, exact-mode guard rails |
| Models | Linear in model count | Linear, but each quantum model is expensive | Parallel execution, caching |
| Configurations (sweeps) | Cheap | Expensive | Kernel cache keyed by config; sweep queueing |

### 33.2 Scalability requirements

| ID | Requirement | Priority |
|---|---|---|
| SC-1 | The classical branch handles ≥ 50,000 rows and ≥ 100 features without code change | SHOULD HAVE |
| SC-2 | The dimensionality-reduction stage is dataset-size-agnostic — introducing a larger cardiovascular dataset requires a config change only | MUST HAVE |
| SC-3 | The quantum branch enforces an explicit training-sample cap with a clear warning and a documented subsampling strategy | MUST HAVE |
| SC-4 | Quantum kernel matrices are cached and reused across models and runs | SHOULD HAVE |
| SC-5 | Circuits are batched and executed in parallel across CPU cores | SHOULD HAVE |
| SC-6 | Runtime and resource estimates are shown before execution | MUST HAVE |
| SC-7 | Pipeline stages are independently re-runnable, so changing only the model does not re-run preprocessing | SHOULD HAVE |
| SC-8 | Intermediate artifacts cached by content hash | SHOULD HAVE |
| SC-9 | Kernel approximation methods (Nyström, random-feature approximations) for large N | FUTURE |
| SC-10 | Distributed or GPU-backed simulation | FUTURE |
| SC-11 | Horizontal scaling of the API with a job queue | FUTURE |

### 33.3 The stated path to a larger dataset (SIH requirement O4)

| Step | What changes | What does not change |
|---|---|---|
| 1. New dataset acquired (larger CVD cohort) | A new schema config file | Ingestion, validation, splitting code |
| 2. Preprocessing tuned | Config values (sentinels, imputation strategy) | Preprocessing implementation |
| 3. Selection and reduction re-tuned | `k` and `n` values | Selector/reducer implementations |
| 4. Quantum branch | Possibly subsampling enabled; same qubit budget | Encoding, feature map, model code |
| 5. Benchmark re-run | New run ID | Benchmarking, explainability, robustness code |

This table is the concrete answer to "is the architecture scalable?" — the answer is that **scaling is a configuration exercise, not a rewrite**, and that the quantum branch's O(N²) cost is the known, documented bottleneck.

---

## 34. Performance Requirements

| ID | Operation | Target | Conditions | Priority |
|---|---|---|---|---|
| PF-1 | Dataset load and validation | ≤ 10 s | UCI-sized dataset | MUST HAVE |
| PF-2 | Full preprocessing + feature engineering + selection + reduction | ≤ 30 s | UCI-sized dataset | MUST HAVE |
| PF-3 | Training all three classical baselines with hyperparameter search | ≤ 3 min | Standard laptop, 4 cores | MUST HAVE |
| PF-4 | QSVM kernel matrix (train/train + test/train) | ≤ 10 min | ≤ 300 training samples, 4–8 qubits, state-vector simulator | MUST HAVE |
| PF-5 | VQC training | ≤ 15 min | ≤ 300 samples, 6 qubits, 2 ansatz layers, ≤ 200 optimizer iterations | MUST HAVE |
| PF-6 | Benchmark computation including bootstrap CIs | ≤ 60 s | 1000 bootstrap resamples | MUST HAVE |
| PF-7 | Classical explainability (SHAP) | ≤ 60 s | UCI-sized dataset | MUST HAVE |
| PF-8 | Quantum explainability (permutation importance) | ≤ 5 min | Cached model, reduced features | SHOULD HAVE |
| PF-9 | Single-record classical inference | ≤ 1 s | — | MUST HAVE |
| PF-10 | Single-record quantum inference | ≤ 10 s | Trained model, simulator | SHOULD HAVE |
| PF-11 | Dashboard page render (non-training pages) | ≤ 2 s | — | MUST HAVE |
| PF-12 | Cached demo scenario load | ≤ 5 s | Pre-computed artifacts | MUST HAVE |
| PF-13 | Robustness suite (repeated CV, noise sweep) for classical models | ≤ 5 min | — | SHOULD HAVE |
| PF-14 | Robustness suite for quantum models | ≤ 30 min | With kernel caching | SHOULD HAVE |

**Demo-safety requirement (MUST HAVE):** the SIH demonstration must never depend on a live quantum training run completing within the presentation window. A pre-computed run is loaded from the registry, and a small live run (reduced samples/qubits) is executed alongside it to prove the pipeline is genuinely live.

---

## 35. Technology Stack

| Layer | Technology | Rationale | Priority |
|---|---|---|---|
| Language | Python 3.10+ | Ecosystem for both ML and quantum SDKs | MUST HAVE |
| Data handling | pandas, NumPy | Standard tabular tooling | MUST HAVE |
| Classical ML | scikit-learn | LR, SVM, pipelines, metrics, CV, calibration | MUST HAVE |
| Gradient boosting | XGBoost (or scikit-learn RandomForest) | Strong tabular baseline | MUST HAVE |
| Imbalance handling | imbalanced-learn | Fold-safe SMOTE | SHOULD HAVE |
| Quantum SDK | **Qiskit** (primary) with Qiskit Machine Learning | Mature feature maps, kernels, VQC, transpiler, simulators, hardware path | MUST HAVE |
| Quantum SDK (alternative) | PennyLane | Autograd-based VQC; useful if gradient-based training is needed | SHOULD HAVE |
| Simulator | Qiskit Aer (state-vector, sampling, noise models) | Meets Section 25 requirements | MUST HAVE |
| Explainability | SHAP, scikit-learn permutation importance, LIME (optional) | Covers both branches | MUST HAVE |
| Statistics | SciPy, statsmodels | McNemar, DeLong/bootstrap, CIs | MUST HAVE |
| Visualization | Matplotlib, Plotly | Static and interactive charts | MUST HAVE |
| Dashboard | **Streamlit** | Fastest route to a complete multi-page workflow UI | MUST HAVE |
| API | **FastAPI** + Pydantic + Uvicorn | Typed, auto-documented REST layer | SHOULD HAVE |
| Experiment tracking | Local JSON/SQLite registry; MLflow optional | Reproducibility without infrastructure overhead | SHOULD HAVE |
| Serialization | joblib, JSON, NPY | Model and artifact persistence | MUST HAVE |
| Configuration | YAML/JSON configs (Pydantic-validated) | Config-driven experiments | MUST HAVE |
| Testing | pytest, pytest-cov | Unit, contract and leakage tests | MUST HAVE |
| Quality | ruff/flake8, black, mypy (partial) | Maintainability | SHOULD HAVE |
| Packaging | requirements.txt / environment.yml; optional Dockerfile | Reproducible environments | MUST HAVE |
| Docs | Markdown + MkDocs (optional) | Deliverable documentation | MUST HAVE |
| Version control | Git + GitHub | Collaboration, run-to-commit traceability | MUST HAVE |
| Frontend SPA (React) | — | FUTURE |
| Database (PostgreSQL) | — | FUTURE |

---

## 36. System Architecture

### 36.1 Layered architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                               │
│   Streamlit Dashboard (15 pages)          CLI (`quantumdx run config.yaml`)│
└───────────────────────────┬───────────────────────────────────────────────┘
                            │  (both call the same core library)
┌───────────────────────────▼───────────────────────────────────────────────┐
│                            SERVICE LAYER                                  │
│   FastAPI  ·  request validation  ·  job orchestration  ·  error envelope  │
└───────────────────────────┬───────────────────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                                │
│   ExperimentRunner: builds the pipeline from config, enforces split &     │
│   preprocessing invariants, runs both branches, writes the run record     │
└───────┬───────────────────────────────────────────────────┬───────────────┘
        │                                                   │
┌───────▼────────────────────────┐                ┌─────────▼─────────────────┐
│      DATA LAYER (shared)       │                │     EVALUATION LAYER      │
│  Ingestion → Validation →      │                │  Benchmarking (metrics,   │
│  Splitting → Preprocessing →   │                │  CIs, significance tests) │
│  Engineering → Selection →     │                │  Explainability           │
│  Reduction → Range-normalizer  │                │  Robustness               │
│  ⇒ ProcessedDataset            │                │  Reporting / Export       │
└───────┬────────────────────────┘                └─────────▲─────────────────┘
        │                                                   │
        ├───────────────────────┬───────────────────────────┤
        │                       │                           │
┌───────▼──────────┐   ┌────────▼─────────────┐             │
│ CLASSICAL BRANCH │   │   QUANTUM BRANCH     │             │
│ LR / RBF-SVM /   │   │ Encoder → FeatureMap │             │
│ XGB or RF        │   │ → QSVM  |  VQC/QNN   │             │
│ (full-d and      │   │                      │             │
│  reduced-n)      │   └────────┬─────────────┘             │
└───────┬──────────┘            │                           │
        │                ┌──────▼──────────────┐            │
        │                │ QUANTUM BACKEND     │            │
        │                │ ABSTRACTION         │            │
        │                │ ├ StatevectorSim    │            │
        │                │ ├ SamplingSim       │            │
        │                │ ├ NoisySim          │            │
        │                │ └ HardwareAdapter   │ (FUTURE)   │
        │                └─────────────────────┘            │
        └────────────────────────┬──────────────────────────┘
                                 │ predictions
                                 ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                          PERSISTENCE LAYER                                │
│  Run Registry (configs, seeds, metrics)  ·  Model Registry (artifacts)     │
│  Artifact Store (runs/{run_id}/…)        ·  Kernel Cache                   │
└───────────────────────────────────────────────────────────────────────────┘
```

### 36.2 Core interfaces (developer contract)

| Interface | Responsibility | Key methods |
|---|---|---|
| `DatasetLoader` | Load raw data against a schema config | `load() -> DataFrame`, `schema() -> Schema` |
| `DataValidator` | Produce a validation report; decide blocking vs warning | `validate(df) -> ValidationReport` |
| `Splitter` | Create and persist split indices before any fitting | `split(df, y) -> SplitIndices` |
| `Preprocessor` | Fit-on-train transformation chain | `fit(X_train)`, `transform(X)`, `save/load` |
| `FeatureEngineer` | Derive clinical features | `fit(X_train)`, `transform(X)` |
| `FeatureSelector` | Rank and select features | `fit(X_train, y_train)`, `transform(X)`, `ranking()` |
| `Reducer` | Reduce to *n* dimensions and normalize into encoding range | `fit(X_train)`, `transform(X)`, `report()` |
| `QuantumEncoder` | Map a classical vector to a circuit | `encode(x) -> QuantumCircuit`, `n_qubits` |
| `FeatureMap` | Parameterized embedding circuit | `build(n_qubits, reps, entanglement)` |
| `QuantumBackend` | Execute circuits | `run(circuits, shots) -> Results`, `capabilities()` |
| `Model` | Uniform model contract for both branches | `fit`, `predict`, `predict_proba`, `save`, `load`, `describe`, `resource_report` |
| `Evaluator` | Compute metrics, CIs and tests | `evaluate(model, X, y) -> MetricSet`, `compare(a, b) -> Comparison` |
| `Explainer` | Global and local explanations | `global_importance(model, X)`, `local_explain(model, x)` |
| `RobustnessSuite` | Stress tests | `run(model, dataset) -> RobustnessReport` |
| `RunRegistry` | Persist and retrieve runs | `create`, `update`, `get`, `list` |

### 36.3 The `ProcessedDataset` object (the fairness contract)

```
ProcessedDataset
├── split_id, preprocessing_config_hash, seed
├── X_train_full,  X_test_full        # d-dimensional processed features
├── X_train_reduced, X_test_reduced   # n-dimensional, range-normalized for encoding
├── y_train, y_test
├── feature_names_full, feature_names_reduced
├── artifacts: {imputer, encoder, scaler, engineer, selector, reducer, range_normalizer}
└── provenance: {dataset_id, dataset_hash, validation_report_ref, warnings_acknowledged}
```

**Every model in every branch receives this object and nothing else.** Because the reduced and full matrices are produced by one chain, from one split, with one seed, the comparison cannot be unfair by accident. The benchmarking engine cross-checks `split_id` and `preprocessing_config_hash` before emitting a comparison (NFR-22, BM-6).

### 36.4 Repository layout (proposed)

```
quantumdx/
├── configs/           # schema_uci_heart.yaml, run configs, device profiles
├── data/raw/          # datasets (git-ignored)
├── quantumdx/
│   ├── data/          # loader, validator, splitter
│   ├── preprocess/    # imputation, encoding, scaling, range normalizer
│   ├── features/      # engineering, selection, reduction
│   ├── models/
│   │   ├── classical/ # logreg, rbf_svm, xgboost
│   │   ├── quantum/   # qsvm, vqc, feature_maps, encoders
│   │   └── registry.py
│   ├── quantum/       # backends, transpilation, diagnostics, kernel cache
│   ├── evaluation/    # metrics, statistics, benchmarking
│   ├── explain/       # classical and quantum explainers
│   ├── robustness/    # cv, noise, subgroups, learning curves
│   ├── tracking/      # run registry, artifact store
│   ├── pipeline.py    # ExperimentRunner
│   └── api/           # FastAPI app
├── dashboard/         # Streamlit pages
├── tests/             # unit, contract, leakage, reproducibility tests
├── runs/              # run artifacts (git-ignored)
└── docs/              # architecture, methods, limitations, demo script
```
---

## 37. User Workflow

### 37.1 The canonical workflow (dashboard)

```
[1] Load data ──▶ [2] Validate ──▶ [3] Configure split & preprocessing
        │                                        │
        │                                        ▼
        │                          [4] Engineer & select features
        │                                        │
        │                                        ▼
        │                          [5] Reduce to n dims (quantum-ready)
        │                                        │
        │                                        ▼
        │                          [6] Choose classical + quantum models
        │                                        │
        │                                        ▼
        │                          [7] Run experiment (both branches)
        │                                        │
        │           ┌────────────────────────────┼────────────────────────────┐
        │           ▼                            ▼                            ▼
        │     [8] Compare                  [9] Explain              [10] Robustness
        │           └────────────────────────────┼────────────────────────────┘
        │                                        ▼
        └────────────────────────────▶ [11] Predict a patient
                                                 │
                                                 ▼
                                        [12] Export / record run ID
```

### 37.2 Gating rules (enforced by DB-1)

| Stage | Unlocked when |
|---|---|
| Validate | A dataset is loaded |
| Configure | Validation has no blocking errors and warnings are acknowledged |
| Features | Split and preprocessing config are set |
| Models | A reduced feature space with dimension *n* exists |
| Run | At least one classical and one quantum model are selected and the resource estimate is accepted |
| Compare / Explain / Robustness | The run has completed (or partially completed with at least two models) |
| Predict | At least one trained model with its bound preprocessing artifacts exists |

### 37.3 CLI workflow (for researchers and sweeps)

```
quantumdx validate  --config configs/run_uci_baseline.yaml
quantumdx run       --config configs/run_uci_baseline.yaml
quantumdx sweep     --config configs/sweep_qubits_featuremaps.yaml
quantumdx report    --run-id run_20260903_141200_a3f9 --format html
quantumdx predict   --run-id run_… --model qsvm --input patient.json
```

CLI parity with the dashboard is **SHOULD HAVE**; `run` and `report` are **MUST HAVE** so the demo can be regenerated headlessly.

---

## 38. MVP Scope

### 38.1 MVP inclusion table

| Capability | In MVP | Notes |
|---|---|---|
| UCI Heart Disease bundled dataset + CSV upload | ✅ MUST | FR-A1, FR-A2 |
| Schema-driven validation with missingness/sentinel detection | ✅ MUST | FR-B1–B6, B9, B10 |
| Stratified split before any fitting; persisted indices | ✅ MUST | FR-C1–C5 |
| Leakage-safe preprocessing chain | ✅ MUST | FR-D1–D6, D8, D10 |
| Clinical feature engineering (toggleable) | ✅ MUST | FR-E1, E5 |
| Feature selection (filter + embedded) with ranked table | ✅ MUST | FR-F1, F2, F6, F7 |
| Dimensionality reduction (PCA + top-k) to n=4–8 with range normalization | ✅ MUST | FR-G1–G3, G6, G7, G9 |
| Classical baselines: LR, RBF-SVM, XGBoost/RF, tuned | ✅ MUST | FR-H1–H5, H8 |
| QSVM with configurable feature map | ✅ MUST | FR-I1, I3, I4, I6, I7, I14 |
| VQC/QNN with configurable ansatz and optimizer | ✅ MUST | FR-I2, I5 |
| Prediction: single record + batch, both branches | ✅ MUST | FR-J1–J8 |
| Benchmark: full metric suite, CIs, McNemar/DeLong, efficiency | ✅ MUST | FR-K1–K8, K11 |
| Explainability: SHAP (classical) + permutation/KernelSHAP (quantum) | ✅ MUST | FR-L1–L4, L9 |
| Robustness: repeated CV, seeds, noise sweep, subgroups, generalization gap | ✅ MUST | FR-M1–M3, M5, M7 |
| Simulator: state-vector + shot-based, seeded, resource-guarded | ✅ MUST | SIM-1, 2, 4, 7, 8, 11, 12 |
| Hardware compatibility: backend abstraction + transpilation report + budgets | ✅ MUST | HW-1–HW-8 |
| Dashboard: pages 1–11 | ✅ MUST | DB-1–DB-9 |
| Run registry with reproducible run IDs | ✅ MUST | ET-1–ET-5, ET-10 |
| Model persistence bound to preprocessing artifacts | ✅ MUST | MM-2, MM-4, MM-5, MM-10, MM-11 |
| Documentation: README, architecture, methods, limitations, demo script | ✅ MUST | NFR-17, QH-5 |
| REST API (full) | ⬜ SHOULD | `/predict` is MUST; the rest SHOULD |
| Diagnostics page (kernel concentration, loss curve, circuit view) | ⬜ SHOULD | FR-I9–I11, QSVM-4, QSVM-5, VQC-5, VQC-7 |
| Kernel caching | ⬜ SHOULD | FR-I8 |
| Noisy simulator evaluation | ⬜ SHOULD | SIM-3, RB-9 |
| Compliance/traceability page | ⬜ SHOULD | Journey C |
| Autoencoder reducer, counterfactuals, stability selection | ⬜ NICE | — |
| Real hardware execution, larger dataset validation, adaptive feature maps | ⬜ FUTURE | Section 39 |

### 38.2 MVP build order (recommended sprints)

| Sprint | Deliverable | Exit criterion |
|---|---|---|
| S1 | Data layer: loader, schema config, validator, splitter | UCI loads; validation report correct; split persisted |
| S2 | Preprocessing + engineering + selection + reduction | `ProcessedDataset` produced; leakage test passes |
| S3 | Classical branch + evaluation metrics | Three tuned baselines with a metrics table |
| S4 | Quantum backend abstraction + QSVM | Quantum kernel computed; QSVM trained and evaluated |
| S5 | VQC/QNN | VQC trains; loss curve recorded |
| S6 | Benchmarking with CIs and significance tests | Fair comparison table with verdict strings |
| S7 | Explainability (both branches) | Global + local explanations rendered |
| S8 | Robustness suite | CV, noise, subgroup reports produced |
| S9 | Dashboard assembly | End-to-end workflow usable by a non-author |
| S10 | Run registry, exports, docs, demo rehearsal | Cached demo loads in ≤ 5 s; docs complete |

---

## 39. Future Scope

| ID | Item | Description | Priority |
|---|---|---|---|
| FUT-1 | **Larger cardiovascular dataset validation** | Ingest a substantially larger CVD cohort through the same pipeline to validate scalability and generalization claims | FUTURE (highest priority post-SIH) |
| FUT-2 | **Full-scale hardware execution** | The scoped validation run is now in the MVP (PRD §26, `MVP_SPEC.md` §8.5). What remains future is the *full* benchmark on hardware — complete kernels, repeated CV, seed and configuration sweeps — which needs a paid plan and roughly 3–4 hours of QPU time per run | FUTURE |
| FUT-3 | **Adaptive / optimized quantum feature maps** | Kernel-target alignment, feature-map architecture search — the intended research novelty, deliberately not claimed now | FUTURE |
| FUT-4 | Error mitigation | Readout-error mitigation, zero-noise extrapolation for hardware runs | FUTURE |
| FUT-5 | Kernel approximation for large N | Nyström or random-feature approximations to break the O(N²) barrier | FUTURE |
| FUT-6 | Additional modalities | ECG waveforms, cardiac imaging, genomics — each requiring a new front-end pipeline feeding the same reduction/quantum stages | FUTURE |
| FUT-7 | Multi-disease / multi-class | Extend beyond binary CVD risk | FUTURE |
| FUT-8 | Survival / time-to-event modelling | Predict time to cardiovascular event, not just binary risk | FUTURE |
| FUT-9 | EHR integration (FHIR) | Pull structured cardiovascular data directly | FUTURE |
| FUT-10 | Federated / privacy-preserving training | Train across hospitals without moving data | FUTURE |
| FUT-11 | Quantum-native interpretability | Attribution grounded in circuit structure and kernel geometry | FUTURE |
| FUT-12 | Production deployment | Auth, RBAC, TLS, containerized deployment, monitoring, drift detection | FUTURE |
| FUT-13 | Prospective clinical validation | Ethics approval, prospective cohort, regulatory pathway | FUTURE |
| FUT-14 | React SPA frontend | Replace Streamlit with a production UI | FUTURE |

---

## 40. Risks and Mitigations

### 40.1 Technical risks

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | **Quantum models do not outperform classical baselines** | High | Low (by design) | The product is framed as a fair measurement platform; a null result is a valid, publishable outcome. Never promise advantage in the pitch. Report honestly (QH-1–QH-5) |
| R2 | **Quantum kernel concentration** — kernel becomes near-constant, SVM degenerates | Medium | High | Diagnostic metric (QSVM-4); keep qubits and reps low; sweep configurations; document the effect as a finding |
| R3 | **Barren plateaus** in VQC training | Medium | Medium | Shallow ansatz, SPSA/COBYLA optimizers, gradient-variance diagnostic (VQC-7), multiple restarts |
| R4 | **QSVM O(N²) cost makes runs too slow** | High | High | Cap training samples, cache kernels, batch circuits, use exact simulation, precompute the demo run |
| R5 | **Information loss in reduction destroys signal** | Medium | High | Report explained variance and pre/post-reduction classical accuracy (FR-G8); run Protocol A so classical models face the same loss |
| R6 | **Data leakage slips into the pipeline** | Medium | Critical | Fit-on-train discipline, single pipeline object, automated leakage test in CI (NFR-7, PRE-17) |
| R7 | **Small dataset → unstable, non-reproducible results** | High | High | Repeated CV, multi-seed runs, bootstrap CIs, explicit variance reporting; never report a single split |
| R8 | **Overfitting through configuration sweeps (multiple comparisons)** | Medium | High | Disclose the number of configurations tried (ST-6); reserve a final validation set; report the sweep, not just the winner |
| R9 | **Simulator memory/runtime blowup at high qubit counts** | Medium | Medium | Hard qubit budget, pre-run resource estimate, guard rails (SIM-8) |
| R10 | **Quantum SDK version churn / API breakage** | Medium | Medium | Pin dependency versions; isolate SDK calls behind the backend abstraction |
| R11 | **Explainability for quantum models is weak or misleading** | Medium | Medium | Use multiple methods (permutation, KernelSHAP, surrogate, kernel-similarity); always state limitations (XI-1) |
| R12 | **Subgroup performance disparity (e.g. lower sensitivity in women)** | Medium | High | Subgroup reporting is mandatory (RB-5, SEC-12); surface rather than hide |

### 40.2 Project and delivery risks

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R13 | Scope creep into imaging/genomics | High | High | Explicit non-goals (Section 6); FUTURE labels |
| R14 | Demo fails live (training too slow, environment issue) | Medium | Critical | Pre-computed cached run (DB-8, PF-12); offline execution (NFR-11); rehearsed demo script |
| R15 | Team bandwidth — quantum work concentrated in one person | High | High | Backend abstraction and `Model` interface let classical and quantum tracks progress independently; documented contracts |
| R16 | Dashboard consumes disproportionate time | Medium | Medium | Streamlit chosen for speed; UI built only after the core library works (Sprint S9) |
| R17 | Over-claiming in the pitch, then being challenged by the jury | Medium | Critical | The honesty requirements are product requirements, not just etiquette; the UI cannot render an unqualified claim (DB-5, QH-1) |
| R18 | Reproducibility breaks near the deadline | Medium | High | Run registry from Sprint S1; seed everything; reproducibility test in CI (NFR-5) |
| R19 | Larger dataset unavailable in time | High | Low | It is explicitly FUTURE; the architecture-readiness argument (Section 33.3) is the Phase 1 deliverable |

---

## 41. Success Metrics

### 41.1 Product/build success (what we control)

| ID | Metric | Target | Priority |
|---|---|---|---|
| SM-1 | End-to-end pipeline completes on UCI data without manual intervention | 100% of demo runs | MUST HAVE |
| SM-2 | Classical baselines trained and evaluated | ≥ 3 models | MUST HAVE |
| SM-3 | Quantum models trained and evaluated | ≥ 2 models (QSVM, VQC) | MUST HAVE |
| SM-4 | Leakage test passes | Always | MUST HAVE |
| SM-5 | Reproducibility test passes (state-vector run reproduces metrics exactly) | Always | MUST HAVE |
| SM-6 | Every reported comparison carries a CI and a p-value | 100% | MUST HAVE |
| SM-7 | Explanations available for at least one classical and one quantum model | Both present | MUST HAVE |
| SM-8 | Robustness suite produces CV, noise and subgroup reports | All three | MUST HAVE |
| SM-9 | Transpilation report generated against a near-term device profile | Present | MUST HAVE |
| SM-10 | Dashboard workflow completed by an outside tester without help | ≤ 15 minutes | SHOULD HAVE |
| SM-11 | Core-library test coverage | ≥ 70% | SHOULD HAVE |
| SM-12 | Documentation completeness (README, architecture, methods, limitations, demo script) | All present | MUST HAVE |

### 41.2 Scientific/experimental outcomes (what we measure, not promise)

| ID | Question | How it is answered | Success definition |
|---|---|---|---|
| SM-13 | Does QSVM differ from RBF-SVM on the same reduced features? | Protocol A with McNemar and DeLong tests | A **measured, qualified answer** — positive, null, or negative — is success |
| SM-14 | Does the hybrid pipeline compete with the best classical pipeline? | Protocol B | A qualified answer with CIs |
| SM-15 | How does each branch degrade under noise and missingness? | Robustness sweeps | Degradation curves produced for all models |
| SM-16 | How does performance vary with qubits, feature map and reps? | Configuration sweep | A documented sensitivity analysis |
| SM-17 | What is the computational cost of the quantum branch? | Efficiency table | Circuit counts, depth and wall-clock reported |

**Explicit statement:** SM-13 through SM-17 are **not** success-conditional on quantum outperforming classical. Success is producing a defensible measurement.

### 41.3 Reference performance context (for calibration, not as a target claim)

Well-tuned classical models on the UCI Heart Disease Cleveland subset typically land in the low-to-mid 80s percent accuracy with ROC-AUC around 0.85–0.92, with wide confidence intervals because the test set holds only ~60 records. **Any quantum result must be interpreted against that variance.** A 2–3 percentage-point difference on this dataset is almost certainly within noise, and the platform must say so.

---

## 42. SIH26139 Requirement-to-Feature Traceability Matrix

### 42.1 Objectives → features

| SIH26139 element | PRD section | Implementing requirement IDs | Dashboard page | Priority |
|---|---|---|---|---|
| **Objective 1** — Design a hybrid quantum-classical ML architecture for early disease detection | 10, 15, 16, 36 | FR-C1–C5, FR-G1, `ProcessedDataset` contract, NFR-22 | Home, Configure, Features | MUST HAVE |
| **Objective 2** — Quantum-enhanced classification models processing high-dimensional biomedical data | 18 | FR-I1–I7, QM-1, QM-2, QE-1–QE-3, QF-1–QF-4, FR-G1–G7 | Models, Run | MUST HAVE |
| **Objective 3** — Improve accuracy, sensitivity, specificity vs classical baselines | 17, 23 | FR-K1–K8, ST-1–ST-7, BM-1–BM-7, QH-1–QH-3 | Compare | MUST HAVE (measurement); outcome not assumed |
| **Objective 4** — Scalable, interpretable, compatible with near-term hardware and simulators | 22, 25, 26, 33 | SIM-1–SIM-12, HW-1–HW-8, SC-1–SC-8, XC/XQ series | Diagnostics, Explain | MUST HAVE |
| **Objective 5** — Incorporate preprocessing, feature selection, explainability modules | 19, 20, 21, 22 | PRE-1–PRE-17, FE-1–FE-8, FS-1–FS-12, XC-1–XC-5, XQ-1–XQ-6 | Validate, Configure, Features, Explain | MUST HAVE |
| **Objective 6** — Benchmark on accuracy, computational efficiency, generalization | 23, 24 | FR-K1–K12, BR-1–BR-8, RB-1–RB-12 | Compare, Robustness | MUST HAVE |

### 42.2 Expected-solution elements → features

| Expected solution element (SIH26139) | Implementing requirements | Priority |
|---|---|---|
| "Fully functional hybrid QML software platform" | Sections 27, 30, 36; FR-N1–N4; NFR-8, NFR-9 | MUST HAVE |
| "Early disease detection on real or benchmark biomedical datasets" | FR-A1–A4 (UCI benchmark; user CSV); Section 19.1 | MUST HAVE |
| "Data handling pipelines" | FR-A*, FR-B*, FR-C*, FR-D*, FR-E*, FR-F*, FR-G* | MUST HAVE |
| "Hybrid quantum-classical model implementation" | FR-H*, FR-I*, QM-1, QM-2, HW-1 | MUST HAVE |
| "Training and inference workflows" | FR-H*, FR-I*, FR-J1–J8; Journeys A and B | MUST HAVE |
| "Performance evaluation" | FR-K1–K12, ST-1–ST-7, BR-1–BR-8 | MUST HAVE |
| "Explainability features" | FR-L1–L9, XC-*, XQ-*, XI-* | MUST HAVE |
| "Comprehensive documentation" | NFR-17, QH-5, MM-9, Section 35 | MUST HAVE |

### 42.3 Description elements → features

| Description phrase | Feature |
|---|---|
| "integrate classical pre-processing and feature engineering" | Modules D and E; leakage-safe chain |
| "quantum-enhanced learning models (QSVM, QNN, VQC)" | QM-1 (QSVM), QM-2 (VQC/QNN) |
| "applied to biomedical datasets for early identification (e.g. cardiovascular disorders)" | Phase 1 disease = cardiovascular; UCI Heart Disease config |
| "support data ingestion" | Module A |
| "hybrid model training" | Modules H and I under one orchestrator |
| "prediction" | Module J |
| "explainability" | Module L |
| "performance evaluation against purely classical baselines" | Module K, Protocols A and B |

### 42.4 Core product checklist (from the project brief) → features

| # | Required user capability | Requirement IDs | MVP |
|---|---|---|---|
| 1 | Upload/provide cardiovascular data | FR-A1, FR-A2, FR-A3 | ✅ |
| 2 | Validate and inspect the dataset | FR-B1–FR-B11 | ✅ |
| 3 | Leakage-safe preprocessing | FR-C1–C5, FR-D1–D10, PRE-14, PRE-17 | ✅ |
| 4 | Feature engineering and selection | FR-E1–E5, FR-F1–F7 | ✅ |
| 5 | Reduce to quantum-compatible dimensions | FR-G1–G9, QE-2 | ✅ |
| 6 | Train classical baselines | FR-H1–H5, CML-1–CML-10 | ✅ |
| 7 | Train hybrid quantum-classical models | FR-I1–I7, I14, QSVM-*, VQC-* | ✅ |
| 8 | Perform predictions | FR-J1–J8 | ✅ |
| 9 | Compare quantum and classical models | FR-K4, BM-1–BM-7, BR-4 | ✅ |
| 10 | Evaluate accuracy, sensitivity, specificity, and other metrics | FR-K1, FR-K2, FR-K3 | ✅ |
| 11 | Explain predictions | FR-L1–L4, L9 | ✅ |
| 12 | Robustness/generalization evaluation | FR-M1–M3, M5, M7, RB-1–RB-7 | ✅ |
| 13 | Display results through a clear dashboard | DB-1–DB-9, Section 27.1 | ✅ |
| 14 | Quantum simulation + near-term hardware compatibility | SIM-1–SIM-12, HW-1–HW-8 | ✅ |

---

## 43. Acceptance Criteria

Each criterion is binary — it passes or it does not. These are the conditions under which the build is declared complete.

### 43.1 Data and pipeline

| ID | Acceptance criterion | Priority |
|---|---|---|
| AC-1 | Loading the bundled UCI Heart Disease dataset produces a validation report that correctly identifies `?` sentinels in `ca`/`thal` and reports class balance | MUST HAVE |
| AC-2 | Uploading a CSV with a missing required column produces a specific, blocking error naming the column | MUST HAVE |
| AC-3 | The train/test split is created before any transformer is fitted, and the split indices are retrievable from the run record | MUST HAVE |
| AC-4 | The automated leakage test passes: scaler/imputer/selector/reducer parameters computed with and without test data present are identical | MUST HAVE |
| AC-5 | The reduced feature matrix has exactly *n* columns, all values within the declared encoding range | MUST HAVE |
| AC-6 | Disabling feature engineering produces a valid run (ablation is possible) | MUST HAVE |
| AC-7 | The ranked feature table is displayed and exportable | MUST HAVE |

### 43.2 Models

| ID | Acceptance criterion | Priority |
|---|---|---|
| AC-8 | Logistic Regression, RBF-SVM and XGBoost/RF all train with hyperparameter search and record their best parameters | MUST HAVE |
| AC-9 | Each classical model is evaluated on both the full-*d* and reduced-*n* feature spaces | MUST HAVE |
| AC-10 | QSVM computes a quantum kernel matrix and trains an SVC with `kernel='precomputed'`, producing predictions and probabilities | MUST HAVE |
| AC-11 | Changing the feature map, reps, entanglement or qubit count demonstrably changes the executed circuit and the kernel matrix | MUST HAVE |
| AC-12 | VQC trains to completion, records a loss curve, and produces predictions | MUST HAVE |
| AC-13 | Every trained model can be saved, reloaded, and produce identical predictions on the same input | MUST HAVE |
| AC-14 | Loading a model with a mismatched feature schema fails with a clear error rather than silently mispredicting | MUST HAVE |

### 43.3 Benchmarking and honesty

| ID | Acceptance criterion | Priority |
|---|---|---|
| AC-15 | All models in a comparison share the same `split_id` and `preprocessing_config_hash`; attempting otherwise raises an explicit error | MUST HAVE |
| AC-16 | The leaderboard reports accuracy, sensitivity, specificity, precision, NPV, F1, balanced accuracy, ROC-AUC, PR-AUC and MCC | MUST HAVE |
| AC-17 | Bootstrap confidence intervals are shown for accuracy and ROC-AUC | MUST HAVE |
| AC-18 | A McNemar test (accuracy) and a DeLong or bootstrap test (AUC) are reported for each quantum-vs-classical pair | MUST HAVE |
| AC-19 | The comparison panel displays one of exactly three verdict strings — "significantly better", "no significant difference", "significantly worse" — derived from the test, never from the point estimate alone | MUST HAVE |
| AC-20 | ROC and PR curves for all models are overlaid on shared axes | MUST HAVE |
| AC-21 | The efficiency table reports training time, inference latency, circuits executed, transpiled depth and total shots | MUST HAVE |
| AC-22 | No screen or export anywhere in the product states or implies quantum superiority without an accompanying CI and p-value | MUST HAVE |

### 43.4 Explainability

| ID | Acceptance criterion | Priority |
|---|---|---|
| AC-23 | Global feature importance is displayed for at least one classical and one quantum model | MUST HAVE |
| AC-24 | Selecting a patient produces a local attribution plot for a classical model and for a quantum model | MUST HAVE |
| AC-25 | When explanations are computed on reduced components, the UI says so and offers the PCA loading map | MUST HAVE |
| AC-26 | Each explanation view states its method and its limitations | MUST HAVE |

### 43.5 Robustness

| ID | Acceptance criterion | Priority |
|---|---|---|
| AC-27 | Repeated stratified CV results (mean ± std) are reported for every model | MUST HAVE |
| AC-28 | The noise-degradation curve is produced for all models across at least four noise levels | MUST HAVE |
| AC-29 | Subgroup sensitivity and specificity are reported by sex and age band | MUST HAVE |
| AC-30 | The generalization gap (train minus test) is reported per model | MUST HAVE |
| AC-31 | Results across at least three seeds are reported with their variance | MUST HAVE |

### 43.6 Quantum execution and hardware readiness

| ID | Acceptance criterion | Priority |
|---|---|---|
| AC-32 | The state-vector backend produces identical metrics across two identical runs | MUST HAVE |
| AC-33 | The shot-based backend runs the same models without any code change, only a config change | MUST HAVE |
| AC-34 | A resource estimate (circuits, estimated time) is displayed before a quantum run begins | MUST HAVE |
| AC-35 | Exceeding the qubit or depth budget produces a warning with concrete remedies | MUST HAVE |
| AC-36 | A transpilation report against a near-term device profile is generated, reporting required qubits, transpiled depth and two-qubit gate count | MUST HAVE |
| AC-37 | The quantum circuit diagram for the current configuration is rendered | SHOULD HAVE |

### 43.7 Product, dashboard and documentation

| ID | Acceptance criterion | Priority |
|---|---|---|
| AC-38 | A first-time user completes upload → validate → configure → train → compare → explain → predict entirely through the dashboard | MUST HAVE |
| AC-39 | The cached demo scenario loads in ≤ 5 seconds | MUST HAVE |
| AC-40 | Every run produces a run record sufficient to reproduce it, including seeds and library versions | MUST HAVE |
| AC-41 | The disclaimer appears on Home, Compare and Predict | MUST HAVE |
| AC-42 | Single-record prediction returns probability, risk band, threshold, model ID and the classical/quantum agreement flag | MUST HAVE |
| AC-43 | The entire demo runs with networking disabled | MUST HAVE |
| AC-44 | README, architecture doc, methods doc, limitations doc and demo script are present and accurate | MUST HAVE |
| AC-45 | The REST `/predict` endpoint returns a valid response for a well-formed record | MUST HAVE |
| AC-46 | The benchmark report exports to CSV and JSON | SHOULD HAVE |

---

## 44. MVP Definition

> **This section is the contract for the SIH 2026 demonstration.** If everything below works, the prototype is complete. If anything below does not work, the prototype is not ready — regardless of how much else has been built.

### 44.1 What MUST be working on demo day

| # | Capability | Verified by |
|---|---|---|
| 1 | **Data ingestion** — the bundled UCI Heart Disease dataset loads with one click, and an arbitrary conforming CSV can be uploaded | AC-1, AC-2 |
| 2 | **Validation and inspection** — a validation report showing schema checks, sentinel/missingness detection (`ca`, `thal`, `chol=0`), class balance, and feature profiles | AC-1 |
| 3 | **Leakage-safe pipeline** — stratified split performed before any fitting; all transformers fitted on training data only; the automated leakage test passes and can be shown to the jury | AC-3, AC-4 |
| 4 | **Feature engineering and selection** — toggleable clinical derived features; a ranked feature table from at least one filter and one embedded method | AC-6, AC-7 |
| 5 | **Dimensionality reduction to a quantum-ready vector** — PCA (or top-*k*) reduces to *n* = 4–8 dimensions, range-normalized into `[0, π]`, with explained variance shown | AC-5 |
| 6 | **Three tuned classical baselines** — Logistic Regression, RBF-SVM and XGBoost/Random Forest, each trained on both the full and reduced feature spaces | AC-8, AC-9 |
| 7 | **QSVM** — a quantum kernel matrix computed from a configurable feature map, feeding a precomputed-kernel SVC that produces predictions and probabilities | AC-10, AC-11 |
| 8 | **VQC/QNN** — a variational classifier that trains to completion, records its loss curve, and predicts | AC-12 |
| 9 | **Prediction** — single-patient form input and batch CSV scoring, using stored preprocessing artifacts, with classical and quantum outputs shown side by side | AC-42 |
| 10 | **Fair benchmark** — one comparison table over an identical split and identical preprocessing, reporting accuracy, sensitivity, specificity, ROC-AUC and the rest of the metric suite, each with bootstrap confidence intervals | AC-15, AC-16, AC-17 |
| 11 | **Statistical verdict** — McNemar and DeLong/bootstrap tests for each quantum-vs-classical pair, rendered as an explicit verdict string; the UI is incapable of showing a bare superiority claim | AC-18, AC-19, AC-22 |
| 12 | **Computational-efficiency reporting** — training time, inference latency, circuits executed, transpiled depth and total shots | AC-21 |
| 13 | **Explainability on both branches** — SHAP for a classical model, permutation importance plus KernelSHAP for a quantum model, global and local, with stated limitations | AC-23, AC-24, AC-26 |
| 14 | **Robustness evidence** — repeated stratified CV with mean ± std, a noise-degradation curve, subgroup sensitivity/specificity by sex and age band, generalization gap, and multi-seed variance | AC-27 to AC-31 |
| 15 | **Simulator execution with reproducibility** — state-vector backend produces identical results on repeat runs; a shot-based backend runs the same models via config change only | AC-32, AC-33 |
| 16 | **Near-term hardware compatibility evidence** — backend abstraction in place plus a transpilation report against a device profile, with qubit and depth budget enforcement | AC-35, AC-36 |
| 17 | **Dashboard** — pages 1 through 11 usable end to end by someone who has not seen the code | AC-38 |
| 18 | **Reproducibility** — every result carries a run ID whose stored record is sufficient to regenerate it | AC-40 |
| 19 | **Offline, demo-safe operation** — the whole demonstration runs with networking disabled, and a cached run loads in ≤ 5 seconds so the demo never depends on live training | AC-39, AC-43 |
| 20 | **Documentation and honest framing** — README, architecture, methods, limitations and demo script present; disclaimers visible; no advantage claimed anywhere without evidence | AC-41, AC-44 |

### 44.2 What is explicitly NOT required for the MVP

Full-scale benchmarking on real quantum hardware (a *scoped* hardware validation run **is** in the MVP — see §26); live QPU execution during the demo; validation on a larger cardiovascular dataset; adaptive or optimized feature maps; imaging or genomics ingestion; multi-class or multi-disease support; authentication and multi-tenancy; error mitigation; federated learning; any regulatory or clinical-validation claim.

### 44.3 The MVP demo script (10 minutes)

| Time | Action | Point being demonstrated |
|---|---|---|
| 0:00–1:00 | Open Home; state the problem, the disease, the dataset, and the honesty position | Framing; SIH26139 alignment |
| 1:00–2:00 | Load UCI dataset; open the validation report; point at the detected `?` sentinels and `chol = 0` | Real data handling, not a toy loader |
| 2:00–3:00 | Configure split and preprocessing; state that the split precedes all fitting; show the passing leakage test | Methodological rigour — the differentiator |
| 3:00–4:00 | Show the ranked feature table and the PCA reduction to 6 dimensions with explained variance and encoding range | Quantum-ready feature construction |
| 4:00–5:00 | Show the model configuration page: feature map, reps, entanglement, qubits, backend, shots, and the resource estimate | Quantum realism and control |
| 5:00–6:30 | Load the cached full run; start a small live quantum run alongside it | Live proof plus demo safety |
| 6:30–8:00 | Walk the Compare page: leaderboard with CIs, ROC overlay, efficiency table, and the verdict string — read the actual verdict aloud, whatever it says | Fair benchmarking and honesty |
| 8:00–9:00 | Explain page: global and local explanations for a classical and a quantum model, with the PCA loading map | Interpretability on both branches |
| 9:00–9:40 | Robustness page: CV variance, noise curve, subgroup slices | Generalization evidence |
| 9:40–10:00 | Predict page: score one patient with both branches; show the disclaimer; show the run ID | End-to-end product completeness |

### 44.4 The single sentence that defines MVP success

> **A person who has never seen the code can, in one sitting, take a cardiovascular dataset from raw file to an explained, benchmarked, statistically qualified risk prediction — with a quantum model and a classical model trained on exactly the same data and compared honestly.**

---

## Appendix A — Glossary

| Term | Definition |
|---|---|
| **Hybrid quantum-classical** | An architecture where classical computation handles data processing and optimization while quantum computation handles a specific sub-task (here, feature mapping / kernel evaluation) |
| **Quantum feature map** | A parameterized circuit that embeds a classical vector into a quantum state |
| **Quantum kernel** | The squared fidelity between two embedded states, used as a similarity measure by an SVM |
| **QSVM** | Support Vector Machine using a precomputed quantum kernel matrix |
| **VQC / QNN** | Variational Quantum Classifier / Quantum Neural Network — an encoded input followed by a trainable ansatz optimized classically |
| **Ansatz** | The trainable parameterized circuit in a variational model |
| **Barren plateau** | A regime where variational gradients vanish exponentially, stalling training |
| **Kernel concentration** | Degeneration of a quantum kernel toward a constant off-diagonal value, destroying discriminative power |
| **Shots** | Number of repeated circuit executions used to estimate a measurement statistic |
| **State-vector simulation** | Exact, noiseless simulation of the quantum state (no sampling noise) |
| **Transpilation** | Rewriting a circuit into a target device's native gates and connectivity |
| **Data leakage** | Any influence of test data on the training process, which inflates reported performance |
| **Sensitivity / Specificity** | True positive rate / true negative rate |
| **Protocol A / B** | The strict like-for-like and practical comparison protocols defined in Section 16.3 |

## Appendix B — Open Decisions

| ID | Open question | Owner | Needed by |
|---|---|---|---|
| OD-1 | Qiskit-only, or Qiskit for QSVM plus PennyLane for gradient-based VQC? | Quantum lead | Sprint S4 |
| OD-2 | XGBoost or Random Forest as the third classical baseline (dependency weight vs performance)? | ML lead | Sprint S3 |
| OD-3 | Default qubit count for the demo: 4, 6 or 8 (accuracy vs runtime)? | Team | Sprint S6 |
| OD-4 | Cleveland-only or the combined multi-site UCI dataset as the primary benchmark? | ML lead | Sprint S1 |
| OD-5 | Which near-term device profile to use for the transpilation report? | Quantum lead | Sprint S4 |
| OD-6 | Which larger cardiovascular dataset to target for Phase 2 scalability validation? | Team | Post-SIH |

## Appendix C — Limitations and Threats to Validity (must appear in the delivered documentation)

1. **Dataset size.** The UCI Heart Disease dataset is small (roughly 300–900 records). Test-set confidence intervals are wide; small metric differences are not meaningful.
2. **Dataset representativeness.** It is a single benchmark cohort, decades old, from a limited set of sites. It does not represent contemporary Indian cardiovascular populations, nor high-dimensional biomedical data generally.
3. **Reduction information loss.** The quantum branch sees a compressed representation. Poor quantum results may reflect the reducer, not the quantum model — which is why Protocol A exists.
4. **Simulator, not hardware.** All Phase 1 results are simulator results. Real devices add noise, decoherence and readout error; hardware results may differ substantially.
5. **Multiple comparisons.** Sweeping configurations and reporting the best inflates apparent performance. The number of configurations tried is disclosed with every sweep.
6. **Explainability limits.** Permutation importance assumes feature independence; SHAP on reduced components explains components, not raw clinical variables; surrogate models explain the surrogate, not the model, up to a stated fidelity.
7. **No clinical validation.** No prospective evaluation, no external cohort, no regulatory assessment. The output is decision support for research purposes only.

