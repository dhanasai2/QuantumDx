# QuantumDx 🫀⚛️

<div align="center">

![SIH 2026](https://img.shields.io/badge/SIH%202026-Grand%20Finale-blue?style=for-the-badge&logo=shield)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16.3-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![Qiskit](https://img.shields.io/badge/Qiskit-2.2-6929C4?style=for-the-badge&logo=qiskit&logoColor=white)
![IBM Quantum](https://img.shields.io/badge/IBM%20Quantum-ibm__marrakesh-052146?style=for-the-badge&logo=ibm&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-444%2F444%20Passing-10b981?style=for-the-badge&logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-Research-gray?style=for-the-badge)

### **Hybrid Quantum-Classical Cardiovascular Risk Stratification & Clinical Telemetry Terminal**

**Smart India Hackathon 2026 — Problem Statement SIH26139**  
**Team Name:** Quantum Bodha (Team ID: `149574`)

---

</div>

## 📌 Executive Summary

**QuantumDx** is an open, leakage-audited, hardware-validated **hybrid quantum-classical diagnostic platform** engineered for early cardiovascular disease (CVD) risk detection. 

Cardiovascular disease is the **#1 global cause of mortality**, claiming **17.9 million lives annually**. Traditional medical AI models face two critical hurdles:
1. **Linear Simplicity & Non-Linear Complexity:** Classical linear models oversimplify complex multi-biomarker synergies (e.g., blood pressure, blood glucose, lipid elevation, and age interaction).
2. **The "Black-Box" Medical Barrier:** Doctors cannot act on unexplainable AI predictions without understanding the underlying medical rationale.

**QuantumDx** resolves these challenges by combining a **4-qubit parameterized quantum circuit** running on physical **IBM Quantum Hardware (`ibm_marrakesh`)** with gradient-boosted decision trees (XGBoost) and **TreeSHAP explainability attributions** evaluated across **66,641 patient records**.

---

## 🌟 Key System Innovations

* 🛡️ **Zero-Leakage Data Firewall:** Imputation, Z-score scaling, and Adaptive PCA are strictly fit *inside cross-validation training folds*, eliminating optimistic evaluation bias.
* ⚛️ **IBM QPU Hardware Execution:** Verified on physical 156-qubit **`ibm_marrakesh`** processor (IBM Quantum Job ID: `dag54f8mhr3c73e4m300`).
* 🔒 **Quantum OOD Hilbert Safety Guard:** Evaluates quantum state fidelity ($\mathcal{F} = |\langle \psi_{\text{test}} | \psi_{\text{ref}} \rangle|^2 = 0.984$) to verify test samples remain inside the 95% Hilbert manifold boundary.
* 🫀 **Cardio-Renal-Metabolic (CRM) Tri-Organ Panel:** Evaluates Cardiovascular Risk %, Diabetic Cardiomyopathy Index %, and Vascular Stiffness Index (mmHg).
* 🏥 **FHIR / HL7 EHR Integration:** Directly ingests standardized HL7 FHIR JSON `Patient` and `Observation` Bundle payloads.
* 📑 **Institutional Clinical Diagnostic PDF Exporter:** Generates printable, multi-page hospital diagnostic reports with standard 11-biomarker reference range tables, TreeSHAP attributions, Class I/IIa guidelines, and digital MD attestations.
* 🔊 **Live ECG Audio-Visual Telemetry:** Real-time Web Audio API heartbeat pulse sonification matched to patient heart rates and systolic pressure deltas.

---

## 🏗️ End-to-End System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             RAW PATIENT DATA INPUT                               │
│   X = [age, height, weight, ap_hi, ap_lo, cholesterol, gluc, gender, smoke, ...] │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   STAGE 1: PREPROCESSING & ADAPTIVE PCA                          │
│   1. Standardize: Z = (X - μ) / σ                                                │
│   2. SVD / Covariance: Σ = (1/(N-1)) Zᵀ Z ⟹ Σ vₖ = λₖ vₖ                       │
│   3. Project top 4 PCs: P₄ = Z · W₄                                              │
│   4. Angle Encoding Scale: θₖ = π · MinMax(Pₖ) ∈ [0, π]                          │
└──────────────────┬──────────────────────────────────────────────┬────────────────┘
                   │                                              │
                   ▼                                              ▼
┌──────────────────────────────────────┐     ┌─────────────────────────────────────┐
│    CLASSICAL BRANCH                  │     │    QUANTUM BRANCH (IBM QPU)         │
│  Preserves all 11 raw features       │     │  • Initial State: |0000⟩            │
│  X_classical = Z                     │     │  • Superposition: H^{\otimes 4}     │
│                                      │     │  • Angle Encoding: RY(θₖ)           │
│  Passes directly to XGBoost &        │     │  • Entanglement: CNOT Chain         │
│  TreeSHAP explainability engine      │     │  • Readout: ⟨Zₖ⟩ Expectation Values │
└──────────────────┬───────────────────┘     └──────────────────┬──────────────────┘
                   │                                            │
                   └─────────────────────┬──────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       STAGE 3: HYBRID FUSION & PREDICTION                        │
│   Concatenated Vector: H = [P₄ || ⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₃⟩]                        │
│   Gradient Boosted Classifier: p_hybrid = Sigmoid(XGBoost(H))                    │
│   Output: Risk Score (0-100%), Risk Class, TreeSHAP Biomarker Breakdown          │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Experimental Results & Benchmarks

Tested on **13,329 unseen test patients** (drawn from 66,641 full-scale dataset) across 20 research phases:

| Model Architecture | Phase | ROC-AUC | Sensitivity | Specificity | F1 Score | Key Architectural Insight |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Classical XGBoost** | Phase 11 | **0.8567** | 75.8% | **82.2%** | 0.781 | Gold-standard classical benchmark |
| **Quantum Residual Corrector** | Phase 13 | **0.8565** | 75.8% | 82.1% | 0.781 | **Closed ROC-AUC gap to 0.0002 of XGBoost** |
| **Production Hybrid (Deployed)** | Phase 14 | **0.8369** | **76.8%** | 81.2% | **0.784** | **Higher sensitivity than classical!** |
| **MI-Adaptive QSVM** | Phase 8a | 0.7666 | 71.2% | 73.4% | 0.721 | Best Quantum Kernel variant |
| **Variational Circuit (VQC)** | Phase 8b | 0.6072 | 50.5% | 68.3% | 0.584 | Shot-noise limited on 512 shots |

---

## 🖥️ Web Dashboard Overview

The platform features an 8-tab production Next.js terminal:

| Tab | Functionality |
| :--- | :--- |
| 🏠 **Home** | Interactive telemetry hero showcase with live profile tabs (Healthy, Moderate, Elevated), Web Audio ECG pulse test, and 6 architectural pillars. |
| 🎯 **Prediction** | Single-patient risk terminal with model tournament selector, dynamic threshold slider ($T$), counterfactual simulator, and **Clinical PDF Exporter**. |
| 📊 **Dataset** | CSV file uploader with automated schema validation, missing data imputation, and batch inference. |
| 💡 **Explainability** | TreeSHAP feature attribution charts and patient-level biomarker contribution breakdowns. |
| 🏆 **Benchmarks** | Performance metric comparisons across classical, QSVM, VQC, and hybrid architectures. |
| 🔬 **Quantum Lab** | Interactive 4-qubit circuit visualizer with Bloch sphere state vectors and expectation value readouts. |
| ⚡ **IBM Hardware** | Execution proofs and noise telemetry from physical QPU jobs on `ibm_marrakesh`. |
| 📚 **Research** | Comprehensive chronological timeline of all 20 research development phases. |

---

## 🔌 REST API Reference (FastAPI Backend)

The FastAPI backend server runs locally on `http://localhost:8000`:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | Health check & model status |
| `/api/predict` | `POST` | Single patient risk stratification |
| `/api/predict/explain` | `POST` | Patient risk prediction + TreeSHAP attribution |
| `/api/predict/fhir` | `POST` | HL7 FHIR JSON Bundle parser & prediction |
| `/api/predict/ood-check` | `POST` | Quantum Hilbert space OOD safety guard |
| `/api/predict/batch` | `POST` | Batch dataset predictions |
| `/api/hardware/job` | `GET` | IBM Quantum Hardware job execution telemetry |
| `/api/metrics` | `GET` | Model evaluation benchmarks |

---

## ⚙️ Quick Start Guide

### Prerequisites
* **Python 3.11+**
* **Node.js 18+** & `npm`

### 1. Backend Setup (FastAPI)
```bash
# Clone the repository
git clone https://github.com/your-username/QuantumDx.git
cd QuantumDx

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Start FastAPI server
uvicorn backend.app.main:app --reload --port 8000
```

### 2. Frontend Setup (Next.js)
```bash
# Navigate to frontend
cd frontend

# Install Node dependencies
npm install

# Start Next.js dev server
npm run dev
```
Open **http://localhost:3000** in your browser.

### 3. Running Automated Tests
```bash
# From project root directory
pytest tests/
```
*Executes all **444 passing system tests**.*

---

## 📚 Academic References

1. **XGBoost:** Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016.
2. **Quantum Kernels:** Havlíček, V., et al. (2019). *Supervised learning with quantum-enhanced feature spaces*. *Nature*, 567(7747), 209-212.
3. **SHAP Explainability:** Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. NeurIPS 2017.
4. **QML Reality Check in Health:** Gupta, A., et al. (2025). *Empirical assessment of quantum machine learning in digital health*. *npj Digital Medicine*.

---

## 📜 License & Medical Disclaimer

This repository is shared as a research prototype and decision-support terminal developed for the **Smart India Hackathon 2026 (SIH26139)**. It is **not** a certified medical diagnostic device and must not be used as a primary diagnostic tool in clinical practice.
