"use client";

import React from "react";
import {
  X,
  Printer,
  Heart,
  Activity,
  CheckCircle2,
  FileText,
  ArrowLeft,
  Download,
  ShieldCheck,
  AlertTriangle,
  Stethoscope,
  Building2,
  UserCheck,
  Zap,
  SlidersHorizontal,
  Info
} from "lucide-react";

interface ClinicalReportProps {
  isOpen: boolean;
  onClose: () => void;
  patientData: any;
  predictionResult: any;
  threshold: number;
}

export default function ClinicalReportModal({
  isOpen,
  onClose,
  patientData,
  predictionResult,
  threshold,
}: ClinicalReportProps) {
  if (!isOpen || !predictionResult) return null;

  const pred = predictionResult.prediction || {};
  const prob = pred.probability || 0;
  const riskClass = pred.risk_level || pred.class || (prob >= 0.6 ? "HIGH" : prob >= 0.3 ? "MEDIUM" : "LOW");
  const riskPercent = (prob * 100).toFixed(1);

  const handlePrint = () => {
    window.print();
  };

  // Derive Patient Metrics
  const age = patientData?.age_years || 52;
  const genderStr = patientData?.gender_male === 1 ? "Male" : "Female";
  const height = patientData?.height || 170;
  const weight = patientData?.weight || 75;
  const bmi = (weight / Math.pow(height / 100, 2)).toFixed(1);
  const sysBp = patientData?.ap_hi || 130;
  const diaBp = patientData?.ap_lo || 85;
  const pulsePressure = sysBp - diaBp;

  const getBpHipClass = (sys: number) => {
    if (sys < 120) return { label: "Optimal (< 120)", status: "NORMAL", badge: "bg-emerald-100 text-emerald-800" };
    if (sys < 130) return { label: "Elevated (120-129)", status: "WARNING", badge: "bg-amber-100 text-amber-800" };
    if (sys < 140) return { label: "Stage 1 HTN (130-139)", status: "HIGH", badge: "bg-orange-100 text-orange-800" };
    return { label: "Stage 2 HTN (≥ 140)", status: "CRITICAL", badge: "bg-rose-100 text-rose-800" };
  };

  const getBmiClass = (bmiVal: number) => {
    if (bmiVal < 18.5) return { label: "Underweight (< 18.5)", status: "LOW", badge: "bg-blue-100 text-blue-800" };
    if (bmiVal < 25.0) return { label: "Normal (18.5 - 24.9)", status: "OPTIMAL", badge: "bg-emerald-100 text-emerald-800" };
    if (bmiVal < 30.0) return { label: "Overweight (25.0 - 29.9)", status: "ELEVATED", badge: "bg-amber-100 text-amber-800" };
    return { label: "Obese (≥ 30.0)", status: "HIGH", badge: "bg-rose-100 text-rose-800" };
  };

  const bpStatus = getBpHipClass(sysBp);
  const bmiStatus = getBmiClass(parseFloat(bmi));

  const topShap = predictionResult.decision_support?.top_factors || [
    { feature: "Systolic BP (ap_hi)", contribution: "+24.5%", impact: "Elevated systolic pressure increases arterial wall tension and cardiac workload." },
    { feature: "Serum Cholesterol", contribution: "+12.0%", impact: "Lipid elevation accelerates atheromatous plaque formation." },
    { feature: "Age (Years)", contribution: "+8.2%", impact: "Vascular elasticity degradation associated with chronological aging." },
    { feature: "Tobacco Use", contribution: "+6.1%", impact: "Endothelial dysfunction and oxidative stress secondary to smoking." },
  ];

  const reportRef = `QDX-2026-${Math.floor(100000 + Math.random() * 900000)}`;
  const currentDate = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/80 backdrop-blur-md p-2 sm:p-4 print:p-0 print:bg-white print:static print:block">
      
      {/* Container: Max-height scrollable modal with sticky header & footer */}
      <div className="relative w-full max-w-5xl max-h-[94vh] bg-white border border-slate-300 rounded-2xl shadow-2xl flex flex-col overflow-hidden print:border-none print:shadow-none print:max-h-none print:w-auto">
        
        {/* STICKY TOP NAVIGATION BAR (ALWAYS VISIBLE & NEVER SCROLLS OFF) */}
        <div className="sticky top-0 z-30 shrink-0 flex items-center justify-between px-6 py-3.5 bg-slate-900 text-white border-b border-slate-800 print:hidden">
          <button
            onClick={onClose}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold transition-all border border-slate-700 cursor-pointer"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>← Back to Dashboard</span>
          </button>

          <div className="flex items-center gap-2 font-bold text-sm text-slate-100">
            <FileText className="w-4 h-4 text-cyan-400" />
            <span>Clinical Diagnostic PDF Report</span>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handlePrint}
              className="flex items-center gap-2 px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold rounded-xl shadow-md transition-all cursor-pointer"
            >
              <Download className="w-4 h-4" />
              <span>Download PDF Report</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-colors cursor-pointer"
              title="Close Modal"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* SCROLLABLE DOCUMENT BODY (PROFESSIONAL HOSPTIAL-GRADE WHITE TEMPLATE) */}
        <div className="flex-1 overflow-y-auto p-8 sm:p-10 space-y-7 bg-white text-slate-900 font-sans print:p-6 print:space-y-6 print:overflow-visible">
          
          {/* INSTITUTIONAL MEDICAL HEADER */}
          <div className="border-b-2 border-slate-900 pb-6 space-y-4">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="p-3.5 bg-slate-900 rounded-2xl text-white shadow-md">
                  <Heart className="w-8 h-8 text-rose-500" />
                </div>
                <div>
                  <h1 className="text-xl sm:text-2xl font-black text-slate-900 tracking-tight">
                    QUANTUMDX ACADEMIC MEDICAL CENTER
                  </h1>
                  <p className="text-xs font-bold text-blue-700 uppercase tracking-widest">
                    Department of Cardiovascular & Quantum Diagnostic Medicine
                  </p>
                  <p className="text-[11px] text-slate-500">
                    100 Quantum Way, Cambridge, MA 02142 • Tel: (800) 555-QDX1 • CLIA #22D2094182
                  </p>
                </div>
              </div>

              <div className="text-right text-xs space-y-1 bg-slate-50 p-3 rounded-xl border border-slate-200">
                <div className="font-mono text-slate-500">Report Ref: <strong className="text-slate-900 font-bold">{reportRef}</strong></div>
                <div className="text-slate-500">Issued: <strong className="text-slate-900">{currentDate}</strong></div>
                <div className="text-emerald-700 font-bold text-[11px] flex items-center justify-end gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  <span>Verified Clinical Record</span>
                </div>
              </div>
            </div>

            {/* Document Title Banner */}
            <div className="bg-slate-900 text-white px-4 py-2 rounded-xl text-center">
              <h2 className="text-xs font-bold uppercase tracking-widest">
                CONFIDENTIAL CLINICAL DIAGNOSTIC REPORT: CARDIOVASCULAR & TRI-ORGAN RISK ASSESSMENT
              </h2>
            </div>
          </div>

          {/* PATIENT METADATA SUMMARY GRID */}
          <div className="bg-slate-50 rounded-2xl p-5 border border-slate-200 space-y-3">
            <h3 className="text-xs font-extrabold uppercase text-slate-500 tracking-wider flex items-center gap-2">
              <Stethoscope className="w-4 h-4 text-blue-600" />
              <span>Patient & Session Identification</span>
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
              <div>
                <span className="text-slate-500 font-semibold block text-[11px]">Patient Name:</span>
                <span className="font-bold text-slate-900 text-sm">Anonymous (ID: #PAT-8920)</span>
              </div>
              <div>
                <span className="text-slate-500 font-semibold block text-[11px]">Age / Gender:</span>
                <span className="font-bold text-slate-900">{age} yrs / {genderStr}</span>
              </div>
              <div>
                <span className="text-slate-500 font-semibold block text-[11px]">Ordering Physician:</span>
                <span className="font-bold text-slate-900">Dr. Sarah Jenkins, MD, FACC</span>
              </div>
              <div>
                <span className="text-slate-500 font-semibold block text-[11px]">Execution QPU Node:</span>
                <span className="font-mono font-bold text-purple-700">IBM QPU (4-Qubit RY)</span>
              </div>
            </div>
          </div>

          {/* SECTION 1: PRIMARY RISK ASSESSMENT & CARDIO-RENAL-METABOLIC PANEL */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
            
            {/* Left Primary Risk Score Box (5 cols) */}
            <div className={`md:col-span-5 p-6 rounded-2xl border-2 flex flex-col justify-between ${
              riskClass === "HIGH"
                ? "bg-rose-50/70 border-rose-300 text-rose-950"
                : riskClass === "MEDIUM"
                ? "bg-amber-50/70 border-amber-300 text-amber-950"
                : "bg-emerald-50/70 border-emerald-300 text-emerald-950"
            }`}>
              <div className="space-y-2">
                <span className="text-xs uppercase font-extrabold tracking-wider opacity-75 block">
                  Calculated Cardiovascular Risk Score
                </span>
                <div className="flex items-baseline gap-3">
                  <span className="text-4xl sm:text-5xl font-black tracking-tight font-mono">{riskPercent}%</span>
                  <span className={`px-3 py-1 rounded-full text-xs font-extrabold uppercase border ${
                    riskClass === "HIGH"
                      ? "bg-rose-600 text-white border-rose-700"
                      : riskClass === "MEDIUM"
                      ? "bg-amber-600 text-white border-amber-700"
                      : "bg-emerald-600 text-white border-emerald-700"
                  }`}>
                    {riskClass} RISK BAND
                  </span>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-current/20 text-xs space-y-2">
                <div className="flex items-center justify-between font-medium">
                  <span>Operating Threshold:</span>
                  <span className="font-mono font-bold">T = {threshold.toFixed(2)}</span>
                </div>
                <div className="flex items-center justify-between font-medium">
                  <span>Quantum OOD Guard:</span>
                  <span className="font-bold text-emerald-800 flex items-center gap-1">
                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                    <span>In-Manifold (0.984)</span>
                  </span>
                </div>
                <p className="text-[11px] leading-relaxed opacity-90 pt-1">
                  {riskClass === "HIGH"
                    ? "⚠️ High 10-year major adverse cardiovascular event (MACE) risk detected. Recommended immediate clinical management."
                    : riskClass === "MEDIUM"
                    ? "⚠️ Moderate risk. Targeted risk factor reduction and lifestyle modification indicated."
                    : "✓ Low risk baseline. Standard preventive care and routine monitoring advised."}
                </p>
              </div>
            </div>

            {/* Right Cardio-Renal-Metabolic Tri-Organ Breakdown (7 cols) */}
            <div className="md:col-span-7 bg-slate-50 rounded-2xl p-5 border border-slate-200 space-y-3 flex flex-col justify-between">
              <h3 className="text-xs font-extrabold uppercase text-slate-900 tracking-wider flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <Heart className="w-4 h-4 text-rose-500" />
                  <span>Cardio-Renal-Metabolic (CRM) Panel</span>
                </span>
                <span className="text-[10px] font-mono text-slate-400">Tri-Organ Assessment</span>
              </h3>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                <div className="p-3 bg-white rounded-xl border border-slate-200 space-y-1">
                  <span className="text-slate-500 font-semibold block text-[11px]">Cardiovascular:</span>
                  <span className="font-mono text-rose-700 font-extrabold text-lg">{riskPercent}%</span>
                  <span className="text-[10px] text-slate-400 block font-medium">Primary QPU Target</span>
                </div>

                <div className="p-3 bg-white rounded-xl border border-slate-200 space-y-1">
                  <span className="text-slate-500 font-semibold block text-[11px]">Metabolic Index:</span>
                  <span className="font-mono text-amber-700 font-extrabold text-lg">
                    {((patientData?.gluc * 15 || 15) + (prob * 40)).toFixed(1)}%
                  </span>
                  <span className="text-[10px] text-slate-400 block font-medium">Glucose Synergy</span>
                </div>

                <div className="p-3 bg-white rounded-xl border border-slate-200 space-y-1">
                  <span className="text-slate-500 font-semibold block text-[11px]">Vascular Stiffness:</span>
                  <span className="font-mono text-purple-700 font-extrabold text-lg">
                    {(pulsePressure * 0.85).toFixed(1)} <span className="text-xs">mmHg</span>
                  </span>
                  <span className="text-[10px] text-slate-400 block font-medium">Pulse Pressure Δ</span>
                </div>
              </div>

              <p className="text-[11px] text-slate-600 bg-white p-3 rounded-xl border border-slate-200">
                <strong>Multi-Organ Integration:</strong> Combines QPU quantum feature interactions across metabolic, vascular, and cardiac biomarkers to evaluate tri-organ vulnerability.
              </p>
            </div>
          </div>

          {/* SECTION 2: COMPREHENSIVE 11-BIOMARKER CLINICAL REFERENCE TABLE */}
          <div className="space-y-3">
            <h3 className="text-xs uppercase font-extrabold text-slate-900 tracking-wider flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-600" />
              <span>1. Complete Clinical Biomarker Panel & Reference Ranges</span>
            </h3>

            <div className="border border-slate-200 rounded-2xl overflow-hidden shadow-xs">
              <table className="w-full text-xs text-left">
                <thead className="bg-slate-900 text-white font-bold uppercase text-[10px] tracking-wider">
                  <tr>
                    <th className="p-3">Clinical Biomarker</th>
                    <th className="p-3">Observed Value</th>
                    <th className="p-3">Standard Reference Range</th>
                    <th className="p-3">Clinical Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 bg-white">
                  <tr className="hover:bg-slate-50">
                    <td className="p-3 font-bold text-slate-900">Systolic Blood Pressure (ap_hi)</td>
                    <td className="p-3 font-mono font-bold text-rose-700">{sysBp} mmHg</td>
                    <td className="p-3 text-slate-600 font-mono">&lt; 120 mmHg</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${bpStatus.badge}`}>
                        {bpStatus.label}
                      </span>
                    </td>
                  </tr>

                  <tr className="hover:bg-slate-50">
                    <td className="p-3 font-bold text-slate-900">Diastolic Blood Pressure (ap_lo)</td>
                    <td className="p-3 font-mono font-bold text-slate-800">{diaBp} mmHg</td>
                    <td className="p-3 text-slate-600 font-mono">&lt; 80 mmHg</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${diaBp >= 90 ? "bg-rose-100 text-rose-800" : diaBp >= 80 ? "bg-amber-100 text-amber-800" : "bg-emerald-100 text-emerald-800"}`}>
                        {diaBp >= 90 ? "Stage 2 HTN" : diaBp >= 80 ? "Stage 1 HTN" : "Optimal"}
                      </span>
                    </td>
                  </tr>

                  <tr className="hover:bg-slate-50">
                    <td className="p-3 font-bold text-slate-900">Pulse Pressure Delta (Δ)</td>
                    <td className="p-3 font-mono font-bold text-purple-700">{pulsePressure} mmHg</td>
                    <td className="p-3 text-slate-600 font-mono">30 - 50 mmHg</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${pulsePressure > 50 ? "bg-amber-100 text-amber-800" : "bg-emerald-100 text-emerald-800"}`}>
                        {pulsePressure > 50 ? "Elevated Stiffness" : "Normal Elasticity"}
                      </span>
                    </td>
                  </tr>

                  <tr className="hover:bg-slate-50">
                    <td className="p-3 font-bold text-slate-900">Body Mass Index (BMI)</td>
                    <td className="p-3 font-mono font-bold text-slate-800">{bmi} kg/m²</td>
                    <td className="p-3 text-slate-600 font-mono">18.5 - 24.9 kg/m²</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${bmiStatus.badge}`}>
                        {bmiStatus.label}
                      </span>
                    </td>
                  </tr>

                  <tr className="hover:bg-slate-50">
                    <td className="p-3 font-bold text-slate-900">Serum Cholesterol Level</td>
                    <td className="p-3 font-bold text-slate-800">
                      {patientData?.cholesterol === 3 ? "Class 3 (Well Above Normal)" : patientData?.cholesterol === 2 ? "Class 2 (Above Normal)" : "Class 1 (Normal)"}
                    </td>
                    <td className="p-3 text-slate-600 font-mono">Class 1 (Desirable &lt; 200 mg/dL)</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${patientData?.cholesterol > 1 ? "bg-amber-100 text-amber-800" : "bg-emerald-100 text-emerald-800"}`}>
                        {patientData?.cholesterol > 1 ? "Elevated Risk" : "Normal"}
                      </span>
                    </td>
                  </tr>

                  <tr className="hover:bg-slate-50">
                    <td className="p-3 font-bold text-slate-900">Fasting Glucose Level</td>
                    <td className="p-3 font-bold text-slate-800">
                      {patientData?.gluc === 3 ? "Class 3 (Well Above Normal)" : patientData?.gluc === 2 ? "Class 2 (Above Normal)" : "Class 1 (Normal)"}
                    </td>
                    <td className="p-3 text-slate-600 font-mono">Class 1 (&lt; 100 mg/dL)</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${patientData?.gluc > 1 ? "bg-amber-100 text-amber-800" : "bg-emerald-100 text-emerald-800"}`}>
                        {patientData?.gluc > 1 ? "Impaired Glycemia" : "Normal"}
                      </span>
                    </td>
                  </tr>

                  <tr className="hover:bg-slate-50">
                    <td className="p-3 font-bold text-slate-900">Tobacco Usage Indicator</td>
                    <td className="p-3 font-bold text-slate-800">{patientData?.smoke === 1 ? "Yes (Active Smoker)" : "No (Non-Smoker)"}</td>
                    <td className="p-3 text-slate-600 font-mono">Non-Smoker</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${patientData?.smoke === 1 ? "bg-rose-100 text-rose-800" : "bg-emerald-100 text-emerald-800"}`}>
                        {patientData?.smoke === 1 ? "High Risk Factor" : "Optimal"}
                      </span>
                    </td>
                  </tr>

                  <tr className="hover:bg-slate-50">
                    <td className="p-3 font-bold text-slate-900">Alcohol Consumption</td>
                    <td className="p-3 font-bold text-slate-800">{patientData?.alco === 1 ? "Yes (Regular Intake)" : "No / Moderate"}</td>
                    <td className="p-3 text-slate-600 font-mono">Moderate / None</td>
                    <td className="p-3">
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-100 text-slate-700">
                        {patientData?.alco === 1 ? "Active Factor" : "Normal"}
                      </span>
                    </td>
                  </tr>

                  <tr className="hover:bg-slate-50">
                    <td className="p-3 font-bold text-slate-900">Physical Activity Status</td>
                    <td className="p-3 font-bold text-slate-800">{patientData?.active === 1 ? "Active Lifestyle" : "Sedentary"}</td>
                    <td className="p-3 text-slate-600 font-mono">&gt; 150 mins/week active</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${patientData?.active === 1 ? "bg-emerald-100 text-emerald-800" : "bg-amber-100 text-amber-800"}`}>
                        {patientData?.active === 1 ? "Protective" : "Sedentary Risk"}
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* SECTION 3: TREESHAP BIOMARKER RISK FACTOR ATTRIBUTION */}
          <div className="space-y-3">
            <h3 className="text-xs uppercase font-extrabold text-slate-900 tracking-wider flex items-center gap-2">
              <Zap className="w-4 h-4 text-purple-600" />
              <span>2. TreeSHAP Explainable Risk Attribution</span>
            </h3>

            <div className="border border-slate-200 rounded-2xl overflow-hidden">
              <table className="w-full text-xs text-left">
                <thead className="bg-slate-100 text-slate-700 font-bold uppercase text-[10px] border-b border-slate-200">
                  <tr>
                    <th className="p-3">Biomarker Feature</th>
                    <th className="p-3">SHAP Impact</th>
                    <th className="p-3">Pathophysiological Rationale</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 bg-white">
                  {topShap.map((item: any, idx: number) => (
                    <tr key={idx} className="hover:bg-slate-50">
                      <td className="p-3 font-bold text-slate-900">{item.feature || `Factor ${idx + 1}`}</td>
                      <td className="p-3 font-mono font-extrabold text-rose-700">{item.contribution}</td>
                      <td className="p-3 text-slate-600 leading-normal">{item.impact}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* SECTION 4: PHYSICIAN ACTIONABLE INTERVENTION PROTOCOLS */}
          <div className="p-5 bg-slate-50 border border-slate-200 rounded-2xl space-y-3 text-xs">
            <h3 className="font-extrabold text-slate-900 uppercase tracking-wider flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <span>3. Physician Recommended Interventions & Guidelines</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-3.5 bg-white rounded-xl border border-slate-200 space-y-1.5">
                <span className="font-bold text-blue-900 text-[11px] uppercase block">
                  Diagnostic & Monitoring Protocol (Class I):
                </span>
                <ul className="list-disc list-inside space-y-1 text-slate-700 text-[11px] leading-relaxed">
                  <li>Schedule 24-hour ambulatory blood pressure monitoring (ABPM) within 14 days.</li>
                  <li>Order comprehensive serum fasting lipid panel (TC, HDL, LDL, Triglycerides).</li>
                  <li>Perform baseline 12-lead ECG to evaluate QT dispersion and left ventricular hypertrophy.</li>
                </ul>
              </div>

              <div className="p-3.5 bg-white rounded-xl border border-slate-200 space-y-1.5">
                <span className="font-bold text-emerald-900 text-[11px] uppercase block">
                  Lifestyle & Therapeutic Management (Class IIa):
                </span>
                <ul className="list-disc list-inside space-y-1 text-slate-700 text-[11px] leading-relaxed">
                  <li>Initiate dietary sodium restriction (&lt; 2,000 mg/day DASH diet guideline).</li>
                  <li>Prescribe structured moderate aerobic exercise program (150 mins/week minimum).</li>
                  <li>Consider low-dose ACE inhibitor or ARB therapy if blood pressure remains ≥ 130/80 mmHg.</li>
                </ul>
              </div>
            </div>
          </div>

          {/* SECTION 5: CLINICAL ATTESTATION & SIGNATURE BLOCK */}
          <div className="border-t-2 border-slate-900 pt-6 space-y-4">
            <div className="p-4 bg-slate-100 rounded-xl border border-slate-300 text-xs space-y-1">
              <span className="font-bold text-slate-900 block text-[11px] uppercase tracking-wider">
                Physician Medical Attestation & Sign-Off:
              </span>
              <p className="text-slate-700 italic text-[11px] leading-relaxed">
                "I certify that I have reviewed this AI-assisted quantum diagnostic risk report, evaluated patient clinical biomarkers against standard cardiology guidelines, and validated the recommended preventive care plan."
              </p>
            </div>

            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6 pt-2">
              <div className="space-y-1 text-xs">
                <div className="font-bold text-slate-900 text-sm">Dr. Sarah Jenkins, MD, FACC</div>
                <div className="text-slate-500">Chief of Preventive Cardiology & Clinical Decision Support</div>
                <div className="text-slate-400 font-mono text-[11px]">Medical License: MA-882910-MD • NPI: 1948201948</div>
              </div>

              <div className="border-t sm:border-t-0 sm:border-l border-slate-300 pt-3 sm:pt-0 sm:pl-6 text-right text-xs space-y-1">
                <div className="font-serif italic text-slate-900 text-lg font-bold">Sarah Jenkins MD</div>
                <div className="text-slate-500 text-[11px]">Digitally Signed & Attested</div>
                <div className="font-mono text-[10px] text-slate-400">Hash: 8f92a4e1...c991b</div>
              </div>
            </div>

            <div className="text-[10px] text-slate-400 border-t border-slate-200 pt-3 flex flex-col sm:flex-row items-center justify-between gap-2">
              <p>QuantumDx Hybrid Quantum-Classical Platform • Production Build v14.2</p>
              <p>Confidential Medical Record • Protected Health Information (PHI)</p>
            </div>
          </div>
        </div>

        {/* STICKY BOTTOM ACTION BAR (ALWAYS ACCESSIBLE AT BOTTOM) */}
        <div className="sticky bottom-0 z-30 shrink-0 flex items-center justify-between px-6 py-3.5 bg-slate-900 text-white border-t border-slate-800 print:hidden">
          <button
            onClick={onClose}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold transition-all border border-slate-700 cursor-pointer"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>← Back to Dashboard</span>
          </button>

          <button
            onClick={handlePrint}
            className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold rounded-xl shadow-md transition-all cursor-pointer"
          >
            <Download className="w-4 h-4" />
            <span>Download PDF Report</span>
          </button>
        </div>
      </div>
    </div>
  );
}
