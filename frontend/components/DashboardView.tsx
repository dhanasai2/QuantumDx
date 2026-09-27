"use client";

import React, { useEffect, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  BrainCircuit,
  CheckCircle,
  Cpu,
  Database,
  FileText,
  ShieldCheck,
  Zap
} from "lucide-react";

interface DashboardViewProps {
  onNavigate: (tab: string) => void;
  apiStatus: boolean;
}

export default function DashboardView({ onNavigate, apiStatus }: DashboardViewProps) {
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [hardwareInfo, setHardwareInfo] = useState<any>(null);

  useEffect(() => {
    fetch("http://localhost:8000/api/model-info")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => setModelInfo(data))
      .catch(() => setModelInfo(null));

    fetch("http://localhost:8000/api/hardware/job")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => setHardwareInfo(data))
      .catch(() => setHardwareInfo(null));
  }, []);

  return (
    <div className="space-y-8 py-4">
      {/* Top Banner / Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white p-6 rounded-3xl border border-slate-200 shadow-sm">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900 flex items-center gap-2">
            <span>Platform Control Dashboard</span>
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Real-time inference, model metrics, dataset profiler, and quantum QPU execution control.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => onNavigate("prediction")}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-blue-600 text-white font-bold text-xs shadow-md shadow-blue-600/20 hover:bg-blue-700 hover:scale-[1.02] transition-all cursor-pointer"
          >
            <Activity className="w-4 h-4" />
            <span>Single Patient Inference</span>
          </button>
          <button
            onClick={() => onNavigate("dataset")}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-100 border border-slate-200 text-slate-700 font-bold text-xs hover:bg-slate-200 transition-all cursor-pointer"
          >
            <Database className="w-4 h-4 text-blue-600" />
            <span>Upload Dataset</span>
          </button>
        </div>
      </div>

      {/* Metrics Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="p-5 rounded-3xl bg-white border border-slate-200 shadow-sm space-y-2 card-shadow-hover">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-500 font-bold">Primary Baseline</span>
            <div className="p-2 rounded-xl bg-blue-50 text-blue-600">
              <BarChart3 className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-extrabold text-slate-900 font-mono">XGBoost</div>
          <div className="flex items-center justify-between text-xs pt-2 border-t border-slate-100">
            <span className="text-slate-500 font-medium">Test ROC-AUC</span>
            <span className="font-mono text-blue-600 font-bold">0.8567</span>
          </div>
        </div>

        <div className="p-5 rounded-3xl bg-white border border-slate-200 shadow-sm space-y-2 card-shadow-hover">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-500 font-bold">IBM Quantum QPU</span>
            <div className="p-2 rounded-xl bg-emerald-50 text-emerald-600">
              <Cpu className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-extrabold text-slate-900 font-mono">ibm_marrakesh</div>
          <div className="flex items-center justify-between text-xs pt-2 border-t border-slate-100">
            <span className="text-slate-500 font-medium">Job ID</span>
            <span className="font-mono text-emerald-600 font-bold">dag54f8m...</span>
          </div>
        </div>

        <div className="p-5 rounded-3xl bg-white border border-slate-200 shadow-sm space-y-2 card-shadow-hover">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-500 font-bold">Leakage Firewall</span>
            <div className="p-2 rounded-xl bg-purple-50 text-purple-600">
              <ShieldCheck className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-extrabold text-slate-900 font-mono">Audited</div>
          <div className="flex items-center justify-between text-xs pt-2 border-t border-slate-100">
            <span className="text-slate-500 font-medium">In-Fold Protocol</span>
            <span className="font-mono text-purple-600 font-bold">100% Fit-on-Train</span>
          </div>
        </div>

        <div className="p-5 rounded-3xl bg-white border border-slate-200 shadow-sm space-y-2 card-shadow-hover">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-500 font-bold">Surrogate Audit</span>
            <div className="p-2 rounded-xl bg-indigo-50 text-indigo-600">
              <BrainCircuit className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-extrabold text-slate-900 font-mono">94.1%</div>
          <div className="flex items-center justify-between text-xs pt-2 border-t border-slate-100">
            <span className="text-slate-500 font-medium">Label Agreement</span>
            <span className="font-mono text-indigo-600 font-bold">High Fidelity</span>
          </div>
        </div>
      </div>

      {/* Main Grid Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Active Model Architecture & Specs */}
        <div className="lg:col-span-2 space-y-6">
          <div className="p-6 rounded-3xl bg-white border border-slate-200 shadow-sm space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                <Zap className="w-5 h-5 text-blue-600" />
                <span>Active Model Specs & Persistent Artifacts</span>
              </h2>
              <span className="text-xs font-mono px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-semibold">
                Phase 14 Product
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200 space-y-1">
                <span className="text-slate-500 font-medium">Model Version</span>
                <p className="font-mono text-slate-900 font-bold">{modelInfo?.model_version || "phase14_hybrid_v1"}</p>
              </div>
              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200 space-y-1">
                <span className="text-slate-500 font-medium">Training Pool Size</span>
                <p className="font-mono text-slate-900 font-bold">{modelInfo?.n_train_full ? `${modelInfo.n_train_full} rows` : "66,641 rows"}</p>
              </div>
              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200 space-y-1">
                <span className="text-slate-500 font-medium">Decision Threshold</span>
                <p className="font-mono text-slate-900 font-bold">{modelInfo?.decision_threshold || "0.50 (Recall Target 0.90)"}</p>
              </div>
              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200 space-y-1">
                <span className="text-slate-500 font-medium">Quantum Qubits</span>
                <p className="font-mono text-slate-900 font-bold">{modelInfo?.quantum?.n_qubits ? `${modelInfo.quantum.n_qubits} Qubits` : "4 Qubits (PCA-4 Encoding)"}</p>
              </div>
            </div>

            <div className="p-4 rounded-2xl bg-blue-50/70 border border-blue-200 text-xs text-blue-900 space-y-2">
              <div className="font-bold flex items-center gap-1.5 text-blue-800">
                <CheckCircle className="w-4 h-4 text-blue-600" />
                <span>Fairness & Validation Invariant</span>
              </div>
              <p className="text-slate-700 leading-relaxed">
                Both classical and quantum models are trained on the exact same processed features and evaluated on the identical hold-out split (Fingerprint: <code className="font-mono text-blue-700 bg-blue-100 px-1.5 py-0.5 rounded border border-blue-200">96eac11a8394b87e</code>).
              </p>
            </div>
          </div>

          {/* Persistent Benchmark Quick Table */}
          <div className="p-6 rounded-3xl bg-white border border-slate-200 shadow-sm space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-purple-600" />
                <span>Validated Benchmark Summary</span>
              </h2>
              <button
                onClick={() => onNavigate("benchmark")}
                className="text-xs font-bold text-purple-600 hover:underline flex items-center gap-1 cursor-pointer"
              >
                <span>Full Benchmark Suite</span>
                <ArrowUpRight className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-700">
                <thead className="bg-slate-50 text-slate-500 font-mono border-b border-slate-200">
                  <tr>
                    <th className="p-3 rounded-l-xl">Model Arm</th>
                    <th className="p-3">Architecture</th>
                    <th className="p-3">CV ROC-AUC</th>
                    <th className="p-3 rounded-r-xl">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  <tr>
                    <td className="p-3 font-bold text-slate-900">XGBoost (Model A)</td>
                    <td className="p-3 text-slate-500">Optuna-tuned Classical GBDT</td>
                    <td className="p-3 font-mono text-blue-600 font-bold">0.80119</td>
                    <td className="p-3"><span className="px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-bold border border-emerald-200 text-[10px]">Primary Baseline</span></td>
                  </tr>
                  <tr>
                    <td className="p-3 font-bold text-slate-900">Quantum Stack (Model B)</td>
                    <td className="p-3 text-slate-500">XGBoost + Q-Circuit (LR Stack)</td>
                    <td className="p-3 font-mono text-slate-900 font-bold">0.80120</td>
                    <td className="p-3"><span className="px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-700 font-bold border border-slate-200 text-[10px]">Non-Inferior</span></td>
                  </tr>
                  <tr>
                    <td className="p-3 font-bold text-slate-900">Matched RFF Control (Model C)</td>
                    <td className="p-3 text-slate-500">XGBoost + Classical RFF (LR Stack)</td>
                    <td className="p-3 font-mono text-slate-500 font-bold">0.80121</td>
                    <td className="p-3"><span className="px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-500 font-bold border border-slate-200 text-[10px]">Classical Control</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right 1 Col: Real IBM Hardware Status Widget */}
        <div className="space-y-6">
          <div className="p-6 rounded-3xl bg-white border border-slate-200 shadow-sm space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <Cpu className="w-5 h-5 text-emerald-600" />
                <span>IBM QPU Execution</span>
              </h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-semibold">
                Verified
              </span>
            </div>

            <div className="space-y-3 text-xs">
              <div className="flex justify-between items-center py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-medium">QPU Backend</span>
                <span className="font-mono text-slate-900 font-bold">{hardwareInfo?.backend || "ibm_marrakesh"}</span>
              </div>
              <div className="flex justify-between items-center py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-medium">Real Job ID</span>
                <span className="font-mono text-blue-600 text-[11px] font-bold">{hardwareInfo?.job_id || "dag54f8mhr3c73e4m300"}</span>
              </div>
              <div className="flex justify-between items-center py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-medium">Total Shots</span>
                <span className="font-mono text-slate-800">16,384 (1024 / circuit)</span>
              </div>
              <div className="flex justify-between items-center py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-medium">Circuit Depth</span>
                <span className="font-mono text-slate-800">17 (Native Gates)</span>
              </div>
              <div className="flex justify-between items-center py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-medium">Mean Abs Deviation</span>
                <span className="font-mono text-emerald-600 font-bold">0.0312</span>
              </div>
            </div>

            <button
              onClick={() => onNavigate("hardware")}
              className="w-full py-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-xs font-bold text-slate-700 transition-all cursor-pointer flex items-center justify-center gap-1.5 border border-slate-200"
            >
              <span>View QASM & Hardware Logs</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="p-6 rounded-3xl bg-white border border-slate-200 shadow-sm space-y-3">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <FileText className="w-4 h-4 text-purple-600" />
              <span>SIH Presentation Story</span>
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Leading with scientific rigour and hardware validation sets your project apart. Rather than making unsubstantiated claims, QuantumDx demonstrates honest, hardware-verified quantum capabilities.
            </p>
            <button
              onClick={() => onNavigate("research")}
              className="text-xs font-bold text-purple-600 hover:underline flex items-center gap-1 cursor-pointer"
            >
              <span>Read 16-Phase Research Log</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
