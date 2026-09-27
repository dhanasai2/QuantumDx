"use client";

import React, { useEffect, useState } from "react";
import {
  CheckCircle2,
  Clock,
  FileText,
  GitBranch,
  HelpCircle,
  Info,
  Layers,
  ShieldAlert,
  Sparkles,
  Zap
} from "lucide-react";

export default function ResearchView() {
  const [objectives, setObjectives] = useState<any[]>([]);
  const [phases, setPhases] = useState<any[]>([]);

  useEffect(() => {
    fetch("http://localhost:8000/api/research/objectives")
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => setObjectives(data))
      .catch(() => setObjectives([]));

    fetch("http://localhost:8000/api/research/phases")
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => setPhases(data))
      .catch(() => setPhases([]));
  }, []);

  return (
    <div className="space-y-8 py-4">
      {/* Top Banner */}
      <div className="p-6 rounded-2xl bg-white border border-slate-200 shadow-sm space-y-1">
        <h1 className="text-2xl font-extrabold text-slate-900 flex items-center gap-2">
          <FileText className="w-6 h-6 text-purple-600" />
          <span>Research Methodology & 16-Phase Timeline</span>
        </h1>
        <p className="text-xs text-slate-500">
          Complete documentation of SIH26139 objectives status, 16 research phase outcomes, and peer-reviewed evidence.
        </p>
      </div>

      {/* SIH Objectives Mapping Section */}
      <div className="p-6 rounded-2xl bg-white border border-slate-200 shadow-sm space-y-4">
        <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2 border-b border-slate-100 pb-3">
          <Zap className="w-5 h-5 text-blue-600" />
          <span>SIH26139 Objectives Traceability Matrix</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {objectives.map((obj) => (
            <div key={obj.id} className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-mono font-bold text-blue-600">{obj.id}</span>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                    obj.badge === "success"
                      ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                      : obj.badge === "warning"
                      ? "bg-amber-50 text-amber-700 border border-amber-200"
                      : "bg-rose-50 text-rose-700 border border-rose-200"
                  }`}
                >
                  {obj.status}
                </span>
              </div>
              <p className="font-bold text-slate-900">{obj.objective}</p>
              <p className="text-slate-600 leading-relaxed text-[11px]">{obj.details}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 16-Phase Experimental Timeline */}
      <div className="p-6 rounded-2xl bg-white border border-slate-200 shadow-sm space-y-4">
        <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2 border-b border-slate-100 pb-3">
          <Clock className="w-5 h-5 text-purple-600" />
          <span>16-Phase Research Trajectory</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
          {phases.map((p) => (
            <div key={p.phase} className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
              <div className="flex items-center justify-between font-mono">
                <span className="text-purple-600 font-bold">Phase {p.phase}</span>
                <span className="text-[10px] text-slate-400 font-semibold">Completed</span>
              </div>
              <div className="font-bold text-slate-900">{p.title}</div>
              <p className="text-slate-600 text-[11px] leading-relaxed">{p.summary}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Primary Academic Citations Box */}
      <div className="p-6 rounded-2xl bg-white border border-slate-200 shadow-sm space-y-3 text-xs">
        <h3 className="font-bold text-slate-900 text-sm">Key Literature References</h3>
        <div className="space-y-2 text-slate-700">
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
            <span className="font-bold text-blue-600">[P10] Gupta et al. (2025)</span> — <em>A systematic review of quantum machine learning for digital health</em>, npj Digital Medicine 8:237.
          </div>
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
            <span className="font-bold text-blue-600">[P5] Setiawan et al. (2026)</span> — <em>Systematic review of QML for medical: trends, datasets, and methods</em>, Int. J. Cogn. Comput. Eng. 7:609-630.
          </div>
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
            <span className="font-bold text-blue-600">[P7] Zorlu & Colak (2026)</span> — <em>Robustness-Oriented Hybrid Pipeline for Breast Cancer Diagnosis</em>, Diagnostics 16:1996.
          </div>
        </div>
      </div>
    </div>
  );
}

