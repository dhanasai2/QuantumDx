"use client";

import React from "react";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  Cpu,
  Database,
  FileText,
  Home,
  LayoutDashboard,
  Zap
} from "lucide-react";

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  apiStatus: boolean;
}

export default function Navbar({ activeTab, setActiveTab, apiStatus }: NavbarProps) {
  const navItems = [
    { id: "landing", label: "Home", icon: Home },
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "dataset", label: "Datasets", icon: Database },
    { id: "prediction", label: "Prediction", icon: Activity },
    { id: "explainability", label: "Explainability", icon: BrainCircuit },
    { id: "benchmark", label: "Benchmarks", icon: BarChart3 },
    { id: "quantum-lab", label: "Quantum Lab", icon: Zap },
    { id: "hardware", label: "IBM QPU", icon: Cpu },
    { id: "research", label: "Research", icon: FileText },
  ];

  return (
    <header className="sticky top-0 z-50 bg-white/95 backdrop-blur-xl border-b border-slate-200/90 px-4 lg:px-8 py-3.5 shadow-xs">
      <div className="max-w-7xl mx-auto flex flex-col lg:flex-row items-center justify-between gap-4">
        {/* Logo & Brand */}
        <div
          className="flex items-center gap-3 cursor-pointer group shrink-0"
          onClick={() => setActiveTab("landing")}
        >
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-700 via-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-blue-500/25 group-hover:scale-105 transition-all">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-xl tracking-tight text-slate-900">
                Quantum<span className="text-blue-600">Dx</span>
              </span>
              <span className="text-[10px] font-extrabold px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200/80 shadow-2xs">
                SIH2026
              </span>
            </div>
            <p className="text-[11px] text-slate-500 hidden sm:block font-medium">
              Hybrid Quantum-Classical Biomedical Intelligence
            </p>
          </div>
        </div>

        {/* Navigation Items (Interactive Glass Pill Bar) */}
        <nav className="flex items-center gap-1 overflow-x-auto max-w-full p-1.5 bg-slate-100/90 rounded-2xl border border-slate-200/90 no-scrollbar shadow-inner">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold transition-all duration-200 whitespace-nowrap rounded-xl cursor-pointer ${
                  isActive
                    ? "bg-white text-blue-600 shadow-sm border border-slate-200/80"
                    : "text-slate-600 hover:text-blue-600 hover:bg-white/60"
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? "text-blue-600" : "text-slate-400"}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* API Connection Indicator */}
        <div className="hidden lg:flex items-center gap-2 shrink-0">
          <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-slate-50 border border-slate-200 text-xs font-bold shadow-2xs">
            <span className={`w-2 h-2 rounded-full ${apiStatus ? "bg-emerald-500 animate-pulse" : "bg-amber-500"}`} />
            <span className="text-slate-800">{apiStatus ? "API Ready" : "Local Mode"}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
