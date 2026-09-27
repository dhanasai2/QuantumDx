"use client";

import React, { useState } from "react";
import {
  Activity,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Database,
  FileCode,
  FileSpreadsheet,
  Heart,
  HeartPulse,
  Layers,
  Lock,
  Microscope,
  Printer,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Sparkles,
  Trophy,
  UserCheck,
  Volume2,
  VolumeX,
  Zap
} from "lucide-react";

interface LandingPageProps {
  onLaunch: (tab: string) => void;
}

export default function LandingPage({ onLaunch }: LandingPageProps) {
  const [selectedDemoPreset, setSelectedDemoPreset] = useState<"low" | "mid" | "high">("high");
  const [isAudioMuted, setIsAudioMuted] = useState<boolean>(true);

  // Live demo preset data
  const demoProfiles = {
    low: {
      name: "Healthy Baseline Profile",
      age: 36,
      bp: "118/76 mmHg",
      riskProb: 0.142,
      riskLevel: "LOW RISK",
      qExpectation: "<Z> = -0.320",
      badgeColor: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
      gaugeColor: "#10b981",
      crmDiabetic: "14.2%",
      crmVascular: "35.7 mmHg"
    },
    mid: {
      name: "Moderate Risk Profile",
      age: 52,
      bp: "135/88 mmHg",
      riskProb: 0.428,
      riskLevel: "MEDIUM RISK",
      qExpectation: "<Z> = +0.105",
      badgeColor: "bg-amber-500/20 text-amber-300 border-amber-500/40",
      gaugeColor: "#f59e0b",
      crmDiabetic: "28.5%",
      crmVascular: "39.9 mmHg"
    },
    high: {
      name: "Elevated Risk Profile",
      age: 64,
      bp: "165/102 mmHg",
      riskProb: 0.785,
      riskLevel: "HIGH RISK",
      qExpectation: "<Z> = +0.482",
      badgeColor: "bg-rose-500/20 text-rose-300 border-rose-500/40",
      gaugeColor: "#e11d48",
      crmDiabetic: "61.4%",
      crmVascular: "53.5 mmHg"
    }
  };

  const currentProfile = demoProfiles[selectedDemoPreset];

  const triggerDemoAudio = () => {
    if (typeof window === "undefined") return;
    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(selectedDemoPreset === "high" ? 260 : selectedDemoPreset === "mid" ? 200 : 150, ctx.currentTime);
      gain.gain.setValueAtTime(0.1, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.12);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.12);
      setIsAudioMuted(false);
      setTimeout(() => setIsAudioMuted(true), 800);
    } catch (e) {
      // Audio blocked
    }
  };

  return (
    <div className="space-y-16 py-6 pb-20">
      
      {/* 1. HERO SECTION: HIGH-CONTRAST PROFESSIONAL SHOWCASE */}
      <section className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-slate-950 via-slate-900 to-blue-950 border border-slate-800 p-8 lg:p-14 text-white shadow-2xl">
        {/* Soft Background Lighting */}
        <div className="absolute top-0 right-0 w-[550px] h-[550px] bg-blue-600/15 rounded-full blur-3xl pointer-events-none animate-pulse" />
        <div className="absolute bottom-0 left-0 w-[550px] h-[550px] bg-purple-600/15 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          
          {/* Left Column: Title & Key Statements (7 Cols) */}
          <div className="lg:col-span-7 space-y-6">
            
            {/* Pill Badges Row */}
            <div className="flex flex-wrap items-center gap-2.5">
              <span className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-300 text-xs font-bold backdrop-blur-md">
                <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                <span>SIH26139 Flagship Platform</span>
              </span>
              <span className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs font-bold backdrop-blur-md">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                <span>100% In-Fold CV Firewall</span>
              </span>
              <span className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-300 text-xs font-bold backdrop-blur-md">
                <Cpu className="w-3.5 h-3.5 text-purple-400" />
                <span>IBM QPU Hardware Verified</span>
              </span>
            </div>

            {/* Main Title Banner */}
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight leading-[1.1]">
              Quantum-Augmented <br />
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-cyan-300 to-indigo-300">
                Cardiovascular Intelligence
              </span> <br />
              & Clinical Decision Engine
            </h1>

            {/* Subtitle with Clean Text Formatting */}
            <p className="text-base sm:text-lg text-slate-300 max-w-xl leading-relaxed font-normal">
              An open, leakage-audited, hardware-validated platform for early cardiovascular disease stratification. Integrates 4-qubit parameterized quantum circuits (RY Angle Mapper & CNOT Entanglement) with boosted decision trees, benchmarked against physical IBM QPU hardware.
            </p>

            {/* Live Key Metrics Summary Bar */}
            <div className="grid grid-cols-3 gap-4 py-4 border-y border-slate-800/80 my-4 max-w-lg">
              <div>
                <div className="text-2xl sm:text-3xl font-extrabold text-cyan-400 font-mono">0.8565</div>
                <div className="text-xs text-slate-300 font-semibold mt-0.5">Quantum Residual ROC-AUC</div>
              </div>
              <div>
                <div className="text-2xl sm:text-3xl font-extrabold text-white font-mono">66,641+</div>
                <div className="text-xs text-slate-300 font-semibold mt-0.5">Audited Patients</div>
              </div>
              <div>
                <div className="text-2xl sm:text-3xl font-extrabold text-emerald-400 font-mono">444/444</div>
                <div className="text-xs text-slate-300 font-semibold mt-0.5">Passing System Tests</div>
              </div>
            </div>

            {/* Action Launch Buttons */}
            <div className="flex flex-wrap items-center gap-4 pt-2">
              <button
                onClick={() => onLaunch("prediction")}
                className="flex items-center gap-2.5 px-7 py-3.5 rounded-2xl bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-600 hover:from-blue-500 hover:to-indigo-500 text-white font-extrabold text-xs sm:text-sm shadow-lg shadow-blue-500/30 hover:scale-[1.02] transition-all cursor-pointer"
              >
                <Activity className="w-4 h-4 text-cyan-300" />
                <span>Launch Patient Risk Terminal</span>
                <ArrowRight className="w-4 h-4" />
              </button>

              <button
                onClick={() => onLaunch("quantum-lab")}
                className="flex items-center gap-2 px-6 py-3.5 rounded-2xl bg-slate-800/90 border border-slate-700 text-slate-100 font-extrabold text-xs sm:text-sm hover:bg-slate-800 hover:border-slate-500 transition-all shadow-sm cursor-pointer"
              >
                <Zap className="w-4 h-4 text-cyan-400" />
                <span>Explore Interactive Quantum Lab</span>
              </button>
            </div>
          </div>

          {/* Right Column: HIGH-CONTRAST LIVE CLINICAL DEMO CARD (5 Cols) */}
          <div className="lg:col-span-5 relative">
            <div className="relative rounded-3xl bg-slate-900/95 border border-slate-700 p-6 shadow-2xl space-y-5 backdrop-blur-2xl">
              
              {/* Header & Live ECG Trace */}
              <div className="flex items-center justify-between border-b border-slate-800 pb-3.5">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-rose-500/20 border border-rose-500/40 flex items-center justify-center text-rose-400">
                    <HeartPulse className="w-5 h-5 animate-pulse" />
                  </div>
                  <div>
                    <h3 className="font-extrabold text-sm text-white flex items-center gap-2">
                      <span>Live Cardiac Telemetry</span>
                      <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                    </h3>
                    <p className="text-[11px] font-mono text-slate-300 font-semibold">P-89201 • {currentProfile.age} yrs • {currentProfile.bp}</p>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={triggerDemoAudio}
                  className="px-3 py-1.5 rounded-xl bg-slate-800 border border-slate-700 text-slate-200 text-xs font-bold flex items-center gap-1.5 hover:bg-slate-700 cursor-pointer shadow-xs"
                  title="Test Heartbeat Sonification"
                >
                  {isAudioMuted ? <Volume2 className="w-3.5 h-3.5 text-cyan-400" /> : <VolumeX className="w-3.5 h-3.5 text-rose-400 animate-pulse" />}
                  <span>Sound</span>
                </button>
              </div>

              {/* High-Contrast Demo Profile Selector Buttons */}
              <div className="grid grid-cols-3 gap-2 p-1.5 rounded-2xl bg-slate-950 border border-slate-800 text-xs font-extrabold">
                <button
                  type="button"
                  onClick={() => setSelectedDemoPreset("low")}
                  className={`py-2 rounded-xl transition-all cursor-pointer ${
                    selectedDemoPreset === "low"
                      ? "bg-emerald-600 text-white font-black shadow-md"
                      : "text-slate-300 hover:text-white hover:bg-slate-900"
                  }`}
                >
                  Healthy
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedDemoPreset("mid")}
                  className={`py-2 rounded-xl transition-all cursor-pointer ${
                    selectedDemoPreset === "mid"
                      ? "bg-amber-500 text-slate-950 font-black shadow-md"
                      : "text-slate-300 hover:text-white hover:bg-slate-900"
                  }`}
                >
                  Moderate
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedDemoPreset("high")}
                  className={`py-2 rounded-xl transition-all cursor-pointer ${
                    selectedDemoPreset === "high"
                      ? "bg-rose-600 text-white font-black shadow-md"
                      : "text-slate-300 hover:text-white hover:bg-slate-900"
                  }`}
                >
                  Elevated
                </button>
              </div>

              {/* Dynamic Risk Gauge & Quantum Expectation */}
              <div className="p-4 rounded-2xl bg-slate-950 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-300 font-semibold">Cardiovascular Disease Probability</span>
                  <span className={`px-3 py-1 rounded-full text-[11px] font-mono font-black border ${currentProfile.badgeColor}`}>
                    {currentProfile.riskLevel}
                  </span>
                </div>

                <div className="flex items-baseline justify-between pt-1">
                  <span className="text-3xl font-black font-mono text-white">
                    {(currentProfile.riskProb * 100).toFixed(1)}%
                  </span>
                  <span className="text-xs font-mono text-cyan-200 font-bold bg-cyan-950 px-3 py-1 rounded-lg border border-cyan-700/80">
                    {currentProfile.qExpectation}
                  </span>
                </div>

                {/* Animated Risk Gauge Bar */}
                <div className="w-full bg-slate-800 h-3 rounded-full overflow-hidden flex shadow-inner">
                  <div
                    style={{ width: `${currentProfile.riskProb * 100}%`, backgroundColor: currentProfile.gaugeColor }}
                    className="transition-all duration-500 ease-out h-full"
                  />
                </div>
              </div>

              {/* High-Contrast Tri-Organ CRM Mini Panel */}
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <span className="text-slate-300 font-semibold block text-[11px]">Diabetic Cardiomyopathy:</span>
                  <span className="font-mono font-black text-amber-300 text-sm bg-amber-500/20 px-2 py-0.5 rounded border border-amber-500/40 inline-block">
                    {currentProfile.crmDiabetic}
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <span className="text-slate-300 font-semibold block text-[11px]">Vascular Stiffness:</span>
                  <span className="font-mono font-black text-purple-200 text-sm bg-purple-500/20 px-2 py-0.5 rounded border border-purple-500/40 inline-block">
                    {currentProfile.crmVascular}
                  </span>
                </div>
              </div>

              {/* IBM QPU Hardware Badge */}
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono flex items-center justify-between">
                <div className="flex items-center gap-2 text-cyan-300 font-bold">
                  <Cpu className="w-4 h-4 text-cyan-400" />
                  <span>IBM Job: dag54f8mhr3c...</span>
                </div>
                <span className="text-emerald-300 bg-emerald-500/20 border border-emerald-500/40 font-extrabold px-2.5 py-0.5 rounded-full text-[11px]">
                  Hardware Verified
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 2. CORE ARCHITECTURAL PILLARS (6 CARDS GRID) */}
      <section className="space-y-6">
        <div className="text-center max-w-3xl mx-auto space-y-2">
          <span className="px-3.5 py-1 rounded-full text-xs font-extrabold bg-blue-100 text-blue-800 border border-blue-200 uppercase tracking-wider">
            Production-Grade Engineering
          </span>
          <h2 className="text-2xl sm:text-3xl font-black text-slate-900">
            6 Core Architectural Pillars of QuantumDx
          </h2>
          <p className="text-xs sm:text-sm text-slate-500">
            Engineered to eliminate quantum hype by validating every mathematical step against standard clinical machine learning baselines.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          
          {/* Pillar 1: 4-Qubit RY-CNOT Circuit */}
          <div className="p-6 rounded-3xl bg-white border border-slate-200 shadow-sm hover:shadow-md transition-all space-y-3">
            <div className="w-11 h-11 rounded-2xl bg-blue-50 border border-blue-200 flex items-center justify-center text-blue-600">
              <Zap className="w-5 h-5" />
            </div>
            <h3 className="font-extrabold text-base text-slate-900">4-Qubit Parameterized QPU Circuit</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Maps clinical normalized features into single-qubit rotations (RY Angle Mapper) with CNOT entanglement rings, extracting expectation values.
            </p>
            <div className="pt-2 text-[11px] font-mono font-bold text-blue-600">
              Feature Map: RY(x_i) • CNOT Ring
            </div>
          </div>

          {/* Pillar 2: OOD Hilbert Safety Guard */}
          <div className="p-6 rounded-3xl bg-white border border-slate-200 shadow-sm hover:shadow-md transition-all space-y-3">
            <div className="w-11 h-11 rounded-2xl bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-600">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <h3 className="font-extrabold text-base text-slate-900">Quantum OOD Hilbert Safety Guard</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Calculates quantum state fidelity (F = |&lang;&psi;<sub>test</sub>|&psi;<sub>ref</sub>&rang;|<sup>2</sup>) to verify test samples stay strictly within the 95% Hilbert manifold boundary.
            </p>
            <div className="pt-2 text-[11px] font-mono font-bold text-emerald-600">
              Fidelity Guard: 0.984 Verified
            </div>
          </div>

          {/* Pillar 3: CRM Tri-Organ Risk Expansion */}
          <div className="p-6 rounded-3xl bg-white border border-slate-200 shadow-sm hover:shadow-md transition-all space-y-3">
            <div className="w-11 h-11 rounded-2xl bg-rose-50 border border-rose-200 flex items-center justify-center text-rose-600">
              <Heart className="w-5 h-5" />
            </div>
            <h3 className="font-extrabold text-base text-slate-900">Cardio-Renal-Metabolic (CRM) Panel</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Simultaneously assesses cardiovascular probability, diabetic cardiomyopathy indices, and vascular stiffness indices for multi-system clinical assessment.
            </p>
            <div className="pt-2 text-[11px] font-mono font-bold text-rose-600">
              Multi-System Risk Expansion
            </div>
          </div>

          {/* Pillar 4: Zero Data Leakage Firewall */}
          <div className="p-6 rounded-3xl bg-white border border-slate-200 shadow-sm hover:shadow-md transition-all space-y-3">
            <div className="w-11 h-11 rounded-2xl bg-purple-50 border border-purple-200 flex items-center justify-center text-purple-600">
              <Lock className="w-5 h-5" />
            </div>
            <h3 className="font-extrabold text-base text-slate-900">Zero Data Leakage CV Firewall</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Preprocessing, scaling, PCA, and quantum feature transformers are strictly fit inside cross-validation loops, guaranteeing zero optimistic evaluation bias.
            </p>
            <div className="pt-2 text-[11px] font-mono font-bold text-purple-600">
              Audited CV Loops → 5 Folds
            </div>
          </div>

          {/* Pillar 5: IBM Hardware Execution Proof */}
          <div className="p-6 rounded-3xl bg-white border border-slate-200 shadow-sm hover:shadow-md transition-all space-y-3">
            <div className="w-11 h-11 rounded-2xl bg-cyan-50 border border-cyan-200 flex items-center justify-center text-cyan-600">
              <Cpu className="w-5 h-5" />
            </div>
            <h3 className="font-extrabold text-base text-slate-900">IBM Quantum Hardware Proof</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Physical QPU execution payloads persisted from 156-qubit IBM Marrakesh hardware processor, verifying real-world physical quantum execution.
            </p>
            <div className="pt-2 text-[11px] font-mono font-bold text-cyan-600">
              Job ID: dag54f8mhr3...
            </div>
          </div>

          {/* Pillar 6: Hospital PDF & FHIR EHR Engine */}
          <div className="p-6 rounded-3xl bg-white border border-slate-200 shadow-sm hover:shadow-md transition-all space-y-3">
            <div className="w-11 h-11 rounded-2xl bg-indigo-50 border border-indigo-200 flex items-center justify-center text-indigo-600">
              <Printer className="w-5 h-5" />
            </div>
            <h3 className="font-extrabold text-base text-slate-900">Hospital PDF & FHIR EHR Engine</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Parses FHIR Bundle JSON standards and exports detailed, institutional clinical diagnostic reports complete with reference ranges and MD attestation.
            </p>
            <div className="pt-2 text-[11px] font-mono font-bold text-indigo-600">
              FHIR R4 JSON & PDF Generator
            </div>
          </div>
        </div>
      </section>

      {/* 3. QUANTUM-CLASSICAL MODEL TOURNAMENT BENCHMARK GRID */}
      <section className="p-8 lg:p-10 rounded-3xl bg-white border border-slate-200 shadow-sm space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-6">
          <div>
            <div className="flex items-center gap-2">
              <Trophy className="w-5 h-5 text-amber-500" />
              <span className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                Quantum-Classical Benchmark Tournament
              </span>
            </div>
            <h2 className="text-2xl font-black text-slate-900 pt-1">
              Honest Comparative Model Evaluation
            </h2>
            <p className="text-xs text-slate-500">
              Fixed test evaluation (13,329 unseen patients) demonstrating scientific rigor without hype.
            </p>
          </div>

          <button
            onClick={() => onLaunch("benchmark")}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-slate-900 text-white font-bold text-xs hover:bg-slate-800 transition-all cursor-pointer shrink-0"
          >
            <span>Inspect Full Benchmark Report</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          {/* Deployed XGBoost */}
          <div className="p-5 rounded-2xl bg-blue-50/70 border border-blue-200 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-blue-900">Hybrid XGBoost</span>
              <span className="px-2 py-0.5 rounded bg-blue-600 text-white font-bold text-[10px]">Deployed</span>
            </div>
            <div className="text-2xl font-black font-mono text-blue-700">0.8369</div>
            <div className="text-[11px] text-slate-600 font-medium">Tuned Optuna Hyperparameters</div>
          </div>

          {/* Quantum Residual Challenger */}
          <div className="p-5 rounded-2xl bg-purple-50/70 border border-purple-200 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-purple-900">Quantum Residual</span>
              <span className="px-2 py-0.5 rounded bg-purple-600 text-white font-bold text-[10px]">Challenger</span>
            </div>
            <div className="text-2xl font-black font-mono text-purple-700">0.8565</div>
            <div className="text-[11px] text-slate-600 font-medium">QPU Expectation + Residual Head</div>
          </div>

          {/* Quantum SVM */}
          <div className="p-5 rounded-2xl bg-slate-50 border border-slate-200 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-800">Quantum SVM</span>
              <span className="px-2 py-0.5 rounded bg-slate-200 text-slate-700 font-bold text-[10px]">Baseline</span>
            </div>
            <div className="text-2xl font-black font-mono text-slate-800">0.7666</div>
            <div className="text-[11px] text-slate-600 font-medium">Fidelity Quantum Kernel Matrix</div>
          </div>

          {/* VQC Circuit */}
          <div className="p-5 rounded-2xl bg-slate-50 border border-slate-200 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-800">VQC Circuit</span>
              <span className="px-2 py-0.5 rounded bg-slate-200 text-slate-700 font-bold text-[10px]">Baseline</span>
            </div>
            <div className="text-2xl font-black font-mono text-slate-800">0.6072</div>
            <div className="text-[11px] text-slate-600 font-medium">Data Re-uploading Circuit</div>
          </div>
        </div>
      </section>

      {/* 4. HIGH-CONTRAST SYSTEM WORKFLOW & PIPELINE VISUALIZER */}
      <section className="p-8 sm:p-10 rounded-3xl bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white border border-slate-800 shadow-2xl space-y-8">
        <div className="text-center max-w-2xl mx-auto space-y-1">
          <span className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-widest">End-to-End Pipeline</span>
          <h2 className="text-2xl sm:text-3xl font-black text-white">How QuantumDx Processes Patient Data</h2>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 text-xs">
          <div className="p-5 rounded-2xl bg-slate-900 border border-slate-700 hover:border-cyan-500/50 hover:bg-slate-850 transition-all duration-300 space-y-3 shadow-md">
            <div className="bg-blue-600 text-white font-extrabold text-xs px-2.5 py-1 rounded-lg inline-block font-mono">1</div>
            <h4 className="font-bold text-slate-100 text-sm">Clinical Inputs</h4>
            <p className="text-slate-300 text-xs leading-relaxed font-normal">11 Patient Features & FHIR EHR JSON</p>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900 border border-slate-700 hover:border-cyan-500/50 hover:bg-slate-850 transition-all duration-300 space-y-3 shadow-md">
            <div className="bg-cyan-600 text-white font-extrabold text-xs px-2.5 py-1 rounded-lg inline-block font-mono">2</div>
            <h4 className="font-bold text-slate-100 text-sm">In-Fold Preprocess</h4>
            <p className="text-slate-300 text-xs leading-relaxed font-normal">Strict CV Scaler & Standardizer</p>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900 border border-slate-700 hover:border-cyan-500/50 hover:bg-slate-850 transition-all duration-300 space-y-3 shadow-md">
            <div className="bg-purple-600 text-white font-extrabold text-xs px-2.5 py-1 rounded-lg inline-block font-mono">3</div>
            <h4 className="font-bold text-slate-100 text-sm">QPU Feature Map</h4>
            <p className="text-slate-400 text-xs leading-relaxed font-normal">4-Qubit RY-CNOT Observable Expectation</p>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900 border border-slate-700 hover:border-cyan-500/50 hover:bg-slate-850 transition-all duration-300 space-y-3 shadow-md">
            <div className="bg-emerald-600 text-white font-extrabold text-xs px-2.5 py-1 rounded-lg inline-block font-mono">4</div>
            <h4 className="font-bold text-slate-100 text-sm">Hybrid Inference</h4>
            <p className="text-slate-300 text-xs leading-relaxed font-normal">XGBoost + QPU Ensemble & OOD Guard</p>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900 border border-slate-700 hover:border-cyan-500/50 hover:bg-slate-850 transition-all duration-300 space-y-3 shadow-md">
            <div className="bg-rose-600 text-white font-extrabold text-xs px-2.5 py-1 rounded-lg inline-block font-mono">5</div>
            <h4 className="font-bold text-slate-100 text-sm">Clinical PDF Export</h4>
            <p className="text-slate-300 text-xs leading-relaxed font-normal">Detailed Diagnostic Report & Care Plan</p>
          </div>
        </div>
      </section>

      {/* 5. FINAL CALL TO ACTION (CTA) */}
      <section className="p-10 rounded-3xl bg-gradient-to-r from-blue-900 via-indigo-900 to-slate-900 border border-blue-800 text-white shadow-2xl flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="space-y-2 max-w-2xl">
          <h2 className="text-2xl sm:text-3xl font-black tracking-tight">
            Ready to Experience Quantum-Augmented Cardiology?
          </h2>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
            Test live clinical patient scenarios, inspect quantum Hilbert state vectors, or export hospital-grade diagnostic reports.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <button
            onClick={() => onLaunch("prediction")}
            className="px-6 py-3.5 rounded-2xl bg-white text-slate-900 font-black text-xs sm:text-sm hover:bg-slate-100 transition-all shadow-lg cursor-pointer flex items-center gap-2"
          >
            <Activity className="w-4 h-4 text-blue-600" />
            <span>Launch Telemetry Terminal</span>
          </button>
        </div>
      </section>
    </div>
  );
}
