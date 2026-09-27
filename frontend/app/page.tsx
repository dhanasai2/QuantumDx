"use client";

import React, { useEffect, useState } from "react";
import Navbar from "@/components/Navbar";
import LandingPage from "@/components/LandingPage";
import DashboardView from "@/components/DashboardView";
import DatasetView, { DatasetState } from "@/components/DatasetView";
import PredictionView, { PatientFormData } from "@/components/PredictionView";
import ExplainabilityView from "@/components/ExplainabilityView";
import BenchmarkView from "@/components/BenchmarkView";
import QuantumLabView from "@/components/QuantumLabView";
import HardwareView from "@/components/HardwareView";
import ResearchView from "@/components/ResearchView";

export default function Home() {
  const [activeTab, setActiveTab] = useState<string>("landing");
  const [apiStatus, setApiStatus] = useState<boolean>(false);

  // Global shared state across tabs
  const [patientFormData, setPatientFormData] = useState<PatientFormData>({
    age_years: 52.0,
    height: 168.0,
    weight: 78.5,
    ap_hi: 130.0,
    ap_lo: 85.0,
    cholesterol: 2,
    gluc: 1,
    gender_male: 1,
    smoke: 0,
    alco: 0,
    active: 1
  });

  const [predictionResult, setPredictionResult] = useState<any>(null);

  const [datasetState, setDatasetState] = useState<DatasetState>({
    selectedFile: null,
    validationResult: null,
    preprocessPreview: null
  });

  useEffect(() => {
    const checkApi = () => {
      fetch("http://localhost:8000/api/health")
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (data && data.status === "ok") {
            setApiStatus(true);
          } else {
            setApiStatus(false);
          }
        })
        .catch(() => setApiStatus(false));
    };

    checkApi();
    const interval = setInterval(checkApi, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans bg-clinical-pattern">
      {/* Top Header Navbar */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} apiStatus={apiStatus} />

      {/* Main View Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8">
        {activeTab === "landing" && <LandingPage onLaunch={setActiveTab} />}
        {activeTab === "dashboard" && <DashboardView onNavigate={setActiveTab} apiStatus={apiStatus} />}
        {activeTab === "dataset" && (
          <DatasetView
            datasetState={datasetState}
            setDatasetState={setDatasetState}
            setFormData={setPatientFormData}
            setActiveTab={setActiveTab}
          />
        )}
        {activeTab === "prediction" && (
          <PredictionView
            formData={patientFormData}
            setFormData={setPatientFormData}
            predictionResult={predictionResult}
            setPredictionResult={setPredictionResult}
          />
        )}
        {activeTab === "explainability" && (
          <ExplainabilityView formData={patientFormData} setFormData={setPatientFormData} />
        )}
        {activeTab === "benchmark" && <BenchmarkView />}
        {activeTab === "quantum-lab" && <QuantumLabView formData={patientFormData} />}
        {activeTab === "hardware" && <HardwareView />}
        {activeTab === "research" && <ResearchView />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-6 px-4 text-center text-xs text-slate-500 shadow-sm mt-12">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="font-bold text-slate-800">QuantumDx</span>
            <span>• SIH Problem Statement SIH26139</span>
          </div>
          <div className="text-slate-600 font-mono text-[11px]">
            Hardware Verified: <code className="text-emerald-600 font-bold bg-emerald-50 px-1 py-0.5 rounded border border-emerald-200">ibm_marrakesh (Job: dag54f8mhr3c73e4m300)</code>
          </div>
          <div>
            <span>Decision Support Prototype • Not a Certified Medical Diagnostic Device</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

