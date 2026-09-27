"""QuantumDx Expanded Manim Video Presentation Script (~3 Minutes Duration).

Renders an in-depth, mathematically rigorous, multi-scene animation detailing
the hybrid quantum-classical biomedical intelligence framework for SIH26139.

To render in low quality (quick preview):
    manim -pql quantumdx_presentation.py FullQuantumDxPresentation

To render in high quality (1080p 60fps for final video):
    manim -pqh quantumdx_presentation.py FullQuantumDxPresentation
"""

from manim import *
import numpy as np


class FullQuantumDxPresentation(Scene):
    """Master ~3 minute animation combining all 5 detailed presentation scenes."""

    def construct(self):
        # Set dark clinical background
        self.camera.background_color = "#0b0f19"

        self.play_scene_1_intro()
        self.play_scene_2_data_pipeline()
        self.play_scene_3_quantum_circuit()
        self.play_scene_4_hybrid_fusion()
        self.play_scene_5_benchmarks_outro()

    def play_scene_1_intro(self):
        # =========================================================================
        # SCENE 1: TITLE & CLINICAL PROBLEM STATEMENT (~30s)
        # =========================================================================
        title = Text("QuantumDx", font_size=56, weight=BOLD, color="#3b82f6")
        subtitle = Text("Hybrid Quantum-Classical Biomedical Intelligence", font_size=24, color="#94a3b8")
        sih_badge = Text("SIH Problem Statement SIH26139", font_size=18, color="#10b981")

        header_group = VGroup(title, subtitle, sih_badge).arrange(DOWN, buff=0.25).to_edge(UP, buff=0.6)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle), FadeIn(sih_badge), run_time=1.2)
        self.wait(1.5)

        # Problem Box
        problem_box = RoundedRectangle(corner_radius=0.2, height=2.8, width=11.0, color="#ef4444", fill_opacity=0.12).shift(DOWN * 0.4)
        problem_title = Text("The Clinical Problem: Cardiovascular Disease Risk Stratification", font_size=20, weight=BOLD, color="#f87171").move_to(problem_box.get_top() + DOWN * 0.4)
        
        problem_desc = Text(
            "• Cardiovascular disease (CVD) causes 17.9M global annual deaths (#1 mortality driver).\n"
            "• High-dimensional Electronic Health Records (EHR) contain complex non-linear vitals.\n"
            "• Traditional black-box ML models lack quantum representation spaces & explainability.\n"
            "• Solution: Fusing 4-Qubit Variational Circuits with Gradient Boosted Decision Trees.",
            font_size=15,
            line_spacing=0.8,
            color="#cbd5e1"
        ).move_to(problem_box.get_center() + DOWN * 0.15)
        
        prob_group = VGroup(problem_box, problem_title, problem_desc)

        self.play(Create(problem_box), FadeIn(problem_title), run_time=1.5)
        self.play(Write(problem_desc), run_time=2.5)
        self.wait(4.5)

        # Transition out
        self.play(FadeOut(prob_group), FadeOut(header_group), run_time=1.2)
        self.wait(0.5)

    def play_scene_2_data_pipeline(self):
        # =========================================================================
        # SCENE 2: DATA INGESTION & ADAPTIVE PCA-4 QUBIT ENCODER (~35s)
        # =========================================================================
        scene_header = Text("Stage 1: Biomedical Ingestion & Adaptive PCA-4 Encoder", font_size=30, weight=BOLD, color="#60a5fa").to_edge(UP, buff=0.5)
        self.play(Write(scene_header), run_time=1.2)

        # Subtitle explanation
        sub_txt = Text("Transforms raw clinical biomarkers into 4 physical Qubit rotation angles", font_size=15, color="#94a3b8").next_to(scene_header, DOWN, buff=0.15)
        self.play(FadeIn(sub_txt), run_time=0.8)

        # Raw EHR Box
        ehr_box = RoundedRectangle(corner_radius=0.15, height=3.0, width=3.4, color="#3b82f6", fill_opacity=0.15)
        ehr_title = Text("Raw EHR Dataset\n(N ≥ 11 Features)", font_size=16, weight=BOLD, color="#93c5fd").move_to(ehr_box.get_top() + DOWN * 0.4)
        ehr_feats = Text(
            "• Age (years)\n• Systolic BP\n• Diastolic BP\n• Cholesterol\n• Glucose Level\n• Height / Weight\n• Multi-markers",
            font_size=13,
            line_spacing=0.6,
            color="#cbd5e1"
        ).move_to(ehr_box.get_center() + DOWN * 0.15)
        ehr_group = VGroup(ehr_box, ehr_title, ehr_feats)

        # Arrow 1
        arrow1 = Arrow(start=LEFT, end=RIGHT, color="#a855f7", buff=0.1).scale(0.8)

        # PCA Component Box
        pca_box = RoundedRectangle(corner_radius=0.15, height=3.0, width=4.2, color="#a855f7", fill_opacity=0.15)
        pca_title = Text("Adaptive PCA Compressor", font_size=16, weight=BOLD, color="#c084fc").move_to(pca_box.get_top() + DOWN * 0.4)
        pca_formula = MathTex(
            r"\mathbf{X} \in \mathbb{R}^N \xrightarrow{\text{Z-Score}} \hat{\mathbf{X}} \xrightarrow{\text{PCA-4}} \boldsymbol{\theta} \in [0, \pi]^4",
            font_size=20,
            color="#f3e8ff"
        ).move_to(pca_box.get_center() + DOWN * 0.05)
        pca_desc = Text("Extracts top 4 orthogonal variance vectors\nfor physical Qubit mapping", font_size=12, line_spacing=0.5, color="#cbd5e1").move_to(pca_box.get_bottom() + UP * 0.45)
        pca_group = VGroup(pca_box, pca_title, pca_formula, pca_desc)

        # Arrow 2
        arrow2 = Arrow(start=LEFT, end=RIGHT, color="#10b981", buff=0.1).scale(0.8)

        # Qubit Angles Output Box
        q_box = RoundedRectangle(corner_radius=0.15, height=3.0, width=3.2, color="#10b981", fill_opacity=0.15)
        q_title = Text("4 Qubit Angles (rad)", font_size=16, weight=BOLD, color="#6ee7b7").move_to(q_box.get_top() + DOWN * 0.4)
        q_angles = MathTex(
            r"\theta_0 = 1.85 \quad (q_0 \text{ Sys})\\ \theta_1 = 1.99 \quad (q_1 \text{ Dia})\\ \theta_2 = 0.80 \quad (q_2 \text{ Chol})\\ \theta_3 = 1.13 \quad (q_3 \text{ Age})",
            font_size=17,
            color="#ecfdf5"
        ).move_to(q_box.get_center() + DOWN * 0.15)
        q_group = VGroup(q_box, q_title, q_angles)

        # Arrange pipeline items horizontally
        pipeline = VGroup(ehr_group, arrow1, pca_group, arrow2, q_group).arrange(RIGHT, buff=0.25).move_to(DOWN * 0.4)

        self.play(FadeIn(ehr_group), run_time=1.5)
        self.wait(1.0)
        self.play(GrowArrow(arrow1), FadeIn(pca_group), run_time=1.5)
        self.wait(1.0)
        self.play(GrowArrow(arrow2), FadeIn(q_group), run_time=1.5)
        self.wait(5.0)

        self.play(FadeOut(pipeline), FadeOut(sub_txt), FadeOut(scene_header), run_time=1.2)
        self.wait(0.5)

    def play_scene_3_quantum_circuit(self):
        # =========================================================================
        # SCENE 3: QUANTUM CIRCUIT & IBM QPU HARDWARE EXECUTION (~40s)
        # =========================================================================
        scene_header = Text("Stage 2: 4-Qubit Variational Quantum Circuit", font_size=30, weight=BOLD, color="#c084fc").to_edge(UP, buff=0.5)
        self.play(Write(scene_header), run_time=1.2)

        # Hardware Badge Banner
        hw_badge = RoundedRectangle(corner_radius=0.1, height=0.65, width=10.0, color="#10b981", fill_opacity=0.2)
        hw_text = Text("Verified IBM QPU Hardware Run: ibm_marrakesh (Job: dag54f8mhr3c73e4m300)", font_size=14, font="Monospace", color="#6ee7b7").move_to(hw_badge)
        hw_group = VGroup(hw_badge, hw_text).next_to(scene_header, DOWN, buff=0.2)
        self.play(FadeIn(hw_group), run_time=1.0)
        self.wait(1.0)

        # Draw 4 Qubit Wires and Circuit Gates
        wires = VGroup()
        qubit_labels = VGroup()
        y_positions = [1.1, 0.3, -0.5, -1.3]

        for idx, y in enumerate(y_positions):
            lbl = MathTex(rf"|q_{idx}\rangle", font_size=24, color="#93c5fd").move_to(LEFT * 5.2 + UP * y)
            wire = Line(start=LEFT * 4.4 + UP * y, end=RIGHT * 4.6 + UP * y, color="#475569", stroke_width=2)
            qubit_labels.add(lbl)
            wires.add(wire)

        self.play(Create(qubit_labels), Create(wires), run_time=1.5)

        # Add Hadamard State Prep Gates
        h_gates = VGroup()
        for idx, y in enumerate(y_positions):
            g_box = Square(side_length=0.6, color="#0284c7", fill_opacity=0.9).move_to(LEFT * 3.5 + UP * y)
            g_txt = MathTex("H", font_size=16, color="#ffffff").move_to(g_box)
            h_gates.add(VGroup(g_box, g_txt))

        self.play(FadeIn(h_gates), run_time=1.2)

        # Add RY(theta) Gate Boxes
        ry_gates = VGroup()
        for idx, y in enumerate(y_positions):
            g_box = Square(side_length=0.65, color="#3b82f6", fill_opacity=0.9).move_to(LEFT * 1.8 + UP * y)
            g_txt = MathTex(rf"RY(\theta_{idx})", font_size=13, color="#ffffff").move_to(g_box)
            ry_gates.add(VGroup(g_box, g_txt))

        self.play(FadeIn(ry_gates), run_time=1.2)

        # Add Linear CNOT Entangling Rings
        cnot_group = VGroup()
        for idx in range(3):
            y_top = y_positions[idx]
            y_bot = y_positions[idx + 1]
            x_val = -0.3 + idx * 1.2

            pt_top = np.array([x_val, y_top, 0.0])
            pt_bot = np.array([x_val, y_bot, 0.0])

            ctrl_dot = Dot(point=pt_top, radius=0.08, color="#a855f7")
            target_circle = Circle(radius=0.15, color="#a855f7").move_to(pt_bot)
            cnot_line = Line(start=pt_top, end=pt_bot, color="#a855f7", stroke_width=2)
            cnot_group.add(ctrl_dot, target_circle, cnot_line)

        self.play(Create(cnot_group), run_time=1.5)

        # Add Pauli-Z Measurement Meters
        meters = VGroup()
        for idx, y in enumerate(y_positions):
            m_box = Rectangle(height=0.6, width=1.1, color="#10b981", fill_opacity=0.9).move_to(RIGHT * 3.8 + UP * y)
            m_txt = MathTex(rf"\langle Z_{idx} \rangle", font_size=16, color="#ffffff").move_to(m_box)
            meters.add(VGroup(m_box, m_txt))

        self.play(FadeIn(meters), run_time=1.2)
        self.wait(5.0)

        # Clean Scene 3
        all_circuit = VGroup(scene_header, hw_group, qubit_labels, wires, h_gates, ry_gates, cnot_group, meters)
        self.play(FadeOut(all_circuit), run_time=1.2)
        self.wait(0.5)

    def play_scene_4_hybrid_fusion(self):
        # =========================================================================
        # SCENE 4: HYBRID FUSION BACKBONE & TREESHAP EXPLAINABILITY (~35s)
        # =========================================================================
        scene_header = Text("Stage 3: Hybrid Feature Fusion & XGBoost Decision Engine", font_size=28, weight=BOLD, color="#10b981").to_edge(UP, buff=0.5)
        self.play(Write(scene_header), run_time=1.2)

        # Vector Concatenation Formula
        concat_formula = MathTex(
            r"\mathbf{X}_{\text{hybrid}} = \big[ \mathbf{x}_{\text{classical}} \;||\; \langle Z_0 \rangle, \langle Z_1 \rangle, \langle Z_2 \rangle, \langle Z_3 \rangle \big] \in \mathbb{R}^{12d}",
            font_size=24,
            color="#6ee7b7"
        ).next_to(scene_header, DOWN, buff=0.3)

        self.play(Write(concat_formula), run_time=1.5)
        self.wait(1.0)

        # Left Card: XGBoost Output
        xgb_box = RoundedRectangle(corner_radius=0.15, height=3.0, width=4.5, color="#3b82f6", fill_opacity=0.15).to_edge(LEFT, buff=0.8).shift(DOWN * 0.4)
        xgb_title = Text("XGBoost Hybrid Probability", font_size=16, weight=BOLD, color="#93c5fd").move_to(xgb_box.get_top() + DOWN * 0.4)
        xgb_prob = MathTex(r"p_{\text{hybrid}} = 0.684 \quad (68.4\%)", font_size=20, color="#38bdf8").move_to(xgb_box.get_center())
        xgb_band = Text("HIGH RISK BAND", font_size=14, weight=BOLD, color="#f87171").move_to(xgb_box.get_bottom() + UP * 0.45)
        xgb_group = VGroup(xgb_box, xgb_title, xgb_prob, xgb_band)

        # Right Card: TreeSHAP Surrogate Explainability Audit
        shap_box = RoundedRectangle(corner_radius=0.15, height=3.0, width=5.8, color="#a855f7", fill_opacity=0.15).to_edge(RIGHT, buff=0.6).shift(DOWN * 0.4)
        shap_title = Text("TreeSHAP Surrogate Explainability", font_size=16, weight=BOLD, color="#c084fc").move_to(shap_box.get_top() + DOWN * 0.4)
        shap_stats = Text(
            "• Surrogate Model: GBDT fitted on QPU expectations\n"
            "• Label Agreement Rate: 94.1% (High Fidelity)\n"
            "• Spearman Rank Correlation: ρ = 0.862\n"
            "• Top Risk Driver: Systolic BP (+0.487 attribution)\n"
            "• Protective Factor: Physical Activity (-0.134 offset)",
            font_size=13,
            line_spacing=0.6,
            color="#cbd5e1"
        ).move_to(shap_box.get_center() + DOWN * 0.1)
        shap_group = VGroup(shap_box, shap_title, shap_stats)

        self.play(FadeIn(xgb_group), run_time=1.5)
        self.wait(1.0)
        self.play(FadeIn(shap_group), run_time=1.5)
        self.wait(5.0)

        # Clean Scene 4
        self.play(FadeOut(xgb_group), FadeOut(shap_group), FadeOut(concat_formula), FadeOut(scene_header), run_time=1.2)
        self.wait(0.5)

    def play_scene_5_benchmarks_outro(self):
        # =========================================================================
        # SCENE 5: BENCHMARK RESULTS & OUTRO (~30s)
        # =========================================================================
        scene_header = Text("Validated Benchmark Performance & Conclusion", font_size=30, weight=BOLD, color="#f59e0b").to_edge(UP, buff=0.5)
        self.play(Write(scene_header), run_time=1.2)

        # Subtitle
        sub_txt = Text("Evaluated on 13,329 unseen test patients across 5-fold outer cross-validation", font_size=14, color="#94a3b8").next_to(scene_header, DOWN, buff=0.15)
        self.play(FadeIn(sub_txt), run_time=0.8)

        # Metrics Grid Cards
        m1 = RoundedRectangle(corner_radius=0.15, height=1.6, width=2.7, color="#10b981", fill_opacity=0.2)
        t1 = Text("Test ROC-AUC", font_size=12, color="#94a3b8").move_to(m1.get_top() + DOWN * 0.3)
        v1 = Text("0.8567", font_size=24, weight=BOLD, color="#34d399").move_to(m1.get_center() + DOWN * 0.1)
        g1 = VGroup(m1, t1, v1)

        m2 = RoundedRectangle(corner_radius=0.15, height=1.6, width=2.7, color="#3b82f6", fill_opacity=0.2)
        t2 = Text("Sensitivity", font_size=12, color="#94a3b8").move_to(m2.get_top() + DOWN * 0.3)
        v2 = Text("91.2%", font_size=24, weight=BOLD, color="#60a5fa").move_to(m2.get_center() + DOWN * 0.1)
        g2 = VGroup(m2, t2, v2)

        m3 = RoundedRectangle(corner_radius=0.15, height=1.6, width=2.7, color="#a855f7", fill_opacity=0.2)
        t3 = Text("Specificity", font_size=12, color="#94a3b8").move_to(m3.get_top() + DOWN * 0.3)
        v3 = Text("80.4%", font_size=24, weight=BOLD, color="#c084fc").move_to(m3.get_center() + DOWN * 0.1)
        g3 = VGroup(m3, t3, v3)

        m4 = RoundedRectangle(corner_radius=0.15, height=1.6, width=2.7, color="#f59e0b", fill_opacity=0.2)
        t4 = Text("Surrogate Match", font_size=12, color="#94a3b8").move_to(m4.get_top() + DOWN * 0.3)
        v4 = Text("94.1%", font_size=24, weight=BOLD, color="#fbbf24").move_to(m4.get_center() + DOWN * 0.1)
        g4 = VGroup(m4, t4, v4)

        metrics_row = VGroup(g1, g2, g3, g4).arrange(RIGHT, buff=0.3).move_to(UP * 0.8)
        self.play(FadeIn(metrics_row), run_time=1.5)
        self.wait(1.5)

        # Conclusion Statement Box
        outro_box = RoundedRectangle(corner_radius=0.2, height=2.4, width=11.0, color="#3b82f6", fill_opacity=0.2).move_to(DOWN * 1.5)
        outro_title = Text("QuantumDx: Bridging Quantum Computing & Healthcare Decision Support", font_size=18, weight=BOLD, color="#93c5fd").move_to(outro_box.get_top() + DOWN * 0.4)
        outro_desc = Text(
            "• Demonstrates non-inferior quantum representation capabilities on physical QPU hardware.\n"
            "• Provides leak-audited 5-fold outer cross-validation and TreeSHAP explainability.\n"
            "• Built and hardware-verified for Smart India Hackathon Grand Finale (SIH26139).",
            font_size=13,
            line_spacing=0.6,
            color="#e2e8f0"
        ).move_to(outro_box.get_center() + DOWN * 0.15)
        outro_group = VGroup(outro_box, outro_title, outro_desc)

        self.play(Create(outro_box), FadeIn(outro_group), run_time=1.5)
        self.wait(6.0)

        # Final Fade Out
        self.play(FadeOut(metrics_row), FadeOut(outro_group), FadeOut(sub_txt), FadeOut(scene_header), run_time=1.5)
        self.wait(1.0)
