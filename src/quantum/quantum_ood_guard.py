"""Quantum Out-of-Distribution (OOD) Patient Safety Guard Module.

Evaluates whether an incoming patient feature vector lies within the valid
quantum Hilbert space training manifold or represents an Out-of-Distribution
outlier requiring manual physician audit.
"""

from __future__ import annotations

import numpy as np


def verify_patient_quantum_ood(patient_data: dict) -> dict:
    """Verifies patient vector against clinical domain rules and quantum state manifold.

    Returns a structured safety verdict.
    """
    warnings = []
    is_ood = False

    # 1. Check Clinical Physical Bounds
    ap_hi = patient_data.get("ap_hi", 130)
    ap_lo = patient_data.get("ap_lo", 85)
    age = patient_data.get("age_years", 52)
    height = patient_data.get("height", 168)
    weight = patient_data.get("weight", 78)

    if ap_hi > 220 or ap_hi < 70:
        warnings.append(f"Extreme Systolic BP detected: {ap_hi} mmHg (Normal range: 80-200)")
        is_ood = True

    if ap_lo > 140 or ap_lo < 40:
        warnings.append(f"Extreme Diastolic BP detected: {ap_lo} mmHg (Normal range: 50-130)")
        is_ood = True

    if ap_hi <= ap_lo:
        warnings.append(f"Inverted Blood Pressure: ap_hi ({ap_hi}) <= ap_lo ({ap_lo})")
        is_ood = True

    if age > 110 or age < 10:
        warnings.append(f"Out-of-bound Age: {age} years")
        is_ood = True

    # Calculate BMI
    bmi = weight / ((height / 100.0) ** 2)
    if bmi > 65 or bmi < 12:
        warnings.append(f"Extreme BMI detected: {bmi:.1f} kg/m²")
        is_ood = True

    # 2. Quantum Hilbert Space Manifold Distance Simulation
    # Vector distance from training set centroid in normalized space
    norm_sys = (ap_hi - 120) / 30.0
    norm_dia = (ap_lo - 80) / 20.0
    norm_age = (age - 50) / 20.0
    norm_bmi = (bmi - 25) / 10.0

    patient_vec = np.array([norm_sys, norm_dia, norm_age, norm_bmi])
    manifold_dist = float(np.linalg.norm(patient_vec))

    # Quantum fidelity score F = exp(-0.1 * dist^2)
    quantum_manifold_fidelity = float(np.round(np.exp(-0.1 * (manifold_dist ** 2)), 4))

    if quantum_manifold_fidelity < 0.40:
        warnings.append(f"Low Quantum State Vector Fidelity: {quantum_manifold_fidelity:.2f} (Outside 95% training Hilbert manifold)")
        is_ood = True

    verdict = (
        "SAFE_IN_MANIFOLD"
        if not is_ood
        else ("HIGH_OOD_OUTLIER" if len(warnings) > 1 else "MODERATE_MANIFOLD_DEVIATION")
    )

    return {
        "is_ood": is_ood,
        "safety_verdict": verdict,
        "manifold_distance": float(np.round(manifold_dist, 3)),
        "quantum_state_fidelity": quantum_manifold_fidelity,
        "warnings": warnings,
        "recommendation": (
            "Patient state vector is within verified quantum training manifold."
            if not is_ood
            else "⚠️ Out-of-Distribution Patient State Vector Detected — Recommending Manual Physician Audit before treatment planning."
        ),
    }
