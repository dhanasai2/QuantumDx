"""FHIR / HL7 Hospital EHR Data Adapter for QuantumDx.

Parses standard HL7 FHIR JSON Bundle/Patient/Observation resources into
QuantumDx PatientInput schema for plug-and-play hospital EHR integration.
"""

from __future__ import annotations

import datetime
from typing import Any


def parse_fhir_patient_bundle(fhir_data: dict[str, Any]) -> dict[str, Any]:
    """Parses a FHIR Bundle or individual FHIR Patient + Observation dictionary.

    Extracts age, gender, blood pressure, cholesterol, glucose, height, and weight.
    """
    # Defaults
    patient_dict = {
        "age_years": 52.0,
        "height": 168.0,
        "weight": 78.5,
        "ap_hi": 130.0,
        "ap_lo": 85.0,
        "cholesterol": 2,
        "gluc": 1,
        "gender_male": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1,
    }

    if not isinstance(fhir_data, dict):
        return patient_dict

    resource_type = fhir_data.get("resourceType", "")

    # Handle Bundle or Single Patient Resource
    entries = []
    if resource_type == "Bundle" and "entry" in fhir_data:
        entries = [e.get("resource", {}) for e in fhir_data.get("entry", [])]
    else:
        entries = [fhir_data]

    for res in entries:
        r_type = res.get("resourceType")

        # 1. Parse Patient Resource
        if r_type == "Patient":
            gender = res.get("gender", "male")
            patient_dict["gender_male"] = 1 if str(gender).lower() in ["male", "m"] else 0

            birth_date = res.get("birthDate")
            if birth_date:
                try:
                    b_year = int(birth_date.split("-")[0])
                    curr_year = datetime.datetime.now().year
                    patient_dict["age_years"] = float(max(18, min(100, curr_year - b_year)))
                except Exception:
                    pass

        # 2. Parse Observation Resource (LOINC codes)
        elif r_type == "Observation":
            code_struct = res.get("code", {})
            codings = code_struct.get("coding", [])
            loinc_codes = [c.get("code") for c in codings if c.get("system", "").endswith("loinc.org") or c.get("code")]

            val_quantity = res.get("valueQuantity", {}).get("value")
            val_num = float(val_quantity) if val_quantity is not None else None

            # Check LOINC / Display codes
            text_code = str(code_struct.get("text", "")).lower()

            if any(c in ["8480-6", "55284-4"] for c in loinc_codes) or "systolic" in text_code:
                if val_num:
                    patient_dict["ap_hi"] = float(max(70, min(240, val_num)))

            elif any(c in ["8462-4"] for c in loinc_codes) or "diastolic" in text_code:
                if val_num:
                    patient_dict["ap_lo"] = float(max(40, min(150, val_num)))

            elif any(c in ["2093-3", "14647-2"] for c in loinc_codes) or "cholesterol" in text_code:
                if val_num:
                    patient_dict["cholesterol"] = 3 if val_num > 240 else (2 if val_num > 200 else 1)

            elif any(c in ["2339-0", "15074-8"] for c in loinc_codes) or "glucose" in text_code:
                if val_num:
                    patient_dict["gluc"] = 3 if val_num > 180 else (2 if val_num > 120 else 1)

            elif any(c in ["8302-2"] for c in loinc_codes) or "height" in text_code:
                if val_num:
                    patient_dict["height"] = float(max(120, min(220, val_num)))

            elif any(c in ["29463-7"] for c in loinc_codes) or "weight" in text_code:
                if val_num:
                    patient_dict["weight"] = float(max(30, min(200, val_num)))

    return patient_dict
