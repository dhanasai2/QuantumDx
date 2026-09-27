"""Pydantic request/response schemas for the Phase 14 API.

Field names match src.large_dataset.schema's feature columns exactly, so
a PatientInput round-trips directly into the trained pipeline with no
translation layer to keep in sync.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    """One patient's raw (pre-preprocessing) feature values -- the exact
    columns src.large_dataset.schema.get_cardio_feature_groups() expects."""

    age_years: float = Field(..., ge=1, le=120, description="Age in years")
    height: float = Field(..., ge=100, le=250, description="Height in cm")
    weight: float = Field(..., ge=20, le=300, description="Weight in kg")
    ap_hi: float = Field(..., ge=60, le=300, description="Systolic blood pressure")
    ap_lo: float = Field(..., ge=30, le=200, description="Diastolic blood pressure")
    cholesterol: int = Field(..., ge=1, le=3, description="1=normal, 2=above normal, 3=well above normal")
    gluc: int = Field(..., ge=1, le=3, description="1=normal, 2=above normal, 3=well above normal")
    gender_male: int = Field(..., ge=0, le=1, description="1=male, 0=female")
    smoke: int = Field(..., ge=0, le=1)
    alco: int = Field(..., ge=0, le=1, description="Alcohol intake")
    active: int = Field(..., ge=0, le=1, description="Physically active")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age_years": 52.0, "height": 168, "weight": 78.5, "ap_hi": 130, "ap_lo": 85,
                "cholesterol": 2, "gluc": 1, "gender_male": 1, "smoke": 0, "alco": 0, "active": 1,
            }
        }
    }


class BatchPredictRequest(BaseModel):
    records: list[PatientInput]


class PredictionBlock(BaseModel):
    probability: float
    class_: str = Field(..., alias="class")
    risk_level: str
    threshold: float
    confidence: float

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    prediction: dict
    classical: dict
    quantum: dict
    decision_support: dict
    model_info: dict


class WhatIfRequest(BaseModel):
    base_patient: PatientInput
    adjustments: dict[str, float]


class ThresholdTuneRequest(BaseModel):
    threshold: float = Field(..., ge=0.05, le=0.95)
    probability: float = Field(..., ge=0.0, le=1.0)


class ModelTournamentRequest(BaseModel):
    patient: PatientInput

