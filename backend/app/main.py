"""Phase 14 FastAPI application entry point.

    uvicorn app.main:app --reload --port 8000   (run from backend/)

Loads the trained artifacts ONCE at startup (see services/inference_service.py)
-- every request reuses the same in-memory models. CORS is open to the
Next.js dev server origin by default; tighten ALLOWED_ORIGINS for production.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.services import inference_service as svc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phase14")

ALLOWED_ORIGINS = os.environ.get("PHASE14_ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app = FastAPI(
    title="QuantumDx Hybrid Risk Intelligence API",
    description="Hybrid quantum-classical disease-risk decision-support API (Phase 14). "
                 "Research/decision-support prototype -- not a certified diagnostic device.",
    version="phase14_hybrid_v1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def _startup() -> None:
    try:
        svc.startup_load_artifacts()
        logger.info("Phase 14 model artifacts loaded successfully.")
    except Exception:
        logger.exception(
            "Failed to load Phase 14 artifacts at startup. /api/health will report model_ready=false "
            "until `python -m src.large_dataset.phase14_hybrid_product` has been run."
        )


@app.get("/")
def root() -> dict:
    return {"service": "QuantumDx Hybrid Risk Intelligence API", "docs": "/docs", "health": "/api/health"}
