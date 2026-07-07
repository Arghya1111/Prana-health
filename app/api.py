"""
PRĀNA Phase 8 — FastAPI REST layer.

Exposes fusion reports from SQLite for the dashboard and external clients.

Run:
    uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations

import json
import socket
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import app as _prana  # noqa: F401 — ensure project root on sys.path
from app.paths import DB_PATH, MODEL_PATH
from database.prana_database import (
    get_all_reports,
    get_fusion_report,
    get_fusion_reports,
    get_fusion_stats,
    get_patient_fusion_history,
    get_patient_history,
    get_stats,
    init_database,
)

app = FastAPI(
    title="PRĀNA Clinical Intelligence API",
    description="REST API for multi-modality AI fusion triage reports",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ───────────────────────────────────────────────

class TriageRequest(BaseModel):
    patient_id: str = Field(..., example="PAT-001")
    module_outputs: Dict[str, Dict[str, Any]] = Field(
        ...,
        example={
            "ECG": {"module": "ECG", "status": "NORMAL", "confidence": 95.0, "severity": "NORMAL"},
            "SpO2": {"module": "SpO2", "spo2": 98, "status": "NORMAL", "confidence": 99.0, "severity": "NORMAL"},
        },
    )


class HealthResponse(BaseModel):
    status: str
    database: str
    nats: str
    ecg_model: str


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup() -> None:
    init_database()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """System health check."""
    import os
    db_ok = "ok" if os.path.exists(DB_PATH) else "missing"
    model_ok = "ok" if os.path.exists(MODEL_PATH) else "missing"
    nats_ok = "unknown"
    try:
        sock = socket.create_connection(("localhost", 4222), timeout=1)
        sock.close()
        nats_ok = "ok"
    except OSError:
        nats_ok = "offline"
    return HealthResponse(status="ok", database=db_ok, nats=nats_ok, ecg_model=model_ok)


@app.get("/stats", tags=["Analytics"])
def clinic_stats() -> Dict[str, Any]:
    """Clinic-wide severity statistics."""
    return {
        "legacy": get_stats(),
        "fusion": get_fusion_stats(),
    }


@app.get("/patients", tags=["Patients"])
def list_patients(
    limit: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """List patients with their latest fusion report summary."""
    reports = get_fusion_reports(limit=limit)
    if not reports:
        return get_all_reports(limit=limit)
    return reports


@app.get("/patients/{patient_id}/reports", tags=["Patients"])
def patient_reports(patient_id: str) -> Dict[str, Any]:
    """Full report history for a patient."""
    fusion = get_patient_fusion_history(patient_id)
    legacy = get_patient_history(patient_id)
    if not fusion and not legacy:
        raise HTTPException(404, f"No reports for patient {patient_id}")
    return {"patient_id": patient_id, "fusion_reports": fusion, "legacy_reports": legacy}


@app.get("/reports", tags=["Reports"])
def list_reports(
    limit: int = Query(50, ge=1, le=500),
    severity: Optional[str] = Query(None, description="Filter by severity"),
) -> List[Dict[str, Any]]:
    """List recent fusion reports."""
    reports = get_fusion_reports(limit=limit)
    if severity:
        reports = [r for r in reports if r.get("overall_severity") == severity.upper()]
    return reports


@app.get("/reports/{report_id}", tags=["Reports"])
def get_report(report_id: int) -> Dict[str, Any]:
    """Get a single fusion report with full JSON payload."""
    report = get_fusion_report(report_id)
    if not report:
        raise HTTPException(404, f"Report {report_id} not found")
    return report


@app.post("/triage", tags=["Triage"])
def manual_triage(request: TriageRequest) -> Dict[str, Any]:
    """
    Manually trigger fusion without NATS (useful for testing).

    Accepts module prediction dicts and returns the fused clinical decision.
    """
    from fusion import run_fusion
    from database.prana_database import save_fusion_report

    result = run_fusion(request.patient_id, request.module_outputs)
    report_id = save_fusion_report(result)
    return {
        "report_id": report_id,
        "patient_id": request.patient_id,
        "risk_score": result.risk_score,
        "overall_severity": result.overall_severity,
        "overall_confidence": result.overall_confidence,
        "recommendation": result.recommendation.to_dict(),
        "report": result.report.to_dict(),
    }


@app.get("/reports/{report_id}/summary", tags=["Reports"])
def report_summary(report_id: int) -> Dict[str, Any]:
    """Executive summary for a fusion report."""
    report = get_fusion_report(report_id)
    if not report:
        raise HTTPException(404, f"Report {report_id} not found")
    data = report.get("report_data") or {}
    return {
        "report_id": report_id,
        "patient_id": report["patient_id"],
        "timestamp": report["timestamp"],
        "risk_score": report["risk_score"],
        "overall_severity": report["overall_severity"],
        "overall_confidence": report["overall_confidence"],
        "action": report["action"],
        "summary": data.get("report", {}).get("summary", report.get("recommendation", "")),
    }
