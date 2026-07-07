# =============================================================
#  PRĀNA — Phase 8 NATS Fusion Pipeline
#
#  Architecture:
#    Patient Devices → NATS → AI Modules → Fusion Engine
#         → SQLite → FastAPI → Dashboard → Doctor
#
#  HOW TO RUN:
#    1. nats-server
#    2. python nats_pipeline.py
# =============================================================

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import neurokit2 as nk

import app  # noqa: F401
from app.module_runner import ModuleRunner
from app.paths import PROJECT_ROOT
from database.prana_database import (
    get_fusion_stats,
    init_database,
    save_fusion_report,
    save_report,
)
from fusion import run_fusion

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── NATS configuration ────────────────────────────────────────────────────────
NATS_URL = "nats://localhost:4222"

TOPICS = {
    "ECG":     "prana.ecg",
    "EEG":     "prana.eeg",
    "PPG":     "prana.ppg",
    "SpO2":    "prana.spo2",
    "THERMAL": "prana.thermal",
    "OPTICAL": "prana.optical",
    "XRAY":    "prana.xray",
    "REPORT":  "prana.report",
}

SAMPLING_RATE = 360
WINDOW_SECONDS = 10
N_PATIENTS = 3

# Modalities required before fusion triggers (all 7 for full demo)
REQUIRED_MODALITIES: Set[str] = {
    "ECG", "EEG", "PPG", "SpO2", "THERMAL", "OPTICAL", "XRAY",
}


# =============================================================
# DEVICE SIMULATORS  (replace with real sensor reads on Pi)
# =============================================================

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def simulate_ecg(patient_id: str) -> Dict[str, Any]:
    scenario = np.random.choice(
        ["normal", "tachycardia", "bradycardia", "irregular"],
        p=[0.5, 0.2, 0.15, 0.15],
    )
    hr = {"normal": (60, 90), "tachycardia": (110, 160),
          "bradycardia": (30, 50), "irregular": (60, 90)}[scenario]
    noise = {"normal": 0.05, "tachycardia": 0.15, "bradycardia": 0.10, "irregular": 0.45}[scenario]
    ecg = nk.ecg_simulate(
        duration=WINDOW_SECONDS, sampling_rate=SAMPLING_RATE,
        heart_rate=np.random.uniform(*hr), noise=noise,
    )
    return {
        "patient_id": patient_id, "timestamp": _ts(),
        "scenario": scenario, "ecg_signal": ecg.tolist(),
        "sampling_rate": SAMPLING_RATE,
    }


def simulate_eeg(patient_id: str) -> Dict[str, Any]:
    """Simulate EEG window (178 samples matching UCI format)."""
    length = 178
    if np.random.random() < 0.15:
        signal = (np.random.randn(length) * 2.5 + np.sin(np.linspace(0, 40, length))).tolist()
        label = "seizure"
    else:
        signal = (np.random.randn(length) * 0.3).tolist()
        label = "normal"
    return {
        "patient_id": patient_id, "timestamp": _ts(),
        "eeg_signal": signal, "sampling_rate": 173.61, "scenario": label,
    }


def simulate_ppg(patient_id: str) -> Dict[str, Any]:
    hr = np.random.uniform(55, 110)
    ppg = nk.ppg_simulate(duration=10, sampling_rate=125, heart_rate=hr)
    return {
        "patient_id": patient_id, "timestamp": _ts(),
        "ppg_signal": ppg.tolist(), "sampling_rate": 125,
    }


def simulate_spo2(patient_id: str) -> Dict[str, Any]:
    spo2 = int(np.random.choice([97, 98, 99, 92, 88, 84, 79], p=[0.3, 0.2, 0.1, 0.15, 0.1, 0.1, 0.05]))
    return {"patient_id": patient_id, "timestamp": _ts(), "spo2": spo2}


def simulate_thermal(patient_id: str) -> Dict[str, Any]:
    scenario = np.random.choice(["normal", "fever", "inflammation"], p=[0.6, 0.25, 0.15])
    mapping = {
        "normal":       ("NORMAL", "NORMAL", "NO"),
        "fever":        ("FEVER", "MODERATE", "YES"),
        "inflammation": ("MODERATE", "MODERATE", "NO"),
    }
    status, severity, fever = mapping[scenario]
    return {
        "patient_id": patient_id, "timestamp": _ts(),
        "status": status, "severity": severity,
        "fever": fever, "confidence": round(np.random.uniform(90, 99), 1),
    }


def simulate_optical(patient_id: str) -> Dict[str, Any]:
    disease = np.random.choice(["NORMAL", "NORMAL", "DRY_EYE", "CONJUNCTIVITIS"], p=[0.5, 0.2, 0.15, 0.15])
    return {
        "patient_id": patient_id, "timestamp": _ts(),
        "disease": disease, "confidence": round(np.random.uniform(90, 99), 1),
    }


def simulate_xray(patient_id: str) -> Dict[str, Any]:
    scenario = np.random.choice(["clear", "mild", "serious"], p=[0.5, 0.3, 0.2])
    if scenario == "clear":
        return {"patient_id": patient_id, "timestamp": _ts(), "status": "NORMAL", "flagged": {}}
    if scenario == "mild":
        flagged = {"Atelectasis": round(np.random.uniform(0.30, 0.45), 2)}
        return {"patient_id": patient_id, "timestamp": _ts(), "status": "ATTENTION", "flagged": flagged}
    flagged = {
        "Pneumonia": round(np.random.uniform(0.55, 0.80), 2),
        "Lung Opacity": round(np.random.uniform(0.50, 0.75), 2),
    }
    return {"patient_id": patient_id, "timestamp": _ts(), "status": "CRITICAL", "flagged": flagged}


SIMULATORS = {
    "ECG": simulate_ecg, "EEG": simulate_eeg, "PPG": simulate_ppg,
    "SpO2": simulate_spo2, "THERMAL": simulate_thermal,
    "OPTICAL": simulate_optical, "XRAY": simulate_xray,
}


# =============================================================
# PUBLISHER
# =============================================================

async def publish_patient(nc, patient_id: str) -> None:
    """Publish all modality messages for one patient."""
    for modality, topic in TOPICS.items():
        if modality == "REPORT":
            continue
        data = SIMULATORS[modality](patient_id)
        await nc.publish(topic, json.dumps(data).encode())
        logger.info("Publisher → %s | %s", topic, patient_id)
        await asyncio.sleep(0.1)


async def publisher(nc, n_patients: int = N_PATIENTS) -> None:
    logger.info("Publishing data for %d patients across %d modalities...", n_patients, len(SIMULATORS))
    for i in range(1, n_patients + 1):
        await publish_patient(nc, f"PAT{i:03d}")
        await asyncio.sleep(0.2)
    logger.info("Publisher complete.")


# =============================================================
# SUBSCRIBER  — Multi-modality buffer + Fusion Engine
# =============================================================

class PranaFusionSubscriber:
    """
    Buffers incoming modality messages per patient, runs AI inference,
    fuses results, saves to SQLite, and publishes the final report.
    """

    def __init__(self, nc=None, runner: Optional[ModuleRunner] = None):
        self.nc = nc
        self.runner = runner or ModuleRunner()
        self.buffers: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.reports_done = 0

    def _buffer_key(self, modality: str) -> str:
        return modality.upper()

    def _store(self, modality: str, data: Dict[str, Any]) -> None:
        pid = data["patient_id"]
        if pid not in self.buffers:
            self.buffers[pid] = {}
        self.buffers[pid][self._buffer_key(modality)] = data

    def _ready(self, patient_id: str) -> bool:
        received = set(self.buffers.get(patient_id, {}).keys())
        return REQUIRED_MODALITIES.issubset(received)

    def process_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Run inference + fusion for a patient when all modalities are buffered."""
        if patient_id not in self.buffers:
            return None

        raw_payloads = self.buffers.pop(patient_id)
        module_outputs = self.runner.run_all(raw_payloads)

        if not module_outputs:
            logger.warning("No module outputs for %s", patient_id)
            return None

        fusion_result = run_fusion(
            patient_id=patient_id,
            module_outputs=module_outputs,
            timestamp=datetime.utcnow().isoformat(),
        )

        save_fusion_report(fusion_result)

        # Legacy triage_reports row for dashboard backward compatibility
        ecg = module_outputs.get("ECG", {})
        xray = module_outputs.get("XRAY", {})
        ecg_features = ecg.get("features", {})
        if isinstance(ecg_features, list) and len(ecg_features) >= 5:
            ecg_legacy = {
                "mean_hr": ecg_features[0], "mean_rr": ecg_features[1] * 1000,
                "sdnn": ecg_features[2] * 1000, "rmssd": ecg_features[3] * 1000,
                "pnn50": ecg_features[4],
                "status": ecg.get("status", "NORMAL"),
                "confidence": ecg.get("confidence", 0),
            }
        else:
            ecg_legacy = {
                "mean_hr": 0, "mean_rr": 0, "sdnn": 0, "rmssd": 0, "pnn50": 0,
                "status": ecg.get("status", "NORMAL"),
                "confidence": ecg.get("confidence", 0),
            }
        xray_legacy = {
            "status": xray.get("status", "NORMAL"),
            "flagged": xray.get("features", {}).get("flagged", xray.get("flagged", {})),
        }
        save_report(
            patient_id=patient_id,
            ecg_result=ecg_legacy,
            xray_result=xray_legacy,
            severity=fusion_result.overall_severity,
            action=fusion_result.recommendation.action,
        )

        sev_icon = {"CRITICAL": "!!", "HIGH": "!", "MODERATE": "~", "NORMAL": "ok", "LOW": "-"}.get(
            fusion_result.overall_severity, "?"
        )
        logger.info(
            "FUSION %s %s — %s | Risk=%.1f | Conf=%.1f%% | %s",
            sev_icon, patient_id, fusion_result.overall_severity,
            fusion_result.risk_score, fusion_result.overall_confidence,
            fusion_result.recommendation.action,
        )
        self.reports_done += 1
        return fusion_result.to_dict()

    async def _on_message(self, modality: str, msg) -> None:
        data = json.loads(msg.data.decode())
        patient_id = data["patient_id"]
        logger.info("Received %s ← %s", modality, patient_id)
        self._store(modality, data)

        if self._ready(patient_id):
            report = self.process_patient(patient_id)
            if report and self.nc:
                await self.nc.publish(
                    TOPICS["REPORT"],
                    json.dumps(report, default=str).encode(),
                )
                logger.info("Published fusion report → %s", TOPICS["REPORT"])

    def make_handler(self, modality: str):
        async def handler(msg):
            await self._on_message(modality, msg)
        return handler


# =============================================================
# OFFLINE MODE  (no NATS server required)
# =============================================================

async def run_offline_simulation(n_patients: int = N_PATIENTS) -> None:
    print("\n" + "=" * 60)
    print("  OFFLINE FUSION SIMULATION")
    print("  Devices -> AI Modules -> Fusion -> SQLite")
    print("=" * 60)

    init_database()
    subscriber = PranaFusionSubscriber()

    for i in range(1, n_patients + 1):
        pid = f"SIM{i:03d}"
        for modality, sim in SIMULATORS.items():
            subscriber._store(modality, sim(pid))
        subscriber.process_patient(pid)
        await asyncio.sleep(0.05)

    print(f"\n  {subscriber.reports_done} fusion reports saved.")
    stats = get_fusion_stats()
    if stats:
        print("  Severity distribution:")
        for level, count in sorted(stats.items()):
            print(f"    {level}: {count}")


# =============================================================
# MAIN
# =============================================================

async def main() -> None:
    print("=" * 60)
    print("  PRANA — Phase 8 NATS Fusion Pipeline")
    print("  Devices -> NATS -> AI -> Fusion -> SQLite")
    print("=" * 60)

    print("\n[1] Initializing database...")
    init_database()

    print("\n[2] Loading AI models...")
    runner = ModuleRunner()

    print(f"\n[3] Connecting to NATS at {NATS_URL}...")
    try:
        import nats
        nc = await nats.connect(NATS_URL, max_reconnect_attempts=0, connect_timeout=2)
        print("  Connected.")
    except Exception as exc:
        print(f"  NATS unavailable ({type(exc).__name__}).")
        print("  Start with: nats-server")
        await run_offline_simulation()
        return

    subscriber = PranaFusionSubscriber(nc=nc, runner=runner)

    print("\n[4] Subscribing to modality topics...")
    for modality, topic in TOPICS.items():
        if modality == "REPORT":
            continue
        await nc.subscribe(topic, cb=subscriber.make_handler(modality))
        print(f"  Listening: {topic}")

    print("\n[5] Publishing simulated device data...")
    await publisher(nc)

    await asyncio.sleep(3)

    print(f"\n[6] Complete — {subscriber.reports_done} fusion reports processed")
    stats = get_fusion_stats()
    if stats:
        for level, count in sorted(stats.items()):
            print(f"    {level}: {count}")

    await nc.close()
    print("\n[Done] Pipeline complete.")
    print("\nNext steps:")
    print("  uvicorn app.api:app --reload")
    print("  streamlit run dashboard.py")


if __name__ == "__main__":
    asyncio.run(main())
