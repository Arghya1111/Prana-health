"""
Module inference runner for the PRĀNA Phase 8 pipeline.

Wraps each standalone AI module and returns fusion-ready prediction dicts.
Models are loaded lazily and cached; missing models fall back to rule-based
defaults so the pipeline remains runnable in demo mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import app  # noqa: F401
from database.prana_database import extract_hrv_features, load_ecg_model
from fusion.utils import standardize_module_output

logger = logging.getLogger(__name__)

SAMPLING_RATE = 360
ECG_FEATURE_COLUMNS = ["Mean_HR", "Mean_RR", "SDNN", "RMSSD", "pNN50"]


@dataclass
class ModuleRunner:
    """
    Runs inference across all PRĀNA diagnostic modules.

    Each ``run_*`` method accepts raw sensor payload (from NATS) and returns
    a standardized fusion-ready dictionary.
    """

    ecg_model: Any = None
    _eeg_model: Any = field(default=None, repr=False)
    _eeg_meta: Dict[str, Any] = field(default_factory=dict, repr=False)
    _ppg_model: Any = field(default=None, repr=False)
    _ppg_meta: Dict[str, Any] = field(default_factory=dict, repr=False)
    _spo2_model: Any = field(default=None, repr=False)
    _spo2_meta: Dict[str, Any] = field(default_factory=dict, repr=False)
    _thermal_model: Any = field(default=None, repr=False)
    _thermal_meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.ecg_model is None:
            try:
                self.ecg_model = load_ecg_model()
            except Exception as exc:
                logger.warning("ECG model unavailable: %s", exc)

    # ── ECG ───────────────────────────────────────────────────────────────

    def run_ecg(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run ECG anomaly detection on a raw NATS ECG payload."""
        signal = np.asarray(payload["ecg_signal"], dtype=np.float64)
        rate = int(payload.get("sampling_rate", SAMPLING_RATE))
        features = extract_hrv_features(signal, rate)

        if features is None or self.ecg_model is None:
            return {
                "module": "ECG",
                "status": "NORMAL",
                "confidence": 0.0,
                "severity": "NORMAL",
                "features": {},
            }

        sample = pd.DataFrame([features], columns=ECG_FEATURE_COLUMNS)
        pred = int(self.ecg_model.predict(sample)[0])
        proba = self.ecg_model.predict_proba(sample)[0]
        confidence = float(max(proba) * 100.0)
        status = "ANOMALOUS" if pred == 1 else "NORMAL"

        return standardize_module_output(
            {
                "module": "ECG",
                "status": status,
                "confidence": confidence,
                "severity": "HIGH" if status == "ANOMALOUS" else "NORMAL",
                "mean_hr": features[0],
                "mean_rr": features[1] * 1000,
                "sdnn": features[2] * 1000,
                "rmssd": features[3] * 1000,
                "pnn50": features[4],
                "features": features,
            }
        ).to_dict()

    # ── EEG ───────────────────────────────────────────────────────────────

    def _ensure_eeg(self) -> bool:
        if self._eeg_model is not None:
            return True
        try:
            from modules.eeg.eeg_inference import load_eeg_model
            self._eeg_model, self._eeg_meta = load_eeg_model()
            return True
        except Exception as exc:
            logger.warning("EEG model unavailable: %s", exc)
            return False

    def run_eeg(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        signal = np.asarray(payload["eeg_signal"], dtype=np.float32)
        if not self._ensure_eeg():
            return standardize_module_output(
                {"module": "EEG", "status": "NORMAL", "confidence": 0.0, "severity": "NORMAL"}
            ).to_dict()
        from modules.eeg.eeg_inference import predict_eeg_sample
        raw = predict_eeg_sample(signal, model=self._eeg_model)
        raw["module"] = "EEG"
        raw["severity"] = "HIGH" if raw.get("status") == "ABNORMAL" else "NORMAL"
        return standardize_module_output(raw).to_dict()

    # ── PPG ───────────────────────────────────────────────────────────────

    def _ensure_ppg(self) -> bool:
        if self._ppg_model is not None:
            return True
        try:
            from modules.ppg.ppg_inference import load_ppg_model
            self._ppg_model, self._ppg_meta = load_ppg_model()
            return True
        except Exception as exc:
            logger.warning("PPG model unavailable: %s", exc)
            return False

    def run_ppg(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        signal = np.asarray(payload["ppg_signal"], dtype=np.float32)
        if not self._ensure_ppg():
            return standardize_module_output(
                {"module": "PPG", "status": "NORMAL", "confidence": 0.0, "severity": "NORMAL"}
            ).to_dict()
        from modules.ppg.ppg_inference import predict_ppg_sample
        raw = predict_ppg_sample(
            signal,
            model=self._ppg_model,
            sampling_rate=float(payload.get("sampling_rate", 125)),
        )
        raw["module"] = "PPG"
        raw["severity"] = "MODERATE" if raw.get("status") == "ABNORMAL" else "NORMAL"
        return standardize_module_output(raw).to_dict()

    # ── SpO₂ ──────────────────────────────────────────────────────────────

    def _ensure_spo2(self) -> bool:
        if self._spo2_model is not None:
            return True
        try:
            from modules.spo2.spo2_inference import load_spo2_model
            self._spo2_model, self._spo2_meta = load_spo2_model()
            return True
        except Exception as exc:
            logger.warning("SpO2 model unavailable: %s", exc)
            return False

    def run_spo2(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "spo2" in payload and "signal" not in payload:
            spo2_val = int(payload["spo2"])
            from modules.spo2.clinical_rules import spo2_to_status
            status = spo2_to_status(spo2_val)
            severity_map = {
                "NORMAL": "NORMAL", "MODERATE": "MODERATE",
                "HIGH": "HIGH", "CRITICAL": "CRITICAL",
            }
            return standardize_module_output({
                "module": "SpO2",
                "spo2": spo2_val,
                "status": status,
                "confidence": float(payload.get("confidence", 95.0)),
                "severity": severity_map.get(status, "NORMAL"),
            }).to_dict()

        signal = np.asarray(payload.get("signal", payload.get("spo2_signal", [])), dtype=np.float32)
        if not self._ensure_spo2() or signal.size == 0:
            return standardize_module_output(
                {"module": "SpO2", "spo2": 98, "status": "NORMAL", "confidence": 0.0, "severity": "NORMAL"}
            ).to_dict()
        from modules.spo2.spo2_inference import predict_spo2_sample
        raw = predict_spo2_sample(signal, model=self._spo2_model)
        raw["module"] = "SpO2"
        return standardize_module_output(raw).to_dict()

    # ── Thermal ───────────────────────────────────────────────────────────

    def _ensure_thermal(self) -> bool:
        if self._thermal_model is not None:
            return True
        try:
            from modules.thermal.thermal_inference import load_thermal_model
            self._thermal_model, self._thermal_meta = load_thermal_model()
            return True
        except Exception as exc:
            logger.warning("Thermal model unavailable: %s", exc)
            return False

    def run_thermal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("status") and "image_path" not in payload:
            return standardize_module_output({
                "module": "THERMAL",
                "status": payload["status"],
                "confidence": float(payload.get("confidence", 90.0)),
                "severity": payload.get("severity", "MODERATE"),
                "fever": payload.get("fever", "YES" if payload["status"] == "FEVER" else "NO"),
            }).to_dict()

        image_ref = payload.get("image_path") or payload.get("image")
        if not image_ref or not self._ensure_thermal():
            return standardize_module_output(
                {"module": "THERMAL", "status": "NORMAL", "confidence": 0.0, "severity": "NORMAL"}
            ).to_dict()
        from modules.thermal.thermal_inference import predict_thermal_image
        raw = predict_thermal_image(image_ref, model=self._thermal_model)
        raw["module"] = "THERMAL"
        return standardize_module_output(raw).to_dict()

    # ── Optical ───────────────────────────────────────────────────────────

    def run_optical(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Optical module stub — accepts pre-classified or defaults to NORMAL."""
        disease = str(payload.get("disease", payload.get("status", "NORMAL"))).upper()
        return standardize_module_output({
            "module": "OPTICAL",
            "disease": disease,
            "status": disease,
            "confidence": float(payload.get("confidence", 95.0)),
            "severity": "NORMAL" if disease == "NORMAL" else "MODERATE",
        }).to_dict()

    # ── X-Ray ─────────────────────────────────────────────────────────────

    def run_xray(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run X-Ray analysis from pre-computed or simulated findings."""
        if payload.get("results"):
            results = payload["results"]
            top = max(results.items(), key=lambda x: x[1])
            status = top[0].upper().replace(" ", "_") if top[1] >= 0.3 else "NORMAL"
            if top[1] >= 0.5:
                severity = "HIGH"
            elif top[1] >= 0.3:
                severity = "MODERATE"
            else:
                severity = "NORMAL"
            return standardize_module_output({
                "module": "XRAY",
                "status": status,
                "confidence": top[1] * 100,
                "severity": severity,
                "results": results,
                "flagged": {k: v for k, v in results.items() if v >= 0.3},
                "critical": {k: v for k, v in results.items() if v >= 0.5},
            }).to_dict()

        flagged = payload.get("flagged", {})
        status = str(payload.get("status", "NORMAL")).upper()
        if status == "NORMAL" and flagged:
            status = max(flagged, key=flagged.get).upper().replace(" ", "_")
        severity_map = {
            "NORMAL": "NORMAL", "ATTENTION": "MODERATE",
            "CRITICAL": "CRITICAL", "PNEUMONIA": "HIGH",
        }
        conf = max(flagged.values()) * 100 if flagged else float(payload.get("confidence", 90.0))
        return standardize_module_output({
            "module": "XRAY",
            "status": status,
            "confidence": conf,
            "severity": severity_map.get(status, "HIGH" if status != "NORMAL" else "NORMAL"),
            "flagged": flagged,
            "critical": {k: v for k, v in flagged.items() if v >= 0.5},
        }).to_dict()

    # ── Dispatcher ────────────────────────────────────────────────────────

    _HANDLERS = {
        "ECG": "run_ecg",
        "EEG": "run_eeg",
        "PPG": "run_ppg",
        "SPO2": "run_spo2",
        "SpO2": "run_spo2",
        "THERMAL": "run_thermal",
        "OPTICAL": "run_optical",
        "XRAY": "run_xray",
    }

    def run_module(self, modality: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch inference for any modality by name."""
        handler_name = self._HANDLERS.get(modality.upper())
        if handler_name is None:
            raise ValueError(f"Unknown modality: {modality}")
        return getattr(self, handler_name)(payload)

    def run_all(self, payloads: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Run inference for every modality present in the payloads dict."""
        results: Dict[str, Dict[str, Any]] = {}
        key_map = {
            "ecg": "ECG", "eeg": "EEG", "ppg": "PPG",
            "spo2": "SpO2", "thermal": "THERMAL",
            "optical": "OPTICAL", "xray": "XRAY",
        }
        for key, payload in payloads.items():
            mod = key_map.get(key.lower(), key.upper())
            try:
                results[mod] = self.run_module(mod, payload)
            except Exception as exc:
                logger.error("Module %s failed: %s", mod, exc)
        return results
