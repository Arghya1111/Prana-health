"""
Lazy AI model health checks for the clinical dashboard.

Verifies checkpoint files exist and can be loaded without blocking startup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st

import app  # noqa: F401 — project root on sys.path
from app.paths import MODULE_MODEL_PATHS

logger = logging.getLogger(__name__)

_MODEL_LOADERS: Dict[str, str] = {
    "EEG": "modules.eeg.eeg_inference.load_eeg_model",
    "PPG": "modules.ppg.ppg_inference.load_ppg_model",
    "SpO2": "modules.spo2.spo2_inference.load_spo2_model",
    "Thermal": "modules.thermal.thermal_inference.load_thermal_model",
}


def _check_ecg_loadable(path: Path) -> Tuple[bool, Optional[str]]:
    """Lightweight ECG check — avoid load_ecg_model() which may auto-train."""
    try:
        import joblib

        joblib.load(path)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _resolve_loader(dotted: str):
    module_name, _, attr = dotted.rpartition(".")
    import importlib

    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


@st.cache_resource(show_spinner=False)
def verify_model(name: str) -> Dict[str, Any]:
    """
    Check whether a modality model file exists and loads.

    Returns a dict with keys: name, path, exists, loadable, error.
    """
    path: Path = MODULE_MODEL_PATHS[name]
    result: Dict[str, Any] = {
        "name": name,
        "path": str(path),
        "exists": path.is_file(),
        "loadable": False,
        "error": None,
    }
    if not result["exists"]:
        result["error"] = f"Model file not found: {path}"
        return result

    if name == "ECG":
        loadable, err = _check_ecg_loadable(path)
        result["loadable"] = loadable
        result["error"] = err
        return result

    loader_path = _MODEL_LOADERS.get(name)
    if not loader_path:
        result["loadable"] = True
        return result

    try:
        loader = _resolve_loader(loader_path)
        loader(model_path=path)
        result["loadable"] = True
    except FileNotFoundError as exc:
        result["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 — surface load failures in UI
        logger.warning("Model load check failed for %s: %s", name, exc)
        result["error"] = str(exc)

    return result


@st.cache_resource(show_spinner=False)
def verify_all_models() -> Dict[str, Dict[str, Any]]:
    """Verify all bundled modality models once per process."""
    return {name: verify_model(name) for name in MODULE_MODEL_PATHS}


def module_status_from_models() -> Dict[str, str]:
    """Fast file-existence check for dashboard badges (no model load at startup)."""
    statuses: Dict[str, str] = {}
    for name, path in MODULE_MODEL_PATHS.items():
        statuses[name] = "online" if path.is_file() else "offline"
    return statuses


def missing_models_summary(*, verify_load: bool = False) -> Tuple[bool, str]:
    """Return (all_ok, human-readable summary for deployment warnings)."""
    if verify_load:
        checks = verify_all_models()
        missing = [n for n, i in checks.items() if not i.get("exists")]
        broken = [n for n, i in checks.items() if i.get("exists") and not i.get("loadable")]
    else:
        missing = [n for n, p in MODULE_MODEL_PATHS.items() if not p.is_file()]
        broken = []
    if not missing and not broken:
        return True, "All AI model files present."
    parts = []
    if missing:
        parts.append(f"Missing weights: {', '.join(missing)}")
    if broken:
        parts.append(f"Failed to load: {', '.join(broken)}")
    return False, " · ".join(parts)
