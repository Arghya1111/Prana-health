"""Backward-compatible entry point. Implementation: modules/ecg/hrv_features.py"""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "modules" / "ecg" / "hrv_features.py"),
    run_name="__main__",
)
