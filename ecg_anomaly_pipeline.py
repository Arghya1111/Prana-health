"""Backward-compatible entry point. Implementation: modules/ecg/ecg_anomaly_pipeline.py"""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "modules" / "ecg" / "ecg_anomaly_pipeline.py"),
    run_name="__main__",
)
