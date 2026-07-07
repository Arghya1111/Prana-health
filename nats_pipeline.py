"""Backward-compatible entry point. Implementation: app/nats_pipeline.py"""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "app" / "nats_pipeline.py"),
    run_name="__main__",
)
