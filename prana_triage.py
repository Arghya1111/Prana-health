"""Backward-compatible entry point. Implementation: fusion/prana_triage.py"""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "fusion" / "prana_triage.py"),
    run_name="__main__",
)
