"""Backward-compatible entry point. Implementation: datasets/cross_validation.py"""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "datasets" / "cross_validation.py"),
    run_name="__main__",
)
