"""Backward-compatible entry point. Implementation: datasets/test_annotations.py"""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "datasets" / "test_annotations.py"),
    run_name="__main__",
)
