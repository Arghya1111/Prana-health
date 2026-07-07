"""Backward-compatible entry point. Implementation: datasets/abc.py"""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "datasets" / "abc.py"),
    run_name="__main__",
)
