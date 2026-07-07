"""Backward-compatible entry point. Implementation: datasets/explore_real_dataset.py"""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "datasets" / "explore_real_dataset.py"),
    run_name="__main__",
)
