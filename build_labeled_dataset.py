"""Backward-compatible entry point. Implementation: datasets/build_labeled_dataset.py"""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "datasets" / "build_labeled_dataset.py"),
    run_name="__main__",
)
