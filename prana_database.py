"""Backward-compatible entry point. Implementation: database/prana_database.py"""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "database" / "prana_database.py"),
    run_name="__main__",
)
