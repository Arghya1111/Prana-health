"""Backward-compatible entry point. Implementation: modules/xray/xray_module.py"""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "modules" / "xray" / "xray_module.py"),
    run_name="__main__",
)
