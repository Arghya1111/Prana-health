#!/usr/bin/env python3
"""
Pre-deploy verification for Render.

Usage:
    python scripts/verify_deployment.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: F401
from app.model_health import missing_models_summary, verify_all_models
from app.paths import DB_PATH, MODULE_MODEL_PATHS, PROJECT_ROOT, ensure_runtime_dirs


def _check_imports() -> list[str]:
    errors: list[str] = []
    modules = [
        "dashboard",
        "dashboard_data",
        "app.paths",
        "app.model_health",
        "database.prana_database",
        "fusion.fusion_engine",
    ]
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Import failed: {name} — {exc}")
    return errors


def main() -> int:
    print("PRANA deployment verification")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Database path: {DB_PATH}")

    ensure_runtime_dirs()
    from database.prana_database import init_database

    init_database(DB_PATH)
    print("  SQLite schema: OK")

    import_errors = _check_imports()
    if import_errors:
        for err in import_errors:
            print(f"  ERROR: {err}")
        return 1
    print("  Core imports: OK")

    ok, summary = missing_models_summary()
    print(f"  Model files: {'OK' if ok else 'WARN'} — {summary}")

    print("\n  Full model load test (optional, may take a minute):")
    checks = verify_all_models()
    for name, info in checks.items():
        path = MODULE_MODEL_PATHS[name]
        if info.get("loadable"):
            status = "LOAD OK"
        elif not path.is_file():
            status = "MISSING"
        else:
            status = f"LOAD FAIL — {info.get('error', 'unknown')}"
        print(f"    {name:8} {status}")

    print("\nDone. Deploy when build + start commands match render.yaml.")
    return 0 if not import_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
