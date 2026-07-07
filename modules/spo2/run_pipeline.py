"""
End-to-end SpO2 AI training pipeline.

Runs all steps in sequence:
  1. Inspect dataset
  2. Preprocess signals
  3. Build SpO2Dataset + DataLoaders
  4. Train 1D CNN
  5. Evaluate metrics + save confusion matrix & ROC curve
  6. Demo single-sample inference

Run from project root:
    python modules/spo2/run_pipeline.py

This module is self-contained and does not depend on fusion, NATS, or dashboard code.
"""

from __future__ import annotations

import sys
from pathlib import Path

_MODULE = Path(__file__).resolve().parent
if str(_MODULE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_MODULE.parent.parent))


def run_step(name: str, fn) -> None:
    """Execute a pipeline step with a labelled banner."""
    print(f"\n{'─' * 60}")
    print(f"  STEP: {name}")
    print(f"{'─' * 60}")
    fn()


def main() -> None:
    from modules.spo2.inspect_dataset import main as inspect_main
    from modules.spo2.train_spo2 import main as train_main
    from modules.spo2.evaluate_model import main as evaluate_main
    from modules.spo2.spo2_inference import main as inference_main

    print("\n" + "=" * 60)
    print("  PRANA SpO2 — Full Training Pipeline")
    print("=" * 60)

    run_step("1. Inspect SpO2 dataset", inspect_main)
    run_step("2–6. Preprocess, train 1D CNN, save spo2_model.pth", train_main)
    run_step("7. Evaluate metrics, confusion matrix, ROC curve", evaluate_main)
    run_step("9. Single-sample inference demo", inference_main)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("  Model  : modules/spo2/model/spo2_model.pth")
    print("  Plots  : training_history.png, confusion_matrix.png, roc_curve.png")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
