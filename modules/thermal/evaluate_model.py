"""
STEP 6 — Evaluate trained thermal CNN.
"""

from __future__ import annotations

import sys
from pathlib import Path

_MODULE = Path(__file__).resolve().parent
if str(_MODULE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_MODULE.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from modules.thermal.clinical_rules import class_index_to_status
from modules.thermal.image_dataset import create_dataloader
from modules.thermal.preprocessing import PreprocessConfig, split_train_val_test
from modules.thermal.thermal_inference import load_thermal_model
from modules.thermal.utils import META_PATH, ThermalDatasetInfo, load_raw_dataset


@torch.no_grad()
def collect(model, loader, device, num_classes):
    preds, labels, probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        p = torch.softmax(model(x), 1).cpu().numpy()
        preds.extend(p.argmax(1).tolist())
        labels.extend(y.numpy().tolist())
        if num_classes == 2:
            probs.extend(p[:, 1].tolist())
        else:
            probs.extend(p.max(1).tolist())
    return np.array(preds), np.array(labels), np.array(probs)


def plot_roc(y_true, y_score, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.legend()
    ax.set_title("Thermal Classifier ROC")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, meta = load_thermal_model(device=device)
        info = ThermalDatasetInfo.load(META_PATH)
        nc = meta["num_classes"]
        images, labels, class_names, _ = load_raw_dataset()
        cfg = PreprocessConfig.from_dataset_info(info)
        _, _, X_test, _, _, y_test = split_train_val_test(
            images, labels, cfg.test_size, cfg.val_size, cfg.random_state,
        )
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    loader = create_dataloader(X_test, y_test, cfg, 32, False, False)
    y_pred, y_true, y_score = collect(model, loader, device, nc)

    print("\n" + "=" * 60)
    print("  PRANA Thermal - Model Evaluation")
    print("=" * 60)
    print(f"\n  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  F1 Score  : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")

    if nc == 2:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"  Sensitivity : {tp / (tp + fn + 1e-8):.4f}")
        print(f"  Specificity : {tn / (tn + fp + 1e-8):.4f}")
        print(f"  ROC AUC     : {roc_auc_score(y_true, y_score):.4f}")
        plot_roc(y_true, y_score, _MODULE / "model" / "roc_curve.png")
        print(f"\n  ROC curve -> {_MODULE / 'model' / 'roc_curve.png'}")

    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    if nc == 2:
        names = ["NORMAL (0)", "ABNORMAL (1)"]
    else:
        names = [class_index_to_status(i, nc) for i in range(nc)]

    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=names, zero_division=0))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
