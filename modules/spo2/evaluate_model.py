"""
STEP 7 — Evaluate the trained SpO2 CNN on the held-out test set.

Computes classification or regression metrics and saves:
  - model/confusion_matrix.png  (classification)
  - model/roc_curve.png         (classification)

Run from project root:
    python modules/spo2/evaluate_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_MODULE = Path(__file__).resolve().parent
if str(_MODULE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_MODULE.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from modules.spo2.preprocessing import preprocess_pipeline, split_train_val_test
from modules.spo2.spo2_dataset import create_dataloader
from modules.spo2.spo2_inference import load_spo2_model
from modules.spo2.utils import META_PATH, SpO2DatasetInfo, ensure_model_dir, load_raw_dataset

CONFUSION_MATRIX_PATH = _MODULE / "model" / "confusion_matrix.png"
ROC_CURVE_PATH = _MODULE / "model" / "roc_curve.png"


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    task_type: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return predictions, true labels, and probability/score arrays."""
    model.eval()
    all_preds: List[float] = []
    all_labels: List[float] = []
    all_probs: List[np.ndarray] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)

        if task_type == "regression":
            preds = logits.squeeze(-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.numpy().tolist())
            all_probs.append(preds.reshape(-1, 1))
        else:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.numpy().tolist())
            all_probs.append(probs)

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    y_prob = np.vstack(all_probs) if all_probs else np.array([])
    return y_pred, y_true, y_prob


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Path,
) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_labels = class_names if len(class_names) == cm.shape[0] else [str(i) for i in range(cm.shape[0])]
    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("SpO2 CNN — Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: Path,
) -> Optional[Path]:
    n_classes = y_prob.shape[1]
    fig, ax = plt.subplots(figsize=(7, 6))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#2980B9", lw=2, label=f"ABNORMAL (AUC = {roc_auc:.3f})")
    else:
        for cls_idx in range(n_classes):
            binary_true = (y_true == cls_idx).astype(int)
            if binary_true.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(binary_true, y_prob[:, cls_idx])
            roc_auc = auc(fpr, tpr)
            label = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
            ax.plot(fpr, tpr, lw=1.5, label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("SpO2 CNN — ROC Curve")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def print_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    print("\n" + "=" * 60)
    print("  PRANA SpO2 - Model Evaluation (Classification)")
    print("=" * 60)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    print(f"\n  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")

    if y_prob.shape[1] == 2:
        try:
            auc_score = roc_auc_score(y_true, y_prob[:, 1])
            print(f"  ROC AUC   : {auc_score:.4f}")
        except ValueError:
            pass

    target_names = ["NORMAL (0)", "ABNORMAL (1)"] if len(np.unique(y_true)) == 2 else class_names

    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    print("=" * 60 + "\n")
    return metrics


def print_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    print("\n" + "=" * 60)
    print("  PRANA SpO2 - Model Evaluation (Regression)")
    print("=" * 60)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    print(f"\n  MAE       : {mae:.4f}")
    print(f"  RMSE      : {rmse:.4f}")
    print(f"  R² Score  : {r2:.4f}")
    print("=" * 60 + "\n")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_model_dir()

    try:
        model, meta = load_spo2_model(device=device)
        info = SpO2DatasetInfo.load(META_PATH)
        task_type = meta.get("task_type", info.task_type)

        signals, labels, _s, spo2_ref, _, _ = load_raw_dataset(auto_generate=True)
        X, y, spo2_ref, cfg = preprocess_pipeline(signals, labels, spo2_ref, info=info)
        _, _, X_test, _, _, y_test = split_train_val_test(
            X, y, test_size=cfg.test_size, val_size=cfg.val_size, random_state=cfg.random_state,
        )
    except Exception as exc:
        print(f"\n[ERROR] Evaluation setup failed: {exc}")
        sys.exit(1)

    test_loader = create_dataloader(X_test, y_test, task_type, batch_size=64, shuffle=False)
    y_pred, y_true, y_prob = collect_predictions(model, test_loader, device, task_type)

    if task_type == "regression":
        print_regression_metrics(y_true, y_pred)
    else:
        class_names = ["NORMAL", "ABNORMAL"] if meta.get("num_classes", 2) == 2 else info.class_names
        print_classification_metrics(y_true, y_pred.astype(int), y_prob, class_names)

        try:
            cm_path = plot_confusion_matrix(
                y_true.astype(int), y_pred.astype(int), class_names, CONFUSION_MATRIX_PATH,
            )
            print(f"  Confusion matrix plot saved -> {cm_path}")
        except Exception as exc:
            print(f"  [WARN] Could not save confusion matrix: {exc}")

        try:
            if y_prob.shape[1] >= 2:
                roc_path = plot_roc_curve(y_true.astype(int), y_prob, class_names, ROC_CURVE_PATH)
                if roc_path:
                    print(f"  ROC curve plot saved -> {roc_path}")
        except Exception as exc:
            print(f"  [WARN] Could not save ROC curve: {exc}")

    print("\n[Done] Evaluation complete.\n")


if __name__ == "__main__":
    main()
