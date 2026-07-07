"""
STEP 6 — Evaluate the trained EEG CNN on the held-out test set.

Computes accuracy, precision, recall, F1-score, and saves:
  - model/confusion_matrix.png
  - model/roc_curve.png

Run from project root:
    python modules/eeg/evaluate_model.py
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
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from modules.eeg.eeg_dataset import create_dataloader
from modules.eeg.eeg_inference import load_eeg_model
from modules.eeg.preprocessing import preprocess_pipeline, split_train_val_test
from modules.eeg.utils import EEGDatasetInfo, META_PATH, ensure_model_dir, load_raw_dataset

# Output paths for evaluation plots
CONFUSION_MATRIX_PATH = _MODULE / "model" / "confusion_matrix.png"
ROC_CURVE_PATH = _MODULE / "model" / "roc_curve.png"


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on a DataLoader.

    Returns
    -------
    y_pred : predicted class indices
    y_true : ground-truth labels
    y_prob : softmax probabilities, shape (N, num_classes)
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(batch_y.numpy().tolist())
        all_probs.append(probs)

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.vstack(all_probs),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Return accuracy, precision, recall, and weighted F1-score."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Path,
) -> Path:
    """Save a labelled confusion-matrix heatmap."""
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
    ax.set_title("EEG CNN — Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )

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
    """
    Save ROC curve plot.

    For binary classification: single ROC with AUC.
    For multi-class: one-vs-rest ROC per class.
    """
    n_classes = y_prob.shape[1]
    fig, ax = plt.subplots(figsize=(7, 6))

    if n_classes == 2:
        # Positive class = index 1 (ABNORMAL)
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#0F4C81", lw=2, label=f"ABNORMAL (AUC = {roc_auc:.3f})")
    else:
        # One-vs-rest ROC for each class
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
    ax.set_title("EEG CNN — ROC Curve")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def print_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """Print evaluation metrics and classification report."""
    print("\n" + "=" * 60)
    print("  PRANA EEG - Model Evaluation")
    print("=" * 60)

    metrics = compute_metrics(y_true, y_pred)
    print(f"\n  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-score  : {metrics['f1']:.4f}")

    n_classes = y_prob.shape[1]
    if n_classes == 2:
        try:
            auc_score = roc_auc_score(y_true, y_prob[:, 1])
            print(f"  ROC AUC   : {auc_score:.4f}")
        except ValueError:
            pass

    target_names = class_names if len(class_names) == len(np.unique(y_true)) else None
    if len(np.unique(y_true)) == 2:
        target_names = ["NORMAL (0)", "ABNORMAL (1)"]

    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    print("=" * 60 + "\n")
    return metrics


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_model_dir()

    try:
        model, meta = load_eeg_model(device=device)
        info = EEGDatasetInfo.load(META_PATH)

        signals, labels, _, _ = load_raw_dataset(auto_download=True)
        X, y, cfg = preprocess_pipeline(signals, labels, info=info)
        _, _, X_test, _, _, y_test = split_train_val_test(
            X, y,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            random_state=cfg.random_state,
        )
    except Exception as exc:
        print(f"\n[ERROR] Evaluation setup failed: {exc}")
        sys.exit(1)

    test_loader = create_dataloader(X_test, y_test, batch_size=64, shuffle=False)
    y_pred, y_true, y_prob = collect_predictions(model, test_loader, device)

    class_names = ["NORMAL", "ABNORMAL"] if meta.get("num_classes", 2) == 2 else info.class_names
    print_metrics(y_true, y_pred, y_prob, class_names)

    try:
        cm_path = plot_confusion_matrix(y_true, y_pred, class_names, CONFUSION_MATRIX_PATH)
        print(f"  Confusion matrix plot saved -> {cm_path}")
    except Exception as exc:
        print(f"  [WARN] Could not save confusion matrix: {exc}")

    try:
        roc_path = plot_roc_curve(y_true, y_prob, class_names, ROC_CURVE_PATH)
        if roc_path:
            print(f"  ROC curve plot saved -> {roc_path}")
    except Exception as exc:
        print(f"  [WARN] Could not save ROC curve: {exc}")

    print("\n[Done] Evaluation complete.\n")


if __name__ == "__main__":
    main()
