"""
STEP 6 — Train the 1D CNN SpO2 classifier.

Run from project root (after inspect_dataset.py):
    python modules/spo2/train_spo2.py

Saves best model to modules/spo2/model/spo2_model.pth
"""

from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path

_MODULE = Path(__file__).resolve().parent
if str(_MODULE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_MODULE.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.spo2.preprocessing import preprocess_pipeline, split_train_val_test
from modules.spo2.spo2_dataset import create_dataloader
from modules.spo2.spo2_model import build_model
from modules.spo2.utils import (
    META_PATH,
    SpO2DatasetInfo,
    ensure_model_dir,
    get_model_path,
    inspect_dataset,
    load_raw_dataset,
)

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 7
MIN_DELTA = 1e-4


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    task_type: str,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)

        if task_type == "regression":
            preds = logits.squeeze(-1)
            loss = criterion(preds, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            total += batch_x.size(0)
        else:
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        loss.backward()
        optimizer.step()

    metric = total_loss / total if task_type == "regression" else correct / total
    return total_loss / total, metric


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    task_type: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)

        if task_type == "regression":
            preds = logits.squeeze(-1)
            loss = criterion(preds, batch_y)
            total_loss += loss.item() * batch_x.size(0)
        else:
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()

        total += batch_x.size(0)

    metric = total_loss / total if task_type == "regression" else correct / total
    return total_loss / total, metric


def plot_training_history(history: dict, save_path: Path, task_type: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    metric_label = "Loss" if task_type == "regression" else "Accuracy"
    ax2.plot(epochs, history["train_acc"], label=f"Train {metric_label}")
    ax2.plot(epochs, history["val_acc"], label=f"Val {metric_label}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(metric_label)
    ax2.set_title(f"Training & Validation {metric_label}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    info: SpO2DatasetInfo,
    signal_length: int,
    in_channels: int,
    num_classes: int,
    history: dict,
) -> None:
    ensure_model_dir()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "signal_length": signal_length,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "task_type": info.task_type,
            "dataset_info": info.to_dict(),
            "history": history,
        },
        path,
    )


def main() -> None:
    print("\n" + "=" * 60)
    print("  PRANA SpO2 - CNN Training")
    print("=" * 60)

    device = get_device()
    print(f"\n  Device: {device}")

    try:
        if not META_PATH.exists():
            print("\n  Metadata not found - running inspection first...")
            info = inspect_dataset(auto_generate=True)
            info.save(META_PATH)
        else:
            info = SpO2DatasetInfo.load(META_PATH)

        signals, labels, _subj, spo2_ref, _, _ = load_raw_dataset(auto_generate=True)
        X, y, spo2_ref, cfg = preprocess_pipeline(signals, labels, spo2_ref, info=info)
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
            X, y,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            random_state=cfg.random_state,
        )
    except Exception as exc:
        print(f"\n[ERROR] Data loading failed: {exc}")
        sys.exit(1)

    task_type = info.task_type
    in_channels = X.shape[1]
    signal_length = X.shape[2]
    num_classes = 1 if task_type == "regression" else len(np.unique(y))

    print(f"\n  Samples - train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
    print(f"  Shape: ({in_channels}, {signal_length}), Task: {task_type}")
    if task_type == "classification":
        print(f"  Classes: {num_classes}")

    train_loader = create_dataloader(X_train, y_train, task_type, BATCH_SIZE, True)
    val_loader = create_dataloader(X_val, y_val, task_type, BATCH_SIZE, False)

    model = build_model(signal_length, in_channels, num_classes).to(device)

    if task_type == "regression":
        criterion = nn.MSELoss()
    else:
        counts = np.bincount(y_train.astype(int), minlength=num_classes).astype(np.float32)
        weights = (counts.sum() / (num_classes * np.maximum(counts, 1))).astype(np.float32)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    model_path = get_model_path()
    plot_path = _MODULE / "model" / "training_history.png"

    print(f"\n  Training for up to {EPOCHS} epochs (early stopping patience={PATIENCE})...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_metric = train_one_epoch(
            model, train_loader, criterion, optimizer, device, task_type,
        )
        val_loss, val_metric = validate(model, val_loader, criterion, device, task_type)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metric)
        history["val_acc"].append(val_metric)

        elapsed = time.time() - t0
        if task_type == "regression":
            print(
                f"  Epoch {epoch:03d}/{EPOCHS}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"({elapsed:.1f}s)"
            )
        else:
            print(
                f"  Epoch {epoch:03d}/{EPOCHS}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"train_acc={train_metric:.3f}  val_acc={val_metric:.3f}  "
                f"({elapsed:.1f}s)"
            )

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        save_checkpoint(model_path, model, info, signal_length, in_channels, num_classes, history)

    try:
        plot_training_history(history, plot_path, task_type)
        print(f"\n  Training curves saved -> {plot_path}")
    except Exception as exc:
        print(f"\n  [WARN] Could not save training plot: {exc}")

    print(f"\n  Best model saved -> {model_path}")
    print("\n[Done] Training complete. Run evaluate_model.py next.\n")


if __name__ == "__main__":
    main()
