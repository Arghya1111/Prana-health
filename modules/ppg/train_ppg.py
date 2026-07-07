"""
STEP 6 — Train the 1D CNN PPG classifier.

Run from project root (after inspect_dataset.py):
    python modules/ppg/train_ppg.py

Saves best model to modules/ppg/model/ppg_model.pth
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

from modules.ppg.ppg_dataset import create_dataloader
from modules.ppg.ppg_model import build_model
from modules.ppg.preprocessing import preprocess_pipeline, split_train_val_test
from modules.ppg.utils import (
    META_PATH,
    PPGDatasetInfo,
    ensure_model_dir,
    get_model_path,
    inspect_dataset,
    load_raw_dataset,
)

# ── Training hyper-parameters ─────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 7
MIN_DELTA = 1e-4
NUM_WORKERS = 0


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
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
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    return total_loss / total, correct / total


def plot_training_history(history: dict, save_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    info: PPGDatasetInfo,
    signal_length: int,
    num_classes: int,
    history: dict,
) -> None:
    ensure_model_dir()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "signal_length": signal_length,
            "num_classes": num_classes,
            "dataset_info": info.to_dict(),
            "history": history,
        },
        path,
    )


def main() -> None:
    print("\n" + "=" * 60)
    print("  PRANA PPG - CNN Training")
    print("=" * 60)

    device = get_device()
    print(f"\n  Device: {device}")

    try:
        if not META_PATH.exists():
            print("\n  Metadata not found - running inspection first...")
            info = inspect_dataset(auto_generate=True)
            info.save(META_PATH)
        else:
            info = PPGDatasetInfo.load(META_PATH)

        signals, labels, _subjects, _, _ = load_raw_dataset(auto_generate=True)
        X, y, cfg = preprocess_pipeline(signals, labels, info=info)
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
            X, y,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            random_state=cfg.random_state,
        )
    except Exception as exc:
        print(f"\n[ERROR] Data loading failed: {exc}")
        sys.exit(1)

    num_classes = len(np.unique(y))
    signal_length = X.shape[1]
    print(f"\n  Samples - train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
    print(f"  Signal length: {signal_length}, Classes: {num_classes}")

    train_loader = create_dataloader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(signal_length=signal_length, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
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
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:03d}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
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
        save_checkpoint(model_path, model, info, signal_length, num_classes, history)

    try:
        plot_training_history(history, plot_path)
        print(f"\n  Training curves saved -> {plot_path}")
    except Exception as exc:
        print(f"\n  [WARN] Could not save training plot: {exc}")

    print(f"\n  Best model saved -> {model_path}")
    print("\n[Done] Training complete. Run evaluate_model.py next.\n")


if __name__ == "__main__":
    main()
