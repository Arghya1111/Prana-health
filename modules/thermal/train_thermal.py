"""
STEP 5 — Train thermal CNN classifier.
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

from modules.thermal.image_dataset import create_dataloader
from modules.thermal.preprocessing import PreprocessConfig, split_train_val_test
from modules.thermal.thermal_model import build_model
from modules.thermal.utils import META_PATH, ThermalDatasetInfo, get_model_path, inspect_dataset, load_raw_dataset

BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3
PATIENCE = 10


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def plot_history(history, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history["train_loss"]) + 1)
    a1.plot(ep, history["train_loss"], label="Train")
    a1.plot(ep, history["val_loss"], label="Val")
    a1.legend()
    a1.set_title("Loss")
    a2.plot(ep, history["train_acc"], label="Train")
    a2.plot(ep, history["val_acc"], label="Val")
    a2.legend()
    a2.set_title("Accuracy")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("\n" + "=" * 60)
    print("  PRANA Thermal - CNN Training")
    print("=" * 60)

    device = get_device()
    print(f"\n  Device: {device}")

    try:
        if not META_PATH.exists():
            info = inspect_dataset(auto_generate=True)
            info.save(META_PATH)
        else:
            info = ThermalDatasetInfo.load(META_PATH)

        images, labels, _, _ = load_raw_dataset(auto_generate=True)
        cfg = PreprocessConfig.from_dataset_info(info)
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
            images, labels, cfg.test_size, cfg.val_size, cfg.random_state,
        )
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    num_classes = len(np.unique(labels))
    print(f"\n  Train/Val/Test: {len(y_train)}/{len(y_val)}/{len(y_test)}, Classes: {num_classes}")

    train_loader = create_dataloader(X_train, y_train, cfg, BATCH_SIZE, True, augment=True)
    val_loader = create_dataloader(X_val, y_val, cfg, BATCH_SIZE, False, augment=False)

    model = build_model(cfg.image_size, 1, num_classes).to(device)
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state, stale = float("inf"), None, 0
    model_path = get_model_path()

    print(f"\n  Training up to {EPOCHS} epochs...\n")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_l, tr_a = train_epoch(model, train_loader, criterion, optimizer, device)
        va_l, va_a = validate(model, val_loader, criterion, device)
        scheduler.step(va_l)
        history["train_loss"].append(tr_l)
        history["val_loss"].append(va_l)
        history["train_acc"].append(tr_a)
        history["val_acc"].append(va_a)
        print(f"  Epoch {epoch:03d}  loss={tr_l:.4f}/{va_l:.4f}  acc={tr_a:.3f}/{va_a:.3f}  ({time.time()-t0:.1f}s)")

        if va_l < best_val - 1e-4:
            best_val, best_state, stale = va_l, deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "image_size": cfg.image_size,
                "in_channels": 1,
                "num_classes": num_classes,
                "dataset_info": info.to_dict(),
                "history": history,
            },
            model_path,
        )

    plot_history(history, _MODULE / "model" / "training_history.png")
    print(f"\n  Best model -> {model_path}")
    print("\n[Done] Run evaluate_model.py next.\n")


if __name__ == "__main__":
    main()
