"""
train.py
Core training and validation loop with early stopping and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


def get_device(cfg):
    """Return the appropriate torch device."""
    if cfg["device"]["force_cpu"]:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        torch.set_num_threads(cfg["device"]["num_threads"])

    print(f"  Using device: {device}")
    return device


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns average loss."""
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Run evaluation. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(loader), correct / total


def run_training(model, train_loader, val_loader, cfg, device):
    """
    Full training loop with early stopping and best-model checkpointing.

    Returns:
        history: dict with train_losses, val_losses, val_accuracies
    """
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["learning_rate"]
    patience = cfg["training"]["early_stopping_patience"]
    save_path = Path(cfg["paths"]["model_save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    history = {"train_losses": [], "val_losses": [], "val_accuracies": []}
    best_val_acc = 0.0
    epochs_no_improve = 0

    print(f"\n{'='*50}")
    print(f"  Starting training for up to {epochs} epochs")
    print(f"  Early stopping patience: {patience}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["val_accuracies"].append(val_acc)

        scheduler.step(val_acc)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            marker = "  <-- best saved"
        else:
            epochs_no_improve += 1

        print(
            f"  Epoch [{epoch+1:>2}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}{marker}"
        )

        if epochs_no_improve >= patience:
            print(f"\n  Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"\n  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Model saved to: {save_path}\n")
    return history
