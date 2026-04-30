"""
evaluate.py
Runs full evaluation on the test set and saves metrics + plots.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
)
import torch.nn.functional as F


def predict_loader(model, loader, device):
    """Run inference on a DataLoader. Returns (all_preds, all_labels, all_probs)."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def save_confusion_matrix(labels, preds, class_names, save_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title("Confusion Matrix - Real vs AI-generated Histopathology")

    path = Path(save_dir) / "confusion_matrix.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_training_curves(history, save_dir):
    """Plot and save training loss and validation accuracy curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_losses"], marker="o", color="steelblue")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_losses"], marker="o", color="orange")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["val_accuracies"], marker="o", color="green")
    axes[2].set_title("Validation Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(save_dir) / "training_curves.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def save_roc_curve(labels, probs, class_names, save_dir):
    """Plot and save ROC curve for the positive class (index 1 = real)."""
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    path = Path(save_dir) / "roc_curve.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return roc_auc


def run_evaluation(model, test_loader, class_names, cfg):
    """
    Full evaluation on test set. Prints classification report,
    saves confusion matrix, ROC curve.
    """
    from src.train import get_device
    device = get_device(cfg)
    save_dir = cfg["paths"]["results_dir"]
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print("  Running evaluation on test set...")
    print(f"{'='*50}")

    preds, labels, probs = predict_loader(model, test_loader, device)

    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print("\n  Classification Report:")
    print(report)

    # Save report to file
    report_path = Path(save_dir) / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    save_confusion_matrix(labels, preds, class_names, save_dir)
    roc_auc = save_roc_curve(labels, probs, class_names, save_dir)
    print(f"\n  Test AUC: {roc_auc:.4f}")
