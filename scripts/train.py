#!/usr/bin/env python3
"""
scripts/train.py
Entry point for training. Run from project root:
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.dataset import get_dataloaders
from src.model import build_model
from src.train import get_device, run_training
from src.evaluate import save_training_curves, save_confusion_matrix, predict_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Train the histopathology AI detector")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to YAML config file")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    print("\n" + "="*50)
    print("  Histopathology AI Detector - Training")
    print("="*50)

    device = get_device(cfg)
    train_loader, val_loader, test_loader, class_names = get_dataloaders(cfg)
    model = build_model(cfg)
    model = model.to(device)

    history = run_training(model, train_loader, val_loader, cfg, device)

    results_dir = cfg["paths"]["results_dir"]
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    save_training_curves(history, results_dir)

    from src.model import load_model
    best_model = load_model(cfg["paths"]["model_save_path"], cfg, device)
    preds, labels, _ = predict_loader(best_model, val_loader, device)
    save_confusion_matrix(labels, preds, class_names, results_dir)

    print("\n  Training complete. Run evaluation with:")
    print("    python scripts/evaluate.py\n")


if __name__ == "__main__":
    main()
