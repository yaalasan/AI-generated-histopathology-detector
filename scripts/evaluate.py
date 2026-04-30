#!/usr/bin/env python3
"""
scripts/evaluate.py
Run full evaluation on the test set. Run from project root:
    python scripts/evaluate.py
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.dataset import get_dataloaders
from src.model import load_model
from src.train import get_device
from src.evaluate import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the histopathology AI detector")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model path from config")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.model:
        cfg["paths"]["model_save_path"] = args.model

    print("\n" + "="*50)
    print("  Histopathology AI Detector - Evaluation")
    print("="*50)

    device = get_device(cfg)
    _, _, test_loader, class_names = get_dataloaders(cfg)
    model = load_model(cfg["paths"]["model_save_path"], cfg, device)

    run_evaluation(model, test_loader, class_names, cfg)


if __name__ == "__main__":
    main()
