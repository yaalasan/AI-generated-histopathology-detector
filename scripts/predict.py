#!/usr/bin/env python3
"""
scripts/predict.py
Run inference on a single image or a folder of images. Run from project root:
    python scripts/predict.py --image path/to/image.png
    python scripts/predict.py --folder path/to/folder/
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.model import load_model
from src.train import get_device
from src.predict import predict_single, predict_batch

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the histopathology AI detector")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model path from config")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--folder", type=str, help="Path to a folder of images")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.model:
        cfg["paths"]["model_save_path"] = args.model

    device = get_device(cfg)
    model = load_model(cfg["paths"]["model_save_path"], cfg, device)
    class_names = ["fake", "real"]
    img_size = cfg["data"]["img_size"]

    if args.image:
        result = predict_single(args.image, model, class_names, device, img_size)
        print(f"\n  Image : {args.image}")
        print(f"  Prediction : {result['label'].upper()}")
        print(f"  Confidence : {result['confidence']*100:.1f}%")
        print(f"  Probabilities:")
        for cls, prob in result["probabilities"].items():
            print(f"    {cls}: {prob*100:.1f}%")

    elif args.folder:
        folder = Path(args.folder)
        paths = [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
        if not paths:
            print(f"  No supported images found in {folder}")
            return

        results = predict_batch(paths, model, class_names, device, img_size)
        print(f"\n  Results for {len(results)} images:\n")
        print(f"  {'File':<40} {'Prediction':<10} {'Confidence'}")
        print(f"  {'-'*65}")
        for r in results:
            fname = Path(r["path"]).name
            print(f"  {fname:<40} {r['label'].upper():<10} {r['confidence']*100:.1f}%")


if __name__ == "__main__":
    main()
