"""
predict.py
Single-image and batch inference utilities.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path


def get_inference_transform(img_size=224):
    """Return the standard inference transform (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict_single(image_path, model, class_names, device, img_size=224):
    """
    Run inference on a single image file.

    Args:
        image_path: path to image (str or Path)
        model: loaded PyTorch model in eval mode
        class_names: list of class label strings
        device: torch.device
        img_size: resize target

    Returns:
        dict with keys: label, confidence, probabilities
    """
    transform = get_inference_transform(img_size)

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).squeeze()
        pred_idx = out.argmax(dim=1).item()

    return {
        "label": class_names[pred_idx],
        "confidence": probs[pred_idx].item(),
        "probabilities": {name: probs[i].item() for i, name in enumerate(class_names)},
    }


def predict_batch(image_paths, model, class_names, device, img_size=224):
    """
    Run inference on a list of image paths.

    Returns:
        list of result dicts (same format as predict_single)
    """
    results = []
    for path in image_paths:
        result = predict_single(path, model, class_names, device, img_size)
        result["path"] = str(path)
        results.append(result)
    return results
