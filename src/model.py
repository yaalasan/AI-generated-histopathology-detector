"""
model.py
Builds the ResNet-based classifier.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights


def build_model(cfg):
    """
    Build and return the model based on config.
    Supports resnet18 and resnet50.
    """
    arch = cfg["model"]["architecture"]
    pretrained = cfg["model"]["pretrained"]
    freeze = cfg["model"]["freeze_backbone"]
    num_classes = cfg["model"]["num_classes"]

    print(f"\n  Building model: {arch} | pretrained={pretrained} | freeze_backbone={freeze}")

    if arch == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif arch == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported architecture: '{arch}'. Choose resnet18 or resnet50.")

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,} | Trainable: {trainable_params:,}\n")

    return model


def load_model(model_path, cfg, device):
    """Load a saved model from disk for inference."""
    model = build_model(cfg)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
