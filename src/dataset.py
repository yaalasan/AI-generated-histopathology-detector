"""
dataset.py
Handles dataset loading, transforms, and train/val/test splitting.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from collections import Counter
from pathlib import Path


def get_transforms(cfg, train=True):
    """
    Build image transforms.
    Augmentation is applied only during training if enabled in config.
    """
    aug = cfg["augmentation"]
    img_size = cfg["data"]["img_size"]

    base = []

    if train and aug.get("enabled", False):
        if aug.get("horizontal_flip"):
            base.append(transforms.RandomHorizontalFlip())
        if aug.get("vertical_flip"):
            base.append(transforms.RandomVerticalFlip())
        if aug.get("rotation_degrees", 0) > 0:
            base.append(transforms.RandomRotation(aug["rotation_degrees"]))
        jitter = aug.get("color_jitter", {})
        if jitter:
            base.append(transforms.ColorJitter(
                brightness=jitter.get("brightness", 0),
                contrast=jitter.get("contrast", 0),
                saturation=jitter.get("saturation", 0),
            ))

    base += [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    return transforms.Compose(base)


def load_datasets(cfg):
    """
    Load dataset from disk and split into train / val / test.
    Returns: train_ds, val_ds, test_ds, class_names
    """
    data_dir = cfg["data"]["data_dir"]
    train_ratio = cfg["data"]["train_split"]
    val_ratio = cfg["data"]["val_split"]
    seed = cfg["data"]["seed"]

    if not Path(data_dir).exists():
        raise FileNotFoundError(
            f"Dataset directory '{data_dir}' not found. "
            "Please create dataset/fake/ and dataset/real/ and populate them."
        )

    full_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(cfg, train=False))
    class_names = full_dataset.classes

    print(f"\n{'='*50}")
    print(f"  Dataset loaded from: {data_dir}")
    print(f"  Classes: {class_names}")
    print(f"  Class to index: {full_dataset.class_to_idx}")
    labels = [y for _, y in full_dataset.samples]
    dist = Counter(labels)
    for idx, name in enumerate(class_names):
        print(f"  '{name}': {dist[idx]} images")
    print(f"  Total: {len(full_dataset)} images")
    print(f"{'='*50}\n")

    n = len(full_dataset)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    test_size = n - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Apply augmentation transforms to train split
    train_ds.dataset.transform = get_transforms(cfg, train=True)

    print(f"  Split -> Train: {train_size} | Val: {val_size} | Test: {test_size}")
    return train_ds, val_ds, test_ds, class_names


def get_dataloaders(cfg):
    """Returns DataLoaders for train, val, and test splits."""
    train_ds, val_ds, test_ds, class_names = load_datasets(cfg)
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names
