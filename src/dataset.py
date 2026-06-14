"""
src/dataset.py — PyTorch Dataset with lazy loading and augmentation.

Why this instead of load_images()?
  - load_images() loads everything into RAM at once (~GBs for large datasets)
  - This loads images on-demand (one at a time during training)
  - Augmentation runs live so every epoch sees slightly different images
  - Works with torch DataLoader for batching, shuffling, multiprocessing

Usage:
    from src.dataset import AlzheimerDataset, get_dataloaders

    train_loader, val_loader, test_loader = get_dataloaders("combined_images")

    for images, labels in train_loader:
        # images: (batch, 1, 128, 128) float tensor
        # labels: (batch,) long tensor
        ...
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 128

CLASS_MAP = {
    "NonDemented": 0,
    "VeryMildDemented": 1,
    "MildDemented": 2,
    "ModerateDemented": 3,
}


# ── Augmentation helpers ──────────────────────────────────────────────────────

def random_horizontal_flip(img: np.ndarray, p: float = 0.5) -> np.ndarray:
    if random.random() < p:
        return cv2.flip(img, 1)
    return img


def random_brightness(img: np.ndarray, low: float = 0.85, high: float = 1.15) -> np.ndarray:
    factor = random.uniform(low, high)
    return np.clip(img * factor, 0.0, 1.0)


def random_rotation(img: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    angle = random.uniform(-max_angle, max_angle)
    h, w = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def augment(img: np.ndarray) -> np.ndarray:
    """Apply all augmentations. img should be float32 in [0, 1]."""
    img = random_horizontal_flip(img)
    img = random_brightness(img)
    img = random_rotation(img)
    return img


# ── Dataset class ─────────────────────────────────────────────────────────────

class AlzheimerDataset(Dataset):
    """
    Lazy-loading dataset for Alzheimer's MRI classification.

    Args:
        dataset_path: path to folder containing class subfolders
        augment_data: if True, applies random augmentation per sample
    """

    def __init__(self, dataset_path: str, augment_data: bool = False):
        self.augment_data = augment_data
        self.samples: list[tuple[str, int]] = []  # (img_path, label)

        for class_name, label in CLASS_MAP.items():
            folder = Path(dataset_path) / class_name
            if not folder.exists():
                print(f"Warning: Missing folder: {folder}")
                continue
            for filename in sorted(folder.iterdir()):
                if filename.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((str(filename), label))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {dataset_path}")

        print(f"Dataset: {len(self.samples)} images across {len(CLASS_MAP)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Could not read image: {img_path}")

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0

        if self.augment_data:
            img = augment(img)

        # Shape: (1, H, W) — channel-first for PyTorch
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor


# ── Convenience: get train/val/test loaders ───────────────────────────────────

def get_dataloaders(
    dataset_path: str,
    batch_size: int = 32,
    val_size: float = 0.1,
    test_size: float = 0.2,
    num_workers: int = 2,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    Augmentation is only applied to the training set.
    """
    # Load full dataset (no augmentation yet — split first)
    full_dataset = AlzheimerDataset(dataset_path, augment_data=False)
    labels = [s[1] for s in full_dataset.samples]
    indices = list(range(len(full_dataset)))

    # Split indices: train+val / test
    train_val_idx, test_idx, train_val_labels, _ = train_test_split(
        indices, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Split train+val into train / val
    relative_val = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        random_state=random_state,
        stratify=train_val_labels,
    )

    # Build augmented training dataset separately
    train_dataset = AlzheimerDataset(dataset_path, augment_data=True)

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"Split — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_loader, val_loader, test_loader