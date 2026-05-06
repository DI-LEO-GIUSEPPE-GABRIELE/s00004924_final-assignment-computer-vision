from __future__ import annotations

"""
Dataset utilities for a joint segmentation + classification task.

Expected on-disk format per split:
- images/*.png (grayscale or RGB, loaded as grayscale)
- masks/*.png (binary mask, loaded as grayscale then thresholded)
- labels.csv with columns: id,label where id matches the image/mask stem.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    """A single sample returned by the dataset and used by the training loop."""

    image: torch.Tensor
    mask: torch.Tensor
    label: torch.Tensor
    sample_id: str


def _read_labels_csv(path: Path) -> dict[str, int]:
    """Reads a split labels.csv into a dict mapping sample_id -> int label (0/1)."""

    labels: dict[str, int] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row["id"])
            labels[sample_id] = int(row["label"])
    return labels


def _load_grayscale(path: Path) -> np.ndarray:
    """Loads an image file and converts it to uint8 grayscale."""

    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def _preprocess_image_u8(img_u8: np.ndarray) -> np.ndarray:
    """Applies simple denoising in uint8 space."""

    img_u8 = cv2.medianBlur(img_u8, 3)
    return img_u8


def _resize(img: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    """Resizes an image/mask to a square size using an interpolation suitable for each type."""

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(img, (size, size), interpolation=interp)


def _random_augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Applies paired geometric augmentation (flip/rotate) to keep image and mask aligned."""

    if rng.random() < 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask)
    if rng.random() < 0.5:
        img = np.flipud(img)
        mask = np.flipud(mask)

    k = int(rng.integers(0, 4))
    if k:
        img = np.rot90(img, k)
        mask = np.rot90(mask, k)
    return img.copy(), mask.copy()


class MedicalSegClsDataset(Dataset[Sample]):
    """PyTorch Dataset for paired image+mask with an additional binary label."""

    def __init__(
        self,
        root: Path,
        split: str,
        image_size: int,
        augment: bool,
        seed: int,
    ) -> None:
        self.root = root
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        split_dir = root / split
        self.images_dir = split_dir / "images"
        self.masks_dir = split_dir / "masks"
        self.labels_path = split_dir / "labels.csv"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing images dir: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Missing masks dir: {self.masks_dir}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Missing labels csv: {self.labels_path}")

        self.labels = _read_labels_csv(self.labels_path)

        image_paths = sorted(self.images_dir.glob("*.png"))
        self.sample_ids = [p.stem for p in image_paths if (
            self.masks_dir / p.name).exists() and p.stem in self.labels]
        if not self.sample_ids:
            raise RuntimeError(f"No samples found in {split_dir}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Sample:
        """Loads and preprocesses a single sample (image normalization is per-image)."""

        sample_id = self.sample_ids[idx]
        img_path = self.images_dir / f"{sample_id}.png"
        mask_path = self.masks_dir / f"{sample_id}.png"

        img_u8 = _load_grayscale(img_path)
        mask_u8 = _load_grayscale(mask_path)

        img_u8 = _preprocess_image_u8(img_u8)
        img_u8 = _resize(img_u8, self.image_size, is_mask=False)
        mask_u8 = _resize(mask_u8, self.image_size, is_mask=True)

        mask_bin = (mask_u8 > 127).astype(np.uint8)

        if self.augment:
            img_u8, mask_bin = _random_augment(img_u8, mask_bin, self.rng)

        img_f32 = img_u8.astype(np.float32) / 255.0
        mean = float(img_f32.mean())
        std = float(img_f32.std() + 1e-6)
        img_f32 = (img_f32 - mean) / std

        image = torch.from_numpy(img_f32).unsqueeze(0)
        mask = torch.from_numpy(mask_bin.astype(np.float32)).unsqueeze(0)
        label = torch.tensor(
            [float(self.labels[sample_id])], dtype=torch.float32)

        return Sample(image=image, mask=mask, label=label, sample_id=sample_id)


def collate_samples(batch: list[Sample]) -> Sample:
    """Batches Samples into a single Sample containing stacked tensors."""

    images = torch.stack([b.image for b in batch], dim=0)
    masks = torch.stack([b.mask for b in batch], dim=0)
    labels = torch.stack([b.label for b in batch], dim=0)
    sample_ids = [b.sample_id for b in batch]
    return Sample(image=images, mask=masks, label=labels, sample_id=",".join(sample_ids))
