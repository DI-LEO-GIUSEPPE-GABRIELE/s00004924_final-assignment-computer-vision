from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


def _draw_lesion(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    h, w = img.shape[:2]
    center = (int(rng.integers(w * 0.2, w * 0.8)), int(rng.integers(h * 0.2, h * 0.8)))
    axes = (int(rng.integers(w * 0.05, w * 0.25)), int(rng.integers(h * 0.05, h * 0.25)))
    angle = float(rng.integers(0, 180))
    intensity = int(rng.integers(60, 200))
    cv2.ellipse(img, center, axes, angle, 0, 360, intensity, thickness=-1)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, thickness=-1)
    area = int(np.count_nonzero(mask))
    return area


def _make_sample(size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
    img = rng.normal(loc=110, scale=18, size=(size, size)).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    num_lesions = int(rng.integers(1, 4))
    total_area = 0
    for _ in range(num_lesions):
        total_area = max(total_area, _draw_lesion(img, mask, rng))

    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.add(img, rng.normal(0, 6, size=img.shape).astype(np.int16), dtype=cv2.CV_8U)

    label = int((total_area / (size * size)) > 0.08 or num_lesions >= 3)
    return img, mask, label


def generate_dataset(out_root: Path, n_train: int, n_val: int, n_test: int, size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    splits = [("train", n_train), ("val", n_val), ("test", n_test)]

    for split, n in splits:
        split_dir = out_root / split
        images_dir = split_dir / "images"
        masks_dir = split_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, str]] = []
        for i in range(n):
            sample_id = f"{split}_{i:05d}"
            img, mask, label = _make_sample(size=size, rng=rng)
            cv2.imwrite(str(images_dir / f"{sample_id}.png"), img)
            cv2.imwrite(str(masks_dir / f"{sample_id}.png"), mask)
            rows.append({"id": sample_id, "label": str(label)})

        with (split_dir / "labels.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "label"])
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--val", type=int, default=50)
    parser.add_argument("--test", type=int, default=50)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(args.out, args.train, args.val, args.test, size=args.size, seed=args.seed)


if __name__ == "__main__":
    main()
