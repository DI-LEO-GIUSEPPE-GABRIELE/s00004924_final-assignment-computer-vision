from __future__ import annotations
import numpy as np
import cv2

"""
Downloads and prepares the Montgomery County TB chest X-ray dataset into the project format.

Source: National Library of Medicine public repository.
Output format per split:
- images/*.png
- masks/*.png (union of left/right lung masks)
- labels.csv (id,label) where label is derived from filename suffix: _0 normal, _1 abnormal
"""

from pathlib import Path
import csv
import re
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


BASE_URL = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"


def _open_url(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _fetch_filenames(index_url: str) -> list[str]:
    """Parses a directory index page and extracts MCUCXR_*.png filenames."""

    html = _open_url(index_url, timeout=60).decode("utf-8", errors="replace")
    names = re.findall(r"(MCUCXR_\d{4}_[01]\.png)", html)
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _download(url: str, out_path: Path) -> None:
    """Downloads a URL to disk if the destination path does not already exist."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    out_path.write_bytes(_open_url(url, timeout=120))


def _label_from_id(sample_id: str) -> int:
    """Returns 0/1 label from the sample id suffix."""

    if sample_id.endswith("_1"):
        return 1
    if sample_id.endswith("_0"):
        return 0
    raise ValueError(f"Unexpected id format: {sample_id}")


def _union_masks(left_path: Path, right_path: Path) -> np.ndarray:
    """Builds a single binary lung mask from left+right lung masks."""

    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise RuntimeError("Failed to read mask images")
    left_bin = (left > 0).astype(np.uint8)
    right_bin = (right > 0).astype(np.uint8)
    union = ((left_bin | right_bin) * 255).astype(np.uint8)
    return union


def _write_split(root: Path, split: str, items: list[str]) -> None:
    """Writes images/masks/labels.csv for a split using already-downloaded raw files."""

    split_dir = root / split
    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for sample_id in items:
        src_img = root / "_raw" / "CXR_png" / f"{sample_id}.png"
        src_left = root / "_raw" / "ManualMask" / \
            "leftMask" / f"{sample_id}.png"
        src_right = root / "_raw" / "ManualMask" / \
            "rightMask" / f"{sample_id}.png"

        img_dst = images_dir / f"{sample_id}.png"
        mask_dst = masks_dir / f"{sample_id}.png"

        img_dst.write_bytes(src_img.read_bytes())
        cv2.imwrite(str(mask_dst), _union_masks(src_left, src_right))

        rows.append({"id": sample_id, "label": str(_label_from_id(sample_id))})

    with (split_dir / "labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerows(rows)


def prepare(out_root: Path, seed: int, max_samples: int | None, train_ratio: float, val_ratio: float) -> None:
    """Downloads a stratified subset (optional) and creates train/val/test splits."""

    raw_root = out_root / "_raw"
    cxr_dir = raw_root / "CXR_png"
    left_dir = raw_root / "ManualMask" / "leftMask"
    right_dir = raw_root / "ManualMask" / "rightMask"

    cxr_dir.mkdir(parents=True, exist_ok=True)
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    cxr_index = f"{BASE_URL}/CXR_png/index.html"
    left_index = f"{BASE_URL}/ManualMask/leftMask/index.html"
    right_index = f"{BASE_URL}/ManualMask/rightMask/index.html"

    image_files = _fetch_filenames(cxr_index)
    left_files = set(_fetch_filenames(left_index))
    right_files = set(_fetch_filenames(right_index))

    sample_ids = [
        Path(f).stem for f in image_files if f in left_files and f in right_files]
    if not sample_ids:
        raise RuntimeError("No samples found from remote index")

    rng = np.random.default_rng(seed)
    pos = [s for s in sample_ids if _label_from_id(s) == 1]
    neg = [s for s in sample_ids if _label_from_id(s) == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    if max_samples is not None and max_samples > 0:
        pos_k = int(round(max_samples * (len(pos) / max(len(sample_ids), 1))))
        neg_k = max_samples - pos_k
        pos = pos[:pos_k]
        neg = neg[:neg_k]

    def split_group(group: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(group)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        train_g = group[:n_train]
        val_g = group[n_train: n_train + n_val]
        test_g = group[n_train + n_val:]
        return train_g, val_g, test_g

    pos_train, pos_val, pos_test = split_group(pos)
    neg_train, neg_val, neg_test = split_group(neg)

    train_ids = pos_train + neg_train
    val_ids = pos_val + neg_val
    test_ids = pos_test + neg_test

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    for sample_id in train_ids + val_ids + test_ids:
        img_url = f"{BASE_URL}/CXR_png/{sample_id}.png"
        left_url = f"{BASE_URL}/ManualMask/leftMask/{sample_id}.png"
        right_url = f"{BASE_URL}/ManualMask/rightMask/{sample_id}.png"

        _download(img_url, cxr_dir / f"{sample_id}.png")
        _download(left_url, left_dir / f"{sample_id}.png")
        _download(right_url, right_dir / f"{sample_id}.png")

    _write_split(out_root, "train", train_ids)
    _write_split(out_root, "val", val_ids)
    _write_split(out_root, "test", test_ids)


def main() -> None:
    """CLI entrypoint."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/montgomery"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    prepare(
        out_root=args.out,
        seed=args.seed,
        max_samples=(args.max_samples if args.max_samples > 0 else None),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
