from __future__ import annotations

"""
Evaluation script for the multi-task segmentation + classification model.

It loads a checkpoint, runs evaluation on the selected split, and writes a JSON
compatible with scripts/make_report.py.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from cv_medical.data import MedicalSegClsDataset, collate_samples
from cv_medical.engine import evaluate
from cv_medical.losses import MultiTaskLoss
from cv_medical.models.mtl_unet import MultiTaskUNet
from cv_medical.utils import get_device, save_json


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/montgomery"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/montgomery/checkpoints/best.pt"))
    parser.add_argument("--out", type=Path, default=Path("runs/montgomery/eval_test.json"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = get_device()
    ds = MedicalSegClsDataset(args.data_root, split=args.split, image_size=args.image_size, augment=False, seed=0)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_samples,
    )

    model = MultiTaskUNet(in_channels=1, base_channels=32).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    loss_fn = MultiTaskLoss(cls_weight=1.0, dice_weight=1.0, bce_seg_weight=1.0)
    res = evaluate(model, loader, device, loss_fn)

    payload = {"split": args.split, "checkpoint": str(args.checkpoint), **res.__dict__}
    save_json(args.out, payload)
    print(payload)


if __name__ == "__main__":
    main()
