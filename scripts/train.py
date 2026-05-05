from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from cv_medical.config import ExperimentConfig
from cv_medical.data import MedicalSegClsDataset, collate_samples
from cv_medical.engine import train
from cv_medical.losses import MultiTaskLoss
from cv_medical.models.mtl_unet import MultiTaskUNet
from cv_medical.utils import get_device, save_experiment_config, set_global_seed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/mtl_unet"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        data_root=args.data_root,
        run_dir=args.run_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
    )

    set_global_seed(cfg.seed)
    device = get_device()

    train_ds = MedicalSegClsDataset(
        cfg.data_root, split="train", image_size=cfg.image_size, augment=True, seed=cfg.seed
    )
    val_ds = MedicalSegClsDataset(
        cfg.data_root, split="val", image_size=cfg.image_size, augment=False, seed=cfg.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_samples,
    )

    model = MultiTaskUNet(in_channels=1, base_channels=32).to(device)
    loss_fn = MultiTaskLoss(
        cls_weight=cfg.cls_loss_weight,
        dice_weight=cfg.dice_loss_weight,
        bce_seg_weight=cfg.bce_seg_loss_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(cfg.run_dir / "config.json", cfg)
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=cfg.epochs,
        run_dir=cfg.run_dir,
    )


if __name__ == "__main__":
    main()
