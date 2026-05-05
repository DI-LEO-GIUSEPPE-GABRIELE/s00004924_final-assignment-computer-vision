from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path
    run_dir: Path
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 2
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    cls_loss_weight: float = 1.0
    dice_loss_weight: float = 1.0
    bce_seg_loss_weight: float = 1.0
