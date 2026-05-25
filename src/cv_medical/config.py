from __future__ import annotations

"""
Configuration objects used by the training and evaluation scripts.

The goal is to keep all hyperparameters in a single, serializable place so that
each run can be reproduced and audited via runs/<run>/config.json.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters and paths for one experiment run."""

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
