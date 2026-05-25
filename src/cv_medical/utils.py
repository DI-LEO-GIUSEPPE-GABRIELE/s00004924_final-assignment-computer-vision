from __future__ import annotations

"""
Small utilities used across the project (reproducibility, device selection, IO).

These helpers keep the scripts thin and make runs easier to reproduce:
- set_global_seed(): consistent randomness across numpy/torch
- get_device(): chooses CUDA (Colab) / MPS (Apple Silicon) / CPU
- atomic_torch_save(): avoids partially-written checkpoints
"""

import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Sets random seeds for python/numpy/torch for better reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Selects the best available device in priority order: CUDA -> MPS -> CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: Path) -> Path:
    """Creates a directory (and parents) if missing and returns the path."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Writes a JSON file with stable formatting (sorted keys, indentation)."""

    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def save_experiment_config(path: Path, cfg: Any) -> None:
    """Serializes a dataclass config to JSON, stringifying Path fields."""

    d = asdict(cfg)
    d["data_root"] = str(d["data_root"])
    d["run_dir"] = str(d["run_dir"])
    save_json(path, d)


def atomic_torch_save(obj: Any, path: Path) -> None:
    """Atomically saves a PyTorch object by writing a temp file and renaming it."""

    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)
