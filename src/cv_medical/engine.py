from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from .losses import MultiTaskLoss
from .metrics import dice_coef, iou_score, sigmoid_to_mask
from .utils import atomic_torch_save, ensure_dir, save_json


@dataclass(frozen=True)
class EpochResult:
    loss_total: float
    dice: float
    miou: float
    cls_accuracy: float
    cls_precision: float
    cls_recall: float
    cls_f1: float
    cls_confusion_matrix: list[list[int]]


def _as_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: MultiTaskLoss,
    seg_threshold: float = 0.5,
) -> EpochResult:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_cls_true: list[int] = []
    all_cls_pred: list[int] = []
    dice_list: list[float] = []
    iou_list: list[float] = []

    for batch in loader:
        image = batch.image.to(device)
        mask = batch.mask.to(device)
        label = batch.label.to(device)

        seg_logits, cls_logits = model(image)
        losses = loss_fn(seg_logits, cls_logits, mask, label)
        total_loss += _as_float(losses["total"])
        n_batches += 1

        seg_prob = torch.sigmoid(seg_logits)
        seg_pred = sigmoid_to_mask(seg_prob, threshold=seg_threshold)
        dice_list.extend(dice_coef(seg_pred, mask).detach().cpu().tolist())
        iou_list.extend(iou_score(seg_pred, mask).detach().cpu().tolist())

        cls_prob = torch.sigmoid(cls_logits).detach().cpu().numpy().reshape(-1)
        cls_pred = (cls_prob >= 0.5).astype(int)
        cls_true = label.detach().cpu().numpy().reshape(-1).astype(int)
        all_cls_true.extend(cls_true.tolist())
        all_cls_pred.extend(cls_pred.tolist())

    y_true = np.asarray(all_cls_true, dtype=int)
    y_pred = np.asarray(all_cls_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return EpochResult(
        loss_total=float(total_loss / max(n_batches, 1)),
        dice=float(np.mean(dice_list) if dice_list else 0.0),
        miou=float(np.mean(iou_list) if iou_list else 0.0),
        cls_accuracy=float(accuracy_score(y_true, y_pred)),
        cls_precision=float(precision_score(y_true, y_pred, zero_division=0)),
        cls_recall=float(recall_score(y_true, y_pred, zero_division=0)),
        cls_f1=float(f1_score(y_true, y_pred, zero_division=0)),
        cls_confusion_matrix=cm,
    )


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    loss_fn: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    run_dir: Path,
    seg_threshold: float = 0.5,
) -> dict[str, EpochResult]:
    ensure_dir(run_dir)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")

    best_val_dice = -1.0
    history: dict[str, EpochResult] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            image = batch.image.to(device)
            mask = batch.mask.to(device)
            label = batch.label.to(device)

            seg_logits, cls_logits = model(image)
            losses = loss_fn(seg_logits, cls_logits, mask, label)
            loss = losses["total"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += _as_float(loss)
            n_batches += 1

        train_res = evaluate(model, train_loader, device, loss_fn, seg_threshold=seg_threshold)
        val_res = evaluate(model, val_loader, device, loss_fn, seg_threshold=seg_threshold)

        history[f"train_epoch_{epoch}"] = train_res
        history[f"val_epoch_{epoch}"] = val_res

        save_json(run_dir / "latest_metrics.json", {"epoch": epoch, "train": train_res.__dict__, "val": val_res.__dict__})

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": val_res.dice,
        }
        atomic_torch_save(ckpt, ckpt_dir / "last.pt")

        if val_res.dice > best_val_dice:
            best_val_dice = val_res.dice
            atomic_torch_save(ckpt, ckpt_dir / "best.pt")

    return history
