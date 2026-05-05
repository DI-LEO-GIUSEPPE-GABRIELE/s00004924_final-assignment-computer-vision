from __future__ import annotations

import numpy as np
import torch


def sigmoid_to_mask(prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (prob >= threshold).to(dtype=torch.float32)


def dice_coef(pred_mask: torch.Tensor, true_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred_mask.reshape(pred_mask.shape[0], -1)
    true = true_mask.reshape(true_mask.shape[0], -1)
    inter = (pred * true).sum(dim=1)
    union = pred.sum(dim=1) + true.sum(dim=1)
    return (2.0 * inter + eps) / (union + eps)


def iou_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred_mask.reshape(pred_mask.shape[0], -1)
    true = true_mask.reshape(true_mask.shape[0], -1)
    inter = (pred * true).sum(dim=1)
    union = pred.sum(dim=1) + true.sum(dim=1) - inter
    return (inter + eps) / (union + eps)


def binary_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = y_true.astype(int).reshape(-1)
    y_pred = y_pred.astype(int).reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
