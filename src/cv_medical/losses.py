from __future__ import annotations

"""
Loss functions for segmentation + classification multi-task learning.

- Classification: BCEWithLogitsLoss on a single logit per image.
- Segmentation: BCEWithLogitsLoss + Soft Dice loss on a single-channel mask.
"""

import torch
from torch import nn


class SoftDiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation using sigmoid probabilities."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes 1 - Dice(logits, targets) averaged over batch."""

        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        inter = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class MultiTaskLoss(nn.Module):
    """Weighted combination of classification BCE and segmentation BCE + Dice."""

    def __init__(
        self,
        cls_weight: float,
        dice_weight: float,
        bce_seg_weight: float,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.dice_weight = dice_weight
        self.bce_seg_weight = bce_seg_weight
        self.cls_bce = nn.BCEWithLogitsLoss()
        self.seg_bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()

    def forward(
        self,
        seg_logits: torch.Tensor,
        cls_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        cls_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Returns a dict with total loss and detached per-component losses."""

        cls_loss = self.cls_bce(cls_logits, cls_targets)
        seg_bce_loss = self.seg_bce(seg_logits, seg_targets)
        seg_dice_loss = self.dice(seg_logits, seg_targets)
        total = (
            self.cls_weight * cls_loss
            + self.bce_seg_weight * seg_bce_loss
            + self.dice_weight * seg_dice_loss
        )
        return {
            "total": total,
            "cls": cls_loss.detach(),
            "seg_bce": seg_bce_loss.detach(),
            "seg_dice": seg_dice_loss.detach(),
        }
