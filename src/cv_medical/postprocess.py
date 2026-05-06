from __future__ import annotations

"""
Post-processing utilities for binary segmentation masks.

The goal is to refine raw probability maps into cleaner binary masks using:
- thresholding
- morphological opening/closing
- largest connected component selection (optional)
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PostProcessConfig:
    """Configuration for postprocess_mask."""

    threshold: float = 0.5
    open_kernel: int = 3
    close_kernel: int = 5
    keep_largest_component: bool = True


def postprocess_mask(prob_mask: np.ndarray, cfg: PostProcessConfig) -> np.ndarray:
    """Converts a probability mask into a binary mask with morphology-based refinement."""

    if prob_mask.dtype != np.float32:
        prob_mask = prob_mask.astype(np.float32)
    mask = (prob_mask >= cfg.threshold).astype(np.uint8) * 255

    if cfg.open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if cfg.close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if cfg.keep_largest_component:
        n, labels = cv2.connectedComponents(mask)
        if n > 1:
            areas = np.bincount(labels.reshape(-1))
            areas[0] = 0
            keep = int(areas.argmax())
            mask = (labels == keep).astype(np.uint8) * 255

    return (mask > 0).astype(np.uint8)
