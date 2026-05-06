from __future__ import annotations

"""
Single-image inference for the multi-task model.

Outputs:
- predicted classification probability
- post-processed binary mask
- overlay visualization
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import torch
from PIL import Image

from cv_medical.models.mtl_unet import MultiTaskUNet
from cv_medical.postprocess import PostProcessConfig, postprocess_mask
from cv_medical.utils import get_device


def _preprocess(image_path: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int], np.ndarray]:
    img = Image.open(image_path).convert("L")
    orig = np.array(img, dtype=np.uint8)
    h, w = orig.shape[:2]
    x = cv2.medianBlur(orig, 3)
    x = cv2.resize(x, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32) / 255.0
    x = (x - float(x.mean())) / float(x.std() + 1e-6)
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    return t, (h, w), orig


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/montgomery/checkpoints/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/montgomery/infer"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = get_device()
    model = MultiTaskUNet(in_channels=1, base_channels=32).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x, (h, w), orig = _preprocess(args.image, args.image_size)
    with torch.no_grad():
        seg_logits, cls_logit = model(x.to(device))
        seg_prob = torch.sigmoid(seg_logits).detach().cpu().numpy()[0, 0]
        cls_prob = float(torch.sigmoid(cls_logit).detach().cpu().numpy().reshape(-1)[0])

    cfg = PostProcessConfig(threshold=args.threshold)
    mask_bin = postprocess_mask(seg_prob.astype(np.float32), cfg)
    mask_u8 = (mask_bin * 255).astype(np.uint8)
    mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mask = out_dir / f"{args.image.stem}_mask.png"
    out_overlay = out_dir / f"{args.image.stem}_overlay.png"

    overlay = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    overlay[mask_u8 > 0] = (0, 0, 255)
    cv2.imwrite(str(out_mask), mask_u8)
    cv2.imwrite(str(out_overlay), overlay)

    print({"cls_prob": cls_prob, "mask_path": str(out_mask), "overlay_path": str(out_overlay)})


if __name__ == "__main__":
    main()
