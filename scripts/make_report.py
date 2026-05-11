from __future__ import annotations

"""
Generates a short Technical Analysis PDF (max 10 pages) from an evaluation JSON.

Input: JSON produced by scripts/evaluate.py
Output: a PDF containing problem statement, methodology, results, failure analysis, and ethics.
"""

from pathlib import Path

import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def _wrap_lines(lines: list[str], width: int = 105) -> list[str]:
    out: list[str] = []
    for line in lines:
        if not line.strip():
            out.append("")
            continue
        out.extend(textwrap.wrap(line, width=width,
                   break_long_words=False, break_on_hyphens=False))
    return out


def _page_text(pdf: PdfPages, title: str, lines: list[str]) -> None:
    """Adds a text-only page to the PDF."""

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(title, fontsize=18, y=0.97)
    wrapped = _wrap_lines(lines, width=105)
    top_y = 0.92
    bottom_y = 0.06
    usable = max(top_y - bottom_y, 1e-6)
    step = min(0.03, usable / max(len(wrapped), 1))
    y = top_y
    for line in wrapped:
        fig.text(0.07, y, line, fontsize=11, va="top")
        y -= step
    pdf.savefig(fig)
    plt.close(fig)


def _page_metrics_table(pdf: PdfPages, title: str, metrics: dict[str, float]) -> None:
    """Adds a metrics table page to the PDF."""

    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_subplot(111)
    ax.axis("off")
    fig.suptitle(title, fontsize=18, y=0.97)

    rows = sorted(metrics.items(), key=lambda kv: kv[0])
    cell_text = [[k, f"{v:.4f}"] for k, v in rows]
    table = ax.table(cellText=cell_text, colLabels=[
                     "Metric", "Value"], cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    pdf.savefig(fig)
    plt.close(fig)


def _page_confusion_matrix(pdf: PdfPages, title: str, cm: list[list[int]]) -> None:
    """Adds a confusion matrix visualization page to the PDF."""

    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_subplot(111)
    fig.suptitle(title, fontsize=18, y=0.97)

    m = np.asarray(cm, dtype=float)
    im = ax.imshow(m, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, str(int(v)), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    """CLI entrypoint."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--out-pdf", type=Path,
                        default=Path("Technical_Analysis.pdf"))
    parser.add_argument("--project-title", type=str,
                        default="Medical Image Analysis Tool (Segmentation + Classification)")
    parser.add_argument(
        "--dataset-note",
        type=str,
        default="Dataset: Montgomery County Chest X-ray (NLM) with lung masks; TB label from filename suffix (0=normal, 1=abnormal).",
    )
    args = parser.parse_args()

    payload = json.loads(args.eval_json.read_text())
    cm = payload.get("cls_confusion_matrix", [[0, 0], [0, 0]])

    scalar_metrics = {
        "loss_total": float(payload.get("loss_total", 0.0)),
        "seg_dice": float(payload.get("dice", 0.0)),
        "seg_miou": float(payload.get("miou", 0.0)),
        "cls_accuracy": float(payload.get("cls_accuracy", 0.0)),
        "cls_precision": float(payload.get("cls_precision", 0.0)),
        "cls_recall": float(payload.get("cls_recall", 0.0)),
        "cls_f1": float(payload.get("cls_f1", 0.0)),
    }

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out_pdf) as pdf:
        _page_text(
            pdf,
            args.project_title,
            [
                "Technical Analysis Document (max 10 pages)",
                "",
                "Problem Statement",
                "This project targets a real-world clinical workflow: automated screening support for chest X-rays.",
                "The system performs two tasks jointly:",
                "- Segmentation: delineate the lung fields (binary mask).",
                "- Classification: predict a binary label (0/1) related to TB findings.",
                "",
                "Why this problem is relevant",
                "- Lung segmentation is a common pre-processing step that reduces background clutter and improves robustness.",
                "- Binary screening helps triage large volumes of studies and prioritize suspicious cases.",
                "- A combined segmentation+classification model allows feature sharing and provides a visual explanation (mask).",
                "",
                "Scope and intended use",
                "- This is a decision-support prototype, not a diagnostic device.",
                "- Output should be reviewed by qualified clinical staff.",
                "- Performance can change under domain shift (hospital/site/device/protocol).",
                "",
                args.dataset_note,
                f"Evaluation split: {payload.get('split', 'unknown')}",
                f"Checkpoint: {payload.get('checkpoint', 'unknown')}",
            ],
        )

        _page_text(
            pdf,
            "Methodology",
            [
                "End-to-end pipeline",
                "",
                "1) Data acquisition and preprocessing",
                "- Images are loaded as grayscale and denoised with a median filter (robust to salt-and-pepper artifacts).",
                "- Each image and mask is resized to a fixed square resolution for batching.",
                "- Intensity normalization is applied per image: (x - mean) / std to reduce scanner/exposure variance.",
                "- Data augmentation is applied only in training: random flips and 90-degree rotations to improve invariance.",
                "",
                "2) Feature engineering / representation",
                "- Learned representation with a CNN backbone (UNet encoder-decoder).",
                "- Shared encoder features support both pixel-wise segmentation and image-level classification.",
                "",
                "3) Core model logic (multi-task learning)",
                "- Segmentation head outputs a single-channel logit map (lung vs background).",
                "- Classification head uses global average pooling on bottleneck features + a linear layer to output 1 logit.",
                "",
                "4) Optimization",
                "- Loss = BCEWithLogits (classification) + BCEWithLogits (segmentation) + Soft Dice loss (segmentation).",
                "- Optimizer: AdamW with weight decay to reduce overfitting.",
                "",
                "5) Post-processing (segmentation)",
                "- Probability thresholding converts logits to a binary mask.",
                "- Morphological opening removes small spurious regions; closing fills small holes.",
                "- Largest connected component selection enforces a single dominant lung region when appropriate.",
                "",
                "Implementation stack",
                "- Python, OpenCV (pre/post-processing), PyTorch (model/training), scikit-learn (classification metrics).",
            ],
        )

        _page_metrics_table(
            pdf, "Experimental Results (Table)", scalar_metrics)
        _page_confusion_matrix(
            pdf, "Experimental Results (Confusion Matrix)", cm)

        _page_text(
            pdf,
            "Failure Analysis",
            [
                "Observed and expected failure modes",
                "",
                "Segmentation failures",
                "- Poor contrast or strong acquisition artifacts may cause under-segmentation (missing peripheral lung).",
                "- Strong field-of-view differences (cropped images) may shift the anatomy and degrade masks.",
                "- Very bright/opaque regions can confuse boundaries and create holes or false regions.",
                "",
                "Classification failures",
                "- Class imbalance can lead to high recall but low specificity (more false positives).",
                "- Domain shift (different hospital/site/device) can change texture statistics and reduce accuracy.",
                "- Label noise: binary TB label is weak supervision and may not perfectly match visible findings.",
                "",
                "Mitigations",
                "- Expand augmentation (contrast/brightness) and use consistent intensity preprocessing.",
                "- Tune the segmentation threshold and morphology kernels per dataset resolution.",
                "- Calibrate classification threshold on validation data (optimize F1 or balanced accuracy).",
                "- Fine-tune on a small labeled set from the target domain; validate cross-site when possible.",
                "",
                "What to report during the oral exam",
                "- Show a few qualitative examples: correct masks, typical failures, and how post-processing changes results.",
                "- Explain why Dice/mIoU and F1/confusion matrix are appropriate for this task.",
            ],
        )

        _page_text(
            pdf,
            "Ethical Considerations",
            [
                "Privacy and data handling",
                "- Use de-identified data and avoid storing patient metadata in logs or filenames.",
                "- Restrict access to raw data and follow institutional policies for medical data handling.",
                "",
                "Bias, fairness, and generalization",
                "- Performance may vary across demographic groups and acquisition devices/protocols.",
                "- Report stratified metrics when subgroup information is available.",
                "- Monitor false negative rate carefully due to the clinical cost of missed cases.",
                "",
                "Clinical safety and limitations",
                "- This tool is intended for decision support and requires human oversight.",
                "- Communicate uncertainty: classification probabilities are not diagnoses.",
                "- Document known failure cases and avoid over-claiming model capability.",
                "",
                "Reproducibility",
                "- Fix random seeds for comparable experiments and keep evaluation protocols consistent.",
                "- Track configuration, checkpoints, and metrics files to enable auditability.",
            ],
        )


if __name__ == "__main__":
    main()
