from __future__ import annotations

"""
Generates a short Technical Analysis PDF (max 10 pages) from an evaluation JSON.

Input: JSON produced by scripts/evaluate.py
Output: a PDF containing problem statement, methodology, results, failure analysis, and ethics.
"""

from pathlib import Path

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def _page_text(pdf: PdfPages, title: str, lines: list[str]) -> None:
    """Adds a text-only page to the PDF."""

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(title, fontsize=18, y=0.97)
    y = 0.92
    for line in lines:
        fig.text(0.07, y, line, fontsize=11, va="top")
        y -= 0.03
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
        default="Dataset: Montgomery County Chest X-ray (NLM) con maschere polmoni e label TB (0=normal, 1=abnormal).",
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
                "Problem Statement:",
                "- Obiettivo: segmentare i polmoni su CXR e classificare la presenza/assenza di segni compatibili con TB.",
                "- Rilevanza: supporto a screening/triage e riduzione del carico manuale per operatori clinici.",
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
                "Pipeline:",
                "- Preprocessing: grayscale, denoise (median blur), resize, normalization; augmentation (flip/rotate) in training.",
                "- Feature representation: feature learned con CNN (encoder UNet).",
                "- Core logic: rete multi-task con head di segmentazione (logits pixel-wise) e head di classificazione (global pooling).",
                "- Loss: BCEWithLogits per classificazione + (BCE + Soft Dice) per segmentazione.",
                "- Post-processing (segmentation): threshold + morphological opening/closing + largest connected component.",
                "",
                "Implementazione: Python + OpenCV + PyTorch + scikit-learn.",
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
                "Casi tipici di fallimento attesi (da verificare su dataset reale):",
                "- Lesioni molto piccole: la segmentazione può essere instabile (basso Dice/mIoU).",
                "- Basso contrasto o artefatti: aumento di falsi positivi/negativi.",
                "- Shift di dominio (scanner/protocollo diverso): degrado sia su segmentazione che classificazione.",
                "",
                "Mitigazioni:",
                "- Data augmentation mirata e normalizzazione coerente.",
                "- Calibrazione soglia e post-processing più conservativo.",
                "- Fine-tuning su dati del dominio target e validazione cross-site.",
            ],
        )

        _page_text(
            pdf,
            "Ethical Considerations",
            [
                "Privacy:",
                "- Usare dataset de-identificati; evitare di salvare metadati sensibili; controllare accessi e log.",
                "",
                "Bias e fairness:",
                "- Verificare performance stratificate (età/sesso/dispositivo) quando disponibili.",
                "- Gestire sbilanciamento classi e valutare metriche robuste (F1, ROC/PR) dove opportuno.",
                "",
                "Uso clinico:",
                "- Strumento di supporto, non sostituzione del giudizio clinico; documentare limiti e failure modes.",
            ],
        )


if __name__ == "__main__":
    main()
