# Medical Image Analysis Tool (Segmentation + Classification)

This project implements an end-to-end Computer Vision pipeline for medical image analysis with:

- Binary segmentation (ROI vs background)
- Binary classification (0/1)

## Requirements

- Python 3

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Dataset (expected format)

Structure:

```
<data_root>/
  train/
    images/   (PNG)
    masks/    (PNG, binary)
    labels.csv (id,label)
  val/
    images/
    masks/
    labels.csv
  test/
    images/
    masks/
    labels.csv
```

`labels.csv` must contain columns: `id,label` where `id` matches the image/mask filename stem and `label` is 0/1.

### Recommended real dataset (Montgomery County CXR)

Download + conversion into the expected format:

```bash
.venv/bin/python scripts/prepare_montgomery.py --out data/montgomery --max-samples 0 --seed 42
```

### Synthetic dataset (demo/smoke-test only)

```bash
.venv/bin/python scripts/generate_synthetic.py --out data/synthetic --train 200 --val 50 --test 50 --size 256 --seed 42
```

## Pipeline

- Data acquisition & preprocessing: grayscale loading, denoise (median blur), resize, normalization
- Feature representation: learned features with a CNN backbone (UNet encoder)
- Core logic: multi-task model (segmentation + classification)
- Post-processing: threshold + morphological ops + largest connected component for the mask

## Training

```bash
.venv/bin/python scripts/train.py --data-root data/montgomery --run-dir runs/montgomery --epochs 10 --image-size 256 --batch-size 8
```

Outputs:

- `runs/<run>/checkpoints/best.pt` and `last.pt`
- `runs/<run>/config.json`
- `runs/<run>/latest_metrics.json`

## Evaluation

Metrics:

- Segmentation: Dice, mIoU
- Classification: Accuracy, Precision, Recall, F1, Confusion Matrix

Run:

```bash
.venv/bin/python scripts/evaluate.py --data-root data/montgomery --split test --checkpoint runs/montgomery/checkpoints/best.pt --out runs/montgomery/eval_test.json
```

## Single-image inference

```bash
.venv/bin/python scripts/infer.py --image path/to/image.png --checkpoint runs/montgomery/checkpoints/best.pt --out-dir runs/montgomery/infer
```

## Technical Analysis Document (PDF, max 10 pages)

Generate the PDF from evaluation results:

```bash
.venv/bin/python scripts/make_report.py --eval-json runs/montgomery/eval_test.json --out-pdf Technical_Analysis.pdf --project-title "Medical Image Analysis Tool"
```
