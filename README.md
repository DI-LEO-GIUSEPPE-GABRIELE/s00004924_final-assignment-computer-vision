# Medical Image Analysis Tool (Segmentation + Classification)

Questo progetto implementa una pipeline completa di Computer Vision per analisi di immagini mediche con:

- Segmentazione binaria (ROI vs background)
- Classificazione binaria (0/1)

## Requisiti

- Python 3

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Dataset (formato atteso)

Struttura:

```
<data_root>/
  train/
    images/   (PNG)
    masks/    (PNG, binaria)
    labels.csv  (id,label)
  val/
    images/
    masks/
    labels.csv
  test/
    images/
    masks/
    labels.csv
```

`labels.csv` ha colonne: `id,label` dove `id` è lo stem del file PNG e `label` è 0/1.

### Dataset reale consigliato (Montgomery County CXR)

Download + conversione nel formato atteso:

```bash
.venv/bin/python scripts/prepare_montgomery.py --out data/montgomery --max-samples 0 --seed 42
```

### Dataset sintetico (solo demo/smoke-test)

```bash
.venv/bin/python scripts/generate_synthetic.py --out data/synthetic --train 200 --val 50 --test 50 --size 256 --seed 42
```

## Pipeline

- Data acquisition & preprocessing: lettura grayscale, denoise (median blur), resize, normalization
- Feature representation: feature learned con backbone CNN (UNet encoder)
- Core logic: modello multi-task (segmentazione + classificazione)
- Post-processing: threshold + operazioni morfologiche + largest connected component per la maschera

## Training

```bash
.venv/bin/python scripts/train.py --data-root data/montgomery --run-dir runs/montgomery --epochs 10 --image-size 256 --batch-size 8
```

Output:

- `runs/<run>/checkpoints/best.pt` e `last.pt`
- `runs/<run>/config.json`
- `runs/<run>/latest_metrics.json`

## Evaluation

Metriche:

- Segmentation: Dice, mIoU
- Classification: Accuracy, Precision, Recall, F1, Confusion Matrix

Esecuzione:

```bash
.venv/bin/python scripts/evaluate.py --data-root data/montgomery --split test --checkpoint runs/montgomery/checkpoints/best.pt --out runs/montgomery/eval_test.json
```

## Inferenza su singola immagine

```bash
.venv/bin/python scripts/infer.py --image path/to/image.png --checkpoint runs/mtl_unet/checkpoints/best.pt --out-dir runs/mtl_unet/infer
```

## Technical Analysis Document (PDF, max 10 pagine)

Generazione automatica del PDF dai risultati di evaluation:

```bash
.venv/bin/python scripts/make_report.py --eval-json runs/montgomery/eval_test.json --out-pdf Technical_Analysis.pdf --project-title "Medical Image Analysis Tool"
```
