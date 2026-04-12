# Robust AI Image Detection (CS 440 Final Project - Reduced Setup)

This repository contains the reduced project pipeline for binary real-vs-fake image classification.

## Scope for current stage

- Active datasets: **Data Set 1, Data Set 2, Data Set 3, Data Set 4**
- Data split policy: use existing folders (`train`, `validation`, `test`), no random split generation
- Label mapping: `real -> 0`, `fake -> 1`
- Baseline model training: pretrained **ResNet-18** only

## 1) Environment setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Expected dataset layout

```text
data/raw/
├── Data Set 1/
│   ├── train/{real,fake}
│   ├── validation/{real,fake}
│   └── test/{real,fake}
├── Data Set 2/
├── Data Set 3/
└── Data Set 4/
```

Images are expected as `.jpg`/`.jpeg` files.

## 3) Configure active dataset

Edit `configs/default.yaml`:

```yaml
dataset:
  root: data/raw
  active_name: Data Set 1
```

Supported values are exactly:
- `Data Set 1`
- `Data Set 2`
- `Data Set 3`
- `Data Set 4`

## 4) Optional metadata indexing (from predefined folders)

```bash
python -m src.scripts.prepare_dataset_index --config configs/default.yaml
```

This writes split-specific CSV metadata to `data/processed/metadata/` for the currently active dataset.

## 5) Dataloader sanity-check

```bash
python -m src.scripts.inspect_dataloader --config configs/default.yaml --split train --save-grid
```

## 6) Training entrypoints

Run end-to-end baseline training:

```bash
python -m src.scripts.train_baseline --config configs/default.yaml
```

Optional overrides:

```bash
python -m src.scripts.train_baseline --config configs/default.yaml --epochs 3 --device cuda
```

What it does:
- loads active dataset from config
- builds train + validation dataloaders
- trains pretrained ResNet-18 (`fc` replaced with 2-class head)
- logs train loss, validation loss, validation accuracy, and validation ROC-AUC
- saves latest and best checkpoints in `outputs/models/`

Checkpoint naming:
- `outputs/models/best_baseline_<dataset_slug>.pt`
- `outputs/models/latest_baseline_<dataset_slug>.pt`

Train the aligned / JPEG-conditioned condition:

```bash
python -m src.scripts.train_aligned --config configs/default.yaml
```

What it changes vs baseline:
- applies in-memory JPEG re-encoding to every training image
- samples quality from `transforms.aligned_jpeg_qualities` (same policy for both classes)
- keeps validation on clean eval transforms

Checkpoint naming:
- `outputs/models/best_aligned_<dataset_slug>.pt`
- `outputs/models/latest_aligned_<dataset_slug>.pt`

## 7) Transform modes

Implemented now:
- Train: `mode="baseline"`
- Train: `mode="aligned"` (JPEG-conditioned training)
- Eval: `mode="clean"`
- Eval: `mode="jpeg_eval"` (fixed JPEG degradation before resize/tensor/normalize)

Relevant config fields:

```yaml
transforms:
  image_size: 224
  interpolation: bilinear
  train_mode: baseline
  eval_mode: clean
  aligned_jpeg_qualities: [95, 75, 55]
  jpeg_eval_quality: 55
```

## 8) Evaluate a saved checkpoint

Evaluate on clean images:

```bash
python -m src.scripts.evaluate_checkpoint \
  --config configs/default.yaml \
  --checkpoint outputs/models/best_baseline_data_set_1.pt \
  --split test \
  --eval-mode clean
```

Evaluate on JPEG-degraded images:

```bash
python -m src.scripts.evaluate_checkpoint \
  --config configs/default.yaml \
  --checkpoint outputs/models/best_baseline_data_set_1.pt \
  --split test \
  --eval-mode jpeg_eval
```

Use the same command pattern for aligned checkpoints to produce the 2x2 comparison:
- baseline checkpoint on `clean`
- baseline checkpoint on `jpeg_eval`
- aligned checkpoint on `clean`
- aligned checkpoint on `jpeg_eval`
