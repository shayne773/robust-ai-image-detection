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

## 6) Baseline training (new)

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

## 7) Transform modes

Implemented now:
- Train: `mode="baseline"`
- Eval: `mode="clean"`

Reserved for later (intentionally not implemented yet):
- Train: `mode="aligned"` -> raises `NotImplementedError`
- Eval: `mode="jpeg_eval"` -> raises `NotImplementedError`
