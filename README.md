# Robust AI Image Detection (CS 440 Final Project - Reduced Setup)

This repository contains the **setup + data pipeline stage** for the reduced project.

## Scope for this stage

The active data pipeline now targets **Data Set 1, Data Set 2, Data Set 3, and Data Set 4** only.

- `cifake` is ignored by the default pipeline.
- Splits are **predefined** by dataset folders: `train`, `validation`, `test`.
- No random split generation is performed.

Label mapping is fixed to:
- `real -> 0`
- `fake -> 1`

## 1) Environment setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Expected dataset layout

Datasets should exist under `data/raw`:

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

Images are expected to be `.jpg`/`.jpeg` files.

## 3) Configure active dataset

Edit `configs/default.yaml`:

```yaml
dataset:
  root: data/raw
  active_name: Data Set 1
```

Supported values for `active_name` are:
- `Data Set 1`
- `Data Set 2`
- `Data Set 3`
- `Data Set 4`

## 4) Optional metadata indexing

You can generate metadata CSVs that mirror existing predefined split folders:

```bash
python -m src.scripts.prepare_dataset_index --config configs/default.yaml
```

This writes split-specific CSVs to `data/processed/metadata/` (for the active dataset), without creating new random splits.

## 5) Sanity-check dataloader

```bash
python -m src.scripts.inspect_dataloader --config configs/default.yaml --split train --save-grid
```

This script:
- resolves active dataset path from config
- scans predefined split folders
- builds dataset + dataloader
- prints split size and class counts
- fetches one batch and prints tensor/label info
- optionally saves a grid to `outputs/figures/`

## 6) Transform modes (current + planned)

Implemented now:
- Train: `mode="baseline"`
- Eval: `mode="clean"`

Reserved for later (explicitly not implemented yet):
- Train: `mode="aligned"` -> raises `NotImplementedError`
- Eval: `mode="jpeg_eval"` -> raises `NotImplementedError`

This keeps the pipeline minimal now and easy to extend for later robustness experiments.
