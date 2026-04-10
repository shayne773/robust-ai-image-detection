# Robust AI Image Detection (CS 440 Final Project - Reduced Setup)

This repository contains the **setup + data pipeline stage** for a reduced (about 20-hour) version of the project:

**Robust Detection of AI-Generated Images Under JPEG Compression and Resizing**.

Current stage focus:
- CIFAKE-only data preparation
- Reproducible train/val/test split generation
- Reusable PyTorch dataset + dataloader pipeline
- Transform modes structured for future robustness experiments

## 1) Environment setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Expected project structure

```text
.
├── configs/default.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── datasets/
│   ├── evaluation/
│   ├── scripts/
│   ├── training/
│   └── utils/
└── outputs/
```

## 3) CIFAKE assumptions and configuration

Set `dataset.cifake_root` in `configs/default.yaml` to your local CIFAKE root.

The scanner is intentionally flexible and supports either:
1. **Explicit class directory names** via config (`dataset.real_dirnames`, `dataset.fake_dirnames`), or
2. **Any directory name containing class keywords** (e.g., `real`, `fake`, `ai`, `generated`, etc.).

Label mapping is fixed to:
- `0 = real`
- `1 = ai-generated`

Supported image extensions are configured in YAML (`dataset.allowed_extensions`).

## 4) Prepare CIFAKE split CSVs

Run:

```bash
python -m src.scripts.prepare_cifake --config configs/default.yaml
```

This script will:
- scan CIFAKE image files
- infer/validate labels
- print class counts
- build stratified train/val/test splits
- save CSVs to `data/splits/`:
  - `cifake_train.csv`
  - `cifake_val.csv`
  - `cifake_test.csv`

Each CSV includes:
- `image_path`
- `label`
- `split`
- `class_name`
- `filename`

## 5) Sanity-check dataloader

Run:

```bash
python -m src.scripts.inspect_dataloader --config configs/default.yaml --split train --save-grid
```

The script will:
- load config and split CSV
- build dataset + dataloader
- print split size and class counts
- fetch one batch and print tensor shape + labels
- optionally save an image grid to `outputs/figures/`

## 6) Transform modes (current + planned)

Implemented now:
- Train: `mode="baseline"`
- Eval: `mode="clean"`

Planned for later (clear errors for now):
- Train: `mode="aligned"` -> currently raises `NotImplementedError`
- Eval: `mode="jpeg_eval"` -> currently raises `NotImplementedError`

This keeps the pipeline explicit and safe while still being easy to expand.
