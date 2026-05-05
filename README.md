# Robust AI Image Detection

This project now supports configurable **ResNet50** training/evaluation on only three generators:

- `biggan` → `data/imagenet_ai_0419_biggan`
- `adm` → `data/imagenet_ai_0508_adm`
- `glide` → `data/imagenet_glide`

`data/raw` and `data/processed` are not used by the new detector pipeline.

## Train detector

Train ResNet50 on raw BigGAN:

```bash
python -m src.scripts.train_detector \
  --config configs/default.yaml \
  --train-generators biggan \
  --train-compression raw \
  --train-percent 1.0 \
  --model resnet50
```

Train ResNet50 on BigGAN/ADM/GLIDE with JPEG96:

```bash
python -m src.scripts.train_detector \
  --config configs/default.yaml \
  --train-generators biggan adm glide \
  --train-compression jpeg96 \
  --train-percent 1.0 \
  --model resnet50
```

## Evaluate detector

Evaluate raw validation images:

```bash
python -m src.scripts.evaluate_detector \
  --config configs/default.yaml \
  --checkpoint outputs/models/best_resnet50.pt \
  --eval-generators biggan adm glide \
  --test-compression raw
```

Evaluate JPEG95 validation images:

```bash
python -m src.scripts.evaluate_detector \
  --config configs/default.yaml \
  --checkpoint outputs/models/best_resnet50.pt \
  --eval-generators biggan adm glide \
  --test-compression jpeg95
```

Evaluation reports: loss, accuracy, ROC-AUC, recall; and saves JSON under `outputs/eval/`.

## Train with size-bias mitigation

After adding the paper-style folders under `data/`:

- `data/imagenet_ai_0424_sdv5`
- `data/imagenet_ai_0419_sdv4`
- `data/imagenet_ai_0424_wukong`

preview the constrained split first:

```bash
python -m src.scripts.train_size_constrained_detector \
  --config configs/default.yaml \
  --dry-run
```

Then train with JPEG96, natural images filtered to `[450, 550]`, generated images assumed to be `512x512`, center-crop `450`, and resize to `224`:

```bash
python -m src.scripts.train_size_constrained_detector \
  --config configs/default.yaml \
  --train-generators sdv5 sdv4 wukong \
  --natural-pool-generators sdv5 sdv4 wukong \
  --train-compression jpeg96 \
  --test-compression jpeg96
```

For quick smoke tests, cap the selected training set:

```bash
python -m src.scripts.train_size_constrained_detector \
  --config configs/default.yaml \
  --max-train-samples 2000 \
  --epochs 1
```

Evaluate those checkpoints with matching preprocessing:

```bash
python -m src.scripts.evaluate_detector \
  --config configs/default.yaml \
  --checkpoint outputs/models/YOUR_CHECKPOINT.pt \
  --eval-generators sdv5 sdv4 wukong \
  --test-compression jpeg96 \
  --preprocess-mode size-constrained
```
