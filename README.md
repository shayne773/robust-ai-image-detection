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
