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

## Visualize detector feature spaces

Compare a raw-trained detector against a JPEG90-trained detector for one generator, with four
groups in 2D: JPEG90-degraded real, JPEG90-degraded fake, raw real, and raw fake.

```bash
python -m src.scripts.plot_detector_embedding_space \
  --config configs/default.yaml \
  --generator adm \
  --method pca \
  --jpeg-compression jpeg90 \
  --max-per-class 250
```

Use `--generator biggan` or `--generator glide` for the other detector pairs. The default
checkpoint names match `outputs/models/best_resnet50_train-{generator}_comp-{raw,jpeg90}_pct-0.1.pt`.
Use `--jpeg-compression jpeg95` or another single JPEG level to compare against a detector trained
at that same level.

## Train with size-bias mitigation

After adding the paper-style folders under `data/`:

- `data/imagenet_ai_0424_sdv5`
- `data/imagenet_ai_0419_sdv4`
- `data/imagenet_ai_0424_wukong`

Create the reusable filtered-natural-image manifest once:

```bash
python -m src.scripts.prepare_size_constrained_natural_manifest \
  --config configs/default.yaml \
  --natural-pool-generators wukong sdv4 sdv5 \
  --natural-min-size 450 \
  --natural-max-size 550
```

Preview the constrained split before training:

```bash
python -m src.scripts.train_size_constrained_detector \
  --config configs/default.yaml \
  --train-generators wukong \
  --natural-pool-generators wukong sdv4 sdv5 \
  --dry-run
```

Then train with JPEG96, natural images loaded from the manifest, generated images assumed to be `512x512`, center-crop `450`, and resize to `224`:

```bash
python -m src.scripts.train_size_constrained_detector \
  --config configs/default.yaml \
  --train-generators wukong \
  --natural-pool-generators wukong sdv4 sdv5 \
  --train-compression jpeg96 \
  --test-compression jpeg96
```

For quick smoke tests, cap the selected training set:

```bash
python -m src.scripts.train_size_constrained_detector \
  --config configs/default.yaml \
  --train-generators wukong \
  --natural-pool-generators wukong sdv4 sdv5 \
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
