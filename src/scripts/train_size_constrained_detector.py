from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import warnings

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.datasets import (
    GeneratorFolderDataset,
    build_generator_records,
    build_size_constrained_training_records,
)
from src.datasets.transforms import (
    build_size_constrained_detector_transforms,
    resolve_compression_quality,
)
from src.models import build_resnet_classifier
from src.training.trainer import evaluate, save_checkpoint, train_one_epoch
from src.utils.io import load_yaml
from src.utils.paths import ensure_dir
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a detector with the GenImage size-bias mitigation protocol."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-generators", nargs="+", default=None)
    parser.add_argument("--natural-pool-generators", nargs="+", default=None)
    parser.add_argument("--eval-generators", nargs="+", default=None)
    parser.add_argument("--train-compression", default=None)
    parser.add_argument("--test-compression", default=None)
    parser.add_argument("--train-percent", type=float, default=None)
    parser.add_argument("--natural-min-size", type=int, default=None)
    parser.add_argument("--natural-max-size", type=int, default=None)
    parser.add_argument("--generated-size", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--inference-resize-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--size-cache", default=None, help="Path for persistent image dimension cache.")
    parser.add_argument("--natural-manifest", default=None, help="Path for reusable filtered natural-image manifest.")
    parser.add_argument("--rebuild-natural-manifest", action="store_true", help="Rebuild the natural manifest even if it exists.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--selection-metric", choices=["roc_auc", "accuracy", "recall"], default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Build and report the constrained split without training.")
    return parser.parse_args()


def _config_list(value: object | None, fallback: list[str]) -> list[str]:
    if value is None:
        return fallback
    if not isinstance(value, list):
        raise ValueError(f"Expected a list in config, got: {value}")
    return [str(item) for item in value]


def _metric_score(metrics: object, metric_name: str) -> float:
    if metric_name == "roc_auc":
        roc_auc = getattr(metrics, "roc_auc")
        return float(roc_auc if roc_auc is not None else getattr(metrics, "accuracy"))
    return float(getattr(metrics, metric_name))


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dcfg = cfg["dataset"]
    pcfg = cfg["preprocessing"]
    mcfg = cfg["model"]
    tcfg = cfg["training"]
    xcfg = cfg.get("size_constraint", {})

    train_generators = args.train_generators or _config_list(
        xcfg.get("train_generators"), [str(x) for x in dcfg["train_generators"]]
    )
    natural_pool_generators = args.natural_pool_generators or _config_list(
        xcfg.get("natural_pool_generators"), train_generators
    )
    eval_generators = args.eval_generators or _config_list(
        xcfg.get("eval_generators"), [str(x) for x in dcfg["eval_generators"]]
    )

    train_compression = str(
        args.train_compression or xcfg.get("train_compression") or pcfg["train_compression"]
    ).lower()
    test_compression = str(
        args.test_compression or xcfg.get("test_compression") or pcfg["test_compression"]
    ).lower()
    resolve_compression_quality(train_compression)
    resolve_compression_quality(test_compression)

    train_percent = float(
        args.train_percent if args.train_percent is not None else xcfg.get("train_percent", dcfg["train_percent"])
    )
    natural_min_size = int(args.natural_min_size or xcfg.get("natural_min_size", 450))
    natural_max_size = int(args.natural_max_size or xcfg.get("natural_max_size", 550))
    generated_size = int(args.generated_size or xcfg.get("generated_size", 512))
    crop_size = int(args.crop_size or xcfg.get("crop_size", natural_min_size))
    inference_resize_size = int(args.inference_resize_size or xcfg.get("inference_resize_size", generated_size))
    max_train_samples = (
        args.max_train_samples
        if args.max_train_samples is not None
        else xcfg.get("max_train_samples")
    )
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
    size_cache = Path(args.size_cache or xcfg.get("size_cache", "outputs/cache/genimage_image_sizes.json"))
    natural_manifest = Path(
        args.natural_manifest
        or xcfg.get(
            "natural_manifest",
            "outputs/splits/size_constrained_natural_wukong-sdv4-sdv5_450-550.json",
        )
    )

    if crop_size > min(natural_min_size, generated_size):
        raise ValueError(
            "crop_size must be <= both natural_min_size and generated_size to avoid padding artifacts."
        )
    if inference_resize_size < crop_size:
        raise ValueError("inference_resize_size must be >= crop_size.")

    seed = int(args.seed if args.seed is not None else tcfg.get("seed", 42))
    set_global_seed(seed)

    model_name = str(args.model or mcfg["architecture"]).lower()
    epochs = int(args.epochs if args.epochs is not None else tcfg["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else tcfg["batch_size"])
    lr = float(args.lr if args.lr is not None else tcfg["lr"])
    weight_decay = float(args.weight_decay if args.weight_decay is not None else tcfg.get("weight_decay", 0.01))
    num_workers = int(args.num_workers if args.num_workers is not None else tcfg["num_workers"])
    selection_metric = str(args.selection_metric or tcfg.get("selection_metric", "roc_auc")).lower()

    root = Path(dcfg["root"])
    allowed_ext = {str(e).lower() for e in dcfg.get("allowed_extensions", [".jpg", ".jpeg", ".png"])}

    train_records, selection_summary = build_size_constrained_training_records(
        dataset_root=root,
        generator_dirs={str(k): str(v) for k, v in dcfg["generator_dirs"].items()},
        train_generators=train_generators,
        natural_pool_generators=natural_pool_generators,
        split=str(dcfg["train_split"]),
        real_class_name=str(dcfg["real_class_name"]),
        fake_class_name=str(dcfg["fake_class_name"]),
        allowed_extensions=allowed_ext,
        natural_min_size=natural_min_size,
        natural_max_size=natural_max_size,
        generated_size=generated_size,
        train_percent=train_percent,
        seed=seed,
        max_train_samples=max_train_samples,
        size_cache_path=size_cache,
        natural_manifest_path=natural_manifest,
        rebuild_natural_manifest=bool(args.rebuild_natural_manifest),
    )
    val_records = build_generator_records(
        dataset_root=root,
        generator_dirs={str(k): str(v) for k, v in dcfg["generator_dirs"].items()},
        generators=eval_generators,
        split=str(dcfg["eval_split"]),
        real_class_name=str(dcfg["real_class_name"]),
        fake_class_name=str(dcfg["fake_class_name"]),
        allowed_extensions=allowed_ext,
    )

    if args.dry_run:
        print("[train_size_constrained] Dry run complete")
        print(f"[train_size_constrained] Train generators: {train_generators}")
        print(f"[train_size_constrained] Natural pool generators: {natural_pool_generators}")
        print(f"[train_size_constrained] Eval generators: {eval_generators}")
        print(f"[train_size_constrained] Train compression: {train_compression}")
        print(f"[train_size_constrained] Eval compression: {test_compression}")
        print(f"[train_size_constrained] Natural size range: [{natural_min_size}, {natural_max_size}]")
        print(f"[train_size_constrained] Generated exact size: {generated_size}x{generated_size}")
        print(f"[train_size_constrained] Max train samples: {max_train_samples}")
        print(f"[train_size_constrained] Natural manifest: {natural_manifest}")
        print(f"[train_size_constrained] Selection summary: {asdict(selection_summary)}")
        print(f"[train_size_constrained] Validation size: {len(val_records)}")
        return

    image_size = int(cfg["transforms"]["image_size"])
    interpolation = str(cfg["transforms"].get("interpolation", "bilinear"))
    train_transform = build_size_constrained_detector_transforms(
        compression_mode=train_compression,
        image_size=image_size,
        crop_size=crop_size,
        interpolation=interpolation,
        is_train=True,
        inference_resize_size=None,
    )
    val_transform = build_size_constrained_detector_transforms(
        compression_mode=test_compression,
        image_size=image_size,
        crop_size=crop_size,
        interpolation=interpolation,
        is_train=False,
        inference_resize_size=inference_resize_size,
    )

    train_ds = GeneratorFolderDataset(train_records, train_transform)
    val_ds = GeneratorFolderDataset(val_records, val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_resnet_classifier(
        model_name,
        num_classes=2,
        pretrained=bool(mcfg["pretrained"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ckpt_dir = ensure_dir(args.checkpoint_dir or tcfg["checkpoint_dir"])
    gen_slug = "-".join(train_generators)
    best_path = (
        ckpt_dir
        / (
            f"best_{model_name}_sizeconstrained_train-{gen_slug}_comp-{train_compression}"
            f"_nat-{natural_min_size}-{natural_max_size}_fake-{generated_size}"
            f"_crop-{crop_size}_pct-{train_percent}.pt"
        )
    )

    print("[train_size_constrained] Training with size-bias mitigation")
    print(f"[train_size_constrained] Train generators: {train_generators}")
    print(f"[train_size_constrained] Natural pool generators: {natural_pool_generators}")
    print(f"[train_size_constrained] Eval generators: {eval_generators}")
    print(f"[train_size_constrained] Train compression: {train_compression}")
    print(f"[train_size_constrained] Eval compression: {test_compression}")
    print(f"[train_size_constrained] Natural size range: [{natural_min_size}, {natural_max_size}]")
    print(f"[train_size_constrained] Generated exact size: {generated_size}x{generated_size}")
    print(f"[train_size_constrained] Max train samples: {max_train_samples}")
    print(f"[train_size_constrained] Natural manifest: {natural_manifest}")
    print(f"[train_size_constrained] Train crop/resize: center-crop {crop_size}, resize {image_size}")
    print(
        "[train_size_constrained] Eval crop/resize: "
        f"resize {inference_resize_size}, center-crop {crop_size}, resize {image_size}"
    )
    print(f"[train_size_constrained] Selection summary: {asdict(selection_summary)}")
    print(f"[train_size_constrained] Validation size: {len(val_ds)}")
    print(f"[train_size_constrained] Device: {device}")

    best = -1.0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        score = _metric_score(val_metrics, selection_metric)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} val_auc={val_metrics.roc_auc} "
            f"val_recall={val_metrics.recall:.4f}"
        )
        if score > best:
            best = score
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best,
                selection_metric,
                extra={
                    "architecture": model_name,
                    "train_condition": "size_constrained",
                    "train_generators": train_generators,
                    "natural_pool_generators": natural_pool_generators,
                    "eval_generators": eval_generators,
                    "train_compression": train_compression,
                    "test_compression": test_compression,
                    "train_percent": train_percent,
                    "natural_min_size": natural_min_size,
                    "natural_max_size": natural_max_size,
                    "generated_size": generated_size,
                    "generated_size_assumption": "generated image dimensions are trusted, not scanned",
                    "max_train_samples": max_train_samples,
                    "natural_manifest": str(natural_manifest),
                    "crop_size": crop_size,
                    "inference_resize_size": inference_resize_size,
                    "selection_summary": asdict(selection_summary),
                    "class_mapping": {dcfg["real_class_name"]: 0, dcfg["fake_class_name"]: 1},
                },
            )
            print(f"[train_size_constrained] New best {selection_metric}: {best:.4f}")

    if best < 0:
        warnings.warn("No checkpoint was saved because no epochs were run.")
    print(f"[train_size_constrained] Saved best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
