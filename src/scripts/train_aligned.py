from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.datasets import FolderImageDataset, build_records_from_predefined_splits
from src.datasets.transforms import build_eval_transforms, build_train_transforms
from src.models import build_resnet18_classifier
from src.training.trainer import evaluate, save_checkpoint, train_one_epoch
from src.utils.dataset_config import resolve_active_dataset_root, validate_dataset_section
from src.utils.io import load_yaml
from src.utils.paths import ensure_dir
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet-18 with JPEG-conditioned aligned train transforms."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override: cpu/cuda")
    return parser.parse_args()


def _build_dataloaders(config: dict) -> tuple[str, Path, DataLoader, DataLoader]:
    dcfg = config["dataset"]
    lcfg = config["loader"]
    tcfg = config["transforms"]

    dataset_name, dataset_root = resolve_active_dataset_root(config)
    records = build_records_from_predefined_splits(
        dataset_root=dataset_root,
        splits=[str(x) for x in dcfg["splits"]],
        class_to_label={str(k): int(v) for k, v in dcfg["class_to_label"].items()},
        allowed_extensions={str(x).lower() for x in dcfg["allowed_extensions"]},
    )

    train_tf = build_train_transforms(
        mode="aligned",
        image_size=int(tcfg["image_size"]),
        interpolation=str(tcfg.get("interpolation", "bilinear")),
        aligned_jpeg_qualities=[int(q) for q in tcfg.get("aligned_jpeg_qualities", [95, 75, 55])],
    )
    val_tf = build_eval_transforms(
        mode="clean",
        image_size=int(tcfg["image_size"]),
        interpolation=str(tcfg.get("interpolation", "bilinear")),
    )

    train_dataset = FolderImageDataset(records=records, split="train", transform=train_tf)
    val_dataset = FolderImageDataset(records=records, split="validation", transform=val_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(lcfg["batch_size"]),
        shuffle=True,
        num_workers=int(lcfg["num_workers"]),
        pin_memory=bool(lcfg["pin_memory"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(lcfg["batch_size"]),
        shuffle=False,
        num_workers=int(lcfg["num_workers"]),
        pin_memory=bool(lcfg["pin_memory"]),
    )

    return dataset_name, dataset_root, train_loader, val_loader


def _select_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    validate_dataset_section(config)

    seed = int(config["project"]["seed"])
    set_global_seed(seed)

    model_cfg = config["model"]
    train_cfg = config["training"]

    epochs = int(args.epochs) if args.epochs is not None else int(train_cfg["epochs"])
    device = _select_device(args.device)

    dataset_name, dataset_root, train_loader, val_loader = _build_dataloaders(config)

    model_name = str(model_cfg["name"]).lower()
    if model_name != "resnet18":
        raise ValueError(f"Only model.name='resnet18' is supported right now. Got: {model_name}")

    model = build_resnet18_classifier(
        num_classes=int(model_cfg["num_classes"]),
        pretrained=bool(model_cfg["pretrained"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    metric_name = str(train_cfg.get("selection_metric", "roc_auc")).lower()
    if metric_name not in {"roc_auc", "accuracy"}:
        raise ValueError("training.selection_metric must be either 'roc_auc' or 'accuracy'.")

    ckpt_dir = ensure_dir(train_cfg["checkpoint_dir"])
    dataset_slug = dataset_name.lower().replace(" ", "_")
    best_ckpt_path = ckpt_dir / f"best_aligned_{dataset_slug}.pt"
    latest_ckpt_path = ckpt_dir / f"latest_aligned_{dataset_slug}.pt"

    print(f"[train_aligned] Active dataset: {dataset_name}")
    print(f"[train_aligned] Dataset root: {dataset_root}")
    print(f"[train_aligned] Train size: {len(train_loader.dataset)}")
    print(f"[train_aligned] Validation size: {len(val_loader.dataset)}")
    print(f"[train_aligned] Device: {device}")
    print(f"[train_aligned] Model: resnet18")
    print(f"[train_aligned] Epochs: {epochs}")
    print(f"[train_aligned] Selection metric: {metric_name}")

    best_metric = float("-inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        selected_val_metric = (
            val_metrics.roc_auc
            if metric_name == "roc_auc" and val_metrics.roc_auc is not None
            else val_metrics.accuracy
        )

        print(
            f"[train_aligned] Epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} "
            f"val_roc_auc={val_metrics.roc_auc if val_metrics.roc_auc is not None else 'N/A'}"
        )

        save_checkpoint(
            path=latest_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_metric,
            selection_metric=metric_name,
            extra={
                "dataset_name": dataset_name,
                "config_path": args.config,
                "train_condition": "aligned",
                "train_mode": "aligned",
                "eval_mode": "clean",
            },
        )

        if selected_val_metric > best_metric:
            best_metric = float(selected_val_metric)
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                selection_metric=metric_name,
                extra={
                    "dataset_name": dataset_name,
                    "config_path": args.config,
                    "train_condition": "aligned",
                    "train_mode": "aligned",
                    "eval_mode": "clean",
                },
            )
            print(
                f"[train_aligned] New best {metric_name}: {best_metric:.4f}. "
                f"Saved checkpoint to {best_ckpt_path}"
            )

    print(f"[train_aligned] Training complete. Best {metric_name}: {best_metric:.4f}")
    print(f"[train_aligned] Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
