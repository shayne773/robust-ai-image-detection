from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from src.datasets import FolderImageDataset, build_records_from_predefined_splits
from src.datasets.transforms import build_eval_transforms, build_train_transforms
from src.utils.io import load_yaml
from src.utils.paths import ensure_dir, resolve_from_root
from src.utils.seed import set_global_seed


SUPPORTED_SPLITS = {"train", "validation", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Data Set 1-4 DataLoader output.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument(
        "--save-grid",
        action="store_true",
        help="Save a sample image grid to outputs/figures.",
    )
    return parser.parse_args()


def _resolve_dataset_root(config: dict) -> tuple[str, Path]:
    dcfg = config["dataset"]
    active_name = str(dcfg["active_name"])
    supported_names = [str(x) for x in dcfg["supported_names"]]
    if active_name not in supported_names:
        raise ValueError(
            f"dataset.active_name='{active_name}' is not supported. "
            f"Choose one of: {supported_names}"
        )

    root = resolve_from_root(dcfg["root"]) / active_name
    return active_name, root


def _build_transform(config: dict, split: str):
    tcfg = config["transforms"]
    image_size = int(tcfg["image_size"])
    interpolation = str(tcfg.get("interpolation", "bilinear"))

    if split == "train":
        return build_train_transforms(
            mode=str(tcfg["train_mode"]),
            image_size=image_size,
            interpolation=interpolation,
        )

    return build_eval_transforms(
        mode=str(tcfg["eval_mode"]),
        image_size=image_size,
        interpolation=interpolation,
    )


def _unnormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def maybe_save_grid(images: torch.Tensor, dataset_name: str, split: str, output_dir: Path, max_images: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    images = _unnormalize(images[:max_images].cpu())
    nrow = min(4, images.shape[0])
    grid = make_grid(images, nrow=nrow)

    slug = dataset_name.lower().replace(" ", "_")
    out_path = output_dir / f"inspect_{slug}_{split}_grid.png"
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    if args.split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split: {args.split}")

    seed = int(config["project"]["seed"])
    set_global_seed(seed)

    dataset_name, dataset_root = _resolve_dataset_root(config)
    dcfg = config["dataset"]
    records = build_records_from_predefined_splits(
        dataset_root=dataset_root,
        splits=[str(x) for x in dcfg["splits"]],
        class_to_label={str(k): int(v) for k, v in dcfg["class_to_label"].items()},
        allowed_extensions={str(x).lower() for x in dcfg["allowed_extensions"]},
    )

    transform = _build_transform(config, args.split)
    dataset = FolderImageDataset(records=records, split=args.split, transform=transform)

    split_records = [r for r in records if r.split == args.split]
    label_counts = Counter(r.label for r in split_records)

    loader_cfg = config["loader"]
    dataloader = DataLoader(
        dataset,
        batch_size=int(loader_cfg["batch_size"]),
        shuffle=(args.split == "train"),
        num_workers=int(loader_cfg["num_workers"]),
        pin_memory=bool(loader_cfg["pin_memory"]),
    )

    print(f"[inspect_dataloader] Active dataset: {dataset_name}")
    print(f"[inspect_dataloader] Dataset root: {dataset_root}")
    print(f"[inspect_dataloader] Split: {args.split}")
    print(f"[inspect_dataloader] Split size: {len(dataset)}")
    print(f"[inspect_dataloader] Split class counts: {dict(label_counts)}")

    batch_images, batch_labels = next(iter(dataloader))
    print(f"[inspect_dataloader] Batch image tensor shape: {tuple(batch_images.shape)}")
    print(f"[inspect_dataloader] Batch labels shape: {tuple(batch_labels.shape)}")
    print(f"[inspect_dataloader] Batch labels sample: {batch_labels[:16].tolist()}")

    if args.save_grid:
        figures_dir = ensure_dir("outputs/figures")
        max_images = int(config.get("inspect", {}).get("grid_max_images", 16))
        out_path = maybe_save_grid(batch_images, dataset_name, args.split, figures_dir, max_images=max_images)
        print(f"[inspect_dataloader] Saved image grid: {out_path}")


if __name__ == "__main__":
    main()
