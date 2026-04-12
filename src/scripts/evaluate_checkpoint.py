from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import FolderImageDataset, build_records_from_predefined_splits
from src.datasets.transforms import build_eval_transforms
from src.models import build_resnet18_classifier
from src.training.trainer import evaluate
from src.utils.dataset_config import resolve_active_dataset_root, validate_dataset_section
from src.utils.io import load_yaml


VALID_SPLITS = {"train", "validation", "test"}
VALID_EVAL_MODES = {"clean", "jpeg_eval"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on one split and eval mode.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=sorted(VALID_SPLITS))
    parser.add_argument(
        "--eval-mode",
        type=str,
        default=None,
        choices=sorted(VALID_EVAL_MODES),
        help="Optional override for transforms.eval_mode",
    )
    parser.add_argument("--device", type=str, default=None, help="Optional device override: cpu/cuda")
    return parser.parse_args()


def _select_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    validate_dataset_section(config)

    dcfg = config["dataset"]
    lcfg = config["loader"]
    tcfg = config["transforms"]
    mcfg = config["model"]

    eval_mode = str(args.eval_mode or tcfg.get("eval_mode", "clean"))

    dataset_name, dataset_root = resolve_active_dataset_root(config)
    records = build_records_from_predefined_splits(
        dataset_root=dataset_root,
        splits=[str(x) for x in dcfg["splits"]],
        class_to_label={str(k): int(v) for k, v in dcfg["class_to_label"].items()},
        allowed_extensions={str(x).lower() for x in dcfg["allowed_extensions"]},
    )

    eval_tf = build_eval_transforms(
        mode=eval_mode,
        image_size=int(tcfg["image_size"]),
        interpolation=str(tcfg.get("interpolation", "bilinear")),
        jpeg_eval_quality=int(tcfg.get("jpeg_eval_quality", 55)),
    )

    eval_dataset = FolderImageDataset(records=records, split=args.split, transform=eval_tf)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(lcfg["batch_size"]),
        shuffle=False,
        num_workers=int(lcfg["num_workers"]),
        pin_memory=bool(lcfg["pin_memory"]),
    )

    model_name = str(mcfg["name"]).lower()
    if model_name != "resnet18":
        raise ValueError(f"Only model.name='resnet18' is supported right now. Got: {model_name}")

    device = _select_device(args.device)
    model = build_resnet18_classifier(
        num_classes=int(mcfg["num_classes"]),
        pretrained=False,
    ).to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(
        model=model,
        dataloader=eval_loader,
        criterion=criterion,
        device=device,
    )

    print("[evaluate_checkpoint] Evaluation complete")
    print(f"[evaluate_checkpoint] Checkpoint: {checkpoint_path}")
    print(f"[evaluate_checkpoint] Dataset: {dataset_name}")
    print(f"[evaluate_checkpoint] Split: {args.split}")
    print(f"[evaluate_checkpoint] Eval mode: {eval_mode}")
    print(f"[evaluate_checkpoint] Loss: {metrics.loss:.6f}")
    print(f"[evaluate_checkpoint] Accuracy: {metrics.accuracy:.6f}")
    print(
        "[evaluate_checkpoint] ROC-AUC: "
        f"{metrics.roc_auc:.6f}" if metrics.roc_auc is not None else "[evaluate_checkpoint] ROC-AUC: N/A"
    )


if __name__ == "__main__":
    main()
