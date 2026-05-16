from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.datasets import GeneratorFolderDataset, GeneratorImageRecord, build_generator_records
from src.datasets.transforms import build_detector_transforms, resolve_compression_quality
from src.models import build_resnet_classifier
from src.utils.io import load_yaml
from src.utils.paths import ensure_dir


LABEL_NAMES = {
    0: "real",
    1: "fake",
}
COLORS = {
    "raw real": "#009E73",
    "raw fake": "#CC79A7",
    "degraded real": "#0072B2",
    "degraded fake": "#D55E00",
}
MARKERS = {
    "raw real": "o",
    "raw fake": "^",
    "degraded real": "s",
    "degraded fake": "X",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize raw-vs-JPEG detector feature spaces for one generator with "
            "four groups: JPEG real/fake and raw real/fake."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--generator", default="adm", help="Generator to visualize, e.g. adm, glide, biggan.")
    parser.add_argument("--split", default=None, help="Dataset split. Defaults to dataset.eval_split.")
    parser.add_argument("--raw-checkpoint", default=None, help="Raw-trained detector checkpoint.")
    parser.add_argument("--jpeg-checkpoint", default=None, help="JPEG-trained detector checkpoint.")
    parser.add_argument(
        "--jpeg-compression",
        default="jpeg90",
        help="Single JPEG degradation mode to compare with raw, e.g. jpeg90.",
    )
    parser.add_argument("--train-percent-tag", default="0.1", help="Checkpoint filename train-percent tag for defaults.")
    parser.add_argument("--max-per-class", type=int, default=250, help="Maximum real and fake records to sample.")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument("--csv-output", default=None, help="Optional projected-coordinate CSV path.")
    return parser.parse_args()


def default_checkpoint(generator: str, compression: str, train_percent_tag: str) -> str:
    return f"outputs/models/best_resnet50_train-{generator}_comp-{compression}_pct-{train_percent_tag}.pt"


def select_balanced_records(
    records: list[GeneratorImageRecord],
    max_per_class: int,
    seed: int,
) -> list[GeneratorImageRecord]:
    if max_per_class <= 0:
        raise ValueError("--max-per-class must be positive.")

    rng = random.Random(seed)
    selected: list[GeneratorImageRecord] = []
    for label in (0, 1):
        label_records = [record for record in records if record.label == label]
        if not label_records:
            raise ValueError(f"No records found for label {label}.")
        count = min(max_per_class, len(label_records))
        selected.extend(rng.sample(label_records, count))

    selected.sort(key=lambda record: (record.label, record.image_path))
    return selected


def load_feature_model(checkpoint_path: Path, architecture: str, device: torch.device) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_resnet_classifier(architecture, num_classes=2, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_features(
    model: nn.Module,
    records: list[GeneratorImageRecord],
    compression: str,
    image_size: int,
    interpolation: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    transform = build_detector_transforms(
        compression,
        image_size,
        interpolation=interpolation,
        is_train=False,
    )
    dataset = GeneratorFolderDataset(records, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    chunks: list[np.ndarray] = []
    for images, _labels in loader:
        features = model(images.to(device))
        chunks.append(features.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def reduce_to_2d(features: np.ndarray, method: str, seed: int, perplexity: float) -> np.ndarray:
    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        return reducer.fit_transform(features)

    max_perplexity = max(1.0, (len(features) - 1) / 3)
    reducer = TSNE(
        n_components=2,
        perplexity=min(perplexity, max_perplexity),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return reducer.fit_transform(features)


def group_name(compression: str, label: int) -> str:
    if compression == "raw":
        return f"raw {LABEL_NAMES[label]}"
    return f"{compression} degraded {LABEL_NAMES[label]}"


def group_order(jpeg_compression: str) -> list[str]:
    return [
        "raw real",
        "raw fake",
        group_name(jpeg_compression, 0),
        group_name(jpeg_compression, 1),
    ]


def group_style(name: str) -> tuple[str, str]:
    if name == "raw real":
        key = "raw real"
    elif name == "raw fake":
        key = "raw fake"
    elif name.endswith(" real"):
        key = "degraded real"
    else:
        key = "degraded fake"
    return COLORS[key], MARKERS[key]


def build_feature_table(
    model: nn.Module,
    records: list[GeneratorImageRecord],
    jpeg_compression: str,
    image_size: int,
    interpolation: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, list[str], list[GeneratorImageRecord], list[str]]:
    compressions = ["raw", jpeg_compression]
    feature_chunks = [
        extract_features(
            model,
            records,
            compression,
            image_size,
            interpolation,
            batch_size,
            num_workers,
            device,
        )
        for compression in compressions
    ]

    features = np.concatenate(feature_chunks, axis=0)
    repeated_records = [
        record
        for _compression in compressions
        for record in records
    ]
    compression_tags = [
        compression
        for compression in compressions
        for _record in records
    ]
    group_labels = [
        group_name(tag, record.label)
        for tag, record in zip(compression_tags, repeated_records, strict=True)
    ]
    return features, group_labels, repeated_records, compression_tags


def plot_embeddings(
    axes: np.ndarray,
    coords_by_detector: dict[str, np.ndarray],
    labels: list[str],
    groups: list[str],
    generator: str,
    jpeg_compression: str,
    method: str,
) -> None:
    for axis, (detector_name, coords) in zip(axes, coords_by_detector.items(), strict=True):
        for name in groups:
            indices = [idx for idx, label in enumerate(labels) if label == name]
            color, marker = group_style(name)
            axis.scatter(
                coords[indices, 0],
                coords[indices, 1],
                c=color,
                marker=marker,
                s=22,
                alpha=0.75,
                linewidths=0.2,
                edgecolors="white",
                label=name,
            )
        axis.set_title(f"{detector_name} detector")
        axis.set_xlabel(f"{method.upper()} 1")
        axis.set_ylabel(f"{method.upper()} 2")
        axis.grid(alpha=0.18, linewidth=0.7)

    axes[0].legend(loc="best", fontsize=8, frameon=True)
    axes[0].figure.suptitle(
        f"{generator.upper()} RAW vs {jpeg_compression.upper()} feature space",
        fontsize=13,
    )


def write_projection_csv(
    path: Path,
    rows_by_detector: dict[str, tuple[np.ndarray, list[str], list[GeneratorImageRecord], list[str]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "detector",
                "x",
                "y",
                "group",
                "compression",
                "label",
                "generator",
                "image_path",
            ]
        )
        for detector_name, (coords, labels, records, compression_tags) in rows_by_detector.items():
            for coord, label, record, compression_tag in zip(
                coords,
                labels,
                records,
                compression_tags,
                strict=True,
            ):
                writer.writerow(
                    [
                        detector_name,
                        float(coord[0]),
                        float(coord[1]),
                        label,
                        compression_tag,
                        record.label,
                        record.generator,
                        record.image_path,
                    ]
                )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dcfg, mcfg, tcfg = cfg["dataset"], cfg["model"], cfg["training"]
    transform_cfg = cfg["transforms"]

    resolve_compression_quality("raw")
    resolve_compression_quality(args.jpeg_compression)
    generator_dirs = dcfg["generator_dirs"]
    if args.generator not in generator_dirs:
        raise ValueError(f"Unsupported generator '{args.generator}'. Supported: {sorted(generator_dirs)}")

    raw_checkpoint = Path(
        args.raw_checkpoint
        or default_checkpoint(args.generator, "raw", args.train_percent_tag)
    )
    jpeg_checkpoint = Path(
        args.jpeg_checkpoint
        or default_checkpoint(args.generator, args.jpeg_compression, args.train_percent_tag)
    )

    split = args.split or dcfg["eval_split"]
    allowed_ext = {extension.lower() for extension in dcfg.get("allowed_extensions", [".jpg", ".jpeg", ".png"])}
    records = build_generator_records(
        Path(dcfg["root"]),
        generator_dirs,
        [args.generator],
        split,
        dcfg["real_class_name"],
        dcfg["fake_class_name"],
        allowed_ext,
    )
    selected_records = select_balanced_records(records, args.max_per_class, args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    image_size = int(transform_cfg["image_size"])
    interpolation = str(transform_cfg.get("interpolation", "bilinear"))
    batch_size = int(args.batch_size or tcfg["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else tcfg["num_workers"])

    detector_paths = {
        "raw-trained": raw_checkpoint,
        f"{args.jpeg_compression}-trained": jpeg_checkpoint,
    }
    coords_by_detector: dict[str, np.ndarray] = {}
    csv_rows: dict[str, tuple[np.ndarray, list[str], list[GeneratorImageRecord], list[str]]] = {}
    shared_labels: list[str] | None = None

    for detector_name, checkpoint_path in detector_paths.items():
        model = load_feature_model(checkpoint_path, str(mcfg["architecture"]), device)
        features, labels, repeated_records, compression_tags = build_feature_table(
            model,
            selected_records,
            args.jpeg_compression,
            image_size,
            interpolation,
            batch_size,
            num_workers,
            device,
        )
        coords = reduce_to_2d(features, args.method, args.seed, args.perplexity)
        coords_by_detector[detector_name] = coords
        csv_rows[detector_name] = (coords, labels, repeated_records, compression_tags)
        shared_labels = labels

    if shared_labels is None:
        raise RuntimeError("No embeddings were computed.")

    out_dir = ensure_dir("outputs/plots")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(
        args.output
        or out_dir / f"detector_embedding_{args.generator}_raw_vs_{args.jpeg_compression}_{args.method}_{timestamp}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    plot_embeddings(
        axes,
        coords_by_detector,
        shared_labels,
        group_order(args.jpeg_compression),
        args.generator,
        args.jpeg_compression,
        args.method,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    csv_path = None
    if args.csv_output is not None:
        csv_path = Path(args.csv_output)
        write_projection_csv(csv_path, csv_rows)

    print(f"Saved plot: {output_path}")
    if csv_path is not None:
        print(f"Saved CSV: {csv_path}")
    print(f"Generator: {args.generator}")
    print(f"Split: {split}")
    print(f"Samples per real/fake class: {len(selected_records) // 2}")
    print(f"Detector checkpoints: {raw_checkpoint}, {jpeg_checkpoint}")


if __name__ == "__main__":
    main()
