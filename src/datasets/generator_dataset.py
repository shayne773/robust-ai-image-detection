from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class GeneratorImageRecord:
    image_path: str
    label: int
    generator: str
    split: str
    class_name: str


class GeneratorFolderDataset(Dataset):
    def __init__(self, records: list[GeneratorImageRecord], transform: Any | None = None) -> None:
        if not records:
            raise ValueError("No records provided to GeneratorFolderDataset.")
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record.label


def build_generator_records(
    dataset_root: Path,
    generator_dirs: dict[str, str],
    generators: list[str],
    split: str,
    real_class_name: str,
    fake_class_name: str,
    allowed_extensions: set[str],
) -> list[GeneratorImageRecord]:
    records: list[GeneratorImageRecord] = []
    label_map = {real_class_name: 0, fake_class_name: 1}

    for generator in generators:
        if generator not in generator_dirs:
            raise ValueError(
                f"Unsupported generator '{generator}'. Supported: {sorted(generator_dirs)}"
            )
        gen_root = dataset_root / generator_dirs[generator]
        split_root = gen_root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Missing split directory: {split_root}")

        for class_name, label in label_map.items():
            class_dir = split_root / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")

            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in allowed_extensions:
                    records.append(
                        GeneratorImageRecord(
                            image_path=str(path.resolve()),
                            label=label,
                            generator=generator,
                            split=split,
                            class_name=class_name,
                        )
                    )

    if not records:
        raise ValueError("No image files found for selected generators/split.")
    return records


def class_balanced_subsample(
    records: list[GeneratorImageRecord],
    train_percent: float,
    seed: int,
) -> list[GeneratorImageRecord]:
    if not 0 < train_percent <= 1.0:
        raise ValueError(f"train_percent must be in (0, 1], got {train_percent}")
    if train_percent >= 1.0:
        return records

    rng = random.Random(seed)
    by_label: dict[int, list[GeneratorImageRecord]] = {0: [], 1: []}
    for r in records:
        by_label[r.label].append(r)

    subsampled: list[GeneratorImageRecord] = []
    for label, label_records in by_label.items():
        if not label_records:
            continue
        k = max(1, int(len(label_records) * train_percent))
        chosen = rng.sample(label_records, k=min(k, len(label_records)))
        subsampled.extend(chosen)

    rng.shuffle(subsampled)
    return subsampled
