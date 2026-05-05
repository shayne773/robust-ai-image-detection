from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import warnings
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
    imagenet_class: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class SizeConstrainedSelectionSummary:
    natural_candidates: int
    fake_candidates: int
    selected_natural: int
    selected_fake: int
    selected_total: int
    fake_per_generator: dict[str, int]


_IMAGENET_INDEX_RE = re.compile(r"^(\d{1,3})(?:[_-]|$)")
_WNID_RE = re.compile(r"^(n\d{8})(?:[_\-.]|$)")


class ImageSizeCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, dict[str, int]] = {}
        self.dirty = False
        if path.exists():
            try:
                raw_entries = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                warnings.warn(f"Ignoring unreadable image-size cache: {path}")
                raw_entries = {}
            if isinstance(raw_entries, dict):
                self.entries = {
                    str(k): v
                    for k, v in raw_entries.items()
                    if isinstance(v, dict)
                    and {"mtime_ns", "size_bytes", "width", "height"}.issubset(v)
                }

    def get(self, path: Path) -> tuple[int, int] | None:
        stat = path.stat()
        key = str(path.resolve())
        entry = self.entries.get(key)
        if entry is None:
            return None
        if entry["mtime_ns"] != stat.st_mtime_ns or entry["size_bytes"] != stat.st_size:
            return None
        return int(entry["width"]), int(entry["height"])

    def set(self, path: Path, size: tuple[int, int]) -> None:
        stat = path.stat()
        self.entries[str(path.resolve())] = {
            "mtime_ns": stat.st_mtime_ns,
            "size_bytes": stat.st_size,
            "width": int(size[0]),
            "height": int(size[1]),
        }
        self.dirty = True

    def flush(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(json.dumps(self.entries), encoding="utf-8")
        temp_path.replace(self.path)


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
    include_image_size: bool = False,
    size_cache: ImageSizeCache | None = None,
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

            for path in sorted(class_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in allowed_extensions:
                    continue
                if path.stat().st_size == 0:
                    warnings.warn(f"Skipping zero-byte image file: {path}")
                    continue
                width = height = None
                if include_image_size:
                    size = _read_image_size(path, size_cache)
                    if size is None:
                        continue
                    width, height = size
                records.append(
                    GeneratorImageRecord(
                        image_path=str(path.resolve()),
                        label=label,
                        generator=generator,
                        split=split,
                        class_name=class_name,
                        imagenet_class=_infer_imagenet_class(path, class_dir),
                        width=width,
                        height=height,
                    )
                )

    if not records:
        raise ValueError("No image files found for selected generators/split.")
    return records


def build_generator_class_records(
    dataset_root: Path,
    generator_dirs: dict[str, str],
    generators: list[str],
    split: str,
    class_name: str,
    label: int,
    allowed_extensions: set[str],
    include_image_size: bool = False,
    size_cache: ImageSizeCache | None = None,
) -> list[GeneratorImageRecord]:
    records: list[GeneratorImageRecord] = []

    for generator in generators:
        if generator not in generator_dirs:
            raise ValueError(
                f"Unsupported generator '{generator}'. Supported: {sorted(generator_dirs)}"
            )
        class_dir = dataset_root / generator_dirs[generator] / split / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in allowed_extensions:
                continue
            if path.stat().st_size == 0:
                warnings.warn(f"Skipping zero-byte image file: {path}")
                continue
            width = height = None
            if include_image_size:
                size = _read_image_size(path, size_cache)
                if size is None:
                    continue
                width, height = size
            records.append(
                GeneratorImageRecord(
                    image_path=str(path.resolve()),
                    label=label,
                    generator=generator,
                    split=split,
                    class_name=class_name,
                    imagenet_class=_infer_imagenet_class(path, class_dir),
                    width=width,
                    height=height,
                )
            )

    if not records:
        raise ValueError(f"No image files found for selected generators/split/class: {class_name}")
    return records


def _read_image_size(path: Path, size_cache: ImageSizeCache | None = None) -> tuple[int, int] | None:
    cached = size_cache.get(path) if size_cache is not None else None
    if cached is not None:
        return cached
    try:
        with Image.open(path) as image:
            size = image.size
            if size_cache is not None:
                size_cache.set(path, size)
            return size
    except Exception as exc:
        warnings.warn(f"Skipping unreadable image file: {path} ({exc})")
        return None


def _infer_imagenet_class(path: Path, class_dir: Path) -> str | None:
    relative_parent = path.parent.relative_to(class_dir)
    if relative_parent.parts:
        return _normalize_class_token(relative_parent.parts[0])
    return _normalize_class_token(path.stem)


def _normalize_class_token(value: str) -> str | None:
    token = value.strip().lower()
    if not token:
        return None

    index_match = _IMAGENET_INDEX_RE.match(token)
    if index_match:
        return f"{int(index_match.group(1)):03d}"

    wnid_match = _WNID_RE.match(token)
    if wnid_match:
        return wnid_match.group(1)

    return token


def normalize_imagenet_class_keys(records: list[GeneratorImageRecord]) -> list[GeneratorImageRecord]:
    wnids = sorted(
        {
            r.imagenet_class
            for r in records
            if r.imagenet_class is not None and re.fullmatch(r"n\d{8}", r.imagenet_class)
        }
    )
    if not wnids:
        return records

    numeric_classes_present = any(
        r.imagenet_class is not None and re.fullmatch(r"\d{3}", r.imagenet_class)
        for r in records
    )
    if numeric_classes_present and len(wnids) != 1000:
        warnings.warn(
            "Inferring WordNet-id to ImageNet-index mapping from the natural classes present "
            f"({len(wnids)} found, expected 1000). Provide a full GenImage split for exact class alignment."
        )

    wnid_to_index = {wnid: f"{idx:03d}" for idx, wnid in enumerate(wnids)}
    return [
        GeneratorImageRecord(
            image_path=r.image_path,
            label=r.label,
            generator=r.generator,
            split=r.split,
            class_name=r.class_name,
            imagenet_class=wnid_to_index.get(r.imagenet_class or "", r.imagenet_class),
            width=r.width,
            height=r.height,
        )
        for r in records
    ]


def _deduplicate_natural_records(records: list[GeneratorImageRecord]) -> list[GeneratorImageRecord]:
    seen: set[tuple[str, str]] = set()
    deduped: list[GeneratorImageRecord] = []
    for record in records:
        key = (record.imagenet_class or "", Path(record.image_path).name.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _has_size_in_range(record: GeneratorImageRecord, min_size: int, max_size: int) -> bool:
    return (
        record.width is not None
        and record.height is not None
        and min_size <= record.width <= max_size
        and min_size <= record.height <= max_size
    )


def build_size_constrained_training_records(
    dataset_root: Path,
    generator_dirs: dict[str, str],
    train_generators: list[str],
    natural_pool_generators: list[str],
    split: str,
    real_class_name: str,
    fake_class_name: str,
    allowed_extensions: set[str],
    natural_min_size: int,
    natural_max_size: int,
    generated_size: int,
    train_percent: float,
    seed: int,
    max_train_samples: int | None = None,
    size_cache_path: Path | None = None,
) -> tuple[list[GeneratorImageRecord], SizeConstrainedSelectionSummary]:
    if not 0 < train_percent <= 1.0:
        raise ValueError(f"train_percent must be in (0, 1], got {train_percent}")
    if natural_min_size > natural_max_size:
        raise ValueError("natural_min_size must be <= natural_max_size")
    if generated_size <= 0:
        raise ValueError("generated_size must be positive.")
    if not train_generators:
        raise ValueError("At least one train generator is required.")
    if not natural_pool_generators:
        raise ValueError("At least one natural-pool generator is required.")
    if max_train_samples is not None and max_train_samples < 2:
        raise ValueError("max_train_samples must be at least 2 when provided.")

    size_cache = ImageSizeCache(size_cache_path) if size_cache_path is not None else None
    try:
        natural_candidates = build_generator_class_records(
            dataset_root=dataset_root,
            generator_dirs=generator_dirs,
            generators=natural_pool_generators,
            split=split,
            class_name=real_class_name,
            label=0,
            allowed_extensions=allowed_extensions,
            include_image_size=True,
            size_cache=size_cache,
        )
    finally:
        if size_cache is not None:
            size_cache.flush()

    natural_records = _deduplicate_natural_records(
        [
            r
            for r in natural_candidates
            if _has_size_in_range(r, natural_min_size, natural_max_size)
        ]
    )
    fake_by_generator = {
        generator: build_generator_class_records(
            dataset_root=dataset_root,
            generator_dirs=generator_dirs,
            generators=[generator],
            split=split,
            class_name=fake_class_name,
            label=1,
            allowed_extensions=allowed_extensions,
            include_image_size=False,
        )
        for generator in train_generators
    }
    fake_records = [record for records in fake_by_generator.values() for record in records]

    rng = random.Random(seed)
    num_fake_generators = len(train_generators)
    per_generator_fake_target = min(
        len(natural_records) // num_fake_generators,
        *(len(records) for records in fake_by_generator.values()),
    )
    if train_percent < 1.0:
        per_generator_fake_target = int(per_generator_fake_target * train_percent)
    if max_train_samples is not None:
        per_generator_fake_target = min(
            per_generator_fake_target,
            max_train_samples // (2 * num_fake_generators),
        )
    if per_generator_fake_target <= 0:
        raise ValueError(
            "No size-constrained training records selected. Check generator names, natural size bounds, "
            "train_percent, and max_train_samples."
        )

    selected_fake: list[GeneratorImageRecord] = []
    for generator in train_generators:
        selected_fake.extend(rng.sample(fake_by_generator[generator], per_generator_fake_target))
    selected_natural = rng.sample(natural_records, per_generator_fake_target * num_fake_generators)
    selected = [*selected_natural, *selected_fake]

    if not selected:
        raise ValueError(
            "No size-constrained training records selected. Check generator names, image dimensions, "
            "and size bounds."
        )

    rng.shuffle(selected)
    summary = SizeConstrainedSelectionSummary(
        natural_candidates=len(natural_records),
        fake_candidates=len(fake_records),
        selected_natural=len(selected_natural),
        selected_fake=len(selected_fake),
        selected_total=len(selected),
        fake_per_generator={
            generator: per_generator_fake_target for generator in train_generators
        },
    )
    return selected, summary


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
