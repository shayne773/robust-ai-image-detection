from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImageRecord:
    image_path: str
    label: int
    split: str
    class_name: str
    filename: str


def _find_case_insensitive_child(parent: Path, expected_name: str) -> Path:
    """Return child folder matching expected_name in a case-insensitive way."""
    expected = expected_name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == expected:
            return child
    raise FileNotFoundError(f"Missing directory '{expected_name}' under: {parent}")


def build_records_from_predefined_splits(
    dataset_root: Path,
    splits: Iterable[str],
    class_to_label: dict[str, int],
    allowed_extensions: set[str],
) -> list[ImageRecord]:
    """Scan `split/class/*.jpg` folders and return image metadata records."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    records: list[ImageRecord] = []
    for split in splits:
        split_dir = _find_case_insensitive_child(dataset_root, split)

        for class_name, label in class_to_label.items():
            class_dir = _find_case_insensitive_child(split_dir, class_name)
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in allowed_extensions:
                    records.append(
                        ImageRecord(
                            image_path=str(path.resolve()),
                            label=int(label),
                            split=split,
                            class_name=class_name,
                            filename=path.name,
                        )
                    )

    if not records:
        raise ValueError(
            f"No image files found under {dataset_root}. "
            f"Check directory structure and allowed extensions {sorted(allowed_extensions)}."
        )

    return records


class FolderImageDataset(Dataset):
    """PyTorch dataset backed by an in-memory list of image records."""

    def __init__(
        self,
        records: list[ImageRecord],
        split: str,
        transform: Any | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.records = [r for r in records if r.split.lower() == split.lower()]
        if not self.records:
            raise ValueError(f"No records found for split '{split}'.")

        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int] | tuple[Tensor, int, dict[str, Any]]:
        record = self.records[index]
        image_path = Path(record.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image missing: {image_path}")

        image = Image.open(image_path).convert("RGB")
        label = record.label

        if self.transform is not None:
            image = self.transform(image)

        if self.return_metadata:
            return image, label, {
                "image_path": record.image_path,
                "split": record.split,
                "class_name": record.class_name,
                "filename": record.filename,
                "label": record.label,
            }

        return image, label
