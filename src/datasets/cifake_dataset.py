from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CIFAKESample:
    image_path: str
    label: int
    split: str
    class_name: str
    filename: str


class CIFAKEDataset(Dataset):
    """CSV-backed CIFAKE dataset.

    Expected columns in split CSV:
    - image_path
    - label
    Optional columns:
    - split, class_name, filename
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: Any | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self._validate_dataframe(self.df)

        self.transform = transform
        self.return_metadata = return_metadata

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        required = {"image_path", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        if df.empty:
            raise ValueError("Dataset CSV has no rows.")

        invalid_labels = set(df["label"].unique()) - {0, 1}
        if invalid_labels:
            raise ValueError(
                f"Detected labels outside binary mapping {{0, 1}}: {sorted(invalid_labels)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[Tensor, int] | tuple[Tensor, int, dict[str, Any]]:
        row = self.df.iloc[index]

        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image missing: {image_path}")

        image = Image.open(image_path).convert("RGB")
        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        if self.return_metadata:
            metadata = {
                "image_path": str(image_path),
                "label": label,
                "split": row.get("split", ""),
                "class_name": row.get("class_name", ""),
                "filename": row.get("filename", image_path.name),
            }
            return image, label, metadata

        return image, label
