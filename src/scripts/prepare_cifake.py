from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.io import load_yaml
from src.utils.paths import ensure_dir, resolve_from_root
from src.utils.seed import set_global_seed

REAL_LABEL = 0
FAKE_LABEL = 1


@dataclass(frozen=True)
class ClassSpec:
    label: int
    class_name: str
    keywords: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CIFAKE split CSV files.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def _normalize_text(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _build_class_specs(cfg: dict) -> list[ClassSpec]:
    dataset_cfg = cfg["dataset"]
    real_keywords = tuple(_normalize_text(x) for x in dataset_cfg.get("real_dirnames", []))
    fake_keywords = tuple(_normalize_text(x) for x in dataset_cfg.get("fake_dirnames", []))

    if not real_keywords:
        real_keywords = ("real", "true", "authentic", "genuine")
    if not fake_keywords:
        fake_keywords = ("fake", "ai", "generated", "synthetic")

    return [
        ClassSpec(label=REAL_LABEL, class_name="real", keywords=real_keywords),
        ClassSpec(label=FAKE_LABEL, class_name="ai-generated", keywords=fake_keywords),
    ]


def _iter_image_files(root: Path, allowed_exts: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed_exts:
            yield path


def _infer_label_from_path(path: Path, specs: list[ClassSpec]) -> tuple[int, str] | None:
    parent_tokens = [_normalize_text(p.name) for p in path.parents]

    for token in parent_tokens:
        for spec in specs:
            if any(keyword in token for keyword in spec.keywords):
                return spec.label, spec.class_name
    return None


def scan_cifake(cifake_root: Path, allowed_exts: set[str], specs: list[ClassSpec]) -> pd.DataFrame:
    if not cifake_root.exists():
        raise FileNotFoundError(f"CIFAKE root does not exist: {cifake_root}")

    image_files = list(_iter_image_files(cifake_root, allowed_exts))
    if not image_files:
        raise ValueError(f"No images found under {cifake_root} for extensions {sorted(allowed_exts)}")

    records: list[dict] = []
    skipped = 0
    for image_path in tqdm(image_files, desc="Scanning CIFAKE files"):
        inferred = _infer_label_from_path(image_path, specs)
        if inferred is None:
            skipped += 1
            continue

        label, class_name = inferred
        records.append(
            {
                "image_path": str(image_path.resolve()),
                "label": int(label),
                "class_name": class_name,
                "filename": image_path.name,
            }
        )

    if not records:
        raise ValueError(
            "No labeled samples found. Update dataset.real_dirnames and "
            "dataset.fake_dirnames in config to match your CIFAKE structure."
        )

    df = pd.DataFrame.from_records(records)
    print(f"[prepare_cifake] Total discovered images: {len(image_files)}")
    print(f"[prepare_cifake] Labeled samples used: {len(df)}")
    print(f"[prepare_cifake] Unlabeled/skipped samples: {skipped}")
    print("[prepare_cifake] Class counts:")
    print(df["label"].value_counts().sort_index())

    return df


def _validate_ratios(ratios: dict[str, float]) -> tuple[float, float, float]:
    train_ratio = float(ratios["train"])
    val_ratio = float(ratios["val"])
    test_ratio = float(ratios["test"])
    total = train_ratio + val_ratio + test_ratio

    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")

    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("All split ratios must be > 0")

    return train_ratio, val_ratio, test_ratio


def create_splits(df: pd.DataFrame, ratios: dict[str, float], seed: int) -> dict[str, pd.DataFrame]:
    train_ratio, val_ratio, test_ratio = _validate_ratios(ratios)

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=df["label"],
    )

    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=seed,
        stratify=temp_df["label"],
    )

    split_map = {
        "train": train_df.copy(),
        "val": val_df.copy(),
        "test": test_df.copy(),
    }

    for split_name, split_df in split_map.items():
        split_df.loc[:, "split"] = split_name
        split_df.sort_values(by=["label", "filename", "image_path"], inplace=True)
        split_df.reset_index(drop=True, inplace=True)

    return split_map


def save_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in splits.items():
        out_path = output_dir / f"cifake_{split_name}.csv"
        columns = ["image_path", "label", "split", "class_name", "filename"]
        split_df.to_csv(out_path, index=False, columns=columns)
        print(f"[prepare_cifake] Wrote {split_name}: {out_path} ({len(split_df)} rows)")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    seed = int(config["project"]["seed"])
    set_global_seed(seed)

    dataset_cfg = config["dataset"]
    cifake_root = resolve_from_root(dataset_cfg["cifake_root"])
    allowed_exts = {ext.lower() for ext in dataset_cfg["allowed_extensions"]}
    specs = _build_class_specs(config)

    df = scan_cifake(cifake_root=cifake_root, allowed_exts=allowed_exts, specs=specs)
    splits = create_splits(df=df, ratios=dataset_cfg["split_ratios"], seed=seed)

    split_output_dir = ensure_dir(dataset_cfg["split_output_dir"])
    save_splits(splits, split_output_dir)


if __name__ == "__main__":
    main()
