from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd

from src.datasets import build_records_from_predefined_splits
from src.utils.io import load_yaml
from src.utils.paths import ensure_dir, resolve_from_root
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index Data Set 1-4 predefined split folders into metadata CSVs."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def _resolve_active_dataset(config: dict) -> tuple[str, pd.DataFrame]:
    dcfg = config["dataset"]
    active_name = str(dcfg["active_name"])
    supported_names = [str(x) for x in dcfg["supported_names"]]
    if active_name not in supported_names:
        raise ValueError(
            f"dataset.active_name='{active_name}' is not supported. "
            f"Choose one of: {supported_names}"
        )

    dataset_root = resolve_from_root(dcfg["root"]) / active_name
    records = build_records_from_predefined_splits(
        dataset_root=dataset_root,
        splits=[str(x) for x in dcfg["splits"]],
        class_to_label={str(k): int(v) for k, v in dcfg["class_to_label"].items()},
        allowed_extensions={str(x).lower() for x in dcfg["allowed_extensions"]},
    )
    return active_name, pd.DataFrame([r.__dict__ for r in records])


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    seed = int(config["project"]["seed"])
    set_global_seed(seed)

    active_name, df = _resolve_active_dataset(config)
    print(f"[prepare_dataset_index] Active dataset: {active_name}")
    print(f"[prepare_dataset_index] Total samples: {len(df)}")
    split_counts = Counter(df["split"].tolist())
    label_counts = Counter(df["label"].tolist())
    print(f"[prepare_dataset_index] Split counts: {dict(split_counts)}")
    print(f"[prepare_dataset_index] Label counts: {dict(label_counts)}")

    out_dir = ensure_dir(config["dataset"]["metadata_output_dir"])
    slug = active_name.lower().replace(" ", "_")

    for split in config["dataset"]["splits"]:
        sdf = df[df["split"] == split].copy()
        out_path = out_dir / f"{slug}_{split}.csv"
        sdf.to_csv(out_path, index=False)
        print(f"[prepare_dataset_index] Wrote {out_path} ({len(sdf)} rows)")


if __name__ == "__main__":
    main()
