from __future__ import annotations

import argparse
import csv
from collections import Counter

from src.datasets import build_records_from_predefined_splits
from src.utils.io import load_yaml
from src.utils.paths import ensure_dir
from src.utils.seed import set_global_seed
from src.utils.dataset_config import resolve_active_dataset_root, validate_dataset_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index Data Set 1-4 predefined split folders into metadata CSVs."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def _resolve_active_dataset(config: dict) -> tuple[str, list[dict[str, str | int]]]:
    dcfg = config["dataset"]
    validate_dataset_section(config)
    active_name, dataset_root = resolve_active_dataset_root(config)
    records = build_records_from_predefined_splits(
        dataset_root=dataset_root,
        splits=[str(x) for x in dcfg["splits"]],
        class_to_label={str(k): int(v) for k, v in dcfg["class_to_label"].items()},
        allowed_extensions={str(x).lower() for x in dcfg["allowed_extensions"]},
    )
    return active_name, [r.__dict__ for r in records]


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    seed = int(config["project"]["seed"])
    set_global_seed(seed)

    active_name, records = _resolve_active_dataset(config)
    print(f"[prepare_dataset_index] Active dataset: {active_name}")
    print(f"[prepare_dataset_index] Total samples: {len(records)}")
    split_counts = Counter(record["split"] for record in records)
    label_counts = Counter(record["label"] for record in records)
    print(f"[prepare_dataset_index] Split counts: {dict(split_counts)}")
    print(f"[prepare_dataset_index] Label counts: {dict(label_counts)}")

    out_dir = ensure_dir(config["dataset"]["metadata_output_dir"])
    slug = active_name.lower().replace(" ", "_")
    fieldnames = ["image_path", "label", "split", "class_name", "filename"]

    for split in config["dataset"]["splits"]:
        split_rows = [row for row in records if row["split"] == split]
        out_path = out_dir / f"{slug}_{split}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"[prepare_dataset_index] Wrote {out_path} ({len(split_rows)} rows)")


if __name__ == "__main__":
    main()
