from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets import load_or_build_size_constrained_natural_records
from src.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or rebuild the filtered natural-image manifest for size-constrained training."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--natural-pool-generators", nargs="+", default=None)
    parser.add_argument("--natural-min-size", type=int, default=None)
    parser.add_argument("--natural-max-size", type=int, default=None)
    parser.add_argument("--size-cache", default=None)
    parser.add_argument("--natural-manifest", default=None)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild even if the manifest already exists.")
    return parser.parse_args()


def _config_list(value: object | None, fallback: list[str]) -> list[str]:
    if value is None:
        return fallback
    if not isinstance(value, list):
        raise ValueError(f"Expected a list in config, got: {value}")
    return [str(item) for item in value]


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    dcfg = cfg["dataset"]
    xcfg = cfg.get("size_constraint", {})

    natural_pool_generators = args.natural_pool_generators or _config_list(
        xcfg.get("natural_pool_generators"), [str(x) for x in dcfg["train_generators"]]
    )
    natural_min_size = int(args.natural_min_size or xcfg.get("natural_min_size", 450))
    natural_max_size = int(args.natural_max_size or xcfg.get("natural_max_size", 550))
    size_cache = Path(args.size_cache or xcfg.get("size_cache", "outputs/cache/genimage_image_sizes.json"))
    natural_manifest = Path(
        args.natural_manifest
        or xcfg.get(
            "natural_manifest",
            "outputs/splits/size_constrained_natural_wukong-sdv4-sdv5_450-550.json",
        )
    )
    root = Path(dcfg["root"])
    allowed_ext = {str(e).lower() for e in dcfg.get("allowed_extensions", [".jpg", ".jpeg", ".png"])}

    print("[prepare_size_constrained_natural_manifest] Preparing filtered natural manifest")
    print(f"[prepare_size_constrained_natural_manifest] Natural pool generators: {natural_pool_generators}")
    print(f"[prepare_size_constrained_natural_manifest] Natural size range: [{natural_min_size}, {natural_max_size}]")
    print(f"[prepare_size_constrained_natural_manifest] Size cache: {size_cache}")
    print(f"[prepare_size_constrained_natural_manifest] Natural manifest: {natural_manifest}")

    records, source = load_or_build_size_constrained_natural_records(
        dataset_root=root,
        generator_dirs={str(k): str(v) for k, v in dcfg["generator_dirs"].items()},
        natural_pool_generators=natural_pool_generators,
        split=str(dcfg["train_split"]),
        real_class_name=str(dcfg["real_class_name"]),
        allowed_extensions=allowed_ext,
        natural_min_size=natural_min_size,
        natural_max_size=natural_max_size,
        size_cache_path=size_cache,
        natural_manifest_path=natural_manifest,
        rebuild_natural_manifest=bool(args.rebuild),
    )

    print(f"[prepare_size_constrained_natural_manifest] Manifest source: {source}")
    print(f"[prepare_size_constrained_natural_manifest] Filtered natural records: {len(records)}")


if __name__ == "__main__":
    main()
