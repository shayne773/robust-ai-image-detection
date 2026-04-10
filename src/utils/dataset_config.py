from __future__ import annotations

from pathlib import Path

from src.utils.paths import resolve_from_root


SUPPORTED_DATASETS = {"Data Set 1", "Data Set 2", "Data Set 3", "Data Set 4"}
SUPPORTED_SPLITS = {"train", "validation", "test"}


def resolve_active_dataset_root(config: dict) -> tuple[str, Path]:
    """Validate active dataset and return its absolute root path."""
    dcfg = config["dataset"]
    active_name = str(dcfg["active_name"])
    configured_names = {str(x) for x in dcfg.get("supported_names", [])}

    if configured_names != SUPPORTED_DATASETS:
        raise ValueError(
            "dataset.supported_names must exactly match the supported datasets: "
            f"{sorted(SUPPORTED_DATASETS)}"
        )

    if active_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset.active_name='{active_name}' is invalid. "
            f"Choose one of: {sorted(SUPPORTED_DATASETS)}"
        )

    dataset_root = resolve_from_root(dcfg["root"]) / active_name
    return active_name, dataset_root


def validate_dataset_section(config: dict) -> None:
    """Validate split and class map constraints for this reduced stage."""
    dcfg = config["dataset"]
    configured_splits = {str(x).lower() for x in dcfg["splits"]}
    if configured_splits != SUPPORTED_SPLITS:
        raise ValueError(
            "dataset.splits must be exactly ['train', 'validation', 'test'] "
            f"for this stage. Got: {sorted(configured_splits)}"
        )

    class_to_label = {str(k).lower(): int(v) for k, v in dcfg["class_to_label"].items()}
    expected_map = {"real": 0, "fake": 1}
    if class_to_label != expected_map:
        raise ValueError(
            "dataset.class_to_label must be exactly {'real': 0, 'fake': 1} "
            f"for this stage. Got: {class_to_label}"
        )
