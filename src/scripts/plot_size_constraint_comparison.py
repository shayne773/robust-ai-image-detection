from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


EVAL_FILE_RE = re.compile(
    r"(?P<model>.+?)_train-(?P<train_generator>[^_]+)_comp-(?P<train_compression>[^_]+)"
    r"_eval-(?P<eval_generator>[^_]+)_test-(?P<test_compression>[^_]+)"
    r"_(?P<timestamp>\d{8}_\d{6})_metrics\.json$"
)

METRIC_CHOICES = ("accuracy", "roc_auc", "recall")
DEFAULT_TRAIN_GENERATORS = ("sdv5", "sdv4", "wukong")
DEFAULT_EVAL_GENERATORS = ("sdv5", "sdv4", "wukong", "biggan", "adm", "glide")
DEFAULT_SIZE_CHECKPOINT_TEMPLATE = (
    "outputs/models/best_resnet50_sizeconstrained_train-{train}_comp-{train_compression}"
    "_nat-450-550_fake-512_crop-450_pct-1.0.pt"
)
DEFAULT_STANDARD_CHECKPOINT_TEMPLATE = (
    "outputs/models/best_resnet50_train-{train}_comp-{train_compression}_pct-0.3.pt"
)

GENERATOR_ALIASES = {
    "adml": "adm",
}

DISPLAY_NAMES = {
    "sdv5": "SDv5\n512px",
    "sdv4": "SDv4\n512px",
    "wukong": "Wukong\n512px",
    "biggan": "BigGAN\n128px",
    "adm": "ADM\n256px",
    "glide": "GLIDE\n256px",
}


@dataclass(frozen=True)
class EvalResult:
    path: Path
    train_generator: str
    train_compression: str
    eval_generator: str
    test_compression: str
    timestamp: str
    metrics: dict[str, Any]
    preprocess_mode: str | None


@dataclass(frozen=True)
class MissingCell:
    train_generator: str
    eval_generator: str
    preprocess_mode: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Figure 5-style size-constrained vs standard cross-generator comparison."
        )
    )
    parser.add_argument("--eval-dir", default="outputs/eval")
    parser.add_argument("--output", default=None)
    parser.add_argument("--metric", choices=METRIC_CHOICES, default="accuracy")
    parser.add_argument("--train-generators", nargs="+", default=list(DEFAULT_TRAIN_GENERATORS))
    parser.add_argument("--eval-generators", nargs="+", default=list(DEFAULT_EVAL_GENERATORS))
    parser.add_argument("--train-compression", default="jpeg96")
    parser.add_argument("--test-compression", default="jpeg96")
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument(
        "--print-missing-commands",
        action="store_true",
        help="Print evaluate_detector commands for missing cells.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if any requested cell is missing instead of plotting N/A cells.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--metric-vmin", type=float, default=50.0)
    parser.add_argument("--metric-vmax", type=float, default=100.0)
    parser.add_argument("--diff-vmin", type=float, default=None)
    parser.add_argument("--diff-vmax", type=float, default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--size-checkpoint-template",
        default=DEFAULT_SIZE_CHECKPOINT_TEMPLATE,
        help="Template used for missing size-constrained eval commands.",
    )
    parser.add_argument(
        "--standard-checkpoint-template",
        default=DEFAULT_STANDARD_CHECKPOINT_TEMPLATE,
        help="Template used for missing standard eval commands.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    return GENERATOR_ALIASES.get(normalized, normalized)


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.upper())


def display_metric(metric: str) -> str:
    mapping = {
        "accuracy": "Accuracy",
        "roc_auc": "ROC-AUC",
        "recall": "Recall",
    }
    return mapping.get(metric, metric)


def load_results(eval_dir: Path) -> list[EvalResult]:
    results: list[EvalResult] = []
    for path in eval_dir.glob("*_metrics.json"):
        match = EVAL_FILE_RE.match(path.name)
        if not match:
            continue

        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        eval_generators = metrics.get("eval_generators")
        if isinstance(eval_generators, list) and len(eval_generators) != 1:
            print(f"Skipping multi-generator eval JSON: {path}")
            continue

        groups = match.groupdict()
        results.append(
            EvalResult(
                path=path,
                train_generator=normalize_name(groups["train_generator"]),
                train_compression=groups["train_compression"].lower(),
                eval_generator=normalize_name(groups["eval_generator"]),
                test_compression=groups["test_compression"].lower(),
                timestamp=groups["timestamp"],
                metrics=metrics,
                preprocess_mode=metrics.get("preprocess_mode"),
            )
        )
    return results


def pick_latest_result(
    results: list[EvalResult],
    train_generator: str,
    eval_generator: str,
    train_compression: str,
    test_compression: str,
    preprocess_mode: str,
) -> EvalResult | None:
    matches = [
        result
        for result in results
        if result.train_generator == train_generator
        and result.eval_generator == eval_generator
        and result.train_compression == train_compression
        and result.test_compression == test_compression
        and result.preprocess_mode == preprocess_mode
    ]
    if not matches:
        return None
    return max(matches, key=lambda result: result.timestamp)


def metric_to_percent(value: float) -> float:
    if 0.0 <= value <= 1.0:
        return value * 100.0
    return value


def build_matrix(
    results: list[EvalResult],
    train_generators: list[str],
    eval_generators: list[str],
    train_compression: str,
    test_compression: str,
    metric: str,
    preprocess_mode: str,
) -> tuple[np.ndarray, list[list[EvalResult | None]], list[MissingCell]]:
    matrix = np.full((len(train_generators), len(eval_generators)), np.nan, dtype=float)
    sources: list[list[EvalResult | None]] = []
    missing: list[MissingCell] = []

    for row_idx, train_generator in enumerate(train_generators):
        source_row: list[EvalResult | None] = []
        for col_idx, eval_generator in enumerate(eval_generators):
            result = pick_latest_result(
                results=results,
                train_generator=train_generator,
                eval_generator=eval_generator,
                train_compression=train_compression,
                test_compression=test_compression,
                preprocess_mode=preprocess_mode,
            )
            source_row.append(result)
            if result is None:
                missing.append(
                    MissingCell(train_generator, eval_generator, preprocess_mode, "missing JSON")
                )
                continue

            value = result.metrics.get(metric)
            if value is None:
                missing.append(
                    MissingCell(train_generator, eval_generator, preprocess_mode, f"missing {metric}")
                )
                continue
            matrix[row_idx, col_idx] = metric_to_percent(float(value))
        sources.append(source_row)

    return matrix, sources, missing


def default_output_path(metric: str) -> Path:
    return Path("outputs/plots") / f"size_constraint_comparison_{metric}.png"


def finite_or_default(values: np.ndarray, default: float) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return default
    return float(np.nanmax(np.abs(finite_values)))


def annotate_cells(
    ax: plt.Axes,
    matrix: np.ndarray,
    fmt: str = ".2f",
    white_threshold: float | None = None,
    use_abs_threshold: bool = False,
) -> None:
    finite_values = matrix[np.isfinite(matrix)]
    if white_threshold is None:
        threshold = float(np.nanmean(finite_values)) if finite_values.size else math.inf
    else:
        threshold = white_threshold

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                text = format(value, fmt)
                contrast_value = abs(value) if use_abs_threshold else value
                color = "white" if contrast_value >= threshold else "#172033"
            else:
                text = "N/A"
                color = "#697386"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color=color, fontsize=9, weight="bold")


def style_heatmap_axes(
    ax: plt.Axes,
    train_generators: list[str],
    eval_generators: list[str],
    title: str,
) -> None:
    ax.set_xticks(np.arange(len(eval_generators)), labels=[display_name(g) for g in eval_generators])
    ax.set_yticks(np.arange(len(train_generators)), labels=[display_name(g) for g in train_generators])
    ax.set_xlabel("Testing Subset", weight="bold")
    ax.set_ylabel("Training Subset", weight="bold")
    ax.set_title(title, weight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_xticks(np.arange(-0.5, len(eval_generators), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(train_generators), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_comparison(
    size_matrix: np.ndarray,
    standard_matrix: np.ndarray,
    train_generators: list[str],
    eval_generators: list[str],
    metric: str,
    output_path: Path,
    title: str | None,
    dpi: int,
    metric_vmin: float,
    metric_vmax: float,
    diff_vmin: float | None,
    diff_vmax: float | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diff_matrix = size_matrix - standard_matrix

    max_abs_diff = finite_or_default(diff_matrix, 50.0)
    if diff_vmin is None:
        diff_vmin = -max_abs_diff
    if diff_vmax is None:
        diff_vmax = max_abs_diff

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.4), constrained_layout=True)
    if title:
        fig.suptitle(title, weight="bold", fontsize=13)
    metric_label = display_metric(metric)

    size_masked = np.ma.masked_invalid(size_matrix)
    size_im = axes[0].imshow(size_masked, cmap="Blues", vmin=metric_vmin, vmax=metric_vmax)
    style_heatmap_axes(
        axes[0],
        train_generators=train_generators,
        eval_generators=eval_generators,
        title=f"{metric_label} for size-constrained training",
    )
    annotate_cells(axes[0], size_matrix, white_threshold=(metric_vmin + metric_vmax) / 2.0)
    size_cb = fig.colorbar(size_im, ax=axes[0], fraction=0.046, pad=0.04)
    size_cb.set_label(f"{metric_label} (%)")

    diff_masked = np.ma.masked_invalid(diff_matrix)
    if diff_vmin < 0.0 < diff_vmax:
        diff_norm = TwoSlopeNorm(vmin=diff_vmin, vcenter=0.0, vmax=diff_vmax)
        diff_im = axes[1].imshow(diff_masked, cmap="coolwarm", norm=diff_norm)
    else:
        diff_im = axes[1].imshow(diff_masked, cmap="coolwarm", vmin=diff_vmin, vmax=diff_vmax)
    style_heatmap_axes(
        axes[1],
        train_generators=train_generators,
        eval_generators=eval_generators,
        title="Difference to standard training",
    )
    diff_white_threshold = max(abs(diff_vmin), abs(diff_vmax)) * 0.5
    annotate_cells(
        axes[1],
        diff_matrix,
        white_threshold=diff_white_threshold,
        use_abs_threshold=True,
    )
    diff_cb = fig.colorbar(diff_im, ax=axes[1], fraction=0.046, pad=0.04)
    diff_cb.set_label("percentage points")

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def print_sources(
    label: str,
    sources: list[list[EvalResult | None]],
    train_generators: list[str],
    eval_generators: list[str],
) -> None:
    print(f"\n[{label}] selected sources")
    for row_idx, train_generator in enumerate(train_generators):
        for col_idx, eval_generator in enumerate(eval_generators):
            result = sources[row_idx][col_idx]
            source = str(result.path) if result else "MISSING"
            print(f"{train_generator} -> {eval_generator}: {source}")


def print_missing_cells(missing: list[MissingCell]) -> None:
    if not missing:
        return
    print("\nMissing requested cells:")
    for cell in missing:
        print(
            f"- {cell.preprocess_mode}: train={cell.train_generator}, "
            f"eval={cell.eval_generator} ({cell.reason})"
        )


def checkpoint_from_template(
    template: str,
    train_generator: str,
    train_compression: str,
) -> str:
    return template.format(train=train_generator, train_compression=train_compression)


def print_missing_commands(
    missing: list[MissingCell],
    config: str,
    train_compression: str,
    test_compression: str,
    size_checkpoint_template: str,
    standard_checkpoint_template: str,
) -> None:
    if not missing:
        return

    seen: set[tuple[str, str, str]] = set()
    print("\nCommands for missing eval JSONs:")
    for cell in missing:
        if cell.reason != "missing JSON":
            continue
        key = (cell.train_generator, cell.eval_generator, cell.preprocess_mode)
        if key in seen:
            continue
        seen.add(key)

        if cell.preprocess_mode == "size-constrained":
            checkpoint = checkpoint_from_template(
                size_checkpoint_template,
                train_generator=cell.train_generator,
                train_compression=train_compression,
            )
        else:
            checkpoint = checkpoint_from_template(
                standard_checkpoint_template,
                train_generator=cell.train_generator,
                train_compression=train_compression,
            )

        print(
            "python -m src.scripts.evaluate_detector "
            f"--config {config} "
            f"--checkpoint {checkpoint} "
            f"--eval-generators {cell.eval_generator} "
            f"--test-compression {test_compression} "
            f"--preprocess-mode {cell.preprocess_mode}"
        )


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    train_generators = [normalize_name(generator) for generator in args.train_generators]
    eval_generators = [normalize_name(generator) for generator in args.eval_generators]
    train_compression = args.train_compression.lower()
    test_compression = args.test_compression.lower()

    results = load_results(eval_dir)
    if not results:
        raise FileNotFoundError(f"No parseable *_metrics.json files found in {eval_dir}")

    size_matrix, size_sources, size_missing = build_matrix(
        results=results,
        train_generators=train_generators,
        eval_generators=eval_generators,
        train_compression=train_compression,
        test_compression=test_compression,
        metric=args.metric,
        preprocess_mode="size-constrained",
    )
    standard_matrix, standard_sources, standard_missing = build_matrix(
        results=results,
        train_generators=train_generators,
        eval_generators=eval_generators,
        train_compression=train_compression,
        test_compression=test_compression,
        metric=args.metric,
        preprocess_mode="standard",
    )
    missing = [*size_missing, *standard_missing]

    if args.show_sources:
        print_sources("size-constrained", size_sources, train_generators, eval_generators)
        print_sources("standard", standard_sources, train_generators, eval_generators)

    print_missing_cells(missing)
    if args.print_missing_commands:
        print_missing_commands(
            missing=missing,
            config=args.config,
            train_compression=train_compression,
            test_compression=test_compression,
            size_checkpoint_template=args.size_checkpoint_template,
            standard_checkpoint_template=args.standard_checkpoint_template,
        )
    if args.strict_missing and missing:
        raise FileNotFoundError("Missing requested eval cells. See messages above.")

    output_path = Path(args.output) if args.output else default_output_path(args.metric)
    plot_comparison(
        size_matrix=size_matrix,
        standard_matrix=standard_matrix,
        train_generators=train_generators,
        eval_generators=eval_generators,
        metric=args.metric,
        output_path=output_path,
        title=args.title,
        dpi=int(args.dpi),
        metric_vmin=float(args.metric_vmin),
        metric_vmax=float(args.metric_vmax),
        diff_vmin=args.diff_vmin,
        diff_vmax=args.diff_vmax,
    )
    print(f"\nSaved comparison plot to {output_path}")


if __name__ == "__main__":
    main()
