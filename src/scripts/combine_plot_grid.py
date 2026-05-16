from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine existing PNG plots into a labeled grid.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input PNG paths in row-major order.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--labels", nargs="+", default=None, help="Optional panel labels.")
    parser.add_argument("--title", default=None, help="Optional figure title.")
    parser.add_argument("--cols", type=int, default=2, help="Number of grid columns.")
    parser.add_argument("--gap", type=int, default=28, help="Pixels between panels.")
    parser.add_argument("--margin", type=int, default=36, help="Outer margin in pixels.")
    parser.add_argument("--label-height", type=int, default=34, help="Reserved label height above each panel.")
    parser.add_argument("--background", default="white", help="Canvas background color.")
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs]
    missing = [path for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input plot(s): " + ", ".join(str(path) for path in missing))

    if args.labels is not None and len(args.labels) != len(input_paths):
        raise ValueError("--labels must have the same number of entries as --inputs")

    images = [Image.open(path).convert("RGB") for path in input_paths]
    cols = max(1, int(args.cols))
    rows = math.ceil(len(images) / cols)
    cell_width = max(image.width for image in images)
    cell_image_height = max(image.height for image in images)
    title_height = 54 if args.title else 0
    label_height = args.label_height if args.labels else 0

    width = args.margin * 2 + cols * cell_width + (cols - 1) * args.gap
    height = (
        args.margin * 2
        + title_height
        + rows * (label_height + cell_image_height)
        + (rows - 1) * args.gap
    )

    canvas = Image.new("RGB", (width, height), args.background)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28, bold=True)
    label_font = load_font(21, bold=True)

    if args.title:
        title_width, title_text_height = text_size(draw, args.title, title_font)
        draw.text(
            ((width - title_width) / 2, args.margin + (title_height - title_text_height) / 2 - 4),
            args.title,
            fill="#172033",
            font=title_font,
        )

    grid_top = args.margin + title_height
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        cell_left = args.margin + col * (cell_width + args.gap)
        cell_top = grid_top + row * (label_height + cell_image_height + args.gap)

        if args.labels:
            label = args.labels[idx]
            label_width, label_text_height = text_size(draw, label, label_font)
            draw.text(
                (cell_left + (cell_width - label_width) / 2, cell_top + (label_height - label_text_height) / 2 - 2),
                label,
                fill="#243044",
                font=label_font,
            )

        image_left = cell_left + (cell_width - image.width) // 2
        image_top = cell_top + label_height + (cell_image_height - image.height) // 2
        canvas.paste(image, (image_left, image_top))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved combined plot to {output_path}")


if __name__ == "__main__":
    main()
