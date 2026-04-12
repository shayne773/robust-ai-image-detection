from __future__ import annotations

import random
from io import BytesIO
from typing import Literal

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class JPEGReencode:
    """Re-encode a PIL image as JPEG in-memory and return the decoded image."""

    def __init__(self, quality: int) -> None:
        if not 1 <= int(quality) <= 100:
            raise ValueError(f"JPEG quality must be in [1, 100], got: {quality}")
        self.quality = int(quality)

    def __call__(self, image: Image.Image) -> Image.Image:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


class RandomJPEGReencode:
    """Sample a JPEG quality from a fixed set and re-encode in-memory."""

    def __init__(self, qualities: list[int]) -> None:
        cleaned = [int(q) for q in qualities]
        if not cleaned:
            raise ValueError("aligned_jpeg_qualities cannot be empty.")
        for q in cleaned:
            if not 1 <= q <= 100:
                raise ValueError(f"Each JPEG quality must be in [1, 100], got: {q}")
        self.qualities = cleaned

    def __call__(self, image: Image.Image) -> Image.Image:
        quality = random.choice(self.qualities)
        return JPEGReencode(quality=quality)(image)


def _get_interpolation(name: str) -> InterpolationMode:
    mapping = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "lanczos": InterpolationMode.LANCZOS,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(
            f"Unsupported interpolation '{name}'. Choose from {sorted(mapping)}."
        )
    return mapping[key]


def build_train_transforms(
    mode: Literal["baseline", "aligned"],
    image_size: int,
    interpolation: str = "bilinear",
    aligned_jpeg_qualities: list[int] | None = None,
) -> transforms.Compose:
    """Build train transforms for CIFAKE experiments."""
    interp = _get_interpolation(interpolation)

    if mode == "baseline":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=interp),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    if mode == "aligned":
        if aligned_jpeg_qualities is None:
            raise ValueError(
                "aligned_jpeg_qualities must be provided when train transform mode='aligned'."
            )
        return transforms.Compose(
            [
                RandomJPEGReencode(qualities=aligned_jpeg_qualities),
                transforms.Resize((image_size, image_size), interpolation=interp),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    raise ValueError(f"Unsupported train transform mode: {mode}")


def build_eval_transforms(
    mode: Literal["clean", "jpeg_eval"],
    image_size: int,
    interpolation: str = "bilinear",
    jpeg_eval_quality: int | None = None,
) -> transforms.Compose:
    """Build eval transforms for clean or degraded conditions."""
    interp = _get_interpolation(interpolation)

    if mode == "clean":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=interp),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    if mode == "jpeg_eval":
        if jpeg_eval_quality is None:
            raise ValueError("jpeg_eval_quality must be provided when eval mode='jpeg_eval'.")
        return transforms.Compose(
            [
                JPEGReencode(quality=int(jpeg_eval_quality)),
                transforms.Resize((image_size, image_size), interpolation=interp),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    raise ValueError(f"Unsupported eval transform mode: {mode}")
