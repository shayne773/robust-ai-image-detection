from __future__ import annotations

from typing import Literal

from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
        raise NotImplementedError(
            "Train transform mode 'aligned' is reserved for future work "
            "(JPEG/resize alignment applied identically across classes)."
        )

    raise ValueError(f"Unsupported train transform mode: {mode}")


def build_eval_transforms(
    mode: Literal["clean", "jpeg_eval"],
    image_size: int,
    interpolation: str = "bilinear",
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
        raise NotImplementedError(
            "Eval transform mode 'jpeg_eval' is reserved for future work "
            "(evaluation-time JPEG degradation pipeline)."
        )

    raise ValueError(f"Unsupported eval transform mode: {mode}")
