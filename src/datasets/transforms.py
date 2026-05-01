from __future__ import annotations

from io import BytesIO
from typing import Literal

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SUPPORTED_COMPRESSION_MODES = {"raw": None, "jpeg96": 96, "jpeg95": 95, "jpeg90": 90}


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


class JPEGCompressionTransform:
    def __init__(self, mode: str) -> None:
        if mode not in SUPPORTED_COMPRESSION_MODES:
            raise ValueError(
                f"Unsupported compression mode '{mode}'. Supported: {sorted(SUPPORTED_COMPRESSION_MODES)}"
            )
        self.mode = mode
        self.quality = SUPPORTED_COMPRESSION_MODES[mode]

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.quality is None:
            return image.convert("RGB")
        return apply_jpeg_compression(image, self.quality)


def _get_interpolation(name: str) -> InterpolationMode:
    mapping = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "lanczos": InterpolationMode.LANCZOS,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported interpolation '{name}'. Choose from {sorted(mapping)}.")
    return mapping[key]


def build_detector_transforms(
    compression_mode: str,
    image_size: int,
    interpolation: str = "bilinear",
    is_train: bool = False,
) -> transforms.Compose:
    interp = _get_interpolation(interpolation)
    ops: list = [JPEGCompressionTransform(compression_mode), transforms.Resize((image_size, image_size), interpolation=interp)]
    if is_train:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transforms.Compose(ops)
