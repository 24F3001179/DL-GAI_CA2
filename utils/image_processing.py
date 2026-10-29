from __future__ import annotations

from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

MODEL_SIZE = 256


def prepare_image(image: Image.Image, size: int = MODEL_SIZE) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert an uploaded image to RGB, fit to square, and normalize to [-1, 1]."""
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image")
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_size = image.size
    # Match the official CycleGAN test preprocessing: resize to the model's 256px input.
    tensor = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])(image).unsqueeze(0)
    return tensor, original_size


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a [-1, 1] BCHW/CHW tensor to an RGB PIL image."""
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Expected a single-image batch")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Expected a 3-channel CHW tensor")
    array = tensor.detach().to(device="cpu", dtype=torch.float32).clamp(-1, 1)
    array = ((array + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def is_valid_output(image: Image.Image, expected_size: tuple[int, int] | None = None) -> bool:
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        return False
    if image.width < 1 or image.height < 1:
        return False
    if expected_size and image.size != expected_size:
        return False
    return np.asarray(image).ndim == 3
