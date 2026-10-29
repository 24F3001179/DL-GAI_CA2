from __future__ import annotations

from pathlib import Path

import torch

from models.cyclegan import ResnetGenerator
from utils.image_processing import MODEL_SIZE, prepare_image, tensor_to_image

MODEL_SOURCES = {
    "Horse → Zebra": {
        "file": "horse2zebra.pth",
        "url": "https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/horse2zebra.pth",
        "direction": "Horse → Zebra",
    },
    "Zebra → Horse": {
        "file": "zebra2horse.pth",
        "url": "https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/zebra2horse.pth",
        "direction": "Zebra → Horse",
    },
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_generator(checkpoint_path: str | Path, device: torch.device | None = None) -> ResnetGenerator:
    device = device or get_device()
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    generator = ResnetGenerator()
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint is not a PyTorch state dictionary")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    generator.load_state_dict(state, strict=True)
    generator.to(device).eval()
    return generator


def translate(generator: ResnetGenerator, image, device: torch.device | None = None):
    device = device or next(generator.parameters()).device
    tensor, original_size = prepare_image(image, MODEL_SIZE)
    with torch.inference_mode():
        output = generator(tensor.to(device))
    return tensor_to_image(output).resize(original_size, resample=ImageResampling.BICUBIC)


from PIL import Image as _Image
ImageResampling = _Image.Resampling
