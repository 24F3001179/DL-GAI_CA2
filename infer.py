from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image

from utils.inference import MODEL_SOURCES, get_device, load_generator, translate

TASK_IDS = {"horse2zebra": "Horse → Zebra", "zebra2horse": "Zebra → Horse"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pretrained ImageLab AI generator on one image")
    parser.add_argument("--task", choices=list(TASK_IDS), default="horse2zebra")
    parser.add_argument("--input", type=Path, default=Path("assets/sample_horse.jpg"))
    parser.add_argument("--output", type=Path, default=Path("results/sample_translation.png"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    details = MODEL_SOURCES[TASK_IDS[args.task]]
    checkpoint = args.checkpoint_dir / details["file"]
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found at {checkpoint}. Run download_model.py first.")
    if not args.input.is_file():
        parser.error(f"Input image not found: {args.input}")
    device = get_device()
    generator = load_generator(checkpoint, device)
    with Image.open(args.input) as source:
        source = source.convert("RGB")
        started = time.perf_counter()
        result = translate(generator, source, device)
        elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    direction = details["direction"].replace("→", "to")
    print(f"Task: {direction}")
    print(f"Input: {source.width}x{source.height}; output: {result.width}x{result.height}")
    print(f"Device: {device}; inference time: {elapsed:.3f}s")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
