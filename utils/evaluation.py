from __future__ import annotations

import math

import numpy as np
from PIL import Image


def paired_metrics(prediction: Image.Image, target: Image.Image) -> dict[str, float]:
    """Compute MAE/PSNR only for a spatially aligned paired reference image."""
    if prediction.size != target.size:
        raise ValueError(
            "Paired metrics require prediction and target images with identical aligned dimensions; "
            f"received {prediction.size} and {target.size}."
        )
    pred = np.asarray(prediction.convert("RGB"), dtype=np.float32) / 255.0
    truth = np.asarray(target.convert("RGB"), dtype=np.float32) / 255.0
    mae = float(np.mean(np.abs(pred - truth)))
    mse = float(np.mean((pred - truth) ** 2))
    psnr = math.inf if mse == 0 else float(10 * math.log10(1.0 / mse))
    return {"mae": mae, "psnr_db": psnr}


def cycle_consistency_mae(original: Image.Image, reconstruction: Image.Image) -> float:
    """Pixel MAE for an image and its cycle reconstruction; not a realism metric."""
    return paired_metrics(original, reconstruction)["mae"]
