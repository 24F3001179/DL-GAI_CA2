from PIL import Image
import pytest

from utils.image_processing import is_valid_output, prepare_image, tensor_to_image
from utils.evaluation import paired_metrics


def test_preprocessing_makes_rgb_256_tensor_normalized():
    tensor, original_size = prepare_image(Image.new("L", (640, 320), color=127))
    assert tuple(tensor.shape) == (1, 3, 256, 256)
    assert original_size == (640, 320)
    assert float(tensor.min()) >= -1.0
    assert float(tensor.max()) <= 1.0


def test_output_conversion_and_validation():
    image = tensor_to_image(__import__("torch").zeros(1, 3, 256, 256))
    assert image.size == (256, 256)
    assert image.mode == "RGB"
    assert is_valid_output(image, (256, 256))
    assert not is_valid_output(Image.new("L", (10, 10)))


def test_preprocessing_rejects_non_pil_input():
    with pytest.raises(TypeError):
        prepare_image("not an image")


def test_paired_metrics_require_identically_aligned_dimensions():
    reference = Image.new("RGB", (8, 8), color=(64, 64, 64))
    assert paired_metrics(reference, reference) == {"mae": 0.0, "psnr_db": float("inf")}
    with pytest.raises(ValueError, match="identical aligned dimensions"):
        paired_metrics(reference, Image.new("RGB", (8, 9)))
