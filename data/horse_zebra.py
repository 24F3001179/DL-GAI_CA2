from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


class UnpairedImageDataset(Dataset):
    """Load independent image sets from trainA/ and trainB/ directories."""

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str | Path, image_size: int = 256, training: bool = True) -> None:
        self.root = Path(root)
        self.domain_a = self._images(self.root / ("trainA" if training else "testA"))
        self.domain_b = self._images(self.root / ("trainB" if training else "testB"))
        if not self.domain_a or not self.domain_b:
            raise FileNotFoundError(
                f"Expected images in {self.root}/{'trainA and trainB' if training else 'testA and testB'}. "
                "Run `python download_dataset.py` first."
            )
        transform_list = [
            transforms.Resize((image_size + 30, image_size + 30), interpolation=transforms.InterpolationMode.BICUBIC),
        ]
        if training:
            transform_list.extend([
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            transform_list.append(transforms.CenterCrop(image_size))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.transform = transforms.Compose(transform_list)

    @classmethod
    def _images(cls, folder: Path) -> list[Path]:
        if not folder.is_dir():
            return []
        return sorted(path for path in folder.iterdir() if path.suffix.lower() in cls.extensions)

    def __len__(self) -> int:
        return max(len(self.domain_a), len(self.domain_b))

    def __getitem__(self, index: int):
        path_a = self.domain_a[index % len(self.domain_a)]
        path_b = random.choice(self.domain_b)
        with Image.open(path_a) as image_a, Image.open(path_b) as image_b:
            image_a = ImageOps.exif_transpose(image_a).convert("RGB")
            image_b = ImageOps.exif_transpose(image_b).convert("RGB")
            return {"A": self.transform(image_a), "B": self.transform(image_b)}
