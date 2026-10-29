from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip"


def download_dataset(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "horse2zebra.zip"
    destination = output_dir / "horse2zebra"
    if not archive.exists():
        print("Downloading Horse2Zebra training dataset (about 111 MB)...")
        urllib.request.urlretrieve(DATASET_URL, archive)
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"Invalid dataset archive: {archive}")
    with zipfile.ZipFile(archive) as zipped:
        base = destination.resolve()
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError("Dataset archive contains an unsafe path")
        zipped.extractall(destination)
    nested = destination / "horse2zebra"
    if nested.is_dir():
        for child in nested.iterdir():
            target = destination / child.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            child.replace(target)
        nested.rmdir()
    required = [destination / name for name in ("trainA", "trainB", "testA", "testB")]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise RuntimeError(f"Unexpected archive layout; missing: {', '.join(missing)}")
    print(f"Dataset ready at {destination.resolve()}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the unpaired Horse2Zebra dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    download_dataset(parser.parse_args().output_dir)
