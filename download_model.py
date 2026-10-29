from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve

from utils.inference import MODEL_SOURCES, load_generator

TASK_IDS = {"horse2zebra": "Horse → Zebra", "zebra2horse": "Zebra → Horse"}


def _download_range(url: str, start: int, end: int, part: Path) -> int:
    request = Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=180) as response:
        data = response.read()
        if response.status != 206 or len(data) != end - start + 1:
            raise RuntimeError(f"Server returned an invalid byte range ({response.status})")
    part.write_bytes(data)
    return len(data)


def _download_checkpoint(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        request = Request(url, method="HEAD")
        with urlopen(request, timeout=30) as response:
            total = int(response.headers.get("Content-Length", "0"))
            supports_ranges = "bytes" in response.headers.get("Accept-Ranges", "").lower()
        if not supports_ranges or total < 10_000_000:
            urlretrieve(url, temporary)
        else:
            workers = 8
            chunk_size = math.ceil(total / workers)
            ranges = [(start, min(total - 1, start + chunk_size - 1))
                      for start in range(0, total, chunk_size)]
            with tempfile.TemporaryDirectory(prefix="imagelab-model-") as temp_dir:
                temp_path = Path(temp_dir)
                parts = [temp_path / f"part-{index:02d}.bin" for index in range(len(ranges))]
                with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
                    futures = [pool.submit(_download_range, url, start, end, parts[index])
                               for index, (start, end) in enumerate(ranges)]
                    received = 0
                    for future in as_completed(futures):
                        received += future.result()
                        print(f"Downloaded {received:,} / {total:,} bytes")
                with temporary.open("wb") as output:
                    for part in parts:
                        with part.open("rb") as stream:
                            shutil.copyfileobj(stream, output)
            if temporary.stat().st_size != total:
                raise RuntimeError("Downloaded checkpoint length does not match the server response")
    except Exception:
        temporary.unlink(missing_ok=True)
        # Fall back to a normal download on servers that do not support byte ranges.
        urlretrieve(url, temporary)
    if temporary.stat().st_size < 10_000_000:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Downloaded file is too small to be a valid model checkpoint")
    temporary.replace(destination)


def download(task: str, output_dir: Path) -> Path:
    details = MODEL_SOURCES[task]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / details["file"]
    if destination.exists() and destination.stat().st_size > 10_000_000:
        print(f"Checkpoint already present: {destination}")
    else:
        direction = details["direction"].replace("→", "to")
        print(f"Downloading official Berkeley {direction} checkpoint (~43 MB)...")
        _download_checkpoint(details["url"], destination)
    # A strict load validates architecture and state-dict compatibility before reporting success.
    load_generator(destination)
    with destination.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(f"Verified checkpoint: {destination}")
    print(f"SHA-256: {digest}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and verify official CycleGAN model weights")
    parser.add_argument("--task", choices=list(TASK_IDS), default="horse2zebra")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    download(TASK_IDS[args.task], args.output_dir)
