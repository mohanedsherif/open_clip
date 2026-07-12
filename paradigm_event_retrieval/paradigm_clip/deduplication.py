from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: str | Path, hash_size: int = 8) -> int:
    """Small dependency-free average hash; distances <= threshold are near duplicates."""
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS).getdata())
    mean = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= mean else "0" for pixel in pixels)
    return int(bits, 2)


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()
