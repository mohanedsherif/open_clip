from __future__ import annotations

import csv
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from .caption_templates import sample_caption
from .deduplication import file_sha256, hamming_distance, perceptual_hash

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
MANIFEST_COLUMNS = ("filepath", "event_id", "category", "caption", "caption_type", "split")


@dataclass(frozen=True)
class ImageRecord:
    filepath: str
    event_id: str
    category: str


def scan_dataset(raw_dir: str | Path) -> tuple[list[ImageRecord], list[dict[str, str]]]:
    raw = Path(raw_dir)
    valid, corrupt = [], []
    if not raw.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw}")
    for path in sorted(p for p in raw.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        relative = path.relative_to(raw)
        if len(relative.parts) < 3:
            corrupt.append({"filepath": relative.as_posix(), "reason": "expected event/category/image layout"})
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            corrupt.append({"filepath": relative.as_posix(), "reason": str(exc)})
            continue
        valid.append(ImageRecord(relative.as_posix(), relative.parts[0], relative.parts[1]))
    return valid, corrupt


def find_duplicates(raw_dir: str | Path, records: Iterable[ImageRecord], near_threshold: int = 6) -> dict[str, list[dict[str, object]]]:
    raw = Path(raw_dir)
    hashes: dict[str, list[str]] = defaultdict(list)
    perceptual: list[tuple[str, int]] = []
    for record in records:
        path = raw / record.filepath
        hashes[file_sha256(path)].append(record.filepath)
        perceptual.append((record.filepath, perceptual_hash(path)))
    exact = [{"sha256": key, "files": value} for key, value in hashes.items() if len(value) > 1]
    near = []
    for i, (left_path, left_hash) in enumerate(perceptual):
        for right_path, right_hash in perceptual[i + 1:]:
            distance = hamming_distance(left_hash, right_hash)
            if distance <= near_threshold:
                near.append({"left": left_path, "right": right_path, "distance": distance})
    return {"exact_duplicates": exact, "near_duplicates": near}


def split_events(records: Iterable[ImageRecord], seed: int = 42, ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> dict[str, str]:
    if len(ratios) != 3 or any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must be three non-negative values summing to 1")
    events = sorted({record.event_id for record in records})
    if len(events) < 3:
        raise ValueError("At least three events are required for train/validation/test event splits")
    random.Random(seed).shuffle(events)
    count = len(events)
    train_count = max(1, round(count * ratios[0]))
    validation_count = max(1, round(count * ratios[1]))
    if train_count + validation_count >= count:
        train_count, validation_count = count - 2, 1
    mapping = {event: "train" for event in events[:train_count]}
    mapping.update({event: "validation" for event in events[train_count:train_count + validation_count]})
    mapping.update({event: "test" for event in events[train_count + validation_count:]})
    return mapping


def write_manifests(records: Iterable[ImageRecord], event_splits: dict[str, str], templates: dict, output_dir: str | Path, seed: int = 42) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, record in enumerate(sorted(records, key=lambda item: item.filepath)):
        caption, caption_type = sample_caption(record.category, templates, seed + index)
        split = event_splits[record.event_id]
        grouped[split].append(dict(zip(MANIFEST_COLUMNS, (record.filepath, record.event_id, record.category, caption, caption_type, split))))
    paths = {}
    for split in ("train", "validation", "test"):
        path = output / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(grouped[split])
        paths[split] = path
    return paths


def dataset_report(records: list[ImageRecord], corrupt: list[dict[str, str]], duplicates: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    per_event = Counter(record.event_id for record in records)
    per_category = Counter(record.category for record in records)
    values = list(per_category.values())
    return {"total_images": len(records) + len(corrupt), "valid_images": len(records), "corrupted_images": corrupt,
            "duplicates": duplicates["exact_duplicates"], "near_duplicates": duplicates["near_duplicates"],
            "images_per_event": dict(sorted(per_event.items())), "images_per_category": dict(sorted(per_category.items())),
            "class_imbalance": {"largest_to_smallest_ratio": max(values) / min(values) if values else None, "category_count": len(values)}}


def save_json(value: object, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
