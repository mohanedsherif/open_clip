from __future__ import annotations

import random
from pathlib import Path

from .config import load_yaml

CAPTION_TYPES = ("category", "detailed", "search")


def load_caption_templates(path: str | Path) -> dict[str, dict[str, list[str]]]:
    raw = load_yaml(path)
    categories = raw.get("categories", raw)
    if not isinstance(categories, dict):
        raise ValueError("caption template configuration must contain a 'categories' mapping")
    result: dict[str, dict[str, list[str]]] = {}
    for category, styles in categories.items():
        if not isinstance(styles, dict):
            raise ValueError(f"templates for {category!r} must be a mapping")
        result[str(category)] = {}
        for style in CAPTION_TYPES:
            values = styles.get(style, [])
            if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
                raise ValueError(f"{category!r}.{style} must be a list of nonempty strings")
            result[str(category)][style] = values
    return result


def candidates(category: str, templates: dict[str, dict[str, list[str]]]) -> list[tuple[str, str]]:
    styles = templates.get(category, templates.get("_default", {}))
    result = [(caption, kind) for kind in CAPTION_TYPES for caption in styles.get(kind, [])]
    if not result:
        raise ValueError(f"No caption templates configured for category {category!r}")
    return result


def sample_caption(category: str, templates: dict[str, dict[str, list[str]]], seed: int) -> tuple[str, str]:
    return random.Random(seed).choice(candidates(category, templates))
