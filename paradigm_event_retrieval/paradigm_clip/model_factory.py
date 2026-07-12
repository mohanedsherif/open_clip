from __future__ import annotations

from typing import Any


def require_open_clip() -> Any:
    try:
        import open_clip
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCLIP is not installed. Run pip install -e '.[training]' from the repository root.") from exc
    return open_clip


def list_pretrained_models() -> list[tuple[str, str]]:
    return list(require_open_clip().list_pretrained())
