"""Manual-review hard-negative candidate format; not injected into the standard CLIP loss."""
from __future__ import annotations
from collections.abc import Iterable


def candidates_from_rankings(query_id: str, ranked_paths: Iterable[str], relevant_paths: set[str], limit: int = 20) -> list[dict[str, object]]:
    return [{"query_id": query_id, "image": path, "review_status": "pending"} for path in ranked_paths if path not in relevant_paths][:limit]
