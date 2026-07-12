from __future__ import annotations

import json
from pathlib import Path

from .inference import ParadigmEmbeddingModel
from .metrics import retrieval_metrics


def evaluate_queries(model: ParadigmEmbeddingModel, image_paths: list[str], queries_path: str | Path, batch_size: int = 32) -> dict[str, float]:
    queries = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    if not isinstance(queries, list): raise ValueError("queries JSON must be a list")
    image_embeddings = model.embed_images(image_paths, batch_size)
    rankings, relevant = [], []
    for item in queries:
        if not isinstance(item, dict) or "query" not in item or "relevant_images" not in item: raise ValueError("each query needs query and relevant_images")
        result = model.retrieve(item["query"], image_embeddings, top_k=len(image_paths))
        rankings.append([image_paths[int(row["index"])] for row in result])
        relevant.append(set(item["relevant_images"]))
    return retrieval_metrics(rankings, relevant)
