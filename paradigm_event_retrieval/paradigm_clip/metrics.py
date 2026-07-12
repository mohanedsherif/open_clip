from __future__ import annotations

import math
from collections.abc import Sequence


def retrieval_metrics(rankings: Sequence[Sequence[str]], relevant: Sequence[set[str]], ks: tuple[int, ...] = (1, 5, 10, 50)) -> dict[str, float]:
    if len(rankings) != len(relevant) or not rankings:
        raise ValueError("rankings and relevant must be non-empty and have equal length")
    result: dict[str, float] = {}
    for k in ks:
        precision = [sum(item in rel for item in ranking[:k]) / k for ranking, rel in zip(rankings, relevant)]
        recall = [sum(item in rel for item in ranking[:k]) / len(rel) if rel else 0.0 for ranking, rel in zip(rankings, relevant)]
        result[f"Precision@{k}"] = sum(precision) / len(precision)
        result[f"Recall@{k}"] = sum(recall) / len(recall)
    reciprocal = []
    ndcg = []
    for ranking, rel in zip(rankings, relevant):
        first = next((index + 1 for index, item in enumerate(ranking) if item in rel), None)
        reciprocal.append(0.0 if first is None else 1.0 / first)
        dcg = sum(1.0 / math.log2(index + 2) for index, item in enumerate(ranking[:10]) if item in rel)
        ideal = sum(1.0 / math.log2(index + 2) for index in range(min(10, len(rel))))
        ndcg.append(dcg / ideal if ideal else 0.0)
    result["MRR"] = sum(reciprocal) / len(reciprocal)
    result["nDCG@10"] = sum(ndcg) / len(ndcg)
    return result
