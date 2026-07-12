from paradigm_event_retrieval.paradigm_clip.metrics import retrieval_metrics


def test_retrieval_ranking_metrics() -> None:
    result = retrieval_metrics([["a", "b", "c"], ["d", "e", "f"]], [{"a"}, {"f"}])
    assert result["Precision@1"] == 0.5
    assert result["Recall@5"] == 1.0
    assert result["MRR"] == (1.0 + 1 / 3) / 2
