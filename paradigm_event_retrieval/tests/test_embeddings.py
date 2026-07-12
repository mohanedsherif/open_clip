import importlib.util
import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires installed torch/OpenCLIP environment")
def test_embedding_api_imports() -> None:
    from paradigm_event_retrieval.paradigm_clip.inference import ParadigmEmbeddingModel
    assert hasattr(ParadigmEmbeddingModel, "embed_images")
