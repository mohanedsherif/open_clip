import importlib.util
import pytest


@pytest.mark.skipif(importlib.util.find_spec("open_clip") is None, reason="requires installed OpenCLIP environment")
def test_model_listing_is_available() -> None:
    from paradigm_event_retrieval.paradigm_clip.model_factory import list_pretrained_models
    assert list_pretrained_models()
