"""Tools for adapting OpenCLIP to Paradigm event-photo retrieval."""

# Keep lightweight utilities (for example retrieval metrics) usable before the
# optional inference dependencies are installed.
__all__ = ["ParadigmEmbeddingModel"]


def __getattr__(name: str):
    if name == "ParadigmEmbeddingModel":
        from .inference import ParadigmEmbeddingModel
        return ParadigmEmbeddingModel
    raise AttributeError(name)
