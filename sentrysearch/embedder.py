"""Embedder factory — selects and caches the active backend.

Provides backward-compatible top-level functions (embed_video_chunk,
embed_query) that delegate to whichever backend is currently active.
Re-exports error classes from gemini_embedder for existing import sites.
"""

from .base_embedder import BaseEmbedder
from .gemini_embedder import GeminiAPIKeyError, GeminiQuotaError  # noqa: F401

_current_embedder: BaseEmbedder | None = None


def get_embedder(backend: str = "gemini", **kwargs) -> BaseEmbedder:
    """Factory to get or create the active embedder."""
    global _current_embedder
    if _current_embedder is None:
        if backend == "gemini":
            from .gemini_embedder import GeminiEmbedder
            _current_embedder = GeminiEmbedder()
        elif backend == "local":
            from .local_embedder import LocalEmbedder
            model = kwargs.get("model", "qwen8b")
            dims = kwargs.get("dimensions", 768)
            quantize = kwargs.get("quantize", None)
            _current_embedder = LocalEmbedder(model_name=model, dimensions=dims, quantize=quantize)
        elif backend == "minimax":
            from .minimax_embedder import MiniMaxEmbedder
            _current_embedder = MiniMaxEmbedder()
        else:
            raise ValueError(f"Unknown backend: {backend}")
    return _current_embedder


def reset_embedder():
    """Reset the cached embedder (for switching backends)."""
    global _current_embedder
    _current_embedder = None


# Convenience functions — backward compatible with existing callers
def embed_video_chunk(chunk_path: str, verbose: bool = False) -> list[float]:
    return get_embedder().embed_video_chunk(chunk_path, verbose=verbose)


def embed_query(query_text: str, verbose: bool = False) -> list[float]:
    return get_embedder().embed_query(query_text, verbose=verbose)
