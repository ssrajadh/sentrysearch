"""Embedder factory — selects and caches the active backend.

Provides backward-compatible top-level functions (embed_video_chunk,
embed_query) that delegate to whichever backend is currently active.
Re-exports error classes from gemini_embedder for existing import sites.

Also home to the backend/model selection and index-time retry policy shared
by every entry point that embeds chunks (`sentrysearch index` and
``benchmarks/accuracy_eval.py``), so both resolve the same flags to the same
backend/model and apply the same backoff.
"""

import time

from .base_embedder import BaseEmbedder
from .gemini_embedder import GeminiAPIKeyError, GeminiQuotaError  # noqa: F401
from .qwen_cloud_embedder import (  # noqa: F401
    DashScopeAPIKeyError,
    DashScopeDependencyError,
    default_dashscope_embedding_model,
)

_current_embedder: BaseEmbedder | None = None


def resolve_backend(
    backend: str | None = None,
    model: str | None = None,
    dashscope_model: str | None = None,
) -> str:
    """Which backend will run, given the user's flags.

    Inference only — no model loading, API call, or filesystem access, so
    callers can run it before a dry-run gate. An explicit ``backend`` wins;
    otherwise ``model`` implies local and ``dashscope_model`` implies
    qwen-cloud (the same precedence as `sentrysearch index`), else gemini.
    """
    if backend is not None:
        return backend
    if model is not None:
        return "local"
    if dashscope_model is not None:
        return "qwen-cloud"
    return "gemini"


def resolve_model(
    backend: str,
    model: str | None = None,
    dashscope_model: str | None = None,
    *,
    echo=None,
) -> str | None:
    """Resolve the model reference for *backend*, or None for gemini.

    Mirrors `sentrysearch index`: qwen-cloud takes ``--dashscope-model``,
    else the provider default -- the generic ``--model`` flag never
    selects a DashScope model, so every entry point reads that flag the
    same way. mlx resolves its full model reference now so it lands in
    the collection metadata, litellm and local fall back to their
    defaults, and local auto-detects from the hardware — reporting that
    choice through *echo* when given.
    """
    if backend == "qwen-cloud":
        return dashscope_model or default_dashscope_embedding_model()
    if backend == "mlx":
        from .mlx_embedder import resolve_model_ref

        return resolve_model_ref(model)
    if backend == "litellm":
        from .litellm_embedder import default_litellm_model

        return model or default_litellm_model()
    if backend == "local":
        from .local_embedder import detect_default_model, normalize_model_key

        if model is None:
            model = detect_default_model()
            if echo is not None:
                echo(f"Auto-detected model: {model}")
        return normalize_model_key(model)
    return None


def is_permanent_failure(exc: Exception) -> bool:
    """Return True for errors that won't resolve by retrying the same chunk."""
    msg = str(exc).lower()
    if isinstance(exc, FileNotFoundError):
        return True
    # OOM — same chunk at same settings will OOM again
    if "out of memory" in msg or "cuda out of memory" in msg:
        return True
    # Decoder failures on specific files
    if "invalid data" in msg or "could not decode" in msg:
        return True
    return False


def embed_with_retry(
    embedder,
    embed_path: str,
    *,
    max_attempts: int = 3,
    verbose: bool = False,
    on_retry=None,
    on_failure=None,
) -> list[float] | None:
    """Embed *embed_path*, retrying transient failures with backoff.

    Quota and configuration errors raise immediately — waiting won't clear
    them, and the user needs to act. Any other error is retried up to
    *max_attempts* times, sleeping ``2 ** attempt`` seconds between tries;
    *on_retry(attempt, exc, wait)* is called before each wait so the caller
    can report progress.

    Errors that are permanent (OOM, an unreadable file) or that survive the
    last attempt are handed to *on_failure(attempt, exc)* if the caller
    supplied one — the CLI records them to the DLQ and carries on — and
    otherwise raised, so a caller that must not silently lose a chunk (the
    accuracy evaluation) fails loud.
    """
    if max_attempts < 1:
        raise ValueError(
            f"max_attempts must be >= 1 (got {max_attempts}); "
            "a run with no attempts could neither succeed nor report a failure"
        )

    from .gemini_embedder import GeminiAPIKeyError, GeminiQuotaError
    from .litellm_embedder import LiteLLMConfigError, LiteLLMQuotaError

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return embedder.embed_video_chunk(embed_path, verbose=verbose)
        except (GeminiQuotaError, GeminiAPIKeyError,
                LiteLLMConfigError, LiteLLMQuotaError):
            raise  # user-facing, stop the run
        except Exception as exc:
            last_exc = exc
            if not is_permanent_failure(exc) and attempt < max_attempts:
                wait = 2 ** attempt
                if on_retry is not None:
                    on_retry(attempt, exc, wait)
                time.sleep(wait)
                continue
            if on_failure is not None:
                on_failure(attempt, exc)
                return None
            raise
    # Unreachable while max_attempts >= 1: every attempt returns or raises.
    raise last_exc


def get_embedder(backend: str = "gemini", **kwargs) -> BaseEmbedder:
    """Factory to get or create the active embedder."""
    global _current_embedder
    if _current_embedder is None:
        if backend == "gemini":
            from .gemini_embedder import GeminiEmbedder
            _current_embedder = GeminiEmbedder(rpm=kwargs.get("rpm"))
        elif backend == "local":
            from .local_embedder import LocalEmbedder
            model = kwargs.get("model", "qwen8b")
            dims = kwargs.get("dimensions", 768)
            quantize = kwargs.get("quantize", None)
            _current_embedder = LocalEmbedder(model_name=model, dimensions=dims, quantize=quantize)
        elif backend == "mlx":
            from .mlx_embedder import MLXEmbedder
            _current_embedder = MLXEmbedder(
                model_name=kwargs.get("model"),
                dimensions=kwargs.get("dimensions", 768),
            )
        elif backend == "qwen-cloud":
            from .qwen_cloud_embedder import QwenCloudEmbedder
            qc_model = kwargs.get("model")
            dims = kwargs.get("dimensions", 768)
            vfps = kwargs.get("video_fps")
            qkw: dict = {"model_name": qc_model, "dimensions": dims}
            if vfps is not None:
                qkw["video_fps"] = vfps
            if kwargs.get("rpm") is not None:
                qkw["rpm"] = kwargs["rpm"]
            _current_embedder = QwenCloudEmbedder(**qkw)
        elif backend == "litellm":
            from .litellm_embedder import LiteLLMEmbedder
            _current_embedder = LiteLLMEmbedder(
                model_name=kwargs.get("model"),
                dimensions=kwargs.get("dimensions"),
                rpm=kwargs.get("rpm"),
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")
    return _current_embedder


def reset_embedder():
    """Reset the cached embedder (for switching backends)."""
    global _current_embedder
    _current_embedder = None
    from .gemini_embedder import reset_limiter
    reset_limiter()


# Convenience functions — backward compatible with existing callers
def embed_video_chunk(chunk_path: str, verbose: bool = False) -> list[float]:
    return get_embedder().embed_video_chunk(chunk_path, verbose=verbose)


def embed_query(query_text: str, verbose: bool = False) -> list[float]:
    return get_embedder().embed_query(query_text, verbose=verbose)


def embed_image(image_path: str, verbose: bool = False) -> list[float]:
    return get_embedder().embed_image(image_path, verbose=verbose)
