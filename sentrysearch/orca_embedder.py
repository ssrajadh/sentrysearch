"""OrcaRouter embedding backend.

Embeds video chunks, text queries, and images through OrcaRouter's
gemini-compatible endpoint using the existing google-genai SDK. OrcaRouter is
an OpenAI-compatible AI gateway that exposes models such as
``google/gemini-embedding-2-preview`` behind a single base URL
(``https://api.orcarouter.ai``), so this backend adds no new runtime
dependency — the same ``google-genai`` package that powers the ``gemini``
backend just points at the gateway.

See: https://www.orcarouter.ai
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque

from dotenv import load_dotenv

from .base_embedder import BaseEmbedder

load_dotenv()

DEFAULT_EMBED_MODEL = "google/gemini-embedding-2-preview"
DEFAULT_RERANK_MODEL = "google/gemini-2.5-flash"
DIMENSIONS = 768
DEFAULT_RPM = int(os.environ.get("ORCAROUTER_RPM", "55"))
API_BASE_URL = "https://api.orcarouter.ai"
API_VERSION = "v1beta"


def default_orca_embedding_model() -> str:
    """Return the OrcaRouter embedding model id from env or the built-in default."""
    return os.environ.get("ORCAROUTER_EMBEDDING_MODEL") or DEFAULT_EMBED_MODEL


def default_orca_rerank_model() -> str:
    """Return the OrcaRouter VLM rerank model id from env or the built-in default."""
    return os.environ.get("ORCAROUTER_RERANK_MODEL") or DEFAULT_RERANK_MODEL


class OrcaAPIKeyError(RuntimeError):
    """Raised when ORCAROUTER_API_KEY is missing."""


class OrcaAPIError(RuntimeError):
    """Raised when the OrcaRouter gateway returns an error response."""


_shared_limiter: _RateLimiter | None = None


def get_limiter(rpm: int | None = None) -> _RateLimiter:
    """Return the process-wide OrcaRouter limiter, creating it on first use.

    Embedding and reranking bill against the same gateway quota, so they
    share one window instead of each running an independent budget.

    Precedence: explicit *rpm* > ``ORCAROUTER_RPM`` > :data:`DEFAULT_RPM`.
    """
    global _shared_limiter
    if _shared_limiter is None:
        _shared_limiter = _RateLimiter(
            max_per_minute=rpm if rpm is not None else DEFAULT_RPM
        )
    elif rpm is not None:
        _shared_limiter.set_rate(rpm)
    return _shared_limiter


def reset_limiter() -> None:
    """Drop the shared limiter (tests, and backend switches)."""
    global _shared_limiter
    _shared_limiter = None


class _RateLimiter:
    """Sliding-window rate limiter (requests per minute)."""

    def __init__(self, max_per_minute: int):
        self._max = max_per_minute
        self._timestamps: deque[float] = deque()

    @property
    def max_per_minute(self) -> int:
        return self._max

    def set_rate(self, max_per_minute: int) -> None:
        """Change the ceiling in place, keeping the current window."""
        if max_per_minute < 1:
            raise ValueError("rpm must be >= 1")
        self._max = max_per_minute

    def wait(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] >= 60:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max:
            sleep_for = 60.0 - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


def _is_transient_transport_error(exc: BaseException) -> bool:
    """True for typical HTTP client / TLS / socket failures worth retrying."""
    try:
        import httpx
    except ImportError:
        httpx = None  # type: ignore[assignment]

    if httpx is not None and isinstance(
        exc, (httpx.TransportError, httpx.TimeoutException)
    ):
        return True
    if isinstance(exc, (TimeoutError, BrokenPipeError, ConnectionResetError)):
        return True
    msg = str(exc).lower()
    return any(
        k in msg
        for k in (
            "connection",
            "timeout",
            "timed out",
            "reset by peer",
            "ssl",
            "eof",
            "broken pipe",
            "temporarily unavailable",
        )
    )


def _format_error(exc: Exception) -> str:
    """Extract a readable message from a google-genai API error."""
    message = getattr(exc, "message", None)
    if message:
        return str(message)
    return str(exc)


def _retry(fn, *, max_retries: int = 5, initial_delay: float = 2.0, max_delay: float = 60.0):
    """Call *fn* with exponential back-off on transient gateway errors (429, 503)."""
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except OrcaAPIError:
            raise
        except Exception as exc:  # noqa: BLE001 — narrow follow-up below
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            msg = str(exc).lower()
            retryable = code in (429, 503) or any(
                k in msg
                for k in ("429", "503", "rate", "throttl", "temporarily unavailable")
            ) or _is_transient_transport_error(exc)
            if not retryable or attempt == max_retries:
                raise OrcaAPIError(_format_error(exc)) from exc
            wait = min(delay, max_delay)
            print(
                f"  Retryable OrcaRouter error (attempt {attempt + 1}/{max_retries}), "
                f"waiting {wait:.0f}s: {_format_error(exc)}",
                file=sys.stderr,
            )
            time.sleep(wait)
            delay *= 2


class OrcaEmbedder(BaseEmbedder):
    """OrcaRouter embedding backend (gemini-compatible gateway)."""

    def __init__(self, rpm: int | None = None):
        from google import genai
        from google.genai import types

        api_key = os.environ.get("ORCAROUTER_API_KEY")
        if not api_key:
            raise OrcaAPIKeyError(
                "ORCAROUTER_API_KEY is not set.\n\n"
                "Get a key at https://www.orcarouter.ai\n\n"
                "Then:\n"
                "  export ORCAROUTER_API_KEY=your-key\n\n"
                "Or add it to ~/.sentrysearch/.env\n\n"
                "Or use a local model instead (no API key needed):\n"
                "  sentrysearch index <directory> --backend local"
            )
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                base_url=API_BASE_URL,
                api_version=API_VERSION,
            ),
        )
        self._types = types
        self._model = default_orca_embedding_model()
        self._limiter = _RateLimiter(max_per_minute=rpm or DEFAULT_RPM)

    @staticmethod
    def _make_video_part(chunk_path: str, types) -> object:
        with open(chunk_path, "rb") as f:
            video_bytes = f.read()
        if hasattr(types.Part, "from_bytes"):
            return types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
        return types.Part(inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"))

    @staticmethod
    def _make_image_part(image_path: str, types) -> object:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        if hasattr(types.Part, "from_bytes"):
            return types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        return types.Part(inline_data=types.Blob(data=image_bytes, mime_type="image/jpeg"))

    def embed_video_chunk(self, chunk_path: str, verbose: bool = False) -> list[float]:
        if not os.path.isfile(chunk_path):
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

        video_part = self._make_video_part(chunk_path, self._types)
        return self._embed(
            video_part,
            task_type="RETRIEVAL_DOCUMENT",
            verbose=verbose,
            label="video",
            size_kb=os.path.getsize(chunk_path) / 1024,
        )

    def embed_query(self, query_text: str, verbose: bool = False) -> list[float]:
        return self._embed(
            query_text,
            task_type="RETRIEVAL_QUERY",
            verbose=verbose,
            label="query",
        )

    def embed_image(self, image_path: str, verbose: bool = False) -> list[float]:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp",
            ".gif": "image/gif",
            ".heic": "image/heic", ".heif": "image/heif",
        }
        if ext not in mime_map:
            raise ValueError(
                f"Unsupported image type {ext!r}. "
                f"OrcaRouter accepts: {', '.join(sorted(mime_map))}."
            )
        mime = mime_map[ext]

        with open(image_path, "rb") as f:
            image_bytes = f.read()
        if hasattr(self._types.Part, "from_bytes"):
            part = self._types.Part.from_bytes(data=image_bytes, mime_type=mime)
        else:
            part = self._types.Part(
                inline_data=self._types.Blob(data=image_bytes, mime_type=mime)
            )
        return self._embed(
            part,
            task_type="RETRIEVAL_QUERY",
            verbose=verbose,
            label="image",
            size_kb=len(image_bytes) / 1024,
        )

    def dimensions(self) -> int:
        return DIMENSIONS

    def _embed(
        self,
        content,
        *,
        task_type: str,
        verbose: bool,
        label: str,
        size_kb: float | None = None,
    ) -> list[float]:
        self._limiter.wait()
        t0 = time.monotonic()
        response = _retry(
            lambda: self._client.models.embed_content(
                model=self._model,
                contents=content,
                config=self._types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=DIMENSIONS,
                ),
            )
        )
        embedding = response.embeddings[0].values
        elapsed = time.monotonic() - t0

        if verbose:
            detail = f"dims={len(embedding)}, api_time={elapsed:.2f}s"
            if size_kb is not None:
                detail = f"size={size_kb:.0f}KB, {detail}"
            print(f"  [verbose] {label} embedding: {detail}", file=sys.stderr)

        return embedding
