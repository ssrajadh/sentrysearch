"""Gemini embedding client using the google-genai SDK.

Embeds video chunks inline via Part.from_bytes — no Files API needed.
"""

import os
import sys
import time
from collections import deque

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

EMBED_MODEL = "gemini-embedding-2-preview"
DIMENSIONS = 768
DEFAULT_RPM = 55

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple sliding-window rate limiter based on request timestamps."""

    def __init__(self, max_per_minute: int = DEFAULT_RPM):
        self._max = max_per_minute
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] >= 60:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max:
            sleep_for = 60.0 - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


_limiter = _RateLimiter()

# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

_client: genai.Client | None = None


class GeminiAPIKeyError(RuntimeError):
    """Raised when GEMINI_API_KEY is missing."""


class GeminiQuotaError(RuntimeError):
    """Raised when API quota is exceeded."""


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise GeminiAPIKeyError(
                "GEMINI_API_KEY is not set.\n\n"
                "Get a free key at https://aistudio.google.com/apikey\n"
                "Then either:\n"
                "  1. Add it to .env:  echo 'GEMINI_API_KEY=your-key' > .env\n"
                "  2. Export it:       export GEMINI_API_KEY=your-key"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _retry(fn, *, max_retries: int = 5, initial_delay: float = 2.0, max_delay: float = 60.0):
    """Call *fn* with exponential back-off on transient errors (429, 503)."""
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc).lower()
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            retryable = status in (429, 503)
            if not retryable:
                retryable = "resource exhausted" in msg or "503" in msg or "429" in msg
            if not retryable or attempt == max_retries:
                # Wrap quota errors with a helpful message
                if "resource exhausted" in msg or status == 429:
                    raise GeminiQuotaError(
                        "Gemini API rate limit exceeded.\n\n"
                        "The free tier allows 60 requests/minute. Options:\n"
                        "  - Wait a minute and retry\n"
                        "  - Use a smaller --chunk-duration to create fewer chunks\n"
                        "  - Upgrade your API plan at https://aistudio.google.com"
                    ) from exc
                raise
            wait = min(delay, max_delay)
            print(
                f"  Retryable error (attempt {attempt + 1}/{max_retries}), "
                f"waiting {wait:.0f}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(wait)
            delay *= 2


def _make_video_part(chunk_path: str) -> types.Part:
    """Read video bytes and build an inline Part."""
    with open(chunk_path, "rb") as f:
        video_bytes = f.read()

    if hasattr(types.Part, "from_bytes"):
        return types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
    return types.Part(inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"))

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_video_chunk(chunk_path: str, verbose: bool = False) -> list[float]:
    """Embed a video chunk inline (no file upload). Returns 768-dim vector."""
    client = _get_client()
    video_part = _make_video_part(chunk_path)

    if verbose:
        size_kb = os.path.getsize(chunk_path) / 1024
        print(
            f"    [verbose] sending {size_kb:.0f}KB to {EMBED_MODEL}",
            file=sys.stderr,
        )

    _limiter.wait()
    t0 = time.monotonic()
    response = _retry(
        lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=types.Content(parts=[video_part]),
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=DIMENSIONS,
            ),
        )
    )
    elapsed = time.monotonic() - t0
    embedding = response.embeddings[0].values

    if verbose:
        size_kb = os.path.getsize(chunk_path) / 1024
        print(
            f"    [verbose] dims={len(embedding)}, "
            f"chunk_size={size_kb:.0f}KB, "
            f"api_time={elapsed:.2f}s",
            file=sys.stderr,
        )

    return embedding


def embed_query(query_text: str, verbose: bool = False) -> list[float]:
    """Embed a text query for retrieval."""
    client = _get_client()
    _limiter.wait()
    t0 = time.monotonic()
    response = _retry(
        lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=query_text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=DIMENSIONS,
            ),
        )
    )
    elapsed = time.monotonic() - t0
    embedding = response.embeddings[0].values

    if verbose:
        print(
            f"  [verbose] query embedding: dims={len(embedding)}, "
            f"api_time={elapsed:.2f}s",
            file=sys.stderr,
        )

    return embedding


def embed_text(text: str, verbose: bool = False) -> list[float]:
    """Embed transcript text as a document for retrieval.

    Uses RETRIEVAL_DOCUMENT task type (vs RETRIEVAL_QUERY for search queries).
    Returns empty list if text is empty.
    """
    if not text.strip():
        return []

    client = _get_client()
    _limiter.wait()
    t0 = time.monotonic()
    response = _retry(
        lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=DIMENSIONS,
            ),
        )
    )
    elapsed = time.monotonic() - t0
    embedding = response.embeddings[0].values

    if verbose:
        print(
            f"    [verbose] text embedding: dims={len(embedding)}, "
            f"text_len={len(text)}, api_time={elapsed:.2f}s",
            file=sys.stderr,
        )

    return embedding
