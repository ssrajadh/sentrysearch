"""Gemini embedding backend using the google-genai SDK.

Embeds video chunks inline via Part.from_bytes — no Files API needed.
"""

import os
import sys
import threading
import time
from collections import deque

from dotenv import load_dotenv

from .base_embedder import BaseEmbedder

load_dotenv()

EMBED_MODEL = "gemini-embedding-2"
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
        self._lock = threading.Lock()

    @property
    def max_per_minute(self) -> int:
        return self._max

    def set_rate(self, max_per_minute: int) -> None:
        """Change the ceiling in place, keeping the current window."""
        if max_per_minute < 1:
            raise ValueError("rpm must be >= 1")
        with self._lock:
            self._max = max_per_minute

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._timestamps and now - self._timestamps[0] >= 60:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._max:
                sleep_for = 60.0 - (now - self._timestamps[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
            self._timestamps.append(time.monotonic())


def _env_rpm() -> int | None:
    """Read GEMINI_RPM, ignoring unset, malformed, or non-positive values."""
    raw = os.environ.get("GEMINI_RPM", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        print(
            f"  Ignoring non-integer GEMINI_RPM={raw!r}; using {DEFAULT_RPM}.",
            file=sys.stderr,
        )
        return None
    if value < 1:
        print(
            f"  Ignoring non-positive GEMINI_RPM={value}; using {DEFAULT_RPM}.",
            file=sys.stderr,
        )
        return None
    return value


_shared_limiter: _RateLimiter | None = None


def get_limiter(rpm: int | None = None) -> _RateLimiter:
    """Return the process-wide Gemini limiter, creating it on first use.

    Embedding and reranking bill against the same per-project quota, so they
    share one window instead of each running an independent budget.

    Precedence: explicit *rpm* > ``GEMINI_RPM`` > :data:`DEFAULT_RPM`.
    """
    global _shared_limiter
    if _shared_limiter is None:
        _shared_limiter = _RateLimiter(
            max_per_minute=rpm if rpm is not None else (_env_rpm() or DEFAULT_RPM)
        )
    elif rpm is not None:
        _shared_limiter.set_rate(rpm)
    return _shared_limiter


def reset_limiter() -> None:
    """Drop the shared limiter (tests, and backend switches)."""
    global _shared_limiter
    _shared_limiter = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class GeminiAPIKeyError(RuntimeError):
    """Raised when GEMINI_API_KEY is missing."""


class GeminiQuotaError(RuntimeError):
    """Raised when API quota is exceeded."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quota_message(msg: str) -> str:
    """Build quota-error guidance, distinguishing per-day from per-minute.

    A per-minute ceiling is something --rpm can pace under; a daily cap is not,
    so the two failures need different advice.
    """
    per_day = "perday" in msg.replace("_", "").replace(" ", "") or "per day" in msg
    if per_day:
        return (
            "Gemini API daily quota exhausted.\n\n"
            "This is a per-day cap, so throttling with --rpm will not help.\n"
            "The quota resets at midnight Pacific. Options:\n"
            "  - Resume tomorrow (indexing is incremental; finished chunks are kept)\n"
            "  - Index a smaller directory at a time\n"
            "  - Use the local backend: --backend local\n"
            "  - Upgrade your API plan at https://aistudio.google.com"
        )
    return (
        "Gemini API rate limit exceeded (requests per minute).\n\n"
        "Free-tier limits vary by model and project; check yours in AI Studio.\n"
        "Options:\n"
        f"  - Re-run with a lower rate, e.g. --rpm 10 (current: {get_limiter().max_per_minute})\n"
        "  - Or set it once: export GEMINI_RPM=10\n"
        "  - Use the local backend: --backend local\n"
        "  - Upgrade your API plan at https://aistudio.google.com"
    )


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
                if "resource exhausted" in msg or status == 429:
                    raise GeminiQuotaError(_quota_message(msg)) from exc
                raise
            wait = min(delay, max_delay)
            print(
                f"  Retryable error (attempt {attempt + 1}/{max_retries}), "
                f"waiting {wait:.0f}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(wait)
            delay *= 2


# ---------------------------------------------------------------------------
# GeminiEmbedder
# ---------------------------------------------------------------------------

class GeminiEmbedder(BaseEmbedder):
    """Gemini Embedding 2 backend (API-based)."""

    def __init__(self, rpm: int | None = None):
        from google import genai
        from google.genai import types  # noqa: F811

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise GeminiAPIKeyError(
                "GEMINI_API_KEY is not set.\n\n"
                "Run: sentrysearch init\n\n"
                "Or set it manually:\n"
                "  export GEMINI_API_KEY=your-key\n\n"
                "Or use a local model instead (no API key needed):\n"
                "  sentrysearch index <directory> --backend local"
            )
        self._client = genai.Client(api_key=api_key)
        self._limiter = get_limiter(rpm)

    def embed_video_chunk(self, chunk_path: str, verbose: bool = False) -> list[float]:
        from google.genai import types

        video_part = self._make_video_part(chunk_path, types)

        if verbose:
            size_kb = os.path.getsize(chunk_path) / 1024
            print(
                f"    [verbose] sending {size_kb:.0f}KB to {EMBED_MODEL}",
                file=sys.stderr,
            )

        self._limiter.wait()
        t0 = time.monotonic()
        response = _retry(
            lambda: self._client.models.embed_content(
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

    def embed_query(self, query_text: str, verbose: bool = False) -> list[float]:
        from google.genai import types

        self._limiter.wait()
        t0 = time.monotonic()
        response = _retry(
            lambda: self._client.models.embed_content(
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

    def embed_image(self, image_path: str, verbose: bool = False) -> list[float]:
        from google.genai import types

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
                f"Gemini accepts: {', '.join(sorted(mime_map))}."
            )
        mime = mime_map[ext]

        with open(image_path, "rb") as f:
            img_bytes = f.read()
        if hasattr(types.Part, "from_bytes"):
            part = types.Part.from_bytes(data=img_bytes, mime_type=mime)
        else:
            part = types.Part(inline_data=types.Blob(data=img_bytes, mime_type=mime))

        self._limiter.wait()
        t0 = time.monotonic()
        response = _retry(
            lambda: self._client.models.embed_content(
                model=EMBED_MODEL,
                contents=types.Content(parts=[part]),
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=DIMENSIONS,
                ),
            )
        )
        elapsed = time.monotonic() - t0
        embedding = response.embeddings[0].values

        if verbose:
            size_kb = len(img_bytes) / 1024
            print(
                f"  [verbose] image embedding: dims={len(embedding)}, "
                f"size={size_kb:.0f}KB, api_time={elapsed:.2f}s",
                file=sys.stderr,
            )

        return embedding

    def dimensions(self) -> int:
        return DIMENSIONS

    @staticmethod
    def _make_video_part(chunk_path: str, types):
        with open(chunk_path, "rb") as f:
            video_bytes = f.read()
        if hasattr(types.Part, "from_bytes"):
            return types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
        return types.Part(inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"))
