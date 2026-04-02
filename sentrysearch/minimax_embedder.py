"""MiniMax embedding backend (Open Platform api.minimaxi.com).

Video chunks: extract a middle frame, caption it via the text chat (vision) API,
then embed the caption with ``embo-01``. Text queries use ``embo-01`` directly.

MiniMax does not expose a native video-to-vector embedding like Gemini; Hailuo
models are for video *generation*, not embeddings.
"""

import base64
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from collections import deque

from dotenv import load_dotenv

from .base_embedder import BaseEmbedder
from .chunker import _get_ffmpeg_executable, _get_video_duration

load_dotenv()

# Defaults match MiniMax Open Platform + LangChain (embo-01 = 1536-dim text vectors).
EMBED_MODEL = os.environ.get("MINIMAX_EMBED_MODEL", "embo-01")
DIMENSIONS = int(os.environ.get("MINIMAX_EMBED_DIMENSIONS", "1536"))
DEFAULT_RPM = 60

# Same host as video_generation and text APIs (see platform docs).
MINIMAX_API_BASE = os.environ.get("MINIMAX_API_BASE", "https://api.minimaxi.com")

MINIMAX_API_KEY_ENV = "MINIMAX_API_KEY"
MINIMAX_GROUP_ID_ENV = "MINIMAX_GROUP_ID"

# Vision model for frame captioning (must be a *text chat* model with image support;
# Hailuo / video_generation models are invalid on /v1/text/chatcompletion_v2).
MINIMAX_VISION_MODEL_ENV = "MINIMAX_VISION_MODEL"
VISION_MODEL_DEFAULT = "MiniMax-Text-01"


def _fallback_caption_policy_chunk(chunk_path: str) -> str:
    """Neutral text when vision API rejects the frame (e.g. content policy 1026)."""
    digest = hashlib.sha256(os.path.abspath(chunk_path).encode()).hexdigest()[:16]
    return (
        "Video clip segment without visual description (content policy). "
        f"segment_id={digest}"
    )


def _resolve_vision_chat_model(raw: str | None) -> tuple[str, str | None]:
    """Return (model_id, stderr_warning_or_none). Video-gen (Hailuo) ids are rejected."""
    default = VISION_MODEL_DEFAULT
    s = (raw or "").strip()
    if not s:
        return default, None
    lowered = s.lower().replace("_", "-")
    if "hailuo" in lowered:
        return default, (
            f"Ignoring MINIMAX_VISION_MODEL={raw!r}: Hailuo models are for "
            f"video_generation, not chat. Using {default} for frame captioning."
        )
    return s, None


class MiniMaxAPIKeyError(RuntimeError):
    """Raised when MINIMAX_API_KEY is missing."""


class MiniMaxQuotaError(RuntimeError):
    """Raised when API quota is exceeded."""


class MiniMaxSensitiveInputError(RuntimeError):
    """Raised when chat/image input is rejected by content policy (e.g. 1026)."""


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


def _retry(fn, *, max_retries: int = 5, initial_delay: float = 2.0, max_delay: float = 60.0):
    """Call *fn* with exponential back-off on transient errors."""
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc).lower()
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            retryable = status in (429, 500, 502, 503)
            if not retryable:
                retryable = "rate limit" in msg or "429" in msg or "500" in msg or "502" in msg or "503" in msg
            if not retryable or attempt == max_retries:
                if "rate limit" in msg or status == 429:
                    raise MiniMaxQuotaError(
                        "MiniMax API rate limit exceeded.\n\n"
                        "Options:\n"
                        "  - Wait a minute and retry\n"
                        "  - Use a smaller --chunk-duration to create fewer chunks\n"
                        "  - Check your API quota at MiniMax dashboard"
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


def _embedding_from_response(parsed: dict) -> list[float]:
    """Extract an embedding vector from MiniMax native or OpenAI-style JSON."""
    base = parsed.get("base_resp")
    if isinstance(base, dict):
        code = base.get("status_code")
        if code not in (None, 0):
            msg = base.get("status_msg", "unknown error")
            raise RuntimeError(
                f"MiniMax API error: status_code={code} status_msg={msg}"
            )

    if "vectors" in parsed:
        vecs = parsed["vectors"]
        if not vecs:
            raise RuntimeError("MiniMax API returned an empty vectors list")
        return vecs[0]

    data = parsed.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and "embedding" in first:
            return first["embedding"]

    emb = parsed.get("embedding")
    if isinstance(emb, list):
        return emb

    raise RuntimeError(
        "Could not parse embedding from MiniMax response "
        f"(top-level keys: {list(parsed.keys())})."
    )


def _assistant_text_from_chat_response(parsed: dict) -> str:
    """Parse non-streaming chatcompletion_v2 JSON."""
    base = parsed.get("base_resp")
    if isinstance(base, dict):
        code = base.get("status_code")
        if code not in (None, 0):
            msg = base.get("status_msg", "unknown error")
            raise RuntimeError(
                f"MiniMax chat API error: status_code={code} status_msg={msg}"
            )

    choices = parsed.get("choices") or []
    if not choices:
        raise RuntimeError("MiniMax chat returned no choices")

    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):
        # Multimodal-style segments; concatenate text parts
        parts = []
        for seg in content:
            if isinstance(seg, dict) and seg.get("type") == "text":
                parts.append(seg.get("text", ""))
            elif isinstance(seg, str):
                parts.append(seg)
        content = "".join(parts)

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("MiniMax chat returned an empty caption for the video frame")

    return content.strip()


def _middle_frame_jpeg_data_url(chunk_path: str) -> str:
    """Extract one JPEG frame near the middle of *chunk_path*; return a data URL."""
    dur = _get_video_duration(chunk_path)
    ss = max(0.0, dur / 2.0)
    ffmpeg = _get_ffmpeg_executable()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        out = f.name
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(ss),
                "-i",
                chunk_path,
                "-frames:v",
                "1",
                "-q:v",
                "3",
                out,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with open(out, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass


CAPTION_PROMPT = (
    "Describe this image in 2-4 short English sentences for video search. "
    "Focus on visible objects, people, vehicles, text, and the overall scene."
)


class MiniMaxEmbedder(BaseEmbedder):
    """MiniMax Open Platform: embo-01 text embeddings + vision caption for video."""

    def __init__(self):
        import httpx

        api_key = os.environ.get(MINIMAX_API_KEY_ENV)
        if not api_key:
            raise MiniMaxAPIKeyError(
                "MINIMAX_API_KEY is not set.\n\n"
                "Run: sentrysearch init\n\n"
                "Or set it manually:\n"
                "  export MINIMAX_API_KEY=your-key\n\n"
                "Or use a local model instead (no API key needed):\n"
                "  sentrysearch index <directory> --backend local"
            )
        self._api_key = api_key
        raw_vm = os.environ.get(MINIMAX_VISION_MODEL_ENV)
        self._vision_model, vm_warn = _resolve_vision_chat_model(raw_vm)
        if vm_warn:
            print(vm_warn, file=sys.stderr)
        self._client = httpx.Client(timeout=120.0)
        self._limiter = _RateLimiter()

    def embed_video_chunk(self, chunk_path: str, verbose: bool = False) -> list[float]:
        """Caption a middle frame via vision chat, then embed the caption (embo-01)."""
        if verbose:
            print(
                f"    [verbose] MiniMax: extracting frame + caption for {EMBED_MODEL}",
                file=sys.stderr,
            )

        data_url = _middle_frame_jpeg_data_url(chunk_path)

        self._limiter.wait()
        t0 = time.monotonic()
        try:
            caption = _retry(lambda: self._call_vision_caption(data_url))
        except MiniMaxSensitiveInputError:
            caption = _fallback_caption_policy_chunk(chunk_path)
            print(
                "  MiniMax: frame caption blocked by content policy; "
                "embedding a generic segment label instead.",
                file=sys.stderr,
            )
        elapsed_v = time.monotonic() - t0

        if verbose:
            preview = caption[:120] + ("…" if len(caption) > 120 else "")
            print(
                f"    [verbose] caption ({elapsed_v:.2f}s): {preview}",
                file=sys.stderr,
            )

        self._limiter.wait()
        t1 = time.monotonic()
        response = _retry(lambda: self._call_embedding_api(caption, embed_type="db"))
        elapsed_e = time.monotonic() - t1

        embedding = _embedding_from_response(response)

        if verbose:
            print(
                f"    [verbose] dims={len(embedding)}, embed_time={elapsed_e:.2f}s",
                file=sys.stderr,
            )

        import math

        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def embed_query(self, query_text: str, verbose: bool = False) -> list[float]:
        """Embed a text query using embo-01 (query vector)."""
        self._limiter.wait()
        t0 = time.monotonic()

        response = _retry(
            lambda: self._call_embedding_api(query_text, embed_type="query")
        )
        elapsed = time.monotonic() - t0

        embedding = _embedding_from_response(response)

        if verbose:
            print(
                f"  [verbose] query embedding: dims={len(embedding)}, "
                f"api_time={elapsed:.2f}s",
                file=sys.stderr,
            )

        import math

        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def dimensions(self) -> int:
        return DIMENSIONS

    def _group_params(self) -> dict:
        group_id = os.environ.get(MINIMAX_GROUP_ID_ENV)
        if group_id:
            return {"GroupId": group_id}
        return {}

    def _call_vision_caption(self, image_data_url: str) -> str:
        """POST /v1/text/chatcompletion_v2 — one frame, short caption."""
        url = f"{MINIMAX_API_BASE}/v1/text/chatcompletion_v2"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CAPTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            "max_completion_tokens": 256,
        }

        response = self._client.post(
            url, headers=headers, json=payload, params=self._group_params()
        )

        if response.status_code == 401:
            raise MiniMaxAPIKeyError(
                "Invalid MINIMAX_API_KEY. Please check your API key."
            )
        if response.status_code == 429:
            raise MiniMaxQuotaError("MiniMax API rate limit exceeded.")
        if response.status_code != 200:
            raise RuntimeError(
                f"MiniMax chat API error: {response.status_code} - {response.text}"
            )

        parsed = response.json()
        br = parsed.get("base_resp")
        if isinstance(br, dict) and br.get("status_code") == 1026:
            raise MiniMaxSensitiveInputError(
                "MiniMax rejected the frame image (content policy). "
                "Try a different backend or clip content."
            )
        return _assistant_text_from_chat_response(parsed)

    def _call_embedding_api(self, content: str, *, embed_type: str) -> dict:
        """POST /v1/embeddings — native ``texts`` + ``type``."""
        url = f"{MINIMAX_API_BASE}/v1/embeddings"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": EMBED_MODEL,
            "type": embed_type,
            "texts": [content],
        }

        response = self._client.post(
            url, headers=headers, json=payload, params=self._group_params()
        )

        if response.status_code == 401:
            raise MiniMaxAPIKeyError(
                "Invalid MINIMAX_API_KEY. Please check your API key."
            )
        elif response.status_code == 429:
            raise MiniMaxQuotaError(
                "MiniMax API rate limit exceeded."
            )
        elif response.status_code != 200:
            raise RuntimeError(
                f"MiniMax API error: {response.status_code} - {response.text}"
            )

        return response.json()
