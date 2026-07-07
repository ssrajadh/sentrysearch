"""Gemini Flash VLM reranker."""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

from .gemini_embedder import (
    GeminiAPIKeyError,
    GeminiEmbedder,
    _RateLimiter,
    _retry,
)
from .reranker import (
    RERANK_SCHEMA,
    RerankScore,
    build_rerank_prompt,
    parse_rerank_response,
)

load_dotenv()

RERANK_MODEL = "gemini-2.5-flash"


class GeminiReranker:
    """Gemini Flash reranker for candidate video clips."""

    def __init__(self):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise GeminiAPIKeyError(
                "GEMINI_API_KEY is not set.\n\n"
                "Run: sentrysearch init\n\n"
                "Or set it manually:\n"
                "  export GEMINI_API_KEY=your-key"
            )
        self._client = genai.Client(api_key=api_key)
        self._limiter = _RateLimiter()

    def score(
        self,
        query: str,
        clip_path: str,
        *,
        verbose: bool = False,
    ) -> RerankScore | None:
        """Return a validated rerank score, or None for unparsable model output."""
        from google.genai import types

        video_part = GeminiEmbedder._make_video_part(clip_path, types)
        prompt_part = types.Part(text=build_rerank_prompt(query))
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=RERANK_SCHEMA,
            temperature=0.0,
        )

        self._limiter.wait()
        t0 = time.monotonic()
        response = _retry(
            lambda: self._client.models.generate_content(
                model=RERANK_MODEL,
                contents=types.Content(parts=[video_part, prompt_part]),
                config=config,
            )
        )
        elapsed = time.monotonic() - t0

        score = parse_rerank_response(getattr(response, "text", None))
        if verbose:
            status = "fallback" if score is None else (
                f"match={score.rerank_match}, "
                f"confidence={score.rerank_confidence:.2f}"
            )
            print(
                f"  [verbose] rerank {RERANK_MODEL}: {status}, "
                f"api_time={elapsed:.2f}s",
                file=sys.stderr,
            )
        return score
