"""OrcaRouter VLM reranker for search results.

Routes the candidate-clip rerank call through OrcaRouter's gemini-compatible
endpoint, so the ``--rerank`` flag works for the ``orca`` backend without any
extra runtime dependency (the same google-genai SDK drives the ``gemini``
backend).
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

from .orca_embedder import (
    API_BASE_URL,
    API_VERSION,
    OrcaAPIError,
    OrcaAPIKeyError,
    OrcaEmbedder,
    _retry,
    default_orca_rerank_model,
    get_limiter,
)
from .reranker import (
    RERANK_SCHEMA,
    RerankScore,
    build_rerank_prompt,
    parse_rerank_response,
)

load_dotenv()


class OrcaReranker:
    """OrcaRouter VLM reranker for candidate video clips."""

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
                "Or add it to ~/.sentrysearch/.env"
            )
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                base_url=API_BASE_URL,
                api_version=API_VERSION,
            ),
        )
        self._types = types
        self._model = default_orca_rerank_model()
        # Share one window with the embedder: both draw on the same gateway quota.
        self._limiter = get_limiter(rpm)

    def score(
        self,
        query: str,
        clip_path: str,
        *,
        verbose: bool = False,
    ) -> RerankScore | None:
        """Return a validated rerank score, or None for unparsable model output."""
        import os as _os

        if not _os.path.isfile(clip_path):
            raise OrcaAPIError(f"Clip file not found: {clip_path}")

        video_part = OrcaEmbedder._make_video_part(clip_path, self._types)
        prompt_part = self._types.Part(text=build_rerank_prompt(query))
        config = self._types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=RERANK_SCHEMA,
            temperature=0.0,
        )

        self._limiter.wait()
        t0 = time.monotonic()
        response = _retry(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=self._types.Content(parts=[video_part, prompt_part]),
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
                f"  [verbose] rerank {self._model}: {status}, "
                f"api_time={elapsed:.2f}s",
                file=sys.stderr,
            )
        return score
