"""VLM reranking helpers for search results."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass

from .trimmer import trim_clip


@dataclass(frozen=True)
class RerankScore:
    """Validated VLM rerank response."""

    rerank_match: bool
    rerank_confidence: float


RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "rerank_match": {"type": "boolean"},
        "rerank_confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": ["rerank_match", "rerank_confidence"],
}


def build_rerank_prompt(query: str) -> str:
    return (
        "You are reranking video search candidates.\n"
        "Look at the clip and decide whether it visually matches this search "
        f"query: {query!r}\n\n"
        "Return JSON only with this exact shape:\n"
        '{"rerank_match": true, "rerank_confidence": 0.0}\n'
        "Use rerank_match=true only when the clip contains the requested event. "
        "Use rerank_confidence as your confidence from 0.0 to 1.0."
    )


def _strip_json_code_fence(text):
    if not isinstance(text, str):
        return text
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return text
    if lines[0].strip() not in ("```", "```json", "```JSON"):
        return text
    return "\n".join(lines[1:-1]).strip()


def parse_rerank_response(text: str) -> RerankScore | None:
    """Parse and validate a VLM rerank JSON response.

    Invalid model output returns ``None`` so callers can fall back to the
    candidate's embedding rank.
    """
    try:
        data = json.loads(_strip_json_code_fence(text))
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    rerank_match = data.get("rerank_match")
    if not isinstance(rerank_match, bool):
        return None

    rerank_confidence = data.get("rerank_confidence")
    if isinstance(rerank_confidence, bool) or not isinstance(
        rerank_confidence, (int, float),
    ):
        return None

    rerank_confidence = float(rerank_confidence)
    if not math.isfinite(rerank_confidence):
        return None
    if rerank_confidence < 0.0 or rerank_confidence > 1.0:
        return None

    return RerankScore(
        rerank_match=rerank_match,
        rerank_confidence=rerank_confidence,
    )


def _sort_key(item: tuple[dict, int, RerankScore | None]) -> tuple:
    _result, original_rank, score = item
    if score is None:
        return (1, original_rank)
    if score.rerank_match:
        return (0, -score.rerank_confidence, original_rank)
    return (2, original_rank)


def rerank_results(
    query: str,
    results: list[dict],
    reranker,
    *,
    candidate_dir: str,
    verbose: bool = False,
) -> list[dict]:
    """Extract candidate clips, score them with *reranker*, and reorder results."""
    if not results:
        return []

    scored: list[tuple[dict, int, RerankScore | None]] = []
    for original_rank, result in enumerate(results):
        reranked = dict(result)
        score = None
        clip_path = os.path.join(candidate_dir, f"candidate_{original_rank:03d}.mp4")
        try:
            clip_path = trim_clip(
                result["source_file"],
                result["start_time"],
                result["end_time"],
                clip_path,
            )
            reranked["_rerank_clip_path"] = clip_path
            score = reranker.score(query, clip_path, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(
                    f"  [verbose] rerank candidate #{original_rank + 1} "
                    f"fallback: {exc}",
                    file=sys.stderr,
                )
        if score is not None:
            reranked["rerank_match"] = score.rerank_match
            reranked["rerank_confidence"] = score.rerank_confidence
        scored.append((reranked, original_rank, score))

    scored.sort(key=_sort_key)
    return [result for result, _rank, _score in scored]
