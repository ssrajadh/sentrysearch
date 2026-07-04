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


def parse_rerank_response(text: str) -> RerankScore | None:
    """Parse and validate the Gemini rerank JSON response.

    Invalid model output returns ``None`` so callers can fall back to the
    candidate's embedding rank.
    """
    try:
        data = json.loads(text)
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
