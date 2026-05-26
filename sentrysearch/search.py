"""Query and retrieval logic."""

from __future__ import annotations

import numpy as np

from .embedder import embed_image, embed_query
from .highlights import _dedupe_indices, _normalize
from .store import SentryStore


def _search_with_embedding(
    embedding: list[float],
    store: SentryStore,
    n_results: int,
    dedupe_threshold: float | None = None,
) -> list[dict]:
    include_embeddings = dedupe_threshold is not None
    hits = store.search(embedding, n_results=n_results,
                        include_embeddings=include_embeddings)
    results = [
        {
            "source_file": hit["source_file"],
            "start_time": hit["start_time"],
            "end_time": hit["end_time"],
            "similarity_score": hit["score"],
        }
        for hit in hits
    ]
    results.sort(key=lambda r: r["similarity_score"], reverse=True)

    if dedupe_threshold is not None and len(results) > 1:
        embeddings = np.array([h["embedding"] for h in hits], dtype=np.float32)
        Xn = _normalize(embeddings)
        ranked = np.arange(len(results))
        kept = _dedupe_indices(ranked, Xn, dedupe_threshold, len(results))
        results = [results[i] for i in kept]

    return results


def search_footage(
    query: str,
    store: SentryStore,
    n_results: int = 5,
    verbose: bool = False,
    dedupe_threshold: float | None = None,
) -> list[dict]:
    """Search indexed footage with a natural language query.

    Args:
        query: Natural language search string.
        store: SentryStore instance to search against.
        n_results: Maximum number of results to return.
        verbose: If True, print debug info to stderr.
        dedupe_threshold: When set, drop results whose cosine similarity
            to a higher-ranked result exceeds this value (0-1).

    Returns:
        List of result dicts sorted by relevance (best first).
        Each dict contains: source_file, start_time, end_time, similarity_score.
    """
    return _search_with_embedding(
        embed_query(query, verbose=verbose), store, n_results,
        dedupe_threshold=dedupe_threshold,
    )


def search_footage_by_image(
    image_path: str,
    store: SentryStore,
    n_results: int = 5,
    verbose: bool = False,
    dedupe_threshold: float | None = None,
) -> list[dict]:
    """Search indexed footage using an image as the query."""
    return _search_with_embedding(
        embed_image(image_path, verbose=verbose), store, n_results,
        dedupe_threshold=dedupe_threshold,
    )
