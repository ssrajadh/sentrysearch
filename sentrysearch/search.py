"""Query and retrieval logic with visual+text score fusion."""

from .embedder import embed_query
from .store import SentryStore

HIGH_CONFIDENCE = 0.7
VISUAL_WEIGHT = 0.5
TEXT_WEIGHT = 0.5
SOURCE_THRESHOLD = 0.1


def fuse_scores(
    visual: float,
    text: float,
    visual_weight: float = VISUAL_WEIGHT,
    text_weight: float = TEXT_WEIGHT,
    high_confidence: float = HIGH_CONFIDENCE,
) -> float:
    """Fuse visual and text similarity scores.

    Uses weighted average normally, but if either score exceeds the
    high-confidence threshold, returns the max instead.
    """
    if visual >= high_confidence or text >= high_confidence:
        return max(visual, text)
    return visual_weight * visual + text_weight * text


def _match_source(visual: float, text: float) -> str:
    """Determine which modality contributed more to the match."""
    diff = abs(visual - text)
    if diff < SOURCE_THRESHOLD:
        return "both"
    return "visual" if visual > text else "audio"


def _merge_results(
    visual_hits: list[dict],
    text_hits: list[dict],
    visual_weight: float = VISUAL_WEIGHT,
    text_weight: float = TEXT_WEIGHT,
    high_confidence: float = HIGH_CONFIDENCE,
) -> list[dict]:
    """Merge visual and text search results by chunk identity."""
    text_by_key: dict[tuple, dict] = {}
    for hit in text_hits:
        key = (hit["source_file"], hit["start_time"])
        text_by_key[key] = hit

    merged: dict[tuple, dict] = {}

    for hit in visual_hits:
        key = (hit["source_file"], hit["start_time"])
        visual_score = hit["score"]
        text_hit = text_by_key.pop(key, None)
        text_score = text_hit["score"] if text_hit else 0.0

        if text_hit:
            fused = fuse_scores(visual_score, text_score, visual_weight,
                                text_weight, high_confidence)
            source = _match_source(visual_score, text_score)
        else:
            fused = visual_score
            source = "visual"

        merged[key] = {
            "source_file": hit["source_file"],
            "start_time": hit["start_time"],
            "end_time": hit["end_time"],
            "similarity_score": fused,
            "fused_score": fused,
            "visual_score": visual_score,
            "text_score": text_score,
            "match_source": source,
            "transcript": text_hit.get("transcript", "") if text_hit else "",
        }

    for key, hit in text_by_key.items():
        text_score = hit["score"]
        merged[key] = {
            "source_file": hit["source_file"],
            "start_time": hit["start_time"],
            "end_time": hit["end_time"],
            "similarity_score": text_score,
            "fused_score": text_score,
            "visual_score": 0.0,
            "text_score": text_score,
            "match_source": "audio",
            "transcript": hit.get("transcript", ""),
        }

    return sorted(merged.values(), key=lambda r: r["fused_score"], reverse=True)


def search_footage(
    query: str,
    store: SentryStore,
    n_results: int = 5,
    verbose: bool = False,
    visual_weight: float = VISUAL_WEIGHT,
    text_weight: float = TEXT_WEIGHT,
    high_confidence: float = HIGH_CONFIDENCE,
) -> list[dict]:
    """Search indexed footage with a natural language query.

    Queries both visual and text collections, fuses scores, and returns
    merged results sorted by relevance.
    """
    query_embedding = embed_query(query, verbose=verbose)
    visual_hits = store.search(query_embedding, n_results=n_results)

    text_hits = []
    if store.has_text_index():
        text_hits = store.search_text(query_embedding, n_results=n_results)

    if not text_hits:
        results = []
        for hit in visual_hits:
            results.append({
                "source_file": hit["source_file"],
                "start_time": hit["start_time"],
                "end_time": hit["end_time"],
                "similarity_score": hit["score"],
                "fused_score": hit["score"],
                "visual_score": hit["score"],
                "text_score": 0.0,
                "match_source": "visual",
                "transcript": "",
            })
        results.sort(key=lambda r: r["similarity_score"], reverse=True)
        return results

    merged = _merge_results(visual_hits, text_hits, visual_weight,
                            text_weight, high_confidence)
    return merged[:n_results]
