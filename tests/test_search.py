"""Tests for sentrysearch.search."""

import math

import pytest

from sentrysearch.search import search_footage


def _make_embedding(seed: float, dim: int = 768) -> list[float]:
    vec = [math.sin(seed + i * 0.1) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


class TestSearchFootage:
    def test_empty_store(self, tmp_store, mock_embed_query):
        results = search_footage("a red car", tmp_store)
        assert results == []

    def test_returns_results(self, tmp_store, mock_embed_query):
        # mock_embed_query returns _fake_embedding(), store a chunk with same vector
        tmp_store.add_chunk("c1", mock_embed_query, {
            "source_file": "video.mp4",
            "start_time": 0.0,
            "end_time": 30.0,
        })
        results = search_footage("anything", tmp_store, n_results=5)
        assert len(results) == 1
        assert results[0]["source_file"] == "video.mp4"
        assert results[0]["similarity_score"] > 0.99

    def test_sorted_by_score(self, tmp_store, mock_embed_query):
        tmp_store.add_chunk("match", mock_embed_query, {
            "source_file": "match.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        tmp_store.add_chunk("diff", _make_embedding(seed=999.0), {
            "source_file": "diff.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        results = search_footage("query", tmp_store, n_results=5)
        assert len(results) == 2
        assert results[0]["source_file"] == "match.mp4"
        assert results[0]["similarity_score"] > results[1]["similarity_score"]

    def test_n_results_limits_output(self, tmp_store, mock_embed_query):
        for i in range(10):
            tmp_store.add_chunk(f"c{i}", _make_embedding(seed=float(i)), {
                "source_file": f"v{i}.mp4",
                "start_time": 0.0,
                "end_time": 30.0,
            })
        results = search_footage("q", tmp_store, n_results=3)
        assert len(results) == 3


class TestSearchDedupe:
    def test_dedupe_drops_near_duplicates(self, tmp_store, mock_embed_query):
        base = mock_embed_query
        near_dup = base.copy()
        near_dup[0] += 0.001
        norm = math.sqrt(sum(x * x for x in near_dup))
        near_dup = [x / norm for x in near_dup]

        distinct = [0.0] * len(base)
        distinct[0] = 1.0

        tmp_store.add_chunk("c0", base, {
            "source_file": "a.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        tmp_store.add_chunk("c1", near_dup, {
            "source_file": "b.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        tmp_store.add_chunk("c2", distinct, {
            "source_file": "c.mp4", "start_time": 0.0, "end_time": 30.0,
        })

        results_no_dedup = search_footage("q", tmp_store, n_results=5)
        assert len(results_no_dedup) == 3

        results_dedup = search_footage("q", tmp_store, n_results=5,
                                       dedupe_threshold=0.9)
        assert len(results_dedup) == 2
        files = {r["source_file"] for r in results_dedup}
        assert "a.mp4" in files
        assert "c.mp4" in files

    def test_dedupe_none_returns_all(self, tmp_store, mock_embed_query):
        base = mock_embed_query
        near_dup = base.copy()
        near_dup[0] += 0.001
        norm = math.sqrt(sum(x * x for x in near_dup))
        near_dup = [x / norm for x in near_dup]

        tmp_store.add_chunk("c0", base, {
            "source_file": "a.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        tmp_store.add_chunk("c1", near_dup, {
            "source_file": "b.mp4", "start_time": 0.0, "end_time": 30.0,
        })

        results = search_footage("q", tmp_store, n_results=5)
        assert len(results) == 2

    def test_dedupe_threshold_1_keeps_all(self, tmp_store, mock_embed_query):
        base = mock_embed_query
        near_dup = base.copy()
        near_dup[0] += 0.001
        norm = math.sqrt(sum(x * x for x in near_dup))
        near_dup = [x / norm for x in near_dup]

        tmp_store.add_chunk("c0", base, {
            "source_file": "a.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        tmp_store.add_chunk("c1", near_dup, {
            "source_file": "b.mp4", "start_time": 0.0, "end_time": 30.0,
        })

        results = search_footage("q", tmp_store, n_results=5,
                                 dedupe_threshold=1.0)
        assert len(results) == 2

    def test_dedupe_single_result(self, tmp_store, mock_embed_query):
        tmp_store.add_chunk("c0", mock_embed_query, {
            "source_file": "a.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        results = search_footage("q", tmp_store, n_results=5,
                                 dedupe_threshold=0.9)
        assert len(results) == 1

    def test_dedupe_still_fills_n_results(self, tmp_store, mock_embed_query):
        """Dedupe used to run on only the top n hits, so three near-copies
        of one moment turned -n 3 into a single result."""
        base = mock_embed_query
        for i in range(3):
            dup = base.copy()
            dup[0] += 0.001 * (i + 1)
            norm = math.sqrt(sum(x * x for x in dup))
            tmp_store.add_chunk(f"d{i}", [x / norm for x in dup], {
                "source_file": f"dup{i}.mp4", "start_time": 0.0, "end_time": 30.0,
            })
        for i in range(3):  # mutually orthogonal, so none dedupes another
            other = [0.0] * len(base)
            other[i + 1] = 1.0
            tmp_store.add_chunk(f"o{i}", other, {
                "source_file": f"other{i}.mp4", "start_time": 0.0, "end_time": 30.0,
            })

        results = search_footage("q", tmp_store, n_results=3,
                                 dedupe_threshold=0.9)
        assert len(results) == 3
        assert sum(r["source_file"].startswith("dup") for r in results) == 1

    def test_dedupe_1_keeps_exact_duplicates(self, tmp_store, mock_embed_query):
        """Float error can put the cosine of identical vectors a hair
        above 1.0; --dedupe 1 must still mean 'off'."""
        for i in range(2):
            tmp_store.add_chunk(f"c{i}", mock_embed_query, {
                "source_file": f"v{i}.mp4", "start_time": 0.0, "end_time": 30.0,
            })
        results = search_footage("q", tmp_store, n_results=5,
                                 dedupe_threshold=1.0)
        assert len(results) == 2
