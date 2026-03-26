"""Tests for score fusion and merged search."""

import pytest


class TestFuseScores:
    """Tests for fuse_scores()."""

    def test_weighted_average_when_both_below_threshold(self):
        from sentrysearch.search import fuse_scores

        score = fuse_scores(0.5, 0.3)
        assert score == pytest.approx(0.4)

    def test_max_fallback_when_visual_high(self):
        from sentrysearch.search import fuse_scores

        score = fuse_scores(0.8, 0.2)
        assert score == pytest.approx(0.8)

    def test_max_fallback_when_text_high(self):
        from sentrysearch.search import fuse_scores

        score = fuse_scores(0.3, 0.75)
        assert score == pytest.approx(0.75)

    def test_max_fallback_when_both_high(self):
        from sentrysearch.search import fuse_scores

        score = fuse_scores(0.8, 0.9)
        assert score == pytest.approx(0.9)

    def test_custom_weights(self):
        from sentrysearch.search import fuse_scores

        score = fuse_scores(0.6, 0.4, visual_weight=0.7, text_weight=0.3)
        assert score == pytest.approx(0.6 * 0.7 + 0.4 * 0.3)

    def test_custom_high_confidence(self):
        from sentrysearch.search import fuse_scores

        score = fuse_scores(0.6, 0.3, high_confidence=0.5)
        assert score == pytest.approx(0.6)


class TestMatchSource:
    """Tests for _match_source()."""

    def test_visual_dominant(self):
        from sentrysearch.search import _match_source

        assert _match_source(0.8, 0.3) == "visual"

    def test_text_dominant(self):
        from sentrysearch.search import _match_source

        assert _match_source(0.3, 0.8) == "audio"

    def test_both_close(self):
        from sentrysearch.search import _match_source

        assert _match_source(0.5, 0.5) == "both"

    def test_both_within_threshold(self):
        from sentrysearch.search import _match_source

        assert _match_source(0.6, 0.55) == "both"


class TestMergeResults:
    """Tests for _merge_results()."""

    def test_merges_visual_and_text_hits(self):
        from sentrysearch.search import _merge_results

        visual_hits = [
            {"source_file": "/v.mp4", "start_time": 0.0, "end_time": 30.0, "score": 0.6},
        ]
        text_hits = [
            {"source_file": "/v.mp4", "start_time": 0.0, "end_time": 30.0, "score": 0.8,
             "transcript": "hello"},
        ]

        merged = _merge_results(visual_hits, text_hits)
        assert len(merged) == 1
        assert merged[0]["fused_score"] == pytest.approx(0.8)
        assert merged[0]["match_source"] == "audio"

    def test_visual_only_chunks(self):
        from sentrysearch.search import _merge_results

        visual_hits = [
            {"source_file": "/v.mp4", "start_time": 0.0, "end_time": 30.0, "score": 0.5},
        ]
        text_hits = []

        merged = _merge_results(visual_hits, text_hits)
        assert len(merged) == 1
        assert merged[0]["fused_score"] == pytest.approx(0.5)
        assert merged[0]["match_source"] == "visual"

    def test_sorted_by_fused_score_descending(self):
        from sentrysearch.search import _merge_results

        visual_hits = [
            {"source_file": "/v.mp4", "start_time": 0.0, "end_time": 30.0, "score": 0.3},
            {"source_file": "/v.mp4", "start_time": 30.0, "end_time": 60.0, "score": 0.8},
        ]
        text_hits = [
            {"source_file": "/v.mp4", "start_time": 0.0, "end_time": 30.0, "score": 0.9,
             "transcript": "important"},
        ]

        merged = _merge_results(visual_hits, text_hits)
        assert merged[0]["start_time"] == 0.0
        assert merged[1]["start_time"] == 30.0
