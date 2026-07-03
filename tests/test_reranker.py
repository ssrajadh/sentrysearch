"""Tests for VLM reranking helpers."""

from sentrysearch.reranker import (
    RerankScore,
    parse_rerank_response,
    rerank_results,
)


def _result(name: str, score: float = 0.8) -> dict:
    return {
        "source_file": f"/videos/{name}.mp4",
        "start_time": 0.0,
        "end_time": 30.0,
        "similarity_score": score,
    }


class TestParseRerankResponse:
    def test_accepts_valid_json(self):
        score = parse_rerank_response(
            '{"rerank_match": true, "rerank_confidence": 0.75}',
        )
        assert score == RerankScore(True, 0.75)

    def test_rejects_malformed_json(self):
        assert parse_rerank_response("{not json") is None

    def test_rejects_non_object_json(self):
        assert parse_rerank_response("[true, 0.8]") is None

    def test_rejects_string_boolean(self):
        assert parse_rerank_response(
            '{"rerank_match": "true", "rerank_confidence": 0.8}',
        ) is None

    def test_rejects_missing_fields(self):
        assert parse_rerank_response('{"rerank_match": true}') is None

    def test_rejects_non_finite_confidence(self):
        assert parse_rerank_response(
            '{"rerank_match": true, "rerank_confidence": NaN}',
        ) is None

    def test_rejects_out_of_range_confidence(self):
        assert parse_rerank_response(
            '{"rerank_match": true, "rerank_confidence": 1.2}',
        ) is None

    def test_rejects_boolean_confidence(self):
        assert parse_rerank_response(
            '{"rerank_match": true, "rerank_confidence": true}',
        ) is None


class TestRerankResults:
    def test_sorts_matches_by_confidence_then_embedding_rank(self, monkeypatch):
        results = [_result("a"), _result("b"), _result("c"), _result("d")]
        scores = [
            None,
            RerankScore(True, 0.8),
            RerankScore(False, 0.99),
            RerankScore(True, 0.8),
        ]
        reranker = type("FakeReranker", (), {
            "score": lambda self, *args, **kwargs: scores.pop(0),
        })()

        monkeypatch.setattr(
            "sentrysearch.reranker.trim_clip",
            lambda *args, **kwargs: args[3],
        )

        reranked = rerank_results("query", results, reranker)
        assert [r["source_file"] for r in reranked] == [
            "/videos/b.mp4",
            "/videos/d.mp4",
            "/videos/a.mp4",
            "/videos/c.mp4",
        ]

    def test_invalid_scores_preserve_embedding_order(self, monkeypatch):
        results = [_result("a"), _result("b"), _result("c")]
        reranker = type("FakeReranker", (), {
            "score": lambda self, *args, **kwargs: None,
        })()

        monkeypatch.setattr(
            "sentrysearch.reranker.trim_clip",
            lambda *args, **kwargs: args[3],
        )

        reranked = rerank_results("query", results, reranker)
        assert reranked == results

    def test_extracts_candidate_before_scoring(self, monkeypatch):
        events = []
        results = [_result("a"), _result("b")]

        def fake_trim(source_file, start_time, end_time, output_path, *, padding):
            events.append(("trim", source_file, padding))
            return output_path

        class FakeReranker:
            def score(self, query, clip_path, *, verbose=False):
                events.append(("score", clip_path, verbose))
                return None

        monkeypatch.setattr("sentrysearch.reranker.trim_clip", fake_trim)

        rerank_results("query", results, FakeReranker(), verbose=True)

        assert events[0] == ("trim", "/videos/a.mp4", 0.0)
        assert events[1][0] == "score"
        assert events[2] == ("trim", "/videos/b.mp4", 0.0)
        assert events[3][0] == "score"

    def test_empty_results_do_not_score(self):
        class FakeReranker:
            def score(self, *args, **kwargs):
                raise AssertionError("should not score empty results")

        assert rerank_results("query", [], FakeReranker()) == []
