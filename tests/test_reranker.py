"""Tests for VLM reranking helpers."""

from sentrysearch.reranker import (
    RERANK_SCHEMA,
    RerankScore,
    build_rerank_prompt,
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
    def test_prompt_is_video_generic(self):
        prompt = build_rerank_prompt("red truck")
        assert "video search candidates" in prompt
        assert "red truck" in prompt

    def test_schema_requires_score_fields(self):
        assert RERANK_SCHEMA["required"] == [
            "rerank_match", "rerank_confidence",
        ]

    def test_accepts_valid_json(self):
        score = parse_rerank_response(
            '{"rerank_match": true, "rerank_confidence": 0.75}',
        )
        assert score == RerankScore(True, 0.75)

    def test_accepts_json_code_fence(self):
        score = parse_rerank_response(
            '```json\n{"rerank_match": true, "rerank_confidence": 0.75}\n```',
        )
        assert score == RerankScore(True, 0.75)

    def test_rejects_malformed_json(self):
        assert parse_rerank_response("{not json") is None

    def test_rejects_missing_text(self):
        assert parse_rerank_response(None) is None

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
    def test_sorts_matches_by_confidence_then_embedding_rank(
        self, monkeypatch, tmp_path,
    ):
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

        reranked = rerank_results(
            "query", results, reranker, candidate_dir=str(tmp_path),
        )
        assert [r["source_file"] for r in reranked] == [
            "/videos/b.mp4",
            "/videos/d.mp4",
            "/videos/a.mp4",
            "/videos/c.mp4",
        ]

    def test_invalid_scores_preserve_embedding_order(self, monkeypatch, tmp_path):
        results = [_result("a"), _result("b"), _result("c")]
        reranker = type("FakeReranker", (), {
            "score": lambda self, *args, **kwargs: None,
        })()

        monkeypatch.setattr(
            "sentrysearch.reranker.trim_clip",
            lambda *args, **kwargs: args[3],
        )

        reranked = rerank_results(
            "query", results, reranker, candidate_dir=str(tmp_path),
        )
        assert [r["source_file"] for r in reranked] == [
            "/videos/a.mp4",
            "/videos/b.mp4",
            "/videos/c.mp4",
        ]
        assert all("_rerank_clip_path" in r for r in reranked)
        assert all("rerank_match" not in r for r in reranked)

    def test_extracts_candidate_before_scoring(self, monkeypatch, tmp_path):
        events = []
        results = [_result("a"), _result("b")]

        def fake_trim(source_file, start_time, end_time, output_path, **kwargs):
            events.append(("trim", source_file, kwargs))
            return output_path

        class FakeReranker:
            def score(self, query, clip_path, *, verbose=False):
                events.append(("score", clip_path, verbose))
                return None

        monkeypatch.setattr("sentrysearch.reranker.trim_clip", fake_trim)

        rerank_results(
            "query", results, FakeReranker(),
            candidate_dir=str(tmp_path),
            verbose=True,
        )

        assert events[0] == ("trim", "/videos/a.mp4", {})
        assert events[1][0] == "score"
        assert events[2] == ("trim", "/videos/b.mp4", {})
        assert events[3][0] == "score"

    def test_trim_failure_falls_back_to_embedding_rank(self, monkeypatch, tmp_path):
        results = [_result("a"), _result("b"), _result("c"), _result("d")]

        def fake_trim(source_file, start_time, end_time, output_path):
            if source_file == "/videos/b.mp4":
                raise RuntimeError("bad trim")
            return output_path

        class FakeReranker:
            def score(self, query, clip_path, *, verbose=False):
                if clip_path.endswith("candidate_002.mp4"):
                    return RerankScore(True, 0.5)
                if clip_path.endswith("candidate_003.mp4"):
                    return RerankScore(False, 0.9)
                return None

        monkeypatch.setattr("sentrysearch.reranker.trim_clip", fake_trim)

        reranked = rerank_results(
            "query", results, FakeReranker(), candidate_dir=str(tmp_path),
        )

        assert [r["source_file"] for r in reranked] == [
            "/videos/c.mp4",
            "/videos/a.mp4",
            "/videos/b.mp4",
            "/videos/d.mp4",
        ]
        assert "rerank_match" not in reranked[2]
        assert "_rerank_clip_path" not in reranked[2]

    def test_score_failure_falls_back_and_valid_scores_still_reorder(
        self, monkeypatch, tmp_path,
    ):
        results = [_result("a"), _result("b"), _result("c")]

        monkeypatch.setattr(
            "sentrysearch.reranker.trim_clip",
            lambda *args, **kwargs: args[3],
        )

        class FakeReranker:
            def score(self, query, clip_path, *, verbose=False):
                if clip_path.endswith("candidate_000.mp4"):
                    raise RuntimeError("quota hit")
                if clip_path.endswith("candidate_001.mp4"):
                    return RerankScore(True, 0.8)
                return RerankScore(False, 0.9)

        reranked = rerank_results(
            "query", results, FakeReranker(), candidate_dir=str(tmp_path),
        )

        assert [r["source_file"] for r in reranked] == [
            "/videos/b.mp4",
            "/videos/a.mp4",
            "/videos/c.mp4",
        ]
        assert "rerank_match" not in reranked[1]
        assert "_rerank_clip_path" in reranked[1]

    def test_candidate_failure_logs_only_when_verbose(
        self, monkeypatch, capsys, tmp_path,
    ):
        results = [_result("a")]

        def fake_trim(*args, **kwargs):
            raise RuntimeError("bad trim")

        class FakeReranker:
            def score(self, *args, **kwargs):
                raise AssertionError("trim failure should skip scoring")

        monkeypatch.setattr("sentrysearch.reranker.trim_clip", fake_trim)

        rerank_results(
            "query", results, FakeReranker(), candidate_dir=str(tmp_path),
        )
        assert capsys.readouterr().err == ""

        rerank_results(
            "query", results, FakeReranker(),
            candidate_dir=str(tmp_path),
            verbose=True,
        )
        assert "rerank candidate #1 fallback: bad trim" in capsys.readouterr().err

    def test_empty_results_do_not_score(self, tmp_path):
        class FakeReranker:
            def score(self, *args, **kwargs):
                raise AssertionError("should not score empty results")

        assert rerank_results(
            "query", [], FakeReranker(), candidate_dir=str(tmp_path),
        ) == []
