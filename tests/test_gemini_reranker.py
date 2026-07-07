"""Tests for Gemini Flash reranker."""

import os
from unittest.mock import MagicMock, patch

import pytest

from sentrysearch.gemini_embedder import GeminiAPIKeyError
from sentrysearch.gemini_reranker import GeminiReranker, RERANK_MODEL
from sentrysearch.reranker import RERANK_SCHEMA, RerankScore


class TestGeminiReranker:
    def test_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(GeminiAPIKeyError, match="GEMINI_API_KEY"):
                GeminiReranker()

    @patch("google.genai.Client")
    def test_creates_client_with_key(self, mock_client_cls):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"}):
            GeminiReranker()
            mock_client_cls.assert_called_once_with(api_key="test-key-123")

    @patch("sentrysearch.gemini_reranker._retry", side_effect=lambda fn: fn())
    @patch("sentrysearch.gemini_reranker.build_rerank_prompt",
           return_value="shared prompt")
    @patch("sentrysearch.gemini_reranker._RateLimiter")
    @patch("google.genai.Client")
    def test_score_uses_flash_json_retry_and_limiter(
        self, mock_client_cls, mock_limiter_cls, mock_prompt, mock_retry,
        tmp_path,
    ):
        clip = tmp_path / "candidate.mp4"
        clip.write_bytes(b"fake-video")
        response = MagicMock()
        response.text = '{"rerank_match": true, "rerank_confidence": 0.91}'
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response
        mock_client_cls.return_value = mock_client

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            reranker = GeminiReranker()
            score = reranker.score("red truck", str(clip))

        assert score == RerankScore(True, 0.91)
        mock_prompt.assert_called_once_with("red truck")
        mock_limiter_cls.return_value.wait.assert_called_once()
        mock_retry.assert_called_once()

        call = mock_client.models.generate_content.call_args
        assert call.kwargs["model"] == RERANK_MODEL
        assert call.kwargs["model"] == "gemini-2.5-flash"
        config = call.kwargs["config"]
        assert config.response_mime_type == "application/json"
        assert config.response_json_schema == RERANK_SCHEMA

    @patch("sentrysearch.gemini_reranker._retry", side_effect=lambda fn: fn())
    @patch("sentrysearch.gemini_reranker._RateLimiter")
    @patch("google.genai.Client")
    def test_score_returns_none_for_unparseable_output(
        self, mock_client_cls, _mock_limiter_cls, _mock_retry, tmp_path,
    ):
        clip = tmp_path / "candidate.mp4"
        clip.write_bytes(b"fake-video")
        response = MagicMock()
        response.text = "not json"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response
        mock_client_cls.return_value = mock_client

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            reranker = GeminiReranker()
            assert reranker.score("red truck", str(clip)) is None
