"""Tests for the OrcaRouter embedder and reranker."""

import os
from unittest.mock import MagicMock, patch

import pytest

from sentrysearch.embedder import get_embedder, reset_embedder
from sentrysearch.orca_embedder import (
    API_BASE_URL,
    OrcaAPIError,
    OrcaAPIKeyError,
    OrcaEmbedder,
    _RateLimiter,
    _retry,
    default_orca_embedding_model,
    default_orca_rerank_model,
)


class TestDefaults:
    def test_default_embedding_model(self, monkeypatch):
        monkeypatch.delenv("ORCAROUTER_EMBEDDING_MODEL", raising=False)
        assert default_orca_embedding_model() == "google/gemini-embedding-2-preview"

    def test_default_embedding_model_env_override(self, monkeypatch):
        monkeypatch.setenv("ORCAROUTER_EMBEDDING_MODEL", "custom/model")
        assert default_orca_embedding_model() == "custom/model"

    def test_default_rerank_model(self, monkeypatch):
        monkeypatch.delenv("ORCAROUTER_RERANK_MODEL", raising=False)
        assert default_orca_rerank_model() == "google/gemini-2.5-flash"


class TestRateLimiter:
    def test_allows_under_limit(self):
        limiter = _RateLimiter(max_per_minute=5)
        for _ in range(5):
            limiter.wait()

    @patch("sentrysearch.orca_embedder.time.sleep")
    @patch("sentrysearch.orca_embedder.time.monotonic")
    def test_blocks_at_limit(self, mock_mono, mock_sleep):
        limiter = _RateLimiter(max_per_minute=2)
        mock_mono.side_effect = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 62.0, 62.0, 63.0, 63.0]
        limiter.wait()
        limiter.wait()
        limiter.wait()
        mock_sleep.assert_called_once()


class TestRetry:
    def test_returns_on_first_success(self):
        fn = MagicMock(return_value="ok")
        assert _retry(fn, max_retries=3, initial_delay=0.01) == "ok"
        fn.assert_called_once()

    @patch("sentrysearch.orca_embedder.time.sleep")
    def test_retries_on_429(self, mock_sleep):
        exc = Exception("429 rate limit")
        exc.code = 429
        fn = MagicMock(side_effect=[exc, exc, "ok"])
        assert _retry(fn, max_retries=3, initial_delay=0.01) == "ok"
        assert fn.call_count == 3

    @patch("sentrysearch.orca_embedder.time.sleep")
    def test_retries_on_503(self, mock_sleep):
        exc = Exception("503 unavailable")
        exc.code = 503
        fn = MagicMock(side_effect=[exc, "ok"])
        assert _retry(fn, max_retries=3, initial_delay=0.01) == "ok"

    def test_non_retryable_wraps_in_orca_error(self):
        fn = MagicMock(side_effect=ValueError("bad input"))
        with pytest.raises(OrcaAPIError, match="bad input"):
            _retry(fn, max_retries=2, initial_delay=0.01)

    @patch("sentrysearch.orca_embedder.time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        exc = Exception("503 error")
        exc.code = 503
        fn = MagicMock(side_effect=[exc, exc, "ok"])
        _retry(fn, max_retries=3, initial_delay=1.0)
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert delays[0] == 1.0
        assert delays[1] == 2.0


class TestOrcaEmbedder:
    def test_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ORCAROUTER_API_KEY", None)
            with pytest.raises(OrcaAPIKeyError, match="ORCAROUTER_API_KEY"):
                OrcaEmbedder()

    @patch("google.genai.Client")
    def test_creates_client_with_base_url(self, mock_client_cls):
        with patch.dict(os.environ, {"ORCAROUTER_API_KEY": "sk-orca-test"}):
            embedder = OrcaEmbedder()
            mock_client_cls.assert_called_once()
            kwargs = mock_client_cls.call_args.kwargs
            assert kwargs["api_key"] == "sk-orca-test"
            assert kwargs["http_options"].base_url == API_BASE_URL
            assert embedder.dimensions() == 768


class TestEmbedderFactory:
    def test_get_embedder_orca_backend(self):
        with patch("sentrysearch.orca_embedder.OrcaEmbedder") as MockOrca:
            MockOrca.return_value = MagicMock()
            reset_embedder()
            result = get_embedder("orca")
            MockOrca.assert_called_once_with(rpm=None)
            assert result is MockOrca.return_value

    def test_get_embedder_orca_backend_with_rpm(self):
        with patch("sentrysearch.orca_embedder.OrcaEmbedder") as MockOrca:
            MockOrca.return_value = MagicMock()
            reset_embedder()
            get_embedder("orca", rpm=12)
            MockOrca.assert_called_once_with(rpm=12)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_embedder("nonexistent")


class TestOrcaReranker:
    def test_raises_without_api_key(self):
        from sentrysearch.orca_reranker import OrcaReranker

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ORCAROUTER_API_KEY", None)
            with pytest.raises(OrcaAPIKeyError, match="ORCAROUTER_API_KEY"):
                OrcaReranker()

    @patch("google.genai.Client")
    def test_creates_client_with_base_url(self, mock_client_cls):
        from sentrysearch.orca_reranker import OrcaReranker

        with patch.dict(os.environ, {"ORCAROUTER_API_KEY": "sk-orca-test"}):
            reranker = OrcaReranker()
            kwargs = mock_client_cls.call_args.kwargs
            assert kwargs["http_options"].base_url == API_BASE_URL
            assert reranker._model == "google/gemini-2.5-flash"
