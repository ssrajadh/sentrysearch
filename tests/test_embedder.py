"""Tests for sentrysearch embedder (factory + gemini backend)."""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from sentrysearch.gemini_embedder import (
    DEFAULT_RPM,
    GeminiAPIKeyError,
    GeminiEmbedder,
    GeminiQuotaError,
    _quota_message,
    _RateLimiter,
    _retry,
    get_limiter,
    reset_limiter,
)
from sentrysearch.embedder import (
    default_dashscope_embedding_model,
    embed_image,
    embed_query,
    embed_video_chunk,
    embed_with_retry,
    get_embedder,
    is_permanent_failure,
    reset_embedder,
    resolve_backend,
    resolve_model,
)


# ---------------------------------------------------------------------------
# _RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_allows_requests_under_limit(self):
        limiter = _RateLimiter(max_per_minute=5)
        for _ in range(5):
            limiter.wait()

    def test_tracks_request_count(self):
        limiter = _RateLimiter(max_per_minute=3)
        for _ in range(3):
            limiter.wait()
        assert len(limiter._timestamps) == 3

    @patch("sentrysearch.gemini_embedder.time.sleep")
    @patch("sentrysearch.gemini_embedder.time.monotonic")
    def test_sleeps_when_limit_reached(self, mock_monotonic, mock_sleep):
        limiter = _RateLimiter(max_per_minute=2)
        # First two at t=0 and t=1
        mock_monotonic.return_value = 0.0
        limiter.wait()
        mock_monotonic.return_value = 1.0
        limiter.wait()
        # Third at t=2: window still has 2 requests
        mock_monotonic.return_value = 2.0
        limiter.wait()
        mock_sleep.assert_called_once()
        assert mock_sleep.call_args[0][0] > 0

    def test_window_slides(self):
        limiter = _RateLimiter(max_per_minute=1)
        limiter._timestamps.append(time.monotonic() - 61)  # expired
        limiter.wait()  # should not block
        assert len(limiter._timestamps) == 1


# ---------------------------------------------------------------------------
# GeminiEmbedder construction
# ---------------------------------------------------------------------------

class TestGeminiEmbedder:
    def test_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(GeminiAPIKeyError, match="GEMINI_API_KEY"):
                GeminiEmbedder()

    @patch("google.genai.Client")
    def test_creates_client_with_key(self, mock_client_cls):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"}):
            embedder = GeminiEmbedder()
            mock_client_cls.assert_called_once_with(api_key="test-key-123")


# ---------------------------------------------------------------------------
# _retry
# ---------------------------------------------------------------------------

class TestRetry:
    def test_returns_on_first_success(self):
        fn = MagicMock(return_value="ok")
        assert _retry(fn, max_retries=3, initial_delay=0.01) == "ok"
        fn.assert_called_once()

    @patch("sentrysearch.gemini_embedder.time.sleep")
    def test_retries_on_429(self, mock_sleep):
        exc = Exception("Resource exhausted")
        exc.status_code = 429
        fn = MagicMock(side_effect=[exc, exc, "ok"])
        assert _retry(fn, max_retries=3, initial_delay=0.01) == "ok"
        assert fn.call_count == 3

    @patch("sentrysearch.gemini_embedder.time.sleep")
    def test_retries_on_503(self, mock_sleep):
        exc = Exception("Service unavailable")
        exc.status_code = 503
        fn = MagicMock(side_effect=[exc, "ok"])
        assert _retry(fn, max_retries=3, initial_delay=0.01) == "ok"

    @patch("sentrysearch.gemini_embedder.time.sleep")
    def test_raises_quota_error_after_max_retries(self, mock_sleep):
        exc = Exception("resource exhausted")
        exc.status_code = 429
        fn = MagicMock(side_effect=exc)
        with pytest.raises(GeminiQuotaError):
            _retry(fn, max_retries=2, initial_delay=0.01)

    def test_raises_non_retryable_immediately(self):
        fn = MagicMock(side_effect=ValueError("bad input"))
        with pytest.raises(ValueError, match="bad input"):
            _retry(fn, max_retries=3, initial_delay=0.01)

    @patch("sentrysearch.gemini_embedder.time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        exc = Exception("503 error")
        exc.status_code = 503
        fn = MagicMock(side_effect=[exc, exc, "ok"])
        _retry(fn, max_retries=3, initial_delay=1.0)
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert delays[0] == 1.0
        assert delays[1] == 2.0


# ---------------------------------------------------------------------------
# Embedder factory
# ---------------------------------------------------------------------------

class TestEmbedderFactory:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_embedder("nonexistent")

    @patch("google.genai.Client")
    def test_embed_query_delegates(self, mock_client_cls):
        fake_values = [0.1] * 768
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=fake_values)]
        mock_client.models.embed_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            reset_embedder()
            result = embed_query("a red car")
            assert result == fake_values
            assert len(result) == 768
            mock_client.models.embed_content.assert_called_once()

    @patch("google.genai.Client")
    def test_embed_video_chunk_delegates(self, mock_client_cls, tiny_video):
        fake_values = [0.2] * 768
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=fake_values)]
        mock_client.models.embed_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            reset_embedder()
            result = embed_video_chunk(tiny_video)
            assert result == fake_values
            assert len(result) == 768

    @patch("google.genai.Client")
    def test_embed_image_delegates(self, mock_client_cls, tmp_path):
        img_path = tmp_path / "q.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
        fake_values = [0.3] * 768
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=fake_values)]
        mock_client.models.embed_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            reset_embedder()
            result = embed_image(str(img_path))
            assert result == fake_values
            # Confirm RETRIEVAL_QUERY task type used (same space as text queries)
            call = mock_client.models.embed_content.call_args
            assert call.kwargs["config"].task_type == "RETRIEVAL_QUERY"

    def test_embed_image_missing_file_raises(self):
        with patch("google.genai.Client"), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            reset_embedder()
            with pytest.raises(FileNotFoundError):
                embed_image("/nonexistent/x.jpg")

    @patch("google.genai.Client")
    def test_get_embedder_caches_instance(self, mock_client_cls):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            reset_embedder()
            e1 = get_embedder("gemini")
            e2 = get_embedder("gemini")
            assert e1 is e2

    @patch("google.genai.Client")
    def test_reset_clears_cache(self, mock_client_cls):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            reset_embedder()
            e1 = get_embedder("gemini")
            reset_embedder()
            e2 = get_embedder("gemini")
            assert e1 is not e2

    def test_get_embedder_local_backend(self):
        with patch("sentrysearch.local_embedder.LocalEmbedder") as MockLocal:
            mock_instance = MagicMock()
            MockLocal.return_value = mock_instance
            reset_embedder()
            result = get_embedder("local", model="test-model", dimensions=512)
            MockLocal.assert_called_once_with(model_name="test-model", dimensions=512, quantize=None)
            assert result is mock_instance

    def test_get_embedder_local_defaults(self):
        with patch("sentrysearch.local_embedder.LocalEmbedder") as MockLocal:
            MockLocal.return_value = MagicMock()
            reset_embedder()
            get_embedder("local")
            MockLocal.assert_called_once_with(model_name="qwen8b", dimensions=768, quantize=None)

    def test_get_embedder_local_with_quantize(self):
        with patch("sentrysearch.local_embedder.LocalEmbedder") as MockLocal:
            MockLocal.return_value = MagicMock()
            reset_embedder()
            get_embedder("local", quantize=True)
            MockLocal.assert_called_once_with(model_name="qwen8b", dimensions=768, quantize=True)

    def test_get_embedder_local_with_quantize_false(self):
        with patch("sentrysearch.local_embedder.LocalEmbedder") as MockLocal:
            MockLocal.return_value = MagicMock()
            reset_embedder()
            get_embedder("local", quantize=False)
            MockLocal.assert_called_once_with(model_name="qwen8b", dimensions=768, quantize=False)

    def test_get_embedder_local_with_model_alias(self):
        with patch("sentrysearch.local_embedder.LocalEmbedder") as MockLocal:
            MockLocal.return_value = MagicMock()
            reset_embedder()
            get_embedder("local", model="qwen2b")
            MockLocal.assert_called_once_with(model_name="qwen2b", dimensions=768, quantize=None)

    def test_get_embedder_qwen_cloud_backend(self):
        with patch("sentrysearch.qwen_cloud_embedder.QwenCloudEmbedder") as MockQC:
            MockQC.return_value = MagicMock()
            reset_embedder()
            result = get_embedder("qwen-cloud", model="qwen3-vl-embedding")
            MockQC.assert_called_once_with(
                model_name="qwen3-vl-embedding", dimensions=768,
            )
            assert result is MockQC.return_value


# ---------------------------------------------------------------------------
# Configurable / shared RPM (issue #85)
# ---------------------------------------------------------------------------

class TestConfigurableRpm:
    def setup_method(self):
        reset_limiter()

    def teardown_method(self):
        reset_limiter()

    def test_defaults_to_module_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_limiter().max_per_minute == DEFAULT_RPM

    def test_env_var_overrides_default(self):
        with patch.dict(os.environ, {"GEMINI_RPM": "7"}):
            assert get_limiter().max_per_minute == 7

    def test_explicit_rpm_beats_env_var(self):
        with patch.dict(os.environ, {"GEMINI_RPM": "7"}):
            assert get_limiter(3).max_per_minute == 3

    @pytest.mark.parametrize("bad", ["abc", "0", "-5", "", "  "])
    def test_malformed_env_falls_back_to_default(self, bad):
        with patch.dict(os.environ, {"GEMINI_RPM": bad}):
            assert get_limiter().max_per_minute == DEFAULT_RPM

    def test_limiter_is_shared_across_calls(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_limiter() is get_limiter()

    def test_later_rpm_retunes_existing_limiter(self):
        with patch.dict(os.environ, {}, clear=True):
            first = get_limiter(10)
            second = get_limiter(4)
            assert first is second
            assert second.max_per_minute == 4

    def test_set_rate_rejects_non_positive(self):
        limiter = _RateLimiter(max_per_minute=5)
        with pytest.raises(ValueError):
            limiter.set_rate(0)

    def test_embedder_and_reranker_share_one_window(self):
        """Both bill the same project quota, so they must not run separate budgets."""
        from sentrysearch.gemini_reranker import GeminiReranker

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            with patch("google.genai.Client", MagicMock()):
                embedder = GeminiEmbedder(rpm=9)
                reranker = GeminiReranker()
        assert embedder._limiter is reranker._limiter
        assert reranker._limiter.max_per_minute == 9


class TestQuotaMessage:
    def test_per_minute_message_points_at_rpm_flag(self):
        msg = _quota_message("429 resource exhausted: quota exceeded")
        assert "--rpm" in msg
        assert "daily" not in msg.lower()

    @pytest.mark.parametrize("raw", [
        "generate_requests_per_model_per_day quota exceeded",
        "quota exceeded: requests per day",
        "GenerateRequestsPerDayPerProject limit",
    ])
    def test_per_day_message_says_throttling_will_not_help(self, raw):
        msg = _quota_message(raw.lower())
        assert "--rpm will not help" in msg
        assert "midnight Pacific" in msg


# ---------------------------------------------------------------------------
# Shared backend/model resolution — `sentrysearch index` and
# benchmarks/accuracy_eval.py must agree on what a flag pair means (issue #78)
# ---------------------------------------------------------------------------


class TestResolveBackendModel:
    def test_explicit_backend_wins(self):
        assert resolve_backend("mlx", "qwen2b", "qc-model") == "mlx"

    def test_model_implies_local(self):
        assert resolve_backend(None, "qwen2b") == "local"

    def test_dashscope_model_implies_qwen_cloud(self):
        assert resolve_backend(None, None, "qc-model") == "qwen-cloud"

    def test_defaults_to_gemini(self):
        assert resolve_backend() == "gemini"

    def test_gemini_ignores_model(self):
        assert resolve_model("gemini", "qwen8b") is None

    def test_local_normalizes_hf_id(self):
        assert resolve_model("local", "Qwen/Qwen3-VL-Embedding-2B") == "qwen2b"

    def test_local_autodetect_reports_through_echo(self):
        with patch("sentrysearch.local_embedder.detect_default_model",
                   return_value="qwen8b"):
            echoed = []
            assert resolve_model("local", echo=echoed.append) == "qwen8b"
        assert echoed == ["Auto-detected model: qwen8b"]

    def test_local_autodetect_is_silent_without_echo(self):
        with patch("sentrysearch.local_embedder.detect_default_model",
                   return_value="qwen8b"):
            assert resolve_model("local") == "qwen8b"

    def test_qwen_cloud_prefers_dashscope_flag(self):
        assert resolve_model("qwen-cloud", "ignored", "qc-model") == "qc-model"

    def test_qwen_cloud_ignores_the_model_flag(self):
        """qwen-cloud reads --dashscope-model only; generic --model is for
        local/mlx/litellm, never for DashScope (CLI parity)."""
        assert (resolve_model("qwen-cloud", "qc-model")
                == default_dashscope_embedding_model())

    def test_qwen_cloud_falls_back_to_provider_default(self):
        assert (resolve_model("qwen-cloud")
                == default_dashscope_embedding_model())

    def test_litellm_uses_model_flag(self):
        assert resolve_model("litellm", "my-model") == "my-model"

    def test_litellm_falls_back_to_default(self):
        from sentrysearch.litellm_embedder import default_litellm_model

        assert resolve_model("litellm") == default_litellm_model()

    def test_mlx_resolves_the_full_model_reference(self):
        from sentrysearch.mlx_embedder import resolve_model_ref

        assert resolve_model("mlx") == resolve_model_ref(None)


# ---------------------------------------------------------------------------
# Shared index-time retry policy (CLI + accuracy evaluation, issue #78)
# ---------------------------------------------------------------------------


class _ScriptedEmbedder:
    """Yields each scripted outcome in order (a result, or an exception)."""

    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def embed_video_chunk(self, path, verbose=False):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestEmbedWithRetry:
    @pytest.fixture(autouse=True)
    def no_backoff(self, monkeypatch):
        """Record sleeps instead of waiting out the backoff."""
        sleeps = []
        monkeypatch.setattr(time, "sleep", sleeps.append)
        return sleeps

    def test_returns_after_a_transient_failure(self, no_backoff):
        embedder = _ScriptedEmbedder(RuntimeError("429"), [0.5])
        assert embed_with_retry(embedder, "chunk.mp4") == [0.5]
        assert embedder.calls == 2
        assert no_backoff == [2]

    def test_raises_the_last_error_when_attempts_run_out(self, no_backoff):
        embedder = _ScriptedEmbedder(*[RuntimeError("boom")] * 3)
        with pytest.raises(RuntimeError, match="boom"):
            embed_with_retry(embedder, "chunk.mp4")
        assert embedder.calls == 3
        assert no_backoff == [2, 4]

    def test_on_retry_reports_each_wait(self, no_backoff):
        embedder = _ScriptedEmbedder(RuntimeError("boom"), [0.5])
        reports = []
        embed_with_retry(
            embedder, "chunk.mp4",
            on_retry=lambda attempt, exc, wait: reports.append(
                (attempt, str(exc), wait)),
        )
        assert reports == [(1, "boom", 2)]

    def test_permanent_failure_is_not_retried(self, no_backoff):
        embedder = _ScriptedEmbedder(FileNotFoundError("gone"))
        failures = []
        result = embed_with_retry(
            embedder, "chunk.mp4",
            on_failure=lambda attempt, exc: failures.append(attempt),
        )
        assert result is None
        assert embedder.calls == 1
        assert failures == [1]
        assert no_backoff == []

    def test_exhausted_retries_hand_error_to_on_failure(self, no_backoff):
        embedder = _ScriptedEmbedder(*[RuntimeError("boom")] * 3)
        failures = []
        result = embed_with_retry(
            embedder, "chunk.mp4",
            on_failure=lambda attempt, exc: failures.append(
                (attempt, str(exc))),
        )
        assert result is None
        assert failures == [(3, "boom")]
        assert no_backoff == [2, 4]

    def test_without_on_failure_the_error_propagates(self, no_backoff):
        """The accuracy evaluation must not silently drop a chunk."""
        embedder = _ScriptedEmbedder(FileNotFoundError("gone"))
        with pytest.raises(FileNotFoundError):
            embed_with_retry(embedder, "chunk.mp4")

    def test_quota_error_stops_the_run_immediately(self, no_backoff):
        embedder = _ScriptedEmbedder(GeminiQuotaError("daily quota"), [0.5])
        with pytest.raises(GeminiQuotaError):
            embed_with_retry(embedder, "chunk.mp4")
        assert embedder.calls == 1
        assert no_backoff == []

    def test_zero_max_attempts_is_rejected(self, no_backoff):
        embedder = _ScriptedEmbedder([0.5])
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            embed_with_retry(embedder, "chunk.mp4", max_attempts=0)
        assert embedder.calls == 0

    def test_negative_max_attempts_is_rejected(self, no_backoff):
        embedder = _ScriptedEmbedder([0.5])
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            embed_with_retry(embedder, "chunk.mp4", max_attempts=-3)
        assert embedder.calls == 0
        assert no_backoff == []


@pytest.mark.parametrize(
    "exc,permanent",
    [
        (FileNotFoundError("nope"), True),
        (RuntimeError("CUDA out of memory"), True),
        (RuntimeError("invalid data"), True),
        (RuntimeError("could not decode frame"), True),
        (RuntimeError("connection reset"), False),
    ],
)
def test_is_permanent_failure(exc, permanent):
    assert is_permanent_failure(exc) is permanent
