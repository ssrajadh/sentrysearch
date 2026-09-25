"""Tests for the LiteLLM embedding backend (proxy and SDK modes)."""

import base64
import sys
import types

import httpx
import pytest

from sentrysearch import litellm_embedder as le
from sentrysearch.litellm_embedder import (
    LiteLLMAPIError,
    LiteLLMConfigError,
    LiteLLMEmbedder,
    LiteLLMQuotaError,
    default_litellm_model,
)

VEC = [0.1] * 768


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "LITELLM_PROXY_API_BASE",
        "LITELLM_PROXY_API_KEY",
        "LITELLM_EMBEDDING_MODEL",
        "LITELLM_EMBEDDING_DIMENSIONS",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def chunk(tmp_path):
    p = tmp_path / "chunk.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42fake-video")
    return p


@pytest.fixture
def proxy(monkeypatch):
    """Point the backend at a fake proxy; record requests, replay responses."""
    monkeypatch.setenv("LITELLM_PROXY_API_BASE", "http://gateway.test/")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "sk-test")
    monkeypatch.setattr(le.time, "sleep", lambda s: None)
    state = {"requests": [], "responses": []}

    def fake_post(url, json=None, headers=None, timeout=None):
        state["requests"].append({"url": url, "json": json, "headers": headers})
        status, payload = state["responses"].pop(0) if state["responses"] else (
            200, {"data": [{"embedding": VEC}]})
        return httpx.Response(status, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    return state


class FakeLiteLLM(types.SimpleNamespace):
    """Stand-in for the litellm module in SDK mode."""


def _fake_litellm(raise_exc=None):
    class _Err(Exception):
        pass

    mod = FakeLiteLLM(
        calls=[],
        AuthenticationError=type("AuthenticationError", (_Err,), {}),
        UnsupportedParamsError=type("UnsupportedParamsError", (_Err,), {}),
        NotFoundError=type("NotFoundError", (_Err,), {}),
        RateLimitError=type("RateLimitError", (_Err,), {}),
    )

    def embedding(**kwargs):
        mod.calls.append(kwargs)
        if raise_exc is not None:
            raise getattr(mod, raise_exc)("boom")
        return types.SimpleNamespace(data=[{"embedding": VEC}])

    mod.embedding = embedding
    return mod


@pytest.fixture
def sdk(monkeypatch):
    mod = _fake_litellm()
    monkeypatch.setattr(le, "_import_litellm", lambda: mod)
    return mod


class TestDefaults:
    def test_default_model(self):
        assert default_litellm_model() == "gemini/gemini-embedding-2"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LITELLM_EMBEDDING_MODEL", "bedrock/amazon.nova-2-multimodal-embeddings-v1:0")
        assert default_litellm_model() == "bedrock/amazon.nova-2-multimodal-embeddings-v1:0"

    def test_gemini_defaults_to_768_dims(self, sdk):
        assert LiteLLMEmbedder().dimensions() == 768

    def test_non_gemini_keeps_native_dims(self, sdk):
        emb = LiteLLMEmbedder(model_name="bedrock/amazon.nova-2-multimodal-embeddings-v1:0")
        assert emb.dimensions() == 0
        emb.embed_query("x")
        assert emb.dimensions() == 768  # learned from the first response

    def test_env_dimensions(self, sdk, monkeypatch):
        monkeypatch.setenv("LITELLM_EMBEDDING_DIMENSIONS", "1024")
        assert LiteLLMEmbedder(model_name="bedrock/x").dimensions() == 1024


class TestProxyMode:
    def test_video_request_shape(self, proxy, chunk):
        assert LiteLLMEmbedder().embed_video_chunk(str(chunk)) == VEC
        req = proxy["requests"][0]
        assert req["url"] == "http://gateway.test/embeddings"
        assert req["headers"] == {"Authorization": "Bearer sk-test"}
        body = req["json"]
        assert body["model"] == "gemini/gemini-embedding-2"
        assert body["encoding_format"] == "float"
        assert body["dimensions"] == 768
        assert body["task_type"] == "RETRIEVAL_DOCUMENT"
        prefix = "data:video/mp4;base64,"
        assert body["input"][0].startswith(prefix)
        assert base64.b64decode(body["input"][0][len(prefix):]) == chunk.read_bytes()

    def test_query_uses_query_task(self, proxy):
        LiteLLMEmbedder().embed_query("red car")
        body = proxy["requests"][0]["json"]
        assert body["input"] == ["red car"]
        assert body["task_type"] == "RETRIEVAL_QUERY"

    def test_image_is_a_query(self, proxy, tmp_path):
        img = tmp_path / "frame.png"
        img.write_bytes(b"\x89PNG fake")
        LiteLLMEmbedder().embed_image(str(img))
        body = proxy["requests"][0]["json"]
        assert body["input"][0].startswith("data:image/png;base64,")
        assert body["task_type"] == "RETRIEVAL_QUERY"

    def test_unsupported_image_type(self, proxy, tmp_path):
        img = tmp_path / "frame.bmp"
        img.write_bytes(b"BM")
        with pytest.raises(ValueError, match="Unsupported image type"):
            LiteLLMEmbedder().embed_image(str(img))

    def test_non_gemini_omits_task_type_and_dims(self, proxy):
        LiteLLMEmbedder(model_name="nova-embed").embed_query("x")
        body = proxy["requests"][0]["json"]
        assert "task_type" not in body
        assert "dimensions" not in body

    def test_no_key_sends_no_auth_header(self, proxy, monkeypatch):
        monkeypatch.delenv("LITELLM_PROXY_API_KEY")
        LiteLLMEmbedder().embed_query("x")
        assert proxy["requests"][0]["headers"] == {}

    def test_does_not_need_litellm_installed(self, proxy, monkeypatch):
        monkeypatch.setitem(sys.modules, "litellm", None)
        assert LiteLLMEmbedder().embed_query("x") == VEC

    @pytest.mark.parametrize("status", [401, 403])
    def test_rejected_key_is_config_error(self, proxy, status):
        proxy["responses"].append((status, {"error": {"message": "bad key"}}))
        with pytest.raises(LiteLLMConfigError, match="rejected the key"):
            LiteLLMEmbedder().embed_query("x")
        assert len(proxy["requests"]) == 1

    def test_unknown_model_is_config_error(self, proxy):
        proxy["responses"].append((404, {"error": {"message": "no such model"}}))
        with pytest.raises(LiteLLMConfigError, match="--model"):
            LiteLLMEmbedder().embed_query("x")

    def test_retries_then_succeeds(self, proxy):
        proxy["responses"] += [(503, {}), (429, {})]
        assert LiteLLMEmbedder().embed_query("x") == VEC
        assert len(proxy["requests"]) == 3

    def test_rate_limit_after_retries(self, proxy):
        proxy["responses"] += [(429, {"error": {"message": "slow down"}})] * 4
        with pytest.raises(LiteLLMQuotaError, match="slow down"):
            LiteLLMEmbedder().embed_query("x")
        assert len(proxy["requests"]) == 4

    def test_bad_request_fails_the_chunk(self, proxy):
        proxy["responses"].append((400, {"error": {"message": "invalid video"}}))
        with pytest.raises(LiteLLMAPIError, match="invalid video"):
            LiteLLMEmbedder().embed_query("x")
        assert len(proxy["requests"]) == 1

    def test_transport_error_retries(self, proxy, monkeypatch):
        calls = {"n": 0}
        real = httpx.post

        def flaky(url, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("refused")
            return real(url, **kw)

        monkeypatch.setattr(httpx, "post", flaky)
        assert LiteLLMEmbedder().embed_query("x") == VEC
        assert calls["n"] == 2


class TestSdkMode:
    def test_call_shape(self, sdk, chunk):
        LiteLLMEmbedder().embed_video_chunk(str(chunk))
        call = sdk.calls[0]
        assert call["model"] == "gemini/gemini-embedding-2"
        assert call["input"][0].startswith("data:video/mp4;base64,")
        assert call["task_type"] == "RETRIEVAL_DOCUMENT"
        assert call["dimensions"] == 768
        assert "encoding_format" not in call

    def test_missing_package(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "litellm", None)
        with pytest.raises(LiteLLMConfigError, match=r"\.\[litellm\]"):
            LiteLLMEmbedder()

    def test_disables_cost_map_download(self, monkeypatch):
        monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
        monkeypatch.setitem(sys.modules, "litellm", types.ModuleType("litellm"))
        le._import_litellm()
        import os
        assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"

    @pytest.mark.parametrize("exc,expected", [
        ("AuthenticationError", LiteLLMConfigError),
        ("UnsupportedParamsError", LiteLLMConfigError),
        ("NotFoundError", LiteLLMConfigError),
        ("RateLimitError", LiteLLMQuotaError),
    ])
    def test_error_mapping(self, monkeypatch, exc, expected):
        monkeypatch.setattr(le, "_import_litellm", lambda: _fake_litellm(raise_exc=exc))
        with pytest.raises(expected):
            LiteLLMEmbedder().embed_query("x")


class TestFactory:
    def test_get_embedder_litellm(self, sdk):
        from sentrysearch.embedder import get_embedder

        emb = get_embedder("litellm", model="gemini/gemini-embedding-2", rpm=10)
        assert isinstance(emb, LiteLLMEmbedder)
        assert emb._limiter is not None


class TestStore:
    def test_collection_name(self, tmp_path):
        from sentrysearch.store import SentryStore

        store = SentryStore(db_path=tmp_path / "db", backend="litellm",
                            model="gemini/gemini-embedding-2")
        assert store.collection.name == "dashcam_chunks_litellm_gemini_gemini-embedding-2"

    def test_separate_from_gemini_backend(self, tmp_path):
        from sentrysearch.store import SentryStore

        a = SentryStore(db_path=tmp_path / "db", backend="gemini")
        b = SentryStore(db_path=tmp_path / "db", backend="litellm")
        assert a.collection.name != b.collection.name

    def test_detect_index(self, tmp_path, monkeypatch):
        from sentrysearch.store import SentryStore, detect_index

        monkeypatch.setenv("LITELLM_PROXY_API_BASE", "http://gateway.test")
        store = SentryStore(db_path=tmp_path / "db", backend="litellm", model="nova-embed")
        store.add_chunk("c1", VEC, {"source_file": "a.mp4", "start_time": 0, "end_time": 30})
        assert detect_index(tmp_path / "db") == ("litellm", "nova-embed")

    def test_backend_installed_via_proxy(self, monkeypatch):
        from sentrysearch.store import _backend_installed

        monkeypatch.setitem(sys.modules, "litellm", None)
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        assert not _backend_installed("litellm")
        monkeypatch.setenv("LITELLM_PROXY_API_BASE", "http://gateway.test")
        assert _backend_installed("litellm")


class TestCli:
    def test_rerank_refuses_to_bypass_gateway(self):
        from sentrysearch.cli import _get_search_reranker

        with pytest.raises(LiteLLMConfigError, match="--rerank"):
            _get_search_reranker("litellm", None, None)

    def test_backend_choice(self):
        from sentrysearch.cli import _BACKEND_CHOICES

        assert "litellm" in _BACKEND_CHOICES
