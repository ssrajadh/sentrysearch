"""Tests for sentrysearch.mlx_embedder (mocked — no mlx required)."""

import types
from unittest.mock import MagicMock, patch

import pytest

from sentrysearch.mlx_embedder import (
    DEFAULT_MODEL,
    MLXEmbedder,
    MLXModelError,
    MODEL_ALIASES,
    _Box,
    _video_capable_class,
    is_supported,
)


class TestModelResolution:
    def test_alias_resolves(self):
        embedder = MLXEmbedder(model_name="qwen2b")
        assert embedder._model_name == MODEL_ALIASES["qwen2b"]

    def test_hf_id_passed_through(self):
        embedder = MLXEmbedder(model_name="someone/Qwen3-VL-Embedding-8B-mlx")
        assert embedder._model_name == "someone/Qwen3-VL-Embedding-8B-mlx"

    def test_local_path_passed_through(self):
        embedder = MLXEmbedder(model_name="/models/qwen-8b-4bit")
        assert embedder._model_name == "/models/qwen-8b-4bit"

    def test_env_var_used_when_no_model_given(self, monkeypatch):
        monkeypatch.setenv("SENTRYSEARCH_MLX_MODEL", "/models/from-env")
        assert MLXEmbedder()._model_name == "/models/from-env"

    def test_explicit_model_beats_env_var(self, monkeypatch):
        monkeypatch.setenv("SENTRYSEARCH_MLX_MODEL", "/models/from-env")
        assert MLXEmbedder(model_name="/explicit")._model_name == "/explicit"

    def test_defaults_to_the_2b_4bit_build(self, monkeypatch):
        """--backend mlx should work with no --model. 4-bit 2B is the
        measured-best configuration, so it is the default."""
        monkeypatch.delenv("SENTRYSEARCH_MLX_MODEL", raising=False)
        assert MLXEmbedder()._model_name == DEFAULT_MODEL
        assert "2B" in DEFAULT_MODEL and "4bit" in DEFAULT_MODEL

    def test_no_8b_alias_is_offered(self):
        """The 8B ties the 2B at 3x the cost; shipping an alias would
        invite users into the worse configuration."""
        assert not any("8b" in k.lower() or "8B" in v for k, v in MODEL_ALIASES.items())


class TestMLXModelError:
    def test_is_runtime_error(self):
        assert issubclass(MLXModelError, RuntimeError)


class TestIsSupported:
    def test_true_on_apple_silicon(self):
        with patch("platform.system", return_value="Darwin"), \
             patch("platform.machine", return_value="arm64"):
            assert is_supported()

    def test_false_on_intel_mac(self):
        with patch("platform.system", return_value="Darwin"), \
             patch("platform.machine", return_value="x86_64"):
            assert not is_supported()

    def test_false_on_linux(self):
        with patch("platform.system", return_value="Linux"), \
             patch("platform.machine", return_value="aarch64"):
            assert not is_supported()


class _StubBase:
    """Stands in for mlx-vlm's embedding Model, which drops video pixels."""

    def __init__(self):
        self.language_model = types.SimpleNamespace(
            _rope_deltas="stale", _position_ids="stale")
        self.seen_call = None
        self.seen_embeddings = None

    def __call__(self, input_ids=None, **kwargs):
        self.seen_call = kwargs
        # Upstream drops pixel_values_videos here: it is never forwarded.
        return self.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=kwargs.get("pixel_values"),
            image_grid_thw=kwargs.get("image_grid_thw"),
            video_grid_thw=kwargs.get("video_grid_thw"),
        )

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        self.seen_embeddings = kwargs
        return "embeddings"


def _make_model():
    cls = _video_capable_class(_StubBase)
    model = cls.__new__(cls)
    _StubBase.__init__(model)
    return model


class TestVideoForwarding:
    """The shim exists because upstream Model.__call__ omits
    pixel_values_videos, so video pixels never reach the vision tower."""

    def test_video_pixels_reach_get_input_embeddings(self):
        model = _make_model()
        model(input_ids="ids", pixel_values_videos="PIXELS",
              video_grid_thw="grid")
        assert model.seen_embeddings["pixel_values_videos"] == "PIXELS"

    def test_stub_base_alone_drops_them(self):
        """Guard the guard: the stub must reproduce the upstream bug, or the
        test above would pass even with the shim removed."""
        base = _StubBase()
        base(input_ids="ids", pixel_values_videos="PIXELS")
        assert "pixel_values_videos" not in base.seen_embeddings

    def test_rope_state_reset_for_video_only_call(self):
        model = _make_model()
        model(input_ids="ids", pixel_values_videos="PIXELS")
        assert model.language_model._rope_deltas is None
        assert model.language_model._position_ids is None

    def test_rope_state_untouched_without_pixels(self):
        model = _make_model()
        model(input_ids="ids")
        assert model.language_model._rope_deltas == "stale"

    def test_grid_still_forwarded(self):
        model = _make_model()
        model(input_ids="ids", pixel_values_videos="PIXELS",
              video_grid_thw="grid")
        assert model.seen_call["video_grid_thw"] == "grid"

    def test_pending_cleared_after_call(self):
        """A stale stash would leak one clip's pixels into the next call."""
        model = _make_model()
        model(input_ids="ids", pixel_values_videos="PIXELS")
        model(input_ids="ids")
        assert "pixel_values_videos" not in model.seen_embeddings

    def test_pending_cleared_when_call_raises(self):
        class Boom(_StubBase):
            def __call__(self, input_ids=None, **kwargs):
                raise ValueError("boom")

        cls = _video_capable_class(Boom)
        model = cls.__new__(cls)
        Boom.__init__(model)
        with pytest.raises(ValueError):
            model(input_ids="ids", pixel_values_videos="PIXELS")
        assert model._pending_video.value is None

    def test_image_path_unaffected(self):
        model = _make_model()
        model(input_ids="ids", pixel_values="IMG", image_grid_thw="grid")
        assert model.seen_call["pixel_values"] == "IMG"
        assert "pixel_values_videos" not in model.seen_embeddings


class TestBox:
    """mlx.nn.Module.__setattr__ registers any mx.array/dict/list/tuple
    assigned to an attribute as a model parameter. _Box must not be one of
    those types, or stashing pixels would corrupt model.parameters()."""

    def test_box_is_not_a_registered_type(self):
        box = _Box("anything")
        assert not isinstance(box, (dict, list, tuple))

    def test_box_round_trips(self):
        assert _Box("v").value == "v"
        assert _Box(None).value is None


class TestLooksLikePath:
    @pytest.mark.parametrize("name", [
        "/models/x", "./x", "../x", "~/models/x", "x", "a/b/c", "a/b/",
    ])
    def test_treated_as_path(self, name):
        from sentrysearch.mlx_embedder import _looks_like_path
        assert _looks_like_path(name)

    @pytest.mark.parametrize("name", [
        "mlx-community/Qwen3-VL-Embedding-8B-mlx",
        "arthurcollet/Qwen3-VL-Embedding-2B-mlx-4bit",
    ])
    def test_treated_as_hub_id(self, name):
        from sentrysearch.mlx_embedder import _looks_like_path
        assert not _looks_like_path(name)


class TestDimensions:
    def test_default_is_768(self):
        assert MLXEmbedder(model_name="/m").dimensions() == 768

    def test_explicit_dimensions(self):
        assert MLXEmbedder(model_name="/m", dimensions=1024).dimensions() == 1024


class TestGuards:
    def test_missing_chunk_file_raises(self, tmp_path):
        embedder = MLXEmbedder(model_name="/m")
        with patch.object(MLXEmbedder, "_load_model"):
            with pytest.raises(MLXModelError, match="Chunk file not found"):
                embedder.embed_video_chunk(str(tmp_path / "nope.mp4"))

    def test_missing_image_file_raises(self, tmp_path):
        embedder = MLXEmbedder(model_name="/m")
        with patch.object(MLXEmbedder, "_load_model"):
            with pytest.raises(MLXModelError, match="Image file not found"):
                embedder.embed_image(str(tmp_path / "nope.jpg"))

    def test_missing_local_dir_is_not_sent_to_the_hub(self, tmp_path):
        """A mistyped path used to surface as 'Repo id must be in the form
        namespace/repo_name', which points nowhere near the cause."""
        missing = tmp_path / "not-converted-yet"
        embedder = MLXEmbedder(model_name=str(missing))
        with patch("sentrysearch.mlx_embedder.is_supported", return_value=True):
            with pytest.raises(MLXModelError, match="Model directory not found"):
                embedder._load_model()

    def test_non_apple_silicon_refuses_to_load(self):
        embedder = MLXEmbedder(model_name="/m")
        with patch("sentrysearch.mlx_embedder.is_supported", return_value=False):
            with pytest.raises(MLXModelError, match="Apple Silicon"):
                embedder._load_model()


class TestFactory:
    def test_factory_builds_mlx_backend(self):
        from sentrysearch import embedder as embedder_mod

        embedder_mod.reset_embedder()
        try:
            built = embedder_mod.get_embedder("mlx", model="/models/x")
            assert isinstance(built, MLXEmbedder)
            assert built._model_name == "/models/x"
        finally:
            embedder_mod.reset_embedder()


class TestRerankGuard:
    def test_rerank_refuses_rather_than_falling_back_to_gemini(self):
        """A user on --backend mlx chose a local backend; silently sending
        their results to an API would be a surprise."""
        from sentrysearch.cli import _get_search_reranker

        with pytest.raises(MLXModelError, match="not supported on the MLX"):
            _get_search_reranker("mlx", None, None)

    def test_local_backend_still_gets_the_qwen_reranker(self):
        from sentrysearch.cli import _get_search_reranker

        with patch("sentrysearch.qwen_reranker.QwenReranker") as qr:
            qr.return_value.load.return_value = "reranker"
            assert _get_search_reranker("local", "qwen2b", None) == "reranker"


class TestCollectionNaming:
    def test_mlx_gets_its_own_collection(self):
        from sentrysearch.store import _collection_name

        assert _collection_name("mlx", "/models/Qwen3-VL-Embedding-8B-mlx-4bit") \
            == "dashcam_chunks_mlx_Qwen3-VL-Embedding-8B-mlx-4bit"

    def test_mlx_does_not_collide_with_local(self):
        from sentrysearch.store import _collection_name

        assert _collection_name("mlx", "qwen8b") != _collection_name("local", "qwen8b")

    def test_path_reduced_to_basename(self):
        from sentrysearch.store import _collection_name

        assert _collection_name("mlx", "/a/b/c/model-x") \
            == _collection_name("mlx", "model-x")
