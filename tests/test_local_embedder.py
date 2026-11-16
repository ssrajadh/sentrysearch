"""Tests for sentrysearch.local_embedder (mocked — no torch required)."""

from unittest.mock import MagicMock, patch

import pytest

from sentrysearch.local_embedder import (
    LocalEmbedder, LocalModelError, MODEL_ALIASES,
    normalize_model_key, detect_default_model, _cpu_fallback_warning,
)


class TestModelAliases:
    def test_qwen8b_alias_resolves(self):
        embedder = LocalEmbedder(model_name="qwen8b")
        assert embedder._model_name == "Qwen/Qwen3-VL-Embedding-8B"

    def test_qwen2b_alias_resolves(self):
        embedder = LocalEmbedder(model_name="qwen2b")
        assert embedder._model_name == "Qwen/Qwen3-VL-Embedding-2B"

    def test_full_hf_id_passed_through(self):
        embedder = LocalEmbedder(model_name="Qwen/Qwen3-VL-Embedding-8B")
        assert embedder._model_name == "Qwen/Qwen3-VL-Embedding-8B"

    def test_custom_model_name_passed_through(self):
        embedder = LocalEmbedder(model_name="custom/my-model")
        assert embedder._model_name == "custom/my-model"

    def test_aliases_dict_has_expected_keys(self):
        assert "qwen8b" in MODEL_ALIASES
        assert "qwen2b" in MODEL_ALIASES


class TestLocalModelError:
    def test_is_runtime_error(self):
        assert issubclass(LocalModelError, RuntimeError)

    def test_message(self):
        err = LocalModelError("missing torch")
        assert "missing torch" in str(err)


class TestLocalEmbedderConstruction:
    def test_default_params(self):
        embedder = LocalEmbedder()
        assert embedder._model_name == "Qwen/Qwen3-VL-Embedding-8B"  # resolved from "qwen8b"
        assert embedder._dimensions == 768
        assert embedder._model is None

    def test_custom_params(self):
        embedder = LocalEmbedder(model_name="custom/model", dimensions=512)
        assert embedder._model_name == "custom/model"
        assert embedder._dimensions == 512

    def test_dimensions_method(self):
        embedder = LocalEmbedder(dimensions=1024)
        assert embedder.dimensions() == 1024

    def test_quantize_none_by_default(self):
        embedder = LocalEmbedder()
        assert embedder._quantize is None

    def test_quantize_true(self):
        embedder = LocalEmbedder(quantize=True)
        assert embedder._quantize is True

    def test_quantize_false(self):
        embedder = LocalEmbedder(quantize=False)
        assert embedder._quantize is False


class TestLocalEmbedderLoadModel:
    def test_missing_torch_raises_local_model_error(self):
        embedder = LocalEmbedder()
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(LocalModelError, match="Missing dependencies"):
                embedder._load_model()

    def test_load_model_called_once(self):
        embedder = LocalEmbedder()
        embedder._model = MagicMock()  # pretend already loaded
        # Should return immediately without reloading
        embedder._load_model()


class TestCpuFallbackWarning:
    def test_intel_mac_points_to_gemini_backend(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("platform.machine", lambda: "x86_64")

        warning = _cpu_fallback_warning()

        assert "Intel Mac" in warning
        assert "--backend gemini" in warning

    def test_linux_cpu_only_uses_general_warning(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        monkeypatch.setattr("platform.machine", lambda: "x86_64")

        warning = _cpu_fallback_warning()

        assert warning == "Warning: No GPU detected, local inference will be very slow."
        assert "--backend gemini" not in warning

    def test_apple_silicon_cpu_only_uses_general_warning(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("platform.machine", lambda: "arm64")

        warning = _cpu_fallback_warning()

        assert warning == "Warning: No GPU detected, local inference will be very slow."
        assert "--backend gemini" not in warning


class TestNormalizeModelKey:
    def test_alias_returned_as_is(self):
        assert normalize_model_key("qwen8b") == "qwen8b"
        assert normalize_model_key("qwen2b") == "qwen2b"

    def test_full_hf_id_reversed_to_alias(self):
        assert normalize_model_key("Qwen/Qwen3-VL-Embedding-8B") == "qwen8b"
        assert normalize_model_key("Qwen/Qwen3-VL-Embedding-2B") == "qwen2b"

    def test_custom_model_sanitized(self):
        assert normalize_model_key("org/my-custom-model") == "org_my_custom_model"


class TestDetectDefaultModel:
    def test_no_torch_returns_qwen8b(self):
        with patch.dict("sys.modules", {"torch": None}):
            # ImportError path
            result = detect_default_model()
            assert result == "qwen8b"

    def test_cuda_returns_qwen8b(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = detect_default_model()
            assert result == "qwen8b"

    def test_cpu_only_returns_qwen2b(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = detect_default_model()
            assert result == "qwen2b"


class TestLocalEmbedderMethods:
    def test_embed_query_calls_load_model(self):
        embedder = LocalEmbedder()
        embedder._load_model = MagicMock(
            side_effect=LocalModelError("no torch in CI")
        )
        with pytest.raises(LocalModelError):
            embedder.embed_query("test query")
        embedder._load_model.assert_called_once()

    def test_embed_video_chunk_calls_load_model(self):
        embedder = LocalEmbedder()
        embedder._load_model = MagicMock(
            side_effect=LocalModelError("no torch in CI")
        )
        with pytest.raises(LocalModelError):
            embedder.embed_video_chunk("/fake/path.mp4")
        embedder._load_model.assert_called_once()


class _StopAfterChatTemplate(Exception):
    pass


def _media_path_sent_to_processor(method_name, media_path, key):
    """Run an embed method up to the chat template and return the path it passes on."""
    import sys
    import types

    embedder = LocalEmbedder()
    embedder._load_model = MagicMock()
    embedder._processor = MagicMock()

    torch_module = types.ModuleType("torch")
    torch_module.nn = types.ModuleType("torch.nn")
    torch_module.nn.functional = types.ModuleType("torch.nn.functional")
    qwen_utils = types.ModuleType("qwen_vl_utils")
    qwen_utils.process_vision_info = MagicMock(side_effect=_StopAfterChatTemplate)

    modules = {
        "torch": torch_module,
        "torch.nn": torch_module.nn,
        "torch.nn.functional": torch_module.nn.functional,
        "qwen_vl_utils": qwen_utils,
    }
    with patch.dict(sys.modules, modules):
        with pytest.raises(_StopAfterChatTemplate):
            getattr(embedder, method_name)(str(media_path))

    conversation = embedder._processor.apply_chat_template.call_args.args[0]
    return conversation[-1]["content"][0][key]


class TestMediaPathsAreUsableOnEveryPlatform:
    """Local paths go to qwen_vl_utils as-is; "file://" + a Windows path is not a valid URI."""

    def test_video_chunk_path_has_no_file_prefix(self, tmp_path):
        chunk = tmp_path / "chunk.mp4"
        chunk.write_bytes(b"x")
        sent = _media_path_sent_to_processor("embed_video_chunk", chunk, "video")
        assert sent == str(chunk.resolve())

    def test_image_path_has_no_file_prefix(self, tmp_path):
        image = tmp_path / "query.png"
        image.write_bytes(b"x")
        sent = _media_path_sent_to_processor("embed_image", image, "image")
        assert sent == str(image.resolve())
