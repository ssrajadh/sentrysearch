"""Tests for local Qwen3-VL reranker (mocked, no model download)."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from sentrysearch.local_embedder import LocalModelError
from sentrysearch.qwen_reranker import (
    MAX_NEW_TOKENS,
    QwenReranker,
    QWEN3_IMAGE_PATCH_SIZE,
    RERANK_MODEL_ALIASES,
)
from sentrysearch.reranker import RerankScore


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class TestQwenReranker:
    def test_aliases_resolve_to_instruct_models(self):
        reranker = QwenReranker("qwen2b")
        assert reranker._model_name == "Qwen/Qwen3-VL-2B-Instruct"
        assert RERANK_MODEL_ALIASES["qwen8b"] == (
            "Qwen/Qwen3-VL-8B-Instruct"
        )

    def test_rejects_custom_model_for_v1(self):
        with pytest.raises(LocalModelError, match="qwen2b or qwen8b"):
            QwenReranker("custom_model")

    def test_load_missing_dependencies_raises_local_model_error(self):
        reranker = QwenReranker("qwen2b")
        with patch.dict(sys.modules, {"torch": None}):
            with pytest.raises(LocalModelError, match="Missing dependencies"):
                reranker.load()

    def test_load_model_wires_processor_model_and_vision_helper(self):
        reranker = QwenReranker("qwen2b")
        processor = MagicMock()
        model = MagicMock()
        model.to.return_value = model

        processor_cls = MagicMock()
        processor_cls.from_pretrained.return_value = processor
        model_cls = MagicMock()
        model_cls.from_pretrained.return_value = model
        process_vision_info = MagicMock()

        torch_module = types.ModuleType("torch")
        torch_module.float16 = "float16"
        torch_module.float32 = "float32"
        torch_module.bfloat16 = "bfloat16"
        torch_module.cuda = MagicMock()
        torch_module.cuda.is_available.return_value = False
        torch_module.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False),
        )

        modeling_module = types.ModuleType(
            "transformers.models.qwen3_vl.modeling_qwen3_vl",
        )
        modeling_module.Qwen3VLForConditionalGeneration = model_cls
        processing_module = types.ModuleType(
            "transformers.models.qwen3_vl.processing_qwen3_vl",
        )
        processing_module.Qwen3VLProcessor = processor_cls
        qwen_utils = types.ModuleType("qwen_vl_utils")
        qwen_utils.process_vision_info = process_vision_info
        huggingface_hub = types.ModuleType("huggingface_hub")
        huggingface_hub.try_to_load_from_cache = MagicMock(return_value=None)

        with patch.dict(sys.modules, {
            "torch": torch_module,
            "transformers": types.ModuleType("transformers"),
            "transformers.models": types.ModuleType("transformers.models"),
            "transformers.models.qwen3_vl": types.ModuleType(
                "transformers.models.qwen3_vl",
            ),
            "transformers.models.qwen3_vl.modeling_qwen3_vl":
                modeling_module,
            "transformers.models.qwen3_vl.processing_qwen3_vl":
                processing_module,
            "qwen_vl_utils": qwen_utils,
            "huggingface_hub": huggingface_hub,
        }):
            assert reranker.load() is reranker

        processor_cls.from_pretrained.assert_called_once_with(
            "Qwen/Qwen3-VL-2B-Instruct", padding_side="left",
        )
        model_cls.from_pretrained.assert_called_once_with(
            "Qwen/Qwen3-VL-2B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch_module.float32,
        )
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()
        assert reranker._processor is processor
        assert reranker._model is model
        assert reranker._process_vision_info is process_vision_info

    def test_score_generates_decodes_and_parses(self, tmp_path):
        clip = tmp_path / "candidate.mp4"
        clip.write_bytes(b"fake-video")

        reranker = QwenReranker("qwen2b")
        model = MagicMock()
        model.device = "cpu"
        model.generate.return_value = [[10, 11, 12, 13]]
        processor = MagicMock()
        processor.apply_chat_template.return_value = "chat prompt"
        processor.return_value = {
            "input_ids": [[10, 11]],
            "attention_mask": [[1, 1]],
        }
        processor.batch_decode.return_value = [
            '```json\n{"rerank_match": true, "rerank_confidence": 0.82}\n```',
        ]
        reranker._model = model
        reranker._processor = processor

        torch_module = types.ModuleType("torch")
        torch_module.no_grad = lambda: _NoGrad()

        qwen_utils = types.ModuleType("qwen_vl_utils")
        qwen_utils.process_vision_info = MagicMock(
            return_value=(None, [("video-data", {"fps": 1.0})], {"fps": 1.0}),
        )

        with patch.dict(
            sys.modules,
            {"torch": torch_module, "qwen_vl_utils": qwen_utils},
        ):
            score = reranker.score("red truck", str(clip), verbose=True)

        assert score == RerankScore(True, 0.82)
        qwen_utils.process_vision_info.assert_called_once()
        assert qwen_utils.process_vision_info.call_args.kwargs == {
            "image_patch_size": QWEN3_IMAGE_PATCH_SIZE,
            "return_video_metadata": True,
            "return_video_kwargs": True,
        }
        processor.apply_chat_template.assert_called_once()
        processor.assert_called_once()
        model.generate.assert_called_once()
        assert model.generate.call_args.kwargs["do_sample"] is False
        assert model.generate.call_args.kwargs["max_new_tokens"] == MAX_NEW_TOKENS
        processor.batch_decode.assert_called_once_with(
            [[12, 13]],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
