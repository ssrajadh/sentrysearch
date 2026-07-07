"""Local Qwen3-VL Instruct reranker."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from .local_embedder import LocalModelError, _cpu_fallback_warning
from .reranker import RerankScore, build_rerank_prompt, parse_rerank_response


RERANK_MODEL_ALIASES: dict[str, str] = {
    "qwen8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen2b": "Qwen/Qwen3-VL-2B-Instruct",
}
MAX_NEW_TOKENS = 128
QWEN3_IMAGE_PATCH_SIZE = 16


class QwenReranker:
    """Local Qwen3-VL Instruct reranker for candidate video clips."""

    def __init__(
        self,
        model_name: str = "qwen8b",
        *,
        quantize: bool | None = None,
    ):
        if model_name not in RERANK_MODEL_ALIASES:
            raise LocalModelError(
                "Local rerank supports qwen2b or qwen8b. "
                f"Got indexed model: {model_name}"
            )
        self._model_name = RERANK_MODEL_ALIASES[model_name]
        self._quantize = quantize
        self._model = None
        self._processor = None
        self._process_vision_info = None

    def load(self):
        self._load_model()
        return self

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLForConditionalGeneration,
            )
            from transformers.models.qwen3_vl.processing_qwen3_vl import (
                Qwen3VLProcessor,
            )
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise LocalModelError(
                f"Missing dependencies for local Qwen rerank: {e}\n\n"
                "Install with: uv tool install \".[local]\"\n"
                "For 4-bit quantization: uv tool install \".[local-quantized]\""
            ) from e

        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(self._model_name, "config.json")
            is_cached = isinstance(cached, str) and os.path.exists(cached)
        except Exception:
            is_cached = False

        if is_cached:
            print(f"Loading {self._model_name}...", file=sys.stderr)
        else:
            print(
                f"Downloading {self._model_name} (this only happens once)...",
                file=sys.stderr,
            )

        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
            print(_cpu_fallback_warning(), file=sys.stderr)

        quantization_config = None
        want_quantize = self._quantize
        if want_quantize is None and device == "cuda":
            props = torch.cuda.get_device_properties(0)
            total_mem = getattr(props, "total_memory", None) or getattr(
                props, "total_mem", 0,
            )
            vram_gb = total_mem / (1024 ** 3)
            needs_gb = 18 if "8B" in self._model_name else 6
            want_quantize = vram_gb < needs_gb
        if want_quantize and device == "cuda":
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                print("Using 4-bit quantization (bitsandbytes)", file=sys.stderr)
            except ImportError as e:
                if self._quantize is True:
                    raise LocalModelError(
                        "4-bit quantization requested but bitsandbytes is not "
                        "installed.\n\n"
                        "Install with: uv tool install \".[local-quantized]\""
                    ) from e
        elif want_quantize and device != "cuda":
            if self._quantize is True:
                raise LocalModelError(
                    "4-bit quantization requires CUDA (NVIDIA GPU). "
                    f"Current device: {device}"
                )

        try:
            self._processor = Qwen3VLProcessor.from_pretrained(
                self._model_name, padding_side="left",
            )
            self._process_vision_info = process_vision_info

            load_kwargs = dict(trust_remote_code=True)
            if device == "mps":
                load_kwargs["attn_implementation"] = "eager"
                os.environ.setdefault("TRANSFORMERS_DISABLE_TORCH_CHECK", "1")
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
            else:
                load_kwargs["torch_dtype"] = dtype

            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self._model_name, **load_kwargs,
            )
            if quantization_config is None:
                self._model = self._model.to(device)
            self._model.eval()
            print(f"Rerank model loaded on {device}", file=sys.stderr)
        except Exception as e:
            raise LocalModelError(
                f"Failed to load {self._model_name}: {e}"
            ) from e

    def _model_device(self):
        device = getattr(self._model, "device", None)
        if device is not None:
            return device
        try:
            return next(self._model.parameters()).device
        except Exception:
            return None

    @staticmethod
    def _trim_generated_ids(input_ids, generated_ids):
        return [
            output_ids[len(input_ids_row):]
            for input_ids_row, output_ids in zip(input_ids, generated_ids)
        ]

    def score(
        self,
        query: str,
        clip_path: str,
        *,
        verbose: bool = False,
    ) -> RerankScore | None:
        """Return a validated rerank score, or None for unparsable model output."""
        self._load_model()

        import torch
        process_vision_info = self._process_vision_info
        if process_vision_info is None:
            from qwen_vl_utils import process_vision_info

        clip = Path(clip_path)
        if not clip.exists():
            raise LocalModelError(f"Clip file not found: {clip}")

        t0 = time.monotonic()
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "file://" + str(clip.resolve()),
                        "fps": 1.0,
                        "max_frames": 32,
                    },
                    {"type": "text", "text": build_rerank_prompt(query)},
                ],
            },
        ]
        text = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )

        images, video_inputs, video_kwargs = process_vision_info(
            conversation,
            image_patch_size=QWEN3_IMAGE_PATCH_SIZE,
            return_video_metadata=True,
            return_video_kwargs=True,
        )
        video_kwargs = video_kwargs or {}

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        inputs = self._processor(
            text=[text],
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            padding=True,
            **video_kwargs,
        )

        device = self._model_device()
        if device is not None:
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        generated_ids = self._trim_generated_ids(
            inputs["input_ids"], generated_ids,
        )
        output_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        elapsed = time.monotonic() - t0
        score = parse_rerank_response(output_text)
        if verbose:
            status = "fallback" if score is None else (
                f"match={score.rerank_match}, "
                f"confidence={score.rerank_confidence:.2f}"
            )
            print(
                f"  [verbose] rerank {self._model_name}: {status}, "
                f"inference_time={elapsed:.2f}s",
                file=sys.stderr,
            )
        return score
