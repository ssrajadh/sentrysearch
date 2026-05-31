"""Local video/text embedding using Qwen3-VL-Embedding.

Uses the Qwen3VLForEmbedding model class adapted from the official
Qwen3-VL-Embedding repository (Apache 2.0 license).
Source: https://github.com/QwenLM/Qwen3-VL-Embedding
"""

import os
import platform
import sys
import time

from .base_embedder import BaseEmbedder


class LocalModelError(RuntimeError):
    """Raised when the local model fails to load or run."""


# Short aliases → full HuggingFace model IDs
MODEL_ALIASES: dict[str, str] = {
    "qwen8b": "Qwen/Qwen3-VL-Embedding-8B",
    "qwen2b": "Qwen/Qwen3-VL-Embedding-2B",
}

# Reverse lookup: full HuggingFace ID → short alias
_REVERSE_ALIASES: dict[str, str] = {v: k for k, v in MODEL_ALIASES.items()}


def normalize_model_key(model: str) -> str:
    """Return canonical short key for a model (used in collection names)."""
    if model in MODEL_ALIASES:
        return model
    if model in _REVERSE_ALIASES:
        return _REVERSE_ALIASES[model]
    # Custom model: sanitize for use as collection name suffix
    return model.replace("/", "_").replace("-", "_").lower()


def detect_default_model() -> str:
    """Pick the best default local model based on available hardware.

    Returns 'qwen8b' for NVIDIA GPUs and high-memory Apple Silicon Macs,
    'qwen2b' for lower-memory Macs and CPU-only systems.
    """
    try:
        import torch
    except ImportError:
        return "qwen8b"

    if torch.cuda.is_available():
        return "qwen8b"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon unified memory — 8B needs ~16 GB in float16
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            mem_gb = int(result.stdout.strip()) / (1024 ** 3)
            return "qwen8b" if mem_gb >= 24 else "qwen2b"
        except Exception:
            return "qwen2b"

    # CPU-only — 2B is more practical
    return "qwen2b"


def _cpu_fallback_warning() -> str:
    if platform.system() == "Darwin" and platform.machine() == "x86_64":
        return (
            "Warning: Intel Macs cannot use Metal acceleration for the "
            "local backend. CPU inference will be impractically slow; "
            "use --backend gemini instead."
        )
    return "Warning: No GPU detected, local inference will be very slow."


class LocalEmbedder(BaseEmbedder):
    """Qwen3-VL-Embedding backend (local GPU inference)."""

    def __init__(
        self,
        model_name: str = "qwen8b",
        dimensions: int = 768,
        quantize: bool | None = None,
    ):
        self._model_name = MODEL_ALIASES.get(model_name, model_name)
        self._dimensions = dimensions
        self._quantize = quantize  # None = auto-detect
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return

        try:
            import torch
            import torch.nn.functional as F  # noqa: F401
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLPreTrainedModel,
                Qwen3VLModel,
                Qwen3VLConfig,
            )
            from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
            from transformers.cache_utils import Cache
            from transformers.utils import TransformersKwargs
            from transformers.processing_utils import Unpack
        except ImportError as e:
            raise LocalModelError(
                f"Missing dependencies for local backend: {e}\n\n"
                "Install with: uv tool install \".[local]\"\n"
                "For 4-bit quantization: uv tool install \".[local-quantized]\""
            ) from e

        # Check if model is already cached locally
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(self._model_name, "config.json")
            is_cached = cached is not None and not isinstance(cached, str) or (isinstance(cached, str) and os.path.exists(cached))
        except Exception:
            is_cached = False

        if is_cached:
            print(f"Loading {self._model_name}...", file=sys.stderr)
        else:
            print(
                f"Downloading {self._model_name} (this only happens once)...",
                file=sys.stderr,
            )

        # Determine device and dtype
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

        # 4-bit quantization: explicit flag or auto-detect
        quantization_config = None
        want_quantize = self._quantize
        if want_quantize is None and device == "cuda":
            # Auto: only quantize when VRAM is tight for the chosen model
            props = torch.cuda.get_device_properties(0)
            # Attribute renamed total_mem → total_memory in recent PyTorch
            total_mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            vram_gb = total_mem / (1024 ** 3)
            # 8B needs ~16 GB in bf16, 2B needs ~4 GB — add headroom
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
            except ImportError:
                if self._quantize is True:
                    raise LocalModelError(
                        "4-bit quantization requested but bitsandbytes is not installed.\n\n"
                        "Install with: uv tool install \".[local-quantized]\""
                    )
        elif want_quantize and device != "cuda":
            if self._quantize is True:
                raise LocalModelError(
                    "4-bit quantization requires CUDA (NVIDIA GPU). "
                    f"Current device: {device}"
                )

        # Build the model class dynamically to avoid top-level imports
        _PreTrained = Qwen3VLPreTrainedModel
        _Model = Qwen3VLModel
        _Config = Qwen3VLConfig
        _Cache = Cache
        _TransformersKwargs = TransformersKwargs
        _Unpack = Unpack

        class _Qwen3VLForEmbedding(_PreTrained):
            """Qwen3-VL wrapper that exposes last_hidden_state for pooling."""
            config: _Config

            def __init__(self, config):
                super().__init__(config)
                self.model = _Model(config)
                self.post_init()

            def get_input_embeddings(self):
                return self.model.get_input_embeddings()

            def set_input_embeddings(self, value):
                self.model.set_input_embeddings(value)

            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                pixel_values=None,
                pixel_values_videos=None,
                image_grid_thw=None,
                video_grid_thw=None,
                cache_position=None,
                **kwargs,
            ):
                return self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    cache_position=cache_position,
                    **kwargs,
                )

        try:
            self._processor = Qwen3VLProcessor.from_pretrained(
                self._model_name, padding_side="right",
            )

            load_kwargs = dict(trust_remote_code=True)
            if device == "mps":
                # MPS SDPA crashes on grouped-query attention (mismatched
                # head counts in batched matmul).  Use eager attention.
                load_kwargs["attn_implementation"] = "eager"
                # transformers >=5.3 has a spurious assertion that reports
                # "Video features and video tokens do not match" even when
                # the counts are identical.  Disable the check on MPS.
                os.environ.setdefault("TRANSFORMERS_DISABLE_TORCH_CHECK", "1")
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
            else:
                load_kwargs["torch_dtype"] = dtype

            self._model = _Qwen3VLForEmbedding.from_pretrained(
                self._model_name, **load_kwargs,
            )
            if quantization_config is None:
                self._model = self._model.to(device)
            self._model.eval()
            print(f"Model loaded on {device}", file=sys.stderr)

        except Exception as e:
            raise LocalModelError(f"Failed to load {self._model_name}: {e}") from e

    @staticmethod
    def _pooling_last(hidden_state, attention_mask):
        """Pool at the last non-padded token position."""
        import torch
        flipped = attention_mask.flip(dims=[1])
        last_pos = flipped.argmax(dim=1)
        col = attention_mask.shape[1] - last_pos - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    @staticmethod
    def _truncate_and_normalize(embedding, target_dims):
        """MRL dimension truncation: slice first N dims, then L2-normalize."""
        import torch
        import torch.nn.functional as F
        truncated = embedding[:target_dims]
        norm = torch.linalg.norm(truncated)
        if norm > 0:
            truncated = truncated / norm
        return truncated.cpu().float().tolist()

    def embed_video_chunk(self, chunk_path: str, verbose: bool = False) -> list[float]:
        self._load_model()

        import torch
        import torch.nn.functional as F
        from pathlib import Path

        from qwen_vl_utils import process_vision_info

        chunk_path = Path(chunk_path)
        if not chunk_path.exists():
            raise LocalModelError(f"Chunk file not found: {chunk_path}")

        if verbose:
            size_kb = os.path.getsize(chunk_path) / 1024
            print(f"    [verbose] embedding {size_kb:.0f}KB chunk locally", file=sys.stderr)

        t0 = time.monotonic()

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Represent the video for retrieval."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "file://" + str(chunk_path.resolve()),
                        "fps": 1.0,
                        "max_frames": 32,
                    },
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )

        images, video_inputs, video_kwargs = process_vision_info(
            conversation,
            return_video_metadata=True,
            return_video_kwargs=True,
        )

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
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = self._pooling_last(
                outputs.last_hidden_state, inputs["attention_mask"],
            )
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        result = self._truncate_and_normalize(embeddings[0], self._dimensions)
        elapsed = time.monotonic() - t0

        if verbose:
            print(
                f"    [verbose] dims={len(result)}, "
                f"inference_time={elapsed:.2f}s",
                file=sys.stderr,
            )

        return result

    def embed_query(self, query_text: str, verbose: bool = False) -> list[float]:
        self._load_model()

        import torch
        import torch.nn.functional as F

        t0 = time.monotonic()

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Retrieve videos relevant to the query."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": query_text}],
            },
        ]

        prompt = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[prompt],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = self._pooling_last(
                outputs.last_hidden_state, inputs["attention_mask"],
            )
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        result = self._truncate_and_normalize(embeddings[0], self._dimensions)
        elapsed = time.monotonic() - t0

        if verbose:
            print(
                f"  [verbose] query embedding: dims={len(result)}, "
                f"inference_time={elapsed:.2f}s",
                file=sys.stderr,
            )

        return result

    def embed_image(self, image_path: str, verbose: bool = False) -> list[float]:
        self._load_model()

        import torch
        import torch.nn.functional as F
        from pathlib import Path

        from qwen_vl_utils import process_vision_info

        image_path = Path(image_path)
        if not image_path.exists():
            raise LocalModelError(f"Image file not found: {image_path}")

        t0 = time.monotonic()

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Retrieve videos relevant to the query."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "file://" + str(image_path.resolve())},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )

        images, video_inputs, video_kwargs = process_vision_info(
            conversation,
            return_video_metadata=True,
            return_video_kwargs=True,
        )

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
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = self._pooling_last(
                outputs.last_hidden_state, inputs["attention_mask"],
            )
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        result = self._truncate_and_normalize(embeddings[0], self._dimensions)
        elapsed = time.monotonic() - t0

        if verbose:
            print(
                f"  [verbose] image embedding: dims={len(result)}, "
                f"inference_time={elapsed:.2f}s",
                file=sys.stderr,
            )

        return result

    def dimensions(self) -> int:
        return self._dimensions
