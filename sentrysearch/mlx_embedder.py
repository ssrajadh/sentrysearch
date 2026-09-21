"""Local video/text embedding on Apple Silicon via MLX.

Same model as ``local_embedder`` (Qwen3-VL-Embedding), different runtime.
MLX quantizes on Metal, which PyTorch on a Mac cannot do -- bitsandbytes is
CUDA-only -- and on a measured comparison the 4-bit 2B matched the
unquantized 2B for retrieval while running about twice as fast in under half
the memory. Quantization is effectively free at this size.

The 8B is deliberately not offered. It ties the 2B at best, needs three
times the time and memory to do it, and is far more sensitive to the
instruction strings below.
"""

import os
import sys
import time
from pathlib import Path

from .base_embedder import BaseEmbedder


class MLXModelError(RuntimeError):
    """Raised when the MLX backend fails to load or run."""


# Short aliases → MLX model repos or local paths.
MODEL_ALIASES: dict[str, str] = {
    "qwen2b": "arthurcollet/Qwen3-VL-Embedding-2B-mlx-4bit",
}

# 4-bit 2B is the default because it measured best: as accurate as the
# unquantized 2B, and as accurate as the 8B at a third of the time and
# memory. An 8B alias is deliberately absent -- see the Task 0c write-up.
DEFAULT_MODEL = MODEL_ALIASES["qwen2b"]

# Escape hatch for pointing at a locally converted model directory.
_DEFAULT_MODEL_ENV = "SENTRYSEARCH_MLX_MODEL"

_VIDEO_PROMPT = "Represent the video for retrieval."
_QUERY_PROMPT = "Retrieve videos relevant to the query."


def resolve_model_ref(model_name: str | None) -> str:
    """Return the full model reference — a local directory or a Hub repo id.

    This is what gets recorded in the index metadata, so that a later search
    can reload the same model. Callers pass the result to both ``SentryStore``
    (which keys the collection on its final path component) and
    ``MLXEmbedder``.
    """
    if model_name:
        return MODEL_ALIASES.get(model_name, model_name)
    return os.environ.get(_DEFAULT_MODEL_ENV) or DEFAULT_MODEL


def _looks_like_path(name: str) -> bool:
    """True when *name* is meant as a filesystem path, not a Hub repo id.

    Hub ids are ``namespace/repo`` — exactly one slash, no leading dot or
    tilde, no trailing slash.
    """
    if name.startswith(("/", "./", "../", "~")):
        return True
    return name.count("/") != 1 or name.endswith("/")


def is_supported() -> bool:
    """True when this machine can run the MLX backend at all."""
    import platform
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class _Box:
    """Holder that MLX will not mistake for a parameter.

    ``mlx.nn.Module.__setattr__`` registers any ``mx.array``, dict, list or
    tuple assigned to an attribute as part of the module's parameter tree.
    Stashing pixel data directly on the model would therefore corrupt
    ``model.parameters()``. A plain object with ``__slots__`` takes the
    non-array branch and is stored as an ordinary Python attribute.
    """

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


def _video_capable_class(base):
    """Build a subclass of mlx-vlm's embedding Model that forwards video.

    Upstream's ``Model.__call__`` declares ``video_grid_thw`` but not
    ``pixel_values_videos``, so decoded video pixels land in ``**kwargs`` and
    never reach the vision tower. Nothing raises: the call returns a
    correctly shaped, normalized vector that carries no information about
    the video, and it indexes and searches without complaint.

    ``get_input_embeddings`` in models/qwen3_vl/qwen3_vl.py already handles
    ``pixel_values_videos`` fully, so the fix is to route the pixels to it.
    Delete this shim once the upstream fix lands.
    """

    class Qwen3VLVideoEmbedding(base):
        def __call__(self, input_ids=None, attention_mask=None,
                     pixel_values=None, pixel_values_videos=None,
                     image_grid_thw=None, video_grid_thw=None, **kwargs):
            self._pending_video = _Box(pixel_values_videos)
            if pixel_values_videos is not None:
                # Upstream resets cached rope state only when pixel_values is
                # set, so a video-only call would reuse stale positions.
                self.language_model._rope_deltas = None
                self.language_model._position_ids = None
            try:
                return super().__call__(
                    input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    **kwargs,
                )
            finally:
                self._pending_video = _Box(None)

        def get_input_embeddings(self, input_ids=None, pixel_values=None,
                                 **kwargs):
            pending = getattr(self, "_pending_video", None)
            if pending is not None and pending.value is not None:
                kwargs["pixel_values_videos"] = pending.value
            return super().get_input_embeddings(
                input_ids=input_ids, pixel_values=pixel_values, **kwargs)

    return Qwen3VLVideoEmbedding


class MLXEmbedder(BaseEmbedder):
    """Qwen3-VL-Embedding backend running on MLX (Apple Silicon only)."""

    def __init__(
        self,
        model_name: str | None = None,
        dimensions: int = 768,
        fps: float = 1.0,
        max_frames: int = 32,
    ):
        self._model_name = resolve_model_ref(model_name)
        self._dimensions = dimensions
        self._fps = fps
        self._max_frames = max_frames
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return

        if not is_supported():
            raise MLXModelError(
                "The MLX backend requires Apple Silicon. Use --backend local "
                "for PyTorch, or a cloud backend."
            )

        path = Path(self._model_name).expanduser()
        if path.exists():
            print(f"Loading {path.name}...", file=sys.stderr)
        elif _looks_like_path(self._model_name):
            # Don't hand a mistyped path to the Hub — the resulting
            # "Repo id must be in the form 'namespace/repo_name'" points
            # nowhere near the actual problem.
            raise MLXModelError(
                f"Model directory not found: {path}\n\n"
                "Pass an existing converted model directory, or a Hugging "
                "Face MLX repo id such as 'someone/Qwen3-VL-Embedding-8B-mlx'."
            )
        else:
            from huggingface_hub import snapshot_download
            print(f"Downloading {self._model_name}...", file=sys.stderr)
            path = Path(snapshot_download(self._model_name))

        try:
            from mlx_vlm.embedding_loader import load_embedding_model
            from mlx_vlm.models.qwen3_vl_embedding.qwen3_vl_embedding import (
                Model as EmbeddingModel,
            )
            from mlx_vlm.utils import load_processor
        except ImportError as e:
            raise MLXModelError(
                f"Missing dependencies for the MLX backend: {e}\n\n"
                'Install with: uv tool install ".[mlx]"'
            ) from e

        t0 = time.monotonic()
        try:
            # Published Qwen3-VL-Embedding conversions declare
            # model_type "qwen3_vl" (inherited from the generative base), and
            # EMBEDDING_MODEL_REMAPPING has no entry for it, so the loader
            # would return the generative model and callers would fail at
            # `.text_embeds`. Force the embedding class.
            model = load_embedding_model(
                path, config_overrides={"model_type": "qwen3_vl_embedding"})
            model.__class__ = _video_capable_class(EmbeddingModel)
            self._model = model
            self._processor = load_processor(path, add_detokenizer=False)
        except Exception as e:
            raise MLXModelError(f"Failed to load {self._model_name}: {e}") from e

        print(f"Model loaded on MLX in {time.monotonic() - t0:.1f}s",
              file=sys.stderr)

    def _prompt(self, system_text: str, content: list[dict]) -> str:
        return self._processor.apply_chat_template(
            [
                {"role": "system",
                 "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _run(self, inputs) -> list[float]:
        import mlx.core as mx

        kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        for key in ("pixel_values", "image_grid_thw",
                    "pixel_values_videos", "video_grid_thw"):
            if key in inputs:
                kwargs[key] = inputs[key]

        embedding = self._model(**kwargs).text_embeds[0]

        # MRL truncation: slice the first N dims, then renormalize.
        truncated = embedding[: self._dimensions].astype(mx.float32)
        norm = mx.linalg.norm(truncated)
        if float(norm) > 0:
            truncated = truncated / norm
        mx.eval(truncated)
        return truncated.tolist()

    def embed_video_chunk(self, chunk_path: str, verbose: bool = False) -> list[float]:
        path = Path(chunk_path)
        if not path.exists():
            raise MLXModelError(f"Chunk file not found: {path}")
        self._load_model()
        from mlx_vlm.utils import prepare_inputs

        if verbose:
            size_kb = os.path.getsize(path) / 1024
            print(f"    [verbose] embedding {size_kb:.0f}KB chunk on MLX",
                  file=sys.stderr)

        t0 = time.monotonic()
        inputs = prepare_inputs(
            self._processor,
            videos=[str(path.resolve())],
            prompts=[self._prompt(_VIDEO_PROMPT, [{"type": "video"}])],
            fps=self._fps,
            max_frames=self._max_frames,
            return_tensors="mlx",
        )
        if "pixel_values_videos" not in inputs:
            raise MLXModelError(
                f"Processor returned no video pixels for {path.name}. "
                "The clip may be unreadable or zero-length."
            )
        result = self._run(inputs)

        if verbose:
            print(f"    [verbose] dims={len(result)}, "
                  f"inference_time={time.monotonic() - t0:.2f}s",
                  file=sys.stderr)
        return result

    def embed_query(self, query_text: str, verbose: bool = False) -> list[float]:
        self._load_model()
        from mlx_vlm.utils import prepare_inputs

        t0 = time.monotonic()
        inputs = prepare_inputs(
            self._processor,
            prompts=[self._prompt(
                _QUERY_PROMPT, [{"type": "text", "text": query_text}])],
            return_tensors="mlx",
        )
        result = self._run(inputs)

        if verbose:
            print(f"  [verbose] query embedding: dims={len(result)}, "
                  f"inference_time={time.monotonic() - t0:.2f}s",
                  file=sys.stderr)
        return result

    def embed_image(self, image_path: str, verbose: bool = False) -> list[float]:
        path = Path(image_path)
        if not path.exists():
            raise MLXModelError(f"Image file not found: {path}")
        self._load_model()
        from mlx_vlm.utils import prepare_inputs

        t0 = time.monotonic()
        inputs = prepare_inputs(
            self._processor,
            images=[str(path.resolve())],
            prompts=[self._prompt(_QUERY_PROMPT, [{"type": "image"}])],
            return_tensors="mlx",
        )
        result = self._run(inputs)

        if verbose:
            print(f"  [verbose] image embedding: dims={len(result)}, "
                  f"inference_time={time.monotonic() - t0:.2f}s",
                  file=sys.stderr)
        return result

    def dimensions(self) -> int:
        return self._dimensions
