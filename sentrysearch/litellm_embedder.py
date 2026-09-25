"""LiteLLM embedding backend: route embeddings through LiteLLM.

Two modes, picked by whether ``LITELLM_PROXY_API_BASE`` is set:

- **Proxy.** POSTs OpenAI-format ``/embeddings`` requests to a LiteLLM proxy,
  such as a company AI gateway. The proxy holds the provider keys and
  enforces budgets and logging; sentrysearch only sends
  ``LITELLM_PROXY_API_KEY``. Needs no extra dependency.
- **SDK.** Calls ``litellm.embedding()`` in-process, with provider keys read
  from the environment (``GEMINI_API_KEY``, AWS credentials, ...). Needs the
  ``litellm`` extra.

Chunks are sent inline as ``data:video/mp4;base64,...`` URIs. The default
model is ``gemini/gemini-embedding-2``, the same model as the gemini
backend, reached through LiteLLM instead of the google-genai SDK.
"""

from __future__ import annotations

import base64
import os
import sys
import time

from dotenv import load_dotenv

from .base_embedder import BaseEmbedder

load_dotenv()

DEFAULT_MODEL = "gemini/gemini-embedding-2"
GEMINI_DIMENSIONS = 768

_IMAGE_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp",
    ".gif": "image/gif",
    ".heic": "image/heic", ".heif": "image/heif",
}

# Proxy statuses worth retrying; everything else fails the call immediately.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def default_litellm_model() -> str:
    """Return the embedding model from env or the built-in default."""
    return os.environ.get("LITELLM_EMBEDDING_MODEL") or DEFAULT_MODEL


def proxy_configured() -> bool:
    """True when a LiteLLM proxy is configured, so the SDK isn't needed."""
    return bool(os.environ.get("LITELLM_PROXY_API_BASE", "").strip())


def _is_gemini(model: str) -> bool:
    return "gemini" in model.lower()


class LiteLLMConfigError(RuntimeError):
    """Setup problem that every chunk would hit: missing dependency,
    rejected key, unknown model. Stops the run instead of filling the DLQ."""


class LiteLLMQuotaError(RuntimeError):
    """Rate limited by the proxy or provider after retries."""


class LiteLLMAPIError(RuntimeError):
    """A single request failed; the chunk goes to the DLQ."""


class LiteLLMEmbedder(BaseEmbedder):
    """Embeddings through LiteLLM, via a proxy or the in-process SDK."""

    def __init__(
        self,
        model_name: str | None = None,
        dimensions: int | None = None,
        rpm: int | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
    ):
        self._model = model_name or default_litellm_model()
        base = api_base or os.environ.get("LITELLM_PROXY_API_BASE", "")
        self._api_base = base.strip().rstrip("/") or None
        self._api_key = api_key or os.environ.get("LITELLM_PROXY_API_KEY")
        self._max_retries = max_retries

        # Gemini defaults to 3072 dims; 768 matches the gemini backend. Other
        # models keep their native size unless LITELLM_EMBEDDING_DIMENSIONS
        # asks for one, since not every provider accepts `dimensions`.
        env_dims = os.environ.get("LITELLM_EMBEDDING_DIMENSIONS", "").strip()
        if dimensions is None and env_dims:
            dimensions = int(env_dims)
        if dimensions is None and _is_gemini(self._model):
            dimensions = GEMINI_DIMENSIONS
        self._dims = dimensions
        self._last_dims: int | None = None

        self._limiter = None
        if rpm is not None:
            from .gemini_embedder import _RateLimiter
            self._limiter = _RateLimiter(max_per_minute=rpm)

        if self._api_base is None:
            self._litellm = _import_litellm()

    # ------------------------------------------------------------------
    # BaseEmbedder
    # ------------------------------------------------------------------

    def embed_video_chunk(self, chunk_path: str, verbose: bool = False) -> list[float]:
        uri = _data_uri(chunk_path, "video/mp4")
        return self._embed(uri, "RETRIEVAL_DOCUMENT", verbose, label=chunk_path)

    def embed_query(self, query_text: str, verbose: bool = False) -> list[float]:
        return self._embed(query_text, "RETRIEVAL_QUERY", verbose, label="query")

    def embed_image(self, image_path: str, verbose: bool = False) -> list[float]:
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in _IMAGE_MIME:
            raise ValueError(
                f"Unsupported image type {ext!r}. "
                f"Accepted: {', '.join(sorted(_IMAGE_MIME))}."
            )
        uri = _data_uri(image_path, _IMAGE_MIME[ext])
        return self._embed(uri, "RETRIEVAL_QUERY", verbose, label=image_path)

    def dimensions(self) -> int:
        return self._dims or self._last_dims or 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _params(self, task_type: str) -> dict:
        params: dict = {}
        if self._dims:
            params["dimensions"] = self._dims
        # Sent for parity with the gemini backend. gemini-embedding-2
        # ignores it (output is bit-identical with or without it), but
        # gemini-embedding-001 honors it, and LiteLLM forwards it only when
        # passed. Other providers reject the parameter, so only send it to
        # Gemini.
        if _is_gemini(self._model):
            params["task_type"] = task_type
        return params

    def _embed(self, inp: str, task_type: str, verbose: bool, label: str) -> list[float]:
        if self._limiter is not None:
            self._limiter.wait()
        params = self._params(task_type)
        t0 = time.monotonic()
        if self._api_base is not None:
            vec = self._embed_via_proxy(inp, params)
        else:
            vec = self._embed_via_sdk(inp, params)
        self._last_dims = len(vec)
        if verbose:
            via = self._api_base or "litellm SDK"
            print(
                f"    [verbose] {label}: {self._model} via {via}, "
                f"dims={len(vec)}, {time.monotonic() - t0:.2f}s",
                file=sys.stderr,
            )
        return vec

    def _embed_via_proxy(self, inp: str, params: dict) -> list[float]:
        import httpx

        # LiteLLM serves /embeddings at the root and under /v1, so this works
        # whether the base URL ends in /v1 or not.
        url = f"{self._api_base}/embeddings"
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        # encoding_format is explicit because OpenAI clients default to
        # "base64", which LiteLLM rejects for Gemini unless the proxy
        # operator enables drop_params.
        body = {
            "model": self._model,
            "input": [inp],
            "encoding_format": "float",
            **params,
        }

        delay = 2.0
        for attempt in range(self._max_retries + 1):
            last = attempt == self._max_retries
            try:
                resp = httpx.post(url, json=body, headers=headers, timeout=120.0)
            except httpx.TransportError as exc:
                if last:
                    raise LiteLLMAPIError(
                        f"Could not reach LiteLLM proxy at {self._api_base}: {exc}"
                    ) from exc
                _wait(delay, attempt, self._max_retries, f"transport error: {exc}")
                delay *= 2
                continue

            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]

            detail = _error_detail(resp)
            if resp.status_code in (401, 403):
                raise LiteLLMConfigError(
                    f"The LiteLLM proxy at {self._api_base} rejected the key "
                    f"(HTTP {resp.status_code}): {detail}\n\n"
                    "Set LITELLM_PROXY_API_KEY to a key the proxy accepts."
                )
            if resp.status_code == 404:
                raise LiteLLMConfigError(
                    f"The LiteLLM proxy at {self._api_base} has no model "
                    f"{self._model!r} (HTTP 404): {detail}\n\n"
                    "Pass the proxy's model name with --model, or set "
                    "LITELLM_EMBEDDING_MODEL."
                )
            if resp.status_code in _RETRYABLE_STATUS and not last:
                _wait(delay, attempt, self._max_retries, f"HTTP {resp.status_code}")
                delay *= 2
                continue
            if resp.status_code == 429:
                raise LiteLLMQuotaError(
                    f"Rate limited by the LiteLLM proxy: {detail}\n\n"
                    "Lower the request rate with --rpm, or try again later."
                )
            raise LiteLLMAPIError(f"LiteLLM proxy HTTP {resp.status_code}: {detail}")
        raise AssertionError("unreachable")

    def _embed_via_sdk(self, inp: str, params: dict) -> list[float]:
        litellm = self._litellm
        try:
            resp = litellm.embedding(
                model=self._model,
                input=[inp],
                num_retries=self._max_retries,
                **params,
            )
        except litellm.AuthenticationError as exc:
            raise LiteLLMConfigError(
                f"{self._model} rejected the credentials: {exc}\n\n"
                "Set the provider's key (GEMINI_API_KEY for gemini/ models)."
            ) from exc
        except litellm.UnsupportedParamsError as exc:
            raise LiteLLMConfigError(str(exc)) from exc
        except litellm.NotFoundError as exc:
            raise LiteLLMConfigError(f"Unknown model {self._model!r}: {exc}") from exc
        except litellm.RateLimitError as exc:
            raise LiteLLMQuotaError(
                f"Rate limited by {self._model}: {exc}\n\n"
                "Lower the request rate with --rpm, or try again later."
            ) from exc
        return resp.data[0]["embedding"]


def _import_litellm():
    # Without this, importing litellm downloads its model cost map from
    # GitHub. Embedding needs no pricing data, and sentrysearch should make
    # no network calls the user didn't ask for.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    try:
        import litellm
    except ImportError as exc:
        raise LiteLLMConfigError(
            "The litellm package is not installed.\n\n"
            'Install optional dependencies: uv tool install ".[litellm]"\n\n'
            "Or point at a LiteLLM proxy instead (no extra install):\n"
            "  export LITELLM_PROXY_API_BASE=https://your-gateway\n"
            "  export LITELLM_PROXY_API_KEY=sk-..."
        ) from exc
    return litellm


def _data_uri(path: str, mime: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _error_detail(resp) -> str:
    try:
        err = resp.json().get("error", {})
        msg = err.get("message") if isinstance(err, dict) else err
        return str(msg or resp.text)[:500]
    except ValueError:
        return resp.text[:500]


def _wait(delay: float, attempt: int, max_retries: int, why: str) -> None:
    wait = min(delay, 60.0)
    print(
        f"  Retryable LiteLLM error (attempt {attempt + 1}/{max_retries}), "
        f"waiting {wait:.0f}s: {why}",
        file=sys.stderr,
    )
    time.sleep(wait)
