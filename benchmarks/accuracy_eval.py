"""Maintainer-run retrieval-accuracy evaluation (issue #78).

Scores how well an embedding backend/model retrieves the *right* clips on
the fixed benchmark workload, using cached human yes/no judgments (pooling).
Unlike the hardware benchmark (issue #68), accuracy is hardware-independent:
the same model produces the same embeddings (barring float precision), so
the maintainer can run this locally for any backend.

Workflow:
  1. Ensure the fixed benchmark clip is on disk
     (https://github.com/ssrajadh/sentrysearch/releases/tag/benchmark-clip-v1,
     cached under ~/.sentrysearch/benchmark/).
  2. Index the clip into a throwaway ``accuracy_eval`` collection, which is
     deleted on exit (success or failure). The user's real index is never
     opened for writing; a pre-existing ``accuracy_eval`` collection without this
     eval's ownership marker aborts the run instead of being deleted.
  3. Load the embedder once, run every workload query, take the top-5 raw
     (non-deduped) results from ``search_footage``.
   4. For each retrieved (query, time-window) not yet in the judgment cache,
      prompt y/n; verdicts are written through to the cache after each answer.
      The cache records the judged clip's identity (URL + SHA-256), so a run
      against any other clip fails loudly instead of silently reusing
      judgments made on the benchmark clip.
  5. Report Precision@1/@3/@5 and MRR per query, with floor queries
     aggregated separately from fine-grained ones (terminology from #68).

Usage (from the repo root, with the desired backend extras installed):

    uv run python benchmarks/accuracy_eval.py --backend local --model qwen8b
    uv run python benchmarks/accuracy_eval.py --backend gemini
    uv run python benchmarks/accuracy_eval.py \
        --backend qwen-cloud --dashscope-model qwen3-vl-embedding

Not a user-facing CLI command — see CONTRIBUTING.md; this is a dev tool.
Judgments accumulate in tests/data/accuracy_judgments.json and are meant to
be committed, so re-runs (and issue #68's reports) reuse them for free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

from sentrysearch.chunker import chunk_video, preprocess_chunk
from sentrysearch.embedder import (
    DashScopeAPIKeyError,
    DashScopeDependencyError,
    GeminiAPIKeyError,
    GeminiQuotaError,
    embed_with_retry,
    get_embedder,
    reset_embedder,
    resolve_backend,
    resolve_model,
)
from sentrysearch.litellm_embedder import LiteLLMConfigError, LiteLLMQuotaError
from sentrysearch.local_embedder import LocalModelError
from sentrysearch.mlx_embedder import MLXModelError
from sentrysearch.search import search_footage
from sentrysearch.store import DEFAULT_DB_PATH, SentryStore

# --------------------------------------------------------------------------
# Fixed workload — identical to issue #68's query list and benchmark clip.
# Keep the two in sync: #68 imports it from benchmarks.accuracy_eval rather
# than redefining it (see the import documented in benchmarks/__init__.py).
# --------------------------------------------------------------------------

BENCHMARK_CLIP_URL = (
    "https://github.com/ssrajadh/sentrysearch/releases/download/"
    "benchmark-clip-v1/benchmark_video.mp4"
)
BENCHMARK_CLIP_PATH = Path.home() / ".sentrysearch" / "benchmark" / "benchmark_video.mp4"

# Floor queries should hit near-ceiling on any working setup; if one scores
# low the pipeline (or the user's setup) is broken, not just the model.
FLOOR_QUERIES = [
    "car driving on road",
    "highway driving",
    "black car",
]

# Fine-grained vehicle IDs, all verified present in the benchmark clip.
FINE_QUERIES = [
    "white toyota pickup truck",
    "amazon prime van",
    "black toyota 4runner",
    "black gmc pickup truck",
    "range rover",
    "tesla suv",
    "black ford explorer",
    "silver acura mdx",
    "black toyota camry",
    "blue garbage truck",
]

# (query, group) pairs; group is "floor" or "fine".
WORKLOAD: list[tuple[str, str]] = (
    [(q, "floor") for q in FLOOR_QUERIES]
    + [(q, "fine") for q in FINE_QUERIES]
)

TOP_K = 5

# Throwaway collection for this eval. ChromaDB requires names to start and
# end with an alphanumeric, so the issue's leading-underscore spelling
# (``_accuracy``) is not accepted; this name still sorts away from the
# dashcam_chunks* production collections. It carries the ownership marker
# below and is deleted on exit.
ACCURACY_COLLECTION = "accuracy_eval"

# Stamped into the eval collection's metadata as proof it belongs to this
# script. Cleanup refuses to delete a collection without it.
EVAL_MARKER_KEY = "sentrysearch_accuracy_eval"

DEFAULT_JUDGMENTS_PATH = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "accuracy_judgments.json"
)

_TIME_WINDOW_DECIMALS = 1

# Retries for a transient embed failure while indexing (2 s, then 4 s) — the
# same policy as `sentrysearch index`, except that exhausted retries abort
# the run instead of being skipped: a silently dropped chunk would skew the
# scores.
_MAX_EMBED_ATTEMPTS = 3

# Reserved top-level key in the judgment cache: records which clip the
# verdicts were made against (issue #78 B1). Queries never collide with it
# because the workload is fixed and human-written.
_META_KEY = "_meta"

# Identity fields compared between cache and clip, in reporting order.
_IDENTITY_FIELDS = ("clip_url", "clip_sha256")


class AccuracyEvalError(RuntimeError):
    """Raised for user-facing failures in the accuracy evaluation."""


# --------------------------------------------------------------------------
# Judgment cache
# --------------------------------------------------------------------------


def judgment_key(start_time: float, end_time: float) -> str:
    """Stable cache key for a clip time window.

    Windows are rounded so float noise across platforms never turns one
    chunk into a "new" clip. Format matches issue #78's agreed structure,
    e.g. ``"124.0-151.0"``.
    """
    start = round(float(start_time), _TIME_WINDOW_DECIMALS)
    end = round(float(end_time), _TIME_WINDOW_DECIMALS)
    return f"{start}-{end}"


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of *path*'s contents (content identity of a clip)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        # 1 MiB blocks: the benchmark clip is ~140 MB.
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clip_identity(
    path: str | Path, *, url: str = BENCHMARK_CLIP_URL,
) -> dict[str, str]:
    """Identity recorded alongside judgments: source URL + content hash."""
    return {"clip_url": url, "clip_sha256": file_sha256(path)}


def _format_identity(identity: dict) -> str:
    return " ".join(f"{field}={identity[field]}" for field in _IDENTITY_FIELDS)


def _read_meta(path: Path) -> dict | None:
    """Return the clip identity recorded in an existing cache, if any."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    meta = raw.get(_META_KEY) if isinstance(raw, dict) else None
    return meta if isinstance(meta, dict) else None


def _verify_clip_identity(
    path: Path,
    meta: dict | None,
    judgments: dict[str, dict[str, bool]],
    expected: dict[str, str],
) -> None:
    """Fail unless *judgments* are safe to reuse for the clip at *expected*.

    An empty cache protects nothing, so it is always adopted (and stamped
    with the current identity on first save). Anything already judged must
    carry both identity fields, and both must equal the clip being
    evaluated: verdicts from an unverified clip would silently poison the
    evaluation.
    """
    verdict_count = sum(len(verdicts) for verdicts in judgments.values())
    if verdict_count == 0:
        return
    recorded = meta if isinstance(meta, dict) else {}
    missing = [field for field in _IDENTITY_FIELDS if field not in recorded]
    if missing:
        raise AccuracyEvalError(
            f"Judgment cache {path} holds {verdict_count} verdict(s) but no "
            f"clip identity ({', '.join(missing)} missing from {_META_KEY!r}), "
            "so it cannot be verified. Delete the cache to start fresh, or "
            f"record the current clip's identity at its top level:\n"
            f"  {json.dumps({_META_KEY: expected})}"
        )
    if any(recorded[field] != expected[field] for field in _IDENTITY_FIELDS):
        raise AccuracyEvalError(
            f"Judgment cache {path} was judged against a different clip.\n"
            f"  cache: {_format_identity(recorded)}\n"
            f"  clip:  {_format_identity(expected)}\n"
            "Point --clip at the judged clip, or move/delete the cache file."
        )


def load_judgments(
    path: str | Path,
    clip_identity: dict[str, str] | None = None,
) -> dict[str, dict[str, bool]]:
    """Load the cached yes/no verdicts from *path*.

    Returns ``{query: {window_key: bool}}``. A missing file is an empty
    cache; malformed content raises :class:`AccuracyEvalError` rather than
    silently discarding the maintainer's existing judgments.

    When *clip_identity* is given (see :func:`clip_identity`), the cache is
    verified against it: judgments recorded for a different clip — or
    recorded before clip identity existed — fail loudly instead of being
    reused. An empty cache is always accepted.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AccuracyEvalError(
            f"Could not read judgment cache {path}: {exc}. "
            "Fix or delete the file, then re-run."
        ) from exc
    if not isinstance(raw, dict):
        raise AccuracyEvalError(
            f"Malformed judgment cache {path}: expected a JSON object "
            "mapping queries to verdict maps."
        )
    meta = None
    if _META_KEY in raw:
        meta = raw[_META_KEY]
        if not isinstance(meta, dict):
            raise AccuracyEvalError(
                f"Malformed judgment cache {path}: {_META_KEY!r} must be an "
                "object recording the judged clip's identity."
            )
    judgments: dict[str, dict[str, bool]] = {}
    for query, verdicts in raw.items():
        if query == _META_KEY:
            continue
        if not isinstance(verdicts, dict):
            raise AccuracyEvalError(
                f"Malformed judgment cache {path}: entry for {query!r} "
                "must map window keys to true/false."
            )
        judgments[query] = {}
        for window, verdict in verdicts.items():
            if not isinstance(verdict, bool):
                raise AccuracyEvalError(
                    f"Malformed judgment cache {path}: {query!r} -> "
                    f"{window!r} must be true or false, got {verdict!r}."
                )
            judgments[query][window] = verdict
    if clip_identity is not None:
        _verify_clip_identity(path, meta, judgments, clip_identity)
    return judgments


def save_judgments(
    path: str | Path,
    judgments: dict[str, dict[str, bool]],
    clip_identity: dict[str, str] | None = None,
) -> None:
    """Atomically write the full judgment cache back to *path*.

    The payload goes to a temporary file in the same directory and is
    swapped in with an atomic replace, so a crash mid-write can never
    truncate — and thereby destroy — the committed human judgments. Any
    temporary file left by a failed write is removed.

    *clip_identity*, when given, is recorded as the cache's clip identity;
    otherwise an identity already present in the file is preserved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        query: dict(sorted(verdicts.items()))
        for query, verdicts in sorted(judgments.items())
    }
    meta = clip_identity if clip_identity is not None else _read_meta(path)
    if meta:
        payload = {_META_KEY: dict(meta), **payload}
    text = json.dumps(payload, indent=2) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def prompt_for_verdict(
    query: str,
    result: dict,
    *,
    input_fn=input,
    out=sys.stdout,
    clip_path: str | Path | None = None,
) -> bool:
    """Ask the human whether *result* is relevant to *query*.

    Shows the absolute clip path, the time window, and a copy-paste
    ``ffplay`` command for watching that segment — printed as text, never
    executed — so the maintainer can actually see the clip before judging
    it. An empty answer counts as "no" so a stray Enter can never inflate a
    score. EOF (non-interactive input) propagates as EOFError.
    """
    start = float(result["start_time"])
    end = float(result["end_time"])
    source = clip_path if clip_path is not None else result.get("source_file") or "?"
    clip = Path(source).expanduser()
    if not clip.is_absolute():
        clip = clip.absolute()
    clip_str = str(clip)
    command = f"ffplay -ss {start:.1f} -t {end - start:.1f} {shlex.quote(clip_str)}"
    print(
        f"\nQuery:  {query!r}\n"
        f"Clip:   {clip_str}\n"
        f"Window: {start:.1f}s-{end:.1f}s ({judgment_key(start, end)})\n"
        f"View:   {command}",
        file=out,
    )
    while True:
        try:
            answer = input_fn("Relevant? [y/N]: ").strip().lower()
        except EOFError:
            print("", file=out)
            raise
        if answer in ("y", "yes"):
            return True
        if answer in ("", "n", "no"):
            return False
        print("Please answer y (relevant) or n (not relevant).", file=out)


def get_verdict(
    query: str,
    result: dict,
    judgments: dict[str, dict[str, bool]],
    judgments_path: str | Path,
    *,
    input_fn=input,
    out=sys.stdout,
    clip_path: str | Path | None = None,
    clip_identity: dict[str, str] | None = None,
) -> bool:
    """Return the cached verdict for *result*, prompting only when unseen.

    New verdicts are written through to *judgments_path* immediately (and
    atomically) so an interrupted run never loses work already judged.
    """
    key = judgment_key(result["start_time"], result["end_time"])
    cached = judgments.get(query)
    if cached is not None and key in cached:
        return cached[key]
    verdict = prompt_for_verdict(
        query, result, input_fn=input_fn, out=out, clip_path=clip_path,
    )
    judgments.setdefault(query, {})[key] = verdict
    save_judgments(judgments_path, judgments, clip_identity)
    return verdict


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def precision_at_k(relevance: list[bool], k: int) -> float:
    """Precision@k over a ranked relevance list (best rank first).

    Denominator is always *k*: a query that returned fewer than k results
    counts its missing slots as non-relevant (standard P@k). ``k`` must be
    positive.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return sum(relevance[:k]) / k


def reciprocal_rank(relevance: list[bool]) -> float:
    """Reciprocal rank of the first relevant result; 0.0 when none."""
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def score_query(
    results: list[dict],
    judgments_for_query: dict[str, bool],
) -> list[bool]:
    """Map retrieved results to a ranked relevance list via the cache."""
    return [
        judgments_for_query.get(
            judgment_key(r["start_time"], r["end_time"]), False
        )
        for r in results
    ]


def aggregate_metrics(
    scored: list[tuple[str, list[bool]]],
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict | None:
    """Mean metrics over ``(query, relevance)`` pairs; None when empty."""
    if not scored:
        return None
    agg: dict = {"n_queries": len(scored)}
    for k in k_values:
        agg[f"p@{k}"] = sum(precision_at_k(rel, k) for _, rel in scored) / len(scored)
    agg["mrr"] = sum(reciprocal_rank(rel) for _, rel in scored) / len(scored)
    return agg


# --------------------------------------------------------------------------
# Throwaway collection lifecycle
# --------------------------------------------------------------------------


def _chroma_client(db_path: str | Path):
    import chromadb

    return chromadb.PersistentClient(path=str(db_path))


def _collection_exists(client, name: str) -> bool:
    return name in {c.name for c in client.list_collections()}


def prepare_accuracy_store(
    db_path: str | Path,
    backend: str,
    model: str | None,
) -> SentryStore:
    """Create the throwaway ``accuracy_eval`` collection for this run.

    A leftover ``accuracy_eval`` collection carrying the eval ownership marker
    (from an interrupted run) is dropped and recreated. One *without* the
    marker might be the user's own data — abort instead of touching it.
    The production ``dashcam_chunks*`` collections are never opened.
    """
    client = _chroma_client(db_path)
    if _collection_exists(client, ACCURACY_COLLECTION):
        existing = client.get_collection(ACCURACY_COLLECTION)
        if not (existing.metadata or {}).get(EVAL_MARKER_KEY):
            raise AccuracyEvalError(
                f"Collection {ACCURACY_COLLECTION!r} exists in {db_path} "
                "without this evaluation's ownership marker — not touching "
                "it. Delete it manually if it is a stale leftover, then re-run."
            )
        client.delete_collection(ACCURACY_COLLECTION)

    # Pre-create the collection with the ownership marker already in its
    # metadata: ChromaDB forbids changing hnsw:space via modify(), and
    # get_or_create ignores metadata on existing collections, so the marker
    # has to be there from the start.
    client.create_collection(
        name=ACCURACY_COLLECTION,
        metadata={
            "hnsw:space": "cosine",
            EVAL_MARKER_KEY: True,
            "embedding_backend": backend,
            **({"embedding_model": model} if model else {}),
        },
    )
    return SentryStore(
        db_path=db_path, backend=backend, model=model,
        collection_name=ACCURACY_COLLECTION,
    )


def cleanup_accuracy_collection(db_path: str | Path) -> bool:
    """Delete the ``accuracy_eval`` collection iff it carries our marker.

    Returns True when a collection was deleted. Unmarked or missing
    collections are left untouched (returns False).
    """
    client = _chroma_client(db_path)
    if not _collection_exists(client, ACCURACY_COLLECTION):
        return False
    collection = client.get_collection(ACCURACY_COLLECTION)
    if not (collection.metadata or {}).get(EVAL_MARKER_KEY):
        print(
            f"warning: leaving {ACCURACY_COLLECTION!r} in place — "
            "no ownership marker, not deleting.",
            file=sys.stderr,
        )
        return False
    client.delete_collection(ACCURACY_COLLECTION)
    return True


# --------------------------------------------------------------------------
# Benchmark clip
# --------------------------------------------------------------------------


def ensure_benchmark_clip(
    clip_path: str | Path | None = None,
    *,
    url: str = BENCHMARK_CLIP_URL,
    err=sys.stderr,
) -> Path:
    """Return the benchmark clip path, downloading it when missing."""
    path = Path(clip_path) if clip_path else BENCHMARK_CLIP_PATH
    if path.is_file() and path.stat().st_size > 0:
        return path
    if clip_path is not None and not path.is_file():
        raise AccuracyEvalError(f"Benchmark clip not found: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading benchmark clip to {path} (~141 MB)...", file=err)
    part = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as resp, open(part, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            read = 0
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                fh.write(block)
                read += len(block)
                if total:
                    print(
                        f"\r  {read / 1e6:.0f}/{total / 1e6:.0f} MB",
                        end="", file=err, flush=True,
                    )
            if total:
                print(file=err)
        part.replace(path)
    except (urllib.error.URLError, OSError) as exc:
        part.unlink(missing_ok=True)
        raise AccuracyEvalError(
            f"Failed to download the benchmark clip from {url}: {exc}. "
            f"Download it manually and pass --clip <path>."
        ) from exc
    return path


# --------------------------------------------------------------------------
# Indexing and evaluation
# --------------------------------------------------------------------------


def index_clip(
    store: SentryStore,
    clip_path: str | Path,
    embedder,
    *,
    verbose: bool = False,
    err=sys.stderr,
) -> int:
    """Chunk, embed, and store the benchmark clip into *store*.

    Uses the default chunking settings (30 s chunks, 5 s overlap) and
    preprocessing, matching `sentrysearch index`. Every chunk is embedded
    (no still-frame skipping) so the eval covers the full clip.

    Transient embed failures are retried with backoff; one that survives
    them raises, so a partial index can never be scored quietly.
    Returns the number of chunks stored.
    """
    chunks = chunk_video(str(clip_path))
    stored = 0
    try:
        for i, chunk in enumerate(chunks, 1):
            print(
                f"Indexing chunk {i}/{len(chunks)} "
                f"({chunk['start_time']:.0f}s-{chunk['end_time']:.0f}s)...",
                file=err,
            )
            embed_path = preprocess_chunk(chunk["chunk_path"])
            embedding = embed_with_retry(
                embedder,
                embed_path,
                verbose=verbose,
                max_attempts=_MAX_EMBED_ATTEMPTS,
                on_retry=lambda attempt, exc, wait: print(
                    f"  Embed error (attempt {attempt}/{_MAX_EMBED_ATTEMPTS}), "
                    f"retrying in {wait}s: {exc}",
                    file=err,
                ),
            )
            store.add_chunk(
                store.make_chunk_id(chunk["source_file"], chunk["start_time"]),
                embedding,
                {
                    "source_file": chunk["source_file"],
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                },
            )
            stored += 1
    finally:
        tmp_dirs = {os.path.dirname(c["chunk_path"]) for c in chunks}
        for tmp_dir in tmp_dirs:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return stored


def collect_results(
    workload: list[tuple[str, str]],
    store: SentryStore,
    judgments: dict[str, dict[str, bool]],
    judgments_path: str | Path,
    *,
    top_k: int = TOP_K,
    input_fn=input,
    out=sys.stdout,
    err=sys.stderr,
    clip_path: str | Path | None = None,
    clip_identity: dict[str, str] | None = None,
) -> list[tuple[str, str, list[bool]]]:
    """Run every workload query, judge retrieved clips, score relevance.

    Returns ``(query, group, relevance)`` triples in workload order.
    """
    scored: list[tuple[str, str, list[bool]]] = []
    for query, group in workload:
        print(f"Querying ({group}): {query!r}", file=err)
        results = search_footage(query, store, n_results=top_k)
        relevance: list[bool] = []
        for result in results:
            verdict = get_verdict(
                query, result, judgments, judgments_path,
                input_fn=input_fn, out=out,
                clip_path=clip_path, clip_identity=clip_identity,
            )
            relevance.append(verdict)
        scored.append((query, group, relevance))
    return scored


def format_report(
    label: str,
    scored: list[tuple[str, str, list[bool]]],
) -> str:
    """Render the per-group metrics and per-query detail as a markdown block."""
    lines = [f"### Retrieval accuracy — {label}", ""]

    groups: dict[str, list[tuple[str, list[bool]]]] = {}
    for query, group, relevance in scored:
        groups.setdefault(group, []).append((query, relevance))
    all_scored = [pair for entries in groups.values() for pair in entries]

    lines.append("| Group | Queries | P@1 | P@3 | P@5 | MRR |")
    lines.append("|---|---|---|---|---|---|")
    rows: list[tuple[str, list[tuple[str, list[bool]]]]] = [
        ("floor", groups.get("floor", [])),
        ("fine-grained", groups.get("fine", [])),
        ("overall", all_scored),
    ]
    for name, entries in rows:
        agg = aggregate_metrics(entries)
        if agg is None:
            lines.append(f"| {name} | 0 | n/a | n/a | n/a | n/a |")
        else:
            lines.append(
                f"| {name} | {agg['n_queries']} "
                f"| {agg['p@1']:.2f} | {agg['p@3']:.2f} | {agg['p@5']:.2f} "
                f"| {agg['mrr']:.2f} |"
            )

    lines.append("")
    lines.append("| Query | Group | P@1 | P@3 | P@5 | RR |")
    lines.append("|---|---|---|---|---|---|")
    for query, group, relevance in scored:
        group_label = "floor" if group == "floor" else "fine"
        lines.append(
            f"| {query} | {group_label} "
            f"| {precision_at_k(relevance, 1):.2f} "
            f"| {precision_at_k(relevance, 3):.2f} "
            f"| {precision_at_k(relevance, 5):.2f} "
            f"| {reciprocal_rank(relevance):.2f} |"
        )
    return "\n".join(lines)


def run_evaluation(
    *,
    backend: str | None = None,
    model: str | None = None,
    dashscope_model: str | None = None,
    clip_path: str | Path | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    judgments_path: str | Path = DEFAULT_JUDGMENTS_PATH,
    workload: list[tuple[str, str]] | None = None,
    quantize: bool | None = None,
    input_fn=input,
    out=sys.stdout,
    err=sys.stderr,
    verbose: bool = False,
) -> str | None:
    """Run the full evaluation and return the markdown report (or None).

    Loads the embedder once, indexes the benchmark clip into the throwaway
    ``accuracy_eval`` collection (always cleaned up afterwards, failure or
    not), retrieves top-5 per query, consults/prompts for judgments bound
    to the clip's identity, and prints a per-group comparison table.
    """
    workload = list(WORKLOAD if workload is None else workload)
    # Same flag resolution as `sentrysearch index`, so both entry points
    # agree on what --backend/--model/--dashscope-model select — including
    # index's rule that --model and --dashscope-model are mutually exclusive.
    if model is not None and dashscope_model is not None:
        raise AccuracyEvalError(
            "Use only one of --model (local backend) or --dashscope-model "
            "(qwen-cloud / DashScope), not both."
        )
    backend = resolve_backend(backend, model, dashscope_model)
    model = resolve_model(backend, model, dashscope_model)

    if not workload:
        print("Workload is empty — nothing to evaluate.", file=err)
        return None

    clip = ensure_benchmark_clip(clip_path, err=err)
    identity = clip_identity(clip)
    judgments = load_judgments(judgments_path, clip_identity=identity)

    store = prepare_accuracy_store(db_path, backend, model)
    try:
        label = f"{backend}" + (f" / {model}" if model else "")
        print(f"Evaluating: {label}", file=err)
        embedder = get_embedder(backend, model=model, quantize=quantize)
        index_clip(store, clip, embedder, verbose=verbose, err=err)
        scored = collect_results(
            workload, store, judgments, judgments_path,
            input_fn=input_fn, out=out, err=err,
            clip_path=clip, clip_identity=identity,
        )
    finally:
        # Plain teardown: drop the ownership-checked throwaway collection
        # first, so an embedder-reset failure can never leak it, then reset
        # the embedder — which still runs even if the cleanup raised.
        try:
            cleanup_accuracy_collection(db_path)
        finally:
            reset_embedder()

    report = format_report(label, scored)
    print(report, file=out)
    print(f"\nJudgment cache: {judgments_path}", file=err)
    return report


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval accuracy (P@1/P@3/P@5, MRR) on the fixed "
            "benchmark clip using cached human judgments."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["gemini", "litellm", "local", "mlx", "qwen-cloud"],
        default=None,
        help="Embedding backend (default: gemini).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (local: qwen8b/qwen2b/HF id, default: auto-detect; "
             "litellm: provider model name). Ignored for qwen-cloud — use "
             "--dashscope-model there.",
    )
    parser.add_argument(
        "--dashscope-model",
        default=None,
        help="DashScope embedding model for the qwen-cloud backend "
             "(default: provider default).",
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="4-bit quantization for the local backend (default: auto-detect).",
    )
    parser.add_argument(
        "--clip",
        default=None,
        help=f"Path to the benchmark clip (default: download to {BENCHMARK_CLIP_PATH}).",
    )
    parser.add_argument(
        "--judgments",
        default=str(DEFAULT_JUDGMENTS_PATH),
        help="Judgment cache file (default: %(default)s).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show debug info.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_evaluation(
            backend=args.backend,
            model=args.model,
            dashscope_model=args.dashscope_model,
            clip_path=args.clip,
            judgments_path=args.judgments,
            quantize=args.quantize,
            verbose=args.verbose,
        )
    except AccuracyEvalError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (GeminiAPIKeyError, GeminiQuotaError,
            LiteLLMConfigError, LiteLLMQuotaError,
            DashScopeAPIKeyError, DashScopeDependencyError,
            LocalModelError, MLXModelError) as exc:
        # Backend setup failures — a missing optional extra, a missing key,
        # exhausted quota — are user-facing, not bugs: report them the way
        # `sentrysearch index` does instead of dumping a traceback.
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except EOFError:
        print(
            "Error: judgment prompt input ended (EOF). Run in a terminal, or "
            "pre-populate the judgment cache (--judgments) so no prompts are "
            "needed.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted — judgments answered so far were saved.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
