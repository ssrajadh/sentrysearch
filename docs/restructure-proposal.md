# Proposal: Restructure SentrySearch into a reusable engine + a thin consumer

**Status:** Proposal — awaiting maintainer validation. No code is changed by this PR; it
contains only this document. The goal is to agree on the *direction* and the *boundaries*
before any modules move.

**Author's intent:** SentrySearch's core value is *semantic search over video content*.
Today that capability is interwoven with Tesla/Sentry-dashcam specifics. This proposal
extracts the generic engine into a standalone, reusable Python package and rebuilds
SentrySearch as a thin consumer of it, **same user-facing behaviour, new internal
architecture**, so other projects can reuse the engine.

> **Working name:** the extracted package is referred to throughout as **`clipsearch`**.
> This is a placeholder — see [§9 Naming](#9-naming-open-for-discussion) for the
> discussion. Wherever you see `clipsearch`, read "the core package, name TBD".

---

## Table of contents

1. [Motivation](#1-motivation)
2. [Goals and non-goals](#2-goals-and-non-goals)
3. [Decisions to validate](#3-decisions-to-validate)
4. [Target architecture](#4-target-architecture)
5. [What goes where (file-by-file map)](#5-what-goes-where-file-by-file-map)
6. [Core API surface](#6-core-api-surface)
7. [Backward compatibility & migration](#7-backward-compatibility--migration)
8. [Phased implementation plan](#8-phased-implementation-plan)
9. [Naming (open for discussion)](#9-naming-open-for-discussion)
10. [Testing & CI](#10-testing--ci)
11. [Risks & open questions](#11-risks--open-questions)
12. [Out of scope](#12-out-of-scope)

---

## 1. Motivation

The interesting, reusable part of this project is the pipeline:

```
discover videos → chunk (ffmpeg) → embed each chunk as video → store vectors (ChromaDB)
→ embed a text/image query → cosine match → trim the matching clip
```

None of that is Tesla-specific. What *is* Tesla/Sentry-specific:

- **SEI telemetry parsing** ([metadata.py](../sentrysearch/metadata.py), [dashcam.proto](../sentrysearch/dashcam.proto), [dashcam_pb2.py](../sentrysearch/dashcam_pb2.py)) — speed/GPS extracted from Tesla firmware metadata.
- **The HUD overlay** ([overlay.py](../sentrysearch/overlay.py)) — a Tesla-styled burn-in of speed/time/location.
- **The sibling-tool handoff** ([_toolkit_cache.py](../sentrysearch/_toolkit_cache.py)) — caches `last_search.json` / `last_clip.json` for SentryMerge and SentryBlur.
- **Dashcam-flavoured naming** in the store (`dashcam_chunks` collections) and CLI copy.

Right now these two concerns share one package and one `cli.py`, so the engine cannot be
reused without dragging the Tesla code along. Splitting them lets a second project import
the engine and build its own UI/UX on top, while SentrySearch keeps doing exactly what it
does today.

## 2. Goals and non-goals

**Goals**

- Extract a generic, dependency-light, **library-only** video semantic-search engine.
- Rebuild SentrySearch as a thin consumer: identical CLI, identical results.
- Keep the engine genuinely domain-agnostic: no `dashcam`/`tesla`/`sentry` strings in it.
- Land it as a reviewable sequence of PRs after this proposal is validated.

**Non-goals**

- No new features. This is a refactor; behaviour is preserved (see [§7](#7-backward-compatibility--migration)).
- No plugin/hook framework in the core (decision in [§3](#3-decisions-to-validate)). The
  core is a set of composable primitives; consumers wire them together.

## 3. Decisions to validate

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | **Distribution** | **Monorepo, two packages.** One git repo, two installable packages wired with a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/). | Easiest to review in one place; shared CI; defers PyPI-release overhead until reuse is proven. |
| D2 | **Backward compatibility** | **Breaking allowed, with migration notes.** | Frees us to clean up internal import paths. In practice we can still avoid an end-user re-index — see [§7](#7-backward-compatibility--migration). |
| D3 | **Core CLI** | **Library-only.** All CLI stays in SentrySearch. | Keeps the engine embeddable and dependency-light (no `click`); consumers own their UX. |
| D4 | **Extensibility model** | **Core = primitives; consumer composes.** No protocols/hooks in core. Tesla metadata + overlay live entirely in SentrySearch, applied *after* the engine returns a clip. | Avoids speculative abstraction (YAGNI). Composition already fits: overlay is a post-trim step today. |

## 4. Target architecture

### 4.1 Repository layout (uv workspace)

```
sentrysearch/                      # repo root = uv workspace
├── pyproject.toml                 # [tool.uv.workspace] members = ["packages/*"]
├── uv.lock                        # single lockfile for the whole workspace
├── README.md / README.zh.md / NOTES.md / CONTRIBUTING.md / LICENSE
├── packages/
│   ├── clipsearch/                # the reusable engine (library-only)
│   │   ├── pyproject.toml         # extras: local, local-quantized, qwen-cloud
│   │   ├── src/clipsearch/
│   │   │   ├── __init__.py        # public API re-exports
│   │   │   ├── config.py          # EngineConfig: db_path, namespace, chunk params…
│   │   │   ├── chunking.py        # ex-chunker.py (generic; configurable extensions)
│   │   │   ├── store.py           # ex-store.py (configurable namespace, no "dashcam")
│   │   │   ├── search.py          # ex-search.py
│   │   │   ├── highlights.py      # ex-highlights.py
│   │   │   ├── trimming.py        # ex-trimmer.py
│   │   │   ├── dlq.py             # ex-dlq.py (configurable path)
│   │   │   ├── indexer.py         # NEW: orchestration extracted from cli.py
│   │   │   └── embedders/
│   │   │       ├── base.py        # ex-base_embedder.py
│   │   │       ├── registry.py    # ex-embedder.py factory → registry
│   │   │       ├── gemini.py      # ex-gemini_embedder.py
│   │   │       ├── local.py       # ex-local_embedder.py
│   │   │       └── qwen_cloud.py  # ex-qwen_cloud_embedder.py
│   │   └── tests/                 # engine tests (dependency-light)
│   └── sentrysearch/              # the consumer (CLI + Tesla/Sentry domain)
│       ├── pyproject.toml         # depends on clipsearch; extras: tesla, local, …
│       ├── src/sentrysearch/
│       │   ├── __init__.py
│       │   ├── cli.py             # thin: wires clipsearch + Tesla extras
│       │   ├── metadata.py        # Tesla SEI parsing
│       │   ├── dashcam.proto / dashcam_pb2.py
│       │   ├── overlay.py         # Tesla HUD (post-trim step)
│       │   └── toolkit_cache.py   # ex-_toolkit_cache.py (SentryMerge/Blur handoff)
│       └── tests/                 # CLI + Tesla tests + an end-to-end parity test
└── .github/workflows/ci.yml       # workspace-aware matrix
```

### 4.2 Responsibility split

- **`clipsearch` (engine):** everything that is true of *any* video corpus. It knows
  nothing about Tesla, dashcams, or the sibling tools. It is configured with a data
  directory, a collection namespace, an embedder backend, and chunking parameters.
- **`sentrysearch` (consumer):** the CLI, the Tesla telemetry + overlay, the sibling-tool
  cache, and the existing `~/.sentrysearch` conventions. It calls into `clipsearch` and
  layers its domain behaviour around the results.

## 5. What goes where (file-by-file map)

| Current file | Destination | Notes |
|---|---|---|
| `base_embedder.py` | `clipsearch/embedders/base.py` | Unchanged `BaseEmbedder` ABC. |
| `embedder.py` | `clipsearch/embedders/registry.py` | Factory → a small **registry** so consumers can register custom backends without editing core (replaces the hardcoded `if/elif`). Keeps `gemini`/`local`/`qwen-cloud` registered by default. |
| `gemini_embedder.py` | `clipsearch/embedders/gemini.py` | Move as-is. |
| `local_embedder.py` | `clipsearch/embedders/local.py` | Move as-is. |
| `qwen_cloud_embedder.py` | `clipsearch/embedders/qwen_cloud.py` | Move as-is. |
| `chunker.py` | `clipsearch/chunking.py` | Generalise: `SUPPORTED_VIDEO_EXTENSIONS` and `scan_directory` become configurable rather than hardcoded `.mp4/.mov` (defaults unchanged). ffmpeg resolution logic unchanged. |
| `store.py` | `clipsearch/store.py` | **Configurable collection namespace** instead of hardcoded `dashcam_chunks`. Per-backend/model isolation + `detect_index`/`check_backend` logic preserved. `SentryStore` → e.g. `VectorStore`. |
| `search.py` | `clipsearch/search.py` | Move as-is (drops the `Sentry`-flavoured naming). |
| `highlights.py` | `clipsearch/highlights.py` | Move as-is. |
| `trimmer.py` | `clipsearch/trimming.py` | Move as-is. **No overlay knowledge** — returns a plain clip path; the consumer applies any overlay afterward. |
| `dlq.py` | `clipsearch/dlq.py` | Configurable path (default no longer `~/.sentrysearch`). |
| *(orchestration inside `cli.py index()`)* | `clipsearch/indexer.py` **(new)** | The scan→chunk→(skip-still)→preprocess→embed-with-retry→store→DLQ loop ([cli.py:441-607](../sentrysearch/cli.py#L441-L607)) is generic. Extract it to an `Indexer`/`index()` callable so the CLI just calls it and renders progress. |
| `metadata.py` | `sentrysearch/metadata.py` | Tesla SEI parsing — stays domain-specific. |
| `dashcam.proto`, `dashcam_pb2.py` | `sentrysearch/` | Generated protobuf; stays with the Tesla parser. Keep `_pb2.py` out of coverage. |
| `overlay.py` | `sentrysearch/overlay.py` | Tesla HUD. Applied by the consumer as a post-trim step. |
| `_toolkit_cache.py` | `sentrysearch/toolkit_cache.py` | SentryMerge/SentryBlur handoff. Its docstring says it's meant to be copied verbatim into sibling tools — that contract is preserved by keeping it in the consumer, **not** in core. |
| `cli.py` | `sentrysearch/cli.py` | Slimmed to wiring: build an `EngineConfig` (db `~/.sentrysearch`, namespace `dashcam`), call `clipsearch` for index/search/highlights, then apply Tesla metadata/overlay and toolkit-cache around the results. Commands and flags unchanged. |

**CLI commands** (`init`, `index`, `search`, `img`, `highlights`, `shell`, `overlay`,
`stats`, `remove`, `reset`, `dlq` group) all stay in `sentrysearch` per D3. `init`
(Gemini-key setup) and `overlay` are inherently consumer-side; the rest become thin
wrappers over `clipsearch`.

## 6. Core API surface

A sketch of what `clipsearch` would expose (illustrative — names to be finalised in
implementation). The shape follows the existing code; the new pieces are `EngineConfig`,
the embedder **registry**, and the extracted `Indexer`.

```python
from clipsearch import EngineConfig, Indexer, search, search_by_image, highlights
from clipsearch.store import VectorStore
from clipsearch.embedders import get_embedder, register_backend  # registry

config = EngineConfig(
    db_path="~/.sentrysearch/db",   # consumer chooses; no hardcoded default dir
    namespace="dashcam",            # → collection naming; consumer chooses
    backend="gemini",
    chunk_duration=30, overlap=5,
    extensions=(".mp4", ".mov"),    # generic default could be broader
)

# Indexing (orchestration that used to live in cli.py)
report = Indexer(config).index("/path/to/footage", retry_failed=False)

# Searching
store = VectorStore(config)
hits = search("a red truck", store, n_results=5, dedupe_threshold=0.9)
clip_path = trim_top_result(hits, output_dir=".")   # plain clip; no overlay

# Custom backend, no core edit needed
register_backend("my-backend", lambda **kw: MyEmbedder(**kw))
```

The consumer (`sentrysearch`) then composes domain behaviour on top:

```python
# sentrysearch/cli.py (sketch)
clip = trim_top_result(hits, output_dir)
if overlay_requested:
    samples = sentrysearch.metadata.extract_metadata(hits[0]["source_file"])  # Tesla SEI
    clip = sentrysearch.overlay.render(clip, samples)                          # Tesla HUD
sentrysearch.toolkit_cache.write_last_clip(clip)                               # sibling tools
```

Note how D4 plays out: the engine returns a clip, the consumer decorates it. No hook
interface is needed in core.

## 7. Backward compatibility & migration

D2 permits breaking changes, but the user-facing surface can stay stable — we spend the
"breaking budget" only on internal Python import paths:

- **CLI:** unchanged. Same commands, flags, output, exit codes. `pip install sentrysearch`
  still works (now pulls `clipsearch` as a dependency).
- **On-disk index:** unchanged. `sentrysearch` configures `clipsearch` with
  `namespace="dashcam"` so the engine produces the **same ChromaDB collection names**
  (`dashcam_chunks`, `dashcam_chunks_local_<model>`, `dashcam_chunks_qwen_cloud_<model>`).
  Existing indexes keep working — **no re-index required**. `detect_index`'s legacy
  fallbacks are preserved.
- **Data dir:** unchanged (`~/.sentrysearch/db`, `dlq.json`, `last_*.json`).
- **What does break (documented):** direct imports of internal modules
  (`from sentrysearch.embedder import ...`, `sentrysearch.store.SentryStore`, etc.). These
  were never a public API, but the migration guide will map old → new paths for anyone who
  relied on them. Optional-dependency extras get re-pointed (below) but keep the same names.

**Extras mapping**

| Install | Before | After |
|---|---|---|
| default | `sentrysearch` | `sentrysearch` → depends on `clipsearch` (Gemini backend) |
| local | `sentrysearch[local]` | `sentrysearch[local]` → `clipsearch[local]` |
| local-quantized | `sentrysearch[local-quantized]` | `sentrysearch[local-quantized]` → `clipsearch[local-quantized]` |
| qwen-cloud | `sentrysearch[qwen-cloud]` | `sentrysearch[qwen-cloud]` → `clipsearch[qwen-cloud]` |
| tesla | `sentrysearch[tesla]` | unchanged — Tesla overlay deps (`geopy`, protobuf) stay in `sentrysearch` |

A short **MIGRATION.md** will accompany the implementation PRs covering: new import paths,
the workspace dev setup, and confirmation that no re-index is needed.

## 8. Phased implementation plan

Each phase is an independently reviewable PR. Behaviour is preserved at every step
(verified by the parity test in [§10](#10-testing--ci)).

1. **Scaffold the workspace.** Create `packages/clipsearch` (empty) and move `sentrysearch`
   under `packages/sentrysearch`; wire the uv workspace and CI. No logic changes.
2. **Extract engine primitives.** Move the generic modules into `clipsearch`, introduce
   `EngineConfig` (db path, namespace, extensions, chunk params) and the embedder registry.
   `sentrysearch` imports them; results identical.
3. **Extract the indexer.** Pull the index loop out of `cli.py` into `clipsearch.indexer`;
   `cli.py index()` becomes a thin caller that renders progress.
4. **Isolate the domain code.** Ensure `metadata.py`, `overlay.py`, `toolkit_cache.py`, and
   the protobuf live only in `sentrysearch`; `trimming.py` in core has no overlay coupling.
5. **Slim the CLI.** `cli.py` reduced to wiring + presentation; confirm flags/UX and
   collection names unchanged; add the parity test.
6. **Docs & release decision.** Add `clipsearch/README.md`, update root README(s),
   `MIGRATION.md`, `CONTRIBUTING.md`. Decide whether to publish `clipsearch` to PyPI now or
   keep it workspace-internal (see [§11](#11-risks--open-questions)).

## 9. Naming (open for discussion)

The engine needs an import + distribution name that reads well as both `pip install X` and
`import X`, and that has no `dashcam`/`tesla`/`sentry` flavour. This is a genuine open
question — `clipsearch` is only the working placeholder used in this document.

| Candidate | Read as | Notes |
|---|---|---|
| **`clipsearch`** *(working name / recommendation)* | "search for clips" | Clear, available-looking, says what it does. |
| `vidsearch` | "video search" | Plainest; possibly too generic / clash-prone. |
| `clipseek` | "seek clips" | Distinctive, brandable. |
| `vidvec` | "video vectors" | Emphasises the embedding nature; more cryptic. |

**Recommendation:** `clipsearch`, falling back to `clipseek` if it's taken on PyPI.
Maintainer's call — happy to rename throughout once decided.

## 10. Testing & CI

- **Per-package suites.** `clipsearch/tests` for the engine, `sentrysearch/tests` for CLI
  + Tesla. The existing dependency-light approach is kept: `conftest.py` injects a fake
  `chromadb` and monkeypatches the SDKs, so the engine suite needs neither torch nor an
  API key (matching today's [tests/conftest.py](../tests/conftest.py)).
- **Parity test.** A golden end-to-end test in `sentrysearch` that indexes a tiny fixture
  video and asserts search ranking/clip output is unchanged versus `master`, guarding the
  refactor.
- **CI.** Extend [.github/workflows/ci.yml](../.github/workflows/ci.yml) to be workspace-aware
  (`uv sync` at the root, run both suites) across the existing Linux/macOS/Windows ×
  Python 3.11/3.12 matrix. Coverage config moves per-package; `dashcam_pb2.py` stays
  omitted. Keep the `check-local-deps` job.

## 11. Risks & open questions

1. **Naming** — see [§9](#9-naming-open-for-discussion). Needs a decision before Phase 2.
2. **PyPI publishing cadence.** The workspace dependency works internally without
   publishing. Do we publish `clipsearch` to PyPI immediately (real external reuse) or keep
   it workspace-internal until a second consumer exists? *Recommendation: keep internal
   until the downstream project actually imports it; publish then.*
3. **Vector store abstraction.** Core hardcodes ChromaDB today. Do we keep that, or define a
   thin `VectorStore` protocol to allow alternative stores later? *Recommendation: keep
   ChromaDB concrete for now (YAGNI, consistent with D4); revisit if a consumer needs it.*
4. **Embedder registry appetite.** Replacing the `if/elif` factory with a registry is a
   small, contained change that enables third-party backends. Confirm this is wanted, or
   keep the explicit factory.
5. **Scope of "generic" defaults.** Should the engine default `extensions` stay `.mp4/.mov`
   or broaden? *Recommendation: keep the current default; let consumers widen it.*
6. **Python support** unchanged (3.11–3.12, PyTorch-gated).

## 12. Out of scope

- New embedding backends, new search features, alternative vector stores.
- Any change to SentryMerge / SentryBlur.

---

**Asks of the maintainer:** confirm D1–D4 in [§3](#3-decisions-to-validate), pick a name in
[§9](#9-naming-open-for-discussion), and weigh in on the open questions in
[§11](#11-risks--open-questions). On approval, the work lands as the phased PR series in
[§8](#8-phased-implementation-plan).
