# Contributing to SentrySearch

Thanks for your interest in contributing. This document covers the dev loop, conventions, and what reviewers will look for.

## Dev setup

Requires Python 3.11 or 3.12 (PyTorch wheels don't support 3.13+ yet).

```bash
git clone https://github.com/ssrajadh/sentrysearch.git
cd sentrysearch
uv sync --group test
```

If needed, also install one of the extras:

```bash
uv sync --extra local            # Mac / Linux+NVIDIA with enough VRAM
uv sync --extra local-quantized  # NVIDIA, 8–16 GB VRAM (bitsandbytes)
uv sync --extra qwen-cloud       # DashScope backend
uv sync --extra tesla            # Tesla metadata overlay
```

Run the CLI from the checkout:

```bash
uv run sentrysearch --help
```

## Running tests

```bash
uv run pytest                       # full suite
uv run pytest tests/test_search.py  # one file
uv run pytest -k highlights         # by keyword
uv run pytest --cov --cov-report=term-missing
```

CI runs on Linux, macOS, and Windows against Python 3.11 and 3.12 (see `.github/workflows/ci.yml`). All three OSes must pass before merge.

Tests that need a real model or API key should be marked and skipped by default — don't burn API credits in CI. Use the fixtures in `tests/conftest.py` for fakes.

## Code conventions

- **Backend isolation.** Embeddings from different backends/models are not interchangeable. Anything that touches the index must respect the backend/model namespacing in `store.py`.
- **CLI flags.** New `index`/`search` flags should be auto-detected from the index where possible (see how `--backend` and `--model` are inferred) so users don't have to repeat themselves.
- **No new top-level dependencies** without a clear reason. Optional features go behind an extra in `pyproject.toml` (`tesla`, `local`, `qwen-cloud`, etc.).
- **Cost-sensitive paths.** Anything that calls the Gemini or DashScope API in a loop must respect `--no-skip-still` / preprocessing settings and document cost impact.

## Branches and PRs

- Branch off `master`. Name branches by topic (`feature/highlights-lof`, `fix/tesla-overlay-tz`).
- Keep PRs focused on one change. Refactors that aren't required for the feature should be split out.
- PR description should include: what changed, why, how you tested it, and any cost/perf implications.
- Update the README when adding user-visible flags or behavior.

## Reporting bugs

Use the issue template at `.github/ISSUE_TEMPLATE/bug_report.md`. The fields there are the minimum needed to reproduce — please fill them all in.

## Questions

Open a GitHub Discussion or a draft PR if you're not sure about the approach before investing implementation time.
