"""Click-based CLI entry point."""

import os
import platform
import shutil
import subprocess

import click
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.expanduser("~"), ".sentrysearch", ".env")

# Load from stable config location first, then cwd as fallback
load_dotenv(_ENV_PATH)
load_dotenv()  # cwd .env can override


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _open_file(path: str) -> None:
    """Open a file with the system's default application."""
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", path])
        elif system == "Windows":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass  # non-critical — clip is already saved


def _overlay_output_path(path: str) -> str:
    """Return the default overlay output path for a source video."""
    base, _ext = os.path.splitext(path)
    return f"{base}_overlay.mp4"


def _handle_error(e: Exception) -> None:
    """Print a user-friendly error and exit."""
    from .gemini_embedder import GeminiAPIKeyError, GeminiQuotaError
    from .local_embedder import LocalModelError
    from .store import BackendMismatchError

    if isinstance(e, GeminiAPIKeyError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, GeminiQuotaError):
        click.secho("Error: " + str(e), fg="yellow", err=True)
        raise SystemExit(1)
    if isinstance(e, LocalModelError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, BackendMismatchError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, PermissionError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, FileNotFoundError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, RuntimeError) and "ffmpeg not found" in str(e).lower():
        click.secho(
            "Error: ffmpeg is not available.\n\n"
            "Install it with one of:\n"
            "  Ubuntu/Debian:  sudo apt install ffmpeg\n"
            "  macOS:          brew install ffmpeg\n"
            "  pip fallback:   uv add imageio-ffmpeg",
            fg="red",
            err=True,
        )
        raise SystemExit(1)
    raise e


def _apply_overlay_to_clip(
    clip_path: str,
    source_file: str,
    start_time: float,
    end_time: float,
    *,
    replace: bool = True,
) -> bool:
    """Apply Tesla telemetry overlay to a clip. Returns True on success.

    When *replace* is True the overlay is written over *clip_path* in-place.
    """
    from .overlay import apply_overlay, get_metadata_samples, reverse_geocode

    samples = get_metadata_samples(source_file, start_time, end_time)
    if samples is None:
        click.secho(
            "No Tesla SEI metadata found — skipping overlay.",
            fg="yellow", err=True,
        )
        return False

    location = None
    mid = samples[len(samples) // 2]
    lat = mid.get("latitude_deg", 0.0)
    lon = mid.get("longitude_deg", 0.0)
    if lat and lon:
        click.echo("Reverse geocoding location...")
        location = reverse_geocode(lat, lon)
        if location is None:
            click.secho(
                "Geocoding failed — continuing without location. "
                "Install deps with: uv tool install \".[tesla]\"",
                fg="yellow", err=True,
            )

    overlay_path = _overlay_output_path(clip_path)
    result_path = apply_overlay(
        clip_path, overlay_path, samples, location,
        source_file=source_file,
        start_time=start_time,
    )
    if result_path == overlay_path and os.path.isfile(overlay_path):
        if replace:
            os.replace(overlay_path, clip_path)
        click.echo("Applied Tesla metadata overlay")
        return True

    click.secho("Overlay failed.", fg="yellow", err=True)
    return False


@click.group()
def cli():
    """Search dashcam footage using natural language queries."""


# -----------------------------------------------------------------------
# init
# -----------------------------------------------------------------------

@cli.command()
def init():
    """Set up your Gemini API key for sentrysearch."""
    env_path = _ENV_PATH
    os.makedirs(os.path.dirname(env_path), exist_ok=True)

    # Check for existing key
    if os.path.exists(env_path):
        with open(env_path) as f:
            contents = f.read()
        if "GEMINI_API_KEY=" in contents:
            if not click.confirm("API key already configured. Overwrite?", default=False):
                return

    api_key = click.prompt(
        "Enter your Gemini API key\n"
        "  Get one at https://aistudio.google.com/apikey\n"
        "  (input is hidden)",
        hide_input=True,
    )

    # Write/update .env
    if os.path.exists(env_path):
        with open(env_path) as f:
            lines = f.readlines()
        with open(env_path, "w") as f:
            found = False
            for line in lines:
                if line.startswith("GEMINI_API_KEY="):
                    f.write(f"GEMINI_API_KEY={api_key}\n")
                    found = True
                else:
                    f.write(line)
            if not found:
                f.write(f"GEMINI_API_KEY={api_key}\n")
    else:
        with open(env_path, "w") as f:
            f.write(f"GEMINI_API_KEY={api_key}\n")

    # Validate by embedding a test string
    os.environ["GEMINI_API_KEY"] = api_key
    click.echo("Validating API key...")
    try:
        from .embedder import get_embedder

        embedder = get_embedder("gemini")
        vec = embedder.embed_query("test")
        if len(vec) != 768:
            click.secho(
                f"Unexpected embedding dimension: {len(vec)} (expected 768). "
                "The key may be valid but something is off.",
                fg="yellow",
                err=True,
            )
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        click.secho(f"Validation failed: {e}", fg="red", err=True)
        click.secho("Please check your API key and try again.", fg="red", err=True)
        raise SystemExit(1)

    click.secho(
        "Setup complete. You're ready to go — run "
        "`sentrysearch index <directory>` to get started.",
        fg="green",
    )
    click.secho(
        "\nTip: Set a spending limit at https://aistudio.google.com/billing "
        "to prevent accidental overspending.",
        fg="yellow",
    )


# -----------------------------------------------------------------------
# index
# -----------------------------------------------------------------------

@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--chunk-duration", default=30, show_default=True,
              help="Chunk duration in seconds.")
@click.option("--overlap", default=5, show_default=True,
              help="Overlap between chunks in seconds.")
@click.option("--preprocess/--no-preprocess", default=True, show_default=True,
              help="Downscale and reduce frame rate before embedding.")
@click.option("--target-resolution", default=480, show_default=True,
              help="Target video height in pixels for preprocessing.")
@click.option("--target-fps", default=5, show_default=True,
              help="Target frames per second for preprocessing.")
@click.option("--skip-still/--no-skip-still", default=True, show_default=True,
              help="Skip chunks with no meaningful visual change.")
@click.option("--backend", type=click.Choice(["gemini", "local"]), default=None,
              help="Embedding backend (default: gemini, or local when --model is set).")
@click.option("--model", default=None, show_default=False,
              help="Model for local backend: qwen8b, qwen2b, or HuggingFace ID "
                   "(default: auto-detect from hardware). Implies --backend local.")
@click.option("--quantize/--no-quantize", default=None,
              help="Enable/disable 4-bit quantization for local backend (default: auto-detect).")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def index(directory, chunk_duration, overlap, preprocess, target_resolution,
          target_fps, skip_still, backend, model, quantize, verbose):
    """Index supported video files in DIRECTORY for searching."""
    from .chunker import (
        SUPPORTED_VIDEO_EXTENSIONS,
        chunk_video,
        is_still_frame_chunk,
        preprocess_chunk,
        scan_directory,
    )
    from .embedder import get_embedder, reset_embedder
    from .local_embedder import detect_default_model, normalize_model_key
    from .store import SentryStore

    try:
        # --model implies --backend local
        if model is not None and backend is None:
            backend = "local"
        if backend is None:
            backend = "gemini"

        # Auto-detect model from hardware when using local backend
        if backend == "local" and model is None:
            model = detect_default_model()
            click.echo(f"Auto-detected model: {model}", err=True)

        # Normalize model key for consistent collection naming
        if backend == "local":
            model = normalize_model_key(model)

        embedder = get_embedder(backend, model=model, quantize=quantize)

        if os.path.isfile(directory):
            videos = [os.path.abspath(directory)]
        else:
            videos = scan_directory(directory)

        if not videos:
            supported = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
            click.echo(f"No supported video files found ({supported}).")
            return

        store = SentryStore(backend=backend, model=model)
        total_files = len(videos)
        new_files = 0
        new_chunks = 0
        skipped_chunks = 0

        if verbose:
            click.echo(f"[verbose] DB path: {store._client._identifier}", err=True)
            click.echo(f"[verbose] backend={backend}, chunk_duration={chunk_duration}s, overlap={overlap}s", err=True)

        for file_idx, video_path in enumerate(videos, 1):
            abs_path = os.path.abspath(video_path)
            basename = os.path.basename(video_path)

            if store.is_indexed(abs_path):
                click.echo(f"Skipping ({file_idx}/{total_files}): {basename} (already indexed)")
                continue

            chunks = chunk_video(abs_path, chunk_duration=chunk_duration, overlap=overlap)
            num_chunks = len(chunks)
            embedded = []

            if verbose:
                click.echo(f"  [verbose] {basename}: duration split into {num_chunks} chunks", err=True)

            # Track files to clean up after processing
            files_to_cleanup = []

            for chunk_idx, chunk in enumerate(chunks, 1):
                if skip_still and is_still_frame_chunk(
                    chunk["chunk_path"], verbose=verbose,
                ):
                    click.echo(
                        f"Skipping chunk {chunk_idx}/{num_chunks} (still frame)"
                    )
                    skipped_chunks += 1
                    # Clean up the skipped chunk file
                    files_to_cleanup.append(chunk["chunk_path"])
                    continue

                click.echo(
                    f"Indexing file {file_idx}/{total_files}: {basename} "
                    f"[chunk {chunk_idx}/{num_chunks}]"
                )

                embed_path = chunk["chunk_path"]
                if preprocess:
                    original_size = os.path.getsize(embed_path)
                    embed_path = preprocess_chunk(
                        embed_path,
                        target_resolution=target_resolution,
                        target_fps=target_fps,
                    )
                    if verbose:
                        new_size = os.path.getsize(embed_path)
                        click.echo(
                            f"    [verbose] preprocess: {original_size / 1024:.0f}KB -> "
                            f"{new_size / 1024:.0f}KB "
                            f"({100 * (1 - new_size / original_size):.0f}% reduction)",
                            err=True,
                        )
                    # Track preprocessed file for cleanup
                    if embed_path != chunk["chunk_path"]:
                        files_to_cleanup.append(embed_path)

                embedding = embedder.embed_video_chunk(embed_path, verbose=verbose)
                embedded.append({**chunk, "embedding": embedding})
                # Clean up chunk file after embedding
                files_to_cleanup.append(chunk["chunk_path"])

            # Clean up temporary chunk files
            for f in files_to_cleanup:
                try:
                    os.unlink(f)
                except OSError:
                    pass

            # Clean up the temporary directory containing chunks
            if chunks:
                tmp_dir = os.path.dirname(chunks[0]["chunk_path"])
                shutil.rmtree(tmp_dir, ignore_errors=True)

            if embedded:
                store.add_chunks(embedded)
                new_files += 1
                new_chunks += len(embedded)

        stats = store.get_stats()
        skipped_msg = f" (skipped {skipped_chunks} still)" if skipped_chunks else ""
        click.echo(
            f"\nIndexed {new_chunks} new chunks from {new_files} files{skipped_msg}. "
            f"Total: {stats['total_chunks']} chunks from "
            f"{stats['unique_source_files']} files."
        )

    except Exception as e:
        _handle_error(e)
    finally:
        reset_embedder()


# -----------------------------------------------------------------------
# search
# -----------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("-n", "--results", "n_results", default=5, show_default=True,
              help="Number of results to return.")
@click.option("-o", "--output-dir", default="~/sentrysearch_clips", show_default=True,
              help="Directory to save trimmed clips.")
@click.option("--trim/--no-trim", default=True, show_default=True,
              help="Auto-trim the top result.")
@click.option("--save-top", default=None, type=click.IntRange(min=1),
              help="Save the top N matching clips instead of just the #1 result (e.g. --save-top 3).")
@click.option("--threshold", default=0.41, show_default=True, type=float,
              help="Minimum similarity score to consider a confident match.")
@click.option("--overlay/--no-overlay", default=False, show_default=True,
              help="Burn Tesla telemetry overlay (speed, GPS, turn signals) onto trimmed clip.")
@click.option("--backend", type=click.Choice(["gemini", "local"]), default=None,
              help="Embedding backend (auto-detected from index if omitted).")
@click.option("--model", default=None, show_default=False,
              help="Model for local backend: qwen8b, qwen2b, or HuggingFace ID "
                   "(default: auto-detect from index). Implies --backend local.")
@click.option("--quantize/--no-quantize", default=None,
              help="Enable/disable 4-bit quantization for local backend (default: auto-detect).")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def search(query, n_results, output_dir, trim, save_top, threshold, overlay, backend, model, quantize, verbose):
    """Search indexed footage with a natural language QUERY."""
    from .embedder import get_embedder, reset_embedder
    from .local_embedder import normalize_model_key
    from .search import search_footage
    from .store import SentryStore, detect_index

    output_dir = os.path.expanduser(output_dir)

    try:
        # --model implies --backend local
        if model is not None and backend is None:
            backend = "local"

        # Normalize model key for consistent collection naming
        if model is not None:
            model = normalize_model_key(model)

        # Auto-detect backend and model from whichever collection has data
        if backend is None:
            detected_backend, detected_model = detect_index()
            backend = detected_backend or "gemini"
            if model is None:
                model = detected_model
        elif backend == "local" and model is None:
            _, detected_model = detect_index()
            model = detected_model

        store = SentryStore(backend=backend, model=model)

        if store.get_stats()["total_chunks"] == 0:
            # Check if data exists under a different model
            det_backend, det_model = detect_index()
            if det_backend == backend and det_model and det_model != model:
                click.echo(
                    f"No footage indexed with the {model} model. "
                    f"Your index uses {det_model}.\n\n"
                    f"Try: sentrysearch search \"{query}\" --model {det_model}"
                )
            elif det_backend and det_backend != backend:
                click.echo(
                    f"No footage indexed with the {backend} backend. "
                    f"Your index uses {det_backend}."
                )
            else:
                click.echo(
                    "No indexed footage found. "
                    "Run `sentrysearch index <directory>` first."
                )
            return

        get_embedder(backend, model=model, quantize=quantize)

        # Ensure we fetch enough results for --save-top
        if save_top is not None and save_top > n_results:
            n_results = save_top

        if verbose:
            click.echo(f"  [verbose] backend={backend}, similarity threshold: {threshold}", err=True)

        results = search_footage(query, store, n_results=n_results, verbose=verbose)

        if not results:
            click.echo(
                "No results found.\n\n"
                "Suggestions:\n"
                "  - Try a broader or different query\n"
                "  - Re-index with smaller --chunk-duration for finer granularity\n"
                "  - Check `sentrysearch stats` to see what's indexed"
            )
            return

        best_score = results[0]["similarity_score"]
        low_confidence = best_score < threshold

        if low_confidence and not trim:
            click.secho(
                f"(low confidence — best score: {best_score:.2f})",
                fg="yellow",
                err=True,
            )

        for i, r in enumerate(results, 1):
            basename = os.path.basename(r["source_file"])
            start_str = _fmt_time(r["start_time"])
            end_str = _fmt_time(r["end_time"])
            score = r["similarity_score"]
            if verbose:
                click.echo(
                    f"  #{i} [{score:.6f}] {basename} "
                    f"@ {start_str}-{end_str}"
                )
            else:
                click.echo(
                    f"  #{i} [{score:.2f}] {basename} "
                    f"@ {start_str}-{end_str}"
                )

        should_trim = trim or save_top is not None
        if should_trim:
            if low_confidence:
                if not click.confirm(
                    f"No confident match found (best score: {best_score:.2f}). "
                    "Show results anyway?",
                    default=False,
                ):
                    return

            from .trimmer import trim_top_results
            count = save_top if save_top is not None else 1
            clip_paths = trim_top_results(results, output_dir, count=count)

            for i, clip_path in enumerate(clip_paths):
                if overlay:
                    r = results[i]
                    _apply_overlay_to_clip(
                        clip_path, r["source_file"],
                        r["start_time"], r["end_time"],
                    )
                click.echo(f"\nSaved clip: {clip_path}")

            if clip_paths:
                _open_file(clip_paths[0])

    except Exception as e:
        _handle_error(e)
    finally:
        reset_embedder()


# -----------------------------------------------------------------------
# overlay
# -----------------------------------------------------------------------

@cli.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", default=None,
              help="Output path (default: <video>_overlay.mp4).")
def overlay(video, output):
    """Apply Tesla telemetry overlay to a VIDEO file for testing."""
    from .chunker import _get_video_duration

    video = os.path.abspath(video)
    if output is None:
        output = _overlay_output_path(video)

    try:
        duration = _get_video_duration(video)
    except Exception as e:
        _handle_error(e)

    success = _apply_overlay_to_clip(
        video, video, 0.0, duration, replace=False,
    )
    if success:
        overlay_path = _overlay_output_path(video)
        if output != overlay_path and os.path.isfile(overlay_path):
            os.replace(overlay_path, output)
        click.secho(f"Saved: {output}", fg="green")
        _open_file(output)
    else:
        raise SystemExit(1)


# -----------------------------------------------------------------------
# stats
# -----------------------------------------------------------------------

@cli.command()
def stats():
    """Print index statistics."""
    from .store import SentryStore, detect_index

    backend, model = detect_index()
    if backend is None:
        backend = "gemini"
    store = SentryStore(backend=backend, model=model)
    s = store.get_stats()

    if s["total_chunks"] == 0:
        click.echo("Index is empty. Run `sentrysearch index <directory>` first.")
        return

    click.echo(f"Total chunks:  {s['total_chunks']}")
    click.echo(f"Source files:  {s['unique_source_files']}")
    backend_label = store.get_backend()
    if model:
        backend_label += f" ({model})"
    click.echo(f"Backend:       {backend_label}")
    click.echo("\nIndexed files:")
    for f in s["source_files"]:
        exists = os.path.exists(f)
        label = "" if exists else "  [missing]"
        click.echo(f"  {f}{label}")


# -----------------------------------------------------------------------
# reset
# -----------------------------------------------------------------------

@cli.command()
@click.option("--backend", type=click.Choice(["gemini", "local"]), default=None,
              help="Backend to reset (auto-detected if omitted).")
@click.option("--model", default=None,
              help="Model to reset (auto-detected if omitted). Implies --backend local.")
@click.confirmation_option(prompt="This will delete all indexed data. Continue?")
def reset(backend, model):
    """Delete all indexed data."""
    from .store import SentryStore, detect_index

    if model is not None and backend is None:
        backend = "local"
    if backend is None:
        backend, detected_model = detect_index()
        backend = backend or "gemini"
        if model is None:
            model = detected_model

    store = SentryStore(backend=backend, model=model)
    s = store.get_stats()

    if s["total_chunks"] == 0:
        click.echo("Index is already empty.")
        return

    for f in s["source_files"]:
        store.remove_file(f)

    click.echo(f"Removed {s['total_chunks']} chunks from {s['unique_source_files']} files.")


# -----------------------------------------------------------------------
# remove
# -----------------------------------------------------------------------

@cli.command()
@click.argument("files", nargs=-1, required=True)
@click.option("--backend", type=click.Choice(["gemini", "local"]), default=None,
              help="Backend to remove from (auto-detected if omitted).")
@click.option("--model", default=None,
              help="Model to remove from (auto-detected if omitted). Implies --backend local.")
def remove(files, backend, model):
    """Remove specific files from the index.

    Accepts full paths or substrings that match indexed file paths.
    """
    from .store import SentryStore, detect_index

    if model is not None and backend is None:
        backend = "local"
    if backend is None:
        backend, detected_model = detect_index()
        backend = backend or "gemini"
        if model is None:
            model = detected_model

    store = SentryStore(backend=backend, model=model)
    s = store.get_stats()

    if s["total_chunks"] == 0:
        click.echo("Index is empty.")
        return

    total_removed = 0
    for pattern in files:
        # Match against indexed source files (substring match)
        matches = [f for f in s["source_files"] if pattern in f]
        if not matches:
            click.echo(f"No indexed files matching '{pattern}'")
            continue
        for source_file in matches:
            removed = store.remove_file(source_file)
            click.echo(f"Removed {removed} chunks from {source_file}")
            total_removed += removed

    if total_removed:
        click.echo(f"\nTotal: removed {total_removed} chunks.")
