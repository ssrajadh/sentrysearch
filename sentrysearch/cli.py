"""Click-based CLI entry point."""

import os
import platform
import shutil
import subprocess

import click
from dotenv import load_dotenv

load_dotenv()


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


def _handle_error(e: Exception) -> None:
    """Print a user-friendly error and exit."""
    from .embedder import GeminiAPIKeyError, GeminiQuotaError

    if isinstance(e, GeminiAPIKeyError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, GeminiQuotaError):
        click.secho("Error: " + str(e), fg="yellow", err=True)
        raise SystemExit(1)
    if isinstance(e, PermissionError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, RuntimeError) and "ffmpeg" in str(e).lower():
        click.secho(
            "Error: ffmpeg is not available.\n\n"
            "Install it with one of:\n"
            "  Ubuntu/Debian:  sudo apt install ffmpeg\n"
            "  macOS:          brew install ffmpeg\n"
            "  pip fallback:   pip install imageio-ffmpeg",
            fg="red",
            err=True,
        )
        raise SystemExit(1)
    raise e


@click.group()
def cli():
    """Search dashcam footage using natural language queries."""


# -----------------------------------------------------------------------
# init
# -----------------------------------------------------------------------

@cli.command()
def init():
    """Set up your Gemini API key for sentrysearch."""
    env_path = os.path.join(os.getcwd(), ".env")

    # Check for existing key
    if os.path.exists(env_path):
        with open(env_path) as f:
            contents = f.read()
        if "GEMINI_API_KEY=" in contents:
            if not click.confirm("API key already configured. Overwrite?", default=False):
                return

    api_key = click.prompt(
        "Enter your Gemini API key (get one at https://aistudio.google.com/apikey)"
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
        from .embedder import embed_query

        vec = embed_query("test")
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
@click.option("--audio/--no-audio", default=True, show_default=True,
              help="Transcribe audio and create text embeddings.")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def index(directory, chunk_duration, overlap, preprocess, target_resolution, target_fps, skip_still, audio, verbose):
    """Index mp4 files in DIRECTORY for searching."""
    from .chunker import chunk_video, is_still_frame_chunk, preprocess_chunk, scan_directory
    from .embedder import embed_video_chunk
    from .store import SentryStore

    try:
        if audio:
            from .transcriber import transcribe_chunk
            from .embedder import embed_text

        if os.path.isfile(directory):
            videos = [os.path.abspath(directory)]
        else:
            videos = scan_directory(directory)

        if not videos:
            click.echo("No mp4 files found.")
            return

        store = SentryStore()
        total_files = len(videos)
        new_files = 0
        new_chunks = 0
        skipped_chunks = 0

        if verbose:
            click.echo(f"[verbose] DB path: {store._client._identifier}", err=True)
            click.echo(f"[verbose] chunk_duration={chunk_duration}s, overlap={overlap}s", err=True)

        for file_idx, video_path in enumerate(videos, 1):
            abs_path = os.path.abspath(video_path)
            basename = os.path.basename(video_path)

            if store.is_indexed(abs_path):
                click.echo(f"Skipping ({file_idx}/{total_files}): {basename} (already indexed)")
                continue

            chunks = chunk_video(abs_path, chunk_duration=chunk_duration, overlap=overlap)
            num_chunks = len(chunks)
            embedded = []
            transcripts = []

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

                embedding = embed_video_chunk(embed_path, verbose=verbose)
                embedded.append({**chunk, "embedding": embedding})

                if audio:
                    click.echo(
                        f"Transcribing file {file_idx}/{total_files}: {basename} "
                        f"[chunk {chunk_idx}/{num_chunks}]"
                    )
                    transcript = transcribe_chunk(chunk["chunk_path"])
                    if transcript and verbose:
                        click.echo(
                            f"    [verbose] transcript: {transcript[:80]}...",
                            err=True,
                        )
                    transcripts.append({**chunk, "transcript": transcript})

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

                if audio:
                    text_to_embed = []
                    for t in transcripts:
                        if t["transcript"]:
                            text_emb = embed_text(t["transcript"], verbose=verbose)
                            if text_emb:
                                text_to_embed.append({**t, "embedding": text_emb})
                    if text_to_embed:
                        store.add_text_chunks(text_to_embed)

        stats = store.get_stats()
        skipped_msg = f" (skipped {skipped_chunks} still)" if skipped_chunks else ""
        click.echo(
            f"\nIndexed {new_chunks} new chunks from {new_files} files{skipped_msg}. "
            f"Total: {stats['total_chunks']} chunks from "
            f"{stats['unique_source_files']} files."
        )

    except Exception as e:
        _handle_error(e)


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
@click.option("--threshold", default=0.41, show_default=True, type=float,
              help="Minimum similarity score to consider a confident match.")
@click.option("--overlay/--no-overlay", default=False, show_default=True,
              help="Burn Tesla telemetry overlay (speed, GPS, turn signals) onto trimmed clip.")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def search(query, n_results, output_dir, trim, threshold, overlay, verbose):
    """Search indexed footage with a natural language QUERY."""
    from .search import search_footage
    from .store import SentryStore

    output_dir = os.path.expanduser(output_dir)

    try:
        store = SentryStore()

        if store.get_stats()["total_chunks"] == 0:
            click.echo(
                "No indexed footage found. "
                "Run `sentrysearch index <directory>` first."
            )
            return

        if verbose:
            click.echo(f"  [verbose] similarity threshold: {threshold}", err=True)

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
            source = r.get("match_source", "visual")
            source_tag = f"[{source}]" if source != "visual" or store.has_text_index() else ""

            if verbose:
                visual_s = r.get("visual_score", score)
                text_s = r.get("text_score", 0.0)
                click.echo(
                    f"  #{i} [{score:.6f}] {source_tag} {basename} "
                    f"@ {start_str}-{end_str} "
                    f"(visual={visual_s:.4f} text={text_s:.4f})"
                )
            else:
                click.echo(
                    f"  #{i} [{score:.2f}] {source_tag} {basename} "
                    f"@ {start_str}-{end_str}"
                )

        if trim:
            if low_confidence:
                if not click.confirm(
                    f"No confident match found (best score: {best_score:.2f}). "
                    "Show results anyway?",
                    default=False,
                ):
                    return

            from .trimmer import trim_top_result
            clip_path = trim_top_result(results, output_dir)

            if overlay:
                from .overlay import apply_overlay, get_metadata_samples, reverse_geocode

                top = results[0]
                samples = get_metadata_samples(
                    top["source_file"], top["start_time"], top["end_time"],
                )
                if samples is None:
                    click.secho(
                        "No Tesla SEI metadata found in source file — skipping overlay.",
                        fg="yellow", err=True,
                    )
                else:
                    location = None
                    mid = samples[len(samples) // 2]
                    lat = mid.get("latitude_deg", 0.0)
                    lon = mid.get("longitude_deg", 0.0)
                    if lat and lon:
                        click.echo("Reverse geocoding location...")
                        location = reverse_geocode(lat, lon)
                        if location is None:
                            click.secho(
                                "Install Tesla overlay dependencies with: "
                                "pip install -e '.[tesla]'",
                                fg="yellow", err=True,
                            )

                    overlay_path = clip_path.replace(".mp4", "_overlay.mp4")
                    result_path = apply_overlay(
                        clip_path, overlay_path, samples, location,
                        source_file=top["source_file"],
                        start_time=top["start_time"],
                    )
                    if result_path == overlay_path and os.path.isfile(overlay_path):
                        os.replace(overlay_path, clip_path)
                        click.echo("Applied Tesla metadata overlay")
                    else:
                        click.secho("Overlay failed — saving plain clip.", fg="yellow", err=True)

            click.echo(f"\nSaved clip: {clip_path}")
            _open_file(clip_path)

    except Exception as e:
        _handle_error(e)


# -----------------------------------------------------------------------
# stats
# -----------------------------------------------------------------------

@cli.command()
def stats():
    """Print index statistics."""
    from .store import SentryStore

    store = SentryStore()
    s = store.get_stats()

    if s["total_chunks"] == 0:
        click.echo("Index is empty. Run `sentrysearch index <directory>` first.")
        return

    click.echo(f"Total chunks:  {s['total_chunks']}")
    click.echo(f"Source files:  {s['unique_source_files']}")
    click.echo("\nIndexed files:")
    for f in s["source_files"]:
        click.echo(f"  {f}")

    if s.get("text_chunks", 0) > 0:
        click.echo(f"Audio indexed: {s['text_chunks']} chunks")
    else:
        click.echo("Audio indexed: none (run `sentrysearch index <dir>` to add)")


# -----------------------------------------------------------------------
# reindex-audio
# -----------------------------------------------------------------------

@cli.command("reindex-audio")
@click.argument("directory", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--chunk-duration", default=30, show_default=True,
              help="Chunk duration in seconds (must match original indexing).")
@click.option("--overlap", default=5, show_default=True,
              help="Overlap between chunks in seconds (must match original indexing).")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def reindex_audio(directory, chunk_duration, overlap, verbose):
    """Add audio transcription to already-indexed videos."""
    from .chunker import chunk_video, scan_directory
    from .embedder import embed_text
    from .store import SentryStore
    from .transcriber import transcribe_chunk

    try:
        if os.path.isfile(directory):
            videos = [os.path.abspath(directory)]
        else:
            videos = scan_directory(directory)

        if not videos:
            click.echo("No mp4 files found.")
            return

        store = SentryStore()
        total_files = len(videos)
        new_text_chunks = 0

        for file_idx, video_path in enumerate(videos, 1):
            abs_path = os.path.abspath(video_path)
            basename = os.path.basename(video_path)

            if not store.is_indexed(abs_path):
                click.echo(f"Skipping ({file_idx}/{total_files}): {basename} (not indexed)")
                continue

            click.echo(f"Processing ({file_idx}/{total_files}): {basename}")
            chunks = chunk_video(abs_path, chunk_duration=chunk_duration, overlap=overlap)

            text_to_embed = []
            for chunk_idx, chunk in enumerate(chunks, 1):
                click.echo(f"  Transcribing chunk {chunk_idx}/{len(chunks)}")
                transcript = transcribe_chunk(chunk["chunk_path"])

                if transcript:
                    if verbose:
                        click.echo(f"    [verbose] transcript: {transcript[:80]}...", err=True)
                    text_emb = embed_text(transcript, verbose=verbose)
                    if text_emb:
                        text_to_embed.append({**chunk, "embedding": text_emb,
                                              "transcript": transcript})

            # Clean up chunk files
            if chunks:
                tmp_dir = os.path.dirname(chunks[0]["chunk_path"])
                shutil.rmtree(tmp_dir, ignore_errors=True)

            if text_to_embed:
                store.add_text_chunks(text_to_embed)
                new_text_chunks += len(text_to_embed)

        click.echo(f"\nAdded {new_text_chunks} audio transcriptions.")

    except Exception as e:
        _handle_error(e)
