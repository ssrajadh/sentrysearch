"""ffmpeg clip extraction."""

import os
import re
import shutil
import subprocess

from .chunker import _get_ffmpeg_executable, _get_video_duration


def trim_clip(
    source_file: str,
    start_time: float,
    end_time: float,
    output_path: str,
    padding: float = 2.0,
) -> str:
    """Extract a segment from the original source video using ffmpeg.

    Adds *padding* seconds before and after the match window, clamped to
    file boundaries.  Tries ``-c copy`` first for speed; falls back to
    re-encoding if the copy fails (e.g. when the seek lands mid-GOP).

    Args:
        source_file: Path to the original mp4 file.
        start_time: Match start time in seconds.
        end_time: Match end time in seconds.
        output_path: Where to write the trimmed clip.
        padding: Extra seconds to include before/after the match window.

    Returns:
        The *output_path* on success.
    """
    if end_time <= start_time:
        raise ValueError(
            f"end_time ({end_time}) must be greater than start_time ({start_time})."
        )

    duration = _get_video_duration(source_file)
    padded_start = max(0.0, start_time - padding)
    padded_end = min(duration, end_time + padding)
    length = padded_end - padded_start

    ffmpeg_exe = _get_ffmpeg_executable()
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Check we can write to the output directory
    if not os.access(out_dir, os.W_OK):
        raise PermissionError(
            f"Cannot write to '{out_dir}'. "
            f"Use --output-dir to specify a writable directory."
        )

    # Attempt 1: stream-copy (fast, no quality loss)
    copy_result = subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-ss", str(padded_start),
            "-i", source_file,
            "-t", str(length),
            "-c", "copy",
            output_path,
        ],
        capture_output=True,
        text=True,
    )

    # Accept if copy produced a non-empty file (ffmpeg may return non-zero
    # but still write a usable file when cutting on non-keyframes)
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 1024:
        return output_path

    # Attempt 2: re-encode with output seeking (more compatible, slower)
    # Use mpeg4/aac which are built into every ffmpeg build (no libx264 needed)
    reencode_result = subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-i", source_file,
            "-ss", str(padded_start),
            "-t", str(length),
            "-c:v", "mpeg4",
            "-q:v", "5",
            "-c:a", "aac",
            "-b:a", "128k",
            output_path,
        ],
        capture_output=True,
        text=True,
    )

    if reencode_result.returncode == 0 and os.path.isfile(output_path):
        return output_path

    # Attempt 3: just copy with output seeking (slower but avoids codec issues)
    final_result = subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-i", source_file,
            "-ss", str(padded_start),
            "-t", str(length),
            "-c", "copy",
            output_path,
        ],
        capture_output=True,
        text=True,
    )

    if final_result.returncode == 0 and os.path.isfile(output_path):
        return output_path

    # All attempts failed - provide helpful error message
    error_msg = (
        f"Failed to trim video clip from {source_file}.\n"
        f"Tried 3 different ffmpeg approaches but none succeeded.\n\n"
        f"ffmpeg stderr from last attempt:\n{final_result.stderr}"
    )
    raise RuntimeError(error_msg)


def _fmt_time(seconds: float) -> str:
    """Format seconds as e.g. '02m15s'."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}m{s:02d}s"


def _safe_filename(source_file: str, start: float, end: float) -> str:
    """Build a filesystem-safe descriptive filename."""
    base = os.path.splitext(os.path.basename(source_file))[0]
    base = re.sub(r"[^\w\-]", "_", base)
    return f"match_{base}_{_fmt_time(start)}-{_fmt_time(end)}.mp4"


def trim_top_results(results: list[dict], output_dir: str, count: int = 1) -> list[str]:
    """Trim the top *count* search results and save them to *output_dir*.

    Args:
        results: List of result dicts from :func:`search_footage`
                 (must contain source_file, start_time, end_time).
        output_dir: Directory to write clips into.
        count: Number of top results to trim (must be >= 1).

    Returns:
        List of paths to saved clips.
    """
    if not results:
        raise ValueError("No results to trim.")
    if count < 1:
        raise ValueError("count must be at least 1.")

    paths = []
    for r in results[:count]:
        filename = _safe_filename(r["source_file"], r["start_time"], r["end_time"])
        output_path = os.path.join(output_dir, filename)
        reusable_clip = r.get("_rerank_clip_path")
        if reusable_clip and os.path.isfile(reusable_clip):
            out_dir = os.path.dirname(output_path) or "."
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(reusable_clip, output_path)
            clip = output_path
        else:
            clip = trim_clip(
                source_file=r["source_file"],
                start_time=r["start_time"],
                end_time=r["end_time"],
                output_path=output_path,
            )
        paths.append(clip)

    return paths


def trim_top_result(results: list[dict], output_dir: str) -> str:
    """Trim the highest-ranked search result and save it to *output_dir*."""
    return trim_top_results(results, output_dir, count=1)[0]
