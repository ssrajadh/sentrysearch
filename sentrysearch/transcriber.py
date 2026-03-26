"""Audio transcription for video chunks.

Extracts audio via ffmpeg, transcribes using yap (macOS 26+) or whisper (fallback).
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .chunker import _get_ffmpeg_executable

YAP_CACHE_PATH = Path.home() / ".cache" / "video-watch" / "yap"


def _find_yap() -> str | None:
    """Find yap binary: cached build or PATH."""
    if YAP_CACHE_PATH.exists():
        return str(YAP_CACHE_PATH)
    which = shutil.which("yap")
    return which


def _find_whisper() -> str | None:
    """Find whisper CLI on PATH."""
    return shutil.which("whisper")


def _extract_audio(video_path: str, output_dir: str) -> str | None:
    """Extract audio from video as 16kHz mono WAV.

    Returns path to the WAV file, or None if the video has no audio track.
    """
    if not os.path.isfile(video_path):
        return None

    # Check if video has an audio stream
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [ffprobe, "-v", "quiet", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-print_format", "json",
             video_path],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            info = json.loads(result.stdout)
            if not info.get("streams"):
                return None

    wav_path = os.path.join(output_dir, "audio.wav")
    ffmpeg_exe = _get_ffmpeg_executable()

    result = subprocess.run(
        [ffmpeg_exe, "-y", "-i", video_path,
         "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
         wav_path],
        capture_output=True,
    )

    if result.returncode != 0 or not os.path.isfile(wav_path):
        return None

    if os.path.getsize(wav_path) <= 44:
        os.unlink(wav_path)
        return None

    return wav_path


def _transcribe_yap(yap_bin: str, audio_path: str) -> str:
    """Transcribe audio using yap. Returns transcript text."""
    result = subprocess.run(
        [yap_bin, "transcribe", "--locale", "en_US", audio_path, "--json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return ""

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return ""

    segments = data.get("segments", [])
    texts = [seg.get("text", "").strip() for seg in segments]
    return " ".join(t for t in texts if t)


def _transcribe_whisper(whisper_bin: str, audio_path: str,
                        output_dir: str) -> str:
    """Transcribe audio using whisper CLI. Returns transcript text."""
    result = subprocess.run(
        [whisper_bin, audio_path,
         "--model", "base", "--language", "en",
         "--output_format", "json", "--output_dir", output_dir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return ""

    basename = Path(audio_path).stem
    json_path = Path(output_dir) / f"{basename}.json"
    if not json_path.exists():
        return ""

    with open(json_path) as f:
        data = json.load(f)

    segments = data.get("segments", [])
    texts = [seg.get("text", "").strip() for seg in segments]
    return " ".join(t for t in texts if t)


def transcribe_chunk(chunk_path: str) -> str:
    """Transcribe audio from a video chunk.

    Extracts audio, runs yap (preferred) or whisper (fallback).
    Returns transcript text, or empty string if no speech / no audio / error.
    """
    if not os.path.isfile(chunk_path):
        return ""

    tmp_dir = tempfile.mkdtemp(prefix="sentrysearch_audio_")
    try:
        wav_path = _extract_audio(chunk_path, tmp_dir)
        if wav_path is None:
            return ""

        yap_bin = _find_yap()
        if yap_bin:
            text = _transcribe_yap(yap_bin, wav_path)
            if text:
                return text

        whisper_bin = _find_whisper()
        if whisper_bin:
            return _transcribe_whisper(whisper_bin, wav_path, tmp_dir)

        return ""
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
