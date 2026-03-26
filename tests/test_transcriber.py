"""Tests for audio extraction and transcription."""

import os
import subprocess

import pytest


class TestExtractAudio:
    """Tests for _extract_audio()."""

    def test_extracts_wav_from_video(self, sample_video, tmp_dir):
        from sentrysearch.transcriber import _extract_audio

        wav_path = _extract_audio(sample_video, tmp_dir)
        assert os.path.isfile(wav_path)
        assert wav_path.endswith(".wav")
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", wav_path],
            capture_output=True,
        )
        assert result.returncode == 0

    def test_returns_none_for_video_without_audio(self, silent_video, tmp_dir):
        from sentrysearch.transcriber import _extract_audio

        result = _extract_audio(silent_video, tmp_dir)
        assert result is None


class TestTranscribeChunk:
    """Tests for transcribe_chunk()."""

    def test_returns_string(self, sample_video):
        from sentrysearch.transcriber import transcribe_chunk

        result = transcribe_chunk(sample_video)
        assert isinstance(result, str)

    def test_returns_empty_string_for_silent_video(self, silent_video):
        from sentrysearch.transcriber import transcribe_chunk

        result = transcribe_chunk(silent_video)
        assert result == ""

    def test_returns_empty_string_for_nonexistent_file(self, tmp_dir):
        from sentrysearch.transcriber import transcribe_chunk

        result = transcribe_chunk(os.path.join(tmp_dir, "nope.mp4"))
        assert result == ""
