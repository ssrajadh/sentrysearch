"""Tests for sentrysearch.chunker."""

import os
import shutil
import sys
from unittest.mock import patch

import pytest

from sentrysearch.chunker import (
    _get_ffmpeg_executable,
    _get_video_duration,
    _parse_duration_from_ffmpeg_output,
    chunk_video,
    preprocess_chunk,
    scan_directory,
)


# ---------------------------------------------------------------------------
# _parse_duration_from_ffmpeg_output
# ---------------------------------------------------------------------------

class TestParseDuration:
    def test_standard_format(self):
        stderr = "  Duration: 01:23:45.67, start: 0.000000, bitrate: 1234 kb/s"
        assert _parse_duration_from_ffmpeg_output(stderr) == pytest.approx(
            1 * 3600 + 23 * 60 + 45.67
        )

    def test_short_duration(self):
        stderr = "Duration: 00:00:03.00, start: 0.000"
        assert _parse_duration_from_ffmpeg_output(stderr) == pytest.approx(3.0)

    def test_no_duration_raises(self):
        with pytest.raises(RuntimeError, match="Could not determine"):
            _parse_duration_from_ffmpeg_output("no duration here")

    def test_ffmpeg_error_message_raised(self):
        stderr = "test.mp4: No such file or directory"
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            _parse_duration_from_ffmpeg_output(stderr)


# ---------------------------------------------------------------------------
# _get_ffmpeg_executable
# ---------------------------------------------------------------------------

class TestGetFfmpegExecutable:
    def test_returns_valid_path(self):
        exe = _get_ffmpeg_executable()
        assert isinstance(exe, str)
        assert os.path.isfile(exe)

    @patch("sentrysearch.chunker.shutil.which", return_value=None)
    def test_falls_back_to_imageio(self, _mock_which):
        exe = _get_ffmpeg_executable()
        assert os.path.isfile(exe)

    @patch("sentrysearch.chunker.shutil.which", return_value=None)
    def test_raises_when_no_ffmpeg(self, _mock_which):
        # Setting a module to None in sys.modules makes import raise ImportError
        with patch.dict(sys.modules, {"imageio_ffmpeg": None}):
            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                _get_ffmpeg_executable()


# ---------------------------------------------------------------------------
# _get_video_duration
# ---------------------------------------------------------------------------

class TestGetVideoDuration:
    def test_returns_duration(self, tiny_video):
        duration = _get_video_duration(tiny_video)
        assert 2.5 <= duration <= 3.5

    def test_nonexistent_file(self):
        with pytest.raises(Exception):
            _get_video_duration("/nonexistent/video.mp4")


# ---------------------------------------------------------------------------
# chunk_video
# ---------------------------------------------------------------------------

class TestChunkVideo:
    def test_short_video_single_chunk(self, tiny_video):
        chunks = chunk_video(tiny_video, chunk_duration=30, overlap=5)
        assert len(chunks) == 1
        assert chunks[0]["start_time"] == 0.0
        assert chunks[0]["end_time"] == pytest.approx(3.0, abs=0.5)
        assert os.path.isfile(chunks[0]["chunk_path"])
        shutil.rmtree(os.path.dirname(chunks[0]["chunk_path"]),
                       ignore_errors=True)

    def test_splits_into_multiple_chunks(self, longer_video):
        chunks = chunk_video(longer_video, chunk_duration=4, overlap=1)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert os.path.isfile(chunk["chunk_path"])
            assert chunk["source_file"] == os.path.abspath(longer_video)
            assert chunk["end_time"] > chunk["start_time"]
        # Verify step size between chunks
        step = 4 - 1
        for i in range(1, len(chunks)):
            assert chunks[i]["start_time"] == pytest.approx(
                chunks[i - 1]["start_time"] + step, abs=0.01
            )
        shutil.rmtree(os.path.dirname(chunks[0]["chunk_path"]),
                       ignore_errors=True)

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            chunk_video("/nonexistent/video.mp4")


# ---------------------------------------------------------------------------
# preprocess_chunk
# ---------------------------------------------------------------------------

class TestPreprocessChunk:
    def test_creates_preprocessed_file(self, tiny_video, tmp_path):
        import shutil as sh
        src = sh.copy2(tiny_video, tmp_path / "input.mp4")
        result = preprocess_chunk(str(src), target_resolution=32, target_fps=5)
        assert os.path.isfile(result)
        assert "preprocessed" in result

    def test_falls_back_on_invalid_input(self, tmp_path):
        fake = tmp_path / "not_a_video.mp4"
        fake.write_text("this is not a video")
        result = preprocess_chunk(str(fake))
        assert result == str(fake)


# ---------------------------------------------------------------------------
# scan_directory
# ---------------------------------------------------------------------------

class TestScanDirectory:
    def test_finds_supported_video_files(self, tmp_path):
        (tmp_path / "a.mp4").write_text("fake")
        (tmp_path / "b.MP4").write_text("fake")
        (tmp_path / "c.mov").write_text("fake")
        (tmp_path / "d.MOV").write_text("fake")
        (tmp_path / "c.txt").write_text("nope")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "e.mp4").write_text("fake")

        results = scan_directory(str(tmp_path))
        assert len(results) == 5
        extensions = {os.path.splitext(r)[1].lower() for r in results}
        assert extensions == {".mp4", ".mov"}

    def test_empty_directory(self, tmp_path):
        assert scan_directory(str(tmp_path)) == []
