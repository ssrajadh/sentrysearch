"""Tests for sentrysearch.trimmer."""

import os

import pytest

from sentrysearch.trimmer import (
    _fmt_time,
    _safe_filename,
    trim_clip,
    trim_top_result,
    trim_top_results,
)


class TestFmtTime:
    def test_zero(self):
        assert _fmt_time(0) == "00m00s"

    def test_seconds(self):
        assert _fmt_time(45) == "00m45s"

    def test_minutes_and_seconds(self):
        assert _fmt_time(125) == "02m05s"

    def test_float_truncates(self):
        assert _fmt_time(59.9) == "00m59s"


class TestSafeFilename:
    def test_basic(self):
        result = _safe_filename("/path/to/video.mp4", 10.0, 40.0)
        assert result.startswith("match_")
        assert result.endswith(".mp4")
        assert "00m10s" in result
        assert "00m40s" in result

    def test_special_chars_cleaned(self):
        result = _safe_filename("/path/to/my video (1).mp4", 0.0, 30.0)
        assert " " not in result
        assert "(" not in result


class TestTrimClip:
    def test_trims_clip(self, tiny_video, tmp_path):
        output = str(tmp_path / "trimmed.mp4")
        result = trim_clip(tiny_video, 0.5, 2.5, output, padding=0.5)
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 0

    def test_end_before_start_raises(self, tiny_video, tmp_path):
        output = str(tmp_path / "bad.mp4")
        with pytest.raises(ValueError, match="end_time.*must be greater"):
            trim_clip(tiny_video, 10.0, 5.0, output)

    def test_creates_output_directory(self, tiny_video, tmp_path):
        deep_output = str(tmp_path / "a" / "b" / "c" / "clip.mp4")
        result = trim_clip(tiny_video, 0.5, 2.0, deep_output)
        assert os.path.isfile(result)


class TestTrimTopResult:
    def test_trims_first_result(self, tiny_video, tmp_path):
        results = [
            {
                "source_file": tiny_video,
                "start_time": 0.5,
                "end_time": 2.5,
                "similarity_score": 0.95,
            },
        ]
        clip = trim_top_result(results, str(tmp_path))
        assert os.path.isfile(clip)
        assert clip.startswith(str(tmp_path))

    def test_empty_results_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No results"):
            trim_top_result([], str(tmp_path))


class TestTrimTopResults:
    def test_trims_multiple_results(self, tiny_video, tmp_path):
        results = [
            {"source_file": tiny_video, "start_time": 0.5, "end_time": 1.5, "similarity_score": 0.95},
            {"source_file": tiny_video, "start_time": 1.0, "end_time": 2.0, "similarity_score": 0.85},
            {"source_file": tiny_video, "start_time": 1.5, "end_time": 2.5, "similarity_score": 0.75},
        ]
        clips = trim_top_results(results, str(tmp_path), count=3)
        assert len(clips) == 3
        for clip in clips:
            assert os.path.isfile(clip)
            assert clip.startswith(str(tmp_path))

    def test_count_limits_output(self, tiny_video, tmp_path):
        results = [
            {"source_file": tiny_video, "start_time": 0.5, "end_time": 1.5, "similarity_score": 0.95},
            {"source_file": tiny_video, "start_time": 1.0, "end_time": 2.0, "similarity_score": 0.85},
            {"source_file": tiny_video, "start_time": 1.5, "end_time": 2.5, "similarity_score": 0.75},
        ]
        clips = trim_top_results(results, str(tmp_path), count=2)
        assert len(clips) == 2

    def test_count_exceeding_results_trims_all(self, tiny_video, tmp_path):
        results = [
            {"source_file": tiny_video, "start_time": 0.5, "end_time": 1.5, "similarity_score": 0.95},
        ]
        clips = trim_top_results(results, str(tmp_path), count=5)
        assert len(clips) == 1
        assert os.path.isfile(clips[0])

    def test_reuses_rerank_clip_path(self, tmp_path, monkeypatch):
        reusable = tmp_path / "candidate.mp4"
        reusable.write_bytes(b"candidate-clip")
        output_dir = tmp_path / "out"
        results = [
            {
                "source_file": "/videos/a.mp4",
                "start_time": 0.5,
                "end_time": 1.5,
                "similarity_score": 0.95,
                "_rerank_clip_path": str(reusable),
            },
        ]

        def fail_trim(*args, **kwargs):
            raise AssertionError("should reuse rerank clip")

        monkeypatch.setattr("sentrysearch.trimmer.trim_clip", fail_trim)

        clips = trim_top_results(results, str(output_dir), count=1)

        assert len(clips) == 1
        assert os.path.isfile(clips[0])
        with open(clips[0], "rb") as f:
            assert f.read() == b"candidate-clip"
        assert reusable.exists()

    def test_missing_rerank_clip_path_falls_back_to_trim(
        self, tmp_path, monkeypatch,
    ):
        output_dir = tmp_path / "out"
        results = [
            {
                "source_file": "/videos/a.mp4",
                "start_time": 0.5,
                "end_time": 1.5,
                "similarity_score": 0.95,
                "_rerank_clip_path": str(tmp_path / "missing.mp4"),
            },
        ]
        calls = []

        def fake_trim(source_file, start_time, end_time, output_path):
            calls.append((source_file, start_time, end_time, output_path))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(b"trimmed")
            return output_path

        monkeypatch.setattr("sentrysearch.trimmer.trim_clip", fake_trim)

        clips = trim_top_results(results, str(output_dir), count=1)

        assert len(calls) == 1
        assert clips == [calls[0][3]]
        with open(clips[0], "rb") as f:
            assert f.read() == b"trimmed"

    def test_empty_results_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No results"):
            trim_top_results([], str(tmp_path))

    def test_zero_count_raises(self, tiny_video, tmp_path):
        results = [
            {"source_file": tiny_video, "start_time": 0.5, "end_time": 1.5, "similarity_score": 0.95},
        ]
        with pytest.raises(ValueError, match="count must be at least 1"):
            trim_top_results(results, str(tmp_path), count=0)
