"""Shared test fixtures."""

import os
import subprocess
import tempfile

import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory, cleaned up after test."""
    with tempfile.TemporaryDirectory(prefix="sentrysearch_test_") as d:
        yield d


@pytest.fixture
def sample_video(tmp_dir):
    """Generate a 5-second silent test video via ffmpeg."""
    path = os.path.join(tmp_dir, "test_video.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-c:v", "libx264",  "-c:a", "aac", "-shortest",
            path,
        ],
        capture_output=True,
        check=True,
    )
    return path


@pytest.fixture
def silent_video(tmp_dir):
    """Generate a 3-second video with no audio track."""
    path = os.path.join(tmp_dir, "silent_video.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
            "-c:v", "libx264", "-an",
            path,
        ],
        capture_output=True,
        check=True,
    )
    return path
