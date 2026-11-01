"""Tests for sentrysearch.cli (Click CLI)."""

import os
import subprocess
from unittest.mock import ANY, MagicMock, patch

import pytest
from click.testing import CliRunner

from sentrysearch.cli import _fmt_time, _overlay_output_path, cli

import importlib.metadata


@pytest.fixture
def runner():
    return CliRunner()


class TestFmtTime:
    def test_zero(self):
        assert _fmt_time(0) == "00:00"

    def test_minutes(self):
        assert _fmt_time(125) == "02:05"


class TestOverlayOutputPath:
    def test_mov_input_outputs_mp4(self):
        assert _overlay_output_path("/tmp/iphone.mov") == "/tmp/iphone_overlay.mp4"


class TestCliGroup:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Search dashcam footage" in result.output or "search" in result.output.lower()

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        expected_version = importlib.metadata.version("sentrysearch")
        assert expected_version in result.output


class TestModelDashscopeFlagConflict:
    def test_index_rejects_both_model_flags(self, runner, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        result = runner.invoke(cli, [
            "index", str(d),
            "--model", "qwen2b",
            "--dashscope-model", "qwen3-vl-embedding",
        ])
        assert result.exit_code == 2
        out = (result.output or "") + (result.stderr or "")
        assert "not both" in out.lower() or "only one of" in out.lower()

    def test_search_rejects_both_model_flags(self, runner):
        result = runner.invoke(cli, [
            "search", "query",
            "--model", "qwen2b",
            "--dashscope-model", "qwen3-vl-embedding",
        ])
        assert result.exit_code == 2


class TestStatsCommand:
    def test_stats_empty(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.get_all_stats", return_value=[]):
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 0
            assert "empty" in result.output.lower()
            # stats must not build a store (which would create collections).
            MockStore.assert_not_called()

    def test_stats_with_data(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.get_all_stats", return_value=[{
                 "backend": "local",
                 "model": "qwen2b",
                 "collection": "dashcam_chunks_local_qwen2b",
                 "total_chunks": 10,
                 "unique_source_files": 2,
                 "source_files": ["/a/video1.mp4", "/b/video2.mp4"],
             }]):
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 0
            assert "10" in result.output
            assert "qwen2b" in result.output
            assert "local" in result.output
            assert "gemini" not in result.output
            MockStore.assert_not_called()

    def test_stats_json_empty_is_a_list(self, runner):
        import json
        with patch("sentrysearch.store.get_all_stats", return_value=[]):
            result = runner.invoke(cli, ["stats", "--json"])
            assert result.exit_code == 0
            assert json.loads(result.output) == []

    def test_stats_json_one_index_is_a_list(self, runner, tmp_path):
        import json
        file_a = tmp_path / "a.mp4"
        file_b = tmp_path / "b.mp4"
        file_gone = tmp_path / "gone.mp4"
        file_a.write_text("a")
        file_b.write_text("b")

        with patch("sentrysearch.store.get_all_stats", return_value=[{
            "backend": "gemini",
            "model": None,
            "collection": "dashcam_chunks",
            "total_chunks": 1234,
            "unique_source_files": 3,
            "source_files": [str(file_a), str(file_b), str(file_gone)],
        }]):
            result = runner.invoke(cli, ["stats", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["backend"] == "gemini"
            assert data[0]["model"] is None
            assert data[0]["total_chunks"] == 1234
            assert data[0]["unique_source_files"] == 3
            assert data[0]["source_files"] == [str(file_a), str(file_b), str(file_gone)]
            assert data[0]["missing_files"] == [str(file_gone)]

    def test_stats_multiple_backends(self, runner):
        with patch("sentrysearch.store.get_all_stats", return_value=[
            {
                "backend": "gemini",
                "model": None,
                "collection": "dashcam_chunks",
                "total_chunks": 812,
                "unique_source_files": 20,
                "source_files": ["/v/gemini.mp4"],
            },
            {
                "backend": "local",
                "model": "qwen8b",
                "collection": "dashcam_chunks_local_qwen8b",
                "total_chunks": 340,
                "unique_source_files": 10,
                "source_files": ["/v/local.mp4"],
            },
        ]):
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 0
            assert "812" in result.output
            assert "340" in result.output
            assert "gemini" in result.output
            assert "qwen8b" in result.output
            assert "/v/gemini.mp4" in result.output
            assert "/v/local.mp4" in result.output

    def test_stats_json_multiple_backends_returns_list(self, runner, tmp_path):
        import json
        file_a = tmp_path / "a.mp4"
        file_a.write_text("a")
        with patch("sentrysearch.store.get_all_stats", return_value=[
            {
                "backend": "gemini",
                "model": None,
                "collection": "dashcam_chunks",
                "total_chunks": 812,
                "unique_source_files": 1,
                "source_files": [str(file_a)],
            },
            {
                "backend": "local",
                "model": "qwen8b",
                "collection": "dashcam_chunks_local_qwen8b",
                "total_chunks": 340,
                "unique_source_files": 1,
                "source_files": [str(file_a)],
            },
        ]):
            result = runner.invoke(cli, ["stats", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert isinstance(data, list)
            assert len(data) == 2
            assert data[0]["backend"] == "gemini"
            assert data[0]["total_chunks"] == 812
            assert data[1]["backend"] == "local"
            assert data[1]["model"] == "qwen8b"
            assert data[1]["total_chunks"] == 340


class TestSearchCommand:
    def test_search_empty_index(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index", return_value=(None, None)):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 0}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["search", "red car"])
            assert result.exit_code == 0
            assert "No indexed footage" in result.output


class TestIndexCommand:
    def test_index_no_supported_videos(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()):
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, ["index", str(empty_dir)])
            assert result.exit_code == 0
            assert "No supported video files found" in result.output

    def test_index_accepts_backend_option(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()):
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, ["index", str(empty_dir), "--backend", "local"])
            assert result.exit_code == 0

    def test_index_scans_mov_files(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "iphone.MOV"
        source.write_bytes(b"fake")

        chunk_dir = tmp_path / "chunks"
        chunk_dir.mkdir()
        chunk_path = chunk_dir / "chunk_000.mp4"
        chunk_path.write_bytes(b"chunk")

        mock_store = MagicMock()
        mock_store.has_chunk.return_value = False
        mock_store.make_chunk_id.return_value = "abc123"
        mock_store.get_stats.return_value = {
            "total_chunks": 1,
            "unique_source_files": 1,
        }
        mock_embedder = MagicMock()
        mock_embedder.embed_video_chunk.return_value = [0.1] * 768

        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.chunker.chunk_video", return_value=[{
                 "chunk_path": str(chunk_path),
                 "source_file": str(source.resolve()),
                 "start_time": 0.0,
                 "end_time": 1.0,
             }]), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(cli, ["index", str(d), "--no-preprocess"])

        assert result.exit_code == 0
        mock_store.add_chunk.assert_called_once()

    def test_index_resumes_skipping_already_indexed_chunks(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "video.mp4"
        source.write_bytes(b"fake")

        chunk_dir = tmp_path / "chunks"
        chunk_dir.mkdir()
        chunks = []
        for i in range(3):
            p = chunk_dir / f"chunk_{i:03d}.mp4"
            p.write_bytes(b"chunk")
            chunks.append({
                "chunk_path": str(p),
                "source_file": str(source.resolve()),
                "start_time": float(i * 30),
                "end_time": float(i * 30 + 30),
            })

        mock_store = MagicMock()
        # First two chunks already indexed, third one is new
        mock_store.has_chunk.side_effect = [True, True, False]
        mock_store.make_chunk_id.side_effect = ["id0", "id1", "id2"]
        mock_store.get_stats.return_value = {
            "total_chunks": 3, "unique_source_files": 1,
        }
        mock_embedder = MagicMock()
        mock_embedder.embed_video_chunk.return_value = [0.1] * 768

        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.chunker.chunk_video", return_value=chunks), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(cli, ["index", str(d), "--no-preprocess"])

        assert result.exit_code == 0
        # Only the third chunk should have been embedded/stored
        assert mock_embedder.embed_video_chunk.call_count == 1
        mock_store.add_chunk.assert_called_once()
        args, _ = mock_store.add_chunk.call_args
        assert args[0] == "id2"


    def test_index_records_failed_chunk_to_dlq(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "video.mp4"
        source.write_bytes(b"fake")

        chunk_dir = tmp_path / "chunks"
        chunk_dir.mkdir()
        chunk_path = chunk_dir / "chunk_000.mp4"
        chunk_path.write_bytes(b"chunk")

        mock_store = MagicMock()
        mock_store.has_chunk.return_value = False
        mock_store.make_chunk_id.return_value = "failing_id"
        mock_store.get_stats.return_value = {
            "total_chunks": 0, "unique_source_files": 0,
        }
        mock_embedder = MagicMock()
        mock_embedder.embed_video_chunk.side_effect = RuntimeError(
            "CUDA out of memory"
        )

        from sentrysearch.dlq import DeadLetterQueue
        dlq_instance = DeadLetterQueue(tmp_path / "dlq.json")

        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.dlq.DeadLetterQueue", return_value=dlq_instance), \
             patch("sentrysearch.chunker.chunk_video", return_value=[{
                 "chunk_path": str(chunk_path),
                 "source_file": str(source.resolve()),
                 "start_time": 0.0,
                 "end_time": 30.0,
             }]), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(cli, ["index", str(d), "--no-preprocess"])

        assert result.exit_code == 0, result.output
        # OOM is permanent — should DLQ on first attempt without retries
        assert mock_embedder.embed_video_chunk.call_count == 1
        mock_store.add_chunk.assert_not_called()
        assert dlq_instance.contains("failing_id")
        entry = dlq_instance.entries()["failing_id"]
        assert "out of memory" in entry["error"].lower()

    def _two_videos_first_unreadable(self, tmp_path, readme=None):
        d = tmp_path / "vids"
        d.mkdir()
        bad = d / "a_bad.mp4"
        good = d / "b_good.mp4"
        bad.write_bytes(b"not a video")
        good.write_bytes(b"fake")
        if readme:
            (d / "-README_en.txt").write_text(readme)
        chunk_path = tmp_path / "chunk_000.mp4"
        chunk_path.write_bytes(b"chunk")

        def chunk_video(path, **kw):
            if path.endswith("a_bad.mp4"):
                raise subprocess.CalledProcessError(1, ["ffprobe"])
            return [{"chunk_path": str(chunk_path), "source_file": path,
                     "start_time": 0.0, "end_time": 30.0}]

        store = MagicMock()
        store.has_chunk.return_value = False
        store.make_chunk_id.side_effect = lambda p, s: f"{p}:{s}"
        store.get_stats.return_value = {"total_chunks": 1, "unique_source_files": 1}
        embedder = MagicMock()
        embedder.embed_video_chunk.return_value = [0.0] * 768
        return d, chunk_video, store, embedder

    def test_index_continues_past_an_unreadable_file(self, runner, tmp_path):
        """One file that won't open used to abort the whole run."""
        d, chunk_video, store, embedder = self._two_videos_first_unreadable(tmp_path)
        with patch("sentrysearch.store.SentryStore", return_value=store), \
             patch("sentrysearch.embedder.get_embedder", return_value=embedder), \
             patch("sentrysearch.chunker.chunk_video", side_effect=chunk_video), \
             patch("sentrysearch.chunker.probe_error",
                   return_value="Invalid data found when processing input"), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(cli, ["index", str(d), "--no-preprocess"])

        assert result.exit_code == 0, result.output
        assert store.add_chunk.call_count == 1  # the good file still indexed
        assert "a_bad.mp4 (unreadable: Invalid data found" in result.output
        assert "1 unreadable skipped" in result.output

    def test_index_explains_tesla_encrypted_clips(self, runner, tmp_path):
        d, chunk_video, store, embedder = self._two_videos_first_unreadable(
            tmp_path, readme="go to https://dashcam.tesla.com to decrypt")
        with patch("sentrysearch.store.SentryStore", return_value=store), \
             patch("sentrysearch.embedder.get_embedder", return_value=embedder), \
             patch("sentrysearch.chunker.chunk_video", side_effect=chunk_video), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(cli, ["index", str(d), "--no-preprocess"])

        assert result.exit_code == 0, result.output
        assert "a_bad.mp4 (Tesla-encrypted clip)" in result.output
        assert "in a browser at https://dashcam.tesla.com" in result.output
        assert "Encrypt Dashcam Recordings" in result.output
        assert store.add_chunk.call_count == 1

    def test_index_explains_macos_folder_block(self, runner, tmp_path):
        d, chunk_video, store, embedder = self._two_videos_first_unreadable(tmp_path)
        with patch("sentrysearch.store.SentryStore", return_value=store), \
             patch("sentrysearch.embedder.get_embedder", return_value=embedder), \
             patch("sentrysearch.chunker.chunk_video", side_effect=chunk_video), \
             patch("sentrysearch.chunker.probe_error",
                   return_value="Operation not permitted"), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(cli, ["index", str(d), "--no-preprocess"])

        assert result.exit_code == 0, result.output
        assert "Privacy & Security" in result.output

    def test_index_skips_dlq_chunks_by_default(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "video.mp4"
        source.write_bytes(b"fake")

        chunk_dir = tmp_path / "chunks"
        chunk_dir.mkdir()
        chunk_path = chunk_dir / "chunk_000.mp4"
        chunk_path.write_bytes(b"chunk")

        mock_store = MagicMock()
        mock_store.has_chunk.return_value = False
        mock_store.make_chunk_id.return_value = "dlq_id"
        mock_store.get_stats.return_value = {
            "total_chunks": 0, "unique_source_files": 0,
        }
        mock_embedder = MagicMock()

        from sentrysearch.dlq import DeadLetterQueue
        dlq_instance = DeadLetterQueue(tmp_path / "dlq.json")
        dlq_instance.record(
            "dlq_id", source_file=str(source.resolve()),
            start_time=0.0, end_time=30.0, error="prior OOM", attempts=1,
        )

        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.dlq.DeadLetterQueue", return_value=dlq_instance), \
             patch("sentrysearch.chunker.chunk_video", return_value=[{
                 "chunk_path": str(chunk_path),
                 "source_file": str(source.resolve()),
                 "start_time": 0.0,
                 "end_time": 30.0,
             }]), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(cli, ["index", str(d), "--no-preprocess"])

        assert result.exit_code == 0
        mock_embedder.embed_video_chunk.assert_not_called()
        mock_store.add_chunk.assert_not_called()


    def test_index_skips_fully_indexed_file_without_chunking(self, runner, tmp_path):
        """When every expected chunk is already stored, chunk_video must not run."""
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "video.mp4"
        source.write_bytes(b"fake")

        mock_store = MagicMock()
        mock_store.has_chunk.return_value = True  # every chunk present
        mock_store.make_chunk_id.side_effect = lambda p, s: f"{p}:{s}"
        mock_store.get_stats.return_value = {
            "total_chunks": 3, "unique_source_files": 1,
        }

        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.chunker._get_video_duration", return_value=90.0), \
             patch("sentrysearch.chunker.chunk_video") as mock_chunk_video:
            result = runner.invoke(cli, ["index", str(d), "--no-preprocess"])

        assert result.exit_code == 0, result.output
        mock_chunk_video.assert_not_called()
        assert "already indexed" in result.output

    def test_index_retry_failed_reattempts_dlq_chunks(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "video.mp4"
        source.write_bytes(b"fake")

        chunk_dir = tmp_path / "chunks"
        chunk_dir.mkdir()
        chunk_path = chunk_dir / "chunk_000.mp4"
        chunk_path.write_bytes(b"chunk")

        mock_store = MagicMock()
        mock_store.has_chunk.return_value = False
        mock_store.make_chunk_id.return_value = "retry_id"
        mock_store.get_stats.return_value = {
            "total_chunks": 1, "unique_source_files": 1,
        }
        mock_embedder = MagicMock()
        mock_embedder.embed_video_chunk.return_value = [0.1] * 768

        from sentrysearch.dlq import DeadLetterQueue
        dlq_instance = DeadLetterQueue(tmp_path / "dlq.json")
        dlq_instance.record(
            "retry_id", source_file=str(source.resolve()),
            start_time=0.0, end_time=30.0, error="prior failure", attempts=1,
        )

        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.dlq.DeadLetterQueue", return_value=dlq_instance), \
             patch("sentrysearch.chunker.chunk_video", return_value=[{
                 "chunk_path": str(chunk_path),
                 "source_file": str(source.resolve()),
                 "start_time": 0.0,
                 "end_time": 30.0,
             }]), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(cli, [
                "index", str(d), "--no-preprocess", "--retry-failed",
            ])

        assert result.exit_code == 0, result.output
        mock_embedder.embed_video_chunk.assert_called_once()
        mock_store.add_chunk.assert_called_once()
        # Successful retry should remove the entry from the DLQ
        assert not dlq_instance.contains("retry_id")


    def test_index_overlap_equal_chunk_duration_errors(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        (d / "test.mp4").write_bytes(b"fake")
        result = runner.invoke(cli, [
            "index", str(d), "--chunk-duration", "5", "--overlap", "5",
        ])
        assert result.exit_code != 0
        assert "overlap" in result.output.lower()

    def test_index_overlap_greater_than_chunk_duration_errors(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        (d / "test.mp4").write_bytes(b"fake")
        result = runner.invoke(cli, [
            "index", str(d), "--chunk-duration", "5", "--overlap", "10",
        ])
        assert result.exit_code != 0
        assert "overlap" in result.output.lower()


class TestIndexDryRun:
    """`index --dry-run` previews chunks and cost without any indexing work."""

    def _invoke(self, runner, directory, durations, *extra):
        """Run index --dry-run with duration probes mocked and every
        indexing side effect wired to a mock we can assert against."""
        def fake_duration(path):
            return durations[os.path.basename(path)]

        with patch("sentrysearch.chunker._get_video_duration",
                   side_effect=fake_duration), \
             patch("sentrysearch.embedder.get_embedder") as mock_get, \
             patch("sentrysearch.store.SentryStore") as mock_store, \
             patch("sentrysearch.dlq.DeadLetterQueue") as mock_dlq, \
             patch("sentrysearch.chunker.chunk_video") as mock_chunk_video, \
             patch("sentrysearch.chunker.is_still_frame_chunk") as mock_still, \
             patch("sentrysearch.local_embedder.detect_default_model") as mock_detect:
            result = runner.invoke(
                cli, ["index", str(directory), "--dry-run", *extra],
            )
        return result, {
            "get_embedder": mock_get,
            "store": mock_store,
            "dlq": mock_dlq,
            "chunk_video": mock_chunk_video,
            "still": mock_still,
            "detect_default_model": mock_detect,
        }

    def _two_video_dir(self, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        # Names from the issue's example output.
        (d / "front_2024-01-15_14-30.mp4").write_bytes(b"fake")
        (d / "left_2024-01-15_14-30.mp4").write_bytes(b"fake")
        return d

    @staticmethod
    def _normalized_lines(output):
        return [" ".join(line.split()) for line in output.splitlines()]

    def test_dry_run_shows_per_file_chunks_and_totals(self, runner, tmp_path):
        d = self._two_video_dir(tmp_path)
        # 120s at the default 30s/5s settings -> 5 spans per file.
        result, _ = self._invoke(
            runner, d,
            {"front_2024-01-15_14-30.mp4": 120.0, "left_2024-01-15_14-30.mp4": 120.0},
        )
        assert result.exit_code == 0, result.output
        lines = self._normalized_lines(result.output)
        assert "front_2024-01-15_14-30.mp4 5 chunks (120s)" in lines
        assert "left_2024-01-15_14-30.mp4 5 chunks (120s)" in lines
        assert "10 chunks total, ~240s of footage" in result.output

    def test_dry_run_gemini_estimate_matches_readme_rate(self, runner, tmp_path):
        """Two 120s files = 10 chunks covering 280 billed span-seconds;
        280 * $0.00079 ~= $0.22 (the issue's $0.19 assumed footage
        seconds, but overlapping seconds are billed per chunk)."""
        d = self._two_video_dir(tmp_path)
        result, _ = self._invoke(
            runner, d,
            {"front_2024-01-15_14-30.mp4": 120.0, "left_2024-01-15_14-30.mp4": 120.0},
        )
        assert result.exit_code == 0, result.output
        assert "Estimated Gemini cost (upper bound): ~$0.22" in result.output
        assert "280 billed chunk-seconds" in result.output

    def test_dry_run_gemini_cost_labels_upper_bound_and_notes_skip(
        self, runner, tmp_path,
    ):
        d = tmp_path / "vids"
        d.mkdir()
        (d / "clip.mp4").write_bytes(b"fake")
        result, _ = self._invoke(runner, d, {"clip.mp4": 60.0})
        assert result.exit_code == 0, result.output
        assert "upper bound" in result.output
        assert "still-frame skipping is on by default" in result.output
        assert "--no-skip-still" in result.output

    def test_dry_run_cost_bills_overlapping_chunk_spans(
        self, runner, tmp_path,
    ):
        """Each chunk is uploaded and billed separately, so the estimate
        covers the sum of the chunk spans: 120s at 30s/5s makes 5 chunks
        covering 140 billed seconds -> ~$0.11 (140 * $0.00079), not the
        footage-based ~$0.09."""
        d = tmp_path / "vids"
        d.mkdir()
        (d / "clip.mp4").write_bytes(b"fake")
        result, _ = self._invoke(runner, d, {"clip.mp4": 120.0})
        assert result.exit_code == 0, result.output
        assert "5 chunks total, ~120s of footage" in result.output
        assert "140 billed chunk-seconds" in result.output
        assert "~$0.11" in result.output
        assert "~$0.09" not in result.output

    def test_dry_run_no_skip_still_hides_still_frame_note(
        self, runner, tmp_path,
    ):
        """With --no-skip-still, still-frame skipping is off, so the dry
        run must not describe it as enabled or advise disabling it."""
        d = tmp_path / "vids"
        d.mkdir()
        (d / "clip.mp4").write_bytes(b"fake")
        result, _ = self._invoke(runner, d, {"clip.mp4": 120.0}, "--no-skip-still")
        assert result.exit_code == 0, result.output
        assert "still-frame skipping" not in result.output
        assert "--no-skip-still" not in result.output
        assert "Estimated Gemini cost (upper bound): ~$0.11" in result.output

    def test_dry_run_notes_already_indexed_files_counted_fully(
        self, runner, tmp_path,
    ):
        """Already-indexed files are charged at full cost (no store
        lookup), so the output must say a real run may skip them."""
        d = self._two_video_dir(tmp_path)
        result, _ = self._invoke(
            runner, d,
            {"front_2024-01-15_14-30.mp4": 120.0, "left_2024-01-15_14-30.mp4": 120.0},
        )
        assert result.exit_code == 0, result.output
        assert "already-indexed files are included in this estimate" in result.output
        assert "a real indexing run may skip them" in result.output

    def test_dry_run_pluralizes_chunk_counts(self, runner, tmp_path):
        """A single chunk must print "1 chunk", never "1 chunks"."""
        d = tmp_path / "vids"
        d.mkdir()
        (d / "clip.mp4").write_bytes(b"fake")
        result, _ = self._invoke(runner, d, {"clip.mp4": 3.0})
        assert result.exit_code == 0, result.output
        lines = self._normalized_lines(result.output)
        assert "clip.mp4 1 chunk (3s)" in lines
        assert "1 chunk total, ~3s of footage" in result.output
        assert "1 chunks" not in result.output

    def test_dry_run_sub_cent_estimate_is_not_zero(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        (d / "clip.mp4").write_bytes(b"fake")
        result, _ = self._invoke(runner, d, {"clip.mp4": 3.0})
        assert result.exit_code == 0, result.output
        assert "1 chunk (3s)" in result.output
        # 3 * 0.00079 = 0.00237 — a positive bound must not print as $0.00
        assert "~$0.0024" in result.output

    def test_dry_run_respects_chunk_duration_and_overlap_flags(
        self, runner, tmp_path,
    ):
        d = tmp_path / "vids"
        d.mkdir()
        (d / "clip.mp4").write_bytes(b"fake")
        result, _ = self._invoke(
            runner, d, {"clip.mp4": 120.0},
            "--chunk-duration", "60", "--overlap", "0",
        )
        assert result.exit_code == 0, result.output
        assert "2 chunks (120s)" in result.output
        assert "2 chunks total" in result.output

    def test_dry_run_empty_directory(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result, mocks = self._invoke(runner, empty_dir, {})
        assert result.exit_code == 0, result.output
        assert "No supported video files found" in result.output
        for mock in mocks.values():
            mock.assert_not_called()

    def test_dry_run_ignores_unsupported_files(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        (d / "notes.txt").write_text("not a video")
        result, _ = self._invoke(runner, d, {})
        assert result.exit_code == 0, result.output
        assert "No supported video files found" in result.output

    def test_dry_run_accepts_a_single_file(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        clip = d / "clip.mp4"
        clip.write_bytes(b"fake")
        result, _ = self._invoke(runner, clip, {"clip.mp4": 60.0})
        assert result.exit_code == 0, result.output
        lines = self._normalized_lines(result.output)
        assert "clip.mp4 3 chunks (60s)" in lines
        assert "3 chunks total, ~60s of footage" in result.output

    def test_dry_run_local_backend_shows_no_dollar_estimate(
        self, runner, tmp_path,
    ):
        d = tmp_path / "vids"
        d.mkdir()
        (d / "clip.mp4").write_bytes(b"fake")
        result, _ = self._invoke(
            runner, d, {"clip.mp4": 120.0}, "--backend", "local", "--model", "qwen2b",
        )
        assert result.exit_code == 0, result.output
        assert "5 chunks total, ~120s of footage" in result.output
        assert "$" not in result.output
        assert "Estimated" not in result.output

    def test_dry_run_local_backend_skips_model_autodetect(
        self, runner, tmp_path,
    ):
        """Without --model, a real local index auto-detects a model (which
        imports torch). Dry-run must not."""
        d = tmp_path / "vids"
        d.mkdir()
        (d / "clip.mp4").write_bytes(b"fake")
        result, mocks = self._invoke(
            runner, d, {"clip.mp4": 60.0}, "--backend", "local",
        )
        assert result.exit_code == 0, result.output
        mocks["detect_default_model"].assert_not_called()
        assert "Auto-detected model" not in result.output
        assert "$" not in result.output

    def test_dry_run_qwen_cloud_shows_no_dollar_estimate(
        self, runner, tmp_path,
    ):
        d = tmp_path / "vids"
        d.mkdir()
        (d / "clip.mp4").write_bytes(b"fake")
        result, _ = self._invoke(
            runner, d, {"clip.mp4": 120.0}, "--backend", "qwen-cloud",
        )
        assert result.exit_code == 0, result.output
        assert "5 chunks total, ~120s of footage" in result.output
        assert "$" not in result.output

    def test_dry_run_mlx_shows_no_dollar_estimate(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        (d / "clip.mp4").write_bytes(b"fake")
        result, _ = self._invoke(runner, d, {"clip.mp4": 120.0}, "--backend", "mlx")
        assert result.exit_code == 0, result.output
        assert "5 chunks total, ~120s of footage" in result.output
        assert "$" not in result.output

    def test_dry_run_model_flag_implies_local_backend(self, runner, tmp_path):
        """--model alone infers the local backend, so no Gemini estimate."""
        d = tmp_path / "vids"
        d.mkdir()
        (d / "clip.mp4").write_bytes(b"fake")
        result, _ = self._invoke(runner, d, {"clip.mp4": 60.0}, "--model", "qwen2b")
        assert result.exit_code == 0, result.output
        assert "3 chunks total, ~60s of footage" in result.output
        assert "$" not in result.output

    def test_dry_run_skips_unreadable_file(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        (d / "a_bad.mp4").write_bytes(b"not a video")
        (d / "b_good.mp4").write_bytes(b"fake")

        def fake_duration(path):
            if os.path.basename(path).startswith("a_bad"):
                raise RuntimeError("ffprobe failed")
            return 60.0

        with patch("sentrysearch.chunker._get_video_duration",
                   side_effect=fake_duration), \
             patch("sentrysearch.chunker.probe_error",
                   return_value="Invalid data found when processing input"), \
             patch("sentrysearch.embedder.get_embedder") as mock_get:
            result = runner.invoke(cli, ["index", str(d), "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "a_bad.mp4  (unreadable: Invalid data found" in result.output
        assert "3 chunks total, ~60s of footage" in result.output
        mock_get.assert_not_called()

    def test_dry_run_labels_tesla_encrypted_clip(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        (d / "encrypted.mp4").write_bytes(b"not a video")

        with patch("sentrysearch.chunker._get_video_duration",
                   side_effect=RuntimeError("ffprobe failed")), \
             patch("sentrysearch.chunker.is_tesla_encrypted",
                   return_value=True):
            result = runner.invoke(cli, ["index", str(d), "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "encrypted.mp4  (Tesla-encrypted clip)" in result.output

    def test_dry_run_makes_zero_side_effects(self, runner, tmp_path):
        """The dry-run path must not construct an embedder, load a model,
        split files, initialize the index, or write anything."""
        d = self._two_video_dir(tmp_path)
        before = sorted(str(p) for p in tmp_path.rglob("*"))

        result, mocks = self._invoke(
            runner, d,
            {"front_2024-01-15_14-30.mp4": 120.0, "left_2024-01-15_14-30.mp4": 120.0},
        )

        assert result.exit_code == 0, result.output
        after = sorted(str(p) for p in tmp_path.rglob("*"))
        assert before == after
        mocks["get_embedder"].assert_not_called()
        mocks["store"].assert_not_called()
        mocks["dlq"].assert_not_called()
        mocks["chunk_video"].assert_not_called()
        mocks["still"].assert_not_called()
        mocks["detect_default_model"].assert_not_called()

    def test_dry_run_still_validates_options(self, runner, tmp_path):
        """Flag validation is unchanged when --dry-run is set."""
        d = tmp_path / "vids"
        d.mkdir()
        (d / "clip.mp4").write_bytes(b"fake")
        result = runner.invoke(cli, [
            "index", str(d), "--dry-run",
            "--chunk-duration", "5", "--overlap", "5",
        ])
        assert result.exit_code != 0
        assert "overlap" in result.output.lower()

        result = runner.invoke(cli, [
            "index", str(d), "--dry-run",
            "--model", "qwen2b", "--dashscope-model", "qwen3-vl-embedding",
        ])
        assert result.exit_code == 2

    def test_index_help_documents_dry_run(self, runner):
        result = runner.invoke(cli, ["index", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output


class TestShellCommand:
    @pytest.fixture(autouse=True)
    def _isolate_history(self, tmp_path):
        """Redirect shell history to tmp_path so tests don't touch real home."""
        with patch("sentrysearch.cli._HISTORY_PATH",
                   str(tmp_path / "history")):
            yield

    def _setup_mocks(self, runner, input_lines, total_chunks=5, search_results=None):
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {
            "total_chunks": total_chunks, "unique_source_files": 1,
        }
        mock_embedder = MagicMock()
        results_side_effect = search_results if search_results is not None else [[]]
        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.store.detect_index", return_value=(None, None)), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder) as mock_get, \
             patch("sentrysearch.search.search_footage", side_effect=results_side_effect) as mock_search:
            result = runner.invoke(cli, ["shell"], input=input_lines)
        return result, mock_get, mock_search, mock_store

    def test_shell_empty_index(self, runner):
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {
            "total_chunks": 0, "unique_source_files": 0,
        }
        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.store.detect_index", return_value=(None, None)), \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()):
            result = runner.invoke(cli, ["shell"])
        assert result.exit_code == 0
        assert "No indexed footage" in result.output

    def test_shell_loads_embedder_once(self, runner):
        fake_results = [
            {"source_file": "/v.mp4", "start_time": 0.0, "end_time": 30.0,
             "similarity_score": 0.8},
        ]
        result, mock_get, mock_search, _ = self._setup_mocks(
            runner,
            input_lines="car\nperson\n:quit\n",
            search_results=[fake_results, fake_results],
        )
        assert result.exit_code == 0, result.output
        # Model loaded exactly once despite multiple queries
        assert mock_get.call_count == 1
        assert mock_search.call_count == 2
        assert "#1 [0.80]" in result.output

    def test_shell_quit_command(self, runner):
        result, _, mock_search, _ = self._setup_mocks(
            runner, input_lines=":quit\n",
        )
        assert result.exit_code == 0
        mock_search.assert_not_called()

    def test_shell_n_command_updates_result_count(self, runner):
        result, _, mock_search, _ = self._setup_mocks(
            runner,
            input_lines=":n 10\ncar\n:quit\n",
            search_results=[[]],
        )
        assert result.exit_code == 0
        assert "n_results = 10" in result.output
        # The one actual query should have used n_results=10
        assert mock_search.call_args.kwargs["n_results"] == 10

    def test_shell_n_command_rejects_bad_input(self, runner):
        result, _, _, _ = self._setup_mocks(
            runner, input_lines=":n abc\n:quit\n",
        )
        assert result.exit_code == 0
        assert "usage: :n" in result.output

    def test_shell_unknown_command(self, runner):
        result, _, mock_search, _ = self._setup_mocks(
            runner, input_lines=":bogus\n:quit\n",
        )
        assert result.exit_code == 0
        assert "unknown command" in result.output
        mock_search.assert_not_called()

    def test_shell_eof_exits_cleanly(self, runner):
        result, _, _, _ = self._setup_mocks(
            runner, input_lines="",  # no input -> immediate EOF
        )
        assert result.exit_code == 0

    def test_shell_empty_results(self, runner):
        result, _, _, _ = self._setup_mocks(
            runner,
            input_lines="ghost\n:quit\n",
            search_results=[[]],
        )
        assert result.exit_code == 0
        assert "(no results)" in result.output

    def test_shell_low_confidence_warning(self, runner):
        low_conf = [
            {"source_file": "/v.mp4", "start_time": 0.0, "end_time": 30.0,
             "similarity_score": 0.10},
        ]
        result, _, _, _ = self._setup_mocks(
            runner,
            input_lines="obscure query\n:quit\n",
            search_results=[low_conf],
        )
        assert result.exit_code == 0
        assert "low confidence" in result.output

    def test_shell_search_error_kept_alive(self, runner):
        """A query failure should print the error but not kill the REPL."""
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {
            "total_chunks": 5, "unique_source_files": 1,
        }
        with patch("sentrysearch.store.SentryStore", return_value=mock_store), \
             patch("sentrysearch.store.detect_index", return_value=(None, None)), \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.search.search_footage",
                   side_effect=[RuntimeError("boom"), []]) as mock_search:
            result = runner.invoke(cli, ["shell"], input="a\nb\n:quit\n")
        assert result.exit_code == 0
        assert "Error: boom" in result.output
        assert mock_search.call_count == 2  # REPL survived first failure


class TestDlqCommand:
    def test_dlq_list_empty(self, runner, tmp_path):
        from sentrysearch.dlq import DeadLetterQueue
        empty = DeadLetterQueue(tmp_path / "dlq.json")
        with patch("sentrysearch.dlq.DeadLetterQueue", return_value=empty):
            result = runner.invoke(cli, ["dlq", "list"])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_dlq_list_shows_entries(self, runner, tmp_path):
        from sentrysearch.dlq import DeadLetterQueue
        q = DeadLetterQueue(tmp_path / "dlq.json")
        q.record(
            "id1", source_file="/tmp/clip.mp4",
            start_time=60.0, end_time=90.0,
            error="cuda out of memory", attempts=2,
        )
        with patch("sentrysearch.dlq.DeadLetterQueue", return_value=q):
            result = runner.invoke(cli, ["dlq", "list"])
        assert result.exit_code == 0
        assert "id1" in result.output
        assert "clip.mp4" in result.output
        assert "out of memory" in result.output
        assert "01:00" in result.output and "01:30" in result.output

    def test_dlq_clear(self, runner, tmp_path):
        from sentrysearch.dlq import DeadLetterQueue
        q = DeadLetterQueue(tmp_path / "dlq.json")
        q.record(
            "id1", source_file="/tmp/clip.mp4",
            start_time=0.0, end_time=30.0,
            error="oom", attempts=1,
        )
        with patch("sentrysearch.dlq.DeadLetterQueue", return_value=q):
            result = runner.invoke(cli, ["dlq", "clear"], input="y\n")
        assert result.exit_code == 0
        assert "Cleared 1" in result.output
        assert len(q) == 0


class TestIndexLocalFlags:
    def test_index_passes_model_to_embedder(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get:
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, [
                "index", str(empty_dir), "--backend", "local", "--model", "qwen2b",
            ])
            assert result.exit_code == 0
            mock_get.assert_called_once()
            assert mock_get.call_args[1]["model"] == "qwen2b"

    def test_index_passes_quantize_to_embedder(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get:
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, [
                "index", str(empty_dir), "--backend", "local", "--quantize",
            ])
            assert result.exit_code == 0
            mock_get.assert_called_once()
            assert mock_get.call_args[1]["quantize"] is True

    def test_index_passes_no_quantize_to_embedder(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get:
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, [
                "index", str(empty_dir), "--backend", "local", "--no-quantize",
            ])
            assert result.exit_code == 0
            mock_get.assert_called_once()
            assert mock_get.call_args[1]["quantize"] is False

    def test_index_auto_detects_model(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.local_embedder.detect_default_model", return_value="qwen2b"):
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, [
                "index", str(empty_dir), "--backend", "local",
            ])
            assert result.exit_code == 0
            assert mock_get.call_args[1]["model"] == "qwen2b"

    def test_index_model_implies_local_backend(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get:
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, [
                "index", str(empty_dir), "--model", "qwen2b",
            ])
            assert result.exit_code == 0
            # Should have inferred backend="local" from --model
            mock_get.assert_called_once_with("local", model="qwen2b", quantize=None, rpm=None)

    def test_index_passes_backend_and_model_to_store(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        (d / "test.mp4").write_bytes(b"fake")
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.local_embedder.detect_default_model", return_value="qwen8b"):
            mock_inst = MagicMock()
            mock_inst.is_indexed.return_value = True
            MockStore.return_value = mock_inst
            runner.invoke(cli, ["index", str(d), "--backend", "local"])
            MockStore.assert_called_once_with(backend="local", model="qwen8b")


class TestSearchShellTip:
    def test_tip_shown_for_local_backend(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.search.search_footage", return_value=[]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, [
                "search", "test", "--backend", "local", "--model", "qwen2b",
            ])
        assert result.exit_code == 0
        assert "shell" in result.output and "keeps the model loaded" in result.output

    def test_tip_not_shown_for_gemini_backend(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)), \
             patch("sentrysearch.search.search_footage", return_value=[]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["search", "test"])
        assert result.exit_code == 0
        assert "keeps the model loaded" not in result.output


class TestSearchLocalFlags:
    def test_search_passes_model_to_embedder(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.search.search_footage", return_value=[]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, [
                "search", "test query", "--backend", "local", "--model", "qwen2b",
            ])
            assert result.exit_code == 0
            mock_get.assert_called_with("local", model="qwen2b", quantize=None, rpm=None)

    def test_search_passes_quantize_to_embedder(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.store.detect_index", return_value=("local", "qwen8b")), \
             patch("sentrysearch.search.search_footage", return_value=[]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, [
                "search", "test query", "--backend", "local", "--quantize",
            ])
            assert result.exit_code == 0
            mock_get.assert_called_with("local", model="qwen8b", quantize=True, rpm=None)

    def test_search_model_implies_local_backend(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.search.search_footage", return_value=[]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, [
                "search", "test query", "--model", "qwen2b",
            ])
            assert result.exit_code == 0
            # --model qwen2b should imply --backend local
            mock_get.assert_called_with("local", model="qwen2b", quantize=None, rpm=None)

    def test_search_auto_detects_backend_and_model(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.store.detect_index", return_value=("local", "qwen2b")), \
             patch("sentrysearch.search.search_footage", return_value=[]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            # No --backend or --model flags
            result = runner.invoke(cli, ["search", "test query"])
            assert result.exit_code == 0
            mock_get.assert_called_with("local", model="qwen2b", quantize=None, rpm=None)
            MockStore.assert_called_once_with(backend="local", model="qwen2b")

    def test_search_explicit_mlx_ignores_local_index(self, runner):
        """With both a local and an mlx index, --backend mlx must resolve the
        mlx model, not the local one that detection finds first."""
        mlx_ref = "/m/Qwen3-VL-Embedding-2B-mlx-4bit"

        def both_indexes(db_path=None, backend=None):
            if backend == "mlx":
                return ("mlx", mlx_ref)
            if backend in (None, "local"):
                return ("local", "qwen2b")
            return (None, None)

        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.store.detect_index", side_effect=both_indexes), \
             patch("sentrysearch.search.search_footage", return_value=[]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["search", "test query", "--backend", "mlx"])
            assert result.exit_code == 0, result.output
            MockStore.assert_called_once_with(backend="mlx", model=mlx_ref)
            mock_get.assert_called_with("mlx", model=mlx_ref, quantize=None, rpm=None)

    def test_mlx_model_suggestion_keeps_the_backend_flag(self, runner):
        """--model alone implies --backend local, so the hint must say
        --backend mlx or following it would switch backends."""
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index", return_value=("mlx", "/m/real")):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 0}
            MockStore.return_value = inst
            result = runner.invoke(cli, [
                "search", "red car", "--backend", "mlx", "--model", "/m/typo",
            ])
            assert "--backend mlx --model /m/real" in result.output

    def test_search_wrong_model_shows_suggestion(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index", return_value=("local", "qwen2b")):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 0}
            MockStore.return_value = inst
            result = runner.invoke(cli, [
                "search", "red car", "--model", "qwen8b",
            ])
            assert result.exit_code == 0
            assert "qwen2b" in result.output
            assert "qwen8b" in result.output

    def test_search_save_top_calls_trim_top_results(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)), \
             patch("sentrysearch.search.search_footage", return_value=[
                 {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0, "similarity_score": 0.9},
                 {"source_file": "/a.mp4", "start_time": 30.0, "end_time": 60.0, "similarity_score": 0.8},
                 {"source_file": "/a.mp4", "start_time": 60.0, "end_time": 90.0, "similarity_score": 0.7},
             ]), \
             patch("sentrysearch.trimmer.trim_top_results", return_value=["/clip1.mp4", "/clip2.mp4", "/clip3.mp4"]) as mock_trim:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["search", "test", "--save-top", "3", "--no-trim"])
            assert result.exit_code == 0
            mock_trim.assert_called_once()
            assert mock_trim.call_args[1]["count"] == 3

    def test_search_save_top_rejects_zero(self, runner):
        result = runner.invoke(cli, ["search", "test", "--save-top", "0"])
        assert result.exit_code != 0


class TestSearchRerank:
    def test_search_rerank_caches_and_presents_reranked_order(self, runner):
        results = [
            {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0,
             "similarity_score": 0.7},
            {"source_file": "/b.mp4", "start_time": 30.0, "end_time": 60.0,
             "similarity_score": 0.9},
        ]
        presented_results = []

        def fake_rerank(query, search_results, reranker, *, candidate_dir, verbose):
            clip_path = os.path.join(candidate_dir, "candidate_001.mp4")
            with open(clip_path, "wb") as f:
                f.write(b"candidate")
            return [{**results[1], "_rerank_clip_path": clip_path}, results[0]]

        def fake_present(search_results, *args, **kwargs):
            presented_results.extend(search_results)
            assert os.path.isfile(search_results[0]["_rerank_clip_path"])

        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)), \
             patch("sentrysearch.search.search_footage", return_value=results) as mock_search, \
             patch("sentrysearch.gemini_reranker.GeminiReranker") as MockReranker, \
             patch("sentrysearch.reranker.rerank_results",
                   side_effect=fake_rerank) as mock_rerank, \
             patch("sentrysearch.cli._cache_last_search") as mock_cache, \
             patch("sentrysearch.cli._present_results",
                   side_effect=fake_present) as mock_present:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst

            result = runner.invoke(cli, ["search", "test", "--rerank"])

        assert result.exit_code == 0, result.output
        mock_search.assert_called_once()
        mock_rerank.assert_called_once_with(
            "test", results, MockReranker.return_value,
            candidate_dir=ANY,
            verbose=False,
        )
        mock_cache.assert_called_once_with([results[1], results[0]], query="test")
        mock_present.assert_called_once()
        assert presented_results[0]["_rerank_clip_path"].endswith(
            "candidate_001.mp4",
        )
        assert mock_present.call_args.kwargs["rerank_enabled"] is True

    def test_search_rerank_skips_empty_results(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)), \
             patch("sentrysearch.search.search_footage", return_value=[]), \
             patch("sentrysearch.gemini_reranker.GeminiReranker") as MockReranker, \
             patch("sentrysearch.reranker.rerank_results") as mock_rerank, \
             patch("sentrysearch.cli._cache_last_search") as mock_cache, \
             patch("sentrysearch.cli._present_results") as mock_present:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst

            result = runner.invoke(cli, ["search", "test", "--rerank"])

        assert result.exit_code == 0, result.output
        MockReranker.assert_not_called()
        mock_rerank.assert_not_called()
        mock_cache.assert_called_once_with([], query="test")
        mock_present.assert_called_once()
        assert mock_present.call_args.kwargs["rerank_enabled"] is True

    def test_search_rerank_missing_key_falls_back_to_embedding_results(
        self, runner,
    ):
        from sentrysearch.gemini_embedder import GeminiAPIKeyError

        results = [
            {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0,
             "similarity_score": 0.7},
        ]
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)), \
             patch("sentrysearch.search.search_footage", return_value=results), \
             patch("sentrysearch.gemini_reranker.GeminiReranker",
                   side_effect=GeminiAPIKeyError("GEMINI_API_KEY is not set.")) as MockReranker, \
             patch("sentrysearch.reranker.rerank_results") as mock_rerank, \
             patch("sentrysearch.cli._cache_last_search") as mock_cache, \
             patch("sentrysearch.cli._present_results") as mock_present:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst

            result = runner.invoke(cli, ["search", "test", "--rerank"])

        assert result.exit_code == 0, result.output
        assert "--rerank skipped: GEMINI_API_KEY is not set" in result.output
        MockReranker.assert_called_once()
        mock_rerank.assert_not_called()
        mock_cache.assert_called_once_with(results, query="test")
        mock_present.assert_called_once()
        assert mock_present.call_args[0][0] == results
        assert mock_present.call_args.kwargs["rerank_enabled"] is False

    def test_search_local_rerank_uses_qwen_and_resets_before_load(
        self, runner,
    ):
        results = [
            {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0,
             "similarity_score": 0.7},
        ]
        events = []

        def fake_reset():
            events.append("reset")

        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.embedder.reset_embedder",
                   side_effect=fake_reset), \
             patch("sentrysearch.store.detect_index",
                   return_value=("local", "qwen2b")), \
             patch("sentrysearch.search.search_footage", return_value=results), \
             patch("sentrysearch.qwen_reranker.QwenReranker") as MockQwen, \
             patch("sentrysearch.reranker.rerank_results",
                   return_value=results) as mock_rerank, \
             patch("sentrysearch.cli._cache_last_search"), \
             patch("sentrysearch.cli._present_results"):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            qwen = MockQwen.return_value

            def fake_load():
                events.append("load")
                return qwen

            qwen.load.side_effect = fake_load

            result = runner.invoke(cli, ["search", "test", "--rerank"])

        assert result.exit_code == 0, result.output
        MockQwen.assert_called_once_with(model_name="qwen2b", quantize=None)
        assert events[:2] == ["reset", "load"]
        mock_rerank.assert_called_once_with(
            "test", results, qwen,
            candidate_dir=ANY,
            verbose=False,
        )

    def test_search_local_rerank_load_error_falls_back_to_embedding_results(
        self, runner,
    ):
        from sentrysearch.local_embedder import LocalModelError

        results = [
            {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0,
             "similarity_score": 0.7},
        ]
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index",
                   return_value=("local", "qwen2b")), \
             patch("sentrysearch.search.search_footage", return_value=results), \
             patch("sentrysearch.qwen_reranker.QwenReranker",
                   side_effect=LocalModelError("missing local deps")), \
             patch("sentrysearch.reranker.rerank_results") as mock_rerank, \
             patch("sentrysearch.cli._cache_last_search") as mock_cache, \
             patch("sentrysearch.cli._present_results") as mock_present:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst

            result = runner.invoke(cli, ["search", "test", "--rerank"])

        assert result.exit_code == 0, result.output
        assert "--rerank skipped: missing local deps" in result.output
        mock_rerank.assert_not_called()
        mock_cache.assert_called_once_with(results, query="test")
        mock_present.assert_called_once()
        assert mock_present.call_args[0][0] == results
        assert mock_present.call_args.kwargs["rerank_enabled"] is False

    def test_search_dedupe_runs_before_rerank(self, runner):
        deduped = [
            {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0,
             "similarity_score": 0.9},
        ]
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)), \
             patch("sentrysearch.search.search_footage", return_value=deduped) as mock_search, \
             patch("sentrysearch.gemini_reranker.GeminiReranker") as MockReranker, \
             patch("sentrysearch.reranker.rerank_results", return_value=deduped) as mock_rerank, \
             patch("sentrysearch.cli._cache_last_search"), \
             patch("sentrysearch.cli._present_results"):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst

            result = runner.invoke(
                cli, ["search", "test", "--dedupe", "0.9", "--rerank"],
            )

        assert result.exit_code == 0, result.output
        assert mock_search.call_args.kwargs["dedupe_threshold"] == 0.9
        mock_rerank.assert_called_once_with(
            "test", deduped, MockReranker.return_value,
            candidate_dir=ANY,
            verbose=False,
        )

    def test_rerank_match_trims_despite_low_embedding_score(self, runner):
        results = [
            {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0,
             "similarity_score": 0.2},
        ]
        reranked = [
            {**results[0], "rerank_match": True, "rerank_confidence": 0.64},
        ]
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)), \
             patch("sentrysearch.search.search_footage", return_value=results), \
             patch("sentrysearch.gemini_reranker.GeminiReranker"), \
             patch("sentrysearch.reranker.rerank_results", return_value=reranked), \
             patch("sentrysearch.trimmer.trim_top_results",
                   return_value=["/clip.mp4"]) as mock_trim, \
             patch("sentrysearch.cli._cache_last_clip"), \
             patch("sentrysearch.cli._open_file"):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst

            result = runner.invoke(cli, ["search", "test", "--rerank"])

        assert result.exit_code == 0, result.output
        assert "No confident match found" not in result.output
        mock_trim.assert_called_once()

    def test_rerank_non_match_prompts_despite_high_embedding_score(self, runner):
        results = [
            {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0,
             "similarity_score": 0.95},
        ]
        reranked = [
            {**results[0], "rerank_match": False, "rerank_confidence": 0.12},
        ]
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)), \
             patch("sentrysearch.search.search_footage", return_value=results), \
             patch("sentrysearch.gemini_reranker.GeminiReranker"), \
             patch("sentrysearch.reranker.rerank_results", return_value=reranked), \
             patch("sentrysearch.trimmer.trim_top_results") as mock_trim:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst

            result = runner.invoke(
                cli, ["search", "test", "--rerank"], input="n\n",
            )

        assert result.exit_code == 0, result.output
        assert "No confident match found" in result.output
        assert "VLM match=false" in result.output
        assert "rerank confidence: 0.12" in result.output
        assert "embedding score: 0.95" in result.output
        mock_trim.assert_not_called()

    def test_rerank_missing_score_fields_falls_back_to_embedding_threshold(
        self, runner,
    ):
        results = [
            {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0,
             "similarity_score": 0.2},
        ]
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)), \
             patch("sentrysearch.search.search_footage", return_value=results), \
             patch("sentrysearch.gemini_reranker.GeminiReranker"), \
             patch("sentrysearch.reranker.rerank_results", return_value=results), \
             patch("sentrysearch.trimmer.trim_top_results") as mock_trim:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst

            result = runner.invoke(
                cli, ["search", "test", "--rerank"], input="n\n",
            )

        assert result.exit_code == 0, result.output
        assert "No confident match found" in result.output
        assert "best score: 0.20" in result.output
        mock_trim.assert_not_called()

    def test_non_rerank_low_embedding_score_still_prompts(self, runner):
        results = [
            {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0,
             "similarity_score": 0.2},
        ]
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)), \
             patch("sentrysearch.search.search_footage", return_value=results), \
             patch("sentrysearch.trimmer.trim_top_results") as mock_trim:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst

            result = runner.invoke(cli, ["search", "test"], input="n\n")

        assert result.exit_code == 0, result.output
        assert "No confident match found" in result.output
        assert "best score: 0.20" in result.output
        mock_trim.assert_not_called()


class TestLastClipCache:
    @pytest.fixture(autouse=True)
    def _isolated_cache(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache" / "last_clip.json"
        monkeypatch.setattr(
            "sentrysearch._toolkit_cache._cache_path", lambda: cache_file,
        )
        return cache_file

    def _read_cache(self, cache_file):
        import json
        return json.loads(cache_file.read_text())

    def test_search_save_top_1_writes_cache(self, runner, _isolated_cache):
        results = [
            {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0,
             "similarity_score": 0.9},
        ]
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)), \
             patch("sentrysearch.search.search_footage", return_value=results), \
             patch("sentrysearch.trimmer.trim_top_results",
                   return_value=["/tmp/clip1.mp4"]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["search", "test", "--save-top", "1"])

        assert result.exit_code == 0, result.output
        assert _isolated_cache.is_file()
        data = self._read_cache(_isolated_cache)
        assert data["path"] == os.path.abspath("/tmp/clip1.mp4")
        assert data["saved_by"] == "sentrysearch"
        assert "Saved clip path cached for sentryblur" in result.output

    def test_search_save_top_3_caches_rank_1(self, runner, _isolated_cache):
        results = [
            {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0, "similarity_score": 0.9},
            {"source_file": "/a.mp4", "start_time": 30.0, "end_time": 60.0, "similarity_score": 0.8},
            {"source_file": "/a.mp4", "start_time": 60.0, "end_time": 90.0, "similarity_score": 0.7},
        ]
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)), \
             patch("sentrysearch.search.search_footage", return_value=results), \
             patch("sentrysearch.trimmer.trim_top_results",
                   return_value=["/tmp/rank1.mp4", "/tmp/rank2.mp4", "/tmp/rank3.mp4"]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["search", "test", "--save-top", "3"])

        assert result.exit_code == 0, result.output
        data = self._read_cache(_isolated_cache)
        assert data["path"] == os.path.abspath("/tmp/rank1.mp4")

    def test_img_save_top_writes_cache(self, runner, tmp_path, _isolated_cache):
        img_path = tmp_path / "q.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0")
        results = [
            {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0,
             "similarity_score": 0.9},
        ]
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)), \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.search.search_footage_by_image", return_value=results), \
             patch("sentrysearch.trimmer.trim_top_results",
                   return_value=["/tmp/img_clip.mp4"]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["img", str(img_path), "--save-top", "1"])

        assert result.exit_code == 0, result.output
        data = self._read_cache(_isolated_cache)
        assert data["path"] == os.path.abspath("/tmp/img_clip.mp4")

    def test_overlay_writes_cache(self, runner, tmp_path, _isolated_cache):
        video = tmp_path / "in.mp4"
        video.write_bytes(b"fake")
        out = tmp_path / "out.mp4"

        with patch("sentrysearch.chunker._get_video_duration", return_value=10.0), \
             patch("sentrysearch.cli._apply_overlay_to_clip", return_value=True), \
             patch("sentrysearch.cli._open_file"):
            # _apply_overlay_to_clip is mocked to True; overlay() then renames
            # the per-source default to `output`, which won't exist. Patch
            # os.path.isfile to skip that branch.
            with patch("sentrysearch.cli.os.path.isfile", return_value=False):
                result = runner.invoke(cli, [
                    "overlay", str(video), "-o", str(out),
                ])

        assert result.exit_code == 0, result.output
        data = self._read_cache(_isolated_cache)
        assert data["path"] == str(out)
        assert data["saved_by"] == "sentrysearch"

    def test_cache_failure_does_not_fail_command(self, runner, monkeypatch, _isolated_cache):
        results = [
            {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0,
             "similarity_score": 0.9},
        ]

        def boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(
            "sentrysearch._toolkit_cache.write_last_clip", boom,
        )

        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)), \
             patch("sentrysearch.search.search_footage", return_value=results), \
             patch("sentrysearch.trimmer.trim_top_results",
                   return_value=["/tmp/clip.mp4"]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["search", "test", "--save-top", "1"])

        assert result.exit_code == 0
        assert "warning" in result.output.lower()
        assert "disk full" in result.output


class TestImgCommand:
    def test_img_empty_index(self, runner, tmp_path):
        img_path = tmp_path / "q.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0")
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index", return_value=(None, None)):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 0}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["img", str(img_path)])
        assert result.exit_code == 0
        assert "No indexed footage" in result.output

    def test_img_calls_search_by_image(self, runner, tmp_path):
        img_path = tmp_path / "q.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0")
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)), \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.search.search_footage_by_image", return_value=[
                 {"source_file": "/a.mp4", "start_time": 0.0, "end_time": 30.0,
                  "similarity_score": 0.85},
             ]) as mock_search:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["img", str(img_path), "--no-trim"])
        assert result.exit_code == 0, result.output
        mock_search.assert_called_once()
        assert "0.85" in result.output
        assert "a.mp4" in result.output

    def test_img_passes_model_to_embedder(self, runner, tmp_path):
        img_path = tmp_path / "q.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0")
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get, \
             patch("sentrysearch.search.search_footage_by_image", return_value=[]):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst
            result = runner.invoke(cli, [
                "img", str(img_path), "--backend", "local", "--model", "qwen2b",
                "--no-trim",
            ])
        assert result.exit_code == 0, result.output
        mock_get.assert_called_with("local", model="qwen2b", quantize=None, rpm=None)

    def test_img_missing_file_errors(self, runner):
        result = runner.invoke(cli, ["img", "/nonexistent/x.jpg"])
        assert result.exit_code != 0


class TestHandleError:
    def test_local_model_error(self, runner):
        from sentrysearch.local_embedder import LocalModelError

        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index", return_value=("local", "qwen8b")):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst

            with patch(
                "sentrysearch.embedder.get_embedder",
                side_effect=LocalModelError("no torch"),
            ):
                result = runner.invoke(cli, ["search", "test query", "--backend", "local"])
                assert result.exit_code == 1
                assert "no torch" in result.output

    def test_backend_mismatch_error(self, runner):
        from sentrysearch.store import BackendMismatchError

        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index", return_value=("local", "qwen8b")):
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 5}
            MockStore.return_value = inst

            with patch(
                "sentrysearch.embedder.get_embedder",
                side_effect=BackendMismatchError("built with gemini"),
            ):
                result = runner.invoke(cli, ["search", "test", "--backend", "local"])
                assert result.exit_code == 1
                assert "gemini" in result.output


class TestResetCommand:
    def test_reset_empty_index(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 0, "unique_source_files": 0, "source_files": [],
            }
            MockStore.return_value = inst
            result = runner.invoke(cli, ["reset", "--yes"])
            assert result.exit_code == 0
            assert "already empty" in result.output.lower()

    def test_reset_removes_all(self, runner):
        with patch("sentrysearch.store.reset_indexes", return_value={
            "total_collections": 1,
            "total_chunks": 10,
            "total_files": 2,
        }):
            result = runner.invoke(cli, ["reset", "--yes"])
            assert result.exit_code == 0
            assert "10" in result.output


class TestRemoveCommand:
    def test_remove_matching_file(self, runner):
        with patch("sentrysearch.store.remove_files", return_value={
            "total_removed": 5, "total_files": 1
        }):
            result = runner.invoke(cli, ["remove", "video1"])
            assert result.exit_code == 0
            assert "Removed 5 chunks" in result.output

    def test_remove_no_match(self, runner):
        with patch("sentrysearch.store.remove_files", return_value={
            "total_removed": 0, "total_files": 0
        }):
            result = runner.invoke(cli, ["remove", "nonexistent"])
            assert result.exit_code == 0
            assert "matching" in result.output.lower()

    def test_remove_empty_index(self, runner):
        with patch("sentrysearch.store.remove_files", return_value={
            "total_removed": 0, "total_files": 0
        }):
            result = runner.invoke(cli, ["remove", "video1"])
            assert result.exit_code == 0
            assert "matching" in result.output.lower()


class TestRpmFlag:
    """--rpm reaches the embedder factory (issue #85)."""

    def test_index_forwards_rpm(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get:
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, ["index", str(empty_dir), "--rpm", "10"])
            assert result.exit_code == 0
            assert mock_get.call_args[1]["rpm"] == 10

    def test_index_rejects_zero_rpm(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = runner.invoke(cli, ["index", str(empty_dir), "--rpm", "0"])
        assert result.exit_code != 0

    def test_rpm_defaults_to_none_when_absent(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()) as mock_get:
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, ["index", str(empty_dir)])
            assert result.exit_code == 0
            assert mock_get.call_args[1]["rpm"] is None

    @pytest.mark.parametrize("command", ["index", "search", "img", "highlights", "shell"])
    def test_rpm_exposed_on_cloud_commands(self, runner, command):
        result = runner.invoke(cli, [command, "--help"])
        assert result.exit_code == 0
        assert "--rpm" in result.output


class TestBackendThreshold:
    """Scores aren't comparable across embedding models, so the default
    low-confidence cutoff depends on the backend."""

    def _search(self, runner, backend, score, *extra):
        hit = [{"source_file": "/v.mp4", "start_time": 0.0, "end_time": 30.0,
                "similarity_score": score}]
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index", return_value=(backend, "m")), \
             patch("sentrysearch.search.search_footage", return_value=hit):
            MockStore.return_value.get_stats.return_value = {"total_chunks": 5}
            return runner.invoke(cli, ["search", "q", "--no-trim", *extra])

    def test_mlx_default_accepts_a_score_that_gemini_flags(self, runner):
        assert "low confidence" not in self._search(runner, "mlx", 0.38).output
        assert "low confidence" in self._search(runner, "gemini", 0.38).output

    def test_mlx_default_still_flags_unrelated_queries(self, runner):
        assert "low confidence" in self._search(runner, "mlx", 0.30).output

    def test_explicit_threshold_wins(self, runner):
        result = self._search(runner, "mlx", 0.38, "--threshold", "0.5")
        assert "low confidence" in result.output

    def test_resolve_threshold(self):
        from sentrysearch.cli import _resolve_threshold

        assert _resolve_threshold(None, "mlx") == 0.35
        assert _resolve_threshold(None, "gemini") == 0.41
        assert _resolve_threshold(None, "local") == 0.41
        assert _resolve_threshold(0.2, "mlx") == 0.2
        assert _resolve_threshold(0.0, "gemini") == 0.0

    def test_help_documents_both_defaults(self, runner):
        out = runner.invoke(cli, ["search", "--help"]).output
        assert "0.41" in out and "0.35" in out


class TestSearchDedupeDefault:
    def _run(self, runner, *args):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)), \
             patch("sentrysearch.search.search_footage", return_value=[]) as mock_search:
            MockStore.return_value.get_stats.return_value = {"total_chunks": 5}
            result = runner.invoke(cli, ["search", "q", "--no-trim", *args])
        assert result.exit_code == 0, result.output
        return mock_search.call_args.kwargs["dedupe_threshold"]

    def test_search_dedupes_by_default(self, runner):
        assert self._run(runner) == 0.9

    def test_search_dedupe_can_be_turned_off(self, runner):
        assert self._run(runner, "--dedupe", "1") == 1.0

    def test_img_and_shell_share_the_default(self, runner):
        for cmd in ("img", "shell"):
            out = runner.invoke(cli, [cmd, "--help"]).output
            assert "--dedupe" in out and "0.9" in out
