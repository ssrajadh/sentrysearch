"""Tests for sentrysearch.cli (Click CLI)."""

import os
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
             patch("sentrysearch.store.detect_index", return_value=(None, None)):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 0, "unique_source_files": 0, "source_files": [],
            }
            MockStore.return_value = inst
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 0
            assert "empty" in result.output.lower() or "0" in result.output

    def test_stats_with_data(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index", return_value=("local", "qwen2b")):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 10,
                "unique_source_files": 2,
                "source_files": ["/a/video1.mp4", "/b/video2.mp4"],
            }
            inst.get_backend.return_value = "local"
            MockStore.return_value = inst
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 0
            assert "10" in result.output
            assert "qwen2b" in result.output


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
            mock_get.assert_called_once_with("local", model="qwen2b", quantize=None)

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
            mock_get.assert_called_with("local", model="qwen2b", quantize=None)

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
            mock_get.assert_called_with("local", model="qwen8b", quantize=True)

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
            mock_get.assert_called_with("local", model="qwen2b", quantize=None)

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
            mock_get.assert_called_with("local", model="qwen2b", quantize=None)
            MockStore.assert_called_once_with(backend="local", model="qwen2b")

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
        mock_get.assert_called_with("local", model="qwen2b", quantize=None)

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
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 10, "unique_source_files": 2,
                "source_files": ["/a/v1.mp4", "/b/v2.mp4"],
            }
            inst.remove_file.return_value = 5
            MockStore.return_value = inst
            result = runner.invoke(cli, ["reset", "--yes"])
            assert result.exit_code == 0
            assert "10" in result.output
            assert inst.remove_file.call_count == 2


class TestRemoveCommand:
    def test_remove_matching_file(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 10, "unique_source_files": 2,
                "source_files": ["/a/video1.mp4", "/b/video2.mp4"],
            }
            inst.remove_file.return_value = 5
            MockStore.return_value = inst
            result = runner.invoke(cli, ["remove", "video1"])
            assert result.exit_code == 0
            assert "Removed 5 chunks" in result.output
            inst.remove_file.assert_called_once_with("/a/video1.mp4")

    def test_remove_no_match(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 10, "unique_source_files": 1,
                "source_files": ["/a/video1.mp4"],
            }
            MockStore.return_value = inst
            result = runner.invoke(cli, ["remove", "nonexistent"])
            assert result.exit_code == 0
            assert "No indexed files matching" in result.output

    def test_remove_empty_index(self, runner):
        with patch("sentrysearch.store.SentryStore") as MockStore, \
             patch("sentrysearch.store.detect_index", return_value=("gemini", None)):
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 0, "unique_source_files": 0, "source_files": [],
            }
            MockStore.return_value = inst
            result = runner.invoke(cli, ["remove", "anything"])
            assert result.exit_code == 0
            assert "empty" in result.output.lower()
