"""Integration tests for CLI commands."""

import os
from unittest.mock import patch

from click.testing import CliRunner

from sentrysearch.cli import cli


class TestIndexWithAudio:
    """Test that indexing produces both visual and text embeddings."""

    @patch("sentrysearch.embedder.embed_text", return_value=[0.5] * 768)
    @patch("sentrysearch.embedder.embed_video_chunk", return_value=[0.5] * 768)
    @patch("sentrysearch.transcriber.transcribe_chunk", return_value="hello world test speech")
    def test_index_creates_text_chunks(self, mock_transcribe, mock_embed_video,
                                        mock_embed_text, sample_video, tmp_dir):
        from sentrysearch.store import SentryStore

        db_path = os.path.join(tmp_dir, "db")
        store = SentryStore(db_path=db_path)

        with patch("sentrysearch.store.SentryStore", return_value=store):
            runner = CliRunner()
            result = runner.invoke(cli, ["index", sample_video])
            assert result.exit_code == 0, f"CLI failed: {result.output}"

            stats = store.get_stats()
            assert stats["total_chunks"] >= 1
            assert stats["text_chunks"] >= 1

    @patch("sentrysearch.embedder.embed_video_chunk", return_value=[0.5] * 768)
    def test_index_no_audio_skips_transcription(self, mock_embed_video,
                                                  sample_video, tmp_dir):
        from sentrysearch.store import SentryStore

        db_path = os.path.join(tmp_dir, "db")
        store = SentryStore(db_path=db_path)

        with patch("sentrysearch.store.SentryStore", return_value=store):
            runner = CliRunner()
            result = runner.invoke(cli, ["index", "--no-audio", sample_video])
            assert result.exit_code == 0, f"CLI failed: {result.output}"

            stats = store.get_stats()
            assert stats["text_chunks"] == 0
