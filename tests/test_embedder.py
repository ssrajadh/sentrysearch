"""Tests for embedder functions."""

from unittest.mock import MagicMock, patch


class TestEmbedText:
    """Tests for embed_text() document embedding."""

    @patch("sentrysearch.embedder._get_client")
    @patch("sentrysearch.embedder._limiter")
    def test_returns_768_dim_vector(self, mock_limiter, mock_get_client):
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=[0.1] * 768)]
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        from sentrysearch.embedder import embed_text

        result = embed_text("hello world")
        assert len(result) == 768

    @patch("sentrysearch.embedder._get_client")
    @patch("sentrysearch.embedder._limiter")
    def test_uses_retrieval_document_task_type(self, mock_limiter, mock_get_client):
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=[0.1] * 768)]
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        from sentrysearch.embedder import embed_text

        embed_text("hello world")

        call_kwargs = mock_client.models.embed_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config.task_type == "RETRIEVAL_DOCUMENT"

    @patch("sentrysearch.embedder._get_client")
    @patch("sentrysearch.embedder._limiter")
    def test_returns_empty_list_for_empty_string(self, mock_limiter, mock_get_client):
        from sentrysearch.embedder import embed_text

        result = embed_text("")
        assert result == []
        mock_get_client.return_value.models.embed_content.assert_not_called()
