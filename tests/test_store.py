"""Tests for SentryStore text collection."""

import pytest


@pytest.fixture
def store(tmp_dir):
    from sentrysearch.store import SentryStore
    return SentryStore(db_path=tmp_dir)


class TestTextCollection:
    """Tests for text embedding storage and search."""

    def test_add_and_search_text_chunks(self, store):
        store.add_text_chunks([
            {
                "source_file": "/video.mp4",
                "start_time": 0.0,
                "end_time": 30.0,
                "embedding": [0.1] * 768,
                "transcript": "talking about vendor bills",
            },
            {
                "source_file": "/video.mp4",
                "start_time": 30.0,
                "end_time": 60.0,
                "embedding": [0.9] * 768,
                "transcript": "discussing design mockups",
            },
        ])

        results = store.search_text([0.9] * 768, n_results=2)
        assert len(results) == 2
        assert results[0]["start_time"] == 30.0

    def test_text_collection_stores_transcript(self, store):
        store.add_text_chunks([
            {
                "source_file": "/video.mp4",
                "start_time": 0.0,
                "end_time": 30.0,
                "embedding": [0.5] * 768,
                "transcript": "hello world",
            },
        ])

        results = store.search_text([0.5] * 768, n_results=1)
        assert results[0]["transcript"] == "hello world"

    def test_text_collection_empty_returns_empty(self, store):
        results = store.search_text([0.5] * 768, n_results=5)
        assert results == []

    def test_has_text_index(self, store):
        assert store.has_text_index() is False

        store.add_text_chunks([
            {
                "source_file": "/video.mp4",
                "start_time": 0.0,
                "end_time": 30.0,
                "embedding": [0.5] * 768,
                "transcript": "test",
            },
        ])

        assert store.has_text_index() is True

    def test_get_stats_includes_text_chunks(self, store):
        store.add_chunks([
            {
                "source_file": "/video.mp4",
                "start_time": 0.0,
                "end_time": 30.0,
                "embedding": [0.5] * 768,
            },
        ])
        store.add_text_chunks([
            {
                "source_file": "/video.mp4",
                "start_time": 0.0,
                "end_time": 30.0,
                "embedding": [0.5] * 768,
                "transcript": "test",
            },
        ])

        stats = store.get_stats()
        assert stats["total_chunks"] == 1
        assert stats["text_chunks"] == 1
