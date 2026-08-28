"""Tests for sentrysearch.store."""

import math

import pytest

from sentrysearch.store import _make_chunk_id


class TestMakeChunkId:
    def test_deterministic(self):
        id1 = _make_chunk_id("video.mp4", 30.0)
        id2 = _make_chunk_id("video.mp4", 30.0)
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        id1 = _make_chunk_id("video.mp4", 30.0)
        id2 = _make_chunk_id("video.mp4", 60.0)
        id3 = _make_chunk_id("other.mp4", 30.0)
        assert id1 != id2
        assert id1 != id3

    def test_returns_hex_string(self):
        cid = _make_chunk_id("test.mp4", 0.0)
        assert len(cid) == 16
        int(cid, 16)  # should not raise


def _make_embedding(seed: float = 1.0, dim: int = 768) -> list[float]:
    vec = [math.sin(seed + i * 0.1) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


class TestSentryStore:
    def test_empty_store_stats(self, tmp_store):
        stats = tmp_store.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["unique_source_files"] == 0
        assert stats["source_files"] == []

    def test_empty_store_search(self, tmp_store):
        results = tmp_store.search(_make_embedding(), n_results=5)
        assert results == []

    def test_add_chunk_and_retrieve(self, tmp_store):
        emb = _make_embedding(seed=1.0)
        tmp_store.add_chunk(
            chunk_id="chunk001",
            embedding=emb,
            metadata={
                "source_file": "/path/to/video.mp4",
                "start_time": 0.0,
                "end_time": 30.0,
            },
        )
        stats = tmp_store.get_stats()
        assert stats["total_chunks"] == 1
        assert stats["unique_source_files"] == 1
        assert "/path/to/video.mp4" in stats["source_files"]

    def test_search_returns_sorted_results(self, tmp_store):
        emb_a = _make_embedding(seed=1.0)
        emb_b = _make_embedding(seed=100.0)
        tmp_store.add_chunk("a", emb_a, {
            "source_file": "vid.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        tmp_store.add_chunk("b", emb_b, {
            "source_file": "vid.mp4", "start_time": 30.0, "end_time": 60.0,
        })
        results = tmp_store.search(emb_a, n_results=2)
        assert len(results) == 2
        assert results[0]["start_time"] == 0.0
        assert results[0]["score"] > results[1]["score"]

    def test_add_chunks_batch(self, tmp_store):
        chunks = [
            {
                "source_file": "batch.mp4",
                "start_time": float(i * 30),
                "end_time": float((i + 1) * 30),
                "embedding": _make_embedding(seed=float(i)),
            }
            for i in range(5)
        ]
        tmp_store.add_chunks(chunks)
        stats = tmp_store.get_stats()
        assert stats["total_chunks"] == 5
        assert stats["unique_source_files"] == 1

    def test_upsert_overwrites(self, tmp_store):
        emb1 = _make_embedding(seed=1.0)
        emb2 = _make_embedding(seed=2.0)
        meta = {"source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0}
        tmp_store.add_chunk("same_id", emb1, meta)
        tmp_store.add_chunk("same_id", emb2, meta)
        assert tmp_store.get_stats()["total_chunks"] == 1

    def test_has_chunk(self, tmp_store):
        cid = tmp_store.make_chunk_id("v.mp4", 30.0)
        assert not tmp_store.has_chunk(cid)
        tmp_store.add_chunk(cid, _make_embedding(), {
            "source_file": "v.mp4", "start_time": 30.0, "end_time": 60.0,
        })
        assert tmp_store.has_chunk(cid)
        assert not tmp_store.has_chunk("nonexistent_id")

    def test_is_indexed(self, tmp_store):
        assert not tmp_store.is_indexed("nonexistent.mp4")
        tmp_store.add_chunk("x", _make_embedding(), {
            "source_file": "found.mp4", "start_time": 0.0, "end_time": 10.0,
        })
        assert tmp_store.is_indexed("found.mp4")
        assert not tmp_store.is_indexed("other.mp4")

    def test_remove_file(self, tmp_store):
        emb = _make_embedding()
        tmp_store.add_chunk("a1", emb, {
            "source_file": "keep.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        tmp_store.add_chunk("b1", emb, {
            "source_file": "drop.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        tmp_store.add_chunk("b2", emb, {
            "source_file": "drop.mp4", "start_time": 30.0, "end_time": 60.0,
        })
        assert tmp_store.get_stats()["total_chunks"] == 3
        removed = tmp_store.remove_file("drop.mp4")
        assert removed == 2
        assert tmp_store.get_stats()["total_chunks"] == 1
        assert tmp_store.is_indexed("keep.mp4")
        assert not tmp_store.is_indexed("drop.mp4")

    def test_remove_file_nonexistent(self, tmp_store):
        removed = tmp_store.remove_file("nope.mp4")
        assert removed == 0

    def test_self_similarity_near_one(self, tmp_store):
        emb = _make_embedding(seed=42.0)
        tmp_store.add_chunk("self", emb, {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        results = tmp_store.search(emb, n_results=1)
        assert len(results) == 1
        assert results[0]["score"] > 0.99


# ---------------------------------------------------------------------------
# Backend support
# ---------------------------------------------------------------------------

class TestStoreBackend:
    def test_default_backend_is_gemini(self, tmp_store):
        assert tmp_store.get_backend() == "gemini"

    def test_local_backend_collection_name(self, tmp_path):
        from sentrysearch.store import SentryStore

        store = SentryStore(db_path=tmp_path / "db", backend="local")
        assert store.collection.name == "dashcam_chunks_local"

    def test_local_model_collection_name(self, tmp_path):
        from sentrysearch.store import SentryStore

        store = SentryStore(db_path=tmp_path / "db", backend="local", model="qwen2b")
        assert store.collection.name == "dashcam_chunks_local_qwen2b"

    def test_gemini_backend_collection_name(self, tmp_path):
        from sentrysearch.store import SentryStore

        store = SentryStore(db_path=tmp_path / "db", backend="gemini")
        assert store.collection.name == "dashcam_chunks"

    def test_qwen_cloud_backend_collection_name(self, tmp_path):
        from sentrysearch.store import SentryStore

        store = SentryStore(
            db_path=tmp_path / "db",
            backend="qwen-cloud",
            model="qwen3-vl-embedding",
        )
        assert store.collection.name == "dashcam_chunks_qwen_cloud_qwen3-vl-embedding"

    def test_qwen_cloud_collection_slug_sanitizes_special_chars(self, tmp_path):
        from sentrysearch.store import SentryStore

        raw = "org/model:v1@special"
        store = SentryStore(db_path=tmp_path / "db", backend="qwen-cloud", model=raw)
        suffix = store.collection.name.removeprefix("dashcam_chunks_qwen_cloud_")
        assert "/" not in suffix and ":" not in suffix and "@" not in suffix
        assert all(c.isalnum() or c in "._-" for c in suffix)

    def test_orca_backend_collection_name(self, tmp_path):
        from sentrysearch.store import SentryStore

        store = SentryStore(
            db_path=tmp_path / "db",
            backend="orca",
            model="google/gemini-embedding-2-preview",
        )
        assert store.collection.name == (
            "dashcam_chunks_orca_google_gemini-embedding-2-preview"
        )

    def test_orca_backend_default_collection_name(self, tmp_path):
        from sentrysearch.store import SentryStore

        store = SentryStore(db_path=tmp_path / "db", backend="orca")
        assert store.collection.name == (
            "dashcam_chunks_orca_google_gemini-embedding-2-preview"
        )

    def test_backends_use_separate_collections(self, tmp_path):
        from sentrysearch.store import SentryStore

        db = tmp_path / "db"
        gemini_store = SentryStore(db_path=db, backend="gemini")
        local_store = SentryStore(db_path=db, backend="local", model="qwen8b")

        emb = _make_embedding(seed=1.0)
        gemini_store.add_chunk("g1", emb, {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })

        assert gemini_store.get_stats()["total_chunks"] == 1
        assert local_store.get_stats()["total_chunks"] == 0

    def test_models_use_separate_collections(self, tmp_path):
        from sentrysearch.store import SentryStore

        db = tmp_path / "db"
        store_8b = SentryStore(db_path=db, backend="local", model="qwen8b")
        store_2b = SentryStore(db_path=db, backend="local", model="qwen2b")

        emb = _make_embedding(seed=1.0)
        store_2b.add_chunk("c1", emb, {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })

        assert store_2b.get_stats()["total_chunks"] == 1
        assert store_8b.get_stats()["total_chunks"] == 0

    def test_get_model(self, tmp_path):
        from sentrysearch.store import SentryStore

        store = SentryStore(db_path=tmp_path / "db", backend="local", model="qwen2b")
        assert store.get_model() == "qwen2b"

    def test_get_model_none_for_gemini(self, tmp_store):
        assert tmp_store.get_model() is None

    def test_check_backend_matching(self, tmp_store):
        tmp_store.check_backend("gemini")  # should not raise

    def test_check_backend_mismatch_raises(self, tmp_store):
        from sentrysearch.store import BackendMismatchError

        with pytest.raises(BackendMismatchError, match="gemini"):
            tmp_store.check_backend("local")


class TestDetectBackend:
    def test_empty_db_returns_none(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_backend

        SentryStore(db_path=tmp_path / "db", backend="gemini")
        assert detect_backend(tmp_path / "db") is None

    def test_detects_gemini(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_backend

        store = SentryStore(db_path=tmp_path / "db", backend="gemini")
        store.add_chunk("c1", _make_embedding(), {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        assert detect_backend(tmp_path / "db") == "gemini"

    def test_detects_local(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_backend

        store = SentryStore(db_path=tmp_path / "db", backend="local", model="qwen8b")
        store.add_chunk("c1", _make_embedding(), {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        assert detect_backend(tmp_path / "db") == "local"

    def test_nonexistent_path_returns_none(self, tmp_path):
        from sentrysearch.store import detect_backend

        assert detect_backend(tmp_path / "no_such_dir") is None


class TestDetectIndex:
    def test_empty_db(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_index

        SentryStore(db_path=tmp_path / "db", backend="gemini")
        assert detect_index(tmp_path / "db") == (None, None)

    def test_detects_gemini(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_index

        store = SentryStore(db_path=tmp_path / "db", backend="gemini")
        store.add_chunk("c1", _make_embedding(), {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        assert detect_index(tmp_path / "db") == ("gemini", None)

    def test_detects_local_model(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_index

        store = SentryStore(db_path=tmp_path / "db", backend="local", model="qwen2b")
        store.add_chunk("c1", _make_embedding(), {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        assert detect_index(tmp_path / "db") == ("local", "qwen2b")

    def test_legacy_local_treated_as_qwen8b(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_index

        # Legacy collection: no model specified
        store = SentryStore(db_path=tmp_path / "db", backend="local")
        store.add_chunk("c1", _make_embedding(), {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        assert detect_index(tmp_path / "db") == ("local", "qwen8b")

    def test_nonexistent_path(self, tmp_path):
        from sentrysearch.store import detect_index

        assert detect_index(tmp_path / "no_such_dir") == (None, None)

    def test_gemini_preferred_over_local(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_index

        db = tmp_path / "db"
        emb = _make_embedding()
        gemini = SentryStore(db_path=db, backend="gemini")
        gemini.add_chunk("g1", emb, {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        local = SentryStore(db_path=db, backend="local", model="qwen2b")
        local.add_chunk("l1", emb, {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        assert detect_index(db) == ("gemini", None)

    def test_gemini_preferred_over_qwen_cloud(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_index

        db = tmp_path / "db"
        emb = _make_embedding()
        gemini = SentryStore(db_path=db, backend="gemini")
        gemini.add_chunk("g1", emb, {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        qc = SentryStore(
            db_path=db, backend="qwen-cloud", model="qwen3-vl-embedding",
        )
        qc.add_chunk("q1", emb, {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        assert detect_index(db) == ("gemini", None)

    def test_detects_qwen_cloud_model(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_index

        store = SentryStore(
            db_path=tmp_path / "db",
            backend="qwen-cloud",
            model="qwen3-vl-embedding",
        )
        store.add_chunk("c1", _make_embedding(), {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        assert detect_index(tmp_path / "db") == ("qwen-cloud", "qwen3-vl-embedding")

    def test_detects_orca_backend(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_index

        store = SentryStore(
            db_path=tmp_path / "db",
            backend="orca",
            model="google/gemini-embedding-2-preview",
        )
        store.add_chunk("c1", _make_embedding(), {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        assert detect_index(tmp_path / "db") == (
            "orca", "google/gemini-embedding-2-preview",
        )

    def test_gemini_preferred_over_orca(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_index

        db = tmp_path / "db"
        emb = _make_embedding()
        gemini = SentryStore(db_path=db, backend="gemini")
        gemini.add_chunk("g1", emb, {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        orca = SentryStore(
            db_path=db, backend="orca", model="google/gemini-embedding-2-preview",
        )
        orca.add_chunk("o1", emb, {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        assert detect_index(db) == ("gemini", None)
