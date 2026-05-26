"""ChromaDB vector store."""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import chromadb


DEFAULT_DB_PATH = Path.home() / ".sentrysearch" / "db"

# Chroma collection names allow [a-zA-Z0-9._-] only.
_CHROMA_NAME_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def _chroma_collection_slug(name: str) -> str:
    """Sanitize *name* for use inside a ChromaDB collection name segment."""
    s = _CHROMA_NAME_UNSAFE.sub("_", (name or "").strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "default"


class BackendMismatchError(RuntimeError):
    """Raised when search backend/model doesn't match the indexed backend/model."""


def _collection_name(backend: str, model: str | None = None) -> str:
    """Return ChromaDB collection name for a backend and optional model."""
    if backend == "gemini":
        return "dashcam_chunks"
    if backend == "qwen-cloud":
        slug = _chroma_collection_slug(model or "qwen3-vl-embedding")
        return f"dashcam_chunks_qwen_cloud_{slug}"
    if model:
        return f"dashcam_chunks_local_{model}"
    # Legacy: local backend without model distinction
    return "dashcam_chunks_local"


def detect_index(db_path: str | Path | None = None) -> tuple[str | None, str | None]:
    """Return ``(backend, model)`` for the first index with data.

    Returns ``(None, None)`` when no index contains data.
    Checks gemini first, then DashScope ``qwen-cloud`` collections, then
    model-specific local collections, then the legacy ``dashcam_chunks_local``
    collection (treated as qwen8b).
    """
    db_path = str(db_path or DEFAULT_DB_PATH)
    if not Path(db_path).exists():
        return None, None
    client = chromadb.PersistentClient(path=db_path)
    existing = {c.name for c in client.list_collections()}

    # Gemini first (default / legacy)
    if "dashcam_chunks" in existing:
        col = client.get_collection("dashcam_chunks")
        if col.count() > 0:
            return "gemini", None

    # DashScope qwen-cloud (dashcam_chunks_qwen_cloud_<model>)
    for name in sorted(existing):
        if name.startswith("dashcam_chunks_qwen_cloud_"):
            col = client.get_collection(name)
            if col.count() > 0:
                meta = col.metadata or {}
                model = meta.get("embedding_model")
                if model is None:
                    model = name.removeprefix("dashcam_chunks_qwen_cloud_")
                return "qwen-cloud", model

    # Model-specific local collections (dashcam_chunks_local_<model>)
    for name in sorted(existing):
        if name.startswith("dashcam_chunks_local_"):
            col = client.get_collection(name)
            if col.count() > 0:
                meta = col.metadata or {}
                model = meta.get("embedding_model")
                if model is None:
                    model = name.removeprefix("dashcam_chunks_local_")
                return "local", model

    # Legacy local collection (no model suffix) — treat as qwen8b
    if "dashcam_chunks_local" in existing:
        col = client.get_collection("dashcam_chunks_local")
        if col.count() > 0:
            meta = col.metadata or {}
            return "local", meta.get("embedding_model", "qwen8b")

    return None, None


def detect_backend(db_path: str | Path | None = None) -> str | None:
    """Return the backend that has indexed data, or None if empty."""
    backend, _ = detect_index(db_path)
    return backend


def _make_chunk_id(source_file: str, start_time: float) -> str:
    """Deterministic chunk ID from source file + start time."""
    raw = f"{source_file}:{start_time}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class SentryStore:
    """Persistent vector store backed by ChromaDB."""

    def __init__(self, db_path: str | Path | None = None, backend: str = "gemini",
                 model: str | None = None):
        db_path = str(db_path or DEFAULT_DB_PATH)
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=db_path)
        self._backend = backend
        self._model = model
        # Separate collection per backend+model so incompatible vectors never mix.
        col_name = _collection_name(backend, model)
        metadata = {"hnsw:space": "cosine", "embedding_backend": backend}
        if model:
            metadata["embedding_model"] = model
        self._collection = self._client.get_or_create_collection(
            name=col_name,
            metadata=metadata,
        )

    @property
    def collection(self) -> chromadb.Collection:
        return self._collection

    def get_backend(self) -> str:
        """Return the backend this index was built with."""
        meta = self._collection.metadata or {}
        return meta.get("embedding_backend", "gemini")

    def get_model(self) -> str | None:
        """Return the model this index was built with, or None."""
        meta = self._collection.metadata or {}
        return meta.get("embedding_model")

    def check_backend(self, backend: str) -> None:
        """Raise BackendMismatchError if *backend* doesn't match the index."""
        indexed_backend = self.get_backend()
        if indexed_backend != backend:
            raise BackendMismatchError(
                f"This index was built with the {indexed_backend} backend. "
                f"Search with --backend {indexed_backend} or re-index with "
                f"--backend {backend}."
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunk(
        self,
        chunk_id: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        """Store a single chunk embedding with metadata.

        Required metadata keys: source_file, start_time, end_time.
        An indexed_at ISO timestamp is added automatically.
        """
        meta = {
            "source_file": metadata["source_file"],
            "start_time": float(metadata["start_time"]),
            "end_time": float(metadata["end_time"]),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        # Carry over any extra metadata the caller provides
        for key in metadata:
            if key not in meta and key != "embedding":
                meta[key] = metadata[key]

        self._collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            metadatas=[meta],
        )

    def add_chunks(self, chunks: list[dict]) -> None:
        """Batch-store chunks. Each dict must have 'embedding' and metadata keys."""
        now = datetime.now(timezone.utc).isoformat()
        ids = []
        embeddings = []
        metadatas = []

        for chunk in chunks:
            chunk_id = _make_chunk_id(chunk["source_file"], chunk["start_time"])
            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            metadatas.append({
                "source_file": chunk["source_file"],
                "start_time": float(chunk["start_time"]),
                "end_time": float(chunk["end_time"]),
                "indexed_at": now,
            })

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        include_embeddings: bool = False,
    ) -> list[dict]:
        """Return top N results with distances and metadata."""
        count = self._collection.count()
        if count == 0:
            return []

        include = ["metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
            include=include,
        )

        hits = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            hit = {
                "source_file": meta["source_file"],
                "start_time": meta["start_time"],
                "end_time": meta["end_time"],
                "score": 1.0 - distance,  # cosine distance → similarity
                "distance": distance,
            }
            if include_embeddings:
                hit["embedding"] = results["embeddings"][0][i]
            hits.append(hit)
        return hits

    def is_indexed(self, source_file: str) -> bool:
        """Check whether any chunks from source_file are already stored."""
        results = self._collection.get(
            where={"source_file": source_file},
            limit=1,
        )
        return len(results["ids"]) > 0

    def has_chunk(self, chunk_id: str) -> bool:
        """Check whether a specific chunk ID is already stored."""
        results = self._collection.get(ids=[chunk_id], limit=1)
        return len(results["ids"]) > 0

    def make_chunk_id(self, source_file: str, start_time: float) -> str:
        """Return the deterministic chunk ID used by this store."""
        return _make_chunk_id(source_file, start_time)

    def remove_file(self, source_file: str) -> int:
        """Remove all chunks for a given source file. Returns count removed."""
        results = self._collection.get(where={"source_file": source_file})
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def get_stats(self) -> dict:
        """Return store statistics."""
        total = self._collection.count()
        if total == 0:
            return {"total_chunks": 0, "unique_source_files": 0, "source_files": []}

        # Fetch all metadata (only the fields we need)
        all_meta = self._collection.get(include=["metadatas"])
        source_files = sorted({m["source_file"] for m in all_meta["metadatas"]})
        return {
            "total_chunks": total,
            "unique_source_files": len(source_files),
            "source_files": source_files,
        }
