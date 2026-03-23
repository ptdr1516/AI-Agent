"""FAISS-backed vector memory: embed conversation summaries, retrieve by similarity."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import faiss
import numpy as np

from core.config import settings
from core.logger import log


def _make_turn_summary(user_message: str, assistant_message: str, max_total: int = 500) -> str:
    u = " ".join(user_message.strip().split())[:240]
    a = " ".join(assistant_message.strip().split())[: max_total - len(u)]
    return f"User: {u} | Assistant: {a}"


def _get_dim() -> int:
    """Return embedding dimension from the shared singleton."""
    from rag.embeddings import get_embedding_model
    sample = get_embedding_model().embed_query("dimension check")
    return len(sample)


def _embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts using the shared singleton. Returns float32 L2-normalised array."""
    from rag.embeddings import get_embedding_model
    vecs = np.array(
        get_embedding_model().embed_documents(texts), dtype=np.float32
    )
    # L2-normalise for cosine similarity via IndexFlatIP
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


class VectorMemoryStore:
    """
    Parallel lists (text, session_id) aligned with FAISS row order.
    IndexFlatIP + L2-normalized embeddings ≈ cosine similarity.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dim: int | None = None
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._index: faiss.Index | None = None
        self._texts: list[str] = []
        self._session_ids: list[str] = []
        self._dir = Path(settings.VECTOR_MEMORY_DIR)
        # NOTE: deliberately NOT loading from disk or embedding model here.
        # Both are lazy — triggered on first actual use, not on import.

    def _get_dim(self) -> int:
        if self._dim is None:
            self._dim = _get_dim()
        return self._dim

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        d = self._get_dim()
        self._index = faiss.IndexFlatIP(d)
        if self._vectors.size == 0:
            self._vectors = np.zeros((0, d), dtype=np.float32)

    def _load_from_disk_if_needed(self) -> None:
        """Lazy disk load — called on first search/add, not at import time."""
        if self._index is not None:
            return  # already loaded
        idx_path = self._dir / "index.faiss"
        meta_path = self._dir / "meta.json"
        vec_path = self._dir / "vectors.npy"
        if not idx_path.is_file() or not meta_path.is_file() or not vec_path.is_file():
            self._ensure_index()
            return
        try:
            self._index = faiss.read_index(str(idx_path))
            self._dim = self._index.d
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._texts = meta.get("texts", [])
            self._session_ids = meta.get("session_ids", [])
            self._vectors = np.load(vec_path).astype(np.float32)
            if (
                self._index.ntotal != len(self._texts)
                or len(self._texts) != len(self._session_ids)
                or (self._vectors.size and self._vectors.shape != (len(self._texts), self._dim))
            ):
                log.warning("Vector memory files inconsistent; starting empty store")
                self._texts, self._session_ids = [], []
                self._vectors = np.zeros((0, self._dim), dtype=np.float32)
                self._index = faiss.IndexFlatIP(self._dim)
            log.info(f"Vector memory loaded: {len(self._texts)} summaries from {self._dir}")
        except Exception as e:
            log.warning(f"Could not load vector memory ({e}); using empty store")
            self._texts, self._session_ids = [], []
            self._index = None
            self._vectors = np.zeros((0, 0), dtype=np.float32)
            self._ensure_index()

    def _persist(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        if self._index is None:
            return
        faiss.write_index(self._index, str(self._dir / "index.faiss"))
        meta = {"texts": self._texts, "session_ids": self._session_ids}
        (self._dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        np.save(self._dir / "vectors.npy", self._vectors)

    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        text = _make_turn_summary(user_message, assistant_message)
        if not text.strip():
            return
        with self._lock:
            self._load_from_disk_if_needed()
            assert self._index is not None and self._dim is not None
            v = _embed([text])
            if v.shape[1] != self._dim:
                log.error("Embedding dimension mismatch; skip vector add")
                return
            self._vectors = np.vstack([self._vectors, v]) if self._vectors.size else v
            self._texts.append(text)
            self._session_ids.append(session_id)
            self._index.add(v)
            self._persist()

    def search(self, session_id: str, query: str, k: int | None = None) -> list[str]:
        k = k if k is not None else settings.VECTOR_MEMORY_TOP_K
        with self._lock:
            self._load_from_disk_if_needed()
            if self._index is None or self._index.ntotal == 0 or not query.strip():
                return []
            q = _embed([query])
            if q.shape[1] != self._dim:
                return []
            n = self._index.ntotal
            fetch = min(max(k * 4, k), n)
            _, indices = self._index.search(q, fetch)
            hits: list[str] = []
            seen: set[str] = set()
            for idx in indices[0]:
                if idx < 0 or idx >= len(self._texts):
                    continue
                if self._session_ids[idx] != session_id:
                    continue
                t = self._texts[idx]
                if t in seen:
                    continue
                seen.add(t)
                hits.append(t)
                if len(hits) >= k:
                    break
            return hits

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._load_from_disk_if_needed()
            if not self._session_ids:
                return
            assert self._index is not None and self._dim is not None
            mask = np.array([sid != session_id for sid in self._session_ids], dtype=bool)
            if mask.all():
                return
            self._vectors = self._vectors[mask]
            self._texts = [t for t, m in zip(self._texts, mask) if m]
            self._session_ids = [s for s, m in zip(self._session_ids, mask) if m]
            self._index = faiss.IndexFlatIP(self._dim)
            if self._vectors.size:
                self._index.add(self._vectors)
            self._persist()
            log.info(f"Vector memory cleared for session: {session_id}")


# Singleton — no heavy work happens here, everything is lazy
vector_memory_store = VectorMemoryStore()