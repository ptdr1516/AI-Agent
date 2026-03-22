"""Persistent FAISS or Chroma vector stores: load/save, add documents, similarity search."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from core.config import settings
from core.logger import log

from .embeddings import get_embedding_model

if TYPE_CHECKING:
    from langchain_community.vectorstores import Chroma, FAISS

Backend = Literal["faiss", "chroma"]


def _normalize_backend(raw: str) -> Backend:
    b = raw.lower().strip()
    if b in ("faiss", "faiss_index"):
        return "faiss"
    if b in ("chroma", "chromadb"):
        return "chroma"
    raise ValueError(f"Unknown RAG_VECTORSTORE_BACKEND: {raw!r}. Use 'faiss' or 'chroma'.")


def _faiss_index_exists(directory: Path) -> bool:
    return (directory / "index.faiss").is_file() and (directory / "index.pkl").is_file()


class PersistentVectorStore:
    """FAISS or Chroma with disk persistence, async add/search, configurable top-k."""

    def __init__(
        self,
        *,
        persist_path: str | None = None,
        backend: str | None = None,
        embedding: Embeddings | None = None,
        collection_name: str | None = None,
        default_top_k: int | None = None,
    ) -> None:
        self._backend = _normalize_backend(backend or settings.RAG_VECTORSTORE_BACKEND)
        self._emb = embedding or get_embedding_model()
        self._persist_path = Path(
            persist_path
            or (
                settings.RAG_CHROMA_DIR
                if self._backend == "chroma"
                else settings.RAG_FAISS_DIR
            )
        )
        self._collection_name = collection_name or settings.RAG_CHROMA_COLLECTION
        self._default_top_k = (
            default_top_k if default_top_k is not None else settings.RAG_RETRIEVAL_K
        )
        self._vs: VectorStore | None = None

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def persist_path(self) -> Path:
        return self._persist_path

    @property
    def has_vectorstore(self) -> bool:
        """True once a backing index exists (FAISS after first ingest; Chroma after open)."""
        return self._vs is not None

    @property
    def vectorstore(self) -> VectorStore:
        if self._vs is None:
            raise RuntimeError(
                "Vector store is empty. Load an existing index or add documents first."
            )
        return self._vs

    @classmethod
    async def open(
        cls,
        *,
        persist_path: str | None = None,
        backend: str | None = None,
        embedding: Embeddings | None = None,
        collection_name: str | None = None,
        default_top_k: int | None = None,
    ) -> PersistentVectorStore:
        """Open or prepare a store: load FAISS index from disk if present; Chroma always opens a collection."""
        inst = cls(
            persist_path=persist_path,
            backend=backend,
            embedding=embedding,
            collection_name=collection_name,
            default_top_k=default_top_k,
        )
        await inst._load_if_exists()
        return inst

    async def _load_if_exists(self) -> None:
        self._persist_path.mkdir(parents=True, exist_ok=True)
        if self._backend == "faiss":
            if _faiss_index_exists(self._persist_path):
                from langchain_community.vectorstores import FAISS

                def _load() -> FAISS:
                    return FAISS.load_local(
                        str(self._persist_path),
                        self._emb,
                        allow_dangerous_deserialization=True,
                    )

                self._vs = await asyncio.to_thread(_load)
                log.info(f"RAG FAISS index loaded from {self._persist_path}")
            else:
                self._vs = None
                log.info(f"No FAISS index at {self._persist_path}; add documents to create one.")
        else:
            from langchain_community.vectorstores import Chroma

            def _open_chroma() -> Chroma:
                return Chroma(
                    embedding_function=self._emb,
                    persist_directory=str(self._persist_path),
                    collection_name=self._collection_name,
                )

            self._vs = await asyncio.to_thread(_open_chroma)
            log.info(
                f"RAG Chroma collection {self._collection_name!r} at {self._persist_path}"
            )

    async def aadd_documents(self, documents: list[Document]) -> list[str]:
        if not documents:
            return []
        if self._backend == "faiss":
            from langchain_community.vectorstores import FAISS

            if self._vs is None:
                self._vs = await FAISS.afrom_documents(documents, self._emb)
                log.info(f"RAG FAISS index created with {len(documents)} chunk(s)")
                return [str(i) for i in range(len(documents))]
            ids = await self._vs.aadd_documents(documents)
            log.info(f"RAG FAISS: added {len(documents)} chunk(s)")
            return ids

        assert self._vs is not None
        ids = await self._vs.aadd_documents(documents)
        log.info(f"RAG Chroma: added {len(documents)} chunk(s)")
        return ids

    def add_documents(self, documents: list[Document]) -> list[str]:
        if not documents:
            return []
        if self._backend == "faiss":
            from langchain_community.vectorstores import FAISS

            if self._vs is None:
                self._vs = FAISS.from_documents(documents, self._emb)
                log.info(f"RAG FAISS index created (sync) with {len(documents)} chunk(s)")
                return [str(i) for i in range(len(documents))]
            return self._vs.add_documents(documents)

        assert self._vs is not None
        return self._vs.add_documents(documents)

    async def asimilarity_search(
        self,
        query: str,
        k: int | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        k = k if k is not None else self._default_top_k
        vs = self.vectorstore
        if self._backend == "faiss":
            fetch_k = kwargs.pop("fetch_k", settings.RAG_FETCH_K)
            return await vs.asimilarity_search(query, k=k, fetch_k=fetch_k, **kwargs)
        return await vs.asimilarity_search(query, k=k, **kwargs)

    def similarity_search(self, query: str, k: int | None = None, **kwargs: Any) -> list[Document]:
        k = k if k is not None else self._default_top_k
        vs = self.vectorstore
        if self._backend == "faiss":
            fetch_k = kwargs.pop("fetch_k", settings.RAG_FETCH_K)
            return vs.similarity_search(query, k=k, fetch_k=fetch_k, **kwargs)
        return vs.similarity_search(query, k=k, **kwargs)

    async def apersist(self) -> None:
        if self._vs is None:
            return
        if self._backend == "faiss":

            def _save() -> None:
                from langchain_community.vectorstores import FAISS

                assert isinstance(self._vs, FAISS)
                self._persist_path.mkdir(parents=True, exist_ok=True)
                self._vs.save_local(str(self._persist_path))

            await asyncio.to_thread(_save)
            log.info(f"RAG FAISS store saved to {self._persist_path}")
        else:

            def _persist_chroma() -> None:
                from langchain_community.vectorstores import Chroma

                assert isinstance(self._vs, Chroma)
                self._vs.persist()

            await asyncio.to_thread(_persist_chroma)
            log.info(f"RAG Chroma persisted at {self._persist_path}")

    def persist(self) -> None:
        if self._vs is None:
            return
        if self._backend == "faiss":
            from langchain_community.vectorstores import FAISS

            assert isinstance(self._vs, FAISS)
            self._persist_path.mkdir(parents=True, exist_ok=True)
            self._vs.save_local(str(self._persist_path))
            log.info(f"RAG FAISS store saved to {self._persist_path}")
        else:
            from langchain_community.vectorstores import Chroma

            assert isinstance(self._vs, Chroma)
            self._vs.persist()
            log.info(f"RAG Chroma persisted at {self._persist_path}")


# ── App singleton (FastAPI RAG ingest) ─────────────────────────────────────────

_app_store: PersistentVectorStore | None = None
_app_store_lock = asyncio.Lock()
_ingest_lock = asyncio.Lock()


async def get_app_vector_store() -> PersistentVectorStore:
    """Shared persistent store for the API process (lazy open)."""
    global _app_store
    async with _app_store_lock:
        if _app_store is None:
            _app_store = await PersistentVectorStore.open()
        return _app_store


async def reset_app_vector_store() -> None:
    """Clear singleton (tests)."""
    global _app_store
    async with _app_store_lock:
        _app_store = None


async def ingest_documents_to_app_store(documents: list[Document]) -> None:
    """Thread-safe add + persist for concurrent uploads."""
    if not documents:
        return
    async with _ingest_lock:
        store = await get_app_vector_store()
        await store.aadd_documents(documents)
        await store.apersist()


async def remove_indexed_document_from_app_store(
    *,
    stored_filename: str | None = None,
    original_filename: str | None = None,
    user_id: str | None = None,
) -> int:
    """Remove all chunks for one upload (matches ingest lock)."""
    from rag.document_index import remove_document_chunks

    async with _ingest_lock:
        store = await get_app_vector_store()
        return await remove_document_chunks(
            store,
            stored_filename=stored_filename,
            original_filename=original_filename,
            user_id=user_id,
        )


async def list_indexed_documents_from_app_store(user_id: str | None = None) -> list[dict]:
    """List grouped indexed documents (read-only; no ingest lock required)."""
    from rag.document_index import list_indexed_documents

    store = await get_app_vector_store()
    return await list_indexed_documents(store, user_id=user_id)


# ── Legacy / low-level helpers (FAISS-backed) ─────────────────────────────────

async def acreate_store(
    documents: list[Document],
    *,
    embedding: Embeddings | None = None,
) -> "FAISS":
    """Embed documents into an in-memory FAISS index (async)."""
    from langchain_community.vectorstores import FAISS

    emb = embedding or get_embedding_model()
    store = await FAISS.afrom_documents(documents, emb)
    log.info(f"RAG FAISS store created with {len(documents)} chunk(s)")
    return store


def create_store_sync(
    documents: list[Document],
    *,
    embedding: Embeddings | None = None,
) -> "FAISS":
    from langchain_community.vectorstores import FAISS

    emb = embedding or get_embedding_model()
    store = FAISS.from_documents(documents, emb)
    log.info(f"RAG FAISS store created (sync) with {len(documents)} chunk(s)")
    return store


async def asave_store(store: "FAISS", directory: str | None = None) -> None:
    path = directory or settings.RAG_FAISS_DIR
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _save() -> None:
        store.save_local(path)

    await asyncio.to_thread(_save)
    log.info(f"RAG FAISS store saved to {path}")


async def aload_store(
    directory: str | None = None,
    *,
    embedding: Embeddings | None = None,
) -> "FAISS":
    from langchain_community.vectorstores import FAISS

    path = directory or settings.RAG_FAISS_DIR
    emb = embedding or get_embedding_model()

    def _load() -> FAISS:
        return FAISS.load_local(path, emb, allow_dangerous_deserialization=True)

    store = await asyncio.to_thread(_load)
    log.info(f"RAG FAISS store loaded from {path}")
    return store
