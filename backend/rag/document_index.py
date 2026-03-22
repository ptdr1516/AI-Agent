"""List and remove indexed RAG documents (by chunk metadata) for FAISS and Chroma."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from langchain_core.documents import Document

from core.logger import log

from .vectorstore import PersistentVectorStore


def _doc_key(meta: dict[str, Any]) -> tuple[str, str]:
    """(original_filename, stored_filename) for grouping."""
    stored = str(meta.get("filename") or "")
    orig = str(meta.get("original_filename") or stored or "unknown")
    return (orig, stored)


def _iter_faiss_doc_ids(vs: Any) -> list[tuple[str, Document]]:
    out: list[tuple[str, Document]] = []
    for _idx, doc_id in vs.index_to_docstore_id.items():
        doc = vs.docstore.search(doc_id)
        if isinstance(doc, Document):
            out.append((doc_id, doc))
    return out


def _list_groups_faiss(vs: Any, *, user_id: str | None = None) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for _doc_id, doc in _iter_faiss_doc_ids(vs):
        md = doc.metadata or {}
        if user_id and md.get("user_id") != user_id:
            continue
        orig, stored = _doc_key(md)
        if not stored:
            continue
        counts[(orig, stored)] += 1
    return [
        {
            "original_filename": orig,
            "stored_filename": stored,
            "chunks": n,
        }
        for (orig, stored), n in sorted(counts.items(), key=lambda x: x[0][0].lower())
    ]


def _faiss_ids_to_delete(
    vs: Any,
    *,
    stored_filename: str | None,
    original_filename: str | None,
    user_id: str | None = None,
) -> list[str]:
    ids: list[str] = []
    for doc_id, doc in _iter_faiss_doc_ids(vs):
        md = doc.metadata or {}
        if user_id and md.get("user_id") != user_id:
            continue
        if stored_filename is not None and md.get("filename") == stored_filename:
            ids.append(doc_id)
        elif original_filename is not None and md.get("original_filename") == original_filename:
            ids.append(doc_id)
    return ids


def _delete_faiss_chunks(vs: Any, ids: list[str]) -> int:
    if not ids:
        return 0
    vs.delete(ids)
    return len(ids)


def _list_groups_chroma(chroma: Any, *, user_id: str | None = None) -> list[dict[str, Any]]:
    col = chroma._collection
    batch = col.get(include=["metadatas"])
    metas = batch.get("metadatas") or []
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for meta in metas:
        if not meta:
            continue
        if user_id and meta.get("user_id") != user_id:
            continue
        orig, stored = _doc_key(meta)
        if not stored:
            continue
        counts[(orig, stored)] += 1
    return [
        {
            "original_filename": orig,
            "stored_filename": stored,
            "chunks": n,
        }
        for (orig, stored), n in sorted(counts.items(), key=lambda x: x[0][0].lower())
    ]


def _delete_chroma_by_filter(
    chroma: Any,
    *,
    stored_filename: str | None,
    original_filename: str | None,
    user_id: str | None = None,
) -> int:
    col = chroma._collection
    data = col.get(include=["metadatas"])
    ids_list = data.get("ids") or []
    metas = data.get("metadatas") or []
    to_delete: list[str] = []
    for doc_id, meta in zip(ids_list, metas):
        if not meta:
            continue
        if user_id and meta.get("user_id") != user_id:
            continue
        if stored_filename is not None and meta.get("filename") == stored_filename:
            to_delete.append(doc_id)
        elif original_filename is not None and meta.get("original_filename") == original_filename:
            to_delete.append(doc_id)
    if not to_delete:
        return 0
    col.delete(ids=to_delete)
    return len(to_delete)


async def list_indexed_documents(
    store: PersistentVectorStore,
    *,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return grouped indexed files for the user_id (global if null)."""
    if not store.has_vectorstore:
        return []

    if store.backend == "faiss":
        vs = store.vectorstore

        def _run() -> list[dict[str, Any]]:
            return _list_groups_faiss(vs, user_id=user_id)

        return await asyncio.to_thread(_run)

    from langchain_community.vectorstores import Chroma

    vs = store.vectorstore
    if not isinstance(vs, Chroma):
        return []

    def _run_c() -> list[dict[str, Any]]:
        return _list_groups_chroma(vs, user_id=user_id)

    return await asyncio.to_thread(_run_c)


async def remove_document_chunks(
    store: PersistentVectorStore,
    *,
    stored_filename: str | None = None,
    original_filename: str | None = None,
    user_id: str | None = None,
) -> int:
    """Delete all chunks matching ``stored_filename`` (preferred) or ``original_filename``."""
    if not store.has_vectorstore:
        return 0
    if not stored_filename and not original_filename:
        return 0

    removed = 0
    if store.backend == "faiss":
        vs = store.vectorstore

        def _run_f() -> int:
            ids = _faiss_ids_to_delete(
                vs,
                stored_filename=stored_filename,
                original_filename=original_filename,
                user_id=user_id,
            )
            return _delete_faiss_chunks(vs, ids)

        removed = await asyncio.to_thread(_run_f)
        if store._vs is not None and len(store.vectorstore.index_to_docstore_id) == 0:
            for name in ("index.faiss", "index.pkl"):
                (store.persist_path / name).unlink(missing_ok=True)
            store._vs = None
            log.info("Cleared empty FAISS index on disk and in memory")
    else:
        from langchain_community.vectorstores import Chroma

        vs = store.vectorstore
        if not isinstance(vs, Chroma):
            return 0

        def _run_c() -> int:
            return _delete_chroma_by_filter(
                vs,
                stored_filename=stored_filename,
                original_filename=original_filename,
                user_id=user_id,
            )

        removed = await asyncio.to_thread(_run_c)

    if removed:
        await store.apersist()
        log.info(
            "Removed %s chunk(s) (stored_filename=%r original_filename=%r)",
            removed,
            stored_filename,
            original_filename,
        )
    return removed
