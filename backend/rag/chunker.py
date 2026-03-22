"""Split documents with LangChain RecursiveCharacterTextSplitter (metadata preserved)."""

from __future__ import annotations

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from core.config import settings


def make_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    *,
    add_start_index: bool = False,
) -> RecursiveCharacterTextSplitter:
    """Build a recursive character splitter.

    Defaults come from settings (``RAG_CHUNK_SIZE`` / ``RAG_CHUNK_OVERLAP``): 500 / 100.
    """
    size = chunk_size if chunk_size is not None else settings.RAG_CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else settings.RAG_CHUNK_OVERLAP
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
        add_start_index=add_start_index,
    )


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    add_start_index: bool = False,
) -> list[Document]:
    """Split documents using LangChain's recursive character splitter.

    Parent metadata is **deep-copied** onto every chunk (LangChain behavior), so keys
    like ``source``, ``filename``, and ``page`` are preserved. Optional
    ``add_start_index=True`` adds ``start_index`` per chunk inside the original text.
    """
    splitter = make_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
    )
    normalized = [
        Document(
            page_content=doc.page_content,
            metadata=dict(doc.metadata or {}),
        )
        for doc in documents
    ]
    return splitter.split_documents(normalized)


def split_text(
    text: str,
    *,
    metadata: dict | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    add_start_index: bool = False,
) -> list[Document]:
    """Split a single string into documents (same splitter defaults as ``split_documents``)."""
    splitter = make_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
    )
    meta = dict(metadata or {})
    return splitter.create_documents([text], metadatas=[meta])
