"""
Runtime RAG retrieval for the LangGraph and HTTP API.

Indexing (upload) uses the same stack elsewhere in this package:

- ``rag.loader`` / ``rag.chunker`` → chunks
- ``rag.embeddings.get_embedding_model`` → vectors
- ``rag.vectorstore.PersistentVectorStore`` / ``ingest_documents_to_app_store`` → persisted FAISS or Chroma

At query time this module only performs **similarity search** on that store via
:class:`rag.retriever.RAGRetriever` (``asimilarity_search`` on the underlying
``VectorStore``). No placeholder chunks and no synthetic context.
"""

from __future__ import annotations

from langchain_core.documents import Document

from core.config import settings
from core.logger import log
from core.tracing import trace_span

from .retriever import RAGRetriever
from .vectorstore import get_app_vector_store


@trace_span("rag.retrieve", metadata_fn=lambda q, **kw: {"query_len": len(q), "user_id": kw.get("user_id", "anon")})
async def retrieve_documents(
    query: str,
    *,
    top_k: int | None = None,
    retriever: RAGRetriever | None = None,
    user_id: str | None = None,
) -> list[Document]:
    """
    Return top-k :class:`~langchain_core.documents.Document` hits for ``query``.

    - If ``retriever`` is provided (e.g. eval script with its own store), use it as-is.
    - Otherwise resolve the process singleton via :func:`get_app_vector_store`, build a
      :class:`RAGRetriever` over ``store.vectorstore``, and run async similarity search.

    If the app store has no index yet, returns an empty list (callers that require an
    index should check ``PersistentVectorStore.has_vectorstore`` before invoking the graph).
    """
    q = (query or "").strip()
    if not q:
        return []

    if retriever is not None:
        docs = await retriever.aretrieve(q, top_k=top_k, user_id=user_id)
        log.debug(f"runtime_retrieval: injected retriever returned {len(docs)} doc(s)")
        return docs

    store = await get_app_vector_store()
    if not store.has_vectorstore:
        log.debug("runtime_retrieval: no vector index — returning no documents")
        return []

    # Lazy-load the backing vector store into RAM only when we know it exists.
    vs = await store.aget_vectorstore()
    r = RAGRetriever(
        vs,
        default_top_k=settings.RAG_RETRIEVAL_K,
    )
    docs = await r.aretrieve(q, top_k=top_k, user_id=user_id)
    log.info(f"runtime_retrieval: app store similarity search → {len(docs)} chunk(s)")
    return docs
