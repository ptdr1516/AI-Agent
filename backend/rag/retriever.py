"""Similarity retrieval over a LangChain vector store (graph + :class:`RAGChain` / tools)."""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from core.config import settings
from core.retrieval_diagnostics import log_retrieval_diagnostics


def documents_to_payloads(documents: Sequence[Document]) -> list[dict[str, Any]]:
    """Expose each hit as ``page_content`` + ``metadata`` dict (JSON-serializable where possible)."""
    out: list[dict[str, Any]] = []
    for doc in documents:
        out.append(
            {
                "page_content": doc.page_content,
                "metadata": dict(doc.metadata or {}),
            }
        )
    return out


class RAGRetriever:
    """Similarity search with configurable ``top_k``; returns LangChain ``Document`` (content + metadata)."""

    def __init__(
        self,
        vectorstore: VectorStore,
        *,
        default_top_k: int | None = None,
        k: int | None = None,
        fetch_k: int | None = None,
    ) -> None:
        self._store = vectorstore
        dk = default_top_k if default_top_k is not None else k
        self._default_top_k = dk if dk is not None else settings.RAG_RETRIEVAL_K
        self._fetch_k = fetch_k if fetch_k is not None else settings.RAG_FETCH_K

    def _effective_top_k(self, top_k: int | None, k: int | None) -> int:
        if top_k is not None:
            return top_k
        if k is not None:
            return k
        return self._default_top_k

    async def aretrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        k: int | None = None,
        user_id: str | None = None,
    ) -> list[Document]:
        """Async similarity search; returns ``Document`` list (``page_content`` + ``metadata``)."""
        n = self._effective_top_k(top_k, k)
        from typing import Any
        kwargs: dict[str, Any] = {"k": n}
        if user_id:
            kwargs["filter"] = {"user_id": user_id}
            
        docs_and_scores = []
        try:
            # Try to get scores if the vector store supports it
            try:
                docs_and_scores = await self._store.asimilarity_search_with_score(
                    query, fetch_k=self._fetch_k, **kwargs
                )
            except AttributeError:
                # Fallback if with_score is not implemented
                docs = await self._store.asimilarity_search(
                    query, fetch_k=self._fetch_k, **kwargs
                )
                docs_and_scores = [(doc, 0.0) for doc in docs]
        except TypeError:
            # Try without fetch_k
            try:
                docs_and_scores = await self._store.asimilarity_search_with_score(
                    query, **kwargs
                )
            except AttributeError:
                docs = await self._store.asimilarity_search(query, **kwargs)
                docs_and_scores = [(doc, 0.0) for doc in docs]

        # Log diagnostics
        log_retrieval_diagnostics(query, docs_and_scores, user_id)
        
        # Return just the documents (LangChain expectation)
        return [doc for doc, _ in docs_and_scores]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        k: int | None = None,
        user_id: str | None = None,
    ) -> list[Document]:
        """Sync similarity search."""
        n = self._effective_top_k(top_k, k)
        from typing import Any
        kwargs: dict[str, Any] = {"k": n}
        if user_id:
            kwargs["filter"] = {"user_id": user_id}
            
        docs_and_scores = []
        try:
            try:
                docs_and_scores = self._store.similarity_search_with_score(
                    query, fetch_k=self._fetch_k, **kwargs
                )
            except AttributeError:
                docs = self._store.similarity_search(
                    query, fetch_k=self._fetch_k, **kwargs
                )
                docs_and_scores = [(doc, 0.0) for doc in docs]
        except TypeError:
            try:
                docs_and_scores = self._store.similarity_search_with_score(
                    query, **kwargs
                )
            except AttributeError:
                docs = self._store.similarity_search(query, **kwargs)
                docs_and_scores = [(doc, 0.0) for doc in docs]

        log_retrieval_diagnostics(query, docs_and_scores, user_id)
        return [doc for doc, _ in docs_and_scores]

    async def aretrieve_payloads(
        self,
        query: str,
        *,
        top_k: int | None = None,
        k: int | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Same as :meth:`aretrieve`, but each hit is a dict with ``page_content`` and ``metadata``."""
        docs = await self.aretrieve(query, top_k=top_k, k=k, user_id=user_id)
        return documents_to_payloads(docs)

    def retrieve_payloads(
        self,
        query: str,
        *,
        top_k: int | None = None,
        k: int | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        docs = self.retrieve(query, top_k=top_k, k=k, user_id=user_id)
        return documents_to_payloads(docs)

    async def aget_relevant_documents(
        self,
        query: str,
        k: int | None = None,
    ) -> list[Document]:
        """LangChain-style name; ``k`` overrides default top_k."""
        return await self.aretrieve(query, k=k)

    def get_relevant_documents(self, query: str, k: int | None = None) -> list[Document]:
        return self.retrieve(query, k=k)

    def as_langchain_retriever(self, *, search_kwargs: dict[str, Any] | None = None) -> VectorStoreRetriever:
        """Wrap the store as a LangChain retriever (fixed ``k`` from ``search_kwargs`` or default)."""
        kwargs = dict(search_kwargs or {})
        if "k" not in kwargs:
            kwargs["k"] = self._default_top_k
        if "fetch_k" not in kwargs:
            kwargs["fetch_k"] = self._fetch_k
        try:
            return self._store.as_retriever(search_kwargs=kwargs)
        except TypeError:
            kwargs.pop("fetch_k", None)
            return self._store.as_retriever(search_kwargs=kwargs)
