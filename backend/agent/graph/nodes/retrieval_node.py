"""retrieval_node — similarity search over the app index (or injected retriever) every turn."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from rag.rag_chain import RAGChunkInfo, build_context, extract_sources, _display_source_name
from rag.runtime_retrieval import retrieve_documents
from rag.retriever import RAGRetriever

from ..state import UnifiedGraphState


async def retrieval_node(state: UnifiedGraphState, config: RunnableConfig) -> dict[str, Any]:
    """Populate ``rag_*`` fields from a real vector-store retrieval (no synthetic chunks)."""
    configurable = (config or {}).get("configurable") or {}
    user_id: str | None = configurable.get("user_id")
    retriever: RAGRetriever | None = configurable.get("retriever")

    question = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            c = m.content
            question = c if isinstance(c, str) else str(c)
            break

    if not question.strip():
        return {
            "rag_docs": [],
            "rag_context": build_context([]),
            "rag_sources": [],
            "rag_chunks": [],
        }

    tk = state.get("rag_top_k")
    docs = await retrieve_documents(question, top_k=tk, retriever=retriever, user_id=user_id)
    chunk_rows = [
        RAGChunkInfo(filename=_display_source_name(d), preview=d.page_content) for d in docs
    ]
    chunks_out = [{"filename": c.filename, "preview": c.preview} for c in chunk_rows]

    return {
        "rag_docs": docs,
        "rag_context": build_context(docs),
        "rag_sources": extract_sources(docs),
        "rag_chunks": chunks_out,
    }
