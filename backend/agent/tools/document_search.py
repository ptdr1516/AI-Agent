"""
Document search tool — RAG retriever over the app vector store.

Returns formatted context (retrieved chunks + source labels), not an LLM answer.
"""
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

from core.logger import log
from rag.rag_chain import build_context
from rag.runtime_retrieval import retrieve_documents
from rag.vectorstore import get_app_vector_store


class DocumentSearchInput(BaseModel):
    query: str = Field(
        description=(
            "Natural-language query to search indexed documents (user uploads). "
            "Be specific about what to find."
        )
    )


from langchain_core.runnables import RunnableConfig

@tool("document_search", args_schema=DocumentSearchInput)
async def document_search_tool(query: str, config: RunnableConfig) -> str:
    """Search locally indexed documents and return relevant passages with source labels for grounding answers."""
    q = (query or "").strip()
    if not q:
        return "Error: empty query."

    try:
        user_id = config.get("configurable", {}).get("user_id")
        store = await get_app_vector_store()
        if not store.has_vectorstore:
            return (
                "No indexed documents are available. The user must upload files first "
                "(e.g. POST /api/upload). Until then, document_search cannot retrieve anything."
            )

        docs = await retrieve_documents(q, top_k=None, retriever=None, user_id=user_id)
        ctx = build_context(docs)
        log.info(f"document_search retrieved {len(docs)} chunk(s) for query len={len(q)}")
        return ctx if ctx.strip() else "(No matching passages found.)"
    except Exception as e:
        log.exception(f"document_search failed: {e}")
        return f"Error searching documents: {e!s}"
