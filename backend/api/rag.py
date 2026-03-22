"""POST /rag/query — retrieve → RAG chain → answer + sources."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from core.auth import verify_token
from core.logger import log
from core.graph_tracer import new_trace_id
from core.metrics_collector import RequestMetrics, bind_metrics
from langchain_core.messages import HumanMessage

from models.schemas import RAGChunkItem, RAGQueryRequest, RAGQueryResponse
from agent.unified_graph import chat_recursion_config, final_assistant_text, get_unified_graph
from rag.vectorstore import get_app_vector_store

router = APIRouter()


@router.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(body: RAGQueryRequest, user_id: str = Depends(verify_token)) -> RAGQueryResponse:
    q = body.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query must not be empty")

    store = await get_app_vector_store()
    if not store.has_vectorstore:
        raise HTTPException(
            status_code=400,
            detail="No vector index yet. Upload documents with POST /api/upload first.",
        )

    graph = get_unified_graph()

    metrics = RequestMetrics.start(
        session_id="rag",
        user_id=user_id,
        query=q,
        endpoint="rag_query",
    )
    bind_metrics(metrics)
    error_msg = ""
    try:
        trace_id = new_trace_id()  # one trace_id for all nodes in this RAG invocation
        config = chat_recursion_config()
        config.setdefault("configurable", {})["user_id"] = user_id
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=q)],
                "memory_context": "",
                "rag_top_k": body.top_k,
            },
            config=config,
        )
    except Exception as e:
        error_msg = str(e)
        log.exception(f"RAG query failed: {e}")
        metrics.finish_and_log(error=error_msg)
        raise HTTPException(status_code=500, detail="RAG query failed") from e

    # Feed metrics from result state
    rag_docs = result.get("rag_docs") or []
    metrics.set_retrieval_docs(len(rag_docs))
    metrics.finish_and_log()

    answer = final_assistant_text(result["messages"])
    sources = result.get("rag_sources") or []
    raw_chunks = result.get("rag_chunks") or []

    return RAGQueryResponse(
        answer=answer,
        sources=sources,
        chunks=[
            RAGChunkItem(filename=c["filename"], preview=c["preview"])
            if isinstance(c, dict)
            else RAGChunkItem(filename=c.filename, preview=c.preview)
            for c in raw_chunks
        ],
    )
