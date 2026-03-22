"""Unit tests for SSE streaming helpers (no LLM)."""

from langchain_core.messages import AIMessageChunk

from api.chat import (
    _event_is_retrieval_node,
    _retrieval_sse_payload,
    _stream_chunk_text,
)


def test_stream_chunk_text_string():
    ch = AIMessageChunk(content="hi")
    assert _stream_chunk_text(ch) == ["hi"]


def test_stream_chunk_text_blocks():
    ch = AIMessageChunk(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert _stream_chunk_text(ch) == ["a", "b"]


def test_retrieval_sse_payload():
    p = _retrieval_sse_payload(
        {
            "rag_sources": ["f.txt"],
            "rag_context": "[1] Source: f.txt\nbody",
            "rag_chunks": [{"filename": "f.txt", "preview": "body"}],
        }
    )
    assert p is not None
    assert p["retrieval"]["status"] == "complete"
    assert p["retrieval"]["sources"] == ["f.txt"]
    assert p["retrieval"]["chunk_count"] == 1
    assert p["retrieval"]["rag_detail"]["sources"] == ["f.txt"]


def test_event_is_retrieval_node():
    assert _event_is_retrieval_node({"name": "retrieval_node"})
    assert _event_is_retrieval_node({"name": "LangGraph/retrieval_node"})
    assert not _event_is_retrieval_node({"name": "llm_node"})
