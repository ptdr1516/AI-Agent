"""Integration test for the FastAPI streaming endpoint.

Note: This test does NOT make a real LLM call. It mocks the agent's
astream_events() so the test runs fast, offline, and deterministically.
"""
import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport
from main import app


async def _fake_stream(*args, **kwargs):
    """A minimal mock of astream_events() that yields two SSE-formatted chunks."""
    # Simulate a 'token' event then a 'done' event
    events = [
        {"type": "token", "content": "4"},
        {"type": "done", "content": ""},
    ]
    for event in events:
        yield event


@pytest.mark.asyncio
async def test_streaming_endpoint_returns_sse():
    """Streaming endpoint should respond 200 with SSE and emit basic events."""

    class _FakeChunk:
        def __init__(self, content: str):
            self.content = content

    class _FakeOutput:
        def __init__(self):
            self.response_metadata = {
                "token_usage": {"prompt_tokens": 1, "completion_tokens": 1}
            }

    class _FakeGraph:
        async def astream_events(self, *args, **kwargs):
            # Retrieval node start + end
            yield {"event": "on_chain_start", "name": "retrieval_node", "data": {}}
            rag_context = "[1] Source: doc1.txt\nhello world"
            yield {
                "event": "on_chain_end",
                "name": "retrieval_node",
                "data": {
                    "output": {
                        "rag_context": rag_context,
                        "rag_sources": ["doc1.txt"],
                        "rag_chunks": [
                            {"filename": "doc1.txt", "preview": "hello world"}
                        ],
                        "rag_docs": [MagicMock()],
                    }
                },
            }

            # Tool start/end
            yield {
                "event": "on_tool_start",
                "name": "calculator_tool",
                "data": {"input": {"expression": "2+2"}},
            }
            yield {
                "event": "on_tool_end",
                "name": "calculator_tool",
                "data": {"output": "4"},
            }

            # Token chunk
            yield {
                "event": "on_chat_model_stream",
                "name": "llm_node",
                "data": {"chunk": _FakeChunk("4")},
            }

            # Final model end (for usage accounting)
            yield {
                "event": "on_chat_model_end",
                "name": "llm_node",
                "data": {"output": _FakeOutput()},
            }

            # Allow chat.py to finalize chain output (optional)
            yield {
                "event": "on_chain_end",
                "name": "output_node",
                "data": {"output": {"output": "4", "messages": []}},
            }

    class _FakeMemory:
        def load_memory_variables(self, _):
            return {"chat_history": []}

        def save_context(self, inputs, outputs):
            return None

    payload = {
        "message": "What is 2+2?",
        "session_id": "test-stream-session-abc",
        "model": "openai/gpt-3.5-turbo",
    }

    with patch("api.chat.get_unified_graph", return_value=_FakeGraph()):
        with patch("api.chat.memory_manager.get_memory", return_value=_FakeMemory()):
            with patch("api.chat.vector_memory_store.search", return_value=[]):
                with patch("api.chat.vector_memory_store.add_turn", return_value=None):
                    async with AsyncClient(
                        transport=ASGITransport(app=app), base_url="http://test"
                    ) as ac:
                        response = await ac.post("/api/chat/stream", json=payload)

                    assert response.status_code == 200, (
                        f"Expected 200, got {response.status_code}"
                    )
                    content_type = response.headers.get("content-type", "")
                    assert (
                        "text/event-stream" in content_type
                    ), f"Expected SSE content-type, got: {content_type}"

                    body = await response.aread()
                    body_text = body.decode("utf-8", errors="replace")
                    assert '"retrieval"' in body_text
                    assert '"content":' in body_text and '"4"' in body_text
                    assert '"tool":' in body_text
                    assert "data: [DONE]" in body_text
                    assert "\n\n" in body_text
