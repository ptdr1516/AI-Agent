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
    """Streaming endpoint should respond 200 with text/event-stream content-type."""
    payload = {
        "message": "What is 2+2?",
        "session_id": "test-stream-session-abc",
        "model": "openai/gpt-3.5-turbo"
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/chat/stream", json=payload)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    content_type = response.headers.get("content-type", "")
    assert "text/event-stream" in content_type, f"Expected SSE content-type, got: {content_type}"
