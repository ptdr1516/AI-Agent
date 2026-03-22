"""POST /api/rag/query."""

import pytest
from langchain_core.messages import AIMessage
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.mark.asyncio
async def test_rag_query_requires_index_or_mocks():
    with patch("api.rag.get_app_vector_store", new_callable=AsyncMock) as m_store:
        mock_store = MagicMock()
        mock_store.has_vectorstore = False
        m_store.return_value = mock_store

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.post("/api/rag/query", json={"query": "hello"})

        assert r.status_code == 400
        assert "upload" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_rag_query_success_mocked():
    with patch("api.rag.get_app_vector_store", new_callable=AsyncMock) as m_store:
        with patch("api.rag.get_unified_graph") as m_graph:
            mock_store = MagicMock()
            mock_store.has_vectorstore = True
            mock_store.vectorstore = MagicMock()
            m_store.return_value = mock_store

            mock_compiled = MagicMock()
            mock_compiled.ainvoke = AsyncMock(
                return_value={
                    "messages": [AIMessage(content="ok")],
                    "rag_sources": ["a.txt"],
                    "rag_chunks": [{"filename": "a.txt", "preview": "chunk body"}],
                }
            )
            m_graph.return_value = mock_compiled

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                r = await ac.post(
                    "/api/rag/query",
                    json={"query": "What is in the doc?", "top_k": 3},
                )

    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "ok"
    assert data["sources"] == ["a.txt"]
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["filename"] == "a.txt"
    assert data["chunks"][0]["preview"] == "chunk body"
