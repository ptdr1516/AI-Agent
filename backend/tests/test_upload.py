"""POST /api/upload — file save + ingest pipeline (ingest mocked)."""

import pytest
from unittest.mock import AsyncMock, patch
from httpx import ASGITransport, AsyncClient

from core.config import settings
from main import app


@pytest.mark.asyncio
async def test_upload_txt_success(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    with patch("api.upload.ingest_documents_to_app_store", new_callable=AsyncMock) as ingest:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post(
                "/api/upload",
                files={"file": ("note.txt", b"hello world " * 50, "text/plain")},
            )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["filename"] == "note.txt"
    assert data["chunks_indexed"] >= 1
    assert "saved_path" in data
    assert data["stored_filename"].endswith("_note.txt")
    ingest.assert_called_once()


@pytest.mark.asyncio
async def test_upload_rejects_bad_extension():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/upload",
            files={"file": ("hack.exe", b"x", "application/octet-stream")},
        )
    assert response.status_code == 400
