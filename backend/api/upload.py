"""POST /upload — save file, chunk, embed, persist to RAG vector store."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from core.auth import verify_token
from core.config import settings
from core.logger import log
from models.schemas import UploadResponse
from rag.chunker import split_documents
from rag.loader import load_file
from rag.vectorstore import ingest_documents_to_app_store

router = APIRouter()

_ALLOWED_SUFFIXES = frozenset({".pdf", ".txt", ".text", ".md", ".markdown"})


def _safe_filename(original: str | None) -> str:
    base = Path(original or "upload.bin").name
    if not base or base in (".", "..") or ".." in base:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return base


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(verify_token),
) -> UploadResponse:
    name = _safe_filename(file.filename)
    suffix = Path(name).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Allowed: {sorted(_ALLOWED_SUFFIXES)}",
        )

    body = await file.read()
    if len(body) > settings.UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {settings.UPLOAD_MAX_BYTES} bytes)",
        )

    upload_root = Path(settings.UPLOAD_DIR)
    upload_root.mkdir(parents=True, exist_ok=True)
    unique = f"{uuid.uuid4().hex}_{name}"
    dest = upload_root / unique

    dest.write_bytes(body)
    log.info(f"Upload saved: {dest}")

    try:
        documents = load_file(dest)
        for d in documents:
            meta = dict(d.metadata or {})
            meta["original_filename"] = name
            meta["user_id"] = user_id
            d.metadata = meta
    except ValueError as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        dest.unlink(missing_ok=True)
        log.exception(f"Loader failed for {dest}: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse file") from e

    chunks = split_documents(documents)
    if not chunks:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No text extracted from file")

    try:
        await ingest_documents_to_app_store(chunks)
    except Exception as e:
        log.exception(f"Vector ingest failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to index document") from e

    return UploadResponse(
        success=True,
        filename=name,
        stored_filename=unique,
        saved_path=str(dest.resolve()),
        chunks_indexed=len(chunks),
    )
