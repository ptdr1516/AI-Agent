"""List and remove globally indexed RAG documents (shared across all chat sessions)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from core.auth import verify_token
from models.schemas import (
    IndexedDocumentsResponse,
    IndexedDocumentItem,
    RemoveIndexedDocumentRequest,
    RemoveIndexedDocumentResponse,
)
from rag.vectorstore import list_indexed_documents_from_app_store, remove_indexed_document_from_app_store

router = APIRouter()


@router.get("/documents", response_model=IndexedDocumentsResponse)
async def list_indexed_documents(user_id: str = Depends(verify_token)) -> IndexedDocumentsResponse:
    rows = await list_indexed_documents_from_app_store(user_id=user_id)
    return IndexedDocumentsResponse(
        documents=[IndexedDocumentItem(**r) for r in rows],
    )


@router.post("/documents/remove", response_model=RemoveIndexedDocumentResponse)
async def remove_indexed_document(
    body: RemoveIndexedDocumentRequest,
    user_id: str = Depends(verify_token),
) -> RemoveIndexedDocumentResponse:
    if not body.stored_filename and not body.original_filename:
        raise HTTPException(
            status_code=400,
            detail="Provide stored_filename (recommended) or original_filename.",
        )
    n = await remove_indexed_document_from_app_store(
        stored_filename=body.stored_filename,
        original_filename=body.original_filename,
        user_id=user_id,
    )
    if n == 0:
        raise HTTPException(
            status_code=404,
            detail="No indexed chunks matched. Check the name or refresh the list.",
        )
    return RemoveIndexedDocumentResponse(removed_chunks=n)
