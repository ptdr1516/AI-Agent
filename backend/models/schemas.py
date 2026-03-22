from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str


class UploadResponse(BaseModel):
    success: bool = True
    filename: str
    stored_filename: str = Field(description="Disk basename (UUID-prefixed); use for /documents/remove.")
    saved_path: str
    chunks_indexed: int


class IndexedDocumentItem(BaseModel):
    original_filename: str
    stored_filename: str
    chunks: int


class IndexedDocumentsResponse(BaseModel):
    documents: list[IndexedDocumentItem]


class RemoveIndexedDocumentRequest(BaseModel):
    stored_filename: str | None = None
    original_filename: str | None = None


class RemoveIndexedDocumentResponse(BaseModel):
    removed_chunks: int


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int | None = None


class RAGChunkItem(BaseModel):
    filename: str
    preview: str


class RAGQueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks: list[RAGChunkItem] = Field(default_factory=list)
