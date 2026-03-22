"""Modular RAG: load → chunk → embed → FAISS → retrieve → answer."""

from .chunker import make_text_splitter, split_documents, split_text
from .embeddings import get_embedding_model, get_embeddings, reset_embedding_model_cache
from .loader import (
    documents_from_strings,
    load_directory,
    load_file,
    load_markdown_file,
    load_pdf_file,
    load_text_file,
    load_text_files,
    standardize_document_metadata,
)
from .rag_chain import (
    DEFAULT_RAG_PROMPT,
    NOT_FOUND_PHRASE,
    RAGAnswer,
    RAG_SYSTEM_PROMPT,
    build_context,
    extract_sources,
    format_documents,
)
from .retriever import RAGRetriever, documents_to_payloads
from .runtime_retrieval import retrieve_documents
from .vectorstore import (
    PersistentVectorStore,
    acreate_store,
    aload_store,
    asave_store,
    create_store_sync,
)

__all__ = [
    "DEFAULT_RAG_PROMPT",
    "NOT_FOUND_PHRASE",
    "RAGAnswer",
    "RAG_SYSTEM_PROMPT",
    "build_context",
    "extract_sources",
    "RAGRetriever",
    "documents_to_payloads",
    "retrieve_documents",
    "PersistentVectorStore",
    "acreate_store",
    "aload_store",
    "asave_store",
    "create_store_sync",
    "documents_from_strings",
    "format_documents",
    "get_embedding_model",
    "get_embeddings",
    "load_directory",
    "load_file",
    "load_markdown_file",
    "load_pdf_file",
    "load_text_file",
    "load_text_files",
    "standardize_document_metadata",
    "make_text_splitter",
    "reset_embedding_model_cache",
    "split_documents",
    "split_text",
]
