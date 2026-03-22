"""Singleton LangChain ``Embeddings`` — HuggingFace (sentence-transformers) or OpenAI-compatible."""

from __future__ import annotations

from threading import Lock
from typing import Any, Optional

from langchain_core.embeddings import Embeddings

from core.config import settings
from core.logger import log

_lock = Lock()
_instance: Optional[Embeddings] = None


def _normalize_provider(raw: str) -> str:
    p = raw.lower().strip()
    if p in ("huggingface", "hf", "sentence-transformers", "sentence_transformers"):
        return "huggingface"
    if p in ("openai", "open_ai"):
        return "openai"
    return p


def _build_huggingface_embeddings() -> Embeddings:
    from langchain_community.embeddings import HuggingFaceEmbeddings

    log.info(f"Embeddings singleton: HuggingFace / sentence-transformers ({settings.EMBEDDING_MODEL})")
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _build_openai_embeddings() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    api_key = settings.EMBEDDING_OPENAI_API_KEY or settings.OPENROUTER_API_KEY
    model = settings.OPENAI_EMBEDDING_MODEL
    log.info(f"Embeddings singleton: OpenAI-compatible ({model})")

    kwargs: dict[str, Any] = {
        "model": model,
        "openai_api_key": api_key,
    }
    # Omit base → official OpenAI API. Set EMBEDDING_OPENAI_API_BASE for OpenRouter, Azure, etc.
    if settings.EMBEDDING_OPENAI_API_BASE:
        kwargs["openai_api_base"] = settings.EMBEDDING_OPENAI_API_BASE
    return OpenAIEmbeddings(**kwargs)


def _create_embeddings() -> Embeddings:
    provider = _normalize_provider(settings.EMBEDDING_PROVIDER)
    if provider == "huggingface":
        return _build_huggingface_embeddings()
    if provider == "openai":
        return _build_openai_embeddings()
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={settings.EMBEDDING_PROVIDER!r}. "
        "Use 'huggingface' or 'openai'."
    )


def get_embedding_model() -> Embeddings:
    """Return the shared ``Embeddings`` instance (thread-safe singleton, lazy init)."""
    global _instance
    with _lock:
        if _instance is None:
            _instance = _create_embeddings()
    return _instance


def get_embeddings() -> Embeddings:
    """Alias for :func:`get_embedding_model`."""
    return get_embedding_model()


def reset_embedding_model_cache() -> None:
    """Clear singleton (e.g. tests or provider switch at runtime)."""
    global _instance
    with _lock:
        _instance = None
