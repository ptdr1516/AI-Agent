"""Modular LangGraph execution pipeline (explicit nodes)."""

from __future__ import annotations

from .builder import build_unified_graph, default_chat_llm, default_llms

_compiled = None
_initialized = False


def get_unified_graph():
    """Singleton compiled graph with lazy initialization."""
    global _compiled, _initialized
    if not _initialized:
        _compiled = build_unified_graph(chat_llm=default_chat_llm())
        _initialized = True
    return _compiled


__all__ = ["build_unified_graph", "default_chat_llm", "default_llms", "get_unified_graph"]
