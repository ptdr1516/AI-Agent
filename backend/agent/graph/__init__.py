"""Modular LangGraph execution pipeline (explicit nodes)."""

from __future__ import annotations

from .builder import build_unified_graph, default_chat_llm, default_llms

_compiled = None


def get_unified_graph():
    """Singleton compiled graph."""
    global _compiled
    if _compiled is None:
        _compiled = build_unified_graph(chat_llm=default_chat_llm())
    return _compiled


__all__ = ["build_unified_graph", "default_chat_llm", "default_llms", "get_unified_graph"]
