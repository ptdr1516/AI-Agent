"""
Compatibility facade for the modular execution graph (``agent.graph``).

The compiled graph uses explicit nodes — see ``agent.graph.builder`` and
``agent.graph.nodes``.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from rag.rag_chain import RAGAnswer, RAGChunkInfo

from .graph import get_unified_graph as _get_unified_graph
from .graph.prompts import SYSTEM_PROMPT
from .graph.state import UnifiedGraphState

# Backward-compatible alias for code/tests expecting ``UnifiedState``.
UnifiedState = UnifiedGraphState

__all__ = [
    "SYSTEM_PROMPT",
    "UnifiedState",
    "UnifiedGraphState",
    "chat_recursion_config",
    "final_assistant_text",
    "get_unified_graph",
    "invoke_unified_graph_turn",
]


def get_unified_graph():
    return _get_unified_graph()


def final_assistant_text(messages: Sequence[BaseMessage]) -> str:
    """Best-effort final reply text from the message list (last non-tool AIMessage)."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            calls = getattr(m, "tool_calls", None) or []
            if calls:
                continue
            c = m.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                parts: list[str] = []
                for block in c:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                    elif isinstance(block, dict) and "text" in block:
                        parts.append(str(block["text"]))
                return "".join(parts)
            return str(c)
    return ""


def chat_recursion_config() -> dict[str, Any]:
    """Recursion limit for tool loops (aligned with prior AgentExecutor max_iterations)."""
    from core.config import settings

    n = max(8, settings.AGENT_MAX_ITERATIONS * 2 + 4)
    return {"recursion_limit": n}


async def invoke_unified_graph_turn(
    *,
    messages: list,
    memory_context: str = "",
    rag_top_k: int | None = None,
    retriever=None,
) -> dict[str, Any]:
    """Single graph execution path (retrieval pre-pass + tool-calling LLM)."""
    graph = get_unified_graph()
    cfg: dict[str, Any] = {**chat_recursion_config()}
    if retriever is not None:
        cfg["configurable"] = {"retriever": retriever}
    return await graph.ainvoke(
        {
            "messages": messages,
            "memory_context": memory_context,
            "rag_top_k": rag_top_k,
        },
        config=cfg,
    )



