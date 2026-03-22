"""Graph state schema for the single execution graph."""

from __future__ import annotations

from typing import Annotated, Any, NotRequired, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep


class UnifiedGraphState(TypedDict):
    """State for the unified graph (retrieval pre-pass + tool-calling LLM)."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep
    memory_context: NotRequired[str]
    rag_top_k: NotRequired[int | None]
    rag_docs: NotRequired[list[Any]]
    rag_context: NotRequired[str]
    rag_sources: NotRequired[list[str]]
    rag_chunks: NotRequired[list[dict[str, str]]]
