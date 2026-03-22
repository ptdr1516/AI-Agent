"""tool_execution_node — LangGraph ToolNode (must match ``llm_node`` bind_tools list)."""

from __future__ import annotations

from langgraph.prebuilt import ToolNode

from ..tool_calling import get_tools_for_graph


def create_tool_execution_node() -> ToolNode:
    """Execute ``AIMessage.tool_calls`` via the same LangChain tools as ``bind_tools``."""
    return ToolNode(get_tools_for_graph())
