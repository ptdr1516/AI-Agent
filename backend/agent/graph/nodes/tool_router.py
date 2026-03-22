"""tool_router_node + routing functions (conditional edges)."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage

from ..state import UnifiedGraphState


def after_llm_route(
    state: UnifiedGraphState,
) -> Literal["tool_router_node"]:
    """After ``llm_node`` always continue to tool routing (tools optional)."""
    _ = state
    return "tool_router_node"


def tool_router_node(state: UnifiedGraphState) -> dict:
    """Explicit pass-through node so the graph has a named routing step (no state change)."""
    return {"memory_context": state.get("memory_context", "")}


def route_tools(
    state: UnifiedGraphState,
) -> Literal["tool_execution_node", "output_node"]:
    """After ``tool_router_node``: run tools if the last model turn requested tool calls."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return "output_node"
    calls = last.tool_calls or []
    if not calls:
        return "output_node"
    if state.get("is_last_step"):
        return "output_node"
    return "tool_execution_node"
