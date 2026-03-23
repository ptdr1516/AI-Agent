"""Compile the modular LangGraph: input → retrieval → llm → tool_router → [tools|output]."""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from core.config import settings
from core.logger import log
from core.graph_tracer import wrap_node

from .nodes import (
    after_llm_route,
    create_llm_node,
    create_tool_execution_node,
    input_node,
    output_node,
    retrieval_node,
    route_tools,
    tool_router_node,
)
from .state import UnifiedGraphState


def build_unified_graph(
    *,
    chat_llm: ChatOpenAI,
):
    """Wire nodes: input → retrieval → llm → tool_router → tool_execution|output (single path).

    Every node is wrapped with ``core.graph_tracer.wrap_node`` which writes a
    structured JSONL trace record to ``graph_traces.jsonl`` per invocation.
    """
    llm_node = create_llm_node(chat_llm)
    tool_execution_node = create_tool_execution_node()

    workflow = StateGraph(UnifiedGraphState)

    # ── Each node is wrapped transparently for per-node tracing ──────────────
    workflow.add_node("input_node",          wrap_node(input_node,          "input_node"))
    workflow.add_node("retrieval_node",      wrap_node(retrieval_node,      "retrieval_node"))
    workflow.add_node("llm_node",            wrap_node(llm_node,            "llm_node"))
    workflow.add_node("tool_router_node",    wrap_node(tool_router_node,    "tool_router_node"))
    workflow.add_node("tool_execution_node", wrap_node(tool_execution_node, "tool_execution_node"))
    workflow.add_node("output_node",         wrap_node(output_node,         "output_node"))

    workflow.add_edge(START, "input_node")
    workflow.add_edge("input_node", "retrieval_node")
    workflow.add_edge("retrieval_node", "llm_node")

    workflow.add_conditional_edges(
        "llm_node",
        after_llm_route,
        {
            "tool_router_node": "tool_router_node",
        },
    )

    workflow.add_conditional_edges(
        "tool_router_node",
        route_tools,
        {
            "tool_execution_node": "tool_execution_node",
            "output_node": "output_node",
        },
    )

    workflow.add_edge("tool_execution_node", "llm_node")
    workflow.add_edge("output_node", END)

    compiled = workflow.compile()
    log.info(
        "Unified LangGraph compiled (with node tracing): "
        "input_node → retrieval_node → llm_node → tool_router_node → "
        "tool_execution_node|output_node"
    )
    return compiled


def default_chat_llm() -> ChatOpenAI:
    """Default streaming chat model (tools + retrieval context)."""
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=settings.LLM_API_BASE,
        streaming=True,
        default_headers={
            "HTTP-Referer": settings.HTTP_REFERER,
            "X-Title": settings.APP_NAME,
        },
    )


def default_llms() -> tuple[ChatOpenAI, ChatOpenAI]:
    """Backward-compatible pair; both entries are the same chat LLM."""
    llm = default_chat_llm()
    return llm, llm
