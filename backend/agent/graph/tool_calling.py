"""
Tools bound to the chat agent for **LangChain OpenAI function calling** (same wire format
as OpenAI ``tools`` / ``tool_choice``).

The chat ``llm_node`` uses ``ChatOpenAI.bind_tools(tools)`` so the model emits
``AIMessage.tool_calls``; ``tool_execution_node`` uses ``langgraph.prebuilt.ToolNode``
with the **identical** tool list so each call ID/name executes the correct ``BaseTool``.

Registered tools (registry keys → LangChain ``.name`` for the model):

- ``calculator`` → ``calculator_tool``
- ``sql_db`` → ``sql_db_tool`` (internal SQL / employees DB)
- ``custom_api`` → ``custom_api_tool`` (REST user profile API)
- ``document_search`` → ``document_search``
- ``web_search`` → ``web_search_tool``

Do not build a second tool list elsewhere — always use :func:`get_tools_for_graph`.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from ..tools.registry import registry


def get_tools_for_graph() -> list[BaseTool]:
    """Tools for ``bind_tools`` + ``ToolNode`` (single source of truth)."""
    return registry.get_all_tools()
