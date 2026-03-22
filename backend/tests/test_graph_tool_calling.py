"""Graph uses the same tools for bind_tools and ToolNode (function calling)."""

from agent.graph.tool_calling import get_tools_for_graph
from agent.tools.registry import registry


def test_graph_tools_match_registry():
    a = [t.name for t in registry.get_all_tools()]
    b = [t.name for t in get_tools_for_graph()]
    assert a == b


def test_expected_tool_names_for_model():
    """LangChain tool names exposed to the LLM (OpenAI function calling)."""
    names = {t.name for t in get_tools_for_graph()}
    assert names == {
        "calculator_tool",
        "sql_db_tool",
        "custom_api_tool",
        "document_search",
        "web_search_tool",
    }
