"""Tests for the ToolRegistry.

Uses the real registry singleton so we're testing actual production behaviour,
not mocks. All assertions are based on the exact registry.py implementation.
"""
from agent.tools.registry import registry


def test_registry_returns_all_tools():
    """Registry should return all registered tools (calculator, sql, api, document_search, web_search)."""
    tools = registry.get_all_tools()
    assert len(tools) == 5


def test_registry_ordering():
    """Tools must be returned in ascending priority order (calculator first, web search last)."""
    tools = registry.get_all_tools()
    tool_names = [t.name for t in tools]

    # calculator_tool is priority=1; document_search before web_search (priorities 4 and 5)
    assert tool_names.index("calculator_tool") < tool_names.index("document_search")
    assert tool_names.index("document_search") < tool_names.index("web_search_tool")


def test_registry_metadata_web_search():
    """get_metadata() should return correct metadata for the web_search registry key."""
    # Registry keys are "web_search", "calculator", etc. — NOT the langchain tool name
    meta = registry.get_metadata("web_search")
    assert meta is not None
    assert meta.is_async is True
    assert "search" in meta.tags
    assert meta.priority == 5
    assert meta.category == "search"


def test_registry_metadata_calculator():
    """Calculator tool metadata should be correctly configured."""
    meta = registry.get_metadata("calculator")
    assert meta is not None
    assert meta.priority == 1
    assert meta.category == "math"
    assert "math" in meta.tags


def test_registry_route_to_search_tag():
    """Routing by 'search' tag should return the web_search tool (not document_search)."""
    tool = registry.route_to_tool("search")
    assert tool is not None
    assert tool.name == "web_search_tool"


def test_registry_route_to_rag_tag():
    """Routing by 'rag' tag should return document_search."""
    tool = registry.route_to_tool("rag")
    assert tool is not None
    assert tool.name == "document_search"


def test_registry_route_to_math_tag():
    """Routing by 'math' tag should return the calculator tool."""
    tool = registry.route_to_tool("math")
    assert tool is not None
    assert tool.name == "calculator_tool"


def test_registry_route_to_unknown_tag():
    """Routing by an unknown tag should return None gracefully."""
    tool = registry.route_to_tool("nonexistent_tag_xyz")
    assert tool is None


def test_registry_list_metadata():
    """list_metadata() should return a serialisable list of dicts sorted by priority."""
    metadata = registry.list_metadata()
    assert len(metadata) == 5
    # Should be priority-sorted ascending
    priorities = [m["priority"] for m in metadata]
    assert priorities == sorted(priorities)
    # Each entry must have the required keys
    for m in metadata:
        assert "name" in m
        assert "tool_name" in m
        assert "is_async" in m
        assert "tags" in m
