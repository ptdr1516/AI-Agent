from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from langchain.tools import BaseTool
from .calculator import calculator_tool
from .custom_api import custom_api_tool
from .sql_db import sql_db_tool
from .web_search import web_search_tool
from .document_search import document_search_tool
from core.logger import log


@dataclass
class ToolMetadata:
    """Structured metadata for every registered tool.
    
    Design Decision: Attaching metadata to tools (rather than scattering
    logic across agent.py) keeps routing logic centralized and makes adding
    new tools or adjusting priority trivially easy without touching the agent.
    """
    tool: BaseTool
    priority: int          # Lower number = higher priority. The agent gets tools sorted by this.
    category: str          # "math" | "search" | "database" | "api"
    description: str       # Human-readable reason this tool exists
    is_async: bool = False
    tags: List[str] = field(default_factory=list)


class ToolRegistry:
    """
    Advanced Tool Registry with metadata, priority routing, and category filtering.
    
    Design Decision: A proper registry decouples *what tools exist* from *how the
    agent uses them*. This mirrors the tool_selector pattern found in production
    frameworks like LangGraph and AutoGPT — you can swap/add tools at runtime
    without rebuilding the entire agent prompt.
    """

    def __init__(self):
        # The lower the priority number, the earlier it appears in the tool list.
        # Agents trained with OpenAI function calling tend to pick the first
        # matching tool, so ordering matters significantly for multi-step tasks.
        self._registry: Dict[str, ToolMetadata] = {
            "calculator": ToolMetadata(
                tool=calculator_tool,
                priority=1,
                category="math",
                description="Evaluates mathematical expressions. Always prefer over text-based guessing.",
                tags=["math", "arithmetic", "compute", "calculation"]
            ),
            "sql_db": ToolMetadata(
                tool=sql_db_tool,
                priority=2,
                category="database",
                description="Executes SQL queries against the internal employee database.",
                tags=["sql", "database", "employees", "salary", "query"]
            ),
            "custom_api": ToolMetadata(
                tool=custom_api_tool,
                priority=3,
                category="api",
                description="Fetches user profile data from public REST API by user_id.",
                is_async=True,
                tags=["api", "user", "profile", "lookup", "json"]
            ),
            "document_search": ToolMetadata(
                tool=document_search_tool,
                priority=4,
                category="rag",
                description="Retrieves relevant passages from user-uploaded indexed documents (local knowledge base).",
                is_async=True,
                tags=["rag", "documents", "upload", "retrieval", "context", "kb", "indexed"]
            ),
            "web_search": ToolMetadata(
                tool=web_search_tool,
                priority=5,
                category="search",
                description="Searches the internet via Tavily API for current events and factual information.",
                is_async=True,
                tags=["search", "internet", "facts", "current_events", "news", "tavily", "realtime"]
            ),
        }
        log.info(
            f"ToolRegistry initialized with {len(self._registry)} tools: "
            f"{[k for k in self._registry.keys()]}"
        )

    def get_all_tools(self) -> List[BaseTool]:
        """Returns all tools sorted by priority (ascending).
        
        The agent sees high-priority deterministic tools (math, SQL) before
        slow non-deterministic tools (web search), making the agent prefer
        cheap/fast local tools over expensive internet calls.
        """
        sorted_entries = sorted(self._registry.values(), key=lambda m: m.priority)
        return [entry.tool for entry in sorted_entries]

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Returns a specific tool by its registry key."""
        meta = self._registry.get(name)
        if not meta:
            log.warning(f"Tool '{name}' requested but not found in registry.")
            return None
        return meta.tool

    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Returns all tools belonging to a specific category (e.g., 'math', 'search')."""
        tools = [m.tool for m in self._registry.values() if m.category == category]
        log.info(f"Routing to {len(tools)} tool(s) in category '{category}'.")
        return tools

    def route_to_tool(self, tag: str) -> Optional[BaseTool]:
        """Semantic tag routing — finds the highest-priority tool matching a tag.
        
        Design Decision: Instead of hardcoding tool selection in the agent prompt,
        we let the registry resolve routing at the component level. The agent can
        hint via tag keywords, and the registry returns the best match.
        """
        matches = [
            m for m in self._registry.values()
            if tag.lower() in m.tags
        ]
        if not matches:
            log.warning(f"No tool found for tag: '{tag}'")
            return None
        best = sorted(matches, key=lambda m: m.priority)[0]
        log.info(f"Tag '{tag}' routed to tool: {best.tool.name} (priority={best.priority})")
        return best.tool

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Returns the full ToolMetadata for a given registry key."""
        return self._registry.get(name)

    def list_metadata(self) -> List[Dict[str, Any]]:
        """Dumps all tool metadata as serializable dicts for API/debug endpoints."""
        return [
            {
                "name": key,
                "tool_name": meta.tool.name,
                "priority": meta.priority,
                "category": meta.category,
                "description": meta.description,
                "is_async": meta.is_async,
                "tags": meta.tags,
            }
            for key, meta in sorted(self._registry.items(), key=lambda x: x[1].priority)
        ]


# Global singleton registry
registry = ToolRegistry()

