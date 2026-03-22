"""
Tool Metadata API Router — exposes the tool registry for frontend panels and debugging.

Design Decision: Exposing tool metadata via a REST endpoint rather than hardcoding it
in the React frontend means the UI is always in sync with the backend. Add a new tool
in registry.py and the frontend tool panel updates automatically on the next load.
"""
from fastapi import APIRouter
from agent.tools.registry import registry
from core.logger import log

router = APIRouter()


@router.get("/tools")
async def list_tools():
    """Returns all registered tools with their priority, category, and tags."""
    log.info("Tool metadata requested via /api/tools")
    return {"tools": registry.list_metadata()}


@router.get("/tools/{tool_name}")
async def get_tool_info(tool_name: str):
    """Returns metadata for a specific tool by its registry key."""
    meta = registry.get_metadata(tool_name)
    if not meta:
        return {"error": f"Tool '{tool_name}' not found"}
    return {
        "name": tool_name,
        "tool_name": meta.tool.name,
        "priority": meta.priority,
        "category": meta.category,
        "description": meta.description,
        "tags": meta.tags,
    }
