"""output_node — terminal step (response materializes from ``messages`` for callers)."""

from __future__ import annotations

from typing import Any

from ..state import UnifiedGraphState


def output_node(state: UnifiedGraphState) -> dict[str, Any]:
    """No-op merge; API layers read ``messages`` / RAG fields from state."""
    return {"memory_context": state.get("memory_context", "")}
