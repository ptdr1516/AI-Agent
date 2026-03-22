"""input_node — normalize incoming graph state."""

from __future__ import annotations

from typing import Any

from ..state import UnifiedGraphState


def input_node(state: UnifiedGraphState) -> dict[str, Any]:
    """Ensure optional fields exist."""
    if "memory_context" not in state or state.get("memory_context") is None:
        return {"memory_context": ""}
    return {"memory_context": state.get("memory_context", "")}
