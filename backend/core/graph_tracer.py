"""
core/graph_tracer.py — Per-node execution tracing for the LangGraph.

Design:
  - `wrap_node(fn, name)` wraps any LangGraph node callable transparently.
  - Each node invocation writes one JSON record to ``graph_traces.jsonl``.
  - No changes required in node files or API endpoints.

Trace record schema (one JSON object per line):
    {
      "ts":               ISO-8601 UTC timestamp,
      "trace_id":         UUID shared across all nodes in one graph invocation,
      "node":             node function name (e.g. "retrieval_node"),
      "session_id":       from state (when present),
      "user_id":          from configurable (when present),
      "latency_ms":       wall-clock time for the node,
      "status":           "ok" | "error",
      "error":            error message (only on "error"),
      "rag_chunks":       count of retrieved RAG chunks (retrieval_node only),
      "rag_sources":      list of source filenames (retrieval_node only),
      "tool_calls":       list of tool names called (llm_node only),
      "tokens":           {"prompt": N, "completion": N} (llm_node only),
      "tools_executed":   list of tool names (tool_execution_node only),
    }
"""
from __future__ import annotations

import json
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.logger import log

TRACE_LOG_FILE = Path("graph_traces.jsonl")

# One trace_id per graph invocation — shared across all nodes in that run
_current_trace_id: ContextVar[str] = ContextVar("_current_trace_id", default="")


def new_trace_id() -> str:
    """Generate a fresh trace ID and store it in the current async context."""
    tid = uuid.uuid4().hex
    _current_trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    tid = _current_trace_id.get("")
    return tid or uuid.uuid4().hex


# ── JSONL writer ──────────────────────────────────────────────────────────────

def _write_trace(record: dict[str, Any]) -> None:
    try:
        with TRACE_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as e:
        log.warning(f"[GRAPH_TRACE] could not write trace: {e}")


# ── State extractors (safe — never throw) ────────────────────────────────────

def _extract_input_meta(state: Any) -> dict[str, Any]:
    """Pull useful fields from the graph state dict before node runs."""
    if not isinstance(state, dict):
        return {}
    meta: dict[str, Any] = {}
    sid = state.get("session_id")
    if sid:
        meta["session_id"] = str(sid)
    return meta


def _extract_output_meta(node_name: str, output: Any) -> dict[str, Any]:
    """Pull node-specific metrics from the node's return dict."""
    if not isinstance(output, dict):
        return {}
    meta: dict[str, Any] = {}

    if node_name == "retrieval_node":
        docs = output.get("rag_docs") or []
        meta["rag_chunks"] = len(docs)
        meta["rag_sources"] = output.get("rag_sources") or []

    if node_name == "llm_node":
        messages = output.get("messages") or []
        tool_calls: list[str] = []
        tokens: dict[str, int] = {"prompt": 0, "completion": 0}
        for msg in messages:
            # Extract tool calls
            calls = getattr(msg, "tool_calls", None) or []
            tool_calls.extend(c.get("name", str(c)) if isinstance(c, dict) else getattr(c, "name", "") for c in calls)
            # Extract token usage (OpenAI usage_metadata)
            usage = getattr(msg, "usage_metadata", None) or {}
            if usage:
                tokens["prompt"] += usage.get("input_tokens", 0)
                tokens["completion"] += usage.get("output_tokens", 0)
        if tool_calls:
            meta["tool_calls"] = tool_calls
        if tokens["prompt"] or tokens["completion"]:
            meta["tokens"] = tokens

    if node_name == "tool_execution_node":
        messages = output.get("messages") or []
        tools_executed = [
            getattr(m, "name", None) or getattr(m, "tool_call_id", "unknown")
            for m in messages
            if hasattr(m, "tool_call_id")  # ToolMessage
        ]
        if tools_executed:
            meta["tools_executed"] = tools_executed

    return meta


# ── Core wrapper ─────────────────────────────────────────────────────────────

def wrap_node(fn: Callable, node_name: str) -> Callable:
    """
    Return a transparent async wrapper around a LangGraph node function.

    Captures: latency, rag_chunks, rag_sources, tool_calls, tokens, status.
    Writes one JSONL record to graph_traces.jsonl per node invocation.
    """
    import functools
    from langchain_core.runnables import RunnableConfig

    from core.config import settings
    if not settings.ENABLE_TRACING:
        return fn

    async def _traced(state, config=None):
        trace_id = get_trace_id()
        configurable = (config or {}).get("configurable") or {}
        user_id = configurable.get("user_id", "")
        input_meta = _extract_input_meta(state)

        t0 = time.perf_counter()
        status = "ok"
        error_msg = ""
        output: Any = {}

        try:
            # Helper to execute the node based on its type (plain fn vs Runnable)
            async def _execute_node():
                # Try calling directly first (functions/methods)
                if callable(fn):
                    try:
                        # Try 2-arg signature first, then 1-arg fallback
                        try:
                            return fn(state, config) if config is not None else fn(state)
                        except TypeError:
                            return fn(state)
                    except Exception:
                        raise
                
                # If not directly callable, it might be a LangChain Runnable (like ToolNode)
                if hasattr(fn, "ainvoke"):
                    return await fn.ainvoke(state, config=config)
                if hasattr(fn, "invoke"):
                    return fn.invoke(state, config=config)
                
                # Final fallback for generic non-callable objects
                raise TypeError(f"Node object of type {type(fn)} is not callable and has no ainvoke/invoke methods.")

            val = await _execute_node()
            import inspect as _ins
            if _ins.iscoroutine(val):
                output = await val
            else:
                output = val
            return output
        except Exception as exc:
            status = "error"
            error_msg = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            output_meta = _extract_output_meta(node_name, output)

            record: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "trace_id": trace_id,
                "node": node_name,
                "user_id": user_id,
                "latency_ms": latency_ms,
                "status": status,
                **input_meta,
                **output_meta,
            }
            if error_msg:
                record["error"] = error_msg

            _write_trace(record)
            _emit_log(node_name, latency_ms, status, output_meta, error_msg)

    # Do not use functools.wraps since it copies __annotations__ but misses __globals__,
    # causing typing.get_type_hints to crash on stringified types like 'UnifiedGraphState'.
    # Do NOT set __wrapped__ — if fn is a class instance (e.g. ToolNode), it causes
    # inspect.signature(follow_wrapped=True) to crash with a descriptor TypeError.
    # CRITICAL: explicitly clear __annotations__ — `from __future__ import annotations`
    # in this module stringifies any annotation (e.g. 'RunnableConfig | None') into
    # _traced.__annotations__. When LangGraph calls get_type_hints(_traced) it evaluates
    # those strings in input_node's globals where RunnableConfig is not defined → NameError.
    _traced.__annotations__ = {}
    _traced.__name__ = getattr(fn, "__name__", node_name)
    _traced.__doc__ = getattr(fn, "__doc__", None)

    return _traced


def _emit_log(node: str, latency_ms: float, status: str, meta: dict[str, Any], error: str) -> None:
    """Print a compact structured log line for the node span."""
    parts = [f"[NODE] {node}", f"status={status}", f"latency_ms={latency_ms}"]
    if "rag_chunks" in meta:
        parts.append(f"rag_chunks={meta['rag_chunks']}")
    if "rag_sources" in meta:
        parts.append(f"sources={meta['rag_sources']}")
    if "tool_calls" in meta:
        parts.append(f"tool_calls={meta['tool_calls']}")
    if "tokens" in meta:
        t = meta["tokens"]
        parts.append(f"tokens(p={t['prompt']},c={t['completion']})")
    if "tools_executed" in meta:
        parts.append(f"tools_executed={meta['tools_executed']}")
    if error:
        parts.append(f"error={error!r}")
    line = " | ".join(parts)
    if status == "error":
        log.error(line)
    else:
        log.info(line)
