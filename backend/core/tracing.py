"""
core/tracing.py — Lightweight observability layer.

Provides:
  1. configure_langsmith()   — enable LangSmith distributed tracing at startup
  2. @trace_span("name")     — async decorator that times any coroutine and emits
                               a structured [TRACE] log line (Datadog/Grafana/CloudWatch-ready)
  3. TraceContext            — context manager for inline span blocks

Architecture note: purely additive. No imports required in nodes/tools — just
decorate the function you want measured.
"""
from __future__ import annotations

import functools
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable

from core.logger import log


# ── LangSmith wiring ─────────────────────────────────────────────────────────

def configure_langsmith() -> None:
    """
    Activate LangSmith distributed tracing if credentials are configured.

    LangChain / LangGraph auto-trace every chain, LLM call, and tool invocation
    when these env-vars are set — no other code change required.
    """
    try:
        from core.config import settings
        import os

        if settings.LANGCHAIN_TRACING_V2.lower() == "true" and settings.LANGCHAIN_API_KEY:
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            os.environ.setdefault("LANGCHAIN_API_KEY", settings.LANGCHAIN_API_KEY)
            os.environ.setdefault("LANGCHAIN_PROJECT", settings.LANGCHAIN_PROJECT)
            os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.LANGCHAIN_ENDPOINT)
            log.info(
                f"[TRACING] LangSmith enabled → project={settings.LANGCHAIN_PROJECT!r}"
            )
        else:
            log.info("[TRACING] LangSmith disabled (set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY)")
    except Exception as e:
        log.warning(f"[TRACING] LangSmith setup skipped: {e}")


# ── Structured span logger ────────────────────────────────────────────────────

def _emit_span(
    name: str,
    duration_ms: float,
    status: str,
    metadata: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """Emit a structured TRACE log line. One line per span → easy to grep/aggregate."""
    parts = [
        f"[TRACE] span={name!r}",
        f"status={status}",
        f"duration_ms={duration_ms:.1f}",
    ]
    if metadata:
        for k, v in metadata.items():
            parts.append(f"{k}={v!r}")
    if error:
        parts.append(f"error={error!r}")

    line = " | ".join(parts)
    if status == "error":
        log.error(line)
    else:
        log.info(line)


# ── Decorator ─────────────────────────────────────────────────────────────────

def trace_span(
    name: str,
    *,
    metadata_fn: Callable[..., dict[str, Any]] | None = None,
) -> Callable:
    """
    Decorator for async functions. Wraps the call in a timed span.

    Usage:
        @trace_span("retrieval")
        async def retrieve_documents(query: str, *, user_id=None, ...): ...

        @trace_span("tool.web_search", metadata_fn=lambda q: {"query_len": len(q)})
        async def search(query: str): ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            meta: dict[str, Any] = {}
            if metadata_fn:
                try:
                    meta = metadata_fn(*args, **kwargs) or {}
                except Exception:
                    pass

            t0 = time.perf_counter()
            try:
                result = await fn(*args, **kwargs)
                duration_ms = (time.perf_counter() - t0) * 1000

                # Enrich with result size hint where possible
                if isinstance(result, (list, tuple)):
                    meta["result_count"] = len(result)
                elif isinstance(result, str):
                    meta["result_len"] = len(result)

                _emit_span(name, duration_ms, "ok", meta)
                return result
            except Exception as exc:
                duration_ms = (time.perf_counter() - t0) * 1000
                err_summary = f"{type(exc).__name__}: {exc}"
                _emit_span(name, duration_ms, "error", meta, error=err_summary)
                raise

        return wrapper
    return decorator


# ── Context manager ───────────────────────────────────────────────────────────

@asynccontextmanager
async def span(name: str, **meta: Any) -> AsyncIterator[dict[str, Any]]:
    """
    Async context manager for inline span blocks.

    Usage:
        async with span("ingest", chunks=len(docs)) as s:
            await vectorstore.aadd_documents(docs)
            s["chunks_written"] = len(docs)
    """
    t0 = time.perf_counter()
    ctx: dict[str, Any] = dict(meta)
    try:
        yield ctx
        duration_ms = (time.perf_counter() - t0) * 1000
        _emit_span(name, duration_ms, "ok", ctx)
    except Exception as exc:
        duration_ms = (time.perf_counter() - t0) * 1000
        err_summary = f"{type(exc).__name__}: {exc}"
        _emit_span(name, duration_ms, "error", ctx, error=err_summary)
        raise
