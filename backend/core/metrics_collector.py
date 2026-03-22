"""
core/metrics_collector.py — Per-request metrics accumulator.

Records per-request:
  - tokens_input / tokens_output
  - total_latency_ms
  - tool_calls_count  (each unique tool invocation)
  - retrieval_docs_count  (number of chunks retrieved)
  - cache_hits  (search cache hits during this request)

Usage:
    # At request start:
    metrics = RequestMetrics.start(session_id=..., user_id=..., query=...)
    bind_metrics(metrics)   # stores in async ContextVar

    # During graph execution (called automatically by cache.py):
    get_current_metrics().add_cache_hit()

    # At request end (call once, does not affect streaming):
    await metrics.finish_and_write(error=None)

The metrics are written to ``request_metrics.jsonl`` — one JSON object per request.
"""
from __future__ import annotations

import json
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from core.logger import log

METRICS_LOG_FILE = Path("request_metrics.jsonl")

# ── ContextVar — one RequestMetrics per async call chain ─────────────────────
_current_metrics: ContextVar["RequestMetrics | None"] = ContextVar(
    "_current_metrics", default=None
)


def get_current_metrics() -> "RequestMetrics | None":
    return _current_metrics.get(None)


def bind_metrics(m: "RequestMetrics") -> None:
    _current_metrics.set(m)


# ── Core dataclass ────────────────────────────────────────────────────────────

@dataclass
class RequestMetrics:
    # Identity
    session_id: str
    user_id: str
    query_preview: str          # first 120 chars of the user query
    endpoint: str               # "chat_stream" | "rag_query" | "chat_sync"

    # Populated during request
    tokens_input: int = 0
    tokens_output: int = 0
    tool_calls_count: int = 0
    tool_names: list[str] = field(default_factory=list)
    retrieval_docs_count: int = 0
    cache_hits: int = 0

    # Timing
    _start_ts: float = field(default_factory=time.perf_counter, repr=False)

    # Written at finish
    total_latency_ms: float = 0.0
    timestamp: str = ""
    status: str = "ok"          # "ok" | "error"
    error: str = ""

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def start(cls, *, session_id: str, user_id: str, query: str, endpoint: str) -> "RequestMetrics":
        m = cls(
            session_id=session_id,
            user_id=user_id,
            query_preview=query[:120],
            endpoint=endpoint,
        )
        return m

    # ── Accumulators (safe to call from anywhere in the async chain) ──────────

    def add_tokens(self, prompt: int, completion: int) -> None:
        self.tokens_input += prompt
        self.tokens_output += completion

    def add_tool_call(self, tool_name: str) -> None:
        self.tool_calls_count += 1
        if tool_name not in self.tool_names:
            self.tool_names.append(tool_name)

    def set_retrieval_docs(self, count: int) -> None:
        """Overwrite with the definitive count (set once by retrieval_node)."""
        self.retrieval_docs_count = max(self.retrieval_docs_count, count)

    def add_cache_hit(self) -> None:
        self.cache_hits += 1

    # ── Finalise + write ──────────────────────────────────────────────────────

    def finish(self, *, error: str = "") -> dict:
        """Compute latency and build the record dict. Call once at request end."""
        self.total_latency_ms = round((time.perf_counter() - self._start_ts) * 1000, 1)
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.status = "error" if error else "ok"
        self.error = error

        record = {
            "timestamp": self.timestamp,
            "endpoint": self.endpoint,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "query_preview": self.query_preview,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "total_tokens": self.tokens_input + self.tokens_output,
            "total_latency_ms": self.total_latency_ms,
            "tool_calls_count": self.tool_calls_count,
            "tool_names": self.tool_names,
            "retrieval_docs_count": self.retrieval_docs_count,
            "cache_hits": self.cache_hits,
            "status": self.status,
        }
        if self.error:
            record["error"] = self.error
        return record

    def write(self, record: dict) -> None:
        """Append the record to request_metrics.jsonl."""
        try:
            with METRICS_LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as e:
            log.warning(f"[METRICS] failed to write metrics: {e}")

    def finish_and_log(self, *, error: str = "") -> None:
        """Compute latency, write JSONL, emit a structured log line. Non-blocking."""
        record = self.finish(error=error)
        self.write(record)
        _emit_log(record)


def _emit_log(r: dict) -> None:
    status = r.get("status", "ok")
    line = (
        f"[METRICS] endpoint={r['endpoint']!r} "
        f"| session={r['session_id']!r} "
        f"| tokens_in={r['tokens_input']} tokens_out={r['tokens_output']} "
        f"| latency_ms={r['total_latency_ms']} "
        f"| tool_calls={r['tool_calls_count']} tools={r['tool_names']} "
        f"| rag_docs={r['retrieval_docs_count']} "
        f"| cache_hits={r['cache_hits']} "
        f"| status={status}"
    )
    if status == "error":
        log.error(line)
    else:
        log.info(line)
