"""
Usage Logger — Structured logging for token consumption, cost estimates, and tool execution.

Design Decision: Usage tracking is isolated from business logic. Swap the sink
(file → DB → Langfuse / Helicone) here without touching agent or API code.
"""
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from core.config import settings
from core.logger import log

USAGE_LOG_FILE = Path("usage_log.jsonl")


def log_usage(
    session_id: str,
    user_message: str,
    prompt_tokens: int,
    completion_tokens: int,
    tools_used: list[str],
    latency_ms: float,
    *,
    user_id: str | None = None,
    rag_chunks_retrieved: int = 0,
    error: str | None = None,
) -> None:
    """
    Appends a structured JSONL record to the usage log.
    JSONL format — one JSON object per line — is stream-parseable and
    compatible with BigQuery, Splunk, and Langfuse out of the box.
    """
    total = prompt_tokens + completion_tokens
    cost = (
        prompt_tokens * settings.COST_PROMPT_PER_TOKEN
        + completion_tokens * settings.COST_COMPLETION_PER_TOKEN
    )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "user_id": user_id or "anonymous",
        "user_message_preview": user_message[:100],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total,
        "estimated_cost_usd": round(cost, 8),
        "tools_used": tools_used,
        "latency_ms": round(latency_ms, 2),
        "rag_chunks_retrieved": rag_chunks_retrieved,
        "error": error,
    }

    try:
        with USAGE_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        log.warning(f"Could not write usage log: {e}")

    log.info(
        f"[USAGE] session={session_id} | "
        f"tokens={total} (p={prompt_tokens}, c={completion_tokens}) | "
        f"cost=${cost:.6f} | tools={tools_used} | latency={latency_ms:.0f}ms"
    )

