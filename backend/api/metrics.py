"""
api/metrics.py — Live observability dashboard endpoint.

GET /api/metrics  →  aggregated stats from usage_log.jsonl (no external service needed).

Reads the append-only JSONL file produced by core/usage_logger.py and returns:
  - request_count      total requests logged
  - requests_last_1h   requests in the last 60 minutes
  - total_tokens       sum of prompt + completion tokens
  - total_cost_usd     estimated total spend
  - avg_latency_ms     mean response latency
  - p95_latency_ms     95th-percentile latency
  - top_tools          ranked list of tool names by usage count
  - rag_hit_rate       fraction of requests that retrieved ≥1 chunk
  - error_rate         fraction of requests with recorded errors
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["Observability"])

USAGE_LOG_FILE = Path("usage_log.jsonl")


def _load_records() -> list[dict[str, Any]]:
    """Read all JSONL usage records. Returns empty list if file is absent."""
    if not USAGE_LOG_FILE.exists():
        return []
    records: list[dict[str, Any]] = []
    with USAGE_LOG_FILE.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * 0.95)
    return round(sorted_v[min(idx, len(sorted_v) - 1)], 2)


@router.get("/metrics")
async def get_metrics() -> dict[str, Any]:
    """
    Live usage metrics aggregated from usage_log.jsonl.
    No authentication required — suitable for internal monitoring.
    """
    records = _load_records()

    if not records:
        return {
            "request_count": 0,
            "requests_last_1h": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "top_tools": [],
            "rag_hit_rate": 0.0,
            "error_rate": 0.0,
            "note": "No usage data recorded yet.",
        }

    now = datetime.now(timezone.utc)
    cutoff_1h = now - timedelta(hours=1)

    total_tokens = 0
    total_cost = 0.0
    latencies: list[float] = []
    tool_counter: Counter[str] = Counter()
    rag_hits = 0
    errors = 0
    requests_last_1h = 0

    for rec in records:
        total_tokens += rec.get("total_tokens", 0)
        total_cost += rec.get("estimated_cost_usd", 0.0)

        lat = rec.get("latency_ms")
        if isinstance(lat, (int, float)):
            latencies.append(float(lat))

        for t in rec.get("tools_used", []):
            tool_counter[t] += 1

        chunks = rec.get("rag_chunks_retrieved", rec.get("chunk_count"))
        if chunks and int(chunks) > 0:
            rag_hits += 1

        if rec.get("error"):
            errors += 1

        ts_str = rec.get("timestamp")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff_1h:
                    requests_last_1h += 1
            except ValueError:
                pass

    n = len(records)
    avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    top_tools = [
        {"tool": name, "count": count}
        for name, count in tool_counter.most_common(10)
    ]

    return {
        "request_count": n,
        "requests_last_1h": requests_last_1h,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "avg_latency_ms": avg_lat,
        "p95_latency_ms": _p95(latencies),
        "top_tools": top_tools,
        "rag_hit_rate": round(rag_hits / n, 4) if n else 0.0,
        "error_rate": round(errors / n, 4) if n else 0.0,
    }
