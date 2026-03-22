"""
core/retrieval_diagnostics.py — Logs detailed retrieval diagnostics for evaluation.

Tracks:
  - query text
  - number of retrieved docs
  - similarity scores
  - source filenames
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.logger import log
from core.config import settings

DIAGNOSTICS_LOG_FILE = Path("retrieval_diagnostics.jsonl")


def log_retrieval_diagnostics(
    query: str,
    docs_and_scores: list[tuple[Any, float]],
    user_id: str | None = None,
) -> None:
    """
    Log retrieval diagnostics to retrieval_diagnostics.jsonl.
    
    docs_and_scores is a list of (Document, float_score) tuples.
    """
    if not settings.ENABLE_EVAL:
        return

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "user_id": user_id or "anonymous",
        "retrieved_count": len(docs_and_scores),
        "docs": [
            {
                "score": round(score, 4) if isinstance(score, (float, int)) else score,
                "filename": doc.metadata.get("source") or doc.metadata.get("filename") or "unknown",
            }
            for doc, score in docs_and_scores
        ],
    }
    
    try:
        with DIAGNOSTICS_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as e:
        log.warning(f"[DIAGNOSTICS] failed to write: {e}")
