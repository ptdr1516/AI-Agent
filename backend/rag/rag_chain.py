"""RAG helpers: context formatting, prompts, and :class:`RAGChain` (graph or LCEL)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from .retriever import RAGRetriever

NOT_FOUND_PHRASE = "Not found in the provided sources."

RAG_SYSTEM_PROMPT = """You are a retrieval-only assistant.

Rules (follow strictly):
1. Answer using ONLY the information in "Context" below. Do not use prior knowledge or guess.
2. If the context does not contain the answer, reply with exactly this sentence and nothing else:
   "{not_found_phrase}"
3. When you answer, name the source(s) you relied on using the "Source:" labels from the context (file names or paths shown in brackets).

Context:
{context}
"""


DEFAULT_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def _display_source_name(doc: Document) -> str:
    meta = doc.metadata or {}
    raw = meta.get("original_filename") or meta.get("filename") or meta.get("source") or "unknown"
    if raw == "unknown":
        return raw
    s = str(raw)
    try:
        return Path(s).name
    except Exception:
        return s


def extract_sources(documents: Sequence[Document]) -> list[str]:
    """Unique source names for attribution (prefer ``filename``, else basename of ``source``)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for doc in documents:
        name = _display_source_name(doc)
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def build_context(documents: Sequence[Document]) -> str:
    """Format retrieved chunks for the prompt; each block includes its source name."""
    if not documents:
        return "(No relevant passages were retrieved.)"
    parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        label = _display_source_name(doc)
        parts.append(f"[{i}] Source: {label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


_CHUNK_BLOCK_RE = re.compile(
    r"^\[(?P<idx>\d+)\]\s+Source:\s+(?P<label>.+?)\n(?P<body>.*)\s*$",
    re.DOTALL,
)


def parse_build_context(text: str) -> dict[str, Any]:
    """Parse strings produced by :func:`build_context` into sources + chunk previews (for UI / SSE)."""
    text = (text or "").strip()
    if not text or text.startswith("(No relevant"):
        return {"sources": [], "chunks": []}
    chunks: list[dict[str, str]] = []
    seen: set[str] = set()
    sources_ordered: list[str] = []
    for block in text.split("\n\n---\n\n"):
        block = block.strip()
        if not block:
            continue
        m = _CHUNK_BLOCK_RE.match(block)
        if not m:
            continue
        filename = m.group("label").strip()
        preview = m.group("body").strip()
        chunks.append({"filename": filename, "preview": preview})
        if filename not in seen:
            seen.add(filename)
            sources_ordered.append(filename)
    return {"sources": sources_ordered, "chunks": chunks}


def format_documents(
    documents: list,
    *,
    include_metadata: bool = False,
) -> str:
    """Backward-compatible alias; prefer :func:`build_context` for RAG prompts."""
    if include_metadata:
        docs = [d for d in documents if hasattr(d, "metadata")]
        return build_context(docs)
    return build_context(documents)


@dataclass
class RAGChunkInfo:
    """One retrieved segment for API / UI display."""

    filename: str
    preview: str


@dataclass
class RAGAnswer:
    """LLM answer plus source names and chunk previews from retrieved documents."""

    answer: str
    sources: list[str] = field(default_factory=list)
    chunks: list[RAGChunkInfo] = field(default_factory=list)



