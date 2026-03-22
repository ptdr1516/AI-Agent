#!/usr/bin/env python3
"""
RAG evaluation: fixture corpus → in-memory FAISS → retrieve → RAG chain → checks.

Usage (from ``backend/``)::

    python scripts/eval_rag.py
    python scripts/eval_rag.py --retrieval-only   # no LLM; retrieval + source labels only
    python scripts/eval_rag.py --json             # write eval_report.json
    python scripts/eval_rag.py --push-langsmith   # upload results to LangSmith dataset

Requires ``OPENROUTER_API_KEY`` in environment or ``.env`` unless ``--retrieval-only``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# Run as ``python scripts/eval_rag.py`` from repo — add backend root to path
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.config import settings  # noqa: E402
from rag.rag_chain import _display_source_name  # noqa: E402
from rag.retriever import RAGRetriever  # noqa: E402
from rag.vectorstore import acreate_store  # noqa: E402


# ── Synthetic corpus (deterministic filenames & facts) ───────────────────────
FIXTURE_DOCUMENTS: list[Document] = [
    Document(
        page_content=(
            "Nova RAG evaluation fixture. Project Alpha uses access code ALPHA-7749. "
            "The Alpha lead is Dr. Chen. Contact: alpha-team@example.internal."
        ),
        metadata={"filename": "project_alpha.md"},
    ),
    Document(
        page_content=(
            "Operations note: The Q3 budget figures live in spreadsheet budget_2024.xlsx. "
            "Shipping discounts use code BLUE-100 only."
        ),
        metadata={"filename": "misc_notes.txt"},
    ),
    Document(
        page_content=(
            "HR policy excerpt: Remote employees must submit timesheets by Friday 17:00 UTC. "
            "Policy document ref HR-REMOTE-2024."
        ),
        metadata={"filename": "hr_policy.pdf"},
    ),
]


@dataclass
class EvalCase:
    id: str
    query: str
    """Filenames that must appear among retrieved chunks (at least one hit each)."""
    must_retrieve_filenames: list[str] = field(default_factory=list)
    """Filenames that must appear in RAGAnswer.sources after full chain."""
    must_list_sources: list[str] = field(default_factory=list)
    """Substrings that must appear in concatenated retrieved chunk text (retrieval relevance)."""
    phrases_in_context: list[str] = field(default_factory=list)
    """Substrings that must appear in the model answer (grounding to retrieved facts)."""
    phrases_in_answer: list[str] = field(default_factory=list)
    min_chunks: int = 1


@dataclass
class EvalResult:
    """Structured result for one eval case — written to JSON report."""
    case_id: str
    query: str
    retrieval_pass: bool
    sources_pass: bool
    grounding_pass: bool
    grounding_score: float        # 0.0–1.0 phrase hit ratio
    retrieval_latency_ms: float
    llm_latency_ms: float
    retrieved_filenames: list[str]
    answer_preview: str
    failure_reason: str = ""

EVAL_CASES: list[EvalCase] = [
    EvalCase(
        id="alpha_access_code",
        query="What is the access code for Project Alpha?",
        must_retrieve_filenames=["project_alpha.md"],
        must_list_sources=["project_alpha.md"],
        phrases_in_context=["ALPHA-7749", "Project Alpha"],
        phrases_in_answer=["ALPHA-7749"],
        min_chunks=1,
    ),
    EvalCase(
        id="q3_budget_file",
        query="Where is the Q3 budget stored?",
        must_retrieve_filenames=["misc_notes.txt"],
        must_list_sources=["misc_notes.txt"],
        phrases_in_context=["budget_2024.xlsx", "Q3"],
        phrases_in_answer=["budget_2024"],
        min_chunks=1,
    ),
    EvalCase(
        id="hr_timesheet_deadline",
        query="When must remote employees submit timesheets?",
        must_retrieve_filenames=["hr_policy.pdf"],
        must_list_sources=["hr_policy.pdf"],
        phrases_in_context=["Friday", "17:00", "UTC"],
        phrases_in_answer=["Friday"],
        min_chunks=1,
    ),
]


def _norm(s: str) -> str:
    return s.lower()


def _retrieved_filenames(docs: list[Document]) -> set[str]:
    return {_display_source_name(d) for d in docs}


def _context_blob(docs: list[Document]) -> str:
    return "\n".join(d.page_content for d in docs)


def check_retrieval(case: EvalCase, docs: list[Document]) -> tuple[bool, str]:
    if len(docs) < case.min_chunks:
        return False, f"expected >= {case.min_chunks} chunks, got {len(docs)}"
    names = _retrieved_filenames(docs)
    for fn in case.must_retrieve_filenames:
        if fn not in names:
            return False, f"missing retrieved source {fn!r}; got {sorted(names)}"
    ctx = _context_blob(docs)
    for phrase in case.phrases_in_context:
        if _norm(phrase) not in _norm(ctx):
            return False, f"context missing phrase {phrase!r}"
    return True, "ok"


def check_sources(case: EvalCase, listed: list[str]) -> tuple[bool, str]:
    listed_set = set(listed)
    for fn in case.must_list_sources:
        if fn not in listed_set:
            return False, f"sources list missing {fn!r}; got {sorted(listed_set)}"
    return True, "ok"


def check_answer_grounding(case: EvalCase, answer: str) -> tuple[bool, str]:
    a = _norm(answer)
    for phrase in case.phrases_in_answer:
        if _norm(phrase) not in a:
            return False, f"answer missing grounding phrase {phrase!r}"
    return True, "ok"


def print_row(cid: str, retrieval: str, sources: str, grounded: str, detail: str) -> None:
    print(f"  {cid:24}  retrieval: {retrieval:5}  sources: {sources:5}  grounded: {grounded:5}")
    if detail:
        print(f"    -> {detail}")


async def run_eval(retrieval_only: bool, verbose: bool, write_json: bool = False, push_langsmith: bool = False) -> int:
    print("Building in-memory FAISS from fixture documents…")
    vs = await acreate_store(FIXTURE_DOCUMENTS)
    retriever = RAGRetriever(vs, default_top_k=settings.RAG_RETRIEVAL_K)

    graph = None
    if not retrieval_only:
        from agent.unified_graph import get_unified_graph
        graph = get_unified_graph()

    check_failures = 0
    eval_results: list[EvalResult] = []
    print()
    print(f"{'case_id':24}  {'retrieval':^12}  {'sources':^10}  {'grounded':^10}  {'score':^6}  {'ret_ms':^8}")
    print("-" * 90)

    for case in EVAL_CASES:
        docs = await retriever.aretrieve(case.query)
        ok_r, msg_r = check_retrieval(case, docs)
        ok_s = ok_g = True
        msg_s = msg_g = ""
        answer = ""
        listed: list[str] = []

        if verbose:
            print(f"\n--- {case.id} ---\nquery: {case.query}")
            for i, d in enumerate(docs, 1):
                print(f"  [{i}] {_display_source_name(d)}: {d.page_content[:200]!r}…")

        if not ok_r:
            check_failures += 1
            print_row(case.id, "FAIL", "SKIP", "SKIP", msg_r)
            continue

        if retrieval_only:
            listed = sorted(_retrieved_filenames(docs))
            ok_s, msg_s = check_sources(case, listed)
            ok_g, msg_g = True, ""
            if not ok_s:
                check_failures += 1
            print_row(
                case.id,
                "PASS",
                "PASS" if ok_s else "FAIL",
                "SKIP",
                msg_s if not ok_s else "(retrieval-only: no LLM)",
            )
            continue

        assert graph is not None
        from agent.unified_graph import chat_recursion_config, final_assistant_text
        from langchain_core.messages import HumanMessage
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=case.query)],
                "memory_context": "",
                "rag_top_k": settings.RAG_RETRIEVAL_K,
            },
            config={"configurable": {"retriever": retriever}, **chat_recursion_config()},
        )
        answer = final_assistant_text(result["messages"])
        listed = result.get("rag_sources") or []
        ok_s, msg_s = check_sources(case, listed)
        ok_g, msg_g = check_answer_grounding(case, answer)
        if not ok_s:
            check_failures += 1
        if not ok_g:
            check_failures += 1

        detail = ""
        if not ok_s:
            detail = msg_s
        if not ok_g:
            detail = f"{detail} {msg_g}".strip()

        print_row(case.id, "PASS", "PASS" if ok_s else "FAIL", "PASS" if ok_g else "FAIL", detail)
        if verbose and answer:
            print(f"    answer: {answer[:300]}{'…' if len(answer) > 300 else ''}")

    print("-" * 90)
    total = len(EVAL_CASES)
    status = "ALL PASSED" if check_failures == 0 else f"{check_failures} CHECK(S) FAILED"
    avg_ret = sum(r.retrieval_latency_ms for r in eval_results) / len(eval_results) if eval_results else 0
    avg_llm = sum(r.llm_latency_ms for r in eval_results if r.llm_latency_ms > 0)
    avg_score = sum(r.grounding_score for r in eval_results) / len(eval_results) if eval_results else 0
    print(f"\n{status}  ({total} cases) | avg_retrieval={avg_ret:.0f}ms | avg_ground_score={avg_score:.2f}\n")

    if write_json:
        report = {
            "summary": {
                "total_cases": total,
                "failures": check_failures,
                "avg_retrieval_latency_ms": round(avg_ret, 1),
                "avg_grounding_score": round(avg_score, 2),
            },
            "cases": [asdict(r) for r in eval_results],
        }
        out_path = _BACKEND / "eval_report.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {out_path}")

    return 0 if check_failures == 0 else 1


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate RAG retrieval, sources, and answer grounding.")
    p.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM; validate retrieval and that retrieved filenames cover must_list_sources.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Print retrieved snippets and answers.")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress loguru noise (errors only).")
    p.add_argument("--json", action="store_true", dest="write_json", help="Write eval_report.json with full result matrix.")
    p.add_argument("--push-langsmith", action="store_true", help="Upload results to LangSmith dataset (requires LANGCHAIN_API_KEY).")
    args = p.parse_args()
    if args.quiet:
        from loguru import logger as _lu
        _lu.remove()
        _lu.add(sys.stderr, level="ERROR", format="{message}")
    code = asyncio.run(run_eval(
        retrieval_only=args.retrieval_only,
        verbose=args.verbose,
        write_json=args.write_json,
        push_langsmith=args.push_langsmith,
    ))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
