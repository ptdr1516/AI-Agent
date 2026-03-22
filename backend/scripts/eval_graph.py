"""
scripts/eval_graph.py — E2E Evaluation against the REAL graph and REAL vector store.

Unlike eval_rag.py (which uses an in-memory FAISS fixture for pure retrieval testing),
this script invokes the fully compiled unified graph against whatever real VectorStore
is currently configured in the environment (e.g., Chroma or on-disk FAISS), and tests
the actual LLM generation output.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage

from agent.unified_graph import chat_recursion_config, final_assistant_text, get_unified_graph
from core.metrics_collector import RequestMetrics, bind_metrics
from core.graph_tracer import new_trace_id
from rag.vectorstore import get_app_vector_store


@dataclass
class EvalCase:
    question: str
    expected_source: str | None = None


def load_eval_cases() -> list[EvalCase]:
    eval_file = Path(__file__).parent.parent / "data" / "eval.json"
    if not eval_file.exists():
        print(f"Dataset not found at {eval_file}")
        return []
    
    with eval_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    cases = []
    for item in data:
        cases.append(EvalCase(
            question=item.get("question", ""),
            expected_source=item.get("expected_source")
        ))
    return cases


async def run_eval() -> int:
    cases = load_eval_cases()
    if not cases:
        print("No eval cases found. Exiting.")
        return 0

    print("Initializing real vector store and compiling unified graph...\n")
    store = await get_app_vector_store()
    if not store.has_vectorstore:
        print("⚠ WARNING: Real vector store is empty. RAG queries will likely return no sources.")
    
    graph = get_unified_graph()
    
    print(f"{'Question prefix':25}  {'Latency':>8}  {'Ans':>4}  {'Srcs':>4}  {'Status':^8}")
    print("-" * 57)
    
    failures = 0
    
    for i, case in enumerate(cases):
        t0 = time.perf_counter()
        
        # Setup metrics/traces identical to production endpoints
        trace_id = new_trace_id()
        metrics = RequestMetrics.start(
            session_id=f"eval_graph_{i}",
            user_id="eval_user",
            query=case.question,
            endpoint="eval_graph"
        )
        bind_metrics(metrics)
        
        config = chat_recursion_config()
        config.setdefault("configurable", {})["user_id"] = "eval_user"

        try:
            result = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=case.question)],
                    "memory_context": "",
                    "rag_top_k": 3,
                },
                config=config,
            )
            
            # Extract outputs
            answer = final_assistant_text(result["messages"])
            sources = result.get("rag_sources") or []
            
            # Feed metrics (optional but keeps logs clean)
            rag_docs = result.get("rag_docs") or []
            metrics.set_retrieval_docs(len(rag_docs))
            metrics.finish_and_log()
            
            # Grading
            has_ans = bool(answer and answer.strip())
            
            passed = True
            err_msg = ""
            
            # Check answer returned at all
            if not has_ans:
                passed = False
                err_msg = "No answer returned"
                
            # Check for expected source
            has_srcs = False
            if case.expected_source:
                source_filenames = [
                    s.metadata.get("source") or s.metadata.get("filename") or ""
                    for s in sources
                ]
                has_srcs = True
                if not any(case.expected_source in fname for fname in source_filenames):
                    passed = False
                    err_msg = f"Expected source {case.expected_source} not found in {source_filenames}"
                    has_srcs = False
                
            latency_ms = int((time.perf_counter() - t0) * 1000)
            
            ans_mark = "✅" if has_ans else "❌"
            src_mark = "✅" if has_srcs else ("-" if not case.expected_source else "❌")
            status_mark = "PASS" if passed else "FAIL"
            
            short_q = (case.question[:22] + "...") if len(case.question) > 25 else case.question
            print(f"{short_q:25}  {latency_ms:6}ms  {ans_mark:>3}  {src_mark:>3}  {status_mark:^8}")
            if not passed:
                print(f"   -> ERROR: {err_msg}")
                failures += 1
                
        except Exception as e:
            metrics.finish_and_log(error=str(e))
            latency_ms = int((time.perf_counter() - t0) * 1000)
            short_q = (case.question[:22] + "...") if len(case.question) > 25 else case.question
            print(f"{short_q:25}  {latency_ms:6}ms  {'❌':>3}  {'❌':>3}   FAIL")
            print(f"   -> EXCEPTION: {e}")
            failures += 1
            
    print("\n" + "=" * 57)
    if failures == 0:
        print(f"✅ All {len(cases)} cases passed.")
        return 0
    else:
        print(f"❌ {failures}/{len(cases)} cases failed.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_eval()))
