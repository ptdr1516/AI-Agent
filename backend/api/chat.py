from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from models.schemas import ChatRequest, ChatResponse
from langchain_core.messages import AIMessage, HumanMessage

from agent.unified_graph import chat_recursion_config, final_assistant_text, get_unified_graph
from agent.memory import memory_manager
from agent.vector_memory import vector_memory_store
from core.auth import verify_token
from core.logger import log
from core.usage_logger import log_usage
from core.graph_tracer import new_trace_id
from core.metrics_collector import RequestMetrics, bind_metrics
from rag.rag_chain import parse_build_context
import json
import time


def _tool_output_str(output) -> str:
    """Normalize LangChain tool results to plain text for SSE and parsers."""
    if output is None:
        return ""
    if hasattr(output, "content"):
        c = output.content
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts: list[str] = []
            for block in c:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
            return "".join(parts)
        return str(c)
    return str(output)

router = APIRouter()


def _stream_chunk_text(chunk) -> list[str]:
    """Extract incremental text from a chat model stream chunk (str, list, or blocks)."""
    if chunk is None:
        return []
    content = getattr(chunk, "content", None)
    if content is None:
        return []
    if isinstance(content, str):
        return [content] if content else []
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    out.append(str(block.get("text", "")))
                elif "text" in block:
                    out.append(str(block["text"]))
        return out
    return [str(content)] if content else []


def _event_is_retrieval_node(event: dict) -> bool:
    """Match LangGraph node events for ``retrieval_node`` (name may be namespaced)."""
    name = (event.get("name") or "").strip()
    if name == "retrieval_node" or name.endswith(":retrieval_node"):
        return True
    return name.split("/")[-1] == "retrieval_node"


def _retrieval_sse_payload(output) -> dict | None:
    """Build JSON-serialisable retrieval summary from ``retrieval_node`` output (no Document objects)."""
    if not isinstance(output, dict):
        return None
    if not any(k in output for k in ("rag_context", "rag_sources", "rag_chunks", "rag_docs")):
        return None
    sources = list(output.get("rag_sources") or [])
    rc = output.get("rag_context") or ""
    chunks_raw = output.get("rag_chunks") or []
    parsed = parse_build_context(rc) if isinstance(rc, str) else {}
    return {
        "retrieval": {
            "status": "complete",
            "sources": sources,
            "rag_detail": parsed,
            "chunk_count": len(chunks_raw),
        }
    }


def _format_vector_memory_context(session_id: str, user_message: str) -> str:
    hits = vector_memory_store.search(session_id, user_message)
    if not hits:
        return "(No similar past summaries.)"
    return "\n".join(f"- {h}" for h in hits)


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, user_id: str = Depends(verify_token)):
    try:
        log.info(f"Sync chat request for session {request.session_id}")
        memory_context = _format_vector_memory_context(request.session_id, request.message)
        memory = memory_manager.get_memory(request.session_id)
        hist = memory.load_memory_variables({}).get("chat_history", [])
        messages = list(hist) + [HumanMessage(content=request.message)]
        graph = get_unified_graph()
        config = chat_recursion_config()
        config.setdefault("configurable", {})["user_id"] = user_id
        result = await graph.ainvoke(
            {
                "messages": messages,
                "memory_context": memory_context,
            },
            config=config,
        )
        out_text = final_assistant_text(result["messages"])
        memory.save_context({"input": request.message}, {"output": out_text})
        vector_memory_store.add_turn(request.session_id, request.message, out_text)
        return ChatResponse(response=out_text)
    except Exception as e:
        log.exception(f"Error handling sync chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, user_id: str = Depends(verify_token)):
    log.info(f"Stream chat request for session {request.session_id}")
    
    async def generate_stream():
        metrics = RequestMetrics.start(
            session_id=request.session_id,
            user_id=user_id,
            query=request.message,
            endpoint="chat_stream",
        )
        bind_metrics(metrics)
        error_msg = ""
        try:
            trace_id = new_trace_id()  # one trace_id for all nodes in this turn
            memory_context = _format_vector_memory_context(request.session_id, request.message)
            memory = memory_manager.get_memory(request.session_id)
            hist = memory.load_memory_variables({}).get("chat_history", [])
            messages = list(hist) + [HumanMessage(content=request.message)]
            graph = get_unified_graph()
            total_prompt_tokens = 0
            total_completion_tokens = 0
            tools_used: list[str] = []
            start_time = time.time()
            assistant_stream_text = ""
            chain_final_output: str | None = None

            config = chat_recursion_config()
            config.setdefault("configurable", {})["user_id"] = user_id

            # Using astream_events to safely stream tokens from the LLM
            async for event in graph.astream_events(
                {
                    "messages": messages,
                    "memory_context": memory_context,
                },
                version="v1",
                config=config,
            ):
                kind = event["event"]
                ev_name = event.get("name") or ""

                if kind == "on_chain_start" and _event_is_retrieval_node(event):
                    log.info(f"[Trace:{trace_id}] Retrieval start")
                    yield f"data: {json.dumps({'retrieval': {'status': 'start'}})}\n\n"

                elif kind == "on_chain_end" and _event_is_retrieval_node(event):
                    out = event.get("data", {}).get("output")
                    payload = _retrieval_sse_payload(out)
                    # Feed retrieval doc count to metrics
                    if isinstance(out, dict):
                        rag_docs = out.get("rag_docs") or []
                        metrics.set_retrieval_docs(len(rag_docs))
                        log.info(f"[Trace:{trace_id}] Retrieval end (docs: {len(rag_docs)})")
                    else:
                        log.info(f"[Trace:{trace_id}] Retrieval end")
                    if payload is not None:
                        yield f"data: {json.dumps(payload)}\n\n"
                    else:
                        yield f"data: {json.dumps({'retrieval': {'status': 'complete', 'sources': [], 'rag_detail': {}, 'chunk_count': 0}})}\n\n"

                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    for text in _stream_chunk_text(chunk):
                        if text:
                            log.info(f"[Trace:{trace_id}] Token chunk: {text!r}")
                        assistant_stream_text += text
                        yield f"data: {json.dumps({'content': text})}\n\n"

                elif kind == "on_chat_model_end":
                    output = event["data"].get("output")
                    if output and hasattr(output, "response_metadata") and output.response_metadata:
                        token_usage = output.response_metadata.get("token_usage", {})
                        if token_usage:
                            pt = token_usage.get("prompt_tokens", 0)
                            ct = token_usage.get("completion_tokens", 0)
                            total_prompt_tokens += pt
                            total_completion_tokens += ct
                            metrics.add_tokens(pt, ct)  # feed metrics
                    
                elif kind == "on_tool_start":
                    tool_name = ev_name
                    if not tool_name:
                        continue
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
                    metrics.add_tool_call(tool_name)  # feed metrics
                    tool_input = event.get("data", {}).get("input", {})
                    log.info(f"[Trace:{trace_id}] Tool start: {tool_name} with input {tool_input}")
                    yield f"data: {json.dumps({'tool': tool_name, 'status': 'start', 'input': str(tool_input)})}\n\n"

                elif kind == "on_tool_end":
                    tool_name = ev_name
                    if not tool_name:
                        continue
                    log.info(f"[Trace:{trace_id}] Tool end: {tool_name}")
                    raw_out = _tool_output_str(event.get("data", {}).get("output"))
                    payload: dict = {
                        "tool": tool_name,
                        "status": "end",
                        "output": raw_out,
                    }
                    if tool_name == "document_search":
                        parsed = parse_build_context(raw_out)
                        if parsed.get("chunks"):
                            payload["rag_detail"] = parsed
                    yield f"data: {json.dumps(payload)}\n\n"

                elif kind == "on_chain_end" and event.get("name") in (
                    "AgentExecutor",
                    "chat_agent",
                    "LangGraph",
                    "output_node",
                    "llm_node",
                ):
                    out = event["data"].get("output")
                    if isinstance(out, dict):
                        ft = out.get("output")
                        if isinstance(ft, str) and ft.strip():
                            chain_final_output = ft
                        msgs = out.get("messages")
                        if isinstance(msgs, list) and msgs:
                            last = msgs[-1]
                            if isinstance(last, AIMessage):
                                t = final_assistant_text([last])
                                if t.strip():
                                    chain_final_output = t

            reply_for_memory = (
                chain_final_output if chain_final_output else assistant_stream_text
            )
            if reply_for_memory.strip():
                memory.save_context({"input": request.message}, {"output": reply_for_memory})
                vector_memory_store.add_turn(
                    request.session_id, request.message, reply_for_memory
                )

            latency_ms = (time.time() - start_time) * 1000
            # Count how many RAG chunks were retrieved this turn (from rag_docs state)
            rag_chunk_count = 0
            for event_result in []:
                pass  # chunks counted below from tools_used heuristic
            log_usage(
                session_id=request.session_id,
                user_message=request.message,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                tools_used=tools_used,
                latency_ms=latency_ms,
                user_id=user_id,
                rag_chunks_retrieved=tools_used.count("retrieval_node") + tools_used.count("document_search"),
            )

            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_msg = str(e)
            log.exception(f"Streaming error: {e}")
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            metrics.finish_and_log(error=error_msg)

    return StreamingResponse(generate_stream(), media_type="text/event-stream")
