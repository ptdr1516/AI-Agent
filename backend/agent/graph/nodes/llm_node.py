"""llm_node — single tool-bound chat model (retrieval context injected in system message)."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI

from ..prompts import SYSTEM_PROMPT
from ..state import UnifiedGraphState
from ..tool_calling import get_tools_for_graph


def _chat_preprocess(state: UnifiedGraphState) -> list[BaseMessage]:
    mc = (state.get("memory_context") or "").strip() or ""
    system = SYSTEM_PROMPT.format(memory_context=mc)
    rc = (state.get("rag_context") or "").strip()
    if rc and not rc.startswith("(No relevant passages"):
        system += (
            "\n\n## Index search results (latest user message)\n"
            "Pre-retrieved passages from the indexed knowledge base. **Use this context when it is relevant.** "
            "Cite **Source:** labels or filenames from the blocks below when you rely on them. "
            "You may still call `document_search` with a refined query if you need different passages.\n\n"
            f"{rc}"
        )
    return [SystemMessage(content=system), *list(state["messages"])]


def build_chat_llm_runnable(chat_llm: ChatOpenAI):
    tools = get_tools_for_graph()
    preprocessor = RunnableLambda(_chat_preprocess)
    return preprocessor | chat_llm.bind_tools(tools)


async def _chat_llm_generate(
    state: UnifiedGraphState,
    config: RunnableConfig,
    chat_runnable,
) -> dict[str, Any]:
    response = await chat_runnable.ainvoke(state, config)
    if not isinstance(response, AIMessage):
        return {"messages": [response]}

    if state.get("is_last_step") and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }

    return {"messages": [response]}


def create_llm_node(
    chat_llm: ChatOpenAI,
) -> Callable[[UnifiedGraphState, RunnableConfig], Awaitable[dict[str, Any]]]:
    """Returns the async ``llm_node`` closure (LCEL preprocess + bind_tools)."""
    chat_runnable = build_chat_llm_runnable(chat_llm)

    async def llm_node(
        state: UnifiedGraphState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        return await _chat_llm_generate(state, config, chat_runnable)

    return llm_node
