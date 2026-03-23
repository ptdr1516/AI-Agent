import pytest
from unittest.mock import AsyncMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnableLambda

from agent.graph.builder import build_unified_graph
from agent.unified_graph import chat_recursion_config, final_assistant_text
from agent.graph.nodes.retrieval_node import retrieval_node


class _FakeChatLLM:
    def bind_tools(self, tools):
        async def _run(messages):
            tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
            if tool_msgs:
                # The last ToolMessage content should be the calculator result.
                return AIMessage(content=str(tool_msgs[-1].content))

            # First pass: request a calculator tool call.
            tool_call = ToolCall(
                name="calculator_tool",
                args={"expression": "2+2"},
                id="call_1",
            )
            return AIMessage(content="", tool_calls=[tool_call])

        # LangChain pipes the output of the preprocess runnable into this bind_tools runnable.
        return RunnableLambda(_run)


@pytest.mark.asyncio
async def test_unified_graph_runs_tool_calls_and_returns_final_answer():
    fake_llm = _FakeChatLLM()

    with patch(
        "agent.graph.nodes.retrieval_node.retrieve_documents",
        new_callable=AsyncMock,
        return_value=[],
    ):
        graph = build_unified_graph(chat_llm=fake_llm)
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="What is 2+2?")],
                "memory_context": "",
                "rag_top_k": None,
                # LangGraph will normally inject this; for direct ainvoke it's fine
                # to set an explicit default.
                "is_last_step": False,
            },
            config={**chat_recursion_config(), "configurable": {"user_id": "u1"}},
        )

    assert final_assistant_text(result["messages"]).strip() == "4"


@pytest.mark.asyncio
async def test_retrieval_node_populates_rag_fields():
    docs = [Document(page_content="hello", metadata={"filename": "doc1.txt"})]

    with patch(
        "agent.graph.nodes.retrieval_node.retrieve_documents",
        new_callable=AsyncMock,
        return_value=docs,
    ):
        out = await retrieval_node(
            {
                "messages": [HumanMessage(content="find doc1")],
                "is_last_step": False,  # not used by retrieval_node
                "memory_context": "",
                "rag_top_k": None,
            },
            config={"configurable": {"user_id": "u1"}},
        )

    assert out["rag_sources"] == ["doc1.txt"]
    assert out["rag_chunks"] == [{"filename": "doc1.txt", "preview": "hello"}]
    assert "[1] Source: doc1.txt" in out["rag_context"]

