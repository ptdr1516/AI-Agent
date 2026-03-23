import asyncio
import traceback
from agent.graph.builder import build_unified_graph, default_chat_llm
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

async def main():
    llm = default_chat_llm()
    graph = build_unified_graph(chat_llm=llm)
    
    state = {
        "messages": [HumanMessage(content="What is 1234 * 56?")],
        "user_id": "test_user",
        "session_id": "test_session",
    }
    
    try:
        async for chunk in graph.astream(state):
            print(chunk)
    except Exception as e:
        with open("error_trace.txt", "w") as f:
            f.write(str(e))
            f.write("\n")
            f.write(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())
