import asyncio
from agent.unified_graph import get_unified_graph, final_assistant_text
from langchain_core.messages import HumanMessage

async def main():
    graph = get_unified_graph()
    queries = [
        ("math", "What is 256 * 14?"),
        ("web", "Search the web for the current stock price of Apple."),
        ("sql", "How many total employees are in the internal SQL database?"),
        ("doc", "Based on my uploaded documents, what is the API url?") # Triggers retrieval
    ]
    
    for test_name, q in queries:
        print(f"\n--- Testing: {test_name} ---")
        try:
            # We use ainvoke here to just get the final state, which includes all intermediate ToolMessages
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=q)], "memory_context": "", "rag_top_k": 3},
                config={"recursion_limit": 10}
            )
            
            # Print which tools were called
            tools_called = set()
            for msg in result["messages"]:
                if getattr(msg, "tool_calls", None):
                    for call in msg.tool_calls:
                        tools_called.add(call["name"])
                        
            print(f"Tools triggered: {list(tools_called)}")
            print(f"Final Answer: {final_assistant_text(result['messages'])[:100]}...")
        except Exception as e:
            print(f"Error testing {test_name}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
