import asyncio
from agent.tools.calculator import calculator_tool

async def main():
    try:
        res = await calculator_tool.ainvoke({"expression": "2 * 3"})
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
