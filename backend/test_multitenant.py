import asyncio
import httpx
import tempfile
import os

BASE_URL = "http://127.0.0.1:8000/api"

async def test_multitenant_isolation():
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("User A creating a secret file...")
        
        # 1. Create a temporary document for User A
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as f:
            f.write("User A's secret launch code is 9988776655. Do not share with User B.")
            user_a_file = f.name
            
        try:
            # 2. Upload document as User A
            print("Uploading document as User A...")
            with open(user_a_file, "rb") as f:
                res = await client.post(
                    f"{BASE_URL}/upload",
                    headers={"Authorization": "Bearer User_A_Token"},
                    files={"file": f}
                )
            print("Upload response:", res.json())
            
            # Wait a moment for indexing to settle
            await asyncio.sleep(1.0)
            
            print("\n--------------------------")
            
            # 3. Query as User A - should see the code
            print("Querying as User A: 'What is my secret launch code?'")
            res_a = await client.post(
                f"{BASE_URL}/rag/query",
                headers={"Authorization": "Bearer User_A_Token"},
                json={"query": "What is my secret launch code?", "top_k": 3}
            )
            data_a = res_a.json()
            print("User A response:", data_a)
            
            print("\n--------------------------")
            
            # 4. Query as User B - should NOT see the code
            print("Querying as User B: 'What is the secret launch code?'")
            res_b = await client.post(
                f"{BASE_URL}/rag/query",
                headers={"Authorization": "Bearer User_B_Token"},
                json={"query": "What is the secret launch code?", "top_k": 3}
            )
            data_b = res_b.json()
            print("User B response:", data_b)
            
            import json
            with open("test_results_clean.json", "w", encoding="utf-8") as f:
                json.dump({"UserA": data_a, "UserB": data_b}, f, indent=2)
            
        finally:
            os.remove(user_a_file)

if __name__ == "__main__":
    asyncio.run(test_multitenant_isolation())
