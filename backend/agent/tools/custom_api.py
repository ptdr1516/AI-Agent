import json
import httpx
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt
from core.logger import log

class UserProfileInput(BaseModel):
    user_id: int = Field(description="The numeric ID of the user profile to fetch (1-10 valid).")

# Global singleton client to reuse connections
_http_client: httpx.AsyncClient | None = None

def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=5.0)
    return _http_client

@retry(wait=wait_exponential(multiplier=1, min=1, max=5), stop=stop_after_attempt(3))
async def _fetch_user_profile(user_id: int) -> dict:
    """Real HTTP call to public REST API with retries and timeout."""
    client = _get_client()
    url = f"https://jsonplaceholder.typicode.com/users/{user_id}"
    log.info(f"Fetching user profile via GET {url}")
    
    response = await client.get(url)
    if response.status_code == 404:
        return {"error": f"User ID {user_id} not found."}
        
    response.raise_for_status()
    # Return minimal subset of data to save LLM context
    data = response.json()
    return {
        "id": data.get("id"),
        "name": data.get("name"),
        "email": data.get("email"),
        "company": data.get("company", {}).get("name"),
        "website": data.get("website")
    }

@tool("custom_api_tool", args_schema=UserProfileInput)
async def custom_api_tool(user_id: int) -> str:
    """Useful for fetching external user profiles by their integer ID from a public REST API. Returns JSON string."""
    try:
        if not (1 <= user_id <= 10):
            return f"Error: Invalid user_id. JSONPlaceholder only has users 1-10."
            
        profile_data = await _fetch_user_profile(user_id)
        if "error" in profile_data:
            return profile_data["error"]
            
        return json.dumps(profile_data)
        
    except httpx.TimeoutException:
        log.error(f"Timeout fetching user {user_id}")
        return "Error fetching API: The request timed out. The server might be down."
    except httpx.HTTPError as e:
        log.error(f"HTTP error fetching user {user_id}: {e}")
        return f"Error fetching API: HTTP connection failed ({str(e)})"
    except Exception as e:
        log.error(f"Unexpected error fetching user {user_id}: {e}")
        return f"Error fetching API: {str(e)}"
