"""Tests for the custom public API tool (JSONPlaceholder)."""
import json
import pytest
from agent.tools.custom_api import custom_api_tool, UserProfileInput

@pytest.mark.asyncio
async def test_custom_api_valid_user():
    """Test fetching a valid user profile (ID 1)."""
    # Tool is async now, so we must await ainvoke
    result = await custom_api_tool.ainvoke({"user_id": 1})
    
    # Needs to be a JSON string containing the required fields
    assert isinstance(result, str)
    data = json.loads(result)
    assert data["id"] == 1
    assert "name" in data
    assert "email" in data

@pytest.mark.asyncio
async def test_custom_api_invalid_user_id():
    """Test schema/function validation catching out-of-bounds user ID."""
    result = await custom_api_tool.ainvoke({"user_id": 99})
    assert "Error" in result or "Invalid" in result

def test_custom_api_schema():
    """Test the Pydantic schema validation."""
    schema = UserProfileInput(user_id=5)
    assert schema.user_id == 5
