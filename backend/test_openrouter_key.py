"""Test script to verify OpenRouter API key is working."""
import os
from dotenv import load_dotenv
import httpx

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    print("❌ ERROR: OPENROUTER_API_KEY not found in .env file")
    exit(1)

print(f"✓ Found API key: {API_KEY[:20]}...{API_KEY[-10:]}")
print("\nTesting OpenRouter API connection...\n")

# Test endpoint: https://openrouter.ai/api/v1/auth/key
url = "https://openrouter.ai/api/v1/auth/key"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

try:
    response = httpx.get(url, headers=headers, timeout=10.0)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}\n")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ SUCCESS! API key is valid!")
        print(f"Key Label: {data.get('data', {}).get('label', 'Unknown')}")
        print(f"Usage Limit: ${data.get('data', {}).get('usage_limit', 'No limit')}/month")
        print(f"Total Usage: ${data.get('data', {}).get('total_usage', 0):.4f}")
    elif response.status_code == 401:
        print("❌ FAILED! API key is invalid or expired.")
        print("\nNext steps:")
        print("1. Go to https://openrouter.ai/keys")
        print("2. Create a new API key")
        print("3. Replace the OPENROUTER_API_KEY in your .env file")
    else:
        print(f"⚠️ Unexpected response: {response.status_code}")
        
except httpx.TimeoutException:
    print("❌ Request timed out. Check your internet connection.")
except Exception as e:
    print(f"❌ Error testing API key: {e}")

print("\n" + "="*60)
