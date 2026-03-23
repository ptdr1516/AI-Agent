"""Test script to verify OpenRouter API key works with actual chat completion."""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://openrouter.ai/api/v1")

if not API_KEY:
    print("❌ ERROR: OPENROUTER_API_KEY not found in .env file")
    exit(1)

print(f"✓ Testing OpenRouter Chat Completion")
print(f"  Model: {LLM_MODEL}")
print(f"  API Base: {LLM_API_BASE}")
print(f"  API Key: {API_KEY[:20]}...{API_KEY[-10:]}")
print("\nInitializing ChatOpenAI...\n")

try:
    # Initialize the LLM exactly as your code does
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.0,
        openai_api_key=API_KEY,
        openai_api_base=LLM_API_BASE,
        streaming=False,  # Use non-streaming for testing
        default_headers={
            "HTTP-Referer": "http://localhost:5173",
            "X-Title": "Nova Agent Test",
        },
    )
    
    print("✓ LLM initialized successfully")
    print("\nSending test message...\n")
    
    # Send a simple test message
    response = llm.invoke([HumanMessage(content="Say 'Hello, this is a test!' and nothing else.")])
    
    print("✅ SUCCESS! Chat completion worked!")
    print(f"\nResponse: {response.content}")
    print(f"\nModel used: {response.response_metadata.get('model', 'Unknown')}")
    
except Exception as e:
    print(f"❌ FAILED! Error: {e}")
    print("\nTroubleshooting steps:")
    print("1. Check if the model name is correct on OpenRouter")
    print("2. Visit https://openrouter.ai/models to see available models")
    print("3. Try a different model (e.g., 'openai/gpt-4', 'anthropic/claude-3-haiku')")
    print("4. Make sure you have credits/quota remaining")

print("\n" + "="*60)
