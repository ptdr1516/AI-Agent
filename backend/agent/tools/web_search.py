"""
Web Search Tool — Tavily Search API.

Design decisions:
  - AsyncTavilyClient singleton: initialised once at import time, not per call.
  - Async tool: FastAPI's event loop never blocks waiting for a network response.
  - Hard timeout via asyncio.wait_for() independent of Tavily's internal timeout.
  - Tenacity retry for transient connectivity errors (network blips, not quota).
  - Structured output: numbered results with title, URL, snippet — easy for the
    LLM to parse and cite specific sources in its final answer.
"""
import asyncio
from typing import Optional

from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

from core.cache import cache
from core.config import settings
from core.logger import log


# ── Singleton client ───────────────────────────────────────────────────────────
# Import is deferred so the server still boots when tavily-python is not yet
# installed. The first actual search call will raise a clear RuntimeError.
_client: Optional[object] = None


def _get_client():
    """Lazy singleton — imports and instantiates AsyncTavilyClient on first use."""
    global _client
    if _client is None:
        if not settings.TAVILY_API_KEY:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. "
                "Get a free key at https://app.tavily.com and add it to your .env file."
            )
        try:
            from tavily import AsyncTavilyClient
            _client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)
            log.info("Tavily AsyncTavilyClient initialised.")
        except ImportError as exc:
            raise RuntimeError(
                "tavily-python is not installed. Run: pip install tavily-python"
            ) from exc
    return _client


# ── Retry policy ──────────────────────────────────────────────────────────────
# Retry transient network errors only — not auth/quota errors (those should surface immediately).
_RETRY_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)


def _retry_policy():
    return retry(
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        reraise=True,
    )


# ── Input schema ──────────────────────────────────────────────────────────────
class WebSearchInput(BaseModel):
    query: str = Field(
        description=(
            "A clear, specific search query for real-time internet lookup. "
            "Be concise and direct. "
            "Good: 'Python FastAPI async best practices 2024'. "
            "Bad: 'tell me about python'."
        )
    )


# ── Inner fetch (retried) ─────────────────────────────────────────────────────
@_retry_policy()
async def _tavily_search(query: str) -> dict:
    """Inner async fetch with retry. Separated so tenacity applies cleanly."""
    client = _get_client()
    return await asyncio.wait_for(
        client.search(
            query=query,
            max_results=settings.TAVILY_MAX_RESULTS,
            search_depth=settings.TAVILY_SEARCH_DEPTH,
            include_answer=True,       # Tavily synthesises a direct answer
            include_raw_content=False, # Snippets only — keeps token count bounded
        ),
        timeout=settings.TAVILY_TIMEOUT,
    )


# ── Tool ──────────────────────────────────────────────────────────────────────
@tool("web_search_tool", args_schema=WebSearchInput)
async def web_search_tool(query: str) -> str:
    """
    Search the internet for real-time information, current events, news, prices,
    and facts that may have changed after your training cutoff.

    Use this tool when:
    - You need up-to-date information (news, stock prices, recent events)
    - You need to verify a specific fact you are not 100% certain about
    - The user explicitly asks to search the web

    Do NOT use this tool for:
    - Math calculations (use calculator_tool)
    - Internal database queries (use sql_db_tool)
    - Questions answerable from conversation history
    """
    log.info(f"Tavily search: '{query}'")

    # ── Cache-aside check ────────────────────────────────────────────────────
    cache_key = f"search:{query.lower().strip()}"
    cached = await cache.get(cache_key)
    if cached is not None:
        log.info(f"Cache HIT for query: '{query}'")
        return cached

    try:
        response = await _tavily_search(query)
    except asyncio.TimeoutError:
        log.warning(f"Tavily search timed out for: '{query}'")
        return f"Search timed out after {settings.TAVILY_TIMEOUT}s. Try a more specific query."
    except RuntimeError as exc:
        # Surface config errors (missing key / missing package) clearly to the agent
        log.error(f"Tavily configuration error: {exc}")
        return f"Search unavailable: {exc}"
    except Exception as exc:
        log.warning(f"Tavily search failed for '{query}': {exc}")
        return f"Search failed ({type(exc).__name__}). Please try a different query."

    parts: list[str] = []

    # Tavily's own synthesised answer is highest quality — cite it first
    direct_answer = (response.get("answer") or "").strip()
    if direct_answer:
        parts.append(f"**Direct Answer:**\n{direct_answer}")

    # Individual source results
    results = response.get("results") or []
    for i, result in enumerate(results, start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        snippet = (result.get("content") or "").strip()
        if snippet:
            # Truncate to keep downstream token usage bounded
            snippet = snippet[:600] + ("…" if len(snippet) > 600 else "")
            parts.append(f"**[{i}] {title}**\nSource: {url}\n{snippet}")

    if not parts:
        return (
            f"No results found for: '{query}'. "
            "Try rephrasing with more specific terms."
        )

    result = "\n\n---\n\n".join(parts)

    # ── Write back to cache ──────────────────────────────────────────────────
    await cache.set(cache_key, result, ttl=settings.SEARCH_CACHE_TTL)
    log.info(f"Cache MISS — stored result for '{query}' (TTL={settings.SEARCH_CACHE_TTL}s)")

    return result
