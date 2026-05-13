import os
import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import httpx
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi_mcp import FastApiMCP

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from mcp_server.auth import StaticTokenVerifier, load_api_keys_from_env
from mcp_server.llm import LLMManager, LLMAllProvidersFailedError

load_dotenv()


# --- Search Provider Enum ====================================================

class SearchProvider(str, Enum):
    """Valid search providers for web_search tool."""
    MIKLIUM = "miklium"
    TAVILY = "tavily"
    BRAVE = "brave"
    GOOGLE = "google"


# Load authentication and host configuration
api_keys = load_api_keys_from_env()
server_host = os.getenv("MCP_HOST", "127.0.0.1")

# --- FastAPI App & Auth =======================================================

app = FastAPI(
    title="WebTool MCP Server",
    version="0.1.0",
)

# DNS rebinding protection for non-localhost hosts
if server_host not in ("127.0.0.1", "localhost", "::1"):
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[server_host],
    )

# Bearer token auth dependency
_bearer = HTTPBearer(auto_error=False)


async def _require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> None:
    if not api_keys:
        return  # Auth disabled
    verifier = StaticTokenVerifier(api_keys)
    token = credentials.credentials if credentials else ""
    result = await verifier.verify_token(token)
    if result is None:
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")

if api_keys:
    print(f"Server initialized with Bearer token auth ({len(api_keys)} key(s))")
else:
    print("Server initialized (no API keys — auth disabled)")

# LLM Manager with multi-provider failover support
llm_manager = LLMManager()

# Default headers for HTTP requests (User-Agent required by many sites)
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def _get_configured_providers() -> list[str]:
    """
    Return a list of configured search providers in priority order.
    Only includes providers that have all required environment variables set.
    MIKLIUM is always available (no API key required).
    """
    providers = []

    # MIKLIUM (priority 1) - no API key required, free API
    providers.append("miklium")

    # Check Tavily (priority 2)
    if os.getenv("TAVILY_API_KEY"):
        providers.append("tavily")

    # Check Brave (priority 3)
    if os.getenv("BRAVE_API_KEY"):
        providers.append("brave")

    # Check Google (priority 4) - requires both API key and search engine ID
    if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
        providers.append("google")

    return providers


# --- MCP Tool Business Logic ==================================================
# These functions are called directly by tests and by the route wrappers below.


async def web_search(
    query: str,
    provider: Optional[str] = None,
    num_results: int = 10,
    days: int = 0,
    offset: int = 0
) -> dict:
    """
    Execute a web search using configured providers with automatic failover.

    When a provider fails (not configured or API error), automatically tries the next
    available configured provider. Priority order: miklium > tavily > brave > google.

    Args:
        query: The search query string (required)
        provider: Preferred provider to use. Only configured providers are valid.
                 If not specified, uses the first available provider with failover support.
        num_results: Number of results to return (default: 10, max: 20)
        days: Filter results to the last N days. 0 means no date filtering.
              Only supported for brave/tavily; ignored for google and miklium.
        offset: Starting index for pagination. Only supported for brave/google;
               tavily and miklium do not support offsets.

    Returns:
        Dict with query, provider, results, and optionally failover_attempts.
        Errors are included inline: {"query": "...", "error": "...", ...}
    """
    if not query:
        return {"query": "", "provider": provider or "miklium", "error": "Missing required field: query"}

    configured_providers = _get_configured_providers()
    num_results = min(num_results, 20)

    # Determine provider order: preferred first, then failover chain
    if provider and provider in configured_providers:
        provider_order = [provider] + [p for p in configured_providers if p != provider]
    elif configured_providers:
        provider_order = configured_providers[:]
    else:
        return {
            "query": query,
            "provider": provider or "miklium",
            "error": "No search providers configured. Set TAVILY_API_KEY, BRAVE_API_KEY, or GOOGLE_API_KEY+GOOGLE_SEARCH_ENGINE_ID in .env"
        }

    failover_attempts = []
    final_result = None
    final_provider = None

    for p in provider_order:
        if p == "miklium":
            result = await _search_miklium(query, num_results)
        elif p == "tavily":
            result = await _search_tavily(query, num_results, days=days)
        elif p == "brave":
            result = await _search_brave(query, num_results, days=days, offset=offset)
        elif p == "google":
            result = await _search_google(query, num_results, offset=offset)
        else:
            continue

        if "error" in result:
            failover_attempts.append({"provider": p, "error": result["error"]})
        else:
            final_result = result
            final_provider = p
            break

    if final_result is not None:
        normalized = {
            "query": query,
            "provider": final_provider,
            "results": [
                {"title": r.get("title", ""), "url": r["url"], "snippet": r.get("description", "")}
                for r in final_result.get("results", [])
            ]
        }
        if failover_attempts:
            normalized["failover_attempts"] = failover_attempts
        return normalized
    else:
        error_response = {
            "query": query,
            "provider": provider_order[0] if provider_order else provider or "miklium",
            "error": f"All search providers failed. Last error: {failover_attempts[-1]['error'] if failover_attempts else 'Unknown'}"
        }
        if failover_attempts:
            error_response["failover_attempts"] = failover_attempts
        return error_response


async def _search_tavily(query: str, num_results: int, days: int = 0) -> dict:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"error": "TAVILY_API_KEY not configured in .env"}

    payload = {"query": query, "search_depth": "basic", "max_results": num_results}

    # Compute start_date from days parameter
    if days > 0:
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        payload["start_date"] = start_date

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            )
            resp.raise_for_status()
            data = resp.json()

        results = [
            {"title": r["title"], "url": r["url"], "description": r.get("content", "")[:200]}
            for r in data.get("results", [])[:num_results]
        ]
        return {"count": len(results), "results": results}
    except Exception as e:
        return {"error": f"Tavily search failed: {str(e)}"}


def _brave_freshness(days: int = 0) -> str:
    """Convert days to Brave freshness period parameter."""
    if days <= 0:
        return ""
    elif days <= 1:
        return "pd"  # 24 hours
    elif days <= 7:
        return "pw"  # 7 days
    elif days <= 31:
        return "pm"  # 31 days
    elif days <= 365:
        return "py"  # 365 days
    else:
        return ""  # No freshness filter for > 365 days


async def _search_brave(query: str, num_results: int, days: int = 0, offset: int = 0) -> dict:
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return {"error": "BRAVE_API_KEY not configured in .env"}

    params = {"q": query, "count": min(num_results, 20)}
    if offset > 0:
        params["offset"] = offset

    freshness = _brave_freshness(days)
    if freshness:
        params["freshness"] = freshness

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params=params,
                headers={"X-Subscription-Token": api_key, "Accept": "application/json"}
            )
            resp.raise_for_status()
            data = resp.json()

        web_results = data.get("web", {}).get("results", [])
        results = [
            {"title": r["title"], "url": r["url"], "description": r.get("description", "")[:200]}
            for r in web_results[:num_results]
        ]
        return {"count": len(results), "results": results}
    except Exception as e:
        return {"error": f"Brave search failed: {str(e)}"}


async def _search_google(query: str, num_results: int, offset: int = 0) -> dict:
    api_key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    if not api_key or not cx:
        return {"error": "GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID must be configured in .env"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            params = {"key": api_key, "cx": cx, "q": query, "num": min(num_results, 10)}
            if offset > 0:
                params["start"] = offset + 1  # Google uses 1-based index

            resp = await client.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params
            )
            resp.raise_for_status()
            data = resp.json()

        results = [
            {"title": r["title"], "url": r["link"], "description": r.get("snippet", "")[:200]}
            for r in data.get("items", [])[:num_results]
        ]
        return {"count": len(results), "results": results}
    except Exception as e:
        return {"error": f"Google search failed: {str(e)}"}


async def _search_miklium(query: str, num_results: int) -> dict:
    """
    Search using the MIKLIUM API (free, no API key required).

    Args:
        query: Search query string
        num_results: Number of results to return

    Returns:
        Dict with "count" and "results" on success,
        or {"error": "..."} on failure.
    """
    max_small = min(num_results, 5)
    max_large = min(max(1, num_results // 3), 2)

    payload = {
        "search": [query],
        "maxSmallSnippets": max_small,
        "maxLargeSnippets": max_large
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://miklium.vercel.app/api/search",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            data = resp.json()

        if not data.get("success", False):
            return {"error": f"MIKLIUM search failed: {data.get('error', 'Unknown error')}"}

        results = []
        for r in data.get("results", [])[:num_results]:
            snippet = r.get("snippet", "")
            results.append({
                "title": snippet[:100] if snippet else "",
                "url": r.get("url", ""),
                "description": snippet[:200] if snippet else ""
            })

        return {"count": len(results), "results": results}
    except Exception as e:
        return {"error": f"MIKLIUM search failed: {str(e)}"}


async def _call_llm(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Call the configured LLM endpoint(s) with failover support.
    Returns the assistant's response content.
    Raises RuntimeError if all providers fail.
    """
    try:
        return await llm_manager.complete(prompt, system_prompt)
    except LLMAllProvidersFailedError as e:
        raise RuntimeError(str(e))


DEFAULT_SUMMARY_PROMPT = """
You are a technical summarizer. Analyze the provided web content and produce a concise, well-structured markdown summary.
Focus on:
- Key findings and main points
- Technical details relevant to the query context
- Important data, statistics, or code examples
Format output as clean Markdown with headers where appropriate.
"""


async def web_summarize(
    url: str,
    summary_prompt: str = "",
    max_words_per_url: int = 800
) -> dict:
    """
    Fetch a URL and summarize content via configured LLM.

    Args:
        url: The URL to fetch and summarize (required)
        summary_prompt: Custom prompt for the summarization step (optional, uses built-in default)
        max_words_per_url: Max words before truncation (default 800)

    Returns:
        Dict with 'url' and 'summary', or 'error' on fetch/LLM failure.
    """
    # Fetch content from the URL
    fetch_result = await web_fetch(
        url,
        num_words=max_words_per_url,
        regex=None  # No filtering; full content for summarization
    )

    system_prompt = summary_prompt if summary_prompt else DEFAULT_SUMMARY_PROMPT

    # Check for fetch errors
    content = fetch_result.get("content", "")
    if "error" in fetch_result or not content:
        error_text = fetch_result.get("error", "") if "error" in fetch_result else content
        return {"url": url, "error": error_text[:100] if len(error_text) > 100 else error_text}

    # Check for "No matches" content (from regex filtering, though we don't pass regex here)
    if "No matches" in content:
        return {"url": url, "error": content[:100]}

    try:
        user_prompt = f"Summarize the following web content:\n\n{content}"
        summary_text = await _call_llm(user_prompt, system_prompt)
        return {"url": url, "summary": summary_text.strip()}
    except RuntimeError as e:
        return {"url": url, "error": str(e)}


async def web_fetch(
    url: str,
    include_links: bool = False,
    start_word: int = 0,
    num_words: int = 1000,
    regex: str = None,
    regex_padding: int = 50
) -> dict:
    """
    Fetch a URL, convert to markdown, and optionally filter via regex.

    Args:
        url: The URL to fetch (required)
        include_links: When True, preserve anchor tag hrefs; when False (default), unwrap anchors keeping only text
        start_word: Starting word index for pagination
        num_words: Maximum words to return (default 1000)
        regex: Regex pattern to filter content
        regex_padding: Characters of context around regex matches (default 50)

    Returns:
        Dict with 'url' and 'content' keys, or 'error' on failure.
    """
    async with httpx.AsyncClient(follow_redirects=True, headers=DEFAULT_HEADERS) as client:
        try:
            resp = await client.get(url, timeout=10.0)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            # Handle include_links option - strip href attributes when False
            if not include_links:
                for a_tag in soup.find_all('a'):
                    a_tag.unwrap()  # Remove anchor tags but keep text content

            # Basic conversion
            content = md(str(soup))

            # Apply Regex filtering/padding if provided
            if regex:
                pattern = re.compile(regex)
                matches = list(pattern.finditer(content))
                if matches:
                    filtered = []
                    for m in matches:
                        start = max(0, m.start() - regex_padding)
                        end = min(len(content), m.end() + regex_padding)
                        filtered.append(content[start:end])
                    content = "\n---\n".join(filtered)
                else:
                    content = "No matches found for regex."

            # Word truncation
            words = content.split()
            truncated = " ".join(words[start_word : start_word + num_words])
            return {"url": url, "content": truncated}
        except Exception as e:
            return {"url": url, "error": f"Error fetching {url}: {str(e)}"}


# --- FastAPI Route Wrappers (MCP Tools via fastapi-mcp) =======================
# These thin wrappers expose the tool functions as FastAPI routes.
# fastapi-mcp auto-generates MCP tools from these routes at mount time.


@app.post(
    "/web_search",
    operation_id="web_search",
    tags=["mcp-tool"],
    summary="Perform web search, general or with optional specified provider",
    dependencies=[Depends(_require_auth)],
)
async def api_web_search(
    query: str,
    provider: Optional[SearchProvider] = None,
    num_results: int = 10,
    days: int = 0,
    offset: int = 0,
) -> dict:
    return await web_search(
        query=query,
        provider=provider.value if provider else None,
        num_results=num_results,
        days=days,
        offset=offset,
    )


@app.post(
    "/web_fetch",
    operation_id="web_fetch",
    tags=["mcp-tool"],
    summary="Fetch, convert to markdown, and/or filter via regex",
    dependencies=[Depends(_require_auth)],
)
async def api_web_fetch(
    url: str,
    include_links: bool = False,
    start_word: int = 0,
    num_words: int = 1000,
    regex: Optional[str] = None,
    regex_padding: int = 50,
) -> dict:
    return await web_fetch(
        url=url,
        include_links=include_links,
        start_word=start_word,
        num_words=num_words,
        regex=regex,
        regex_padding=regex_padding,
    )


@app.post(
    "/web_summarize",
    operation_id="web_summarize",
    tags=["mcp-tool"],
    summary="Fetch AI-summarized URL content via configured LLM",
    dependencies=[Depends(_require_auth)],
)
async def api_web_summarize(
    url: str,
    summary_prompt: str = "",
    max_words_per_url: int = 800,
) -> dict:
    return await web_summarize(
        url=url,
        summary_prompt=summary_prompt,
        max_words_per_url=max_words_per_url,
    )


@app.get("/")
async def health() -> dict:
    """Health check endpoint for service discovery."""
    return {"status": "ok", "name": "WebTool MCP Server"}


# --- Mount MCP StreamableHTTP at /mcp ========================================

fastapi_mcp = FastApiMCP(app, name="WebTool", include_tags=["mcp-tool"])
fastapi_mcp.mount_http(mount_path="/mcp")


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import uvicorn

    print("Starting MCP server...", flush=True)

    parser = argparse.ArgumentParser(description="WebTool MCP Server")
    parser.add_argument("--host", default=server_host, help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)