import os
import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi_mcp import FastApiMCP

from pathlib import Path

from dotenv import load_dotenv

from mcp_server.auth import StaticTokenVerifier, load_api_keys_from_env
from mcp_server.llm import LLMManager, LLMAllProvidersFailedError
from mcp_server.llm.parser import DOCLING_SUPPORTED_EXTENSIONS
from mcp_server.extraction import ContentExtractionPipeline
from mcp_server.agentic import AgenticFetchAgent

# Load .env from project root (one level up from src/)
_ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)


# --- Search Provider Enum ====================================================

class SearchProvider(str, Enum):
    """Valid search providers for searchWeb tool."""
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
if server_host not in ("127.0.0.1", "localhost", "::1"):  # pragma: no cover
    app.add_middleware(  # pragma: no cover
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
    print(f"Server initialized with Bearer token auth ({len(api_keys)} key(s))")  # pragma: no cover
else:
    print("Server initialized (no API keys — auth disabled)")

# LLM Manager with multi-provider failover support
llm_manager = LLMManager()

# Content extraction pipeline (Playwright → Trafilatura → Docling → BS4)
_extraction_pipeline = ContentExtractionPipeline()

# Default headers for HTTP requests (User-Agent required by many sites)
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# File extensions treated as binary documents – routed directly to Docling.
# HTML and plain-text variants are handled by the full HTML pipeline instead.
_BINARY_DOC_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp",
    ".csv", ".json", ".xml",
}


def _get_url_extension(url: str) -> str:
    """Return the lower-case file extension from a URL path, or empty string."""
    path = urlparse(url).path
    filename = path.rsplit("/", 1)[-1]
    if "." in filename:
        return "." + filename.rsplit(".", 1)[1].lower().split("?")[0]
    return ""


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


async def search_web(
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


async def summarize_text(
    text: str,
    summary_prompt: str = "",
    max_words: int = 800
) -> dict:
    """
    Summarize raw text content via configured LLM.

    Args:
        text: The raw text to summarize (required)
        summary_prompt: Custom prompt for the summarization step (optional, uses built-in default)
        max_words: Maximum words in output summary (default 800)

    Returns:
        Dict with 'summary', or 'error' on LLM failure.
    """
    system_prompt = DEFAULT_SUMMARY_PROMPT + (f"\n\n**More importantly: {summary_prompt}**" if summary_prompt else "") + f"\n**You produce summaries using no more than {max_words} words.**"
    
    # Add current date/time in YYYY-MM-DD HH:MM:SS format to system prompt for better context in summarization
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    system_prompt += f"\n\n The current date and time is: {current_time} UTC."

    try:
        user_prompt = f"Summarize the following content in no more than {max_words} words:\n\n{text}"
        summary_text = await _call_llm(user_prompt, system_prompt)
        return {"summary": summary_text.strip()}
    except RuntimeError as e:
        return {"error": str(e)}


async def fetch_web_content(
    url: str,
    include_links: bool = False,
    start_word: int = 0,
    num_words: int = 1000,
    regex: str = None,
    regex_padding: int = 50,
    use_llm_refinement: Optional[bool] = None,
    summarize: bool = False,
    summary_prompt: str = "",
) -> dict:
    """
    Fetch a URL, extract its main content as Markdown, and optionally filter via regex.

    Uses a multi-tiered extraction pipeline that cascades through increasingly
    powerful (and slower) extraction strategies until sufficient content is
    obtained or all tiers are exhausted:

      1. Dynamic Rendering (Playwright) – executes JavaScript for SPAs.
      2. Heuristic/Text-Density (Trafilatura → Readability-lxml) – removes boilerplate.
      3. Layout-Aware (Docling) – handles tables and complex document structures.
      4. Fallback (BeautifulSoup) – always-succeeds minimal HTML converter.
      5. Cognitive Refinement (LLM, optional) – semantic cleanup pass.

    Binary documents (PDF, DOCX, PPTX, XLSX, images) are routed directly to
    Docling (Tier 3) and bypass the HTML pipeline.

    Args:
        url: The URL to fetch (required)
        include_links: When True, preserve anchor tag hrefs; when False (default), unwrap anchors keeping only text
        start_word: Starting word index for pagination
        num_words: Maximum words to return (default 1000)
        regex: Regex pattern to filter content
        regex_padding: Characters of context around regex matches (default 50)
        use_llm_refinement: When True, apply an optional LLM cleanup pass if
            content quality is still poor after all structural tiers.
            When None (default), uses per-extension defaults: enabled for textual
            types (.html, .md) and disabled for data/binary types (.pdf, .docx,
            .pptx, .xlsx, .csv, .json, .xml).

    Returns:
        Dict with 'url' and 'content' keys, or 'error' on failure.
    """
    file_ext = _get_url_extension(url)
    is_binary = file_ext in _BINARY_DOC_EXTENSIONS

    # Apply per-extension defaults for LLM refinement if not explicitly set
    if use_llm_refinement is None:
        textual_extensions = {".html", ".md"}
        data_extensions = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".json", ".xml"}

        if file_ext in textual_extensions:
            use_llm_refinement = True
        elif file_ext in data_extensions:
            use_llm_refinement = False
        else:
            # Unknown or no extension - default to True for backward compatibility
            use_llm_refinement = True

    async with httpx.AsyncClient(follow_redirects=True, headers=DEFAULT_HEADERS) as client:
        try:
            extraction = None
            try:
                resp = await client.get(url, timeout=10.0)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code not in (403, 429, 503):
                    raise
                # Bot-protection response – retry with a real browser.
                extraction = await _extraction_pipeline.extract_from_html(
                    html="",
                    url=url,
                    include_links=include_links,
                    use_playwright=True,
                    use_llm_refinement=use_llm_refinement,
                    llm_manager=llm_manager if use_llm_refinement else None,
                )

            if extraction is None:
                # Determine routing based on URL extension AND actual response type.
                # A binary-extension URL may return HTML (e.g. a JS loading/redirect
                # page) rather than the actual document bytes.
                actual_content_type = resp.headers.get("content-type", "")
                is_html_response = (
                    "text/html" in actual_content_type
                    or "application/xhtml" in actual_content_type
                )

                if is_binary and not is_html_response:
                    # URL directly served binary data – parse it immediately.
                    extraction = await _extraction_pipeline.extract_from_bytes(
                        resp.content, file_ext, include_links
                    )
                elif is_binary:
                    # Binary-extension URL returned HTML (loading/redirect page).
                    # Use Playwright to execute JS and capture the real binary payload.
                    binary_bytes = await _extraction_pipeline.playwright_fetch_binary(url)
                    if binary_bytes is not None:
                        extraction = await _extraction_pipeline.extract_from_bytes(
                            binary_bytes, file_ext, include_links
                        )
                    else:
                        extraction = await _extraction_pipeline.extract_from_html(
                            html=resp.text,
                            url=url,
                            include_links=include_links,
                            use_playwright=True,
                            use_llm_refinement=use_llm_refinement,
                            llm_manager=llm_manager if use_llm_refinement else None,
                        )
                else:
                    extraction = await _extraction_pipeline.extract_from_html(
                        html=resp.text,
                        url=url,
                        include_links=include_links,
                        use_playwright=True,
                        use_llm_refinement=use_llm_refinement,
                        llm_manager=llm_manager if use_llm_refinement else None,
                    )

            content = extraction.content

            # Apply regex filtering/padding if provided
            if regex:
                pattern = re.compile(regex)
                matches = list(pattern.finditer(content))
                if matches:
                    filtered = []
                    for m in matches:
                        start_idx = max(0, m.start() - regex_padding)
                        end_idx = min(len(content), m.end() + regex_padding)
                        filtered.append(content[start_idx:end_idx])
                    content = "\n---\n".join(filtered)
                else:
                    content = "No matches found for regex."

            # If summarize is requested, use LLM to summarize instead of returning raw content
            if summarize:
                # Fetch unlimited content for summarization; num_words becomes the summary word cap
                words = content.split()
                full_content = " ".join(words)  # No truncation for summarize mode

                summary_result = await summarize_text(full_content, summary_prompt, num_words)
                if "error" in summary_result:
                    return {"url": url, "error": summary_result["error"]}
                return {"url": url, "summary": summary_result["summary"]}

            # Word-based pagination / truncation
            words = content.split()
            truncated = " ".join(words[start_word: start_word + num_words])
            return {"url": url, "content": truncated}
        except Exception as e:
            return {"url": url, "error": f"Error fetching {url}: {str(e)}"}


# --- FastAPI Route Wrappers (MCP Tools via fastapi-mcp) =======================
# These thin wrappers expose the tool functions as FastAPI routes.
# fastapi-mcp auto-generates MCP tools from these routes at mount time.


@app.post(
    "/searchWeb",
    operation_id="searchWeb",
    tags=["mcp-tool"],
    summary="Perform web search, general or with optional specified provider",
    dependencies=[Depends(_require_auth)],
)
async def api_search_web(
    query: str,
    provider: Optional[SearchProvider] = None,
    num_results: int = 10,
    days: int = 0,
    offset: int = 0,
) -> dict:
    return await search_web(
        query=query,
        provider=provider.value if provider else None,
        num_results=num_results,
        days=days,
        offset=offset,
    )


@app.post(
    "/fetchWebContent",
    operation_id="fetchWebContent",
    tags=["mcp-tool"],
    summary="Fetch a URL, extract content as Markdown, and optionally summarize via LLM",
    dependencies=[Depends(_require_auth)],
)
async def api_fetch_web_content(
    url: Annotated[str, Body(description="The URL to fetch and convert to markdown (required)")],
    include_links: Annotated[bool, Body(description="Preserve anchor tag hrefs; otherwise unwrap anchors keeping only text")] = True,
    start_word: Annotated[int, Body(description="Starting word index for pagination")] = 0,
    num_words: Annotated[int, Body(description="Max words to return. When summarize=True, this is the max summary word count.")] = 1000,
    regex: Annotated[Optional[str], Body(description="Regex pattern to filter content")] = None,
    regex_padding: Annotated[int, Body(description="Characters of context around regex matches (default 50)")] = 50,
    use_llm_refinement: Annotated[bool, Body(description="Apply an optional LLM cleanup pass for content quality")] = False,
    summarize: Annotated[bool, Body(description="If true, return an LLM-generated summary instead of raw content")] = False,
    summary_prompt: Annotated[str, Body(description="Custom prompt to guide the summarization (optional)")] = "",
) -> dict:
    return await fetch_web_content(
        url=url,
        include_links=include_links,
        start_word=start_word,
        num_words=num_words,
        regex=regex,
        regex_padding=regex_padding,
        use_llm_refinement=use_llm_refinement,
        summarize=summarize,
        summary_prompt=summary_prompt,
    )


async def agentic_fetch(
    prompt: str,
    max_steps: int = 10
) -> dict:
    """
    Agentic AI fetch mode that autonomously searches and browses the web.

    Takes a natural language prompt like "find the most recent Federal Reserve
    meeting minutes" and uses AI to plan and execute a series of web searches/fetches.

    Returns detailed JSON with:
    - success: Whether content was successfully found
    - content: The extracted content if successful
    - url: The primary URL where content was found
    - urls_visited: List of all URLs attempted with titles and actions taken
    - steps_taken: Detailed log of each agent step including actions and results
    - error_message: Error description if not successful

    Args:
        prompt: Natural language request describing what to find
        max_steps: Maximum number of agent steps (default 10, prevents infinite loops)

    Returns:
        Dict with detailed results including URLs visited and step-by-step actions
    """
    agent = AgenticFetchAgent(
        llm_manager=llm_manager,
        extraction_pipeline=_extraction_pipeline,
        search_func=lambda q, num_results: search_web(q, num_results=num_results),
        fetch_func=lambda url: fetch_web_content(url, include_links=True),
        max_steps=max_steps
    )

    result = await agent.execute(prompt)
    return result.to_dict()


async def capture_screenshot_endpoint(url: str) -> dict:
    """
    Capture a screenshot of a URL using Playwright.

    Args:
        url: The URL to capture

    Returns:
        Dict with success status, image_base64, and url
    """
    try:
        screenshot = await _extraction_pipeline.capture_screenshot(url)
        if screenshot is None:
            return {"success": False, "error": "Failed to capture screenshot", "url": url}
        return {"success": True, "image_base64": screenshot, "url": url}
    except Exception as e:
        return {"success": False, "error": str(e), "url": url}


@app.post(
    "/agenticFetch",
    operation_id="agenticFetch",
    tags=["mcp-tool"],
    summary="Agentic AI fetch mode - autonomously search and browse to find information",
    dependencies=[Depends(_require_auth)],
)
async def api_agentic_fetch(
    prompt: Annotated[str, Body(description="Natural language request describing what to find (required)")],
    max_steps: Annotated[int, Body(description="Maximum agent steps before giving up (default 10)")] = 10,
) -> dict:
    """
    Agentic AI fetch endpoint.

    Takes a natural language prompt and uses AI to autonomously:
    1. Plan search queries
    2. Execute web searches and fetches
    3. Navigate pages as needed (clicking, scrolling via browser automation)
    4. Evaluate whether content matches the request
    5. Return findings or detailed report of URLs visited if not found

    Returns:
        JSON with success status, content/found information,
        URLs visited list, and detailed step-by-step actions taken.
    """
    return await agentic_fetch(prompt=prompt, max_steps=max_steps)


@app.post(
    "/screenshot",
    operation_id="captureScreenshot",
    tags=["mcp-tool"],
    summary="Capture a screenshot of a URL using Playwright",
    dependencies=[Depends(_require_auth)],
)
async def api_capture_screenshot(
    url: Annotated[str, Body(description="The URL to capture (required)")],
) -> dict:
    """
    Capture a screenshot of a URL using Playwright.

    Returns base64-encoded PNG image for analysis by vision-capable LLMs.
    Useful for CAPTCHAs, visual web elements, and interactive forms.

    Returns:
        JSON with success status, image_base64 (PNG), and url.
    """
    return await capture_screenshot_endpoint(url=url)


@app.get("/")
async def health() -> dict:
    """Health check endpoint for service discovery."""
    return {"status": "ok", "name": "WebTool MCP Server"}


# --- Mount MCP at /mcp =======================================================
#
# Support both POST (for JSON-RPC requests) and GET (for SSE streaming responses).
# Roo Code uses GET /mcp to establish an SSE stream for receiving server messages.
#
# We use mount_sse() which provides:
#   - GET  /mcp         -> SSE stream for server-to-client messages
#   - POST /mcp/messages/  -> send JSON-RPC requests to the session
#
# This avoids race conditions in HTTP transport's stateless=False mode where
# GET requests would fail with "Missing session ID" errors.

fastapi_mcp = FastApiMCP(app, name="WebTool", include_tags=["mcp-tool"])
fastapi_mcp.mount_sse(mount_path="/mcp")


async def async_main() -> None:
    """Async entry point for stdio-based MCP server (Roo Code / process mode)."""
    from mcp.server.stdio import stdio_server

    # Get the underlying MCP server and its initialization options
    mcp_server = fastapi_mcp.server
    init_options = mcp_server.create_initialization_options()

    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(read_stream, write_stream, init_options)


def main() -> None:  # pragma: no cover
    """Entry point for the webtool-mcp console script.
    
    Defaults to HTTP mode when run directly. Use --stdio for Roo Code / process mode.
    """
    import argparse
    import asyncio
    import uvicorn

    parser = argparse.ArgumentParser(description="WebTool MCP Server")
    parser.add_argument("--stdio", action="store_true", help="Run as stdio server instead of HTTP")
    parser.add_argument("--host", default=server_host, help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    args = parser.parse_args()

    if args.stdio:
        print("Starting MCP server (stdio mode)...", flush=True)
        asyncio.run(async_main())
    else:
        print(f"Starting MCP server (HTTP mode on {args.host}:{args.port})...", flush=True)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()  # pragma: no cover