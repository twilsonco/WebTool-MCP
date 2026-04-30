import os
import re
import json
from typing import Optional, Union
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from markdownify import markdownify as md

load_dotenv()
mcp = FastMCP("WebTool")

# Configuration from .env
BASE_URL = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Default headers for HTTP requests (User-Agent required by many sites)
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

@mcp.tool()
async def web_fetch(
    urls: list[str], 
    include_links: bool = False, 
    start_word: int = 0, 
    num_words: int = 1000, 
    regex: str = None, 
    regex_padding: int = 50
) -> dict:
    """Fetch URLs, convert to markdown, and optionally filter via regex."""
    results = {}
    async with httpx.AsyncClient(follow_redirects=True, headers=DEFAULT_HEADERS) as client:
        for url in urls:
            try:
                resp = await client.get(url, timeout=10.0)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, 'html.parser')
                
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
                results[url] = " ".join(words[start_word : start_word + num_words])
            except Exception as e:
                results[url] = f"Error fetching {url}: {str(e)}"
    return results

@mcp.tool()
async def web_search(searches: list[dict]) -> list[dict]:
    """
    Execute one or more web searches using Brave, Google, or Tavily.

    Args:
        searches: List of search specifications. Each dict supports:
            - query (str): The search query string (required)
            - provider (str): Which provider to use: "brave", "google", "tavily" (default: "tavily")
            - num_results (int): Number of results to return (default: 10, max varies by provider)
            - start_date (str): YYYY-MM-DD format. Results after this date.
                              Only supported for brave/tavily; ignored for google.
            - end_date (str): YYYY-MM-DD format. Results before this date.
                             Only supported for brave/tavily; ignored for google.
            - offset (int): Starting index for pagination. Only supported for brave/google;
                           tavily does not support offsets (use multiple searches instead).

    Returns:
        List of results, one per search spec:
        [{"query": "...", "provider": "...", "results": [{"title": "...", "url": "...", "snippet": "..."}, ...]}, ...]
        Errors are included inline: {"query": "...", "error": "...", "provider": "..."}
    """
    results = []

    for search_spec in searches:
        query = search_spec.get("query", "")
        if not query:
            results.append({"error": "Missing required field: query"})
            continue

        provider = search_spec.get("provider", "tavily")
        num_results = min(search_spec.get("num_results", 10), 20)
        start_date = search_spec.get("start_date") or None
        end_date = search_spec.get("end_date") or None
        offset = search_spec.get("offset") or 0

        if provider == "tavily":
            result = await _search_tavily(query, num_results, start_date=start_date, end_date=end_date)
        elif provider == "brave":
            result = await _search_brave(query, num_results, start_date=start_date, end_date=end_date, offset=offset)
        elif provider == "google":
            result = await _search_google(query, num_results, offset=offset)
        else:
            results.append({"query": query, "provider": provider, "error": f"Unknown provider: {provider}"})
            continue

        # Normalize to flat format with snippet field name
        if "error" in result:
            results.append({"query": query, "provider": provider, **result})
        else:
            normalized = {
                "query": query,
                "provider": provider,
                "results": [
                    {"title": r.get("title", ""), "url": r["url"], "snippet": r.get("description", "")}
                    for r in result.get("results", [])
                ]
            }
            if start_date and provider != "google":
                normalized["start_date"] = start_date
            if end_date and provider != "google":
                normalized["end_date"] = end_date
            results.append(normalized)

    return results

async def _search_tavily(query: str, num_results: int, start_date: str = None, end_date: str = None) -> dict:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"error": "TAVILY_API_KEY not configured in .env"}

    payload = {"query": query, "search_depth": "basic", "max_results": num_results}
    if start_date:
        payload["start_date"] = start_date
    if end_date:
        payload["end_date"] = end_date

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

def _brave_freshness(start_date: str = None, end_date: str = None) -> str:
    """Convert date range to Brave freshness parameter."""
    if start_date and end_date:
        return f"{start_date}to{end_date}"
    elif start_date:
        # Default end to today for single-sided ranges
        from datetime import date
        return f"{start_date}to{date.today().isoformat()}"
    elif end_date:
        return f"1900-01-01to{end_date}"
    return ""


async def _search_brave(query: str, num_results: int, start_date: str = None, end_date: str = None, offset: int = 0) -> dict:
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return {"error": "BRAVE_API_KEY not configured in .env"}

    params = {"q": query, "count": min(num_results, 20)}
    if offset > 0:
        params["offset"] = offset

    freshness = _brave_freshness(start_date, end_date)
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

async def _call_llm(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Call the configured LLM endpoint (OpenAI-compatible).
    Returns the assistant's response content.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    headers = {"Content-Type": "application/json"}
    if OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{BASE_URL}/chat/completions",
                json={"model": MODEL_NAME, "messages": messages},
                headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"LLM API error {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        raise RuntimeError(f"LLM inference failed: {str(e)}")

DEFAULT_SUMMARY_PROMPT = """
You are a technical summarizer. Analyze the provided web content and produce a concise, well-structured markdown summary.
Focus on:
- Key findings and main points
- Technical details relevant to the query context
- Important data, statistics, or code examples
Format output as clean Markdown with headers where appropriate.
"""

DEFAULT_REDUCTION_PROMPT = """
You are an analytical assistant. Synthesize multiple document summaries into one coherent overview.
Identify common themes, differences, and provide a unified assessment.
Return the result as structured Markdown.
"""

@mcp.tool()
async def web_summarize(
    urls: list[str],
    summary_prompt: str = "",
    reduce: bool = False,
    reduction_prompt: str = "",
    max_words_per_url: int = 800
) -> dict:
    """
    Fetch URLs and summarize content via configured LLM.

    Args:
        urls: List of URLs to fetch and summarize
        summary_prompt: Custom prompt for individual URL summaries (optional)
        reduce: If True, synthesize all summaries into one combined summary
        reduction_prompt: Custom prompt for the synthesis step (used when reduce=True)
        max_words_per_url: Max words per URL before truncation (default 800)

    Returns:
        Dict with 'summaries' (per-URL) and optionally 'combined' (synthesis)
    """
    # Fetch content from all URLs
    content_map = await web_fetch(
        urls,
        num_words=max_words_per_url,
        regex=None  # No filtering; full content for summarization
    )

    summaries = {}
    system_prompt = summary_prompt if summary_prompt else DEFAULT_SUMMARY_PROMPT

    for url, content in content_map.items():
        if "Error" in content or "No matches" in content:
            summaries[url] = {"error": content[:100]}
            continue

        try:
            user_prompt = f"Summarize the following web content:\n\n{content}"
            summary_text = await _call_llm(user_prompt, system_prompt)
            summaries[url] = {"summary": summary_text.strip()}
        except RuntimeError as e:
            summaries[url] = {"error": str(e)}

    result = {"summaries": summaries}

    # Optional reduction step: synthesize all into one overview
    if reduce and len(urls) > 1:
        synthesis_prompt_text = reduction_prompt if reduction_prompt else DEFAULT_REDUCTION_PROMPT
        individual_summaries = []
        for url, data in summaries.items():
            if "summary" in data:
                individual_summaries.append(f"## Source: {url}\n\n{data['summary']}")

        if individual_summaries:
            combined_text = "\n\n---\n\n".join(individual_summaries)
            try:
                synthesis_input = f"The following are summaries from {len(individual_summaries)} sources:\n\n{combined_text}"
                combined_summary = await _call_llm(synthesis_input, synthesis_prompt_text)
                result["combined"] = {"summary": combined_summary.strip()}
            except RuntimeError as e:
                result["combined"] = {"error": str(e)}

    return result

if __name__ == "__main__":
    mcp.run()
