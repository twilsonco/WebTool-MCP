import asyncio
import httpx
import json
import os
from typing import Optional

# Get server port from env or default to 8000 (uvicorn default when using --http)
SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8000"))
BASE_URL = f"http://localhost:{SERVER_PORT}"

async def call_mcp_tool(tool_name, arguments):
    """Call an MCP tool via the streamable-http transport."""
    # FastMCP's streamable-http uses /tools/{tool_name} with query params for session
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/tools/{tool_name}",
            json={"arguments": arguments},
            headers={"Content-Type": "application/json"},
            timeout=60.0
        )
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text[:500]}"}
        return response.json()


async def test_web_search():
    print("\nTesting web_search (default provider - miklium)...")
    response = await call_mcp_tool(
        "web_search",
        {
            "searches": [
                {
                    "query": "Python MCP server implementation",
                    "num_results": 3
                }
            ]
        }
    )
    print(json.dumps(response, indent=2))


async def test_web_search_tavily():
    print("\nTesting web_search (tavily)...")
    response = await call_mcp_tool(
        "web_search",
        {
            "searches": [
                {
                    "query": "Python async programming",
                    "provider": "tavily",
                    "num_results": 2
                }
            ]
        }
    )
    print(json.dumps(response, indent=2))


async def test_web_search_brave():
    print("\nTesting web_search (brave)...")
    response = await call_mcp_tool(
        "web_search",
        {
            "searches": [
                {
                    "query": "FastMCP tutorial",
                    "provider": "brave",
                    "num_results": 2
                }
            ]
        }
    )
    print(json.dumps(response, indent=2))


async def test_web_search_google():
    print("\nTesting web_search (google)...")
    response = await call_mcp_tool(
        "web_search",
        {
            "searches": [
                {
                    "query": "MCP protocol specification",
                    "provider": "google",
                    "num_results": 2
                }
            ]
        }
    )
    print(json.dumps(response, indent=2))


async def test_web_fetch():
    print("\nTesting web_fetch...")
    response = await call_mcp_tool(
        "web_fetch",
        {
            "urls": ["https://example.com"],
            "num_words": 50
        }
    )
    print(json.dumps(response, indent=2))


async def test_web_summarize():
    print("\nTesting web_summarize...")
    response = await call_mcp_tool(
        "web_summarize",
        {
            "urls": ["https://example.com"],
            "max_words_per_url": 300
        }
    )
    print(json.dumps(response, indent=2))


async def main():
    print(f"Connecting to MCP server at {BASE_URL}")
    await test_web_search()
    await test_web_search_tavily()
    await test_web_search_brave()
    await test_web_search_google()
    await test_web_fetch()
    await test_web_summarize()


if __name__ == "__main__":
    asyncio.run(main())
