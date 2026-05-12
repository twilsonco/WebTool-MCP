import asyncio
import json
import os

from mcp.client.streamable_http import streamable_http_client
from mcp.client.session import ClientSession

# Get server port from env or default to 8000 (uvicorn default when using --http)
SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8000"))
# FastMCP's streamable_http_app mounts at /mcp (default mount_path)
BASE_URL = f"http://localhost:{SERVER_PORT}/mcp"


async def call_mcp_tool(session: ClientSession, tool_name: str, arguments: dict):
    """Call an MCP tool via the streamable-http transport using the MCP client SDK."""
    result = await session.call_tool(tool_name, arguments=arguments)
    # Convert CallToolResult to a plain dict for display
    content = []
    for item in result.content:
        if hasattr(item, "text"):
            content.append({"type": "text", "text": item.text})
        elif hasattr(item, "data"):
            content.append({"type": "data", "data": item.data})
        else:
            content.append(str(item))
    return {"content": content, "is_error": result.isError}


async def test_web_search(session: ClientSession):
    print("\nTesting web_search (default provider - miklium)...")
    response = await call_mcp_tool(
        session,
        "web_search",
        {
            "query": "Python MCP server implementation",
            "num_results": 3
        },
    )
    print(json.dumps(response, indent=2))


async def test_web_search_tavily(session: ClientSession):
    print("\nTesting web_search (tavily)...")
    response = await call_mcp_tool(
        session,
        "web_search",
        {
            "query": "Python async programming",
            "provider": "tavily",
            "num_results": 2
        },
    )
    print(json.dumps(response, indent=2))


async def test_web_search_brave(session: ClientSession):
    print("\nTesting web_search (brave)...")
    response = await call_mcp_tool(
        session,
        "web_search",
        {
            "query": "FastMCP tutorial",
            "provider": "brave",
            "num_results": 2
        },
    )
    print(json.dumps(response, indent=2))


async def test_web_search_google(session: ClientSession):
    print("\nTesting web_search (google)...")
    response = await call_mcp_tool(
        session,
        "web_search",
        {
            "query": "MCP protocol specification",
            "provider": "google",
            "num_results": 2
        },
    )
    print(json.dumps(response, indent=2))


async def test_web_fetch(session: ClientSession):
    print("\nTesting web_fetch...")
    response = await call_mcp_tool(
        session,
        "web_fetch",
        {
            "url": "https://example.com",
            "num_words": 50
        },
    )
    print(json.dumps(response, indent=2))


async def test_web_summarize(session: ClientSession):
    print("\nTesting web_summarize...")
    response = await call_mcp_tool(
        session,
        "web_summarize",
        {
            "url": "https://example.com",
            "max_words_per_url": 300
        },
    )
    print(json.dumps(response, indent=2))


async def main():
    print(f"Connecting to MCP server at {BASE_URL}")
    async with streamable_http_client(BASE_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            await test_web_search(session)
            await test_web_search_tavily(session)
            await test_web_search_brave(session)
            await test_web_search_google(session)
            await test_web_fetch(session)
            await test_web_summarize(session)


if __name__ == "__main__":
    asyncio.run(main())