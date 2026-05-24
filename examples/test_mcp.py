import asyncio
import json
import os
from mcp.client.streamable_http import streamable_http_client
from mcp.client.session import ClientSession

# Get server port from env or default to 8000
SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8000"))
# fastapi-mcp mounts the StreamableHTTP endpoint at /mcp
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


async def test_search_web(session: ClientSession):
    return await call_mcp_tool(
        session,
        "searchWeb",
        {
            "query": "Python MCP server implementation",
            "num_results": 3
        },
    )


async def test_search_web_tavily(session: ClientSession):
    return await call_mcp_tool(
        session,
        "searchWeb",
        {
            "query": "Python async programming",
            "provider": "tavily",
            "num_results": 2
        },
    )


async def test_search_web_brave(session: ClientSession):
    return await call_mcp_tool(
        session,
        "searchWeb",
        {
            "query": "FastMCP tutorial",
            "provider": "brave",
            "num_results": 2
        },
    )


async def test_search_web_google(session: ClientSession):
    return await call_mcp_tool(
        session,
        "searchWeb",
        {
            "query": "MCP protocol specification",
            "provider": "google",
            "num_results": 2
        },
    )


async def test_fetch_web_content(session: ClientSession):
    return await call_mcp_tool(
        session,
        "fetchWebContent",
        {
            "url": "https://example.com",
            "num_words": 50
        },
    )


async def test_fetch_web_content_summarize(session: ClientSession):
    return await call_mcp_tool(
        session,
        "fetchWebContent",
        {
            "url": "https://example.com",
            "num_words": 300,
            "summarize": True,
        },
    )


def format_search_results(test_name: str, provider: str, response: dict):
    """Format search results nicely for display."""
    print(f"\n{'='*60}")
    print(f"  {test_name}")
    print(f"{'='*60}")
    
    if response.get("is_error"):
        print(f"✗ Error: {response}")
        return
    
    try:
        content = response.get("content", [])
        if not content:
            print(f"✗ No content returned")
            return
        
        text_content = content[0].get("text", "{}")
        data = json.loads(text_content)
        
        results = data.get("results", [])
        if not results:
            print(f"✓ Search completed but no results found")
            return
        
        print(f"✓ Search successful - Found {len(results)} result(s)")
        print(f"{'-'*60}")
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            snippet = result.get("snippet", "")[:100]
            print(f"  {i}. {title}")
            print(f"     URL: {url}")
            if snippet:
                print(f"     Snippet: {snippet}...")
            print()
            
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"✗ Failed to parse response: {e}")
        print(f"  Raw response: {response}")


def format_fetch_results(test_name: str, response: dict):
    """Format fetch_web_content results nicely for display."""
    print(f"\n{'='*60}")
    print(f"  {test_name}")
    print(f"{'='*60}")
    
    if response.get("is_error"):
        print(f"✗ Error: {response}")
        return
    
    try:
        content = response.get("content", [])
        if not content:
            print(f"✗ No content returned")
            return
        
        text_content = content[0].get("text", "{}")
        data = json.loads(text_content)
        
        url = data.get("url", "Unknown URL")
        web_content = data.get("content", "")
        
        print(f"✓ Fetch successful")
        print(f"{'-'*60}")
        print(f"  URL: {url}")
        print(f"  Content snippet (~200 chars):")
        print(f"  {web_content[:200]}...")
            
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"✗ Failed to parse response: {e}")
        print(f"  Raw response: {response}")


async def main():
    print(f"\n{'#'*60}")
    print(f"  MCP Server Test Suite")
    print(f"  Connecting to: {BASE_URL}")
    print(f"{'#'*60}")
    
    async with streamable_http_client(BASE_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            # Run all tests with formatted output
            response = await test_search_web(session)
            format_search_results("searchWeb (default - miklium)", "miklium", response)
            
            response = await test_search_web_tavily(session)
            format_search_results("searchWeb (tavily)", "tavily", response)
            
            response = await test_search_web_brave(session)
            format_search_results("searchWeb (brave)", "brave", response)
            
            response = await test_search_web_google(session)
            format_search_results("searchWeb (google)", "google", response)
            
            response = await test_fetch_web_content(session)
            format_fetch_results("fetchWebContent", response)
            
            response = await test_fetch_web_content_summarize(session)
            format_fetch_results("fetchWebContent (summarize mode)", response)
            
            print(f"\n{'#'*60}")
            print(f"  All tests completed!")
            print(f"{'#'*60}\n")


if __name__ == "__main__":
    asyncio.run(main())