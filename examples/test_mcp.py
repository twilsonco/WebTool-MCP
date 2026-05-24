import argparse
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
            "url": "https://blog.comma.ai/011release/",
            "num_words": 300,
            "summarize": True,
        },
    )


def parse_example_selection(selection: str | None) -> set[int]:
    """Parse comma-separated/range example selection like '1-3,5,7' into a set of integers.
    
    Args:
        selection: Comma-separated list with optional ranges, e.g., '1-3,5,7' or '1,2,3'
        
    Returns:
        Set of example numbers (1-indexed)
        
    Examples:
        '1-3,5,7' -> {1, 2, 3, 5, 7}
        '1,2,3'   -> {1, 2, 3}
        '5'       -> {5}
    """
    if not selection:
        return set()
    
    selected = set()
    parts = selection.split(',')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        if '-' in part:
            range_parts = part.split('-')
            if len(range_parts) == 2:
                try:
                    start, end = int(range_parts[0]), int(range_parts[1])
                    if start <= end:
                        selected.update(range(start, end + 1))
                except ValueError:
                    pass
        else:
            try:
                selected.add(int(part))
            except ValueError:
                pass
    
    return selected


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MCP Server test suite - Run integration tests against the MCP server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run all 6 tests
  %(prog)s 1                    Run only test 1 (searchWeb default)
  %(prog)s 1-3                  Run tests 1, 2, and 3
  %(prog)s 1,3,5                Run tests 1, 3, and 5
  %(prog)s 2-4,6                Run tests 2, 3, 4, and 6
 
Tests (numbered for selection):
  1. test_search_web           - searchWeb with default provider (miklium)
  2. test_search_web_tavily    - searchWeb with tavily provider
  3. test_search_web_brave     - searchWeb with brave provider
  4. test_search_web_google    - searchWeb with google provider
  5. test_fetch_web_content    - fetchWebContent basic
  6. test_fetch_web_content_summarize - fetchWebContent with summarize
        """
    )
    parser.add_argument(
        "tests",
        nargs="?",
        help="Comma-separated test numbers or ranges (e.g., '1,3,5' or '1-4')"
    )
    return parser.parse_args()


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
        # Check for summarization mode (returns 'summary' key) vs regular content
        web_content = data.get("content", "")
        summary_content = data.get("summary", "")
        
        print(f"✓ Fetch successful")
        print(f"{'-'*60}")
        print(f"  URL: {url}")
        
        if summary_content:
            # Summarization mode
            print(f"  Mode: summarization")
            print(f"  Summary snippet (~500 chars):")
            print(f"  {summary_content[:500]}...")
            print(f"\n  Word count of summary: {len(summary_content.split())} words")
        else:
            # Regular fetch mode
            print(f"  Mode: regular content")
            print(f"  Content snippet (~500 chars):")
            print(f"  {web_content[:500]}...")
            
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"✗ Failed to parse response: {e}")
        print(f"  Raw response: {response}")


async def main(selected_tests: set[int] | None = None):
    print(f"\n{'#'*60}")
    print(f"  MCP Server Test Suite")
    print(f"  Connecting to: {BASE_URL}")
    if selected_tests:
        print(f"  Running selected tests: {sorted(selected_tests)}")
    print(f"{'#'*60}")
    
    async with streamable_http_client(BASE_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            # Define test functions and their indices
            tests = [
                (1, "searchWeb (default - miklium)", lambda: test_search_web(session), "search"),
                (2, "searchWeb (tavily)", lambda: test_search_web_tavily(session), "search"),
                (3, "searchWeb (brave)", lambda: test_search_web_brave(session), "search"),
                (4, "searchWeb (google)", lambda: test_search_web_google(session), "search"),
                (5, "fetchWebContent", lambda: test_fetch_web_content(session), "fetch"),
                (6, "fetchWebContent (summarize mode)", lambda: test_fetch_web_content_summarize(session), "fetch"),
            ]
            
            # Run selected tests or all tests if none specified
            for test_num, test_name, test_func, test_type in tests:
                if selected_tests is None or test_num in selected_tests:
                    response = await test_func()
                    if test_type == "search":
                        format_search_results(test_name, "", response)
                    else:
                        format_fetch_results(test_name, response)
            
            print(f"\n{'#'*60}")
            print(f"  All tests completed!")
            print(f"{'#'*60}\n")


if __name__ == "__main__":
    args = parse_args()
    selected_tests = parse_example_selection(args.tests)
    asyncio.run(main(selected_tests))