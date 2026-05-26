#!/usr/bin/env python3
"""
MCP Server test suite for WebTool-MCP.

Consolidates functionality from both MCP SDK client and httpx-based testing
to provide comprehensive integration testing against the running MCP server.

Usage:
    1. Start the server: uv run python src/mcp_server/server.py --http
       (or with custom port: --http --port 8001)
    2. Run this script: uv run python examples/test_mcp.py

Features:
    - MCP SDK for proper protocol handling
    - Health check / server connectivity test
    - Tools listing with descriptions
    - Selective test execution by number or category
    - Multiple search provider tests (default, tavily, brave, google)
    - Fetch content and summarization tests
"""

import argparse
import asyncio
import json
import os
from typing import Any

# Get server port from env or default to 8000 (matches test_server_connection.py)
SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8000"))
# fastapi-mcp mounts the StreamableHTTP endpoint at /mcp
BASE_URL = f"http://localhost:{SERVER_PORT}/mcp"

# Import MCP SDK components (matches test_mcp.py approach)
from mcp.client.streamable_http import streamable_http_client
from mcp.client.session import ClientSession


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


# ============================================================================
# Test Functions
# ============================================================================

async def test_health(session: ClientSession) -> dict[str, Any]:
    """Test server connectivity via MCP session initialization."""
    print(f"\n{'='*60}")
    print("  Testing Server Health (via Initialize)")
    print(f"  Target: {BASE_URL}")
    print('='*60)
    
    try:
        # Initialize should succeed if server is running
        print("\n✓ Server is responding and MCP session established!")
        return {"success": True}
    except Exception as e:
        print(f"\n✗ Cannot connect to server at {BASE_URL}")
        print(f"  Error: {e}")
        return {"success": False, "error": str(e)}


async def test_list_tools(session: ClientSession) -> dict[str, Any]:
    """List available tools from the MCP server."""
    print(f"\n{'='*60}")
    print("  Testing Tools List")
    print('='*60)
    
    result = await session.list_tools()
    tools = result.tools
    
    print(f"\n✓ Found {len(tools)} tool(s):")
    
    for tool in tools:
        name = getattr(tool, 'name', 'unknown')
        desc = str(getattr(tool, 'description', 'No description'))[:60]
        print(f"   • {name}: {desc}...")
    
    return {"success": True, "tools": len(tools)}


async def test_search_web(session: ClientSession):
    """Test searchWeb with default provider (miklium)."""
    return await call_mcp_tool(
        session,
        "searchWeb",
        {
            "query": "Python MCP server implementation",
            "num_results": 3
        },
    )


async def test_search_web_tavily(session: ClientSession):
    """Test searchWeb with tavily provider."""
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
    """Test searchWeb with brave provider."""
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
    """Test searchWeb with google provider."""
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
    """Test fetchWebContent basic."""
    return await call_mcp_tool(
        session,
        "fetchWebContent",
        {
            "url": "https://example.com",
            "num_words": 50
        },
    )


async def test_fetch_web_content_summarize(session: ClientSession):
    """Test fetchWebContent with summarization."""
    return await call_mcp_tool(
        session,
        "fetchWebContent",
        {
            "url": "https://blog.comma.ai/011release/",
            "num_words": 300,
            "summarize": True,
        },
    )


# ============================================================================
# Output Formatting
# ============================================================================

def format_search_results(test_name: str, response: dict):
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
        
        provider = data.get("provider", "unknown")
        print(f"✓ Search successful - Found {len(results)} result(s)")
        if provider:
            print(f"  Provider: {provider}")
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
        word_count = data.get("word_count", 0)
        
        # Check for summarization mode (returns 'summary' key) vs regular content
        web_content = data.get("content", "")
        summary_content = data.get("summary", "")
        
        print(f"✓ Fetch successful")
        print(f"{'-'*60}")
        print(f"  URL: {url}")
        if word_count:
            print(f"  Word count: {word_count}")
        
        if summary_content:
            # Summarization mode
            print(f"  Mode: summarization")
            print(f"  Summary snippet (~500 chars):")
            print(f"  {summary_content[:500]}...")
        else:
            # Regular fetch mode
            print(f"  Mode: regular content")
            if web_content:
                print(f"  Content snippet (~500 chars):")
                print(f"  {web_content[:500]}...")
            
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"✗ Failed to parse response: {e}")
        print(f"  Raw response: {response}")


# ============================================================================
# Argument Parsing
# ============================================================================

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
  %(prog)s                      Run all tests (health, tools, and numbered tests)
  %(prog)s health               Only test server connectivity
  %(prog)s tools                Only list available tools
  %(prog)s search               Run all search tests (1-4)
  %(prog)s fetch                Run both fetch tests (5-6)
  %(prog)s all                  Same as no args - run everything
  %(prog)s 1                    Run only test 1 (searchWeb default)
  %(prog)s 2-4                  Run tests 2, 3, and 4
  %(prog)s 1-6                  Run all numbered tests (skips health/tools)
 
Prerequisites:
  Start the server on port {PORT}:
    uv run python src/mcp_server/server.py --http --port {PORT}

Tests (numbered for selection):
  1. test_search_web           - searchWeb with default provider (miklium)
  2. test_search_web_tavily    - searchWeb with tavily provider
  3. test_search_web_brave     - searchWeb with brave provider
  4. test_search_web_google    - searchWeb with google provider
  5. test_fetch_web_content    - fetchWebContent basic (50 words)
  6. test_fetch_web_content_summarize - fetchWebContent with summarization

Special categories:
  health    - Test server connectivity (always runs first if 'all')
  tools     - List available MCP tools
  search    - Run all provider tests (1-4)
  fetch     - Run both fetch tests (5-6)
        """.format(PORT=SERVER_PORT)
    )
    parser.add_argument(
        "tests",
        nargs="?",
        help=(
            "Test selection: 'health', 'tools', 'search', 'fetch', 'all', "
            "or comma-separated/range test numbers (e.g., '1,3,5' or '1-4')"
        )
    )
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

async def main(selected_tests: set[int] | None = None, run_health: bool = True, 
               run_tools: bool = False, test_category: str | None = None):
    """Run the MCP server tests.
    
    Args:
        selected_tests: Set of numbered test indices to run (1-6)
        run_health: Whether to run the health check first
        run_tools: Whether to list available tools
        test_category: Category-based selection ('search', 'fetch')
    """
    print(f"\n{'#'*60}")
    print(f"  MCP Server Test Suite")
    print(f"  Connecting to: {BASE_URL}")
    
    if test_category:
        print(f"  Category filter: {test_category}")
    elif selected_tests:
        print(f"  Running tests: {sorted(selected_tests)}")
    
    # Determine which numbered tests to run based on category
    if test_category == "search":
        selected_tests = {1, 2, 3, 4}
    elif test_category == "fetch":
        selected_tests = {5, 6}
    
    print(f"{'#'*60}")
    
    async with streamable_http_client(BASE_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            # Health check always runs first if requested
            if run_health:
                health_result = await test_health(session)
                if not health_result.get("success"):
                    print("\n✗ Server health check failed. Exiting.")
                    return 1
            
            # Tools listing
            if run_tools:
                await test_list_tools(session)
            
            # Define numbered test functions and their indices
            tests = [
                (1, "searchWeb (default - miklium)", lambda: test_search_web(session), "search"),
                (2, "searchWeb (tavily)", lambda: test_search_web_tavily(session), "search"),
                (3, "searchWeb (brave)", lambda: test_search_web_brave(session), "search"),
                (4, "searchWeb (google)", lambda: test_search_web_google(session), "search"),
                (5, "fetchWebContent", lambda: test_fetch_web_content(session), "fetch"),
                (6, "fetchWebContent (summarize mode)", lambda: test_fetch_web_content_summarize(session), "fetch"),
            ]
            
            # Run selected tests or all if none specified
            for test_num, test_name, test_func, test_type in tests:
                if selected_tests is None or test_num in selected_tests:
                    response = await test_func()
                    if test_type == "search":
                        format_search_results(test_name, response)
                    else:
                        format_fetch_results(test_name, response)
            
            print(f"\n{'#'*60}")
            print(f"  All tests completed!")
            print(f"{'#'*60}\n")
    
    return 0


if __name__ == "__main__":
    args = parse_args()
    
    # Parse test selection
    run_health = True
    run_tools = False
    selected_tests: set[int] | None = None
    
    test_arg = args.tests.lower() if args.tests else "all"
    
    if test_arg in ("all", ""):
        # Run everything (health, tools, all numbered tests)
        run_health = True
        run_tools = True
    elif test_arg == "health":
        run_health = True
        run_tools = False
    elif test_arg == "tools":
        run_health = True  # Need session for tools, but won't list them if not requested
        run_tools = True
    elif test_arg == "search":
        selected_tests = {1, 2, 3, 4}
    elif test_arg == "fetch":
        selected_tests = {5, 6}
    else:
        # Try parsing as numbered selection
        selected_tests = parse_example_selection(args.tests)
    
    exit(asyncio.run(main(selected_tests, run_health, run_tools)))