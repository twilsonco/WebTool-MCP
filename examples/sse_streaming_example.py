#!/usr/bin/env python3
"""
Example demonstrating SSE (Server-Sent Events) streaming with WebTool-MCP.

This example shows how to:
1. Connect to the MCP server using streamable-http transport
2. Establish an SSE streaming connection via GET /mcp
3. Send JSON-RPC requests to the server and receive responses

The MCP HTTP transport supports:
- GET /mcp  -> SSE stream for server-to-client messages (streaming responses)
- POST /mcp/messages/  -> send JSON-RPC requests to the session

This is particularly useful for:
- Roo Code integration (uses GET /mcp for SSE streaming)
- Long-running operations that benefit from real-time updates
- Event-driven workflows where the server pushes data to clients

Requirements:
    - Server must be running on http://localhost:8000 (or set MCP_SERVER_PORT env var)
    - Start the server with: uv run python src/mcp_server/server.py --http
"""

import asyncio
import os

# MCP client SDK imports for streamable-http transport with SSE support
from mcp.client.streamable_http import streamable_http_client
from mcp.client.session import ClientSession

# Get server port from env or default to 8000
SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8000"))
# The server mounts the MCP SSE endpoint at /mcp
BASE_URL = f"http://localhost:{SERVER_PORT}/mcp"


async def demonstrate_sse_streaming():
    """
    Demonstrates establishing an SSE streaming connection to the MCP server.
    
    The streamable_http_client handles:
    - GET requests to establish SSE streams for receiving server messages
    - POST requests to send JSON-RPC commands
    
    This example shows the full lifecycle of an MCP session including:
    1. Initializing the connection
    2. Listing available tools
    3. Calling a tool (search)
    4. Properly closing the session
    """
    print(f"\n{'#'*60}")
    print("  WebTool-MCP SSE Streaming Demo")
    print(f"  Connecting to: {BASE_URL}")
    print(f"{'#'*60}\n")
    
    # The streamable_http_client context manager handles both:
    # - GET /mcp: Establishes SSE stream for server-to-client events
    #   (automatically handled by the MCP client SDK)
    # - POST /mcp/messages/: Sends JSON-RPC requests
    async with streamable_http_client(BASE_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the MCP session - this exchanges capabilities
            await session.initialize()
            print("✓ Session initialized successfully")
            
            # List available tools - shows what MCP resources are available
            print("\n--- Available Tools ---")
            
            # The tools() call goes through POST /mcp/messages/
            # but responses can come back via the SSE stream (GET)
            tools = await session.list_tools()
            
            for tool in tools.tools:
                print(f"  • {tool.name}: {getattr(tool, 'description', 'No description')[:50]}...")
            
            print(f"\n✓ Found {len(tools.tools)} tool(s)")
            
            # Call the search tool to demonstrate a complete round-trip
            print("\n--- Calling search Tool ---")
            
            result = await session.call_tool(
                "search",
                arguments={
                    "query": "Model Context Protocol MCP specification",
                    "num_results": 3
                }
            )
            
            # Process the response - MCP returns content as a list of content blocks
            for item in result.content:
                if hasattr(item, "text"):
                    import json
                    data = json.loads(item.text)
                    
                    results = data.get("results", [])
                    print(f"✓ Search returned {len(results)} result(s)")
                    
                    for i, r in enumerate(results, 1):
                        print(f"\n  Result {i}:")
                        print(f"    Title: {r.get('title', 'N/A')}")
                        print(f"    URL:   {r.get('url', 'N/A')}")
            
            if result.isError:
                print(f"✗ Tool call returned an error")
            else:
                print(f"\n✓ Tool execution successful!")
    
    # Session closed automatically when exiting the context managers
    print("\n✓ Connection closed gracefully")


async def demonstrate_sse_with_fetch():
    """
    Demonstrates SSE streaming with the fetch tool.
    
    This shows how to use SSE streams for content fetching operations,
    which can be useful when dealing with longer-running fetches.
    """
    print(f"\n{'#'*60}")
    print("  WebTool-MCP SSE Streaming with Fetch Demo")
    print(f"  Connecting to: {BASE_URL}")
    print(f"{'#'*60}\n")
    
    async with streamable_http_client(BASE_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            print("✓ Session initialized")
            print("\n--- Fetching Web Content via SSE ---")
            
            result = await session.call_tool(
                "fetch",
                arguments={
                    "url": "https://example.com",
                    "num_words": 50
                }
            )
            
            for item in result.content:
                if hasattr(item, "text"):
                    import json
                    data = json.loads(item.text)
                    
                    url = data.get("url", "N/A")
                    content_preview = data.get("content", "")[:200]
                    
                    print(f"✓ Fetched: {url}")
                    print(f"\n  Content preview:")
                    print(f"    {content_preview}...")
    
    print("\n✓ Fetch completed successfully")


def main():
    """Run the SSE streaming examples."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="WebTool-MCP SSE Streaming Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all SSE streaming demos
  %(prog)s search       Only run the search demo (search)
  %(prog)s fetch        Only run the fetch demo (fetch)

The server must be running before executing this script:
  uv run python src/mcp_server/server.py --http
        """
    )
    
    parser.add_argument(
        "demo",
        nargs="?",
        choices=["all", "search", "fetch"],
        default="all",
        help="Which demo to run (default: all)"
    )
    
    args = parser.parse_args()
    
    if args.demo in ("all", "search"):
        asyncio.run(demonstrate_sse_streaming())
    
    if args.demo in ("all", "fetch"):
        asyncio.run(demonstrate_sse_with_fetch())
    
    print(f"\n{'#'*60}")
    print("  All SSE demos completed!")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()