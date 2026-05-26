#!/usr/bin/env python3
"""
Manual test script for WebTool-MCP server at localhost:8001.

This is an EXAMPLE SCRIPT (not a pytest test) for manual testing.
It makes actual HTTP requests to the running MCP server.

Usage:
    1. Start the server: uv run python src/mcp_server/server.py --http
       (The user mentioned port 8001, so use: uv run python src/mcp_server/server.py --http --port 8001)
    2. Run this script: uv run python examples/test_server_connection.py

This script tests the MCP HTTP endpoint directly using httpx.
"""

import asyncio
import json
from typing import Any

import httpx


# Server configuration - must match where the server is running
SERVER_HOST = "localhost"
SERVER_PORT = 8001
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


async def test_server_health():
    """Test basic server connectivity."""
    print(f"\n{'='*60}")
    print("  Testing Server Health")
    print(f"  Target: {BASE_URL}")
    print('='*60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{BASE_URL}/mcp")
            print(f"\n✓ Server is responding!")
            print(f"  Status: {response.status_code}")
            print(f"  Headers: {dict(response.headers)}")
        except httpx.ConnectError as e:
            print(f"\n✗ Cannot connect to server at {BASE_URL}")
            print(f"  Error: {e}")
            print("\n  Make sure the server is running with --http flag:")
            print(f"    uv run python src/mcp_server/server.py --http --port {SERVER_PORT}")
            return False
    
    return True


async def send_mcp_request(
    client: httpx.AsyncClient,
    method: str,
    params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Send a JSON-RPC request to the MCP server."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method
    }
    if params:
        payload["params"] = params
    
    response = await client.post(
        f"{BASE_URL}/mcp/messages/",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code}: {response.text}")
    
    return response.json()


async def test_list_tools():
    """Test the tools/list endpoint."""
    print(f"\n{'='*60}")
    print("  Testing MCP Initialize + List Tools")
    print('='*60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # First, initialize the session
        print("\n1. Sending Initialize request...")
        
        init_result = await send_mcp_request(
            client,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"roots": {}, "sampling": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        )
        
        print(f"   ✓ Initialize successful")
        if "result" in init_result:
            server_info = init_result["result"].get("serverInfo", {})
            print(f"   Server: {server_info.get('name', 'unknown')} v{server_info.get('version', '?')}")
        
        # Now list tools
        print("\n2. Sending tools/list request...")
        
        result = await send_mcp_request(client, "tools/list")
        
        if "result" in result:
            tools = result["result"].get("tools", [])
            print(f"\n   ✓ Found {len(tools)} tool(s):")
            
            for tool in tools:
                name = tool.get("name", "unknown")
                desc = tool.get("description", "No description")[:60]
                print(f"   • {name}: {desc}...")
        else:
            print(f"\n   ✗ Error: {result}")
    
    return True


async def test_call_search_tool():
    """Test calling the searchWeb tool."""
    print(f"\n{'='*60}")
    print("  Testing Call Tool: searchWeb")
    print('='*60)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Initialize
        await send_mcp_request(
            client,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"roots": {}, "sampling": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        )
        
        print("\nCalling searchWeb tool...")
        
        result = await send_mcp_request(
            client,
            "tools/call",
            {
                "name": "searchWeb",
                "arguments": {
                    "query": "Model Context Protocol MCP specification",
                    "num_results": 3
                }
            }
        )
        
        if "result" in result:
            content = result["result"].get("content", [])
            
            for item in content:
                if item.get("type") == "text":
                    data = json.loads(item["text"])
                    results = data.get("results", [])
                    
                    print(f"\n   ✓ Search returned {len(results)} result(s)")
                    
                    for i, r in enumerate(results[:3], 1):
                        title = r.get("title", "N/A")
                        url = r.get("url", "N/A")
                        print(f"\n   Result {i}:")
                        print(f"      Title: {title}")
                        print(f"      URL:   {url}")
        else:
            print(f"\n   ✗ Error or not implemented: {result}")
    
    return True


async def test_call_fetch_tool():
    """Test calling the fetchWebContent tool."""
    print(f"\n{'='*60}")
    print("  Testing Call Tool: fetchWebContent")
    print('='*60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Initialize
        await send_mcp_request(
            client,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"roots": {}, "sampling": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        )
        
        print("\nCalling fetchWebContent tool...")
        
        result = await send_mcp_request(
            client,
            "tools/call",
            {
                "name": "fetchWebContent",
                "arguments": {
                    "url": "https://example.com",
                    "num_words": 50
                }
            }
        )
        
        if "result" in result:
            content = result["result"].get("content", [])
            
            for item in content:
                if item.get("type") == "text":
                    data = json.loads(item["text"])
                    
                    url = data.get("url", "N/A")
                    word_count = data.get("word_count", 0)
                    
                    print(f"\n   ✓ Fetched: {url}")
                    print(f"   Word count: {word_count}")
                    
                    # Show preview
                    text_content = data.get("content", "")
                    if text_content:
                        print(f"\n   Content preview (first 200 chars):")
                        print(f"      {text_content[:200]}...")
        else:
            print(f"\n   ✗ Error or not implemented: {result}")
    
    return True


async def main():
    """Run all connection tests."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Manual test script for WebTool-MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all connection tests
  %(prog)s health       Only test server connectivity
  %(prog)s tools        Only test list_tools
  %(prog)s search       Only test searchWeb tool
  %(prog)s fetch        Only test fetchWebContent tool

Prerequisites:
  Start the server on port {PORT}:
    uv run python src/mcp_server/server.py --http --port {PORT}
        """.format(PORT=SERVER_PORT)
    )
    
    parser.add_argument(
        "test",
        nargs="?",
        choices=["all", "health", "tools", "search", "fetch"],
        default="all",
        help="Which test to run (default: all)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  WebTool-MCP Server Connection Test")
    print(f"  Target: http://localhost:{SERVER_PORT}")
    print("="*60)
    
    # Always test health first
    if args.test in ("all", "health"):
        success = await test_server_health()
        if not success:
            print("\n✗ Server health check failed. Exiting.")
            return 1
    
    if args.test in ("all", "tools"):
        await test_list_tools()
    
    if args.test in ("all", "search"):
        await test_call_search_tool()
    
    if args.test in ("all", "fetch"):
        await test_call_fetch_tool()
    
    print("\n" + "="*60)
    print("  All tests completed!")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))