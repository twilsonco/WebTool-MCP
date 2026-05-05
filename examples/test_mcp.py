import asyncio
import httpx
import json

async def call_mcp_tool(tool_name, arguments):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:5000",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tool_call",
                "params": {
                    "tool": tool_name,
                    "arguments": arguments
                }
            }
        )
        return response.json()

async def test_web_search():
    print("\nTesting web_search...")
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
    await test_web_search()
    await test_web_fetch()
    await test_web_summarize()

if __name__ == "__main__":
    asyncio.run(main())