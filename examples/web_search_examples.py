#!/usr/bin/env python3
"""
web_search examples - Demonstrates usage of the web_search MCP tool.
This script imports and calls the actual implementation directly.
Loads API keys from .env in project root.
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from dotenv import load_dotenv

load_dotenv(project_root / ".env")

# Import the actual implementation from server.py
# web_search is the main tool; _search_tavily/_search_brave/_search_google are internal helpers
from src.mcp_server.server import web_search as real_web_search

query = "Python asyncio tutorial"


def print_results(result: dict):
    """Pretty-print search results."""
    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    print(f"\nProvider: {result.get('provider', 'unknown')}")
    print(f"Query: '{result.get('query', '')}'")
    print(f"Results found: {result.get('count', len(result.get('results', [])))}")
    print("-" * 50)

    for i, r in enumerate(result.get("results", []), 1):
        print(f"\n[{i}] {r['title']}")
        print(f"    URL: {r['url']}")
        if r.get('description'):
            desc = r['description'][:120]
            print(f"    Desc: {desc}..." if len(r['description']) > 120 else f"    Desc: {r['description']}")


async def example_tavily():
    """Example 1: Tavily search (default provider)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Tavily Search (default)")
    print("=" * 60)

    if not os.getenv("TAVILY_API_KEY"):
        print("\nSkipping: TAVILY_API_KEY not set in .env")
        return

    result = await real_web_search(query, provider="tavily", num_results=3)
    print(f"\nSearching: '{query}' with Tavily Search")
    print_results(result)


async def example_brave():
    """Example 2: Brave Search API."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Brave Search")
    print("=" * 60)

    if not os.getenv("BRAVE_API_KEY"):
        print("\nSkipping: BRAVE_API_KEY not set in .env")
        return

    result = await real_web_search(query, provider="brave", num_results=5)
    print(f"\nSearching: '{query}' with Brave Search")
    print_results(result)


async def example_google():
    """Example 3: Google Custom Search JSON API."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Google Custom Search")
    print("=" * 60)

    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
        print("\nSkipping: GOOGLE_API_KEY and/or GOOGLE_SEARCH_ENGINE_ID not set in .env")
        return

    result = await real_web_search(query, provider="google", num_results=5)
    print(f"\nSearching: '{query}' with Google Custom Search")
    print_results(result)


async def example_unknown_provider():
    """Example 4: Error handling for unknown provider."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Unknown Provider (error handling)")
    print("=" * 60)

    result = await real_web_search("test", provider="unknown_provider")
    print(f"\nTrying provider 'unknown_provider':")
    print_results(result)


async def example_config_check():
    """Example 5: Check which providers are configured."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Provider Configuration Status")
    print("=" * 60)

    print("\nConfigured API keys:")
    for key in ["TAVILY_API_KEY", "BRAVE_API_KEY", "GOOGLE_API_KEY"]:
        status = "SET" if os.getenv(key) else "NOT SET"
        print(f"  {key}: {status}")


async def main():
    print("\n" + "#" * 60)
    print("# web_search Examples (using real implementation)")
    print("#" * 60)

    await example_tavily()
    await example_brave()
    await example_google()
    await example_unknown_provider()
    await example_config_check()

    print("\n" + "#" * 60)
    print("# Done!")
    print("#" * 60)


if __name__ == "__main__":
    asyncio.run(main())
