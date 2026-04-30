#!/usr/bin/env python3
"""
web_search examples - Demonstrates usage of the web_search MCP tool.
This script imports and calls the actual implementation directly.
Loads API keys from .env in project root.

The new multi-query API accepts a list of search specifications:
    searches = [
        {"query": "...", "provider": "tavily", "num_results": 5},
        {"query": "...", "provider": "brave", "num_results": 3, "start_date": "2025-01-01"},
    ]
    results = await real_web_search(searches)

Each search dict supports:
    - query (str): The search query string (required)
    - provider (str): Which provider to use: "brave", "google", "tavily" (default: "tavily")
    - num_results (int): Number of results to return (default: 10, max varies by provider)
    - start_date (str): YYYY-MM-DD format. Results after this date.
                      Only supported for brave/tavily; ignored for google.
    - end_date (str): YYYY-MM-DD format. Results before this date.
                     Only supported for brave/tavily; ignored for google.
    - offset (int): Starting index for pagination. Only supported for brave/google;
                   tavily does not support offsets.
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


def print_results(result: dict):
    """Pretty-print a single search result."""
    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    print(f"\nProvider: {result.get('provider', 'unknown')}")
    print(f"Query: '{result.get('query', '')}'")
    if result.get("start_date"):
        print(f"Start Date Filter: {result['start_date']}")
    if result.get("end_date"):
        print(f"End Date Filter: {result['end_date']}")
    print(f"Results found: {len(result.get('results', []))}")
    print("-" * 50)

    for i, r in enumerate(result.get("results", []), 1):
        print(f"\n[{i}] {r['title']}")
        print(f"    URL: {r['url']}")
        if r.get('snippet'):
            snippet = r['snippet'][:120]
            print(f"    Snippet: {snippet}..." if len(r['snippet']) > 120 else f"    Snippet: {r['snippet']}")


async def example_tavily():
    """Example 1: Tavily search with new multi-query API."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Tavily Search (new multi-query format)")
    print("=" * 60)

    if not os.getenv("TAVILY_API_KEY"):
        print("\nSkipping: TAVILY_API_KEY not set in .env")
        return

    # New API: pass a list of search specifications
    searches = [
        {"query": "Python asyncio tutorial", "provider": "tavily", "num_results": 3}
    ]
    results = await real_web_search(searches)
    
    print(f"\nSearching with Tavily Search (new API format)")
    for result in results:
        print_results(result)


async def example_brave():
    """Example 2: Brave Search API with new multi-query API."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Brave Search (new multi-query format)")
    print("=" * 60)

    if not os.getenv("BRAVE_API_KEY"):
        print("\nSkipping: BRAVE_API_KEY not set in .env")
        return

    # New API: pass a list of search specifications
    searches = [
        {"query": "Python asyncio tutorial", "provider": "brave", "num_results": 5}
    ]
    results = await real_web_search(searches)
    
    print(f"\nSearching with Brave Search (new API format)")
    for result in results:
        print_results(result)


async def example_google():
    """Example 3: Google Custom Search JSON API with new multi-query API."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Google Custom Search (new multi-query format)")
    print("=" * 60)

    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
        print("\nSkipping: GOOGLE_API_KEY and/or GOOGLE_SEARCH_ENGINE_ID not set in .env")
        return

    # New API: pass a list of search specifications
    searches = [
        {"query": "Python asyncio tutorial", "provider": "google", "num_results": 5}
    ]
    results = await real_web_search(searches)
    
    print(f"\nSearching with Google Custom Search (new API format)")
    for result in results:
        print_results(result)


async def example_date_filtering():
    """Example 4: Date filtering with brave/tavily providers."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Date Filtering (brave/tavily)")
    print("=" * 60)

    # Check which providers are available for date filtering demo
    has_tavily = bool(os.getenv("TAVILY_API_KEY"))
    has_brave = bool(os.getenv("BRAVE_API_KEY"))

    if not (has_tavily or has_brave):
        print("\nSkipping: No Tavily or Brave API keys set in .env")
        return

    searches = []
    
    # Date filtering example for Tavily (start_date and end_date)
    if has_tavily:
        searches.append({
            "query": "Python asyncio tutorial",
            "provider": "tavily",
            "num_results": 3,
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        })
    
    # Date filtering example for Brave (freshness parameter)
    if has_brave:
        searches.append({
            "query": "Python asyncio tutorial",
            "provider": "brave",
            "num_results": 3,
            "start_date": "2025-01-01"
        })

    print(f"\nSearching with date filters applied:")
    results = await real_web_search(searches)
    for result in results:
        print_results(result)


async def example_offset_pagination():
    """Example 5: Offset pagination for brave/google providers."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Offset Pagination (brave/google)")
    print("=" * 60)

    # Check which providers are available for offset demo
    has_brave = bool(os.getenv("BRAVE_API_KEY"))
    has_google = bool(os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_SEARCH_ENGINE_ID"))

    if not (has_brave or has_google):
        print("\nSkipping: No Brave or Google API keys set in .env")
        return

    searches = []
    
    # Offset pagination example for Brave
    if has_brave:
        searches.append({
            "query": "Python tutorial",
            "provider": "brave",
            "num_results": 3,
            "offset": 0  # First page
        })
        searches.append({
            "query": "Python tutorial",
            "provider": "brave",
            "num_results": 3,
            "offset": 3  # Second page (skip first 3)
        })
    
    # Offset pagination example for Google
    if has_google:
        searches.append({
            "query": "Python tutorial",
            "provider": "google",
            "num_results": 5,
            "offset": 0  # First page
        })
        searches.append({
            "query": "Python tutorial",
            "provider": "google",
            "num_results": 5,
            "offset": 5  # Second page (skip first 5)
        })

    print(f"\nSearching with offset pagination:")
    results = await real_web_search(searches)
    for result in results:
        print_results(result)


async def example_google_ignores_dates():
    """Example 6: Demonstrate that Google ignores date filters silently."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Google Ignores Date Filters (graceful degradation)")
    print("=" * 60)

    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
        print("\nSkipping: GOOGLE_API_KEY and/or GOOGLE_SEARCH_ENGINE_ID not set in .env")
        return

    # Google doesn't support date filtering, but the API accepts it gracefully
    searches = [
        {
            "query": "Python asyncio tutorial",
            "provider": "google",
            "num_results": 5,
            "start_date": "2025-01-01",  # This will be ignored by Google
            "end_date": "2025-12-31"     # This will also be ignored by Google
        }
    ]
    
    print(f"\nNote: Google Custom Search API does not support date filtering.")
    print("The start_date/end_date parameters are silently ignored for google provider.")
    results = await real_web_search(searches)
    for result in results:
        # Note: result will NOT have start_date/end_date fields since Google doesn't support them
        print_results(result)


async def example_multiple_queries():
    """Example 7: Execute multiple queries in a single API call."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Multiple Queries (single API call)")
    print("=" * 60)

    # Build list of searches based on available API keys
    searches = []
    
    if os.getenv("TAVILY_API_KEY"):
        searches.append({"query": "Python asyncio tutorial", "provider": "tavily", "num_results": 2})
    
    if os.getenv("BRAVE_API_KEY"):
        searches.append({"query": "Latest Python news", "provider": "brave", "num_results": 2, "start_date": "2025-01-01"})
    
    if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
        searches.append({"query": "Python best practices", "provider": "google", "num_results": 2})

    if not searches:
        print("\nSkipping: No API keys set in .env")
        return

    print(f"\nExecuting {len(searches)} queries in a single API call:")
    results = await real_web_search(searches)
    
    for result in results:
        print_results(result)


async def example_error_handling():
    """Example 8: Error handling for unknown provider and missing query."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Error Handling (graceful degradation)")
    print("=" * 60)

    # Test with unknown provider - should return error in results
    searches = [
        {"query": "test", "provider": "unknown_provider"}
    ]
    
    print(f"\nTesting unknown provider:")
    results = await real_web_search(searches)
    for result in results:
        print_results(result)

    # Test with missing query - should return error
    searches_missing_query = [
        {"provider": "tavily"}  # Missing 'query' field
    ]
    
    print(f"\nTesting missing query field:")
    results = await real_web_search(searches_missing_query)
    for result in results:
        print_results(result)


async def example_config_check():
    """Example 9: Check which providers are configured."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Provider Configuration Status")
    print("=" * 60)

    print("\nConfigured API keys:")
    for key in ["TAVILY_API_KEY", "BRAVE_API_KEY", "GOOGLE_API_KEY", "GOOGLE_SEARCH_ENGINE_ID"]:
        status = "SET" if os.getenv(key) else "NOT SET"
        print(f"  {key}: {status}")


async def main():
    print("\n" + "#" * 60)
    print("# web_search Examples (using new multi-query API)")
    print("#" * 60)

    await example_tavily()
    await example_brave()
    await example_google()
    await example_date_filtering()
    await example_offset_pagination()
    await example_google_ignores_dates()
    await example_multiple_queries()
    await example_error_handling()
    await example_config_check()

    print("\n" + "#" * 60)
    print("# Done!")
    print("#" * 60)


if __name__ == "__main__":
    asyncio.run(main())
