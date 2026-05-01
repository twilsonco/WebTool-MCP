#!/usr/bin/env python3
"""
web_search examples - Demonstrates usage of the web_search MCP tool.
This script imports and calls the actual implementation directly.
Loads API keys from .env in project root.

The multi-query API accepts a list of search specifications:
    searches = [
        {"query": "...", "provider": "tavily", "num_results": 5},
        {"query": "...", "provider": "brave", "num_results": 3, "days": 90},
    ]
    results = await real_web_search(searches)

Each search dict supports:
    - query (str): The search query string (required)
    - provider (str): Which provider to use: "brave", "google", "tavily" (default: "tavily")
    - num_results (int): Number of results to return (default: 10, max varies by provider)
    - days (int): Filter results to last N days. Simpler than start_date/end_date.
                  - Tavily: computes a start_date internally based on the days value
                  - Brave: uses freshness period codes (pd=1, pw=7, pm=31, py=365)
                  - Google: ignores this parameter silently
                  - Omit or set to 0 for no date filtering
    - offset (int): Starting index for pagination. Only supported for brave/google;
                   tavily does not support offsets.
"""
import argparse
import json
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

DRY_RUN = False


def dry_run_searches(searches: list[dict], title: str | None = None):
    """Print searches that would be executed (dry-run mode)."""
    if title:
        print(f"\n{title}")
    print(f"  Would execute {len(searches)} search(es) with web_search():")
    for s in searches:
        # Pretty-print each search spec
        provider = s.get("provider", "tavily")
        query = s.get("query", "(missing)")
        extras = {k: v for k, v in s.items() if k not in ("query", "provider")}
        extra_str = f", {extras}" if extras else ""
        print(f"    - {json.dumps(s)}")


def print_results(result: dict):
    """Pretty-print a single search result."""
    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    print(f"\nProvider: {result.get('provider', 'unknown')}")
    print(f"Query: '{result.get('query', '')}'")
    if result.get("days"):
        print(f"Date Filter: Last {result['days']} days")
    elif result.get("days") == 0:
        print(f"Date Filter: None (all results)")
    print(f"Results found: {len(result.get('results', []))}")
    print("-" * 50)

    for i, r in enumerate(result.get("results", []), 1):
        print(f"\n[{i}] {r['title']}")
        print(f"    URL: {r['url']}")
        if r.get('snippet'):
            snippet = r['snippet'][:120]
            print(f"    Snippet: {snippet}..." if len(r['snippet']) > 120 else f"    Snippet: {r['snippet']}")


async def example_tavily():
    """Example 1: Tavily search"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Tavily Search")
    print("=" * 60)

    if not os.getenv("TAVILY_API_KEY"):
        print("\nSkipping: TAVILY_API_KEY not set in .env")
        return

    searches = [
        {"query": "Python asyncio tutorial", "provider": "tavily", "num_results": 3}
    ]

    if DRY_RUN:
        dry_run_searches(searches, "Tavily Search")
        return

    results = await real_web_search(searches)

    print(f"\nSearching with Tavily Search")
    for result in results:
        print_results(result)


async def example_brave():
    """Example 2: Brave Search API."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Brave Search")
    print("=" * 60)

    if not os.getenv("BRAVE_API_KEY"):
        print("\nSkipping: BRAVE_API_KEY not set in .env")
        return

    searches = [
        {"query": "Python asyncio tutorial", "provider": "brave", "num_results": 5}
    ]

    if DRY_RUN:
        dry_run_searches(searches, "Brave Search")
        return

    results = await real_web_search(searches)

    print(f"\nSearching with Brave Search")
    for result in results:
        print_results(result)


async def example_google():
    """Example 3: Google Custom Search JSON API."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Google Custom Search")
    print("=" * 60)

    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
        print("\nSkipping: GOOGLE_API_KEY and/or GOOGLE_SEARCH_ENGINE_ID not set in .env")
        return

    searches = [
        {"query": "Python asyncio tutorial", "provider": "google", "num_results": 5}
    ]

    if DRY_RUN:
        dry_run_searches(searches, "Google Search")
        return

    results = await real_web_search(searches)

    print(f"\nSearching with Google Custom Search")
    for result in results:
        print_results(result)


async def example_date_filtering():
    """Example 4: Date filtering with brave/tavily providers.
    
    The days parameter is simpler than start_date/end_date:
    - days=1: Last 24 hours (Brave uses 'pd' freshness code)
    - days=7: Last 7 days (Brave uses 'pw')
    - days=31: Last 31 days (Brave uses 'pm')
    - days=365: Last 365 days (Brave uses 'py')
    - days=0 or omitted: No date filtering
    """
    num_days = 730  # Last 2 years
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Date Filtering with 'days' parameter")
    print(f"Filtering results to the last {num_days} days (if supported by provider)")
    print("=" * 60)

    # Check which providers are available for date filtering demo
    has_tavily = bool(os.getenv("TAVILY_API_KEY"))
    has_brave = bool(os.getenv("BRAVE_API_KEY"))

    if not (has_tavily or has_brave):
        print("\nSkipping: No Tavily or Brave API keys set in .env")
        return

    searches = []
    
    # Date filtering examples for different providers using the simpler 'days' parameter
    if has_tavily:
        searches.append({
            "query": "Python asyncio tutorial",
            "provider": "tavily",
            "num_results": 3,
            "days": num_days  # Last 2 years - Tavily computes start_date internally
        })
    
    if has_brave:
        searches.append({
            "query": "Python asyncio tutorial",
            "provider": "brave",
            "num_results": 3,
            "days": num_days  # Last 2 years - Brave uses 'py' freshness code
        })

    print(f"\nSearching with date filters (using 'days' parameter):")

    if DRY_RUN:
        dry_run_searches(searches)
        return

    results = await real_web_search(searches)
    for result in results:
        print_results(result)


async def example_date_filtering_options():
    """Example 5: Demonstrate all date filtering options.
    
    Shows the different freshness period values and their Brave equivalents:
    - days=1 or 'pd': Past day (24 hours)
    - days=7 or 'pw': Past week
    - days=31 or 'pm': Past month
    - days=365 or 'py': Past year
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Date Filtering Options (Brave freshness periods)")
    print("=" * 60)

    if not os.getenv("BRAVE_API_KEY"):
        print("\nSkipping: BRAVE_API_KEY not set in .env")
        return

    # Demonstrate different date filtering options with Brave
    searches = [
        {"query": "Python tutorial", "provider": "brave", "num_results": 2, "days": 1},   # Past day (pd)
        {"query": "Python tutorial", "provider": "brave", "num_results": 2, "days": 7},   # Past week (pw)
        {"query": "Python tutorial", "provider": "brave", "num_results": 2, "days": 31},  # Past month (pm)
        {"query": "Python tutorial", "provider": "brave", "num_results": 2, "days": 365}, # Past year (py)
    ]

    print(f"\nSearching with various date filters:")

    if DRY_RUN:
        dry_run_searches(searches)
        return

    results = await real_web_search(searches)
    for result in results:
        print_results(result)


async def example_offset_pagination():
    """Example 6: Offset pagination for brave/google providers."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Offset Pagination (brave/google)")
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

    if DRY_RUN:
        dry_run_searches(searches)
        return

    results = await real_web_search(searches)
    for result in results:
        print_results(result)


async def example_google_ignores_days():
    """Example 7: Demonstrate that Google ignores the days parameter silently."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Google Ignores 'days' Parameter (graceful degradation)")
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
            "days": 7  # This will be ignored by Google silently
        }
    ]
    
    print(f"\nNote: Google Custom Search API does not support date filtering.")
    print("The 'days' parameter is silently ignored for google provider.")

    if DRY_RUN:
        dry_run_searches(searches)
        return

    results = await real_web_search(searches)
    for result in results:
        # Note: result will NOT have days field since Google doesn't support it
        print_results(result)


async def example_multiple_queries(providers_to_test: list[str] | None = None):
    """Example 8: Execute multiple queries in a single API call."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Multiple Queries (single API call)")
    print(f"[DEBUG] providers_to_test = {providers_to_test}")
    print("=" * 60)

    # Build list of searches based on REQUESTED providers, not available API keys
    searches = []

    if os.getenv("TAVILY_API_KEY"):
        searches.append({"query": "Python asyncio tutorial", "provider": "tavily", "num_results": 2})

    if os.getenv("BRAVE_API_KEY"):
        searches.append({"query": "Latest Python news", "provider": "brave", "num_results": 2, "days": 7})

    if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
        searches.append({"query": "Python best practices", "provider": "google", "num_results": 2})

    if not searches:
        print("\nSkipping: No API keys set in .env")
        return

    if DRY_RUN:
        dry_run_searches(searches, f"Executing {len(searches)} queries in a single API call")
        return

    print(f"\nExecuting {len(searches)} queries in a single API call:")
    results = await real_web_search(searches)

    for result in results:
        print_results(result)


async def example_error_handling():
    """Example 9: Error handling for unknown provider and missing query."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Error Handling (graceful degradation)")
    print("=" * 60)

    # Test with unknown provider - should return error in results
    searches = [
        {"query": "test", "provider": "unknown_provider"}
    ]
    
    print(f"\nTesting unknown provider:")

    if DRY_RUN:
        dry_run_searches(searches)
    else:
        results = await real_web_search(searches)
        for result in results:
            print_results(result)

    # Test with missing query - should return error
    searches_missing_query = [
        {"provider": "tavily"}  # Missing 'query' field
    ]
    
    print(f"\nTesting missing query field:")

    if DRY_RUN:
        dry_run_searches(searches_missing_query)
    else:
        results = await real_web_search(searches_missing_query)
        for result in results:
            print_results(result)


async def example_config_check():
    """Example 10: Check which providers are configured."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Provider Configuration Status")
    print("=" * 60)

    print("\nConfigured API keys:")
    for key in ["TAVILY_API_KEY", "BRAVE_API_KEY", "GOOGLE_API_KEY", "GOOGLE_SEARCH_ENGINE_ID"]:
        status = "SET" if os.getenv(key) else "NOT SET"
        print(f"  {key}: {status}")


async def main(providers: list[str] | None = None):
    """
    Run web search examples.
    
    Args:
        providers: Optional list of provider names to test. If None or empty, all available
                   providers are tested. Valid values: miklium, tavily, brave, google.
    """
    # Define which providers map to which example functions
    provider_examples = {
        "tavily": [example_tavily],
        "brave": [example_brave],
        "google": [example_google],
        "miklium": [],  # miklium examples would go here if added
    }
    
    # If no providers specified (None or empty list), test all available ones
    original_providers = providers.copy() if providers else []
    if not providers:
        providers = ["tavily", "brave", "google"]
    
    # Validate and filter provider names
    valid_providers = set(provider_examples.keys())
    requested_providers = set(providers)
    invalid_providers = requested_providers - valid_providers
    
    if invalid_providers:
        print(f"Warning: Unknown provider(s) skipped: {', '.join(sorted(invalid_providers))}")
        print(f"Valid providers: {', '.join(sorted(valid_providers))}")
    
    # Filter to only valid and requested providers
    providers_to_test = [p for p in providers if p in valid_providers]
    
    if not providers_to_test:
        print("No valid providers specified. Use --help for usage information.")
        return

    print("\n" + "#" * 60)
    print(f"# web_search Examples (providers: {', '.join(providers_to_test)})")
    print("#" * 60)

    # Run examples based on selected providers
    for provider in providers_to_test:
        if provider == "tavily":
            await example_tavily()
        elif provider == "brave":
            await example_brave()
        elif provider == "google":
            await example_google()

    # These examples work with multiple providers, run if any relevant provider is selected
    has_tavily_or_brave = "tavily" in providers_to_test or "brave" in providers_to_test
    has_brave = "brave" in providers_to_test
    has_google = "google" in providers_to_test
    
    if has_tavily_or_brave:
        await example_date_filtering()
    
    if has_brave:
        await example_date_filtering_options()
        await example_offset_pagination()
    
    if has_google:
        await example_google_ignores_days()

    # Run multiple queries example only when no provider args passed (demo mode)
    if not original_providers:
        await example_multiple_queries()

    await example_config_check()

    print("\n" + "#" * 60)
    print("# Done!")
    print("#" * 60)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Web search examples demonstrating various providers and features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                  Run all provider examples
  %(prog)s tavily           Run only Tavily examples
  %(prog)s google brave     Run Google and Brave examples
  %(prog)s miklium tavily   Run Miklium (if exists) and Tavily, skip invalid
        """
    )
    
    parser.add_argument(
        "providers",
        nargs="*",
        default=None,
        metavar="PROVIDER",
        help="Provider names to test: miklium, tavily, brave, google (default: all). Unknown providers are skipped with a warning."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without making API calls"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DRY_RUN = args.dry_run
    asyncio.run(main(providers=args.providers))
