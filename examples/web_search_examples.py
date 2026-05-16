#!/usr/bin/env python3
"""
searchWeb examples - Demonstrates usage of the searchWeb MCP tool.
This script imports and calls the actual implementation directly.
Loads API keys from .env in project root.

The atomic API accepts individual search parameters:
    result = await real_web_search("query", provider="tavily", num_results=5)

Each call supports:
    - query (str): The search query string (required)
    - provider (str): Which provider to use: "miklium", "brave", "google", or "tavily"
                      Only configured providers are valid. If not specified, uses
                      the first available provider with failover support.
    - num_results (int): Number of results to return (default: 10, max varies by provider)
    - days (int): Filter results to last N days. Simpler than start_date/end_date.
                  - Tavily: computes a start_date internally based on the days value
                  - Brave: uses freshness period codes (pd=1, pw=7, pm=31, py=365)
                  - Google/Miklium: ignores this parameter silently
                  - Omit or set to 0 for no date filtering
    - offset (int): Starting index for pagination. Only supported for brave/google;
                   tavily and miklium do not support offsets.
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
from src.mcp_server.server import web_search as real_web_search

DRY_RUN = False


def dry_run_search(query: str, provider: str | None = None, title: str | None = None, **kwargs):
    """Print search that would be executed (dry-run mode)."""
    if title:
        print(f"\n{title}")
    params = {"query": query}
    if provider:
        params["provider"] = provider
    extras = {k: v for k, v in kwargs.items() if v}
    params.update(extras)
    print(f"  Would execute searchWeb() with: {json.dumps(params)}")


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


async def example_miklium():
    """Example 1: Miklium search (default, no API key required)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Miklium Search (default, free)")
    print("=" * 60)

    if DRY_RUN:
        dry_run_search("Python MCP server implementation", title="Miklium Search")
        return

    result = await real_web_search("Python MCP server implementation", num_results=3)
    print_results(result)


async def example_tavily():
    """Example 2: Tavily search."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Tavily Search")
    print("=" * 60)

    if not os.getenv("TAVILY_API_KEY"):
        print("\nSkipping: TAVILY_API_KEY not set in .env")
        return

    if DRY_RUN:
        dry_run_search("Python asyncio tutorial", provider="tavily", title="Tavily Search")
        return

    result = await real_web_search("Python asyncio tutorial", provider="tavily", num_results=3)
    print_results(result)


async def example_brave():
    """Example 3: Brave Search API."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Brave Search")
    print("=" * 60)

    if not os.getenv("BRAVE_API_KEY"):
        print("\nSkipping: BRAVE_API_KEY not set in .env")
        return

    if DRY_RUN:
        dry_run_search("Python asyncio tutorial", provider="brave", title="Brave Search")
        return

    result = await real_web_search("Python asyncio tutorial", provider="brave", num_results=5)
    print_results(result)


async def example_google():
    """Example 4: Google Custom Search JSON API."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Google Custom Search")
    print("=" * 60)

    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
        print("\nSkipping: GOOGLE_API_KEY and/or GOOGLE_SEARCH_ENGINE_ID not set in .env")
        return

    if DRY_RUN:
        dry_run_search("Python asyncio tutorial", provider="google", title="Google Search")
        return

    result = await real_web_search("Python asyncio tutorial", provider="google", num_results=5)
    print_results(result)


async def example_date_filtering():
    """Example 5: Date filtering with brave/tavily providers.

    The days parameter is simpler than start_date/end_date:
    - days=1: Last 24 hours (Brave uses 'pd' freshness code)
    - days=7: Last 7 days (Brave uses 'pw')
    - days=31: Last 31 days (Brave uses 'pm')
    - days=365: Last 365 days (Brave uses 'py')
    - days=0 or omitted: No date filtering
    """
    num_days = 730  # Last 2 years
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Date Filtering with 'days' parameter")
    print(f"Filtering results to the last {num_days} days (if supported by provider)")
    print("=" * 60)

    # Check which providers are available for date filtering demo
    has_tavily = bool(os.getenv("TAVILY_API_KEY"))
    has_brave = bool(os.getenv("BRAVE_API_KEY"))

    if not (has_tavily or has_brave):
        print("\nSkipping: No Tavily or Brave API keys set in .env")
        return

    print(f"\nSearching with date filters (using 'days' parameter):")

    if has_tavily:
        if DRY_RUN:
            dry_run_search("Python asyncio tutorial", provider="tavily", days=num_days)
        else:
            result = await real_web_search("Python asyncio tutorial", provider="tavily", num_results=3, days=num_days)
            print_results(result)

    if has_brave:
        if DRY_RUN:
            dry_run_search("Python asyncio tutorial", provider="brave", days=num_days)
        else:
            result = await real_web_search("Python asyncio tutorial", provider="brave", num_results=3, days=num_days)
            print_results(result)


async def example_date_filtering_options():
    """Example 6: Demonstrate all date filtering options.

    Shows the different freshness period values and their Brave equivalents:
    - days=1 or 'pd': Past day (24 hours)
    - days=7 or 'pw': Past week
    - days=31 or 'pm': Past month
    - days=365 or 'py': Past year
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Date Filtering Options (Brave freshness periods)")
    print("=" * 60)

    if not os.getenv("BRAVE_API_KEY"):
        print("\nSkipping: BRAVE_API_KEY not set in .env")
        return

    date_options = [
        (1, "pd", "Past day"),
        (7, "pw", "Past week"),
        (31, "pm", "Past month"),
        (365, "py", "Past year"),
    ]

    print(f"\nSearching with various date filters:")

    for days, code, label in date_options:
        if DRY_RUN:
            dry_run_search("Python tutorial", provider="brave", num_results=2, days=days)
        else:
            result = await real_web_search("Python tutorial", provider="brave", num_results=2, days=days)
            print(f"\n--- {label} (days={days}, freshness='{code}') ---")
            print_results(result)


async def example_offset_pagination():
    """Example 7: Offset pagination for brave/google providers."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Offset Pagination (brave/google)")
    print("=" * 60)

    has_brave = bool(os.getenv("BRAVE_API_KEY"))
    has_google = bool(os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_SEARCH_ENGINE_ID"))

    if not (has_brave or has_google):
        print("\nSkipping: No Brave or Google API keys set in .env")
        return

    print(f"\nSearching with offset pagination:")

    if has_brave:
        if DRY_RUN:
            dry_run_search("Python tutorial", provider="brave", num_results=3, offset=0)
            dry_run_search("Python tutorial", provider="brave", num_results=3, offset=3)
        else:
            result_p1 = await real_web_search("Python tutorial", provider="brave", num_results=3, offset=0)
            print(f"\n--- Brave Page 1 (offset=0) ---")
            print_results(result_p1)
            result_p2 = await real_web_search("Python tutorial", provider="brave", num_results=3, offset=3)
            print(f"\n--- Brave Page 2 (offset=3) ---")
            print_results(result_p2)

    if has_google:
        if DRY_RUN:
            dry_run_search("Python tutorial", provider="google", num_results=5, offset=0)
            dry_run_search("Python tutorial", provider="google", num_results=5, offset=5)
        else:
            result_p1 = await real_web_search("Python tutorial", provider="google", num_results=5, offset=0)
            print(f"\n--- Google Page 1 (offset=0) ---")
            print_results(result_p1)
            result_p2 = await real_web_search("Python tutorial", provider="google", num_results=5, offset=5)
            print(f"\n--- Google Page 2 (offset=5) ---")
            print_results(result_p2)


async def example_google_ignores_days():
    """Example 8: Demonstrate that Google ignores the days parameter silently."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Google Ignores 'days' Parameter (graceful degradation)")
    print("=" * 60)

    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
        print("\nSkipping: GOOGLE_API_KEY and/or GOOGLE_SEARCH_ENGINE_ID not set in .env")
        return

    # Google doesn't support date filtering, but the API accepts it gracefully
    print(f"\nNote: Google Custom Search API does not support date filtering.")
    print("The 'days' parameter is silently ignored for google provider.")

    if DRY_RUN:
        dry_run_search("Python asyncio tutorial", provider="google", num_results=5, days=7)
        return

    result = await real_web_search("Python asyncio tutorial", provider="google", num_results=5, days=7)
    # Note: result will NOT have days field since Google doesn't support it
    print_results(result)


async def example_error_handling():
    """Example 9: Error handling for unknown provider and empty query."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Error Handling (graceful degradation)")
    print("=" * 60)

    # Test with unknown provider - should return error in results
    print(f"\nTesting unknown provider:")

    if DRY_RUN:
        dry_run_search("test", provider="unknown_provider")
    else:
        result = await real_web_search("test", provider="unknown_provider")
        print_results(result)

    # Test with missing query - should return error
    print(f"\nTesting empty query:")

    if DRY_RUN:
        dry_run_search("")
    else:
        result = await real_web_search("")
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
        "miklium": [example_miklium],
        "tavily": [example_tavily],
        "brave": [example_brave],
        "google": [example_google],
    }

    # If no providers specified (None or empty list), test all available ones
    original_providers = providers.copy() if providers else []
    if not providers:
        providers = ["miklium", "tavily", "brave", "google"]

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
    print(f"# searchWeb Examples (providers: {', '.join(providers_to_test)})")
    print("#" * 60)

    # Run examples based on selected providers
    for provider in providers_to_test:
        if provider == "miklium":
            await example_miklium()
        elif provider == "tavily":
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

    # Error handling always relevant
    await example_error_handling()
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
  %(prog)s miklium          Run Miklium (free, no API key) examples
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