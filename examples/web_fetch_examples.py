#!/usr/bin/env python3
"""
web_fetch examples - Demonstrates usage of the web_fetch MCP tool.
This script imports and calls the actual implementation directly.
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from dotenv import load_dotenv

# Load .env for any provider API keys (not needed for fetch, but good practice)
load_dotenv(project_root / ".env")

# Import the actual implementation functions from server.py
from src.mcp_server.server import web_fetch as real_web_fetch


async def example_basic():
    """Example 1: Basic fetch - just get content as markdown."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Fetch")
    print("=" * 60)

    urls = ["https://example.com", "https://httpbin.org/html"]

    results = await real_web_fetch(urls)

    for url, content in results.items():
        preview = content[:300] if len(content) > 300 else content
        print(f"\n[{url}]")
        print(preview + ("..." if len(content) > 300 else ""))


async def example_with_truncation():
    """Example 2: Fetch with word-level truncation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Word Truncation")
    print("=" * 60)

    # Fetch some quotes
    urls = ["https://quotes.toscrape.com"]

    results = await real_web_fetch(urls, num_words=50)

    for url, content in results.items():
        word_count = len(content.split())
        print(f"\n[{url}]")
        print(f"Truncated to {word_count} words:")
        print("-" * 40)
        print(content[:500] + "..." if len(content) > 500 else content)


async def example_with_regex():
    """Example 3: Fetch with regex filtering."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Regex Filtering")
    print("=" * 60)

    # Get a page and filter for specific patterns
    urls = ["https://httpbin.org/html"]

    # Try different patterns - some may match, some not
    results_no_match = await real_web_fetch(urls, regex=r"NONEXISTENT_PATTERN_XYZ", regex_padding=20)
    print(f"\nPattern 'NONEXISTENT_PATTERN_XYZ' on httpbin.org:")
    print(f"  Result: {results_no_match['https://httpbin.org/html']}")

    # Pattern that should match like "the" or "is"
    results_matching = await real_web_fetch(
        urls,
        regex=r"the|is",
        regex_padding=30
    )
    print(f"\nPattern 'the|is' on httpbin.org:")
    content = results_matching['https://httpbin.org/html'][:500]
    print(content + ("..." if len(results_matching['https://httpbin.org/html']) > 500 else ""))


async def example_start_offset():
    """Example 4: Fetch starting from a specific word offset."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Word Offset (start_word)")
    print("=" * 60)

    # Get the same page twice with different offsets to show pagination effect
    url = "https://httpbin.org/html"

    first_100 = await real_web_fetch([url], num_words=20, start_word=0)
    second_100 = await real_web_fetch([url], num_words=20, start_word=50)

    print(f"\n[{url}]")
    print("-" * 40)
    print(f"Words 1-20: {first_100[url]}")
    print()
    print(f"Words 51-70: {second_100[url]}")


async def main():
    print("\n" + "#" * 60)
    print("# web_fetch Examples (using real implementation)")
    print("#" * 60)

    await example_basic()
    await example_with_truncation()
    await example_with_regex()
    await example_start_offset()

    print("\n" + "#" * 60)
    print("# Done!")
    print("#" * 60)


if __name__ == "__main__":
    asyncio.run(main())
