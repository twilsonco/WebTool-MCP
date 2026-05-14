#!/usr/bin/env python3
"""
fetchWebContent examples - Demonstrates usage of the fetchWebContent MCP tool.
This script imports and calls the actual implementation directly.

The atomic API fetches one URL per call:
    result = await real_web_fetch("https://example.com", num_words=500)

Returns a dict with 'url' and 'content', or 'error' on failure.
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
from src.mcp_server.server import fetchWebContent as real_web_fetch


def print_result(result: dict):
    """Display a fetch result."""
    if "error" in result:
        print(f"  Error: {result['error']}")
        return
    content = result.get("content", "")
    preview = content[:300] if len(content) > 300 else content
    print(f"\n[{result['url']}]")
    print(preview + ("..." if len(content) > 300 else ""))


async def example_basic():
    """Example 1: Basic fetch - just get content as markdown."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Fetch")
    print("=" * 60)

    result = await real_web_fetch("https://example.com")
    print_result(result)


async def example_with_truncation():
    """Example 2: Fetch with word-level truncation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Word Truncation")
    print("=" * 60)

    result = await real_web_fetch("https://quotes.toscrape.com", num_words=50)

    if "content" in result:
        word_count = len(result["content"].split())
        print(f"\n[{result['url']}]")
        print(f"Truncated to {word_count} words:")
        print("-" * 40)
        content = result["content"]
        print(content[:500] + "..." if len(content) > 500 else content)


async def example_with_regex():
    """Example 3: Fetch with regex filtering."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Regex Filtering")
    print("=" * 60)

    # Try different patterns - some may match, some not
    result_no_match = await real_web_fetch("https://httpbin.org/html", regex=r"NONEXISTENT_PATTERN_XYZ", regex_padding=20)
    print(f"\nPattern 'NONEXISTENT_PATTERN_XYZ' on httpbin.org:")
    if "content" in result_no_match:
        print(f"  Result: {result_no_match['content']}")
    else:
        print_result(result_no_match)

    # Pattern that should match like "the" or "is"
    result_matching = await real_web_fetch(
        "https://httpbin.org/html",
        regex=r"the|is",
        regex_padding=30
    )
    print(f"\nPattern 'the|is' on httpbin.org:")
    if "content" in result_matching:
        content = result_matching["content"][:500]
        print(content + ("..." if len(result_matching["content"]) > 500 else ""))
    else:
        print_result(result_matching)


async def example_start_offset():
    """Example 4: Fetch starting from a specific word offset."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Word Offset (start_word)")
    print("=" * 60)

    url = "https://httpbin.org/html"

    first_100 = await real_web_fetch(url, num_words=20, start_word=0)
    second_100 = await real_web_fetch(url, num_words=20, start_word=50)

    print(f"\n[{url}]")
    print("-" * 40)
    if "content" in first_100:
        print(f"Words 1-20: {first_100['content']}")
    if "content" in second_100:
        print(f"\nWords 51-70: {second_100['content']}")


async def main():
    print("\n" + "#" * 60)
    print("# fetchWebContent Examples (using real implementation)")
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