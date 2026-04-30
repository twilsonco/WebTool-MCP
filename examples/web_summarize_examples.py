#!/usr/bin/env python3
"""
web_summarize examples - Demonstrates usage of the web_summarize MCP tool.
This script imports and calls the actual implementations directly.
Loads API keys from .env in project root.
Requires OPENAI_COMPATIBLE_BASE_URL (e.g., OpenWebUI or Ollama).
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from dotenv import load_dotenv

load_dotenv(project_root / ".env")

# Import the actual implementations from server.py
from src.mcp_server.server import web_summarize as real_web_summarize
from src.mcp_server.server import BASE_URL, MODEL_NAME


async def example_single_url():
    """Example 1: Summarize a single URL with default prompt."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Single URL Summary")
    print("=" * 60)

    urls = ["https://example.com"]

    print(f"\nSummarizing {urls[0]}...")
    result = await real_web_summarize(urls, max_words_per_url=500)

    if "summaries" in result:
        for url, data in result["summaries"].items():
            print(f"\nURL: {url}")
            print("-" * 40)
            if "summary" in data:
                print(data["summary"])
            elif "error" in data:
                print(f"Error: {data['error']}")


async def example_multiple_urls_reduce():
    """Example 2: Summarize multiple URLs with synthesis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multiple URLs with Synthesis (reduce=True)")
    print("=" * 60)

    urls = [
        "https://example.com",
        "https://httpbin.org/html",
    ]

    print(f"\nSummarizing {len(urls)} URLs and synthesizing...")
    result = await real_web_summarize(
        urls,
        reduce=True,
        max_words_per_url=300
    )

    if "summaries" in result:
        print("\nIndividual Summaries:")
        print("-" * 40)
        for url, data in result["summaries"].items():
            preview = data.get("summary", data.get("error", "unknown"))[:150]
            print(f"\n[{url}]")
            print(preview + "..." if len(data.get("summary", "")) > 150 else preview)

    if "combined" in result:
        print("\n" + "=" * 40)
        print("Combined/Synthesized Summary:")
        print("-" * 40)
        print(result["combined"].get("summary", "N/A"))


async def example_custom_summary_prompt():
    """Example 3: Custom prompt for technical focus."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Summary Prompt (technical focus)")
    print("=" * 60)

    urls = ["https://httpbin.org/html"]

    custom_prompt = """
Focus on extracting:
1. Key programming languages or frameworks mentioned
2. Code examples or APIs described
3. Performance characteristics
4. Setup/installation instructions
Format as: ## Overview\n## Technical Details\n## Examples
""".strip()

    print(f"\nSummarizing with custom technical prompt...")
    result = await real_web_summarize(
        urls,
        summary_prompt=custom_prompt,
        max_words_per_url=400
    )

    if "summaries" in result:
        for url, data in result["summaries"].items():
            print(f"\nURL: {url}")
            print("-" * 40)
            if "summary" in data:
                print(data["summary"])


async def example_custom_reduction_prompt():
    """Example 4: Custom prompt for synthesis/comparison."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Custom Reduction Prompt (comparison)")
    print("=" * 60)

    urls = [
        "https://example.com",
        "https://httpbin.org/html",
    ]

    reduction_prompt = """
Compare and contrast these document summaries. Identify:
- Common themes or overlapping information
- Unique contributions from each source
- Any conflicting claims or approaches
Format as: ## Shared Content\n## Unique Points\n## Conclusion
""".strip()

    print(f"\nSummarizing with custom comparison prompt...")
    result = await real_web_summarize(
        urls,
        reduce=True,
        reduction_prompt=reduction_prompt,
        max_words_per_url=200  # Smaller for faster demo
    )

    if "combined" in result:
        print(f"\nCombined Analysis:")
        print("-" * 40)
        print(result["combined"].get("summary", "N/A"))


async def example_config_check():
    """Example 5: Check LLM configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: LLM Configuration Status")
    print("=" * 60)

    print(f"\nLLM Endpoint: {BASE_URL}")
    print(f"Model Name:   {MODEL_NAME}")
    print(f"API Key Set:  {'Yes' if os.getenv('OPENAI_API_KEY') else 'No (local endpoint assumed)'}")

    # Quick connectivity check
    import httpx
    try:
        base = BASE_URL.rsplit('/', 1)[0]
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base}/models")
            if resp.is_success:
                print("Connection:   OK - LLM endpoint reachable")
            else:
                print(f"Connection:   Warning - status {resp.status_code}")
    except Exception as e:
        print(f"Connection:   Error - cannot reach endpoint ({e})")


async def main():
    print("\n" + "#" * 60)
    print("# web_summarize Examples (using real implementation)")
    print(f"# LLM Endpoint: {BASE_URL}")
    print(f"# Model: {MODEL_NAME}")
    print("#" * 60)

    await example_config_check()
    await example_single_url()

    # These take longer due to multiple LLM calls - uncomment as needed:
    # await example_multiple_urls_reduce()
    # await example_custom_summary_prompt()
    # await example_custom_reduction_prompt()

    print("\n" + "#" * 60)
    print("# Done!")
    print("#" * 60)


if __name__ == "__main__":
    asyncio.run(main())
