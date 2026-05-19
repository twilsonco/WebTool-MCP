#!/usr/bin/env python3
"""
summarizeWebContent examples - Demonstrates usage of the summarizeWebContent MCP tool.
This script imports and calls the actual implementations directly.
Loads API keys from .env in project root.
Requires LLM_PROVIDER_1_* variables to be configured (multi-provider support).

The atomic API summarizes one URL per call:
    result = await real_summarize_web_content("https://example.com", max_num_words=500)

Returns a dict with 'url' and 'summary', or 'error' on failure.
"""
import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from dotenv import load_dotenv

load_dotenv(project_root / ".env")

# Import the actual implementations from server.py
from src.mcp_server.server import summarize_web_content as real_summarize_web_content


async def example_single_url():
    """Example 1: Summarize a single URL with default prompt."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Single URL Summary")
    print("=" * 60)

    url = "https://blog.comma.ai/011release/"

    print(f"\nSummarizing {url}...")
    result = await real_summarize_web_content(url, max_num_words=500)

    if "summary" in result:
        print(f"\nURL: {result['url']}")
        print("-" * 40)
        print(result["summary"])
    elif "error" in result:
        print(f"\nError: {result['error']}")


async def example_custom_summary_prompt():
    """Example 2: Custom prompt for technical focus."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Summary Prompt (technical focus)")
    print("=" * 60)

    url = "https://httpbin.org/html"

    custom_prompt = """
Focus on extracting:
1. Key programming languages or frameworks mentioned
2. Code examples or APIs described
3. Performance characteristics
4. Setup/installation instructions
Format as: ## Overview\n## Technical Details\n## Examples
""".strip()

    print(f"\nSummarizing with custom technical prompt...")
    result = await real_summarize_web_content(
        url,
        summary_prompt=custom_prompt,
        max_num_words=400
    )

    if "summary" in result:
        print(f"\nURL: {result['url']}")
        print("-" * 40)
        print(result["summary"])
    elif "error" in result:
        print(f"\nError: {result['error']}")


async def example_config_check():
    """Example 3: Check LLM configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: LLM Configuration Status")
    print("=" * 60)

    from src.mcp_server.llm import LLMManager

    try:
        manager = LLMManager()
        providers = manager.providers

        if providers:
            provider = providers[0]  # Primary provider
            print(f"\nLLM Provider: {provider.name}")
            print(f"Base URL:     {provider.config.base_url}")
            print(f"Model:        {provider.config.model}")
            print(f"API Key Set:  {'Yes' if provider.config.api_key else 'No'}")
        else:
            print("\nNo LLM providers configured.")
    except Exception as e:
        print(f"Error checking configuration: {e}")


async def example_multi_provider_failover():
    """
    Example 4: Multi-provider LLM failover demonstration.

    This example shows how to configure multiple LLM providers with automatic
    failover. When the first provider fails, the system automatically tries
    the next provider in sequence.

    Configuration via environment variables:
        # Provider 1 (Primary - highest priority)
        LLM_PROVIDER_1_NAME=primary-ollama
        LLM_PROVIDER_1_BASE_URL=http://localhost:11434/v1
        LLM_PROVIDER_1_API_KEY=
        LLM_PROVIDER_1_MODEL=llama3.2

        # Provider 2 (Fallback)
        LLM_PROVIDER_2_NAME=secondary-ollama
        LLM_PROVIDER_2_BASE_URL=http://192.168.1.100:11434/v1
        LLM_PROVIDER_2_API_KEY=
        LLM_PROVIDER_2_MODEL=mistral

        # Provider 3 (Last resort - could be cloud API like OpenRouter)
        LLM_PROVIDER_3_NAME=cloud-backup
        LLM_PROVIDER_3_BASE_URL=https://openrouter.ai/api/v1
        LLM_PROVIDER_3_API_KEY=sk-or-v1-...
        LLM_PROVIDER_3_MODEL=anthropic/claude-3-haiku
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Multi-Provider Failover Configuration")
    print("=" * 60)

    from src.mcp_server.llm import LLMManager, LLMAllProvidersFailedError

    try:
        manager = LLMManager()
        providers = manager.providers

        print(f"\nConfigured {len(providers)} LLM provider(s):")
        for i, provider in enumerate(providers, 1):
            print(f"\n  Provider {i}: {provider.name}")
            print(f"    URL:   {provider.config.base_url}")
            print(f"    Model: {provider.config.model}")
            has_key = "Yes" if provider.config.api_key else "No"
            print(f"    API Key: {has_key}")

        # Demonstrate failover behavior with a test prompt
        print("\n" + "-" * 40)
        print("Testing LLM completion (with potential failover)...")

        try:
            result = await manager.complete(
                "Say 'Hello' if you receive this.",
                system_prompt="You are a helpful assistant. Keep responses very brief."
            )
            print(f"\nSuccess! Response: {result[:100]}...")
        except LLMAllProvidersFailedError as e:
            print(f"\nAll providers failed:\n  {e}")

    except Exception as e:
        print(f"Error initializing LLM manager: {e}")


def parse_example_selection(selection: str | None) -> set[int]:
    """Parse comma-separated/range example selection like '1-3,5,7' into a set of integers.

    Args:
        selection: Comma-separated list with optional ranges, e.g., '1-3,5,7' or '1,2,3'

    Returns:
        Set of example numbers (1-indexed)

    Examples:
        '1-3,5,7' -> {1, 2, 3, 5, 7}
        '1,2,3'   -> {1, 2, 3}
        '5'       -> {5}
    """
    if not selection:
        return set()

    selected = set()
    parts = selection.split(',')

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if '-' in part:
            range_parts = part.split('-')
            if len(range_parts) == 2:
                try:
                    start, end = int(range_parts[0]), int(range_parts[1])
                    if start <= end:
                        selected.update(range(start, end + 1))
                except ValueError:
                    pass
        else:
            try:
                selected.add(int(part))
            except ValueError:
                pass

    return selected


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="summarizeWebContent examples demonstrating various features and use cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run all 4 examples
  %(prog)s 1                    Run only example 1 (single URL summary)
  %(prog)s 1-3                  Run examples 1, 2, and 3
  %(prog)s 1,3                  Run examples 1 and 3

Example functions (numbered for selection):
  1. example_single_url()           - Summarize a single URL with default prompt
  2. example_custom_summary_prompt() - Custom prompt for technical focus
  3. example_config_check()         - Check LLM configuration status
  4. example_multi_provider_failover() - Multi-provider LLM failover demonstration
        """
    )

    parser.add_argument(
        "examples",
        nargs="?",
        default=None,
        metavar="EXAMPLES",
        help="Comma-separated list or range of example numbers to run, e.g., '1-3,5,7'. "
             "Omit or leave empty to run all examples. Example: '1,2,3' or '1-4'."
    )

    return parser.parse_args()


async def main(selected_examples: set[int] | None = None):
    """Run summarizeWebContent examples.

    Args:
        selected_examples: Set of example numbers to run. If None or empty, run all.
    """
    from src.mcp_server.llm import LLMManager

    # All available examples in execution order
    all_examples = [
        (1, "example_single_url", example_single_url),
        (2, "example_custom_summary_prompt", example_custom_summary_prompt),
        (3, "example_config_check", example_config_check),
        (4, "example_multi_provider_failover", example_multi_provider_failover),
    ]

    # Determine which examples to run
    if selected_examples:
        examples_to_run = [(num, name, func) for num, name, func in all_examples if num in selected_examples]
    else:
        examples_to_run = all_examples

    print("\n" + "#" * 60)
    print("# summarizeWebContent Examples (using real implementation)")
    try:
        manager = LLMManager()
        if manager.providers:
            p = manager.providers[0]
            print(f"# LLM Provider: {p.name}")
            print(f"# Model: {p.config.model}")
    except Exception:
        print("# LLM Provider: Not configured")
    if selected_examples:
        example_nums = sorted(selected_examples)
        print(f"# Running examples: {example_nums}")
    else:
        print("# Running all 4 examples")
    print("#" * 60)

    for example_num, example_name, example_func in examples_to_run:
        try:
            await example_func()
        except Exception as e:
            print(f"\nError in {example_name}: {e}")

    print("\n" + "#" * 60)
    print("# Done!")
    print("#" * 60)


if __name__ == "__main__":
    args = parse_args()
    selected = parse_example_selection(args.examples)
    asyncio.run(main(selected_examples=selected if selected else None))