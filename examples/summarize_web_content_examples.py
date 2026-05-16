#!/usr/bin/env python3
"""
summarizeWebContent examples - Demonstrates usage of the summarizeWebContent MCP tool.
This script imports and calls the actual implementations directly.
Loads API keys from .env in project root.
Requires LLM_PROVIDER_1_* variables to be configured (multi-provider support).

The atomic API summarizes one URL per call:
    result = await real_summarize_web_content("https://example.com", max_words_per_url=500)

Returns a dict with 'url' and 'summary', or 'error' on failure.
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
from src.mcp_server.server import summarize_web_content as real_summarize_web_content


async def example_single_url():
    """Example 1: Summarize a single URL with default prompt."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Single URL Summary")
    print("=" * 60)

    url = "https://blog.comma.ai/011release/"

    print(f"\nSummarizing {url}...")
    result = await real_summarize_web_content(url, max_words_per_url=500)

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
        max_words_per_url=400
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


async def main():
    from src.mcp_server.llm import LLMManager

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
    print("#" * 60)

    await example_config_check()
    # Uncomment to test multi-provider failover:
    # await example_multi_provider_failover()
    await example_single_url()

    # This takes longer due to LLM call - uncomment as needed:
    # await example_custom_summary_prompt()

    print("\n" + "#" * 60)
    print("# Done!")
    print("#" * 60)


if __name__ == "__main__":
    asyncio.run(main())