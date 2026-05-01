#!/usr/bin/env python3
"""
web_summarize examples - Demonstrates usage of the web_summarize MCP tool.
This script imports and calls the actual implementations directly.
Loads API keys from .env in project root.
Requires LLM_PROVIDER_1_* variables to be configured (multi-provider support).
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


async def example_single_url():
    """Example 1: Summarize a single URL with default prompt."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Single URL Summary")
    print("=" * 60)

    urls = ["https://blog.comma.ai/011release/"]

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
    Example 6: Multi-provider LLM failover demonstration.
    
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
    
    How failover works:
        1. The system tries provider 1 (primary)
        2. If it fails (connection error, API error, timeout), it logs the error
           and moves to provider 2
        3. This continues through all configured providers
        4. If ALL providers fail, LLMAllProvidersFailedError is raised with
           details about what went wrong with each
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Multi-Provider Failover Configuration")
    print("=" * 60)

    # Import the LLM manager to check provider configuration
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
    print("# web_summarize Examples (using real implementation)")
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

    # These take longer due to multiple LLM calls - uncomment as needed:
    # await example_multiple_urls_reduce()
    # await example_custom_summary_prompt()
    # await example_custom_reduction_prompt()

    print("\n" + "#" * 60)
    print("# Done!")
    print("#" * 60)


if __name__ == "__main__":
    asyncio.run(main())
