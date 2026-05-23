#!/usr/bin/env python3
"""
Test LLM Providers - Tests all configured LLM providers by sending a simple message.

This script iterates through all LLM providers configured via environment variables
(LLM_PROVIDER_1_*, LLM_PROVIDER_2_*, etc.) and sends a test message to each one,
reporting success or failure for each provider.

Environment variables for each provider (replace N with 1, 2, etc.):
    LLM_PROVIDER_N_NAME     - Unique name for the provider
    LLM_PROVIDER_N_BASE_URL - OpenAI-compatible API base URL
    LLM_PROVIDER_N_API_KEY  - Authentication token (optional)
    LLM_PROVIDER_N_MODEL    - Model identifier for chat completions

Example:
    LLM_PROVIDER_1_NAME=ollama
    LLM_PROVIDER_1_BASE_URL=http://localhost:11434/v1
    LLM_PROVIDER_1_API_KEY=
    LLM_PROVIDER_1_MODEL=llama3.2

Usage:
    python examples/test_llm_providers.py              Test all providers
    python examples/test_llm_providers.py 1            Test only provider 1
    python examples/test_llm_providers.py 1-3          Test providers 1, 2, and 3
    python examples/test_llm_providers.py 1,3          Test providers 1 and 3
    python examples/test_llm_providers.py 1,2-4        Test providers 1, 2, 3, and 4
    python examples/test_llm_providers.py --dry-run    Show what would be tested
"""
import argparse
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load .env for LLM provider configuration
load_dotenv(project_root / ".env")

from src.mcp_server.llm.manager import LLMManager
from src.mcp_server.llm.exceptions import LLMProviderError

TEST_MESSAGE = "Hello, please respond with exactly 'Test successful' if you can understand this message."
TEST_SYSTEM_PROMPT = "You are a helpful assistant. Keep responses brief and concise."


async def test_provider(provider) -> tuple[bool, str]:
    """
    Test a single LLM provider with a simple message.
    
    Args:
        provider: An LLMProvider instance to test.
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # First check if provider is available
        is_avail = await provider.is_available(timeout=5.0)
        if not is_avail:
            return False, "Provider endpoint is not available or not responding"
        
        # Send test message
        response = await provider.complete(TEST_MESSAGE, TEST_SYSTEM_PROMPT)
        
        # Check if response is non-empty
        if not response or not response.strip():
            return False, "Provider returned empty response"
        
        return True, f"Response received ({len(response)} chars): {response[:100]}..."
        
    except LLMProviderError as e:
        return False, f"LLMProviderError: {e}"
    except Exception as e:
        return False, f"Unexpected error: {type(e).__name__}: {e}"


def parse_example_selection(selection: str | None) -> set[int] | None:
    """
    Parse comma-separated/range example selection string.
    
    Supports formats like:
        '1-3,5,7'  -> {1, 2, 3, 5, 7}
        '1,2,3'    -> {1, 2, 3}
        '1-4'      -> {1, 2, 3, 4}
        '5'        -> {5}
    
    Args:
        selection: Comma-separated list or range of numbers, e.g. '1-3,5,7'
        
    Returns:
        Set of selected indices, or None if selection is empty/None
    """
    if not selection:
        return None
    
    selected: set[int] = set()
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
        description="Test configured LLM providers by sending a simple message.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Test all configured providers
  %(prog)s 1            Test only provider 1
  %(prog)s 1-3          Test providers 1, 2, and 3
  %(prog)s 1,3          Test providers 1 and 3
  %(prog)s --dry-run    Show what would be tested without making API calls
        """
    )
    
    parser.add_argument(
        "providers",
        nargs="?",
        default=None,
        metavar="PROVIDERS",
        help="Comma-separated list or range of provider numbers to test, e.g., '1-3,5,7'. "
             "Omit or leave empty to test all providers. Example: '1,2,3' or '1-4'."
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be tested without making API calls"
    )
    
    return parser.parse_args()


async def main(selected_providers: set[int] | None = None, dry_run: bool = False):
    """Main entry point - test configured LLM providers.
    
    Args:
        selected_providers: Set of provider numbers to test. If None, test all.
        dry_run: If True, only show what would be tested without making API calls.
    """
    print("=" * 60)
    print("LLM Provider Test Script")
    print("=" * 60)
    
    # Create LLM manager which loads all providers from environment
    manager = LLMManager()
    providers = manager.providers
    
    if not providers:
        print("\nNo LLM providers configured!")
        print("\nPlease set the following environment variables:")
        print("  LLM_PROVIDER_1_NAME=provider_name")
        print("  LLM_PROVIDER_1_BASE_URL=http://localhost:11434/v1")
        print("  LLM_PROVIDER_1_API_KEY=optional_api_key")
        print("  LLM_PROVIDER_1_MODEL=model_name")
        print("\nAdd more providers with LLM_PROVIDER_2_*, LLM_PROVIDER_3_*, etc.")
        sys.exit(1)
    
    print(f"\nFound {len(providers)} configured provider(s)")
    
    # Filter to selected providers if specified
    if selected_providers:
        max_provider_num = len(providers)
        invalid = [p for p in selected_providers if p < 1 or p > max_provider_num]
        if invalid:
            print(f"\nError: Invalid provider number(s): {invalid}")
            print(f"Valid provider numbers are 1-{max_provider_num}")
            sys.exit(1)
        providers_to_test = [(i, providers[i-1]) for i in sorted(selected_providers)]
        print(f"Testing {len(providers_to_test)} selected provider(s): {sorted(selected_providers)}")
    else:
        providers_to_test = list(enumerate(providers, 1))
        print(f"Testing all {len(providers)} provider(s)")
    
    print("-" * 60)
    
    results = []
    
    for i, provider in providers_to_test:
        config = provider.config
        print(f"\n[{i}] Testing: {config.name}")
        print(f"    Base URL: {config.base_url}")
        print(f"    Model: {config.model}")
        print(f"    Test message: {TEST_MESSAGE}")
        
        if dry_run:
            print(f"    [DRY RUN] Would test provider and wait for response")
            results.append((config.name, True, "Dry run - would test"))
            continue
        
        success, message = await test_provider(provider)
        results.append((config.name, success, message))
        
        if success:
            print(f"    ✓ SUCCESS: {message}")
        else:
            print(f"    ✗ FAILED: {message}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    success_count = sum(1 for _, success, _ in results if success)
    fail_count = len(results) - success_count
    
    print(f"\nTotal providers tested: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    
    if fail_count > 0:
        print("\nFailed providers:")
        for name, success, message in results:
            if not success:
                print(f"  - {name}: {message}")
        sys.exit(1)
    else:
        print("\nAll providers passed!")
        sys.exit(0)


if __name__ == "__main__":
    args = parse_args()
    selected = parse_example_selection(args.providers)
    asyncio.run(main(selected_providers=selected, dry_run=args.dry_run))