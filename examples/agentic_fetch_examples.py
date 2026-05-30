"""
Examples demonstrating agentic AI fetch mode.

This module shows how to use the agentic fetch endpoint that autonomously
searches and browses the web using AI decision-making.

Prerequisites:
- Set up LLM_PROVIDER_1_* environment variables in .env
- Optionally install browser-use for AI-controlled browser actions:
  pip install browser-use

Example .env configuration:
```
LLM_PROVIDER_1_NAME=openai
LLM_PROVIDER_1_BASE_URL=https://api.openai.com/v1
LLM_PROVIDER_1_API_KEY=sk-...
LLM_PROVIDER_1_MODEL=gpt-4o
```

Usage:
```python
import asyncio
from mcp_server.agentic import agentic_fetch

async def main():
    result = await agentic_fetch(
        prompt="Find the most recent Federal Reserve meeting minutes",
        max_steps=10
    )
    
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Content found at: {result.get('url')}")
        print(result['content'][:500])
    else:
        print(f"Error: {result.get('error_message')}")
    
    print("\nURLs visited:")
    for url_info in result.get('urls_visited', []):
        print(f"  - {url_info['title']}: {url_info['url']}")
    
    print("\nSteps taken:")
    for step in result.get('steps_taken', []):
        print(f"  Step {step['step']}: {step['action']} - {step.get('description', '')[:50]}")

asyncio.run(main())
```

HTTP API usage:
```bash
curl -X POST http://localhost:8000/agenticFetch \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Find the most recent Federal Reserve meeting minutes", "max_steps": 10}'
```

Example response:
{
    "success": true,
    "content": "# Federal Reserve Meeting Minutes\n\nThe Federal Open Market Committee...",
    "url": "https://www.federalreserve.gov/monetarypolicy/fomcminutes/2024xxxx.htm",
    "urls_visited": [
        {
            "url": "https://www.google.com/search?q=Federal+Reserve+meeting+minutes",
            "title": "Federal Reserve Meeting Minutes 2024 - Official Site",
            "action": "Search result at step 1"
        },
        {
            "url": "https://www.federalreserve.gov/monetarypolicy/fomcminutes/2024xxxx.htm",
            "title": "Federal Reserve FOMC Minutes",
            "action": "Navigated at step 3"
        }
    ],
    "steps_taken": [
        {
            "step": 1,
            "action": "search",
            "description": "Search for Federal Reserve meeting minutes 2024",
            "url": null
        },
        {
            "step": 2,
            "action": "navigate",
            "description": "Fetching the official Federal Reserve minutes page",
            "url": "https://www.federalreserve.gov/..."
        },
        {
            "step": 3,
            "action": "done",
            "description": "Successfully extracted meeting minutes content"
        }
    ]
}
"""

import argparse
import asyncio
import sys
from typing import Dict, Any

from pathlib import Path
from dotenv import load_dotenv
# Load .env for LLM provider configuration
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env", override=True)

from mcp_server.agentic import agentic_fetch


def parse_example_selection(selection: str | None) -> set[int]:
    """Parse comma-separated/range example selection like '1-3,5' into a set of integers.

    Args:
        selection: Comma-separated list with optional ranges, e.g., '1-3,5' or '1,2,3'

    Returns:
        Set of example numbers (1-indexed)

    Examples:
        '1-3,5' -> {1, 2, 3, 5}
        '1,2,3' -> {1, 2, 3}
        '5'     -> {5}
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
        description="agentic_fetch examples demonstrating various features and use cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run all examples
  %(prog)s 1                    Run only example 1 (basic agentic fetch)
  %(prog)s 1-3                  Run examples 1, 2, and 3
  %(prog)s 1,3                  Run examples 1 and 3

Example functions (numbered for selection):
  1. example_1_basic()          - Basic agentic fetch with sample result
  2. example_2_not_found()      - Example of a failed agentic fetch result
  3. example_3_streaming()      - Agentic fetch with streaming callback
        """
    )

    parser.add_argument(
        "examples",
        nargs="?",
        default=None,
        metavar="EXAMPLES",
        help="Comma-separated list or range of example numbers to run, e.g., '1-3,5'. "
             "Omit or leave empty to run all examples. Example: '1,2' or '1-3'."
    )

    return parser.parse_args()


async def run_agentic_search(prompt: str, max_steps: int = 15) -> Dict[str, Any]:
    """
    Actually calls the agentic fetch API with a given prompt.
    
    Args:
        prompt: The natural language search query
        max_steps: Maximum number of agentic steps to take
        
    Returns:
        The result from the agentic fetch API
    """
    result = await agentic_fetch(prompt=prompt, max_steps=max_steps)
    
    print_example_result(result)
    return result


async def example_1_basic():
    """
    Example 1: Basic agentic fetch.

    Returns a sample result structure for documentation purposes.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Agentic Fetch (Sample Result)")
    print("=" * 60)

    result = {
        "success": True,
        "content": "# Federal Reserve Meeting Minutes\n\nThe Federal Open Market Committee (FOMC) held its meeting on... [content truncated]",
        "url": "https://www.federalreserve.gov/monetarypolicy/fomcminutes/2024xxxx.htm",
        "urls_visited": [
            {
                "url": "https://www.google.com/search?q=Federal+Reserve+meeting+minutes",
                "title": "Federal Reserve Meeting Minutes 2024 - Official Site",
                "action": "Search result at step 1"
            },
            {
                "url": "https://www.federalreserve.gov/monetarypolicy/fomcminutes/2024xxxx.htm",
                "title": "Federal Reserve FOMC Minutes",
                "action": "Navigated at step 3"
            }
        ],
        "steps_taken": [
            {
                "step": 1,
                "action": "search",
                "description": "Search for Federal Reserve meeting minutes 2024",
                "result_preview": None
            },
            {
                "step": 2,
                "action": "navigate",
                "description": "Fetching the official Federal Reserve minutes page",
                "result_preview": None
            },
            {
                "step": 3,
                "action": "done",
                "description": "Successfully extracted meeting minutes content"
            }
        ],
        "error_message": None
    }

    print_example_result(result)
    return result


async def example_2_not_found():
    """
    Example 2: Failed agentic fetch result.

    Returns a sample failure structure for documentation purposes.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Failed Agentic Fetch (Sample Result)")
    print("=" * 60)

    result = {
        "success": False,
        "content": None,
        "url": None,
        "urls_visited": [
            {
                "url": "https://www.google.com/search?q=nonexistent+topic",
                "title": "",
                "action": "Search result at step 1"
            },
            {
                "url": "https://example.com/related-page",
                "title": "",
                "action": "Navigated at step 2"
            }
        ],
        "steps_taken": [
            {
                "step": 1,
                "action": "search",
                "description": "Search for nonexistent topic",
                "result_preview": None
            },
            {
                "step": 2,
                "action": "navigate",
                "description": "Navigating to promising result",
                "result_preview": None
            },
            {
                "step": 3,
                "action": "done",
                "description": "Could not find requested content after trying multiple sources"
            }
        ],
        "error_message": "Could not find requested content after 10 steps. See urls_visited for URLs that were attempted."
    }

    print_example_result(result)
    return result


async def example_3_streaming():
    """
    Example 3: Agentic fetch with streaming callback.

    Demonstrates the stream_callback parameter which receives progress
    updates after each agent step completes.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Streaming Callback")
    print("=" * 60)

    async def stream_callback(step_num: int, action: str, description: str, result: str | None) -> None:
        print(f"  [Step {step_num}] {action}: {description}")
        if result:
            print(f"           -> {result[:100]}...")

    prompt = "What is the latest on updated Apple TV hardware?"

    print(f"\nPrompt: {prompt}")
    print("\nStreaming progress:")
    print("-" * 40)

    result = await agentic_fetch(
        prompt=prompt,
        max_steps=10,
        stream_callback=stream_callback
    )

    print("\n" + "-" * 40)
    print("Final result:")
    print_example_result(result)

    return result


def print_example_result(result: Dict[str, Any]) -> None:
    """Print an agentic fetch result in a readable format."""
    print("=" * 60)
    print("AGENTIC FETCH RESULT")
    print("=" * 60)
    
    print(f"\nSuccess: {result.get('success', False)}")
    
    if result.get('error_message'):
        print(f"Error: {result['error_message']}")
    
    if result.get('url'):
        print(f"\nPrimary URL: {result['url']}")
    
    if result.get('content'):
        content = result['content']
        print(f"\nContent ({len(content)} chars):")
        print("-" * 40)
        preview = content[:2000] + "..." if len(content) > 2000 else content
        print(preview)
    
    urls = result.get('urls_visited', [])
    if urls:
        print(f"\n\nURLs Visited ({len(urls)}):")
        print("-" * 40)
        for url_info in urls:
            title = url_info.get('title', 'No title')
            url = url_info.get('url', '')
            action = url_info.get('action', '')
            print(f"  • {title[:100]}")
            print(f"    URL: {url}")
            print(f"    Action: {action}")
    
    steps = result.get('steps_taken', [])
    if steps:
        print(f"\n\nSteps Taken ({len(steps)}):")
        print("-" * 40)
        for step in steps:
            step_num = step.get('step', '?')
            action = step.get('action', '')
            desc = str(step.get('description', ''))[:500]
            print(f"  Step {step_num}: [{action}] {desc}...")
    
    print("\n" + "=" * 60)


async def main(selected_examples: set[int] | None = None):
    """Run agentic_fetch examples.

    Args:
        selected_examples: Set of example numbers to run. If None or empty, run all.
    """
    all_examples = [
        (1, "example_1_basic", example_1_basic),
        (2, "example_2_not_found", example_2_not_found),
        (3, "example_3_streaming", example_3_streaming),
    ]

    if selected_examples:
        examples_to_run = [(num, name, func) for num, name, func in all_examples if num in selected_examples]
    else:
        examples_to_run = all_examples

    print("\n" + "#" * 60)
    print("# agentic_fetch Examples")
    if selected_examples:
        example_nums = sorted(selected_examples)
        print(f"# Running examples: {example_nums}")
    else:
        print("# Running all 3 examples")
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