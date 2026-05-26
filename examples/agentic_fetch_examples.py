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

import asyncio
from typing import Dict, Any

from mcp_server.agentic import agentic_fetch


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


async def agentic_fetch_example_basic():
    """
    Basic example of using agentic fetch.
    
    Returns a sample result structure for documentation purposes.
    """
    return {
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


async def agentic_fetch_example_not_found():
    """
    Example of a failed agentic fetch result.
    
    Returns a sample failure structure for documentation purposes.
    """
    return {
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
        preview = content[:500] + "..." if len(content) > 500 else content
        print(preview)
    
    urls = result.get('urls_visited', [])
    if urls:
        print(f"\n\nURLs Visited ({len(urls)}):")
        print("-" * 40)
        for url_info in urls:
            title = url_info.get('title', 'No title')
            url = url_info.get('url', '')
            action = url_info.get('action', '')
            print(f"  • {title[:50]}")
            print(f"    URL: {url}")
            print(f"    Action: {action}")
    
    steps = result.get('steps_taken', [])
    if steps:
        print(f"\n\nSteps Taken ({len(steps)}):")
        print("-" * 40)
        for step in steps:
            step_num = step.get('step', '?')
            action = step.get('action', '')
            desc = str(step.get('description', ''))[:50]
            print(f"  Step {step_num}: [{action}] {desc}...")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import asyncio
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Summarize the most recent comma.ai blog post"
        print(f"Running agentic search: {prompt}")
        result = asyncio.run(run_agentic_search(prompt))
    else:
        print("Example: Successful agentic fetch")
        result = asyncio.run(agentic_fetch_example_basic())
        print_example_result(result)
        
        print("\n\nExample: Failed agentic fetch")
        result = asyncio.run(agentic_fetch_example_not_found())
        print_example_result(result)