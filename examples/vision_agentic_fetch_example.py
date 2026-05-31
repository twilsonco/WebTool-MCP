"""
Examples demonstrating vision-enabled agentic AI fetch mode.

This module shows two approaches:
1. Full vision agentic fetch - uses the agent with screenshot capture and vision analysis
2. Screenshot-only example - captures a screenshot for external vision processing

Prerequisites:
- Set up LLM_PROVIDER_1_* environment variables in .env with a vision-capable model (e.g., gpt-4o)
- Install Playwright: pip install playwright && python -m playwright install chromium

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
    # Vision-enabled agentic fetch
    result = await agentic_fetch(
        prompt="Describe the layout and main elements of Hacker News",
        vision_enabled=True,
        max_steps=5
    )
    print(f"Success: {result['success']}")
    if result.get('description'):
        print(result['description'])

asyncio.run(main())
```

HTTP API usage:
```bash
curl -X POST http://localhost:8000/agenticFetch \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Describe the layout of this page", "max_steps": 5}'
```
"""

import argparse
import asyncio
import sys

from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env", override=True)

from mcp_server.agentic import agentic_fetch
from mcp_server.extraction.pipeline import ContentExtractionPipeline


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Vision-enabled agentic fetch examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                  Run all examples
  %(prog)s vision           Run the vision agentic fetch example (default)
  %(prog)s screenshot       Run the standalone screenshot capture example
        """
    )

    parser.add_argument(
        "example",
        nargs="?",
        default=None,
        choices=["vision", "screenshot"],
        help="Which example to run: 'vision' for full vision agentic fetch, "
             "'screenshot' for standalone screenshot capture"
    )

    return parser.parse_args()


async def example_vision_agentic_fetch():
    """
    Example: Vision-enabled agentic fetch.

    Uses the full agentic fetch with vision_enabled=True. The agent can
    decide to capture and analyze screenshots as part of its reasoning.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE: Vision Agentic Fetch")
    print("=" * 60)

    prompt = "Describe the layout and main elements of the Greyhound Bus Museum website. Describe the colors, images, and structure you see in the screenshots."
    print(f"\nPrompt: {prompt}")

    result = await agentic_fetch(
        prompt=prompt,
        vision_enabled=True,
        max_steps=10
    )

    print_example_result(result)
    return result


async def example_screenshot_capture():
    """
    Example: Standalone screenshot capture.

    Demonstrates capturing a screenshot directly using ContentExtractionPipeline,
    for cases where you want to handle vision analysis separately.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE: Standalone Screenshot Capture")
    print("=" * 60)

    url = "https://news.ycombinator.com"
    print(f"\nURL: {url}")

    pipeline = ContentExtractionPipeline()
    screenshot_base64 = await pipeline.capture_screenshot(url)

    if screenshot_base64 is None:
        print("\nFailed to capture screenshot. Is Playwright installed?")
        return {"success": False, "error": "Screenshot capture failed"}

    print(f"\nScreenshot captured: {len(screenshot_base64)} chars base64")
    print("(In a real use case, this would be sent to a vision-capable LLM)")

    return {
        "success": True,
        "url": url,
        "screenshot_length": len(screenshot_base64)
    }


def print_example_result(result):
    """Print an agentic fetch result in a readable format."""
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    success = result.get('success', False)
    print(f"\nSuccess: {success}")

    if not success:
        error = result.get('error_message') or result.get('error')
        if error:
            print(f"Error: {error}")
        return

    # Vision result fields
    if result.get('description'):
        print("\nDescription:")
        print("-" * 40)
        print(result['description'])

    if result.get('url'):
        print(f"\nPrimary URL: {result['url']}")

    if result.get('content'):
        content = result['content']
        print(f"\nContent ({len(content)} chars):")
        preview = content[:1000] + "..." if len(content) > 1000 else content
        print("-" * 40)
        print(preview)

    urls = result.get('urls_visited', [])
    if urls:
        print(f"\n\nURLs Visited ({len(urls)}):")
        for url_info in urls:
            title = url_info.get('title', 'No title')[:60]
            action = url_info.get('action', '')
            print(f"  • [{action}] {title}")

    steps = result.get('steps_taken', [])
    if steps:
        print(f"\n\nSteps Taken ({len(steps)}):")
        for step in steps:
            action = step.get('action', '')
            desc = str(step.get('description', ''))[:60]
            print(f"  [{action}] {desc}")


async def main(example_name: str | None = None):
    """Run the specified example or all examples."""
    if example_name is None:
        await example_vision_agentic_fetch()
    elif example_name == "vision":
        await example_vision_agentic_fetch()
    elif example_name == "screenshot":
        await example_screenshot_capture()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(example_name=args.example))