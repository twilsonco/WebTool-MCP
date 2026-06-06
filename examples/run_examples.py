#!/usr/bin/env python3
"""
Run example scripts demonstrating WebTool MCP server capabilities.
These examples import and call the actual implementation functions directly.

Usage:
    uv run python examples/run_examples.py [fetch|search|all]

Examples:
    uv run python examples/run_examples.py fetch     # Run fetch examples (includes summarize via summarize=true)
    uv run python examples/run_examples.py search    # Run search examples
    uv run python examples/run_examples.py all       # Run everything
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
examples_dir = Path(__file__).parent

sys.path.insert(0, str(project_root))


async def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nNo command specified. Available examples:")
        print("  fetch     - fetch tool demonstrations (normal and summarize modes)")
        print("  search    - search (Tavily/Brave/Google) demos")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "fetch":
        from examples.fetch_web_content_examples import main as run_main
        await run_main()

    elif command == "search":
        from examples.search_web_examples import main as run_main
        await run_main()

    elif command == "all":

        from examples.fetch_web_content_examples import main as fetch_main
        from examples.search_web_examples import main as search_main


        print("\n" + "#" * 60)
        print("# Running Fetch Examples")
        print("#" * 60)
        await fetch_main()

        input("\nPress Enter to continue to Search examples...")

        print("\n" + "#" * 60)
        print("# Running Search Examples")
        print("#" * 60)
        await search_main()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
