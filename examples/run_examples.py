#!/usr/bin/env python3
"""
Run example scripts demonstrating WebTool MCP server capabilities.
These examples import and call the actual implementation functions directly.

Usage:
    uv run python examples/run_examples.py [fetch|search|summarize|all]

Examples:
    uv run python examples/run_examples.py fetch     # Run fetchWebContent examples
    uv run python examples/run_examples.py search    # Run searchWeb examples
    uv run python examples/run_examples.py summarize # Run summarizeWebContent examples
    uv run python examples/run_examples.py all       # Run everything
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
examples_dir = Path(__file__).parent

sys.path.insert(0, str(project_root))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nNo command specified. Available examples:")
        print("  fetch     - fetchWebContent tool demonstrations")
        print("  search    - searchWeb (Tavily/Brave/Google) demos")
        print("  summarize - summarizeWebContent with LLM inference demos")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "fetch":
        from examples.web_fetch_examples import main as run_main
        run_main()

    elif command == "search":
        from examples.web_search_examples import main as run_main
        run_main()

    elif command == "summarize":
        from examples.web_summarize_examples import main as run_main
        run_main()

    elif command == "all":

        from examples.web_fetch_examples import main as fetch_main
        from examples.web_search_examples import main as search_main
        from examples.web_summarize_examples import main as summarize_main

        
        print("\n" + "#" * 60)
        print("# Running Fetch Examples")
        print("#" * 60)
        fetch_main()

        print("\n" + "#" * 60)
        print("# Running Search Examples")
        print("#" * 60)
        search_main()

        print("\n" + "#" * 60)
        print("# Running Summarize Examples")
        print("#" * 60)
        summarize_main()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
