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
        print("\n" + "#" * 60)
        print("# Running All Examples")
        print("#" * 60)

        from examples.web_fetch_examples import main as fetch_main
        from examples.web_search_examples import main as search_main
        from examples.web_summarize_examples import main as summarize_main

        fetch_main()
        response = input("\nPress Enter to continue to search examples (or 'q' to quit)... ")
        if response.lower() == 'q':
            return

        search_main()
        response = input("\nPress Enter to continue to summarize examples (or 'q' to quit)... ")
        if response.lower() == 'q':
            return

        summarize_main()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
