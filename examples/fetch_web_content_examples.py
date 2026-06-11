#!/usr/bin/env python3
"""
fetch examples - Demonstrates usage of the fetch MCP tool.
This script imports and calls the actual implementation directly.

The atomic API fetches one URL per call:
    result = await real_fetch_web_content("https://example.com", num_words=500)

Returns a dict with 'url' and 'content', or 'error' on failure.

Extraction Pipeline (applied in order, best result wins):
0. Firecrawl   - AI-powered scraping (when USE_FIRECRAWL=true)
1. Playwright  - dynamic rendering for JS-heavy / SPA pages
2. Trafilatura - heuristic text-density extraction (fast, no JS)
3. Readability - Mozilla-style article extraction
4. Docling     - layout-aware parsing for PDFs, DOCX, images, etc.
5. BeautifulSoup - universal HTML fallback (always succeeds)
6. LLM refinement - optional semantic cleanup pass (use_llm_refinement=True)
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from dotenv import load_dotenv

# Load .env for any provider API keys (not needed for fetch, but good practice)
load_dotenv(project_root / ".env")

# Import the actual implementation functions from server.py
from src.mcp_server.server import fetch_web_content as real_fetch_web_content, _BINARY_DOC_EXTENSIONS
from src.mcp_server.extraction import get_firecrawl_client


async def _is_firecrawl_available() -> bool:
    """Check if Firecrawl is configured and available."""
    import os
    if os.getenv("USE_FIRECRAWL", "false").lower() != "true":
        return False
    try:
        client = get_firecrawl_client()
        if client is None:
            return False
        result = await client.scrape("https://example.com", timeout=5)
        return result is not None and result.word_count > 0
    except Exception:
        return False


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
        description="fetch examples demonstrating various features and use cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run all examples
  %(prog)s 1                    Run only example 1 (basic fetch)
  %(prog)s 1-3                  Run examples 1, 2, and 3
  %(prog)s 1,3,5                Run examples 1, 3, and 5
  %(prog)s 2-4,7                Run examples 2, 3, 4, and 7
    
Example functions (numbered for selection):
  1. example_basic()          - Basic fetch with include_links option
  2. example_with_truncation()- Fetch with word-level truncation (num_words)
  3. example_with_regex()     - Regex filtering on fetched content
  4. example_start_offset()   - Fetch starting from specific word offset
  5. example_binary_document_formats() - Binary document format info (Docling)
  6. example_pdf_fetch()      - Fetch all files in examples/files/ via Docling
  7. example_llm_refinement() - LLM refinement pass for semantic cleanup
  8. example_full_content_fetch() - Full content fetch of real-world URLs
  9. example_summary()        - Summarize content with LLM (summarize=True)
 10. example_firecrawl_scrape()     - Basic Firecrawl scrape (requires USE_FIRECRAWL=true)
 11. example_firecrawl_with_options() - Firecrawl with screenshot_full_page, use_clean_content
 12. example_firecrawl_map()        - Discover URLs via /map endpoint (requires USE_FIRECRAWL=true)
 13. example_firecrawl_batch_scrape() - Batch scrape multiple URLs (requires USE_FIRECRAWL=true)

Note: Examples 10-13 require Firecrawl to be running (USE_FIRECRAWL=true).
      Start Firecrawl via: ./start-firecrawl.sh or manually
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


def print_result(result: dict):
    """Display a fetch result."""
    if "error" in result:
        print(f"  Error: {result['error']}")
        return
    content = result.get("content", "")
    preview = content[:300] if len(content) > 300 else content
    print(f"\n[{result['url']}]")
    print(preview + ("..." if len(content) > 300 else ""))


async def example_basic():
    """Example 1: Basic fetch - just get content as markdown."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Fetch")
    print("=" * 60)

    result = await real_fetch_web_content("https://example.com", include_links=True)
    print_result(result)
    
    # Then without links
    print("\n" + "-" * 60)
    print("Without links:")
    result_no_links = await real_fetch_web_content("https://example.com", include_links=False)
    print_result(result_no_links)


async def example_with_truncation():
    """Example 2: Fetch with word-level truncation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Word Truncation")
    print("=" * 60)

    result = await real_fetch_web_content("https://quotes.toscrape.com", num_words=50)

    if "content" in result:
        word_count = len(result["content"].split())
        print(f"\n[{result['url']}]")
        print(f"Truncated to {word_count} words:")
        print("-" * 40)
        content = result["content"]
        print(content[:500] + "..." if len(content) > 500 else content)


async def example_with_regex():
    """Example 3: Fetch with regex filtering."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Regex Filtering")
    print("=" * 60)

    # Try different patterns - some may match, some not
    result_no_match = await real_fetch_web_content("https://httpbin.org/html", regex=r"NONEXISTENT_PATTERN_XYZ", regex_padding=20)
    print(f"\nPattern 'NONEXISTENT_PATTERN_XYZ' on httpbin.org:")
    if "content" in result_no_match:
        print(f"  Result: {result_no_match['content']}")
    else:
        print_result(result_no_match)

    # Pattern that should match like "the" or "is"
    result_matching = await real_fetch_web_content(
        "https://httpbin.org/html",
        regex=r"the|is",
        regex_padding=30
    )
    print(f"\nPattern 'the|is' on httpbin.org:")
    if "content" in result_matching:
        content = result_matching["content"][:500]
        print(content + ("..." if len(result_matching["content"]) > 500 else ""))
    else:
        print_result(result_matching)


async def example_start_offset():
    """Example 4: Fetch starting from a specific word offset."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Word Offset (start_word)")
    print("=" * 60)

    url = "https://httpbin.org/html"

    first_100 = await real_fetch_web_content(url, num_words=20, start_word=0)
    second_100 = await real_fetch_web_content(url, num_words=20, start_word=50)

    print(f"\n[{url}]")
    print("-" * 40)
    if "content" in first_100:
        print(f"Words 1-20: {first_100['content']}")
    if "content" in second_100:
        print(f"\nWords 51-70: {second_100['content']}")


async def example_binary_document_formats():
    """Example 5: Binary document formats routed directly to Docling.

    URLs whose extension matches a binary document type bypass the HTML
    pipeline and go straight to Docling for layout-aware parsing.
    Supported formats: PDF, DOCX, PPTX, XLSX, images (PNG, JPG, TIFF, BMP),
    and structured data files (CSV, JSON, XML).

    Note: The URLs below are placeholders. Replace with real document URLs to
    exercise the pipeline.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Binary Document Formats (Docling via pipeline)")
    print("=" * 60)

    # Example document URLs (replace with actual documents to test)
    binary_urls = [
        "https://example.com/document.pdf",
        "https://example.com/presentation.pptx",
        "https://example.com/spreadsheet.xlsx",
        "https://example.com/document.docx",
    ]

    print("\nBinary document extensions routed to Docling:")
    print(f"  {', '.join(sorted(_BINARY_DOC_EXTENSIONS))}")
    print("\nExample URLs that would use the Docling (binary) path:")
    for url in binary_urls:
        print(f"  - {url}")


async def example_pdf_fetch():
    """Example 6: Fetch and parse all files in examples/files/ via Docling.

    Iterates over all files in the examples/files/ directory, fetching each
    from GitHub raw URL. Runs real_fetch_web_content() twice per file:
    - Once with default settings (no LLM refinement)
    - Once with use_llm_refinement=True for content that supports it.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: File Fetch (Docling binary path)")
    print("=" * 60)

    # Files available in examples/files/
    files_dir = project_root / "examples" / "files"
    filenames = [
        "Banana Split Decoded.pdf",
        "creative resume.doc",
        "creative resume.docx",
        "Family Budget.csv",
        "Family Budget.xls",
        "Family Budget.xlsx",
        "Family Budget.xml",
        "file-sample_150kB.pdf",
        "Presentation.ppt",
        "Presentation.pptx",
    ]

    for filename in filenames:
        # GitHub raw URL for each file
        url = f"https://raw.githubusercontent.com/twilsonco/WebTool-MCP/main/examples/files/{filename}"

        # Fetch without LLM refinement
        print(f"\n{'=' * 60}")
        print(f"FILE: {filename}")
        print(f"LLM Refinement: DISABLED")
        print(f"{'=' * 60}")
        result = await real_fetch_web_content(url, num_words=20000, include_links=True)

        if "error" in result:
            print(f"\n  Error: {result['error']}")
        else:
            content = result.get("content", "")
            if content:
                print(f"\n[{result['url']}]")
                print("-" * 40)
                preview = content[:1000] if len(content) > 1000 else content
                print(preview + ("..." if len(content) > 1000 else ""))
            else:
                print("No content extracted")

        # Fetch with LLM refinement enabled
        print(f"\n{'=' * 60}")
        print(f"FILE: {filename}")
        print(f"LLM Refinement: ENABLED")
        print(f"{'=' * 60}")
        result_llm = await real_fetch_web_content(
            url, num_words=20000, include_links=True, use_llm_refinement=True
        )

        if "error" in result_llm:
            print(f"\n  Error: {result_llm['error']}")
        else:
            content = result_llm.get("content", "")
            if content:
                print(f"\n[{result_llm['url']}]")
                print("-" * 40)
                preview = content[:1000] if len(content) > 1000 else content
                print(preview + ("..." if len(content) > 1000 else ""))
            else:
                print("No content extracted")

        # Pause to avoid GitHub rate limiting
        await asyncio.sleep(3)


async def example_llm_refinement():
    """Example 7: Optional LLM refinement pass.

    When use_llm_refinement=True, an LLM does a final semantic cleanup of the
    extracted Markdown.  Requires at least one LLM_PROVIDER_*_BASE_URL to be
    configured in .env; silently skipped otherwise.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 7: LLM Refinement (use_llm_refinement=True)")
    print("=" * 60)

    result = await real_fetch_web_content(
        "https://github.com/docling-project/docling/discussions/1953",
        use_llm_refinement=True,
    )

    print("\nFetched with LLM refinement enabled:")
    if "content" in result:
        print(f"  URL: {result['url']}")
        print(f"  Extraction method: {result.get('extraction_method', 'unknown')}")
        print(f"  Content preview: {result['content'][:200]}...")
    else:
        print_result(result)


async def example_full_content_fetch():
    """Example 8: Fetch URLs and print full extracted content.

    Defines a list of URLs to fetch and prints the entire extracted content
    for each one. Useful for testing extraction on diverse URLs in the wild.

    This example includes:
      - https://github.com/docling-project/docling/discussions/1953
        A GitHub discussion page with user comments.
      - https://file-examples.com/wp-content/storage/2017/10/file-sample_150kB.pdf
        A PDF file behind a 1-second loading page.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Full Content Fetch (Real-World URLs)")
    print("=" * 60)

    urls_to_fetch = [
        "https://github.com/docling-project/docling/discussions/1953",
        "https://file-examples.com/wp-content/storage/2017/10/file-sample_150kB.pdf",
        "https://stackoverflow.com/questions/9919509/need-help-to-generate-report-in-pdf-or-doc-using-python",
    ]

    for url in urls_to_fetch:
        print(f"\n{'=' * 60}")
        print(f"Fetching: {url}")
        print(f"{'=' * 60}")

        result = await real_fetch_web_content(url)

        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            content = result.get("content", "")
            print(f"URL: {result['url']}")
            print(f"Extraction method: {result.get('extraction_method', 'unknown')}")
            print(f"Content length: {len(content)} characters")
            print(f"\n{'─' * 60}")
            print("FULL CONTENT:")
            print(f"{'─' * 60}")
            print(content)
            print(f"{'─' * 60}")


async def example_summary():
    """Example 9: Summarize content with LLM (summarize=True).

    When summarize=True, an LLM generates a summary of the fetched content
    instead of returning the full content. The summary is returned in the
    'summary' key of the result (not 'content').

    The summary_prompt parameter allows customizing how the LLM summarizes
    the content by providing specific instructions.

    Note: Requires at least one LLM_PROVIDER_*_BASE_URL to be configured
    in .env; returns an error otherwise.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Summarize Content (summarize=True)")
    print("=" * 60)

    # Example URL - Linux kernel README
    url = "https://blog.comma.ai/011release/"

    # Part 1: Basic summarize fetch
    print("\n" + "-" * 60)
    print("Part 1: Basic summarize=True")
    print("-" * 60)
    print(f"\nFetching and summarizing: {url}")

    result = await real_fetch_web_content(
        url,
        summarize=True,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        # When summarize=True, result has 'summary' key instead of 'content'
        summary = result.get("summary", "")
        if summary:
            print(f"\nURL: {result['url']}")
            print(f"Extraction method: {result.get('extraction_method', 'unknown')}")
            print(f"\n{'─' * 60}")
            print("SUMMARY:")
            print(f"{'─' * 60}")
            print(summary)
            print(f"{'─' * 60}")
        else:
            print("No summary extracted (LLM may not be configured)")


async def example_firecrawl_scrape():
    """Example 10: Basic Firecrawl scrape with markdown format.

    Demonstrates direct use of the Firecrawl client for AI-powered scraping.
    Requires USE_FIRECRAWL=true and Firecrawl running on FIRECRAWL_API_URL
    (defaults to http://localhost:3002).
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Firecrawl Scrape")
    print("=" * 60)

    if not await _is_firecrawl_available():
        print("\nSkipped: Firecrawl is not available.")
        print("Set USE_FIRECRAWL=true and ensure Firecrawl is running.")
        return

    client = get_firecrawl_client()
    result = await client.scrape("https://example.com", timeout=30)

    if result:
        print(f"\n[Firecrawl] {result.url}")
        print(f"Method: {result.method}")
        print(f"Word count: {result.word_count}")
        print("-" * 40)
        preview = result.content[:500] if len(result.content) > 500 else result.content
        print(preview + ("..." if len(result.content) > 500 else ""))
    else:
        print("\nFirecrawl scrape failed")


async def example_firecrawl_with_options():
    """Example 11: Firecrawl with screenshot and content options.

    Demonstrates Firecrawl's screenshot_full_page option for capturing
    full-page screenshots, and only_main_content for clean extraction.
    Requires USE_FIRECRAWL=true and Firecrawl running.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Firecrawl with Options")
    print("=" * 60)

    if not await _is_firecrawl_available():
        print("\nSkipped: Firecrawl is not available.")
        print("Set USE_FIRECRAWL=true and ensure Firecrawl is running.")
        return

    client = get_firecrawl_client()
    url = "https://example.com"

    result = await client.scrape(
        url,
        formats=["markdown"],
        only_main_content=False,
        timeout=30,
    )

    if result:
        print(f"\n[Firecrawl] {result.url}")
        print(f"Method: {result.method}")
        print("-" * 40)
        preview = result.content[:800] if len(result.content) > 800 else result.content
        print(preview + ("..." if len(result.content) > 800 else ""))
    else:
        print("\nFirecrawl scrape failed")


async def example_firecrawl_map():
    """Example 12: Discover URLs on a site using Firecrawl /map endpoint.

    Uses Firecrawl's map_site method to discover and list URLs starting
    from a root URL. Requires USE_FIRECRAWL=true and Firecrawl running.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Firecrawl Map Site")
    print("=" * 60)

    if not await _is_firecrawl_available():
        print("\nSkipped: Firecrawl is not available.")
        print("Set USE_FIRECRAWL=true and ensure Firecrawl is running.")
        return

    client = get_firecrawl_client()
    urls = await client.map_site("https://example.com", search_depth=1)

    if urls:
        print(f"\nDiscovered {len(urls)} URLs from https://example.com:")
        for i, discovered_url in enumerate(urls[:20], 1):
            print(f"  {i}. {discovered_url}")
        if len(urls) > 20:
            print(f"  ... and {len(urls) - 20} more")
    else:
        print("\nNo URLs discovered or Firecrawl map failed")


async def example_firecrawl_batch_scrape():
    """Example 13: Batch scrape multiple URLs with Firecrawl.

    Demonstrates submitting a batch scrape job and polling for results.
    Requires USE_FIRECRAWL=true and Firecrawl running.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 13: Firecrawl Batch Scrape")
    print("=" * 60)

    if not await _is_firecrawl_available():
        print("\nSkipped: Firecrawl is not available.")
        print("Set USE_FIRECRAWL=true and ensure Firecrawl is running.")
        return

    client = get_firecrawl_client()
    urls = [
        "https://example.com",
        "https://httpbin.org/html",
    ]

    job_response = await client.batch_scrape(urls, timeout=30)

    if not job_response or "jobId" not in job_response:
        print("\nBatch scrape submission failed")
        return

    job_id = job_response["jobId"]
    print(f"\nSubmitted batch job: {job_id}")
    print("Polling for results...")

    import asyncio
    max_attempts = 10
    for attempt in range(max_attempts):
        await asyncio.sleep(2)
        status = await client.get_batch_status(job_id)

        if not status:
            continue

        status_val = status.get("status", "")
        print(f"  Attempt {attempt + 1}: status={status_val}")

        if status_val == "completed":
            data = status.get("data", [])
            for item in data[:5]:
                url_item = item.get("url", "unknown")
                content = item.get("markdown", "")[:200]
                print(f"\n  URL: {url_item}")
                print(f"  Content preview: {content}...")
            break
        elif status_val in ("failed", "cancelled"):
            print("\nBatch job failed or was cancelled")
            break


async def main(selected_examples: set[int] | None = None):
    """Run fetch examples.
    
    Args:
        selected_examples: Set of example numbers to run. If None or empty, run all.
    """
    # All available examples in execution order
    all_examples = [
        (1, "example_basic", example_basic),
        (2, "example_with_truncation", example_with_truncation),
        (3, "example_with_regex", example_with_regex),
        (4, "example_start_offset", example_start_offset),
        (5, "example_binary_document_formats", example_binary_document_formats),
        (6, "example_pdf_fetch", example_pdf_fetch),
        (7, "example_llm_refinement", example_llm_refinement),
        (8, "example_full_content_fetch", example_full_content_fetch),
        (9, "example_summary", example_summary),
        (10, "example_firecrawl_scrape", example_firecrawl_scrape),
        (11, "example_firecrawl_with_options", example_firecrawl_with_options),
        (12, "example_firecrawl_map", example_firecrawl_map),
        (13, "example_firecrawl_batch_scrape", example_firecrawl_batch_scrape),
    ]
    
    # Determine which examples to run
    if selected_examples:
        examples_to_run = [(num, name, func) for num, name, func in all_examples if num in selected_examples]
    else:
        examples_to_run = all_examples
    
    print("\n" + "#" * 60)
    print("# fetch Examples (using real implementation)")
    if selected_examples:
        example_nums = sorted(selected_examples)
        print(f"# Running examples: {example_nums}")
    else:
        print("# Running all 13 examples")
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