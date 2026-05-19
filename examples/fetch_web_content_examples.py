#!/usr/bin/env python3
"""
fetchWebContent examples - Demonstrates usage of the fetchWebContent MCP tool.
This script imports and calls the actual implementation directly.

The atomic API fetches one URL per call:
    result = await real_fetch_web_content("https://example.com", num_words=500)

Returns a dict with 'url' and 'content', or 'error' on failure.

Extraction Pipeline (applied in order, best result wins):
1. Playwright  - dynamic rendering for JS-heavy / SPA pages
2. Trafilatura - heuristic text-density extraction (fast, no JS)
3. Readability - Mozilla-style article extraction
4. Docling     - layout-aware parsing for PDFs, DOCX, images, etc.
5. BeautifulSoup - universal HTML fallback (always succeeds)
6. LLM refinement - optional semantic cleanup pass (use_llm_refinement=True)
"""
import os
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
    """Example 6: Fetch and parse a real PDF via Docling.

    Fetches an actual PDF document; the binary-document path routes it
    directly to Docling for layout-aware parsing.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: PDF Fetch (Docling binary path)")
    print("=" * 60)

    # Real sample PDF from GitHub (replace with any accessible PDF URL to test)
    pdf_url = "https://raw.githubusercontent.com/twilsonco/WebTool-MCP/main/examples/files/file-sample_150kB.pdf"

    print(f"\nFetching PDF: {pdf_url}")
    result = await real_fetch_web_content(pdf_url, num_words=20000, include_links=True)

    if "error" in result:
        print(f"\n  Error: {result['error']}")
    else:
        print(f"\n[{result['url']}]")
        print("-" * 40)
        content = result.get("content", "")
        if content:
            print(content)
        else:
            print("No content extracted")


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


async def main():
    print("\n" + "#" * 60)
    print("# fetchWebContent Examples (using real implementation)")
    print("#" * 60)

    await example_basic()
    await example_with_truncation()
    await example_with_regex()
    await example_start_offset()
    await example_binary_document_formats()
    await example_pdf_fetch()
    await example_llm_refinement()
    await example_full_content_fetch()

    print("\n" + "#" * 60)
    print("# Done!")
    print("#" * 60)


if __name__ == "__main__":
    asyncio.run(main())