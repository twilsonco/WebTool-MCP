#!/usr/bin/env python3
"""
fetchWebContent examples - Demonstrates usage of the fetchWebContent MCP tool.
This script imports and calls the actual implementation directly.

The atomic API fetches one URL per call:
    result = await real_fetch_web_content("https://example.com", num_words=500)

Returns a dict with 'url' and 'content', or 'error' on failure.

Docling Integration:
- fetchWebContent automatically uses Docling for supported document formats
- Supported formats: PDF, DOCX, PPTX, XLSX, PNG, JPG, TIFF, BMP, MD, CSV, JSON, XML, HTML
- Falls back to BeautifulSoup for regular web pages
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
from src.mcp_server.server import fetch_web_content as real_fetch_web_content


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

    result = await real_fetch_web_content("https://example.com")
    print_result(result)


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


async def example_docling_formats():
    """Example 5: Docling-supported document formats.
    
    This example demonstrates fetching documents in formats supported by Docling:
    - PDF, DOCX, PPTX, XLSX
    - Images (PNG, JPG, TIFF)
    - Markdown, CSV, JSON, XML
    
    Note: These URLs are placeholders. Replace with actual document URLs to test.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Docling Document Formats")
    print("=" * 60)

    # Example document URLs (replace with actual documents to test)
    docling_urls = [
        "https://example.com/document.pdf",
        "https://example.com/presentation.pptx",
        "https://example.com/spreadsheet.xlsx",
        "https://example.com/document.docx",
    ]

    print("\nDocling automatically parses these formats when detected by URL extension.")
    print("Supported extensions: .pdf, .docx, .pptx, .xlsx, .png, .jpg, .md, .csv, .json, .xml")
    print("\nExample URLs that would use Docling:")
    for url in docling_urls:
        print(f"  - {url}")


async def example_docling_pdf_fetch():
    """Example 6: Fetch and parse a real PDF using Docling.
    
    This example demonstrates fetching an actual PDF document from the web
    and parsing it using Docling's advanced document understanding.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Docling PDF Fetch (Real Document)")
    print("=" * 60)

    # Real sample PDF from file-examples.com
    pdf_url = "https://file-examples.com/wp-content/storage/2017/10/file-sample_150kB.pdf"
    
    print(f"\nFetching PDF: {pdf_url}")
    print("Using Docling for advanced PDF parsing...")
    
    result = await real_fetch_web_content(pdf_url, num_words=200)
    
    if "error" in result:
        print(f"\n  Error: {result['error']}")
    else:
        print(f"\n[{result['url']}]")
        print("-" * 40)
        content = result.get("content", "")
        if content:
            print(f"Content preview (first 500 chars):")
            print(content[:500] + ("..." if len(content) > 500 else ""))
        else:
            print("No content extracted")


async def example_docling_fallback():
    """Example 6: Docling fallback to BeautifulSoup.
    
    When Docling is not available or fails, the system falls back
    to BeautifulSoup for HTML parsing.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Docling Fallback Behavior")
    print("=" * 60)

    # Regular HTML pages always use BeautifulSoup
    result = await real_fetch_web_content("https://example.com")
    
    print("\nRegular HTML pages use BeautifulSoup for parsing:")
    if "content" in result:
        print(f"  URL: {result['url']}")
        print(f"  Content preview: {result['content'][:100]}...")
    else:
        print_result(result)


async def main():
    print("\n" + "#" * 60)
    print("# fetchWebContent Examples (using real implementation)")
    print("#" * 60)

    await example_basic()
    await example_with_truncation()
    await example_with_regex()
    await example_start_offset()
    await example_docling_formats()
    await example_docling_pdf_fetch()  # Real PDF fetch with Docling
    await example_docling_fallback()

    print("\n" + "#" * 60)
    print("# Done!")
    print("#" * 60)


if __name__ == "__main__":
    asyncio.run(main())