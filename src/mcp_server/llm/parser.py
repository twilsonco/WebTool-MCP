"""
Document parsing module using Docling for multi-format document processing.

Docling supports 16+ formats including:
- HTML, PDF, DOCX, PPTX, XLSX
- Images (PNG, JPG, TIFF, etc.)
- Markdown, CSV, JSON, XML
- And more...

This module provides async document parsing with fallback to BeautifulSoup for HTML.
"""

import io
from typing import Optional, Set
from urllib.parse import urlparse

# Docling imports for document parsing
try:
    from docling.datamodel.base_models import DocumentStream
    from docling.document_converter import DocumentConverter
    
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

# Supported file extensions for Docling parsing
DOCLING_SUPPORTED_EXTENSIONS: Set[str] = {
    ".html",
    # Documents
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    # Other formats
    ".md",
    ".csv",
    ".json",
    ".xml",
}


def is_docling_supported_url(url: str) -> bool:
    """
    Check if a URL points to a file format supported by Docling.
    
    URLs without extensions are treated as HTML (since most web pages are HTML
    and Docling supports HTML parsing).
    
    Args:
        url: The URL to check
        
    Returns:
        True if the URL has a Docling-supported file extension or no extension
    """
    
    # Parse URL and get just the path portion
    parsed = urlparse(url.lower())
    path = parsed.path
    
    # Remove query parameters and fragment from URL before checking extension
    url_clean = path.split("?")[0].split("#")[0]
    
    # Check if URL has any extension (by checking the last path segment)
    # An extension is a "." in the final path component after the last "/"
    filename = url_clean.split("/")[-1]
    
    if not filename or "." not in filename:
        # No extension found - treat as HTML since most web pages are HTML
        return True
    
    # Check if URL ends with a supported extension
    return any(url_clean.endswith(ext) for ext in DOCLING_SUPPORTED_EXTENSIONS)


def is_html_content(content_type: Optional[str]) -> bool:
    """
    Check if content type indicates HTML.
    
    Args:
        content_type: The Content-Type header value
        
    Returns:
        True if the content type indicates HTML
    """
    if not content_type:
        return False
    return "text/html" in content_type.lower() or "application/xhtml" in content_type.lower()


async def parse_with_docling(
    content: bytes,
    file_extension: str,
    include_links: bool = False,
) -> Optional[str]:
    """
    Parse document content using Docling.
    
    Args:
        content: Raw binary content of the document
        file_extension: File extension (e.g., ".pdf", ".docx")
        include_links: Whether to preserve hyperlinks in the output
        
    Returns:
        Parsed text content as markdown, or None if parsing fails
    """
    if not DOCLING_AVAILABLE:
        return None
    
    try:
        # Initialize the document converter (use default settings)
        converter = DocumentConverter()
        
        # Create a DocumentStream from the binary content
        doc_stream = DocumentStream(
            name=f"document{file_extension}",
            stream=io.BytesIO(content),
        )
        
        # Convert the document - returns a single ConversionResult
        result = converter.convert(
            source=doc_stream,
            raises_on_error=False,
        )
        
        if result and hasattr(result, 'document') and result.document:
            # Always export to HTML first for consistent link handling
            html_content = result.document.export_to_html()
            
            if include_links:
                # Convert HTML to markdown with links preserved
                from markdownify import markdownify as md
                return md(html_content)
            else:
                # Strip link tags from HTML before converting to markdown
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                for a_tag in soup.find_all('a'):
                    a_tag.unwrap()  # Remove anchor tags but keep text content
                from markdownify import markdownify as md
                return md(str(soup))
        
        return None
    except Exception:
        # Docling parsing failed, will fall back to other methods
        return None


async def parse_html_with_beautifulsoup(
    html_content: str,
    include_links: bool = False,
) -> str:
    """
    Parse HTML content using BeautifulSoup and markdownify.
    
    This is the fallback method when Docling is not available or fails,
    and also used for regular HTML web pages.
    
    Args:
        html_content: Raw HTML string
        include_links: Whether to preserve anchor tag hrefs
        
    Returns:
        Markdown-formatted text content
    """
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Handle include_links option - strip href attributes when False
    if not include_links:
        for a_tag in soup.find_all('a'):
            a_tag.unwrap()  # Remove anchor tags but keep text content
    
    return md(str(soup))