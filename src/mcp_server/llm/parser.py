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
import re
from typing import Optional, Set

# Docling imports for document parsing
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.backend.pypdf_backend import PyPdfDocumentBackend
    from docling.backend.markdown_backend import MarkdownBackend
    
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

# Supported file extensions for Docling parsing
DOCLING_SUPPORTED_EXTENSIONS: Set[str] = {
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
    
    Args:
        url: The URL to check
        
    Returns:
        True if the URL has a Docling-supported file extension
    """
    # Remove query parameters and fragment from URL before checking extension
    url_clean = url.lower().split("?")[0].split("#")[0]
    
    for ext in DOCLING_SUPPORTED_EXTENSIONS:
        if url_clean.endswith(ext):
            return True
    return False


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
) -> Optional[str]:
    """
    Parse document content using Docling.
    
    Args:
        content: Raw binary content of the document
        file_extension: File extension (e.g., ".pdf", ".docx")
        
    Returns:
        Parsed text content as markdown, or None if parsing fails
    """
    if not DOCLING_AVAILABLE:
        return None
    
    try:
        # Initialize the document converter with appropriate backend
        if file_extension.lower() == ".pdf":
            # Use PyPDF for PDF files
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        backend=PyPdfDocumentBackend,
                    )
                }
            )
        else:
            # Use default converter for other formats
            converter = DocumentConverter()
        
        # Create an input document from the binary content
        doc = converter.convert(
            source=io.BytesIO(content),
            raises_on_error=False,
        )
        
        if doc and doc.export_to_markdown():
            return doc.export_to_markdown()
        
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


def extract_text_from_markdown(markdown_content: str) -> str:
    """
    Extract plain text from markdown content.
    
    Removes markdown formatting while preserving the underlying text.
    
    Args:
        markdown_content: Markdown formatted text
        
    Returns:
        Plain text with markdown formatting removed
    """
    # Remove headers (keep the text)
    text = re.sub(r'^#{1,6}\s+', '', markdown_content, flags=re.MULTILINE)
    
    # Remove bold/italic markers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    
    # Remove links but keep text
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    
    # Remove image syntax
    text = re.sub(r'!\[.*?\]\(.+?\)', '', text)
    
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    
    # Remove blockquotes markers
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    
    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
    
    # Remove list markers
    text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()