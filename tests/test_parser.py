"""
Unit tests for mcp_server.llm.parser module.

Covers:
  - _get_docling_converter singleton creation and reuse
  - is_html_content utility
  - parse_with_docling: unavailable / no-converter / success / no-document / exception
  - parse_html_with_beautifulsoup: basic conversion, links stripped/preserved
"""

import pytest
from unittest.mock import MagicMock, patch

import mcp_server.llm.parser as parser_module


# ---------------------------------------------------------------------------
# _get_docling_converter
# ---------------------------------------------------------------------------

class TestGetDoclingConverter:
    def test_returns_none_when_docling_unavailable(self):
        with patch.object(parser_module, "DOCLING_AVAILABLE", False):
            result = parser_module._get_docling_converter()
        assert result is None

    def test_creates_converter_on_first_call(self):
        original = parser_module._docling_converter
        parser_module._docling_converter = None
        mock_converter = MagicMock()
        try:
            with patch("mcp_server.llm.parser.DocumentConverter", return_value=mock_converter):
                result = parser_module._get_docling_converter()
            assert result is mock_converter
            assert parser_module._docling_converter is mock_converter
        finally:
            parser_module._docling_converter = original

    def test_reuses_existing_converter(self):
        original = parser_module._docling_converter
        mock_converter = MagicMock()
        parser_module._docling_converter = mock_converter
        try:
            result = parser_module._get_docling_converter()
            assert result is mock_converter
        finally:
            parser_module._docling_converter = original


# ---------------------------------------------------------------------------
# is_html_content
# ---------------------------------------------------------------------------

class TestIsHtmlContent:
    def test_none_content_type_returns_false(self):
        assert parser_module.is_html_content(None) is False

    def test_empty_string_returns_false(self):
        assert parser_module.is_html_content("") is False

    def test_text_html_returns_true(self):
        assert parser_module.is_html_content("text/html; charset=utf-8") is True

    def test_application_xhtml_returns_true(self):
        assert parser_module.is_html_content("application/xhtml+xml") is True

    def test_application_json_returns_false(self):
        assert parser_module.is_html_content("application/json") is False


# ---------------------------------------------------------------------------
# parse_with_docling
# ---------------------------------------------------------------------------

class TestParseWithDocling:
    @pytest.mark.asyncio
    async def test_returns_none_when_docling_unavailable(self):
        with patch.object(parser_module, "DOCLING_AVAILABLE", False):
            result = await parser_module.parse_with_docling(b"content", ".pdf")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_converter_is_none(self):
        with patch("mcp_server.llm.parser._get_docling_converter", return_value=None):
            result = await parser_module.parse_with_docling(b"content", ".pdf")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_markdown_without_links(self):
        mock_result = MagicMock()
        mock_result.document = MagicMock()
        mock_result.document.export_to_html.return_value = (
            '<html><body><p>Content with <a href="https://link.com">link text</a></p></body></html>'
        )
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        original = parser_module._docling_converter
        parser_module._docling_converter = mock_converter
        try:
            result = await parser_module.parse_with_docling(b"data", ".pdf", include_links=False)
            assert result is not None
            assert isinstance(result, str)
            assert "link text" in result
            assert "https://link.com" not in result
        finally:
            parser_module._docling_converter = original

    @pytest.mark.asyncio
    async def test_returns_markdown_with_links(self):
        mock_result = MagicMock()
        mock_result.document = MagicMock()
        mock_result.document.export_to_html.return_value = (
            '<html><body><p>See <a href="https://example.com">this page</a></p></body></html>'
        )
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        original = parser_module._docling_converter
        parser_module._docling_converter = mock_converter
        try:
            result = await parser_module.parse_with_docling(b"data", ".pdf", include_links=True)
            assert result is not None
            assert "https://example.com" in result
        finally:
            parser_module._docling_converter = original

    @pytest.mark.asyncio
    async def test_returns_none_when_result_document_is_none(self):
        mock_result = MagicMock()
        mock_result.document = None

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        original = parser_module._docling_converter
        parser_module._docling_converter = mock_converter
        try:
            result = await parser_module.parse_with_docling(b"data", ".pdf")
            assert result is None
        finally:
            parser_module._docling_converter = original

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        mock_converter = MagicMock()
        mock_converter.convert.side_effect = Exception("conversion failed")

        original = parser_module._docling_converter
        parser_module._docling_converter = mock_converter
        try:
            result = await parser_module.parse_with_docling(b"data", ".pdf")
            assert result is None
        finally:
            parser_module._docling_converter = original


# ---------------------------------------------------------------------------
# parse_html_with_beautifulsoup
# ---------------------------------------------------------------------------

class TestParseHtmlWithBeautifulSoup:
    @pytest.mark.asyncio
    async def test_converts_html_to_markdown(self):
        html = "<html><body><h1>Title</h1><p>Some paragraph content.</p></body></html>"
        result = await parser_module.parse_html_with_beautifulsoup(html)
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    @pytest.mark.asyncio
    async def test_links_stripped_by_default(self):
        html = '<html><body><p>Visit <a href="https://example.com">this site</a></p></body></html>'
        result = await parser_module.parse_html_with_beautifulsoup(html, include_links=False)
        assert "this site" in result
        assert "https://example.com" not in result

    @pytest.mark.asyncio
    async def test_links_preserved_when_requested(self):
        html = '<html><body><p>Visit <a href="https://example.com">this site</a></p></body></html>'
        result = await parser_module.parse_html_with_beautifulsoup(html, include_links=True)
        assert "https://example.com" in result


import pytest
from unittest.mock import MagicMock, patch

import mcp_server.llm.parser as parser_module


# ---------------------------------------------------------------------------
# _get_docling_converter
# ---------------------------------------------------------------------------

class TestGetDoclingConverter:
    def test_returns_none_when_docling_unavailable(self):
        with patch.object(parser_module, "DOCLING_AVAILABLE", False):
            result = parser_module._get_docling_converter()
        assert result is None

    def test_creates_converter_on_first_call(self):
        original = parser_module._docling_converter
        parser_module._docling_converter = None
        mock_converter = MagicMock()
        try:
            with patch("mcp_server.llm.parser.DocumentConverter", return_value=mock_converter):
                result = parser_module._get_docling_converter()
            assert result is mock_converter
            assert parser_module._docling_converter is mock_converter
        finally:
            parser_module._docling_converter = original

    def test_reuses_existing_converter(self):
        original = parser_module._docling_converter
        mock_converter = MagicMock()
        parser_module._docling_converter = mock_converter
        try:
            result = parser_module._get_docling_converter()
            assert result is mock_converter
        finally:
            parser_module._docling_converter = original


# ---------------------------------------------------------------------------
# parse_with_docling
# ---------------------------------------------------------------------------

class TestParseWithDocling:
    @pytest.mark.asyncio
    async def test_returns_none_when_docling_unavailable(self):
        with patch.object(parser_module, "DOCLING_AVAILABLE", False):
            result = await parser_module.parse_with_docling(b"content", ".pdf")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_converter_is_none(self):
        with patch("mcp_server.llm.parser._get_docling_converter", return_value=None):
            result = await parser_module.parse_with_docling(b"content", ".pdf")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_markdown_without_links(self):
        mock_result = MagicMock()
        mock_result.document = MagicMock()
        mock_result.document.export_to_html.return_value = (
            '<html><body><p>Content with <a href="https://link.com">link text</a></p></body></html>'
        )
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        original = parser_module._docling_converter
        parser_module._docling_converter = mock_converter
        try:
            result = await parser_module.parse_with_docling(b"data", ".pdf", include_links=False)
            assert result is not None
            assert isinstance(result, str)
            assert "link text" in result
            assert "https://link.com" not in result
        finally:
            parser_module._docling_converter = original

    @pytest.mark.asyncio
    async def test_returns_markdown_with_links(self):
        mock_result = MagicMock()
        mock_result.document = MagicMock()
        mock_result.document.export_to_html.return_value = (
            '<html><body><p>See <a href="https://example.com">this page</a></p></body></html>'
        )
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        original = parser_module._docling_converter
        parser_module._docling_converter = mock_converter
        try:
            result = await parser_module.parse_with_docling(b"data", ".pdf", include_links=True)
            assert result is not None
            assert "https://example.com" in result
        finally:
            parser_module._docling_converter = original

    @pytest.mark.asyncio
    async def test_returns_none_when_result_document_is_none(self):
        mock_result = MagicMock()
        mock_result.document = None

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        original = parser_module._docling_converter
        parser_module._docling_converter = mock_converter
        try:
            result = await parser_module.parse_with_docling(b"data", ".pdf")
            assert result is None
        finally:
            parser_module._docling_converter = original

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        mock_converter = MagicMock()
        mock_converter.convert.side_effect = Exception("conversion failed")

        original = parser_module._docling_converter
        parser_module._docling_converter = mock_converter
        try:
            result = await parser_module.parse_with_docling(b"data", ".pdf")
            assert result is None
        finally:
            parser_module._docling_converter = original


# ---------------------------------------------------------------------------
# parse_html_with_beautifulsoup
# ---------------------------------------------------------------------------

class TestParseHtmlWithBeautifulSoup:
    @pytest.mark.asyncio
    async def test_converts_html_to_markdown(self):
        html = "<html><body><h1>Title</h1><p>Some paragraph content.</p></body></html>"
        result = await parser_module.parse_html_with_beautifulsoup(html)
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    @pytest.mark.asyncio
    async def test_links_stripped_by_default(self):
        html = '<html><body><p>Visit <a href="https://example.com">this site</a></p></body></html>'
        result = await parser_module.parse_html_with_beautifulsoup(html, include_links=False)
        assert "this site" in result
        assert "https://example.com" not in result

    @pytest.mark.asyncio
    async def test_links_preserved_when_requested(self):
        html = '<html><body><p>Visit <a href="https://example.com">this site</a></p></body></html>'
        result = await parser_module.parse_html_with_beautifulsoup(html, include_links=True)
        assert "https://example.com" in result
