"""
Unit tests for FirecrawlClient.

Tests cover:
  - Client initialization and environment variable handling
  - scrape() method with various response formats
  - batch_scrape() job submission
  - get_batch_status() polling
  - map_site() URL discovery
  - crawl_async() job creation
  - get_crawl_status() polling
  - Error handling and graceful degradation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestFirecrawlClientInit:
    """Tests for FirecrawlClient initialization."""

    def test_default_values(self):
        """Test default URL and timeout values."""
        with patch.dict("os.environ", {}, clear=True):
            from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

            client = FirecrawlClient()
            assert client.api_url == "http://localhost:3002"
            assert client.timeout == 60
            assert client._client is None

    def test_env_var_api_url(self):
        """Test FIRECRAWL_API_URL env var is read."""
        with patch.dict("os.environ", {"FIRECRAWL_API_URL": "http://custom:9000"}):
            from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

            client = FirecrawlClient()
            assert client.api_url == "http://custom:9000"

    def test_env_var_api_key(self):
        """Test FIRECRAWL_API_KEY env var is read."""
        with patch.dict("os.environ", {"FIRECRAWL_API_KEY": "secret-key"}):
            from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

            client = FirecrawlClient()
            assert client.api_key == "secret-key"

    def test_explicit_api_url_overrides_env(self):
        """Test explicit api_url parameter takes precedence over env var."""
        with patch.dict("os.environ", {"FIRECRAWL_API_URL": "http://env-url:3002"}):
            from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

            client = FirecrawlClient(api_url="http://explicit:4000")
            assert client.api_url == "http://explicit:4000"

    def test_explicit_api_key_overrides_env(self):
        """Test explicit api_key parameter takes precedence over env var."""
        with patch.dict("os.environ", {"FIRECRAWL_API_KEY": "env-key"}):
            from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

            client = FirecrawlClient(api_key="explicit-key")
            assert client.api_key == "explicit-key"

    def test_url_strip_trailing_slash(self):
        """Test API URL trailing slash is stripped."""
        with patch.dict("os.environ", {}, clear=True):
            from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

            client = FirecrawlClient(api_url="http://localhost:3002/")
            assert client.api_url == "http://localhost:3002"


class TestFirecrawlScrape:
    """Tests for FirecrawlClient.scrape() method."""

    @pytest.mark.asyncio
    async def test_scrape_success_markdown(self):
        """Test successful scrape returning markdown content."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "markdown": "# Hello World\n\nThis is the content.",
            "metadata": {"title": "Test Page"},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        client = FirecrawlClient()
        client._client = mock_client

        result = await client.scrape("https://example.com")

        assert result is not None
        assert result.method == "firecrawl"
        assert "# Hello World" in result.content
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_success_html(self):
        """Test successful scrape returning HTML content."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "html": "<html><body><p>Hello World</p></body></html>",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        client = FirecrawlClient()
        client._client = mock_client

        result = await client.scrape("https://example.com", formats=["html"])

        assert result is not None
        assert "Hello World" in result.content

    @pytest.mark.asyncio
    async def test_scrape_fallback_to_content_field(self):
        """Test fallback to 'content' field when format-specific field is missing."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": "Fallback content here",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        client = FirecrawlClient()
        client._client = mock_client

        result = await client.scrape("https://example.com")

        assert result is not None
        assert "Fallback content" in result.content

    @pytest.mark.asyncio
    async def test_scrape_returns_none_on_http_error(self):
        """Test scrape returns None on HTTP error."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        ))
        mock_client.is_closed = False

        client = FirecrawlClient()
        client._client = mock_client

        result = await client.scrape("https://example.com")

        assert result is None

    @pytest.mark.asyncio
    async def test_scrape_returns_none_on_timeout(self):
        """Test scrape returns None on timeout."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient
        import httpx

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client.is_closed = False

        client = FirecrawlClient(timeout=5)
        client._client = mock_client

        result = await client.scrape("https://example.com")

        assert result is None

    @pytest.mark.asyncio
    async def test_scrape_includes_auth_header_when_api_key_set(self):
        """Test Authorization header is included when api_key is set."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"markdown": "content"}

        captured_request = {}

        async def mock_post(url, **kwargs):
            captured_request["url"] = url
            captured_request["headers"] = kwargs.get("headers", {})
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.is_closed = False

        client = FirecrawlClient(api_key="test-key-123")
        client._client = mock_client

        await client.scrape("https://example.com")

        assert "Authorization" in captured_request["headers"]
        assert captured_request["headers"]["Authorization"] == "Bearer test-key-123"

    @pytest.mark.asyncio
    async def test_scrape_respects_formats_parameter(self):
        """Test scrape sends correct formats to API."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"markdown": "content"}

        captured_body = {}

        async def mock_post(url, **kwargs):
            captured_body.update(kwargs.get("json", {}))
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.is_closed = False

        client = FirecrawlClient()
        client._client = mock_client

        await client.scrape("https://example.com", formats=["markdown", "html"])

        assert captured_body["formats"] == ["markdown", "html"]

    @pytest.mark.asyncio
    async def test_scrape_includes_actions(self):
        """Test scrape includes browser actions when provided."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"markdown": "content"}

        captured_body = {}

        async def mock_post(url, **kwargs):
            captured_body.update(kwargs.get("json", {}))
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.is_closed = False

        client = FirecrawlClient()
        client._client = mock_client

        actions = [{"type": "scroll", "direction": "down"}]
        await client.scrape("https://example.com", actions=actions)

        assert captured_body["actions"] == actions


class TestFirecrawlBatchScrape:
    """Tests for FirecrawlClient.batch_scrape() method."""

    @pytest.mark.asyncio
    async def test_batch_scrape_returns_job_id(self):
        """Test batch scrape returns job information."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jobId": "batch-123"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        client = FirecrawlClient()
        client._client = mock_client

        result = await client.batch_scrape(["https://a.com", "https://b.com"])

        assert result == {"jobId": "batch-123"}
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_scrape_returns_empty_on_error(self):
        """Test batch scrape returns empty dict on HTTP error."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 400

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=mock_response
        ))
        mock_client.is_closed = False

        client = FirecrawlClient()
        client._client = mock_client

        result = await client.batch_scrape(["https://a.com"])

        assert result == {}


class TestFirecrawlGetBatchStatus:
    """Tests for FirecrawlClient.get_batch_status() method."""

    @pytest.mark.asyncio
    async def test_get_batch_status_success(self):
        """Test successful batch status retrieval."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jobId": "batch-123",
            "status": "completed",
            "data": [{"url": "https://a.com", "content": "..."}],
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        client = FirecrawlClient()
        client._client = mock_client

        result = await client.get_batch_status("batch-123")

        assert result["status"] == "completed"
        assert len(result["data"]) == 1


class TestFirecrawlMapSite:
    """Tests for FirecrawlClient.map_site() method."""

    @pytest.mark.asyncio
    async def test_map_site_returns_urls(self):
        """Test map_site returns discovered URLs."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "urls": ["https://example.com", "https://example.com/about"],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        client = FirecrawlClient()
        client._client = mock_client

        result = await client.map_site("https://example.com")

        assert len(result) == 2
        assert "https://example.com" in result

    @pytest.mark.asyncio
    async def test_map_site_returns_empty_on_error(self):
        """Test map_site returns empty list on error."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        ))
        mock_client.is_closed = False

        client = FirecrawlClient()
        client._client = mock_client

        result = await client.map_site("https://example.com")

        assert result == []

    @pytest.mark.asyncio
    async def test_map_site_handles_list_response(self):
        """Test map_site handles list response format."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            "https://a.com",
            "https://b.com",
        ]

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        client = FirecrawlClient()
        client._client = mock_client

        result = await client.map_site("https://example.com")

        assert len(result) == 2


class TestFirecrawlCrawlAsync:
    """Tests for FirecrawlClient.crawl_async() method."""

    @pytest.mark.asyncio
    async def test_crawl_async_returns_job_id(self):
        """Test crawl_async returns job ID."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jobId": "crawl-456"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        client = FirecrawlClient()
        client._client = mock_client

        result = await client.crawl_async("https://example.com")

        assert result == "crawl-456"

    @pytest.mark.asyncio
    async def test_crawl_async_returns_none_on_error(self):
        """Test crawl_async returns None on HTTP error."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 400

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=mock_response
        ))
        mock_client.is_closed = False

        client = FirecrawlClient()
        client._client = mock_client

        result = await client.crawl_async("https://example.com")

        assert result is None


class TestFirecrawlGetCrawlStatus:
    """Tests for FirecrawlClient.get_crawl_status() method."""

    @pytest.mark.asyncio
    async def test_get_crawl_status_success(self):
        """Test successful crawl status retrieval."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jobId": "crawl-456",
            "status": "completed",
            "data": [],
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        client = FirecrawlClient()
        client._client = mock_client

        result = await client.get_crawl_status("crawl-456")

        assert result["status"] == "completed"


class TestFirecrawlClose:
    """Tests for FirecrawlClient.close() method."""

    @pytest.mark.asyncio
    async def test_close_closes_client(self):
        """Test close properly closes the HTTP client."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        mock_aclose = AsyncMock()
        mock_client = MagicMock()
        mock_client.aclose = mock_aclose
        mock_client.is_closed = False

        client = FirecrawlClient()
        client._client = mock_client

        await client.close()

        mock_aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_handles_none_client(self):
        """Test close handles None client gracefully."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        client = FirecrawlClient()
        client._client = None

        await client.close()

        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_handles_already_closed_client(self):
        """Test close handles already closed client gracefully."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        mock_client = MagicMock()
        mock_client.is_closed = True

        client = FirecrawlClient()
        client._client = mock_client

        await client.close()

        assert client._client is None


class TestExtractContentFromResponse:
    """Tests for _extract_content_from_response helper."""

    def test_extracts_markdown_format(self):
        """Test extraction from markdown format."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        data = {"markdown": "# Title\n\nSome content here."}
        client = FirecrawlClient()

        result = client._extract_content_from_response(data, ["markdown"])

        assert result == "# Title\n\nSome content here."

    def test_extracts_html_format(self):
        """Test extraction from HTML format."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        data = {"html": "<p>Hello</p><p>World</p>"}
        client = FirecrawlClient()

        result = client._extract_content_from_response(data, ["html"])

        assert result == "<p>Hello</p><p>World</p>"

    def test_falls_back_to_content_field(self):
        """Test fallback to generic content field."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        data = {"content": "Generic content"}
        client = FirecrawlClient()

        result = client._extract_content_from_response(data, ["markdown"])

        assert result == "Generic content"

    def test_returns_none_for_empty_response(self):
        """Test returns None for empty response."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        data = {}
        client = FirecrawlClient()

        result = client._extract_content_from_response(data, ["markdown"])

        assert result is None

    def test_returns_none_for_short_content(self):
        """Test returns None for content below word threshold."""
        from src.mcp_server.extraction.firecrawl_client import FirecrawlClient

        data = {"markdown": "short"}
        client = FirecrawlClient()

        result = client._extract_content_from_response(data, ["markdown"])

        assert result is None


class TestGetFirecrawlClient:
    """Tests for get_firecrawl_client factory function."""

    @pytest.mark.asyncio
    async def test_returns_configured_client(self):
        """Test factory returns a configured FirecrawlClient."""
        with patch.dict("os.environ", {"FIRECRAWL_API_URL": "http://custom:3002"}):
            from src.mcp_server.extraction.firecrawl_client import get_firecrawl_client

            client = await get_firecrawl_client()

            assert client.api_url == "http://custom:3002"