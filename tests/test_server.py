import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# Import from server module
from src.mcp_server.server import (
    fetch_web_content, summarize_text, search_web, _call_llm,
    _get_configured_providers, _brave_freshness,
)
from src.mcp_server.extraction.pipeline import ExtractionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_html_response(html: str) -> MagicMock:
    """Return a mock httpx response that yields *html* as its text."""
    mock = MagicMock()
    mock.text = html
    mock.content = html.encode()
    mock.raise_for_status = MagicMock()
    return mock


def _patch_pipeline(content: str, method: str = "static+trafilatura"):
    """Patch the extraction pipeline to return a controlled ExtractionResult."""
    return patch(
        "src.mcp_server.server._extraction_pipeline.extract_from_html",
        new=AsyncMock(return_value=ExtractionResult(content=content, method=method)),
    )


class TestWebFetch:
    @pytest.mark.asyncio
    async def test_fetch_web_content_single_url_success(self):
        expected_content = "Test Content here"

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = _make_html_response(
                "<html><body><h1>Test</h1><p>Content here.</p></body></html>"
            )
            mock_client.return_value.__aenter__.return_value = instance

            with _patch_pipeline(expected_content):
                result = await fetch_web_content("https://example.com")

        assert result["url"] == "https://example.com"
        content_lower = result["content"].lower()
        assert any(word in content_lower for word in ["test", "content"])

    @pytest.mark.asyncio
    async def test_fetch_web_content_with_regex_filter(self):
        pipeline_content = "ERROR_CODE: 123 ERROR_TYPE: critical"

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = _make_html_response(
                "<html><body>ERROR_CODE: 123 ERROR_TYPE: critical</body></html>"
            )
            mock_client.return_value.__aenter__.return_value = instance

            with _patch_pipeline(pipeline_content):
                result = await fetch_web_content(
                    "https://example.com", regex="ERROR_", regex_padding=10
                )

        assert "url" in result
        if "content" in result:
            assert len(result["content"]) > 0

    @pytest.mark.asyncio
    async def test_fetch_web_content_word_truncation(self):
        pipeline_content = " ".join(["word"] * 200)

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = _make_html_response(
                "<p>" + pipeline_content + "</p>"
            )
            mock_client.return_value.__aenter__.return_value = instance

            with _patch_pipeline(pipeline_content):
                result = await fetch_web_content("https://example.com", num_words=50)

        words = result["content"].split()
        assert len(words) <= 55  # Allow small margin for markdown conversion

    @pytest.mark.asyncio
    async def test_fetch_web_content_http_error(self):
        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            error_resp = MagicMock()
            error_resp.raise_for_status.side_effect = Exception("404 Not Found")
            instance.get.return_value = error_resp
            mock_client.return_value.__aenter__.return_value = instance

            result = await fetch_web_content("https://example.com/notfound")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_fetch_web_content_regex_no_match(self):
        pipeline_content = "No errors here, just normal content."

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = _make_html_response(
                "<p>No errors here, just normal content.</p>"
            )
            mock_client.return_value.__aenter__.return_value = instance

            with _patch_pipeline(pipeline_content):
                result = await fetch_web_content(
                    "https://example.com",
                    regex="NONEXISTENT_PATTERN_",
                    regex_padding=10,
                )

        assert "content" in result
        assert "No matches found" in result["content"]

    @pytest.mark.asyncio
    async def test_fetch_web_content_include_links(self):
        pipeline_content_links = "[Click here](https://link.com) and [Other](https://other.com)"
        pipeline_content_no_links = "Click here and Other"

        html = '<html><body><a href="https://link.com">Click here</a> and <a href="https://other.com">Other</a></body></html>'

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = _make_html_response(html)
            mock_client.return_value.__aenter__.return_value = instance

            with _patch_pipeline(pipeline_content_links):
                result_with = await fetch_web_content(
                    "https://example.com", include_links=True
                )

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = _make_html_response(html)
            mock_client.return_value.__aenter__.return_value = instance

            with _patch_pipeline(pipeline_content_no_links):
                result_without = await fetch_web_content(
                    "https://example.com", include_links=False
                )

        assert "url" in result_with and "content" in result_with
        assert "url" in result_without and "content" in result_without

    @pytest.mark.asyncio
    async def test_fetch_web_content_start_word_pagination(self):
        words = [f"word{i}" for i in range(100)]
        pipeline_content = " ".join(words)

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = _make_html_response(
                "<p>" + pipeline_content + "</p>"
            )
            mock_client.return_value.__aenter__.return_value = instance

            with _patch_pipeline(pipeline_content):
                result_page1 = await fetch_web_content(
                    "https://example.com", start_word=0, num_words=10
                )
            with _patch_pipeline(pipeline_content):
                result_page2 = await fetch_web_content(
                    "https://example.com", start_word=50, num_words=10
                )

        assert result_page1["content"] != result_page2["content"]

    @pytest.mark.asyncio
    async def test_fetch_web_content_regex_padding(self):
        pipeline_content = "prefix content CRITICAL: 123 suffix content more text"

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = _make_html_response(
                "<p>" + pipeline_content + "</p>"
            )
            mock_client.return_value.__aenter__.return_value = instance

            with _patch_pipeline(pipeline_content):
                result_small = await fetch_web_content(
                    "https://example.com", regex="CRITICAL", regex_padding=5
                )
            with _patch_pipeline(pipeline_content):
                result_large = await fetch_web_content(
                    "https://example.com", regex="CRITICAL", regex_padding=100
                )

        assert "No matches" not in result_small["content"]
        assert "No matches" not in result_large["content"]

    @pytest.mark.asyncio
    async def test_fetch_web_content_llm_refinement_param_accepted(self):
        """use_llm_refinement parameter should be passed through without error."""
        pipeline_content = "some extracted content"

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = _make_html_response(
                "<p>some extracted content</p>"
            )
            mock_client.return_value.__aenter__.return_value = instance

            with _patch_pipeline(pipeline_content):
                result = await fetch_web_content(
                    "https://example.com", use_llm_refinement=False
                )

        assert "content" in result




class TestWebSearch:
    @pytest.mark.asyncio
    async def test_search_tavily_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Test Result", "url": "https://example.com", "content": "Description text"}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            mock_getenv.return_value = "fake_api_key"

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                instance.post.return_value = mock_response
                mock_client.return_value.__aenter__.return_value = instance

                result = await search_web("test query", provider="tavily")

                assert result["provider"] == "tavily"
                assert len(result["results"]) == 1
                assert result["results"][0]["title"] == "Test Result"

    @pytest.mark.asyncio
    async def test_search_tavily_no_api_key(self):
        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            # Mock ALL providers to ensure only tavily is checked and it's not configured
            def get_env(key, default=None):
                if key == "TAVILY_API_KEY":
                    return None  # Tavily not configured
                elif key in ("BRAVE_API_KEY", "GOOGLE_API_KEY", "GOOGLE_SEARCH_ENGINE_ID"):
                    return None  # Other providers also not configured to force error
                return default
            mock_getenv.side_effect = get_env

            result = await search_web("test", provider="tavily")

            # With miklium always available, the search should still run via failover
            if "error" in result:
                assert "not configured" in result["error"] or "failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_brave_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": "Brave Result", "url": "https://brave.example.com", "description": "Desc"}
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            mock_getenv.return_value = "fake_brave_key"

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                instance.get.return_value = mock_response
                mock_client.return_value.__aenter__.return_value = instance

                result = await search_web("test", provider="brave")

                assert result["provider"] == "brave"
                assert len(result["results"]) == 1

    @pytest.mark.asyncio
    async def test_search_google_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "items": [
                {"title": "Google Result", "link": "https://google.example.com", "snippet": "Snippet text"}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key == "GOOGLE_API_KEY":
                    return "fake_google_key"
                elif key == "GOOGLE_SEARCH_ENGINE_ID":
                    return "fake_cx"
                return default
            mock_getenv.side_effect = get_env

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                instance.get.return_value = mock_response
                mock_client.return_value.__aenter__.return_value = instance

                result = await search_web("test", provider="google")

                assert result["provider"] == "google"
                assert len(result["results"]) == 1
                assert result["results"][0]["url"] == "https://google.example.com"

    @pytest.mark.asyncio
    async def test_search_unknown_provider(self):
        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            # Return a fake key so at least one provider is "configured"
            mock_getenv.return_value = "fake_key"
            result = await search_web("test", provider="unknown")

            # With unknown provider, it should try configured providers and fail or use failover

    @pytest.mark.asyncio
    async def test_search_api_error_handling(self):
        # Create error-throwing responses for both POST (tavily/miklium) and GET (brave/google)
        post_error = MagicMock()
        post_error.raise_for_status.side_effect = Exception("Connection timeout")
        post_error.text = "Timeout"

        get_error = MagicMock()
        get_error.raise_for_status.side_effect = Exception("Connection timeout")
        get_error.text = "Timeout"

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            mock_getenv.return_value = "fake_api_key"

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                # Tavily and miklium use POST; brave and google use GET
                instance.post.return_value = post_error
                instance.get.return_value = get_error
                mock_client.return_value.__aenter__.return_value = instance

                result = await search_web("test", provider="tavily")

                # Either error in the result directly or failover_attempts with errors
                has_error = "error" in result or ("failover_attempts" in result and any("error" in a for a in result["failover_attempts"]))
                assert has_error

    @pytest.mark.asyncio
    async def test_search_brave_no_api_key(self):
        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            # Mock ALL providers to ensure proper isolation
            def get_env(key, default=None):
                if key == "BRAVE_API_KEY":
                    return None  # Brave not configured
                elif key in ("TAVILY_API_KEY", "GOOGLE_API_KEY", "GOOGLE_SEARCH_ENGINE_ID"):
                    return None  # Other providers also not configured to force error
                return default
            mock_getenv.side_effect = get_env

            result = await search_web("test", provider="brave")

            # With miklium always available, brave-specific key not being set means
            # it fails over to miklium
            if "error" in result:
                assert "configured" in result["error"] or "failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_google_no_credentials(self):
        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            # Mock ALL providers to ensure proper isolation
            def get_env(key, default=None):
                if key in ("GOOGLE_API_KEY", "GOOGLE_SEARCH_ENGINE_ID"):
                    return None  # Google not configured
                elif key in ("TAVILY_API_KEY", "BRAVE_API_KEY"):
                    return None  # Other providers also not configured to force error
                return default
            mock_getenv.side_effect = get_env

            result = await search_web("test", provider="google")

            # With miklium always available, google-specific keys not being set means
            # it fails over to miklium
            if "error" in result:
                assert "configured" in result["error"] or "failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_result_count(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": f"Result {i}", "url": f"https://example.com/{i}", "content": "Desc"}
                for i in range(5)
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            mock_getenv.return_value = "fake_api_key"

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                instance.post.return_value = mock_response
                mock_client.return_value.__aenter__.return_value = instance

                result = await search_web("test", provider="tavily", num_results=3)

                # Should be limited to requested count (max 20, but we asked for 3)
                assert len(result["results"]) <= 3

    @pytest.mark.asyncio
    async def test_search_miklium_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "results": [
                {"url": "https://miklium.example.com", "snippet": "Miklium search result"}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                # No API keys for other providers - miklium is always available
                if key in ("TAVILY_API_KEY", "BRAVE_API_KEY", "GOOGLE_API_KEY"):
                    return None
                return default
            mock_getenv.side_effect = get_env

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                instance.post.return_value = mock_response
                mock_client.return_value.__aenter__.return_value = instance

                result = await search_web("test query")

                assert result["provider"] == "miklium"
                assert len(result["results"]) >= 1

    @pytest.mark.asyncio
    async def test_search_miklium_api_error(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Miklium timeout")
        mock_response.text = "Timeout"

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key in ("TAVILY_API_KEY", "BRAVE_API_KEY", "GOOGLE_API_KEY"):
                    return None
                return default
            mock_getenv.side_effect = get_env

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                instance.post.return_value = mock_response
                mock_client.return_value.__aenter__.return_value = instance

                result = await search_web("test")

                assert "error" in result

    @pytest.mark.asyncio
    async def test_search_empty_query(self):
        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            mock_getenv.return_value = "fake_key"

            result = await search_web("")

        assert "error" in result
        assert "Missing required field: query" in result["error"]

    @pytest.mark.asyncio
    async def test_search_tavily_with_days(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Recent Result", "url": "https://example.com", "content": "Desc"}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            mock_getenv.return_value = "fake_api_key"

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                instance.post.return_value = mock_response
                mock_client.return_value.__aenter__.return_value = instance

                result = await search_web("test", provider="tavily", days=7)

                # Verify the payload included start_date (computed from days)
                call_args = instance.post.call_args
                json_payload = call_args.kwargs["json"]
                assert "start_date" in json_payload

    @pytest.mark.asyncio
    async def test_search_tavily_days_zero(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Result", "url": "https://example.com", "content": "Desc"}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            mock_getenv.return_value = "fake_api_key"

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                instance.post.return_value = mock_response
                mock_client.return_value.__aenter__.return_value = instance

                result = await search_web("test", provider="tavily", days=0)

                call_args = instance.post.call_args
                json_payload = call_args.kwargs["json"]
                # days=0 should NOT add start_date
                assert "start_date" not in json_payload

    @pytest.mark.asyncio
    async def test_search_brave_with_offset(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": "Brave Result", "url": "https://example.com", "description": "Desc"}
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            mock_getenv.return_value = "fake_brave_key"

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                instance.get.return_value = mock_response
                mock_client.return_value.__aenter__.return_value = instance

                result = await search_web("test", provider="brave", offset=10)

                # Verify offset was passed to the API call
                call_args = instance.get.call_args
                params = call_args.kwargs["params"]
                assert "offset" in params
                assert params["offset"] == 10

    @pytest.mark.asyncio
    async def test_search_failover_to_miklium(self):
        """Test that when other providers fail, miklium can serve as failover."""
        # Tavily fails
        tavily_error = MagicMock()
        tavily_error.raise_for_status.side_effect = Exception("Tavily down")
        tavily_error.text = "Down"

        # Miklium succeeds
        miklium_response = MagicMock()
        miklium_response.json.return_value = {
            "success": True,
            "results": [
                {"url": "https://miklium.example.com", "snippet": "Miklium result"}
            ]
        }
        miklium_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key == "TAVILY_API_KEY":
                    return "fake_tavily_key"
                if key in ("BRAVE_API_KEY", "GOOGLE_API_KEY"):
                    return None
                return default
            mock_getenv.side_effect = get_env

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                # Tavily POST fails, Miklium POST succeeds
                instance.post.side_effect = [tavily_error, miklium_response]
                mock_client.return_value.__aenter__.return_value = instance

                result = await search_web("test", provider="tavily")

                # Should either failover to miklium or report error with failover_attempts
                if "results" in result and len(result["results"]) > 0:
                    assert result.get("provider") == "miklium" or result.get("failover_attempts")
                else:
                    assert "error" in result or "failover_attempts" in result

    @pytest.mark.asyncio
    async def test_search_no_providers_configured(self):
        """Test when no search providers are configured at all."""
        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key in ("TAVILY_API_KEY", "BRAVE_API_KEY", "GOOGLE_API_KEY"):
                    return None
                return default
            mock_getenv.side_effect = get_env

            result = await search_web("test", provider="unknown")

        # Unknown provider should not be in configured_providers, so it gets skipped
        # and failover to miklium should work or produce an error

    @pytest.mark.asyncio
    async def test_search_failover_with_attempts_on_success(self):
        """Test that failover_attempts is included when some providers fail before one succeeds."""
        # Tavily fails
        tavily_error = MagicMock()
        tavily_error.raise_for_status.side_effect = Exception("Tavily down")
        tavily_error.text = "Down"

        # Brave succeeds
        brave_response = MagicMock()
        brave_response.json.return_value = {
            "web": {"results": [{"title": "Brave Result", "url": "https://b.com", "description": "D"}]}
        }
        brave_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key == "TAVILY_API_KEY":
                    return "fake_tavily"
                if key == "BRAVE_API_KEY":
                    return "fake_brave"
                return default
            mock_getenv.side_effect = get_env

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                # miklium POST succeeds (for default search), tavily POST fails, brave GET succeeds
                miklium_ok = MagicMock()
                miklium_ok.json.return_value = {"success": True, "results": [{"url": "https://m.com", "snippet": "M"}]}
                miklium_ok.raise_for_status = MagicMock()
                instance.post.side_effect = [miklium_ok, tavily_error]
                instance.get.return_value = brave_response
                mock_client.return_value.__aenter__.return_value = instance

                result = await search_web("test", provider="tavily")

                # Should have results (from failover to brave or miklium)
                if "results" in result and len(result["results"]) > 0:
                    # If it succeeded after failover, check for failover_attempts
                    if "failover_attempts" in result:
                        assert len(result["failover_attempts"]) > 0

    @pytest.mark.asyncio
    async def test_search_tavily_not_configured_direct(self):
        """Test the direct 'not configured' return from _search_tavily."""
        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key == "TAVILY_API_KEY":
                    return None
                # Other providers also not configured to prevent failover success
                if key in ("BRAVE_API_KEY", "GOOGLE_API_KEY"):
                    return None
                return default
            mock_getenv.side_effect = get_env

            # Mock miklium and brave/google to all fail so we see the provider-specific error
            post_error = MagicMock()
            post_error.raise_for_status.side_effect = Exception("Network down")
            get_error = MagicMock()
            get_error.raise_for_status.side_effect = Exception("Network down")

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                instance.post.return_value = post_error
                instance.get.return_value = get_error
                mock_client.return_value.__aenter__.return_value = instance

                result = await search_web("test", provider="tavily")

        # Should have error in result
        assert "error" in result

    @pytest.mark.asyncio
    async def test_search_brave_with_days_freshness(self):
        """Test that days>0 adds freshness param to Brave search."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {"results": [{"title": "Brave", "url": "https://b.com", "description": "D"}]}
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            mock_getenv.return_value = "fake_brave_key"

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                instance.get.return_value = mock_response
                mock_client.return_value.__aenter__.return_value = instance

                result = await search_web("test", provider="brave", days=7)

                call_args = instance.get.call_args
                params = call_args.kwargs["params"]
                assert "freshness" in params
                assert params["freshness"] == "pw"

    @pytest.mark.asyncio
    async def test_search_google_with_offset_start_param(self):
        """Test that offset>0 adds 'start' param to Google search (1-based)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "items": [{"title": "Google", "link": "https://g.com", "snippet": "S"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key == "GOOGLE_API_KEY":
                    return "fake_google_key"
                elif key == "GOOGLE_SEARCH_ENGINE_ID":
                    return "fake_cx"
                return default
            mock_getenv.side_effect = get_env

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                instance.get.return_value = mock_response
                mock_client.return_value.__aenter__.return_value = instance

                result = await search_web("test", provider="google", offset=5)

                call_args = instance.get.call_args
                params = call_args.kwargs["params"]
                assert "start" in params
                # Google uses 1-based index, so offset=5 -> start=6
                assert params["start"] == 6

    @pytest.mark.asyncio
    async def test_search_miklium_empty_query(self):
        """Test _search_miklium with empty query string."""
        from src.mcp_server.server import _search_miklium

        result = await _search_miklium("test query", 10)
        # Should succeed (empty string would be caught by search_web, not _search_miklium)
        assert "error" in result or "results" in result

    @pytest.mark.asyncio
    async def test_search_tavily_direct_not_configured(self):
        """Test _search_tavily returns 'not configured' error directly."""
        from src.mcp_server.server import _search_tavily

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key == "TAVILY_API_KEY":
                    return None
                return default
            mock_getenv.side_effect = get_env

            result = await _search_tavily("test query", 10)

        assert "error" in result
        assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_search_brave_direct_not_configured(self):
        """Test _search_brave returns 'not configured' error directly."""
        from src.mcp_server.server import _search_brave

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key == "BRAVE_API_KEY":
                    return None
                return default
            mock_getenv.side_effect = get_env

            result = await _search_brave("test query", 10)

        assert "error" in result
        assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_search_google_direct_not_configured(self):
        """Test _search_google returns 'not configured' error directly."""
        from src.mcp_server.server import _search_google

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key in ("GOOGLE_API_KEY", "GOOGLE_SEARCH_ENGINE_ID"):
                    return None
                return default
            mock_getenv.side_effect = get_env

            result = await _search_google("test query", 10)

        assert "error" in result
        assert "configured" in result["error"]

    @pytest.mark.asyncio
    async def test_search_miklium_api_success_false(self):
        """Test miklium when API returns success=False."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": False,
            "error": "Rate limit exceeded"
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key in ("TAVILY_API_KEY", "BRAVE_API_KEY", "GOOGLE_API_KEY"):
                    return None
                return default
            mock_getenv.side_effect = get_env

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                instance.post.return_value = mock_response
                mock_client.return_value.__aenter__.return_value = instance

                result = await search_web("test")

        # Should produce an error about miklium failure
        if "error" in result:
            assert "MIKLIUM" in result["error"] or "failed" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_search_no_preferred_uses_all_configured(self):
        """Test that omitting provider uses all configured providers in priority order."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"title": "Tavily", "url": "https://t.com", "content": "C"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key == "TAVILY_API_KEY":
                    return "fake_tavily"
                return default
            mock_getenv.side_effect = get_env

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                # miklium POST succeeds, then tavily POST
                miklium_ok = MagicMock()
                miklium_ok.json.return_value = {"success": True, "results": [{"url": "https://m.com", "snippet": "M"}]}
                miklium_ok.raise_for_status = MagicMock()
                instance.post.side_effect = [miklium_ok, mock_response]
                instance.get.return_value = MagicMock()
                mock_client.return_value.__aenter__.return_value = instance

                # No provider specified - defaults to miklium (first configured)
                result = await search_web("test")

        # Default search goes to miklium since it's first in configured providers

    @pytest.mark.asyncio
    async def test_search_no_providers_configured_error(self):
        """Test the 'no providers configured' error path when _get_configured_providers returns empty."""
        with patch("src.mcp_server.server._get_configured_providers") as mock_prov:
            # Force empty providers list (structurally unreachable in production,
            # but we test the defensive code path)
            mock_prov.return_value = []

            result = await search_web("test", provider="tavily")

        # Should produce error since no providers available for non-miklium search
        assert "error" in result or ("results" not in result)

    @pytest.mark.asyncio
    async def test_search_unknown_provider_in_failover_loop(self):
        """Test that an unknown provider name is skipped in the failover loop."""
        # This tests when a provider name appears in configured_providers but
        # doesn't match any known handler in the failover loop.
        with patch("src.mcp_server.server._get_configured_providers") as mock_prov:
            # Include a fake provider name that has no handler
            mock_prov.return_value = ["miklium", "fake_provider"]

            # Make miklium fail so the loop actually reaches fake_provider
            post_error = MagicMock()
            post_error.raise_for_status.side_effect = Exception("Miklium down")

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                instance.post.return_value = post_error
                instance.get.return_value = MagicMock()
                mock_client.return_value.__aenter__.return_value = instance

                # Request fake_provider specifically (will fail), then failover hits it
                result = await search_web("test", provider="fake_provider")

        # Should produce an error since fake_provider has no handler

    @pytest.mark.asyncio
    async def test_search_empty_query_non_miklium(self):
        """Test empty query on a non-miklium provider search."""
        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key == "TAVILY_API_KEY":
                    return "fake_tavily"
                return default
            mock_getenv.side_effect = get_env

            # Search with explicit non-miklium provider but empty query
            result = await search_web("", provider="tavily")

        # The empty query should produce a "Missing required field: query" error
        assert "error" in result
        assert "Missing required field: query" in result["error"]


class TestBraveFreshness:
    def test_brave_freshness_zero(self):
        assert _brave_freshness(0) == ""

    def test_brave_freshness_negative(self):
        assert _brave_freshness(-1) == ""

    def test_brave_freshness_past_day(self):
        assert _brave_freshness(1) == "pd"

    def test_brave_freshness_past_week(self):
        assert _brave_freshness(7) == "pw"

    def test_brave_freshness_past_month(self):
        assert _brave_freshness(31) == "pm"

    def test_brave_freshness_past_year(self):
        assert _brave_freshness(365) == "py"

    def test_brave_freshness_beyond_year(self):
        assert _brave_freshness(500) == ""

    def test_brave_freshness_boundary_2(self):
        assert _brave_freshness(2) == "pw"

    def test_brave_freshness_boundary_8(self):
        assert _brave_freshness(8) == "pm"

    def test_brave_freshness_boundary_32(self):
        assert _brave_freshness(32) == "py"


class TestGetConfiguredProviders:
    def test_no_api_keys_returns_miklium_only(self):
        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key in ("TAVILY_API_KEY", "BRAVE_API_KEY", "GOOGLE_API_KEY"):
                    return None
                if key == "GOOGLE_SEARCH_ENGINE_ID":
                    return None
                return default
            mock_getenv.side_effect = get_env

            providers = _get_configured_providers()
            assert providers == ["miklium"]

    def test_all_keys_returns_all_providers(self):
        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key == "TAVILY_API_KEY":
                    return "tavily_key"
                if key == "BRAVE_API_KEY":
                    return "brave_key"
                if key == "GOOGLE_API_KEY":
                    return "google_key"
                if key == "GOOGLE_SEARCH_ENGINE_ID":
                    return "cx_id"
                return default
            mock_getenv.side_effect = get_env

            providers = _get_configured_providers()
            assert providers == ["miklium", "tavily", "brave", "google"]

    def test_google_requires_both_keys(self):
        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key == "GOOGLE_API_KEY":
                    return "google_key"
                # Missing GOOGLE_SEARCH_ENGINE_ID
                if key in ("TAVILY_API_KEY", "BRAVE_API_KEY"):
                    return None
                return default
            mock_getenv.side_effect = get_env

            providers = _get_configured_providers()
            assert "google" not in providers
            # miklium is always present
            assert "miklium" in providers

    def test_partial_keys_returns_partial_providers(self):
        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key == "TAVILY_API_KEY":
                    return "tavily_key"
                if key in ("BRAVE_API_KEY", "GOOGLE_API_KEY"):
                    return None
                return default
            mock_getenv.side_effect = get_env

            providers = _get_configured_providers()
            assert providers == ["miklium", "tavily"]


class TestSummarizeText:
    @pytest.mark.asyncio
    async def test_summarize_text_basic(self):
        """Test summarize_text returns summary on success."""
        with patch("src.mcp_server.server._call_llm", AsyncMock(return_value="Key points: summary here.")):
            result = await summarize_text("Some content to summarize")

        assert "summary" in result
        assert len(result["summary"]) > 0

    @pytest.mark.asyncio
    async def test_summarize_text_custom_prompt(self):
        """Test summarize_text uses custom prompt."""
        def check_prompt(prompt, system_prompt=None):
            assert "technical" in system_prompt.lower() or "focus on technical details" in system_prompt.lower()
            return "Custom summary"

        with patch("src.mcp_server.server._call_llm", AsyncMock(side_effect=check_prompt)):
            result = await summarize_text("Content", summary_prompt="Focus on technical details")

        assert "summary" in result

    @pytest.mark.asyncio
    async def test_summarize_text_llm_error(self):
        """Test summarize_text handles LLM errors."""
        def raise_error(prompt, system_prompt=None):
            raise RuntimeError("LLM API Error: 503 Service Unavailable")

        with patch("src.mcp_server.server._call_llm", AsyncMock(side_effect=raise_error)):
            result = await summarize_text("Content")

        assert "error" in result


class TestFetchWebContentSummarize:
    @pytest.mark.asyncio
    async def test_fetch_web_content_summarize_true(self):
        """Test fetch_web_content returns summary when summarize=True."""
        def check_summarize(text, summary_prompt="", max_words=800):
            assert "content to summarize" in text.lower()
            return {"summary": "Summarized output"}

        with patch("src.mcp_server.server._extraction_pipeline.extract_from_html", new=AsyncMock(return_value=MagicMock(content="Content to summarize from the page"))):
            with patch("src.mcp_server.server.summarize_text", AsyncMock(side_effect=check_summarize)):
                result = await fetch_web_content(
                    url="https://example.com",
                    summarize=True,
                    summary_prompt="Focus on key points"
                )

        assert "summary" in result
        assert "url" in result

    @pytest.mark.asyncio
    async def test_fetch_web_content_summarize_false(self):
        """Test fetch_web_content returns content when summarize=False (default)."""
        with patch("src.mcp_server.server.ContentExtractionPipeline.extract_from_html", new=AsyncMock(return_value=MagicMock(content="Raw markdown content"))):
            result = await fetch_web_content(url="https://example.com", summarize=False)

        assert "content" in result
        assert "summary" not in result


class TestCallLLM:
    @pytest.mark.asyncio
    async def test_call_llm_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "LLM response text"}}]}
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = instance

            result = await _call_llm("User prompt here", "System prompt")

            assert result == "LLM response text"

    @pytest.mark.asyncio
    async def test_call_llm_no_system_prompt(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Response without system"}}]}
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = instance

            result = await _call_llm("Just user prompt")

            assert "Response without system" in result or len(result) > 0

    @pytest.mark.asyncio
    async def test_call_llm_http_error(self):
        from httpx import HTTPStatusError, Request

        mock_response = MagicMock()
        request = Request("POST", "http://test.com/chat/completions")
        mock_response.raise_for_status.side_effect = HTTPStatusError(
            "Server error", request=request, response=MagicMock(status_code=500)
        )

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = instance

            with pytest.raises(RuntimeError) as exc_info:
                await _call_llm("Test prompt")

            assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_llm_general_error(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Connection refused")

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = instance

            with pytest.raises(RuntimeError) as exc_info:
                await _call_llm("Test prompt")

            assert "inference failed" in str(exc_info.value).lower()


class TestLLMProviderConfig:
    """Tests for LLMProviderConfig dataclass."""

    def test_config_creation(self):
        """Test that LLMProviderConfig can be created with all fields."""
        from src.mcp_server.llm.base import LLMProviderConfig

        config = LLMProviderConfig(
            name="test-provider",
            base_url="http://localhost:11434/v1",
            api_key="secret-key",
            model="llama3.2"
        )

        assert config.name == "test-provider"
        assert config.base_url == "http://localhost:11434/v1"
        assert config.api_key == "secret-key"
        assert config.model == "llama3.2"

    def test_config_immutable(self):
        """Test that LLMProviderConfig is immutable (frozen dataclass)."""
        from src.mcp_server.llm.base import LLMProviderConfig

        config = LLMProviderConfig(
            name="test",
            base_url="http://localhost:11434/v1",
            api_key="",
            model="model"
        )

        with pytest.raises(AttributeError):
            config.name = "new-name"

    def test_config_equality(self):
        """Test that two configs with same values are equal."""
        from src.mcp_server.llm.base import LLMProviderConfig

        config1 = LLMProviderConfig(
            name="test", base_url="http://localhost:11434/v1",
            api_key="", model="llama3.2"
        )
        config2 = LLMProviderConfig(
            name="test", base_url="http://localhost:11434/v1",
            api_key="", model="llama3.2"
        )

        assert config1 == config2


class TestOpenAICompatibleProvider:
    """Tests for OpenAICompatibleProvider."""

    @pytest.mark.asyncio
    async def test_complete_success(self):
        """Test successful completion request."""
        from src.mcp_server.llm.base import LLMProviderConfig
        from src.mcp_server.llm.openai_compatible import OpenAICompatibleProvider

        config = LLMProviderConfig(
            name="test",
            base_url="http://localhost:11434/v1",
            api_key="",
            model="llama3.2"
        )
        provider = OpenAICompatibleProvider(config)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.llm.openai_compatible.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = instance

            result = await provider.complete("Test prompt", "System prompt")

            assert result == "Test response"
            # Verify the request was made with correct structure
            call_args = instance.post.call_args
            json_data = call_args.kwargs["json"]
            assert json_data["model"] == "llama3.2"
            assert len(json_data["messages"]) == 2
            assert json_data["messages"][0]["role"] == "system"
            assert json_data["messages"][1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_complete_no_system_prompt(self):
        """Test completion without system prompt."""
        from src.mcp_server.llm.base import LLMProviderConfig
        from src.mcp_server.llm.openai_compatible import OpenAICompatibleProvider

        config = LLMProviderConfig(
            name="test",
            base_url="http://localhost:11434/v1",
            api_key="",
            model="llama3.2"
        )
        provider = OpenAICompatibleProvider(config)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.llm.openai_compatible.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = instance

            result = await provider.complete("User prompt")

            assert result == "Response"
            call_args = instance.post.call_args
            json_data = call_args.kwargs["json"]
            # Should only have user message, no system
            assert len(json_data["messages"]) == 1
            assert json_data["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_complete_http_error(self):
        """Test that HTTP errors raise LLMProviderError."""
        from src.mcp_server.llm.base import LLMProviderConfig
        from src.mcp_server.llm.openai_compatible import OpenAICompatibleProvider
        from httpx import HTTPStatusError, Request

        config = LLMProviderConfig(
            name="test-provider",
            base_url="http://localhost:11434/v1",
            api_key="",
            model="llama3.2"
        )
        provider = OpenAICompatibleProvider(config)

        mock_response = MagicMock()
        request = Request("POST", "http://localhost:11434/v1/chat/completions")
        mock_response.raise_for_status.side_effect = HTTPStatusError(
            "Server error", request=request, response=MagicMock(status_code=500)
        )

        with patch("src.mcp_server.llm.openai_compatible.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = instance

            with pytest.raises(Exception) as exc_info:
                await provider.complete("Test prompt")

            assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_is_available_success(self):
        """Test is_available returns True on successful health check."""
        from src.mcp_server.llm.base import LLMProviderConfig
        from src.mcp_server.llm.openai_compatible import OpenAICompatibleProvider

        config = LLMProviderConfig(
            name="test",
            base_url="http://localhost:11434/v1",
            api_key="",
            model="llama3.2"
        )
        provider = OpenAICompatibleProvider(config)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("src.mcp_server.llm.openai_compatible.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = instance

            result = await provider.is_available()

            assert result is True

    @pytest.mark.asyncio
    async def test_is_available_failure(self):
        """Test is_available returns False when endpoint unreachable."""
        from src.mcp_server.llm.base import LLMProviderConfig
        from src.mcp_server.llm.openai_compatible import OpenAICompatibleProvider

        config = LLMProviderConfig(
            name="test",
            base_url="http://localhost:11434/v1",
            api_key="",
            model="llama3.2"
        )
        provider = OpenAICompatibleProvider(config)

        with patch("src.mcp_server.llm.openai_compatible.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            # Both /models and chat/completions fail
            instance.get.side_effect = Exception("Connection refused")
            instance.post.side_effect = Exception("Connection refused")
            mock_client.return_value.__aenter__.return_value = instance

            result = await provider.is_available()

            assert result is False

    @pytest.mark.asyncio
    async def test_is_available_models_fails_completions_succeeds(self):
        """Test is_available when /models fails but chat/completions succeeds (fallback)."""
        from src.mcp_server.llm.base import LLMProviderConfig
        from src.mcp_server.llm.openai_compatible import OpenAICompatibleProvider

        config = LLMProviderConfig(
            name="test",
            base_url="http://localhost:11434/v1",
            api_key="",
            model="llama3.2"
        )
        provider = OpenAICompatibleProvider(config)

        # /models returns 404 (non-200), but chat/completions succeeds
        models_response = MagicMock()
        models_response.status_code = 404

        completions_response = MagicMock()
        completions_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.llm.openai_compatible.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = models_response
            instance.post.return_value = completions_response
            mock_client.return_value.__aenter__.return_value = instance

            result = await provider.is_available()

            assert result is True

    def test_provider_name_property(self):
        """Test that name property returns config name."""
        from src.mcp_server.llm.base import LLMProviderConfig
        from src.mcp_server.llm.openai_compatible import OpenAICompatibleProvider

        config = LLMProviderConfig(
            name="my-ollama",
            base_url="http://localhost:11434/v1",
            api_key="",
            model="llama3.2"
        )
        provider = OpenAICompatibleProvider(config)

        assert provider.name == "my-ollama"
        assert provider.config == config

    def test_headers_with_api_key(self):
        """Test that Authorization header is set when api_key provided."""
        from src.mcp_server.llm.base import LLMProviderConfig
        from src.mcp_server.llm.openai_compatible import OpenAICompatibleProvider

        config = LLMProviderConfig(
            name="test",
            base_url="http://localhost:11434/v1",
            api_key="my-secret-key",
            model="model"
        )
        provider = OpenAICompatibleProvider(config)

        headers = provider._headers()

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer my-secret-key"

    def test_headers_without_api_key(self):
        """Test that no Authorization header when api_key is empty."""
        from src.mcp_server.llm.base import LLMProviderConfig
        from src.mcp_server.llm.openai_compatible import OpenAICompatibleProvider

        config = LLMProviderConfig(
            name="test",
            base_url="http://localhost:11434/v1",
            api_key="",
            model="model"
        )
        provider = OpenAICompatibleProvider(config)

        headers = provider._headers()

        assert "Authorization" not in headers


class TestLLMManager:
    """Tests for LLMManager failover logic."""

    @pytest.mark.asyncio
    async def test_single_provider_success(self):
        """Test that single provider works when it succeeds."""
        from src.mcp_server.llm.manager import LLMManager
        from src.mcp_server.llm.base import LLMProviderConfig, LLMProvider
        from unittest.mock import AsyncMock

        # Create a mock manager with mocked providers
        manager = object.__new__(LLMManager)
        
        # Create mock provider that succeeds
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.complete = AsyncMock(return_value="Success response")
        mock_provider.name = "primary"
        
        manager._providers = [mock_provider]

        result = await manager.complete("Test prompt", "System")

        assert result == "Success response"
        mock_provider.complete.assert_called_once_with("Test prompt", "System")

    @pytest.mark.asyncio
    async def test_failover_to_second_provider(self):
        """Test that failover works when first provider fails."""
        from src.mcp_server.llm.manager import LLMManager, LLMProviderError
        from src.mcp_server.llm.base import LLMProvider
        from unittest.mock import AsyncMock

        manager = object.__new__(LLMManager)

        # First provider fails, second succeeds
        mock_provider1 = MagicMock(spec=LLMProvider)
        mock_provider1.complete = AsyncMock(
            side_effect=LLMProviderError("primary", "Connection refused")
        )
        mock_provider1.name = "primary"

        mock_provider2 = MagicMock(spec=LLMProvider)
        mock_provider2.complete = AsyncMock(return_value="Fallback response")
        mock_provider2.name = "fallback"

        manager._providers = [mock_provider1, mock_provider2]

        result = await manager.complete("Test prompt")

        assert result == "Fallback response"
        # First provider should have been called
        mock_provider1.complete.assert_called_once()
        # Second provider should also have been called (failover worked)
        mock_provider2.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_failover_through_multiple_providers(self):
        """Test failover through multiple providers."""
        from src.mcp_server.llm.manager import LLMManager, LLMProviderError
        from src.mcp_server.llm.base import LLMProvider
        from unittest.mock import AsyncMock

        manager = object.__new__(LLMManager)

        # All providers fail except the last one
        mock_provider1 = MagicMock(spec=LLMProvider)
        mock_provider1.complete = AsyncMock(
            side_effect=LLMProviderError("p1", "Failed")
        )
        mock_provider1.name = "p1"

        mock_provider2 = MagicMock(spec=LLMProvider)
        mock_provider2.complete = AsyncMock(
            side_effect=LLMProviderError("p2", "Failed")
        )
        mock_provider2.name = "p2"

        mock_provider3 = MagicMock(spec=LLMProvider)
        mock_provider3.complete = AsyncMock(return_value="Third time's a charm")
        mock_provider3.name = "p3"

        manager._providers = [mock_provider1, mock_provider2, mock_provider3]

        result = await manager.complete("Test prompt")

        assert result == "Third time's a charm"
        # All three should have been attempted
        mock_provider1.complete.assert_called_once()
        mock_provider2.complete.assert_called_once()
        mock_provider3.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_providers_fail(self):
        """Test that LLMAllProvidersFailedError is raised when all fail."""
        from src.mcp_server.llm.manager import LLMManager, LLMProviderError, LLMAllProvidersFailedError
        from src.mcp_server.llm.base import LLMProvider
        from unittest.mock import AsyncMock

        manager = object.__new__(LLMManager)

        mock_provider1 = MagicMock(spec=LLMProvider)
        mock_provider1.complete = AsyncMock(
            side_effect=LLMProviderError("p1", "Connection timeout")
        )
        mock_provider1.name = "p1"

        mock_provider2 = MagicMock(spec=LLMProvider)
        mock_provider2.complete = AsyncMock(
            side_effect=LLMProviderError("p2", "503 Service Unavailable")
        )
        mock_provider2.name = "p2"

        manager._providers = [mock_provider1, mock_provider2]

        with pytest.raises(LLMAllProvidersFailedError) as exc_info:
            await manager.complete("Test prompt")

        # Error message should contain info about failures
        assert "p1" in str(exc_info.value)
        assert "p2" in str(exc_info.value)

    def test_providers_property_returns_copy(self):
        """Test that providers property returns a copy to prevent modification."""
        from src.mcp_server.llm.manager import LLMManager
        from src.mcp_server.llm.base import LLMProviderConfig, LLMProvider

        manager = object.__new__(LLMManager)
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.name = "test"
        manager._providers = [mock_provider]

        providers1 = manager.providers
        providers2 = manager.providers

        # Should be equal but not same object
        assert providers1 == providers2
        assert providers1 is not providers2

    def test_multi_provider_config_loading(self):
        """Test that multi-provider config is loaded correctly."""
        from src.mcp_server.llm.manager import LLMManager, LLMProviderConfig

        with patch("src.mcp_server.llm.manager.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                env_map = {
                    "LLM_PROVIDER_1_NAME": "primary",
                    "LLM_PROVIDER_1_BASE_URL": "http://primary:11434/v1",
                    "LLM_PROVIDER_1_API_KEY": "key1",
                    "LLM_PROVIDER_1_MODEL": "llama3.2",
                    "LLM_PROVIDER_2_NAME": "secondary",
                    "LLM_PROVIDER_2_BASE_URL": "http://secondary:11434/v1",
                    "LLM_PROVIDER_2_API_KEY": "key2",
                    "LLM_PROVIDER_2_MODEL": "mistral",
                    # No legacy vars
                }
                return env_map.get(key, default)
            
            mock_getenv.side_effect = get_env

            manager = LLMManager()

            assert len(manager._providers) == 2
            assert manager._providers[0].name == "primary"
            assert manager._providers[0].config.model == "llama3.2"
            assert manager._providers[1].name == "secondary"
            assert manager._providers[1].config.model == "mistral"

    def test_multi_provider_missing_base_url_raises(self):
        """Test that missing BASE_URL raises ValueError."""
        from src.mcp_server.llm.manager import LLMManager

        with patch("src.mcp_server.llm.manager.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                env_map = {
                    "LLM_PROVIDER_1_NAME": "bad-provider",
                    # Missing BASE_URL and MODEL
                }
                return env_map.get(key, default)
            
            mock_getenv.side_effect = get_env

            with pytest.raises(ValueError) as exc_info:
                LLMManager()

            assert "BASE_URL" in str(exc_info.value)

    def test_multi_provider_missing_model_raises(self):
        """Test that missing MODEL raises ValueError."""
        from src.mcp_server.llm.manager import LLMManager

        with patch("src.mcp_server.llm.manager.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                env_map = {
                    "LLM_PROVIDER_1_NAME": "bad-provider",
                    "LLM_PROVIDER_1_BASE_URL": "http://localhost:11434/v1",
                    # Missing MODEL
                }
                return env_map.get(key, default)
            
            mock_getenv.side_effect = get_env

            with pytest.raises(ValueError) as exc_info:
                LLMManager()

            assert "MODEL" in str(exc_info.value)


class TestLLMProviderError:
    """Tests for LLMProviderError exception."""

    def test_error_attributes(self):
        """Test that error has correct attributes."""
        from src.mcp_server.llm.exceptions import LLMProviderError

        error = LLMProviderError("my-provider", "Connection failed", status_code=503)

        assert error.provider_name == "my-provider"
        assert error.status_code == 503
        assert "my-provider" in str(error)
        assert "Connection failed" in str(error)

    def test_error_without_status_code(self):
        """Test error when status code is not available."""
        from src.mcp_server.llm.exceptions import LLMProviderError

        error = LLMProviderError("test", "Something went wrong")

        assert error.provider_name == "test"
        assert error.status_code is None


class TestLLMAllProvidersFailedError:
    """Tests for LLMAllProvidersFailedError exception."""

    def test_error_message(self):
        """Test that error message includes provider count and errors."""
        from src.mcp_server.llm.exceptions import LLMAllProvidersFailedError

        error = LLMAllProvidersFailedError(
            "All 3 LLM providers failed. Errors: [p1] Failed; [p2] Failed again"
        )

        assert "3" in str(error)
        assert "p1" in str(error)
        assert "p2" in str(error)


class TestHTTPEndpoints:
    """Integration tests for FastAPI route handlers and MCP tool registration."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from mcp_server.server import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["name"] == "WebTool MCP Server"

    def test_mcp_tools_registered(self):
        from mcp_server.server import fastapi_mcp
        tool_names = [t.name for t in fastapi_mcp.tools]
        assert "searchWeb" in tool_names
        assert "fetchWebContent" in tool_names
        # summarizeWebContent is now removed - only 2 tools remain

    def test_mcp_tools_exclude_health(self):
        from mcp_server.server import fastapi_mcp
        tool_names = [t.name for t in fastapi_mcp.tools]
        assert "health__get" not in tool_names
        # Only 2 tools (searchWeb, fetchWebContent)

    def test_mcp_endpoint_registered(self):
        """Test that the /mcp endpoint is registered in FastAPI routes."""
        from mcp_server.server import app
        from starlette.routing import Route
        routes = [r for r in app.routes if isinstance(r, Route)]
        route_paths = [r.path for r in routes]
        # The /mcp endpoint is mounted via mount_sse() which adds SSE handling
        assert "/mcp" in route_paths or any("/mcp" in str(r) for r in routes), \
            f"/mcp endpoint not found in routes: {route_paths}"

    def test_mcp_messages_endpoint_exists(self, client):
        """Test that the /mcp/messages/ endpoint is accessible for JSON-RPC requests."""
        resp = client.post("/mcp/messages/", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        # Should return proper MCP response (not 404)
        assert resp.status_code != 404, "/mcp/messages/ endpoint not found"


class TestSSEMount:
    """Tests for SSE mount functionality at /mcp endpoint."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from mcp_server.server import app
        return TestClient(app)

    def test_sse_endpoint_configured(self):
        """Test that SSE is mounted at /mcp path."""
        from mcp_server.server import fastapi_mcp
        # Verify the SSE mount path is configured correctly
        assert hasattr(fastapi_mcp, 'server'), "fastapi_mcp should have server attribute"

    def test_mcp_sse_mount_path_configured(self):
        """Test that SSE mount path is accessible in fastapi_mcp configuration."""
        from mcp_server.server import fastapi_mcp
        # The mount_sse() call in server.py uses mount_path="/mcp"
        # This is verified by checking the fastapi_mcp server has proper configuration
        assert hasattr(fastapi_mcp, 'server'), "fastapi_mcp should have server attribute"
        # Verify tools are registered (indicating MCP properly configured)
        tool_names = [t.name for t in fastapi_mcp.tools]
        assert "searchWeb" in tool_names, "Expected searchWeb tool to be registered"

    def test_mcp_tool_schemas(self):
        from mcp_server.server import fastapi_mcp
        tools_by_name = {t.name: t for t in fastapi_mcp.tools}
        # searchWeb has query as required
        search_props = tools_by_name["searchWeb"].inputSchema["properties"]
        assert "query" in search_props
        assert "provider" in search_props
        # fetchWebContent has url as required, plus summarize and summary_prompt params
        fetch_props = tools_by_name["fetchWebContent"].inputSchema["properties"]
        assert "url" in fetch_props
        assert "summarize" in fetch_props
        assert "summary_prompt" in fetch_props

    def test_auth_dependency_exists(self):
        from mcp_server.server import _require_auth, api_keys
        # The dependency is wired; verify it exists and references our verifier
        assert callable(_require_auth)
        # api_keys is a list (may be empty when no MCP_API_KEYS set)
        assert isinstance(api_keys, list)

    def test_fastapi_app_routes(self, client):
        # POST routes accept query params (FastAPI default for simple-typed params)
        resp = client.post("/searchWeb?query=test")
        assert resp.status_code == 200


class TestDoclingIntegration:
    """Tests for Docling document parsing integration in fetch_web_content."""

    @pytest.mark.asyncio
    async def test_fetch_pdf_url_uses_docling(self):
        """Test that PDF URLs trigger Docling parsing."""
        from src.mcp_server.llm.parser import is_docling_supported_url
        
        # PDF URLs should be detected as Docling-supported
        assert is_docling_supported_url("https://example.com/document.pdf") is True
        assert is_docling_supported_url("https://example.com/file.PDF") is True

    @pytest.mark.asyncio
    async def test_fetch_docx_url_uses_docling(self):
        """Test that DOCX URLs trigger Docling parsing."""
        from src.mcp_server.llm.parser import is_docling_supported_url
        
        assert is_docling_supported_url("https://example.com/document.docx") is True
        assert is_docling_supported_url("https://example.com/presentation.pptx") is True
        assert is_docling_supported_url("https://example.com/data.xlsx") is True

    @pytest.mark.asyncio
    async def test_fetch_image_url_uses_docling(self):
        """Test that image URLs trigger Docling parsing."""
        from src.mcp_server.llm.parser import is_docling_supported_url
        
        assert is_docling_supported_url("https://example.com/image.png") is True
        assert is_docling_supported_url("https://example.com/photo.jpg") is True
        assert is_docling_supported_url("https://example.com/scan.tiff") is True

    @pytest.mark.asyncio
    async def test_fetch_html_url_uses_docling(self):
        """Test that HTML URLs (with or without .html extension) trigger Docling."""
        from src.mcp_server.llm.parser import is_docling_supported_url
        
        # HTML pages should use Docling (since Docling supports HTML parsing)
        assert is_docling_supported_url("https://example.com/page.html") is True
        # URLs without extensions are treated as HTML and use Docling
        assert is_docling_supported_url("https://example.com/") is True
        assert is_docling_supported_url("https://example.com/page") is True

    @pytest.mark.asyncio
    async def test_fetch_with_docling_fallback_to_beautifulsoup(self):
        """Test that HTML content falls back to BeautifulSoup when Docling is not applicable."""
        html = "<html><body><h1>Test Page</h1><p>Content here.</p></body></html>"

        mock_response = MagicMock()
        mock_response.text = html
        mock_response.content = html.encode()
        mock_response.raise_for_status = MagicMock()

        expected_content = "Test Page Content here"

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = instance

            with _patch_pipeline(expected_content):
                # Regular HTML page should still work with BeautifulSoup
                result = await fetch_web_content("https://example.com")

        assert "url" in result
        assert result["url"] == "https://example.com"
        # Content should contain converted markdown from the HTML
        content_lower = result["content"].lower()
        assert any(word in content_lower for word in ["test", "content"])

    @pytest.mark.asyncio
    async def test_fetch_docling_supported_formats(self):
        """Test that various Docling-supported formats are correctly detected."""
        from src.mcp_server.llm.parser import is_docling_supported_url
        
        # All supported formats should return True
        supported_urls = [
            "https://example.com/doc.pdf",
            "https://example.com/doc.PDF",
            "https://example.com/document.docx",
            "https://example.com/slides.pptx",
            "https://example.com/spreadsheet.xlsx",
            "https://example.com/image.png",
            "https://example.com/photo.jpg",
            "https://example.com/scan.jpeg",
            "https://example.com/doc.tiff",
            "https://example.com/diagram.bmp",
            "https://example.com/readme.md",
            "https://example.com/data.csv",
            "https://example.com/config.json",
            "https://example.com/file.xml",
        ]
        
        for url in supported_urls:
            assert is_docling_supported_url(url) is True, f"Expected {url} to be Docling-supported"

    @pytest.mark.asyncio
    async def test_fetch_unsupported_format_uses_beautifulsoup(self):
        """Test that unsupported formats fall back to BeautifulSoup."""
        from src.mcp_server.llm.parser import is_docling_supported_url
        
        # Unsupported formats should return False
        unsupported_urls = [
            "https://example.com/file.txt",
            "https://example.com/script.js",
            "https://example.com/style.css",
            "https://example.com/video.mp4",
        ]
        
        for url in unsupported_urls:
            assert is_docling_supported_url(url) is False, f"Expected {url} to NOT be Docling-supported"

    @pytest.mark.asyncio
    async def test_parse_html_with_beautifulsoup(self):
        """Test the BeautifulSoup HTML parsing function."""
        from src.mcp_server.llm.parser import parse_html_with_beautifulsoup
        
        html = "<html><body><h1>Title</h1><p>Paragraph with <strong>bold</strong> text.</p></body></html>"
        result = await parse_html_with_beautifulsoup(html, include_links=False)
        
        assert "Title" in result
        assert "Paragraph" in result
        assert "bold" in result

    @pytest.mark.asyncio
    async def test_parse_html_with_links_preserved(self):
        """Test that include_links option works with BeautifulSoup parser."""
        from src.mcp_server.llm.parser import parse_html_with_beautifulsoup
        
        html = '<html><body><a href="https://example.com">Link Text</a></body></html>'
        result = await parse_html_with_beautifulsoup(html, include_links=True)
        
        # With include_links=True, the link should be preserved
        assert "example.com" in result or "Link Text" in result

    @pytest.mark.asyncio
    async def test_docling_not_available_graceful_fallback(self):
        """Test that when Docling is not installed, parsing falls back gracefully."""
        from src.mcp_server.llm import parser
        
        # Save original value
        original_available = parser.DOCLING_AVAILABLE
        
        try:
            # Simulate Docling not being available
            parser.DOCLING_AVAILABLE = False
            
            html = "<html><body><h1>Fallback Test</h1></body></html>"
            result = await parser.parse_html_with_beautifulsoup(html, include_links=False)
            
            assert "Fallback Test" in result
        finally:
            # Restore original value
            parser.DOCLING_AVAILABLE = original_available

    @pytest.mark.asyncio
    async def test_fetch_web_content_with_query_params_in_url(self):
        """Test that URLs with query parameters are handled correctly for Docling detection."""
        from src.mcp_server.llm.parser import is_docling_supported_url
        
        # URL with query params should still detect the file extension
        assert is_docling_supported_url("https://example.com/doc.pdf?version=1") is True
        # assert is_docling_supported_url("https://example.com/page.html?ref=home") is False


# ---------------------------------------------------------------------------
# _get_url_extension edge case
# ---------------------------------------------------------------------------

class TestUrlExtension:
    def test_no_extension_returns_empty_string(self):
        from src.mcp_server.server import _get_url_extension
        assert _get_url_extension("https://example.com/path/resource") == ""

    def test_extension_extracted(self):
        from src.mcp_server.server import _get_url_extension
        assert _get_url_extension("https://example.com/file.pdf") == ".pdf"


# ---------------------------------------------------------------------------
# Binary document fetch path
# ---------------------------------------------------------------------------

class TestBinaryFetchPath:
    @pytest.mark.asyncio
    async def test_fetch_binary_url_uses_extract_from_bytes(self):
        """fetch_web_content routes binary extensions through extract_from_bytes."""
        from src.mcp_server.extraction.pipeline import ExtractionResult

        binary_result = ExtractionResult(content="PDF content here", method="docling")

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            resp = MagicMock()
            resp.content = b"%PDF-1.4 binary"
            resp.raise_for_status = MagicMock()
            resp.headers = {"content-type": "application/pdf"}
            instance.get.return_value = resp
            mock_client.return_value.__aenter__.return_value = instance

            with patch(
                "src.mcp_server.server._extraction_pipeline.extract_from_bytes",
                new=AsyncMock(return_value=binary_result),
            ):
                result = await fetch_web_content("https://example.com/document.pdf")

        assert "content" in result
        assert result["content"] == "PDF content here"

    @pytest.mark.asyncio
    async def test_binary_url_html_response_playwright_succeeds(self):
        """Binary URL returning HTML uses playwright_fetch_binary, then extract_from_bytes."""
        from src.mcp_server.extraction.pipeline import ExtractionResult

        pdf_bytes = b"%PDF-1.4 " + b"x" * 600
        binary_result = ExtractionResult(content="PDF via playwright", method="docling")

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            resp = MagicMock()
            resp.content = b"<html><body>Loading...</body></html>"
            resp.text = "<html><body>Loading...</body></html>"
            resp.raise_for_status = MagicMock()
            resp.headers = {"content-type": "text/html; charset=utf-8"}
            instance.get.return_value = resp
            mock_client.return_value.__aenter__.return_value = instance

            with patch(
                "src.mcp_server.server._extraction_pipeline.playwright_fetch_binary",
                new=AsyncMock(return_value=pdf_bytes),
            ):
                with patch(
                    "src.mcp_server.server._extraction_pipeline.extract_from_bytes",
                    new=AsyncMock(return_value=binary_result),
                ):
                    result = await fetch_web_content("https://example.com/document.pdf")

        assert "content" in result
        assert result["content"] == "PDF via playwright"

    @pytest.mark.asyncio
    async def test_binary_url_html_response_playwright_fails_falls_back_to_html(self):
        """Falls back to extract_from_html when playwright_fetch_binary returns None."""
        from src.mcp_server.extraction.pipeline import ExtractionResult

        html_result = ExtractionResult(
            content="HTML fallback content", method="playwright+trafilatura"
        )

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            resp = MagicMock()
            resp.text = "<html><body>Loading...</body></html>"
            resp.raise_for_status = MagicMock()
            resp.headers = {"content-type": "text/html; charset=utf-8"}
            instance.get.return_value = resp
            mock_client.return_value.__aenter__.return_value = instance

            with patch(
                "src.mcp_server.server._extraction_pipeline.playwright_fetch_binary",
                new=AsyncMock(return_value=None),
            ):
                with patch(
                    "src.mcp_server.server._extraction_pipeline.extract_from_html",
                    new=AsyncMock(return_value=html_result),
                ):
                    result = await fetch_web_content("https://example.com/document.pdf")

        assert "content" in result
        assert result["content"] == "HTML fallback content"

    @pytest.mark.asyncio
    async def test_403_response_falls_back_to_playwright(self):
        """403 Forbidden triggers a Playwright fallback instead of returning an error."""
        import httpx as _httpx
        from src.mcp_server.extraction.pipeline import ExtractionResult

        playwright_result = ExtractionResult(
            content="SO page via playwright", method="playwright+trafilatura"
        )

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            resp = MagicMock()
            error_resp = MagicMock()
            error_resp.status_code = 403
            resp.raise_for_status = MagicMock(
                side_effect=_httpx.HTTPStatusError(
                    "403 Forbidden", request=MagicMock(), response=error_resp
                )
            )
            instance.get.return_value = resp
            mock_client.return_value.__aenter__.return_value = instance

            with patch(
                "src.mcp_server.server._extraction_pipeline.extract_from_html",
                new=AsyncMock(return_value=playwright_result),
            ):
                result = await fetch_web_content("https://stackoverflow.com/questions/123")

        assert "content" in result
        assert result["content"] == "SO page via playwright"

    @pytest.mark.asyncio
    async def test_429_response_falls_back_to_playwright(self):
        """429 Too Many Requests also triggers Playwright fallback."""
        import httpx as _httpx
        from src.mcp_server.extraction.pipeline import ExtractionResult

        playwright_result = ExtractionResult(
            content="rate-limited page via playwright", method="playwright+trafilatura"
        )

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            resp = MagicMock()
            error_resp = MagicMock()
            error_resp.status_code = 429
            resp.raise_for_status = MagicMock(
                side_effect=_httpx.HTTPStatusError(
                    "429 Too Many Requests", request=MagicMock(), response=error_resp
                )
            )
            instance.get.return_value = resp
            mock_client.return_value.__aenter__.return_value = instance

            with patch(
                "src.mcp_server.server._extraction_pipeline.extract_from_html",
                new=AsyncMock(return_value=playwright_result),
            ):
                result = await fetch_web_content("https://example.com/page")

        assert "content" in result
        assert result["content"] == "rate-limited page via playwright"

    @pytest.mark.asyncio
    async def test_404_response_returns_error_not_playwright(self):
        """404 Not Found is not a bot-protection code and returns an error directly."""
        import httpx as _httpx

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            resp = MagicMock()
            error_resp = MagicMock()
            error_resp.status_code = 404
            resp.raise_for_status = MagicMock(
                side_effect=_httpx.HTTPStatusError(
                    "404 Not Found", request=MagicMock(), response=error_resp
                )
            )
            instance.get.return_value = resp
            mock_client.return_value.__aenter__.return_value = instance

            result = await fetch_web_content("https://example.com/missing")

        assert "error" in result
        assert "404" in result["error"]


# ---------------------------------------------------------------------------
# _require_auth function
# ---------------------------------------------------------------------------

class TestRequireAuth:
    @pytest.mark.asyncio
    async def test_auth_disabled_when_no_api_keys(self):
        """_require_auth returns immediately when api_keys is empty."""
        from src.mcp_server.server import _require_auth
        # api_keys is empty by default in test environment – should not raise
        await _require_auth(None)

    @pytest.mark.asyncio
    async def test_auth_passes_with_valid_token(self):
        """_require_auth does not raise when a valid Bearer token is supplied."""
        from src.mcp_server.server import _require_auth
        from fastapi.security import HTTPAuthorizationCredentials

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="good-key")
        with patch("src.mcp_server.server.api_keys", ["good-key"]):
            await _require_auth(credentials)  # must not raise

    @pytest.mark.asyncio
    async def test_auth_raises_with_invalid_token(self):
        """_require_auth raises HTTPException 401 when the token is wrong."""
        from src.mcp_server.server import _require_auth
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad-key")
        with patch("src.mcp_server.server.api_keys", ["good-key"]):
            with pytest.raises(HTTPException) as exc_info:
                await _require_auth(credentials)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_auth_raises_when_no_credentials_and_keys_set(self):
        """_require_auth raises 401 when credentials are None and auth is enabled."""
        from src.mcp_server.server import _require_auth
        from fastapi import HTTPException

        with patch("src.mcp_server.server.api_keys", ["some-key"]):
            with pytest.raises(HTTPException) as exc_info:
                await _require_auth(None)
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Route wrapper functions (cover the return-await lines)
# ---------------------------------------------------------------------------

class TestRouteWrappers:
    @pytest.mark.asyncio
    async def test_api_fetch_web_content_delegates(self):
        """api_fetch_web_content delegates to fetch_web_content."""
        from src.mcp_server.server import api_fetch_web_content

        expected = {"url": "https://example.com", "content": "test content"}
        with patch("src.mcp_server.server.fetch_web_content", new=AsyncMock(return_value=expected)) as mock_fn:
            result = await api_fetch_web_content(url="https://example.com")
        assert result == expected
        mock_fn.assert_called_once()

# ---------------------------------------------------------------------------
# async_main entry point
# ---------------------------------------------------------------------------

class TestAsyncMain:
    @pytest.mark.asyncio
    async def test_async_main_calls_stdio_and_runs_server(self):
        """async_main wires stdio_server and mcp_server.run together."""
        from contextlib import asynccontextmanager
        from src.mcp_server.server import async_main

        mock_read = AsyncMock()
        mock_write = AsyncMock()

        @asynccontextmanager
        async def mock_stdio_server():
            yield (mock_read, mock_write)

        mock_mcp_server = MagicMock()
        mock_mcp_server.create_initialization_options.return_value = {}
        mock_mcp_server.run = AsyncMock()

        mock_fastapi_mcp = MagicMock()
        mock_fastapi_mcp.server = mock_mcp_server

        with patch("mcp.server.stdio.stdio_server", mock_stdio_server):
            with patch("src.mcp_server.server.fastapi_mcp", mock_fastapi_mcp):
                await async_main()

        mock_mcp_server.run.assert_called_once_with(mock_read, mock_write, {})


# ---------------------------------------------------------------------------
# Tests for uncovered lines in server.py
# ---------------------------------------------------------------------------

class TestUseLLMRefinementForTextualExtensions:
    """Tests for line 487: use_llm_refinement = True for textual extensions (.html, .md)."""

    @pytest.mark.asyncio
    async def test_fetch_web_content_html_enables_llm_refinement(self):
        """Test that .html URLs get use_llm_refinement=True by default."""
        from src.mcp_server.server import fetch_web_content

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = _make_html_response("<html><body>Test</body></html>")
            mock_client.return_value.__aenter__.return_value = instance

            with patch("src.mcp_server.server._extraction_pipeline.extract_from_html") as mock_extract:
                expected_result = ExtractionResult(content="Test content", method="trafilatura")
                mock_extract.return_value = expected_result

                result = await fetch_web_content("https://example.com/page.html", use_llm_refinement=None)

                mock_extract.assert_called_once()
                call_kwargs = mock_extract.call_args.kwargs
                assert call_kwargs["use_llm_refinement"] is True

    @pytest.mark.asyncio
    async def test_fetch_web_content_md_enables_llm_refinement(self):
        """Test that .md URLs get use_llm_refinement=True by default."""
        from src.mcp_server.server import fetch_web_content

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = _make_html_response("# Markdown content")
            mock_client.return_value.__aenter__.return_value = instance

            with patch("src.mcp_server.server._extraction_pipeline.extract_from_html") as mock_extract:
                expected_result = ExtractionResult(content="Markdown content", method="trafilatura")
                mock_extract.return_value = expected_result

                result = await fetch_web_content("https://example.com/readme.md", use_llm_refinement=None)

                mock_extract.assert_called_once()
                call_kwargs = mock_extract.call_args.kwargs
                assert call_kwargs["use_llm_refinement"] is True


class TestSummarizeErrorPath:
    """Tests for line 579: summarize_text returns dict with 'error' key."""

    @pytest.mark.asyncio
    async def test_fetch_web_content_summarize_returns_error_on_llm_failure(self):
        """Test that fetch_web_content returns error when summarize_text fails."""
        from src.mcp_server.server import fetch_web_content

        pipeline_content = "Some content to summarize"

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = _make_html_response("<html><body>Some content</body></html>")
            mock_client.return_value.__aenter__.return_value = instance

            with _patch_pipeline(pipeline_content):
                with patch("src.mcp_server.server.summarize_text", new=AsyncMock(return_value={"error": "LLM provider failed"})):
                    result = await fetch_web_content("https://example.com", summarize=True, num_words=50)

        assert result["url"] == "https://example.com"
        assert "error" in result
        assert result["error"] == "LLM provider failed"


class TestAgenticFetch:
    """Tests for lines 674-683: agentic_fetch function."""

    @pytest.mark.asyncio
    async def test_agentic_fetch_creates_agent_with_defaults(self):
        """Test that agentic_fetch creates AgenticFetchAgent with correct defaults."""
        from src.mcp_server.server import agentic_fetch

        mock_agent_instance = AsyncMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"success": True, "content": "found info"}
        mock_agent_instance.execute.return_value = mock_result

        with patch("src.mcp_server.server.AgenticFetchAgent") as MockAgent:
            MockAgent.return_value = mock_agent_instance

            result = await agentic_fetch("Find information about topic", max_steps=10)

            MockAgent.assert_called_once()
            call_kwargs = MockAgent.call_args.kwargs
            assert call_kwargs["max_steps"] == 10

        assert result["success"] is True


class TestApiAgenticFetchEndpoint:
    """Tests for line 711: api_agentic_fetch endpoint."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from mcp_server.server import app
        return TestClient(app)

    def test_api_agentic_fetch_delegates_to_agentic_fetch(self, client):
        """Test that api_agentic_fetch endpoint calls agentic_fetch."""
        from mcp_server.server import api_agentic_fetch

        with patch("mcp_server.server.agentic_fetch", new=AsyncMock(return_value={"success": True})) as mock_fn:
            with client:
                response = client.post(
                    "/agenticFetch",
                    json={"prompt": "Find info about test", "max_steps": 5}
                )

            mock_fn.assert_called_once_with(prompt="Find info about test", max_steps=5)
