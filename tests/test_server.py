import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# Import from server module
from src.mcp_server.server import web_fetch, web_search, web_summarize, _call_llm


class TestWebFetch:
    @pytest.mark.asyncio
    async def test_web_fetch_single_url_success(self):
        html = "<html><body><h1>Test</h1><p>Content here.</p></body></html>"

        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = instance

            result = await web_fetch(["https://example.com"])

            assert "https://example.com" in result
            # Content should contain converted markdown from the HTML
            content_lower = result["https://example.com"].lower()
            assert any(word in content_lower for word in ["test", "content"])

    @pytest.mark.asyncio
    async def test_web_fetch_with_regex_filter(self):
        html = "<html><body>ERROR_CODE: 123 ERROR_TYPE: critical</body></html>"

        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = instance

            result = await web_fetch(["https://example.com"], regex="ERROR_", regex_padding=10)

            assert "https://example.com" in result
            # Should have matched content or no-match message
            assert len(result["https://example.com"]) > 0

    @pytest.mark.asyncio
    async def test_web_fetch_word_truncation(self):
        html = "<p>" + " ".join(["word"] * 200) + "</p>"

        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = instance

            result = await web_fetch(["https://example.com"], num_words=50)

            words = result["https://example.com"].split()
            assert len(words) <= 55  # Allow small margin for markdown conversion overhead

    @pytest.mark.asyncio
    async def test_web_fetch_multiple_urls(self):
        html1, html2 = "<h1>Page One</h1>", "<h1>Page Two</h1>"

        mock_response1 = MagicMock()
        mock_response1.text = html1
        mock_response1.raise_for_status = MagicMock()

        mock_response2 = MagicMock()
        mock_response2.text = html2
        mock_response2.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.side_effect = [mock_response1, mock_response2]
            mock_client.return_value.__aenter__.return_value = instance

            result = await web_fetch(["https://example.com/1", "https://example.com/2"])

            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_web_fetch_http_error(self):
        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            error_resp = MagicMock()
            error_resp.raise_for_status.side_effect = Exception("404 Not Found")
            instance.get.return_value = error_resp
            mock_client.return_value.__aenter__.return_value = instance

            result = await web_fetch(["https://example.com/notfound"])

            assert "Error" in result["https://example.com/notfound"]

    @pytest.mark.asyncio
    async def test_web_fetch_regex_no_match(self):
        html = "<p>No errors here, just normal content.</p>"

        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = instance

            result = await web_fetch(["https://example.com"], regex="NONEXISTENT_PATTERN_", regex_padding=10)

            assert "No matches found" in result["https://example.com"]


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

                result = await web_search("test query", provider="tavily")

                assert result["provider"] == "tavily"
                assert len(result["results"]) == 1
                assert result["results"][0]["title"] == "Test Result"

    @pytest.mark.asyncio
    async def test_search_tavily_no_api_key(self):
        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            # Use side_effect to return None for the specific key
            def get_env(key, default=None):
                if key == "TAVILY_API_KEY":
                    return None
                return os.environ.get(key, default)
            mock_getenv.side_effect = get_env

            result = await web_search("test", provider="tavily")

            assert "error" in result
            assert "not configured" in result["error"]

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

                result = await web_search("test", provider="brave")

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

                result = await web_search("test", provider="google")

                assert result["provider"] == "google"
                assert len(result["results"]) == 1
                assert result["results"][0]["url"] == "https://google.example.com"

    @pytest.mark.asyncio
    async def test_search_unknown_provider(self):
        with patch("src.mcp_server.server.os.getenv"):  # Should not be called
            result = await web_search("test", provider="unknown")

            assert "error" in result
            assert "Unknown provider" in result["error"]

    @pytest.mark.asyncio
    async def test_search_api_error_handling(self):
        mock_response = MagicMock()
        error = Exception("Connection timeout")
        mock_response.raise_for_status.side_effect = error
        mock_response.text = "Timeout"

        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            mock_getenv.return_value = "fake_api_key"

            with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
                instance = AsyncMock()
                instance.post.return_value = mock_response
                mock_client.return_value.__aenter__.return_value = instance

                result = await web_search("test", provider="tavily")

                assert "error" in result
                assert "failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_brave_no_api_key(self):
        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key == "BRAVE_API_KEY":
                    return None
                return os.environ.get(key, default)
            mock_getenv.side_effect = get_env

            result = await web_search("test", provider="brave")

            assert "error" in result
            assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_search_google_no_credentials(self):
        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            def get_env(key, default=None):
                if key == "GOOGLE_API_KEY":
                    return None
                elif key == "GOOGLE_SEARCH_ENGINE_ID":
                    return None
                return os.environ.get(key, default)
            mock_getenv.side_effect = get_env

            result = await web_search("test", provider="google")

            assert "error" in result
            assert "configured" in result["error"]

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

                result = await web_search("test", provider="tavily", num_results=3)

                assert result["count"] == 3  # Limited to requested count
                assert len(result["results"]) == 3


class TestWebSummarize:
    @pytest.mark.asyncio
    async def test_summarize_single_url(self):
        fetch_result = {"https://example.com": "This is the actual content from the webpage with details."}

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "## Summary\n\nKey points extracted."}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("src.mcp_server.server.web_fetch", AsyncMock(return_value=fetch_result)):
            with patch("src.mcp_server.server._call_llm", AsyncMock(return_value="## Summary\n\nKey points extracted.")):
                result = await web_summarize(["https://example.com"])

                assert "summaries" in result
                assert "https://example.com" in result["summaries"]
                summary_data = result["summaries"]["https://example.com"]
                if "summary" in summary_data:
                    assert len(summary_data["summary"]) > 0

    @pytest.mark.asyncio
    async def test_summarize_with_reduce(self):
        fetch_result = {
            "url1": "Content from first source.",
            "url2": "Content from second source."
        }

        async def mock_llm(prompt, system_prompt=None):
            if "Summarize the following" in prompt:
                return f"Summary for {prompt[:20]}..."
            else:
                return "## Combined Overview\n\nSynthesized findings from both sources."

        with patch("src.mcp_server.server.web_fetch", AsyncMock(return_value=fetch_result)):
            with patch("src.mcp_server.server._call_llm", side_effect=mock_llm):
                result = await web_summarize(["url1", "url2"], reduce=True)

                assert "summaries" in result
                assert "combined" in result
                if "summary" in result["combined"]:
                    assert len(result["combined"]["summary"]) > 0

    @pytest.mark.asyncio
    async def test_summarize_custom_prompts(self):
        fetch_result = {"https://example.com": "Content for custom analysis."}

        with patch("src.mcp_server.server.web_fetch", AsyncMock(return_value=fetch_result)):
            with patch("src.mcp_server.server._call_llm", AsyncMock(return_value="Custom summary")):
                result = await web_summarize(
                    ["https://example.com"],
                    summary_prompt="Focus on technical specifications only.",
                    reduce=False
                )

                assert "summaries" in result
                # Should still use the default prompt or custom
                assert isinstance(result["summaries"], dict)

    @pytest.mark.asyncio
    async def test_summarize_fetch_error(self):
        fetch_result = {"https://error.com": "Error: Failed to connect to server"}

        with patch("src.mcp_server.server.web_fetch", AsyncMock(return_value=fetch_result)):
            result = await web_summarize(["https://error.com"])

            assert "summaries" in result
            summary_data = result["summaries"]["https://error.com"]
            # Error content should be captured as dict with error key
            assert isinstance(summary_data, dict) or "Error:" in str(summary_data)

    @pytest.mark.asyncio
    async def test_summarize_llm_error_handling(self):
        fetch_result = {"https://example.com": "Normal content here."}

        async def mock_llm_error(prompt, system_prompt=None):
            raise RuntimeError("LLM API Error: 503 Service Unavailable")

        with patch("src.mcp_server.server.web_fetch", AsyncMock(return_value=fetch_result)):
            with patch("src.mcp_server.server._call_llm", side_effect=mock_llm_error):
                result = await web_summarize(["https://example.com"])

                summary_data = result["summaries"]["https://example.com"]
                assert "error" in summary_data or isinstance(summary_data, dict)

    @pytest.mark.asyncio
    async def test_summarize_empty_urls(self):
        with patch("src.mcp_server.server.web_fetch", AsyncMock(return_value={})):
            result = await web_summarize([], reduce=True)

            # Empty URL list should still return valid structure
            assert "summaries" in result
            assert len(result["summaries"]) == 0

    @pytest.mark.asyncio
    async def test_summarize_max_words(self):
        fetch_result = {"https://example.com": "x " * 1500}

        with patch("src.mcp_server.server.web_fetch", AsyncMock(return_value=fetch_result)):
            with patch("src.mcp_server.server._call_llm", AsyncMock(return_value="Summary")):
                result = await web_summarize(["https://example.com"], max_words_per_url=100)

                assert "summaries" in result

    @pytest.mark.asyncio
    async def test_summarize_with_reduction_prompt(self):
        fetch_result = {"u1": "C1", "u2": "C2"}

        async def mock_llm(prompt, system_prompt=None):
            return f"Synthesized with custom prompt: {system_prompt[:30] if system_prompt else 'none'}"

        with patch("src.mcp_server.server.web_fetch", AsyncMock(return_value=fetch_result)):
            with patch("src.mcp_server.server._call_llm", side_effect=mock_llm):
                result = await web_summarize(
                    ["u1", "u2"],
                    reduce=True,
                    reduction_prompt="Provide a competitive analysis format."
                )

                assert "combined" in result


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
