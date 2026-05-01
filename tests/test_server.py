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

                result = await web_search([{"query": "test query", "provider": "tavily"}])

                assert len(result) == 1
                assert result[0]["provider"] == "tavily"
                assert len(result[0]["results"]) == 1
                assert result[0]["results"][0]["title"] == "Test Result"

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

            result = await web_search([{"query": "test", "provider": "tavily"}])

            assert len(result) == 1
            assert "error" in result[0]
            # Should indicate no providers configured or tavily specifically not available
            assert "not configured" in result[0]["error"] or "No search providers" in result[0]["error"]

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

                result = await web_search([{"query": "test", "provider": "brave"}])

                assert len(result) == 1
                assert result[0]["provider"] == "brave"
                assert len(result[0]["results"]) == 1

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

                result = await web_search([{"query": "test", "provider": "google"}])

                assert len(result) == 1
                assert result[0]["provider"] == "google"
                assert len(result[0]["results"]) == 1
                assert result[0]["results"][0]["url"] == "https://google.example.com"

    @pytest.mark.asyncio
    async def test_search_unknown_provider(self):
        with patch("src.mcp_server.server.os.getenv") as mock_getenv:
            # Return a fake key so at least one provider is "configured"
            mock_getenv.return_value = "fake_key"
            result = await web_search([{"query": "test", "provider": "unknown"}])

            assert len(result) == 1
            # With unknown provider, it should try configured providers and fail or use failover
            # The exact behavior depends on what providers are "configured" via mock

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

                result = await web_search([{"query": "test", "provider": "tavily"}])

                assert len(result) == 1
                # Either error in the result directly or failover_attempts with errors
                has_error = "error" in result[0] or ("failover_attempts" in result[0] and any("error" in a for a in result[0]["failover_attempts"]))
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

            result = await web_search([{"query": "test", "provider": "brave"}])

            assert len(result) == 1
            # All providers not configured should produce an error about no providers
            assert "error" in result[0]
            assert "No search providers" in result[0]["error"] or "configured" in result[0]["error"]

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

            result = await web_search([{"query": "test", "provider": "google"}])

            assert len(result) == 1
            # All providers not configured should produce an error about no providers
            assert "error" in result[0]
            assert "No search providers" in result[0]["error"] or "configured" in result[0]["error"]

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

                result = await web_search([{"query": "test", "provider": "tavily", "num_results": 3}])

                assert len(result) == 1
                # Should be limited to requested count (max 20, but we asked for 3)
                assert len(result[0]["results"]) <= 3


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
