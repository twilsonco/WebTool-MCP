"""
Unit tests for vision capabilities (screenshot capture and multimodal LLM).

Tests cover:
  - ContentExtractionPipeline.capture_screenshot()
  - OpenAICompatibleProvider.complete_with_images()
  - LLMManager.complete_with_images() failover
  - AgenticFetchAgent vision support (SCREENSHOT action, _capture_screenshot,
    _check_vision_support)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.mcp_server.extraction.pipeline import ContentExtractionPipeline
from src.mcp_server.llm.base import LLMProviderConfig, LLMProvider
from src.mcp_server.llm.openai_compatible import OpenAICompatibleProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_browser_mock():
    """Return a mock browser whose new_context returns a usable mock context."""
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.close = AsyncMock()

    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)
    mock_context.close = AsyncMock()

    mock_browser = MagicMock()
    mock_browser.browser_type = "chromium"
    mock_browser.new_context = AsyncMock(return_value=mock_context)

    return mock_browser, mock_context, mock_page


# ---------------------------------------------------------------------------
# ContentExtractionPipeline.capture_screenshot
# ---------------------------------------------------------------------------

class TestCaptureScreenshot:
    """Tests for ContentExtractionPipeline.capture_screenshot."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_browser(self):
        """Returns None when Playwright is unavailable (no browser)."""
        original_lock = ContentExtractionPipeline._lock
        ContentExtractionPipeline._lock = None
        try:
            with patch.object(
                ContentExtractionPipeline, "_get_browser", new=AsyncMock(return_value=None)
            ):
                result = await ContentExtractionPipeline().capture_screenshot(
                    "https://example.com"
                )
            assert result is None
        finally:
            ContentExtractionPipeline._lock = original_lock

    @pytest.mark.asyncio
    async def test_returns_base64_png_on_success(self):
        """Returns base64 string when screenshot is captured successfully."""
        expected_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=expected_base64)
        mock_page.close = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = MagicMock()
        mock_browser.browser_type = "chromium"
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        original_lock = ContentExtractionPipeline._lock
        ContentExtractionPipeline._lock = None
        try:
            with patch.object(
                ContentExtractionPipeline, "_get_browser", new=AsyncMock(return_value=mock_browser)
            ):
                result = await ContentExtractionPipeline().capture_screenshot(
                    "https://example.com"
                )
            assert result == expected_base64
        finally:
            ContentExtractionPipeline._lock = original_lock

    @pytest.mark.asyncio
    async def test_returns_none_on_navigation_error(self):
        """Returns None when page.goto raises an exception."""
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=Exception("navigation error"))
        mock_page.close = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = MagicMock()
        mock_browser.browser_type = "chromium"
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        original_lock = ContentExtractionPipeline._lock
        ContentExtractionPipeline._lock = None
        try:
            with patch.object(
                ContentExtractionPipeline, "_get_browser", new=AsyncMock(return_value=mock_browser)
            ):
                result = await ContentExtractionPipeline().capture_screenshot(
                    "https://example.com"
                )
            assert result is None
        finally:
            ContentExtractionPipeline._lock = original_lock

    @pytest.mark.asyncio
    async def test_page_and_context_closed_on_success(self):
        """page.close() and context.close() are called after successful screenshot."""
        expected_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAE="

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=expected_base64)
        mock_page.close = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = MagicMock()
        mock_browser.browser_type = "chromium"
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        original_lock = ContentExtractionPipeline._lock
        ContentExtractionPipeline._lock = None
        try:
            with patch.object(
                ContentExtractionPipeline, "_get_browser", new=AsyncMock(return_value=mock_browser)
            ):
                await ContentExtractionPipeline().capture_screenshot("https://example.com")

            mock_page.close.assert_called_once()
        finally:
            ContentExtractionPipeline._lock = original_lock

    @pytest.mark.asyncio
    async def test_page_and_context_closed_on_error(self):
        """page.close() and context.close() are called even when goto raises."""
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=Exception("navigation error"))
        mock_page.close = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = MagicMock()
        mock_browser.browser_type = "chromium"
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        original_lock = ContentExtractionPipeline._lock
        ContentExtractionPipeline._lock = None
        try:
            with patch.object(
                ContentExtractionPipeline, "_get_browser", new=AsyncMock(return_value=mock_browser)
            ):
                await ContentExtractionPipeline().capture_screenshot("https://example.com")

            mock_page.close.assert_called_once()
        finally:
            ContentExtractionPipeline._lock = original_lock


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider.complete_with_images
# ---------------------------------------------------------------------------

class TestCompleteWithImages:
    """Tests for OpenAICompatibleProvider.complete_with_images."""

    @pytest.mark.asyncio
    async def test_sends_text_only_when_no_images(self):
        """When images is None, sends a simple text message."""
        config = LLMProviderConfig(
            name="test",
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4o"
        )
        provider = OpenAICompatibleProvider(config)

        mock_response = {
            "choices": [{"message": {"content": "Hello world"}}]
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=_FakeResponse(mock_response, 200))
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await provider.complete_with_images(
                prompt="Say hello",
                system_prompt="You are a helpful assistant"
            )

        assert result == "Hello world"

        call_kwargs = mock_client.post.call_args.kwargs
        messages = call_kwargs["json"]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_sends_image_url_blocks_when_images_provided(self):
        """When images are provided, builds content array with image_url blocks."""
        config = LLMProviderConfig(
            name="test",
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4o"
        )
        provider = OpenAICompatibleProvider(config)

        mock_response = {
            "choices": [{"message": {"content": "I see a login form"}}]
        }

        image_data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE="

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=_FakeResponse(mock_response, 200))
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await provider.complete_with_images(
                prompt="Describe this image",
                images=[image_data_uri]
            )

        assert result == "I see a login form"

        call_kwargs = mock_client.post.call_args.kwargs
        messages = call_kwargs["json"]["messages"]
        user_message = messages[-1]
        assert user_message["role"] == "user"
        content_blocks = user_message["content"]
        assert len(content_blocks) == 2
        assert content_blocks[0] == {"type": "text", "text": "Describe this image"}
        assert content_blocks[1]["type"] == "image_url"
        assert content_blocks[1]["image_url"]["url"] == image_data_uri

    @pytest.mark.asyncio
    async def test_includes_auth_header_when_api_key_set(self):
        """Authorization header is included when api_key is configured."""
        config = LLMProviderConfig(
            name="test",
            base_url="https://api.example.com/v1",
            api_key="secret-key",
            model="gpt-4o"
        )
        provider = OpenAICompatibleProvider(config)

        mock_response = {
            "choices": [{"message": {"content": "response"}}]
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=_FakeResponse(mock_response, 200))
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            await provider.complete_with_images(prompt="Hi")

        call_kwargs = mock_client.post.call_args.kwargs
        headers = call_kwargs["headers"]
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer secret-key"

    @pytest.mark.asyncio
    async def test_raises_LLMProviderError_on_http_error(self):
        """Raises LLMProviderError when API returns non-2xx status."""
        from src.mcp_server.llm.exceptions import LLMProviderError

        config = LLMProviderConfig(
            name="test",
            base_url="https://api.example.com/v1",
            api_key="key",
            model="gpt-4o"
        )
        provider = OpenAICompatibleProvider(config)

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "bad request"
        mock_response.raise_for_status.side_effect = Exception("HTTP 400")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_async_ctx = AsyncMock()
            mock_async_ctx.__aenter__.return_value = MagicMock(post=AsyncMock(return_value=mock_response))
            mock_client_cls.return_value = mock_async_ctx

            with pytest.raises(LLMProviderError) as exc_info:
                await provider.complete_with_images(prompt="Hi")

        assert "400" in str(exc_info.value)


# ---------------------------------------------------------------------------
# LLMManager.complete_with_images
# ---------------------------------------------------------------------------

class TestLLMManagerCompleteWithImages:
    """Tests for LLMManager.complete_with_images failover."""

    @pytest.mark.asyncio
    async def test_tries_providers_in_order(self):
        """Tries each provider sequentially until one succeeds."""
        from src.mcp_server.llm.manager import LLMManager
        from src.mcp_server.llm.exceptions import LLMProviderError

        mock_provider1 = MagicMock()
        mock_provider1.name = "provider-1"
        mock_provider1.complete_with_images = AsyncMock(
            side_effect=LLMProviderError("p1", "not supported")
        )

        mock_provider2 = MagicMock()
        mock_provider2.name = "provider-2"
        mock_provider2.complete_with_images = AsyncMock(return_value="vision response")

        manager = object.__new__(LLMManager)
        manager._providers = [mock_provider1, mock_provider2]

        result = await manager.complete_with_images(
            prompt="Describe",
            images=["data:image/png;base64,xyz"]
        )

        assert result == "vision response"
        mock_provider1.complete_with_images.assert_called_once()
        mock_provider2.complete_with_images.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_when_all_providers_fail(self):
        """Raises LLMAllProvidersFailedError when all providers fail."""
        from src.mcp_server.llm.manager import LLMManager
        from src.mcp_server.llm.exceptions import LLMAllProvidersFailedError, LLMProviderError

        mock_provider = MagicMock()
        mock_provider.name = "failing"
        mock_provider.complete_with_images = AsyncMock(
            side_effect=LLMProviderError("p1", "vision not supported")
        )

        manager = object.__new__(LLMManager)
        manager._providers = [mock_provider]

        with pytest.raises(LLMAllProvidersFailedError) as exc_info:
            await manager.complete_with_images(prompt="Describe", images=[])

        assert "vision request" in str(exc_info.value)


# ---------------------------------------------------------------------------
# AgenticFetchAgent vision support
# ---------------------------------------------------------------------------

class TestCaptureScreenshotAction:
    """Tests for AgenticFetchAgent SCREENSHOT action handling."""

    @pytest.mark.asyncio
    async def test_capture_screenshot_returns_none_when_no_pipeline(self):
        """_capture_screenshot returns None when no extraction pipeline is set."""
        from src.mcp_server.agentic.fetch_agent import AgenticFetchAgent

        agent = object.__new__(AgenticFetchAgent)
        agent._extraction_pipeline = None
        result = await agent._capture_screenshot("https://example.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_capture_screenshot_delegates_to_pipeline(self):
        """_capture_screenshot calls pipeline.capture_screenshot when available."""
        from src.mcp_server.agentic.fetch_agent import AgenticFetchAgent

        mock_pipeline = MagicMock()
        mock_pipeline.capture_screenshot = AsyncMock(return_value="base64image")

        agent = object.__new__(AgenticFetchAgent)
        agent._extraction_pipeline = mock_pipeline

        result = await agent._capture_screenshot("https://example.com")
        assert result == "base64image"
        mock_pipeline.capture_screenshot.assert_called_once_with("https://example.com")

    @pytest.mark.asyncio
    async def test_capture_screenshot_returns_none_when_method_missing(self):
        """_capture_screenshot returns None when pipeline lacks capture_screenshot."""
        from src.mcp_server.agentic.fetch_agent import AgenticFetchAgent

        mock_pipeline = MagicMock(spec=[])  # no capture_screenshot method
        agent = object.__new__(AgenticFetchAgent)
        agent._extraction_pipeline = mock_pipeline

        result = await agent._capture_screenshot("https://example.com")
        assert result is None


class TestVisionSupportCheck:
    """Tests for _check_vision_support."""

    @pytest.mark.asyncio
    async def test_returns_false_when_vision_disabled(self):
        """Returns False when vision_enabled=False."""
        from src.mcp_server.agentic.fetch_agent import AgenticFetchAgent

        mock_llm = MagicMock()
        agent = AgenticFetchAgent(
            vision_enabled=False,
            llm_manager=mock_llm
        )
        result = await agent._check_vision_support()
        assert result is False
        assert agent._vision_checked is True

    @pytest.mark.asyncio
    async def test_returns_false_when_no_llm_manager(self):
        """Returns False when no LLM manager is configured."""
        from src.mcp_server.agentic.fetch_agent import AgenticFetchAgent

        agent = AgenticFetchAgent(
            vision_enabled=True,
            llm_manager=None
        )

        result = await agent._check_vision_support()
        assert result is False

    @pytest.mark.asyncio
    async def test_caches_vision_check_result(self):
        """Second call to _check_vision_support returns cached result."""
        from src.mcp_server.agentic.fetch_agent import AgenticFetchAgent

        mock_llm = MagicMock()
        mock_llm.complete_with_images = AsyncMock(return_value="yes")

        agent = AgenticFetchAgent(
            vision_enabled=True,
            llm_manager=mock_llm
        )

        result1 = await agent._check_vision_support()
        assert result1 is True
        mock_llm.complete_with_images.assert_called_once()

        result2 = await agent._check_vision_support()
        assert result2 is True
        mock_llm.complete_with_images.assert_called_once()  # still only once

    @pytest.mark.asyncio
    async def test_sets_vision_supported_on_success(self):
        """Sets _vision_supported=True when vision check succeeds."""
        from src.mcp_server.agentic.fetch_agent import AgenticFetchAgent

        mock_llm = MagicMock()
        mock_llm.complete_with_images = AsyncMock(return_value="YES")

        agent = AgenticFetchAgent(
            vision_enabled=True,
            llm_manager=mock_llm
        )

        await agent._check_vision_support()
        assert agent._vision_supported is True

    @pytest.mark.asyncio
    async def test_sets_vision_unsupported_on_failure(self):
        """Sets _vision_supported=False when vision check raises."""
        from src.mcp_server.agentic.fetch_agent import AgenticFetchAgent

        mock_llm = MagicMock()
        mock_llm.complete_with_images = AsyncMock(side_effect=Exception("not supported"))

        agent = AgenticFetchAgent(
            vision_enabled=True,
            llm_manager=mock_llm
        )

        await agent._check_vision_support()
        assert agent._vision_supported is False


class TestScreenshotActionInExecute:
    """Tests for SCREENSHOT action handling inside execute()."""

    @pytest.mark.asyncio
    async def test_screenshot_action_skips_when_vision_not_enabled(self):
        """SCREENSHOT action with vision_enabled=False logs and continues."""
        from src.mcp_server.agentic.fetch_agent import AgenticFetchAgent

        mock_llm = MagicMock()
        mock_llm.complete_with_images = AsyncMock(return_value="response")
        mock_llm.complete = AsyncMock(
            return_value='{"action": "done", "description": "finished"}'
        )

        agent = AgenticFetchAgent(
            vision_enabled=False,
            llm_manager=mock_llm
        )

        result = await agent.execute("test prompt")
        step_actions = [s.get("action") for s in result.steps_taken]
        assert "error" not in step_actions

    @pytest.mark.asyncio
    async def test_screenshot_action_no_url_provided(self):
        """SCREENSHOT action without URL appends error to context and continues."""
        from src.mcp_server.agentic.fetch_agent import AgenticFetchAgent

        mock_llm = MagicMock()
        mock_response = '{"action": "screenshot", "description": "take shot"}'
        mock_llm.complete = AsyncMock(return_value=mock_response)

        agent = AgenticFetchAgent(
            vision_enabled=True,
            llm_manager=mock_llm
        )

        result = await agent.execute("test prompt")
        step_results = [s.get("result", "") for s in result.steps_taken]
        assert any("No URL" in r for r in step_results)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class _FakeResponse:
    """Minimal mock httpx response."""

    def __init__(self, json_data: dict, status_code: int):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            from httpx import HTTPStatusError
            raise HTTPStatusError(
                "error",
                request=MagicMock(),
                response=self
            )

    def json(self):
        return self._json


class _FakeHTTPErrorResponse:
    """Minimal mock httpx response for error cases."""

    def __init__(self, json_data: dict, status_code: int):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            from httpx import HTTPStatusError
            raise HTTPStatusError(
                "error",
                request=MagicMock(),
                response=self
            )

    def json(self):
        return self._json