"""
Tests for Firecrawl integration in agentic fetch mode.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestFirecrawlFetchAgent:
    """Tests for AgenticFetchAgent with Firecrawl integration."""

    @pytest.mark.asyncio
    async def test_agent_initialization_with_firecrawl_client(self):
        """Test agent can be initialized with firecrawl_client."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_firecrawl = MagicMock()
        agent = AgenticFetchAgent(
            max_steps=5,
            firecrawl_client=mock_firecrawl
        )

        assert agent.max_steps == 5
        assert agent._firecrawl_client is mock_firecrawl

    @pytest.mark.asyncio
    async def test_agent_initialization_without_firecrawl_client(self):
        """Test agent can be initialized without firecrawl_client."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent(max_steps=5)

        assert agent.max_steps == 5
        assert agent._firecrawl_client is None


class TestFirecrawlFetch:
    """Tests for _firecrawl_fetch method."""

    @pytest.mark.asyncio
    async def test_firecrawl_fetch_success(self):
        """Test successful Firecrawl fetch."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_result = MagicMock()
        mock_result.content = "# Test Content\n\nThis is the page content."
        mock_result.title = "Test Page"

        mock_firecrawl = AsyncMock()
        mock_firecrawl.scrape.return_value = mock_result

        agent = AgenticFetchAgent(
            firecrawl_client=mock_firecrawl,
            max_steps=3
        )

        result = await agent._firecrawl_fetch("https://example.com")

        assert "content" in result
        assert result["method"] == "firecrawl"
        mock_firecrawl.scrape.assert_called_once()

    @pytest.mark.asyncio
    async def test_firecrawl_fetch_no_client(self):
        """Test Firecrawl fetch when no client is configured."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent(max_steps=3)
        assert agent._firecrawl_client is None

        result = await agent._firecrawl_fetch("https://example.com")

        assert "error" in result
        assert "Firecrawl not available" in result["error"]

    @pytest.mark.asyncio
    async def test_firecrawl_fetch_failure(self):
        """Test Firecrawl fetch when scrape fails."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_firecrawl = AsyncMock()
        mock_firecrawl.scrape.return_value = None  # Simulates failure

        agent = AgenticFetchAgent(
            firecrawl_client=mock_firecrawl,
            max_steps=3
        )

        result = await agent._firecrawl_fetch("https://example.com")

        assert "error" in result


class TestFirecrawlMap:
    """Tests for _firecrawl_map method."""

    @pytest.mark.asyncio
    async def test_firecrawl_map_success(self):
        """Test successful Firecrawl map_site."""
        from mcp_server.agentic import AgenticFetchAgent

        discovered_urls = [
            "https://example.com/page1",
            "https://example.com/page2"
        ]

        mock_firecrawl = AsyncMock()
        mock_firecrawl.map_site.return_value = discovered_urls

        agent = AgenticFetchAgent(
            firecrawl_client=mock_firecrawl,
            max_steps=3
        )

        result = await agent._firecrawl_map("https://example.com")

        assert result == discovered_urls
        mock_firecrawl.map_site.assert_called_once_with("https://example.com", search_depth=1)

    @pytest.mark.asyncio
    async def test_firecrawl_map_no_client(self):
        """Test Firecrawl map when no client is configured."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent(max_steps=3)
        assert agent._firecrawl_client is None

        result = await agent._firecrawl_map("https://example.com")

        assert result == []


class TestFirecrawlEnhancedFetch:
    """Tests for Firecrawl-enhanced fetch action in execute()."""

    @pytest.mark.asyncio
    async def test_fetch_action_uses_firecrawl_when_available(self):
        """Test that 'fetch' action uses Firecrawl when client is available."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_result = MagicMock()
        mock_result.content = "Firecrawl extracted content"
        mock_result.title = "Page Title"

        mock_firecrawl = AsyncMock()
        mock_firecrawl.scrape.return_value = mock_result

        mock_llm = AsyncMock()

        call_count = 0

        async def llm_side_effect(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return '{"action": "fetch", "description": "Get content from URL", "url": "https://example.com"}'
            else:
                return '{"action": "done", "content": "Found it!"}'

        mock_llm.complete.side_effect = llm_side_effect

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            firecrawl_client=mock_firecrawl,
            max_steps=3
        )

        result = await agent.execute("Find content on example.com")

        assert result.success is True
        mock_firecrawl.scrape.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_action_skips_firecrawl_when_not_available(self):
        """Test that 'fetch' action skips Firecrawl when client is not set."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()
        mock_fetch_func = AsyncMock(return_value={"content": "HTTP fetched content"})

        call_count = 0

        async def llm_side_effect(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return '{"action": "fetch", "description": "Get content from URL", "url": "https://example.com"}'
            else:
                return '{"action": "done", "content": "Found it!"}'

        mock_llm.complete.side_effect = llm_side_effect

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            fetch_func=mock_fetch_func,
            max_steps=3
        )

        result = await agent.execute("Find content on example.com")

        assert result.success is True


class TestFirecrawlClientIntegration:
    """Tests for Firecrawl client integration patterns."""

    @pytest.mark.asyncio
    async def test_firecrawl_client_passed_to_agent(self):
        """Test that firecrawl_client is properly passed to AgenticFetchAgent."""
        from mcp_server.agentic import AgenticFetchResult
        from unittest.mock import patch

        mock_result = AgenticFetchResult(success=True, content="ok")

        with patch("mcp_server.agentic.fetch_agent.AgenticFetchAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = mock_result
            MockAgent.return_value = mock_instance

            from mcp_server.agentic.fetch_agent import agentic_fetch

            mock_firecrawl = MagicMock()

            await agentic_fetch(
                prompt="Test",
                firecrawl_client=mock_firecrawl
            )

            call_kwargs = MockAgent.call_args.kwargs
            assert "firecrawl_client" in call_kwargs
            assert call_kwargs["firecrawl_client"] is mock_firecrawl

    @pytest.mark.asyncio
    async def test_agentic_fetch_without_firecrawl_client(self):
        """Test agentic_fetch works without firecrawl_client."""
        from mcp_server.agentic import AgenticFetchResult
        from unittest.mock import patch

        mock_result = AgenticFetchResult(success=True, content="ok")

        with patch("mcp_server.agentic.fetch_agent.AgenticFetchAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = mock_result
            MockAgent.return_value = mock_instance

            from mcp_server.agentic.fetch_agent import agentic_fetch

            result_dict = await agentic_fetch(prompt="Test")

            call_kwargs = MockAgent.call_args.kwargs
            assert "firecrawl_client" in call_kwargs
            assert call_kwargs["firecrawl_client"] is None