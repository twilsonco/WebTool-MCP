"""
Tests for agentic fetch mode.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestAgenticFetchResult:
    """Tests for AgenticFetchResult dataclass."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        from mcp_server.agentic import AgenticFetchResult
        
        result = AgenticFetchResult(
            success=True,
            content="Test content",
            url="https://example.com"
        )
        
        d = result.to_dict()
        assert d["success"] is True
        assert d["content"] == "Test content"
        assert d["url"] == "https://example.com"

    def test_to_dict_with_urls_visited(self):
        """Test to_dict with URLs visited."""
        from mcp_server.agentic import AgenticFetchResult
        
        result = AgenticFetchResult(
            success=True,
            content="Test",
            url="https://example.com"
        )
        result.urls_visited.append({
            "url": "https://search-result.com",
            "title": "Search Result 1",
            "action": "Search result at step 1"
        })
        
        d = result.to_dict()
        assert len(d["urls_visited"]) == 1
        assert d["urls_visited"][0]["url"] == "https://search-result.com"

    def test_to_dict_with_steps(self):
        """Test to_dict with steps taken."""
        from mcp_server.agentic import AgenticFetchResult
        
        result = AgenticFetchResult(success=False)
        result.steps_taken.append({
            "step": 1,
            "action": "search",
            "description": "Searching for query",
            "result": {"count": 10}
        })
        
        d = result.to_dict()
        assert len(d["steps_taken"]) == 1
        assert d["steps_taken"][0]["action"] == "search"


class TestAgenticFetchAgent:
    """Tests for AgenticFetchAgent class."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent can be initialized."""
        from mcp_server.agentic import AgenticFetchAgent
        
        agent = AgenticFetchAgent(max_steps=5)
        
        assert agent.max_steps == 5
        assert agent._llm_manager is None
        assert agent._search_func is None
        assert agent._fetch_func is None

    @pytest.mark.asyncio
    async def test_agent_with_mocked_llm(self):
        """Test agent execution with mocked LLM."""
        from mcp_server.agentic import AgenticFetchAgent
        
        # Create mock LLM manager
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = '{"action": "done", "description": "Found it!", "content": "The answer is 42"}'
        
        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            max_steps=3
        )
        
        result = await agent.execute("What is the answer to life?")
        
        assert result.success is True
        assert "42" in (result.content or "")
        # LLM may be called multiple times: once for action decision,
        # and again for content relevance validation
        assert mock_llm.complete.call_count >= 1

    @pytest.mark.asyncio
    async def test_agent_falls_back_to_regular_search(self):
        """Test agent falls back to regular search when browser-use unavailable."""
        from mcp_server.agentic import AgenticFetchAgent
        
        # Create mock LLM that requests a search
        mock_llm = AsyncMock()
        
        call_count = 0
        
        async def llm_side_effect(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First call - return search with query
                return '{"action": "search", "description": "Search for it", "query": "test query"}'
            else:
                # Subsequent calls - return done
                return '{"action": "done", "description": "Search complete"}'
        
        mock_llm.complete.side_effect = llm_side_effect
        
        # Create mock search function
        async def mock_search(query, num_results=10):
            return {
                "results": [
                    {"title": f"Result {i}", "url": f"https://example.com/{i}"}
                    for i in range(3)
                ]
            }
        
        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            search_func=mock_search,
            max_steps=3
        )
        
        result = await agent.execute("Find me some results")
        
        assert len(result.urls_visited) > 0


class TestAgenticFetchFunction:
    """Tests for agentic_fetch() convenience function."""

    @pytest.mark.asyncio
    async def test_agentic_fetch_with_mocked_agent(self):
        """Test agentic_fetch with mocked AgenticFetchAgent."""
        from mcp_server.agentic import AgenticFetchResult
        from unittest.mock import patch
        
        mock_result = AgenticFetchResult(
            success=True,
            content="Test content",
            url="https://example.com"
        )
        
        with patch("mcp_server.agentic.fetch_agent.AgenticFetchAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = mock_result
            MockAgent.return_value = mock_instance
            
            from mcp_server.agentic.fetch_agent import agentic_fetch
            result_dict = await agentic_fetch("Find me content")
            
            # Verify AgenticFetchAgent was instantiated
            MockAgent.assert_called_once()
            call_kwargs = MockAgent.call_args.kwargs
            
            # Verify default max_steps
            assert call_kwargs.get("max_steps") == 10
            
            # Verify execute was called with the prompt
            mock_instance.execute.assert_called_once_with("Find me content")
            
            # Verify result is converted to dict
            assert isinstance(result_dict, dict)
            assert result_dict["success"] is True
            assert result_dict["content"] == "Test content"

    @pytest.mark.asyncio
    async def test_agentic_fetch_passes_through_arguments(self):
        """Test that agentic_fetch properly passes arguments to AgenticFetchAgent."""
        from mcp_server.agentic import AgenticFetchResult
        from unittest.mock import patch, MagicMock
        
        mock_result = AgenticFetchResult(success=True, content="ok")
        
        custom_llm = MagicMock()
        custom_pipeline = MagicMock()
        custom_search = AsyncMock(return_value={"results": []})
        custom_fetch = AsyncMock(return_value={"content": "fetched"})
        
        with patch("mcp_server.agentic.fetch_agent.AgenticFetchAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = mock_result
            MockAgent.return_value = mock_instance
            
            from mcp_server.agentic.fetch_agent import agentic_fetch
            await agentic_fetch(
                prompt="Test prompt",
                max_steps=5,
                llm_manager=custom_llm,
                extraction_pipeline=custom_pipeline,
                search_func=custom_search,
                fetch_func=custom_fetch
            )
            
            call_kwargs = MockAgent.call_args.kwargs
            
            assert call_kwargs.get("max_steps") == 5
            assert call_kwargs.get("llm_manager") is custom_llm
            assert call_kwargs.get("extraction_pipeline") is custom_pipeline
            # Note: search_func and fetch_func are wrapped, so we check they were passed
            assert "search_func" in call_kwargs
            assert "fetch_func" in call_kwargs

    @pytest.mark.asyncio
    async def test_agentic_fetch_custom_max_steps(self):
        """Test agentic_fetch with custom max_steps."""
        from mcp_server.agentic import AgenticFetchResult
        from unittest.mock import patch
        
        mock_result = AgenticFetchResult(success=False, error_message="Max steps reached")
        
        with patch("mcp_server.agentic.fetch_agent.AgenticFetchAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = mock_result
            MockAgent.return_value = mock_instance
            
            from mcp_server.agentic.fetch_agent import agentic_fetch
            result_dict = await agentic_fetch(
                prompt="Long task",
                max_steps=3
            )
            
            call_kwargs = MockAgent.call_args.kwargs
            assert call_kwargs.get("max_steps") == 3

    @pytest.mark.asyncio
    async def test_agentic_fetch_returns_dict(self):
        """Test that agentic_fetch returns a dictionary (result.to_dict())."""
        from mcp_server.agentic import AgenticFetchResult
        from unittest.mock import patch
        
        mock_result = AgenticFetchResult(
            success=True,
            content="Content here",
            url="https://test.com"
        )
        
        with patch("mcp_server.agentic.fetch_agent.AgenticFetchAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = mock_result
            MockAgent.return_value = mock_instance
            
            from mcp_server.agentic.fetch_agent import agentic_fetch
            result_dict = await agentic_fetch("test prompt")
            
            # Should return a plain dict, not an AgenticFetchResult
            assert isinstance(result_dict, dict)
            assert "success" in result_dict
            assert "content" in result_dict