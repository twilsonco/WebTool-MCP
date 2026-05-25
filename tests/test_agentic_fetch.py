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
        mock_llm.complete.assert_called_once()

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
            
            if "action" not in prompt:
                return '{"action": "search", "description": "Search for it", "query": "test query"}'
            elif call_count > 1:
                return '{"action": "done", "description": "Search complete"}'
            else:
                return '{"action": "search", "description": "Need to search more"}'
        
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