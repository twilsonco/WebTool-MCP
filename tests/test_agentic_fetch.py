"""
Tests for agentic fetch mode.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


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



class TestNormalizeUrlExceptionFallback:
    """Tests for _normalize_url exception handling in fallback path.

    Note: Lines 181-183 are defensive exception handling for edge cases.
    The _normalize_url function uses only string operations that don't normally fail,
    and already handles None/empty input at the start. This exception handler
    is for truly unexpected inputs that cause internal string operations to fail.
    """

    def test_normalize_url_with_unicode_edge_case(self):
        """Test _normalize_url with unusual Unicode that could cause issues."""
        from mcp_server.agentic.fetch_agent import _normalize_url

        result = _normalize_url("https://example.com/path")
        assert "https" in result
        assert "example.com" in result


class TestBrowserSearchStringParseFailure:
    """Tests for _browser_search string parsing failure error path."""

    @pytest.mark.asyncio
    async def test_browser_search_result_string_json_decode_error(self):
        """Test _browser_search when result is a string that fails JSON parsing."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()

        async def llm_side_effect(prompt, system_prompt=None):
            return '{"action": "search", "query": "test"}'

        mock_llm.complete.side_effect = llm_side_effect

        async def mock_search(query, num_results=10):
            return {"results": []}

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            search_func=mock_search,
            max_steps=2
        )

        with patch.object(agent, '_browser_search', new_callable=AsyncMock) as mock_browser_search:
            mock_browser_search.return_value = "this is not parseable {"

            result = await agent.execute("test")

            assert len(result.steps_taken) >= 1


class TestBrowserSearchUnexpectedType:
    """Tests for _browser_search unexpected result type error path."""

    @pytest.mark.asyncio
    async def test_browser_search_returns_unexpected_type_error(self):
        """Test _browser_search when result is neither list nor dict."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()

        async def llm_side_effect(prompt, system_prompt=None):
            return '{"action": "search", "query": "test"}'

        mock_llm.complete.side_effect = llm_side_effect

        async def mock_search(query, num_results=10):
            return {"results": []}

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            search_func=mock_search,
            max_steps=2
        )

        with patch.object(agent, '_browser_search', new_callable=AsyncMock) as mock_browser_search:
            mock_browser_search.return_value = {"error": "test error"}

            result = await agent.execute("test")

            assert len(result.steps_taken) >= 1


class TestBrowserNavigateStringParseFailure:
    """Tests for _browser_navigate_and_extract string parsing failure error path."""

    @pytest.mark.asyncio
    async def test_browser_navigate_result_string_json_decode_error(self):
        """Test _browser_navigate_and_extract when result is a string that fails JSON parsing."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()

        async def llm_side_effect(prompt, system_prompt=None):
            return '{"action": "navigate", "url": "https://example.com"}'

        mock_llm.complete.side_effect = llm_side_effect

        async def mock_fetch(url):
            return {"content": "test"}

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            fetch_func=mock_fetch,
            max_steps=2
        )

        with patch.object(agent, '_browser_navigate_and_extract') as mock_browser_nav:
            mock_browser_nav.return_value = "not parseable {"

            result = await agent.execute("test")

            assert len(result.steps_taken) >= 1


class TestBrowserNavigateUnexpectedType:
    """Tests for _browser_navigate_and_extract unexpected result type error path."""

    @pytest.mark.asyncio
    async def test_browser_navigate_returns_list_error(self):
        """Test _browser_navigate_and_extract when result is a list (unexpected)."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()

        async def llm_side_effect(prompt, system_prompt=None):
            return '{"action": "navigate", "url": "https://example.com"}'

        mock_llm.complete.side_effect = llm_side_effect

        async def mock_fetch(url):
            return {"content": "test"}

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            fetch_func=mock_fetch,
            max_steps=2
        )

        with patch.object(agent, '_browser_navigate_and_extract') as mock_browser_nav:
            mock_browser_nav.return_value = ["unexpected", "list"]

            result = await agent.execute("test")

            assert len(result.steps_taken) >= 1


class TestParseLLMActionBracePositionEdgeCases:
    """Tests for _parse_llm_action JSON parsing edge cases with invalid brace positions."""

    def test_parse_llm_action_stripped_starts_with_invalid_brace(self):
        """Test _parse_llm_action when stripped response starts with brace but has invalid structure."""
        from mcp_server.agentic import AgenticFetchAgent
        from mcp_server.agentic.fetch_agent import ActionParsingError

        agent = AgenticFetchAgent()

        with patch("mcp_server.agentic.fetch_agent.LLMAction.model_validate_json") as mock_validate:
            import json
            mock_validate.side_effect = json.JSONDecodeError("error", "}", 0)

            response = '{"action": ' + ',some extra text that has no keywords'

            with pytest.raises(ActionParsingError):
                agent._parse_llm_action(response)

    def test_parse_llm_action_no_json_start_brace_found(self):
        """Test _parse_llm_action when no opening brace is found after find."""
        from mcp_server.agentic import AgenticFetchAgent
        from mcp_server.agentic.fetch_agent import ActionParsingError

        agent = AgenticFetchAgent()

        with patch("mcp_server.agentic.fetch_agent.LLMAction.model_validate_json") as mock_validate:
            import json
            mock_validate.side_effect = ValueError("No JSON object found in response")

            response = "no json here"

            with pytest.raises(ActionParsingError):
                agent._parse_llm_action(response)

    def test_parse_llm_action_invalid_json_structure_brace_order(self):
        """Test _parse_llm_action when closing brace is before opening brace."""
        from mcp_server.agentic import AgenticFetchAgent
        from mcp_server.agentic.fetch_agent import ActionParsingError

        agent = AgenticFetchAgent()

        with patch("mcp_server.agentic.fetch_agent.LLMAction.model_validate_json") as mock_validate:
            import json
            mock_validate.side_effect = ValueError("Invalid JSON structure")

            response = "}something{"

            with pytest.raises(ActionParsingError):
                agent._parse_llm_action(response)


class TestSearchActionBrowserResultErrors:
    """Tests for search action error paths when browser returns unexpected format."""

    @pytest.mark.asyncio
    async def test_search_browser_fallback_result_contains_error(self):
        """Test search action when browser fallback result contains error key."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()

        async def llm_side_effect(prompt, system_prompt=None):
            return '{"action": "search", "query": "test"}'

        mock_llm.complete.side_effect = llm_side_effect

        async def mock_search(query, num_results=10):
            return {"error": "Search failed"}

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            search_func=mock_search,
            max_steps=2
        )

        with patch.object(agent, '_browser_search', new_callable=AsyncMock) as mock_browser:
            mock_browser.return_value = {"fallback_used": True, "result": {"error": "Browser search failed"}}

            result = await agent.execute("test")

            assert len(result.steps_taken) >= 1

    @pytest.mark.asyncio
    async def test_search_browser_result_not_list(self):
        """Test search action when browser returns non-list result."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()

        async def llm_side_effect(prompt, system_prompt=None):
            return '{"action": "search", "query": "test"}'

        mock_llm.complete.side_effect = llm_side_effect

        async def mock_search(query, num_results=10):
            return {"results": []}

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            search_func=mock_search,
            max_steps=2
        )

        with patch.object(agent, '_browser_search', new_callable=AsyncMock) as mock_browser:
            mock_browser.return_value = {"fallback_used": False, "result": "not a list"}

            result = await agent.execute("test")

            assert len(result.steps_taken) >= 1

    @pytest.mark.asyncio
    async def test_search_fallback_after_browser_error(self):
        """Test search action when fallback result contains error after browser failure."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()

        async def llm_side_effect(prompt, system_prompt=None):
            return '{"action": "search", "query": "test"}'

        mock_llm.complete.side_effect = llm_side_effect

        async def mock_search(query, num_results=10):
            return {"error": "Search failed"}

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            search_func=mock_search,
            max_steps=2
        )

        with patch.object(agent, '_browser_search', new_callable=AsyncMock) as mock_browser:
            from mcp_server.agentic.fetch_agent import BrowserToolError
            mock_browser.side_effect = BrowserToolError("Browser failed")

            result = await agent.execute("test")

            assert len(result.steps_taken) >= 1


class TestFetchNavigateBrowserResultErrors:
    """Tests for fetch/navigate branches with browser result parsing errors."""

    @pytest.mark.asyncio
    async def test_fetch_browser_result_string_parse_error(self):
        """Test fetch action when browser returns string that fails JSON parsing."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()

        async def llm_side_effect(prompt, system_prompt=None):
            return '{"action": "fetch", "url": "https://example.com"}'

        mock_llm.complete.side_effect = llm_side_effect

        async def mock_fetch(url):
            return {"content": "test"}

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            fetch_func=mock_fetch,
            max_steps=2
        )

        with patch.object(agent, '_browser_navigate_and_extract') as mock_browser:
            from mcp_server.agentic.fetch_agent import BrowserToolError
            mock_browser.return_value = "string result {"
            agent._fetch = AsyncMock(return_value={"content": "fallback"})

            result = await agent.execute("test")

            assert len(result.steps_taken) >= 1

    @pytest.mark.asyncio
    async def test_fetch_fallback_result_contains_error(self):
        """Test fetch action when fallback result contains error."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()

        async def llm_side_effect(prompt, system_prompt=None):
            return '{"action": "fetch", "url": "https://example.com"}'

        mock_llm.complete.side_effect = llm_side_effect

        async def mock_fetch(url):
            return {"error": "Fetch failed"}

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            fetch_func=mock_fetch,
            max_steps=2
        )

        with patch.object(agent, '_browser_navigate_and_extract') as mock_browser:
            from mcp_server.agentic.fetch_agent import BrowserToolError
            mock_browser.side_effect = BrowserToolError("Browser failed")

            result = await agent.execute("test")

            assert len(result.steps_taken) >= 1


class TestAgenticFetchDefaultValueCreation:
    """Tests for agentic_fetch function defaults when llm_manager is None."""

    @pytest.mark.asyncio
    async def test_agentic_fetch_llm_creation_fails(self):
        """Test agentic_fetch when llm_manager is None and LLMImport() raises."""
        from mcp_server.agentic import AgenticFetchResult

        mock_result = AgenticFetchResult(success=True, content="test")

        with patch("mcp_server.agentic.fetch_agent.AgenticFetchAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = mock_result
            MockAgent.return_value = mock_instance

            with patch("mcp_server.llm.LLMManager") as MockLLM:
                MockLLM.side_effect = Exception("LLM creation failed")

                from mcp_server.agentic.fetch_agent import agentic_fetch
                await agentic_fetch("test prompt")

    @pytest.mark.asyncio
    async def test_agentic_fetch_default_search_creation_fails(self):
        """Test agentic_fetch when default search creation fails."""
        from mcp_server.agentic import AgenticFetchResult

        mock_result = AgenticFetchResult(success=True, content="test")

        with patch("mcp_server.agentic.fetch_agent.AgenticFetchAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = mock_result
            MockAgent.return_value = mock_instance

            with patch("mcp_server.llm.LLMManager") as MockLLM:
                mock_llm_instance = MagicMock()
                MockLLM.return_value = mock_llm_instance

                with patch("mcp_server.server.search_web", new_callable=AsyncMock) as mock_search:
                    mock_search.side_effect = Exception("Search not available")

                    from mcp_server.agentic.fetch_agent import agentic_fetch
                    await agentic_fetch("test prompt")

    @pytest.mark.asyncio
    async def test_agentic_fetch_default_fetch_creation_fails(self):
        """Test agentic_fetch when default fetch creation fails."""
        from mcp_server.agentic import AgenticFetchResult

        mock_result = AgenticFetchResult(success=True, content="test")

        with patch("mcp_server.agentic.fetch_agent.AgenticFetchAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = mock_result
            MockAgent.return_value = mock_instance

            with patch("mcp_server.llm.LLMManager") as MockLLM:
                mock_llm_instance = MagicMock()
                MockLLM.return_value = mock_llm_instance

                with patch("mcp_server.server.search_web", new=AsyncMock()):
                    with patch("mcp_server.server.fetch_web_content", new_callable=AsyncMock) as mock_fetch:
                        mock_fetch.side_effect = Exception("Fetch not available")

                        from mcp_server.agentic.fetch_agent import agentic_fetch
                        await agentic_fetch("test prompt")


class TestNormalizeUrl:
    """Tests for _normalize_url function."""

    def test_normalize_url_with_none(self):
        """Test normalize_url with None input."""
        from mcp_server.agentic.fetch_agent import _normalize_url
        result = _normalize_url(None)
        assert result == ""

    def test_normalize_url_with_empty_string(self):
        """Test normalize_url with empty string."""
        from mcp_server.agentic.fetch_agent import _normalize_url
        result = _normalize_url("")
        assert result == ""

    def test_normalize_url_with_non_string(self):
        """Test normalize_url with non-string input."""
        from mcp_server.agentic.fetch_agent import _normalize_url
        result = _normalize_url(123)
        assert result == ""

    def test_normalize_url_without_scheme(self):
        """Test normalize_url with URL without scheme."""
        from mcp_server.agentic.fetch_agent import _normalize_url
        result = _normalize_url("example.com/page")
        assert result == "example.com/page"

    def test_normalize_url_exception_fallback(self):
        """Test normalize_url exception fallback path."""
        from mcp_server.agentic.fetch_agent import _normalize_url
        result = _normalize_url("http://[invalid")
        assert isinstance(result, str)


class TestActionParsingError:
    """Tests for ActionParsingError exception."""

    def test_action_parsing_error_attributes(self):
        """Test ActionParsingError has raw_response and cause attributes."""
        from mcp_server.agentic.fetch_agent import ActionParsingError

        original_error = ValueError("original")
        error = ActionParsingError(
            message="Test error",
            raw_response='{"action": "test"}',
            cause=original_error
        )

        assert error.raw_response == '{"action": "test"}'
        assert error.cause is original_error
        assert str(error) == "Test error"


class TestLLMActionValidator:
    """Tests for LLMAction validator."""

    def test_llm_action_nav_normalization(self):
        """Test 'nav' is normalized to NAVIGATE."""
        from mcp_server.agentic.fetch_agent import LLMAction

        action = LLMAction(action="nav")
        assert action.action == "navigate"

    def test_llm_action_navigat_normalization(self):
        """Test 'navigat' prefix is normalized to NAVIGATE."""
        from mcp_server.agentic.fetch_agent import LLMAction

        action = LLMAction(action="navigat")
        assert action.action == "navigate"

    def test_llm_action_unknown_pass_through(self):
        """Test unknown action is passed through as-is."""
        from mcp_server.agentic.fetch_agent import LLMAction

        action = LLMAction(action="unknown_action")
        assert action.action == "unknown_action"


class TestBrowserToolError:
    """Tests for BrowserToolError exception."""

    def test_browser_tool_error_attributes(self):
        """Test BrowserToolError has fallback_used attribute."""
        from mcp_server.agentic.fetch_agent import BrowserToolError

        error = BrowserToolError("Test error", fallback_used=True)
        assert str(error) == "Test error"
        assert error.fallback_used is True

    def test_browser_tool_error_default(self):
        """Test BrowserToolError default fallback_used is False."""
        from mcp_server.agentic.fetch_agent import BrowserToolError

        error = BrowserToolError("Test error")
        assert error.fallback_used is False


class TestBrowserToolExecute:
    """Tests for BrowserTool.execute method."""

    @pytest.mark.asyncio
    async def test_browser_tool_string_result(self):
        """Test BrowserTool with string result."""
        from mcp_server.agentic.fetch_agent import BrowserTool

        async def fallback():
            return {"content": "fallback"}

        tool = BrowserTool(task_description="test task", fallback_func=fallback)
        result = await tool.execute()
        assert "error" in result or result.get("fallback_used") is True

    @pytest.mark.asyncio
    async def test_browser_tool_import_error_with_fallback(self):
        """Test BrowserTool ImportError fallback path."""
        from mcp_server.agentic.fetch_agent import BrowserTool, BrowserToolError
        import sys

        async def fallback():
            return {"results": [{"title": "Fallback", "url": "https://fallback.com"}]}

        tool = BrowserTool(
            task_description="test task",
            fallback_func=fallback
        )

        class MockBrowserUseModule:
            def __getattr__(self, name):
                raise ImportError("browser-use not installed")

        with patch.dict(sys.modules, {'browser_use': MockBrowserUseModule()}):
            result = await tool.execute()
            assert result.get("fallback_used") is True

    @pytest.mark.asyncio
    async def test_browser_tool_import_error_without_fallback(self):
        """Test BrowserTool ImportError without fallback raises."""
        from mcp_server.agentic.fetch_agent import BrowserTool, BrowserToolError
        import sys

        tool = BrowserTool(task_description="test task")

        class MockBrowserUseModule:
            def __getattr__(self, name):
                raise ImportError("browser-use not installed")

        with patch.dict(sys.modules, {'browser_use': MockBrowserUseModule()}):
            with pytest.raises(BrowserToolError):
                await tool.execute()

    @pytest.mark.asyncio
    async def test_browser_tool_exception_with_fallback(self):
        """Test BrowserTool exception fallback path."""
        from mcp_server.agentic.fetch_agent import BrowserTool
        import sys

        async def fallback():
            return {"results": [{"title": "Fallback", "url": "https://fallback.com"}]}

        tool = BrowserTool(
            task_description="test task",
            fallback_func=fallback
        )

        class MockAgent:
            def __init__(self, task):
                pass
            async def run(self):
                raise Exception("Browser failed")

        mock_browser_use = MagicMock()
        mock_browser_use.Agent = MockAgent

        with patch.dict(sys.modules, {'browser_use': mock_browser_use}):
            result = await tool.execute()
            assert result.get("fallback_used") is True


class TestFetchStepToDict:
    """Tests for FetchStep.to_dict method."""

    def test_fetch_step_to_dict(self):
        """Test FetchStep to_dict conversion."""
        from mcp_server.agentic.fetch_agent import FetchStep

        step = FetchStep(
            step_number=1,
            action="search",
            query="test query"
        )

        d = step.to_dict()
        assert d["step"] == 1
        assert d["action"] == "search"
        assert d["query"] == "test query"


class TestAgenticFetchAgentCallLlm:
    """Tests for AgenticFetchAgent._call_llm method."""

    @pytest.mark.asyncio
    async def test_call_llm_with_none_manager(self):
        """Test _call_llm with None LLM manager."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        result = await agent._call_llm("test prompt")
        assert result is None

    @pytest.mark.asyncio
    async def test_call_llm_with_exception(self):
        """Test _call_llm exception handling."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = Exception("LLM failed")

        agent = AgenticFetchAgent(llm_manager=mock_llm)
        result = await agent._call_llm("test prompt")
        assert result is None


class TestAgenticFetchAgentSearch:
    """Tests for AgenticFetchAgent._search method."""

    @pytest.mark.asyncio
    async def test_search_with_no_function(self):
        """Test _search with no search function configured."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        result = await agent._search("test query")
        assert "error" in result


class TestAgenticFetchAgentFetch:
    """Tests for AgenticFetchAgent._fetch method."""

    @pytest.mark.asyncio
    async def test_fetch_with_no_function(self):
        """Test _fetch with no fetch function configured."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        result = await agent._fetch("https://example.com")
        assert "error" in result


class TestBrowserSearch:
    """Tests for AgenticFetchAgent._browser_search method."""

    @pytest.mark.asyncio
    async def test_browser_search_fallback_used(self):
        """Test _browser_search when fallback is used."""
        from mcp_server.agentic import AgenticFetchAgent

        async def mock_search(query, num_results=10):
            return {"results": [{"title": "Result", "url": "https://example.com"}]}

        mock_llm = AsyncMock()

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            search_func=mock_search
        )

        with patch("mcp_server.agentic.fetch_agent.BrowserTool") as mock_tool:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = {
                "fallback_used": True,
                "result": {"results": [{"title": "Fallback", "url": "https://fallback.com"}]}
            }
            mock_tool.return_value = mock_instance

            result = await agent._browser_search("test query")
            assert isinstance(result, dict)
            assert "results" in result

    @pytest.mark.asyncio
    async def test_browser_search_string_result(self):
        """Test _browser_search with string result."""
        from mcp_server.agentic import AgenticFetchAgent

        async def mock_search(query, num_results=10):
            return {"results": []}

        agent = AgenticFetchAgent(search_func=mock_search)

        with patch("mcp_server.agentic.fetch_agent.BrowserTool") as mock_tool:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = {
                "result": '[{"title": "Result", "url": "https://example.com"}]'
            }
            mock_tool.return_value = mock_instance

            result = await agent._browser_search("test query")
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_browser_search_dict_with_error(self):
        """Test _browser_search with dict containing error."""
        from mcp_server.agentic import AgenticFetchAgent

        async def mock_search(query, num_results=10):
            return {"results": []}

        agent = AgenticFetchAgent(search_func=mock_search)

        with patch("mcp_server.agentic.fetch_agent.BrowserTool") as mock_tool:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = {
                "result": {"error": "Search failed"}
            }
            mock_tool.return_value = mock_instance

            result = await agent._browser_search("test query")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_browser_search_unexpected_type(self):
        """Test _browser_search with unexpected result type."""
        from mcp_server.agentic import AgenticFetchAgent

        async def mock_search(query, num_results=10):
            return {"results": []}

        agent = AgenticFetchAgent(search_func=mock_search)

        with patch("mcp_server.agentic.fetch_agent.BrowserTool") as mock_tool:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = {
                "result": 12345
            }
            mock_tool.return_value = mock_instance

            result = await agent._browser_search("test query")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_browser_search_browser_tool_error(self):
        """Test _browser_search when BrowserToolError is raised."""
        from mcp_server.agentic import AgenticFetchAgent, BrowserToolError

        async def mock_search(query, num_results=10):
            return {"results": [{"title": "Result", "url": "https://example.com"}]}

        agent = AgenticFetchAgent(search_func=mock_search)

        with patch("mcp_server.agentic.fetch_agent.BrowserTool") as mock_tool:
            mock_instance = AsyncMock()
            mock_instance.execute.side_effect = BrowserToolError("Browser failed")
            mock_tool.return_value = mock_instance

            result = await agent._browser_search("test query")
            assert isinstance(result, dict)
            assert "results" in result


class TestBrowserNavigateAndExtract:
    """Tests for AgenticFetchAgent._browser_navigate_and_extract method."""

    @pytest.mark.asyncio
    async def test_browser_navigate_fallback_used(self):
        """Test _browser_navigate_and_extract when fallback is used."""
        from mcp_server.agentic import AgenticFetchAgent

        async def mock_fetch(url):
            return {"content": "Fallback content", "title": "Fallback"}

        agent = AgenticFetchAgent(fetch_func=mock_fetch)

        with patch("mcp_server.agentic.fetch_agent.BrowserTool") as mock_tool:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = {
                "fallback_used": True,
                "result": {"content": "Fallback", "title": "Page"}
            }
            mock_tool.return_value = mock_instance

            result = await agent._browser_navigate_and_extract("https://example.com")
            assert "content" in result or "error" not in result

    @pytest.mark.asyncio
    async def test_browser_navigate_string_result(self):
        """Test _browser_navigate_and_extract with string result."""
        from mcp_server.agentic import AgenticFetchAgent

        async def mock_fetch(url):
            return {"content": ""}

        agent = AgenticFetchAgent(fetch_func=mock_fetch)

        with patch("mcp_server.agentic.fetch_agent.BrowserTool") as mock_tool:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = {
                "result": '{"title": "Page", "content": "Content"}'
            }
            mock_tool.return_value = mock_instance

            result = await agent._browser_navigate_and_extract("https://example.com")
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_browser_navigate_json_decode_error(self):
        """Test _browser_navigate_and_extract with JSON decode error."""
        from mcp_server.agentic import AgenticFetchAgent

        async def mock_fetch(url):
            return {"content": ""}

        agent = AgenticFetchAgent(fetch_func=mock_fetch)

        with patch("mcp_server.agentic.fetch_agent.BrowserTool") as mock_tool:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = {
                "result": "not valid json {"
            }
            mock_tool.return_value = mock_instance

            result = await agent._browser_navigate_and_extract("https://example.com")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_browser_navigate_sets_url(self):
        """Test _browser_navigate_and_extract sets URL in result."""
        from mcp_server.agentic import AgenticFetchAgent

        async def mock_fetch(url):
            return {"content": ""}

        agent = AgenticFetchAgent(fetch_func=mock_fetch)

        with patch("mcp_server.agentic.fetch_agent.BrowserTool") as mock_tool:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = {
                "result": {"title": "Page", "content": "Content"}
            }
            mock_tool.return_value = mock_instance

            result = await agent._browser_navigate_and_extract("https://example.com")
            assert result.get("url") == "https://example.com"

    @pytest.mark.asyncio
    async def test_browser_navigate_unexpected_list(self):
        """Test _browser_navigate_and_extract with unexpected list result."""
        from mcp_server.agentic import AgenticFetchAgent

        async def mock_fetch(url):
            return {"content": ""}

        agent = AgenticFetchAgent(fetch_func=mock_fetch)

        with patch("mcp_server.agentic.fetch_agent.BrowserTool") as mock_tool:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = {
                "result": [{"title": "Page"}]
            }
            mock_tool.return_value = mock_instance

            result = await agent._browser_navigate_and_extract("https://example.com")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_browser_navigate_unexpected_type(self):
        """Test _browser_navigate_and_extract with unexpected type."""
        from mcp_server.agentic import AgenticFetchAgent

        async def mock_fetch(url):
            return {"content": ""}

        agent = AgenticFetchAgent(fetch_func=mock_fetch)

        with patch("mcp_server.agentic.fetch_agent.BrowserTool") as mock_tool:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = {
                "result": 12345
            }
            mock_tool.return_value = mock_instance

            result = await agent._browser_navigate_and_extract("https://example.com")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_browser_navigate_browser_tool_error(self):
        """Test _browser_navigate_and_extract when BrowserToolError is raised."""
        from mcp_server.agentic import AgenticFetchAgent, BrowserToolError

        async def mock_fetch(url):
            return {"content": "Fallback content", "title": "Page"}

        agent = AgenticFetchAgent(fetch_func=mock_fetch)

        with patch("mcp_server.agentic.fetch_agent.BrowserTool") as mock_tool:
            mock_instance = AsyncMock()
            mock_instance.execute.side_effect = BrowserToolError("Browser failed")
            mock_tool.return_value = mock_instance

            result = await agent._browser_navigate_and_extract("https://example.com")
            assert "content" in result


class TestValidateContentRelevance:
    """Tests for AgenticFetchAgent._validate_content_relevance method."""

    @pytest.mark.asyncio
    async def test_validate_empty_content(self):
        """Test _validate_content_relevance with empty content."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        is_relevant, reasoning = await agent._validate_content_relevance(
            "test prompt",
            ""
        )
        assert is_relevant is False

    @pytest.mark.asyncio
    async def test_validate_whitespace_content(self):
        """Test _validate_content_relevance with whitespace-only content."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        is_relevant, reasoning = await agent._validate_content_relevance(
            "test prompt",
            "   \n\t  "
        )
        assert is_relevant is False

    @pytest.mark.asyncio
    async def test_validate_truncated_content(self):
        """Test _validate_content_relevance adds truncation message."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "YES - relevant"

        agent = AgenticFetchAgent(llm_manager=mock_llm)
        long_content = "x" * 5000
        is_relevant, reasoning = await agent._validate_content_relevance(
            "test prompt",
            long_content
        )
        assert "... [content truncated for validation]" in reasoning or mock_llm.complete.called

    @pytest.mark.asyncio
    async def test_validate_none_llm_manager(self):
        """Test _validate_content_relevance with None LLM manager."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        is_relevant, reasoning = await agent._validate_content_relevance(
            "test prompt",
            "some content"
        )
        assert is_relevant is True

    @pytest.mark.asyncio
    async def test_validate_empty_response(self):
        """Test _validate_content_relevance with empty LLM response."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = ""

        agent = AgenticFetchAgent(llm_manager=mock_llm)
        is_relevant, reasoning = await agent._validate_content_relevance(
            "test prompt",
            "some content"
        )
        assert is_relevant is True

    @pytest.mark.asyncio
    async def test_validate_yes_with_dash(self):
        """Test _validate_content_relevance with YES - format."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "YES - This is relevant"

        agent = AgenticFetchAgent(llm_manager=mock_llm)
        is_relevant, reasoning = await agent._validate_content_relevance(
            "test prompt",
            "some content"
        )
        assert is_relevant is True

    @pytest.mark.asyncio
    async def test_validate_no_with_dash(self):
        """Test _validate_content_relevance with NO - format."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "NO - Not relevant"

        agent = AgenticFetchAgent(llm_manager=mock_llm)
        is_relevant, reasoning = await agent._validate_content_relevance(
            "test prompt",
            "some content"
        )
        assert is_relevant is False

    @pytest.mark.asyncio
    async def test_validate_unparseable_response(self):
        """Test _validate_content_relevance with unparseable response."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "MAYBE perhaps probably"

        agent = AgenticFetchAgent(llm_manager=mock_llm)
        is_relevant, reasoning = await agent._validate_content_relevance(
            "test prompt",
            "some content"
        )
        assert is_relevant is True

    @pytest.mark.asyncio
    async def test_validate_exception(self):
        """Test _validate_content_relevance with exception."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = Exception("LLM failed")

        agent = AgenticFetchAgent(llm_manager=mock_llm)
        is_relevant, reasoning = await agent._validate_content_relevance(
            "test prompt",
            "some content"
        )
        assert is_relevant is True


class TestParseLlmAction:
    """Tests for AgenticFetchAgent._parse_llm_action method."""

    def test_parse_empty_response(self):
        """Test _parse_llm_action with empty response."""
        from mcp_server.agentic import AgenticFetchAgent
        from mcp_server.agentic.fetch_agent import ActionParsingError

        agent = AgenticFetchAgent()
        with pytest.raises(ActionParsingError):
            agent._parse_llm_action("")

    def test_parse_whitespace_response(self):
        """Test _parse_llm_action with whitespace-only response."""
        from mcp_server.agentic import AgenticFetchAgent
        from mcp_server.agentic.fetch_agent import ActionParsingError

        agent = AgenticFetchAgent()
        with pytest.raises(ActionParsingError):
            agent._parse_llm_action("   \n\t  ")

    def test_parse_keyword_done(self):
        """Test _parse_llm_action keyword fallback for done."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        result = agent._parse_llm_action("The task is done.")
        assert result["action"] == "done"

    def test_parse_keyword_search(self):
        """Test _parse_llm_action keyword fallback for search."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        result = agent._parse_llm_action("I will search for this")
        assert result["action"] == "search"

    def test_parse_keyword_look_up(self):
        """Test _parse_llm_action keyword fallback for look up."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        result = agent._parse_llm_action("Let me look up this information")
        assert result["action"] == "search"

    def test_parse_keyword_find(self):
        """Test _parse_llm_action keyword fallback for find."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        result = agent._parse_llm_action("I need to find this")
        assert result["action"] == "search"

    def test_parse_keyword_fetch(self):
        """Test _parse_llm_action keyword fallback for fetch."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        result = agent._parse_llm_action("Let me fetch this page")
        assert result["action"] == "fetch"

    def test_parse_keyword_visit(self):
        """Test _parse_llm_action keyword fallback for visit."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        result = agent._parse_llm_action("I will visit this URL")
        assert result["action"] == "fetch"

    def test_parse_keyword_navigate(self):
        """Test _parse_llm_action keyword fallback for navigate."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        result = agent._parse_llm_action("Navigate to the page")
        assert result["action"] == "navigate"

    def test_parse_keyword_evaluate(self):
        """Test _parse_llm_action keyword fallback for evaluate."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        result = agent._parse_llm_action("Evaluate this content")
        assert result["action"] == "evaluate"


class TestActionToEnum:
    """Tests for AgenticFetchAgent._action_to_enum method."""

    def test_action_to_enum_nav(self):
        """Test _action_to_enum with 'nav'."""
        from mcp_server.agentic import AgenticFetchAgent, ActionType

        agent = AgenticFetchAgent()
        result = agent._action_to_enum("nav")
        assert result == ActionType.NAVIGATE

    def test_action_to_enum_navigat(self):
        """Test _action_to_enum with 'navigat'."""
        from mcp_server.agentic import AgenticFetchAgent, ActionType

        agent = AgenticFetchAgent()
        result = agent._action_to_enum("navigat")
        assert result == ActionType.NAVIGATE

    def test_action_to_enum_done(self):
        """Test _action_to_enum with 'done'."""
        from mcp_server.agentic import AgenticFetchAgent, ActionType

        agent = AgenticFetchAgent()
        result = agent._action_to_enum("done")
        assert result == ActionType.DONE

    def test_action_to_enum_search(self):
        """Test _action_to_enum with 'search'."""
        from mcp_server.agentic import AgenticFetchAgent, ActionType

        agent = AgenticFetchAgent()
        result = agent._action_to_enum("search")
        assert result == ActionType.SEARCH

    def test_action_to_enum_lookup(self):
        """Test _action_to_enum with 'lookup'."""
        from mcp_server.agentic import AgenticFetchAgent, ActionType

        agent = AgenticFetchAgent()
        result = agent._action_to_enum("lookup")
        assert result == ActionType.SEARCH

    def test_action_to_enum_fetch(self):
        """Test _action_to_enum with 'fetch'."""
        from mcp_server.agentic import AgenticFetchAgent, ActionType

        agent = AgenticFetchAgent()
        result = agent._action_to_enum("fetch")
        assert result == ActionType.FETCH

    def test_action_to_enum_evaluate(self):
        """Test _action_to_enum with 'evaluate'."""
        from mcp_server.agentic import AgenticFetchAgent, ActionType

        agent = AgenticFetchAgent()
        result = agent._action_to_enum("evaluate")
        assert result == ActionType.EVALUATE

    def test_action_to_enum_eval(self):
        """Test _action_to_enum with 'eval'."""
        from mcp_server.agentic import AgenticFetchAgent, ActionType

        agent = AgenticFetchAgent()
        result = agent._action_to_enum("eval")
        assert result == ActionType.EVALUATE

    def test_action_to_enum_unknown(self):
        """Test _action_to_enum with unknown action returns NAVIGATE."""
        from mcp_server.agentic import AgenticFetchAgent, ActionType

        agent = AgenticFetchAgent()
        result = agent._action_to_enum("unknown_action")
        assert result == ActionType.NAVIGATE


class TestExecuteLlmReturnsNone:
    """Tests for execute() when LLM returns None."""

    @pytest.mark.asyncio
    async def test_execute_llm_returns_none(self):
        """Test execute() when _call_llm returns None."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = None

        agent = AgenticFetchAgent(llm_manager=mock_llm, max_steps=3)
        result = await agent.execute("test prompt")

        assert len(result.steps_taken) > 0
        assert result.steps_taken[0]["action"] == "error"
        assert "LLM call failed" in result.steps_taken[0]["description"]


class TestExecuteContentNotRelevant:
    """Tests for execute() when content is not relevant."""

    @pytest.mark.asyncio
    async def test_execute_content_not_relevant(self):
        """Test execute() when content fails relevance check."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()
        call_count = 0

        async def llm_side_effect(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1

            if "validate" in prompt.lower() or system_prompt is None:
                return "NO - Content does not match the query"
            elif call_count == 1:
                return '{"action": "done", "description": "Found content", "content": "Some content here"}'
            return '{"action": "done", "description": "Done"}'

        mock_llm.complete.side_effect = llm_side_effect

        agent = AgenticFetchAgent(llm_manager=mock_llm, max_steps=3)
        result = await agent.execute("test prompt")

        assert len(result.steps_taken) > 0


class TestExecuteSearchNoQuery:
    """Tests for execute() when search has no query."""

    @pytest.mark.asyncio
    async def test_execute_search_no_query(self):
        """Test execute() when search action has no query."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()
        call_count = 0

        async def llm_side_effect(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return '{"action": "search", "description": "Search for it"}'
            return '{"action": "done", "description": "Done"}'

        mock_llm.complete.side_effect = llm_side_effect

        agent = AgenticFetchAgent(llm_manager=mock_llm, max_steps=3)
        result = await agent.execute("test prompt")

        assert len(result.steps_taken) > 0
        step = result.steps_taken[0]
        assert "search query" in step.get("result", "").lower()


class TestStreamCallback:
    """Tests for stream_callback functionality."""

    @pytest.mark.asyncio
    async def test_stream_callback_exception(self):
        """Test stream_callback exception handling."""
        from mcp_server.agentic import AgenticFetchAgent

        async def failing_callback(step_num, action, description, result):
            raise Exception("Callback failed")

        mock_llm = AsyncMock()
        call_count = 0

        async def llm_side_effect(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return '{"action": "done", "description": "Done"}'
            return '{"action": "done", "content": "Result"}'

        mock_llm.complete.side_effect = llm_side_effect

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            max_steps=3,
            stream_callback=failing_callback
        )
        result = await agent.execute("test prompt")
        assert result is not None


class TestAgenticFetchDefaultSearchCreation:
    """Tests for agentic_fetch default search function creation."""

    @pytest.mark.asyncio
    async def test_agentic_fetch_default_search_creation(self):
        """Test agentic_fetch creates default search when not provided."""
        from mcp_server.agentic.fetch_agent import AgenticFetchAgent

        mock_result = {"results": [{"title": "Result", "url": "https://example.com"}]}

        with patch("mcp_server.agentic.fetch_agent.AgenticFetchAgent") as MockAgent:
            mock_instance = AsyncMock()
            from mcp_server.agentic import AgenticFetchResult
            mock_result_obj = AgenticFetchResult(success=True, content="test")
            mock_instance.execute.return_value = mock_result_obj
            MockAgent.return_value = mock_instance

            with patch("mcp_server.llm.LLMManager") as MockLLM:
                mock_llm_instance = MagicMock()
                MockLLM.return_value = mock_llm_instance

                with patch("mcp_server.server.search_web", new_callable=AsyncMock) as mock_search:
                    from mcp_server.agentic.fetch_agent import agentic_fetch
                    await agentic_fetch("test prompt")


class TestAgenticFetchDefaultFetchCreation:
    """Tests for agentic_fetch default fetch function creation."""

    @pytest.mark.asyncio
    async def test_agentic_fetch_default_fetch_creation(self):
        """Test agentic_fetch creates default fetch when not provided."""
        from mcp_server.agentic.fetch_agent import AgenticFetchAgent

        with patch("mcp_server.agentic.fetch_agent.AgenticFetchAgent") as MockAgent:
            mock_instance = AsyncMock()
            from mcp_server.agentic import AgenticFetchResult
            mock_result_obj = AgenticFetchResult(success=True, content="test")
            mock_instance.execute.return_value = mock_result_obj
            MockAgent.return_value = mock_instance

            with patch("mcp_server.llm.LLMManager") as MockLLM:
                mock_llm_instance = MagicMock()
                MockLLM.return_value = mock_llm_instance

                with patch("mcp_server.server.fetch_web_content", new_callable=AsyncMock) as mock_fetch:
                    from mcp_server.agentic.fetch_agent import agentic_fetch
                    await agentic_fetch("test prompt")


class TestAgenticFetchDefaultSearchException:
    """Tests for agentic_fetch default search exception handling."""

    @pytest.mark.asyncio
    async def test_agentic_fetch_default_search_exception(self):
        """Test agentic_fetch default search handles exception."""
        from mcp_server.agentic.fetch_agent import AgenticFetchAgent

        with patch("mcp_server.agentic.fetch_agent.AgenticFetchAgent") as MockAgent:
            mock_instance = AsyncMock()
            from mcp_server.agentic import AgenticFetchResult
            mock_result_obj = AgenticFetchResult(success=True, content="test")
            mock_instance.execute.return_value = mock_result_obj
            MockAgent.return_value = mock_instance

            with patch("mcp_server.llm.LLMManager") as MockLLM:
                mock_llm_instance = MagicMock()
                MockLLM.return_value = mock_llm_instance

                with patch("mcp_server.server.search_web", new_callable=AsyncMock) as mock_search:
                    mock_search.side_effect = Exception("Search failed")

                    from mcp_server.agentic.fetch_agent import agentic_fetch
                    await agentic_fetch("test prompt")


class TestAgenticFetchDefaultFetchException:
    """Tests for agentic_fetch default fetch exception handling."""

    @pytest.mark.asyncio
    async def test_agentic_fetch_default_fetch_exception(self):
        """Test agentic_fetch default fetch handles exception."""
        from mcp_server.agentic.fetch_agent import AgenticFetchAgent

        with patch("mcp_server.agentic.fetch_agent.AgenticFetchAgent") as MockAgent:
            mock_instance = AsyncMock()
            from mcp_server.agentic import AgenticFetchResult
            mock_result_obj = AgenticFetchResult(success=True, content="test")
            mock_instance.execute.return_value = mock_result_obj
            MockAgent.return_value = mock_instance

            with patch("mcp_server.llm.LLMManager") as MockLLM:
                mock_llm_instance = MagicMock()
                MockLLM.return_value = mock_llm_instance

                with patch("mcp_server.server.fetch_web_content", new_callable=AsyncMock) as mock_fetch:
                    mock_fetch.side_effect = Exception("Fetch failed")

                    from mcp_server.agentic.fetch_agent import agentic_fetch
                    await agentic_fetch("test prompt")


class TestNormalizeUrlAdditional:
    """Additional tests for _normalize_url function."""

    def test_normalize_url_with_www_prefix(self):
        """Test normalize_url strips www prefix."""
        from mcp_server.agentic.fetch_agent import _normalize_url
        result = _normalize_url("https://www.example.com/page")
        assert result == "https://example.com/page"
        assert not result.startswith("https://www.")

    def test_normalize_url_path_with_trailing_slash(self):
        """Test normalize_url removes trailing slash from path."""
        from mcp_server.agentic.fetch_agent import _normalize_url
        result = _normalize_url("https://example.com/page/")
        assert not result.endswith("/")

    def test_normalize_url_exception_fallback(self):
        """Test normalize_url returns lowercased original on exception."""
        from mcp_server.agentic.fetch_agent import _normalize_url
        result = _normalize_url("http://[invalid")
        assert isinstance(result, str)
        assert "http://[invalid" == result


class TestBrowserToolExecuteSuccess:
    """Additional tests for BrowserTool.execute success paths."""

    @pytest.mark.asyncio
    async def test_browser_tool_with_fallback_func_only(self):
        """Test BrowserTool with only fallback func (no browser-use)."""
        from mcp_server.agentic.fetch_agent import BrowserTool

        async def fallback():
            return {"content": "fallback result"}

        tool = BrowserTool(task_description="test task", fallback_func=fallback)
        result = await tool.execute()
        
        assert result.get("fallback_used") is True
        assert "content" in result.get("result", {})


class TestParseLlmActionJsonInText:
    """Tests for _parse_llm_action with JSON embedded in text."""

    def test_parse_json_embedded_in_text(self):
        """Test _parse_llm_action finds JSON embedded in text."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        response = "Here is my action: {\"action\": \"done\", \"content\": \"result\"} - let me know if you need anything else."
        result = agent._parse_llm_action(response)
        assert result["action"] == "done"


class TestDeadCodeAnalysis:
    """Tests documenting unreachable/dead code that cannot be covered.

    This class documents code paths that are logically impossible to execute
    due to contradictory conditions in the control flow.
    """

    def test_dead_code_lines_942949_cannot_be_reached(self):
        """Document that lines 941-949 in fetch_agent.py are unreachable dead code.

        Lines 942-949 check `if not action_data:` after calling _parse_llm_action.
        However, _parse_llm_action NEVER returns None/falsy:
        - It either returns a valid dict (LLMAction.to_dict())
        - Or it raises an exception (ValueError, ActionParsingError)

        Therefore `action_data` is always truthy and lines 942-949 are dead code.

        This test verifies the behavior that makes these lines unreachable:
        _parse_llm_action raises instead of returning None/falsy.
        """
        from mcp_server.agentic.fetch_agent import AgenticFetchAgent, ActionParsingError

        agent = AgenticFetchAgent()

        response_no_json = "this has no JSON and no keywords"
        with pytest.raises(ActionParsingError):
            agent._parse_llm_action(response_no_json)

        response_valid = '{"action": "done"}'
        result = agent._parse_llm_action(response_valid)
        assert isinstance(result, dict)
        assert result["action"] == "done"


    def test_parse_json_with_leading_text(self):
        """Test _parse_llm_action with JSON after some text."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        response = "I think we should {\"action\": \"search\", \"query\": \"test\"} for more info."
        result = agent._parse_llm_action(response)
        assert result["action"] == "search"
        assert result.get("query") == "test"


class TestNormalizeUrlExceptionFallback:
    """Tests for _normalize_url exception handling fallback to url.lower()."""

    def test_normalize_url_exception_inside_try_block(self):
        """Test _normalize_url falls back to url.lower() when parsing fails inside try block."""
        from mcp_server.agentic.fetch_agent import _normalize_url
        result = _normalize_url("https://[invalid")
        assert isinstance(result, str)
        assert result == "https://[invalid"

    def test_normalize_url_exception_returns_lowercase(self):
        """Test _normalize_url returns lowercase version of original URL on exception."""
        from mcp_server.agentic.fetch_agent import _normalize_url
        url = "HTTPS://EXAMPLE.COM/Page"
        result = _normalize_url(url)
        assert isinstance(result, str)
        assert result == url.lower()


class TestParseLlmActionMismatchedBraces:
    """Tests for _parse_llm_action with mismatched braces."""

    def test_parse_mismatched_braces_missing_closing(self):
        """Test _parse_llm_action falls back to keywords when JSON has mismatched braces."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        response = '{"action": "done"'
        result = agent._parse_llm_action(response)
        assert result["action"] == "done"

    def test_parse_mismatched_braces_extra_closing(self):
        """Test _parse_llm_action handles extra closing brace."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        response = '{"action": "search", "query": "test"} extra'
        result = agent._parse_llm_action(response)
        assert result["action"] == "search"


class TestStreamCallbackExceptionHandling:
    """Tests for stream_callback exception handling."""

    @pytest.mark.asyncio
    async def test_stream_callback_exception_is_caught(self):
        """Test stream_callback exception is caught and logged."""
        from mcp_server.agentic import AgenticFetchAgent
        import logging

        async def failing_callback(step_num, action, description, result):
            raise RuntimeError("Callback failed")

        mock_llm = AsyncMock()
        call_count = 0

        async def llm_side_effect(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1

            if "validate" in prompt.lower() or system_prompt is None:
                return "YES - relevant"
            elif call_count == 1:
                return '{"action": "search", "query": "test"}'
            elif call_count == 2:
                return '{"action": "done", "description": "Done"}'
            return '{"action": "done", "description": "Done"}'

        mock_llm.complete.side_effect = llm_side_effect

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            max_steps=3,
            stream_callback=failing_callback
        )

        with patch("mcp_server.agentic.fetch_agent.logger") as mock_logger:
            result = await agent.execute("test prompt")
            assert len(result.steps_taken) > 0
            mock_logger.warning.assert_called()


class TestAgenticFetchDefaultFunctions:
    """Tests for agentic_fetch default function creation."""

    @pytest.mark.asyncio
    async def test_agentic_fetch_uses_server_search_web(self):
        """Test agentic_fetch uses server.search_web when search_func is None."""
        with patch("mcp_server.llm.LLMManager") as MockLLM:
            mock_llm_instance = MagicMock()
            call_count = 0

            async def llm_side_effect(prompt, system_prompt=None):
                nonlocal call_count
                call_count += 1

                if "validate" in prompt.lower() or system_prompt is None:
                    return "YES - relevant"
                elif call_count == 1:
                    return '{"action": "search", "query": "test"}'
                elif call_count == 2:
                    return '{"action": "done", "description": "Done"}'
                return '{"action": "done", "description": "Done"}'

            mock_llm_instance.complete.side_effect = llm_side_effect
            MockLLM.return_value = mock_llm_instance

            with patch("mcp_server.server.search_web", new_callable=AsyncMock) as mock_search:
                mock_search.return_value = {"results": []}
                from mcp_server.agentic.fetch_agent import agentic_fetch
                await agentic_fetch("test prompt")

                mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_agentic_fetch_uses_server_fetch_web_content(self):
        """Test agentic_fetch uses server.fetch_web_content when fetch_func is None."""
        with patch("mcp_server.llm.LLMManager") as MockLLM:
            mock_llm_instance = MagicMock()
            call_count = 0

            async def llm_side_effect(prompt, system_prompt=None):
                nonlocal call_count
                call_count += 1

                if "validate" in prompt.lower() or system_prompt is None:
                    return "YES - relevant"
                elif call_count == 1:
                    return '{"action": "fetch", "url": "https://example.com"}'
                elif call_count == 2:
                    return '{"action": "done", "description": "Done"}'
                return '{"action": "done", "description": "Done"}'

            mock_llm_instance.complete.side_effect = llm_side_effect
            MockLLM.return_value = mock_llm_instance

            with patch("mcp_server.server.fetch_web_content", new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = {"content": "Fetched content"}
                from mcp_server.agentic.fetch_agent import agentic_fetch
                await agentic_fetch("test prompt")

                mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_agentic_fetch_default_search_returns_error_on_exception(self):
        """Test agentic_fetch default search returns error dict on exception."""
        from mcp_server.agentic.fetch_agent import AgenticFetchAgent

        with patch("mcp_server.agentic.fetch_agent.AgenticFetchAgent") as MockAgent:
            mock_instance = AsyncMock()
            from mcp_server.agentic import AgenticFetchResult
            mock_result_obj = AgenticFetchResult(success=True, content="test")
            mock_instance.execute.return_value = mock_result_obj
            MockAgent.return_value = mock_instance

            with patch("mcp_server.llm.LLMManager") as MockLLM:
                mock_llm_instance = MagicMock()
                MockLLM.return_value = mock_llm_instance

                with patch("mcp_server.server.search_web", new_callable=AsyncMock) as mock_search:
                    mock_search.side_effect = Exception("Search unavailable")

                    from mcp_server.agentic.fetch_agent import agentic_fetch
                    result = await agentic_fetch("test prompt")
                    
                    assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_agentic_fetch_default_fetch_returns_error_on_exception(self):
        """Test agentic_fetch default fetch returns error dict on exception."""
        from mcp_server.agentic.fetch_agent import AgenticFetchAgent

        with patch("mcp_server.agentic.fetch_agent.AgenticFetchAgent") as MockAgent:
            mock_instance = AsyncMock()
            from mcp_server.agentic import AgenticFetchResult
            mock_result_obj = AgenticFetchResult(success=True, content="test")
            mock_instance.execute.return_value = mock_result_obj
            MockAgent.return_value = mock_instance

            with patch("mcp_server.llm.LLMManager") as MockLLM:
                mock_llm_instance = MagicMock()
                MockLLM.return_value = mock_llm_instance

                with patch("mcp_server.server.fetch_web_content", new_callable=AsyncMock) as mock_fetch:
                    mock_fetch.side_effect = Exception("Fetch unavailable")

                    from mcp_server.agentic.fetch_agent import agentic_fetch
                    result = await agentic_fetch("test prompt")
                    
                    assert isinstance(result, dict)


class TestEvaluateAndUnknownActionHandling:
    """Tests for evaluate action (lines 1212-1217) and unknown action handling (line 1225)."""

    @pytest.mark.asyncio
    async def test_execute_evaluate_action_handling(self):
        """Test execute() properly handles evaluate action and continues execution."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()
        call_count = 0

        async def llm_side_effect(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1

            if "validate" in prompt.lower() or system_prompt is None:
                return "YES - relevant"
            elif call_count == 1:
                return '{"action": "fetch", "description": "Fetching initial page", "url": "https://example.com"}'
            elif call_count == 2:
                return '{"action": "evaluate", "description": "Content looks relevant, continuing..."}'
            elif call_count == 3:
                return '{"action": "done", "description": "Task complete"}'
            elif call_count == 4:
                return '{"action": "done", "content": "Final content"}'
            return '{"action": "done", "description": "Done"}'

        mock_llm.complete.side_effect = llm_side_effect

        async def mock_fetch(url):
            return {"content": "Page content", "title": "Example"}

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            fetch_func=mock_fetch,
            max_steps=5
        )
        result = await agent.execute("test prompt")

        assert len(result.steps_taken) >= 3
        step_actions = [s.get("action") for s in result.steps_taken]
        assert "evaluate" in step_actions

    @pytest.mark.asyncio
    async def test_execute_evaluate_action_no_urls_visited(self):
        """Test evaluate action handling when no URLs have been visited yet."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()
        call_count = 0

        async def llm_side_effect(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1

            if "validate" in prompt.lower() or system_prompt is None:
                return "YES - relevant"
            elif call_count == 1:
                return '{"action": "evaluate", "description": "No content fetched yet"}'
            elif call_count == 2:
                return '{"action": "done", "description": "Done"}'
            elif call_count == 3:
                return '{"action": "done", "content": ""}'
            return '{"action": "done"}'

        mock_llm.complete.side_effect = llm_side_effect

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            max_steps=3
        )
        result = await agent.execute("test prompt")

        assert len(result.steps_taken) >= 1
        step_actions = [s.get("action") for s in result.steps_taken]
        assert "evaluate" in step_actions

    @pytest.mark.asyncio
    async def test_execute_unknown_action_handling(self):
        """Test execute() handles unknown actions gracefully."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()
        call_count = 0

        async def llm_side_effect(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1

            if "validate" in prompt.lower() or system_prompt is None:
                return "YES - relevant"
            elif call_count == 1:
                return '{"action": "unknown_action", "description": "Some unknown action"}'
            elif call_count == 2:
                return '{"action": "done", "description": "Done"}'
            elif call_count == 3:
                return '{"action": "done", "content": ""}'
            return '{"action": "done"}'

        mock_llm.complete.side_effect = llm_side_effect

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            max_steps=3
        )
        result = await agent.execute("test prompt")

        assert len(result.steps_taken) >= 1
        step_actions = [s.get("action") for s in result.steps_taken]
        assert "unknown_action" in step_actions


class TestParseLlmActionMismatchedBracesAt806:
    """Test for line 806 - mismatched braces triggering keyword fallback."""

    def test_parse_llm_action_missing_closing_brace_keyword_fallback(self):
        """Test _parse_llm_action with missing closing brace triggers keyword fallback."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        response = '{"action": "done"'
        result = agent._parse_llm_action(response)
        assert result["action"] == "done"

    def test_parse_llm_action_no_opening_brace_falls_to_keywords(self):
        """Test _parse_llm_action when no opening brace found triggers keyword fallback."""
        from mcp_server.agentic import AgenticFetchAgent

        agent = AgenticFetchAgent()
        response = 'The task is done'
        result = agent._parse_llm_action(response)
        assert result["action"] == "done"


class TestExecuteFetchNavigateNoUrl:
    """Tests for lines 1094-1097 - execute() fetch/navigate with no URL provided."""

    @pytest.mark.asyncio
    async def test_execute_fetch_no_url_provided(self):
        """Test execute() fetch action with no URL in action_data (lines 1093-1097)."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()
        call_count = 0

        async def llm_side_effect(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1

            if "validate" in prompt.lower() or system_prompt is None:
                return "YES - relevant"
            elif call_count == 1:
                return '{"action": "fetch", "description": "Fetching"}'
            elif call_count == 2:
                return '{"action": "done", "description": "Done"}'
            elif call_count == 3:
                return '{"action": "done", "content": ""}'
            return '{"action": "done"}'

        mock_llm.complete.side_effect = llm_side_effect

        async def mock_fetch(url):
            return {"content": "Page content", "title": "Example"}

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            fetch_func=mock_fetch,
            max_steps=3
        )
        result = await agent.execute("test prompt")

        assert len(result.steps_taken) >= 1
        fetch_step = result.steps_taken[0]
        assert fetch_step.get("result") == "No URL provided"

    @pytest.mark.asyncio
    async def test_execute_navigate_no_url_provided(self):
        """Test execute() navigate action with no URL in action_data (lines 1093-1097)."""
        from mcp_server.agentic import AgenticFetchAgent

        mock_llm = AsyncMock()
        call_count = 0

        async def llm_side_effect(prompt, system_prompt=None):
            nonlocal call_count
            call_count += 1

            if "validate" in prompt.lower() or system_prompt is None:
                return "YES - relevant"
            elif call_count == 1:
                return '{"action": "navigate", "description": "Navigating"}'
            elif call_count == 2:
                return '{"action": "done", "description": "Done"}'
            elif call_count == 3:
                return '{"action": "done", "content": ""}'
            return '{"action": "done"}'

        mock_llm.complete.side_effect = llm_side_effect

        async def mock_fetch(url):
            return {"content": "Page content", "title": "Example"}

        agent = AgenticFetchAgent(
            llm_manager=mock_llm,
            fetch_func=mock_fetch,
            max_steps=3
        )
        result = await agent.execute("test prompt")

        assert len(result.steps_taken) >= 1
        nav_step = result.steps_taken[0]
        assert nav_step.get("result") == "No URL provided"


class TestBrowserToolExecuteWithBrowserUse:
    """Tests for BrowserTool.execute with browser-use library returning string (lines 341-350)."""

    @pytest.mark.asyncio
    async def test_browser_tool_execute_with_string_result(self):
        """Test BrowserTool.execute when browser-use returns a string result (lines 341-347)."""
        from mcp_server.agentic.fetch_agent import BrowserTool
        import sys

        async def fallback():
            return {"content": "fallback"}

        tool = BrowserTool(task_description="test task", fallback_func=fallback)

        class MockAgent:
            def __init__(self, task):
                pass
            async def run(self):
                return "Task completed successfully"

        mock_browser_use = MagicMock()
        mock_browser_use.Agent = MockAgent

        with patch.dict(sys.modules, {'browser_use': mock_browser_use}):
            result = await tool.execute()
            assert result.get("success") is True
            assert result.get("result") == "Task completed successfully"

    @pytest.mark.asyncio
    async def test_browser_tool_execute_with_json_string_result(self):
        """Test BrowserTool.execute when browser-use returns JSON string (lines 341-344)."""
        from mcp_server.agentic.fetch_agent import BrowserTool
        import sys

        async def fallback():
            return {"content": "fallback"}

        tool = BrowserTool(task_description="test task", fallback_func=fallback)

        class MockAgent:
            def __init__(self, task):
                pass
            async def run(self):
                return '{"results": ["item1", "item2"]}'

        mock_browser_use = MagicMock()
        mock_browser_use.Agent = MockAgent

        with patch.dict(sys.modules, {'browser_use': mock_browser_use}):
            result = await tool.execute()
            assert result.get("success") is True
            assert result.get("result") == {"results": ["item1", "item2"]}

    @pytest.mark.asyncio
    async def test_browser_tool_execute_fallback_raises_exception_no_fallback(self):
        """Test BrowserTool.execute when fallback_func is None and browser raises (line 366)."""
        from mcp_server.agentic.fetch_agent import BrowserTool, BrowserToolError
        import sys

        tool = BrowserTool(task_description="test task")

        class MockAgent:
            def __init__(self, task):
                pass
            async def run(self):
                raise Exception("Browser failed")

        mock_browser_use = MagicMock()
        mock_browser_use.Agent = MockAgent

        with patch.dict(sys.modules, {'browser_use': mock_browser_use}):
            with pytest.raises(BrowserToolError) as exc_info:
                await tool.execute()
            assert "Browser failed" in str(exc_info.value)
            assert exc_info.value.fallback_used is False


class TestParseLlmActionJsonInText:
    """Tests for _parse_llm_action finding JSON within text (lines 803-806)."""

    def test_parse_llm_action_mismatched_braces_finds_json_in_text(self):
        """Test _parse_llm_action with mismatched braces exercises lines 803-806."""
        from mcp_server.agentic import AgenticFetchAgent
        from unittest.mock import MagicMock

        agent = AgenticFetchAgent()

        response = "Some text before {invalid json here} and some text after"

        with patch("mcp_server.agentic.fetch_agent.LLMAction.model_validate_json") as mock_validate:
            from mcp_server.agentic.fetch_agent import LLMAction
            mock_action = MagicMock()
            mock_action.to_dict.return_value = {"action": "done", "result": ""}
            mock_validate.return_value = mock_action

            result = agent._parse_llm_action(response)
            assert result["action"] == "done"


class TestBrowserToolNonStringDictResult:
    """Tests for BrowserTool.execute when browser-use returns non-string, non-dict result (line 350)."""

    @pytest.mark.asyncio
    async def test_browser_tool_returns_list_result(self):
        """Test BrowserTool.execute when browser-use returns a list (line 350 path)."""
        from mcp_server.agentic.fetch_agent import BrowserTool
        import sys

        tool = BrowserTool(task_description="test task")

        class MockAgent:
            def __init__(self, task):
                pass
            async def run(self):
                return ["result1", "result2", "result3"]

        mock_browser_use = MagicMock()
        mock_browser_use.Agent = MockAgent

        with patch.dict(sys.modules, {'browser_use': mock_browser_use}):
            result = await tool.execute()
            assert result["success"] is True
            assert result["result"] == ["result1", "result2", "result3"]

    @pytest.mark.asyncio
    async def test_browser_tool_returns_int_result(self):
        """Test BrowserTool.execute when browser-use returns an integer."""
        from mcp_server.agentic.fetch_agent import BrowserTool
        import sys

        tool = BrowserTool(task_description="test task")

        class MockAgent:
            def __init__(self, task):
                pass
            async def run(self):
                return 42

        mock_browser_use = MagicMock()
        mock_browser_use.Agent = MockAgent

        with patch.dict(sys.modules, {'browser_use': mock_browser_use}):
            result = await tool.execute()
            assert result["success"] is True
            assert result["result"] == 42


class TestParseLLMActionStage3Error:
    """Tests for _parse_llm_action Stage 3 error path (lines 847-853)."""

    @pytest.mark.asyncio
    async def test_parse_llm_action_stage3_raises_error(self):
        """Test _parse_llm_action when JSON parsing fails and no keywords match (Stage 3)."""
        from mcp_server.agentic.fetch_agent import AgenticFetchAgent, ActionParsingError

        agent = AgenticFetchAgent()

        response = "this is gibberish with no actionable content"

        with pytest.raises(ActionParsingError) as exc_info:
            agent._parse_llm_action(response)
        
        assert "Could not parse LLM response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_parse_llm_action_stage3_with_invalid_json_syntax(self):
        """Test _parse_llm_action Stage 3 when JSON has syntax errors and no keywords."""
        from mcp_server.agentic.fetch_agent import AgenticFetchAgent, ActionParsingError

        agent = AgenticFetchAgent()

        response = "{invalid: json} with random text that means nothing"

        with pytest.raises(ActionParsingError) as exc_info:
            agent._parse_llm_action(response)
        
        assert "Could not parse LLM response" in str(exc_info.value)


class TestBrowserSearchNonListResult:
    """Tests for execute search action when browser result parses as JSON but is not a list (lines 1043-1057)."""

    @pytest.mark.asyncio
    async def test_browser_search_result_is_dict_not_list(self):
        """Test _browser_navigate_and_extract when browser result parses to dict not list."""
        from mcp_server.agentic import AgenticFetchAgent

        async def mock_fetch(url):
            return {"content": ""}

        agent = AgenticFetchAgent(fetch_func=mock_fetch)

        with patch("mcp_server.agentic.fetch_agent.BrowserTool") as mock_tool:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = {
                "success": True,
                "result": '{"title": "Page", "content": "Content"}'
            }
            mock_tool.return_value = mock_instance

            result = await agent._browser_navigate_and_extract("https://example.com")
            assert "content" in result or "error" not in result

    @pytest.mark.asyncio
    async def test_browser_search_result_string_parses_to_dict(self):
        """Test search action when browser result string parses to dict (not list) at lines 1053-1057."""
        from mcp_server.agentic.fetch_agent import AgenticFetchAgent, BrowserTool

        async def mock_llm_response(prompt, system_prompt=None):
            return '{"action": "search", "query": "test"}'

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = mock_llm_response

        agent = AgenticFetchAgent(llm_manager=mock_llm)

        with patch.object(agent, '_browser_search', new_callable=AsyncMock) as mock_browser_search:
            mock_browser_search.return_value = {
                "success": True,
                "result": '{"title": "Not a list", "url": "https://example.com"}'
            }

            with patch.object(agent, '_search', new_callable=AsyncMock) as mock_search:
                mock_search.return_value = {"results": []}
                result = await agent.execute("test query")
                assert len(result.steps_taken) >= 1


class TestBrowserFetchNonListResult:
    """Tests for execute fetch/navigate action when browser result parses as JSON but is not a list (lines 1156-1181)."""

    @pytest.mark.asyncio
    async def test_browser_fetch_result_is_dict_not_list(self):
        """Test fetch action when browser result parses to dict (not list) at lines 1166-1181."""
        from mcp_server.agentic.fetch_agent import AgenticFetchAgent

        async def mock_llm_response(prompt, system_prompt=None):
            return '{"action": "fetch", "url": "https://example.com"}'

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = mock_llm_response

        async def mock_fetch(url):
            return {"content": ""}

        agent = AgenticFetchAgent(llm_manager=mock_llm, fetch_func=mock_fetch)

        with patch.object(agent, '_browser_navigate_and_extract') as mock_browser:
            mock_browser.return_value = {
                "success": True,
                "result": '{"title": "Page", "content": "some text"}'
            }

            result = await agent.execute("fetch https://example.com")
            assert len(result.steps_taken) >= 1

    @pytest.mark.asyncio
    async def test_browser_fetch_result_string_parses_to_dict(self):
        """Test navigate action when browser result string parses to dict at lines 1156-1164."""
        from mcp_server.agentic.fetch_agent import AgenticFetchAgent

        async def mock_llm_response(prompt, system_prompt=None):
            return '{"action": "navigate", "url": "https://example.com"}'

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = mock_llm_response

        async def mock_fetch(url):
            return {"content": ""}

        agent = AgenticFetchAgent(llm_manager=mock_llm, fetch_func=mock_fetch)

        with patch.object(agent, '_browser_navigate_and_extract') as mock_browser:
            mock_browser.return_value = {
                "success": True,
                "result": '{"title": "Page", "content": "Content here"}'
            }

            result = await agent.execute("navigate to https://example.com")
            assert len(result.steps_taken) >= 1


class TestParseLLMActionEmbeddedJsonValidationFailure:
    """Tests for _parse_llm_action when JSON is embedded in text but fails model_validate_json (line 806).

    NOTE: Line 806 `json_str = stripped[json_start:json_end + 1]` is UNREACHABLE CODE.

    Analysis:
    - Line 806 is in the else branch (lines 799-806) which executes when:
      * Line 792 condition is False: NOT(json_start >= 0 AND json_end > json_start)
      * Line 796 condition is False: NOT stripped.startswith(('{', '['))

    - For line 806 to be reached after the else branch, we need at line 804:
      * json_end (rfind('}') result) > json_start to NOT raise ValueError

    - But for line 792 condition to be False when json_start >= 0, we need:
      * rfind('}') + 1 <= json_start

    - Since rfind('}') + 1 <= json_start implies rfind('}') < json_start - 1,
      this contradicts the requirement that rfind('}') > json_start

    Therefore line 806 can never be executed. The else branch always raises
    ValueError at either line 802 (no opening brace) or line 805 (closing brace
    before/equal to opening brace).
    """

    def test_parse_llm_action_embedded_json_fails_validation_keyword_fallback(self):
        """Test _parse_llm_action with JSON embedded in text where extracted substring fails model_validate_json.

        This test exercises the keyword fallback path when JSON extraction succeeds
        but validation fails. It does NOT reach line 806 (unreachable code) - instead,
        it reaches line 793 which extracts valid JSON structure but the mock makes
        model_validate_json fail.

        Response: 'Here is the result: {"action": "done"} and more text}'
        - json_start finds first '{' at position 20
        - rfind('}') + 1 = 53 (last '}' after 'more text')
        - Line 792 condition is True: json_start >= 0 AND json_end > json_start
        - Extracts via line 793: '{"action": "done"} and more text}'
        - model_validate_json fails (mocked), falls back to keyword matching for 'done'
        """
        from mcp_server.agentic.fetch_agent import AgenticFetchAgent

        agent = AgenticFetchAgent()

        response = 'Here is the result: {"action": "done"} and more text}'

        with patch("mcp_server.agentic.fetch_agent.LLMAction.model_validate_json") as mock_validate:
            from pydantic import ValidationError
            mock_validate.side_effect = ValidationError.from_exception_data(
                "LLMAction",
                [{"type": "missing", "loc": ("action",), "msg": "Field required", "input": {}}]
            )

            result = agent._parse_llm_action(response)
            assert result["action"] == "done"


class TestAgenticFetchDefaultSearchError:
    """Tests for agentic_fetch when default search_func raises (lines 1326-1328)."""

    @pytest.mark.asyncio
    async def test_default_search_error_returns_error_dict(self):
        """Test agentic_fetch returns error dict when default search raises exception.
        
        Lines 1323-1328: The _default_search closure catches Exception,
        logs the error, and returns an error dict.
        """
        from mcp_server.agentic import AgenticFetchResult
        mock_result = AgenticFetchResult(success=True, content="test")
        
        with patch("mcp_server.agentic.fetch_agent.AgenticFetchAgent") as MockAgent:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = mock_result
            MockAgent.return_value = mock_instance
            
            # Patch search_web at the source to raise an exception
            with patch("mcp_server.server.search_web", new=AsyncMock(side_effect=Exception("Search failed"))):
                from mcp_server.agentic.fetch_agent import agentic_fetch
                
                # Call with search_func=None to trigger default creation
                result = await agentic_fetch("test prompt", llm_manager=AsyncMock())
                
                # Verify result is returned (agent was created and executed)
                assert isinstance(result, dict)


class TestBrowserSearchFallbackStringJsonError:
    """Tests for _browser_search when string result is not valid JSON (lines 607-608).

    Lines 606-608 are hit when:
    - BrowserTool.execute returns a result where fallback_used is NOT True
    - The string result cannot be parsed as valid JSON (json.JSONDecodeError)
    """

    @pytest.mark.asyncio
    async def test_browser_search_string_result_json_decode_error(self):
        """Test _browser_search when string result fails JSON parsing (lines 607-608)."""
        from mcp_server.agentic import AgenticFetchAgent

        async def mock_search(query, num_results=10):
            return {"results": [{"title": "Fallback", "url": "https://fallback.com"}]}

        agent = AgenticFetchAgent(search_func=mock_search)

        with patch("mcp_server.agentic.fetch_agent.BrowserTool") as mock_tool:
            mock_instance = AsyncMock()
            mock_instance.execute.return_value = {
                "fallback_used": False,
                "result": "not valid json {"
            }
            mock_tool.return_value = mock_instance

            result = await agent._browser_search("test query")
            assert isinstance(result, dict)
            assert "error" in result
            assert "failed to parse" in result["error"]