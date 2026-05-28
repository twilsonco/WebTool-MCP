"""
Agentic AI fetch agent using browser-use for autonomous web browsing.

This module provides an agent that can:
- Take a natural language prompt
- Plan and execute web searches/fetches using AI decision-making
- Navigate, click, scroll as needed using browser automation
- Return found content or a detailed report of URLs visited if not found
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Enumeration of valid action types for the agent."""
    
    SEARCH = "search"
    FETCH = "fetch"
    NAVIGATE = "navigate"
    EVALUATE = "evaluate"
    DONE = "done"


class ActionParsingError(Exception):
    """
    Exception raised when the LLM response cannot be parsed into an action.
    
    This provides detailed context about parsing failures for debugging
    and error handling purposes.
    """
    
    def __init__(self, message: str, raw_response: str = "", cause: Optional[Exception] = None):
        super().__init__(message)
        self.raw_response = raw_response
        self.cause = cause


class LLMAction(BaseModel):
    """
    Pydantic model representing an action parsed from LLM response.
    
    This provides type-safe validation of LLM outputs with clear field
    documentation and validation rules.
    
    Attributes:
        action: The type of action to perform (search, fetch, navigate, evaluate, done)
        query: Optional search query for SEARCH actions
        url: Optional URL for FETCH or NAVIGATE actions
        result: Final content for DONE actions when successful
        reasoning: Explanation of why this action was chosen (used internally)
    """
    
    action: str = Field(
        ...,
        description="The type of action to perform",
        examples=["search", "fetch", "navigate", "evaluate", "done"]
    )
    query: Optional[str] = Field(
        default=None,
        description="Search query for SEARCH actions"
    )
    url: Optional[str] = Field(
        default=None,
        description="URL to fetch or navigate for FETCH/NAVIGATE actions"
    )
    result: Optional[str] = Field(
        default=None,
        validation_alias="content",  # Accept "content" from LLM JSON output
        description="Final content for DONE actions when successfully completed"
    )
    
    reasoning: Optional[str] = Field(
        default=None,
        validation_alias="description",  # Accept "description" from LLM JSON
        description="Explanation of why this action was chosen"
    )
    
    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Validate action is one of the known action types."""
        normalized = v.lower().strip()
        
        # Accept any valid action type even if not in our enum
        valid_actions = {a.value for a in ActionType}
        
        # Support legacy spellings/variations
        if normalized == "nav" or normalized.startswith("navigat"):
            return ActionType.NAVIGATE.value
        elif normalized in valid_actions:
            return normalized
        
        # Return as-is for backward compatibility; execute() will handle unknown actions
        return normalized
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "action": self.action,
            "description": self.reasoning or "",
            "query": self.query,
            "url": self.url,
            "content": self.result
        }


class BrowserToolError(Exception):
    """Exception raised when browser tool operations fail."""
    
    def __init__(self, message: str, fallback_used: bool = False):
        super().__init__(message)
        self.fallback_used = fallback_used


class BrowserTool:
    """
    Reusable browser automation tool using browser-use Agent.
    
    This class encapsulates the common pattern of:
    1. Creating a browser-use Agent with a task description
    2. Running the agent to completion
    3. Parsing and returning the result
    4. Falling back gracefully if browser-use is unavailable
    
    Example:
        async def fallback_search(query):
            return await regular_search_func(query)
        
        tool = BrowserTool(
            task_description=f"Search for: {query}",
            fallback_func=lambda: fallback_search(query)
        )
        result = await tool.execute()
    """
    
    def __init__(
        self,
        task_description: str,
        fallback_func: Optional[Callable[[], Any]] = None
    ):
        """
        Initialize BrowserTool.
        
        Args:
            task_description: The natural language task for the browser agent
            fallback_func: Optional async function to call if browser-use fails
        """
        self.task_description = task_description
        self.fallback_func = fallback_func
    
    async def execute(self) -> Dict[str, Any]:
        """
        Execute the browser agent task.
        
        Returns:
            Dictionary with either:
            - {"success": True, "result": <parsed result>}
            - {"success": False, "error": <error message>} if fallback also fails
            
        If browser-use is unavailable or fails, calls fallback_func if provided.
        """
        try:
            from browser_use import Agent as BrowserAgent
            
            agent = BrowserAgent(task=self.task_description)
            raw_result = await agent.run()
            
            # If result is a string, try to parse as JSON
            if isinstance(raw_result, str):
                try:
                    parsed = json.loads(raw_result)
                    return {"success": True, "result": parsed}
                except json.JSONDecodeError:
                    # Return raw string if not valid JSON
                    return {"success": True, "result": raw_result}
            
            # Otherwise return as-is
            return {"success": True, "result": raw_result}
            
        except ImportError:
            logger.info("browser-use not installed, using fallback")
            if self.fallback_func is not None:
                result = await self.fallback_func()
                return {"success": True, "result": result, "fallback_used": True}
            raise BrowserToolError(
                "browser-use not installed and no fallback provided",
                fallback_used=False
            )
        except Exception as e:
            logger.warning("BrowserTool execution failed: %s", str(e))
            if self.fallback_func is not None:
                result = await self.fallback_func()
                return {"success": True, "result": result, "fallback_used": True}
            raise BrowserToolError(str(e), fallback_used=False)


class FetchStep(BaseModel):
    """
    Pydantic model representing a single step in the agent's execution.
    
    Records what action was taken, any parameters used, and a preview
    of the result for logging and debugging purposes.
    
    Attributes:
        step_number: Sequential step number in the agent execution
        action: The type of action performed (search, fetch, navigate, evaluate)
        query: Search query if action was SEARCH
        url: URL if action involved a specific URL
        result_preview: First 500 characters of the step's result
        timestamp: When this step was executed (UTC)
    """
    
    step_number: int = Field(
        ...,
        ge=1,
        description="Sequential step number in agent execution"
    )
    action: str = Field(
        ...,
        description="Type of action performed (search, fetch, navigate, evaluate)"
    )
    query: Optional[str] = Field(
        default=None,
        description="Search query for SEARCH actions"
    )
    url: Optional[str] = Field(
        default=None,
        description="URL involved in this step (fetch/navigate URL or search query)"
    )
    result_preview: str = Field(
        default="",
        max_length=500,
        description="First 500 characters of result for preview"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this step was executed (UTC)"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility with existing code."""
        return {
            "step": self.step_number,
            "action": self.action,
            "description": "",  # No description field in FetchStep model
            "query": self.query,
            "url": self.url,
            "result_preview": self.result_preview[:500] if len(self.result_preview) > 500 else self.result_preview,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AgenticFetchResult:
    """Result of an agentic fetch operation."""
    
    success: bool
    content: Optional[str] = None
    url: Optional[str] = None
    urls_visited: List[Dict[str, str]] = field(default_factory=list)
    steps_taken: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary for JSON serialization."""
        return {
            "success": self.success,
            "content": self.content,
            "url": self.url,
            "urls_visited": self.urls_visited,
            "steps_taken": [
                {
                    "step": s.get("step", 0),
                    "action": s.get("action", ""),
                    "description": s.get("description", ""),
                    "url": s.get("url"),
                    "result_preview": str(s.get("result", ""))[:500] if s.get("result") else None
                }
                for s in self.steps_taken
            ],
            "error_message": self.error_message
        }


class AgenticFetchAgent:
    """
    AI agent that autonomously searches and browses the web to find information.
    
    The agent uses a loop-based approach:
    1. Analyze the user's prompt and decide on an action
    2. Execute the action (search, navigate, extract content)
    3. Evaluate results and decide next step or conclude
    4. Repeat until content is found or max_steps reached
    
    Falls back to using existing search/fetch tools when browser-use actions
    are not available or fail.
    """
    
    SYSTEM_PROMPT = """You are a web browsing agent that helps users find information on the internet.

Your capabilities:
1. Search the web using search_web tool
2. Fetch and extract content from URLs using fetch_web_content  
3. Navigate to URLs directly
4. Evaluate whether found content matches the user's request

Your approach:
1. First understand what information the user is looking for
2. Formulate a search query to find relevant pages
3. Execute searches and examine results
4. Fetch promising URLs to get detailed content
5. Evaluate if the content satisfies the user's request
6. If not found, try alternative searches or URLs
7. Report back with findings or detailed report of what was tried

Always be thorough but efficient. Try to find the most relevant and authoritative sources.

Output your next action as a JSON object with:
{
  "action": "search|fetch|navigate|evaluate|done",
  "description": "What you're doing and why",
  "query": "search query (for search action)",
  "url": "URL to fetch/navigate (for fetch/navigate actions)"
}

When you have found content that satisfies the user's request, output:
{
  "action": "done",
  "description": "Summary of what was found and why it matches the request",
  "content": "The relevant content that answers the user's question"
}

When you cannot find what was requested after multiple attempts:
{
  "action": "done",
  "description": "Explanation of what was tried and why it failed"
}
"""
    
    def __init__(
        self,
        llm_manager: Any = None,
        extraction_pipeline: Any = None,
        search_func: Optional[Callable[[str, int], Any]] = None,
        fetch_func: Optional[Callable[[str], Any]] = None,
        max_steps: int = 10
    ):
        """
        Initialize the agent.
        
        Args:
            llm_manager: LLMManager instance for AI decision-making
            extraction_pipeline: ContentExtractionPipeline instance  
            search_func: Async function for web searches (query, num_results) -> dict
            fetch_func: Async function for content fetching (url) -> dict
            max_steps: Maximum number of agent steps before giving up
        """
        self._llm_manager = llm_manager
        self._extraction_pipeline = extraction_pipeline
        self._search_func = search_func
        self._fetch_func = fetch_func
        self.max_steps = max_steps
        
        # Add current UTC-0 date/time in YYYY-MM-DD HH:MM:SS format to system prompt for better context in decision-making
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self.SYSTEM_PROMPT += f"\n\n The current date and time is: {current_time} UTC."
    
    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM with a prompt."""
        if self._llm_manager is None:
            logger.error("No LLM manager configured")
            return None
        try:
            return await self._llm_manager.complete(prompt, self.SYSTEM_PROMPT)
        except Exception as e:
            logger.error("LLM call failed: %s", str(e))
            return None
    
    async def _search(self, query: str) -> Dict[str, Any]:
        """Execute a web search."""
        if self._search_func:
            return await self._search_func(query, num_results=10)
        else:
            logger.error("No search function configured")
            return {"error": "Search not available"}
    
    async def _fetch(self, url: str) -> Dict[str, Any]:
        """Fetch content from a URL."""
        if self._fetch_func:
            return await self._fetch_func(url)
        else:
            logger.error("No fetch function configured")
            return {"error": "Fetch not available"}
    
    async def _browser_search(self, query: str) -> Dict[str, Any]:
        """
        Perform a web search using browser-use if available.
        
        Falls back to regular search if browser-use is not installed
        or fails.
        """
        task = f"Search for: {query}. Return a JSON array of search results with title, url, and snippet fields."
        
        tool = BrowserTool(
            task_description=task,
            fallback_func=lambda: self._search(query)
        )
        
        try:
            result = await tool.execute()
            
            if result.get("fallback_used"):
                # Browser failed, fallback was used - return that result
                return result.get("result", {"error": "Fallback failed"})
            
            parsed = result.get("result")
            
            # Validate we got a list (search results)
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except json.JSONDecodeError:
                    return {"error": f"Browser search failed to parse: {parsed}"}
            
            if isinstance(parsed, list):
                return parsed
            
            # If we got a dict with error info
            if isinstance(parsed, dict) and "error" in parsed:
                return {"error": f"Browser search failed: {parsed.get('error')}"}
            
            return {"error": f"Unexpected browser search result type: {type(parsed)}"}
            
        except BrowserToolError as e:
            logger.warning("Browser search failed: %s, falling back to regular search", str(e))
            return await self._search(query)
    
    async def _browser_navigate_and_extract(self, url: str) -> Dict[str, Any]:
        """
        Navigate to a URL and extract content using browser-use.
        
        Falls back to regular fetch if browser-use is not installed or fails.
        """
        task = f"""Navigate to {url} and extract the main content.

Wait for any dynamic content to load. Extract:
1. The page title
2. The main article/content body as plain text
3. Any important details relevant to finding information

Return a JSON object with:
{{"title": "...", "content": "...", "url": "{url}"}}"""

        tool = BrowserTool(
            task_description=task,
            fallback_func=lambda: self._fetch(url)
        )
        
        try:
            result = await tool.execute()
            
            if result.get("fallback_used"):
                # Browser failed, fallback was used - return that result
                return result.get("result", {"error": "Fallback failed"})
            
            parsed = result.get("result")
            
            # Validate we got a dict (page data)
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except json.JSONDecodeError:
                    return {"error": f"Browser navigate failed to parse: {parsed}"}
            
            if isinstance(parsed, dict):
                # Ensure URL is set in result
                if "url" not in parsed:
                    parsed["url"] = url
                return parsed
            
            # If we got a list (unexpected)
            if isinstance(parsed, list):
                return {"error": f"Browser navigate returned unexpected list"}
            
            return {"error": f"Unexpected browser navigate result type: {type(parsed)}"}
            
        except BrowserToolError as e:
            logger.warning("Browser navigate failed: %s, falling back to regular fetch", str(e))
            return await self._fetch(url)
    
    async def _validate_content_relevance(
        self,
        prompt: str,
        fetched_content: str
    ) -> tuple[bool, str]:
        """
        Validate that the fetched content actually answers the user's prompt.
        
        Uses an LLM to evaluate whether the content is relevant to the original request,
        returning a tuple of (is_relevant, reasoning).
        
        Args:
            prompt: The user's original request
            fetched_content: The content to validate
            
        Returns:
            Tuple of (is_relevant: bool, reasoning: str)
            
        Note:
            If validation fails for any reason (LLM unavailable, parse error, etc.),
            we default to allowing the content but log a warning.
        """
        if not fetched_content or len(fetched_content.strip()) == 0:
            return False, "Content is empty"
        
        # Truncate content if too long for validation (avoid token limits)
        max_content_len = 4000
        truncated_content = fetched_content[:max_content_len]
        if len(fetched_content) > max_content_len:
            truncated_content += "... [content truncated for validation]"
        
        # Simple prompt that doesn't use the agent's system prompt
        validation_prompt = f"""Given the user's original request: '{prompt}'

And the fetched content:
---
{truncated_content}
---

Does this content adequately answer the request? Respond with YES or NO followed by a brief explanation (1-2 sentences max).

Format your response as:
YES - [brief reason]  or  NO - [brief reason]
"""
        
        try:
            # Call LLM without the agent's system prompt (which expects JSON action output)
            if self._llm_manager is None:
                logger.warning("Content relevance validation: no LLM manager, allowing content by default")
                return True, "No LLM available for validation"
            
            response = await self._llm_manager.complete(validation_prompt, system_prompt=None)
            
            if not response:
                logger.warning("Content relevance validation: LLM call returned empty, defaulting to allow")
                return True, "Validation failed (empty response), allowing content by default"
            
            # Parse the YES/NO response
            response_clean = response.strip().upper()
            
            if response_clean.startswith("YES"):
                # Extract reasoning after "YES"
                reason = response[len("YES"):].strip()
                if reason.startswith("-"):
                    reason = reason[1:].strip()
                return True, f"Content validated as relevant: {reason}"
            elif response_clean.startswith("NO"):
                # Extract reasoning after "NO"
                reason = response[len("NO"):].strip()
                if reason.startswith("-"):
                    reason = reason[1:].strip()
                return False, f"Content validated as not relevant: {reason}"
            else:
                # Couldn't parse response, log warning and allow by default
                logger.warning("Content relevance validation: could not parse response '%s', allowing content",
                             response[:100])
                return True, f"Validation unclear ('{response[:50]}...'), allowing content by default"
                
        except Exception as e:
            logger.warning("Content relevance validation failed: %s, defaulting to allow content", str(e))
            return True, f"Validation error ('{str(e)[:50]}...'), allowing content by default"
    
    def _parse_llm_action(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse the LLM's JSON action response into an LLMAction.
        
        Uses a multi-stage parsing approach:
        1. First attempts proper JSON parsing with Pydantic validation
        2. Falls back to keyword extraction only if both:
           - JSON parsing fails AND
           - Keywords indicating intent are found
        3. Raises ActionParsingError only if all parsing attempts fail
        
        Args:
            response: Raw string response from the LLM
            
        Returns:
            Dictionary representation of LLMAction for backward compatibility,
            or None only when keyword fallback finds no matches (to continue execution)
            
        Raises:
            ActionParsingError: When neither JSON parsing nor keyword fallback succeeds
        """
        if not response or not response.strip():
            raise ActionParsingError(
                "Empty LLM response",
                raw_response=response
            )
        
        stripped = response.strip()
        
        # Stage 1: Try proper JSON parsing with Pydantic
        try:
            json_start = stripped.find("{")
            json_end = stripped.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = stripped[json_start:json_end]
                
            # Try to parse directly if no braces found
            elif stripped.startswith(('{', '[')):
                json_str = stripped.rstrip(',').rstrip('}')
            else:
                # Try finding JSON within text
                json_start = stripped.find('{')
                if json_start == -1:
                    raise ValueError("No JSON object found in response")
                json_end = stripped.rfind('}')
                if json_end <= json_start:
                    raise ValueError("Invalid JSON structure")
                json_str = stripped[json_start:json_end + 1]
            
            # Use Pydantic for validation, then convert to dict
            action = LLMAction.model_validate_json(json_str)
            return action.to_dict()
            
        except Exception as json_error:
            logger.debug("JSON parsing failed: %s", str(json_error))
            
        # Stage 2: Keyword-based fallback (only if JSON failed)
        response_lower = stripped.lower()
        
        # Define keywords that indicate specific actions
        if "done" in response_lower:
            return LLMAction(
                action=ActionType.DONE.value,
                result="",
                reasoning=f"Keyword fallback: 'done' found in response"
            ).to_dict()
        elif any(term in response_lower for term in ["search", "look up", "find"]):
            return LLMAction(
                action=ActionType.SEARCH.value,
                query="",
                reasoning=f"Keyword fallback: search-related term found in response"
            ).to_dict()
        elif any(term in response_lower for term in ["fetch", "visit"]):
            return LLMAction(
                action=ActionType.FETCH.value,
                reasoning=f"Keyword fallback: fetch-related term found in response"
            ).to_dict()
        elif "navigate" in response_lower:
            return LLMAction(
                action=ActionType.NAVIGATE.value,
                reasoning=f"Keyword fallback: 'navigate' found in response"
            ).to_dict()
        elif "evaluate" in response_lower:
            return LLMAction(
                action=ActionType.EVALUATE.value,
                reasoning=f"Keyword fallback: 'evaluate' found in response"
            ).to_dict()
        
        # Stage 3: All parsing attempts failed
        raise ActionParsingError(
            f"Could not parse LLM response into any known action type. "
            f"No JSON structure found and no keyword matches.",
            raw_response=response[:500],  # Truncate for logging
            cause=None
        )
    
    def _action_to_enum(self, action_str: str) -> ActionType:
        """
        Convert a string action to an ActionType enum value.
        
        Provides consistent handling of action strings throughout the codebase
        and normalizes variations (e.g., "nav" -> "navigate").
        
        Args:
            action_str: The action string to convert
            
        Returns:
            ActionType enum value
        """
        normalized = action_str.lower().strip()
        
        # Handle common variations and abbreviations
        if normalized in ("nav", "navigat"):
            return ActionType.NAVIGATE
        elif normalized == ActionType.DONE.value:
            return ActionType.DONE
        elif normalized in (ActionType.SEARCH.value, "lookup"):
            return ActionType.SEARCH
        elif normalized == ActionType.FETCH.value:
            return ActionType.FETCH
        elif normalized in (ActionType.EVALUATE.value, "eval"):
            return ActionType.EVALUATE
        else:
            # Return NAVIGATE as default fallback for unknown actions (backward compat)
            return ActionType.NAVIGATE
    
    async def execute(self, prompt: str) -> AgenticFetchResult:
        """
        Execute the agent to find information based on a natural language prompt.
        
        Args:
            prompt: The user's request in natural language
            
        Returns:
            AgenticFetchResult with success status, content if found,
            URLs visited, and detailed steps taken
        """
        result = AgenticFetchResult(success=False)
        
        logger.info("Starting agentic fetch for prompt: %s", prompt[:100])
        
        current_context = f"The user wants to find the following information: {prompt}\n\n"
        
        for step_num in range(1, self.max_steps + 1):
            logger.info("Agent step %d/%d", step_num, self.max_steps)
            
            # Create a detailed prompt for the LLM
            agent_prompt = f"""{current_context}

Please decide on your next action. Output a JSON object with:
- "action": one of: search, fetch, navigate, evaluate, done
- "description": explanation of what you're doing and why  
- For search: include a "query" field with the search query
- For fetch/navigate: include a "url" field with the URL to visit
- When done and successful: include a "content" field with findings

Think about:
1. What search terms would best find this information?
2. Are there any URLs already discovered that might be promising?
3. Have we gathered enough information to answer the user's question?

Be strategic and try different approaches if initial searches don't work."""
            
            # Call LLM to decide next action
            llm_response = await self._call_llm(agent_prompt)
            
            if not llm_response:
                result.steps_taken.append({
                    "step": step_num,
                    "action": "error",
                    "description": f"LLM call failed at step {step_num}"
                })
                result.error_message = "Failed to get LLM response"
                break
            
            # Parse the action
            action_data = self._parse_llm_action(llm_response)
            
            if not action_data:
                result.steps_taken.append({
                    "step": step_num,
                    "action": "error",
                    "description": f"Could not parse LLM response at step {step_num}",
                    "result": llm_response[:500]
                })
                current_context += f"\nStep {step_num}: Could not determine action.\n"
                continue
            
            action = action_data.get("action", "").lower()
            
            logger.info("Step %d: Taking action '%s' - %s", step_num, action, 
                       action_data.get("description", "")[:100])
            
            # Execute the action
            step_result = {
                "step": step_num,
                "action": action,
                "description": action_data.get("description", ""),
            }
            
            try:
                if action == "done":
                    # Check if we have content
                    content = action_data.get("content")
                    
                    step_result["description"] += f"\n\nLLM concluded: {action_data.get('description', '')}"
                    
                    if content:
                        # Validate that the content actually answers the prompt
                        is_relevant, relevance_reasoning = await self._validate_content_relevance(
                            prompt=prompt,
                            fetched_content=content
                        )
                        
                        # Add relevance check to step result for transparency
                        step_result["relevance_check"] = {
                            "is_relevant": is_relevant,
                            "reasoning": relevance_reasoning
                        }
                        
                        if is_relevant:
                            result.success = True
                            result.content = content
                            step_result["result"] = f"Successfully found relevant content ({len(content)} chars)"
                            logger.info("Content relevance validated: %s", relevance_reasoning)
                        else:
                            # Content is not relevant - still set success=True but log warning
                            result.success = True  # Still return content since agent worked hard
                            result.content = content
                            step_result["result"] = f"Content may not fully answer prompt ({len(content)} chars) - {relevance_reasoning}"
                            logger.warning("Content relevance check failed: %s | Prompt: %s",
                                         relevance_reasoning, prompt[:100])
                    else:
                        step_result["result"] = "Agent concluded without finding content"
                    
                    result.steps_taken.append(step_result)
                    break
                    
                elif action == "search":
                    query = action_data.get("query", "")
                    
                    if not query:
                        current_context += f"\nStep {step_num}: Search action but no query provided.\n"
                        step_result["result"] = "No search query provided"
                        result.steps_taken.append(step_result)
                        continue
                    
                    step_result["url"] = f"Search: {query}"
                    
                    # Use browser search if available, fallback to regular
                    task = f"""Search for: {query}

Use a web search to find relevant pages about this topic.

Return a JSON array of results with:
[{{"title": "...", "url": "...", "snippet": "..."}}]

Only return valid JSON array, no other text."""

                    tool = BrowserTool(
                        task_description=task,
                        fallback_func=lambda: self._search(query)
                    )
                    
                    try:
                        browser_result = await tool.execute()
                        
                        if browser_result.get("fallback_used"):
                            # Browser failed, fallback was used
                            search_results = browser_result.get("result", {})
                            if "error" in search_results:
                                step_result["result"] = f"Search error: {search_results['error']}"
                            else:
                                step_result["result"] = f"Found {len(search_results.get('results', []))} results via API"
                                search_results = search_results.get("results", [])
                        else:
                            # Browser succeeded
                            parsed = browser_result.get("result")
                            if isinstance(parsed, str):
                                try:
                                    search_results = json.loads(parsed)
                                except json.JSONDecodeError:
                                    logger.warning("Could not parse browser search result as JSON")
                                    raise BrowserToolError("JSON parse failed")
                            else:
                                search_results = parsed
                            
                            if isinstance(search_results, list):
                                step_result["result"] = f"Found {len(search_results)} results via browser"
                            else:
                                step_result["result"] = f"Found 0 results (unexpected format)"
                                search_results = []
                                
                    except BrowserToolError:
                        # Fall back to regular search if browser failed
                        logger.info("Using regular search after BrowserTool failure")
                        search_results = await self._search(query)
                        
                        if "error" in search_results:
                            step_result["result"] = f"Search error: {search_results['error']}"
                        else:
                            step_result["result"] = f"Found {len(search_results.get('results', []))} results via API"
                            search_results = search_results.get("results", [])
                    
                    # Track URLs visited
                    if isinstance(search_results, list):
                        for r in search_results:
                            url = r.get("url", "")
                            if url and not any(u.get("url") == url for u in result.urls_visited):
                                result.urls_visited.append({
                                    "url": url,
                                    "title": r.get("title", "")[:100],
                                    "action": f"Search result at step {step_num}"
                                })
                    
                    current_context += f"\nStep {step_num} ({action}): Searched for '{query}'.\n"
                    
                    if isinstance(search_results, list) and len(search_results) > 0:
                        current_context += f"Top results:\n"
                        for r in search_results[:3]:
                            current_context += f"- {r.get('title', '')[:80]}: {r.get('url', '')}\n"
                    
                elif action in ("fetch", "navigate"):
                    url = action_data.get("url", "")
                    
                    if not url:
                        current_context += f"\nStep {step_num}: Fetch/navigate action but no URL provided.\n"
                        step_result["result"] = "No URL provided"
                        result.steps_taken.append(step_result)
                        continue
                    
                    step_result["url"] = url
                    
                    # Use browser navigate if available, fallback to regular fetch
                    task = f"""Navigate to {url} and extract the main content.

Wait for dynamic content to fully load. Extract:
1. The page title
2. Main article/body text as plain markdown
3. Any important details relevant to the user's query: {prompt}

Return a JSON object:
{{"title": "...", "content": "...", "url": "{url}"}}

Only return valid JSON."""

                    tool = BrowserTool(
                        task_description=task,
                        fallback_func=lambda: self._fetch(url)
                    )
                    
                    try:
                        browser_result = await tool.execute()
                        
                        if browser_result.get("fallback_used"):
                            # Browser failed, fallback was used
                            fetch_data = browser_result.get("result", {})
                            
                            if "error" in fetch_data:
                                step_result["result"] = f"Fetch error: {fetch_data['error']}"
                            else:
                                content = fetch_data.get("content", "")
                                step_result["result"] = f"Extracted {len(content) if content else 0} chars via HTTP"
                                
                                result.urls_visited.append({
                                    "url": url,
                                    "title": fetch_data.get("title", "")[:100] if isinstance(fetch_data, dict) else "",
                                    "action": f"Fetched at step {step_num}"
                                })
                                
                                current_context += f"\nStep {step_num} ({action}): Fetched content from '{url}'.\n"
                                current_context += f"Extracted {len(content) if content else 0} characters.\n"
                                
                                if content:
                                    current_context += f"Content preview: {content[:500]}...\n"
                        else:
                            # Browser succeeded
                            parsed = browser_result.get("result")
                            if isinstance(parsed, str):
                                try:
                                    fetch_data = json.loads(parsed)
                                except json.JSONDecodeError:
                                    logger.warning("Could not parse browser fetch result as JSON")
                                    raise BrowserToolError("JSON parse failed")
                            else:
                                fetch_data = parsed
                            
                            content = fetch_data.get("content", "") if isinstance(fetch_data, dict) else ""
                            step_result["result"] = f"Extracted {len(content) if content else 0} chars via browser"
                            
                            # Add to URLs visited
                            result.urls_visited.append({
                                "url": url,
                                "title": fetch_data.get("title", "")[:100] if isinstance(fetch_data, dict) else "",
                                "action": f"Navigated at step {step_num}"
                            })
                            
                            current_context += f"\nStep {step_num} ({action}): Fetched content from '{url}'.\n"
                            current_context += f"Extracted {len(content) if content else 0} characters.\n"
                            
                            # Check for relevant content
                            if content:
                                current_context += f"Content preview: {content[:500]}...\n"
                                
                    except BrowserToolError:
                        # Fall back to regular fetch if browser failed
                        logger.info("Using regular fetch after BrowserTool failure")
                        
                        fetch_result = await self._fetch(url)
                        
                        if "error" in fetch_result:
                            step_result["result"] = f"Fetch error: {fetch_result['error']}"
                        else:
                            content = fetch_result.get("content", "")
                            step_result["result"] = f"Extracted {len(content) if content else 0} chars via HTTP"
                            
                            result.urls_visited.append({
                                "url": url,
                                "title": fetch_result.get("title", "")[:100] if isinstance(fetch_result, dict) else "",
                                "action": f"Fetched at step {step_num}"
                            })
                            
                            current_context += f"\nStep {step_num} ({action}): Fetched content from '{url}'.\n"
                            current_context += f"Extracted {len(content) if content else 0} characters.\n"
                            
                            if content:
                                current_context += f"Content preview: {content[:500]}...\n"
                
                elif action == "evaluate":
                    current_context += f"\nStep {step_num} (evaluate): LLM evaluated progress.\n"
                    current_context += f"Evaluation: {action_data.get('description', '')}\n"
                    
                else:
                    current_context += f"\nStep {step_num}: Unknown action '{action}'.\n"
                    
            except Exception as e:
                logger.error("Error executing step %d: %s", step_num, str(e))
                current_context += f"\nStep {step_num}: Error executing action: {str(e)}\n"
                step_result["result"] = f"Error: {str(e)}"
            
            result.steps_taken.append(step_result)
        
        # Check if we ran out of steps
        if not result.success:
            logger.info("Agent failed to find content after %d steps", self.max_steps)
            
            # Check if there's any partial content we can return
            last_step = result.steps_taken[-1] if result.steps_taken else None
            last_action = last_step.get("action", "") if last_step else ""
            
            result.error_message = f"Could not find requested content after {self.max_steps} steps. See urls_visited for URLs that were attempted."
            
            # If we have a final content from LLM conclusion, use it
            if last_action == "done":
                result.success = False  # Still marked as not fully successful
        else:
            logger.info("Agent successfully found content")
        
        return result


async def agentic_fetch(
    prompt: str,
    max_steps: int = 10,
    llm_manager: Any = None,
    extraction_pipeline: Any = None,
    search_func: Optional[Callable[[str, int], Any]] = None,
    fetch_func: Optional[Callable[[str], Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to perform agentic fetch.
    
    Args:
        prompt: Natural language request
        max_steps: Maximum agent steps (default 10)
        llm_manager: LLMManager instance
        extraction_pipeline: ContentExtractionPipeline instance
        search_func: Async function for web searches (query, num_results) -> dict
        fetch_func: Async function for content fetching (url) -> dict
        
    Returns:
        Dict with agentic fetch result
    """
    # Import here to avoid circular imports at module level
    from ..llm import LLMManager as LLMImport
    
    if llm_manager is None:
        try:
            llm_manager = LLMImport()
        except Exception as e:
            logger.error("Could not create LLM manager: %s", str(e))
    
    # Create default search/fetch functions if not provided
    if search_func is None:
        async def _default_search(query, num_results=10):
            try:
                from ..server import search_web
                return await search_web(query, num_results=num_results)
            except Exception as e:
                logger.error("Default search failed: %s", str(e))
                return {"error": f"Search not available: {str(e)}"}
        
        search_func = _default_search
    
    if fetch_func is None:
        async def _default_fetch(url):
            try:
                from ..server import fetch_web_content
                return await fetch_web_content(url, include_links=True)
            except Exception as e:
                logger.error("Default fetch failed: %s", str(e))
                return {"error": f"Fetch not available: {str(e)}"}
        
        fetch_func = _default_fetch
    
    agent = AgenticFetchAgent(
        llm_manager=llm_manager,
        extraction_pipeline=extraction_pipeline,
        search_func=search_func,
        fetch_func=fetch_func,
        max_steps=max_steps
    )
    
    result = await agent.execute(prompt)
    return result.to_dict()