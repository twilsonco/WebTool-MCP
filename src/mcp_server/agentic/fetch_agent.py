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
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


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
            return await self._search_func(query)
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
        try:
            from browser_use import Agent as BrowserAgent
            
            # Create a simple agent to execute the search
            task = f"Search for: {query}. Return a JSON array of search results with title, url, and snippet fields."
            
            agent = BrowserAgent(task=task)
            result = await agent.run()
            
            # Parse the result
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    pass
            
            return {"error": f"Browser search failed: {result}"}
            
        except ImportError:
            logger.info("browser-use not installed, falling back to regular search")
            return await self._search(query)
        except Exception as e:
            logger.warning("Browser search failed: %s, falling back to regular search", str(e))
            return await self._search(query)
    
    async def _browser_navigate_and_extract(self, url: str) -> Dict[str, Any]:
        """
        Navigate to a URL and extract content using browser-use.
        
        Falls back to regular fetch if browser-use is not installed or fails.
        """
        try:
            from browser_use import Agent as BrowserAgent
            
            task = f"""Navigate to {url} and extract the main content.
            
Wait for any dynamic content to load. Extract:
1. The page title
2. The main article/content body as plain text
3. Any important details relevant to finding information

Return a JSON object with:
{{"title": "...", "content": "...", "url": "{url}"}}"""

            agent = BrowserAgent(task=task)
            result = await agent.run()
            
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    pass
            
            return {"error": f"Browser navigate failed: {result}"}
            
        except ImportError:
            logger.info("browser-use not installed, falling back to regular fetch")
            return await self._fetch(url)
        except Exception as e:
            logger.warning("Browser navigate failed: %s, falling back to regular fetch", str(e))
            return await self._fetch(url)
    
    def _parse_llm_action(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM's JSON action response."""
        try:
            # Try to find a JSON block in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            
            # Try parsing the whole response as JSON
            return json.loads(response.strip())
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse LLM action response: %s", str(e))
            
            # Try a simple fallback - look for keywords
            response_lower = response.lower()
            if "done" in response_lower:
                return {"action": "done", "description": response}
            elif any(term in response_lower for term in ["search", "look up", "find"]):
                return {"action": "search", "description": response}
            
            return None
    
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
                        result.success = True
                        result.content = content
                        step_result["result"] = f"Successfully found content ({len(content)} chars)"
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
                    try:
                        from browser_use import Agent as BrowserAgent
                        
                        task = f"""Search for: {query}

Use a web search to find relevant pages about this topic.

Return a JSON array of results with:
[{{"title": "...", "url": "...", "snippet": "..."}}]

Only return valid JSON array, no other text."""

                        agent = BrowserAgent(task=task)
                        search_result_raw = await agent.run()
                        
                        # Parse result
                        try:
                            if isinstance(search_result_raw, str):
                                search_results = json.loads(search_result_raw)
                            else:
                                search_results = search_result_raw
                        except (json.JSONDecodeError, TypeError):
                            logger.warning("Could not parse browser search result as JSON")
                            # Fall back to regular search
                            raise ImportError("Parse failed")
                        
                        step_result["result"] = f"Found {len(search_results) if isinstance(search_results, list) else 0} results via browser"
                        
                    except (ImportError, Exception) as e:
                        logger.info("Using regular search: %s", str(e))
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
                    try:
                        from browser_use import Agent as BrowserAgent
                        
                        task = f"""Navigate to {url} and extract the main content.

Wait for dynamic content to fully load. Extract:
1. The page title
2. Main article/body text as plain markdown
3. Any important details relevant to the user's query: {prompt}

Return a JSON object:
{{"title": "...", "content": "...", "url": "{url}"}}

Only return valid JSON."""

                        agent = BrowserAgent(task=task)
                        fetch_result_raw = await agent.run()
                        
                        # Parse result
                        try:
                            if isinstance(fetch_result_raw, str):
                                fetch_data = json.loads(fetch_result_raw)
                            else:
                                fetch_data = fetch_result_raw
                        except (json.JSONDecodeError, TypeError):
                            logger.warning("Could not parse browser fetch result as JSON")
                            raise ImportError("Parse failed")
                        
                        content = fetch_data.get("content", "")
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
                        
                    except (ImportError, Exception) as e:
                        logger.info("Using regular fetch: %s", str(e))
                        
                        # Regular HTTP fetch
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