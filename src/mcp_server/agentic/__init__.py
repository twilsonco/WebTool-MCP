"""
Agentic AI fetch mode using browser-use for autonomous web browsing.

This module provides an agent that can:
- Take a natural language prompt
- Plan and execute web searches/fetches using AI decision-making
- Navigate, click, scroll as needed using browser automation
- Return found content or a detailed report of URLs visited if not found
"""

from .fetch_agent import (
    ActionParsingError,
    AgenticFetchAgent,
    AgenticFetchResult,
    agentic_fetch,
)
from .fetch_agent import ActionType, FetchStep, LLMAction
from .fetch_agent import BrowserTool, BrowserToolError

__all__ = [
    "ActionParsingError",
    "AgenticFetchAgent",
    "AgenticFetchResult",
    "agentic_fetch",
    "ActionType",
    "BrowserTool",
    "BrowserToolError",
    "FetchStep",
    "LLMAction"
]