"""
Agentic AI fetch mode using browser-use for autonomous web browsing.

This module provides an agent that can:
- Take a natural language prompt
- Plan and execute web searches/fetches using AI decision-making
- Navigate, click, scroll as needed using browser automation
- Return found content or a detailed report of URLs visited if not found
"""

from .fetch_agent import AgenticFetchAgent, AgenticFetchResult, agentic_fetch

__all__ = ["AgenticFetchAgent", "AgenticFetchResult", "agentic_fetch"]