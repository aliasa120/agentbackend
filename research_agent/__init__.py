"""Deep Research Agent Example.

This module demonstrates building a research agent using the deepagents package
with custom tools for web search and strategic thinking.
"""

from research_agent.prompts import (
    RESEARCHER_INSTRUCTIONS,
    NEWS_TO_SOCIAL_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from research_agent.tools import linkup_search, think_tool

__all__ = [
    "linkup_search",
    "think_tool",
    "RESEARCHER_INSTRUCTIONS",
    "NEWS_TO_SOCIAL_WORKFLOW_INSTRUCTIONS",
    "SUBAGENT_DELEGATION_INSTRUCTIONS",
]
