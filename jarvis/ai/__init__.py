"""
jarvis.ai — Public API for the JARVIS AI layer.

Exposes the key classes and singleton accessors so the rest of the
application can do::

    from jarvis.ai import get_agent, get_ai_client, AgentResponse
"""

from ai.agent import AgentResponse, ConversationMemory, JarvisAgent, get_agent
from ai.client import AIClient, get_ai_client
from ai.command_parser import CommandParser, ParsedResponse, KNOWN_ACTIONS
from ai.prompts import (
    JARVIS_SYSTEM_PROMPT,
    TASK_SYSTEM_PROMPT,
    TELEGRAM_REPLY_PROMPT,
    DAILY_BRIEFING_PROMPT,
    build_system_prompt,
    format_conversation_history,
)

__all__ = [
    # Agent
    "AgentResponse",
    "ConversationMemory",
    "JarvisAgent",
    "get_agent",
    # Client
    "AIClient",
    "get_ai_client",
    # Parser
    "CommandParser",
    "ParsedResponse",
    "KNOWN_ACTIONS",
    # Prompts
    "JARVIS_SYSTEM_PROMPT",
    "TASK_SYSTEM_PROMPT",
    "TELEGRAM_REPLY_PROMPT",
    "DAILY_BRIEFING_PROMPT",
    "build_system_prompt",
    "format_conversation_history",
]
