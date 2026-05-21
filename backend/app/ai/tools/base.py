"""
Tool base classes and registry for JARVIS agent tool calling.

Provides:
- ToolDefinition: metadata + OpenAI format serialiser
- BaseTool: abstract base for all tools
- ToolRegistry: register, look up, and execute tools
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------


class ToolDefinition(BaseModel):
    """
    Serialisable definition of a tool for the OpenAI function calling API.

    ``parameters`` must be a valid JSON Schema object describing the function
    arguments.
    """

    name: str = Field(..., description="Unique tool name (snake_case)")
    description: str = Field(..., description="Human-readable description for the model")
    parameters: dict[str, Any] = Field(
        ...,
        description="JSON Schema for the tool's input parameters",
    )

    def to_openai_format(self) -> dict[str, Any]:
        """
        Render the definition in the format expected by the OpenAI tools array.

        Returns:
            ::

                {
                    "type": "function",
                    "function": {
                        "name": ...,
                        "description": ...,
                        "parameters": {...}
                    }
                }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# BaseTool
# ---------------------------------------------------------------------------


class BaseTool(ABC):
    """
    Abstract base class that every JARVIS tool must inherit from.

    Subclasses must:
    1. Set class-level ``name`` and ``description`` attributes.
    2. Define ``parameters_schema`` as a JSON Schema dict describing inputs.
    3. Implement ``async execute(params)`` which returns a result dict.
    """

    name: str = ""
    description: str = ""
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    @property
    def definition(self) -> ToolDefinition:
        """Return the ToolDefinition for this tool."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the tool with the given parameters.

        Args:
            params: Dict matching the tool's ``parameters_schema``.

        Returns:
            A dict containing at minimum a ``result`` key.  On error, include
            ``error`` key with a human-readable message.
        """

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r}>"


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """
    Central registry for all JARVIS tools.

    Usage::

        registry = ToolRegistry()
        registry.register(CreateTaskTool())
        registry.register(GetCurrentTimeTool())

        # Get OpenAI-formatted tools list
        tools = registry.to_openai_tools()

        # Execute a tool by name
        result = await registry.execute("create_task", {"title": "Buy milk"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool instance.

        Args:
            tool: A concrete BaseTool subclass instance.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if not tool.name:
            raise ValueError(f"Tool {type(tool).__name__} has no name set.")
        if tool.name in self._tools:
            raise ValueError(
                f"A tool named '{tool.name}' is already registered. "
                "Unregister it first or use a unique name."
            )
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool | None:
        """
        Retrieve a registered tool by name.

        Returns:
            The BaseTool instance, or None if not found.
        """
        return self._tools.get(name)

    def list_all(self) -> list[ToolDefinition]:
        """
        Return a list of ToolDefinition objects for all registered tools.
        """
        return [tool.definition for tool in self._tools.values()]

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """
        Return all registered tools formatted for the OpenAI ``tools`` parameter.

        Returns:
            List of dicts in OpenAI ``{"type": "function", "function": {...}}`` format.
        """
        return [tool.definition.to_openai_format() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a registered tool by name.

        Args:
            name: Tool name as registered.
            params: Parameter dict to pass to ``tool.execute()``.

        Returns:
            The tool's result dict, or an error dict if the tool is not found
            or execution raises an exception.
        """
        tool = self._tools.get(name)
        if tool is None:
            logger.error("ToolRegistry.execute: unknown tool '%s'", name)
            return {
                "error": f"Tool '{name}' is not registered.",
                "tool_name": name,
            }

        logger.info("Executing tool: %s | params=%s", name, params)
        try:
            result = await tool.execute(params)
            logger.debug("Tool %s returned: %s", name, result)
            return result
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception: %s", name, exc)
            return {
                "error": f"Tool execution failed: {exc}",
                "tool_name": name,
            }

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        names = list(self._tools.keys())
        return f"<ToolRegistry tools={names}>"
