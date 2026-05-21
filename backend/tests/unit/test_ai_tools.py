"""Unit tests for AI tool registry."""
import pytest
from unittest.mock import AsyncMock


@pytest.mark.unit
class TestToolRegistry:
    def test_register_and_retrieve_tool(self):
        from app.ai.tools.base import ToolRegistry, BaseTool, ToolDefinition

        class MockTool(BaseTool):
            name = "mock_tool"
            description = "A mock tool for testing"

            def _get_parameters(self):
                return {"type": "object", "properties": {}, "required": []}

            async def execute(self, params: dict) -> dict:
                return {"result": "ok"}

        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)

        retrieved = registry.get("mock_tool")
        assert retrieved is not None
        assert retrieved.name == "mock_tool"

    def test_list_all_tools(self):
        from app.ai.tools.base import ToolRegistry, BaseTool

        class ToolA(BaseTool):
            name = "tool_a"
            description = "Tool A"

            def _get_parameters(self):
                return {"type": "object", "properties": {}, "required": []}

            async def execute(self, params: dict) -> dict:
                return {}

        registry = ToolRegistry()
        registry.register(ToolA())
        tools = registry.list_all()
        assert len(tools) >= 1
        assert any(t.name == "tool_a" for t in tools)

    def test_to_openai_format(self):
        from app.ai.tools.base import ToolRegistry, BaseTool

        class SampleTool(BaseTool):
            name = "sample"
            description = "Sample tool"

            def _get_parameters(self):
                return {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"]
                }

            async def execute(self, params: dict) -> dict:
                return {"output": params.get("input", "")}

        registry = ToolRegistry()
        registry.register(SampleTool())
        openai_tools = registry.to_openai_tools()

        assert len(openai_tools) >= 1
        tool_def = openai_tools[0]
        assert tool_def["type"] == "function"
        assert "function" in tool_def
        assert tool_def["function"]["name"] == "sample"
