"""Tests for tool system."""

import pytest
import asyncio
from src.researcher.tools.base import (
    Tool, ToolDescription, ToolResult, ToolStatus,
    ToolRegistry, ParallelToolExecutor
)
from src.researcher.tools.mock_search import MockSearchTool


class SlowTool(Tool):
    """Tool that simulates slow execution."""
    
    def __init__(self, delay: float = 1.0):
        super().__init__("slow_tool", "A slow tool for testing")
        self.delay = delay
        
    def get_description(self) -> ToolDescription:
        return ToolDescription(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}},
            timeout=0.5  # Timeout shorter than delay
        )
        
    async def _execute(self) -> str:
        await asyncio.sleep(self.delay)
        return "Slow operation complete"


class ErrorTool(Tool):
    """Tool that always fails."""
    
    def __init__(self, error_message: str = "Tool failed"):
        super().__init__("error_tool", "A tool that fails")
        self.error_message = error_message
        
    def get_description(self) -> ToolDescription:
        return ToolDescription(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}}
        )
        
    async def _execute(self) -> None:
        raise RuntimeError(self.error_message)


@pytest.mark.asyncio
async def test_mock_search_tool():
    """Test mock search tool functionality."""
    tool = MockSearchTool()
    result = await tool.execute(query="AI research", limit=5)
    
    assert result.status == ToolStatus.SUCCESS
    assert isinstance(result.data, list)
    assert len(result.data) == 3  # Mock returns max 3 results
    assert result.data[0]["title"] == "Result 1 for: AI research"
    assert tool.search_history == ["AI research"]


@pytest.mark.asyncio
async def test_tool_with_predefined_results():
    """Test mock search with predefined results."""
    predefined = [
        {"title": "Custom Result 1", "url": "https://custom1.com"},
        {"title": "Custom Result 2", "url": "https://custom2.com"}
    ]
    
    tool = MockSearchTool(results=predefined)
    result = await tool.execute(query="test", limit=10)
    
    assert result.status == ToolStatus.SUCCESS
    assert result.data == predefined


@pytest.mark.asyncio
async def test_tool_timeout():
    """Test tool timeout handling."""
    tool = SlowTool(delay=2.0)
    result = await tool.execute()
    
    assert result.status == ToolStatus.TIMEOUT
    assert "timed out" in result.error
    assert result.execution_time < 1.0  # Should timeout at 0.5s


@pytest.mark.asyncio
async def test_tool_error_handling():
    """Test tool error handling."""
    tool = ErrorTool("Custom error")
    result = await tool.execute()
    
    assert result.status == ToolStatus.FAILURE
    assert result.error == "Custom error"


@pytest.mark.asyncio
async def test_rate_limit_detection():
    """Test rate limit error detection."""
    tool = ErrorTool("Rate limit exceeded")
    result = await tool.execute()
    
    assert result.status == ToolStatus.RATE_LIMITED
    assert result.retry_after == 60.0


def test_tool_metrics():
    """Test tool metrics tracking."""
    tool = MockSearchTool()
    
    # Initial metrics
    metrics = tool.get_metrics()
    assert metrics["call_count"] == 0
    assert metrics["total_time"] == 0
    assert metrics["average_time"] == 0


def test_tool_registry():
    """Test tool registry functionality."""
    registry = ToolRegistry()
    
    # Register tools
    search_tool = MockSearchTool()
    registry.register(search_tool, category="search")
    
    error_tool = ErrorTool()
    registry.register(error_tool, category="test")
    
    # Test retrieval
    assert registry.get_tool("web_search") == search_tool
    assert registry.get_tool("error_tool") == error_tool
    assert registry.get_tool("nonexistent") is None
    
    # Test listing
    all_tools = registry.list_tools()
    assert len(all_tools) == 2
    
    search_tools = registry.list_tools(category="search")
    assert len(search_tools) == 1
    assert search_tools[0].name == "web_search"


def test_tool_search():
    """Test tool search functionality."""
    registry = ToolRegistry()
    
    tool1 = MockSearchTool()
    registry.register(tool1)
    
    # Search by name
    results = registry.search_tools("web")
    assert len(results) == 1
    assert results[0].name == "web_search"
    
    # Search by description
    results = registry.search_tools("information")
    assert len(results) == 1
    
    # No matches
    results = registry.search_tools("xyz")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_parallel_tool_execution():
    """Test parallel tool execution."""
    registry = ToolRegistry()
    registry.register(MockSearchTool())
    registry.register(ErrorTool())
    
    executor = ParallelToolExecutor(registry)
    
    tool_calls = [
        {"tool": "web_search", "parameters": {"query": "test1"}},
        {"tool": "web_search", "parameters": {"query": "test2"}},
        {"tool": "error_tool", "parameters": {}},
        {"tool": "nonexistent", "parameters": {}}
    ]
    
    results = await executor.execute_parallel(tool_calls)
    
    assert len(results) == 4
    assert results[0].status == ToolStatus.SUCCESS
    assert results[1].status == ToolStatus.SUCCESS
    assert results[2].status == ToolStatus.FAILURE
    assert results[3].status == ToolStatus.FAILURE
    assert "not found" in results[3].error