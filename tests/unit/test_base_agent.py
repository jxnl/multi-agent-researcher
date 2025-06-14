"""Tests for base agent functionality."""

import pytest
from datetime import datetime
from src.researcher.agents.base import (
    Agent, AgentContext, AgentResult, AgentState, ToolCall
)


class MockAgent(Agent[str]):
    """Mock agent implementation for testing."""
    
    def __init__(self, should_fail=False, **kwargs):
        super().__init__(**kwargs)
        self.should_fail = should_fail
        self.plan_called = False
        self.execute_called = False
        
    async def plan(self, context: AgentContext) -> list:
        self.plan_called = True
        self.add_thinking("Analyzing the query")
        if self.should_fail:
            raise ValueError("Planning failed")
        return [{"action": "search", "query": context.query}]
        
    async def execute(self, plan: list) -> str:
        self.execute_called = True
        self.add_thinking("Executing search")
        
        # Simulate tool call
        tool_call = ToolCall(
            tool_name="search",
            parameters={"query": plan[0]["query"]},
            result="Search results"
        )
        self.record_tool_call(tool_call)
        
        if self.should_fail:
            raise RuntimeError("Execution failed")
        return "Research complete"


@pytest.mark.asyncio
async def test_agent_successful_execution():
    """Test successful agent execution flow."""
    agent = MockAgent(name="TestAgent")
    context = AgentContext(
        query="What are AI agents?",
        objective="Research AI agents"
    )
    
    result = await agent.run(context)
    
    assert agent.plan_called
    assert agent.execute_called
    assert result.status == AgentState.COMPLETED
    assert result.output == "Research complete"
    assert len(result.thinking) == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "search"
    assert result.end_time is not None
    assert result.error is None


@pytest.mark.asyncio
async def test_agent_planning_failure():
    """Test agent behavior when planning fails."""
    agent = MockAgent(should_fail=True, name="FailingAgent")
    context = AgentContext(
        query="Test query",
        objective="Test objective"
    )
    
    with pytest.raises(ValueError, match="Planning failed"):
        await agent.run(context)
        
    assert agent.plan_called
    assert not agent.execute_called
    assert agent.state == AgentState.FAILED
    assert agent._result.status == AgentState.FAILED
    assert "Planning failed" in agent._result.error


def test_agent_context_creation():
    """Test AgentContext model creation and validation."""
    context = AgentContext(
        query="Research query",
        objective="Find information",
        constraints=["Limit to 5 sources", "Use only recent data"],
        metadata={"priority": "high"},
        parent_agent_id="parent-123"
    )
    
    assert context.query == "Research query"
    assert context.objective == "Find information"
    assert len(context.constraints) == 2
    assert context.metadata["priority"] == "high"
    assert context.parent_agent_id == "parent-123"


def test_tool_call_model():
    """Test ToolCall model functionality."""
    tool_call = ToolCall(
        tool_name="web_search",
        parameters={"query": "AI research", "limit": 10}
    )
    
    assert tool_call.tool_name == "web_search"
    assert tool_call.parameters["query"] == "AI research"
    assert tool_call.result is None
    assert tool_call.error is None
    assert isinstance(tool_call.timestamp, datetime)


def test_agent_result_model():
    """Test AgentResult model."""
    result = AgentResult(
        agent_id="agent-123",
        status=AgentState.COMPLETED,
        output="Research findings",
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        tokens_used=1500
    )
    
    assert result.agent_id == "agent-123"
    assert result.status == AgentState.COMPLETED
    assert result.output == "Research findings"
    assert result.tokens_used == 1500
    assert len(result.tool_calls) == 0
    assert len(result.thinking) == 0