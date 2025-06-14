"""Tests for LeadResearcherV2 with Instructor integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os

from src.researcher.agents.lead_v2 import LeadResearcherV2, LeadResearcherConfig
from src.researcher.agents.models import (
    ResearchDecomposition, ResearchApproach, SubagentTaskSpec,
    QueryComplexity, ResearchSynthesis
)
from src.researcher.agents.base import AgentContext, AgentResult, AgentState
from datetime import datetime
from src.researcher.memory.base import InMemoryStorage, ResearchMemory
from src.researcher.tools.base import ToolRegistry
from src.researcher.agents.mock_subagent import MockSubagent


@pytest.fixture
def memory():
    """Create test memory."""
    storage = InMemoryStorage()
    return ResearchMemory(storage)


@pytest.fixture
def tool_registry():
    """Create test tool registry."""
    return ToolRegistry()


@pytest.fixture
def mock_decomposition():
    """Create mock decomposition response."""
    return ResearchDecomposition(
        original_query="What are the latest AI agent architectures?",
        main_objective="Research current AI agent architectures and implementations",
        approach=ResearchApproach(
            complexity=QueryComplexity.MODERATE,
            estimated_subagents=3,
            approach_steps=[
                "Search for recent papers on AI agents",
                "Find implementation examples",
                "Compare different architectures"
            ],
            key_aspects=["multi-agent systems", "tool use", "memory systems"],
            search_strategy="broad-first then deep-dive"
        ),
        subagent_tasks=[
            SubagentTaskSpec(
                objective="Research academic papers on AI agent architectures",
                search_queries=["AI agent architectures 2024", "multi-agent systems"],
                focus_areas=["recent research", "novel architectures"],
                expected_findings="Academic papers and research summaries",
                priority="high"
            ),
            SubagentTaskSpec(
                objective="Find open-source implementations",
                search_queries=["AI agent github", "LangChain agents", "AutoGPT"],
                focus_areas=["code examples", "popular frameworks"],
                expected_findings="GitHub repositories and documentation",
                priority="medium"
            ),
            SubagentTaskSpec(
                objective="Analyze industry applications",
                search_queries=["AI agents production", "enterprise AI agents"],
                focus_areas=["real-world use cases", "performance metrics"],
                expected_findings="Case studies and benchmarks",
                priority="medium"
            )
        ],
        synthesis_instructions="Compare findings across academic, open-source, and industry perspectives"
    )


@pytest.fixture
def mock_synthesis():
    """Create mock synthesis response."""
    return ResearchSynthesis(
        executive_summary="AI agent architectures have evolved significantly with multi-agent systems",
        main_findings=[
            "Multi-agent systems show 90% better performance on complex tasks",
            "Tool use is now standard in production AI agents",
            "Memory systems are critical for long-running agents"
        ],
        detailed_sections=[
            {
                "title": "Academic Research",
                "content": "Recent papers focus on..."
            }
        ],
        connections_found=["All modern systems use some form of tool integration"],
        gaps_remaining=["Limited benchmarks for multi-agent systems"],
        recommendations=["Implement tool use early in agent design"]
    )


@pytest.mark.asyncio
async def test_lead_researcher_initialization(memory, tool_registry):
    """Test LeadResearcherV2 initialization."""
    config = LeadResearcherConfig(
        max_subagents=3,
        anthropic_api_key="test-key"
    )
    
    lead = LeadResearcherV2(
        memory=memory,
        tool_registry=tool_registry,
        subagent_class=MockSubagent,
        config=config,
        name="TestLead"
    )
    
    assert lead.name == "TestLead"
    assert lead.config.max_subagents == 3
    assert lead.subagents == []


@pytest.mark.asyncio
async def test_lead_researcher_planning(memory, tool_registry, mock_decomposition):
    """Test planning with mocked Instructor response."""
    config = LeadResearcherConfig(anthropic_api_key="test-key")
    
    lead = LeadResearcherV2(
        memory=memory,
        tool_registry=tool_registry,
        subagent_class=MockSubagent,
        config=config
    )
    
    # Mock the Instructor client
    mock_create = AsyncMock(return_value=mock_decomposition)
    lead.client.chat.completions.create = mock_create
    
    context = AgentContext(
        query="What are the latest AI agent architectures?",
        objective="Research AI agents",
        constraints=["Focus on 2024 developments"]
    )
    
    plan = await lead.plan(context)
    
    # Verify planning
    assert len(plan) == 1
    assert plan[0]["action"] == "execute_research"
    assert plan[0]["decomposition"] == mock_decomposition
    
    # Verify research plan was saved
    saved_plan = await memory.get_research_plan(lead.agent_id)
    assert saved_plan is not None
    assert saved_plan.query == context.query
    assert len(saved_plan.subagent_tasks) == 3


@pytest.mark.asyncio
async def test_lead_researcher_execution_parallel(
    memory, tool_registry, mock_decomposition, mock_synthesis
):
    """Test parallel execution of subagents."""
    config = LeadResearcherConfig(
        parallel_execution=True,
        anthropic_api_key="test-key"
    )
    
    lead = LeadResearcherV2(
        memory=memory,
        tool_registry=tool_registry,
        subagent_class=MockSubagent,
        config=config
    )
    
    # Mock synthesis
    lead.client.chat.completions.create = AsyncMock(return_value=mock_synthesis)
    
    # Start the agent
    context = AgentContext(
        query="Test query",
        objective="Test objective"
    )
    
    # Manually set the plan to skip planning phase
    plan = [{"action": "execute_research", "decomposition": mock_decomposition}]
    
    # Execute
    lead._result = AgentResult(
        agent_id=lead.agent_id,
        status=AgentState.EXECUTING,
        start_time=datetime.utcnow()
    )
    
    result = await lead.execute(plan)
    
    # Verify execution
    assert "Executive Summary" in result
    assert "Multi-agent systems show 90% better" in result
    assert len(lead.subagents) == 3
    
    # Verify intermediate results were stored
    intermediate = await memory.get_intermediate_results(lead.agent_id)
    assert len(intermediate) == 3


@pytest.mark.asyncio
async def test_lead_researcher_execution_sequential(
    memory, tool_registry, mock_decomposition, mock_synthesis
):
    """Test sequential execution of subagents."""
    config = LeadResearcherConfig(
        parallel_execution=False,
        anthropic_api_key="test-key"
    )
    
    lead = LeadResearcherV2(
        memory=memory,
        tool_registry=tool_registry,
        subagent_class=MockSubagent,
        config=config
    )
    
    # Mock synthesis
    lead.client.chat.completions.create = AsyncMock(return_value=mock_synthesis)
    
    # Execute
    plan = [{"action": "execute_research", "decomposition": mock_decomposition}]
    lead._result = AgentResult(
        agent_id=lead.agent_id,
        status=AgentState.EXECUTING,
        start_time=datetime.utcnow()
    )
    
    result = await lead.execute(plan)
    
    # Verify
    assert "Executive Summary" in result
    assert len(lead.subagents) == 3


@pytest.mark.asyncio
async def test_lead_researcher_with_failed_subagent(
    memory, tool_registry, mock_decomposition, mock_synthesis
):
    """Test handling of failed subagents."""
    config = LeadResearcherConfig(
        parallel_execution=True,
        anthropic_api_key="test-key",
        subagent_timeout=0.1  # Very short timeout
    )
    
    # Create a slow subagent that will timeout
    class SlowMockSubagent(MockSubagent):
        def __init__(self, **kwargs):
            super().__init__(delay=1.0, **kwargs)  # 1 second delay
    
    lead = LeadResearcherV2(
        memory=memory,
        tool_registry=tool_registry,
        subagent_class=SlowMockSubagent,
        config=config
    )
    
    # Mock synthesis
    lead.client.chat.completions.create = AsyncMock(return_value=mock_synthesis)
    
    # Execute
    plan = [{"action": "execute_research", "decomposition": mock_decomposition}]
    lead._result = AgentResult(
        agent_id=lead.agent_id,
        status=AgentState.EXECUTING,
        start_time=datetime.utcnow()
    )
    
    result = await lead.execute(plan)
    
    # Should still produce results even with timeouts
    assert "Executive Summary" in result


@pytest.mark.asyncio 
async def test_decomposition_fallback_on_error(memory, tool_registry):
    """Test fallback behavior when decomposition fails."""
    config = LeadResearcherConfig(anthropic_api_key="test-key")
    
    lead = LeadResearcherV2(
        memory=memory,
        tool_registry=tool_registry,
        subagent_class=MockSubagent,
        config=config
    )
    
    # Mock client to raise exception
    lead.client.chat.completions.create = AsyncMock(
        side_effect=Exception("API Error")
    )
    
    context = AgentContext(
        query="Test query",
        objective="Test objective"
    )
    
    with pytest.raises(Exception, match="API Error"):
        await lead.plan(context)


def test_synthesis_formatting(mock_synthesis):
    """Test synthesis formatting."""
    lead = LeadResearcherV2(
        memory=MagicMock(),
        tool_registry=MagicMock(),
        subagent_class=MockSubagent,
        config=LeadResearcherConfig()
    )
    
    formatted = lead._format_synthesis(mock_synthesis)
    
    assert "# Research Results" in formatted
    assert "## Executive Summary" in formatted
    assert "## Main Findings" in formatted
    assert "1. Multi-agent systems show" in formatted
    assert "## Patterns and Connections" in formatted
    assert "## Recommendations" in formatted
    assert "## Information Gaps" in formatted