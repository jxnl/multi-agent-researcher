"""Integration test demonstrating the research system."""

import pytest
import os
from unittest.mock import AsyncMock

from src.researcher.agents.lead_v2 import LeadResearcherV2, LeadResearcherConfig
from src.researcher.agents.models import (
    ResearchDecomposition, ResearchApproach, SubagentTaskSpec,
    QueryComplexity, ResearchSynthesis
)
from src.researcher.agents.base import AgentContext
from src.researcher.agents.mock_subagent import MockSubagent
from src.researcher.memory.base import InMemoryStorage, ResearchMemory
from src.researcher.tools.base import ToolRegistry
from src.researcher.tools.mock_search import MockSearchTool


@pytest.fixture
def research_system():
    """Create a complete research system."""
    # Memory
    storage = InMemoryStorage()
    memory = ResearchMemory(storage)
    
    # Tools
    registry = ToolRegistry()
    search_tool = MockSearchTool(results=[
        {"title": "AI Agents Paper", "url": "http://arxiv.org/1", "snippet": "Recent advances in AI agents..."},
        {"title": "Multi-Agent Systems", "url": "http://arxiv.org/2", "snippet": "Coordination in multi-agent..."},
        {"title": "LangChain Docs", "url": "http://docs.langchain.com", "snippet": "Building AI agents with..."}
    ])
    registry.register(search_tool, category="search")
    
    # Config
    config = LeadResearcherConfig(
        max_subagents=2,
        parallel_execution=True,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "test-key")
    )
    
    # Lead researcher
    lead = LeadResearcherV2(
        memory=memory,
        tool_registry=registry,
        subagent_class=MockSubagent,
        config=config
    )
    
    return {
        "lead": lead,
        "memory": memory,
        "registry": registry
    }


def create_mock_decomposition():
    """Create a simple mock decomposition."""
    return ResearchDecomposition(
        original_query="What are AI agents?",
        main_objective="Understand AI agent architectures and applications",
        approach=ResearchApproach(
            complexity=QueryComplexity.MODERATE,
            estimated_subagents=2,
            approach_steps=["Research definitions", "Find examples"],
            key_aspects=["architecture", "applications"],
            search_strategy="broad overview"
        ),
        subagent_tasks=[
            SubagentTaskSpec(
                objective="Research AI agent definitions and architectures",
                search_queries=["AI agents definition", "agent architectures"],
                focus_areas=["technical definitions", "system design"],
                expected_findings="Clear definitions and architectural patterns"
            ),
            SubagentTaskSpec(
                objective="Find AI agent applications and examples",
                search_queries=["AI agent applications", "agent use cases"],
                focus_areas=["real-world examples", "industry applications"],
                expected_findings="Concrete examples and case studies"
            )
        ],
        synthesis_instructions="Combine technical understanding with practical applications"
    )


def create_mock_synthesis():
    """Create a simple mock synthesis."""
    return ResearchSynthesis(
        executive_summary="AI agents are autonomous systems that perceive, decide, and act to achieve goals.",
        main_findings=[
            "AI agents use perception-reasoning-action loops",
            "Multi-agent systems enable complex problem solving",
            "Tool use is essential for practical AI agents"
        ],
        detailed_sections=[
            {
                "title": "Architecture",
                "content": "Modern AI agents typically include perception, reasoning, and action components..."
            },
            {
                "title": "Applications", 
                "content": "AI agents are used in customer service, research, coding assistance..."
            }
        ],
        connections_found=["All successful agents incorporate some form of memory"],
        recommendations=["Start with simple reactive agents before building complex systems"]
    )


@pytest.mark.asyncio
async def test_research_system_flow(research_system):
    """Test the complete research flow."""
    lead = research_system["lead"]
    memory = research_system["memory"]
    
    # Mock the LLM calls
    lead.client.chat.completions.create = AsyncMock(
        side_effect=[create_mock_decomposition(), create_mock_synthesis()]
    )
    
    # Create research context
    context = AgentContext(
        query="What are AI agents?",
        objective="Learn about AI agents",
        constraints=["Focus on recent developments", "Include practical examples"]
    )
    
    # Run the complete research
    result = await lead.run(context)
    
    # Verify the result
    assert result.status == "completed"
    assert result.output is not None
    assert "Executive Summary" in result.output
    assert "AI agents are autonomous systems" in result.output
    
    # Verify memory was used
    saved_plan = await memory.get_research_plan(lead.agent_id)
    assert saved_plan is not None
    assert saved_plan.query == "What are AI agents?"
    assert len(saved_plan.subagent_tasks) == 2
    
    # Verify intermediate results
    intermediate = await memory.get_intermediate_results(lead.agent_id)
    assert len(intermediate) == 2
    
    # Verify metrics
    assert result.tokens_used > 0
    assert len(result.thinking) > 0
    assert len(result.subagent_results) == 0  # Subagent results stored separately


@pytest.mark.asyncio
async def test_research_with_failing_subagent(research_system):
    """Test research with one failing subagent."""
    lead = research_system["lead"]
    
    # Use mixed subagents - one normal, one failing
    class MixedMockSubagent(MockSubagent):
        call_count = 0
        
        def __init__(self, **kwargs):
            MixedMockSubagent.call_count += 1
            # First subagent succeeds, second fails
            should_fail = MixedMockSubagent.call_count % 2 == 0
            super().__init__(should_fail=should_fail, **kwargs)
    
    lead.subagent_class = MixedMockSubagent
    
    # Mock the LLM calls
    lead.client.chat.completions.create = AsyncMock(
        side_effect=[create_mock_decomposition(), create_mock_synthesis()]
    )
    
    context = AgentContext(
        query="Test query with failure",
        objective="Test resilience"
    )
    
    # Should still complete despite one failure
    result = await lead.run(context)
    assert result.status == "completed"
    assert result.output is not None


@pytest.mark.asyncio
async def test_memory_persistence(research_system):
    """Test that research state persists in memory."""
    lead = research_system["lead"]
    memory = research_system["memory"]
    
    # Mock decomposition with specific ID
    decomposition = create_mock_decomposition()
    lead.client.chat.completions.create = AsyncMock(
        side_effect=[decomposition, create_mock_synthesis()]
    )
    
    # Run research
    context = AgentContext(query="Memory test", objective="Test memory")
    await lead.run(context)
    
    # Create new lead with same memory
    new_lead = LeadResearcherV2(
        memory=memory,
        tool_registry=research_system["registry"],
        subagent_class=MockSubagent,
        config=lead.config
    )
    
    # Should be able to retrieve previous research
    old_plan = await memory.get_research_plan(lead.agent_id)
    assert old_plan is not None
    assert old_plan.query == "Memory test"
    
    # Should have intermediate results
    results = await memory.get_intermediate_results(lead.agent_id)
    assert len(results) > 0