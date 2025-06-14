"""Demo with fully mocked LLM responses for testing without API keys."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

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

console = Console()


def create_mock_decomposition():
    """Create mock decomposition response."""
    return ResearchDecomposition(
        original_query="What is RAG and how is it used?",
        main_objective="Understand Retrieval Augmented Generation",
        approach=ResearchApproach(
            complexity=QueryComplexity.MODERATE,
            estimated_subagents=2,
            approach_steps=[
                "Define RAG and its components",
                "Explore practical applications"
            ],
            key_aspects=["definition", "architecture", "use cases"],
            search_strategy="focused search on RAG concepts"
        ),
        subagent_tasks=[
            SubagentTaskSpec(
                objective="Research RAG definition and architecture",
                search_queries=["RAG retrieval augmented generation", "RAG architecture"],
                focus_areas=["technical definition", "system components"],
                expected_findings="Clear explanation of RAG"
            ),
            SubagentTaskSpec(
                objective="Find RAG applications and examples",
                search_queries=["RAG applications", "RAG use cases"],
                focus_areas=["real-world applications", "implementation examples"],
                expected_findings="Practical RAG implementations"
            )
        ],
        synthesis_instructions="Combine technical understanding with practical applications"
    )


def create_mock_synthesis():
    """Create mock synthesis response."""
    return ResearchSynthesis(
        executive_summary="RAG (Retrieval Augmented Generation) is a technique that enhances LLMs by retrieving relevant information from external sources.",
        main_findings=[
            "RAG combines the power of retrieval systems with generative AI models",
            "It reduces hallucinations by grounding responses in retrieved facts",
            "Common applications include question answering, chatbots, and knowledge systems",
            "Key components: Vector database, Embedding model, Retrieval mechanism, and LLM"
        ],
        detailed_sections=[
            {
                "title": "How RAG Works",
                "content": "1. User query is embedded into a vector\n2. Similar documents are retrieved from vector store\n3. Retrieved context is provided to LLM\n4. LLM generates response based on context"
            },
            {
                "title": "Benefits",
                "content": "- More accurate and factual responses\n- Ability to cite sources\n- Can work with private/proprietary data\n- Reduces model hallucinations"
            }
        ],
        connections_found=["All modern AI assistants use some form of retrieval augmentation"],
        recommendations=["Implement RAG for any system requiring factual accuracy"]
    )


async def demo_research():
    """Run demo with mocked responses."""
    
    # Setup
    memory = ResearchMemory(InMemoryStorage())
    registry = ToolRegistry()
    
    # Mock search tool
    search_tool = MockSearchTool(results=[
        {
            "title": "What is RAG? - Retrieval Augmented Generation Explained",
            "url": "https://example.com/rag-explained",
            "snippet": "RAG enhances large language models by retrieving relevant information..."
        },
        {
            "title": "Building RAG Applications with LangChain",
            "url": "https://example.com/rag-langchain",
            "snippet": "Step-by-step guide to implementing RAG using LangChain and vector databases..."
        }
    ])
    registry.register(search_tool, category="search")
    
    # Configure
    config = LeadResearcherConfig(
        max_subagents=2,
        parallel_execution=True,
        anthropic_api_key="mock-key"  # Won't be used
    )
    
    # Create lead researcher
    lead = LeadResearcherV2(
        memory=memory,
        tool_registry=registry,
        subagent_class=MockSubagent,
        config=config
    )
    
    # Mock the LLM calls
    lead.client.chat.completions.create = AsyncMock(
        side_effect=[create_mock_decomposition(), create_mock_synthesis()]
    )
    
    # Research context
    context = AgentContext(
        query="What is Retrieval Augmented Generation (RAG) and how is it used?",
        objective="Understand RAG technology",
        constraints=["Include practical examples", "Explain technical architecture"]
    )
    
    console.print(Panel("🎯 Demo: Multi-Agent Research System", style="bold cyan"))
    console.print(f"\n🔍 Query: {context.query}\n")
    
    # Run research
    with console.status("[bold green]Running research demo...", spinner="dots"):
        result = await lead.run(context)
    
    # Display results
    console.print("✅ Research completed!\n")
    console.print(f"📊 Status: {result.status}")
    console.print(f"⏱️  Time: {(result.end_time - result.start_time).total_seconds():.2f}s")
    console.print(f"🪙 Tokens: {result.tokens_used:,}")
    
    console.print("\n💭 Thinking Process:")
    for thought in result.thinking:
        console.print(f"  • {thought}")
    
    console.print("\n" + "="*80)
    console.print(result.output)
    console.print("="*80)
    
    # Show what was saved in memory
    console.print("\n💾 Memory Contents:")
    plan = await memory.get_research_plan(lead.agent_id)
    if plan:
        console.print(f"  • Saved research plan with {len(plan.subagent_tasks)} tasks")
        for i, task in enumerate(plan.subagent_tasks):
            console.print(f"    {i+1}. {task['objective']}")
    
    intermediate = await memory.get_intermediate_results(lead.agent_id)
    console.print(f"  • {len(intermediate)} intermediate results from subagents")


async def main():
    """Run the demo."""
    console.print(Panel(
        "This demo shows the multi-agent research system with mocked LLM responses.\n"
        "No API keys required - perfect for testing and understanding the flow.",
        title="Mock Demo",
        style="bold yellow"
    ))
    
    try:
        await demo_research()
    except Exception as e:
        console.print(f"\n❌ Error: {e}", style="red")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())