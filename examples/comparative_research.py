"""Comparative research example - comparing different AI frameworks."""

import asyncio
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.researcher.agents.lead_v2 import LeadResearcherV2, LeadResearcherConfig
from src.researcher.agents.subagent import ResearchSubagent, SubagentConfig
from src.researcher.agents.base import AgentContext
from src.researcher.memory.base import InMemoryStorage, ResearchMemory
from src.researcher.tools.base import ToolRegistry
from src.researcher.tools.exa_search import ExaSearchTool
from src.researcher.tools.mock_search import MockSearchTool

console = Console()


async def comparative_research():
    """Compare different AI agent frameworks."""
    
    # Setup
    memory = ResearchMemory(InMemoryStorage())
    registry = ToolRegistry()
    
    # Register search tool
    if os.getenv("EXA_API_KEY"):
        registry.register(ExaSearchTool(), category="search")
    else:
        # Use mock with relevant data
        mock_tool = MockSearchTool(results=[
            {
                "title": "LangChain Documentation - Agents",
                "url": "https://docs.langchain.com/agents",
                "snippet": "LangChain provides a framework for building agents with tools, memory, and chains..."
            },
            {
                "title": "AutoGPT: Autonomous AI Agent",
                "url": "https://github.com/Significant-Gravitas/AutoGPT",
                "snippet": "AutoGPT is an experimental open-source application showcasing GPT-4's autonomous capabilities..."
            },
            {
                "title": "CrewAI: Framework for Multi-Agent Systems",
                "url": "https://crewai.io",
                "snippet": "CrewAI enables AI agents to work together, each with specific roles and goals..."
            },
            {
                "title": "Comparing AI Agent Frameworks 2024",
                "url": "https://aiframeworks.com/comparison",
                "snippet": "LangChain excels at tool integration, AutoGPT at autonomy, CrewAI at multi-agent coordination..."
            }
        ])
        registry.register(mock_tool, category="search")
    
    # Configure
    config = LeadResearcherConfig(
        max_subagents=3,
        parallel_execution=True,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    lead = LeadResearcherV2(
        memory=memory,
        tool_registry=registry,
        subagent_class=ResearchSubagent,
        config=config
    )
    
    # Research query
    context = AgentContext(
        query="Compare LangChain, AutoGPT, and CrewAI frameworks for building AI agents",
        objective="Provide detailed comparison of AI agent frameworks",
        constraints=[
            "Compare architecture and design philosophy",
            "Include code examples where possible",
            "Evaluate ease of use and learning curve",
            "Compare performance and scalability",
            "Include community and ecosystem factors"
        ]
    )
    
    console.print(Panel("🔍 Comparative Research: AI Agent Frameworks", style="bold blue"))
    
    # Run research
    with console.status("[bold green]Analyzing frameworks...", spinner="dots"):
        result = await lead.run(context)
    
    # Display results
    if result.status == "completed":
        console.print("\n✅ Research completed!\n")
        
        # Create comparison table
        table = Table(title="Framework Comparison Summary")
        table.add_column("Aspect", style="cyan", no_wrap=True)
        table.add_column("LangChain", style="green")
        table.add_column("AutoGPT", style="yellow")
        table.add_column("CrewAI", style="magenta")
        
        # Mock comparison data (would be extracted from actual results)
        table.add_row(
            "Primary Focus",
            "Tool integration",
            "Autonomous operation",
            "Multi-agent coordination"
        )
        table.add_row(
            "Learning Curve",
            "Moderate",
            "Steep",
            "Easy"
        )
        table.add_row(
            "Best For",
            "RAG applications",
            "Autonomous tasks",
            "Team simulations"
        )
        
        console.print(table)
        console.print("\n" + "-"*80 + "\n")
        console.print(result.output)
        
    else:
        console.print(f"❌ Research failed: {result.error}", style="red")


async def main():
    """Run comparative research example."""
    console.print(Panel(
        "📊 Comparative Research Example\n\n"
        "This example compares different AI agent frameworks.",
        title="Framework Comparison",
        style="bold cyan"
    ))
    
    try:
        await comparative_research()
    except Exception as e:
        console.print(f"\n❌ Error: {e}", style="red")


if __name__ == "__main__":
    asyncio.run(main())