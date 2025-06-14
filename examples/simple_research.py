"""Simple research example using the multi-agent system."""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.researcher.agents.lead_v2 import LeadResearcherV2, LeadResearcherConfig
from src.researcher.agents.subagent import ResearchSubagent, SubagentConfig
from src.researcher.agents.base import AgentContext
from src.researcher.memory.base import InMemoryStorage, ResearchMemory
from src.researcher.tools.base import ToolRegistry
from src.researcher.tools.exa_search import ExaSearchTool

console = Console()


async def simple_research_example():
    """Run a simple research query."""
    
    # Setup components
    console.print(Panel("🔧 Setting up research system...", style="blue"))
    
    # Memory
    storage = InMemoryStorage()
    memory = ResearchMemory(storage)
    
    # Tools
    registry = ToolRegistry()
    
    # Use Exa if API key is available
    if os.getenv("EXA_API_KEY"):
        console.print("✅ Using Exa.ai for real web search")
        registry.register(ExaSearchTool(), category="search")
    else:
        console.print("⚠️  No EXA_API_KEY found - using mock search")
        from src.researcher.tools.mock_search import MockSearchTool
        registry.register(MockSearchTool(), category="search")
    
    # Configure agents
    lead_config = LeadResearcherConfig(
        max_subagents=2,
        parallel_execution=True,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    subagent_config = SubagentConfig(
        max_search_iterations=2,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Create lead researcher
    lead = LeadResearcherV2(
        memory=memory,
        tool_registry=registry,
        subagent_class=ResearchSubagent,
        config=lead_config,
        name="SimpleResearchLead"
    )
    
    # Define research query
    query = "What is Retrieval Augmented Generation (RAG) and how is it used in AI applications?"
    
    context = AgentContext(
        query=query,
        objective="Understand RAG technology and its applications",
        constraints=[
            "Include practical examples",
            "Explain the technical architecture",
            "Focus on recent implementations"
        ]
    )
    
    console.print(Panel(f"🔍 Research Query: {query}", style="green"))
    
    # Run research
    with console.status("[bold blue]Researching...", spinner="dots"):
        result = await lead.run(context)
    
    # Display results
    console.print("\n" + "="*80 + "\n")
    console.print(Panel("📊 Research Results", style="bold magenta"))
    
    if result.status == "completed":
        console.print(f"✅ Status: [green]{result.status}[/green]")
        console.print(f"⏱️  Time: {(result.end_time - result.start_time).total_seconds():.2f}s")
        console.print(f"🪙 Tokens: {result.tokens_used:,}")
        
        console.print("\n[bold]Thinking Process:[/bold]")
        for thought in result.thinking:
            console.print(f"  💭 {thought}")
        
        console.print("\n" + "-"*80 + "\n")
        console.print(Markdown(result.output))
        
        # Show memory usage
        console.print("\n[bold]Memory Contents:[/bold]")
        plan = await memory.get_research_plan(lead.agent_id)
        if plan:
            console.print(f"  📋 Research plan with {len(plan.subagent_tasks)} tasks")
            
        intermediate = await memory.get_intermediate_results(lead.agent_id)
        console.print(f"  📄 {len(intermediate)} intermediate results stored")
        
    else:
        console.print(f"❌ Research failed: {result.error}", style="red")


async def main():
    """Main entry point."""
    console.print(Panel(
        "🚀 Simple Research Example\n\n"
        "This example demonstrates basic usage of the multi-agent research system.",
        title="Multi-Agent Research System",
        style="bold blue"
    ))
    
    # Check for API keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print("\n⚠️  Warning: No ANTHROPIC_API_KEY found", style="yellow")
        console.print("The system will use mock responses for testing.\n")
        
    try:
        await simple_research_example()
    except Exception as e:
        console.print(f"\n❌ Error: {e}", style="red")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())