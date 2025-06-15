"""Time-bounded research example - finding recent developments."""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.researcher.agents.lead_v2 import LeadResearcherV2, LeadResearcherConfig
from src.researcher.agents.mock_subagent import MockSubagent
from src.researcher.agents.base import AgentContext
from src.researcher.memory.base import InMemoryStorage, ResearchMemory
from src.researcher.tools.base import ToolRegistry
from src.researcher.tools.exa_search import ExaSearchTool, ExaResearchTool
from src.researcher.tools.mock_search import MockSearchTool

console = Console()


async def recent_developments_research():
    """Research recent AI developments."""
    
    # Setup
    memory = ResearchMemory(InMemoryStorage())
    registry = ToolRegistry()
    
    # Calculate date range
    days_back = 30
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    if os.getenv("EXA_API_KEY"):
        # Use Exa research tool for comprehensive search
        research_tool = ExaResearchTool()
        registry.register(research_tool, category="research")
        registry.register(ExaSearchTool(), category="search")
    else:
        # Mock recent results
        mock_tool = MockSearchTool(results=[
            {
                "title": "OpenAI Announces GPT-5 Development",
                "url": "https://openai.com/gpt5",
                "snippet": f"Breaking: OpenAI reveals GPT-5 is in development with enhanced reasoning... (Published: {datetime.now().strftime('%Y-%m-%d')})"
            },
            {
                "title": "Google's Gemini 2.0 Shows Multi-Modal Excellence",
                "url": "https://deepmind.google/gemini2",
                "snippet": f"Gemini 2.0 demonstrates unprecedented multi-modal capabilities... (Published: {(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')})"
            },
            {
                "title": "Anthropic's Claude 3.5 Achieves New Benchmarks",
                "url": "https://anthropic.com/claude35",
                "snippet": f"Claude 3.5 sets new standards in helpfulness and harmlessness... (Published: {(datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')})"
            }
        ])
        registry.register(mock_tool, category="search")
    
    # Configure for recent search
    config = LeadResearcherConfig(
        max_subagents=4,  # More subagents for different aspects
        parallel_execution=True
    )
    
    lead = LeadResearcherV2(
        memory=memory,
        tool_registry=registry,
        subagent_class=MockSubagent,
        config=config
    )
    
    # Time-bounded query
    context = AgentContext(
        query=f"What are the most significant AI breakthroughs and announcements in the last {days_back} days?",
        objective="Identify and analyze recent AI developments",
        constraints=[
            f"Only include developments from {start_date} onwards",
            "Focus on major announcements from leading AI companies",
            "Include technical breakthroughs and product launches",
            "Highlight implications for the AI industry",
            "Prioritize verified sources"
        ],
        metadata={
            "start_date": start_date,
            "time_sensitive": True
        }
    )
    
    console.print(Panel(
        f"📅 Time-Bounded Research\n\n"
        f"Searching for AI developments from: [bold]{start_date}[/bold] to [bold]today[/bold]",
        style="bold yellow"
    ))
    
    # Run with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Searching recent developments...", total=None)
        result = await lead.run(context)
        progress.update(task, completed=True)
    
    # Display timeline
    if result.status == "completed":
        console.print("\n✅ Recent developments found!\n")
        
        # Create timeline visualization
        console.print(Panel("📰 AI Developments Timeline", style="bold"))
        console.print(result.output)
        
        # Show search metadata
        console.print(f"\n📊 Search Metadata:")
        console.print(f"  • Time range: {days_back} days")
        console.print(f"  • Start date: {start_date}")
        console.print(f"  • Subagents used: {config.max_subagents}")
        console.print(f"  • Total tokens: {result.tokens_used:,}")
        
    else:
        console.print(f"❌ Search failed: {result.error}", style="red")


async def main():
    """Run time-bounded research example."""
    console.print(Panel(
        "⏰ Time-Bounded Research Example\n\n"
        "This example finds recent AI developments within a specific time window.",
        title="Recent AI News",
        style="bold yellow"
    ))
    
    try:
        await recent_developments_research()
    except Exception as e:
        console.print(f"\n❌ Error: {e}", style="red")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())