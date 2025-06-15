"""Academic research example - finding and analyzing research papers."""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from src.researcher.agents.lead_v2 import LeadResearcherV2, LeadResearcherConfig
from src.researcher.agents.mock_subagent import MockSubagent
from src.researcher.agents.base import AgentContext
from src.researcher.memory.base import InMemoryStorage, ResearchMemory
from src.researcher.tools.base import ToolRegistry
from src.researcher.tools.exa_search import ExaSearchTool
from src.researcher.tools.mock_search import MockSearchTool

console = Console()


async def academic_research():
    """Research academic papers on a specific topic."""
    
    # Setup
    memory = ResearchMemory(InMemoryStorage())
    registry = ToolRegistry()
    
    # Configure search for academic sources
    if os.getenv("EXA_API_KEY"):
        search_tool = ExaSearchTool()
        registry.register(search_tool, category="search")
    else:
        # Mock academic results
        mock_tool = MockSearchTool(results=[
            {
                "title": "Attention Is All You Need",
                "url": "https://arxiv.org/abs/1706.03762",
                "snippet": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks..."
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "url": "https://arxiv.org/abs/1810.04805",
                "snippet": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations..."
            },
            {
                "title": "Language Models are Few-Shot Learners",
                "url": "https://arxiv.org/abs/2005.14165",
                "snippet": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on large corpus..."
            },
            {
                "title": "Constitutional AI: Harmlessness from AI Feedback",
                "url": "https://arxiv.org/abs/2212.08073",
                "snippet": "We propose a method for training harmless AI assistants without human labels..."
            }
        ])
        registry.register(mock_tool, category="search")
    
    # Configure for academic research
    config = LeadResearcherConfig(
        max_subagents=3,
        parallel_execution=True
    )
    
    lead = LeadResearcherV2(
        memory=memory,
        tool_registry=registry,
        subagent_class=MockSubagent,
        config=config
    )
    
    # Academic query
    topic = "transformer architectures in natural language processing"
    context = AgentContext(
        query=f"Find and analyze key research papers on {topic}",
        objective="Comprehensive literature review of transformer architectures",
        constraints=[
            "Focus on seminal papers and recent advances",
            "Include papers from arxiv.org, aclweb.org, and major conferences",
            "Extract key contributions and innovations",
            "Identify research trends and future directions",
            "Cite papers properly with authors and years"
        ],
        metadata={
            "domains": ["arxiv.org", "aclweb.org", "openai.com", "deepmind.com"],
            "academic_focus": True
        }
    )
    
    console.print(Panel(
        f"🎓 Academic Research\n\n"
        f"Topic: [bold]{topic}[/bold]",
        style="bold magenta"
    ))
    
    # Run research
    with console.status("[bold blue]Analyzing academic literature...", spinner="dots"):
        result = await lead.run(context)
    
    # Display results as a tree
    if result.status == "completed":
        console.print("\n✅ Literature review completed!\n")
        
        # Create a tree visualization of papers
        tree = Tree("📚 Research Papers Found")
        
        # Categories (would be extracted from actual results)
        foundational = tree.add("🏛️ Foundational Papers")
        foundational.add("• Attention Is All You Need (Vaswani et al., 2017)")
        foundational.add("• BERT (Devlin et al., 2018)")
        
        recent = tree.add("🆕 Recent Advances")
        recent.add("• GPT-3 (Brown et al., 2020)")
        recent.add("• Constitutional AI (Anthropic, 2022)")
        
        applications = tree.add("💡 Applications")
        applications.add("• Vision Transformers")
        applications.add("• Multimodal Transformers")
        
        console.print(tree)
        console.print("\n" + "-"*80 + "\n")
        console.print(result.output)
        
        # Show research metrics
        console.print(f"\n📊 Research Metrics:")
        console.print(f"  • Papers analyzed: ~{result.tokens_used // 500}")
        console.print(f"  • Domains searched: {len(context.metadata.get('domains', []))}")
        console.print(f"  • Execution time: {(result.end_time - result.start_time).total_seconds():.2f}s")
        
    else:
        console.print(f"❌ Research failed: {result.error}", style="red")


async def main():
    """Run academic research example."""
    console.print(Panel(
        "🎓 Academic Research Example\n\n"
        "This example demonstrates finding and analyzing academic papers\n"
        "on a specific research topic.",
        title="Literature Review",
        style="bold magenta"
    ))
    
    try:
        await academic_research()
    except Exception as e:
        console.print(f"\n❌ Error: {e}", style="red")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())