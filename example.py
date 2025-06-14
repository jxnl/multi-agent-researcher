"""Example demonstrating the multi-agent research system with Instructor."""

import asyncio
import os
from src.researcher.agents.lead_v2 import LeadResearcherV2, LeadResearcherConfig
from src.researcher.agents.subagent import ResearchSubagent, SubagentConfig
from src.researcher.agents.base import AgentContext
from src.researcher.memory.base import InMemoryStorage, ResearchMemory
from src.researcher.tools.base import ToolRegistry
from src.researcher.tools.mock_search import MockSearchTool


async def main():
    """Run a research task using the multi-agent system."""
    
    # Setup memory
    storage = InMemoryStorage()
    memory = ResearchMemory(storage)
    
    # Setup tools
    registry = ToolRegistry()
    
    # Add mock search tool with sample results
    search_tool = MockSearchTool(results=[
        {
            "title": "Introduction to AI Agents",
            "url": "https://example.com/ai-agents-intro",
            "snippet": "AI agents are autonomous systems that perceive their environment and take actions..."
        },
        {
            "title": "Multi-Agent Systems Architecture",
            "url": "https://arxiv.org/multi-agent",
            "snippet": "Multi-agent systems consist of multiple interacting intelligent agents..."
        },
        {
            "title": "Building AI Agents with LangChain",
            "url": "https://langchain.com/agents",
            "snippet": "LangChain provides tools for building AI agents with memory and tools..."
        },
        {
            "title": "AutoGPT: An Autonomous AI Agent",
            "url": "https://github.com/autogpt",
            "snippet": "AutoGPT demonstrates autonomous AI agent capabilities..."
        },
        {
            "title": "AI Agents in Production",
            "url": "https://techblog.com/ai-agents-production",
            "snippet": "Deploying AI agents at scale requires careful consideration of..."
        }
    ])
    registry.register(search_tool, category="search")
    
    # Configure the system
    # Note: Set ANTHROPIC_API_KEY environment variable to use real API
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    lead_config = LeadResearcherConfig(
        max_subagents=3,
        parallel_execution=True,
        anthropic_api_key=api_key,
        model="claude-3-5-sonnet-20241022"
    )
    
    subagent_config = SubagentConfig(
        max_search_iterations=2,
        anthropic_api_key=api_key,
        enable_thinking=True
    )
    
    # Create lead researcher
    lead = LeadResearcherV2(
        memory=memory,
        tool_registry=registry,
        subagent_class=ResearchSubagent,
        config=lead_config,
        name="LeadResearcher"
    )
    
    # Create research context
    context = AgentContext(
        query="What are the latest developments in AI agent architectures, and how do they compare?",
        objective="Provide a comprehensive overview of modern AI agent architectures",
        constraints=[
            "Focus on developments from 2023-2024",
            "Include both academic research and practical implementations",
            "Compare at least 3 different approaches"
        ]
    )
    
    print("🔍 Starting multi-agent research...")
    print(f"Query: {context.query}\n")
    
    try:
        # Run the research
        result = await lead.run(context)
        
        print("\n✅ Research completed!")
        print(f"Status: {result.status}")
        print(f"Tokens used: {result.tokens_used:,}")
        print(f"Execution time: {(result.end_time - result.start_time).total_seconds():.2f}s")
        
        print("\n📊 Thinking process:")
        for thought in result.thinking:
            print(f"  - {thought}")
            
        print("\n📄 Research Results:")
        print("-" * 80)
        print(result.output)
        print("-" * 80)
        
        # Show memory contents
        print("\n💾 Memory Contents:")
        plan = await memory.get_research_plan(lead.agent_id)
        if plan:
            print(f"  Research plan saved with {len(plan.subagent_tasks)} tasks")
            
        intermediate = await memory.get_intermediate_results(lead.agent_id)
        print(f"  {len(intermediate)} intermediate results stored")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Note: This example uses mock data by default
    # To use real Anthropic API:
    # 1. Set ANTHROPIC_API_KEY environment variable
    # 2. The system will use Instructor to get structured outputs from Claude
    
    print("Multi-Agent Research System Demo")
    print("=" * 40)
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  No ANTHROPIC_API_KEY found - using mock responses")
        print("   Set the environment variable to use real AI decomposition\n")
    
    asyncio.run(main())