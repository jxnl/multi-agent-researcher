# Multi-Agent Research System Documentation

A sophisticated multi-agent research system that uses Instructor for structured LLM outputs and Exa.ai for neural web search.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Components](#components)
6. [Examples](#examples)
7. [API Reference](#api-reference)
8. [Best Practices](#best-practices)

## Overview

This system implements a multi-agent architecture for conducting comprehensive research tasks. It features:

- **Structured Query Decomposition**: Uses Instructor to break down complex queries into subtasks
- **Parallel Agent Execution**: Multiple subagents work simultaneously on different aspects
- **Neural Web Search**: Integrates Exa.ai for high-quality semantic search
- **Persistent Memory**: Stores research plans and intermediate results
- **Flexible Tool System**: Easy to add new data sources and capabilities

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    User     │────▶│Lead Research│────▶│   Memory    │
└─────────────┘     │    Agent    │     └─────────────┘
                    └──────┬──────┘
                           │
                ┌──────────┴──────────┐
                │                     │
         ┌──────▼──────┐      ┌──────▼──────┐
         │  Subagent 1 │      │  Subagent 2 │
         └──────┬──────┘      └──────┬──────┘
                │                     │
         ┌──────▼──────┐      ┌──────▼──────┐
         │   Exa.ai    │      │   Exa.ai    │
         │   Search    │      │   Search    │
         └─────────────┘      └─────────────┘
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd researcher

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Environment Setup

Set your API keys:

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export EXA_API_KEY="your-exa-key"
```

## Quick Start

```python
import asyncio
from src.researcher.agents.lead_v2 import LeadResearcherV2, LeadResearcherConfig
from src.researcher.agents.subagent import ResearchSubagent
from src.researcher.agents.base import AgentContext
from src.researcher.memory.base import InMemoryStorage, ResearchMemory
from src.researcher.tools.base import ToolRegistry
from src.researcher.tools.exa_search import ExaSearchTool

async def research_topic():
    # Setup
    memory = ResearchMemory(InMemoryStorage())
    registry = ToolRegistry()
    registry.register(ExaSearchTool(), category="search")
    
    # Configure lead researcher
    config = LeadResearcherConfig(
        max_subagents=3,
        parallel_execution=True
    )
    
    lead = LeadResearcherV2(
        memory=memory,
        tool_registry=registry,
        subagent_class=ResearchSubagent,
        config=config
    )
    
    # Run research
    context = AgentContext(
        query="What are the latest AI agent architectures?",
        objective="Comprehensive overview of modern AI agents"
    )
    
    result = await lead.run(context)
    print(result.output)

# Run
asyncio.run(research_topic())
```

## Components

### 1. Lead Researcher Agent

The orchestrator that manages the research process:

```python
from src.researcher.agents.lead_v2 import LeadResearcherV2

lead = LeadResearcherV2(
    memory=memory,
    tool_registry=registry,
    subagent_class=ResearchSubagent,
    config=LeadResearcherConfig(
        max_subagents=5,
        parallel_execution=True,
        model="claude-3-5-sonnet-20241022"
    )
)
```

### 2. Research Subagent

Specialized agents that perform iterative searches:

```python
from src.researcher.agents.subagent import ResearchSubagent

subagent = ResearchSubagent(
    tool_registry=registry,
    config=SubagentConfig(
        max_search_iterations=3,
        enable_thinking=True
    )
)
```

### 3. Exa.ai Search Tools

Neural search capabilities:

```python
from src.researcher.tools.exa_search import ExaSearchTool

# Basic search
search_tool = ExaSearchTool()
results = await search_tool.execute(
    query="AI agent architectures",
    num_results=10,
    search_type="neural"
)

# Date-filtered search
results = await search_tool.execute(
    query="multi-agent systems",
    start_published_date="2024-01-01",
    include_domains=["arxiv.org"]
)
```

### 4. Memory System

Persistent storage for research data:

```python
from src.researcher.memory.base import ResearchMemory, InMemoryStorage

# Create memory
memory = ResearchMemory(InMemoryStorage())

# Save research plan
await memory.save_research_plan(agent_id, plan)

# Retrieve intermediate results
results = await memory.get_intermediate_results(agent_id)
```

## Examples

### Example 1: Simple Research Task

```python
# examples/simple_research.py
import asyncio
from src.researcher import create_research_system

async def main():
    system = create_research_system()
    
    result = await system.research(
        "What is Retrieval Augmented Generation (RAG)?",
        constraints=["Focus on practical implementations"]
    )
    
    print(result)

asyncio.run(main())
```

### Example 2: Comparative Research

```python
# examples/comparative_research.py
async def compare_frameworks():
    system = create_research_system()
    
    result = await system.research(
        "Compare LangChain, AutoGPT, and CrewAI frameworks",
        constraints=[
            "Include code examples",
            "Focus on agent capabilities",
            "Compare performance metrics"
        ]
    )
    
    return result
```

### Example 3: Time-Bounded Research

```python
# examples/recent_developments.py
async def recent_ai_news():
    system = create_research_system()
    
    # Configure for recent results only
    search_tool = ExaSearchTool()
    system.registry.register(search_tool)
    
    result = await system.research(
        "Latest breakthroughs in AI agents",
        constraints=["Only include developments from last 30 days"]
    )
    
    return result
```

### Example 4: Domain-Specific Research

```python
# examples/academic_research.py
async def academic_papers():
    system = create_research_system()
    
    result = await system.research(
        "Multi-agent reinforcement learning",
        constraints=[
            "Focus on academic papers",
            "Include arxiv.org and openai.com",
            "Prioritize 2024 publications"
        ]
    )
    
    return result
```

## API Reference

### AgentContext

```python
class AgentContext(BaseModel):
    query: str                    # The research query
    objective: str               # Clear objective statement
    constraints: List[str]       # Constraints and requirements
    metadata: Dict[str, Any]     # Additional context
    parent_agent_id: Optional[str]  # For subagents
```

### LeadResearcherConfig

```python
class LeadResearcherConfig(BaseModel):
    max_subagents: int = 5          # Maximum parallel subagents
    max_iterations: int = 3         # Max refinement iterations
    parallel_execution: bool = True # Enable parallel execution
    context_window_limit: int = 200000  # Token limit
    subagent_timeout: float = 300.0 # Timeout per subagent
    model: str = "claude-3-5-sonnet-20241022"  # LLM model
```

### Tool Registration

```python
# Register a tool
registry.register(tool, category="search")

# List available tools
tools = registry.list_tools(category="search")

# Search for tools
matching = registry.search_tools("exa")
```

## Best Practices

### 1. Query Formulation

- Be specific about what you want to research
- Include temporal constraints when relevant
- Specify output format preferences

```python
# Good
context = AgentContext(
    query="How do transformer-based AI agents handle tool use?",
    objective="Technical analysis of tool integration in LLM agents",
    constraints=["Include implementation details", "Focus on 2024 research"]
)

# Too vague
context = AgentContext(
    query="Tell me about AI",
    objective="Learn about AI"
)
```

### 2. Managing Costs

- Set appropriate `max_subagents` based on query complexity
- Use `subagent_timeout` to prevent runaway searches
- Monitor token usage through results

```python
config = LeadResearcherConfig(
    max_subagents=2,  # Start small
    subagent_timeout=60.0,  # 1 minute limit
    parallel_execution=True
)
```

### 3. Error Handling

```python
try:
    result = await lead.run(context)
except Exception as e:
    # Check memory for partial results
    intermediate = await memory.get_intermediate_results(lead.agent_id)
    if intermediate:
        print(f"Partial results available: {len(intermediate)}")
```

### 4. Custom Tools

Extend the system with custom tools:

```python
class CustomAPITool(Tool):
    def __init__(self):
        super().__init__("custom_api", "Fetch data from custom API")
        
    async def _execute(self, endpoint: str, **kwargs):
        # Implementation
        pass
        
# Register
registry.register(CustomAPITool(), category="custom")
```

### 5. Production Deployment

For production use:

1. Use persistent storage (SQLite/PostgreSQL) instead of InMemoryStorage
2. Implement proper logging and monitoring
3. Set up rate limiting for API calls
4. Use environment-specific configurations
5. Implement retry logic for transient failures

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   export ANTHROPIC_API_KEY="sk-..."
   export EXA_API_KEY="..."
   ```

2. **Timeout Errors**
   - Increase `subagent_timeout` in config
   - Reduce `max_subagents` for complex queries

3. **Memory Issues**
   - Use SQLite for large research tasks
   - Implement cleanup for old research data

4. **Rate Limiting**
   - Implement exponential backoff
   - Use multiple API keys if available

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[Your License]