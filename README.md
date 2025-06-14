# Multi-Agent Research System

A sophisticated multi-agent research system that uses [Instructor](https://github.com/jxnl/instructor) for structured LLM outputs and [Exa.ai](https://exa.ai) for neural web search.

## Features

- 🤖 **Multi-Agent Architecture**: Lead researcher decomposes queries and coordinates parallel subagents
- 🔍 **Neural Search**: Integrates Exa.ai for semantic web search 
- 📊 **Structured Outputs**: Uses Instructor to get typed, validated responses from LLMs
- 💾 **Persistent Memory**: Stores research plans and intermediate results
- 🔄 **Iterative Refinement**: Subagents evaluate and refine their searches
- ⚡ **Parallel Execution**: Multiple agents work simultaneously for faster results

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd researcher

# Install with uv (recommended)
pip install uv
uv sync

# Or with pip
pip install -e .
```

### Environment Setup

```bash
# Required for LLM decomposition
export ANTHROPIC_API_KEY="your-anthropic-key"

# Required for web search (optional - will use mock data without it)
export EXA_API_KEY="your-exa-key" 
```

### Run Examples

```bash
# Simple research query
uv run python examples/simple_research.py

# Compare AI frameworks
uv run python examples/comparative_research.py

# Find recent AI developments
uv run python examples/time_bounded_research.py

# Academic paper research
uv run python examples/academic_research.py

# Demo with mock data (no API keys needed)
uv run python examples/demo_with_mocks.py
```

## Architecture

The system implements a hierarchical multi-agent architecture:

```
User Query → Lead Researcher → Query Decomposition (via Instructor)
                ↓
        Parallel Subagents → Iterative Search → Result Synthesis
                ↓
        Memory Storage → Citation Addition → Final Report
```

### Key Components

1. **LeadResearcherV2**: Orchestrates the research process using Instructor for structured task decomposition
2. **ResearchSubagent**: Performs iterative searches with self-evaluation
3. **Exa.ai Integration**: High-quality neural search for web content
4. **Memory System**: SQLite-backed persistence for research data
5. **Tool Registry**: Pluggable architecture for adding new data sources

## Usage

```python
import asyncio
from src.researcher.agents.lead_v2 import LeadResearcherV2, LeadResearcherConfig
from src.researcher.agents.subagent import ResearchSubagent
from src.researcher.agents.base import AgentContext
from src.researcher.memory.base import InMemoryStorage, ResearchMemory
from src.researcher.tools.base import ToolRegistry
from src.researcher.tools.exa_search import ExaSearchTool

async def research():
    # Setup
    memory = ResearchMemory(InMemoryStorage())
    registry = ToolRegistry()
    registry.register(ExaSearchTool(), category="search")
    
    # Configure
    config = LeadResearcherConfig(
        max_subagents=3,
        parallel_execution=True
    )
    
    # Create lead researcher
    lead = LeadResearcherV2(
        memory=memory,
        tool_registry=registry,
        subagent_class=ResearchSubagent,
        config=config
    )
    
    # Run research
    context = AgentContext(
        query="What are the latest advances in AI agents?",
        objective="Comprehensive overview of AI agent technology"
    )
    
    result = await lead.run(context)
    print(result.output)

asyncio.run(research())
```

## How It Works

1. **Query Decomposition**: The lead researcher uses Instructor to break down your query into structured subtasks
2. **Parallel Research**: Multiple subagents work simultaneously on different aspects
3. **Iterative Search**: Each subagent evaluates results and refines searches
4. **Synthesis**: Results are combined into a comprehensive report with citations

## Documentation

See the [docs/](docs/) directory for comprehensive documentation including:
- Architecture details
- API reference  
- Best practices
- Troubleshooting guide

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test file
uv run pytest tests/unit/test_lead_researcher_v2.py -v
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [Instructor](https://github.com/jxnl/instructor) for reliable structured outputs
- Powered by [Exa.ai](https://exa.ai) for neural search capabilities
- Inspired by Anthropic's multi-agent research architecture