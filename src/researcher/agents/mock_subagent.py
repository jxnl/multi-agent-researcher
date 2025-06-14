"""Mock subagent for testing."""

from typing import Any, Dict, List
from src.researcher.agents.base import Agent, AgentContext


class MockSubagent(Agent[str]):
    """Simple mock subagent for testing lead researcher."""
    
    def __init__(self, should_fail: bool = False, delay: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.should_fail = should_fail
        self.delay = delay
        
    async def plan(self, context: AgentContext) -> List[Dict[str, Any]]:
        """Create simple search plan."""
        self.add_thinking(f"Planning search for: {context.query}")
        
        if self.should_fail:
            raise ValueError("Mock planning failure")
            
        return [{"action": "search", "query": context.query}]
        
    async def execute(self, plan: List[Dict[str, Any]]) -> str:
        """Execute mock search."""
        import asyncio
        
        if self.delay > 0:
            await asyncio.sleep(self.delay)
            
        if self.should_fail:
            raise RuntimeError("Mock execution failure")
            
        query = plan[0]["query"]
        self.add_thinking(f"Searching for: {query}")
        
        # Simulate some token usage
        self._result.tokens_used = 100
        
        return f"Mock findings for '{query}': This is simulated research data with relevant information."