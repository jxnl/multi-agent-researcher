"""Mock search tool for testing."""

from typing import Any, Dict, List, Optional
from src.researcher.tools.base import Tool, ToolDescription


class MockSearchTool(Tool):
    """Mock web search tool for testing."""
    
    def __init__(self, results: Optional[List[Dict[str, str]]] = None):
        super().__init__(
            name="web_search",
            description="Search the web for information"
        )
        self.results = results or []
        self.search_history: List[str] = []
        
    def get_description(self) -> ToolDescription:
        return ToolDescription(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 10
                    }
                },
                "required": ["query"]
            },
            examples=[
                {
                    "query": "AI agents research",
                    "limit": 5
                }
            ],
            cost_per_call=0.001,
            timeout=10.0
        )
        
    async def _execute(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """Execute mock search."""
        self.search_history.append(query)
        
        # Return predefined results or generate based on query
        if self.results:
            return self.results[:limit]
            
        # Generate mock results
        mock_results = []
        for i in range(min(limit, 3)):
            mock_results.append({
                "title": f"Result {i+1} for: {query}",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a mock search result for query: {query}"
            })
            
        return mock_results