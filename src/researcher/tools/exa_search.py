"""Exa.ai search tool integration."""

from typing import Any, Dict, List, Optional
import os
from exa_py import Exa
from datetime import datetime, timedelta

from src.researcher.tools.base import Tool, ToolDescription


class ExaSearchTool(Tool):
    """Real web search using Exa.ai neural search API."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="exa_search",
            description="Neural web search using Exa.ai for high-quality results"
        )
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY must be provided or set as environment variable")
        
        self.client = Exa(api_key=self.api_key)
        
    def get_description(self) -> ToolDescription:
        return ToolDescription(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query - can be natural language"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["neural", "keyword"],
                        "description": "Type of search - neural for semantic, keyword for traditional",
                        "default": "neural"
                    },
                    "include_text": {
                        "type": "boolean",
                        "description": "Whether to include page text content",
                        "default": True
                    },
                    "start_published_date": {
                        "type": "string",
                        "description": "Filter by start publication date (YYYY-MM-DD)",
                        "default": None
                    },
                    "include_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only include results from these domains",
                        "default": []
                    },
                    "exclude_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exclude results from these domains",
                        "default": []
                    }
                },
                "required": ["query"]
            },
            examples=[
                {
                    "query": "latest developments in AI agents 2024",
                    "num_results": 5,
                    "search_type": "neural"
                },
                {
                    "query": "multi-agent systems research papers",
                    "num_results": 10,
                    "include_domains": ["arxiv.org", "openai.com"],
                    "start_published_date": "2023-01-01"
                }
            ],
            cost_per_call=0.01,  # Approximate cost
            timeout=30.0
        )
        
    async def _execute(
        self, 
        query: str,
        num_results: int = 10,
        search_type: str = "neural",
        include_text: bool = True,
        start_published_date: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute Exa search with specified parameters."""
        
        # Build search parameters
        search_params = {
            "query": query,
            "num_results": num_results,
            "type": search_type
        }
        
        # Add date filter if specified
        if start_published_date:
            search_params["start_published_date"] = start_published_date
            
        # Add domain filters
        if include_domains:
            search_params["include_domains"] = include_domains
        if exclude_domains:
            search_params["exclude_domains"] = exclude_domains
            
        # Execute search
        if include_text:
            # Search and get content in one call
            results = self.client.search_and_contents(
                **search_params,
                text={"max_characters": 2000}  # Limit text length
            )
        else:
            # Just search without content
            results = self.client.search(**search_params)
            
        # Format results
        formatted_results = []
        for result in results.results:
            formatted = {
                "title": result.title,
                "url": result.url,
                "snippet": getattr(result, "text", "")[:500] if include_text else "",
                "published_date": getattr(result, "published_date", None),
                "score": getattr(result, "score", None),
                "highlights": getattr(result, "highlights", [])
            }
            
            # Add full text if available
            if include_text and hasattr(result, "text"):
                formatted["text"] = result.text
                
            formatted_results.append(formatted)
            
        return formatted_results


class ExaSimilarityTool(Tool):
    """Find similar content using Exa.ai."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="exa_find_similar",
            description="Find similar web pages to a given URL using Exa.ai"
        )
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY must be provided or set as environment variable")
        
        self.client = Exa(api_key=self.api_key)
        
    def get_description(self) -> ToolDescription:
        return ToolDescription(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to find similar content for"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of similar results",
                        "default": 10
                    },
                    "exclude_source_domain": {
                        "type": "boolean",
                        "description": "Exclude results from the same domain",
                        "default": True
                    }
                },
                "required": ["url"]
            },
            examples=[
                {
                    "url": "https://arxiv.org/abs/2301.00234",
                    "num_results": 5,
                    "exclude_source_domain": True
                }
            ],
            timeout=30.0
        )
        
    async def _execute(
        self,
        url: str,
        num_results: int = 10,
        exclude_source_domain: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Find similar content to a URL."""
        
        results = self.client.find_similar(
            url=url,
            num_results=num_results,
            exclude_source_domain=exclude_source_domain
        )
        
        # Format results
        formatted_results = []
        for result in results.results:
            formatted = {
                "title": result.title,
                "url": result.url,
                "score": getattr(result, "score", None),
                "published_date": getattr(result, "published_date", None)
            }
            formatted_results.append(formatted)
            
        return formatted_results


class ExaResearchTool(Tool):
    """Advanced research using Exa.ai with multiple searches."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="exa_research",
            description="Comprehensive research using multiple Exa searches"
        )
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        self.search_tool = ExaSearchTool(api_key=self.api_key)
        
    def get_description(self) -> ToolDescription:
        return ToolDescription(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Research topic"
                    },
                    "aspects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Different aspects to research",
                        "default": ["overview", "recent developments", "examples"]
                    },
                    "date_range_days": {
                        "type": "integer",
                        "description": "Search within last N days (0 for all time)",
                        "default": 0
                    },
                    "max_results_per_aspect": {
                        "type": "integer",
                        "description": "Results per aspect",
                        "default": 5
                    }
                },
                "required": ["topic"]
            },
            timeout=60.0  # Longer timeout for multiple searches
        )
        
    async def _execute(
        self,
        topic: str,
        aspects: List[str] = None,
        date_range_days: int = 0,
        max_results_per_aspect: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute comprehensive research on a topic."""
        
        if aspects is None:
            aspects = ["overview", "recent developments", "examples", "best practices"]
            
        # Calculate date filter
        start_date = None
        if date_range_days > 0:
            start_date = (datetime.now() - timedelta(days=date_range_days)).strftime("%Y-%m-%d")
            
        research_results = {
            "topic": topic,
            "aspects": {},
            "summary": []
        }
        
        # Search each aspect
        for aspect in aspects:
            query = f"{topic} {aspect}"
            
            try:
                results = await self.search_tool._execute(
                    query=query,
                    num_results=max_results_per_aspect,
                    search_type="neural",
                    include_text=True,
                    start_published_date=start_date
                )
                
                research_results["aspects"][aspect] = results
                
                # Add to summary
                if results:
                    research_results["summary"].append(
                        f"{aspect}: Found {len(results)} relevant results"
                    )
                    
            except Exception as e:
                research_results["aspects"][aspect] = {"error": str(e)}
                research_results["summary"].append(f"{aspect}: Search failed - {str(e)}")
                
        return research_results