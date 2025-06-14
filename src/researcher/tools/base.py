"""Base tool interface for the multi-agent system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, TypeVar
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field
import asyncio
import time


class ToolStatus(str, Enum):
    """Tool execution status."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


class ToolResult(BaseModel):
    """Result from tool execution."""
    status: ToolStatus
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_after: Optional[float] = None  # For rate limiting
    

class ToolDescription(BaseModel):
    """Tool metadata for agent discovery."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    cost_per_call: float = 0.0  # Token/API cost estimate
    max_retries: int = 3
    timeout: float = 30.0
    

T = TypeVar('T')


class Tool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._call_count = 0
        self._total_time = 0.0
        
    @abstractmethod
    def get_description(self) -> ToolDescription:
        """Return tool description for agent discovery."""
        pass
        
    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """Internal execution logic to be implemented by subclasses."""
        pass
        
    async def execute(self, **kwargs) -> ToolResult:
        """Execute tool with error handling and metrics."""
        start_time = time.time()
        self._call_count += 1
        
        try:
            # Get timeout from description
            timeout = self.get_description().timeout
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute(**kwargs),
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            self._total_time += execution_time
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                error=f"Tool execution timed out after {timeout}s",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._total_time += execution_time
            
            # Check if it's a rate limit error
            if "rate limit" in str(e).lower():
                return ToolResult(
                    status=ToolStatus.RATE_LIMITED,
                    error=str(e),
                    execution_time=execution_time,
                    retry_after=60.0  # Default retry after 60s
                )
            
            return ToolResult(
                status=ToolStatus.FAILURE,
                error=str(e),
                execution_time=execution_time
            )
            
    def get_metrics(self) -> Dict[str, float]:
        """Return tool usage metrics."""
        avg_time = self._total_time / self._call_count if self._call_count > 0 else 0
        return {
            "call_count": self._call_count,
            "total_time": self._total_time,
            "average_time": avg_time
        }


class ToolRegistry:
    """Registry for discovering and managing tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}
        
    def register(self, tool: Tool, category: Optional[str] = None) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool
        
        if category:
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(tool.name)
            
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self._tools.get(name)
        
    def list_tools(self, category: Optional[str] = None) -> List[ToolDescription]:
        """List available tools, optionally filtered by category."""
        if category:
            tool_names = self._categories.get(category, [])
            tools = [self._tools[name] for name in tool_names if name in self._tools]
        else:
            tools = list(self._tools.values())
            
        return [tool.get_description() for tool in tools]
        
    def search_tools(self, query: str) -> List[ToolDescription]:
        """Search tools by name or description."""
        query_lower = query.lower()
        matching_tools = []
        
        for tool in self._tools.values():
            desc = tool.get_description()
            if (query_lower in desc.name.lower() or 
                query_lower in desc.description.lower()):
                matching_tools.append(desc)
                
        return matching_tools


class ParallelToolExecutor:
    """Executes multiple tools in parallel."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        
    async def execute_parallel(
        self, 
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolResult]:
        """Execute multiple tool calls in parallel."""
        tasks = []
        
        for call in tool_calls:
            tool_name = call.get("tool")
            params = call.get("parameters", {})
            
            tool = self.registry.get_tool(tool_name)
            if tool:
                tasks.append(tool.execute(**params))
            else:
                # Create failed result for missing tool
                tasks.append(asyncio.create_task(
                    self._create_error_result(f"Tool '{tool_name}' not found")
                ))
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to ToolResult
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                final_results.append(ToolResult(
                    status=ToolStatus.FAILURE,
                    error=str(result)
                ))
            else:
                final_results.append(result)
                
        return final_results
        
    async def _create_error_result(self, error: str) -> ToolResult:
        """Create an error result."""
        return ToolResult(status=ToolStatus.FAILURE, error=error)