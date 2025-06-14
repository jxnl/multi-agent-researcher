"""Base agent abstractions for the multi-agent research system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar, Generic
from datetime import datetime
from enum import Enum
import uuid
from pydantic import BaseModel, Field


class AgentState(str, Enum):
    """Agent execution states."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolCall(BaseModel):
    """Represents a single tool invocation."""
    tool_name: str
    parameters: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    result: Optional[Any] = None
    error: Optional[str] = None


class AgentContext(BaseModel):
    """Context passed between agents."""
    query: str
    objective: str
    constraints: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_agent_id: Optional[str] = None
    
    
class AgentResult(BaseModel):
    """Standardized result from agent execution."""
    agent_id: str
    status: AgentState
    output: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    thinking: List[str] = Field(default_factory=list)
    subagent_results: List['AgentResult'] = Field(default_factory=list)
    error: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    tokens_used: int = 0


T = TypeVar('T')


class Agent(ABC, Generic[T]):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, agent_id: Optional[str] = None, name: Optional[str] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.state = AgentState.IDLE
        self._context: Optional[AgentContext] = None
        self._result: Optional[AgentResult] = None
        
    @abstractmethod
    async def plan(self, context: AgentContext) -> List[Dict[str, Any]]:
        """Plan the approach for handling the given context."""
        pass
        
    @abstractmethod
    async def execute(self, plan: List[Dict[str, Any]]) -> T:
        """Execute the planned actions."""
        pass
        
    async def run(self, context: AgentContext) -> AgentResult:
        """Main entry point for agent execution."""
        self._context = context
        self.state = AgentState.PLANNING
        start_time = datetime.utcnow()
        
        self._result = AgentResult(
            agent_id=self.agent_id,
            status=self.state,
            start_time=start_time
        )
        
        try:
            # Planning phase
            plan = await self.plan(context)
            
            # Execution phase
            self.state = AgentState.EXECUTING
            self._result.status = self.state
            
            output = await self.execute(plan)
            
            # Completion
            self.state = AgentState.COMPLETED
            self._result.status = self.state
            self._result.output = str(output)
            self._result.end_time = datetime.utcnow()
            
        except Exception as e:
            self.state = AgentState.FAILED
            self._result.status = self.state
            self._result.error = str(e)
            self._result.end_time = datetime.utcnow()
            raise
            
        return self._result
        
    def add_thinking(self, thought: str) -> None:
        """Add a thinking step to the agent's trace."""
        if self._result:
            self._result.thinking.append(thought)
            
    def record_tool_call(self, tool_call: ToolCall) -> None:
        """Record a tool invocation."""
        if self._result:
            self._result.tool_calls.append(tool_call)
            self._result.tokens_used += 100  # Placeholder token counting