"""Memory system for persistent agent state."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from datetime import datetime
from pydantic import BaseModel, Field
import json


class MemoryEntry(BaseModel):
    """Single memory entry."""
    key: str
    value: Any
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    

class ResearchPlan(BaseModel):
    """Research plan stored in memory."""
    query: str
    objective: str
    approach: List[str]
    subagent_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    

class IntermediateResult(BaseModel):
    """Intermediate results from subagents."""
    subagent_id: str
    task_description: str
    findings: List[Dict[str, Any]]
    sources: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tokens_used: int = 0
    

T = TypeVar('T')


class Memory(ABC, Generic[T]):
    """Abstract base class for memory storage."""
    
    @abstractmethod
    async def store(self, key: str, value: T, agent_id: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a value in memory."""
        pass
        
    @abstractmethod
    async def retrieve(self, key: str, agent_id: Optional[str] = None) -> Optional[T]:
        """Retrieve a value from memory."""
        pass
        
    @abstractmethod
    async def delete(self, key: str, agent_id: Optional[str] = None) -> bool:
        """Delete a value from memory."""
        pass
        
    @abstractmethod
    async def list_keys(self, prefix: Optional[str] = None, 
                       agent_id: Optional[str] = None) -> List[str]:
        """List all keys, optionally filtered by prefix."""
        pass
        
    @abstractmethod
    async def clear(self, agent_id: Optional[str] = None) -> None:
        """Clear all memory, optionally for specific agent."""
        pass


class InMemoryStorage(Memory[Any]):
    """Simple in-memory storage for testing."""
    
    def __init__(self):
        self._storage: Dict[str, MemoryEntry] = {}
        
    def _make_key(self, key: str, agent_id: Optional[str] = None) -> str:
        """Create composite key with optional agent scope."""
        if agent_id:
            return f"{agent_id}:{key}"
        return key
        
    async def store(self, key: str, value: Any, agent_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        composite_key = self._make_key(key, agent_id)
        entry = MemoryEntry(
            key=key,
            value=value,
            agent_id=agent_id,
            metadata=metadata or {}
        )
        self._storage[composite_key] = entry
        
    async def retrieve(self, key: str, agent_id: Optional[str] = None) -> Optional[Any]:
        composite_key = self._make_key(key, agent_id)
        entry = self._storage.get(composite_key)
        return entry.value if entry else None
        
    async def delete(self, key: str, agent_id: Optional[str] = None) -> bool:
        composite_key = self._make_key(key, agent_id)
        if composite_key in self._storage:
            del self._storage[composite_key]
            return True
        return False
        
    async def list_keys(self, prefix: Optional[str] = None,
                       agent_id: Optional[str] = None) -> List[str]:
        keys = []
        for composite_key, entry in self._storage.items():
            # Filter by agent if specified
            if agent_id and entry.agent_id != agent_id:
                continue
                
            # Filter by prefix if specified
            if prefix and not entry.key.startswith(prefix):
                continue
                
            keys.append(entry.key)
        return keys
        
    async def clear(self, agent_id: Optional[str] = None) -> None:
        if agent_id:
            # Clear only entries for specific agent
            to_delete = [k for k, v in self._storage.items() 
                        if v.agent_id == agent_id]
            for key in to_delete:
                del self._storage[key]
        else:
            # Clear everything
            self._storage.clear()


class ResearchMemory:
    """High-level memory interface for research agents."""
    
    def __init__(self, storage: Memory[Any]):
        self.storage = storage
        
    async def save_research_plan(self, agent_id: str, plan: ResearchPlan) -> None:
        """Save research plan to memory."""
        await self.storage.store(
            key="research_plan",
            value=plan.model_dump(),
            agent_id=agent_id,
            metadata={"type": "research_plan"}
        )
        
    async def get_research_plan(self, agent_id: str) -> Optional[ResearchPlan]:
        """Retrieve research plan from memory."""
        data = await self.storage.retrieve("research_plan", agent_id)
        if data:
            return ResearchPlan(**data)
        return None
        
    async def add_intermediate_result(self, agent_id: str, 
                                    result: IntermediateResult) -> None:
        """Add intermediate result from subagent."""
        # Store with unique key based on subagent and timestamp
        key = f"result:{result.subagent_id}:{result.timestamp.isoformat()}"
        await self.storage.store(
            key=key,
            value=result.model_dump(),
            agent_id=agent_id,
            metadata={"type": "intermediate_result", "subagent_id": result.subagent_id}
        )
        
    async def get_intermediate_results(self, agent_id: str) -> List[IntermediateResult]:
        """Get all intermediate results."""
        keys = await self.storage.list_keys(prefix="result:", agent_id=agent_id)
        results = []
        
        for key in keys:
            data = await self.storage.retrieve(key, agent_id)
            if data:
                results.append(IntermediateResult(**data))
                
        # Sort by timestamp
        results.sort(key=lambda r: r.timestamp)
        return results
        
    async def checkpoint_state(self, agent_id: str, state: Dict[str, Any]) -> None:
        """Save agent state checkpoint for recovery."""
        await self.storage.store(
            key=f"checkpoint:{datetime.utcnow().isoformat()}",
            value=state,
            agent_id=agent_id,
            metadata={"type": "checkpoint"}
        )
        
    async def get_latest_checkpoint(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get most recent checkpoint."""
        keys = await self.storage.list_keys(prefix="checkpoint:", agent_id=agent_id)
        if not keys:
            return None
            
        # Get the latest by key (ISO format sorts correctly)
        latest_key = sorted(keys)[-1]
        return await self.storage.retrieve(latest_key, agent_id)