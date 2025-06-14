"""Tests for memory system."""

import pytest
from datetime import datetime
from src.researcher.memory.base import (
    InMemoryStorage, MemoryEntry, ResearchPlan, 
    IntermediateResult, ResearchMemory
)


@pytest.mark.asyncio
async def test_in_memory_storage_basic():
    """Test basic in-memory storage operations."""
    storage = InMemoryStorage()
    
    # Test store and retrieve
    await storage.store("key1", "value1")
    assert await storage.retrieve("key1") == "value1"
    
    # Test with agent_id
    await storage.store("key2", "value2", agent_id="agent-123")
    assert await storage.retrieve("key2", agent_id="agent-123") == "value2"
    assert await storage.retrieve("key2") is None  # Different scope
    
    # Test delete
    assert await storage.delete("key1") is True
    assert await storage.retrieve("key1") is None
    assert await storage.delete("key1") is False  # Already deleted


@pytest.mark.asyncio
async def test_storage_with_metadata():
    """Test storage with metadata."""
    storage = InMemoryStorage()
    
    await storage.store(
        "config",
        {"setting": "value"},
        agent_id="agent-1",
        metadata={"version": "1.0", "type": "config"}
    )
    
    # Verify internal storage structure
    key = storage._make_key("config", "agent-1")
    entry = storage._storage[key]
    assert entry.metadata["version"] == "1.0"
    assert entry.metadata["type"] == "config"


@pytest.mark.asyncio
async def test_list_keys():
    """Test listing keys with filters."""
    storage = InMemoryStorage()
    
    # Add various keys
    await storage.store("config:main", "value1", agent_id="agent-1")
    await storage.store("config:backup", "value2", agent_id="agent-1")
    await storage.store("data:results", "value3", agent_id="agent-1")
    await storage.store("config:other", "value4", agent_id="agent-2")
    
    # List all keys for agent-1
    keys = await storage.list_keys(agent_id="agent-1")
    assert set(keys) == {"config:main", "config:backup", "data:results"}
    
    # List with prefix filter
    keys = await storage.list_keys(prefix="config:", agent_id="agent-1")
    assert set(keys) == {"config:main", "config:backup"}
    
    # List all keys (no agent filter)
    keys = await storage.list_keys()
    assert len(keys) == 4


@pytest.mark.asyncio
async def test_clear_storage():
    """Test clearing storage."""
    storage = InMemoryStorage()
    
    # Add data for multiple agents
    await storage.store("key1", "value1", agent_id="agent-1")
    await storage.store("key2", "value2", agent_id="agent-1")
    await storage.store("key3", "value3", agent_id="agent-2")
    
    # Clear specific agent
    await storage.clear(agent_id="agent-1")
    assert await storage.retrieve("key1", agent_id="agent-1") is None
    assert await storage.retrieve("key3", agent_id="agent-2") == "value3"
    
    # Clear all
    await storage.clear()
    assert len(storage._storage) == 0


@pytest.mark.asyncio
async def test_research_memory_plan():
    """Test research plan storage."""
    storage = InMemoryStorage()
    memory = ResearchMemory(storage)
    
    plan = ResearchPlan(
        query="What are AI agents?",
        objective="Research AI agent architectures",
        approach=["Search for papers", "Analyze implementations"],
        subagent_tasks=[
            {"task": "search_papers", "keywords": ["AI agents", "multi-agent"]},
            {"task": "search_code", "repositories": ["github"]}
        ],
        constraints=["Focus on recent work", "Include practical examples"]
    )
    
    await memory.save_research_plan("agent-123", plan)
    
    # Retrieve plan
    retrieved = await memory.get_research_plan("agent-123")
    assert retrieved is not None
    assert retrieved.query == plan.query
    assert retrieved.objective == plan.objective
    assert len(retrieved.subagent_tasks) == 2


@pytest.mark.asyncio
async def test_intermediate_results():
    """Test storing and retrieving intermediate results."""
    storage = InMemoryStorage()
    memory = ResearchMemory(storage)
    
    # Add multiple results
    result1 = IntermediateResult(
        subagent_id="subagent-1",
        task_description="Search for papers",
        findings=[
            {"title": "Paper 1", "url": "http://example.com/1"},
            {"title": "Paper 2", "url": "http://example.com/2"}
        ],
        sources=["scholar.google.com"],
        tokens_used=500
    )
    
    result2 = IntermediateResult(
        subagent_id="subagent-2",
        task_description="Search for code",
        findings=[
            {"repo": "example/agent", "stars": 100}
        ],
        sources=["github.com"],
        tokens_used=300
    )
    
    await memory.add_intermediate_result("agent-123", result1)
    await memory.add_intermediate_result("agent-123", result2)
    
    # Retrieve all results
    results = await memory.get_intermediate_results("agent-123")
    assert len(results) == 2
    assert results[0].subagent_id == "subagent-1"
    assert results[1].subagent_id == "subagent-2"
    assert sum(r.tokens_used for r in results) == 800


@pytest.mark.asyncio
async def test_checkpointing():
    """Test checkpoint save and restore."""
    storage = InMemoryStorage()
    memory = ResearchMemory(storage)
    
    # Save checkpoints
    state1 = {
        "step": 1,
        "status": "planning",
        "subagents": []
    }
    await memory.checkpoint_state("agent-123", state1)
    
    # Add another checkpoint
    import asyncio
    await asyncio.sleep(0.01)  # Ensure different timestamp
    
    state2 = {
        "step": 2,
        "status": "executing",
        "subagents": ["sub-1", "sub-2"]
    }
    await memory.checkpoint_state("agent-123", state2)
    
    # Get latest checkpoint
    latest = await memory.get_latest_checkpoint("agent-123")
    assert latest is not None
    assert latest["step"] == 2
    assert latest["status"] == "executing"
    assert len(latest["subagents"]) == 2


@pytest.mark.asyncio
async def test_memory_isolation():
    """Test that agent memories are properly isolated."""
    storage = InMemoryStorage()
    memory = ResearchMemory(storage)
    
    # Create plans for different agents
    plan1 = ResearchPlan(
        query="Query 1",
        objective="Objective 1",
        approach=["Approach 1"]
    )
    
    plan2 = ResearchPlan(
        query="Query 2", 
        objective="Objective 2",
        approach=["Approach 2"]
    )
    
    await memory.save_research_plan("agent-1", plan1)
    await memory.save_research_plan("agent-2", plan2)
    
    # Verify isolation
    retrieved1 = await memory.get_research_plan("agent-1")
    retrieved2 = await memory.get_research_plan("agent-2")
    
    assert retrieved1.query == "Query 1"
    assert retrieved2.query == "Query 2"