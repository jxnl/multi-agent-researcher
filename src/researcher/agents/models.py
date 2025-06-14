"""Pydantic models for structured LLM outputs using Instructor."""

from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum


class QueryComplexity(str, Enum):
    """Query complexity assessment."""
    SIMPLE = "simple"  # Single fact-finding, direct answer
    MODERATE = "moderate"  # Multi-aspect, comparison, or analysis
    COMPLEX = "complex"  # Open-ended research, comprehensive investigation


class ResearchApproach(BaseModel):
    """Structured research approach."""
    complexity: QueryComplexity = Field(
        description="Assessed complexity of the research query"
    )
    estimated_subagents: int = Field(
        description="Recommended number of subagents (1-10)",
        ge=1,
        le=10
    )
    approach_steps: List[str] = Field(
        description="High-level steps for the research approach"
    )
    key_aspects: List[str] = Field(
        description="Key aspects or themes to investigate"
    )
    search_strategy: str = Field(
        description="Overall search strategy (broad-first, deep-dive, targeted, etc.)"
    )


class SubagentTaskSpec(BaseModel):
    """Specification for a single subagent task."""
    objective: str = Field(
        description="Clear, specific objective for this subagent"
    )
    search_queries: List[str] = Field(
        description="Initial search queries to explore (2-5 queries)",
        min_length=1,
        max_length=5
    )
    focus_areas: List[str] = Field(
        description="Specific areas or aspects to focus on"
    )
    constraints: List[str] = Field(
        description="Any specific constraints or requirements",
        default_factory=list
    )
    expected_findings: str = Field(
        description="Description of expected output/findings"
    )
    priority: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Task priority"
    )


class ResearchDecomposition(BaseModel):
    """Complete decomposition of research query into subagent tasks."""
    original_query: str = Field(description="The original user query")
    main_objective: str = Field(
        description="Clear statement of the main research objective"
    )
    approach: ResearchApproach = Field(
        description="Overall research approach and strategy"
    )
    subagent_tasks: List[SubagentTaskSpec] = Field(
        description="Detailed tasks for each subagent",
        min_length=1,
        max_length=10
    )
    synthesis_instructions: str = Field(
        description="Instructions for synthesizing subagent results"
    )


class SubagentSearchPlan(BaseModel):
    """Search plan for a subagent."""
    refined_queries: List[str] = Field(
        description="Refined search queries based on task",
        min_length=1,
        max_length=5
    )
    search_sequence: List[str] = Field(
        description="Sequence of search actions to take"
    )
    evaluation_criteria: List[str] = Field(
        description="Criteria for evaluating search results"
    )


class SearchResultEvaluation(BaseModel):
    """Evaluation of search results by subagent."""
    relevance_score: float = Field(
        description="Relevance score 0-1",
        ge=0,
        le=1
    )
    key_findings: List[str] = Field(
        description="Key findings from the search"
    )
    gaps_identified: List[str] = Field(
        description="Information gaps that need further search",
        default_factory=list
    )
    next_queries: List[str] = Field(
        description="Suggested queries for next iteration",
        default_factory=list
    )
    sufficient_information: bool = Field(
        description="Whether sufficient information has been gathered"
    )


class SubagentReport(BaseModel):
    """Final report from a subagent."""
    task_summary: str = Field(
        description="Summary of the task completed"
    )
    findings: List[Dict[str, str]] = Field(
        description="List of findings with title, content, and source"
    )
    key_insights: List[str] = Field(
        description="Key insights discovered"
    )
    sources_used: List[str] = Field(
        description="List of sources consulted"
    )
    confidence_level: Literal["high", "medium", "low"] = Field(
        description="Confidence in the findings"
    )
    limitations: List[str] = Field(
        description="Any limitations or caveats",
        default_factory=list
    )


class ResearchSynthesis(BaseModel):
    """Synthesis of all subagent findings."""
    executive_summary: str = Field(
        description="High-level summary of research findings"
    )
    main_findings: List[str] = Field(
        description="Main findings organized by importance"
    )
    detailed_sections: List[Dict[str, str]] = Field(
        description="Detailed sections with title and content"
    )
    connections_found: List[str] = Field(
        description="Connections and patterns across subagent findings"
    )
    gaps_remaining: List[str] = Field(
        description="Any remaining information gaps",
        default_factory=list
    )
    recommendations: List[str] = Field(
        description="Recommendations based on findings",
        default_factory=list
    )


class CitationMapping(BaseModel):
    """Mapping of claims to citations."""
    claim: str = Field(description="The claim or statement")
    source_title: str = Field(description="Title of the source")
    source_url: str = Field(description="URL of the source")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the citation match"
    )