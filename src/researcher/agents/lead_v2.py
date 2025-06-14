"""Lead researcher agent using Instructor for structured decomposition."""

from typing import Any, Dict, List, Optional, Type
import asyncio
import uuid
from pydantic import BaseModel
import instructor
from anthropic import AsyncAnthropic

from src.researcher.agents.base import Agent, AgentContext, AgentResult, AgentState
from src.researcher.agents.models import (
    ResearchDecomposition, ResearchApproach, SubagentTaskSpec,
    ResearchSynthesis, SubagentReport
)
from src.researcher.memory.base import ResearchMemory, ResearchPlan, IntermediateResult
from src.researcher.tools.base import ToolRegistry


class LeadResearcherConfig(BaseModel):
    """Configuration for lead researcher."""
    max_subagents: int = 5
    max_iterations: int = 3
    parallel_execution: bool = True
    context_window_limit: int = 200000
    subagent_timeout: float = 300.0
    anthropic_api_key: Optional[str] = None
    model: str = "claude-3-5-sonnet-20241022"


class LeadResearcherV2(Agent[str]):
    """Lead researcher using Instructor for structured outputs."""
    
    def __init__(
        self,
        memory: ResearchMemory,
        tool_registry: ToolRegistry,
        subagent_class: Type[Agent],
        config: Optional[LeadResearcherConfig] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.memory = memory
        self.tool_registry = tool_registry
        self.subagent_class = subagent_class
        self.config = config or LeadResearcherConfig()
        
        # Initialize Anthropic client with Instructor
        client = AsyncAnthropic(api_key=self.config.anthropic_api_key)
        self.client = instructor.from_anthropic(client)
        
        self.subagents: List[Agent] = []
        self.decomposition: Optional[ResearchDecomposition] = None
        
    async def plan(self, context: AgentContext) -> List[Dict[str, Any]]:
        """Use Instructor to decompose query into structured research plan."""
        self.add_thinking(f"Analyzing query: {context.query}")
        
        # Get structured decomposition from LLM
        decomposition_prompt = f"""
        You are a lead research agent. Analyze this query and create a comprehensive research plan.
        
        Query: {context.query}
        Objective: {context.objective}
        Constraints: {', '.join(context.constraints) if context.constraints else 'None'}
        
        Break this down into specific subtasks for parallel research agents. Consider:
        1. What are the key aspects that need investigation?
        2. How can we divide this into independent research tasks?
        3. What search strategies would be most effective?
        4. How many agents do we need (1-{self.config.max_subagents})?
        
        Create diverse, complementary tasks that together will provide comprehensive coverage.
        """
        
        try:
            self.decomposition = await self.client.chat.completions.create(
                model=self.config.model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": "You are an expert research orchestrator."},
                    {"role": "user", "content": decomposition_prompt}
                ],
                response_model=ResearchDecomposition,
                max_retries=2
            )
            
            self.add_thinking(f"Decomposed into {len(self.decomposition.subagent_tasks)} tasks")
            
            # Save research plan to memory
            plan = ResearchPlan(
                query=context.query,
                objective=self.decomposition.main_objective,
                approach=self.decomposition.approach.approach_steps,
                subagent_tasks=[task.model_dump() for task in self.decomposition.subagent_tasks],
                constraints=context.constraints
            )
            await self.memory.save_research_plan(self.agent_id, plan)
            
            return [{"action": "execute_research", "decomposition": self.decomposition}]
            
        except Exception as e:
            self.add_thinking(f"Failed to decompose query: {str(e)}")
            raise
            
    async def execute(self, plan: List[Dict[str, Any]]) -> str:
        """Execute research plan with subagents."""
        decomposition = plan[0]["decomposition"]
        
        # Execute subagents based on decomposition
        if self.config.parallel_execution:
            self.add_thinking("Executing subagents in parallel")
            subagent_results = await self._execute_parallel(decomposition.subagent_tasks)
        else:
            self.add_thinking("Executing subagents sequentially")
            subagent_results = await self._execute_sequential(decomposition.subagent_tasks)
            
        # Store intermediate results
        total_tokens = 0
        for i, (task, result) in enumerate(zip(decomposition.subagent_tasks, subagent_results)):
            if result.status == AgentState.COMPLETED and result.output:
                intermediate = IntermediateResult(
                    subagent_id=result.agent_id,
                    task_description=task.objective,
                    findings=[{"content": result.output}],
                    sources=[],
                    tokens_used=result.tokens_used
                )
                await self.memory.add_intermediate_result(self.agent_id, intermediate)
                total_tokens += result.tokens_used
                
        # Update token count
        if self._result:
            self._result.tokens_used = total_tokens + 1000  # Add estimate for LLM calls
                
        # Synthesize results using Instructor
        synthesis = await self._synthesize_with_llm(decomposition, subagent_results)
        
        return self._format_synthesis(synthesis)
        
    async def _execute_parallel(self, tasks: List[SubagentTaskSpec]) -> List[AgentResult]:
        """Execute subagents in parallel."""
        coroutines = []
        
        for i, task in enumerate(tasks):
            subagent = self.subagent_class(
                agent_id=str(uuid.uuid4()),
                name=f"Subagent-{i+1}"
            )
            self.subagents.append(subagent)
            
            context = AgentContext(
                query=' '.join(task.search_queries),
                objective=task.objective,
                constraints=task.constraints,
                metadata={
                    "focus_areas": task.focus_areas,
                    "expected_findings": task.expected_findings,
                    "priority": task.priority
                },
                parent_agent_id=self.agent_id
            )
            
            coro = asyncio.wait_for(
                subagent.run(context),
                timeout=self.config.subagent_timeout
            )
            coroutines.append(coro)
            
        # Execute all subagents
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed = AgentResult(
                    agent_id=self.subagents[i].agent_id,
                    status=AgentState.FAILED,
                    error=str(result),
                    start_time=self._result.start_time
                )
                final_results.append(failed)
            else:
                final_results.append(result)
                
        return final_results
        
    async def _execute_sequential(self, tasks: List[SubagentTaskSpec]) -> List[AgentResult]:
        """Execute subagents sequentially."""
        results = []
        
        for i, task in enumerate(tasks):
            try:
                subagent = self.subagent_class(
                    agent_id=str(uuid.uuid4()),
                    name=f"Subagent-{i+1}"
                )
                self.subagents.append(subagent)
                
                context = AgentContext(
                    query=' '.join(task.search_queries),
                    objective=task.objective,
                    constraints=task.constraints,
                    metadata={
                        "focus_areas": task.focus_areas,
                        "expected_findings": task.expected_findings,
                        "priority": task.priority
                    },
                    parent_agent_id=self.agent_id
                )
                
                result = await asyncio.wait_for(
                    subagent.run(context),
                    timeout=self.config.subagent_timeout
                )
                results.append(result)
                
            except Exception as e:
                failed = AgentResult(
                    agent_id=str(uuid.uuid4()),
                    status=AgentState.FAILED,
                    error=str(e),
                    start_time=self._result.start_time
                )
                results.append(failed)
                
        return results
        
    async def _synthesize_with_llm(
        self, 
        decomposition: ResearchDecomposition,
        results: List[AgentResult]
    ) -> ResearchSynthesis:
        """Use LLM to synthesize research findings."""
        # Prepare findings summary
        findings_text = []
        for i, (task, result) in enumerate(zip(decomposition.subagent_tasks, results)):
            if result.status == AgentState.COMPLETED:
                findings_text.append(f"""
                Task {i+1}: {task.objective}
                Status: Completed
                Findings: {result.output}
                """)
            else:
                findings_text.append(f"""
                Task {i+1}: {task.objective}
                Status: Failed - {result.error}
                """)
                
        synthesis_prompt = f"""
        You are synthesizing research findings from multiple subagents.
        
        Original Query: {decomposition.original_query}
        Main Objective: {decomposition.main_objective}
        
        Synthesis Instructions: {decomposition.synthesis_instructions}
        
        Subagent Findings:
        {''.join(findings_text)}
        
        Create a comprehensive synthesis that:
        1. Identifies key patterns and connections across findings
        2. Highlights the most important discoveries
        3. Notes any gaps or contradictions
        4. Provides actionable insights
        """
        
        try:
            synthesis = await self.client.chat.completions.create(
                model=self.config.model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": "You are an expert research synthesizer."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                response_model=ResearchSynthesis,
                max_retries=2
            )
            
            return synthesis
            
        except Exception as e:
            self.add_thinking(f"Synthesis failed: {str(e)}")
            # Return a basic synthesis
            return ResearchSynthesis(
                executive_summary="Failed to synthesize results with LLM",
                main_findings=["Synthesis error occurred"],
                detailed_sections=[],
                connections_found=[],
                gaps_remaining=["Unable to process findings"],
                recommendations=[]
            )
            
    def _format_synthesis(self, synthesis: ResearchSynthesis) -> str:
        """Format synthesis into readable output."""
        output = []
        
        output.append(f"# Research Results\n")
        output.append(f"## Executive Summary\n{synthesis.executive_summary}\n")
        
        output.append(f"## Main Findings")
        for i, finding in enumerate(synthesis.main_findings, 1):
            output.append(f"{i}. {finding}")
            
        if synthesis.detailed_sections:
            output.append(f"\n## Detailed Analysis")
            for section in synthesis.detailed_sections:
                output.append(f"\n### {section.get('title', 'Section')}")
                output.append(section.get('content', ''))
                
        if synthesis.connections_found:
            output.append(f"\n## Patterns and Connections")
            for connection in synthesis.connections_found:
                output.append(f"- {connection}")
                
        if synthesis.recommendations:
            output.append(f"\n## Recommendations")
            for rec in synthesis.recommendations:
                output.append(f"- {rec}")
                
        if synthesis.gaps_remaining:
            output.append(f"\n## Information Gaps")
            for gap in synthesis.gaps_remaining:
                output.append(f"- {gap}")
                
        return "\n".join(output)