"""Lead researcher agent that orchestrates the research process."""

from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass
import asyncio
import uuid
from pydantic import BaseModel, Field

from src.researcher.agents.base import Agent, AgentContext, AgentResult, AgentState
from src.researcher.memory.base import ResearchMemory, ResearchPlan, IntermediateResult
from src.researcher.tools.base import ToolRegistry


@dataclass
class SubagentTask:
    """Task definition for a subagent."""
    task_id: str
    objective: str
    search_focus: str
    constraints: List[str]
    tools_to_use: List[str]
    expected_output: str
    

class LeadResearcherConfig(BaseModel):
    """Configuration for lead researcher."""
    max_subagents: int = 5
    max_iterations: int = 3
    parallel_execution: bool = True
    context_window_limit: int = 200000
    subagent_timeout: float = 300.0  # 5 minutes per subagent
    

class QueryComplexity(str):
    """Query complexity levels."""
    SIMPLE = "simple"  # Single fact-finding
    MODERATE = "moderate"  # Comparison or multi-aspect
    COMPLEX = "complex"  # Open-ended research
    

class LeadResearcher(Agent[str]):
    """Orchestrator agent that manages the research process."""
    
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
        self.subagents: List[Agent] = []
        
    async def plan(self, context: AgentContext) -> List[Dict[str, Any]]:
        """Analyze query and create research plan."""
        self.add_thinking(f"Analyzing query: {context.query}")
        
        # Determine query complexity
        complexity = self._assess_complexity(context.query)
        self.add_thinking(f"Query complexity assessed as: {complexity}")
        
        # Create research plan
        plan = ResearchPlan(
            query=context.query,
            objective=context.objective,
            approach=self._determine_approach(complexity),
            constraints=context.constraints
        )
        
        # Save plan to memory
        await self.memory.save_research_plan(self.agent_id, plan)
        
        # Decompose into subagent tasks
        tasks = self._create_subagent_tasks(context, complexity)
        plan.subagent_tasks = [self._task_to_dict(t) for t in tasks]
        
        self.add_thinking(f"Created {len(tasks)} subagent tasks")
        
        return [{"action": "execute_research", "tasks": tasks}]
        
    async def execute(self, plan: List[Dict[str, Any]]) -> str:
        """Execute research plan with subagents."""
        action = plan[0]
        tasks = action["tasks"]
        
        # Create subagents
        subagent_results = []
        
        if self.config.parallel_execution:
            # Execute subagents in parallel
            self.add_thinking("Executing subagents in parallel")
            subagent_results = await self._execute_parallel(tasks)
        else:
            # Execute sequentially
            self.add_thinking("Executing subagents sequentially")
            subagent_results = await self._execute_sequential(tasks)
            
        # Store intermediate results
        for result in subagent_results:
            if result.status == AgentState.COMPLETED and result.output:
                intermediate = IntermediateResult(
                    subagent_id=result.agent_id,
                    task_description=f"Task {result.agent_id}",
                    findings=[{"content": result.output}],
                    sources=[],
                    tokens_used=result.tokens_used
                )
                await self.memory.add_intermediate_result(self.agent_id, intermediate)
                
        # Synthesize results
        self.add_thinking("Synthesizing research findings")
        synthesis = self._synthesize_results(subagent_results)
        
        return synthesis
        
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity based on heuristics."""
        query_lower = query.lower()
        
        # Simple indicators
        if any(word in query_lower for word in ["what is", "define", "when did"]):
            return QueryComplexity.SIMPLE
            
        # Complex indicators  
        if any(word in query_lower for word in ["compare", "analyze", "evaluate", "all"]):
            return QueryComplexity.COMPLEX
            
        # Word count heuristic
        if len(query.split()) > 20:
            return QueryComplexity.COMPLEX
            
        return QueryComplexity.MODERATE
        
    def _determine_approach(self, complexity: str) -> List[str]:
        """Determine research approach based on complexity."""
        if complexity == QueryComplexity.SIMPLE:
            return ["Direct search for specific information"]
        elif complexity == QueryComplexity.MODERATE:
            return [
                "Search for multiple aspects",
                "Compare and contrast findings",
                "Verify from multiple sources"
            ]
        else:
            return [
                "Broad initial exploration",
                "Identify key themes and subtopics", 
                "Deep dive into specific areas",
                "Synthesize comprehensive findings"
            ]
            
    def _create_subagent_tasks(self, context: AgentContext, 
                               complexity: str) -> List[SubagentTask]:
        """Create tasks for subagents based on query."""
        tasks = []
        
        # Determine number of subagents
        if complexity == QueryComplexity.SIMPLE:
            num_agents = 1
        elif complexity == QueryComplexity.MODERATE:
            num_agents = min(3, self.config.max_subagents)
        else:
            num_agents = self.config.max_subagents
            
        # Create diverse tasks
        # This is a simplified version - in production, would use LLM for decomposition
        for i in range(num_agents):
            task = SubagentTask(
                task_id=str(uuid.uuid4()),
                objective=f"Research aspect {i+1} of: {context.query}",
                search_focus=self._get_search_focus(context.query, i),
                constraints=context.constraints,
                tools_to_use=["web_search"],  # Could be expanded
                expected_output="Detailed findings with sources"
            )
            tasks.append(task)
            
        return tasks
        
    def _get_search_focus(self, query: str, index: int) -> str:
        """Generate search focus for subagent."""
        # Simplified - in production would use LLM
        focuses = [
            "recent developments and news",
            "technical details and implementations",
            "comparisons and alternatives",
            "best practices and recommendations",
            "case studies and examples"
        ]
        return focuses[index % len(focuses)]
        
    def _task_to_dict(self, task: SubagentTask) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": task.task_id,
            "objective": task.objective,
            "search_focus": task.search_focus,
            "constraints": task.constraints,
            "tools": task.tools_to_use,
            "expected_output": task.expected_output
        }
        
    async def _execute_parallel(self, tasks: List[SubagentTask]) -> List[AgentResult]:
        """Execute subagents in parallel."""
        coroutines = []
        
        for task in tasks:
            subagent = self.subagent_class(
                agent_id=task.task_id,
                name=f"Subagent-{task.task_id[:8]}"
            )
            self.subagents.append(subagent)
            
            context = AgentContext(
                query=task.search_focus,
                objective=task.objective,
                constraints=task.constraints,
                parent_agent_id=self.agent_id
            )
            
            # Add timeout wrapper
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
                # Create failed result
                failed = AgentResult(
                    agent_id=tasks[i].task_id,
                    status=AgentState.FAILED,
                    error=str(result),
                    start_time=self._result.start_time
                )
                final_results.append(failed)
            else:
                final_results.append(result)
                
        return final_results
        
    async def _execute_sequential(self, tasks: List[SubagentTask]) -> List[AgentResult]:
        """Execute subagents sequentially."""
        results = []
        
        for task in tasks:
            try:
                subagent = self.subagent_class(
                    agent_id=task.task_id,
                    name=f"Subagent-{task.task_id[:8]}"
                )
                self.subagents.append(subagent)
                
                context = AgentContext(
                    query=task.search_focus,
                    objective=task.objective,
                    constraints=task.constraints,
                    parent_agent_id=self.agent_id
                )
                
                result = await asyncio.wait_for(
                    subagent.run(context),
                    timeout=self.config.subagent_timeout
                )
                results.append(result)
                
            except Exception as e:
                # Create failed result
                failed = AgentResult(
                    agent_id=task.task_id,
                    status=AgentState.FAILED,
                    error=str(e),
                    start_time=self._result.start_time
                )
                results.append(failed)
                
        return results
        
    def _synthesize_results(self, results: List[AgentResult]) -> str:
        """Synthesize findings from all subagents."""
        # Collect successful results
        findings = []
        total_tokens = 0
        
        for result in results:
            if result.status == AgentState.COMPLETED and result.output:
                findings.append(f"[{result.agent_id[:8]}]: {result.output}")
                total_tokens += result.tokens_used
                
        if not findings:
            return "No results were successfully gathered."
            
        # Simple synthesis - in production would use LLM
        synthesis = f"Research completed with {len(findings)} successful subagents.\n\n"
        synthesis += "Key Findings:\n"
        synthesis += "\n\n".join(findings)
        synthesis += f"\n\nTotal tokens used: {total_tokens}"
        
        return synthesis