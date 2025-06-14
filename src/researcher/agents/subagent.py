"""Research subagent using Instructor for structured search planning."""

from typing import Any, Dict, List, Optional
import asyncio
import instructor
from anthropic import AsyncAnthropic
from pydantic import BaseModel

from src.researcher.agents.base import Agent, AgentContext, ToolCall
from src.researcher.agents.models import (
    SubagentSearchPlan, SearchResultEvaluation, SubagentReport
)
from src.researcher.tools.base import ToolRegistry, ToolStatus


class SubagentConfig(BaseModel):
    """Configuration for research subagent."""
    max_search_iterations: int = 3
    max_parallel_searches: int = 3
    anthropic_api_key: Optional[str] = None
    model: str = "claude-3-5-sonnet-20241022"
    enable_thinking: bool = True


class ResearchSubagent(Agent[str]):
    """Subagent that performs iterative research with structured planning."""
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        config: Optional[SubagentConfig] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tool_registry = tool_registry
        self.config = config or SubagentConfig()
        
        # Initialize Anthropic client with Instructor
        client = AsyncAnthropic(api_key=self.config.anthropic_api_key)
        self.client = instructor.from_anthropic(client)
        
        self.search_results: List[Dict[str, Any]] = []
        self.iterations_completed = 0
        
    async def plan(self, context: AgentContext) -> List[Dict[str, Any]]:
        """Create structured search plan using LLM."""
        self.add_thinking(f"Creating search plan for: {context.objective}")
        
        # Extract metadata from context
        focus_areas = context.metadata.get("focus_areas", [])
        expected_findings = context.metadata.get("expected_findings", "")
        
        plan_prompt = f"""
        You are a research subagent. Create a detailed search plan.
        
        Objective: {context.objective}
        Initial Queries: {context.query}
        Focus Areas: {', '.join(focus_areas) if focus_areas else 'General research'}
        Expected Findings: {expected_findings}
        Constraints: {', '.join(context.constraints) if context.constraints else 'None'}
        
        Design a search strategy that:
        1. Starts broad then narrows based on findings
        2. Uses diverse query formulations
        3. Targets the specific focus areas
        4. Can be completed in {self.config.max_search_iterations} iterations
        """
        
        try:
            search_plan = await self.client.chat.completions.create(
                model=self.config.model,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": "You are an expert research agent."},
                    {"role": "user", "content": plan_prompt}
                ],
                response_model=SubagentSearchPlan,
                max_retries=2
            )
            
            self.add_thinking(f"Created plan with {len(search_plan.refined_queries)} queries")
            
            return [{
                "action": "iterative_search",
                "plan": search_plan,
                "context": context
            }]
            
        except Exception as e:
            self.add_thinking(f"Failed to create search plan: {str(e)}")
            # Fallback to simple search
            return [{
                "action": "simple_search",
                "queries": [context.query],
                "context": context
            }]
            
    async def execute(self, plan: List[Dict[str, Any]]) -> str:
        """Execute iterative search based on plan."""
        action = plan[0]
        
        if action["action"] == "iterative_search":
            return await self._execute_iterative_search(
                action["plan"],
                action["context"]
            )
        else:
            return await self._execute_simple_search(
                action["queries"],
                action["context"]
            )
            
    async def _execute_iterative_search(
        self,
        search_plan: SubagentSearchPlan,
        context: AgentContext
    ) -> str:
        """Execute iterative search with evaluation and refinement."""
        all_findings = []
        queries_used = search_plan.refined_queries.copy()
        
        for iteration in range(self.config.max_search_iterations):
            self.add_thinking(f"Starting search iteration {iteration + 1}")
            
            # Execute searches in parallel
            batch_size = min(len(queries_used), self.config.max_parallel_searches)
            current_queries = queries_used[:batch_size]
            queries_used = queries_used[batch_size:]
            
            if not current_queries:
                break
                
            # Search with current queries
            results = await self._execute_parallel_searches(current_queries)
            all_findings.extend(results)
            
            # Evaluate results and decide next steps
            evaluation = await self._evaluate_results(
                results,
                context,
                search_plan.evaluation_criteria
            )
            
            self.add_thinking(
                f"Iteration {iteration + 1} - Relevance: {evaluation.relevance_score:.2f}"
            )
            
            # Check if we have sufficient information
            if evaluation.sufficient_information:
                self.add_thinking("Sufficient information gathered")
                break
                
            # Add new queries for next iteration
            if evaluation.next_queries:
                queries_used.extend(evaluation.next_queries[:self.config.max_parallel_searches])
                
            self.iterations_completed = iteration + 1
            
        # Generate final report
        report = await self._generate_report(all_findings, context)
        return self._format_report(report)
        
    async def _execute_simple_search(
        self,
        queries: List[str],
        context: AgentContext
    ) -> str:
        """Fallback simple search execution."""
        results = await self._execute_parallel_searches(queries)
        
        # Simple aggregation of results
        findings = []
        for query, result in zip(queries, results):
            if result.get("status") == "success":
                findings.append({
                    "query": query,
                    "results": result.get("data", [])
                })
                
        return f"Found {len(findings)} results for queries: {', '.join(queries)}"
        
    async def _execute_parallel_searches(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Execute multiple searches in parallel."""
        search_tool = self.tool_registry.get_tool("web_search")
        if not search_tool:
            self.add_thinking("No search tool available")
            return []
            
        # Create search tasks
        tasks = []
        for query in queries:
            task = search_tool.execute(query=query, limit=10)
            tasks.append(task)
            
        # Execute searches
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                processed_results.append({
                    "query": query,
                    "status": "error",
                    "error": str(result)
                })
            else:
                # Record tool call
                tool_call = ToolCall(
                    tool_name="web_search",
                    parameters={"query": query},
                    result=result.data if result.status == ToolStatus.SUCCESS else None,
                    error=result.error if result.status != ToolStatus.SUCCESS else None
                )
                self.record_tool_call(tool_call)
                
                processed_results.append({
                    "query": query,
                    "status": "success" if result.status == ToolStatus.SUCCESS else "failed",
                    "data": result.data if result.status == ToolStatus.SUCCESS else [],
                    "error": result.error
                })
                
        return processed_results
        
    async def _evaluate_results(
        self,
        results: List[Dict[str, Any]],
        context: AgentContext,
        criteria: List[str]
    ) -> SearchResultEvaluation:
        """Evaluate search results using LLM."""
        # Prepare results summary
        results_text = []
        for result in results:
            if result["status"] == "success":
                results_text.append(f"Query: {result['query']}")
                for item in result.get("data", [])[:3]:  # First 3 results
                    results_text.append(f"- {item.get('title', 'No title')}")
                    results_text.append(f"  {item.get('snippet', 'No snippet')}")
                    
        eval_prompt = f"""
        Evaluate these search results for the research objective.
        
        Objective: {context.objective}
        Focus Areas: {context.metadata.get('focus_areas', [])}
        
        Search Results:
        {chr(10).join(results_text)}
        
        Evaluation Criteria:
        {chr(10).join(f'- {c}' for c in criteria)}
        
        Assess:
        1. How relevant are these results to the objective?
        2. What key findings can be extracted?
        3. What information gaps remain?
        4. What queries would help fill those gaps?
        5. Is there sufficient information to complete the research?
        """
        
        try:
            evaluation = await self.client.chat.completions.create(
                model=self.config.model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": "You are evaluating research results."},
                    {"role": "user", "content": eval_prompt}
                ],
                response_model=SearchResultEvaluation,
                max_retries=2
            )
            
            return evaluation
            
        except Exception as e:
            # Fallback evaluation
            return SearchResultEvaluation(
                relevance_score=0.5,
                key_findings=["Evaluation failed"],
                gaps_identified=["Unable to evaluate"],
                next_queries=[],
                sufficient_information=False
            )
            
    async def _generate_report(
        self,
        all_findings: List[Dict[str, Any]],
        context: AgentContext
    ) -> SubagentReport:
        """Generate structured report from findings."""
        # Aggregate successful results
        findings_text = []
        sources = set()
        
        for finding in all_findings:
            if finding["status"] == "success":
                for item in finding.get("data", []):
                    findings_text.append({
                        "title": item.get("title", ""),
                        "content": item.get("snippet", ""),
                        "source": item.get("url", "")
                    })
                    if item.get("url"):
                        sources.add(item.get("url"))
                        
        report_prompt = f"""
        Create a comprehensive research report.
        
        Research Objective: {context.objective}
        Focus Areas: {context.metadata.get('focus_areas', [])}
        Expected Findings: {context.metadata.get('expected_findings', '')}
        
        Raw Findings: {len(findings_text)} items from {len(sources)} sources
        
        Synthesize these findings into:
        1. A clear task summary
        2. Key insights discovered
        3. Confidence level in the findings
        4. Any limitations or gaps
        
        Be specific and cite sources where possible.
        """
        
        try:
            # Include sample findings in prompt
            sample_findings = findings_text[:10]  # First 10 findings
            
            report = await self.client.chat.completions.create(
                model=self.config.model,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": "You are creating a research report."},
                    {"role": "user", "content": report_prompt},
                    {"role": "assistant", "content": f"I'll analyze these findings: {sample_findings}"}
                ],
                response_model=SubagentReport,
                max_retries=2
            )
            
            # Add actual findings and sources
            report.findings = findings_text[:20]  # Limit to 20 findings
            report.sources_used = list(sources)[:10]  # Limit to 10 sources
            
            return report
            
        except Exception as e:
            # Fallback report
            return SubagentReport(
                task_summary=f"Research on: {context.objective}",
                findings=findings_text[:10],
                key_insights=[f"Found {len(findings_text)} results"],
                sources_used=list(sources)[:10],
                confidence_level="low",
                limitations=["Report generation failed"]
            )
            
    def _format_report(self, report: SubagentReport) -> str:
        """Format report as readable text."""
        output = []
        
        output.append(f"## Task Summary")
        output.append(report.task_summary)
        
        output.append(f"\n## Key Insights")
        for i, insight in enumerate(report.key_insights, 1):
            output.append(f"{i}. {insight}")
            
        output.append(f"\n## Findings")
        for finding in report.findings[:5]:  # Top 5 findings
            output.append(f"- **{finding.get('title', 'Finding')}**")
            output.append(f"  {finding.get('content', '')}")
            if finding.get('source'):
                output.append(f"  Source: {finding.get('source')}")
                
        output.append(f"\n## Confidence Level: {report.confidence_level}")
        
        if report.limitations:
            output.append(f"\n## Limitations")
            for limitation in report.limitations:
                output.append(f"- {limitation}")
                
        output.append(f"\n## Sources ({len(report.sources_used)})")
        for source in report.sources_used[:5]:
            output.append(f"- {source}")
            
        output.append(f"\n## Metrics")
        output.append(f"- Search iterations: {self.iterations_completed + 1}")
        output.append(f"- Total findings: {len(report.findings)}")
        
        return "\n".join(output)