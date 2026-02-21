# src/extensions/agents/master_agent.py
"""Master Agent - orchestrates research and data analysis agents."""
import uuid
import asyncio
from typing import Dict, Any
from datetime import datetime
import os

from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool

# Import our agents
from extensions.agents.research_wrapper import ResearchAgent
from extensions.agents.data_analysis_agent import DataAnalysisAgent

# Import research enhancement utility
from extensions.utils.research_enhancement import EnhancedResearchWrapper

# Import state manager and models
from extensions.utils.state_manager import StateManager
from extensions.models.extended_state import MasterAgentState

# Import LLM factory
from extensions.utils.llm_factory import get_extensions_llm


class MasterAgent:
    """Master agent that orchestrates research and data analysis.

    This agent decides which specialized agents to call based on the query.
    When research output is available and analysis is requested, the data
    analysis agent uses the enforced sequential pipeline:
        extract -> profile -> plan charts -> create charts -> outliers
    """

    def __init__(self, enable_state_persistence: bool = True, storage_type: str = "sqlite",
                 use_enhanced_research: bool = False, provider: str = None, model: str = None):
        """Initialize master agent.

        Args:
            enable_state_persistence: Enable state saving to database
            storage_type: Type of storage ('sqlite', 'memory')
            use_enhanced_research: Use enhanced research (3-4x more comprehensive, but slower)
            provider: LLM provider override. Falls back to env vars.
            model: LLM model override. Falls back to env vars.
        """
        self.name = "master"

        # Initialize sub-agents - choose research mode
        if use_enhanced_research:
            print("[MasterAgent] Using Enhanced Research Mode (3-4x more comprehensive)")
            self.research_agent = EnhancedResearchWrapper(provider=provider, model=model)
        else:
            self.research_agent = ResearchAgent(provider=provider, model=model)

        self.data_agent = DataAnalysisAgent(provider=provider, model=model)

        # Initialize state manager (optional)
        self.state_manager = StateManager(storage_type=storage_type) if enable_state_persistence else None

        # Create agent-as-tools
        self.tools = [
            self._create_research_tool(),
            self._create_data_analysis_tool(),
        ]

        # Create orchestrator LLM using factory (provider-agnostic)
        self.llm = get_extensions_llm(provider=provider, model=model, temperature=0.7)

        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a master research and analysis coordinator.

You have access to the following specialized agents:

1. **deep_research**: Conducts comprehensive research on topics using web search.
   - Use for: finding information, researching topics, gathering facts.
   - Returns: very detailed research report with sources.

2. **data_analysis**: Analyzes data, creates visualizations, performs calculations.
   - Use for: profiling data, creating charts, extracting data, math operations.
   - When research has been completed, this tool will automatically extract tables
     from the research, profile the data, and create appropriate visualizations.
   - Returns: analysis summary, chart files, extracted data, and data profile.

**Decision Logic:**
- Query about research/information only -> Use deep_research.
- Query about data/analysis/visualization only -> Use data_analysis (with actual data).
- Query needs both research AND visualizations -> Call deep_research FIRST, then call data_analysis.

**RESEARCH-TO-VISUALIZATION PIPELINE:**
When the user asks for research WITH charts/plots/visualizations:

Step 1: Call deep_research to get comprehensive findings.
Step 2: Call data_analysis. The tool will automatically:
   - Extract all quantitative data from the research (tables, numbers, statistics)
   - Profile the extracted data (statistics, distributions)
   - Plan and create appropriate Plotly visualizations
   - Detect outliers in numeric columns

**FOLLOW-UP / CONVERSATIONAL QUERIES:**
If the input includes a "=== PREVIOUS CONVERSATION CONTEXT ===" block, it contains
results from earlier queries in this session (research reports, extracted data,
data profiles, chart explanations, sources). Use this context to:
- Answer follow-up questions DIRECTLY without re-running tools when the answer
  is already in the prior context (e.g., "what did the bar chart show?",
  "summarize just the data for India", "which sources mentioned AI?").
- Only call deep_research or data_analysis again if the user explicitly asks
  for NEW research or NEW analysis that is not covered by prior context.

**STRICT RULES -- NEVER VIOLATE:**
1. NEVER invent, fabricate, or hallucinate chart file paths. Only use paths that are explicitly returned by the data_analysis tool.
2. ALWAYS use the data_analysis tool to create charts. NEVER try to create charts yourself.
3. Charts created by data_analysis are ALWAYS saved as interactive Plotly HTML files at outputs/charts/<type>_<id>.html.

Always explain your reasoning and provide clear results."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # Create agent
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
        )

    def _create_research_tool(self) -> StructuredTool:
        """Wrap research agent as a structured tool with both sync and async implementations."""

        class ResearchInput(BaseModel):
            query: str = Field(description="Research question/topic")

        def _store_research_results(result: dict, state_ref: dict):
            """Helper to store research results in state."""
            if result.get("status") == "completed":
                state_ref["final_report"] = result.get("output", "")
                state_ref["sources"] = result.get("sources", [])
                if result.get("sub_queries_used"):
                    state_ref["sub_queries"] = result.get("sub_queries_used", [])
                sources_text = []
                for i, source in enumerate(result.get("sources", []), 1):
                    sources_text.append(f"{i}. [{source}]({source})")
                state_ref["sources_text"] = sources_text

        def research_func_sync(query: str) -> str:
            result = self.research_agent.run(query)
            if not hasattr(self, '_current_state'):
                self._current_state = {}
            _store_research_results(result, self._current_state)

            if result["status"] == "completed":
                sources = result.get("sources", [])
                return (
                    "Research completed!\n\n"
                    f"Report:\n{result['output']}\n\n"
                    f"Sources ({len(sources)}): {', '.join(sources[:3])}" +
                    (f"... and {len(sources)-3} more" if len(sources) > 3 else "")
                )
            else:
                return f"Research failed: {result['error']}"

        async def research_func_async(query: str) -> str:
            result = await self.research_agent.run_async(query)
            if not hasattr(self, '_current_state'):
                self._current_state = {}
            _store_research_results(result, self._current_state)

            if result["status"] == "completed":
                sources = result.get("sources", [])
                return (
                    "Research completed!\n\n"
                    f"Report:\n{result['output']}\n\n"
                    f"Sources ({len(sources)}): {', '.join(sources[:3])}" +
                    (f"... and {len(sources)-3} more" if len(sources) > 3 else "")
                )
            else:
                return f"Research failed: {result['error']}"

        return StructuredTool(
            name="deep_research",
            description="Conduct comprehensive research on a topic using web search. Returns detailed research report with sources.",
            func=research_func_sync,
            coroutine=research_func_async,
            args_schema=ResearchInput,
        )

    def _create_data_analysis_tool(self) -> StructuredTool:
        """Wrap data analysis agent as a structured tool.

        When research output is available (from a prior deep_research call),
        this tool uses the enforced pipeline: extract -> profile -> charts -> outliers.
        Otherwise it uses free-form mode.
        """

        class AnalysisInput(BaseModel):
            query: str = Field(description="Data analysis / visualization / math task")

        def _store_analysis_results(result: dict, state_ref: dict):
            """Helper to store analysis results in state."""
            if result.get("status") == "completed":
                state_ref["analysis_output"] = result.get("output", "")
                state_ref["charts"] = result.get("charts", [])
                state_ref["chart_explanations"] = result.get("chart_explanations", {})
                # Store pipeline-specific outputs
                if result.get("extracted_data"):
                    state_ref["extracted_data"] = result.get("extracted_data", "")
                if result.get("data_profile"):
                    state_ref["data_profile"] = result.get("data_profile", "")

        def analysis_func_sync(query: str) -> str:
            # Prevent regeneration if charts already exist
            if hasattr(self, '_current_state') and self._current_state.get("charts"):
                print("[MasterAgent] WARNING: data_analysis called again. Charts already exist. Returning cached.")
                return self._current_state.get("analysis_output", "Analysis already completed.")

            if not hasattr(self, '_current_state'):
                self._current_state = {}

            # If research output is available, use the enforced pipeline
            if self._current_state.get("final_report"):
                research_context = self._current_state["final_report"]
                print(f"[MasterAgent] Running analysis pipeline on {len(research_context)} chars of research...")
                result = self.data_agent.run_pipeline(research_context)
            else:
                # No research context -- use free-form mode
                print("[MasterAgent] No research context. Using free-form analysis mode.")
                result = self.data_agent.run(query)

            _store_analysis_results(result, self._current_state)
            print(f"[MasterAgent] Analysis complete. Charts: {self._current_state.get('charts', [])}")

            if result["status"] == "completed":
                return result["output"]
            else:
                return f"Analysis failed: {result['error']}"

        async def analysis_func_async(query: str) -> str:
            # Prevent regeneration if charts already exist
            if hasattr(self, '_current_state') and self._current_state.get("charts"):
                print("[MasterAgent] WARNING: data_analysis called again. Charts already exist. Returning cached.")
                return self._current_state.get("analysis_output", "Analysis already completed.")

            if not hasattr(self, '_current_state'):
                self._current_state = {}

            # If research output is available, use the enforced pipeline
            if self._current_state.get("final_report"):
                research_context = self._current_state["final_report"]
                print(f"[MasterAgent] Running analysis pipeline on {len(research_context)} chars of research...")
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, self.data_agent.run_pipeline, research_context)
            else:
                # No research context -- use free-form mode
                print("[MasterAgent] No research context. Using free-form analysis mode.")
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, self.data_agent.run, query)

            _store_analysis_results(result, self._current_state)
            print(f"[MasterAgent] Analysis complete. Charts: {self._current_state.get('charts', [])}")

            if result["status"] == "completed":
                return result["output"]
            else:
                return f"Analysis failed: {result['error']}"

        return StructuredTool(
            name="data_analysis",
            description="Analyze data, create visualizations (bar, line, scatter, pie, histogram, box, heatmap, density, bubble, violin), detect outliers (IQR, Z-score, Isolation Forest), extract structured data, or perform calculations. When research has been completed, automatically extracts data and creates appropriate visualizations.",
            func=analysis_func_sync,
            coroutine=analysis_func_async,
            args_schema=AnalysisInput,
        )

    async def run_async(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Run master agent asynchronously."""
        conversation_id = str(uuid.uuid4())
        start_time = datetime.now()

        self._current_state = {
            "query": query
        }

        state: MasterAgentState = {
            "conversation_id": conversation_id,
            "query": query,
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "agents_used": [],
        }

        if session_id:
            state["session_id"] = session_id

        if self.state_manager:
            self.state_manager.save_state(conversation_id, state)

        # Inject prior session context so the LLM can answer follow-up
        # questions about previous research, data, charts, and analysis.
        effective_input = query
        if session_id and self.state_manager:
            prior_context = self.state_manager.build_session_context(session_id)
            if prior_context:
                effective_input = f"{prior_context}\n\nCurrent query: {query}"

        try:
            result = await self.agent_executor.ainvoke({"input": effective_input})

            output = result.get('output', '')

            # Merge tracked state from tools
            if hasattr(self, '_current_state'):
                for key in ['final_report', 'sources', 'sources_text', 'sub_queries',
                           'analysis_output', 'charts', 'chart_explanations',
                           'extracted_data', 'data_profile',
                           'report_html']:
                    if key in self._current_state:
                        state[key] = self._current_state[key]

            if state.get("final_report") or state.get("sources"):
                state["agents_used"].append("research")
            if state.get("analysis_output") or state.get("charts"):
                state["agents_used"].append("data_analysis")

            state["status"] = "completed"
            state["execution_time"] = (datetime.now() - start_time).total_seconds()

            if self.state_manager:
                self.state_manager.save_state(conversation_id, state)

            if hasattr(self, '_current_state'):
                delattr(self, '_current_state')

            return {
                "conversation_id": conversation_id,
                "status": "completed",
                "output": output,
                "state": state,
                "agents_used": state.get("agents_used", []),
                "execution_time": state.get("execution_time", 0),
                "error": None,
            }

        except Exception as e:
            state["status"] = "error"
            state["error"] = str(e)
            state["execution_time"] = (datetime.now() - start_time).total_seconds()

            if self.state_manager:
                self.state_manager.save_state(conversation_id, state)

            if hasattr(self, '_current_state'):
                delattr(self, '_current_state')

            return {
                "conversation_id": conversation_id,
                "status": "error",
                "output": "",
                "state": state,
                "agents_used": state.get("agents_used", []),
                "execution_time": state.get("execution_time", 0),
                "error": str(e),
            }

    def run(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Synchronous convenience wrapper. Prefer run_async in notebooks."""
        return asyncio.run(self.run_async(query, session_id=session_id))

    def get_state(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation state."""
        if not self.state_manager:
            return None
        return self.state_manager.get_state(conversation_id)

    def list_conversations(self, limit: int = 50) -> list:
        """List recent conversations."""
        if not self.state_manager:
            return []
        return self.state_manager.list_conversations(limit)


__all__ = ['MasterAgent']
