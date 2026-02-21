import operator
from dataclasses import dataclass, field
from typing_extensions import TypedDict, Annotated
from typing import List, Dict, Optional, Any, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
import json
import asyncio


def replace_list(old_list, new_list):
    """Custom reducer that replaces the old list with the new list completely.
    This allows us to completely clear or replace lists in the state.
    """
    # If the new list is explicitly set to an empty list, we want to clear the data
    return new_list


class SummaryState(BaseModel):
    """
    State for the research summary graph.
    """

    research_topic: str = Field(description="The main research topic")
    search_query: str = Field(default="", description="Current search query")
    running_summary: str = Field(default="", description="Running summary of research")
    research_complete: bool = Field(
        default=False, description="Whether research is complete"
    )
    knowledge_gap: str = Field(default="", description="Identified knowledge gaps")
    research_loop_count: int = Field(
        default=0, description="Number of research loops completed"
    )
    sources_gathered: List[str] = Field(
        default_factory=list, description="List of sources gathered"
    )
    web_research_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Web research results"
    )
    search_results_empty: bool = Field(
        default=False, description="Whether search results were empty"
    )
    selected_search_tool: str = Field(
        default="general_search", description="Selected search tool"
    )
    source_citations: Dict[str, Any] = Field(
        default_factory=dict, description="Source citations"
    )
    subtopic_queries: List[str] = Field(
        default_factory=list, description="Subtopic queries"
    )
    subtopics_metadata: List[Dict[str, Any]] = Field(
        default_factory=list, description="Subtopics metadata"
    )
    research_plan: Optional[Dict[str, Any]] = Field(
        default=None, description="Current research plan with decomposition data"
    )
    # Reflection metadata fields (for trajectory capture)
    priority_section: Optional[str] = Field(
        default=None, description="Priority section from reflection (for logging)"
    )
    section_gaps: Optional[Dict[str, str]] = Field(
        default=None, description="Section gaps from reflection (for logging)"
    )
    evaluation_notes: Optional[str] = Field(
        default=None, description="Evaluation notes from reflection (for logging)"
    )
    # Tool call logging (for trajectory capture)
    tool_calls_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Log of all tool calls made during research (for trajectory)",
    )
    # Complete execution trace (for trajectory capture)
    execution_trace: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Complete chronological trace of all LLM calls and tool executions",
    )
    extra_effort: bool = Field(default=False, description="Whether to use extra effort")
    minimum_effort: bool = Field(
        default=False, description="Whether to use minimum effort"
    )
    qa_mode: bool = Field(
        default=False, description="Whether in QA mode (simple question-answering)"
    )
    benchmark_mode: bool = Field(
        default=False,
        description="Whether in benchmark mode (with full citation processing)",
    )

    llm_provider: Optional[str] = Field(default=None, description="LLM provider")
    llm_model: Optional[str] = Field(default=None, description="LLM model")
    uploaded_knowledge: Optional[str] = Field(
        default=None, description="User-uploaded external knowledge"
    )
    uploaded_files: List[str] = Field(
        default_factory=list, description="List of uploaded file IDs"
    )
    analyzed_files: List[Dict[str, Any]] = Field(
        default_factory=list, description="Analysis results from uploaded files"
    )

    # Additional fields for enhanced functionality
    formatted_sources: List[Dict[str, Any]] = Field(
        default_factory=list, description="Formatted sources for UI"
    )
    useful_information: str = Field(
        default="", description="Useful information extracted"
    )
    missing_information: str = Field(
        default="", description="Missing information identified"
    )
    needs_refinement: bool = Field(
        default=False, description="Whether query needs refinement"
    )
    current_refined_query: str = Field(default="", description="Current refined query")
    refinement_reasoning: str = Field(
        default="", description="Reasoning for refinement"
    )
    previous_answers: List[str] = Field(
        default_factory=list, description="Previous answers"
    )
    reflection_history: List[str] = Field(
        default_factory=list, description="Reflection history"
    )

    # Visualization fields
    visualization_disabled: bool = Field(
        default=True, description="Whether to have visualizations"
    )

    # Parallel search execution controls
    parallel_search_enabled: bool = Field(
        default=False, description="Whether to run search subtasks in parallel"
    )
    parallel_search_max_concurrency: int = Field(
        default=4, description="Max concurrent search subtasks when parallelized"
    )

    # Database fields for text2sql functionality
    database_info: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Information about uploaded databases for text2sql functionality",
    )

    # Intent detection and analysis pipeline fields
    analysis_required: bool = Field(
        default=False,
        description="Whether user requested data analysis and visualization",
    )
    analysis_output: Optional[str] = Field(
        default=None,
        description="Text summary from DataAnalysisAgent pipeline",
    )
    analysis_report_path: Optional[str] = Field(
        default=None,
        description="File path to generated HTML analysis report",
    )
    analysis_charts: List[str] = Field(
        default_factory=list,
        description="File paths to generated chart HTML files",
    )
    analysis_chart_explanations: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chart metadata: {path: {title, explanation}}",
    )

    def model_post_init(self, __context):
        """Post-initialization hook to log database_info"""
        logger.info(
            f"[STATE_INIT] SummaryState created with database_info: {self.database_info}"
        )
        return super().model_post_init(__context)

    visualizations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Generated visualizations"
    )
    base64_encoded_images: List[str] = Field(
        default_factory=list, description="Base64 encoded images"
    )
    visualization_html: str = Field(
        default="", description="Visualization HTML content"
    )
    visualization_paths: List[str] = Field(
        default_factory=list, description="Paths to visualization files"
    )

    # Code snippet fields
    code_snippets: List[Dict[str, Any]] = Field(
        default_factory=list, description="Generated code snippets"
    )

    # Report format fields
    markdown_report: Optional[str] = Field(
        default="",
        description="Plain markdown version of the report without HTML elements",
    )

    # Benchmark mode fields
    benchmark_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Benchmark result"
    )

    # Configuration
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Configuration settings"
    )

    # Simple Steering System
    steering_enabled: bool = Field(
        default=False, description="Whether steering is enabled for this session"
    )
    steering_todo: Optional[Any] = Field(
        default=None,
        description="Simple todo manager for steering",
        exclude=False,  # Allow in schema but handle gracefully
    )
    pending_steering_messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Queue of pending steering messages to process between research loops",
    )

    def log_tool_call(
        self, tool_name: str, params: Dict[str, Any], result_summary: str = None
    ):
        """
        Log a tool call for trajectory capture (non-invasive).

        Args:
            tool_name: Name of the tool being called
            params: Parameters passed to the tool
            result_summary: Optional summary of the result
        """
        from datetime import datetime

        tool_call_entry = {
            "tool": tool_name,
            "params": params,
            "timestamp": datetime.now().isoformat(),
            "research_loop": self.research_loop_count,
        }
        if result_summary:
            tool_call_entry["result_summary"] = result_summary

        self.tool_calls_log.append(tool_call_entry)

    def log_execution_step(
        self,
        step_type: str,
        action: str,
        input_data: Any = None,
        output_data: Any = None,
        metadata: Dict[str, Any] = None,
    ):
        """
        Log a complete execution step in chronological order.

        CRITICAL: This is for trajectory capture only and should NEVER affect research logic.
        All errors are caught and suppressed to ensure research continues normally.

        Args:
            step_type: Type of step ("llm_call", "tool_execution", "reflection", etc.)
            action: What action was taken (e.g., "decompose_query", "general_search", "reflect")
            input_data: Input to the step (e.g., query, prompt)
            output_data: Output from the step (e.g., decomposition result, search results)
            metadata: Additional metadata (e.g., model used, num_sources)
        """
        try:
            from datetime import datetime

            step_entry = {
                "step_type": step_type,
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "research_loop": self.research_loop_count,
            }

            if input_data is not None:
                step_entry["input"] = input_data
            if output_data is not None:
                # Truncate large outputs to minimize memory/storage impact
                if isinstance(output_data, str) and len(output_data) > 1000:
                    step_entry["output_preview"] = output_data[:1000] + "..."
                    step_entry["output_length"] = len(output_data)
                else:
                    step_entry["output"] = output_data
            if metadata:
                step_entry["metadata"] = metadata

            self.execution_trace.append(step_entry)
        except Exception:
            # Silently fail - trajectory logging should never break research
            pass

    def __init__(self, **data):
        super().__init__(**data)

        # CRITICAL DEBUG: Check what we receive
        logger.info(f"[STATE_INIT] ===== DEBUGGING STATE CREATION =====")
        logger.info(f"[STATE_INIT] data keys: {list(data.keys())}")
        logger.info(
            f"[STATE_INIT] steering_enabled in data: {'steering_enabled' in data}"
        )
        logger.info(
            f"[STATE_INIT] steering_enabled value: {data.get('steering_enabled', 'MISSING')}"
        )
        logger.info(f"[STATE_INIT] ==========================================")

        # Initialize simple steering if enabled
        if data.get("steering_enabled", False):
            from src.simple_steering import ResearchTodoManager, TaskStatus

            topic = data.get("research_topic", "Research")
            logger.info(f"[STEERING] About to create ResearchTodoManager for: {topic}")
            self.steering_todo = ResearchTodoManager(topic)
            logger.info(f"[STEERING] Initialized simple todo manager for: {topic}")
        else:
            logger.info(
                f"[STEERING] NOT initializing steering (steering_enabled={data.get('steering_enabled', False)})"
            )
            self.steering_todo = None

        # Log the uploaded_knowledge when state is created
        if hasattr(self, "uploaded_knowledge") and self.uploaded_knowledge:
            print(f"[UPLOAD_TRACE] SummaryState.__init__: uploaded_knowledge set")
            print(
                f"[UPLOAD_TRACE] SummaryState.__init__: uploaded_knowledge length: {len(self.uploaded_knowledge)}"
            )
            print(
                f"[UPLOAD_TRACE] SummaryState.__init__: uploaded_knowledge preview: {self.uploaded_knowledge[:100]}..."
            )
        else:
            print(
                f"[UPLOAD_TRACE] SummaryState.__init__: uploaded_knowledge not set (value: {getattr(self, 'uploaded_knowledge', 'MISSING_ATTR')})"
            )

    async def add_steering_message(self, message: str) -> Dict[str, Any]:
        """Add a natural language steering message from user (queued like Cursor)"""
        if not self.steering_todo:
            raise ValueError("Steering not enabled for this research session")

        await self.steering_todo.add_user_message(message)

        return {
            "message_queued": True,
            "queue_size": len(self.steering_todo.pending_messages),
            "todo_version": self.steering_todo.todo_version,
            "pending_tasks": len(self.steering_todo.get_pending_tasks()),
        }

    async def prepare_steering_for_next_loop(self) -> Dict[str, Any]:
        """
        SIMPLIFIED: Only fetch messages from API session store.
        The actual todo updates happen in reflect_on_report() via LLM.

        This function just queues messages for the reflection phase.
        """
        if not self.steering_todo:
            return {"steering_enabled": False}

        # Fetch any new messages from the API session store
        await self._fetch_pending_messages_from_session_store()

        # Create initial plan if not done yet
        if not self.steering_todo.initial_plan_created:
            initial_query = getattr(self, "search_query", None) or getattr(
                self, "research_topic", ""
            )
            await self.steering_todo.create_initial_plan(
                initial_query=initial_query,
                research_context=f"Research loop {self.research_loop_count}, summary: {self.running_summary[:500]}...",
            )

        # Log message queue status
        if self.steering_todo.pending_messages:
            logger.info(
                f"[STEERING] Queued {len(self.steering_todo.pending_messages)} messages for reflection phase"
            )

        # Get current plan status
        updated_plan = self.steering_todo.get_todo_md()

        return {
            "steering_enabled": True,
            "todo_updated": False,  # Will be updated in reflect_on_report
            "todo_version": self.steering_todo.todo_version,
            "pending_tasks": len(self.steering_todo.get_pending_tasks()),
            "completed_tasks": len(
                [
                    t
                    for t in self.steering_todo.tasks.values()
                    if t.status.name == "COMPLETED"
                ]
            ),
            "current_plan": updated_plan,
            "loop_guidance": self.steering_todo.get_current_loop_guidance(),
        }

    def get_steering_plan(self) -> str:
        """Get current steering plan in todo.md format"""
        if not self.steering_todo:
            return "# No steering enabled\n\nSteering is not enabled for this research session."

        return self.steering_todo.get_todo_md()

    def should_cancel_search(self, search_query: str) -> bool:
        """Check if search should be cancelled due to steering"""
        if not self.steering_todo:
            return False

        return self.steering_todo.should_cancel_search(search_query)

    def get_search_priority_boost(self, search_query: str) -> int:
        """Get priority boost for search query"""
        if not self.steering_todo:
            return 0

        return self.steering_todo.get_search_priority_boost(search_query)

    async def _fetch_pending_messages_from_session_store(self):
        """Fetch pending messages from the API session store"""
        if not self.steering_todo:
            return

        try:
            # Import here to avoid circular imports
            from routers.simple_steering_api import active_research_sessions

            # Find our session by looking for matching state object
            session_id = None
            for sid, session_info in active_research_sessions.items():
                if session_info.get("state") is self:
                    session_id = sid
                    break

            if not session_id:
                # Try file store if available
                try:
                    from session_store import session_store

                    all_sessions = session_store.get_all_sessions()
                    for sid, session_info in all_sessions.items():
                        if (
                            session_info.get("steering_enabled")
                            and session_info.get("research_topic")
                            == self.research_topic
                        ):
                            session_id = sid
                            break
                except ImportError:
                    pass

            if session_id:
                # Get pending messages from session
                session_info = active_research_sessions.get(session_id)
                if session_info and "pending_messages" in session_info:
                    pending_messages = session_info["pending_messages"]
                    logger.info(
                        f"[STEERING] Found {len(pending_messages)} pending messages in session {session_id}"
                    )

                    # Add messages to todo manager (DON'T mark as processed yet)
                    for msg_data in pending_messages:
                        if not msg_data.get("fetched", False):
                            message_content = msg_data.get("content", "")
                            await self.steering_todo.add_user_message(message_content)
                            msg_data["fetched"] = True  # Mark as fetched, NOT processed
                            logger.info(
                                f"[STEERING] Fetched message for processing: {message_content}"
                            )

                    # DON'T clear messages here - they stay in session until reflect_on_report processes them
                else:
                    logger.info(
                        f"[STEERING] No pending messages found for session {session_id}"
                    )
            else:
                logger.warning(
                    "[STEERING] Could not find session ID for current research state"
                )

        except Exception as e:
            logger.error(f"[STEERING] Error fetching messages from session store: {e}")
            import traceback

            traceback.print_exc()


class SummaryStateInput(BaseModel):
    """
    Input model for the research process.
    """

    research_topic: str
    extra_effort: bool = False
    minimum_effort: bool = False
    qa_mode: bool = Field(
        default=False,
        description="Whether to run in QA mode (simple question-answering)",
    )
    benchmark_mode: bool = Field(
        default=False,
        description="Whether to run in benchmark mode (with full citation processing)",
    )
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    uploaded_knowledge: Optional[str] = None
    uploaded_files: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    # CRITICAL: Include steering fields in input schema so LangGraph doesn't drop them
    steering_enabled: bool = False
    steering_todo: Optional[Any] = None
    # CRITICAL: Include database_info so LangGraph doesn't drop it
    database_info: Optional[List[Dict[str, Any]]] = None
    # CRITICAL: Include analysis_required so LangGraph doesn't drop it
    # (set by intent detector before graph starts; controls route_after_finalize_report)
    analysis_required: bool = False


class SummaryStateOutput(BaseModel):
    """
    Output model for the research process.
    """

    running_summary: str
    research_complete: bool
    research_loop_count: int
    sources_gathered: List[str]
    web_research_results: List[Dict[str, Any]] = []
    source_citations: Dict[str, Dict[str, str]]
    qa_mode: bool = Field(default=False, description="Whether ran in QA mode")
    benchmark_mode: bool = Field(
        default=False, description="Whether ran in benchmark mode"
    )
    benchmark_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Results from benchmark testing"
    )
    visualizations: List[Dict[str, Any]] = []
    base64_encoded_images: List[Dict[str, Any]] = []
    visualization_paths: List[str] = []
    code_snippets: List[Dict[str, Any]] = []
    markdown_report: str = ""
    uploaded_knowledge: Optional[str] = None
    analyzed_files: List[Dict[str, Any]] = []
    # Analysis pipeline output fields (from run_analysis node)
    analysis_required: bool = False
    analysis_output: Optional[str] = None
    analysis_report_path: Optional[str] = None
    analysis_charts: List[str] = []
    analysis_chart_explanations: Dict[str, Any] = {}
