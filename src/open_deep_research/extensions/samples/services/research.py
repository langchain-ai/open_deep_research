import os
import logging
import datetime
import asyncio
import concurrent.futures
import uuid
import json
from typing import Dict, Any, AsyncGenerator, Optional, List, Set, Tuple
import time
from asyncio import Queue
import re
import random

from src.graph import create_graph
from src.state import SummaryState  # Import the SummaryState class for type checking
from src.configuration import Configuration, ActivityVerbosity
from services.activity_generator import (
    ActivityManager,
)  # Import the new ActivityManager class instead of ActivityGenerator

from models.research import ResearchRequest, ResearchResponse, ResearchEvent

logger = logging.getLogger(__name__)

# Import simple steering session management
try:
    from routers.simple_steering_api import (
        register_research_session,
        unregister_research_session,
        update_session_activity,
    )

    STEERING_AVAILABLE = True
except ImportError:
    logger.warning(
        "Simple steering router not available - steering functionality disabled"
    )
    STEERING_AVAILABLE = False

    # Dummy functions if steering not available
    def register_research_session(
        session_id: str, state=None, steering_enabled: bool = False
    ):
        pass

    def unregister_research_session(session_id: str):
        pass

    def update_session_activity(session_id: str):
        pass


class EventBuffer:
    """
    Buffer, filter, and transform LangGraph events into higher-level state updates.
    This reduces the number of events sent to the frontend and prevents overwhelm/delay.
    """

    def __init__(self, callback=None):
        self.callback = callback
        self.current_state = {
            "node_status": {},  # Track active nodes
            "search_sources": set(),  # Track unique sources found
            "knowledge_gaps": set(),  # Track knowledge gaps
            "research_complete": False,
            "report_generating": False,
            "report_improving": False,
            "last_emitted": {},  # Track last emitted state for each category
        }
        self.buffer = []
        self.token_buffer = ""
        self.last_emit_time = {}
        self.last_heartbeat_time = time.time()
        self.events = []
        self.node_end_counter = 0
        self.nodes_reporting_end = set()
        self.current_nodes = set()
        self.last_node = None
        self.step_results = {}
        self.collected_results = {}
        self.report_content = None
        self.updated_report_content = None
        self.has_report = False
        self.report_time = None
        self.report_needs_improvement = False
        self.activity_history = []
        self.last_event_time = None

    def _should_emit_state_update(self, category, min_interval=2.0):
        """Check if enough time has passed since the last emission for this category"""
        current_time = time.time()
        if category not in self.last_emit_time:
            self.last_emit_time[category] = current_time
            return True

        time_since_last = current_time - self.last_emit_time[category]
        if time_since_last >= min_interval:
            self.last_emit_time[category] = current_time
            return True

        return False

    def check_heartbeat(self, current_timestamp=None):
        """Check if we need to send a heartbeat to keep the connection alive"""
        current_time = time.time()
        if (
            current_time - self.last_heartbeat_time > 5.0
        ):  # Send heartbeat every 5 seconds
            self.last_heartbeat_time = current_time
            return {
                "event_type": "heartbeat",
                "data": {"timestamp": current_time},
                "timestamp": current_timestamp or datetime.datetime.now().isoformat(),
            }
        return None

    def process_event(self, event, time_now=None):
        """
        Process an event and update the state.

        Args:
            event: The event to process
            time_now: The current time, used for testing

        Returns:
            A tuple of (done, result)
        """
        if time_now is None:
            time_now = time.time()

        event_type = event.get("type")

        # Special logging for activity_generated events
        if event_type == "activity_generated":
            activity_text = event.get("activity", "No activity text")
            logger.info(f"ACTIVITY EVENT FOUND: {activity_text[:100]}...")

        # Track the event time
        self.last_event_time = time_now

        if event_type == "node_start":
            node_name = event.get("node_name", "unknown")
            self.current_nodes.add(node_name)
            if not self.last_node:
                self.last_node = node_name

            # Update steering session with current node
            if self.stream_id:
                try:
                    from routers.simple_steering_api import active_research_sessions
                    if self.stream_id in active_research_sessions:
                        active_research_sessions[self.stream_id]["current_node"] = node_name
                        logger.info(f"[STEERING] Updated current_node to {node_name} for session {self.stream_id}")
                except Exception as e:
                    logger.debug(f"Could not update session current_node: {e}")

            # Log node start
            logger.info(f"Node started: {node_name}")

        elif event_type == "node_end":
            node_name = event.get("node_name", "unknown")

            # Enhanced logging for node_end events
            logger.info(f"Node ended: {node_name} (Last node: {self.last_node})")

            if node_name in self.current_nodes:
                self.current_nodes.remove(node_name)

            self.nodes_reporting_end.add(node_name)
            self.node_end_counter += 1

            # Is this the last node to finish?
            is_last_node = node_name == self.last_node
            logger.info(f"Is this the last node? {is_last_node}")

            if node_name == "generate_report":
                self.has_report = True
                self.report_time = time_now

                # Set the report content
                self.report_content = event.get("report", "")
                logger.info(
                    f"Report received - length: {len(self.report_content) if self.report_content else 0} chars"
                )

            if node_name == "reflect_on_report":
                reflections = event.get("reflections", None)
                needs_improvement = (
                    reflections
                    and "needs_improvement" in reflections
                    and reflections["needs_improvement"]
                )
                self.report_needs_improvement = needs_improvement
                logger.info(
                    f"Report reflection - needs improvement: {needs_improvement}"
                )

            if node_name == "finalize_report":
                self.report_content = event.get("report", self.report_content)
                logger.info(
                    f"Report finalized - length: {len(self.report_content) if self.report_content else 0} chars"
                )

            if node_name == self.last_node and not self.current_nodes:
                logger.info("Final node completed - research process finished")
                # Done with the research
                return (True, self._get_final_result())

        elif event_type == "activity_generated":
            # Extract activity data from the event
            activity = event.get("activity")
            node_name = event.get("node_name")
            enriched_data = event.get("enriched_data", {})

            # Check if we're getting the new event structure with data nested under 'data'
            if not activity and "data" in event:
                activity = event["data"].get("activity")
                node_name = event["data"].get("node_name")
                enriched_data = event["data"].get("enriched_data", {})

            if activity:
                # Log activity and enriched data for debugging
                logger.info(f"Adding activity to history: {activity[:50]}...")
                if enriched_data:
                    logger.info(f"With enriched data: {list(enriched_data.keys())}")
                    if "domains" in enriched_data:
                        logger.info(
                            f"Domain count: {len(enriched_data.get('domains', []))}"
                        )
                    if "code_snippets" in enriched_data:
                        logger.info(
                            f"Code snippets found: {len(enriched_data.get('code_snippets', []))} snippets"
                        )

                # Add to activity history with enriched data
                self.activity_history.append(
                    {
                        "activity": activity,
                        "timestamp": time_now,
                        "node_name": node_name,
                        "enriched_data": enriched_data,
                    }
                )

        return (False, event)

    def flush(self):
        """Flush any buffered events that should be emitted"""
        # Future enhancement: implement batching logic here
        pass

    def _clean_content_for_frontend(self, content):
        """Clean content that might contain JSON or code blocks before sending to frontend."""
        if not content:
            return ""

        logger.info(
            f"Starting cleaning for content (first 150 chars): {content[:150]}..."
        )

        # Define patterns for the problematic JSON blocks
        topic_complexity_pattern = r'\{\s*"topic_complexity":[\s\S]*?\}\s*'
        visualization_pattern = r'\{\s*"visualization_needed":[\s\S]*?\}\s*'
        python_code_pattern = r"```python[\s\S]*?```"
        json_code_pattern = r"```json[\s\S]*?```"
        loose_imports_pattern = r"^(?:import|from)\s+.*$"
        backend_logs_pattern = r"backend logs:.*$"

        # Remove these patterns iteratively
        cleaned = content
        patterns_to_remove = [
            topic_complexity_pattern,
            visualization_pattern,
            python_code_pattern,
            json_code_pattern,
            loose_imports_pattern,
            backend_logs_pattern,
        ]

        for pattern in patterns_to_remove:
            # Use re.DOTALL to make . match newlines
            # Use re.MULTILINE for ^ matching start of lines
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.MULTILINE)

        # Remove any remaining generic JSON objects as a fallback
        def remove_json_objects(text):
            # This pattern looks for JSON-like structures with balanced braces
            # It's simplified here to avoid excessive complexity, focusing on top-level objects
            json_pattern = r"(\{(?:[^{}]|(?:\\{[^{}]*\\}))*\})"
            result = text
            # Repeatedly remove JSON until no more are found
            prev_result = ""
            while result != prev_result:
                prev_result = result
                result = re.sub(json_pattern, "", result, flags=re.DOTALL)
            return result

        cleaned = remove_json_objects(cleaned)

        # Final cleanup: remove excessive newlines and trim whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # Replace 3+ newlines with 2
        cleaned = cleaned.strip()

        logger.info(f"Finished cleaning. Result (first 150 chars): {cleaned[:150]}...")

        # If cleaning resulted in an empty string, return a placeholder message
        if not cleaned:
            logger.warning("Cleaning removed all content. Returning placeholder.")
            return "# Research Summary\n\nThe research process has completed, but the generated report content could not be displayed properly after cleaning. Please review the activity stream for details."

        return cleaned

    # Add this new method for emergency cleaning
    def _clean_all_string_content(self, event):
        """
        Emergency filter to clean ANY string content in an event to prevent
        JSON or code blocks from reaching the frontend.
        """
        if not event or not isinstance(event, dict):
            return

        # Before cleaning, check if this is a visualization activity event directly from the activity manager
        # These are special events that should be sent to the UI without further processing
        is_viz_activity = False
        if isinstance(event, dict) and event.get("event_type") == "activity_generated":
            data = event.get("data", {})
            if (
                data.get("related_event_type") == "visualization_generated"
                and data.get("node_name") == "visualization"
            ):
                is_viz_activity = True
                logger.info("DIRECT VISUALIZATION ACTIVITY DETECTED")
                images = data.get("enriched_data", {}).get("images", [])
                logger.info(f"Visualization activity has {len(images)} images")
                for i, img in enumerate(images):
                    has_src = bool(img.get("src"))
                    src_type = (
                        "base64"
                        if has_src and img["src"].startswith("data:image")
                        else "file_path" if has_src else "none"
                    )
                    src_length = len(img["src"]) if has_src else 0
                    logger.info(
                        f"  - Direct Viz Image {i+1} has src: {has_src} (type: {src_type}, length: {src_length})"
                    )
                code_snippets = data.get("enriched_data", {}).get("code_snippets", [])
                logger.info(
                    f"Visualization activity has {len(code_snippets)} code snippets"
                )
                for i, snippet in enumerate(code_snippets):
                    filename = snippet.get("filename", "unknown")
                    logger.info(
                        f"  - Direct Viz Code Snippet {i+1} has filename: {filename}"
                    )

                # Send visualization activities directly to the UI
                # Apply cleaning recursively to all string values in the event
                def clean_json_from_strings(obj):
                    if isinstance(obj, str):
                        # Remove topic_complexity JSON
                        obj = re.sub(r'{"topic_complexity"[\s\S]*?}}}', "", obj)

                        # Remove visualization_needed JSON
                        obj = re.sub(r'{"visualization_needed"[\s\S]*?}}}', "", obj)

                        # Remove Python code blocks - BUT NOT if this is a code snippet event
                        if not has_code_snippets:
                            obj = re.sub(r"```python[\s\S]*?```", "", obj)

                        # Remove loose import statements - BUT NOT if this is a code snippet event
                        if not has_code_snippets:
                            obj = re.sub(r"import\s+\w+[\s\S]*?\n", "", obj)

                        # Remove backend logs
                        obj = re.sub(r"backend logs:.*$", "", obj, flags=re.MULTILINE)

                        # Clean up general JSON objects but preserve code snippets
                        # Skip cleaning if the string contains code_snippets or if this is a code snippet event
                        if "code_snippets" not in obj and not has_code_snippets:
                            obj = re.sub(r'{"[^"]*?":[^}]*?}', "", obj)

                        return obj.strip()
                    elif isinstance(obj, dict):
                        return {k: clean_json_from_strings(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [clean_json_from_strings(item) for item in obj]
                    else:
                        return obj

                # Apply the cleaning to the entire event - UNLESS it has code snippets
                if not has_code_snippets:
                    clean_json_from_strings(event)

                # Specifically check the event content field for direct JSON
                if event.get("data") and isinstance(event["data"], dict):
                    data = event["data"]

                    # Special check for content fields that might contain JSON
                    for field in [
                        "content",
                        "report",
                        "message",
                        "running_summary",
                        "activity",
                    ]:
                        if field in data and isinstance(data[field], str):
                            # Extra aggressive cleaning for visible content fields
                            content = data[field]

                            # If it looks like pure JSON, remove it completely - BUT NOT if this is a code snippet event
                            if not has_code_snippets and (
                                (
                                    content.strip().startswith("{")
                                    and content.strip().endswith("}")
                                )
                                or (
                                    content.strip().startswith("[")
                                    and content.strip().endswith("]")
                                )
                            ):
                                try:
                                    json.loads(content.strip())
                                    # It parsed as valid JSON, so either remove it or replace with a message
                                    if field == "report" or field == "running_summary":
                                        data[field] = (
                                            "# Research Results\n\nThe research process has completed, but some content could not be displayed properly."
                                        )
                                    else:
                                        data[field] = ""
                                except:
                                    # Not valid JSON, apply normal cleaning
                                    if not has_code_snippets:
                                        data[field] = clean_json_from_strings(content)


class ResearchService:
    """
    Service for conducting deep research using the research graph.
    """

    # Store active queues instead of generators
    _active_queues: Dict[str, Queue] = {}
    # Add a lock for thread-safe access to the dictionary (good practice)
    _queue_lock = asyncio.Lock()

    # Track cancellation requests for active research sessions
    _cancellation_flags: Dict[str, bool] = {}
    _cancellation_lock = asyncio.Lock()

    @staticmethod
    def _make_json_serializable(obj: Any) -> Any:
        """
        Recursively convert objects to JSON serializable types.

        Args:
            obj: Any object that needs to be converted to a JSON serializable type

        Returns:
            A JSON serializable version of the object
        """
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, SummaryState):
            # Convert SummaryState to dictionary by getting all public attributes
            result = {}
            # Get all attributes that don't start with underscore
            for attr in dir(obj):
                if not attr.startswith("_"):
                    try:
                        value = getattr(obj, attr)
                        # Skip methods and callables
                        if not callable(value):
                            result[attr] = ResearchService._make_json_serializable(
                                value
                            )
                    except Exception as e:
                        # If attribute access fails, just skip it
                        logger.info(f"Skipping attribute {attr} in SummaryState: {e}")
            return result
        elif isinstance(obj, (list, tuple)):
            return [ResearchService._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {
                str(k): ResearchService._make_json_serializable(v)
                for k, v in obj.items()
            }
        elif hasattr(obj, "__dict__"):
            # Convert custom objects to dict
            try:
                result = {}
                for k, v in obj.__dict__.items():
                    if not k.startswith("_"):
                        result[k] = ResearchService._make_json_serializable(v)
                return result
            except:
                # If __dict__ access fails, try dir() approach
                return {
                    attr: ResearchService._make_json_serializable(getattr(obj, attr))
                    for attr in dir(obj)
                    if not attr.startswith("_")
                    and not callable(getattr(obj, attr, None))
                }
        else:
            # For other types, convert to string
            try:
                return str(obj)
            except:
                return f"<Object of type {type(obj).__name__} (not serializable)>"

    @staticmethod
    async def conduct_research(
        query: str,
        extra_effort: bool = False,
        minimum_effort: bool = False,
        benchmark_mode: bool = False,
        streaming: bool = False,
        stream_id: str = None,
        queue: Optional[Queue] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        uploaded_data_content: Optional[str] = None,
        uploaded_files: Optional[List[str]] = None,
        steering_enabled: bool = False,
        parallel_search_enabled: bool = False,
        parallel_search_max_concurrency: int = 2,
        database_info: Optional[List[Dict[str, Any]]] = None,
    ):
        """Conduct research on a given query, putting events onto a queue if provided."""
        if streaming and not queue:
            logger.error(
                f"Streaming research requested for {stream_id} but no queue provided."
            )
            # Cannot proceed without a queue for streaming
            return

        event_buffer = (
            EventBuffer()
        )  # Create buffer for processing events before queuing

        # Store state in a dictionary to avoid nonlocal issues
        state_dict = {"final_report_content": None}

        # Create a configuration object for simplified activity management
        activity_config = {
            "enable_activity_filtering": True,
            "send_important_events_only": True,
        }

        # Reset domain and image tracking for new research session
        ActivityManager.reset_image_tracking()

        # Enhanced logging for uploaded content
        logger.info(f"[UPLOAD_TRACE] ResearchService.conduct_research called")
        if uploaded_data_content:
            logger.info(f"[UPLOAD_TRACE] Uploaded data content received in service")
            logger.info(
                f"[UPLOAD_TRACE] Content length: {len(uploaded_data_content)} characters"
            )
            logger.info(
                f"[UPLOAD_TRACE] Content preview (first 100 chars): {uploaded_data_content[:100]}..."
            )
            logger.info(f"[UPLOAD_TRACE] Content type: {type(uploaded_data_content)}")
        else:
            logger.info(
                f"[UPLOAD_TRACE] No uploaded data content in service (value: {uploaded_data_content})"
            )

        if uploaded_files:
            logger.info(
                f"[UPLOAD_TRACE] Uploaded files received: {len(uploaded_files)} file(s)"
            )
            logger.info(f"[UPLOAD_TRACE] File IDs: {uploaded_files}")
        else:
            logger.info(f"[UPLOAD_TRACE] No uploaded files provided")

        # Log configuration
        logger.info(f"Starting research on topic: {query}")
        logger.info(
            f"Extra effort: {extra_effort}, Minimum effort: {minimum_effort}, Benchmark mode: {benchmark_mode}"
        )
        logger.info(f"Database info: {database_info} (type: {type(database_info)})")
        logger.info(
            f"Activity filtering enabled: {activity_config['enable_activity_filtering']}"
        )
        logger.info(
            f"[MODEL SELECTION] User selected → provider={provider}, model={model}"
        )
        logger.info(
            f"[MODEL SELECTION] Env fallback  → LLM_PROVIDER={os.environ.get('LLM_PROVIDER')}, LLM_MODEL={os.environ.get('LLM_MODEL')}"
        )
        logger.info(f"Uploaded data content present: {bool(uploaded_data_content)}")

        # --- Intent Detection Agent ---
        from services.intent_detector import detect_intent

        try:
            intent_result = await detect_intent(query, provider, model)
            analysis_required = intent_result.get("analysis_required", False)
        except Exception as e:
            logger.warning(f"Intent detection failed ({e}), defaulting to research only")
            analysis_required = False

        logger.info(f"[INTENT] analysis_required={analysis_required}")

        try:
            # Create a fresh graph instance
            fresh_graph = create_graph()

            # Configure settings based on environment variables
            with_extra_config = (
                os.environ.get("RESEARCH_EXTRA_EFFORT", "false").lower() == "true"
            )
            extra_effort = extra_effort or with_extra_config

            logger.info(f"[UPLOAD_TRACE] Creating SummaryState with uploaded_knowledge")

            # Define the initial state for the graph using SummaryState class
            logger.info(
                f"[RESEARCH_SERVICE] Creating SummaryState with database_info: {database_info}"
            )
            state = SummaryState(
                research_topic=query,
                search_query=query,
                running_summary="",
                research_complete=False,
                knowledge_gap="",
                research_loop_count=0,
                sources_gathered=[],
                web_research_results=[],
                search_results_empty=False,
                selected_search_tool="general_search",
                source_citations={},
                subtopic_queries=[],
                subtopics_metadata=[],
                extra_effort=extra_effort,
                minimum_effort=minimum_effort,
                benchmark_mode=benchmark_mode,
                llm_provider=provider,  # Pass provider to state
                llm_model=model,  # Pass model to state
                uploaded_knowledge=uploaded_data_content,  # Pass uploaded content to state
                uploaded_files=uploaded_files or [],  # Pass uploaded file IDs to state
                steering_enabled=steering_enabled,  # Enable steering if requested
                parallel_search_enabled=parallel_search_enabled,
                parallel_search_max_concurrency=parallel_search_max_concurrency,
                database_info=database_info,  # Pass database information to state
                analysis_required=analysis_required,  # Intent detection result
            )
            logger.info(
                f"[RESEARCH_SERVICE] SummaryState created with database_info: {state.database_info}"
            )

            # Register session for steering if enabled
            session_id = stream_id or str(uuid.uuid4())
            logger.info(
                f"[RESEARCH_SERVICE] About to register session {session_id}, STEERING_AVAILABLE={STEERING_AVAILABLE}, steering_enabled={steering_enabled}"
            )

            if STEERING_AVAILABLE:
                logger.info(
                    f"[RESEARCH_SERVICE] Calling register_research_session for {session_id}"
                )
                register_research_session(
                    session_id, state, steering_enabled=steering_enabled
                )
                if steering_enabled:
                    logger.info(
                        f"[STEERING] Registered session {session_id} with steering enabled"
                    )
                else:
                    logger.info(
                        f"[RESEARCH] Registered session {session_id} (no steering)"
                    )
            else:
                logger.warning(
                    f"[RESEARCH_SERVICE] STEERING_AVAILABLE is False - session not registered"
                )

            # Process uploaded files if any
            if uploaded_files:
                logger.info(
                    f"[UPLOAD_TRACE] Processing {len(uploaded_files)} uploaded files"
                )
                try:
                    from services.content_analysis import ContentAnalysisService

                    analyzed_files = []

                    for file_id in uploaded_files:
                        # Get existing analysis or trigger new analysis
                        analysis = await ContentAnalysisService.get_analysis(file_id)
                        if not analysis:
                            logger.info(
                                f"[UPLOAD_TRACE] Triggering analysis for file {file_id}"
                            )
                            analysis = await ContentAnalysisService.analyze_file(
                                file_id, "comprehensive"
                            )

                        if analysis and analysis.status == "completed":
                            analyzed_files.append(
                                {
                                    "file_id": file_id,
                                    "content_description": analysis.content_description,
                                    "metadata": analysis.metadata,
                                }
                            )
                            logger.info(
                                f"[UPLOAD_TRACE] Successfully processed file {file_id}"
                            )
                        else:
                            logger.warning(
                                f"[UPLOAD_TRACE] Failed to analyze file {file_id}"
                            )

                    # Add analyzed files to state
                    state.analyzed_files = analyzed_files
                    logger.info(
                        f"[UPLOAD_TRACE] Added {len(analyzed_files)} analyzed files to state"
                    )

                except Exception as e:
                    logger.error(
                        f"[UPLOAD_TRACE] Error processing uploaded files: {str(e)}"
                    )

            # Log the state creation
            if hasattr(state, "uploaded_knowledge") and state.uploaded_knowledge:
                logger.info(f"[UPLOAD_TRACE] State created with uploaded_knowledge")
                logger.info(
                    f"[UPLOAD_TRACE] State.uploaded_knowledge length: {len(state.uploaded_knowledge)} characters"
                )
                logger.info(
                    f"[UPLOAD_TRACE] State.uploaded_knowledge preview: {state.uploaded_knowledge[:100]}..."
                )
            else:
                logger.info(
                    f"[UPLOAD_TRACE] State created without uploaded_knowledge (value: {getattr(state, 'uploaded_knowledge', 'MISSING_ATTR')})"
                )

            if hasattr(state, "analyzed_files") and state.analyzed_files:
                logger.info(
                    f"[UPLOAD_TRACE] State created with {len(state.analyzed_files)} analyzed files"
                )

            # WORKAROUND: Store database_info in the state's config field to bypass LangGraph state serialization
            state.config = {
                "configurable": {
                    "thread_id": str(uuid.uuid4()),
                    "llm_provider": provider,
                    "llm_model": model,
                    "user_prompt": query,
                    "database_info": database_info,  # Store database_info in config
                }
            }

            # HACKY FIX: Set session-specific database_info before invoking graph
            from src.graph import set_database_info

            set_database_info(database_info, session_id=stream_id)

            # Create LangGraph config for the graph invocation
            graph_config = {
                "configurable": {
                    "thread_id": str(uuid.uuid4()),
                    "stream_id": stream_id,  # Pass stream_id for session-specific data
                    "llm_provider": provider,  # Pass provider from request
                    "llm_model": model,  # Pass model from request
                    "user_prompt": query,  # Pass query from request
                    "database_info": database_info,  # Pass database_info to LangGraph
                    # Add other request-specific config here if needed
                    # e.g., "search_depth": search_depth_from_request
                },
                "recursion_limit": 100,  # Set recursion limit to 100
            }
            logger.info(
                f"[MODEL SELECTION] graph_config.configurable → "
                f"llm_provider={graph_config['configurable']['llm_provider']}, "
                f"llm_model={graph_config['configurable']['llm_model']}"
            )
            logger.info(
                f"[MODEL SELECTION] state → "
                f"llm_provider={state.llm_provider}, llm_model={state.llm_model}"
            )

            # Utility function to update state from events
            def update_state_from_event(event):
                """Update the state object based on event data"""
                event_type = event.get("event_type")
                event_data = event.get("data", {})

                try:
                    # Update the state based on event type and data
                    if event_type == "node_end":
                        node_name = event_data.get("node_name", "unknown")
                        logger.info(f"Processing node_end for node: {node_name}")

                        # Extract output data from node_end events
                        output_data = event_data.get("output", {})
                        if isinstance(output_data, dict):
                            # Update sources gathered from deep_search_parallel node
                            if (
                                node_name == "deep_search_parallel"
                                and "sources_gathered" in output_data
                            ):
                                sources = output_data.get("sources_gathered", [])
                                if isinstance(sources, list) and sources:
                                    # Update sources in state
                                    state.sources_gathered = sources

                            # Also look for any web_research_results to include as sources
                            if (
                                "web_research_results" in output_data
                                and output_data["web_research_results"]
                            ):
                                web_results = output_data["web_research_results"]
                                if isinstance(web_results, list) and web_results:
                                    # Update the web_research_results in state
                                    state.web_research_results = web_results

                                    # Process web_research_results to extract sources if sources_gathered is empty
                                    if (
                                        not hasattr(state, "sources_gathered")
                                        or not state.sources_gathered
                                    ):
                                        state.sources_gathered = []

                                    # Process source_citations if needed
                                    if (
                                        not hasattr(state, "source_citations")
                                        or not state.source_citations
                                    ):
                                        state.source_citations = {}

                                    # Extract sources from each result in web_research_results
                                    sources_gathered = []
                                    source_citations = {}
                                    citation_index = 1

                                    # First, add any existing sources to avoid duplication
                                    if (
                                        hasattr(state, "sources_gathered")
                                        and state.sources_gathered
                                    ):
                                        sources_gathered.extend(state.sources_gathered)

                                    if (
                                        hasattr(state, "source_citations")
                                        and state.source_citations
                                    ):
                                        source_citations.update(state.source_citations)
                                        # Find the highest citation index to continue numbering
                                        highest_index = 0
                                        for key in source_citations.keys():
                                            if (
                                                key.isdigit()
                                                and int(key) > highest_index
                                            ):
                                                highest_index = int(key)
                                        citation_index = highest_index + 1

                                    # Process each web_research_result to extract sources
                                    for result in web_results:
                                        if isinstance(result, dict):
                                            # Extract sources from 'sources' field
                                            if "sources" in result and isinstance(
                                                result["sources"], list
                                            ):
                                                for source in result["sources"]:
                                                    if (
                                                        isinstance(source, dict)
                                                        and "title" in source
                                                        and "url" in source
                                                    ):
                                                        # Add to sources_gathered if not already there
                                                        source_str = f"{source['title']} : {source['url']}"
                                                        if (
                                                            source_str
                                                            not in sources_gathered
                                                        ):
                                                            sources_gathered.append(
                                                                source_str
                                                            )

                                                        # Add to source_citations if not already there
                                                        source_url = source["url"]
                                                        found = False
                                                        for (
                                                            citation_key,
                                                            citation_data,
                                                        ) in source_citations.items():
                                                            if (
                                                                citation_data.get("url")
                                                                == source_url
                                                            ):
                                                                found = True
                                                                break

                                                        if not found:
                                                            citation_key = str(
                                                                citation_index
                                                            )
                                                            source_citations[
                                                                citation_key
                                                            ] = {
                                                                "title": source[
                                                                    "title"
                                                                ],
                                                                "url": source["url"],
                                                            }
                                                            citation_index += 1

                                    # Update state with extracted sources
                                    if sources_gathered:
                                        logger.info(
                                            f"[update_state_from_event] Extracted {len(sources_gathered)} sources from web_research_results"
                                        )
                                        state.sources_gathered = sources_gathered

                                    if source_citations:
                                        logger.info(
                                            f"[update_state_from_event] Created {len(source_citations)} source citations"
                                        )
                                        state.source_citations = source_citations

                            # Update running summary from report generation nodes
                            if "running_summary" in output_data and isinstance(
                                output_data["running_summary"], str
                            ):
                                if len(output_data["running_summary"]) > 100:
                                    state.running_summary = output_data[
                                        "running_summary"
                                    ]

                            # Update knowledge_gap from search agent or other nodes
                            if (
                                "knowledge_gap" in output_data
                                and output_data["knowledge_gap"]
                            ):
                                state.knowledge_gap = output_data["knowledge_gap"]

                            # Update research_loop_count from deep_search_parallel
                            if "research_loop_count" in output_data:
                                state.research_loop_count = output_data[
                                    "research_loop_count"
                                ]

                            # Update any additional state fields that exist in the output
                            for field in [
                                "subtopic_queries",
                                "source_citations",
                                "citations",
                            ]:
                                if field in output_data and hasattr(state, field):
                                    setattr(state, field, output_data[field])

                    # Update on source-related events
                    elif event_type == "source_added" and "source" in event_data:
                        source = event_data.get("source")
                        if source:
                            if (
                                not hasattr(state, "sources_gathered")
                                or not state.sources_gathered
                            ):
                                state.sources_gathered = []
                            state.sources_gathered.append(source)

                    # Handle search_sources_found events that contain domain information
                    elif event_type == "search_sources_found":
                        # These events contain information about sources being searched
                        domains = event_data.get("search_domains", [])
                        if domains:
                            if not hasattr(state, "domains") or not state.domains:
                                state.domains = []
                            state.domains.extend(domains)

                    # Handle sources_update events
                    elif event_type == "sources_update":
                        sources = event_data.get("sources", [])
                        citations = event_data.get("citations", {})
                        source_citations = event_data.get("source_citations", {})

                        # Update sources_gathered if available
                        if sources:
                            state.sources_gathered = sources
                            logger.info(
                                f"[update_state_from_event] Updated sources_gathered with {len(sources)} sources"
                            )

                        # Update source_citations if available
                        if source_citations:
                            state.source_citations = source_citations
                            logger.info(
                                f"[update_state_from_event] Updated source_citations with {len(source_citations)} citations"
                            )
                        elif citations:
                            state.source_citations = citations
                            logger.info(
                                f"[update_state_from_event] Updated source_citations from citations with {len(citations)} citations"
                            )

                    # Update from knowledge gap events
                    elif event_type == "knowledge_gap":
                        gaps = event_data.get("gaps")
                        if gaps:
                            state.knowledge_gap = gaps

                    # Update from research_complete events
                    elif event_type == "research_complete":
                        state.research_complete = True

                        # Also update the final report if included
                        if "report" in event_data and event_data["report"]:
                            state.running_summary = event_data["report"]

                except Exception as e:
                    logger.error(f"Error updating state: {e}")

            async def _put_event_on_queue(event):
                if queue:
                    try:
                        # Handle None events (sentinel value for end of stream)
                        if event is None:
                            await queue.put(None)
                            return

                        # First, check if this is a visualization activity event directly from the activity manager
                        # These are special events that should be sent to the UI without further processing
                        is_viz_activity = False
                        if (
                            isinstance(event, dict)
                            and event.get("event_type") == "activity_generated"
                        ):
                            data = event.get("data", {})
                            if (
                                data.get("related_event_type")
                                == "visualization_generated"
                                and data.get("node_name") == "visualization"
                            ):
                                is_viz_activity = True
                                logger.info("DIRECT VISUALIZATION ACTIVITY DETECTED")
                                images = data.get("enriched_data", {}).get("images", [])
                                logger.info(
                                    f"Visualization activity has {len(images)} images"
                                )
                                for i, img in enumerate(images):
                                    has_src = bool(img.get("src"))
                                    src_type = (
                                        "base64"
                                        if has_src
                                        and img["src"].startswith("data:image")
                                        else "file_path" if has_src else "none"
                                    )
                                    src_length = len(img["src"]) if has_src else 0
                                    logger.info(
                                        f"  - Direct Viz Image {i+1} has src: {has_src} (type: {src_type}, length: {src_length})"
                                    )
                                code_snippets = data.get("enriched_data", {}).get(
                                    "code_snippets", []
                                )
                                logger.info(
                                    f"Visualization activity has {len(code_snippets)} code snippets"
                                )
                                for i, snippet in enumerate(code_snippets):
                                    filename = snippet.get("filename", "unknown")
                                    logger.info(
                                        f"  - Direct Viz Code Snippet {i+1} has filename: {filename}"
                                    )
                                # Send visualization activities directly to the UI
                                await queue.put(event)
                                logger.info(
                                    f"Sent visualization activity directly to UI: {data.get('activity', '')[:50]}..."
                                )
                                logger.info(
                                    f"Event data structure before queuing: {event.get('data', {}).get('enriched_data', {}).keys()}"
                                )
                                logger.info(
                                    f"Code snippets in event data: {event.get('data', {}).get('enriched_data', {}).get('code_snippets', [])}"
                                )
                                return

                        # Process the event through the buffer first
                        done, processed_event = event_buffer.process_event(event)

                        # If the buffer returns None for the processed event, skip this event
                        if processed_event is None:
                            return

                        # Update state based on the event before activity generation
                        if processed_event:
                            update_state_from_event(processed_event)

                        # Quick check if this is an event type we want to process at all
                        event_type = processed_event.get("event_type")
                        event_data = processed_event.get("data", {})

                        # Check if we have node_end events from multi_agents_network that might contain search results
                        if (
                            event_type == "node_end"
                            and event_data.get("node_name") == "multi_agents_network"
                        ):
                            # Extract sources from web_research_results if present
                            output_data = event_data.get("output", {})
                            if (
                                isinstance(output_data, dict)
                                and "web_research_results" in output_data
                            ):
                                web_results = output_data["web_research_results"]

                                # Process web_research_results to extract sources
                                if isinstance(web_results, list):
                                    sources_gathered = []
                                    source_citations = {}
                                    citation_index = 1

                                    # Extract sources from each result
                                    for result in web_results:
                                        if (
                                            isinstance(result, dict)
                                            and "sources" in result
                                        ):
                                            result_sources = result.get("sources", [])
                                            for source in result_sources:
                                                if (
                                                    isinstance(source, dict)
                                                    and "title" in source
                                                    and "url" in source
                                                ):
                                                    # Add to sources_gathered
                                                    source_str = f"{source['title']} : {source['url']}"
                                                    if (
                                                        source_str
                                                        not in sources_gathered
                                                    ):
                                                        sources_gathered.append(
                                                            source_str
                                                        )

                                                    # Add to source_citations
                                                    source_url = source.get("url")
                                                    found = False
                                                    for (
                                                        citation_key,
                                                        citation_data,
                                                    ) in source_citations.items():
                                                        if (
                                                            citation_data.get("url")
                                                            == source_url
                                                        ):
                                                            found = True
                                                            break

                                                    if not found:
                                                        citation_key = str(
                                                            citation_index
                                                        )
                                                        source_citations[
                                                            citation_key
                                                        ] = {
                                                            "title": source["title"],
                                                            "url": source["url"],
                                                        }
                                                        citation_index += 1

                                    # Update state with extracted sources
                                    if sources_gathered:
                                        logger.info(
                                            f"Extracted {len(sources_gathered)} sources from web_research_results"
                                        )
                                        state.sources_gathered = sources_gathered

                                    if source_citations:
                                        logger.info(
                                            f"Created {len(source_citations)} source citations"
                                        )
                                        state.source_citations = source_citations

                                    # Send a sources update event to the UI
                                    if sources_gathered or source_citations:
                                        sources_event = {
                                            "event_type": "sources_update",
                                            "data": {
                                                "sources": sources_gathered,
                                                "citations": source_citations,
                                                "source_citations": source_citations,
                                            },
                                            "timestamp": datetime.datetime.now().isoformat(),
                                        }
                                        await queue.put(sources_event)
                                        logger.info(
                                            f"Sent sources update event with {len(sources_gathered)} sources and {len(source_citations)} citations"
                                        )

                        # Check if this is an important event that should generate an activity
                        if ActivityManager.should_process_event(event_type, event_data):
                            # Only examine events that might be important for UI
                            if ActivityManager.is_important_activity(
                                event_type, event_data
                            ):
                                # Create a simplified activity event
                                # Data to be passed to ActivityManager.create_activity_event

                                node_input_state = {}
                                if event_type == "node_start":
                                    node_input_state = processed_event.get(
                                        "data", {}
                                    ).get("input", {})
                                    if not isinstance(
                                        node_input_state, dict
                                    ):  # Ensure it's a dict
                                        logger.warning(
                                            f"[_put_event_on_queue] node_input_state for node_start is not a dict: {type(node_input_state)}. Defaulting to empty dict."
                                        )
                                        node_input_state = {}

                                # Ensure formatted_sources is a list
                                fs_from_node_input = node_input_state.get(
                                    "formatted_sources"
                                )
                                fs_from_global_state = getattr(
                                    state, "formatted_sources", None
                                )

                                payload_formatted_sources = []
                                if isinstance(fs_from_node_input, list):
                                    payload_formatted_sources = fs_from_node_input
                                elif isinstance(fs_from_global_state, list):
                                    payload_formatted_sources = fs_from_global_state

                                payload_for_activity_generator = {
                                    "research_topic": node_input_state.get(
                                        "research_topic", state.research_topic
                                    ),
                                    "node_name": processed_event.get("data", {}).get(
                                        "node_name", processed_event.get("name", "")
                                    ),
                                    "timestamp": processed_event.get("timestamp"),
                                    "sources": node_input_state.get(
                                        "sources_gathered", state.sources_gathered
                                    ),
                                    "citations": node_input_state.get(
                                        "source_citations", state.source_citations
                                    ),  # Use source_citations for "citations" key too
                                    "source_citations": node_input_state.get(
                                        "source_citations", state.source_citations
                                    ),
                                    "web_research_results": node_input_state.get(
                                        "web_research_results",
                                        state.web_research_results,
                                    ),
                                    "formatted_sources": payload_formatted_sources,
                                    "research_loop_count": node_input_state.get(
                                        "research_loop_count", state.research_loop_count
                                    ),
                                    # Spread original event data (contains node inputs etc.)
                                    # This is important for ActivityManager to get the full context
                                    **(processed_event.get("data", {})),
                                    # Re-assert critical fields from node_input_state or global state to ensure they are prioritized
                                    "research_topic": node_input_state.get(
                                        "research_topic", state.research_topic
                                    ),  # Re-assert
                                    "sources": node_input_state.get(
                                        "sources_gathered", state.sources_gathered
                                    ),  # Re-assert
                                    "citations": node_input_state.get(
                                        "source_citations", state.source_citations
                                    ),  # Re-assert
                                    "source_citations": node_input_state.get(
                                        "source_citations", state.source_citations
                                    ),  # Re-assert
                                    "web_research_results": node_input_state.get(
                                        "web_research_results",
                                        state.web_research_results,
                                    ),  # Re-assert
                                    "formatted_sources": payload_formatted_sources,  # Re-assert
                                    "research_loop_count": node_input_state.get(
                                        "research_loop_count", state.research_loop_count
                                    ),  # Re-assert
                                    # Ensure 'input' key itself is correctly passed if it was part of original event_data
                                    # and not just its contents spread out.
                                    "input": (
                                        node_input_state
                                        if event_type == "node_start"
                                        else processed_event.get("data", {}).get(
                                            "input", {}
                                        )
                                    ),
                                }

                                # Ensure code_snippets are also passed correctly, prioritizing node_input_state
                                node_input_code_snippets = node_input_state.get(
                                    "code_snippets"
                                )
                                event_data_code_snippets = processed_event.get(
                                    "data", {}
                                ).get("code_snippets")
                                event_data_enriched_code_snippets = (
                                    processed_event.get("data", {})
                                    .get("enriched_data", {})
                                    .get("code_snippets")
                                )

                                if node_input_code_snippets:
                                    payload_for_activity_generator["code_snippets"] = (
                                        node_input_code_snippets
                                    )
                                elif event_data_code_snippets:
                                    payload_for_activity_generator["code_snippets"] = (
                                        event_data_code_snippets
                                    )
                                elif event_data_enriched_code_snippets:
                                    payload_for_activity_generator["code_snippets"] = (
                                        event_data_enriched_code_snippets
                                    )

                                logger.info(
                                    f"[_put_event_on_queue] Preparing payload_for_activity_generator for event_type: {event_type}"
                                )
                                logger.info(
                                    f"[_put_event_on_queue] Using node_input_state: {bool(node_input_state)}"
                                )
                                logger.info(
                                    f"[_put_event_on_queue] payload_for_activity_generator['sources'] count: {len(payload_for_activity_generator.get('sources', []))}"
                                )
                                logger.info(
                                    f"[_put_event_on_queue] payload_for_activity_generator['web_research_results'] type: {type(payload_for_activity_generator.get('web_research_results'))}, len: {len(payload_for_activity_generator.get('web_research_results', [])) if isinstance(payload_for_activity_generator.get('web_research_results'), list) else 'N/A'}"
                                )
                                logger.info(
                                    f"[_put_event_on_queue] payload_for_activity_generator['formatted_sources'] type: {type(payload_for_activity_generator.get('formatted_sources'))}, len: {len(payload_for_activity_generator.get('formatted_sources', [])) if isinstance(payload_for_activity_generator.get('formatted_sources'), list) else 'N/A'}"
                                )
                                logger.info(
                                    f"[_put_event_on_queue] payload_for_activity_generator keys: {list(payload_for_activity_generator.keys())}"
                                )

                                # Create the activity event - this is now async
                                activity_events = (
                                    await ActivityManager.create_activity_event(
                                        event_type, payload_for_activity_generator
                                    )
                                )

                                # Handle both single events and lists of events
                                if activity_events:
                                    # Convert to list if it's a single event
                                    if not isinstance(activity_events, list):
                                        activity_events = [activity_events]

                                    # Process and send each activity event
                                    for activity_event in activity_events:
                                        logger.info(
                                            f"Generated activity for {event_type}: {activity_event.get('data', {}).get('activity', '')}"
                                        )
                                        # Add source information and code snippets to the event
                                        activity_event["data"][
                                            "sources"
                                        ] = state.sources_gathered
                                        activity_event["data"][
                                            "citations"
                                        ] = state.source_citations
                                        activity_event["data"][
                                            "source_citations"
                                        ] = state.source_citations

                                        # Ensure code snippets are preserved in enriched_data
                                        if (
                                            "enriched_data"
                                            not in activity_event["data"]
                                        ):
                                            activity_event["data"]["enriched_data"] = {}
                                        if (
                                            "code_snippets"
                                            not in activity_event["data"][
                                                "enriched_data"
                                            ]
                                        ):
                                            activity_event["data"]["enriched_data"][
                                                "code_snippets"
                                            ] = []
                                        # Preserve any code snippets from the original event
                                        if event_data.get("enriched_data", {}).get(
                                            "code_snippets"
                                        ):
                                            activity_event["data"]["enriched_data"][
                                                "code_snippets"
                                            ].extend(
                                                event_data["enriched_data"][
                                                    "code_snippets"
                                                ]
                                            )

                                        # Check if this is a visualization activity
                                        is_viz = False
                                        if (
                                            activity_event.get("data", {}).get(
                                                "related_event_type"
                                            )
                                            == "visualization_generated"
                                        ):
                                            is_viz = True
                                            logger.info(
                                                "VISUALIZATION ACTIVITY DETECTED in activity_events list"
                                            )
                                            images = (
                                                activity_event.get("data", {})
                                                .get("enriched_data", {})
                                                .get("images", [])
                                            )
                                            logger.info(
                                                f"Visualization activity has {len(images)} images"
                                            )
                                            for i, img in enumerate(images):
                                                logger.info(
                                                    f"  - Image {i+1} has src attribute: {bool(img.get('src'))}"
                                                )

                                        await queue.put(activity_event)
                                        logger.info(
                                            f"Sent {'visualization ' if is_viz else ''}activity to UI queue: {activity_event.get('data', {}).get('activity', '')[:50]}..."
                                        )

                        # We no longer send most LangGraph events to the UI
                        # Only send these specific event types that the UI needs
                        important_ui_events = [
                            "research_complete",
                            "reconnecting",
                            "error",
                            "connection_interrupted",
                            "heartbeat",
                            "search_sources_found",
                            "sources_update",
                        ]

                        if event_type in important_ui_events:
                            # Add source information to important events
                            if processed_event.get("data") is None:
                                processed_event["data"] = {}
                            processed_event["data"]["sources"] = state.sources_gathered
                            processed_event["data"][
                                "citations"
                            ] = state.source_citations
                            processed_event["data"][
                                "source_citations"
                            ] = state.source_citations

                            # Still send critical events to the UI
                            await queue.put(processed_event)
                            logger.info(f"Sending important UI event: {event_type}")

                            # If we have sources, send a separate sources update event
                            if state.sources_gathered or state.source_citations:
                                sources_event = {
                                    "event_type": "sources_update",
                                    "data": {
                                        "sources": state.sources_gathered,
                                        "citations": state.source_citations,
                                        "source_citations": state.source_citations,
                                    },
                                    "timestamp": datetime.datetime.now().isoformat(),
                                }
                                await queue.put(sources_event)
                                logger.info("Sending sources update event")

                        # Also check if we need to send a heartbeat
                        heartbeat_event = event_buffer.check_heartbeat(
                            processed_event.get("timestamp")
                        )
                        if heartbeat_event:
                            await queue.put(heartbeat_event)

                    except Exception as e:
                        logger.error(
                            f"Failed to put event on queue for {stream_id}: {e}"
                        )
                        import traceback

                        logger.error(
                            f"Event queue error traceback: {traceback.format_exc()}"
                        )

            try:
                # Stream events directly from LangGraph
                async for raw_event in fresh_graph.astream_events(
                    state, graph_config, version="v2"
                ):
                    # Check for cancellation at the start of each event
                    if stream_id and await ResearchService.is_cancelled(stream_id):
                        logger.info(
                            f"[CANCEL] Research cancelled for stream {stream_id}, stopping execution"
                        )
                        # Send cancellation event to frontend
                        if queue:
                            await queue.put(
                                {
                                    "event_type": "research_cancelled",
                                    "data": {"message": "Research was stopped by user"},
                                    "timestamp": datetime.datetime.now().isoformat(),
                                }
                            )
                        # Clean up and exit
                        await ResearchService.clear_cancellation(stream_id)
                        await ResearchService.remove_queue(stream_id)
                        return None

                    event_type = raw_event["event"]
                    timestamp = datetime.datetime.now().isoformat()
                    event = None  # Initialize event to None

                    # Process different event types from LangGraph
                    if event_type == "on_chain_start":
                        # Make the input JSON serializable
                        input_data = ResearchService._make_json_serializable(
                            raw_event.get("data", {}).get("input", {})
                        )

                        # Check if this is our custom event
                        if raw_event.get("name") == "custom_event":
                            # This is a custom event from our emit_event function
                            custom_event = raw_event.get("data", {})
                            event_data = ResearchService._make_json_serializable(
                                custom_event.get("data", {}).get("data", {})
                            )

                            # Pass through the custom event with its type
                            event = {
                                "event_type": custom_event.get("data", {}).get(
                                    "event_type", "custom_event"
                                ),
                                "data": event_data,
                                "timestamp": custom_event.get("data", {}).get(
                                    "timestamp", timestamp
                                ),
                            }
                        # Handle legacy custom_search_sources_found format
                        elif (
                            raw_event.get("name") == "custom_event"
                            and "custom_event_type" in input_data
                        ):
                            # Extract the custom event data
                            custom_event_type = input_data.get("custom_event_type")

                            # Pass through the custom event
                            event = {
                                "event_type": custom_event_type,
                                "data": {
                                    "query": input_data.get("query", ""),
                                    "search_domains": input_data.get(
                                        "search_domains", []
                                    ),
                                    "search_query_index": input_data.get(
                                        "search_query_index", 0
                                    ),
                                },
                                "timestamp": timestamp,
                            }
                        else:
                            # Regular node_start event
                            event = {
                                "event_type": "node_start",
                                "data": {
                                    "node_name": raw_event.get("name", "unknown"),
                                    "input": input_data,
                                    "status": "running",
                                },
                                "timestamp": timestamp,
                            }
                    elif event_type == "on_chain_end":
                        # Make the output JSON serializable
                        output_data = ResearchService._make_json_serializable(
                            raw_event.get("data", {}).get("output", {})
                        )

                        # Check for benchmark_result first (for benchmark mode)
                        if "benchmark_result" in output_data and isinstance(
                            output_data["benchmark_result"], dict
                        ):
                            benchmark_result = output_data["benchmark_result"]
                            # Use the direct answer from benchmark result for final report content
                            direct_answer = benchmark_result.get("answer", "")
                            confidence_level = benchmark_result.get(
                                "confidence_level", ""
                            )
                            evidence = benchmark_result.get(
                                "evidence", ""
                            ) or benchmark_result.get("supporting_evidence", "")
                            sources = benchmark_result.get("sources", [])
                            reasoning = benchmark_result.get("reasoning", "")
                            limitations = benchmark_result.get("limitations", "")
                            expected_answer = benchmark_result.get(
                                "expected_answer", ""
                            )
                            is_correct = benchmark_result.get("is_correct", None)

                            # Format the benchmark result as a clean, structured final report
                            final_report_content = f"**Answer:** {direct_answer}\n\n"

                            if confidence_level:
                                final_report_content += (
                                    f"**Confidence:** {confidence_level}\n\n"
                                )

                            if evidence:
                                final_report_content += f"**Evidence:** {evidence}\n\n"

                            if reasoning:
                                final_report_content += (
                                    f"**Reasoning:** {reasoning}\n\n"
                                )

                            if sources and any(sources):
                                final_report_content += f"**Sources:** {', '.join(filter(None, sources))}\n\n"

                            if limitations and limitations.lower() != "none":
                                final_report_content += (
                                    f"**Limitations:** {limitations}\n\n"
                                )

                            if expected_answer:
                                final_report_content += (
                                    f"**Expected Answer:** {expected_answer}\n\n"
                                )

                            if is_correct is not None:
                                final_report_content += f"**Correctness:** {'Correct' if is_correct else 'Incorrect'}\n\n"

                            logger.info(
                                f"Found benchmark_result in {raw_event.get('name', 'unknown')}, using enhanced structured answer for final report"
                            )
                            state_dict["final_report_content"] = final_report_content
                        # Check if there's a running_summary that could be a final report (for non-benchmark mode)
                        elif (
                            "running_summary" in output_data
                            and isinstance(output_data["running_summary"], str)
                            and len(output_data["running_summary"]) > 200
                        ):
                            # This could be part of the final report - store it
                            logger.info(
                                f"Found substantial running_summary in {raw_event.get('name', 'unknown')}, length: {len(output_data['running_summary'])}"
                            )
                            state_dict["final_report_content"] = output_data[
                                "running_summary"
                            ]

                        # Capture analysis pipeline results from run_analysis node
                        if "analysis_report_path" in output_data:
                            state_dict["analysis_report_path"] = output_data.get("analysis_report_path")
                            state_dict["analysis_charts"] = output_data.get("analysis_charts", [])
                            state_dict["analysis_output"] = output_data.get("analysis_output", "")
                            logger.info(
                                f"[ANALYSIS] Captured analysis results: report_path={state_dict.get('analysis_report_path')}, charts={len(state_dict.get('analysis_charts', []))}"
                            )

                        # Check if this is deep_search_parallel node with search domains info
                        if (
                            raw_event.get("name") == "deep_search_parallel"
                            and "search_domains_info" in output_data
                        ):
                            search_domains_list = output_data.get(
                                "search_domains_info", []
                            )
                            print(
                                f"DEBUG: Found {len(search_domains_list)} search domains sets"
                            )

                            # Emit an event for each set of domains
                            for domain_set in search_domains_list:
                                query = domain_set.get("query", "")
                                domains = domain_set.get("domains", [])

                                if domains:
                                    print(
                                        f"DEBUG: Emitting search sources event for query: {query}"
                                    )
                                    event = {
                                        "event_type": "search_sources_found",
                                        "data": {
                                            "query": query,
                                            "search_domains": domains,
                                        },
                                        "timestamp": timestamp,
                                    }

                                    # Process through event buffer before queuing
                                    done, high_level_event = event_buffer.process_event(
                                        event
                                    )
                                    if high_level_event:
                                        await _put_event_on_queue(high_level_event)
                                    event = None  # Reset base event as we already handled the specific one

                        if event:  # If not reset, construct standard node_end
                            event = {
                                "event_type": "node_end",
                                "data": {
                                    "node_name": raw_event.get("name", "unknown"),
                                    "status": "complete",
                                    "output": output_data,
                                },
                                "timestamp": timestamp,
                            }
                    elif event_type == "on_chat_model_stream":
                        chunk_content = (
                            raw_event.get("data", {}).get("chunk", {}).content
                        )
                        if chunk_content:
                            event = {
                                "event_type": "token_stream",
                                "data": {"token": chunk_content},
                                "timestamp": timestamp,
                            }
                    elif event_type == "on_tool_start":
                        # Make the tool input JSON serializable
                        tool_input = ResearchService._make_json_serializable(
                            raw_event.get("data", {}).get("input", {})
                        )

                        # Forward our custom tool events directly
                        if "event_type" in raw_event.get("data", {}):
                            # This is a custom event from our enhanced tools
                            custom_event = raw_event.get("data", {})
                            event = {
                                "event_type": custom_event.get(
                                    "event_type", "tool_event"
                                ),
                                "data": ResearchService._make_json_serializable(
                                    custom_event.get("data", {})
                                ),
                                "timestamp": custom_event.get("timestamp", timestamp),
                            }
                        # Handle custom_event container for our updated emit_event function
                        elif raw_event.get("name") == "custom_event":
                            # This is a custom event from our enhanced emit_event function
                            custom_event = raw_event.get("data", {})
                            custom_data = custom_event.get("data", {})
                            event = {
                                "event_type": custom_event.get(
                                    "event_type", "custom_event"
                                ),
                                "data": ResearchService._make_json_serializable(
                                    custom_data
                                ),
                                "timestamp": custom_event.get("timestamp", timestamp),
                            }
                        else:
                            # Regular tool_start event
                            event = {
                                "event_type": "tool_start",
                                "data": {
                                    "tool_name": raw_event.get("name", "unknown_tool"),
                                    "input": tool_input,
                                },
                                "timestamp": timestamp,
                            }
                    elif event_type == "on_tool_end":
                        # Make the tool output JSON serializable
                        tool_output = ResearchService._make_json_serializable(
                            raw_event.get("data", {}).get("output", {})
                        )

                        # Forward our custom tool events directly
                        if "event_type" in raw_event.get("data", {}):
                            # This is a custom event from our enhanced tools
                            custom_event = raw_event.get("data", {})
                            event = {
                                "event_type": custom_event.get(
                                    "event_type", "tool_event"
                                ),
                                "data": ResearchService._make_json_serializable(
                                    custom_event.get("data", {})
                                ),
                                "timestamp": custom_event.get("timestamp", timestamp),
                            }
                        # Handle custom_event container for our updated emit_event function
                        elif raw_event.get("name") == "custom_event":
                            # This is a custom event from our enhanced emit_event function
                            custom_event = raw_event.get("data", {})
                            custom_data = custom_event.get("data", {})
                            event = {
                                "event_type": custom_event.get(
                                    "event_type", "custom_event"
                                ),
                                "data": ResearchService._make_json_serializable(
                                    custom_data
                                ),
                                "timestamp": custom_event.get("timestamp", timestamp),
                            }
                        else:
                            # Regular tool_end event
                            event = {
                                "event_type": "tool_end",
                                "data": {
                                    "tool_name": raw_event.get("name", "unknown_tool"),
                                    "output": tool_output,
                                },
                                "timestamp": timestamp,
                            }
                    else:
                        # Unknown event type - create a generic event
                        event = {
                            "event_type": event_type,
                            "data": ResearchService._make_json_serializable(
                                raw_event.get("data", {})
                            ),
                            "timestamp": timestamp,
                        }

                    # Process the constructed event through the buffer
                    if event:
                        done, high_level_event = event_buffer.process_event(event)
                        if high_level_event:
                            await _put_event_on_queue(high_level_event)

                    # Check if we should send a heartbeat based on inactivity
                    heartbeat = event_buffer.check_heartbeat(timestamp)
                    if heartbeat:
                        await _put_event_on_queue(heartbeat)
            except Exception as e:
                logger.error(f"Error processing events from LangGraph: {e}")
                # Rethrow the exception so it can be caught by the outer try-except
                raise

            # Final completion event
            if state_dict["final_report_content"]:
                logger.info(
                    f"Sending research_complete with report content, length: {len(state_dict['final_report_content'])}"
                )
            else:
                logger.warning(
                    f"No final report content found for {stream_id} - sending research_complete without report"
                )

            # Clean the report content before sending it to the frontend
            cleaned_report = None
            if state_dict["final_report_content"]:
                # Extract only the markdown content and remove any JSON or debug information
                report_content = state_dict["final_report_content"]

                # Clean the report content using our helper method
                cleaned_report = event_buffer._clean_content_for_frontend(
                    report_content
                )

                # Log the final result
                logger.info(f"Final cleaned report length: {len(cleaned_report or '')}")

            # Create a complete event with properly cleaned report
            complete_event = {
                "event_type": "research_complete",
                "data": {
                    "message": "Research process completed successfully. You can view the full report.",
                    "report": cleaned_report,
                    "analysis_required": analysis_required,
                    "analysis_report_path": state_dict.get("analysis_report_path"),
                    "analysis_charts": state_dict.get("analysis_charts", []),
                    "analysis_output": state_dict.get("analysis_output", ""),
                },
                "timestamp": datetime.datetime.now().isoformat(),
            }

            # Put the complete event directly on the queue to avoid EventBuffer filtering
            try:
                # Apply one final cleaning pass to ensure no JSON or code blocks are present
                if complete_event["data"]["report"]:
                    complete_event["data"]["report"] = (
                        event_buffer._clean_content_for_frontend(
                            complete_event["data"]["report"]
                        )
                    )

                # EMERGENCY FINAL FILTER: Use the emergency cleaner on the complete event
                # to ensure NO JSON or Python code slips through
                event_buffer._clean_all_string_content(complete_event)

                # Log the first part of the report being sent to verify it's clean
                if complete_event["data"]["report"]:
                    report_preview = (
                        complete_event["data"]["report"][:200] + "..."
                        if len(complete_event["data"]["report"]) > 200
                        else complete_event["data"]["report"]
                    )
                    logger.info(f"FINAL REPORT (PREVIEW): {report_preview}")

                if queue:  # Check if queue exists before putting
                    await queue.put(complete_event)
                elif not streaming:
                    logger.info(
                        f"Non-streaming research complete for {stream_id}, no queue to send complete_event."
                    )

                    # For non-streaming requests, we need to return a ResearchResponse object
                    from models.research import ResearchResponse

                    # Create and return a ResearchResponse object
                    return ResearchResponse(
                        running_summary=cleaned_report or "No report generated.",
                        research_complete=True,
                        research_loop_count=(
                            state.research_loop_count
                            if hasattr(state, "research_loop_count")
                            else 0
                        ),
                        sources_gathered=(
                            state.sources_gathered
                            if hasattr(state, "sources_gathered")
                            else []
                        ),
                        web_research_results=(
                            state.web_research_results
                            if hasattr(state, "web_research_results")
                            else []
                        ),
                        source_citations=(
                            state.source_citations
                            if hasattr(state, "source_citations")
                            else {}
                        ),
                        benchmark_mode=(
                            state.benchmark_mode
                            if hasattr(state, "benchmark_mode")
                            else False
                        ),
                        benchmark_result=(
                            state.benchmark_result
                            if hasattr(state, "benchmark_result")
                            else None
                        ),
                        visualizations=(
                            state.visualizations
                            if hasattr(state, "visualizations")
                            else []
                        ),
                        base64_encoded_images=(
                            state.base64_encoded_images
                            if hasattr(state, "base64_encoded_images")
                            else []
                        ),
                        visualization_paths=(
                            state.visualization_paths
                            if hasattr(state, "visualization_paths")
                            else []
                        ),
                        code_snippets=(
                            state.code_snippets
                            if hasattr(state, "code_snippets")
                            else []
                        ),
                        uploaded_knowledge=uploaded_data_content,
                        analyzed_files=(
                            state.analyzed_files
                            if hasattr(state, "analyzed_files")
                            else []
                        ),
                    )
                else:  # streaming but no queue
                    logger.warning(
                        f"Streaming research for {stream_id}, but queue is None. Cannot send complete_event."
                    )
            except Exception as e:
                logger.error(f"Error sending research_complete event: {e}")

            # Signal the end of the stream (moved from finally block to here)
            if queue:  # Check if queue exists before putting sentinel
                await queue.put(None)  # Use None as sentinel value
                logger.info(
                    f"Research task for {stream_id} finished processing and sentinel sent."
                )
            elif not streaming:
                logger.info(
                    f"Non-streaming research for {stream_id}, no queue for sentinel."
                )
            else:  # streaming but no queue
                logger.warning(
                    f"Streaming research for {stream_id}, but queue is None. Cannot send sentinel."
                )
        except (asyncio.CancelledError, concurrent.futures._base.CancelledError) as ce:
            # Extract the core message to help identify if this is a LangGraph cancel scope error
            error_message = str(ce)
            logger.warning(f"Research task for {stream_id} cancelled: {error_message}")

            # Special handling for LangGraph cancel scope errors which are causing frontend retries
            if "Cancelled by cancel scope" in error_message:
                logger.info(
                    f"LangGraph cancel scope error detected. Sending graceful termination event."
                )
                # Send a graceful completion event rather than an error to prevent frontend retries
                completion_event = {
                    "event_type": "research_complete",
                    "data": {
                        "message": "Research process completed.",
                        "research_topic": (
                            state.research_topic
                            if hasattr(state, "research_topic")
                            else query
                        ),
                        "report": "The research process completed. Results will be available shortly.",
                    },
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                if queue:
                    await _put_event_on_queue(completion_event)  # Check queue exists
            else:
                # Regular cancellation - send interrupted connection event
                error_event = {
                    "event_type": "connection_interrupted",
                    "data": {
                        "message": "Research was interrupted due to connection loss",
                        "research_topic": (
                            state.research_topic
                            if hasattr(state, "research_topic")
                            else query
                        ),
                        "can_resume": True,
                        "interruption_time": datetime.datetime.now().isoformat(),
                    },
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                if queue:
                    await _put_event_on_queue(error_event)  # Check queue exists

                # For non-streaming requests, return an error response
                if not streaming:
                    from models.research import ResearchResponse

                    return ResearchResponse(
                        running_summary="Research was interrupted due to connection loss",
                        research_complete=False,
                        research_loop_count=(
                            state.research_loop_count
                            if hasattr(state, "research_loop_count")
                            else 0
                        ),
                        sources_gathered=[],
                        analyzed_files=(
                            state.analyzed_files
                            if hasattr(state, "analyzed_files")
                            else []
                        ),
                    )
        except Exception as e:
            logger.error(
                f"Error during research process for {stream_id}: {e}", exc_info=True
            )
            error_event = {
                "event_type": "error",
                "data": {
                    "error": str(e),
                    "research_topic": (
                        state.research_topic
                        if hasattr(state, "research_topic")
                        else query
                    ),
                },
                "timestamp": datetime.datetime.now().isoformat(),
            }
            if queue:  # Check queue exists
                await _put_event_on_queue(error_event)
                await queue.put(None)  # Also send sentinel value in case of error
            # For non-streaming requests, return an error response
            if not streaming:
                from models.research import ResearchResponse

                return ResearchResponse(
                    running_summary=f"Error during research: {str(e)}",
                    research_complete=False,
                    research_loop_count=0,
                    sources_gathered=[],
                    analyzed_files=[],
                )
        finally:
            logger.info(
                f"Research task for {stream_id} finished processing in finally block."
            )
            # Clean up cancellation flag
            if stream_id:
                await ResearchService.clear_cancellation(stream_id)
                logger.info(f"Cleaned up cancellation flag for stream {stream_id}")
            # Ensure queue is marked as finished if it exists
            if queue and not queue.empty():
                try:
                    pass
                except Exception as fin_e:
                    logger.error(f"Error during final queue handling: {fin_e}")
            # Optionally remove queue from active queues if managing centrally
            if stream_id:
                await ResearchService.remove_queue(stream_id)

            # Cleanup session registration
            if STEERING_AVAILABLE:
                unregister_research_session(session_id)
                logger.info(f"[RESEARCH] Unregistered session {session_id}")

            # NOTE: Removed the fallback return from finally block as it was overriding successful returns

    @staticmethod
    async def add_queue(stream_id: str, queue: Queue):
        """Add a queue for a new stream."""
        async with ResearchService._queue_lock:
            if stream_id in ResearchService._active_queues:
                logger.warning(
                    f"Queue already exists for stream {stream_id}. Overwriting."
                )
            ResearchService._active_queues[stream_id] = queue
            logger.info(f"Added queue for stream {stream_id}")

    @staticmethod
    async def get_queue(stream_id: str) -> Optional[Queue]:
        """Get the queue for a given stream ID."""
        async with ResearchService._queue_lock:
            return ResearchService._active_queues.get(stream_id)

    @staticmethod
    async def remove_queue(stream_id: str):
        """Remove the queue for a finished or cancelled stream."""
        async with ResearchService._queue_lock:
            if stream_id in ResearchService._active_queues:
                del ResearchService._active_queues[stream_id]
                logger.info(f"Removed queue for stream {stream_id}")
            else:
                logger.warning(
                    f"Attempted to remove non-existent queue for stream {stream_id}"
                )

    @staticmethod
    async def request_cancellation(stream_id: str):
        """Request cancellation of an active research session."""
        async with ResearchService._cancellation_lock:
            ResearchService._cancellation_flags[stream_id] = True
            logger.info(f"[CANCEL] Cancellation requested for stream {stream_id}")

    @staticmethod
    async def is_cancelled(stream_id: str) -> bool:
        """Check if cancellation was requested for a stream."""
        async with ResearchService._cancellation_lock:
            return ResearchService._cancellation_flags.get(stream_id, False)

    @staticmethod
    async def clear_cancellation(stream_id: str):
        """Clear cancellation flag after research completes or is cleaned up."""
        async with ResearchService._cancellation_lock:
            if stream_id in ResearchService._cancellation_flags:
                del ResearchService._cancellation_flags[stream_id]
                logger.info(f"Cleared cancellation flag for stream {stream_id}")


# Remove the old get_stream method if it exists
# (Ensure it's removed or commented out if refactoring)
# @staticmethod
# def get_stream(stream_id: str): ...
