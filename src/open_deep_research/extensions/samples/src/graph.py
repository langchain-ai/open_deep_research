import json
import time
import re
import os
import traceback
import asyncio
import concurrent.futures
import logging
import base64
from functools import partial
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable
from pydantic import BaseModel
import uuid

from typing_extensions import Literal

# Set up logger
logger = logging.getLogger(__name__)

# Global dictionary to store database_info per session (workaround for LangGraph state serialization issue)
_session_database_info = {}


def set_database_info(database_info, session_id=None):
    """Set the database_info for a specific session or globally"""
    global _session_database_info
    if session_id:
        _session_database_info[session_id] = database_info
        logger.info(
            f"[set_database_info] Set database_info for session {session_id}: {database_info}"
        )
    else:
        # Fallback to global key if no session_id provided
        _session_database_info["__global__"] = database_info
        logger.info(f"[set_database_info] Set global database_info: {database_info}")


def get_database_info(session_id=None):
    """Get the database_info for a specific session or globally"""
    global _session_database_info
    if session_id and session_id in _session_database_info:
        db_info = _session_database_info[session_id]
        logger.info(
            f"[get_database_info] Retrieved database_info for session {session_id}: {db_info}"
        )
        return db_info
    # Fallback to global key
    db_info = _session_database_info.get("__global__")
    logger.info(f"[get_database_info] Retrieved global database_info: {db_info}")
    return db_info


def clear_database_info(session_id=None):
    """Clear database_info for a specific session"""
    global _session_database_info
    if session_id and session_id in _session_database_info:
        del _session_database_info[session_id]
        logger.info(
            f"[clear_database_info] Cleared database_info for session {session_id}"
        )
    elif session_id is None:
        # Clear all if no session specified
        _session_database_info.clear()
        logger.info(f"[clear_database_info] Cleared all session database_info")


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph


def _resolve_provider_model(configurable, state, node_name: str = "unknown"):
    """Resolve LLM provider and model from configurable + state, with clear logging.

    Priority order:
      1. state.llm_provider / state.llm_model  (set by the user's frontend selection)
      2. configurable.llm_provider / configurable.llm_model  (from graph_config or env)

    Never falls back to a hardcoded provider — always uses what the user chose.

    Returns:
        (provider: str, model: str)
    """
    # --- provider ---
    # Start with configurable (already reads env vars as fallback)
    if isinstance(configurable.llm_provider, str):
        provider = configurable.llm_provider
    else:
        provider = configurable.llm_provider.value  # Enum → str

    logger.info(f"[{node_name}] configurable.llm_provider={configurable.llm_provider} (type={type(configurable.llm_provider).__name__}) → provider='{provider}'")

    # State override takes priority (set from the frontend request)
    state_provider = getattr(state, "llm_provider", None)
    if state_provider:
        logger.info(f"[{node_name}] state.llm_provider='{state_provider}' overrides configurable")
        provider = state_provider
    else:
        logger.info(f"[{node_name}] state.llm_provider is empty/None, using configurable value")

    # --- model ---
    model = configurable.llm_model or ""
    state_model = getattr(state, "llm_model", None)
    if state_model:
        model = state_model

    logger.info(f"[{node_name}] FINAL LLM → provider='{provider}', model='{model}'")
    return provider, model

# Import prompts for QA mode (simple question-answering)
from src.prompts_qa import (
    QUESTION_DECOMPOSITION_PROMPT as QA_QUESTION_DECOMPOSITION_PROMPT,
    ANSWER_GENERATION_PROMPT as QA_ANSWER_GENERATION_PROMPT,
    ANSWER_REFLECTION_PROMPT as QA_ANSWER_REFLECTION_PROMPT,
    FINAL_ANSWER_PROMPT as QA_FINAL_ANSWER_PROMPT,
    ANSWER_VERIFICATION_PROMPT as QA_ANSWER_VERIFICATION_PROMPT,
    VALIDATE_RETRIEVAL_PROMPT as QA_VALIDATE_RETRIEVAL_PROMPT,
    REFINE_QUERY_PROMPT as QA_REFINE_QUERY_PROMPT,
)

# Import prompts for benchmark mode (with full citation processing)
from src.prompts_benchmark import (
    QUESTION_DECOMPOSITION_PROMPT as BENCHMARK_QUESTION_DECOMPOSITION_PROMPT,
    ANSWER_GENERATION_PROMPT as BENCHMARK_ANSWER_GENERATION_PROMPT,
    ANSWER_REFLECTION_PROMPT as BENCHMARK_ANSWER_REFLECTION_PROMPT,
    FINAL_ANSWER_PROMPT as BENCHMARK_FINAL_ANSWER_PROMPT,
    ANSWER_VERIFICATION_PROMPT as BENCHMARK_ANSWER_VERIFICATION_PROMPT,
    VALIDATE_RETRIEVAL_PROMPT as BENCHMARK_VALIDATE_RETRIEVAL_PROMPT,
    REFINE_QUERY_PROMPT as BENCHMARK_REFINE_QUERY_PROMPT,
)

from src.tools.tool_schema import (
    SEARCH_TOOL_FUNCTIONS,
    TOPIC_DECOMPOSITION_FUNCTION,
    Tool,
    ToolParameter,
    ToolParameterType,
)
from src.tools import SearchToolRegistry as ToolRegistry, ToolExecutor

# Import our LLM clients module for integration
from llm_clients import (
    get_llm_client,
    get_model_response,
    get_formatted_system_prompt,
    CURRENT_DATE,
    CURRENT_YEAR,
    ONE_YEAR_AGO,
)

from src.configuration import Configuration, SearchAPI, LLMProvider
from src.utils import (
    deduplicate_and_format_sources,
    general_deep_search,
    linkedin_search,
    github_search,
    academic_search,
    format_sources,
    deduplicate_sources_list,
    generate_numbered_sources,
    extract_domain,
)
from src.state import SummaryState, SummaryStateInput, SummaryStateOutput
from src.simple_steering import TaskStatus
from src.prompts import (
    query_writer_instructions,
    summarizer_instructions,
    reflection_instructions,
    finalize_report_instructions,
)

# State class is now defined in src.state as SummaryState

# Define callbacks for event emissions
callbacks = {
    "on_event": lambda event_type, data: print(
        f"EVENT: {event_type} - {json.dumps(data, indent=2)}"
    )
}


# Helper function to get the proper callback object from config
def get_callback_from_config(config):
    """
    Get the appropriate callback object from config, handling both single callback
    objects and lists of callbacks.

    Args:
        config: Configuration object which may contain callbacks

    Returns:
        tuple: (callback_obj, has_callbacks) where callback_obj is the object to call
               on_event on, and has_callbacks is a boolean indicating if a valid
               callback was found.
    """
    # If config is None, return a default empty callback
    if config is None:
        return {"on_event": lambda event_type, data: None}, False

    callbacks = config.get("callbacks", [])

    # First check if callbacks is a single object with on_event
    if callbacks and hasattr(callbacks, "on_event"):
        return callbacks, True

    # Then check if it's a list with at least one item that has on_event
    if isinstance(callbacks, list) and callbacks and hasattr(callbacks[-1], "on_event"):
        return callbacks[-1], True

    # No valid callback found
    return {"on_event": lambda event_type, data: None}, False


# Helper function to emit event if callbacks exist
def emit_event(callbacks, event_type, data=None, error_message=None):
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"emit_event called: event_type={event_type}, data={data}, callbacks={'present' if callbacks else 'none'}"
    )
    """
    Emit an event to the specified callbacks.
    
    Args:
        callbacks: Callback object or None
        event_type: Type of event to emit
        data: Data to include with the event
        error_message: Custom error message to display if emission fails
    """
    try:
        if callbacks and isinstance(callbacks, dict) and "on_event" in callbacks:
            callbacks["on_event"](event_type, data or {})
        elif callbacks and hasattr(callbacks, "on_event"):
            callbacks.on_event(event_type, data or {})
    except Exception as e:
        error_msg = error_message or f"Warning: Failed to emit event {event_type}"
        print(f"{error_msg}: {str(e)}")


# Utility function to get max loops with consistent logic
def get_max_loops(
    configurable,
    extra_effort=False,
    minimum_effort=False,
    benchmark_mode=False,
    qa_mode=False,
):
    """Get maximum number of research loops with consistent handling of effort flags.

    Args:
        configurable: Configuration object containing max_web_research_loops
        extra_effort: Boolean flag indicating if extra effort (more loops) should be used
        minimum_effort: Boolean flag indicating if minimum effort (1 loop) should be used
        benchmark_mode: Boolean flag indicating if running in benchmark mode
        qa_mode: Boolean flag indicating if running in QA mode (1 loop)

    Returns:
        int: Maximum number of research loops to perform
    """
    # Minimum effort or QA mode overrides everything - use only 1 loop
    if minimum_effort or qa_mode:
        if qa_mode:
            print("  - Using QA mode (1 loop)")
        else:
            print("  - Using minimum effort (1 loop)")
        return 1

    env_max_loops = os.environ.get("MAX_WEB_RESEARCH_LOOPS")
    base_max_loops = (
        int(env_max_loops)
        if env_max_loops
        else int(configurable.max_web_research_loops)
    )
    print(f"  - Reading MAX_WEB_RESEARCH_LOOPS from environment: {env_max_loops}")

    max_loops = base_max_loops

    print(
        f"  - Using max_loops={max_loops} (extra_effort={extra_effort}, base={base_max_loops})"
    )
    return max_loops


# Nodes
def reset_state(state: SummaryState):
    """Reset the state for a new research topic"""

    # Use the research_topic from the input but completely reset everything else
    # We need to create a brand new state instead of modifying the existing one
    return {
        "research_loop_count": 0,
        "sources_gathered": [],  # Empty array, not appending
        "web_research_results": [],  # Empty array, not appending
        "running_summary": "",
        "search_query": "",
        "research_complete": False,  # Initialize research_complete flag
        "knowledge_gap": "",  # Initialize knowledge_gap
        "search_results_empty": False,  # Initialize search_results_empty flag
        "selected_search_tool": "general_search",  # Initialize search tool tracker
        # Copy the research_topic from the input state
        "research_topic": state.research_topic,
        # Preserve analysis settings so they survive the state reset
        "analysis_required": getattr(state, "analysis_required", False),
        "llm_provider": getattr(state, "llm_provider", None),
        "llm_model": getattr(state, "llm_model", None),
    }


async def heartbeat_task(callbacks, interval=5):
    try:
        from src.graph import emit_event

        while True:
            emit_event(callbacks, "heartbeat", {"message": "still working"})
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


async def async_multi_agents_network(state: SummaryState, callbacks=None):
    """
    Asynchronously execute research using the new agent-based architecture.
    This function represents the multi-agent network entry point.

    Args:
        state: The current state containing research parameters

    Returns:
        Updated state with research results
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("[async_multi_agents_network] Starting research")
    logger.info(f"[async_multi_agents_network] state type: {type(state)}")
    logger.info(
        f"[async_multi_agents_network] state.steering_enabled: {getattr(state, 'steering_enabled', 'MISSING')}"
    )
    logger.info(
        f"[async_multi_agents_network] state.steering_todo: {getattr(state, 'steering_todo', 'MISSING')}"
    )
    logger.info(
        f"[async_multi_agents_network] hasattr steering_enabled: {hasattr(state, 'steering_enabled')}"
    )
    logger.info(
        f"[async_multi_agents_network] hasattr steering_todo: {hasattr(state, 'steering_todo')}"
    )
    logger.info("=" * 70)

    logger.info(
        "[async_multi_agents_network] Starting research with agent architecture"
    )
    logger.info(f"callbacks at entry: {'present' if callbacks else 'none'}")
    logger.info(
        f"[async_multi_agents_network] Research loop count: {state.research_loop_count}"
    )

    try:
        # Process steering messages and update todo.md at the start of each research loop
        if hasattr(state, "steering_todo") and state.steering_todo:
            from src.steering_integration import (
                integrate_steering_with_research_loop,
                get_steering_summary_for_agent,
            )

            logger.info(
                "[STEERING] Processing steering messages and updating todo.md before research loop"
            )

            # Process queued steering messages and update todo.md
            steering_result = await state.prepare_steering_for_next_loop()
            if steering_result.get("steering_enabled"):
                logger.info(
                    f"[STEERING] Todo.md updated to version {steering_result.get('todo_version')}"
                )
                logger.info(
                    f"[STEERING] Pending tasks: {steering_result.get('pending_tasks')}, "
                    f"Completed tasks: {steering_result.get('completed_tasks')}"
                )

                # Emit steering update event for UI
                if callbacks:
                    await callbacks.emit_event(
                        "steering_updated",
                        {
                            "todo_version": steering_result.get("todo_version"),
                            "current_plan": steering_result.get("current_plan"),
                            "pending_tasks": steering_result.get("pending_tasks"),
                            "completed_tasks": steering_result.get("completed_tasks"),
                            "loop_guidance": steering_result.get("loop_guidance"),
                            "research_loop_count": state.research_loop_count,
                        },
                    )

            # Get steering summary for agent context
            steering_context = get_steering_summary_for_agent(state)
            if steering_context:
                logger.info(
                    f"[STEERING] Active constraints: {steering_context.strip()}"
                )

        # Import the master agent
        from src.agent_architecture import MasterResearchAgent

        # Initialize the master agent with config from state
        # Use state for configuration: Create a config object that contains the llm_provider and llm_model
        config = getattr(state, "config", None)
        if not config:
            config = {
                "configurable": {
                    "thread_id": str(uuid.uuid4()),
                    "llm_provider": state.llm_provider,
                    "llm_model": state.llm_model,
                }
            }
        elif "configurable" not in config:
            config["configurable"] = {
                "thread_id": str(uuid.uuid4()),
                "llm_provider": state.llm_provider,
                "llm_model": state.llm_model,
            }
        else:
            # Ensure llm_provider and llm_model are in configurable
            config["configurable"]["llm_provider"] = state.llm_provider
            config["configurable"]["llm_model"] = state.llm_model

        # Log the provider and model being used
        logger.info(
            f"[async_multi_agents_network] Using provider: {state.llm_provider}, model: {state.llm_model}"
        )

        master_agent = MasterResearchAgent(config)

        # Start heartbeat
        heartbeat = asyncio.create_task(heartbeat_task(callbacks))
        # Execute research using the master agent asynchronously
        # The 'results' from master_agent.execute_research should be a list of dictionaries,
        # where each dictionary is a search result.
        # MODIFICATION: master_agent_output is now a dictionary
        # WORKAROUND: LangGraph is losing the database_info field during state serialization
        # Use session-specific global variable instead of trying to get from state.config
        stream_id = None
        if config and "configurable" in config:
            stream_id = config["configurable"].get("stream_id")
        database_info = get_database_info(session_id=stream_id)
        logger.info(
            f"[async_multi_agents_network] Database info from global variable (session {stream_id}): {database_info}"
        )
        master_agent_output = await master_agent.execute_research(
            state, callbacks=callbacks, database_info=database_info
        )
        # Cancel heartbeat when done
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass
        logger.info(
            "[async_multi_agents_network] Research completed successfully by master_agent."
        )

        # Initialize results dictionary that will be returned.
        # It should preserve existing state fields and update with new research data.
        current_state_dict = state.__dict__.copy() if state else {}
        updated_results = current_state_dict  # Start with all current state

        # Process master_agent_output which is now a dictionary
        if isinstance(master_agent_output, dict):
            logger.info(
                f"[async_multi_agents_network] master_agent returned a dictionary. Keys: {list(master_agent_output.keys())}"
            )

            # Primary search results
            raw_agent_results = master_agent_output.get("web_research_results", [])
            if isinstance(raw_agent_results, list):
                updated_results["web_research_results"] = raw_agent_results
                if raw_agent_results:
                    logger.info(
                        f"[async_multi_agents_network] First item in web_research_results is of type: {type(raw_agent_results[0])}"
                    )
                    if isinstance(raw_agent_results[0], dict):
                        logger.info(
                            f"[async_multi_agents_network] Keys in first search result item: {raw_agent_results[0].keys()}"
                        )

                    sources_gathered = []
                    source_citations = {}
                    citation_index = 1

                    existing_source_citations = getattr(state, "source_citations", {})
                    if existing_source_citations:
                        source_citations.update(existing_source_citations)
                        highest_index = 0
                        for key in source_citations.keys():
                            if (
                                isinstance(key, str)
                                and key.isdigit()
                                and int(key) > highest_index
                            ):
                                highest_index = int(key)
                            elif (
                                isinstance(key, int) and key > highest_index
                            ):  # handle int keys too
                                highest_index = key
                        citation_index = highest_index + 1

                    existing_urls_in_citations = {
                        sc_data.get("url")
                        for sc_data in source_citations.values()
                        if isinstance(sc_data, dict)
                    }

                    for result in raw_agent_results:
                        if isinstance(result, dict):
                            if "sources" in result and isinstance(
                                result["sources"], list
                            ):
                                for source in result["sources"]:
                                    if isinstance(source, dict):
                                        if "title" in source and "url" in source:
                                            source_str = (
                                                f"{source['title']} : {source['url']}"
                                            )
                                            if (
                                                source_str not in sources_gathered
                                            ):  # Basic dedupe for sources_gathered
                                                sources_gathered.append(source_str)

                                            source_url = source["url"]
                                            if (
                                                source_url
                                                not in existing_urls_in_citations
                                            ):
                                                citation_key = str(citation_index)
                                                source_dict = {
                                                    "title": source["title"],
                                                    "url": source["url"],
                                                }
                                                # Preserve source_type if it exists
                                                if "source_type" in source:
                                                    source_dict["source_type"] = source[
                                                        "source_type"
                                                    ]
                                                source_citations[citation_key] = (
                                                    source_dict
                                                )
                                                existing_urls_in_citations.add(
                                                    source_url
                                                )
                                                citation_index += 1

                    if sources_gathered:
                        logger.info(
                            f"[async_multi_agents_network] Extracted {len(sources_gathered)} sources from search results"
                        )
                        # Append to existing sources_gathered, ensuring uniqueness
                        current_sources_gathered = updated_results.get(
                            "sources_gathered", []
                        )
                        for sg in sources_gathered:
                            if sg not in current_sources_gathered:
                                current_sources_gathered.append(sg)
                        updated_results["sources_gathered"] = current_sources_gathered

                    if (
                        source_citations
                    ):  # source_citations already includes existing ones
                        logger.info(
                            f"[async_multi_agents_network] Updated source_citations, total: {len(source_citations)}"
                        )
                        updated_results["source_citations"] = source_citations
            else:
                logger.warning(
                    "[async_multi_agents_network] 'web_research_results' from master_agent was not a list."
                )
                updated_results["web_research_results"] = []

            # Merge visualization outputs
            new_visualizations = master_agent_output.get(
                "visualizations_generated_this_loop", []
            )
            if new_visualizations:
                current_visualizations = updated_results.get("visualizations", [])
                if not isinstance(current_visualizations, list):
                    current_visualizations = []
                current_visualizations.extend(new_visualizations)
                updated_results["visualizations"] = current_visualizations
                logger.info(
                    f"[async_multi_agents_network] Added {len(new_visualizations)} new visualizations. Total: {len(current_visualizations)}"
                )

            new_base64_images = master_agent_output.get(
                "base64_images_generated_this_loop", []
            )
            if new_base64_images:
                current_base64_images = updated_results.get("base64_encoded_images", [])
                if not isinstance(current_base64_images, list):
                    current_base64_images = []
                current_base64_images.extend(new_base64_images)
                updated_results["base64_encoded_images"] = current_base64_images
                logger.info(
                    f"[async_multi_agents_network] Added {len(new_base64_images)} new base64 images. Total: {len(current_base64_images)}"
                )

            # Update visualization_paths from the 'filepath' attribute of new_visualizations
            new_viz_paths = [
                viz.get("filepath")
                for viz in new_visualizations
                if isinstance(viz, dict) and viz.get("filepath")
            ]
            if new_viz_paths:
                current_viz_paths = updated_results.get("visualization_paths", [])
                if not isinstance(current_viz_paths, list):
                    current_viz_paths = []
                for path in new_viz_paths:
                    if path not in current_viz_paths:  # Ensure uniqueness
                        current_viz_paths.append(path)
                updated_results["visualization_paths"] = current_viz_paths
                logger.info(
                    f"[async_multi_agents_network] Added {len(new_viz_paths)} new visualization paths. Total unique: {len(current_viz_paths)}"
                )

            new_code_snippets = master_agent_output.get(
                "code_snippets_generated_this_loop", []
            )
            if new_code_snippets:
                current_code_snippets = updated_results.get("code_snippets", [])
                if not isinstance(current_code_snippets, list):
                    current_code_snippets = []
                # Simple deduplication for code snippets based on code content
                existing_code_hashes = {
                    hash(cs.get("code"))
                    for cs in current_code_snippets
                    if isinstance(cs, dict) and cs.get("code")
                }
                for snippet in new_code_snippets:
                    if isinstance(snippet, dict) and snippet.get("code"):
                        if hash(snippet.get("code")) not in existing_code_hashes:
                            current_code_snippets.append(snippet)
                            existing_code_hashes.add(hash(snippet.get("code")))
                updated_results["code_snippets"] = current_code_snippets
                logger.info(
                    f"[async_multi_agents_network] Added/updated code snippets. Total: {len(current_code_snippets)}"
                )

            # Preserve research_plan if master_agent_output contains it
            if "research_plan" in master_agent_output:
                updated_results["research_plan"] = master_agent_output["research_plan"]
                logger.info(
                    "[async_multi_agents_network] Updated research_plan from master_agent_output."
                )

        else:
            logger.warning(
                f"[async_multi_agents_network] master_agent returned an unexpected type: {type(master_agent_output)}. Converting to string and placing in web_research_results."
            )
            updated_results["web_research_results"] = (
                [str(master_agent_output)] if master_agent_output else []
            )

        # Log what visualization data is available (if master_agent provided it directly in a dict)
        if "visualization_html" in updated_results:
            logger.info(
                f"[async_multi_agents_network] Visualization HTML is present: {len(updated_results['visualization_html'])} chars"
            )
        if "base64_encoded_images" in updated_results:
            logger.info(
                f"[async_multi_agents_network] Base64 images: {len(updated_results.get('base64_encoded_images', []))} items"
            )
        if "visualization_paths" in updated_results:
            logger.info(
                f"[async_multi_agents_network] Visualization paths: {len(updated_results.get('visualization_paths', []))} items"
            )
        if "visualizations" in updated_results:
            logger.info(
                f"[async_multi_agents_network] Visualizations: {len(updated_results.get('visualizations', []))} items"
            )

        # Ensure research_loop_count is preserved (it should be part of current_state_dict already)
        if (
            "research_loop_count" not in updated_results
        ):  # Should not happen if current_state_dict was used as base
            updated_results["research_loop_count"] = current_state_dict.get(
                "research_loop_count", 0
            )

        # Explicitly preserve benchmark fields from the original state, as they are critical for flow
        benchmark_fields = [
            "benchmark_mode",
            "benchmark_result",
            "previous_answers",
            "reflection_history",
            "config",
        ]
        for field in benchmark_fields:
            if field in current_state_dict:  # Prioritize original state for these
                if updated_results.get(field) != current_state_dict[field]:
                    logger.info(
                        f"[async_multi_agents_network] Preserving benchmark field '{field}' from original state."
                    )
                updated_results[field] = current_state_dict[field]
            elif (
                field not in updated_results
            ):  # If not in current_state_dict and not set by agent
                updated_results[field] = (
                    None  # Or some default like [] for lists, {} for dicts
                )

        # Visualization fields should ideally be part of raw_agent_results if it's a dict,
        # or handled within MasterResearchAgent to be part of its structured output.
        # For now, we assume they might be top-level keys in raw_agent_results if it was a dict.
        # If raw_agent_results was a list, visualization data would need to be part of the search result items
        # or handled differently by MasterResearchAgent.

        logger.info(
            f"[async_multi_agents_network] Final web_research_results going to next node is a list of {len(updated_results.get('web_research_results', []))} items."
        )
        if updated_results.get("web_research_results"):
            logger.info(
                f"[async_multi_agents_network] Type of first item in final web_research_results: {type(updated_results['web_research_results'][0])}"
            )

        # CRITICAL: Ensure steering fields are preserved
        if hasattr(state, "steering_enabled"):
            updated_results["steering_enabled"] = state.steering_enabled
        if hasattr(state, "steering_todo"):
            updated_results["steering_todo"] = state.steering_todo

        logger.info(
            f"[async_multi_agents_network] Preserving steering_enabled: {updated_results.get('steering_enabled', 'MISSING')}"
        )
        logger.info(
            f"[async_multi_agents_network] Preserving steering_todo: {updated_results.get('steering_todo', 'MISSING')}"
        )

        return updated_results

    except asyncio.CancelledError as ce:
        # Gracefully handle client disconnection (e.g., laptop lid close)
        logger.warning(
            f"[async_multi_agents_network] Research cancelled due to client disconnection: {str(ce)}"
        )

        # Create a partial result state with what we have so far
        # This maintains the current state while marking it as interrupted
        interrupted_state = {
            "status": "interrupted",
            "error": f"Research was interrupted: {str(ce)}",
            "interrupted_at": datetime.now().isoformat(),
            "research_topic": state.research_topic,
            "running_summary": (
                state.running_summary if hasattr(state, "running_summary") else ""
            ),
            "research_loop_count": state.research_loop_count,
            "sources_gathered": getattr(state, "sources_gathered", []),
            "web_research_results": getattr(state, "web_research_results", []),
            "selected_search_tool": getattr(
                state, "selected_search_tool", "general_search"
            ),
            # Preserve visualization fields
            "visualization_html": getattr(state, "visualization_html", ""),
            "base64_encoded_images": getattr(state, "base64_encoded_images", []),
            "visualization_paths": getattr(state, "visualization_paths", []),
            "visualizations": getattr(state, "visualizations", []),
            "code_snippets": getattr(
                state, "code_snippets", []
            ),  # Preserve code_snippets
        }

        # Merge with existing state to avoid losing fields
        current_state_dict = state.__dict__ if state else {}
        return {**current_state_dict, **interrupted_state}

    except Exception as e:
        logger.error(
            f"[async_multi_agents_network] Error in agent-based research: {str(e)}"
        )
        logger.error(traceback.format_exc())

        # Return an error state
        error_result = {
            "error": f"Async multi-agent network failed: {str(e)}",
            "status": "failed",
            "research_topic": state.research_topic if state else "Unknown",
            "running_summary": (
                state.running_summary if state else ""
            ),  # Keep existing summary
            "research_loop_count": state.research_loop_count if state else 0,
            # Preserve visualization fields even in error cases
            "visualization_html": getattr(state, "visualization_html", ""),
            "base64_encoded_images": getattr(state, "base64_encoded_images", []),
            "visualization_paths": getattr(state, "visualization_paths", []),
            "visualizations": getattr(state, "visualizations", []),
            "code_snippets": getattr(
                state, "code_snippets", []
            ),  # Preserve code_snippets
        }
        # Merge error dict with existing state to avoid losing other fields
        current_state_dict = state.__dict__ if state else {}
        updated_state = {**current_state_dict, **error_result}
        # Ensure the state remains valid, e.g., don't clear essential fields needed later
        return updated_state


def _generate_database_report(
    state: SummaryState, database_content: str, source_citations: dict
):
    """
    Generate a comprehensive data report with LLM synthesis for database query results.
    This function presents the data WITH intelligent analysis and insights.
    """
    logger.info(
        "[DATABASE_REPORT] Starting database report generation with LLM synthesis"
    )
    logger.info(
        f"[DATABASE_REPORT] Database content length: {len(database_content)} characters"
    )
    logger.info(f"[DATABASE_REPORT] Research topic: {state.research_topic}")

    # Use LLM to generate insightful analysis from the data
    from llm_clients import get_llm_client
    from langchain_core.messages import HumanMessage, SystemMessage

    provider = getattr(state, "llm_provider", None) or os.environ.get("LLM_PROVIDER", "azure")
    model = getattr(state, "llm_model", None) or os.environ.get("LLM_MODEL", None)
    logger.info(f"[DATABASE_REPORT] Using LLM: provider='{provider}', model='{model}' (state.llm_provider={getattr(state, 'llm_provider', 'N/A')}, state.llm_model={getattr(state, 'llm_model', 'N/A')})")
    llm = get_llm_client(provider, model)

    # Check for steering messages or additional context
    steering_context = ""
    if hasattr(state, "pending_steering_messages") and state.pending_steering_messages:
        steering_messages = [msg["message"] for msg in state.pending_steering_messages]
        steering_context = f"\n\nUser also asked: {', '.join(steering_messages)}\nMake sure to address these additional questions in your analysis."
        logger.info(
            f"[DATABASE_REPORT] Including {len(steering_messages)} steering messages in analysis"
        )

    # Create analysis prompt
    analysis_prompt = f"""
You are a data analyst. Analyze the following database query results and provide:
1. A concise Executive Summary (2-3 sentences)
2. Key Findings (3-5 bullet points with specific numbers and insights)
3. Detailed Analysis (2-3 paragraphs explaining patterns, trends, or notable observations)
4. Recommendations or Conclusions (2-3 actionable insights or takeaways)

Research Question: {state.research_topic}{steering_context}

Database Query Results:
{database_content}

Generate a professional, insightful analysis. Focus on WHAT THE DATA MEANS, not just what it shows.
Be specific with numbers and percentages. Highlight surprises or notable patterns.

Format your response as plain text with clear section headers.
"""

    try:
        logger.info("[DATABASE_REPORT] Calling LLM for analysis...")
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        analysis_text = (
            response.content if hasattr(response, "content") else str(response)
        )
        logger.info(
            f"[DATABASE_REPORT] LLM analysis generated successfully, length: {len(analysis_text)} characters"
        )
        logger.info(f"[DATABASE_REPORT] Analysis preview: {analysis_text[:200]}...")
    except Exception as e:
        logger.error(f"[DATABASE_REPORT] Error generating LLM analysis: {e}")
        analysis_text = (
            "Analysis could not be generated. Please review the data results above."
        )

    # Use inline styles to avoid CSS being stripped by the cleaner
    report_content = f"""<div style="font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; line-height: 1.6;">
    
    <h1 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 20px;">Data Analysis Report</h1>
    
    <div style="background-color: #e8f5e8; border: 1px solid #4caf50; border-radius: 5px; padding: 15px; margin: 20px 0;">
        <h3 style="color: #2c3e50; margin-top: 0;">Research Overview</h3>
        <p style="margin: 5px 0;"><strong>Question:</strong> {state.research_topic}</p>
        <p style="margin: 5px 0;"><strong>Analysis Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>
        <p style="margin: 5px 0;"><strong>Data Source:</strong> Uploaded Database</p>
    </div>
    
    <div style="background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 5px; padding: 15px; margin: 20px 0;">
        <h3 style="color: #856404; margin-top: 0;">Analysis Summary</h3>
        <div style="white-space: pre-wrap; color: #333;">{analysis_text}</div>
    </div>
    
    <h2 style="color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 15px;">Query Results</h2>
    
    {database_content}
    
    <div style="margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-left: 4px solid #6c757d; border-radius: 4px;">
        <p style="margin: 0; font-size: 14px; color: #666;">
            <strong>Note:</strong> This analysis is based on data extracted from your uploaded database using SQL queries. 
            The insights above are generated using AI analysis of the query results.
        </p>
    </div>
    
    </div>"""

    logger.info(
        f"[DATABASE_REPORT] Final report content length: {len(report_content)} characters"
    )

    # Check if there are pending steering messages - if so, don't mark as complete yet
    has_pending_steering = False
    if hasattr(state, "pending_steering_messages") and state.pending_steering_messages:
        has_pending_steering = True
        logger.info(
            f"[DATABASE_REPORT] Found {len(state.pending_steering_messages)} pending steering messages - continuing research"
        )

    # Update the state with the database report
    # Only mark complete if no steering messages are pending
    updated_state = state.model_copy(
        update={
            "running_summary": report_content,
            "research_complete": not has_pending_steering,  # Don't complete if steering messages exist
            "research_loop_count": getattr(state, "research_loop_count", 0) + 1,
        }
    )

    if has_pending_steering:
        logger.info(
            "[DATABASE_REPORT] Report generated, but continuing research to process steering messages"
        )
    else:
        logger.info("[DATABASE_REPORT] Database report generation complete!")

    return updated_state


def generate_report(state: SummaryState, config: RunnableConfig):
    """
    Generate a research report based on the current state.

    This function takes the current research state and generates a comprehensive
    report by summarizing the findings and integrating new web research results.

    Args:
        state: The current state with research results
        config: Configuration for the report generation

    Returns:
        Updated state with the generated report
    """
    print(f"[UPLOAD_TRACE] generate_report: Function called")
    print(f"[UPLOAD_TRACE] generate_report: State type: {type(state)}")
    print(
        f"[UPLOAD_TRACE] generate_report: State has uploaded_knowledge attr: {hasattr(state, 'uploaded_knowledge')}"
    )
    if hasattr(state, "uploaded_knowledge"):
        print(
            f"[UPLOAD_TRACE] generate_report: State.uploaded_knowledge value: {getattr(state, 'uploaded_knowledge', 'MISSING')}"
        )

    # CRITICAL DEBUG: Check steering state at function entry
    logger.info(f"[generate_report] ===== STEERING DEBUG =====")
    logger.info(
        f"[generate_report] state.steering_enabled: {getattr(state, 'steering_enabled', 'MISSING')}"
    )
    logger.info(
        f"[generate_report] state.steering_todo: {getattr(state, 'steering_todo', 'MISSING')}"
    )
    logger.info(f"[generate_report] ===========================")

    # Get the current research loop count
    research_loop_count = getattr(state, "research_loop_count", 0)
    print(f"--- ENTERING generate_report (Loop {research_loop_count}) ---")

    # Step 1: Check if we have any new web research content
    # Combine ALL raw content strings from the last loop's results
    all_new_raw_content = (
        "\n\n---\n\n".join(
            item.get("content", "")
            for item in state.web_research_results
            if isinstance(item, dict) and item.get("content")
        )
        if state.web_research_results
        else ""
    )

    # Optionally remove base64 images from the textual content
    base64_pattern = r"data:image/[a-zA-Z]+;base64,[a-zA-Z0-9+/=]+"
    cleaned_web_research = re.sub(
        base64_pattern, "[Image Data Removed]", all_new_raw_content
    )

    existing_summary = state.running_summary or ""
    knowledge_gap = getattr(state, "knowledge_gap", "")

    print(
        f"DEBUG: Length of combined cleaned_web_research: {len(cleaned_web_research)}"
    )
    print(f"DEBUG: Length of existing_summary: {len(existing_summary)}")
    print(f"DEBUG: Length of knowledge_gap: {len(knowledge_gap)}")

    # Ensure we have source_citations
    source_citations = getattr(state, "source_citations", {})
    if (
        not source_citations
        and hasattr(state, "web_research_results")
        and state.web_research_results
    ):
        # Extract sources from web_research_results if source_citations is empty
        print(
            "SOURCE EXTRACTION: No source_citations found, extracting from web_research_results"
        )
        source_citations = {}
        citation_index = 1

        # Extract sources from each web_research_result
        for result in state.web_research_results:
            if isinstance(result, dict) and "sources" in result:
                result_sources = result.get("sources", [])
                for source in result_sources:
                    if (
                        isinstance(source, dict)
                        and "title" in source
                        and "url" in source
                    ):
                        # Add to source_citations if not already present
                        source_url = source.get("url")
                        found = False
                        for citation_key, citation_data in source_citations.items():
                            if citation_data.get("url") == source_url:
                                found = True
                                break

                        if not found:
                            citation_key = str(citation_index)
                            source_dict = {
                                "title": source["title"],
                                "url": source["url"],
                            }
                            # Preserve source_type if it exists
                            if "source_type" in source:
                                source_dict["source_type"] = source["source_type"]
                            source_citations[citation_key] = source_dict
                            citation_index += 1

        # Update state.source_citations with the newly extracted sources
        state.source_citations = source_citations
        print(
            f"SOURCE EXTRACTION: Extracted {len(source_citations)} sources from web_research_results"
        )

    # If source_citations is still empty, check sources_gathered
    if (
        not source_citations
        and hasattr(state, "sources_gathered")
        and state.sources_gathered
    ):
        print(
            "SOURCE EXTRACTION: No source_citations found, creating from sources_gathered"
        )
        source_citations = {}
        for idx, source_str in enumerate(state.sources_gathered):
            if isinstance(source_str, str) and " : " in source_str:
                try:
                    title, url = source_str.split(" : ", 1)
                    source_citations[str(idx + 1)] = {"title": title, "url": url}
                except Exception as e:
                    print(
                        f"SOURCE EXTRACTION: Error parsing source {source_str}: {str(e)}"
                    )

        # Update state.source_citations
        state.source_citations = source_citations
        print(
            f"SOURCE EXTRACTION: Created {len(source_citations)} source citations from sources_gathered"
        )

    uploaded_knowledge_content = getattr(state, "uploaded_knowledge", None)
    external_knowledge_section = ""

    # Enhanced logging for uploaded knowledge
    print(f"[UPLOAD_TRACE] generate_report: Checking for uploaded_knowledge")
    print(
        f"[UPLOAD_TRACE] generate_report: uploaded_knowledge_content = {uploaded_knowledge_content}"
    )
    print(
        f"[UPLOAD_TRACE] generate_report: uploaded_knowledge_content type = {type(uploaded_knowledge_content)}"
    )

    if uploaded_knowledge_content:
        print(f"[UPLOAD_TRACE] generate_report: uploaded_knowledge_content is truthy")
        print(
            f"[UPLOAD_TRACE] generate_report: uploaded_knowledge_content.strip() = '{uploaded_knowledge_content.strip()}'"
        )

    if uploaded_knowledge_content and uploaded_knowledge_content.strip():
        print(
            f"DEBUG: Including uploaded_knowledge in generate_report. Length: {len(uploaded_knowledge_content)}"
        )
        print(f"[UPLOAD_TRACE] generate_report: Creating external_knowledge_section")
        external_knowledge_section = f"""User-Provided External Knowledge:
------------------------------------------------------------
{uploaded_knowledge_content}

"""
        print(
            f"[UPLOAD_TRACE] generate_report: external_knowledge_section created, length: {len(external_knowledge_section)}"
        )
    else:
        print("DEBUG: No uploaded_knowledge to include in generate_report.")
        print(f"[UPLOAD_TRACE] generate_report: No external knowledge section created")

    if source_citations:
        print(f"Using {len(source_citations)} source citations for summarizer")

        # Check if we have database query results
        database_sources = [
            source
            for source in source_citations.values()
            if source.get("source_type") == "database"
        ]
        # Database results (from text2sql) are already well-formatted HTML tables
        # They flow through normal report generation just like web search results
        print(
            f"[DEBUG] Total sources: {len(source_citations)}, Database sources: {len([s for s in source_citations if 'database://' in s])}"
        )
    else:
        print("WARNING: No source citations found for summarizer. We'll still proceed.")

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    provider, model = _resolve_provider_model(configurable, state, "generate_report")

    from llm_clients import get_llm_client
    llm = get_llm_client(provider, model)

    # Build the system prompt from summarizer_instructions
    AUGMENT_KNOWLEDGE_CONTEXT = ""
    if uploaded_knowledge_content and uploaded_knowledge_content.strip():
        AUGMENT_KNOWLEDGE_CONTEXT = f"""
USER-PROVIDED EXTERNAL KNOWLEDGE AVAILABLE:
The user has provided external knowledge/documentation that should be treated as highly authoritative and trustworthy. This uploaded knowledge should form the foundation of your research synthesis, with web search results used to complement, validate, or provide recent updates.

Uploaded Knowledge Preview: {uploaded_knowledge_content[:500]}{'...' if len(uploaded_knowledge_content) > 500 else ''}
"""
    else:
        AUGMENT_KNOWLEDGE_CONTEXT = "No user-provided external knowledge available. Rely on web search results as primary sources."

    system_prompt = summarizer_instructions.format(
        research_topic=state.research_topic,
        current_date=CURRENT_DATE,
        current_year=CURRENT_YEAR,
        one_year_ago=ONE_YEAR_AGO,
        AUGMENT_KNOWLEDGE_CONTEXT=AUGMENT_KNOWLEDGE_CONTEXT,
    )

    # Provide the existing summary, new content, knowledge_gap for merging
    human_message = f"""Please integrate newly fetched content and any user-provided external knowledge into our running summary.

{external_knowledge_section}Existing Summary (previous round):
------------------------------------------------------------
{existing_summary}

Newly Fetched Web Research (current round):
------------------------------------------------------------
{cleaned_web_research}

Knowledge Gap or Additional Context:
------------------------------------------------------------
{knowledge_gap}

Citations in state: {json.dumps(source_citations, indent=2)}

Generate an updated summary that merges the newly fetched content into the existing summary, removing redundancies while retaining important details. Keep or add citation markers as needed, but do not finalize references section here. This is an internal incremental summary.

Return the updated summary as plain text:
"""

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_message)]
    )

    if hasattr(response, "content"):
        updated_summary = response.content
    else:
        updated_summary = str(response)

    # Replace the old running_summary with the updated merged summary
    state.running_summary = updated_summary

    # Clear the web_research_results so we don't keep huge raw content
    # The summary now incorporates important details from these search results
    cleared_web_research_results = []

    # Return updated state
    return {
        "running_summary": state.running_summary,
        "web_research_results": cleared_web_research_results,  # Cleared after use
        "knowledge_gap": knowledge_gap,
        "source_citations": source_citations,  # Use the updated source_citations
        "research_loop_count": state.research_loop_count,
        "research_topic": state.research_topic,
        "formatted_sources": getattr(
            state, "formatted_sources", ""
        ),  # Preserve formatted_sources if it exists
        "sources_gathered": getattr(
            state, "sources_gathered", []
        ),  # Preserve sources_gathered
        "visualizations": getattr(state, "visualizations", []),
        "base64_encoded_images": getattr(state, "base64_encoded_images", []),
        "visualization_paths": getattr(state, "visualization_paths", []),
        "selected_search_tool": state.selected_search_tool,
        "code_snippets": getattr(state, "code_snippets", []),
        # Preserve analysis settings so routing works after generate_report
        "analysis_required": getattr(state, "analysis_required", False),
        "llm_provider": getattr(state, "llm_provider", None),
        "llm_model": getattr(state, "llm_model", None),
    }


def reflect_on_report(state: SummaryState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Reflect on the current research report and decide if further research is needed.
    This function analyzes the current research, identifies knowledge gaps,
    and determines the next search query.

    Args:
        state: The current state with research results
        config: Configuration for the reflection process

    Returns:
        Updated state with research_loop_count incremented and possibly new search_query
    """
    try:
        # CRITICAL DEBUG: Check steering state at function entry
        logger.info(f"[reflect_on_report] ===== STEERING DEBUG =====")
        logger.info(
            f"[reflect_on_report] state.steering_enabled: {getattr(state, 'steering_enabled', 'MISSING')}"
        )
        logger.info(
            f"[reflect_on_report] state.steering_todo: {getattr(state, 'steering_todo', 'MISSING')}"
        )
        logger.info(
            f"[reflect_on_report] state.steering_todo type: {type(getattr(state, 'steering_todo', None))}"
        )
        if hasattr(state, "steering_todo") and state.steering_todo:
            logger.info(
                f"[reflect_on_report] steering_todo.tasks count: {len(state.steering_todo.tasks)}"
            )
            logger.info(
                f"[reflect_on_report] steering_todo.pending_messages: {len(state.steering_todo.pending_messages)}"
            )
        logger.info(f"[reflect_on_report] ===========================")

        # IMPORTANT: If we have a database report, skip LLM reflection and mark complete
        # All reports go through normal reflection, including database results

        configurable = get_configurable(config)

        # Get current research state
        research_loop_count = getattr(state, "research_loop_count", 0)
        extra_effort = getattr(state, "extra_effort", False)
        minimum_effort = getattr(
            state, "minimum_effort", False
        )  # Get minimum_effort flag
        max_research_loops = get_max_loops(
            configurable, extra_effort, minimum_effort, state.benchmark_mode
        )  # Pass minimum_effort and benchmark_mode
        research_topic = state.research_topic

        print(f"\n--- REFLECTION START (Loop {research_loop_count+1}) ---")
        print(f"  - Current Loop Count: {research_loop_count}")
        print(
            f"  - Max Research Loops: {max_research_loops} (extra_effort={extra_effort}, minimum_effort={minimum_effort}, benchmark_mode={state.benchmark_mode})"
        )  # Update log
        print(f"  - Research Topic: {research_topic}")

        # Increment the research loop counter
        next_research_loop_count = research_loop_count + 1

        # Check if we've reached the maximum number of research loops
        if next_research_loop_count > max_research_loops:
            # PRIORITY CHECK: Only steering messages prevent max_loop stop (NOT tasks!)
            # User requirement: "if no steering messages left to process, but if max_loops reached then research stops"
            pending_messages_count = 0

            if (
                hasattr(state, "pending_steering_messages")
                and state.pending_steering_messages
            ):
                pending_messages_count = len(state.pending_steering_messages)

            # Also check todo manager for pending messages
            if hasattr(state, "steering_todo") and state.steering_todo:
                pending_messages_count += len(state.steering_todo.pending_messages)

            if pending_messages_count > 0:
                logger.warning(
                    f"[REFLECT] Max loops reached but {pending_messages_count} steering messages still need processing - MUST continue"
                )
                research_complete = False  # MUST process all steering messages
            else:
                logger.info(
                    f"[REFLECT] Max loops reached with no steering messages - STOPPING research"
                )
                research_complete = True  # Hard stop at max loops

            print(
                f"REFLECTION DECISION: Reached maximum research loops ({max_research_loops}). Pending steering messages: {pending_messages_count}, research_complete={research_complete}"
            )
            return {
                # Fields calculated/updated by this node
                "research_loop_count": next_research_loop_count,
                "research_complete": research_complete,
                "knowledge_gap": "",
                "search_query": "",
                "extra_effort": extra_effort,
                "minimum_effort": minimum_effort,
                # Reflection metadata (max loops reached)
                "priority_section": None,
                "section_gaps": None,
                "evaluation_notes": "Max loops reached, forcing completion",
                # <<< START FIX: Preserve fields from input state >>>
                "research_topic": state.research_topic,
                "running_summary": state.running_summary,
                "sources_gathered": state.sources_gathered,
                "source_citations": state.source_citations,
                "visualization_paths": getattr(state, "visualization_paths", []),
                "web_research_results": getattr(state, "web_research_results", []),
                "visualizations": getattr(state, "visualizations", []),
                "base64_encoded_images": getattr(state, "base64_encoded_images", []),
                "code_snippets": getattr(
                    state, "code_snippets", []
                ),  # Preserve code_snippets
                # Preserve analysis settings so routing works after max loops
                "analysis_required": getattr(state, "analysis_required", False),
                "llm_provider": getattr(state, "llm_provider", None),
                "llm_model": getattr(state, "llm_model", None),
                # <<< END FIX >>>
            }

        # Get LLM client
        provider, model = _resolve_provider_model(configurable, state, "reflect_on_report")

        from llm_clients import get_llm_client
        llm = get_llm_client(provider, model)

        # Format the reflection_instructions with the appropriate context
        uploaded_knowledge_content = getattr(state, "uploaded_knowledge", None)
        AUGMENT_KNOWLEDGE_CONTEXT = ""
        if uploaded_knowledge_content and uploaded_knowledge_content.strip():
            AUGMENT_KNOWLEDGE_CONTEXT = f"""
USER-PROVIDED EXTERNAL KNOWLEDGE AVAILABLE:
The user has provided external knowledge/documentation that should be considered when evaluating research completeness. This uploaded knowledge is highly authoritative and may already cover significant portions of the research topic.

Uploaded Knowledge Preview: {uploaded_knowledge_content[:500]}{'...' if len(uploaded_knowledge_content) > 500 else ''}

When evaluating completeness, consider:
- What aspects are already well-covered by the uploaded knowledge
- Whether web research has successfully complemented the uploaded knowledge
- Focus knowledge gaps on areas not covered by either uploaded knowledge or web research
"""
        else:
            AUGMENT_KNOWLEDGE_CONTEXT = "No user-provided external knowledge available. Evaluate completeness based solely on web research results."

        # CRITICAL: Collect todo context and steering messages for unified reflection
        # NEW ARCHITECTURE: Send ONLY pending tasks for completion evaluation
        # Completed tasks sent separately when creating NEW tasks (to avoid duplicates)
        pending_tasks_for_reflection = ""
        completed_tasks_context = ""
        steering_messages = ""
        print(f"[reflect_on_report] state.steering_todo: {state.steering_todo}")
        if hasattr(state, "steering_todo") and state.steering_todo:
            # Get ONLY pending tasks for LLM to evaluate completion
            pending_tasks_for_reflection = (
                state.steering_todo.get_pending_tasks_for_llm()
            )
            logger.info(
                f"[reflect_on_report] pending_tasks_for_reflection length: {len(pending_tasks_for_reflection)}"
            )
            logger.debug(
                f"[reflect_on_report] pending_tasks_for_reflection content:\n{pending_tasks_for_reflection[:500]}"
            )

            # Get completed tasks context (for creating new tasks without duplicates)
            # IMPORTANT: Show ALL completed tasks to prevent duplicates!
            completed_tasks_context = state.steering_todo.get_completed_tasks_for_llm(
                limit=None  # Show ALL completed tasks, not just 10
            )
            completed_count = len(state.steering_todo.get_completed_tasks())
            logger.info(
                f"[reflect_on_report] Showing {completed_count} completed tasks to LLM (length: {len(completed_tasks_context)} chars)"
            )
            logger.info(
                f"[reflect_on_report] completed_tasks_context preview:\n{completed_tasks_context[:800]}"
            )

            # CRITICAL: Snapshot the message queue to prevent race conditions
            # If user sends messages DURING reflection, we need to preserve them
            messages_snapshot = list(state.steering_todo.pending_messages)

            # Get pending steering messages (queued by prepare_steering_for_next_loop)
            # Index messages with [0], [1], etc. for LLM to reference in clear_messages
            if messages_snapshot:
                steering_messages = "\n".join(
                    [f'[{i}] "{msg}"' for i, msg in enumerate(messages_snapshot)]
                )
                logger.info(
                    f"[reflect_on_report] Snapshotted {len(messages_snapshot)} steering messages for LLM processing"
                )
            else:
                steering_messages = "No new steering messages this loop"

            logger.info(
                f"[reflect_on_report] Pending tasks: {len(state.steering_todo.get_pending_tasks())}"
            )
            logger.info(
                f"[reflect_on_report] Completed tasks: {len(state.steering_todo.get_completed_tasks())}"
            )
            logger.info(
                f"[reflect_on_report] Steering messages: {len(state.steering_todo.pending_messages)}"
            )
        else:
            pending_tasks_for_reflection = "No todo list active (steering disabled)"
            completed_tasks_context = ""
            steering_messages = "No steering system active"

        formatted_prompt = reflection_instructions.format(
            research_topic=research_topic,
            current_date=CURRENT_DATE,
            current_year=CURRENT_YEAR,
            one_year_ago=ONE_YEAR_AGO,
            AUGMENT_KNOWLEDGE_CONTEXT=AUGMENT_KNOWLEDGE_CONTEXT,
            pending_tasks=pending_tasks_for_reflection,
            completed_tasks=completed_tasks_context,
            steering_messages=steering_messages,
        )

        # Prepare the current summary for analysis
        current_summary = (
            state.running_summary if hasattr(state, "running_summary") else ""
        )

        # Call LLM with the properly formatted system prompt
        response = llm.invoke(
            [
                SystemMessage(content=formatted_prompt),
                HumanMessage(
                    content=f"Analyze this research summary and determine if more research is needed:\n\n{current_summary}"
                ),
            ]
        )

        print("  - Raw LLM Reflection Response:")
        print(f"    {response}")

        # Extract content based on the response type
        if hasattr(response, "content"):
            content = response.content
        else:
            content = (
                response  # fallback: raw string response
            )

        # Parse the response - extract JSON from <answer> tags
        def parse_wrapped_response(reg_exp, text_phrase):
            match = re.search(reg_exp, text_phrase, re.DOTALL)
            if match:
                return match.group(1)
            return ""

        try:
            # First try to extract from <answer> tags
            json_str = parse_wrapped_response(r"<answer>\s*(.*?)\s*</answer>", content)

            if json_str:
                # Clean up the JSON string
                json_str = json_str.strip()
                result = json.loads(json_str)
            else:
                # Fallback: Look for JSON block in markdown code blocks
                json_match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)
                else:
                    # Last resort: Try parsing the entire content as JSON
                    result = json.loads(content)

            # Log the reflection result
            print("  - Parsed LLM Reflection Result:")
            print(f"    {json.dumps(result, indent=4)}")

            # Extract the key information from the result
            research_complete = result.get("research_complete", False)
            knowledge_gap = result.get("knowledge_gap", "")
            search_query = result.get("follow_up_query", "")

            # Check for pending steering messages - override research_complete if messages pending
            has_pending_steering = (
                hasattr(state, "pending_steering_messages")
                and state.pending_steering_messages
            )

            # CRITICAL: Check for pending steering messages in todo manager
            if hasattr(state, "steering_todo") and state.steering_todo:
                todo_pending_messages = len(state.steering_todo.pending_messages)
                if todo_pending_messages > 0:
                    has_pending_steering = True
                    logger.info(
                        f"[REFLECT] Found {todo_pending_messages} pending steering messages in todo manager"
                    )

            if has_pending_steering and research_complete:
                logger.info(
                    f"[REFLECT] LLM marked complete but {len(state.pending_steering_messages) if hasattr(state, 'pending_steering_messages') else 0} steering messages pending - continuing research"
                )
                research_complete = False
                # Create knowledge gap for steering messages
                if not knowledge_gap:
                    knowledge_gap = "Process pending steering messages"

            # Extract the research_topic field, or use the original if not provided
            preserved_research_topic = result.get("research_topic", research_topic)

            # If research is complete, ensure search_query is empty
            if research_complete:
                print(
                    "  - LLM determined research is complete. Clearing knowledge gap and search query."
                )
                search_query = ""
                knowledge_gap = ""
            else:
                print("  - LLM determined research should continue.")

            # CRITICAL: Process todo_updates from LLM response
            if hasattr(state, "steering_todo") and state.steering_todo:
                todo_updates = result.get("todo_updates", {})
                logger.info(
                    f"[reflect_on_report] LLM result keys: {list(result.keys())}"
                )
                logger.info(
                    f"[reflect_on_report] todo_updates present: {bool(todo_updates)}"
                )
                logger.info(f"[reflect_on_report] todo_updates content: {todo_updates}")

                if todo_updates:

                    # Mark tasks as completed (only if they're currently PENDING or IN_PROGRESS)
                    from src.simple_steering import TaskStatus

                    mark_completed_list = todo_updates.get("mark_completed", [])
                    logger.info(
                        f"[reflect_on_report] mark_completed list: {mark_completed_list}"
                    )

                    for task_id in mark_completed_list:
                        logger.info(f"[DEBUG] Processing task_id: {task_id}")

                        if task_id in state.steering_todo.tasks:
                            logger.info(
                                f"[DEBUG] Task {task_id} found in tasks dict"
                            )
                            task = state.steering_todo.tasks[task_id]
                            logger.info(f"[DEBUG] Task status: {task.status}")

                            # Only mark as completed if it's NOT already completed
                            if task.status == TaskStatus.COMPLETED:
                                logger.debug(
                                    f"[reflect_on_report] Task {task_id} already COMPLETED, skipping"
                                )
                                continue

                            # Only mark as completed if it's PENDING or IN_PROGRESS
                            if task.status in [
                                TaskStatus.PENDING,
                                TaskStatus.IN_PROGRESS,
                            ]:
                                state.steering_todo.mark_task_completed(
                                    task_id,
                                    completion_note="Addressed in research loop",
                                )
                                logger.info(
                                    f"[reflect_on_report] (done) Marked task {task_id} as completed"
                                )
                            else:
                                logger.debug(
                                    f"[reflect_on_report] Task {task_id} has status {task.status.name}, not marking completed"
                                )
                        else:
                            logger.warning(
                                f"[reflect_on_report] Task {task_id} not found in tasks dict. Available tasks: {list(state.steering_todo.tasks.keys())[:5]}"
                            )

                    # Cancel tasks
                    cancel_tasks_list = todo_updates.get("cancel_tasks", [])
                    logger.info(
                        f"[reflect_on_report] cancel_tasks list: {cancel_tasks_list}"
                    )
                    for task_id in cancel_tasks_list:
                        if task_id in state.steering_todo.tasks:
                            state.steering_todo.mark_task_cancelled(
                                task_id, reason="No longer relevant based on findings"
                            )
                            logger.info(
                                f"[reflect_on_report] (cancelled) Cancelled task {task_id}"
                            )

                    # Add new tasks with source-based priority
                    # Priority mapping: steering_message=10, original_query=9, knowledge_gap=7
                    SOURCE_PRIORITY = {
                        "steering_message": 10,  # User explicitly requested
                        "original_query": 9,  # From initial research query
                        "knowledge_gap": 7,  # System-identified gaps
                    }

                    add_tasks_list = todo_updates.get("add_tasks", [])
                    logger.info(
                        f"[reflect_on_report] add_tasks list length: {len(add_tasks_list)}"
                    )
                    for i, new_task in enumerate(add_tasks_list):
                        source = new_task.get("source", "knowledge_gap")
                        priority = SOURCE_PRIORITY.get(source, 8)

                        logger.info(
                            f"[reflect_on_report] Processing new task {i+1}/{len(add_tasks_list)}: {new_task}"
                        )
                        task_id = state.steering_todo.create_task(
                            description=new_task.get("description", ""),
                            priority=priority,
                            source=source,
                            created_from_message=new_task.get(
                                "rationale", "Added by reflection"
                            ),
                        )
                        logger.info(
                            f"[reflect_on_report] + Added task {task_id} (source: {source}, priority: {priority}): {new_task.get('description', '')[:60]}"
                        )

                    # SMART MESSAGE CLEARING: Only clear messages LLM says are fully addressed
                    # Use the snapshot to avoid race conditions with messages added during reflection
                    clear_message_indices = todo_updates.get("clear_messages", [])
                    if clear_message_indices:
                        original_snapshot_count = len(messages_snapshot)

                        # Clear from snapshot (not live list!)
                        remaining_snapshot_messages = [
                            msg
                            for i, msg in enumerate(messages_snapshot)
                            if i not in clear_message_indices
                        ]

                        # Now merge: Keep any NEW messages added during reflection + remaining snapshot messages
                        current_live_messages = state.steering_todo.pending_messages
                        new_messages_during_reflection = [
                            msg
                            for msg in current_live_messages
                            if msg not in messages_snapshot
                        ]

                        # Final queue = remaining snapshot + new messages
                        state.steering_todo.pending_messages = (
                            remaining_snapshot_messages + new_messages_during_reflection
                        )

                        cleared_count = original_snapshot_count - len(
                            remaining_snapshot_messages
                        )
                        if new_messages_during_reflection:
                            logger.info(
                                f"[reflect_on_report] {len(new_messages_during_reflection)} new messages arrived during reflection - preserved!"
                            )
                        logger.info(
                            f"[reflect_on_report] Cleared {cleared_count}/{original_snapshot_count} steering messages: indices {clear_message_indices}"
                        )
                    else:
                        logger.info(
                            f"[reflect_on_report] No messages cleared (LLM didn't specify any in clear_messages)"
                        )

                    # Session store is automatically synced since we modified state.steering_todo.pending_messages directly
                    # The UI polling endpoint reads from state.steering_todo.pending_messages
                    # No need to manually update session store

                    # Update todo version
                    state.steering_todo.todo_version += 1
                    logger.info(
                        f"[reflect_on_report] Updated todo version to {state.steering_todo.todo_version}"
                    )

                    # CRITICAL: Update session store so UI polling picks up the changes
                    # Retry logic to ensure UI gets the update
                    session_update_success = False
                    max_retries = 3

                    for attempt in range(max_retries):
                        try:
                            from routers.simple_steering_api import (
                                active_research_sessions,
                            )

                            # Get session ID directly from config (much cleaner!)
                            session_id = config.get("configurable", {}).get("stream_id")

                            if not session_id:
                                logger.warning(
                                    f"[reflect_on_report] No session ID in config (attempt {attempt + 1}). Config keys: {list(config.get('configurable', {}).keys())}"
                                )
                                break  # No point retrying if no session ID

                            if session_id not in active_research_sessions:
                                logger.warning(
                                    f"[reflect_on_report] Session {session_id} not in active_research_sessions (attempt {attempt + 1}). Active sessions: {list(active_research_sessions.keys())}"
                                )
                                if attempt < max_retries - 1:
                                    from time import sleep

                                    sleep(0.1)  # Wait for registration
                                continue

                            # IMPORTANT: Update the state reference in the session
                            # LangGraph creates new state instances, so the stored reference gets stale
                            active_research_sessions[session_id]["state"] = state
                            session_update_success = True
                            logger.info(
                                f"[reflect_on_report] Updated session {session_id} state reference for UI polling (attempt {attempt + 1})"
                            )
                            logger.info(
                                f"[reflect_on_report] Pending messages after update: {len(state.steering_todo.pending_messages)}"
                            )
                            break  # Success - exit retry loop

                        except ImportError as e:
                            logger.error(
                                f"[reflect_on_report] Failed to import active_research_sessions (attempt {attempt + 1}): {e}"
                            )
                            break  # Import error won't fix itself
                        except KeyError as e:
                            logger.error(
                                f"[reflect_on_report] KeyError accessing session (attempt {attempt + 1}): {e}"
                            )
                            if attempt < max_retries - 1:
                                from time import sleep

                                sleep(0.1)
                        except Exception as e:
                            logger.error(
                                f"[reflect_on_report] Unexpected error updating session (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}"
                            )
                            if attempt < max_retries - 1:
                                from time import sleep

                                sleep(0.1)  # Brief delay before retry

                    if not session_update_success and clear_message_indices:
                        logger.error(
                            "[reflect_on_report] CRITICAL: Session state update failed after all retries - UI queue may not clear!"
                        )

            # CRITICAL: Final validation - check ONLY pending steering messages (NOT tasks!)
            # User requirement: "research can stop only if steered_message_queue is empty"
            final_pending_messages = 0
            if hasattr(state, "steering_todo") and state.steering_todo:
                final_pending_messages = len(state.steering_todo.pending_messages)

                if final_pending_messages > 0 and research_complete:
                    logger.warning(
                        f"[REFLECT] LLM marked complete but {final_pending_messages} steering messages still need processing - OVERRIDING to continue"
                    )
                    research_complete = False
                    if not knowledge_gap:
                        knowledge_gap = f"Process {final_pending_messages} pending steering messages"
                    if not search_query:
                        search_query = "Continue research to process steering messages"

            # Log the reflection decision
            print(
                f"REFLECTION DECISION: Proceeding to loop {next_research_loop_count}."
            )
            print(f"  - research_complete set to: {research_complete}")
            print(f"  - Pending steering messages: {final_pending_messages}")
            print(f"  - Identified knowledge gap: '{knowledge_gap}'")
            print(f"  - New search query: '{search_query}'")
            print(f"  - Research topic: '{preserved_research_topic}'")
            print("--- REFLECTION END ---")

            # Log complete reflection execution step (non-invasive, never fails research)
            try:
                if hasattr(state, "log_execution_step"):
                    state.log_execution_step(
                        step_type="llm_call",
                        action="reflect_on_report",
                        input_data={
                            "running_summary": (
                                current_summary[:500] + "..."
                                if len(current_summary) > 500
                                else current_summary
                            )
                        },
                        output_data={
                            "research_complete": research_complete,
                            "knowledge_gap": knowledge_gap,
                            "priority_section": result.get("priority_section"),
                            "section_gaps": result.get("section_gaps"),
                            "evaluation_notes": result.get("evaluation_notes"),
                            "follow_up_query": search_query,
                        },
                        metadata={"provider": provider, "model": model},
                    )
            except Exception:
                pass  # Logging errors should never break research

            # Return updated state
            return {
                # Fields calculated/updated by this node
                "research_loop_count": next_research_loop_count,
                "knowledge_gap": knowledge_gap,
                "search_query": search_query,
                "research_complete": research_complete,
                "research_topic": preserved_research_topic,
                "extra_effort": extra_effort,
                "minimum_effort": minimum_effort,
                # Reflection metadata (for trajectory capture)
                "priority_section": result.get("priority_section"),
                "section_gaps": result.get("section_gaps"),
                "evaluation_notes": result.get("evaluation_notes"),
                # <<< START FIX: Preserve fields from input state >>>
                "research_topic": state.research_topic,
                "running_summary": state.running_summary,
                "sources_gathered": state.sources_gathered,
                "source_citations": state.source_citations,
                "visualization_paths": getattr(state, "visualization_paths", []),
                "web_research_results": getattr(state, "web_research_results", []),
                "visualizations": getattr(state, "visualizations", []),
                "base64_encoded_images": getattr(state, "base64_encoded_images", []),
                "code_snippets": getattr(
                    state, "code_snippets", []
                ),  # Preserve code_snippets
                "analysis_required": getattr(state, "analysis_required", False),  # Preserve for routing
                "llm_provider": getattr(state, "llm_provider", None),
                "llm_model": getattr(state, "llm_model", None),
                # CRITICAL: Preserve steering state
                "steering_enabled": getattr(state, "steering_enabled", False),
                "steering_todo": getattr(state, "steering_todo", None),
                # <<< END FIX >>>
            }

        except Exception as e:
            print(f"REFLECTION ERROR: Failed to parse LLM response: {str(e)}")
            print(f"  - Raw response: {content}")

            # Fallback to basic approach
            current_sources = getattr(state, "sources_gathered", [])
            num_sources = len(current_sources)

            if num_sources < 3:
                knowledge_gap = "Need more comprehensive sources"
                search_query = f"More detailed information about {research_topic}"
            else:
                knowledge_gap = "Need specific examples and case studies"
                search_query = f"Examples and case studies of {research_topic}"

            print(f"REFLECTION DECISION: Using fallback approach due to parsing error.")
            print(f"  - Identified knowledge gap: '{knowledge_gap}'")
            print(f"  - New search query: '{search_query}'")
            print(f"  - Research topic: '{research_topic}' (preserved from original)")
            print("--- REFLECTION END (Fallback) ---")

            # Return updated state with fallback values
            return {
                # Fields calculated/updated by this node
                "research_loop_count": next_research_loop_count,
                "knowledge_gap": knowledge_gap,
                "search_query": search_query,
                "research_complete": False,
                "research_topic": research_topic,
                "extra_effort": extra_effort,
                "minimum_effort": minimum_effort,
                # Reflection metadata (fallback - empty for trajectory capture)
                "priority_section": None,
                "section_gaps": None,
                "evaluation_notes": "Reflection parsing failed, using fallback",
                # <<< START FIX: Preserve fields from input state >>>
                "research_topic": state.research_topic,
                "running_summary": state.running_summary,
                "sources_gathered": state.sources_gathered,
                "source_citations": state.source_citations,
                "visualization_paths": getattr(state, "visualization_paths", []),
                "web_research_results": getattr(state, "web_research_results", []),
                "visualizations": getattr(state, "visualizations", []),
                "base64_encoded_images": getattr(state, "base64_encoded_images", []),
                "code_snippets": getattr(
                    state, "code_snippets", []
                ),  # Preserve code_snippets
                "analysis_required": getattr(state, "analysis_required", False),  # Preserve for routing
                "llm_provider": getattr(state, "llm_provider", None),
                "llm_model": getattr(state, "llm_model", None),
                # <<< END FIX >>>
            }

    except Exception as e:
        print(f"REFLECTION FATAL ERROR: {str(e)}")
        # On error, increment research loop but mark as complete to avoid infinite loops
        print("  - Marking research as complete to avoid infinite loops.")
        print("--- REFLECTION END (Fatal Error) ---")
        return {
            # Fields calculated/updated by this node
            "research_loop_count": research_loop_count + 1,
            "research_complete": True,
            "knowledge_gap": "",
            "search_query": "",
            "research_topic": research_topic,
            "extra_effort": extra_effort,
            "minimum_effort": minimum_effort,
            # <<< START FIX: Preserve fields from input state >>>
            "research_topic": state.research_topic,
            "running_summary": state.running_summary,
            "sources_gathered": state.sources_gathered,
            "source_citations": state.source_citations,
            "visualization_paths": getattr(state, "visualization_paths", []),
            "web_research_results": getattr(state, "web_research_results", []),
            "visualizations": getattr(state, "visualizations", []),
            "base64_encoded_images": getattr(state, "base64_encoded_images", []),
            "code_snippets": getattr(
                state, "code_snippets", []
            ),  # Preserve code_snippets
            "analysis_required": getattr(state, "analysis_required", False),  # Preserve for routing
            "llm_provider": getattr(state, "llm_provider", None),
            "llm_model": getattr(state, "llm_model", None),
            # <<< END FIX >>>
        }


def finalize_report(state: SummaryState, config: RunnableConfig):
    """Finalize the summary into a publication-quality document"""

    print(
        f"[finalize_report] START — analysis_required={getattr(state, 'analysis_required', False)}, "
        f"summary_len={len(state.running_summary or '')}, topic={state.research_topic!r}"
    )

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Note: Steering messages that arrive during finalization are queued but don't interrupt the finalization process.
    if hasattr(state, "steering_todo") and state.steering_todo:
        pending_count = len(state.steering_todo.pending_messages)
        if pending_count > 0:
            logger.info(
                f"[finalize_report] {pending_count} steering message(s) queued but not interrupting finalization."
            )

    # All reports go through normal finalization, including database results
    current_summary = state.running_summary or ""
    web_research_results = state.web_research_results or []

    input_content_for_finalization = ""
    using_raw_content = False

    # If running summary is empty, fallback to using raw web results
    if not current_summary.strip() and web_research_results:
        print(
            "FINALIZE_REPORT: Running summary is empty. Using raw web research results."
        )
        using_raw_content = True
        # Combine and clean raw content, similar to generate_report
        all_new_raw_content = (
            "\n\n---\n\n".join(
                item.get("content", "")
                for item in web_research_results
                if isinstance(item, dict) and item.get("content")
            )
            if isinstance(web_research_results, list)
            else "\n\n---\n\n".join(web_research_results)
        )

        base64_pattern = r"data:image/[a-zA-Z]+;base64,[a-zA-Z0-9+/=]+"
        input_content_for_finalization = re.sub(
            base64_pattern, "[Image Data Removed]", all_new_raw_content
        )
        print(
            f"FINALIZE_REPORT: Using combined raw content of length {len(input_content_for_finalization)}"
        )
    else:
        print(
            f"FINALIZE_REPORT: Using existing running summary of length {len(current_summary)}"
        )
        input_content_for_finalization = current_summary

    # If even raw content is empty, we might have an issue, but proceed anyway
    if not input_content_for_finalization.strip():
        print(
            "FINALIZE_REPORT WARNING: Both running summary and raw results are empty!"
        )
        # Assign an empty string or some placeholder if necessary
        input_content_for_finalization = "(No content available to finalize)"

    # Ensure we have source_citations
    source_citations = getattr(state, "source_citations", {})
    if (
        not source_citations
        and hasattr(state, "web_research_results")
        and state.web_research_results
    ):
        # Extract sources from web_research_results if source_citations is empty
        print(
            "SOURCE EXTRACTION: No source_citations found, extracting from web_research_results"
        )
        source_citations = {}
        citation_index = 1

        # Extract sources from each web_research_result
        for result in state.web_research_results:
            if isinstance(result, dict) and "sources" in result:
                result_sources = result.get("sources", [])
                for source in result_sources:
                    if (
                        isinstance(source, dict)
                        and "title" in source
                        and "url" in source
                    ):
                        # Add to source_citations if not already present
                        source_url = source.get("url")
                        found = False
                        for citation_key, citation_data in source_citations.items():
                            if citation_data.get("url") == source_url:
                                found = True
                                break

                        if not found:
                            citation_key = str(citation_index)
                            source_dict = {
                                "title": source["title"],
                                "url": source["url"],
                            }
                            # Preserve source_type if it exists
                            if "source_type" in source:
                                source_dict["source_type"] = source["source_type"]
                            source_citations[citation_key] = source_dict
                            citation_index += 1

        # Update state.source_citations with the newly extracted sources
        state.source_citations = source_citations
        print(
            f"SOURCE EXTRACTION: Extracted {len(source_citations)} sources from web_research_results"
        )

    # If source_citations is still empty, check sources_gathered
    if (
        not source_citations
        and hasattr(state, "sources_gathered")
        and state.sources_gathered
    ):
        print(
            "SOURCE EXTRACTION: No source_citations found, creating from sources_gathered"
        )
        source_citations = {}
        for idx, source_str in enumerate(state.sources_gathered):
            if isinstance(source_str, str) and " : " in source_str:
                try:
                    title, url = source_str.split(" : ", 1)
                    source_citations[str(idx + 1)] = {"title": title, "url": url}
                except Exception as e:
                    print(
                        f"SOURCE EXTRACTION: Error parsing source {source_str}: {str(e)}"
                    )

        # Update state.source_citations
        state.source_citations = source_citations
        print(
            f"SOURCE EXTRACTION: Created {len(source_citations)} source citations from sources_gathered"
        )

    # Create a properly formatted references section
    if source_citations:
        # Format the source citations into a numbered list (These are already deduplicated by generate_numbered_sources)
        numbered_sources = [
            f"{num}. {src['title']}, [{src['url']}]"
            for num, src in sorted(source_citations.items())
        ]
        formatted_sources_for_prompt = "\n".join(numbered_sources)
        print(f"USING {len(numbered_sources)} UNIQUE NUMBERED SOURCES IN FINAL REPORT")

        # Also log sources that were gathered but not included in citations
        cited_urls = set(src["url"] for src in source_citations.values())
        all_source_texts = (
            state.sources_gathered
        )  # This list might still contain duplicates
        unused_sources = []
        seen_unused_urls = set()  # Deduplicate unused sources as well for logging
        for source_text in all_source_texts:
            if " : " in source_text:
                url = source_text.split(" : ", 1)[1].strip()
                if url not in cited_urls and url not in seen_unused_urls:
                    unused_sources.append(source_text)
                    seen_unused_urls.add(url)
        if unused_sources:
            print(
                f"NOTE: {len(unused_sources)} unique sources were gathered but not cited in the final report"
            )

    else:
        # Fallback to simple formatting if no citations were tracked
        print(
            "WARNING: No source citations found, using basic source list from sources_gathered."
        )
        all_sources_raw = state.sources_gathered
        print(
            f"DEBUG: Fallback - processing {len(all_sources_raw)} raw gathered sources."
        )

        # --- START FIX: Deduplicate sources_gathered in fallback ---
        deduplicated_sources = []
        seen_urls_fallback = set()
        for source in all_sources_raw:
            if source and ":" in source:
                try:
                    url = source.split(" : ", 1)[1].strip()
                    if url not in seen_urls_fallback:
                        deduplicated_sources.append(source)  # Keep original format
                        seen_urls_fallback.add(url)
                except Exception:
                    print(f"DEBUG: Fallback - could not parse source for URL: {source}")
                    deduplicated_sources.append(source)  # Keep unparsable ones?
            elif source:  # Keep non-empty, non-parsable sources
                deduplicated_sources.append(source)
        # --- END FIX ---

        formatted_sources_for_prompt = "\n".join(deduplicated_sources)
        print(
            f"DEBUG: Fallback - using {len(deduplicated_sources)} deduplicated sources for prompt."
        )

    # Resolve provider/model from config + state (respects user's frontend selection)
    provider, model = _resolve_provider_model(configurable, state, "finalize_report")

    llm = get_llm_client(provider, model)

    # Generate the finalized summary with date information
    uploaded_knowledge_content = getattr(state, "uploaded_knowledge", None)
    AUGMENT_KNOWLEDGE_CONTEXT = ""
    if uploaded_knowledge_content and uploaded_knowledge_content.strip():
        AUGMENT_KNOWLEDGE_CONTEXT = f"""
USER-PROVIDED EXTERNAL KNOWLEDGE AVAILABLE:
The user has provided external knowledge/documentation that should form the authoritative foundation of your final report. This uploaded knowledge is highly trustworthy and should be given precedence over web search results.

Uploaded Knowledge Preview: {uploaded_knowledge_content[:500]}{'...' if len(uploaded_knowledge_content) > 500 else ''}

Integration Instructions:
- Use uploaded knowledge as the primary structural foundation
- Integrate web research to enhance, validate, or update uploaded knowledge
- Clearly distinguish between uploaded knowledge and web source information
- Give precedence to uploaded knowledge when conflicts arise
"""
    else:
        AUGMENT_KNOWLEDGE_CONTEXT = "No user-provided external knowledge available. Base the final report on web research results."

    system_prompt = finalize_report_instructions.format(
        research_topic=state.research_topic,
        current_date=CURRENT_DATE,
        current_year=CURRENT_YEAR,
        one_year_ago=ONE_YEAR_AGO,
        AUGMENT_KNOWLEDGE_CONTEXT=AUGMENT_KNOWLEDGE_CONTEXT,
    )

    # Construct the human message based on whether we used raw content or the summary
    uploaded_knowledge_section = ""
    if uploaded_knowledge_content and uploaded_knowledge_content.strip():
        uploaded_knowledge_section = f"""

USER-PROVIDED EXTERNAL KNOWLEDGE (HIGHEST AUTHORITY):
------------------------------------------------------------
{uploaded_knowledge_content}

INTEGRATION INSTRUCTIONS: Use the above uploaded knowledge as your primary foundation. The web research content below should complement, validate, or provide recent updates to this authoritative knowledge.

"""

    # CRITICAL: Include todo completion status and user steering intentions in final report
    todo_completion_section = ""
    if hasattr(state, "steering_todo") and state.steering_todo:
        completed_tasks = state.steering_todo.get_completed_tasks()
        pending_tasks = state.steering_todo.get_pending_tasks()
        all_messages = getattr(state.steering_todo, "all_user_messages", [])

        if completed_tasks or pending_tasks or all_messages:
            todo_completion_section = f"""

USER STEERING AND TODO COMPLETION STATUS:
------------------------------------------------------------
The user provided {len(all_messages)} steering messages during research to guide the process.
These messages represent the user's TRUE NEEDS and PRIORITIES for the final report.

COMPLETED RESEARCH TASKS ({len(completed_tasks)} tasks):
"""
            for task in completed_tasks[-10:]:  # Last 10 completed tasks
                todo_completion_section += f"[done] {task.description}\n"
                if task.completed_note:
                    todo_completion_section += f"  > {task.completed_note}\n"

            if pending_tasks:
                todo_completion_section += (
                    f"\n\nREMAINING TASKS NOT COMPLETED ({len(pending_tasks)} tasks):\n"
                )
                for task in pending_tasks[:5]:  # First 5 pending tasks
                    todo_completion_section += (
                        f"[!] {task.description} (Priority: {task.priority})\n"
                    )

            if all_messages:
                todo_completion_section += (
                    f"\n\nUSER'S STEERING MESSAGES (in chronological order):\n"
                )
                for i, msg in enumerate(all_messages[-10:], 1):  # Last 10 messages
                    todo_completion_section += f'{i}. "{msg}"\n'

            todo_completion_section += f"""

CRITICAL INSTRUCTION FOR FINAL REPORT:
Your final report MUST address all completed tasks and respect all user steering messages above.
The report should reflect the user's refined intentions throughout the research process.
If any high-priority pending tasks remain, acknowledge them as areas for future research.
The report should feel like it was written specifically to answer the user's evolving needs.

"""

    if using_raw_content:
        human_message = (
            f"Please create a polished final report with a descriptive title based on the following sources.\n\n"
            f"IMPORTANT: Begin your report with a clear, descriptive title in the format 'Profile of [Person]: [Role/Position]' or similar format appropriate to the topic. For example: 'Profile of Dr. Caiming Xiong: AI Research Leader at Salesforce' or 'State-of-the-Art Data Strategies for Pretraining 7B Parameter LLMs from Scratch'.\n\n"
            f"{uploaded_knowledge_section}"
            f"{todo_completion_section}"
            f"Raw Research Content from Web Search:\n{input_content_for_finalization}\n\n"
            f"Numbered Sources for Citation:\n{formatted_sources_for_prompt}"
        )
    else:
        human_message = (
            f"Please finalize this research summary into a polished document with a descriptive title.\n\n"
            f"IMPORTANT: Begin your report with a clear, descriptive title in the format 'Profile of [Person]: [Role/Position]' or similar format appropriate to the topic. For example: 'Profile of Dr. Caiming Xiong: AI Research Leader at Salesforce' or 'State-of-the-Art Data Strategies for Pretraining 7B Parameter LLMs from Scratch'.\n\n"
            f"{uploaded_knowledge_section}"
            f"{todo_completion_section}"
            f"Working Summary from Web Research:\n{input_content_for_finalization}\n\n"
            f"Numbered Sources for Citation:\n{formatted_sources_for_prompt}"
        )

    # Add visualization information to the prompt if available
    if (
        state.base64_encoded_images
        or hasattr(state, "visualizations")
        and state.visualizations
    ):
        visualization_info = []
        seen_filenames = set()  # Track seen filenames to avoid duplication

        # Add base64 encoded image info
        for idx, img in enumerate(
            state.base64_encoded_images[:5]
        ):  # Limit to first 5 to avoid overwhelming prompt
            # Skip if we've already processed this image
            filename = img.get("filename", "")
            if filename and filename in seen_filenames:
                continue

            if filename:
                seen_filenames.add(filename)

            title = img.get("title", f"Visualization {idx+1}")
            description = img.get("description", "")
            visualization_info.append(
                f"Image {len(visualization_info) + 1}: {title}\nDescription: {description}"
            )

        # Use full visualization objects instead of just paths
        if hasattr(state, "visualizations"):
            for idx, viz in enumerate(state.visualizations[:5]):  # Limit to first 5
                # Skip if we've already processed this image
                if "filename" in viz and viz["filename"] in seen_filenames:
                    continue

                if "filename" in viz:
                    seen_filenames.add(viz["filename"])

                # Extract title from subtask_name or filename
                title = viz.get("subtask_name", "")
                if not title and "filename" in viz:
                    # Generate title from filename if no subtask_name
                    filename = viz["filename"]
                    title_base = os.path.splitext(filename)[0].replace("_", " ").title()
                    title = title_base

                # Build enhanced description with whatever metadata is available
                description_parts = []
                if "description" in viz and viz["description"]:
                    description_parts.append(viz["description"])
                if "chart_type" in viz and viz["chart_type"]:
                    description_parts.append(f"Chart type: {viz['chart_type']}")
                if "data_summary" in viz and viz["data_summary"]:
                    description_parts.append(f"Data summary: {viz['data_summary']}")

                description = "\n".join(description_parts) if description_parts else ""

                # Add to visualization info
                visualization_info.append(
                    f"Image {len(visualization_info) + 1}: {title}\nDescription: {description}"
                )

        # Also add visualization_paths as fallback if we have no visualizations object
        # (This covers the transition period where old code might still be using visualization_paths)
        elif hasattr(state, "visualization_paths") and state.visualization_paths:
            for idx, path in enumerate(state.visualization_paths[:5]):
                # Extract just the filename to check for duplicates
                filename = os.path.basename(path)
                if filename in seen_filenames:
                    continue

                seen_filenames.add(filename)

                title_base = os.path.splitext(filename)[0]
                if len(title_base) > 6 and title_base[-6:].isalnum():
                    title_base = title_base[:-6]

                # Capitalize and replace underscores
                title = title_base.replace("_", " ").title()

                # Use the current count of visualization_info for image numbering to avoid gaps
                visualization_info.append(
                    f"Image {len(visualization_info) + 1}: {title}"
                )

        if visualization_info:
            visualization_prompt = "\n\nAvailable Visualizations:\n" + "\n\n".join(
                visualization_info
            )
            visualization_prompt += "\n\nPlease indicate where these visualizations should be placed in the report by adding [INSERT IMAGE X] markers at appropriate locations in your text. Choose the most relevant locations based on content."
            human_message += visualization_prompt

    result = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_message)]
    )

    # Extract content based on the client type
    if hasattr(result, "content"):
        finalized_summary = result.content
    else:
        finalized_summary = (
            result  # fallback: raw string response
        )

    # Post-process the final report to ensure citation consistency and include References section
    finalized_summary = post_process_report(finalized_summary, source_citations)

    # Enhance the markdown conversion for better HTML styling, particularly for the Table of Contents
    finalized_summary = re.sub(
        r"^## Table of Contents",
        "<h2>Table of Contents</h2>",
        finalized_summary,
        flags=re.MULTILINE,
    )

    # Format date on a new line with proper spacing
    finalized_summary = re.sub(
        r"<h1>(.*?)</h1>\s*(\w+ \d+, \d{4})",
        r'<h1>\1</h1>\n<p class="report-date">\2</p>',
        finalized_summary,
    )

    # Convert horizontal table of contents to vertical list format with clickable anchor links
    # Handle both bullet-separated and dash-separated formats
    def _slugify(text):
        """Convert heading text to a URL-safe slug for anchor links."""
        slug = text.lower().strip()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s]+', '-', slug)
        slug = re.sub(r'-+', '-', slug)
        return slug.strip('-')

    toc_pattern = r"<h2>Table of Contents</h2>\s*\n*([^#<]+?)(?=\n\n|<h2>|##)"
    toc_match = re.search(toc_pattern, finalized_summary, re.DOTALL)
    if toc_match:
        toc_content = toc_match.group(1).strip()

        # First, split by newlines to get individual lines
        lines = [line.strip() for line in toc_content.split("\n") if line.strip()]

        # Then process each line - split by " - " if present
        all_toc_items = []
        for line in lines:
            # Remove leading bullet markers (-, •, *, etc.)
            line = re.sub(r"^[\-•*]\s*", "", line)
            # Strip any existing markdown link syntax: [text](url) → text
            line = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line)

            # Split by " - " to handle concatenated sections
            if " - " in line:
                items = [item.strip() for item in line.split(" - ") if item.strip()]
                all_toc_items.extend(items)
            elif line:
                all_toc_items.append(line)

        # Create properly formatted vertical TOC with clickable anchor links
        if all_toc_items:
            vertical_toc = (
                "<ul>\n"
                + "\n".join([f'<li><a href="#{_slugify(item)}">{item}</a></li>' for item in all_toc_items])
                + "\n</ul>"
            )
            # Replace the entire TOC section
            finalized_summary = re.sub(
                toc_pattern,
                f"<h2>Table of Contents</h2>\n{vertical_toc}",
                finalized_summary,
                flags=re.DOTALL,
            )

    # create a copy for markdown report
    import copy

    markdown_final_summary = copy.deepcopy(finalized_summary)

    # ---- START NEW APPROACH: Process LLM-directed visualization placement ----
    # Define the maximum number of visualizations to embed in the LLM prompt
    MAX_VISUALIZATIONS_TO_EMBED = 5

    # Prepare visualization items for insertion into the main content
    inline_visualizations = []

    # --- START FIX: Re-add definition of base64_images ---
    # Check for base64 encoded images stored in the state
    base64_images = []
    # Try to get base64_encoded_images directly from state
    base64_images = getattr(state, "base64_encoded_images", [])
    if base64_images:
        print(
            f"Found {len(base64_images)} base64-encoded images directly from state for final report"
        )
    else:
        # Try to extract from result_combiner in state (fallback)
        result_combiner = getattr(state, "result_combiner", None)
        if result_combiner and hasattr(result_combiner, "_base64_encoded_images"):
            base64_images = result_combiner._base64_encoded_images
            print(
                f"Found {len(base64_images)} base64-encoded images from ResultCombiner instance (fallback)"
            )
    # --- END FIX ---

    # Add CSS for styling the report
    visualization_css = """
    <style>
    .report-container {
        font-family: Arial, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .report-container h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .report-container h2 {
        font-size: 2rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .report-container h3 {
        font-size: 1.5rem;
        margin-top: 1.5rem;
    }
    .report-container ul {
        list-style-type: none;
        margin-left: 1rem;
        padding-left: 0.5rem;
    }
    .report-container ul li {
        margin-bottom: 0.25rem;
        position: relative;
    }
    .report-container ul li:before {
        content: "•";
        position: absolute;
        left: -1rem;
    }
    .report-container ul ul li:before {
        content: "◦";
    }
    .report-date {
        font-size: 1.1rem;
        margin-top: 0;
        margin-bottom: 2rem;
        color: #555;
    }
    .inline-visualization {
        margin: 1.5rem 0;
        padding: 1rem;
        background-color: #f9f9f9;
        border-radius: 8px;
    }
    .inline-visualization h4 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .inline-visualization img {
        max-width: 100%;
        height: auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        display: block;
        margin: 0 auto;
    }
    </style>
    """

    # Process base64 and file path visualizations into a list
    all_visualizations = []

    # First add any base64 encoded images (these are most reliable)
    for idx, img in enumerate(base64_images):
        try:
            filename = img.get("filename", "")
            title = img.get("title", f"Visualization {idx+1}")
            img_data = img.get("base64_data", "")
            img_format = img.get("format", "png")

            if img_data:
                all_visualizations.append(
                    {
                        "id": idx + 1,
                        "title": title,
                        "html": f'<div class="inline-visualization">'
                        f"<h4>{title}</h4>"
                        f'<img src="data:image/{img_format};base64,{img_data}" alt="{title}" />'
                        f"</div>",
                    }
                )
        except Exception as e:
            print(f"Error processing base64 image: {e}")

    # Process images from the visualizations object (preferred way)
    seen_filenames = set()
    if hasattr(state, "visualizations") and state.visualizations:
        for idx, viz in enumerate(state.visualizations):
            try:
                # Skip if we've already processed this file
                if "filename" in viz and viz["filename"] in seen_filenames:
                    continue

                if "filename" in viz:
                    seen_filenames.add(viz["filename"])

                # Get filepath
                filepath = viz.get("filepath", "")
                if not filepath and "filename" in viz:
                    # Try to reconstruct path if only filename is available
                    filepath = os.path.join("visualizations", viz["filename"])

                if not filepath or not os.path.exists(filepath):
                    print(f"Warning: Visualization file not found: {filepath}")
                    continue

                # Get title from metadata or fallback to filename
                title = viz.get("subtask_name", "")
                if not title and "filename" in viz:
                    # Generate title from filename
                    filename = viz["filename"]
                    title_base = os.path.splitext(filename)[0].replace("_", " ").title()
                    title = title_base

                # Try to read file and encode as base64
                with open(filepath, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode("utf-8")

                # Determine image format
                img_format = os.path.splitext(filepath)[1][1:].lower()
                if img_format not in ["png", "jpg", "jpeg", "gif", "svg"]:
                    img_format = "png"  # Default to png

                # Include description if available
                description_html = ""
                if "description" in viz and viz["description"]:
                    description_html = (
                        f'<p class="visualization-description">{viz["description"]}</p>'
                    )

                # Add visualization to our collection
                all_visualizations.append(
                    {
                        "id": len(base64_images) + idx + 1,
                        "title": title,
                        "html": f'<div class="inline-visualization">'
                        f"<h4>{title}</h4>"
                        f'<img src="data:image/{img_format};base64,{img_data}" alt="{title}" />'
                        f"{description_html}"
                        f"</div>",
                    }
                )
            except Exception as e:
                print(f"Error processing visualization: {e}")

    # Fallback to visualization_paths if no visualizations were processed
    if not any(
        viz.get("id", 0) > len(base64_images) for viz in all_visualizations
    ) and hasattr(state, "visualization_paths"):
        for idx, path in enumerate(state.visualization_paths[:5]):
            # Skip if we've already processed this file
            filename = os.path.basename(path)
            if filename in seen_filenames:
                continue

            seen_filenames.add(filename)

            try:
                # Extract a title from the filename
                title_base = os.path.splitext(filename)[0]
                if len(title_base) > 6 and title_base[-6:].isalnum():
                    title_base = title_base[:-6]

                # Capitalize and replace underscores
                title = title_base.replace("_", " ").title()

                # Use the current count of visualization_info for image numbering to avoid gaps
                visualization_info.append(
                    f"Image {len(visualization_info) + 1}: {title}"
                )
            except Exception as e:
                print(f"Error processing visualization path {path}: {e}")

    print(f"Total visualizations prepared for embedding: {len(all_visualizations)}")

    # Look for LLM-provided placement markers [INSERT IMAGE X] and replace them with visualizations
    placed_visualization_ids = set()

    for viz in all_visualizations:
        viz_id = viz["id"]
        marker = f"[INSERT IMAGE {viz_id}]"

        if marker in finalized_summary:
            # Replace the marker with the visualization HTML
            finalized_summary = finalized_summary.replace(marker, viz["html"])
            placed_visualization_ids.add(viz_id)
            print(f"Placed visualization {viz_id} at LLM-specified location")

    # MODIFIED: Only place visualizations via explicit markers to prevent duplicates
    # Visualizations not placed via markers will be handled by the activity events system
    # and don't need to be added to the report content directly
    if not placed_visualization_ids:
        print(
            f"No visualizations were placed via markers. They will be shown through the activity events system instead."
        )
    elif len(placed_visualization_ids) < len(all_visualizations):
        unplaced_count = len(all_visualizations) - len(placed_visualization_ids)
        print(
            f"{unplaced_count} visualizations were not placed via markers and will be shown through the activity events system instead."
        )

    # CLEANUP: Remove any unreplaced [INSERT IMAGE X] markers from the report
    # This prevents placeholder text from appearing in the final output
    marker_pattern = r"\[INSERT IMAGE \d+\]"
    original_markers = re.findall(marker_pattern, finalized_summary)
    if original_markers:
        finalized_summary = re.sub(marker_pattern, "", finalized_summary)
        print(
            f"Cleaned up {len(original_markers)} unreplaced image markers: {original_markers}"
        )

    # Check if we need a report container
    has_container = '<div class="report-container">' in finalized_summary

    # Make sure we have the container and CSS
    if '<div class="report-container">' not in finalized_summary:
        finalized_summary = f'<div class="report-container">{visualization_css}{finalized_summary}</div>'

    # ---- END NEW APPROACH ----

    # Generate clean markdown version of the report
    markdown_report = generate_markdown_report(markdown_final_summary)
    print(f"Generated markdown report with length: {len(markdown_report)}")
    print(
        f"[finalize_report] DONE — research report ready ({len(finalized_summary)} chars HTML, "
        f"{len(markdown_report)} chars MD). Returning analysis_required={getattr(state, 'analysis_required', False)}"
    )
    # Ensure correct indentation for the return statement
    return {
        "running_summary": finalized_summary,
        "markdown_report": markdown_report,  # Add the clean markdown version
        "web_research_results": [],  # Completely clear web_research_results as it's no longer needed
        "selected_search_tool": state.selected_search_tool,
        "source_citations": source_citations,  # Preserve the source citations
        "visualization_paths": getattr(
            state, "visualization_paths", []
        ),  # Preserve visualization paths from state
        "extra_effort": getattr(state, "extra_effort", False),  # Preserve extra_effort
        "minimum_effort": getattr(
            state, "minimum_effort", False
        ),  # Preserve minimum_effort
        # <<< START FIX: Preserve fields from input state >>>
        "research_topic": state.research_topic,
        "research_loop_count": state.research_loop_count,
        "sources_gathered": state.sources_gathered,
        "knowledge_gap": getattr(state, "knowledge_gap", ""),
        "visualizations": getattr(state, "visualizations", []),
        "base64_encoded_images": getattr(state, "base64_encoded_images", []),
        "code_snippets": getattr(state, "code_snippets", []),  # Preserve code_snippets
        "analysis_required": getattr(state, "analysis_required", False),  # Preserve for routing
        "llm_provider": getattr(state, "llm_provider", None),  # Preserve for run_analysis
        "llm_model": getattr(state, "llm_model", None),  # Preserve for run_analysis
        # <<< END FIX >>>
    }


def generate_markdown_report(report):
    """
    Generate a clean markdown version of the report without HTML elements.
    The output is formatted as a JSON-serializable string suitable for dumping as a JSON field.

    Args:
        report (str): The generated report (may contain HTML)

    Returns:
        str: Clean markdown version of the report formatted for JSON serialization
    """
    # Start with the original report
    markdown_report = report

    # Remove HTML tags and convert to clean markdown
    # Convert HTML headers back to markdown
    markdown_report = re.sub(
        r"<h1[^>]*>(.*?)</h1>", r"# \1", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<h2[^>]*>(.*?)</h2>", r"## \1", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<h3[^>]*>(.*?)</h3>", r"### \1", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<h4[^>]*>(.*?)</h4>", r"#### \1", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<h5[^>]*>(.*?)</h5>", r"##### \1", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<h6[^>]*>(.*?)</h6>", r"###### \1", markdown_report, flags=re.DOTALL
    )

    # Convert HTML lists to markdown
    markdown_report = re.sub(r"<ul[^>]*>", "", markdown_report)
    markdown_report = re.sub(r"</ul>", "", markdown_report)
    markdown_report = re.sub(r"<ol[^>]*>", "", markdown_report)
    markdown_report = re.sub(r"</ol>", "", markdown_report)
    markdown_report = re.sub(
        r"<li[^>]*>(.*?)</li>", r"* \1", markdown_report, flags=re.DOTALL
    )

    # Convert HTML formatting to markdown
    markdown_report = re.sub(
        r"<strong[^>]*>(.*?)</strong>", r"**\1**", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<b[^>]*>(.*?)</b>", r"**\1**", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<em[^>]*>(.*?)</em>", r"*\1*", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<i[^>]*>(.*?)</i>", r"*\1*", markdown_report, flags=re.DOTALL
    )

    # Convert HTML code blocks to markdown
    markdown_report = re.sub(
        r"<code[^>]*>(.*?)</code>", r"`\1`", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<pre[^>]*>(.*?)</pre>", r"```\n\1\n```", markdown_report, flags=re.DOTALL
    )

    # Remove other HTML tags
    markdown_report = re.sub(
        r'<p[^>]*class="report-date"[^>]*>(.*?)</p>',
        r"\1",
        markdown_report,
        flags=re.DOTALL,
    )
    markdown_report = re.sub(
        r"<div[^>]*>(.*?)</div>", r"\1", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<p[^>]*>(.*?)</p>", r"\1", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(r"<br\s*/?>", "\n", markdown_report)

    # Remove any remaining HTML tags (after table conversion)
    markdown_report = re.sub(r"<[^>]+>", "", markdown_report)

    # Remove any leftover HTML escape sequences or artifacts
    markdown_report = re.sub(r"&[a-zA-Z0-9#]+;", "", markdown_report)

    # Clean up any CSS or JavaScript that might have snuck in
    markdown_report = re.sub(
        r"<style[^>]*>.*?</style>", "", markdown_report, flags=re.DOTALL
    )
    markdown_report = re.sub(
        r"<script[^>]*>.*?</script>", "", markdown_report, flags=re.DOTALL
    )

    # Clean up extra whitespace and newlines
    markdown_report = re.sub(
        r"\n\s*\n\s*\n+", "\n\n", markdown_report
    )  # Multiple empty lines to double
    markdown_report = re.sub(
        r"^\s+", "", markdown_report, flags=re.MULTILINE
    )  # Leading whitespace
    markdown_report = markdown_report.strip()

    # Remove base64 image data that might be embedded in the text (if any)
    base64_pattern = r"data:image/[a-zA-Z]+;base64,[a-zA-Z0-9+/=]+"
    markdown_report = re.sub(base64_pattern, "", markdown_report)

    # Remove any HTML attributes that might be left behind
    markdown_report = re.sub(r'class="[^"]*"', "", markdown_report)
    markdown_report = re.sub(r'style="[^"]*"', "", markdown_report)
    markdown_report = re.sub(r'id="[^"]*"', "", markdown_report)

    # Remove HTML entities
    markdown_report = markdown_report.replace("&nbsp;", " ")
    markdown_report = markdown_report.replace("&amp;", "&")
    markdown_report = markdown_report.replace("&lt;", "<")
    markdown_report = markdown_report.replace("&gt;", ">")
    markdown_report = markdown_report.replace("&quot;", '"')
    markdown_report = markdown_report.replace("&#39;", "'")

    # Remove any remaining HTML comments
    markdown_report = re.sub(r"<!--.*?-->", "", markdown_report, flags=re.DOTALL)

    # Convert HTML tables to markdown tables (preserve table structure)
    def convert_html_table_to_markdown(match):
        """Convert a single HTML table to markdown format"""
        table_html = match.group(0)

        # Extract rows
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL)
        if not rows:
            return ""

        markdown_rows = []
        is_header_row = True

        for row in rows:
            # Extract cells (th or td)
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.DOTALL)
            if not cells:
                continue

            # Clean cell content of any remaining HTML
            clean_cells = []
            for cell in cells:
                clean_cell = re.sub(r"<[^>]+>", "", cell).strip()
                clean_cells.append(clean_cell)

            # Create markdown row
            markdown_row = "| " + " | ".join(clean_cells) + " |"
            markdown_rows.append(markdown_row)

            # Add separator after header row
            if is_header_row and clean_cells:
                separator = "| " + " | ".join(["---"] * len(clean_cells)) + " |"
                markdown_rows.append(separator)
                is_header_row = False

        return "\n" + "\n".join(markdown_rows) + "\n"

    # Apply table conversion
    markdown_report = re.sub(
        r"<table[^>]*>.*?</table>",
        convert_html_table_to_markdown,
        markdown_report,
        flags=re.DOTALL,
    )

    # Final cleanup: ensure proper JSON escaping for special characters
    # Replace problematic characters that might break JSON
    markdown_report = markdown_report.replace("\r\n", "\n")  # Normalize line endings
    markdown_report = markdown_report.replace("\r", "\n")  # Normalize line endings

    # Clean up excessive whitespace while preserving intentional formatting
    markdown_report = re.sub(
        r"\n\s*\n\s*\n+", "\n\n", markdown_report
    )  # Multiple empty lines to double
    markdown_report = re.sub(
        r"[ \t]+$", "", markdown_report, flags=re.MULTILINE
    )  # Trailing whitespace
    markdown_report = markdown_report.strip()

    return markdown_report


def post_process_benchmark_answer(answer, source_citations):
    """
    Post-process the benchmark answer to ensure citation consistency and include a References section
    with the specific format: [cite number] title. authors. [link]

    Args:
        answer (str): The generated benchmark answer
        source_citations (dict): Dictionary mapping citation numbers to source metadata

    Returns:
        str: The post-processed answer with properly formatted citations
    """
    if not source_citations:
        return answer  # No citations to check or add

    # Check if a References section already exists in the answer
    references_section_patterns = [
        "References",
        "References:",
        "## References",
        "# References",
        "**References:**",
    ]

    has_references_section = any(
        pattern in answer for pattern in references_section_patterns
    )

    # Create the references section if needed
    if not has_references_section:
        print("Adding missing References section to the benchmark answer")
        # Format the references section with benchmark-specific format
        references_section = "\n\n**References:**\n"

        # Add each reference in the academic format: [cite number] First Author et al. (year) Title. [link]
        for num, src in sorted(source_citations.items()):
            title = src.get("title", "Unknown Title")
            url = src.get("url", "")
            author = src.get("author")
            year = src.get("year")

            # Format: [cite number] First Author et al. (year) Title. [link]
            if author and year:
                references_section += f"[{num}] {author} et al. ({year}) {title}\n"
            elif author:
                references_section += f"[{num}] {author} et al. {title}\n"
            elif year:
                references_section += f"[{num}] ({year}) {title}\n"
            else:
                # Fallback to original format if no author/year available
                references_section += f"[{num}] {title}\n"

        # Append to the answer
        answer += references_section

    # Fix any generic citations that might have been generated
    import re

    # Multiple patterns to catch different variations of generic citations
    generic_citation_patterns = [
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited\s+in\s+the\s+provided\s+research\s+summary",
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited\s+in\s+the\s+research\s+summary",
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited\s+in\s+the\s+provided\s+research",
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited\s+in\s+research",
        r"\[(\d+)\]\s*Source\s+\d+",
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited",
    ]

    all_matches = []
    for pattern in generic_citation_patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        all_matches.extend(matches)

    matches = list(set(all_matches))  # Remove duplicates

    if matches:
        print(f"Fixing {len(matches)} generic citations in benchmark answer")
        for citation_num in matches:
            if citation_num in source_citations:
                src = source_citations[citation_num]
                title = src.get("title", "Unknown Title")
                url = src.get("url", "")
                author = src.get("author")
                year = src.get("year")

                # Replace with proper format
                if author and year:
                    replacement = f"[{citation_num}] {author} et al. ({year}) {title}"
                elif author:
                    replacement = f"[{citation_num}] {author} et al. {title}"
                elif year:
                    replacement = f"[{citation_num}] ({year}) {title}"
                else:
                    replacement = f"[{citation_num}] {title}"

                # Replace all variations of generic citations
                for pattern in generic_citation_patterns:
                    answer = re.sub(
                        pattern.replace(r"(\d+)", citation_num),
                        replacement,
                        answer,
                        flags=re.IGNORECASE,
                    )

    # Check for citation consistency (same logic as regular mode)
    # Get all citation numbers used in the answer
    citation_pattern = r"\[(\d+(?:,\s*\d+)*)\]"
    found_citations = set()

    for match in re.finditer(citation_pattern, answer):
        # Handle both single citations [1] and multiple citations [1,2,3]
        for citation in re.split(r",\s*", match.group(1)):
            if citation.isdigit():
                found_citations.add(citation)  # Keep as string

    # Check which citations were used but not in source_citations
    source_citations_keys = set(str(k) for k in source_citations.keys())
    missing_citations = [c for c in found_citations if c not in source_citations_keys]
    if missing_citations:
        print(
            f"WARNING: Benchmark answer contains citations {missing_citations} not found in source_citations"
        )

    # Check which citations from source_citations were not used
    unused_citations = [c for c in source_citations_keys if c not in found_citations]
    if unused_citations:
        print(
            f"WARNING: Benchmark answer doesn't use citations {unused_citations} from source_citations"
        )

    return answer


def post_process_report(report, source_citations):
    """
    Post-process the report to ensure citation consistency and include a References section
    if it's missing.

    Args:
        report (str): The generated report
        source_citations (dict): Dictionary mapping citation numbers to source metadata

    Returns:
        str: The post-processed report
    """
    # Convert markdown headers to proper HTML for better styling

    # First identify and convert the main title (first # header)
    import re

    title_pattern = r"^#\s+(.*?)$"
    match = re.search(title_pattern, report, re.MULTILINE)
    if match:
        title = match.group(1)
        report = re.sub(
            title_pattern, f"<h1>{title}</h1>", report, count=1, flags=re.MULTILINE
        )

        # Remove any duplicate titles that match exactly
        report = re.sub(f"<h1>{re.escape(title)}</h1>", "", report)
        report = re.sub(f"#\\s+{re.escape(title)}\\s*\n", "", report)

        # Also remove similar titles (those that contain the main title)
        similar_title_pattern = f"<h1>.*?{re.escape(title)}.*?</h1>"
        report = re.sub(similar_title_pattern, "", report)
        report = re.sub(f"#\\s+.*?{re.escape(title)}.*?\\s*\n", "", report)

    # Fix the table of contents formatting if needed

    if not source_citations:
        return report  # No citations to check or add

    # Check if a References section already exists in the report
    references_section_patterns = [
        "References",
        "References:",
        "## References",
        "# References",
    ]

    has_references_section = any(
        pattern in report for pattern in references_section_patterns
    )

    # Create the references section if needed
    if not has_references_section:
        print("Adding missing References section to the report")
        # Format the references section
        references_section = "\n\n──────────────────────────────\nReferences\n\n"

        # Add each reference in order
        for num, src in sorted(source_citations.items()):
            references_section += f"{num}. {src['title']}, [{src['url']}]\n"

        # Append to the report
        report += references_section

    # Fix any generic citations that might have been generated
    # Multiple patterns to catch different variations of generic citations
    generic_citation_patterns = [
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited\s+in\s+the\s+provided\s+research\s+summary",
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited\s+in\s+the\s+research\s+summary",
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited\s+in\s+the\s+provided\s+research",
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited\s+in\s+research",
        r"\[(\d+)\]\s*Source\s+\d+",
        r"\[(\d+)\]\s*Source\s+\d+,\s*as\s+cited",
    ]

    all_matches = []
    for pattern in generic_citation_patterns:
        matches = re.findall(pattern, report, re.IGNORECASE)
        all_matches.extend(matches)

    matches = list(set(all_matches))  # Remove duplicates

    if matches:
        print(f"Fixing {len(matches)} generic citations in report")
        for citation_num in matches:
            if citation_num in source_citations:
                src = source_citations[citation_num]
                title = src.get("title", "Unknown Title")
                url = src.get("url", "")

                # Replace with proper format
                replacement = f"[{citation_num}] {title}"

                # Replace all variations of generic citations
                for pattern in generic_citation_patterns:
                    report = re.sub(
                        pattern.replace(r"(\d+)", citation_num),
                        replacement,
                        report,
                        flags=re.IGNORECASE,
                    )

    # Check for citation consistency
    # Get all citation numbers used in the report
    citation_pattern = r"\[(\d+(?:,\s*\d+)*)\]"
    found_citations = set()

    for match in re.finditer(citation_pattern, report):
        # Handle both single citations [1] and multiple citations [1,2,3]
        for citation in re.split(r",\s*", match.group(1)):
            if citation.isdigit():
                found_citations.add(citation)  # Keep as string

    # Check which citations were used but not in source_citations
    # Ensure source_citations keys are strings if they aren't already (they should be)
    source_citations_keys = set(str(k) for k in source_citations.keys())
    missing_citations = [c for c in found_citations if c not in source_citations_keys]
    if missing_citations:
        print(
            f"WARNING: Report contains citations {missing_citations} not found in source_citations"
        )

    # Check which citations from source_citations were not used
    unused_citations = [c for c in source_citations_keys if c not in found_citations]
    if unused_citations:
        print(
            f"WARNING: Report doesn't use citations {unused_citations} from source_citations"
        )

    return report


def route_research(state: SummaryState, config: RunnableConfig):
    """Determines if research is complete or should continue"""

    configurable = Configuration.from_runnable_config(config)

    # Debug logging
    print(f"ROUTING STATE EXAMINATION:")
    print(f"  - research_complete: {state.research_complete}")
    print(
        f"  - search_query: '{state.search_query if hasattr(state, 'search_query') else ''}'"
    )
    print(
        f"  - research_loop_count: {state.research_loop_count}/{getattr(configurable, 'max_web_research_loops', 'N/A')}"
    )
    print(
        f"  - running_summary length: {len(state.running_summary) if state.running_summary else 0} chars"
    )

    # Check if we've reached the maximum number of research loops
    # Get effort flags from state
    extra_effort = getattr(state, "extra_effort", False)
    minimum_effort = getattr(state, "minimum_effort", False)  # Get minimum_effort flag

    # Get max_loops using the utility function
    max_loops = get_max_loops(
        configurable, extra_effort, minimum_effort, state.benchmark_mode, state.qa_mode
    )  # Pass minimum_effort, benchmark_mode, and qa_mode
    if state.research_loop_count >= max_loops:
        print(f"ROUTING OVERRIDE: Max loops reached ({max_loops}), finalizing report")
        return "finalize_report"

    # BUGFIX: Check LLM's decision about research completeness first - this takes priority
    if state.research_complete:
        print("ROUTING DECISION: Research marked as complete by LLM, finalizing report")
        return "finalize_report"

    # First iteration: always continue research, regardless of what LLM says
    # (Only runs if research_complete is False)
    if state.research_loop_count == 1:
        print(
            "ROUTING OVERRIDE: First iteration - forcing research to continue regardless of flags"
        )
        return "multi_agents_network"

    # If no follow-up query was generated, finalize the report
    if (
        not hasattr(state, "search_query")
        or not state.search_query
        or len(state.search_query.strip()) == 0
    ):
        print("ROUTING DECISION: No follow-up query generated, finalizing report")
        return "finalize_report"

    # Otherwise, continue with research by going directly to the research agent
    # This preserves the carefully crafted query from reflection instead of regenerating it
    print(
        "ROUTING DECISION: Continuing with research, going to multi-agent network with reflection's query"
    )
    return "multi_agents_network"


def route_after_search(
    state: SummaryState,
) -> Literal["generate_report", "reflect_on_report"]:
    """Route after search based on whether we have results or not"""

    # Check if the search_results_empty flag is set
    if getattr(state, "search_results_empty", False):
        print(
            "ROUTING: Search returned no results, skipping summarization and going directly to reflection"
        )
        return "reflect_on_report"

    # Normal flow - proceed to summarization
    print("ROUTING: Search returned results, proceeding to summarization")
    return "generate_report"


# NEW ROUTING FUNCTION
def route_after_multi_agents(
    state: SummaryState,
) -> Literal["generate_report", "reflect_on_report", "finalize_report"]:
    """
    Determines the next step after the multi_agents_network based on minimum_effort
    and search results.
    """
    minimum_effort = getattr(state, "minimum_effort", False)
    if minimum_effort:
        # If minimum effort is requested, skip reflection and go directly to finalize
        print(
            "ROUTING: Minimum effort requested, skipping reflection, finalizing report"
        )
        return "finalize_report"
    else:
        # Otherwise, use the existing routing logic based on search results
        return route_after_search(state)


def generate_answer(state: SummaryState, config: RunnableConfig):
    """
    Generate a concise, fact-based answer for QA and benchmark questions.
    This node is used when qa_mode or benchmark_mode is True.
    """
    print(f"--- ENTERING generate_answer (Loop {state.research_loop_count}) ---")
    print(
        f"[generate_answer] qa_mode={state.qa_mode}, benchmark_mode={state.benchmark_mode}"
    )
    print(f"[generate_answer] research_topic={state.research_topic}")

    # Start timer for performance logging
    start_time = time.time()

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Get LLM client
    provider, model = _resolve_provider_model(configurable, state, "generate_answer")
    llm = get_llm_client(provider, model)

    # Get previous answers with reasoning if available
    previous_answers = []
    if hasattr(state, "previous_answers") and state.previous_answers:
        previous_answers = state.previous_answers

    previous_answers_text = ""
    if previous_answers:
        previous_answers_text = "\n\n".join(
            [
                f"LOOP {idx+1}:\n{answer.get('answer', '')}\nConfidence: {answer.get('confidence', 'UNKNOWN')}\nReasoning: {answer.get('reasoning', 'None provided')}"
                for idx, answer in enumerate(previous_answers)
            ]
        )

    # Use running_summary as the primary context for answer generation
    accumulated_context = getattr(state, "running_summary", "")
    if not accumulated_context:
        print(
            "[generate_answer] No running_summary available. Using research_topic as minimal context."
        )
        accumulated_context = f"Initial query: {state.research_topic}. No information has been gathered yet."

    # Optionally, include any very recent, unsummarized info from web_research_results if it exists
    # (though ideally, this would have been processed by validate_context_sufficiency already)
    recent_unprocessed_info = getattr(state, "web_research_results", [])
    if recent_unprocessed_info:
        recent_text = "\n\n---\n\n".join(
            item.get("content", "")
            for item in recent_unprocessed_info
            if isinstance(item, dict) and item.get("content")
        )
        if recent_text:
            accumulated_context += f"\n\nADDITIONAL RECENTLY FETCHED CONTENT (May not be fully validated):\n{recent_text}"

    print(
        f"[generate_answer] Context for answer generation (running_summary, first 300 chars): {accumulated_context[:300]}..."
    )

    # Generate date constants for time context
    from datetime import datetime

    today = datetime.now()
    current_date = today.strftime("%B %d, %Y")
    current_year = str(today.year)
    one_year_ago = str(today.year - 1)

    # Choose the appropriate prompt based on mode
    if state.benchmark_mode:
        answer_prompt = BENCHMARK_ANSWER_GENERATION_PROMPT
        print(
            f"[generate_answer] Using BENCHMARK mode prompts with full citation processing"
        )
    elif state.qa_mode:
        answer_prompt = QA_ANSWER_GENERATION_PROMPT
        print(f"[generate_answer] Using QA mode prompts")
    else:
        # Fallback to QA mode if neither is specified
        answer_prompt = QA_ANSWER_GENERATION_PROMPT
        print(f"[generate_answer] Fallback to QA mode prompts")

    # Generate focused answer using the selected prompt
    prompt = answer_prompt.format(
        current_date=current_date,
        current_year=current_year,
        one_year_ago=one_year_ago,
        research_topic=state.research_topic,
        web_research_results=accumulated_context,  # Use accumulated_context (running_summary)
        previous_answers_with_reasoning=previous_answers_text,
    )

    print(f"[generate_answer] Sending prompt to {provider}/{model}")

    response = llm.invoke(prompt)

    # Parse the response to extract structured information
    answer_text = response.content

    # Log the raw response
    print(f"[generate_answer] Raw LLM response preview:")
    preview = answer_text[:300] + "..." if len(answer_text) > 300 else answer_text
    print(f"{preview}")

    # Initialize fields
    direct_answer = None
    confidence_level = None
    supporting_evidence = None
    sources = []
    reasoning = None
    missing_info = None

    # Parse the structured response
    direct_answer = None
    confidence_level = None
    supporting_evidence = None
    sources = []
    reasoning = None
    missing_info = None

    # Split response into lines and process
    lines = answer_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Handle different formats for each section
        if (
            line.startswith("# Direct Answer")
            or line.startswith("1. Direct Answer:")
            or line.startswith("Direct Answer:")
        ):
            # Extract answer from same line if present
            if ":" in line:
                direct_answer = line.split(":", 1)[1].strip()
            # Look for answer on next line if not on same line
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if (
                    next_line
                    and not next_line.startswith("#")
                    and not next_line.startswith("2.")
                    and not next_line.startswith("Confidence:")
                ):
                    direct_answer = next_line
                    i += 1  # Skip the next line since we processed it

        elif (
            line.startswith("# Confidence")
            or line.startswith("2. Confidence:")
            or line.startswith("Confidence:")
        ):
            if ":" in line:
                confidence_level = line.split(":", 1)[1].strip()
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if (
                    next_line
                    and not next_line.startswith("#")
                    and not next_line.startswith("3.")
                    and not next_line.startswith("Supporting Evidence:")
                ):
                    confidence_level = next_line
                    i += 1

        elif (
            line.startswith("# Supporting Evidence")
            or line.startswith("3. Supporting Evidence:")
            or line.startswith("Supporting Evidence:")
        ):
            if ":" in line:
                supporting_evidence = line.split(":", 1)[1].strip()
            elif i + 1 < len(lines):
                # Collect multi-line supporting evidence
                evidence_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if (
                        next_line.startswith("#")
                        or next_line.startswith("4.")
                        or next_line.startswith("Sources:")
                        or next_line.startswith("Reasoning:")
                        or next_line.startswith("Missing Information:")
                    ):
                        break
                    if next_line:
                        evidence_lines.append(next_line)
                    j += 1
                if evidence_lines:
                    supporting_evidence = " ".join(evidence_lines)
                    i = j - 1  # Set i to the last processed line

        elif (
            line.startswith("# Sources")
            or line.startswith("4. Sources:")
            or line.startswith("Sources:")
        ):
            # Collect numbered sources
            sources = []
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if (
                    next_line.startswith("#")
                    or next_line.startswith("5.")
                    or next_line.startswith("Reasoning:")
                    or next_line.startswith("Missing Information:")
                ):
                    break
                if next_line and (
                    next_line.startswith("1.")
                    or next_line.startswith("2.")
                    or next_line.startswith("3.")
                    or next_line.startswith("4.")
                    or next_line.startswith("5.")
                    or "http" in next_line
                    or next_line.startswith("-")
                    or next_line.startswith("*")
                ):
                    # Clean up source formatting
                    source = next_line.lstrip("1234567890.-* ").strip()
                    if source:
                        sources.append(source)
                j += 1
            i = j - 1

        elif (
            line.startswith("# Reasoning")
            or line.startswith("5. Reasoning:")
            or line.startswith("Reasoning:")
        ):
            if ":" in line:
                reasoning = line.split(":", 1)[1].strip()
            elif i + 1 < len(lines):
                # Collect multi-line reasoning
                reasoning_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if (
                        next_line.startswith("#")
                        or next_line.startswith("6.")
                        or next_line.startswith("Missing Information:")
                    ):
                        break
                    if next_line:
                        reasoning_lines.append(next_line)
                    j += 1
                if reasoning_lines:
                    reasoning = " ".join(reasoning_lines)
                    i = j - 1

        elif (
            line.startswith("# Missing Information")
            or line.startswith("6. Missing Information:")
            or line.startswith("Missing Information:")
        ):
            if ":" in line:
                missing_info = line.split(":", 1)[1].strip()
            elif i + 1 < len(lines):
                # Collect multi-line missing info
                missing_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if next_line.startswith("#") or next_line.startswith("7."):
                        break
                    if next_line:
                        missing_lines.append(next_line)
                    j += 1
                if missing_lines:
                    missing_info = " ".join(missing_lines)
                    i = j - 1

        i += 1

    # Fallback: if direct_answer is still None, try to extract from the beginning of the response
    if not direct_answer:
        # Look for the first substantial line that's not a header
        for line in lines:
            line = line.strip()
            if (
                line
                and not line.startswith("#")
                and not line.startswith("1.")
                and not line.startswith("Direct Answer:")
                and not line.startswith("Confidence:")
                and len(line) > 10
            ):  # Ensure it's substantial
                direct_answer = line
                break

        # If still no answer, use the first line
        if not direct_answer and lines:
            direct_answer = lines[0].strip()

    # Convert confidence level to numeric value if possible
    numeric_confidence = 0.5  # default mid-range value
    if confidence_level:
        if confidence_level == "HIGH":
            numeric_confidence = 0.9
        elif confidence_level == "MEDIUM":
            numeric_confidence = 0.6
        elif confidence_level == "LOW":
            numeric_confidence = 0.3

    # Create structured answer object
    answer_result = {
        "answer": direct_answer if direct_answer else answer_text.split("\n")[0],
        "confidence": numeric_confidence,
        "confidence_level": confidence_level,
        "supporting_evidence": supporting_evidence,
        "sources": sources,
        "reasoning": reasoning,
        "missing_info": missing_info,
        "full_response": answer_text,
    }

    # Log the structured answer
    print(f"[generate_answer] Structured answer:")
    print(f"  - Answer: {answer_result['answer']}")
    print(
        f"  - Confidence: {answer_result['confidence']} ({answer_result['confidence_level']})"
    )
    print(f"  - Sources: {answer_result['sources']}")
    if reasoning:
        print(
            f"  - Reasoning: {reasoning[:100]}..."
            if len(reasoning) > 100
            else reasoning
        )

    # Create new previous_answers list by copying the existing one and adding the new answer
    previous_answers_updated = list(previous_answers)
    previous_answers_updated.append(answer_result)

    # Log performance timing
    end_time = time.time()
    print(f"[generate_answer] Processing time: {end_time - start_time:.2f} seconds")
    print(f"--- EXITING generate_answer ---\n")

    # Return the updates as a dictionary instead of modifying state directly
    return {
        "benchmark_result": answer_result,
        "qa_mode": state.qa_mode,
        "benchmark_mode": state.benchmark_mode,
        "research_loop_count": getattr(state, "research_loop_count", 0) + 1,
        "previous_answers": previous_answers_updated,
    }


def reflect_answer(state: SummaryState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Reflect on the generated answer and determine if more research is needed.
    """
    print(f"--- REFLECTION START (Loop {state.research_loop_count}) ---")
    print(f"[reflect_answer] benchmark_mode={state.benchmark_mode}")

    # Start timer for performance logging
    start_time = time.time()

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Get LLM client
    provider, model = _resolve_provider_model(configurable, state, "reflect_answer")
    llm = get_llm_client(provider, model)

    # Get current answer and research state
    current_answer = ""
    benchmark_result = getattr(state, "benchmark_result", {})
    if benchmark_result:
        # Use full structured response if available
        if "full_response" in benchmark_result:
            current_answer = benchmark_result["full_response"]
        else:
            # Fall back to simple format
            answer_text = benchmark_result.get("answer", "")
            confidence = benchmark_result.get("confidence", 0.0)
            sources = benchmark_result.get("sources", [])
            current_answer = f"Answer: {answer_text}\nConfidence: {confidence}\nSources: {', '.join(sources)}"

    # Log the current answer for reflection
    print(f"[reflect_answer] Current answer to reflect on:")
    preview = (
        current_answer[:200] + "..." if len(current_answer) > 200 else current_answer
    )
    print(f"  {preview}")

    # Get effort flags and max loops for the combined reflection prompt
    extra_effort = getattr(state, "extra_effort", False)
    minimum_effort = getattr(state, "minimum_effort", False)

    # Get max loops using the utility function
    max_loops = get_max_loops(
        configurable, extra_effort, minimum_effort, state.benchmark_mode, state.qa_mode
    )
    research_loop_count = state.research_loop_count

    print(f"[reflect_answer] Research statistics:")
    print(f"  - Current loop: {research_loop_count} of maximum {max_loops}")
    print(f"  - Extra effort: {extra_effort}")
    print(f"  - Minimum effort: {minimum_effort}")

    # Get web_research_results safely
    web_research_results = getattr(state, "web_research_results", [])

    # Generate date constants for time context
    from datetime import datetime

    today = datetime.now()
    current_date = today.strftime("%B %d, %Y")
    current_year = str(today.year)
    one_year_ago = str(today.year - 1)

    # Choose the appropriate reflection prompt based on mode
    if state.benchmark_mode:
        reflection_prompt = BENCHMARK_ANSWER_REFLECTION_PROMPT
        print(f"[reflect_answer] Using BENCHMARK mode reflection prompts")
    elif state.qa_mode:
        reflection_prompt = QA_ANSWER_REFLECTION_PROMPT
        print(f"[reflect_answer] Using QA mode reflection prompts")
    else:
        # Fallback to QA mode
        reflection_prompt = QA_ANSWER_REFLECTION_PROMPT
        print(f"[reflect_answer] Fallback to QA mode reflection prompts")

    # Prepare reflection prompt with additional parameters including time context
    prompt = reflection_prompt.format(
        current_date=current_date,
        current_year=current_year,
        one_year_ago=one_year_ago,
        research_topic=state.research_topic,
        current_answer=current_answer,
        web_research_results=(
            web_research_results
            if web_research_results
            else "No research results available yet."
        ),
        research_loop_count=research_loop_count,
        max_loops=max_loops,
    )

    print(f"[reflect_answer] Sending reflection prompt to {provider}/{model}")

    response = llm.invoke(prompt)

    # Parse reflection response
    reflection_text = response.content

    # Log the raw reflection response
    print(f"[reflect_answer] Raw reflection response preview:")
    preview = (
        reflection_text[:300] + "..." if len(reflection_text) > 300 else reflection_text
    )
    print(f"{preview}")

    # Initialize the reflection result
    decision = None
    confidence = None
    missing_facts = []
    follow_up = None

    # Parse the structured response - handle both JSON and text formats
    should_continue = False
    justification = ""

    # Try to parse as JSON first (since prompt asks for function call format)
    try:
        import json

        # Look for JSON in the response
        json_start = reflection_text.find("{")
        json_end = reflection_text.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            json_text = reflection_text[json_start:json_end]
            parsed_json = json.loads(json_text)

            # Extract decision from JSON structure
            if "evaluation" in parsed_json:
                evaluation = parsed_json["evaluation"]

                # Look for final decision
                if "final_decision" in evaluation:
                    final_decision = evaluation["final_decision"]
                    continue_research = final_decision.get(
                        "continueResearch", final_decision.get("continue_research", "")
                    )

                    if isinstance(continue_research, str):
                        should_continue = continue_research.lower() in [
                            "yes",
                            "true",
                            "1",
                        ]
                    elif isinstance(continue_research, bool):
                        should_continue = continue_research

                    justification = final_decision.get("justification", "")
                    follow_up = final_decision.get(
                        "followUpQuery", final_decision.get("follow_up_query", "")
                    )

                # If no final_decision, look for other decision indicators
                if not justification and "answer_quality" in evaluation:
                    answer_quality = evaluation["answer_quality"]
                    # If answer has logical flaws or inappropriate confidence, continue research
                    logical_flaws = answer_quality.get(
                        "logical_flaws", answer_quality.get("logicalFlaws", "")
                    )
                    confidence_appropriate = answer_quality.get(
                        "confidence_appropriate",
                        answer_quality.get("confidenceAppropriate", ""),
                    )

                    if logical_flaws and logical_flaws.lower() == "yes":
                        should_continue = True
                        justification = (
                            "Answer contains logical flaws that need to be addressed"
                        )
                    elif (
                        confidence_appropriate
                        and confidence_appropriate.lower() == "no"
                    ):
                        should_continue = True
                        justification = "Confidence level is not appropriate for the evidence provided"

                # Extract missing facts if available
                if "evidence_evaluation" in evaluation or "evidence" in evaluation:
                    evidence_eval = evaluation.get(
                        "evidence_evaluation", evaluation.get("evidence", {})
                    )
                    missing_info = evidence_eval.get(
                        "missingCriticalInformation",
                        evidence_eval.get(
                            "missing_critical_information",
                            evidence_eval.get("missing_critical_info", ""),
                        ),
                    )
                    if missing_info and missing_info.lower() == "yes":
                        # Try to extract what's missing from justification or other fields
                        if not justification:
                            justification = (
                                "Critical information is missing from the sources"
                            )
                        if not missing_facts:
                            missing_facts = [justification]

            print(f"[reflect_answer] Successfully parsed JSON response")
            print(f"[reflect_answer] Extracted continue_research: {should_continue}")
            print(f"[reflect_answer] Extracted justification: {justification}")

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"[reflect_answer] Failed to parse JSON response: {e}")
        print(f"[reflect_answer] Falling back to text pattern parsing")

        # Fall back to text pattern parsing
        for line in reflection_text.split("\n"):
            line = line.strip()
            if line.startswith("DECISION:"):
                decision = line.replace("DECISION:", "").strip()
            elif line.startswith("Confidence:"):
                confidence = line.replace("Confidence:", "").strip()
            elif line.startswith("Missing facts:"):
                missing_facts_text = line.replace("Missing facts:", "").strip()
                missing_facts = (
                    [fact.strip() for fact in missing_facts_text.split(",")]
                    if missing_facts_text
                    else []
                )
            elif line.startswith("Follow-up query:"):
                follow_up = line.replace("Follow-up query:", "").strip()
            elif "Should research continue?" in line:
                if "Yes" in line:
                    should_continue = True
                elif "No" in line:
                    should_continue = False
            elif line.startswith("Justification:"):
                justification = line.replace("Justification:", "").strip()

        # If no clear decision was found, check for key phrases in the response
        if should_continue is None:
            if (
                "needs more research" in reflection_text.lower()
                or "continue research" in reflection_text.lower()
            ):
                should_continue = True
                print(
                    f"[reflect_answer] No explicit continue/terminate decision found, defaulting to CONTINUE based on text analysis"
                )
            elif (
                "complete" in reflection_text.lower()
                or "sufficient" in reflection_text.lower()
            ):
                should_continue = False
                print(
                    f"[reflect_answer] No explicit continue/terminate decision found, defaulting to TERMINATE based on text analysis"
                )
            else:
                # Default to continuing research if unclear
                should_continue = True
                print(
                    f"[reflect_answer] No explicit continue/terminate decision found, defaulting to CONTINUE as safest option"
                )

    # Create a dictionary for the reflection result
    reflection_result = {
        "should_continue": should_continue,
        "justification": justification,
        "missing_facts": missing_facts,
        "follow_up_query": follow_up,
        "loop_count": research_loop_count,
    }

    # Get existing reflection history safely
    reflection_history = getattr(state, "reflection_history", [])

    # Create a new reflection history list with the current reflection added
    reflection_history_updated = list(reflection_history)
    reflection_history_updated.append(reflection_result)

    # Log the reflection decision and next steps
    print(
        f"[reflect_answer] REFLECTION DECISION: {'Continue research' if should_continue else 'Complete research'}"
    )
    print(f"  - Justification: {justification}")

    if missing_facts:
        print(f"  - Missing facts identified:")
        for fact in missing_facts:
            print(f"    * {fact}")
    else:
        print(f"  - No specific missing facts identified")

    if follow_up:
        print(f'  - Follow-up query: "{follow_up}"')
    else:
        print(f"  - No follow-up query generated")

    print(f"  - Research complete: {not should_continue}")
    print(f"  - Research loop count: {research_loop_count}")

    # Log performance timing
    end_time = time.time()
    print(f"[reflect_answer] Processing time: {end_time - start_time:.2f} seconds")
    print(f"--- REFLECTION END ---\n")

    # Return the updates as a dictionary instead of modifying state directly
    return {
        "research_complete": not should_continue,
        "knowledge_gap": "\n".join(missing_facts) if missing_facts else "",
        "search_query": follow_up if follow_up else "",
        "reflection_history": reflection_history_updated,
    }


def route_after_multi_agents_benchmark(
    state: SummaryState,
) -> Literal["generate_answer", "reflect_answer", "finalize_answer"]:
    """
    Determines the next step after the multi_agents_network in benchmark mode.
    """
    minimum_effort = getattr(state, "minimum_effort", False)
    if minimum_effort:
        print(
            "ROUTING: Minimum effort requested, skipping reflection, finalizing answer"
        )
        return "finalize_answer"
    else:
        # Check if search returned results
        if getattr(state, "search_results_empty", False):
            print("ROUTING: Search returned no results, going directly to reflection")
            return "reflect_answer"
        else:
            print("ROUTING: Search returned results, proceeding to answer generation")
            return "generate_answer"


def route_after_generate_answer(state: SummaryState) -> Literal["reflect_answer"]:
    """Route after generating answer to reflection"""
    print("ROUTING: Moving to answer reflection")
    return "reflect_answer"


def route_after_reflect_answer(state: SummaryState, config: RunnableConfig):
    """Determines if research should continue or answer should be finalized"""
    # Create default config if none provided
    if not config:
        config = {"configurable": {"max_web_research_loops": 3}}

    configurable = Configuration.from_runnable_config(config)

    # Get effort flags from state
    extra_effort = getattr(state, "extra_effort", False)
    minimum_effort = getattr(state, "minimum_effort", False)

    print(f"ROUTING STATE CHECK:")
    print(f"  - research_complete: {state.research_complete}")
    print(f"  - research_loop_count: {state.research_loop_count}")
    print(
        f"  - has search_query: {hasattr(state, 'search_query') and bool(state.search_query)}"
    )
    print(f"  - extra_effort: {extra_effort}")
    print(f"  - minimum_effort: {minimum_effort}")

    # Get max_loops using the utility function
    env_max_loops = os.environ.get("MAX_WEB_RESEARCH_LOOPS")
    if env_max_loops:
        print(f"  - Reading MAX_WEB_RESEARCH_LOOPS from environment: {env_max_loops}")

    max_loops = get_max_loops(
        configurable, extra_effort, minimum_effort, state.benchmark_mode, state.qa_mode
    )
    print(
        f"  - Using max_loops={max_loops} (extra_effort={extra_effort}, base={configurable.max_web_research_loops or 3})"
    )

    # First check if we've hit max loops
    if state.research_loop_count >= max_loops:
        print(f"ROUTING OVERRIDE: Max loops reached ({max_loops}), finalizing answer")
        return "finalize_answer"

    # If research is marked as complete, finalize the answer
    if state.research_complete:
        print("ROUTING DECISION: Research marked as complete, finalizing answer")
        return "finalize_answer"

    # If minimum effort is requested, skip further research and finalize
    if minimum_effort:
        print("ROUTING DECISION: Minimum effort requested, finalizing answer")
        return "finalize_answer"

    # If we have high confidence in our current answer, finalize it
    if hasattr(state, "benchmark_result") and state.benchmark_result:
        confidence = state.benchmark_result.get("confidence", 0)
        confidence_level = state.benchmark_result.get("confidence_level", "")

        # Check if confidence exceeds threshold from config
        confidence_threshold = 0.8  # Default threshold
        if hasattr(state, "config") and state.config:
            benchmark_config = state.config.get("benchmark", {})
            if "confidence_threshold" in benchmark_config:
                confidence_threshold = benchmark_config.get("confidence_threshold")

        if confidence >= confidence_threshold or confidence_level == "HIGH":
            print(
                f"ROUTING DECISION: High confidence answer ({confidence} >= {confidence_threshold}), finalizing"
            )
            return "finalize_answer"

    # Check if we have a search query to continue research
    has_search_query = hasattr(state, "search_query") and bool(state.search_query)

    # If no search query but we need more research, generate a default one based on state
    if not has_search_query:
        # Generate a search query based on the missing information or follow up
        if hasattr(state, "reflection_history") and state.reflection_history:
            latest_reflection = state.reflection_history[-1]
            missing_facts = latest_reflection.get("missing_facts", [])

            if missing_facts:
                # Use missing facts to formulate a query
                missing_facts_text = ", ".join(missing_facts)
                state.search_query = f"{state.research_topic} {missing_facts_text}"
                state.knowledge_gap = missing_facts_text
                print(
                    f"ROUTING DECISION: Generated search query from missing facts: {state.search_query}"
                )
            else:
                # Create a generic search query from the research topic
                state.search_query = f"{state.research_topic} detailed information"
                state.knowledge_gap = "Need more comprehensive information"
                print(
                    f"ROUTING DECISION: Generated generic search query: {state.search_query}"
                )
        else:
            # Create a generic search query from the research topic
            state.search_query = f"{state.research_topic} detailed information"
            state.knowledge_gap = "Need more comprehensive information"
            print(
                f"ROUTING DECISION: Generated default search query: {state.search_query}"
            )

        return "multi_agents_network"

    # Continue research with the existing search query
    print("ROUTING DECISION: Research not complete, continuing with existing query")
    return "multi_agents_network"


def verify_answer(state: SummaryState, config: RunnableConfig):
    """
    Verify the generated answer against the expected benchmark answer.
    """
    print("[verify_answer] Starting answer verification")

    # Get the benchmark result
    result = state.benchmark_result
    if not result:
        print("[verify_answer] No benchmark result found to verify")
        return state

    # Get expected answer from config
    expected = state.config.get("benchmark", {}).get("expected_answer")
    if not expected:
        print("[verify_answer] No expected answer found in config")
        return state

    # Parse the generated answer
    answer_lines = result.get("answer", "").split("\n")
    parsed_answer = None
    confidence = result.get("confidence", 0.0)
    sources = result.get("sources", [])

    for line in answer_lines:
        if line.strip():
            parsed_answer = line.strip()
            break

    if not parsed_answer:
        print("[verify_answer] Could not parse generated answer")
        return state

    # Compare answers
    is_correct = parsed_answer.lower() == expected.lower()
    print(f"[verify_answer] Generated answer: {parsed_answer}")
    print(f"[verify_answer] Expected answer: {expected}")
    print(f"[verify_answer] Match: {is_correct}")
    print(f"[verify_answer] Confidence: {confidence}")
    print(f"[verify_answer] Sources: {sources}")

    # Update state with verification results
    state.benchmark_result = {
        "answer": parsed_answer,
        "expected": expected,
        "is_correct": is_correct,
        "confidence": confidence,
        "sources": sources,
    }

    return state


def finalize_answer(state: SummaryState, config: RunnableConfig):
    """
    Finalize the answer for benchmark questions using all research findings.
    """
    print("[finalize_answer] Starting answer finalization")
    print(f"[finalize_answer] benchmark_mode={state.benchmark_mode}")

    # Start timer for performance logging
    start_time = time.time()

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Get LLM client
    provider, model = _resolve_provider_model(configurable, state, "finalize_answer")
    llm = get_llm_client(provider, model)

    # Get all previous answers with reasoning from the benchmark workflow
    previous_answers = getattr(state, "previous_answers", [])
    reflection_history = getattr(state, "reflection_history", [])

    print(
        f"[finalize_answer] Processing {len(previous_answers)} previous answers and {len(reflection_history)} reflections"
    )

    # Debug: Log the actual data structures
    if previous_answers:
        print(f"[finalize_answer] Previous answers structure:")
        for i, answer in enumerate(previous_answers):
            print(
                f"  Answer {i+1}: {list(answer.keys()) if isinstance(answer, dict) else type(answer)}"
            )
            if isinstance(answer, dict):
                print(f"    - answer: {answer.get('answer', 'MISSING')[:100]}...")
                print(
                    f"    - confidence_level: {answer.get('confidence_level', 'MISSING')}"
                )
                print(
                    f"    - reasoning: {answer.get('reasoning', 'MISSING')[:100] if answer.get('reasoning') else 'MISSING'}..."
                )
    else:
        print(f"[finalize_answer] No previous answers found in state")

    if reflection_history:
        print(f"[finalize_answer] Reflection history structure:")
        for i, reflection in enumerate(reflection_history):
            print(
                f"  Reflection {i+1}: {list(reflection.keys()) if isinstance(reflection, dict) else type(reflection)}"
            )
            if isinstance(reflection, dict):
                print(
                    f"    - justification: {reflection.get('justification', 'MISSING')[:100]}..."
                )
                print(
                    f"    - missing_facts: {reflection.get('missing_facts', 'MISSING')}"
                )
    else:
        print(f"[finalize_answer] No reflection history found in state")

    # Format all answers with reasoning for the FINAL_ANSWER_PROMPT
    all_answers_with_reasoning = ""
    if previous_answers:
        for i, answer in enumerate(previous_answers):
            loop_num = i + 1

            # Handle both dict and non-dict answer formats
            if isinstance(answer, dict):
                answer_text = answer.get("answer", "No answer provided")
                confidence = answer.get(
                    "confidence_level", answer.get("confidence", "UNKNOWN")
                )
                reasoning = answer.get("reasoning", "No reasoning provided")
                supporting_evidence = answer.get(
                    "supporting_evidence", "No evidence provided"
                )
                sources = answer.get("sources", [])
            else:
                # Fallback for non-dict answers
                answer_text = str(answer)
                confidence = "UNKNOWN"
                reasoning = "No reasoning provided"
                supporting_evidence = "No evidence provided"
                sources = []

            all_answers_with_reasoning += f"RESEARCH LOOP {loop_num}:\n"
            all_answers_with_reasoning += f"Answer: {answer_text}\n"
            all_answers_with_reasoning += f"Confidence: {confidence}\n"
            all_answers_with_reasoning += (
                f"Supporting Evidence: {supporting_evidence}\n"
            )
            all_answers_with_reasoning += (
                f"Sources: {', '.join(sources) if sources else 'No sources listed'}\n"
            )
            all_answers_with_reasoning += f"Reasoning: {reasoning}\n"

            # Add reflection for this loop if available
            if i < len(reflection_history) and isinstance(reflection_history[i], dict):
                reflection = reflection_history[i]
                justification = reflection.get(
                    "justification", "No reflection justification"
                )
                missing_facts = reflection.get("missing_facts", [])
                all_answers_with_reasoning += f"Reflection: {justification}\n"
                if missing_facts:
                    all_answers_with_reasoning += (
                        f"Missing Facts Identified: {', '.join(missing_facts)}\n"
                    )
            else:
                all_answers_with_reasoning += (
                    f"Reflection: No reflection available for this loop\n"
                )

            all_answers_with_reasoning += "\n---\n\n"
    else:
        all_answers_with_reasoning = "No previous research answers available."

    # Get the most recent web research results as final context
    web_research_results = getattr(state, "web_research_results", [])
    final_search_results = ""
    if web_research_results:
        # Convert web_research_results to text format for the prompt
        if isinstance(web_research_results, list):
            final_search_results = "\n\n---\n\n".join(
                item.get("content", str(item))
                for item in web_research_results
                if isinstance(item, dict) and item.get("content")
            )
        else:
            final_search_results = str(web_research_results)

    # Use running_summary if available as additional context
    running_summary = getattr(state, "running_summary", "")
    if running_summary:
        final_search_results = f"ACCUMULATED RESEARCH SUMMARY:\n{running_summary}\n\n---\n\nLATEST SEARCH RESULTS:\n{final_search_results}"

    # Add source citations context for benchmark mode
    source_citations = getattr(state, "source_citations", {})
    if source_citations and state.benchmark_mode:
        citations_context = "\n\n---\n\nAVAILABLE SOURCES FOR CITATION:\n"
        for cite_num, cite_data in sorted(source_citations.items()):
            title = cite_data.get("title", "Unknown Title")
            url = cite_data.get("url", "No URL")
            author = cite_data.get("author")
            year = cite_data.get("year")

            # Format with author and year if available (academic style)
            if author and year:
                citations_context += f"[{cite_num}] {author} et al. ({year}) {title}\n"
            elif author:
                citations_context += f"[{cite_num}] {author} et al. {title}\n"
            elif year:
                citations_context += f"[{cite_num}] ({year}) {title}\n"
            else:
                citations_context += f"[{cite_num}] {title}\n"
        final_search_results += citations_context

    if not final_search_results:
        final_search_results = "No search results available."

    print(f"[finalize_answer] Context lengths:")
    print(f"  - All answers with reasoning: {len(all_answers_with_reasoning)} chars")
    print(f"  - Final search results: {len(final_search_results)} chars")

    # Generate date constants for time context
    from datetime import datetime

    today = datetime.now()
    current_date = today.strftime("%B %d, %Y")
    current_year = str(today.year)
    one_year_ago = str(today.year - 1)

    # Choose the appropriate final answer prompt based on mode
    if state.benchmark_mode:
        final_prompt = BENCHMARK_FINAL_ANSWER_PROMPT
        print(f"[finalize_answer] Using BENCHMARK mode final answer prompts")
    elif state.qa_mode:
        final_prompt = QA_FINAL_ANSWER_PROMPT
        print(f"[finalize_answer] Using QA mode final answer prompts")
    else:
        # Fallback to QA mode
        final_prompt = QA_FINAL_ANSWER_PROMPT
        print(f"[finalize_answer] Fallback to QA mode final answer prompts")

    # Use the selected FINAL_ANSWER_PROMPT
    prompt = final_prompt.format(
        current_date=current_date,
        current_year=current_year,
        one_year_ago=one_year_ago,
        research_topic=state.research_topic,
        all_answers_with_reasoning=all_answers_with_reasoning,
        web_research_results=final_search_results,
    )

    print(f"[finalize_answer] Sending FINAL_ANSWER_PROMPT to {provider}/{model}")

    response = llm.invoke(prompt)

    # Parse the response to extract structured information
    answer_text = response.content if hasattr(response, "content") else str(response)

    # Log the raw response
    print(f"[finalize_answer] Raw LLM response preview:")
    preview = answer_text[:300] + "..." if len(answer_text) > 300 else answer_text
    print(f"{preview}")

    # Initialize fields for parsing the FINAL_ANSWER_PROMPT format
    direct_answer = None
    confidence_level = None
    key_evidence = None
    sources = []
    limitations = None

    # Parse the structured response according to FINAL_ANSWER_PROMPT format
    for line in answer_text.split("\n"):
        line = line.strip()
        if (
            line.startswith("# Direct Answer:")
            or line.startswith("1. Direct Answer:")
            or line.startswith("## Direct Answer:")
            or line.startswith("Direct Answer:")
        ):
            direct_answer = (
                line.replace("# Direct Answer:", "")
                .replace("1. Direct Answer:", "")
                .replace("## Direct Answer:", "")
                .replace("Direct Answer:", "")
                .strip()
            )
        elif (
            line.startswith("# Overall Confidence:")
            or line.startswith("2. Overall Confidence:")
            or line.startswith("## Overall Confidence:")
            or line.startswith("Overall Confidence:")
        ):
            confidence_level = (
                line.replace("# Overall Confidence:", "")
                .replace("2. Overall Confidence:", "")
                .replace("## Overall Confidence:", "")
                .replace("Overall Confidence:", "")
                .strip()
            )
        elif (
            line.startswith("# Key Evidence:")
            or line.startswith("3. Key Evidence:")
            or line.startswith("## Key Evidence:")
            or line.startswith("Key Evidence:")
        ):
            key_evidence = (
                line.replace("# Key Evidence:", "")
                .replace("3. Key Evidence:", "")
                .replace("## Key Evidence:", "")
                .replace("Key Evidence:", "")
                .strip()
            )
        elif (
            line.startswith("# Sources:")
            or line.startswith("4. Sources:")
            or line.startswith("## Sources:")
            or line.startswith("Sources:")
        ):
            sources_text = (
                line.replace("# Sources:", "")
                .replace("4. Sources:", "")
                .replace("## Sources:", "")
                .replace("Sources:", "")
                .strip()
            )
            sources = [src.strip() for src in sources_text.split("\n") if src.strip()]
        elif (
            line.startswith("# Limitations:")
            or line.startswith("5. Limitations:")
            or line.startswith("## Limitations:")
            or line.startswith("Limitations:")
        ):
            limitations = (
                line.replace("# Limitations:", "")
                .replace("5. Limitations:", "")
                .replace("## Limitations:", "")
                .replace("Limitations:", "")
                .strip()
            )

    # If parsing didn't work well, try a more comprehensive approach
    if not direct_answer or not confidence_level:
        lines = answer_text.split("\n")
        current_section = None
        content_lines = []

        for i, line in enumerate(lines):
            line = line.strip()

            # Identify section headers
            if (
                line.startswith("# Direct Answer")
                or line.startswith("## Direct Answer")
                or line.startswith("1. Direct Answer")
                or line.startswith("Direct Answer")
            ):
                if current_section == "direct_answer" and content_lines:
                    direct_answer = " ".join(content_lines).strip()
                current_section = "direct_answer"
                content_lines = []
                # Check if answer is on the same line
                if ":" in line:
                    after_colon = line.split(":", 1)[1].strip()
                    if after_colon:
                        content_lines.append(after_colon)

            elif (
                line.startswith("# Overall Confidence")
                or line.startswith("## Overall Confidence")
                or line.startswith("2. Overall Confidence")
                or line.startswith("Overall Confidence")
            ):
                if current_section == "direct_answer" and content_lines:
                    direct_answer = " ".join(content_lines).strip()
                elif current_section == "confidence" and content_lines:
                    confidence_level = " ".join(content_lines).strip()
                current_section = "confidence"
                content_lines = []
                # Check if confidence is on the same line
                if ":" in line:
                    after_colon = line.split(":", 1)[1].strip()
                    if after_colon:
                        content_lines.append(after_colon)

            elif (
                line.startswith("# Key Evidence")
                or line.startswith("## Key Evidence")
                or line.startswith("3. Key Evidence")
                or line.startswith("Key Evidence")
            ):
                if current_section == "confidence" and content_lines:
                    confidence_level = " ".join(content_lines).strip()
                elif current_section == "evidence" and content_lines:
                    key_evidence = " ".join(content_lines).strip()
                current_section = "evidence"
                content_lines = []
                # Check if evidence is on the same line
                if ":" in line:
                    after_colon = line.split(":", 1)[1].strip()
                    if after_colon:
                        content_lines.append(after_colon)

            elif (
                line.startswith("# Sources")
                or line.startswith("## Sources")
                or line.startswith("4. Sources")
                or line.startswith("Sources")
            ):
                if current_section == "evidence" and content_lines:
                    key_evidence = " ".join(content_lines).strip()
                elif current_section == "sources" and content_lines:
                    sources = [src.strip() for src in content_lines if src.strip()]
                current_section = "sources"
                content_lines = []
                # Check if sources are on the same line
                if ":" in line:
                    after_colon = line.split(":", 1)[1].strip()
                    if after_colon:
                        content_lines.append(after_colon)

            elif (
                line.startswith("# Limitations")
                or line.startswith("## Limitations")
                or line.startswith("5. Limitations")
                or line.startswith("Limitations")
            ):
                if current_section == "sources" and content_lines:
                    sources = [src.strip() for src in content_lines if src.strip()]
                elif current_section == "limitations" and content_lines:
                    limitations = " ".join(content_lines).strip()
                current_section = "limitations"
                content_lines = []
                # Check if limitations are on the same line
                if ":" in line:
                    after_colon = line.split(":", 1)[1].strip()
                    if after_colon:
                        content_lines.append(after_colon)

            elif line and not line.startswith("#") and current_section:
                # This is content for the current section
                content_lines.append(line)

            # Handle the last section
        if current_section == "direct_answer" and content_lines:
            direct_answer = " ".join(content_lines).strip()
        elif current_section == "confidence" and content_lines:
            confidence_level = " ".join(content_lines).strip()
        elif current_section == "evidence" and content_lines:
            key_evidence = " ".join(content_lines).strip()
        elif current_section == "sources" and content_lines:
            sources = [src.strip() for src in content_lines if src.strip()]
        elif current_section == "limitations" and content_lines:
            limitations = " ".join(content_lines).strip()

    # Convert confidence level to numeric value if possible
    numeric_confidence = 0.5  # default mid-range value
    if confidence_level:
        if confidence_level.upper() == "HIGH":
            numeric_confidence = 0.9
        elif confidence_level.upper() == "MEDIUM":
            numeric_confidence = 0.6
        elif confidence_level.upper() == "LOW":
            numeric_confidence = 0.3

    # If direct answer wasn't parsed, use the first meaningful line as the answer
    if not direct_answer and answer_text:
        lines = answer_text.split("\n")
        for line in lines:
            line = line.strip()
            if (
                line
                and not line.startswith("1.")
                and not line.startswith("2.")
                and not line.startswith("3.")
                and not line.startswith("4.")
                and not line.startswith("5.")
            ):
                direct_answer = line
                break

        # If still no answer, use the first line
        if not direct_answer:
            direct_answer = (
                lines[0].strip() if lines else "No answer could be extracted"
            )

    # Verify against expected answer if available in benchmark config
    expected_answer = None
    is_correct = False

    if hasattr(state, "config") and state.config:
        benchmark_config = state.config.get("benchmark", {})
        expected_answer = benchmark_config.get("expected_answer")

        if expected_answer and direct_answer:
            # Simple string matching for now, could be enhanced with semantic matching
            is_correct = (
                expected_answer.lower() in direct_answer.lower()
                or direct_answer.lower() in expected_answer.lower()
            )
            print(f"[finalize_answer] Comparing answers:")
            print(f"  - Generated: {direct_answer}")
            print(f"  - Expected:  {expected_answer}")
            print(f"  - Match:     {is_correct}")

    # Apply citation processing for benchmark mode
    processed_answer_text = answer_text
    if state.benchmark_mode:
        print(f"[finalize_answer] Applying citation processing for benchmark mode")

        # Get source citations from state
        source_citations = getattr(state, "source_citations", {})

        # Apply benchmark-specific citation processing with custom format
        processed_answer_text = post_process_benchmark_answer(
            answer_text, source_citations
        )

        print(f"[finalize_answer] Citation processing completed")
        print(f"  - Original length: {len(answer_text)} chars")
        print(f"  - Processed length: {len(processed_answer_text)} chars")
        print(f"  - Available citations: {len(source_citations)}")

    # Create the final benchmark result
    final_result = {
        "answer": direct_answer,
        "confidence": numeric_confidence,
        "confidence_level": confidence_level,
        "evidence": key_evidence,
        "sources": sources,
        "limitations": limitations,
        "expected_answer": expected_answer,
        "is_correct": is_correct,
        "full_response": processed_answer_text,  # Use processed text with citations
        "raw_response": answer_text,  # Keep original for debugging
        "synthesis_of_all_loops": all_answers_with_reasoning,  # Include the synthesis for debugging
    }

    # Log the final structured answer
    print(f"[finalize_answer] Final structured answer:")
    print(f"  - Answer: {final_result['answer']}")
    print(
        f"  - Confidence: {final_result['confidence']} ({final_result['confidence_level']})"
    )
    if sources:
        print(f"  - Sources: {sources}")
    if key_evidence:
        print(
            f"  - Evidence: {key_evidence[:100]}..."
            if len(key_evidence) > 100
            else key_evidence
        )
    if limitations:
        print(f"  - Limitations: {limitations}")

    # Log performance timing
    end_time = time.time()
    print(f"[finalize_answer] Processing time: {end_time - start_time:.2f} seconds")
    print(f"[finalize_answer] Complete\n")

    # Return the updated state with the final result
    return {
        "benchmark_result": final_result,
        "qa_mode": state.qa_mode,
        "benchmark_mode": state.benchmark_mode,
        "research_complete": True,
        "previous_answers": previous_answers,  # Preserve previous answers
        "reflection_history": reflection_history,  # Preserve reflection history
        "config": getattr(state, "config", {}),
        "source_citations": getattr(
            state, "source_citations", {}
        ),  # Preserve citations
        "running_summary": getattr(
            state, "running_summary", ""
        ),  # Preserve running summary
        "research_loop_count": getattr(
            state, "research_loop_count", 0
        ),  # Preserve loop count
    }


def validate_context_sufficiency(
    state: SummaryState, config: RunnableConfig
) -> Dict[str, Any]:
    """
    Validate if the retrieved context is sufficient to answer the question.
    Updates state with useful information, missing information, and refinement needs.
    Appends useful information to the running_summary.
    """
    print(
        f"--- ENTERING validate_context_sufficiency (Loop {state.research_loop_count}) ---"
    )
    logger.info(
        f"[validate_context_sufficiency] Called at loop {state.research_loop_count}"
    )
    running_summary_preview = (state.running_summary or "")[:100]
    logger.info(
        f"[validate_context_sufficiency] Initial state.running_summary (first 100 chars): '{running_summary_preview}...'"
    )
    logger.info(
        f"[validate_context_sufficiency] Initial state.web_research_results type: {type(state.web_research_results)}"
    )
    if isinstance(state.web_research_results, list):
        logger.info(
            f"[validate_context_sufficiency] Initial state.web_research_results length: {len(state.web_research_results)}"
        )
        if state.web_research_results:
            logger.info(
                f"[validate_context_sufficiency] First item type in web_research_results: {type(state.web_research_results[0])}"
            )

    start_time = time.time()

    configurable = Configuration.from_runnable_config(config)
    provider, model = _resolve_provider_model(configurable, state, "validate_context_sufficiency")
    llm = get_llm_client(provider, model)

    # Current web_research_results (list of dicts, each with 'content')
    new_search_results_list = getattr(state, "web_research_results", [])
    new_search_content = "\n\n---\n\n".join(
        item.get("content", "")
        for item in new_search_results_list
        if isinstance(item, dict) and item.get("content")
    )
    logger.debug(
        f"[validate_context_sufficiency] Constructed new_search_content. Length: {len(new_search_content)}. Preview (first 300 chars): '{new_search_content[:300]}...'"
    )

    # Previously accumulated knowledge
    current_running_summary = getattr(state, "running_summary", "")

    # Get uploaded knowledge
    uploaded_knowledge = getattr(state, "uploaded_knowledge", None)

    # Construct the full context for validation: uploaded knowledge + running summary + new results
    # Separator logic
    context_parts = []
    if uploaded_knowledge and uploaded_knowledge.strip():
        context_parts.append(
            f"USER-PROVIDED EXTERNAL KNOWLEDGE (HIGHEST AUTHORITY):\\n{uploaded_knowledge}"
        )
        print(
            f"[validate_context_sufficiency] Including uploaded knowledge in validation: {len(uploaded_knowledge)} characters"
        )
    if current_running_summary.strip():
        context_parts.append(
            f"PREVIOUSLY ACCUMULATED KNOWLEDGE (Running Summary):\\n{current_running_summary}"
        )
    if new_search_content.strip():
        context_parts.append(
            f"NEWLY FETCHED CONTENT (From latest web search):\\n{new_search_content}"
        )

    context_to_validate_llm = "\n\n---\n\n".join(context_parts)

    # If, after all processing, there's still no context to send to the LLM,
    # it means the initial search might have yielded nothing usable, or the running_summary was empty.
    if not context_to_validate_llm.strip():
        print(
            f"[validate_context_sufficiency] No context (running_summary or new web_research_results) to validate. Assuming incomplete."
        )
        running_summary_empty_check = not (
            state.running_summary and state.running_summary.strip()
        )
        new_content_empty_check = (
            not new_search_content
        )  # new_search_content is a string by this point
        logger.warning(
            f"[validate_context_sufficiency] current_context_to_validate is empty. "
            f"Skipping LLM call. Running summary was empty: {running_summary_empty_check}, "
            f"new_search_content was empty: {new_content_empty_check}"
        )
        # Fallback: if no context at all, assume refinement is needed.
        return {
            "running_summary": current_running_summary,  # Preserve existing running_summary
            "useful_information": "",  # No new useful info this round
            "missing_information": "No context was available for validation (neither running_summary nor new search results). The original query needs to be addressed.",
            "refinement_reasoning": "No context provided for validation.",
            "needs_refinement": True,
            "web_research_results": [],
            # Preserve analysis settings
            "analysis_required": getattr(state, "analysis_required", False),
            "llm_provider": getattr(state, "llm_provider", None),
            "llm_model": getattr(state, "llm_model", None),
        }

    # Choose the appropriate validation prompt based on mode
    if state.benchmark_mode:
        validate_prompt = BENCHMARK_VALIDATE_RETRIEVAL_PROMPT
    elif state.qa_mode:
        validate_prompt = QA_VALIDATE_RETRIEVAL_PROMPT
    else:
        validate_prompt = QA_VALIDATE_RETRIEVAL_PROMPT  # Fallback

    prompt = validate_prompt.format(
        current_date=CURRENT_DATE,
        current_year=CURRENT_YEAR,
        one_year_ago=ONE_YEAR_AGO,
        question=state.research_topic,
        retrieved_context=context_to_validate_llm,  # Pass combined context
    )

    print(
        f"[validate_context_sufficiency] Sending validation prompt to {provider}/{model}"
    )
    print(
        f"[validate_context_sufficiency] Context for validation LLM (first 300 chars): {context_to_validate_llm[:300]}..."
    )
    response = llm.invoke(prompt)
    raw_llm_output = response.content if hasattr(response, "content") else str(response)
    print(
        f"[validate_context_sufficiency] Raw LLM response preview: {raw_llm_output[:300]}..."
    )

    reasoning = ""
    json_response_str = raw_llm_output.strip()

    if "<think>" in json_response_str and "</think>" in json_response_str:
        parts = json_response_str.split("</think>", 1)
        think_part = parts[0]
        if "<think>" in think_part:
            reasoning = think_part.split("<think>", 1)[1].strip()
        json_response_str = parts[1].strip() if len(parts) > 1 else ""

    if json_response_str.startswith("```json"):
        json_response_str = json_response_str[7:]
    if json_response_str.startswith("```"):
        json_response_str = json_response_str[3:]
    if json_response_str.endswith("```"):
        json_response_str = json_response_str[:-3]
    json_response_str = json_response_str.strip()

    updated_running_summary = current_running_summary  # Default to existing
    newly_identified_useful_info_this_round = ""

    try:
        parsed_output = json.loads(json_response_str)
        status = parsed_output.get("status", "INCOMPLETE").upper()
        llm_identified_missing_info = parsed_output.get(
            "missing_information", "No specific missing information provided by LLM."
        )
        # This is useful info specifically from the *newly_fetched_content*
        llm_identified_useful_info_from_new = parsed_output.get(
            "useful_information", ""
        )
        needs_refinement_flag = status == "INCOMPLETE"

        print(f"[validate_context_sufficiency] Validation Result:")
        print(f"  - Status: {status}")
        print(f"  - Needs Refinement: {needs_refinement_flag}")
        print(f"  - LLM Missing Info: {llm_identified_missing_info[:200]}...")
        print(
            f"  - LLM Useful Info (from new content): {llm_identified_useful_info_from_new[:200]}..."
        )
        print(f"  - Reasoning: {reasoning[:200]}...")

        newly_identified_useful_info_this_round = (
            llm_identified_useful_info_from_new.strip()
        )

        # Append newly identified useful info to the running_summary
        if newly_identified_useful_info_this_round:
            if updated_running_summary:  # If there's existing summary, add a separator
                updated_running_summary += f"\\n\\n---\\nNEW FINDINGS (Loop {state.research_loop_count}):\\n{newly_identified_useful_info_this_round}"
            else:  # First time adding useful info
                updated_running_summary = f"INITIAL FINDINGS (Loop {state.research_loop_count}):\\n{newly_identified_useful_info_this_round}"
            print(
                f"[validate_context_sufficiency] Appended new useful info to running_summary. New length: {len(updated_running_summary)}"
            )
        else:
            print(
                "[validate_context_sufficiency] No new useful information identified by LLM from the latest search results."
            )
            if (
                not needs_refinement_flag
                and not updated_running_summary
                and new_search_content.strip()
            ):  # If complete, and summary is empty, use new content
                updated_running_summary = f"INITIAL FINDINGS (Loop {state.research_loop_count}):\\n{new_search_content.strip()}"
                newly_identified_useful_info_this_round = new_search_content.strip()

        updates = {
            "running_summary": updated_running_summary,
            "useful_information": newly_identified_useful_info_this_round,  # Info from THIS validation step
            "missing_information": llm_identified_missing_info,
            "refinement_reasoning": reasoning,
            "needs_refinement": needs_refinement_flag,
            "web_research_results": [],
            # Preserve analysis settings
            "analysis_required": getattr(state, "analysis_required", False),
            "llm_provider": getattr(state, "llm_provider", None),
            "llm_model": getattr(state, "llm_model", None),
        }

    except json.JSONDecodeError as e:
        print(
            f"[validate_context_sufficiency] ERROR: Failed to parse JSON from LLM output: {e}"
        )
        print(
            f"[validate_context_sufficiency] Raw non-JSON output: {json_response_str}"
        )
        # Fallback: assume incomplete, keep existing running_summary, pass raw output as missing info
        updates = {
            "running_summary": current_running_summary,
            "useful_information": "",  # No new useful info parsed
            "missing_information": f"LLM validation output was not valid JSON. Raw output: {raw_llm_output}",
            "refinement_reasoning": reasoning + " (Error parsing LLM JSON response)",
            "needs_refinement": True,
            "web_research_results": [],
            # Preserve analysis settings
            "analysis_required": getattr(state, "analysis_required", False),
            "llm_provider": getattr(state, "llm_provider", None),
            "llm_model": getattr(state, "llm_model", None),
        }

    end_time = time.time()
    print(
        f"[validate_context_sufficiency] Processing time: {end_time - start_time:.2f} seconds"
    )
    print(f"--- EXITING validate_context_sufficiency ---")
    return updates


def refine_query(state: SummaryState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Refine the search query based on the current state, including running_summary and missing_information.
    """
    print(
        f"--- ENTERING refine_query (Loop {state.research_loop_count}) ---"
    )  # research_loop_count is already incremented before this node
    start_time = time.time()

    next_loop_count = state.research_loop_count + 1
    print(
        f"[refine_query] Incremented research_loop_count from {state.research_loop_count} to {next_loop_count}"
    )

    configurable = Configuration.from_runnable_config(config)
    provider, model = _resolve_provider_model(configurable, state, "refine_query")
    llm = get_llm_client(provider, model)

    original_query = state.research_topic
    # This is identified useful info from the *last* validation step
    useful_info_from_last_validation = getattr(state, "useful_information", "")
    # This is what's still missing after the last validation, considering all info so far
    missing_info_after_last_validation = getattr(
        state, "missing_information", "No specific missing information identified yet."
    )
    # This is all useful information accumulated over all previous loops
    current_running_summary = getattr(state, "running_summary", "")
    # Reasoning for why refinement might be needed (from last validation)
    refinement_reasoning_from_last_validation = getattr(
        state, "refinement_reasoning", ""
    )

    # Construct a comprehensive context for the LLM to refine the query
    refinement_context_parts = [f"Original Research Topic/Question: {original_query}"]
    if current_running_summary.strip():
        refinement_context_parts.append(
            f"CUMULATIVE KNOWLEDGE SO FAR (Running Summary):\\n{current_running_summary}"
        )
    if useful_info_from_last_validation.strip():
        refinement_context_parts.append(
            f"NEWLY IDENTIFIED USEFUL INFORMATION (From Last Validation Step):\\n{useful_info_from_last_validation}"
        )
    if missing_info_after_last_validation.strip():
        refinement_context_parts.append(
            f"CURRENTLY MISSING INFORMATION (Based on all knowledge so far):\\n{missing_info_after_last_validation}"
        )
    if refinement_reasoning_from_last_validation.strip():
        refinement_context_parts.append(
            f"REASONING FOR REFINEMENT (From Last Validation Step):\\n{refinement_reasoning_from_last_validation}"
        )

    full_refinement_context_for_llm = "\n\n---\n\n".join(refinement_context_parts)

    # Choose the appropriate refinement prompt based on mode
    if state.benchmark_mode:
        refine_prompt = BENCHMARK_REFINE_QUERY_PROMPT
    elif state.qa_mode:
        refine_prompt = QA_REFINE_QUERY_PROMPT
    else:
        refine_prompt = QA_REFINE_QUERY_PROMPT  # Fallback

    prompt = refine_prompt.format(
        current_date=CURRENT_DATE,
        current_year=CURRENT_YEAR,
        one_year_ago=ONE_YEAR_AGO,
        question=original_query,  # Keep original question for top-level context
        # retrieved_context is now a more structured block of info
        retrieved_context=full_refinement_context_for_llm,
    )

    print(f"[refine_query] Sending refinement prompt to {provider}/{model}")
    print(
        f"[refine_query] Context for LLM (first 300 chars): {full_refinement_context_for_llm[:300]}..."
    )

    invoke_kwargs = {}
    if provider == "azure":
        invoke_kwargs["max_tokens"] = 500

    try:
        print(
            f"[refine_query] Calling llm.invoke with provider-specific kwargs: {invoke_kwargs}"
        )
        response = llm.invoke(prompt, **invoke_kwargs)
    except TypeError as te:
        # This catch might now be for other unexpected TypeErrors,
        # as we are trying to use correct param names above.
        print(
            f"[refine_query] Warning: llm.invoke failed with TypeError. Provider: {provider}, Error: {te}"
        )
        print(
            f"[refine_query] Attempting invoke without explicit max token override for this call."
        )
        response = llm.invoke(prompt)  # Fallback to calling without it

    raw_llm_output = response.content if hasattr(response, "content") else str(response)
    print(f"[refine_query] Raw LLM response: {raw_llm_output[:300]}...")

    # Attempt to parse JSON, but also handle plain text if JSON fails
    refined_query_str = original_query  # Default to original if parsing fails
    refinement_reasoning_llm = ""

    # Robustly strip ```json and ``` markers and <think> tags
    json_part_str = raw_llm_output.strip()
    if "<think>" in json_part_str and "</think>" in json_part_str:
        parts = json_part_str.split("</think>", 1)
        # We don't need the <think> part for refine_query's primary output
        json_part_str = parts[1].strip() if len(parts) > 1 else ""

    if json_part_str.startswith("```json"):
        json_part_str = json_part_str[7:]
    if json_part_str.startswith("```"):
        json_part_str = json_part_str[3:]
    if json_part_str.endswith("```"):
        json_part_str = json_part_str[:-3]
    json_part_str = json_part_str.strip()

    try:
        parsed_output = json.loads(json_part_str)
        if isinstance(parsed_output, dict):
            refined_query_str = parsed_output.get("refined_query", original_query)
            refinement_reasoning_llm = parsed_output.get("reasoning", "")
            print(
                f"[refine_query] Parsed refined_query from JSON object: '{refined_query_str}'"
            )
            if refinement_reasoning_llm:
                print(
                    f"[refine_query] LLM reasoning for refinement: {refinement_reasoning_llm[:200]}..."
                )
        elif isinstance(parsed_output, str):
            # This handles the case where the LLM returns a JSON string literal, e.g., ""actual query""
            refined_query_str = parsed_output
            refinement_reasoning_llm = ""  # No reasoning if it was just a string
            print(
                f"[refine_query] Parsed refined_query from JSON string literal: '{refined_query_str}'"
            )
        else:
            # Should not happen if json.loads worked, but as a safe fallback
            print(
                f"[refine_query] json.loads returned an unexpected type: {type(parsed_output)}. Falling back to original query."
            )
            refined_query_str = original_query
            refinement_reasoning_llm = "LLM returned an unexpected JSON structure."

    except json.JSONDecodeError:
        print(
            f"[refine_query] Failed to parse JSON from LLM refinement output. Treating raw output as refined query."
        )
        # If not JSON, assume the entire (stripped) response is the refined query
        refined_query_str = raw_llm_output.strip()
        if not refined_query_str:  # Safety net if LLM returns empty or only whitespace
            print(
                "[refine_query] LLM returned empty refined query, falling back to original query."
            )
            refined_query_str = original_query
        else:
            # Extract first non-empty line if output is multi-line non-JSON
            first_line_query = refined_query_str.split("\\n")[0].strip()
            if first_line_query:
                refined_query_str = first_line_query
            else:  # If first line is empty, take original_query
                refined_query_str = original_query

            print(
                f"[refine_query] Using raw output as refined_query: '{refined_query_str}'"
            )

    # Update the research_topic in the state to guide the next round of multi_agents_network
    # Or, if you prefer to keep original research_topic pristine, use a new state field like 'current_search_focus'

    updated_state = {
        # 'research_topic': refined_query_str, # DO NOT UPDATE research_topic here
        "search_query": refined_query_str,  # Update search_query to reflect the new focus for routing
        "research_loop_count": next_loop_count,  # Increment loop count
        "refinement_reasoning": refinement_reasoning_llm,  # Store LLM's reasoning for this refinement
        # research_topic (the original question) is preserved implicitly from the input state
        # Other state fields like running_summary, useful_information, missing_information are preserved implicitly
    }

    end_time = time.time()
    print(f"[refine_query] Original Query: {original_query}")
    print(f"[refine_query] Refined Query for next loop: {refined_query_str}")
    print(f"[refine_query] Processing time: {end_time - start_time:.2f} seconds")
    print(f"--- EXITING refine_query ---")
    return updated_state


async def process_steering(
    state: SummaryState, config: RunnableConfig
) -> Dict[str, Any]:
    """
    Process steering messages using the advanced memory system and planning agent.
    This node implements the Manus-style agent loop for adaptive research.
    """
    logger.info(
        "[STEERING_NODE] Processing steering messages and adapting research plan"
    )

    start_time = datetime.now()

    # Initialize memory system if not already done
    if not state.memory_system and state.steering_enabled:
        from src.memory_system import AdvancedMemorySystem

        state.memory_system = AdvancedMemorySystem()
        state.steering_session_id = state.memory_system.session_id
        logger.info(
            f"[STEERING_NODE] Initialized memory system with session ID: {state.steering_session_id}"
        )

    # If no memory system and steering not enabled, skip processing
    if not state.memory_system:
        logger.info(
            "[STEERING_NODE] No memory system available, skipping steering processing"
        )
        return {"steering_processed": False, "reason": "Memory system not initialized"}

    try:
        # Initialize steering planning agent
        from src.steering_agent import SteeringPlanningAgent

        planning_agent = SteeringPlanningAgent(state)

        # Execute the agent loop
        loop_results = await planning_agent.execute_agent_loop()

        # Update state with results
        processing_time = (datetime.now() - start_time).total_seconds()

        # Update research plan in state
        state.current_research_plan = state.memory_system.get_todo_md()

        # Log steering processing results
        logger.info(
            f"[STEERING_NODE] Completed steering processing in {processing_time:.2f}s"
        )
        logger.info(
            f"[STEERING_NODE] Steering messages processed: {loop_results.get('steering_processed', 0)}"
        )
        logger.info(
            f"[STEERING_NODE] Tools executed: {loop_results.get('tools_executed', 0)}"
        )
        logger.info(
            f"[STEERING_NODE] Tools cancelled: {loop_results.get('tools_cancelled', 0)}"
        )
        logger.info(
            f"[STEERING_NODE] Plan adaptations: {loop_results.get('plan_adaptations', 0)}"
        )

        return {
            "steering_processed": True,
            "processing_time": processing_time,
            "loop_results": loop_results,
            "memory_summary": state.get_memory_summary(),
            "updated_plan": state.current_research_plan,
        }

    except Exception as e:
        logger.error(f"[STEERING_NODE] Error processing steering: {str(e)}")
        logger.error(f"[STEERING_NODE] Traceback: {traceback.format_exc()}")

        return {
            "steering_processed": False,
            "error": str(e),
            "processing_time": (datetime.now() - start_time).total_seconds(),
        }


# Steering routing function removed - using standard research flow


def route_after_steering(state: SummaryState) -> str:
    """
    Route after steering processing to determine next action.
    """
    logger.info("[ROUTING] Determining next action after steering processing")

    # Check if research should continue based on memory system
    if state.memory_system:
        # Get next tasks from memory system
        next_tasks = state.memory_system.task_planner.get_next_tasks(max_tasks=1)

        if next_tasks:
            logger.info(
                f"[ROUTING] Found {len(next_tasks)} pending tasks - continuing research"
            )
            return "multi_agents_network"
        else:
            logger.info("[ROUTING] No pending tasks - proceeding to report generation")
            if state.benchmark_mode:
                return "validate_context_sufficiency"
            else:
                return "generate_report"

    # Default routing
    if state.benchmark_mode:
        return "validate_context_sufficiency"
    else:
        return "generate_report"


# ---------------------------------------------------------------------------
# Analysis Pipeline Node
# ---------------------------------------------------------------------------
async def run_analysis(state: SummaryState, config: RunnableConfig):
    """
    Run data analysis and visualization on the finalized research report.
    Only executes when state.analysis_required is True.

    Pipeline: extract_data → profile_data → plan_charts → create_chart → build HTML report
    """
    logger.info("[ANALYSIS] Starting analysis pipeline...")
    print(f"[run_analysis] START — analysis_required={getattr(state, 'analysis_required', False)}")

    try:
        from extensions.agents.data_analysis_agent import DataAnalysisAgent
        from extensions.utils.report_builder import build_html_report

        configurable = Configuration.from_runnable_config(config)
        provider, model = _resolve_provider_model(configurable, state, "run_analysis")
        print(f"[run_analysis] Using provider={provider!r}, model={model!r}")

        # Initialize the analysis agent with the user-selected provider/model
        agent = DataAnalysisAgent(provider=provider, model=model)

        # Run the enforced pipeline: extract → profile → visualize
        # Prefer clean markdown (no HTML tags/CSS) over finalized HTML
        report_text = state.markdown_report or state.running_summary or ""
        logger.info(f"[ANALYSIS] Running pipeline on report ({len(report_text)} chars)")
        print(f"[run_analysis] Running pipeline on {len(report_text)} chars of research report")

        result = agent.run_pipeline(report_text)

        status = result.get("status", "error")
        logger.info(f"[ANALYSIS] Pipeline status: {status}")

        if status == "completed":
            charts = result.get("charts", [])
            chart_explanations = result.get("chart_explanations", {})
            extracted_data = result.get("extracted_data", "")
            data_profile = result.get("data_profile", "")
            analysis_output = result.get("output", "")

            logger.info(f"[ANALYSIS] Charts generated: {len(charts)}")
            logger.info(f"[ANALYSIS] Data points extracted: {len(extracted_data.splitlines()) if extracted_data else 0}")

            # Build HTML analysis report
            sources = state.sources_gathered or []
            sub_queries = state.subtopic_queries or []

            try:
                report_path = build_html_report(
                    display_text=report_text,
                    analysis_output=analysis_output,
                    figures=charts,
                    chart_explanations=chart_explanations,
                    sources=sources,
                    query=state.research_topic,
                    sub_queries=sub_queries,
                    extracted_data_summary=extracted_data,
                    data_profile_summary=data_profile,
                )
                logger.info(f"[ANALYSIS] HTML report generated: {report_path}")
                print(f"[run_analysis] Analysis HTML report generated: {report_path}")
            except Exception as e:
                logger.error(f"[ANALYSIS] Report builder failed: {e}")
                print(f"[run_analysis] ERROR: Analysis report builder failed: {e}")
                report_path = None

            print(
                f"[run_analysis] DONE — charts={len(charts)}, "
                f"report_path={report_path!r}, output_len={len(analysis_output)}"
            )
            return {
                "analysis_output": analysis_output,
                "analysis_report_path": report_path,
                "analysis_charts": charts,
                "analysis_chart_explanations": chart_explanations,
                "analysis_required": getattr(state, "analysis_required", False),
                "llm_provider": getattr(state, "llm_provider", None),
                "llm_model": getattr(state, "llm_model", None),
            }
        else:
            error_msg = result.get("error", "Unknown error")
            logger.warning(f"[ANALYSIS] Pipeline failed: {error_msg}")
            print(f"[run_analysis] FAILED — pipeline status={status!r}, error={error_msg!r}")
            return {
                "analysis_output": f"Analysis pipeline failed: {error_msg}",
                "analysis_report_path": None,
                "analysis_charts": [],
                "analysis_chart_explanations": {},
                "analysis_required": getattr(state, "analysis_required", False),
                "llm_provider": getattr(state, "llm_provider", None),
                "llm_model": getattr(state, "llm_model", None),
            }

    except Exception as e:
        logger.error(f"[ANALYSIS] Exception: {e}", exc_info=True)
        print(f"[run_analysis] EXCEPTION — {type(e).__name__}: {e}")
        return {
            "analysis_output": f"Analysis error: {str(e)}",
            "analysis_report_path": None,
            "analysis_charts": [],
            "analysis_chart_explanations": {},
            "analysis_required": getattr(state, "analysis_required", False),
            "llm_provider": getattr(state, "llm_provider", None),
            "llm_model": getattr(state, "llm_model", None),
        }


def route_after_finalize_report(state: SummaryState):
    """Route after finalize_report: run analysis if requested, otherwise end."""
    if state.analysis_required:
        logger.info("[ROUTING] analysis_required=True → run_analysis")
        print("[route_after_finalize_report] analysis_required=True → routing to run_analysis")
        return "run_analysis"
    else:
        logger.info("[ROUTING] analysis_required=False → END")
        print("[route_after_finalize_report] analysis_required=False → routing to END (no analysis)")
        return "__end__"


def create_graph():
    """
    Factory function that creates a fresh graph instance with steering capabilities.
    """
    # Create a new builder
    builder = StateGraph(
        SummaryState,
        input=SummaryStateInput,
        output=SummaryStateOutput,
        config_schema=Configuration,
    )

    # Add all nodes
    builder.add_node("multi_agents_network", async_multi_agents_network)
    builder.add_node("generate_report", generate_report)
    builder.add_node("reflect_on_report", reflect_on_report)
    builder.add_node("finalize_report", finalize_report)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("reflect_answer", reflect_answer)
    builder.add_node("finalize_answer", finalize_answer)
    builder.add_node(
        "validate_context_sufficiency", validate_context_sufficiency
    )  # New node
    builder.add_node("refine_query", refine_query)  # New node
    builder.add_node("run_analysis", run_analysis)  # Analysis pipeline node

    # Add edges
    builder.add_edge(START, "multi_agents_network")

    # Add conditional routing based on qa_mode and benchmark_mode
    def route_after_multi_agents_decision(state):
        if state.qa_mode or state.benchmark_mode:
            return "validate_context_sufficiency"
        else:
            return "generate_report"

    # Add conditional routing after multi_agents_network
    builder.add_conditional_edges(
        "multi_agents_network",
        route_after_multi_agents_decision,  # Use the standard routing function
        {
            "validate_context_sufficiency": "validate_context_sufficiency",  # Benchmark path
            "generate_report": "generate_report",  # Regular path
        },
    )

    # Routing for validate_context_sufficiency (Benchmark Path)
    builder.add_conditional_edges(
        "validate_context_sufficiency",
        lambda state: "refine_query" if state.needs_refinement else "generate_answer",
        {"refine_query": "refine_query", "generate_answer": "generate_answer"},
    )

    # Routing for refine_query (Benchmark Path) - loops back to search
    builder.add_edge("refine_query", "multi_agents_network")

    # Regular report flow
    builder.add_edge("generate_report", "reflect_on_report")
    # OLD: builder.add_edge("reflect_on_report", "finalize_report")
    # NEW: Conditional routing after reflect_on_report (to allow looping for regular research)
    builder.add_conditional_edges(
        "reflect_on_report",
        route_research,  # Using the existing route_research logic
        {
            "multi_agents_network": "multi_agents_network",  # Loop back for more research
            "finalize_report": "finalize_report",  # Finalize if research is complete
        },
    )
    # Conditional routing after finalize_report: run analysis if user requested it
    builder.add_conditional_edges(
        "finalize_report",
        route_after_finalize_report,
        {
            "run_analysis": "run_analysis",
            "__end__": END,
        },
    )
    builder.add_edge("run_analysis", END)

    # Benchmark flow after generate_answer (no change here, existing logic)
    builder.add_edge("generate_answer", "reflect_answer")

    # Add conditional routing after reflect_answer (no change here, existing logic)
    builder.add_conditional_edges(
        "reflect_answer",
        lambda state: route_after_reflect_answer(
            state, {}
        ),  # Pass empty config or get from state
        {
            "multi_agents_network": "multi_agents_network",  # Continue research
            "finalize_answer": "finalize_answer",  # Complete research
        },
    )

    # End benchmark flow after finalize_answer
    builder.add_edge("finalize_answer", END)

    return builder.compile()


# ... existing code ...

# Export a function to get a fresh graph instead of a single graph instance
# This way each research topic gets a completely fresh state
# Replace the single graph instance with a factory function
graph = create_graph()  # Export a compiled instance for LangGraph Studio

# Keep the factory function available for programmatic use
create_fresh_graph = (
    create_graph  # Use this when you need a fresh instance programmatically
)


# Helper function to get configuration from the runnable config
def get_configurable(config):
    """
    Helper function to extract configuration from the runnable config.

    Args:
        config: The runnable config object or None

    Returns:
        Configuration object with the settings from config
    """
    # If config is None, return a default configuration
    if config is None:
        return Configuration()

    # If config already has a configurable, use it
    if "configurable" in config and isinstance(config["configurable"], Configuration):
        return config["configurable"]

    # Otherwise create a new configurable from the config
    return Configuration.from_runnable_config(config)


# Helper function to process response for UI display
def process_response_for_ui(response_text):
    """
    Process a raw response into a format suitable for UI display.

    Args:
        response_text: The raw response text from the agent

    Returns:
        Processed response text ready for UI display
    """
    # If response is empty, return an empty string
    if not response_text:
        return ""

    # Remove any <think> tags and their contents
    while "<think>" in response_text and "</think>" in response_text:
        start = response_text.find("<think>")
        end = response_text.find("</think>") + len("</think>")
        response_text = response_text[:start] + response_text[end:]

    # Clean up markdown formatting for UI
    response_text = response_text.strip()

    return response_text


def route_research_flow(state: SummaryState, config: Dict[str, Any]) -> str:
    """
    Route the research flow based on state and configuration.
    Returns the next node to execute.
    """
    print(f"[route_research_flow] Determining next research step")
    print(f"[route_research_flow] benchmark_mode={state.benchmark_mode}")
    print(f"[route_research_flow] research_complete={state.research_complete}")

    # Check if in benchmark mode
    if state.benchmark_mode:
        print("[route_research_flow] Operating in benchmark mode")

        # If we have search results, generate a focused answer
        if state.web_research_results:
            print(
                "[route_research_flow] Search returned results, proceeding to answer generation"
            )
            return "generate_answer"

        # If we have an answer, verify it
        if hasattr(state, "benchmark_result"):
            print("[route_research_flow] Answer generated, proceeding to verification")
            return "verify_answer"

        # If research is complete but no answer, reflect
        if state.research_complete:
            print("[route_research_flow] Research complete, proceeding to reflection")
            return "reflect_answer"

        # Otherwise continue research
        print("[route_research_flow] Continuing research")
        return "multi_agents_network"

    # Regular research flow
    print("[route_research_flow] Operating in regular research mode")
    if state.research_complete:
        return "finalize_report"
    elif state.web_research_results:
        return "generate_report"
    else:
        return "multi_agents_network"
