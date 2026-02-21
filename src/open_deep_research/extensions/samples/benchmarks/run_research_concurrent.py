#!/usr/bin/env python3
"""
Multi-threaded research script for benchmarking.
"""

import asyncio
import argparse
import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import threading
from typing import List, Dict, Optional, Tuple
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("research_concurrent.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Add the deep-research directory to the Python path
script_dir = Path(__file__).parent
deep_research_dir = script_dir.parent
sys.path.insert(0, str(deep_research_dir))

# Load environment variables
env_file_path = deep_research_dir / ".env"
logger.info(f"Loading environment from: {env_file_path}")
load_dotenv(dotenv_path=env_file_path)

# Import HTML-to-markdown converter
from src.graph import generate_markdown_report

STATS_LOCK = threading.Lock()
GLOBAL_STATS = {
    "completed": 0,
    "failed": 0,
    "start_time": None,
    "tasks_started": 0,
    "tasks_in_progress": 0,
}


# ==================== Trajectory Recorder ====================
class ResearchTrajectoryRecorder:
    """Records structured research trajectory with OpenAI-format tool calls"""

    def __init__(self):
        self.query = ""
        self.iterations = []
        self.current_iteration = None
        self.tool_call_counter = 0
        self.previous_sources = set()

    def start_iteration(self, iteration_number: int):
        """Start a new research iteration"""
        self.current_iteration = {
            "iteration": iteration_number,
            "timestamp_start": datetime.now().isoformat(),
            "tool_calls": [],  # OpenAI format tool calls
            "running_summary": "",
            "num_sources": 0,
        }

    def end_iteration(self):
        """End the current iteration and save it"""
        if self.current_iteration:
            self.current_iteration["timestamp_end"] = datetime.now().isoformat()
            self.iterations.append(self.current_iteration)
            self.current_iteration = None

    def add_tool_call(self, function_name: str, arguments: Dict, result: any):
        """Add a tool call in OpenAI format"""
        if self.current_iteration:
            self.tool_call_counter += 1
            tool_call = {
                "id": f"call_{self.tool_call_counter}",
                "type": "function",
                "function": {"name": function_name, "arguments": arguments},
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }
            self.current_iteration["tool_calls"].append(tool_call)

    def record_running_summary(self, summary):
        """Record the running summary after this iteration (with HTML converted to markdown)"""
        if self.current_iteration:
            # Handle case where summary might be a list (convert to string)
            if isinstance(summary, list):
                summary = "\n\n".join(str(s) for s in summary)
            elif not isinstance(summary, str):
                summary = str(summary)

            # Convert HTML to clean markdown using the existing graph.py function
            cleaned_summary = generate_markdown_report(summary)
            self.current_iteration["running_summary"] = cleaned_summary

    def record_sources(self, sources: List[str]):
        """Record number of NEW sources gathered in this iteration"""
        if self.current_iteration:
            current_sources = set(sources)
            new_sources = current_sources - self.previous_sources
            self.current_iteration["num_sources"] = len(new_sources)
            self.previous_sources = current_sources

    def get_summary(self, total_unique_sources: int = 0) -> Dict:
        """Get a summary of the trajectory"""
        return {
            "query": self.query,
            "num_iterations": len(self.iterations),
            "total_num_sources": total_unique_sources,
            "iterations_summary": [
                {
                    "iteration": it["iteration"],
                    "num_tool_calls": len(it.get("tool_calls", [])),
                    "num_sources": it.get("num_sources", 0),
                }
                for it in self.iterations
            ],
        }

    def to_dict(self, total_unique_sources: int = 0) -> Dict:
        """Convert trajectory to dictionary for JSON serialization"""
        return {
            "query": self.query,
            "summary": self.get_summary(total_unique_sources),
            "iterations": self.iterations,
        }


# ==================== Benchmark Dataset Managers ====================
class BenchmarkDatasetManager(ABC):
    """Abstract base class for managing different benchmark datasets."""

    @abstractmethod
    def load_queries(
        self,
        file_path: str,
        task_ids: Optional[List[int]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Load queries from the dataset file."""
        pass

    @abstractmethod
    def get_query_field(self) -> str:
        """Return the field name that contains the query text."""
        pass

    @abstractmethod
    def get_processing_config(self) -> Dict:
        """Return the configuration for processing this benchmark."""
        pass

    @abstractmethod
    def format_result(self, task_data: Dict, result_data: Dict) -> Dict:
        """Format the result according to benchmark requirements."""
        pass

    @abstractmethod
    def get_output_filename(self, task_id: str) -> str:
        """Get the output filename for a task result."""
        pass


class DRBDatasetManager(BenchmarkDatasetManager):
    """Manager for Deep Research Benchmark (DRB) dataset."""

    def load_queries(
        self,
        file_path: str,
        task_ids: Optional[List[int]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Load queries from DRB JSONL file."""
        queries = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            query_data = json.loads(line)
                            queries.append(query_data)
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON on line {line_num}: {e}")

            logger.info(f"Loaded {len(queries)} queries from {file_path}")

            # Filter by task IDs if specified
            if task_ids:
                queries = [q for q in queries if q["id"] in task_ids]
                logger.info(f"Filtered to {len(queries)} specific tasks: {task_ids}")

            # Apply limit if specified
            if limit:
                queries = queries[:limit]
                logger.info(f"Limited to first {len(queries)} tasks")

            return queries

        except FileNotFoundError:
            logger.error(f"Query file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading queries: {e}")
            raise

    def get_query_field(self) -> str:
        return "prompt"

    def get_processing_config(self) -> Dict:
        return {
            "max_loops": 5,
            "extra_effort": False,
            "qa_mode": False,
            "benchmark_mode": False,
        }

    def format_result(self, task_data: Dict, result_data: Dict) -> Dict:
        """Format result for DRB."""
        return {
            "id": result_data["id"],
            "prompt": result_data["query"],  # DRB uses "prompt" not "query"
            "article": result_data["article"],
            "metadata": {
                "timing": result_data["timing"],
                "debug_info": result_data["debug_info"],
                "content_stats": result_data["content_stats"],
            },
        }

    def get_output_filename(self, task_id: str) -> str:
        return f"{task_id}.json"


class DeepConsultDatasetManager(BenchmarkDatasetManager):
    """Dataset manager for DeepConsult CSV dataset."""

    def load_queries(
        self,
        file_path: str,
        task_ids: Optional[List[int]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Load queries from CSV file."""
        import csv

        queries = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    # Create query dict with index as ID
                    # Ensure query is a string
                    query = row["query"]
                    if isinstance(query, list):
                        query = " ".join(str(item) for item in query)
                    query_data = {"id": i, "index": i, "query": str(query).strip()}
                    queries.append(query_data)

            logger.info(f"Loaded {len(queries)} queries from {file_path}")

            # Filter by task IDs if specified
            if task_ids:
                queries = [q for q in queries if q["id"] in task_ids]
                logger.info(f"Filtered to {len(queries)} specific tasks: {task_ids}")

            # Apply limit if specified
            if limit:
                queries = queries[:limit]
                logger.info(f"Limited to first {len(queries)} tasks")

            return queries

        except FileNotFoundError:
            logger.error(f"Query file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading queries: {e}")
            raise

    def get_query_field(self) -> str:
        return "query"

    def get_processing_config(self) -> Dict:
        return {
            "max_loops_default": 10,  # Max 10 loops as requested
            "benchmark_mode": False,  # Regular mode as requested
            "qa_mode": False,
            "visualization_disabled": True,
        }

    def format_result(self, task_data: Dict, result_data: Dict) -> Dict:
        """Format result for DeepConsult - keep existing structure."""
        return result_data

    def get_output_filename(self, task_id: str) -> str:
        return f"deepconsult_{task_id}.json"


class HealthBenchDatasetManager(BenchmarkDatasetManager):
    """Dataset manager for HealthBench evaluation."""

    def load_queries(
        self,
        file_path: str,
        task_ids: Optional[List[int]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Load queries from HealthBench JSON file (final_run_100 or final_run_1000)."""
        queries = []
        try:
            logger.info(f"Loading HealthBench data from {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data = data

            # data is a list of HealthBench examples
            for example in data:
                query_data = {
                    "id": example["id"],  # UUID from HealthBench
                    "query": example["problem"],  # Formatted problem string
                    "original_data": example,  # Keep all original data for eval
                }
                queries.append(query_data)

            logger.info(f"Loaded {len(queries)} HealthBench queries")

            # Filter by task IDs if specified (using string IDs for HealthBench)
            if task_ids:
                # Convert task_ids to strings for comparison
                task_ids_str = [str(tid) for tid in task_ids]
                queries = [q for q in queries if q["id"] in task_ids_str]
                logger.info(f"Filtered to {len(queries)} specific tasks")

            # Apply limit if specified
            if limit:
                queries = queries[:limit]
                logger.info(f"Limited to first {len(queries)} tasks")

            return queries

        except FileNotFoundError:
            logger.error(f"HealthBench file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading HealthBench data: {e}")
            raise

    def get_query_field(self) -> str:
        return "query"

    def get_processing_config(self) -> Dict:
        return {
            "max_loops": 5,  # 3 loops for medical questions
            "extra_effort": False,
            "qa_mode": False,
            "benchmark_mode": False,  # Use benchmark mode for citations
        }

    def format_result(self, task_data: Dict, result_data: Dict) -> Dict:
        """Format result for HealthBench - preserve all data needed for DR Tulu eval."""
        return {
            "id": result_data["id"],
            "query": result_data["query"],
            "article": result_data["article"],
            "original_data": task_data.get(
                "original_data", {}
            ),  # Preserve HealthBench metadata
            "metadata": {
                "timing": result_data["timing"],
                "debug_info": result_data["debug_info"],
                "content_stats": result_data["content_stats"],
            },
        }

    def get_output_filename(self, task_id: str) -> str:
        return f"{task_id}.json"


class HFDatasetManager(BenchmarkDatasetManager):
    """Dataset manager for HuggingFace datasets (e.g., LiveResearchBench)."""

    def __init__(
        self,
        dataset_name: str,
        query_column: str,
        config_name: Optional[str] = None,
        split: str = "test",
        mode: str = "regular",
        max_loops: int = 5,
        provider: str = "google",
        model: str = "gemini-2.5-pro",
    ):
        """
        Initialize HuggingFace dataset manager.

        Args:
            dataset_name: HF dataset path (e.g., "Salesforce/LiveResearchBench")
            query_column: Column name containing queries (e.g., "question_no_placeholder")
            config_name: Optional config name for the dataset (e.g., "question_only")
            split: Dataset split to load (default: "test")
            mode: Processing mode - "regular", "qa", or "benchmark" (default: "regular")
            max_loops: Maximum research loops (default: 5)
            provider: LLM provider (default: "google")
            model: LLM model name (default: "gemini-2.5-pro")
        """
        self.dataset_name = dataset_name
        self.query_column = query_column
        self.config_name = config_name
        self.split = split
        self.mode = mode
        self.max_loops = max_loops
        self.provider = provider
        self.model = model

    def load_queries(
        self,
        file_path: str,  # Not used for HF datasets, kept for interface compatibility
        task_ids: Optional[List[int]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Load queries from HuggingFace dataset."""
        try:
            from datasets import load_dataset

            logger.info(f"Loading HuggingFace dataset: {self.dataset_name}")
            if self.config_name:
                logger.info(f"   Config: {self.config_name}")
            logger.info(f"   Split: {self.split}")
            logger.info(f"   Query column: {self.query_column}")

            # Load dataset
            if self.config_name:
                dataset = load_dataset(
                    self.dataset_name, self.config_name, split=self.split
                )
            else:
                dataset = load_dataset(self.dataset_name, split=self.split)

            queries = []
            for i, example in enumerate(dataset):
                # Get query from specified column
                query = example.get(self.query_column, "")
                if isinstance(query, list):
                    query = " ".join(str(item) for item in query)
                elif not isinstance(query, str):
                    query = str(query)

                query = query.strip()
                if not query:
                    logger.warning(f"[WARN] Skipping empty query at index {i}")
                    continue

                # Use qid field as ID if available (for LiveResearchBench), otherwise use index
                qid = example.get("qid", i)
                query_data = {"id": qid, "index": i, "query": query}
                queries.append(query_data)

            logger.info(f"Loaded {len(queries)} queries from HuggingFace dataset")

            # Filter by task IDs if specified
            if task_ids:
                queries = [q for q in queries if q["id"] in task_ids]
                logger.info(f"Filtered to {len(queries)} specific tasks: {task_ids}")

            # Apply limit if specified
            if limit:
                queries = queries[:limit]
                logger.info(f"Limited to first {len(queries)} tasks")

            return queries

        except ImportError:
            logger.error(
                "[ERROR] HuggingFace datasets library not installed. Run: pip install datasets"
            )
            raise
        except Exception as e:
            logger.error(f"[ERROR] Error loading HuggingFace dataset: {e}")
            raise

    def get_query_field(self) -> str:
        return "query"

    def get_processing_config(self) -> Dict:
        return {
            "max_loops_default": self.max_loops,
            "benchmark_mode": self.mode == "benchmark",
            "qa_mode": self.mode == "qa",
            "visualization_disabled": True,
        }

    def format_result(self, task_data: Dict, result_data: Dict) -> Dict:
        """Format result for HuggingFace dataset."""
        return result_data

    def get_output_filename(self, task_id: str) -> str:
        # Use qid directly as filename (for LiveResearchBench compatibility)
        return f"{task_id}.json"


# ==================== Task Execution ====================
async def run_single_research_task_with_trajectory(
    task_data: Dict,
    dataset_manager: BenchmarkDatasetManager,
    output_dir: str,
    provider: str = None,
    model: str = None,
    max_web_search_loops: int = 5,
    extra_effort: bool = False,
    minimum_effort: bool = False,
    qa_mode: bool = False,
    benchmark_mode: bool = False,
    visualization_disabled: bool = True,
    steering_enabled: bool = False,
    parallel_search_enabled: bool = True,
    parallel_search_max_concurrency: int = 4,
    task_manager=None,
    collect_trajectory: bool = False,
    save_md: bool = False,
) -> Tuple[bool, Dict, str]:
    """
    Run a single research task with optional trajectory capture.

    Args:
        collect_trajectory: If True, collect detailed trajectory data (disabled by default for benchmarks)
        save_md: If True, save markdown report as .md file immediately after generation

    Returns: (success: bool, result: Dict, error_message: str)
    """
    task_id = task_data.get("id", task_data.get("index", "unknown"))
    query_field = dataset_manager.get_query_field()
    query = task_data[query_field]

    # Ensure query is a string (handle cases where it might be a list)
    if isinstance(query, list):
        query = " ".join(str(item) for item in query)
    elif not isinstance(query, str):
        query = str(query)

    # Preserve the original query
    original_query = query

    task_start_time = datetime.now()

    if task_manager:
        await task_manager.rate_limit()

    log_msg = f"[Task {task_id}] Starting"
    if collect_trajectory:
        log_msg += " with trajectory capture"
    logger.info(log_msg)

    with STATS_LOCK:
        GLOBAL_STATS["tasks_started"] += 1
        GLOBAL_STATS["tasks_in_progress"] += 1

    # Initialize trajectory recorder only if requested
    recorder = None
    if collect_trajectory:
        recorder = ResearchTrajectoryRecorder()
        recorder.query = original_query  # Use original query for recorder

    try:
        from src.state import SummaryState
        from src.graph import create_graph

        if not provider:
            provider = os.environ.get("LLM_PROVIDER", "openai")
        if not model:
            model = os.environ.get("LLM_MODEL", "o3-mini")

        # Set environment variables (match working version)
        os.environ["MAX_WEB_RESEARCH_LOOPS"] = str(max_web_search_loops)
        os.environ["LLM_PROVIDER"] = provider
        os.environ["LLM_MODEL"] = model

        fresh_graph = create_graph()

        if isinstance(dataset_manager, DeepConsultDatasetManager):
            benchmark_type = "DEEPCONSULT"
        elif isinstance(dataset_manager, HFDatasetManager):
            benchmark_type = "LRB"
        elif isinstance(dataset_manager, HealthBenchDatasetManager):
            benchmark_type = "HEALTHBENCH"
        elif isinstance(dataset_manager, DRBDatasetManager):
            benchmark_type = "DRB"
        else:
            benchmark_type = "UNKNOWN"

        run_ref = f"EVAL_{benchmark_type}_{task_id}"

        graph_config = {
            "configurable": {
                "llm_provider": provider,
                "llm_model": model,
                "max_web_research_loops": max_web_search_loops,
                "user_prompt": query,
                "parallel_search_enabled": parallel_search_enabled,
                "parallel_search_max_concurrency": parallel_search_max_concurrency,
            },
            "recursion_limit": 100,
            "tags": [
                f"provider:{provider}",
                f"model:{model}",
                f"loops:{max_web_search_loops}",
                f"task_id:{task_id}",
                f"benchmark:{benchmark_type}",
                f"parallel:{parallel_search_enabled}",
                f"parallel_max:{parallel_search_max_concurrency}",
                "eval_trajectory",
            ],
            "metadata": {
                "run_ref": run_ref,
                "query": query,
                "provider": provider,
                "model": model,
                "max_loops": max_web_search_loops,
                "benchmark": benchmark_type,
            },
        }

        initial_state = SummaryState(
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
            qa_mode=qa_mode,
            benchmark_mode=benchmark_mode,
            visualization_disabled=visualization_disabled,
            llm_provider=provider,
            llm_model=model,
            uploaded_knowledge=None,
            uploaded_files=[],
            uploaded_images=[],
            current_node=None,
            previous_node=None,
            steering_enabled=steering_enabled,
            parallel_search_enabled=parallel_search_enabled,
            parallel_search_max_concurrency=parallel_search_max_concurrency,
        )

        graph_start_time = datetime.now()
        logger.info(f"[Task {task_id}] Starting graph execution...")

        result = None
        last_started_iteration = -1
        previous_state = {}

        # Stream through graph and capture trajectory (same as working version)
        async for state_update in fresh_graph.astream(initial_state, graph_config):
            # state_update is a dict with node name as key and state changes as value
            for node_name, state_data in state_update.items():
                # Skip if not actual state data
                if not isinstance(state_data, dict):
                    continue

                # Get current research loop count
                current_loop = state_data.get("research_loop_count", 0)

                # Start a new iteration if not already started for this loop
                # (Don't end previous iteration here - let reflection node handle it)
                if recorder and (
                    recorder.current_iteration is None
                    and current_loop != last_started_iteration
                ):
                    recorder.start_iteration(current_loop)
                    last_started_iteration = current_loop
                    logger.info(f"[Task {task_id}] Started iteration {current_loop}")

                # Capture query decomposition from research_plan
                research_plan = state_data.get("research_plan")
                if research_plan and research_plan != previous_state.get(
                    "research_plan"
                ):
                    if recorder and recorder.current_iteration is not None:
                        # Use research_topic for initial query, search_query for follow-ups
                        query = state_data.get("search_query") or state_data.get(
                            "research_topic", ""
                        )
                        recorder.add_tool_call(
                            function_name="decompose_query",
                            arguments={
                                "query": query,
                                "knowledge_gap": state_data.get("knowledge_gap", ""),
                            },
                            result=research_plan,
                        )
                        logger.info(
                            f"[Task {task_id}] Captured decompose_query tool call"
                        )

                # Capture search tool calls from web_research_results
                web_results = state_data.get("web_research_results", [])
                prev_web_results = previous_state.get("web_research_results", [])
                if web_results and len(web_results) > len(prev_web_results):
                    if recorder and recorder.current_iteration is not None:
                        # Capture only NEW search results
                        new_results = web_results[len(prev_web_results) :]
                        for result in new_results:
                            query = result.get("query", "")
                            tool_name = result.get("tool", "general_search")
                            sources = result.get("sources", [])

                            recorder.add_tool_call(
                                function_name=tool_name,
                                arguments={"query": query},
                                result={
                                    "num_sources": len(sources),
                                    "sources": sources,
                                },
                            )
                        logger.info(
                            f"[Task {task_id}] Captured {len(new_results)} search tool calls"
                        )

                # Capture generate_report (synthesis) when running_summary changes
                running_summary = state_data.get("running_summary", "")
                prev_running_summary = previous_state.get("running_summary", "")
                if running_summary and running_summary != prev_running_summary:
                    if recorder and recorder.current_iteration is not None:
                        # Capture the synthesis step as a tool call
                        recorder.add_tool_call(
                            function_name="generate_report",
                            arguments={
                                "existing_summary_length": len(prev_running_summary),
                                "new_research_results": len(web_results),
                                "knowledge_gap": state_data.get("knowledge_gap", ""),
                            },
                            result={
                                "updated_summary_length": len(running_summary),
                                "num_sources_cited": len(
                                    state_data.get("source_citations", {})
                                ),
                            },
                        )
                        logger.info(
                            f"[Task {task_id}] Captured generate_report synthesis call"
                        )

                    # Also record in running_summary field
                    if recorder:
                        recorder.record_running_summary(running_summary)

                # Capture finalize_report when final_summary is created
                final_summary = state_data.get("final_summary", "")
                prev_final_summary = previous_state.get("final_summary", "")
                if final_summary and not prev_final_summary:
                    if recorder and recorder.current_iteration is not None:
                        # Capture the finalization step
                        recorder.add_tool_call(
                            function_name="finalize_report",
                            arguments={
                                "running_summary_length": len(running_summary),
                                "total_sources": len(
                                    state_data.get("sources_gathered", [])
                                ),
                                "has_visualizations": len(
                                    state_data.get("visualizations", [])
                                )
                                > 0,
                            },
                            result={
                                "final_report_length": len(final_summary),
                                "formatted_sources": len(
                                    state_data.get("source_citations", {})
                                ),
                            },
                        )
                        logger.info(
                            f"[Task {task_id}] Captured finalize_report formatting call"
                        )

                # Record sources when they change - compare actual content for accuracy
                sources = state_data.get("sources_gathered", [])
                previous_sources = previous_state.get("sources_gathered", [])

                if sources:
                    # Convert to sets for accurate comparison (handles additions AND replacements)
                    current_sources_set = set(sources)
                    previous_sources_set = (
                        set(previous_sources) if previous_sources else set()
                    )

                    # Only record if the actual source content changed
                    if current_sources_set != previous_sources_set:
                        new_sources = current_sources_set - previous_sources_set
                        logger.info(
                            f"[Task {task_id}] Iter {current_loop}: "
                            f"Sources updated: +{len(new_sources)} new (total: {len(current_sources_set)})"
                        )
                        if recorder:
                            recorder.record_sources(sources)
                    else:
                        # Log when sources exist but haven't changed (for debugging)
                        logger.debug(
                            f"[Task {task_id}] Iter {current_loop}: "
                            f"Sources unchanged ({len(current_sources_set)} total)"
                        )

                # Check for reflection completion (capture and end iteration)
                if (
                    node_name == "reflect_on_research"
                    or node_name == "reflect_on_report"
                    or "reflect" in node_name.lower()
                ):
                    if recorder and recorder.current_iteration is not None:
                        research_complete = state_data.get("research_complete", False)
                        logger.info(f"[Task {task_id}] Reflection completed")
                        logger.info(
                            f"[Task {task_id}]    Research complete: {research_complete}"
                        )

                        # Capture reflection output as a tool call
                        reflection_result = {
                            "research_complete": research_complete,
                            "knowledge_gap": state_data.get("knowledge_gap", ""),
                            "follow_up_query": state_data.get("search_query", ""),
                            "section_gaps": state_data.get("section_gaps", {}),
                            "priority_section": state_data.get("priority_section", ""),
                            "evaluation_notes": state_data.get("evaluation_notes", ""),
                            "research_topic": state_data.get("research_topic", ""),
                        }

                        # Add todo_updates if present (from steering system)
                        if "todo_updates" in state_data:
                            reflection_result["todo_updates"] = state_data[
                                "todo_updates"
                            ]

                        # Add reflection as tool call with empty arguments (OpenAI format)
                        recorder.add_tool_call(
                            function_name="reflect_on_report",
                            arguments={},  # Empty args as requested
                            result=reflection_result,
                        )
                        logger.info(
                            f"[Task {task_id}]    [OK] Captured reflection tool call"
                        )

                        # Log iteration summary before ending
                        if recorder:
                            iter_num = recorder.current_iteration.get("iteration", "?")
                            iter_sources = recorder.current_iteration.get(
                                "num_sources", 0
                            )
                            iter_tools = len(
                                recorder.current_iteration.get("tool_calls", [])
                            )
                            logger.info(
                                f"[Task {task_id}] Ending iteration {iter_num} (after reflection): "
                                f"{iter_tools} tool_calls, {iter_sources} new sources"
                            )
                            recorder.end_iteration()
                        # DON'T start the next iteration here - let the loop count change handle it

                # Update previous state
                previous_state = state_data.copy()

                # Store final result
                result = state_data

        # End final iteration if still active
        if recorder and recorder.current_iteration is not None:
            recorder.end_iteration()

        graph_end_time = datetime.now()
        graph_duration = graph_end_time - graph_start_time

        if not result:
            raise Exception("Graph execution produced no final result")

        logger.info(
            f"[Task {task_id}] Graph completed in {graph_duration.total_seconds():.2f}s"
        )

        # Extract final content (match working version logic)
        final_summary = result.get("running_summary", "No summary generated")

        # Handle case where summary might be a list (convert to string)
        if isinstance(final_summary, list):
            final_summary = "\n\n".join(str(s) for s in final_summary)
        elif not isinstance(final_summary, str):
            final_summary = str(final_summary)

        if qa_mode or benchmark_mode:
            benchmark_result = result.get("benchmark_result", {})
            final_content = (
                benchmark_result.get("full_response", final_summary)
                if benchmark_result
                else final_summary
            )
        else:
            markdown_report = result.get("markdown_report", "")
            # Handle if markdown_report is also a list or other non-string type
            if isinstance(markdown_report, list):
                markdown_report = "\n\n".join(str(s) for s in markdown_report)
            elif not isinstance(markdown_report, str):
                markdown_report = str(markdown_report) if markdown_report else ""

            if markdown_report and markdown_report.strip():
                # Find the start of Executive Summary section and trim TOC
                exec_summary_start = markdown_report.find("## Executive Summary\n")
                if exec_summary_start >= 0:
                    final_content = markdown_report[exec_summary_start:]
                    logger.info(
                        f"[Task {task_id}] Using clean markdown report (from Executive Summary)"
                    )
                else:
                    final_content = markdown_report
                    logger.info(
                        f"[Task {task_id}] Using complete markdown report (no Executive Summary found)"
                    )
            else:
                final_content = final_summary
                logger.info(
                    f"[Task {task_id}] Using running summary (no markdown report available)"
                )

        task_end_time = datetime.now()
        total_duration = task_end_time - task_start_time

        # Calculate total unique sources
        total_unique_sources = len(result.get("sources_gathered", []))

        # Create result data
        result_data = {
            "id": task_id,
            "query": original_query,  # Use original query
            "article": final_content,
            "summary": final_summary,
            "timing": {
                "start_time": task_start_time.isoformat(),
                "end_time": task_end_time.isoformat(),
                "total_duration_seconds": total_duration.total_seconds(),
                "graph_execution_seconds": graph_duration.total_seconds(),
            },
            "debug_info": {
                "research_loops": result.get("research_loop_count", 0),
                "sources_gathered": total_unique_sources,
                "knowledge_gap": result.get("knowledge_gap", ""),
                "selected_search_tool": result.get("selected_search_tool", "unknown"),
                "research_complete": result.get("research_complete", False),
            },
            "content_stats": {
                "final_content_length": len(final_content.split()),
                "final_summary_length": len(final_summary.split()),
            },
        }

        # Format and save result
        formatted_result = dataset_manager.format_result(task_data, result_data)
        output_filename = dataset_manager.get_output_filename(str(task_id))
        output_file = os.path.join(output_dir, output_filename)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(formatted_result, f, indent=2, ensure_ascii=False)

        # Save markdown file if requested
        if save_md:
            md_filename = Path(output_filename).stem + ".md"
            md_file = os.path.join(output_dir, md_filename)
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(final_content)
            logger.info(f"[Task {task_id}] Markdown saved to: {md_file}")

        # Save trajectory
        trajectory_data = {
            "run_ref": run_ref,
            "query": original_query,  # Use original query for trajectory too
            "final_report_markdown": final_content,
            "start_time": task_start_time.isoformat(),
            "end_time": task_end_time.isoformat(),
            "duration_seconds": total_duration.total_seconds(),
            "configuration": {
                "provider": provider,
                "model": model,
                "max_loops": max_web_search_loops,
                "extra_effort": extra_effort,
                "qa_mode": qa_mode,
                "benchmark": benchmark_type,
            },
            "trajectory": (
                recorder.to_dict(total_unique_sources=total_unique_sources)
                if recorder
                else None
            ),
        }

        # Save trajectory data if collection was enabled
        if recorder:
            trajectory_filename = f"trajectory_{task_id}.json"
            trajectory_file = os.path.join(output_dir, trajectory_filename)
            with open(trajectory_file, "w", encoding="utf-8") as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
            logger.info(f"[Task {task_id}] Trajectory saved to: {trajectory_file}")

        logger.info(f"[Task {task_id}] [OK] Completed successfully")
        logger.info(f"[Task {task_id}] Result saved to: {output_file}")

        with STATS_LOCK:
            GLOBAL_STATS["completed"] += 1
            GLOBAL_STATS["tasks_in_progress"] -= 1

        return True, formatted_result, ""

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[Task {task_id}] [ERROR] Failed: {error_msg}")

        with STATS_LOCK:
            GLOBAL_STATS["failed"] += 1
            GLOBAL_STATS["tasks_in_progress"] -= 1

        # Save error info
        error_data = {
            "id": task_id,
            "query": query,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }
        error_file = os.path.join(output_dir, f"error_{task_id}.json")
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(error_data, f, indent=2, ensure_ascii=False)

        return False, {}, error_msg


class ConcurrentTaskManager:
    """Manages concurrent task execution with rate limiting."""

    def __init__(self, max_concurrent: int = 5, requests_per_minute: int = 60):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limit_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.lock = asyncio.Lock()

    async def rate_limit(self):
        """Apply rate limiting."""
        async with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_interval:
                await asyncio.sleep(self.rate_limit_interval - time_since_last)
            self.last_request_time = time.time()

    async def run_task(self, coro):
        """Run a task with semaphore control."""
        async with self.semaphore:
            return await coro


async def process_tasks_concurrently(
    tasks: List[Dict],
    dataset_manager: BenchmarkDatasetManager,
    output_dir: str,
    provider: str,
    model: str,
    max_concurrent: int = 5,
    **kwargs,
):
    """Process multiple tasks concurrently with progress monitoring."""
    logger.info(f"process_tasks_concurrently called with {len(tasks)} tasks")

    # Initialize global stats
    with STATS_LOCK:
        GLOBAL_STATS["start_time"] = time.time()
        GLOBAL_STATS["completed"] = 0
        GLOBAL_STATS["failed"] = 0
        GLOBAL_STATS["tasks_started"] = 0
        GLOBAL_STATS["tasks_in_progress"] = 0

    task_manager = ConcurrentTaskManager(max_concurrent=max_concurrent)

    # Start progress monitoring task
    async def monitor_progress():
        """Monitor and log progress periodically."""
        while True:
            await asyncio.sleep(10)  # Update every 10 seconds
            with STATS_LOCK:
                if GLOBAL_STATS["start_time"]:
                    elapsed = time.time() - GLOBAL_STATS["start_time"]
                    total_tasks = len(tasks)
                    completed = GLOBAL_STATS["completed"]
                    failed = GLOBAL_STATS["failed"]
                    in_progress = GLOBAL_STATS["tasks_in_progress"]

                    completion_rate = completed / elapsed if elapsed > 0 else 0
                    eta = (
                        (total_tasks - completed - failed) / completion_rate
                        if completion_rate > 0
                        else 0
                    )

                    logger.info(
                        f"Progress: {completed}/{total_tasks} completed, "
                        f"{failed} failed, {in_progress} in progress, "
                        f"ETA: {eta/60:.1f}min"
                    )

                    if completed + failed >= total_tasks:
                        break

    # Start monitoring
    monitor_task = asyncio.create_task(monitor_progress())

    try:
        # Create task coroutines
        logger.info(f"Creating coroutines for {len(tasks)} tasks...")
        coroutines = [
            task_manager.run_task(
                run_single_research_task_with_trajectory(
                    task_data=task,
                    dataset_manager=dataset_manager,
                    output_dir=output_dir,
                    provider=provider,
                    model=model,
                    task_manager=task_manager,
                    **kwargs,
                )
            )
            for task in tasks
        ]
        logger.info(f"Created {len(coroutines)} coroutines, executing...")

        # Execute all tasks
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Cancel monitoring
        monitor_task.cancel()

        # Process results
        successful_results = []
        failed_results = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append(f"Task {tasks[i]['id']}: {str(result)}")
                logger.error(f"Task {tasks[i]['id']} failed with exception: {result}")
            else:
                success, data, error_msg = result
                if success:
                    successful_results.append(data)
                else:
                    failed_results.append(error_msg)

        # Final statistics
        with STATS_LOCK:
            total_time = time.time() - GLOBAL_STATS["start_time"]

        logger.info(f"\n{'='*80}")
        logger.info(f"CONCURRENT PROCESSING COMPLETE!")
        logger.info(f"{'='*80}")
        logger.info(f"[OK] Successful: {len(successful_results)}")
        logger.info(f"[ERROR] Failed: {len(failed_results)}")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Average time per task: {total_time/len(tasks):.2f} seconds")
        logger.info(
            f"Throughput: {len(successful_results)/total_time*60:.2f} tasks/minute"
        )

        if failed_results:
            logger.warning(f"\n[ERROR] Failed tasks:")
            for error in failed_results:
                logger.warning(f"  - {error}")

        return successful_results, failed_results

    except Exception as e:
        monitor_task.cancel()
        raise


# ==================== Helper Functions ====================
def get_default_file_paths(benchmark_type: str) -> Dict[str, str]:
    """Get default file paths for different benchmarks."""
    # Script directory (benchmarks repos should be cloned here)
    script_dir = Path(__file__).parent

    if benchmark_type == "drb":
        return {
            "queries_file": str(
                script_dir
                / "deep_research_bench"
                / "data"
                / "prompt_data"
                / "query.jsonl"
            ),
            "output_dir": str(
                script_dir / "deep_research_bench" / "results" / "edr_reports_gemini"
            ),
        }
    elif benchmark_type == "deepconsult":
        return {
            "queries_file": str(
                script_dir
                / "ydc-deep-research-evals"
                / "datasets"
                / "DeepConsult"
                / "queries.csv"
            ),
            "output_dir": str(
                script_dir
                / "ydc-deep-research-evals"
                / "results"
                / "edr_reports_gemini"
            ),
        }
    elif benchmark_type == "lrb":
        return {
            "queries_file": "",  # Not used for HuggingFace datasets
            "output_dir": str(
                script_dir / "liveresearchbench" / "results" / "edr_reports_gemini"
            ),
        }
    elif benchmark_type == "healthbench":
        return {
            "queries_file": str(
                Path.home()
                / "Documents"
                / "scratch"
                / "healthbench_data"
                / "final_run_100.json"
            ),
            "output_dir": str(script_dir / "healthbench"),
        }
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(
        description="Run research benchmarks concurrently (DRB, DeepConsult, LiveResearchBench, and HealthBench). Optional trajectory capture via --collect-traj."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["drb", "deepconsult", "lrb", "healthbench"],
        help="Benchmark to run (drb, deepconsult, lrb for LiveResearchBench, or healthbench)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input file (CSV for DeepConsult, JSONL for DRB). If not specified, uses default path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. If not specified, uses default path.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "groq", "google"],
        default="google",
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="LLM model name (e.g., 'o3-mini', 'claude-3-5-sonnet', 'gemini-2.5-pro')",
    )
    parser.add_argument(
        "--max_loops",
        type=int,
        default=None,
        help="Max research loops (defaults to benchmark-specific value)",
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=5, help="Max concurrent tasks"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument(
        "--task_ids", type=str, default=None, help="Comma-separated task IDs"
    )
    parser.add_argument("--extra_effort", action="store_true", help="Extra effort mode")
    parser.add_argument(
        "--minimum_effort", action="store_true", help="Minimum effort mode"
    )
    parser.add_argument(
        "--collect-traj",
        action="store_true",
        help="Collect detailed trajectory data (disabled by default for benchmarks)",
    )
    parser.add_argument(
        "--save_md",
        action="store_true",
        help="Save markdown report as .md file immediately after generation",
    )
    parser.add_argument(
        "--enable-steering",
        action="store_true",
        help="Enable steering/task management during benchmark runs",
    )
    parser.add_argument(
        "--parallel-search",
        action="store_true",
        help="Enable parallel search execution for all tasks",
    )
    parser.add_argument(
        "--parallel-max",
        type=int,
        default=4,
        help="Max concurrent search subtasks when parallelized",
    )

    args = parser.parse_args()

    # Get default paths if not specified
    default_paths = get_default_file_paths(args.benchmark)
    input_file = args.input or default_paths["queries_file"]
    output_dir = args.output_dir or default_paths["output_dir"]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Select dataset manager
    if args.benchmark == "drb":
        dataset_manager = DRBDatasetManager()
    elif args.benchmark == "deepconsult":
        dataset_manager = DeepConsultDatasetManager()
    elif args.benchmark == "lrb":
        # LiveResearchBench with defaults: max_loops=5, gemini model, regular mode
        dataset_manager = HFDatasetManager(
            dataset_name="Salesforce/LiveResearchBench",
            query_column="question_no_placeholder",
            config_name="question_only",
            split="test",
            mode="regular",
            max_loops=5,
            provider=args.provider,
            model=args.model,
        )
    elif args.benchmark == "healthbench":
        dataset_manager = HealthBenchDatasetManager()
    else:
        raise ValueError(f"Unknown benchmark type: {args.benchmark}")

    # Parse task IDs if provided
    task_ids = None
    if args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",")]

    # Load tasks
    logger.info(f"Loading tasks from {input_file}")
    tasks = dataset_manager.load_queries(
        input_file, task_ids=task_ids, limit=args.limit
    )
    logger.info(f"Loaded {len(tasks)} tasks")

    if not tasks:
        logger.error("No tasks to process!")
        sys.exit(1)

    # Filter out already-processed tasks (check if output file exists in output_dir)
    original_task_count = len(tasks)
    filtered_tasks = []
    skipped_tasks = []

    logger.info(f"Checking for existing files in: {output_dir}")

    for task in tasks:
        task_id = task.get("id", task.get("index", "unknown"))
        output_filename = dataset_manager.get_output_filename(str(task_id))
        output_file = os.path.join(output_dir, output_filename)

        if os.path.exists(output_file):
            skipped_tasks.append((task_id, "already processed"))
            logger.info(f"Skipping task {task_id}: already processed")
        else:
            filtered_tasks.append(task)

    # Update tasks list
    tasks = filtered_tasks

    logger.info(f"\n{'='*80}")
    logger.info(f"TASK FILTERING SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Original tasks: {original_task_count}")
    logger.info(f"Already processed (skipped): {len(skipped_tasks)}")
    logger.info(f"Remaining tasks to process: {len(tasks)}")
    logger.info(f"{'='*80}\n")

    if not tasks:
        logger.info("[OK] All tasks have already been processed!")
        sys.exit(0)

    # Get processing config from dataset manager
    config = dataset_manager.get_processing_config()

    # Determine max_loops: use command-line arg if provided, otherwise use dataset default
    if args.max_loops is None:
        max_loops = config.get("max_loops", config.get("max_loops_default", 10))
        logger.info(f"Using dataset default max_loops: {max_loops}")
    else:
        max_loops = args.max_loops
        logger.info(f"Using user-specified max_loops: {max_loops}")

    # Get other config values
    benchmark_mode = config.get("benchmark_mode", False)
    qa_mode = config.get("qa_mode", False)
    logger.info(f"Using benchmark_mode: {benchmark_mode}, qa_mode: {qa_mode}")

    # Start execution
    GLOBAL_STATS["start_time"] = time.time()
    logger.info(f"Starting concurrent execution with {args.max_concurrent} workers")

    # Run tasks
    try:
        successful_results, failed_results = asyncio.run(
            process_tasks_concurrently(
                tasks=tasks,
                dataset_manager=dataset_manager,
                output_dir=output_dir,
                provider=args.provider,
                model=args.model,
                max_concurrent=args.max_concurrent,
                max_web_search_loops=max_loops,
                extra_effort=args.extra_effort,
                minimum_effort=args.minimum_effort,
                qa_mode=qa_mode,  # Use config value
                benchmark_mode=benchmark_mode,  # Use config value
                visualization_disabled=True,
                steering_enabled=args.enable_steering,
                parallel_search_enabled=args.parallel_search,
                parallel_search_max_concurrency=args.parallel_max,
                collect_trajectory=args.collect_traj,
                save_md=args.save_md,
            )
        )
        logger.info(
            f"Completed execution: {len(successful_results)} successful, {len(failed_results)} failed"
        )
    except Exception as e:
        logger.error(f"Fatal error during execution: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Print final stats
    end_time = time.time()
    total_duration = end_time - GLOBAL_STATS["start_time"]
    logger.info("\n" + "=" * 60)
    logger.info("FINAL STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total completed: {GLOBAL_STATS['completed']}")
    logger.info(f"Total failed: {GLOBAL_STATS['failed']}")
    logger.info(f"Total duration: {total_duration:.2f}s")
    logger.info(f"Average time per task: {total_duration / len(tasks):.2f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
