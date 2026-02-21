#!/usr/bin/env python3
"""
Clean script to run the deep research agent with custom parameters.
This script bypasses the UI and directly calls the backend to get a markdown report.

Usage:
    python run_research.py "Your research query here"
    python run_research.py "Climate change" --extra-effort --provider openai --model gpt-4o
    python run_research.py "Who is the current CEO of Tesla?" --qa-mode
    python run_research.py "AI research" --enable-steering --steering-message "Focus on practical applications"
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime
import uuid
from typing import List

# Add the deep-research directory to the Python path
script_dir = Path(__file__).parent
deep_research_dir = script_dir.parent
sys.path.insert(0, str(deep_research_dir))


# Load environment variables from the correct .env file
env_file_path = deep_research_dir / ".env"
print(f"Loading environment from: {env_file_path}")
load_dotenv(dotenv_path=env_file_path)


async def run_research_sync(
    query: str,
    max_web_search_loops: int = 1,
    visualization_disabled: bool = True,
    extra_effort: bool = False,
    minimum_effort: bool = False,
    qa_mode: bool = False,
    benchmark_mode: bool = False,
    provider: str = None,
    model: str = None,
    output_file: str = None,
    file_path: str = None,
    steering_enabled: bool = False,
    steering_messages: List[str] = None,
):
    """
    Run the research agent with specified parameters.

    Args:
        query: The research query/topic
        max_web_search_loops: Maximum number of web search loops (default: 10)
        visualization_disabled: Whether to disable visualizations (default: True)
        extra_effort: Whether to use extra effort mode (default: False)
        minimum_effort: Whether to use minimum effort mode (default: False)
        qa_mode: Whether to run in QA mode (simple question-answering) (default: False)
        benchmark_mode: Whether to run in benchmark mode (QA with full citations) (default: False)
        provider: LLM provider ('openai', 'anthropic', 'groq', 'google')
        model: LLM model name
        output_file: Optional file path to save the result
        file_path: Optional file path to analyze and include in research
        steering_enabled: Whether to enable steering for this research session
        steering_messages: List of steering messages to apply during research

    Returns:
        The research result (markdown or JSON)
    """

    # Start timing
    start_time = datetime.now()
    print(f"Starting research on: {query}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Visualizations: {'Disabled' if visualization_disabled else 'Enabled'}")
    print(f"Max search loops: {max_web_search_loops}")
    print(f"Extra effort: {extra_effort}")
    print(f"Minimum effort: {minimum_effort}")
    print(f"QA mode: {qa_mode}")
    print(f"Benchmark mode: {benchmark_mode}")
    print(f"Steering enabled: {steering_enabled}")
    if steering_enabled and steering_messages:
        print(f"Steering messages: {len(steering_messages)}")
        for i, msg in enumerate(steering_messages, 1):
            print(f"   {i}. {msg}")

    # Set up provider and model with defaults
    if not provider:
        provider = os.environ.get("LLM_PROVIDER", "openai")
    if not model:
        model = os.environ.get("LLM_MODEL", "o3-mini")

    print(f"Using {provider}/{model}")

    run_name = f"{provider}_{model}_{max_web_search_loops}"
    print(f"LangSmith run name: {run_name}")

    # Process uploaded file if provided
    uploaded_data_content = None
    uploaded_files = []
    analyzed_files = []

    if file_path:
        print(f"Processing file: {file_path}")

        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            return None

        try:
            # Import file services
            from services.file_storage import FileStorageService
            from services.content_analysis import ContentAnalysisService
            import shutil
            from pathlib import Path

            # Store the file by copying it to the upload directory
            file_id = str(uuid.uuid4())
            original_name = os.path.basename(file_path)

            print(f"Processing file: {original_name}")

            # Copy file to uploads directory and register it
            FileStorageService._ensure_upload_directory()
            file_extension = Path(original_name).suffix.lower()
            sanitized_name = FileStorageService._sanitize_filename(original_name)
            stored_filename = f"{file_id}_{sanitized_name}"
            stored_path = os.path.join(FileStorageService.UPLOAD_DIR, stored_filename)

            # Copy the file
            shutil.copy2(file_path, stored_path)

            # Get file stats
            file_stats = os.stat(stored_path)
            file_size = file_stats.st_size

            # Register file in the storage service
            file_metadata = {
                "file_id": file_id,
                "filename": stored_filename,
                "original_name": original_name,
                "file_path": stored_path,
                "file_type": file_extension[1:] if file_extension else "unknown",
                "file_size": file_size,
                "upload_timestamp": datetime.now(),
                "content_type": "application/octet-stream",
            }

            FileStorageService._file_registry[file_id] = file_metadata
            print(f"[OK] File stored with ID: {file_id}")

            # Analyze the file
            print("Analyzing file content...")
            analysis_result = await ContentAnalysisService.analyze_file(
                file_id, "comprehensive"
            )

            if analysis_result and analysis_result.status.value == "completed":
                uploaded_data_content = analysis_result.content_description
                uploaded_files = [file_id]

                # Create analyzed_files structure like UI mode
                analyzed_files = [
                    {
                        "file_id": file_id,
                        "content_description": analysis_result.content_description,
                        "metadata": analysis_result.metadata,
                    }
                ]

                print("[OK] File analyzed successfully")
            else:
                print(
                    f"[ERROR] File analysis failed: {analysis_result.error_message if analysis_result else 'Unknown error'}"
                )

        except Exception as e:
            print(f"[ERROR] Error processing file: {str(e)}")
            print("Continuing research without file content...")
            uploaded_data_content = None
            uploaded_files = []
            analyzed_files = []

    # Set environment variables for configuration
    original_max_loops = os.environ.get("MAX_WEB_RESEARCH_LOOPS")
    # LangSmith is optional - only used if LANGCHAIN_API_KEY is configured
    original_langsmith_project = os.environ.get("LANGCHAIN_PROJECT")

    os.environ["MAX_WEB_RESEARCH_LOOPS"] = str(max_web_search_loops)
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model

    try:
        from src.state import SummaryState
        from src.graph import create_graph

        fresh_graph = create_graph()

        # Generate unique run reference
        run_ref = f"test_{provider}_{model}_{max_web_search_loops}"

        # Create graph configuration
        graph_config = {
            "configurable": {
                "llm_provider": provider,
                "llm_model": model,
                "max_web_search_loops": max_web_search_loops,
                "user_prompt": query,
            },
            "recursion_limit": 100,
            "run_name": run_name,
            "tags": [
                f"provider:{provider}",
                f"model:{model}",
                f"loops:{max_web_search_loops}",
            ],
            "metadata": {
                "benchmark": "rqa_test",
                "run_ref": run_ref,
                "query": query,
                "provider": provider,
                "model": model,
                "max_loops": max_web_search_loops,
                "visualization_disabled": visualization_disabled,
            },
        }

        # Create initial state as SummaryState object (not dictionary)
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
            uploaded_knowledge=uploaded_data_content,
            uploaded_files=uploaded_files,
            analyzed_files=analyzed_files,
            steering_enabled=steering_enabled,  # Enable steering
        )

        # Apply steering messages if provided
        if steering_enabled and steering_messages:
            print(f"\nApplying {len(steering_messages)} steering messages...")
            for i, message in enumerate(steering_messages, 1):
                print(f"   {i}. Processing: '{message}'")
                try:
                    result = await initial_state.add_steering_message(message)
                    print(f"      → Created {result['pending_tasks']} tasks")
                except Exception as e:
                    print(f"      [ERROR] Error: {e}")

            # Show the steering plan
            if initial_state.steering_todo:
                print(f"\nCurrent Steering Plan:")
                plan_lines = initial_state.get_steering_plan().split("\n")
                for line in plan_lines[:15]:  # Show first 15 lines
                    print(f"   {line}")
                if len(plan_lines) > 15:
                    print(f"   ... ({len(plan_lines) - 15} more lines)")
                print()

        print(f"\n{'='*80}")
        print("Starting research execution...")
        print(f"{'='*80}\n")

        # Time the graph execution
        graph_start_time = datetime.now()

        # Run the graph using async method
        result = await fresh_graph.ainvoke(initial_state, config=graph_config)

        graph_end_time = datetime.now()
        graph_duration = graph_end_time - graph_start_time

        print(f"\n{'='*80}")
        print("[OK] Research completed!")
        print(f"Graph execution time: {graph_duration.total_seconds():.2f} seconds")
        print(f"{'='*80}\n")

        # Extract the result based on mode
        final_summary = result.get("running_summary", "No summary generated")

        # Handle different modes appropriately
        if qa_mode or benchmark_mode:
            # For QA and benchmark modes, use the benchmark_result if available
            benchmark_result = result.get("benchmark_result", {})
            if benchmark_result:
                # Use the full_response (which includes citations for benchmark mode)
                final_content = benchmark_result.get("full_response", "")
                if not final_content:
                    # Fallback to structured answer if full_response is not available
                    answer = benchmark_result.get("answer", "")
                    confidence = benchmark_result.get("confidence_level", "")
                    evidence = benchmark_result.get("evidence", "")
                    limitations = benchmark_result.get("limitations", "")

                    final_content = f"**Answer:** {answer}\n\n"
                    if confidence:
                        final_content += f"**Confidence:** {confidence}\n\n"
                    if evidence:
                        final_content += f"**Supporting Evidence:** {evidence}\n\n"
                    if limitations:
                        final_content += f"**Limitations:** {limitations}\n\n"

                mode_name = "benchmark mode" if benchmark_mode else "QA mode"
                print(
                    f"Using {mode_name} result with {'citations' if benchmark_mode else 'basic sources'}"
                )
            else:
                final_content = final_summary
                print("No benchmark result available, using running summary")
        else:
            # Regular mode - prioritize markdown_report if available
            markdown_report = result.get("markdown_report", "")
            if markdown_report and markdown_report.strip():
                # Find the start of Executive Summary section
                exec_summary_start = markdown_report.find("## Executive Summary\n")
                if exec_summary_start >= 0:
                    final_content = markdown_report[exec_summary_start:]
                    print("Using clean markdown report (from Executive Summary)")
                else:
                    final_content = markdown_report
                    print(
                        "Using complete markdown report (no Executive Summary found)"
                    )
            else:
                final_content = final_summary
                print("Using running summary (no markdown report available)")

        # Calculate total execution time
        end_time = datetime.now()
        total_duration = end_time - start_time

        # Extract debugging information from result
        debug_info = {
            "research_loops": result.get("research_loop_count", 0),
            "sources_gathered": len(result.get("sources_gathered", [])),
            "visualizations": len(result.get("visualizations", [])),
            "knowledge_gap": result.get("knowledge_gap", ""),
            "selected_search_tool": result.get("selected_search_tool", "unknown"),
            "research_complete": result.get("research_complete", False),
        }

        # Add benchmark-specific debug info
        if qa_mode or benchmark_mode:
            benchmark_result = result.get("benchmark_result", {})
            debug_info.update(
                {
                    "benchmark_result_available": bool(benchmark_result),
                    "benchmark_confidence": (
                        benchmark_result.get("confidence", 0) if benchmark_result else 0
                    ),
                    "benchmark_answer_length": (
                        len(benchmark_result.get("answer", ""))
                        if benchmark_result
                        else 0
                    ),
                    "citations_processed": (
                        bool(benchmark_result.get("full_response", "").find("[") != -1)
                        if benchmark_result
                        else False
                    ),
                }
            )

        # Create comprehensive result data
        data_point = {
            "id": None,  # Will be set by caller if needed
            "run_ref": run_ref,  # Unique reference for this run
            "research_topic": query,
            "prompt": query,
            "article": final_content,
            "summary": final_summary,
            "timing": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_duration.total_seconds(),
                "graph_execution_seconds": graph_duration.total_seconds(),
                "setup_time_seconds": (graph_start_time - start_time).total_seconds(),
            },
            "configuration": {
                "provider": provider,
                "model": model,
                "max_web_search_loops": max_web_search_loops,
                "visualization_disabled": visualization_disabled,
                "extra_effort": extra_effort,
                "minimum_effort": minimum_effort,
                "qa_mode": qa_mode,
                "benchmark_mode": benchmark_mode,
            },
            "debug_info": debug_info,
            "content_stats": {
                "final_content_length": len(final_content),
                "final_summary_length": len(final_summary),
                "markdown_report_available": (
                    bool(
                        result.get("markdown_report", "")
                        and result.get("markdown_report", "").strip()
                    )
                    if not (qa_mode or benchmark_mode)
                    else False
                ),
                "benchmark_result_available": (
                    bool(result.get("benchmark_result", {}))
                    if (qa_mode or benchmark_mode)
                    else False
                ),
                "content_type": (
                    "benchmark_result"
                    if (qa_mode or benchmark_mode)
                    and result.get("benchmark_result", {})
                    else (
                        "markdown_report"
                        if (
                            result.get("markdown_report", "")
                            and result.get("markdown_report", "").strip()
                            and not (qa_mode or benchmark_mode)
                        )
                        else "running_summary"
                    )
                ),
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_point, f, indent=2, ensure_ascii=False)
        print(f"JSON result saved to: {output_file}")

        # Print comprehensive summary stats
        print("\nResearch Summary:")
        print(
            f"   Total time: {total_duration.total_seconds():.2f} seconds ({total_duration.total_seconds()/60:.2f} minutes)"
        )
        print(
            f"   Setup time: {(graph_start_time - start_time).total_seconds():.2f} seconds"
        )
        print(f"   Graph execution: {graph_duration.total_seconds():.2f} seconds")
        print(f"   Research loops: {debug_info['research_loops']}")
        print(f"   Sources gathered: {debug_info['sources_gathered']}")
        print(f"   Content length: {len(final_content.split()):,} words")
        print(f"   [OK] Research complete: {debug_info['research_complete']}")

        if debug_info["knowledge_gap"]:
            print(
                f"   Final knowledge gap: {debug_info['knowledge_gap'][:200]}{'...' if len(debug_info['knowledge_gap']) > 200 else ''}"
            )

        # Performance analysis
        word_count = len(final_content.split())
        throughput = (
            word_count / total_duration.total_seconds()
            if total_duration.total_seconds() > 0
            else 0
        )
        print(f"   Content throughput: {throughput:.1f} words/second")

        return data_point
    finally:
        # Restore original environment variables
        if original_max_loops is not None:
            os.environ["MAX_WEB_RESEARCH_LOOPS"] = original_max_loops
        elif "MAX_WEB_RESEARCH_LOOPS" in os.environ:
            del os.environ["MAX_WEB_RESEARCH_LOOPS"]

        if original_langsmith_project is not None:
            os.environ["LANGCHAIN_PROJECT"] = original_langsmith_project
        elif "LANGCHAIN_PROJECT" in os.environ:
            del os.environ["LANGCHAIN_PROJECT"]


async def main():
    """Main function to handle command line arguments and run research."""

    parser = argparse.ArgumentParser(
        description="Run deep research agent directly with custom parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_research.py "Climate change solutions" --max-loops 15 --extra-effort
  python run_research.py "Space exploration" --minimum-effort --model gpt-4o
  python run_research.py "Who is the current president of France?" --qa-mode
  python run_research.py "What is quantum entanglement?" --benchmark-mode --provider anthropic
  python run_research.py "AI research" --enable-steering --steering-message "Focus on ethics" --steering-message "Include recent papers"
        """,
    )

    # Required arguments
    parser.add_argument("query", help="The research query/topic to investigate")

    # Configuration arguments
    parser.add_argument(
        "--max-loops",
        type=int,
        default=1,
        help="Maximum number of web search loops (default: 10)",
    )
    parser.add_argument(
        "--disable-visualizations",
        action="store_true",
        help="Disable visualization generation",
    )
    parser.add_argument(
        "--extra-effort",
        action="store_true",
        help="Use extra effort mode for more thorough research",
    )
    parser.add_argument(
        "--minimum-effort",
        action="store_true",
        help="Use minimum effort mode for faster research",
    )
    parser.add_argument(
        "--qa-mode",
        action="store_true",
        help="Run in QA mode (simple question-answering without full citations)",
    )
    parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        help="Run in benchmark mode (QA with full citation processing)",
    )

    # LLM configuration
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "groq", "google"],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        help="LLM model name (e.g., 'o3-mini', 'claude-3-5-sonnet', 'gemini-2.5-pro')",
    )

    # Output configuration
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path to save the result",
        default="research_result.json",
    )

    # File analysis configuration
    parser.add_argument(
        "--file",
        "-f",
        help="File path to analyze and include in research",
        type=str,
    )

    # Steering configuration (disabled by default for benchmarks)
    parser.add_argument(
        "--enable-steering",
        action="store_true",
        help="Enable steering/task management (disabled by default for benchmarks)",
    )
    parser.add_argument(
        "--steering-message",
        action="append",
        help="Add a steering message (can be used multiple times)",
    )

    args = parser.parse_args()

    # Validate mutually exclusive modes
    if args.qa_mode and args.benchmark_mode:
        print("[ERROR] --qa-mode and --benchmark-mode are mutually exclusive.")
        print(
            "   Use --qa-mode for simple Q&A or --benchmark-mode for Q&A with full citations."
        )
        return 1

    # Check for required environment variables
    required_vars = []
    provider = args.provider or os.environ.get("LLM_PROVIDER", "openai")

    if provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif provider == "anthropic":
        required_vars.append("ANTHROPIC_API_KEY")
    elif provider == "groq":
        required_vars.append("GROQ_API_KEY")
    elif provider == "google":
        required_vars.extend(["GOOGLE_CLOUD_PROJECT"])

    # Always need search API
    required_vars.append("TAVILY_API_KEY")

    missing_vars = [var for var in required_vars if not os.environ.get(var)]

    if missing_vars:
        print("[ERROR] Missing required environment variables:")
        for var in missing_vars:
            print(f"   {var}")
        print("\nPlease set these environment variables before running.")
        return 1

    # Run the research
    try:
        result = await run_research_sync(
            query=args.query,
            max_web_search_loops=args.max_loops,
            visualization_disabled=args.disable_visualizations,
            extra_effort=args.extra_effort,
            minimum_effort=args.minimum_effort,
            qa_mode=args.qa_mode,
            benchmark_mode=args.benchmark_mode,
            provider=args.provider,
            model=args.model,
            output_file=args.output,
            file_path=args.file,
            steering_enabled=args.enable_steering,
            steering_messages=args.steering_message if args.steering_message else None,
        )

        if result:
            if not args.output:
                # Print result to console if no output file specified
                print(json.dumps(result, indent=2, ensure_ascii=False))

            # Print final success message with timing
            total_time = result.get("timing", {}).get("total_duration_seconds", 0)
            print(f"\nSUCCESS: Research completed in {total_time:.2f} seconds!")

            # Additional performance insights
            if result.get("debug_info", {}).get("research_loops", 0) > 1:
                print(
                    f"   Used {result['debug_info']['research_loops']} research loops"
                )
            if result.get("debug_info", {}).get("sources_gathered", 0) > 0:
                print(
                    f"   Gathered {result['debug_info']['sources_gathered']} sources"
                )

            return 0
        else:
            print("\n[ERROR] FAILED: Research could not be completed")
            return 1

    except KeyboardInterrupt:
        print("\nResearch interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
