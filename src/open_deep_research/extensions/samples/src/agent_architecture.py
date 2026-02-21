"""
Agent Architecture for Deep Research

This module implements a modular, agent-based architecture for research:
- Master Agent: Planning, query decomposition, and coordination
- Search Agent: Specialized for executing search queries
- Visualization Agent: Creates data visualizations from search results
- Result Combiner: Integrates findings from multiple sources

The architecture provides better separation of concerns while
maintaining backward compatibility with the existing codebase.
"""

import os
import json
import logging
import traceback
import re
import base64
from typing import Dict, List, Any, Optional
from openai import OpenAI
import asyncio
import uuid
from src.tools.executor import ToolExecutor
from src.visualization_agent import VisualizationAgent
import time


class MasterResearchAgent:
    """
    Master agent responsible for planning research, decomposing topics,
    and coordinating specialized agents.

    This agent serves as the "central brain" that breaks down complex queries
    into manageable subtasks and delegates them to specialized agents.
    """

    def __init__(self, config=None):
        """
        Initialize the Master Research Agent.

        Args:
            config: Configuration object containing LLM settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def decompose_topic(
        self,
        query,
        knowledge_gap,
        research_loop_count,
        uploaded_knowledge=None,
        existing_tasks=None,
    ):
        """
        Analyze a topic and decompose it if complex.
        Uses query_writer_instructions as the central brain for decomposition.

        Args:
            query: The main research query or topic
            knowledge_gap: Additional context about knowledge gaps to address
            research_loop_count: Current iteration of research loop
            uploaded_knowledge: User-provided external knowledge (optional)
            existing_tasks: List of existing pending tasks to avoid duplicates (optional)

        Returns:
            Dict containing topic_complexity, query/subtopics, and analysis
        """
        # Import necessary functions
        import os
        import json
        import traceback
        from llm_clients import get_llm_client, MODEL_CONFIGS

        try:
            # Get configuration
            if self.config is not None:
                from src.graph import get_configurable

                configurable = get_configurable(self.config)
                provider = configurable.llm_provider
                model = configurable.llm_model
            else:
                # Try environment variables first
                provider = os.environ.get("LLM_PROVIDER")
                model = os.environ.get("LLM_MODEL")

                # Use defaults if not found in environment
                if not provider:
                    provider = "azure"
                # Will use the default model from llm_clients.py if model is None

            # Get the model name from the configuration or default
            if not model and provider in MODEL_CONFIGS:
                model = MODEL_CONFIGS[provider]["default_model"]

            self.logger.info(
                f"[MasterAgent] Using provider='{provider}' (type={type(provider).__name__}), model='{model}' for topic decomposition"
            )

            # Generate date constants
            from datetime import datetime

            today = datetime.now()
            CURRENT_DATE = today.strftime("%B %d, %Y")
            CURRENT_YEAR = str(today.year)
            ONE_YEAR_AGO = str(today.year - 1)

            # Create a combined query that incorporates knowledge gaps if available
            combined_topic = query
            research_context = ""
            if knowledge_gap and knowledge_gap.strip():
                combined_topic = f"{query} - {knowledge_gap}"
                research_context = (
                    f"Knowledge gap identified in previous research: {knowledge_gap}"
                )

            # Prepare uploaded knowledge context
            AUGMENT_KNOWLEDGE_CONTEXT = ""
            if uploaded_knowledge and uploaded_knowledge.strip():
                AUGMENT_KNOWLEDGE_CONTEXT = f"""
USER-PROVIDED EXTERNAL KNOWLEDGE AVAILABLE:
The user has provided external knowledge/documentation that should be treated as highly authoritative and trustworthy. This uploaded knowledge should guide query generation to complement rather than duplicate existing information.

Uploaded Knowledge Preview: {uploaded_knowledge[:500]}{'...' if len(uploaded_knowledge) > 500 else ''}

Query Generation Strategy:
- Identify what information is already covered in the uploaded knowledge
- Generate queries that fill gaps or provide additional context to the uploaded knowledge
- Focus on recent developments, validation, or areas not covered in uploaded knowledge
- Use uploaded knowledge to inform more targeted and specific search queries
"""
            else:
                AUGMENT_KNOWLEDGE_CONTEXT = "No user-provided external knowledge available. Generate queries based on the research topic and knowledge gaps."

            # Import the query_writer_instructions prompt
            from src.prompts import query_writer_instructions

            # Import async client getter
            from llm_clients import get_async_llm_client

            # Get steering context if available
            steering_context = ""
            if (
                hasattr(self, "state")
                and self.state
                and hasattr(self.state, "steering_todo")
                and self.state.steering_todo
            ):
                steering_context = self.state.get_steering_plan()
                self.logger.info(f"[STEERING] Including steering context in prompt")
            else:
                steering_context = "No steering instructions provided. Follow standard research approach."

            # Create database context if databases are available
            database_context = ""
            self.logger.info(
                f"[MasterAgent.decompose_topic] Checking database_info: hasattr={hasattr(self, 'database_info')}, value={getattr(self, 'database_info', 'NOT_SET')}"
            )
            if hasattr(self, "database_info") and self.database_info:
                self.logger.info(
                    f"[MasterAgent.decompose_topic] Database info is available, generating database context"
                )
                database_list = []
                for db in self.database_info:
                    db_info = f"- {db.get('filename', 'Unknown')} ({db.get('file_type', 'unknown').upper()}) with {len(db.get('tables', []))} tables: {', '.join(db.get('tables', []))}"
                    database_list.append(db_info)

                database_context = f"""
DATABASE AVAILABLE:
{chr(10).join(database_list)}

TOOL SELECTION:
- Questions about this uploaded data → use "text2sql" tool
- Questions about external/general information → use search tools
- NEVER pass SQL code to search tools
"""
                self.logger.info(
                    f"[MasterAgent.decompose_topic] Generated database context: {database_context[:200]}..."
                )
            else:
                self.logger.info(
                    f"[MasterAgent.decompose_topic] No database info available, using default context"
                )
                database_context = (
                    "No database files are available. Use standard web search tools."
                )

            # Build existing tasks context for duplicate prevention
            existing_tasks_context = ""
            if existing_tasks and len(existing_tasks) > 0:
                self.logger.info(
                    f"[MasterAgent] Including {len(existing_tasks)} existing tasks for duplicate prevention"
                )
                existing_tasks_context = "\n\nIMPORTANT - EXISTING PENDING TASKS:\n"
                existing_tasks_context += "The following research tasks are already pending or in-progress. DO NOT create duplicate or highly similar tasks:\n\n"
                for task in existing_tasks:
                    existing_tasks_context += f"- [{task.id}] {task.description} (Priority: {task.priority}, Source: {task.source})\n"
                existing_tasks_context += "\nQuery Generation Strategy:\n"
                existing_tasks_context += "- Only generate queries for NEW aspects NOT covered by existing tasks\n"
                existing_tasks_context += "- Avoid creating semantically similar queries to existing task descriptions\n"
                existing_tasks_context += "- If existing tasks already cover the knowledge gap, generate fewer or zero new queries\n"

            # Format the prompt with the appropriate context
            formatted_prompt = query_writer_instructions.format(
                research_topic=combined_topic,
                research_context=research_context,
                current_date=CURRENT_DATE,
                current_year=CURRENT_YEAR,
                one_year_ago=ONE_YEAR_AGO,
                AUGMENT_KNOWLEDGE_CONTEXT=AUGMENT_KNOWLEDGE_CONTEXT,
                DATABASE_CONTEXT=database_context,
                steering_context=steering_context,
            )

            # Append existing_tasks_context after formatting (so it's not part of the template)
            if existing_tasks_context:
                formatted_prompt += existing_tasks_context

            self.logger.info(
                f"[MasterAgent.decompose_topic] Database context being passed to LLM: {database_context[:200]}..."
            )

            # Get the appropriate ASYNC LLM client based on provider
            # llm = get_llm_client(provider, model) # Old sync call
            llm = await get_async_llm_client(provider, model)  # Use async client

            # Format messages for the LLM client (Using dictionary format for broader compatibility)
            messages = [
                {"role": "system", "content": formatted_prompt},
                {
                    "role": "user",
                    "content": f"Generate search queries for the following research topic: {combined_topic}",
                },
            ]

            # Import the topic decomposition function schema
            from src.tools.tool_schema import TOPIC_DECOMPOSITION_FUNCTION

            # Bind the decomposition tool
            langchain_tools = [
                {"type": "function", "function": TOPIC_DECOMPOSITION_FUNCTION}
            ]

            # Handle tool choice format based on provider
            if provider in ("google", "gemini"):
                # Use string format for Google Generative AI / Gemini
                tool_choice = "decompose_research_topic"
            else:
                # Azure OpenAI uses standard OpenAI function format
                tool_choice = {
                    "type": "function",
                    "function": {"name": "decompose_research_topic"},
                }

            llm_with_tool = llm.bind_tools(
                tools=langchain_tools, tool_choice=tool_choice
            )

            # Call LLM API with function calling ASYNCHRONOUSLY
            model_str = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
            self.logger.info(
                f"[MasterAgent] Making ASYNC tool call with model {model_str} to decompose topic..."
            )
            response = await llm_with_tool.ainvoke(messages)  # Use await and ainvoke
            self.logger.info(
                f"[MasterAgent] Raw ASYNC LLM response for decomposition: {response}"
            )

            # Process the response (consistent across providers due to Langchain abstraction)
            function_args = None  # Initialize to None
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_call = response.tool_calls[0]
                self.logger.info(f"[MasterAgent] Extracted tool call: {tool_call}")
                try:
                    # Standard Langchain way: access parsed args directly
                    if hasattr(tool_call, "args") and isinstance(tool_call.args, dict):
                        function_args = tool_call.args
                        self.logger.info(
                            f"[MasterAgent] Using pre-parsed args (from tool_call.args): {function_args}"
                        )
                    # Fallback for string arguments (less common)
                    elif (
                        hasattr(tool_call, "function")
                        and hasattr(tool_call.function, "arguments")
                        and isinstance(tool_call.function.arguments, str)
                    ):
                        raw_args = tool_call.function.arguments
                        self.logger.warning(
                            f"[MasterAgent] Received string arguments, attempting JSON parse: {raw_args}"
                        )
                        function_args = json.loads(raw_args)
                    else:
                        # Handle unexpected structures, including the dict case if needed
                        if (
                            isinstance(tool_call, dict)
                            and "args" in tool_call
                            and isinstance(tool_call["args"], dict)
                        ):
                            function_args = tool_call["args"]
                            self.logger.info(
                                f"[MasterAgent] Using args from dict key (standard structure): {function_args}"
                            )
                        else:
                            self.logger.warning(
                                f"[MasterAgent] Tool call structure not recognized or args not found/parsed: {tool_call}"
                            )

                except json.JSONDecodeError as e:
                    self.logger.error(
                        f"[MasterAgent] Failed to parse JSON arguments: {getattr(locals(), 'raw_args', 'N/A')} - Error: {e}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"[MasterAgent] Error processing tool call arguments: {e}"
                    )

            if function_args:
                self.logger.info(f"[MasterAgent] Parsed function_args: {function_args}")

                # Log what LLM returned for duplicate tracking
            else:
                # Fallback if no tool call or parsing failed
                self.logger.warning(
                    "[MasterAgent] No valid tool call arguments found. Falling back to simple search."
                )
                function_args = {
                    "topic_complexity": "simple",
                    "simple_topic": {
                        "query": query,
                        "aspect": "general information",
                        "rationale": "Fallback: No tool call or argument parsing failed",
                        "suggested_tool": "general_search",
                    },
                }

            # Process function call result based on topic complexity
            if function_args.get("topic_complexity") == "simple":
                simple_topic = function_args.get("simple_topic", {})
                query_text = simple_topic.get("query", query)
                suggested_tool = simple_topic.get("suggested_tool", "general_search")

                result = {
                    "topic_complexity": "simple",
                    "query": query_text,
                    "aspect": simple_topic.get("aspect", "general information"),
                    "rationale": simple_topic.get("rationale", ""),
                    "suggested_tool": suggested_tool,
                }
                # Log tool call for trajectory capture (non-invasive, never fails research)
                try:
                    if hasattr(self, "state") and self.state:
                        self.state.log_tool_call(
                            tool_name="decompose_research_topic",
                            params={"query": query, "knowledge_gap": knowledge_gap},
                            result_summary=f"simple topic: {result['query']}",
                        )
                        # Log complete execution step
                        self.state.log_execution_step(
                            step_type="llm_call",
                            action="decompose_query",
                            input_data={"query": query, "knowledge_gap": knowledge_gap},
                            output_data=result,
                            metadata={"provider": provider, "model": model},
                        )
                except Exception:
                    pass  # Logging errors should never break research
                return result
            elif function_args.get("topic_complexity") == "complex":
                complex_topic = function_args.get("complex_topic", {})
                self.logger.info(
                    f"[MasterAgent] Topic decomposed into {len(complex_topic.get('subtopics', []))} subtopics."
                )  # Added log
                result = {
                    "topic_complexity": "complex",
                    "main_query": complex_topic.get("main_query", query),
                    "main_tool": complex_topic.get("main_tool", "general_search"),
                    "subtopics": complex_topic.get("subtopics", []),
                }
                # Log tool call for trajectory capture (non-invasive, never fails research)
                try:
                    if hasattr(self, "state") and self.state:
                        self.state.log_tool_call(
                            tool_name="decompose_research_topic",
                            params={"query": query, "knowledge_gap": knowledge_gap},
                            result_summary=f"complex topic: {len(result['subtopics'])} subtopics",
                        )
                        # Log complete execution step
                        self.state.log_execution_step(
                            step_type="llm_call",
                            action="decompose_query",
                            input_data={"query": query, "knowledge_gap": knowledge_gap},
                            output_data=result,
                            metadata={
                                "provider": provider,
                                "model": model,
                                "num_subtopics": len(result["subtopics"]),
                            },
                        )
                except Exception:
                    pass  # Logging errors should never break research
                return result
            else:
                # Fallback if the function call didn't return expected format
                self.logger.warning(
                    f"[MasterAgent] Function call did not return expected format (complexity: {function_args.get('topic_complexity')}). Falling back to simple topic."
                )
                return {
                    "topic_complexity": "simple",
                    "query": query,
                    "aspect": "general information",
                    "rationale": "Fallback due to unexpected function call format",
                    "suggested_tool": "general_search",
                }

        except Exception as e:
            self.logger.error(f"[MasterAgent] Error in topic decomposition: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                "topic_complexity": "simple",
                "query": query,
                "aspect": "general information",
                "rationale": "Error during execution",
                "suggested_tool": "general_search",
            }

    async def plan_research(
        self,
        query,
        knowledge_gap,
        research_loop_count,
        uploaded_knowledge=None,
        existing_tasks=None,
    ):
        """
        Create a research plan based on the topic decomposition.

        Args:
            query: The main research query or topic
            knowledge_gap: Additional context about knowledge gaps to address
            research_loop_count: Current iteration of research loop
            uploaded_knowledge: User-provided external knowledge (optional)
            existing_tasks: List of existing pending tasks to avoid duplicates (optional)

        Returns:
            Dict containing the research plan with tasks for specialized agents
        """
        # Decompose the topic (pass existing_tasks for LLM awareness)
        topic_info = await self.decompose_topic(
            query,
            knowledge_gap,
            research_loop_count,
            uploaded_knowledge,
            existing_tasks,
        )

        # Log decomposition in benchmark mode
        if hasattr(self, "state") and getattr(self.state, "benchmark_mode", False):
            print(f"[plan_research] Benchmark mode: Question decomposition")
            if "subtopics" in topic_info:
                print(f"[plan_research] Identified subtopics:")
                for i, subtopic in enumerate(topic_info.get("subtopics", [])):
                    print(f"  {i+1}. {subtopic}")
            if "key_entities" in topic_info:
                print(f"[plan_research] Key entities:")
                for entity in topic_info.get("key_entities", []):
                    print(f"  - {entity}")

        # Create a research plan based on the topic complexity
        if topic_info.get("topic_complexity") == "complex":
            # For complex topics, create a plan with subtasks
            subtasks = []

            # Add search tasks based on subtopics
            for i, subtopic in enumerate(topic_info.get("subtopics", [])):
                # Use subtopic name/aspect for concise description instead of full dict
                if isinstance(subtopic, dict):
                    desc = (
                        subtopic.get("name")
                        or subtopic.get("aspect")
                        or subtopic.get("query", "")
                    )
                else:
                    desc = str(subtopic)

                subtasks.append(
                    {
                        "index": i,
                        "type": "search",
                        "query": subtopic,
                        "description": desc,
                        "source_type": topic_info.get(
                            "recommended_sources", ["general"]
                        )[
                            min(
                                i,
                                len(topic_info.get("recommended_sources", ["general"]))
                                - 1,
                            )
                        ],
                    }
                )
            # Skip visualization tasks if in benchmark mode
            if not (
                hasattr(self, "state")
                and (
                    getattr(self.state, "benchmark_mode", False)
                    or getattr(self.state, "visualization_disabled", False)
                )
            ):
                # Add visualization tasks based on recommended visualizations
                for i, viz in enumerate(
                    topic_info.get("recommended_visualizations", [])
                ):
                    search_task_index = viz.get("search_task_index")
                    if search_task_index is not None:
                        viz_task = {
                            "index": len(subtasks),
                            "type": "visualization",
                            "description": viz.get(
                                "description",
                                f"Create visualization based on subtopic {search_task_index}",
                            ),
                            "visualization_type": viz.get("type", "default"),
                            "search_task_index": search_task_index,
                        }
                        subtasks.append(viz_task)

            # Create the plan
            plan = {
                "title": topic_info.get("title", query),
                "description": topic_info.get(
                    "description", "A comprehensive research plan"
                ),
                "subtasks": subtasks,
                "subtopics": topic_info.get("subtopics", []),
            }

        else:
            # For simple topics, create a basic plan with a single search task
            search_query = topic_info.get("query", query)
            suggested_tool = topic_info.get("suggested_tool", "general_search")

            # Map suggested_tool to source_type (for backwards compatibility)
            source_type_map = {
                "text2sql": "text2sql",
                "general_search": "general",
                "academic_search": "academic",
                "github_search": "github",
                "linkedin_search": "linkedin",
            }
            source_type = source_type_map.get(suggested_tool, "general")

            plan = {
                "title": topic_info.get("title", query),
                "description": topic_info.get("description", "A simple research plan"),
                "subtasks": [
                    {
                        "index": 0,
                        "type": "search",
                        "query": search_query,
                        "description": f"Research information about {search_query}",
                        "source_type": source_type,
                    }
                ],
                "subtopics": [search_query],
            }

            # Skip visualization tasks if in benchmark mode or visualization mode is disabled
            if not (
                hasattr(self, "state")
                and (
                    getattr(self.state, "benchmark_mode", False)
                    or getattr(self.state, "visualization_disabled", False)
                )
            ):
                # Add a generic visualization task for simple topics
                if topic_info.get("recommended_visualizations"):
                    viz = topic_info.get("recommended_visualizations")[0]
                    viz_task = {
                        "index": 1,
                        "type": "visualization",
                        "description": viz.get(
                            "description", "Create visualization of the topic"
                        ),
                        "visualization_type": viz.get("type", "default"),
                        "search_task_index": 0,  # Reference the single search task
                    }
                    plan["subtasks"].append(viz_task)

        # Log the plan
        subtask_count = len(plan.get("subtasks", []))
        search_task_count = len(
            [t for t in plan.get("subtasks", []) if t.get("type") == "search"]
        )
        visualization_task_count = len(
            [t for t in plan.get("subtasks", []) if t.get("type") == "visualization"]
        )

        self.logger.info(
            f"[MasterAgent] Created research plan with {subtask_count} subtasks:"
        )
        self.logger.info(f"[MasterAgent] - {search_task_count} search tasks")
        self.logger.info(
            f"[MasterAgent] - {visualization_task_count} visualization tasks"
        )

        if hasattr(self, "state") and getattr(self.state, "benchmark_mode", False):
            print(f"[plan_research] Benchmark mode: Research plan created")
            print(f"[plan_research] Plan contains {search_task_count} search tasks")
            if search_task_count > 0:
                print(f"[plan_research] Search queries:")
                for i, task in enumerate(
                    [t for t in plan.get("subtasks", []) if t.get("type") == "search"]
                ):
                    print(
                        f"  {i+1}. \"{task.get('query')}\" (source: {task.get('source_type', 'general')})"
                    )

        return plan

    async def plan_research_from_tasks(
        self,
        query: str,
        tasks: List,
        knowledge_gap: str,
        research_loop_count: int,
        state=None,
    ):
        """
        Generate research plan that explicitly targets completing specific todo tasks.
        This is the KEY method that makes tasks drive research - inspired by Manus AI's iterative agent loop.

        Like Cursor/Claude Code, this creates explicit task → query mappings so we can:
        1. Generate queries specifically to complete each task
        2. Track which query completes which task
        3. Mark tasks as completed after successful searches
        4. Verify all user requirements are met
        """
        from src.simple_steering import SteeringTask

        self.logger.info(
            f"[TASK_PLANNING] Planning research to complete {len(tasks)} specific tasks"
        )

        # Format tasks for the prompt with clear IDs and priorities
        task_list_lines = []
        for i, task in enumerate(tasks):
            task_list_lines.append(
                f"  {i+1}. [{task.id}] (Priority {task.priority}/10) {task.description}"
            )
        task_list = "\n".join(task_list_lines)

        # Get research context
        research_context = ""
        if state:
            research_context = f"""
Research History:
- Loop: {research_loop_count}
- Previous findings: {state.running_summary[-500:] if state.running_summary else 'Starting fresh'}
- Sources gathered: {len(state.sources_gathered)} sources
- Knowledge gap: {knowledge_gap}
"""

        # Add database context if available
        database_context = ""
        if self.database_info:
            database_list = []
            for db in self.database_info:
                db_info = f"- {db.get('filename', 'Unknown')} ({db.get('file_type', 'unknown').upper()}) with {len(db.get('tables', []))} tables: {', '.join(db.get('tables', []))}"
                database_list.append(db_info)
            database_context = f"""
DATABASE AVAILABLE:
{chr(10).join(database_list)}

TOOL SELECTION:
- Questions about uploaded data → "text2sql"
- Questions about external info → other search tools (general_search, academic_search, etc.)
"""

        # Build prompt following Manus AI's approach: analyze current state, select actions
        planning_prompt = f"""
You are a research planning agent operating in an iterative task completion loop.

CURRENT STATE:
{research_context}

{database_context}

MAIN RESEARCH TOPIC: {query}

TODO LIST (Tasks You MUST Complete):
{task_list}

YOUR JOB:
For EACH task in the todo list above, generate ONE specific search query/code that will complete that task.


CRITICAL RULES:
1. Write queries in NATURAL LANGUAGE (plain English), NEVER SQL code
2. Select appropriate tool for each query
3. If database available and task is about the data → tool="text2sql"
4. If task is about external info → tool="general_search"

Return JSON with THIS EXACT format:
{{
    "reasoning": "Your strategy",
    "queries": [
        {{
            "query": "natural language question here",
            "tool": "text2sql",
            "completes_task_id": "task_xxx",
            "task_description": "brief description",
            "rationale": "rationale",
            "priority": 8
        }}
    ]
}}

REQUIRED: Every query MUST have a "tool" field ("text2sql" or "general_search")!
Generate {len(tasks)} queries, one for each task.
"""

        try:
            # Use LLM to create task-driven plan
            from llm_clients import get_async_llm_client
            import os

            provider = getattr(state, "llm_provider", None) or os.environ.get("LLM_PROVIDER", "azure")
            model = getattr(state, "llm_model", None) or os.environ.get("LLM_MODEL", None)
            self.logger.info(f"[MasterAgent.create_task_driven_plan] Using provider='{provider}', model='{model}'")

            llm = await get_async_llm_client(provider, model)

            messages = [{"role": "user", "content": planning_prompt}]

            response = await llm.ainvoke(messages)

            # Parse response
            import json
            import re

            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())

                self.logger.info(
                    f"[TASK_PLANNING] Strategy: {plan_data.get('reasoning', '')}"
                )

                # Convert to research plan format with task mappings
                subtasks = []
                for query_item in plan_data.get("queries", []):
                    # Get tool from LLM response
                    tool = query_item.get("tool", "general_search")
                    # Map to source_type
                    source_type = "text2sql" if tool == "text2sql" else "general"

                    self.logger.info(
                        f"[TASK_PLANNING] Query tool selection: tool={tool}, source_type={source_type}, query={query_item['query'][:60]}..."
                    )

                    subtasks.append(
                        {
                            "index": len(subtasks),
                            "type": "search",
                            "query": query_item["query"],
                            "description": query_item.get(
                                "task_description", query_item["query"]
                            ),
                            "source_type": source_type,  # Use LLM's tool selection
                            "priority": "high",
                            "completes_task_id": query_item[
                                "completes_task_id"
                            ],  # KEY: Task tracking
                            "task_rationale": query_item.get("rationale", ""),
                        }
                    )

                self.logger.info(
                    f"[TASK_PLANNING] Created {len(subtasks)} task-driven search queries"
                )

                # Log the task → query mappings
                for subtask in subtasks:
                    self.logger.info(
                        f"[TASK_PLANNING]   Query: '{subtask['query'][:60]}...' → Task: {subtask['completes_task_id']}"
                    )

                return {
                    "topic_complexity": "task_driven",
                    "subtasks": subtasks,
                    "task_planning_strategy": plan_data.get("reasoning", ""),
                    "tasks_targeted": [t.id for t in tasks],
                }
            else:
                self.logger.warning("[TASK_PLANNING] Failed to parse LLM response")

        except Exception as e:
            self.logger.error(f"[TASK_PLANNING] Error: {e}")
            import traceback

            traceback.print_exc()

        # Fallback: create simple queries from task descriptions
        self.logger.info(
            "[TASK_PLANNING] Using fallback: extract queries from task descriptions"
        )
        subtasks = []
        for i, task in enumerate(tasks):
            query = state.steering_todo.extract_search_query_from_task(task.description)
            subtasks.append(
                {
                    "index": i,
                    "type": "search",
                    "query": f"{query} {query}",
                    "description": task.description,
                    "source_type": "general",
                    "priority": "high",
                    "completes_task_id": task.id,
                    "task_rationale": "Fallback: extracted from task description",
                }
            )

        return {
            "topic_complexity": "task_driven",
            "subtasks": subtasks,
            "task_planning_strategy": "Fallback to simple extraction",
            "tasks_targeted": [t.id for t in tasks],
        }

    async def plan_adaptive_research(
        self,
        query,
        knowledge_gap,
        research_loop_count,
        uploaded_knowledge=None,
        steering_guidance="",
        todo_plan="",
        state=None,
    ):
        """
        Create an adaptive research plan that considers steering guidance and todo.md.

        This method implements Cursor/Claude-style adaptive planning where the agent
        dynamically adjusts its approach based on user feedback and current context.

        Args:
            query: The main research query or topic
            knowledge_gap: Additional context about knowledge gaps to address
            research_loop_count: Current iteration of research loop
            uploaded_knowledge: User-provided external knowledge (optional)
            steering_guidance: Current loop guidance from steering system
            todo_plan: Current todo.md plan content
            state: Research state for context

        Returns:
            Dict containing the adaptive research plan with tasks for specialized agents
        """
        self.logger.info(
            f"[ADAPTIVE_PLANNING] Starting adaptive planning for loop {research_loop_count}"
        )

        # Get research history context
        research_context = ""
        if state:
            research_context = f"""
Research History:
- Loop: {research_loop_count}
- Previous summary: {state.running_summary[-1000:] if state.running_summary else 'None'}
- Sources gathered: {len(state.sources_gathered)} sources
- Knowledge gap: {knowledge_gap}
"""

        # Create adaptive planning prompt
        adaptive_prompt = f"""
You are an adaptive research planning agent. Your job is to create the next set of research actions based on:

1. CURRENT RESEARCH CONTEXT:
{research_context}

2. USER STEERING GUIDANCE:
{steering_guidance}

3. CURRENT TODO PLAN:
{todo_plan}

4. ORIGINAL QUERY: {query}

TASK: Based on the above context, determine what specific research actions should be taken next. 
Consider:
- What the user has specifically requested via steering messages
- What tasks in the todo.md are highest priority
- What gaps exist in the current research
- What searches or tools would be most valuable right now

You should adapt your approach like Cursor or Claude - if the user says "focus on recent work", 
prioritize recent publications. If they say "exclude entertainment", avoid entertainment-related searches.

Generate a focused research plan with 2-4 specific, targeted search queries that directly address 
the user's steering guidance and todo priorities.

Respond with a JSON object containing:
{{
    "reasoning": "Brief explanation of your adaptive strategy",
    "priority_focus": "What you're prioritizing based on steering",
    "search_queries": ["specific search query 1", "specific search query 2", ...],
    "adaptations_made": ["adaptation 1", "adaptation 2", ...]
}}
"""

        try:
            # Use LLM to create adaptive plan
            from llm_clients import get_async_llm_client
            import os

            adaptive_provider = (getattr(state, "llm_provider", None) or os.environ.get("LLM_PROVIDER", "azure")) if state else os.environ.get("LLM_PROVIDER", "azure")
            adaptive_model = (getattr(state, "llm_model", None) or os.environ.get("LLM_MODEL", None)) if state else os.environ.get("LLM_MODEL", None)
            self.logger.info(f"[MasterAgent.create_adaptive_plan] Using provider='{adaptive_provider}', model='{adaptive_model}'")
            llm_client = await get_async_llm_client(adaptive_provider, adaptive_model)

            response = await llm_client.ainvoke(
                [{"role": "user", "content": adaptive_prompt}]
            )

            # Parse the response
            import json
            import re

            # Extract JSON from response (ainvoke returns AIMessage, need .content)
            response_text = response.content if hasattr(response, "content") else str(response)
            self.logger.info(f"[MasterAgent.create_adaptive_plan] LLM response type={type(response).__name__}, length={len(response_text)}")
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                adaptive_plan = json.loads(json_match.group())

                self.logger.info(
                    f"[ADAPTIVE_PLANNING] Reasoning: {adaptive_plan.get('reasoning', '')}"
                )
                self.logger.info(
                    f"[ADAPTIVE_PLANNING] Priority Focus: {adaptive_plan.get('priority_focus', '')}"
                )
                self.logger.info(
                    f"[ADAPTIVE_PLANNING] Adaptations: {adaptive_plan.get('adaptations_made', [])}"
                )

                # Convert to research plan format
                subtasks = []
                for i, search_query in enumerate(
                    adaptive_plan.get("search_queries", [])
                ):
                    subtasks.append(
                        {
                            "index": i,
                            "type": "search",
                            "query": search_query,
                            "description": f"Adaptive search: {search_query}",
                            "source_type": "general",
                            "priority": "high",
                            "steering_adapted": True,
                            "adaptation_reason": adaptive_plan.get("reasoning", ""),
                        }
                    )

                return {
                    "topic_complexity": "adaptive",
                    "subtasks": subtasks,
                    "adaptive_reasoning": adaptive_plan.get("reasoning", ""),
                    "priority_focus": adaptive_plan.get("priority_focus", ""),
                    "adaptations_made": adaptive_plan.get("adaptations_made", []),
                    "steering_guidance_used": steering_guidance,
                    "todo_plan_considered": bool(todo_plan),
                }

            else:
                self.logger.warning(
                    "[ADAPTIVE_PLANNING] Failed to parse LLM response, falling back to original planning"
                )

        except Exception as e:
            self.logger.error(f"[ADAPTIVE_PLANNING] Error in adaptive planning: {e}")

        # Fallback to original planning if adaptive planning fails
        self.logger.info("[ADAPTIVE_PLANNING] Falling back to original planning method")
        return await self.plan_research(
            query, knowledge_gap, research_loop_count, uploaded_knowledge
        )

    async def execute_research(self, state, callbacks=None, database_info=None):
        """
        Execute research tasks based on the research plan.
        """
        # Store state for access by all methods in this execution
        self.state = state

        # Store database_info for access by all methods in this execution
        self.database_info = database_info
        self.logger.info(
            f"[MasterAgent.execute_research] Received database_info parameter: {database_info}"
        )

        # Step 1: Create initial research plan as todo items if steering is enabled
        if hasattr(state, "steering_todo") and state.steering_todo:
            await self._create_initial_research_plan(state)
            self.logger.info(f"[STEERING] Created initial research plan")

        try:
            # Initialize visualization tracking variables
            viz_tasks_created_this_loop = 0
            total_viz_in_state = (
                len(state.visualizations)
                if hasattr(state, "visualizations") and state.visualizations
                else 0
            )
            # 5 or 1
            max_viz_for_this_loop = self._get_max_viz_for_loop(
                state.research_loop_count
            )

            # Always get query, knowledge_gap, and loop_count from state for planning
            # Use refined search_query if available, otherwise the original research_topic
            query = (
                state.search_query
                if hasattr(state, "search_query") and state.search_query
                else state.research_topic
            )
            knowledge_gap = getattr(state, "knowledge_gap", "")
            research_loop_count = getattr(state, "research_loop_count", 0)
            uploaded_knowledge = getattr(state, "uploaded_knowledge", None)

            # Log all tasks at start of research loop
            if hasattr(state, "steering_todo") and state.steering_todo:
                print("\n" + "=" * 80)
                print(f"[RESEARCH LOOP {research_loop_count}] TASK STATUS SNAPSHOT")
                print("=" * 80)

                from src.simple_steering import TaskStatus

                pending = [
                    t
                    for t in state.steering_todo.tasks.values()
                    if t.status == TaskStatus.PENDING
                ]
                in_progress = [
                    t
                    for t in state.steering_todo.tasks.values()
                    if t.status == TaskStatus.IN_PROGRESS
                ]
                completed = [
                    t
                    for t in state.steering_todo.tasks.values()
                    if t.status == TaskStatus.COMPLETED
                ]
                cancelled = [
                    t
                    for t in state.steering_todo.tasks.values()
                    if t.status == TaskStatus.CANCELLED
                ]

                print(
                    f"\nSUMMARY: {len(pending)} pending | {len(in_progress)} in-progress | {len(completed)} completed | {len(cancelled)} cancelled"
                )

                if pending:
                    print(f"\nPENDING TASKS ({len(pending)}):")
                    for task in sorted(pending, key=lambda t: t.priority, reverse=True):
                        print(
                            f"  [ ] [{task.id}] P{task.priority} - {task.description[:80]}"
                        )

                if in_progress:
                    print(f"\nIN PROGRESS ({len(in_progress)}):")
                    for task in in_progress:
                        print(f"  [{task.id}] - {task.description[:80]}")

                if completed:
                    print(f"\nCOMPLETED ({len(completed)}):")
                    for task in completed:  # Show ALL
                        print(f"  [done] [{task.id}] - {task.description[:80]}")

                if cancelled:
                    print(f"\nCANCELLED ({len(cancelled)}):")
                    for task in cancelled:  # Show ALL
                        print(f"  [cancelled] [{task.id}] - {task.description[:80]}")

                print("=" * 80 + "\n")

            self.logger.info(
                f"[MasterAgent.execute_research] Planning with query: '{query[:100]}...'"
            )

            # Log uploaded knowledge availability
            if uploaded_knowledge and uploaded_knowledge.strip():
                self.logger.info(
                    f"[MasterAgent.execute_research] Uploaded knowledge available: {len(uploaded_knowledge)} characters"
                )
            else:
                self.logger.info(
                    "[MasterAgent.execute_research] No uploaded knowledge available"
                )

            # Log database information availability
            if self.database_info and len(self.database_info) > 0:
                self.logger.info(
                    f"[MasterAgent.execute_research] Database info available: {len(self.database_info)} database(s)"
                )
                for db in self.database_info:
                    self.logger.info(
                        f"[MasterAgent.execute_research] Database: {db.get('filename', 'Unknown')} with {len(db.get('tables', []))} tables"
                    )
            else:
                self.logger.info(
                    "[MasterAgent.execute_research] No database info available"
                )

            research_plan = None
            if (
                hasattr(state, "research_plan")
                and state.research_plan
                and research_loop_count == 0
            ):
                # Only use existing plan if it's the very first loop AND a plan was somehow pre-loaded
                # Otherwise, always re-plan to use the latest query/knowledge_gap
                research_plan = state.research_plan
                print(
                    "[execute_research] Using pre-existing research plan for initial loop"
                )

            # If no pre-existing plan for loop 0, or if it's a subsequent loop, always (re)plan.
            if not research_plan or research_loop_count > 0:
                if research_loop_count > 0:
                    print(
                        f"[execute_research] Re-planning research for loop {research_loop_count}."
                    )
                else:
                    print("[execute_research] Creating initial research plan.")

                # Step 2: Process any pending steering messages and update todo.md
                # This happens at the beginning of each research loop to incorporate user guidance
                if hasattr(state, "steering_todo") and state.steering_todo:
                    steering_result = await state.prepare_steering_for_next_loop()
                    if steering_result.get("steering_enabled"):
                        self.logger.info(
                            f"[STEERING] Prepared todo.md for loop {state.research_loop_count}:"
                        )
                        self.logger.info(
                            f"  - Todo version: {steering_result.get('todo_version')}"
                        )
                        self.logger.info(
                            f"  - Pending tasks: {steering_result.get('pending_tasks')}"
                        )
                        self.logger.info(
                            f"  - Loop guidance: {steering_result.get('loop_guidance', 'None')}"
                        )
                        print(
                            f"  - Completed tasks: {steering_result.get('completed_tasks')}"
                        )
                        print(
                            f"  - Queue processed: {steering_result.get('todo_updated')}"
                        )

                        # Show current loop guidance
                        loop_guidance = steering_result.get("loop_guidance", "")
                        if loop_guidance:
                            print(
                                f"\n[STEERING] Current Loop Guidance:\n{loop_guidance}"
                            )

                        # Show snippet of current plan
                        current_plan = steering_result.get("current_plan", "")
                        if current_plan:
                            plan_lines = current_plan.split("\n")
                            print(f"\n[STEERING] Updated todo.md (first 10 lines):")
                            for line in plan_lines[:10]:
                                print(f"  {line}")
                            if len(plan_lines) > 10:
                                print(f"  ... ({len(plan_lines) - 10} more lines)")
                    else:
                        # Legacy behavior for existing code
                        await self._process_pending_steering_messages(state)
                        current_plan = state.get_steering_plan()
                        print(
                            f"[STEERING] Current todo.md plan:\n{current_plan[:500]}..."
                        )

                # Step 3: Create research plan - TASK-DRIVEN if we have pending tasks
                if hasattr(state, "steering_todo") and state.steering_todo:
                    # Get pending tasks
                    pending_tasks = state.steering_todo.get_pending_tasks()

                    if pending_tasks:
                        # Sort by priority (highest first)
                        pending_tasks.sort(key=lambda t: t.priority, reverse=True)

                        # Take top 3-4 tasks for this loop (like Manus AI's iterative approach)
                        top_tasks = pending_tasks[: min(4, len(pending_tasks))]

                        self.logger.info(
                            f"[STEERING] Found {len(pending_tasks)} pending tasks, targeting top {len(top_tasks)} this loop"
                        )

                        # Mark tasks as in-progress
                        for task in top_tasks:
                            state.steering_todo.mark_task_in_progress(task.id)

                        # Use TASK-DRIVEN planning (explicit task → query mapping)
                        research_plan = await self.plan_research_from_tasks(
                            query=query,
                            tasks=top_tasks,
                            knowledge_gap=knowledge_gap,
                            research_loop_count=research_loop_count,
                            state=state,
                        )

                        self.logger.info(
                            f"[TASK_PLANNING] Created task-driven plan with {len(research_plan.get('subtasks', []))} queries"
                        )
                    else:
                        # No pending tasks - use adaptive planning
                        self.logger.info(
                            "[STEERING] No pending tasks, using adaptive planning"
                        )
                        loop_guidance = state.steering_todo.get_current_loop_guidance()
                        todo_md = state.steering_todo.get_todo_md()

                        research_plan = await self.plan_adaptive_research(
                            query,
                            knowledge_gap,
                            research_loop_count,
                            uploaded_knowledge,
                            steering_guidance=loop_guidance,
                            todo_plan=todo_md,
                            state=state,
                        )
                else:
                    # No steering - fallback to original planning
                    research_plan = await self.plan_research(
                        query, knowledge_gap, research_loop_count, uploaded_knowledge
                    )

                # Add/update the research plan in state if possible
                try:
                    state.research_plan = research_plan
                    print("[execute_research] Updated research plan in state")
                except (ValueError, AttributeError):
                    print(
                        "[execute_research] Unable to update research plan in state, using local copy"
                    )

            # Execute search tasks
            search_results_list = await self._execute_search_tasks(
                research_plan, state
            )  # Renamed to avoid conflict

            if hasattr(state, "steering_todo") and state.steering_todo:
                plan_complexity = research_plan.get("topic_complexity", "")

                if plan_complexity == "task_driven":
                    # For task-driven plans, we have explicit task → query mappings
                    self.logger.info(
                        "[TASK_COMPLETION] Checking task completion based on search results"
                    )

                    for subtask in research_plan.get("subtasks", []):
                        task_id = subtask.get("completes_task_id")
                        if not task_id:
                            continue

                        # Find corresponding search result
                        subtask_index = subtask.get("index", -1)
                        if subtask_index < len(search_results_list):
                            search_result = search_results_list[subtask_index]

                            # Check if search was successful and has content
                            if search_result.get("success", False):
                                has_content = bool(
                                    search_result.get("content")
                                    or search_result.get("sources", [])
                                )

                                if has_content:
                                    # Mark task as COMPLETED
                                    state.steering_todo.mark_task_completed(
                                        task_id=task_id,
                                        completion_note=f"[done] Found via search: '{subtask['query'][:50]}...'",
                                    )
                                    self.logger.info(
                                        f"[TASK_COMPLETION] [done] Task {task_id} completed successfully"
                                    )
                                else:
                                    # Search succeeded but found nothing - cancel task or leave pending
                                    self.logger.warning(
                                        f"[TASK_COMPLETION] [WARN] Search for task {task_id} returned no results"
                                    )
                            else:
                                # Search failed - leave task in progress, will retry
                                error_msg = search_result.get("error", "Unknown error")
                                self.logger.warning(
                                    f"[TASK_COMPLETION] [FAIL] Search for task {task_id} failed: {error_msg}"
                                )

                elif plan_complexity == "adaptive":
                    # For adaptive plans, use existing heuristic-based completion
                    await self._update_todo_based_on_results(
                        research_plan, search_results_list, state
                    )

            successful_search_indices = [
                i
                for i, result in enumerate(search_results_list)
                if result.get("success", False)
            ]

            self.logger.info(
                f"[MasterAgent] Completed {len(successful_search_indices)} successful search tasks out of {len(search_results_list)}."
            )

            # Initialize lists to store visualization outputs from this loop
            visualizations_generated_this_loop = []
            base64_images_generated_this_loop = []
            code_snippets_generated_this_loop = []

            # Skip visualization tasks in QA mode or benchmark mode
            if state.qa_mode or state.benchmark_mode:
                mode_name = "QA mode" if state.qa_mode else "benchmark mode"
                self.logger.info(
                    f"[MasterAgent] Skipping visualization tasks because {mode_name} is enabled."
                )
            elif state.visualization_disabled:
                self.logger.info(
                    "[MasterAgent] Skipping visualization tasks because visualization mode is disabled."
                )
            # do visualization tasks
            else:
                self.logger.info(
                    f"[MasterAgent] Attempting to generate visualizations. Max for this loop: {max_viz_for_this_loop}"
                )
                visualization_agent = VisualizationAgent(self.config)

                for i, search_result_item in enumerate(search_results_list):
                    if not search_result_item.get("success", False):
                        continue  # Skip failed search tasks

                    if viz_tasks_created_this_loop >= max_viz_for_this_loop:
                        self.logger.info(
                            f"[MasterAgent] Reached max visualizations ({max_viz_for_this_loop}) for this loop."
                        )
                        break

                    self.logger.info(
                        f"[MasterAgent] Considering search result index {search_result_item.get('index')} for visualization."
                    )

                    # Pass the actual search result content to determine_visualization_needs
                    # search_result_item is already a dict like {'index': ..., 'query': ..., 'success': ..., 'content': ..., 'sources': ..., 'error': ...}
                    vis_needs = await visualization_agent.determine_visualization_needs(
                        search_result_item
                    )

                    if vis_needs and vis_needs.get("visualization_needed"):
                        self.logger.info(
                            f"[MasterAgent] Visualization needed for search result index {search_result_item.get('index')}. Rationale: {vis_needs.get('rationale')}"
                        )
                        code_data = (
                            await visualization_agent.generate_visualization_code(
                                search_result_item, vis_needs
                            )
                        )

                        if code_data and code_data.get("code"):
                            execution_output = (
                                await visualization_agent.execute_visualization_code(
                                    code_data
                                )
                            )

                            if execution_output and not execution_output.get("error"):
                                viz_tasks_created_this_loop += 1
                                # execution_output["results"] is a list of dicts like:
                                # [{"type": "image", "filepath": "path/to/img.png", "filename": "img.png", "description": "desc", "format": "png", "data": "base64data", "src": "datauri"}]
                                generated_visualizations = execution_output.get(
                                    "results", []
                                )
                                visualizations_generated_this_loop.extend(
                                    generated_visualizations
                                )

                                # Extract base64 data for state
                                for viz_item_detail in generated_visualizations:
                                    if viz_item_detail.get(
                                        "data"
                                    ) and viz_item_detail.get("filename"):
                                        base64_images_generated_this_loop.append(
                                            {
                                                "filename": viz_item_detail.get(
                                                    "filename"
                                                ),
                                                "title": viz_item_detail.get(
                                                    "description"
                                                )
                                                or viz_item_detail.get(
                                                    "filename"
                                                ),  # Use description or filename as title
                                                "base64_data": viz_item_detail.get(
                                                    "data"
                                                ),
                                                "format": viz_item_detail.get(
                                                    "format", "png"
                                                ),
                                            }
                                        )
                                # Collect code snippets
                                if execution_output.get("code_snippets"):
                                    code_snippets_generated_this_loop.extend(
                                        execution_output.get("code_snippets", [])
                                    )
                                elif code_data.get(
                                    "code_snippets"
                                ):  # Fallback to code_data if not in execution_output
                                    code_snippets_generated_this_loop.extend(
                                        code_data.get("code_snippets", [])
                                    )

                                self.logger.info(
                                    f"[MasterAgent] Successfully generated {len(generated_visualizations)} visualizations for search result index {search_result_item.get('index')}."
                                )
                            elif execution_output and execution_output.get("error"):
                                self.logger.error(
                                    f"[MasterAgent] Error executing visualization code for search result index {search_result_item.get('index')}: {execution_output.get('error')}"
                                )
                                # Collect code snippets even on error
                                if execution_output.get("code_snippets"):
                                    code_snippets_generated_this_loop.extend(
                                        execution_output.get("code_snippets", [])
                                    )
                                elif code_data and code_data.get("code_snippets"):
                                    code_snippets_generated_this_loop.extend(
                                        code_data.get("code_snippets", [])
                                    )
                        else:
                            self.logger.info(
                                f"[MasterAgent] No visualization code generated for search result index {search_result_item.get('index')}."
                            )
                    else:
                        self.logger.info(
                            f"[MasterAgent] No visualization deemed necessary for search result index {search_result_item.get('index')}. Rationale: {vis_needs.get('rationale') if vis_needs else 'No vis_needs determined'}"
                        )

            self.logger.info(
                f"[MasterAgent] Generated {viz_tasks_created_this_loop} visualization(s) in this loop. Total viz in state (before this loop): {total_viz_in_state}"
            )

            # Prepare the return dictionary
            # The primary output is still the search_results_list
            # We add new keys for the visualization outputs from this loop.
            return_value = {
                "web_research_results": search_results_list,  # This is what subsequent nodes expect
                "visualizations_generated_this_loop": visualizations_generated_this_loop,
                "base64_images_generated_this_loop": base64_images_generated_this_loop,
                "code_snippets_generated_this_loop": code_snippets_generated_this_loop,
                # Ensure other state fields are preserved if they were part of the input `state`
                # This is crucial if `execute_research` is expected to return a full state-like dict
                # For now, focusing on returning the direct outputs of this agent's actions.
            }
            # Preserve essential state fields that might have been updated by planning
            if hasattr(state, "research_plan"):
                return_value["research_plan"] = state.research_plan

            return return_value

        except Exception as e:
            self.logger.error(f"[MasterAgent] Error in research execution: {str(e)}")
            self.logger.error(f"[MasterAgent] {traceback.format_exc()}")
            # Return search_results_list even in case of error to allow flow to continue if possible
            # And add error information.
            return {
                "web_research_results": getattr(
                    self, "search_results_list", []
                ),  # search_results_list might not be defined if error is early
                "error": f"MasterAgent execution failed: {str(e)}",
                "visualizations_generated_this_loop": [],
                "base64_images_generated_this_loop": [],
                "code_snippets_generated_this_loop": [],
            }

    def _get_max_viz_for_loop(self, research_loop_count):
        """
        Determine the maximum number of visualizations allowed for a given research loop.

        Args:
            research_loop_count: Current research loop iteration

        Returns:
            int: Maximum number of visualizations allowed for this loop
        """
        if research_loop_count == 0:
            # Allow more visualizations in the initial loop
            return 5  # Up to 5 in first loop
        else:
            # Allow fewer visualizations in subsequent loops
            return 1  # Only 1 in later loops

    async def _execute_search_tasks(self, research_plan, state):
        """
        Execute search tasks from the research plan.

        Args:
            research_plan: Dictionary containing the research plan
            state: Current state object

        Returns:
            list: Results from search tasks
        """
        # Start timer for performance logging
        search_start_time = time.time()

        # State is accessible via self.state set in execute_research

        # Track if we're in benchmark mode
        benchmark_mode = getattr(state, "benchmark_mode", False)
        if benchmark_mode:
            print(f"[_execute_search_tasks] Executing search tasks in benchmark mode")

        visualization_disabled = getattr(state, "visualization_disabled", False)
        if visualization_disabled:
            print(
                f"[_execute_search_tasks] Executing search tasks in visualization disabled mode"
            )

        # Initialize specialized agents and tools
        search_agent = SearchAgent(self.config, database_info=self.database_info)

        # Initialize result mapping to track completion by subtask index
        task_results_map = {}

        # If there are no research tasks, exit early
        if "subtasks" not in research_plan or not research_plan["subtasks"]:
            print("No search tasks found in research plan.")
            return []

        def _build_skipped_result(
            task_index: int,
            query_text: str,
            tool_name: str,
            reason: str,
        ) -> Dict[str, Any]:
            return {
                "index": task_index,
                "query": query_text,
                "success": False,
                "content": "",
                "sources": [],
                "error": reason,
                "tool_used": tool_name,
                "skipped": True,
            }

        async def _execute_single_task(
            task_index: int, query_text: str, tool_name: str
        ) -> Dict[str, Any]:
            # Log this to help with tracing
            print(f"Executing search task {task_index} with query: '{query_text}'")
            print(f"Using search tool: {tool_name}")

            try:
                # Execute the search based on the tool_name
                search_result = None
                if tool_name == "general_search":
                    search_result = await search_agent.general_search(query_text)
                elif tool_name == "academic_search":
                    search_result = await search_agent.academic_search(query_text)
                elif tool_name == "github_search":
                    search_result = await search_agent.github_search(query_text)
                elif tool_name == "linkedin_search":
                    search_result = await search_agent.linkedin_search(query_text)
                elif tool_name == "text2sql":
                    # Handle text2sql tool execution
                    search_result = await search_agent.text2sql_search(query_text)
                else:
                    # Default to general search if tool is unknown
                    search_result = await search_agent.general_search(query_text)

                # Log search tool call for trajectory capture (non-invasive, never fails research)
                try:
                    if hasattr(self, "state") and self.state:
                        num_sources = 0
                        sources_list = []
                        if isinstance(search_result, dict):
                            if "formatted_sources" in search_result:
                                sources_list = search_result.get(
                                    "formatted_sources", []
                                )
                                num_sources = len(sources_list)
                            elif "sources" in search_result:
                                sources_list = search_result.get("sources", [])
                                num_sources = len(sources_list)

                        self.state.log_tool_call(
                            tool_name=tool_name,
                            params={"query": query_text},
                            result_summary=f"{num_sources} sources",
                        )

                        # Log complete execution step
                        self.state.log_execution_step(
                            step_type="tool_execution",
                            action=tool_name,
                            input_data={"query": query_text},
                            output_data={
                                "num_sources": num_sources,
                                "sources": (
                                    sources_list[:10]
                                    if len(sources_list) > 10
                                    else sources_list
                                ),  # First 10 sources
                            },
                            metadata={"total_sources": num_sources},
                        )
                except Exception:
                    pass  # Logging errors should never break research

                # Extract sources from search_result for easy access in results
                sources = []

                # Handle different search_result formats
                if isinstance(search_result, dict):
                    # Format from newer search tools that return a dict with 'formatted_sources'
                    if "formatted_sources" in search_result:
                        formatted_sources = search_result.get(
                            "formatted_sources", []
                        )
                        # Extract source details from formatted_sources
                        for src in formatted_sources:
                            if isinstance(src, str) and " : " in src:
                                # Parse title and URL from source string (format: "title : url")
                                parts = src.split(" : ", 1)
                                if len(parts) == 2:
                                    title, url = parts
                                    sources.append({"title": title, "url": url})

                    # If search_result directly contains 'sources' key with structured data
                    if "sources" in search_result and isinstance(
                        search_result["sources"], list
                    ):
                        for src in search_result["sources"]:
                            if (
                                isinstance(src, dict)
                                and "title" in src
                                and "url" in src
                            ):
                                sources.append(
                                    {"title": src["title"], "url": src["url"]}
                                )

                    # Extract content from search_result
                    if "content" in search_result:
                        content = search_result["content"]
                    elif "raw_contents" in search_result:
                        # Join multiple raw contents into a single string
                        raw_contents = search_result.get("raw_contents", [])
                        if isinstance(raw_contents, list):
                            content = "\n\n".join(
                                [str(item) for item in raw_contents if item]
                            )
                        else:
                            content = str(raw_contents)
                    else:
                        # Fallback: convert the entire result to a string
                        content = str(search_result)
                else:
                    # Fallback for unexpected search_result type
                    content = str(search_result)
                    # No sources can be extracted in this case

                # Build the result object with both content and sources
                task_result = {
                    "index": task_index,
                    "query": query_text,  # Store the actual query text, not the original task_query
                    "success": True,
                    "content": content,
                    "sources": sources,
                    "error": None,
                    "tool_used": tool_name,  # Include the tool used for tracking
                }

                print(
                    f"Search task {task_index} completed with {len(sources)} sources"
                )

                # Mark query as executed to prevent future duplicates
                if (
                    hasattr(self.state, "steering_todo")
                    and self.state.steering_todo
                ):
                    self.state.steering_todo.mark_query_executed(query_text)
                    self.logger.info(
                        f"[DEDUP] Marked query as executed: '{query_text[:60]}...'"
                    )

                return task_result

            except Exception as e:
                # Log the error and continue with other tasks
                print(f"Error in search task {task_index}: {str(e)}")
                return {
                    "index": task_index,
                    "query": query_text,  # Store the actual query text, not the original task_query
                    "success": False,
                    "content": "",
                    "sources": [],
                    "error": str(e),
                    "tool_used": tool_name,  # Include the tool used for tracking even on error
                }

        async def _execute_with_semaphore(
            semaphore: asyncio.Semaphore,
            task_index: int,
            query_text: str,
            tool_name: str,
        ) -> Dict[str, Any]:
            async with semaphore:
                return await _execute_single_task(
                    task_index, query_text, tool_name
                )

        parallel_enabled = getattr(state, "parallel_search_enabled", False)
        max_concurrency = getattr(state, "parallel_search_max_concurrency", 4)
        if isinstance(getattr(state, "config", None), dict):
            parallel_enabled = state.config.get(
                "parallel_search_enabled", parallel_enabled
            )
            max_concurrency = state.config.get(
                "parallel_search_max_concurrency", max_concurrency
            )
        if not isinstance(max_concurrency, int) or max_concurrency < 1:
            max_concurrency = 1

        parallel_tasks = []
        sequential_tasks = []

        # Loop through each search task and build execution lists
        for task in research_plan["subtasks"]:
            if task.get("type") != "search":
                continue

            task_index = task.get("index", 0)
            task_query = task.get("query", {})

            # Handle both string and dict query formats
            if isinstance(task_query, str):
                # If query is a string, use it directly
                query_text = task_query
                tool_name = task.get("source_type", "general_search")
                # Map source_type to tool names
                if tool_name == "general":
                    tool_name = "general_search"
                elif tool_name == "academic":
                    tool_name = "academic_search"
                elif tool_name == "github":
                    tool_name = "github_search"
                elif tool_name == "linkedin":
                    tool_name = "linkedin_search"
                elif tool_name == "text2sql":
                    tool_name = "text2sql"
                else:
                    tool_name = "general_search"  # Default fallback

                # WORKAROUND: If database_info is available and query contains SQL, use text2sql
                if (
                    hasattr(self, "database_info")
                    and self.database_info
                    and (
                        "SELECT" in query_text.upper()
                        or "FROM" in query_text.upper()
                        or "JOIN" in query_text.upper()
                    )
                ):
                    self.logger.info(
                        f"[MasterAgent._execute_search_tasks] Detected SQL query, switching to text2sql tool"
                    )
                    tool_name = "text2sql"
            elif isinstance(task_query, dict):
                # If query is a dict, extract the query text and tool
                query_text = task_query.get("query", "")
                tool_name = task_query.get("suggested_tool", "general_search")
            else:
                # Fallback for unexpected format
                query_text = str(task_query)
                tool_name = "general_search"

            # Step 3: Apply steering constraints to filter/modify queries
            if hasattr(self.state, "steering_todo") and self.state.steering_todo:
                # Check if this query should be cancelled due to steering
                if self.state.steering_todo.should_cancel_search(query_text):
                    print(
                        f"[STEERING] Cancelled search task {task_index}: '{query_text}' (filtered by constraints)"
                    )
                    task_results_map[task_index] = _build_skipped_result(
                        task_index,
                        query_text,
                        tool_name,
                        "cancelled_by_steering",
                    )
                    continue

                # Check for duplicate queries (avoid redundant searches)
                if self.state.steering_todo.is_query_duplicate(query_text):
                    self.logger.info(
                        f"[DEDUP] Skipping duplicate query: '{query_text[:60]}...'"
                    )
                    print(
                        f"[STEERING] Skipped duplicate search task {task_index}: '{query_text}' (already executed)"
                    )
                    task_results_map[task_index] = _build_skipped_result(
                        task_index,
                        query_text,
                        tool_name,
                        "duplicate_query",
                    )
                    continue

                # Apply priority boost if relevant
                priority_boost = self.state.steering_todo.get_search_priority_boost(
                    query_text
                )
                if priority_boost > 0:
                    print(
                        f"[STEERING] Boosted priority for search task {task_index}: '{query_text}' (+{priority_boost})"
                    )

            if not parallel_enabled or tool_name == "text2sql":
                sequential_tasks.append((task_index, query_text, tool_name))
            else:
                parallel_tasks.append((task_index, query_text, tool_name))

        # Execute text2sql tasks sequentially (no parallelization)
        for task_index, query_text, tool_name in sequential_tasks:
            task_results_map[task_index] = await _execute_single_task(
                task_index, query_text, tool_name
            )

        # Execute remaining search tasks in parallel with a concurrency cap
        if parallel_tasks:
            semaphore = asyncio.Semaphore(max_concurrency)
            parallel_results = await asyncio.gather(
                *[
                    _execute_with_semaphore(
                        semaphore, task_index, query_text, tool_name
                    )
                    for task_index, query_text, tool_name in parallel_tasks
                ]
            )
            for result in parallel_results:
                task_results_map[result["index"]] = result

        # Rebuild ordered results list to preserve index-based assumptions downstream
        if task_results_map:
            max_index = max(task_results_map.keys())
            ordered_results = []
            for i in range(max_index + 1):
                result = task_results_map.get(i)
                if result is None:
                    ordered_results.append(
                        _build_skipped_result(
                            i,
                            "",
                            "unknown",
                            "no_search_result",
                        )
                    )
                else:
                    ordered_results.append(result)
        else:
            ordered_results = []

        # Log performance
        search_end_time = time.time()
        search_duration = search_end_time - search_start_time
        print(f"All search tasks completed in {search_duration:.2f} seconds")
        print(f"Total search tasks completed: {len(ordered_results)}")

        return ordered_results

    async def _create_initial_research_plan(self, state):
        """Create initial research plan as todo items"""
        try:
            # Get existing pending tasks for duplicate prevention (Loop 1+)
            existing_pending = []
            if state.research_loop_count > 0:
                existing_pending = state.steering_todo.get_pending_tasks()
                if existing_pending:
                    self.logger.info(
                        f"[STEERING] Loop {state.research_loop_count}: "
                        f"Found {len(existing_pending)} existing pending tasks, "
                        f"will avoid creating duplicates"
                    )

            # Generate initial research plan based on the query
            # Pass existing tasks to LLM for awareness (only in Loop 1+)
            research_plan = await self.plan_research(
                state.search_query,
                state.knowledge_gap,
                state.research_loop_count,
                getattr(state, "uploaded_knowledge", None),
                existing_tasks=existing_pending if existing_pending else None,
            )

            # Determine source based on research loop count
            # First loop (0) = original query, subsequent loops = knowledge gaps identified by system
            is_first_loop = state.research_loop_count == 0
            task_source = "original_query" if is_first_loop else "knowledge_gap"

            # Convert research plan to todo items
            if "subtasks" in research_plan:

                for i, subtask in enumerate(research_plan["subtasks"], 1):
                    if subtask.get("type") == "search":
                        query = subtask.get("query", "")
                        if isinstance(query, dict):
                            query = query.get("query", str(query))

                        # Create todo task for this research item
                        task_description = f"Research: {query}"

                        # Use LLM-suggested priority if available, otherwise calculate
                        priority = subtask.get("priority")
                        if priority is None:
                            # Fallback: Earlier tasks get higher priority
                            priority = 5 + (len(research_plan["subtasks"]) - i)

                        state.steering_todo.create_task(
                            task_description,
                            priority=priority,
                            search_queries=[str(query)],
                            created_from_message="Auto-generated research subtask",
                            source=task_source,  # original_query for first loop, knowledge_gap for follow-ups
                        )

            # Create overall research goal task (only if we have a meaningful query)
            research_query = (
                state.search_query or state.research_topic or "research objectives"
            )

            # Skip creating this task if the query is "none" or empty
            if research_query and research_query.lower().strip() not in [
                "none",
                "",
                "research objectives",
            ]:
                high_level_description = f"Complete research on: {research_query}"
                state.steering_todo.create_task(
                    high_level_description,
                    priority=10,  # Highest priority
                    search_queries=[research_query],
                    created_from_message="Primary research objective",
                    source="original_query",
                )

            self.logger.info(
                f"[STEERING] Initial research plan created with {len(state.steering_todo.tasks)} tasks"
            )

        except Exception as e:
            self.logger.error(f"[STEERING] Error creating initial research plan: {e}")

    async def _update_todo_based_on_results(self, research_plan, search_results, state):
        """
        Update todo.md tasks based on research results - similar to how Cursor/Claude
        mark tasks as completed and adapt their approach based on outcomes.
        """
        try:
            self.logger.info(
                "[ADAPTIVE_FEEDBACK] Updating todo tasks based on research results"
            )

            # Analyze search results to determine task completion
            completed_searches = []
            failed_searches = []

            for i, result in enumerate(search_results):
                subtask = (
                    research_plan.get("subtasks", [])[i]
                    if i < len(research_plan.get("subtasks", []))
                    else {}
                )
                search_query = subtask.get("query", "")

                if result.get("success", False):
                    sources_found = len(result.get("sources", []))
                    completed_searches.append(
                        {
                            "query": search_query,
                            "sources_found": sources_found,
                            "description": subtask.get("description", ""),
                        }
                    )
                else:
                    failed_searches.append(
                        {
                            "query": search_query,
                            "error": result.get("error", "Unknown error"),
                        }
                    )

            # Create feedback message for the todo system
            feedback_message = f"""Research loop {state.research_loop_count} completed:

SUCCESSFUL SEARCHES ({len(completed_searches)}):
"""
            for search in completed_searches:
                feedback_message += f"- [OK] '{search['query']}' - Found {search['sources_found']} sources\n"

            if failed_searches:
                feedback_message += f"\nFAILED SEARCHES ({len(failed_searches)}):\n"
                for search in failed_searches:
                    feedback_message += (
                        f"- [FAIL] '{search['query']}' - {search['error']}\n"
                    )

            feedback_message += f"\nCurrent knowledge gaps: {state.knowledge_gap}\n"
            feedback_message += (
                f"Sources gathered so far: {len(state.sources_gathered)}\n"
            )

            # Add this as a system message to update the todo
            await state.steering_todo.add_user_message(
                f"SYSTEM_FEEDBACK: {feedback_message}"
            )

            self.logger.info(
                f"[ADAPTIVE_FEEDBACK] Updated todo with research results: {len(completed_searches)} successful, {len(failed_searches)} failed"
            )

        except Exception as e:
            self.logger.error(
                f"[ADAPTIVE_FEEDBACK] Error updating todo based on results: {e}"
            )

    async def _process_pending_steering_messages(self, state):
        """Process any pending steering messages and update research plan"""
        try:
            # Check if there are pending steering messages in the state
            if (
                hasattr(state, "pending_steering_messages")
                and state.pending_steering_messages
            ):
                self.logger.info(
                    f"[STEERING] Processing {len(state.pending_steering_messages)} pending messages"
                )

                # Process each message
                for message_data in state.pending_steering_messages:
                    try:
                        message_content = message_data.get(
                            "content", message_data.get("message", "")
                        )
                        await state.add_steering_message(message_content)
                        self.logger.info(
                            f"[STEERING] Processed message: {message_content}"
                        )
                    except Exception as e:
                        self.logger.error(f"[STEERING] Error processing message: {e}")

                # Clear processed messages
                state.pending_steering_messages.clear()

                # Update research focus based on steering constraints
                await self._adapt_research_plan_for_steering(state)

            elif (
                state.steering_todo and len(state.steering_todo.get_pending_tasks()) > 0
            ):
                # Even if no new messages, check if we need to adapt the research plan
                self.logger.info(
                    "[STEERING] Checking existing todo tasks for research adaptation"
                )
                await self._adapt_research_plan_for_steering(state)

        except Exception as e:
            self.logger.error(
                f"[STEERING] Error processing pending steering messages: {e}"
            )

    async def _adapt_research_plan_for_steering(self, state):
        """Adapt the research plan based on current steering constraints"""
        try:
            if not state.steering_todo:
                return

            constraints = state.steering_todo.get_current_constraints()

            # If we have focus constraints, update the main search query
            if constraints.get("focus_on"):
                focus_items = constraints["focus_on"]
                # Modify the search query to include focus constraints
                original_query = state.search_query
                focused_query = f"{original_query} {' '.join(focus_items)}"

                # Update state query for this research loop
                state.search_query = focused_query
                self.logger.info(
                    f"[STEERING] Adapted query: '{original_query}' → '{focused_query}'"
                )

            # Mark relevant todo tasks as in-progress
            pending_tasks = state.steering_todo.get_pending_tasks()
            for task in pending_tasks[:3]:  # Process top 3 pending tasks
                if any(
                    keyword in task.description.lower()
                    for keyword in ["research", "complete"]
                ):
                    from src.simple_steering import TaskStatus

                    state.steering_todo.mark_task_in_progress(task.id)
                    self.logger.info(f"[STEERING] Started task: {task.description}")

        except Exception as e:
            self.logger.error(f"[STEERING] Error adapting research plan: {e}")


class SearchAgent:
    """
    Specialized agent for executing search queries.

    This agent is responsible for executing specific search tasks
    using the appropriate search tool based on the subtopic.
    """

    def __init__(self, config=None, database_info=None):
        """
        Initialize the Search Agent.

        Args:
            config: Configuration object containing search settings
            database_info: Database context information for text2sql queries
        """
        self.config = config
        self.database_info = database_info
        self.logger = logging.getLogger(__name__)

    async def general_search(self, query):
        """Execute a general search query"""
        from src.graph import ToolRegistry, ToolExecutor

        self.logger.info(f"SearchAgent.general_search called with query: {query}")

        # Create tool registry and executor
        tool_registry = ToolRegistry(self.config)
        tool_executor = ToolExecutor(tool_registry)

        # Execute the search
        return await tool_executor.execute_tool(
            tool_name="general_search", params={"query": query, "top_k": 5}
        )

    async def academic_search(self, query):
        """Execute an academic search query"""
        from src.graph import ToolRegistry, ToolExecutor

        self.logger.info(f"SearchAgent.academic_search called with query: {query}")

        # Create tool registry and executor
        tool_registry = ToolRegistry(self.config)
        tool_executor = ToolExecutor(tool_registry)

        # Execute the search
        return await tool_executor.execute_tool(
            tool_name="academic_search", params={"query": query, "top_k": 5}
        )

    async def github_search(self, query):
        """Execute a GitHub search query"""
        from src.graph import ToolRegistry, ToolExecutor

        self.logger.info(f"SearchAgent.github_search called with query: {query}")

        # Create tool registry and executor
        tool_registry = ToolRegistry(self.config)
        tool_executor = ToolExecutor(tool_registry)

        # Execute the search
        return await tool_executor.execute_tool(
            tool_name="github_search", params={"query": query, "top_k": 5}
        )

    async def linkedin_search(self, query):
        """Execute a LinkedIn search query"""
        from src.graph import ToolRegistry, ToolExecutor

        self.logger.info(f"SearchAgent.linkedin_search called with query: {query}")

        # Create tool registry and executor
        tool_registry = ToolRegistry(self.config)
        tool_executor = ToolExecutor(tool_registry)

        # Execute the search
        return await tool_executor.execute_tool(
            tool_name="linkedin_search", params={"query": query, "top_k": 5}
        )

    async def text2sql_search(self, query):
        """Execute a text2sql query"""
        # Import the global text2sql tool instance from the database router
        from routers.database import text2sql_tool

        self.logger.info(f"SearchAgent.text2sql_search called with query: {query}")

        # Get database context from the agent's database_info
        db_id = None
        if hasattr(self, "database_info") and self.database_info:
            # Use the first database from the database_info
            if isinstance(self.database_info, list) and len(self.database_info) > 0:
                db_id = self.database_info[0].get("database_id")
                self.logger.info(f"Using database_id from database_info: {db_id}")

        # Use the global text2sql tool instance that has access to uploaded databases
        try:
            result = text2sql_tool._run(query, db_id=db_id)

            self.logger.info(f"text2sql_tool._run returned: {result}")

            # Format the result to match the expected search result format
            if "error" in result:
                return {
                    "index": 0,
                    "query": query,
                    "success": False,
                    "content": f"Error: {result['error']}",
                    "sources": [],
                    "error": result["error"],
                }
            else:
                # Format the successful result with HTML for better display
                content = "<div class='database-results' style='margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;'>\n"
                content += "<h3 style='color: #2c3e50; margin-top: 0;'>Database Analysis Results</h3>\n"
                content += (
                    f"<p style='color: #555;'><strong>Query:</strong> {query}</p>\n"
                )

                if "sql" in result:
                    content += "<div style='background-color: #f4f4f4; border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 15px 0; font-family: monospace; font-size: 14px; overflow-x: auto;'>\n"
                    content += "<h4 style='margin-top: 0; color: #555;'>Generated SQL Query:</h4>\n"
                    content += f"<pre style='margin: 0; white-space: pre-wrap;'><code>{result['sql']}</code></pre>\n"
                    content += "</div>\n"

                if "results" in result and result["results"]:
                    results_data = result["results"]
                    if results_data.get("type") == "select" and results_data.get(
                        "rows"
                    ):
                        content += f"<h4 style='color: #2c3e50; margin-top: 20px;'>Query Results ({results_data.get('row_count', 0)} rows)</h4>\n"

                        # Create HTML table with inline styles
                        columns = results_data.get("columns", [])
                        if columns:
                            content += "<table style='width: 100%; border-collapse: collapse; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); background-color: white;'>\n"
                            content += "<thead><tr>\n"
                            for col in columns:
                                content += f"<th style='background-color: #3498db; color: white; padding: 12px; text-align: left; font-weight: bold; border-bottom: 2px solid #2980b9;'>{col}</th>\n"
                            content += "</tr></thead>\n"
                            content += "<tbody>\n"

                            # Add data rows
                            for idx, row in enumerate(results_data.get("rows", [])):
                                bg_color = "#f8f9fa" if idx % 2 == 0 else "white"
                                content += (
                                    f"<tr style='background-color: {bg_color};'>\n"
                                )
                                for col in columns:
                                    value = row.get(col, "")
                                    # Format numbers nicely
                                    if isinstance(value, float):
                                        value = f"{value:.2f}"
                                    content += f"<td style='padding: 10px; border-bottom: 1px solid #ddd;'>{value}</td>\n"
                                content += "</tr>\n"

                            content += "</tbody>\n"
                            content += "</table>\n"
                    else:
                        content += f"<p><strong>Results:</strong> {results_data}</p>\n"

                if "database" in result:
                    content += f"<p style='color: #555; margin-top: 15px;'><strong>Source Database:</strong> {result['database']}</p>\n"

                if "executed_at" in result:
                    content += f"<p style='color: #888; font-size: 0.9em;'><strong>Executed at:</strong> {result['executed_at']}</p>\n"

                content += "</div>\n"

                self.logger.info(f"text2sql formatted content: {content[:200]}...")

                return {
                    "index": 0,
                    "query": query,
                    "success": True,
                    "content": content,
                    "sources": [
                        {
                            "title": f'Database Query Results - {result.get("database", "Unknown Database")}',
                            "url": f'database://{result.get("database", "unknown")}',
                            "snippet": f'SQL: {result.get("sql", "N/A")}\nResults: {len(result.get("results", {}).get("rows", []))} rows returned',
                            "source_type": "database",
                        }
                    ],
                    "error": None,
                }

        except Exception as e:
            self.logger.error(f"Error in text2sql_search: {e}")
            return {
                "index": 0,
                "query": query,
                "success": False,
                "content": f"Error executing text2sql query: {str(e)}",
                "sources": [],
                "error": str(e),
            }

    async def execute(self, subtask, tool_executor=None):
        """
        Execute a search for a specific subtask asynchronously using the provided tool executor.
        This method directly uses the query and tool name from the subtask description
        without making an additional LLM call.

        Args:
            subtask: Dict containing task details (query, tool, name, aspect, etc.)
            tool_executor: An initialized ToolExecutor instance.

        Returns:
            Dict containing the search results or an error.
        """
        # Import necessary components
        from src.graph import ToolRegistry, ToolExecutor  # Keep ToolExecutor import
        import traceback

        # Ensure tool_executor is provided (as it's essential now)
        if tool_executor is None:
            self.logger.error("[SearchAgent] ToolExecutor instance must be provided.")
            # Consider raising an error or handling appropriately
            # For now, let's try creating one, but this indicates a potential design issue
            try:
                from src.graph import ToolRegistry

                tool_registry = ToolRegistry(self.config)
                tool_executor = ToolExecutor(tool_registry)
                self.logger.warning(
                    "[SearchAgent] Created a default ToolExecutor instance as none was provided."
                )
            except Exception as te_err:
                self.logger.error(
                    f"[SearchAgent] Failed to create default ToolExecutor: {te_err}"
                )
                return {
                    "error": "ToolExecutor instance required but not provided and creation failed."
                }

        # Directly extract parameters from the subtask dictionary
        query = subtask.get("query")
        # Default to general_search if not specified in subtask
        tool_name = subtask.get("suggested_tool", subtask.get("tool", "general_search"))
        subtask_name = subtask.get("name", "Unnamed subtask")

        # Validate extracted parameters
        if not query:
            self.logger.error(
                f"[SearchAgent] Subtask '{subtask_name}' is missing a query."
            )
            return {"error": f"Subtask '{subtask_name}' missing query."}
        if not tool_name:
            self.logger.error(
                f"[SearchAgent] Subtask '{subtask_name}' is missing a tool name."
            )
            # Defaulting again just in case, but log the error
            tool_name = "general_search"

        self.logger.info(
            f"[SearchAgent] Directly executing tool '{tool_name}' for subtask '{subtask_name}' with query: '{query}'"
        )

        try:
            # --- LLM Call Removed ---
            # No need to call LLM again to determine parameters; use directly from subtask.

            # Define standard parameters (can be extended if MasterAgent provides more)
            # Example: if MasterAgent decided top_k based on decomposition:
            # top_k = subtask.get("parameters", {}).get("top_k", 5)
            params = {
                "query": query,
                "top_k": 5,  # Using a default, could be passed in subtask if needed
            }

            # Execute search with the specified tool asynchronously using ToolExecutor
            self.logger.info(
                f"[SearchAgent] Calling tool_executor.execute_tool with tool_name='{tool_name}', params={params}"
            )
            search_result = await tool_executor.execute_tool(
                tool_name=tool_name, params=params
            )

            # Ensure search_result is a dictionary (ToolExecutor should ideally ensure this)
            if not isinstance(search_result, dict):
                self.logger.warning(
                    f"[SearchAgent] ToolExecutor result is not a dictionary: {type(search_result)}. Converting."
                )
                search_result = {
                    "formatted_sources": str(search_result),
                    "search_string": query,
                    "domains": [],
                }

            # Add metadata to the result
            search_result["subtask"] = subtask  # Include original subtask for context
            search_result["tool_used"] = tool_name  # Log the tool used
            # No "tool_used_llm" as LLM didn't choose the tool here

            self.logger.info(
                f"[SearchAgent] Successfully executed tool '{tool_name}' for subtask '{subtask_name}'"
            )
            return search_result

        except Exception as e:
            self.logger.error(
                f"[SearchAgent] Error executing tool '{tool_name}' for subtask '{subtask_name}': {str(e)}"
            )
            self.logger.error(traceback.format_exc())

            # Return error result
            return {
                "error": str(e),
                "formatted_sources": f"Error executing search: {str(e)}",
                "search_string": query,
                "domains": [],
                "subtask": subtask,  # Include original subtask even on error
                "tool_used": tool_name,
            }


class ResultCombiner:
    """
    Agent for combining results from multiple specialized agents.

    This agent takes the outputs from various specialized agents
    and combines them into a cohesive research result.
    """

    def __init__(self, config=None):
        """
        Initialize the Result Combiner.

        Args:
            config: Configuration object containing combining settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def combine(
        self,
        research_plan,
        subtask_results,
        query,
        research_loop_count,
        original_research_topic,
        current_state=None,
    ):
        """
        Combine results from multiple specialized agents.

        Args:
            research_plan: The original research plan
            subtask_results: List of results from specialized agents
            query: The current search query being used
            research_loop_count: Current research loop count
            original_research_topic: The original research topic from the user (preserved across iterations)
            current_state: The current state object containing existing sources

        Returns:
            Dict containing the combined research results in a format compatible with the existing codebase
        """
        self.logger.info(
            f"[ResultCombiner] Combining {len(subtask_results)} results for loop {research_loop_count}"
        )

        # --- START FIX: Load existing visualizations from state ---
        existing_visualizations = getattr(current_state, "visualizations", [])
        existing_viz_paths = getattr(current_state, "visualization_paths", [])
        existing_base64 = getattr(current_state, "base64_encoded_images", [])
        self.logger.info(
            f"[ResultCombiner] Loaded {len(existing_visualizations)} existing visualizations, {len(existing_viz_paths)} paths, {len(existing_base64)} base64 images from previous state."
        )

        # Initialize collection variables with existing data
        all_formatted_sources = []  # Formatted sources are typically loop-specific
        all_search_strings = []  # Search strings are loop-specific
        all_tools_used = set(
            getattr(current_state, "tools", [])
        )  # Accumulate tools used
        all_visualizations = (
            existing_visualizations.copy()
        )  # Start with existing visualizations
        # Use sets for paths and filenames to handle duplicates easily
        seen_viz_paths = set(existing_viz_paths)
        # Need a way to uniquely identify base64 images if structure allows (e.g., by filename if present)
        # Using filename for now, assuming base64_encoded_images is a list of dicts with 'filename'
        seen_base64_filenames = set(
            img.get("filename")
            for img in existing_base64
            if isinstance(img, dict) and img.get("filename")
        )
        current_loop_base64_images = []
        # --- END FIX ---

        self.logger.info(
            f"[ResultCombiner] Processing {len(subtask_results)} subtask results:"
        )
        for i, result in enumerate(subtask_results):
            keys = (
                list(result.keys())
                if isinstance(result, dict)
                else ["result is not a dict"]
            )
            if isinstance(result, dict) and "error" in result:
                self.logger.warning(
                    f"[ResultCombiner] Result {i+1}: ERROR={result.get('error')}"
                )
            # Check for visualization results (assuming structure from VisualizationAgent.execute)
            elif (
                isinstance(result, dict)
                and "results" in result
                and isinstance(result["results"], dict)
                and "results" in result["results"]
            ):
                viz_count = len(result["results"].get("results", []))
                self.logger.info(
                    f"[ResultCombiner] Result {i+1}: VISUALIZATION with {viz_count} images. Keys: {keys}"
                )
            elif isinstance(result, dict):
                self.logger.info(
                    f"[ResultCombiner] Result {i+1}: Regular result with keys: {keys}"
                )
            else:
                self.logger.warning(
                    f"[ResultCombiner] Result {i+1}: Unexpected type {type(result)}"
                )

        # Process each subtask result
        subtopic_results = []
        all_code_snippets = []  # Initialize list for code snippets

        # NEW: accumulate raw contents for storing in state.web_research_results
        all_raw_contents = []

        # --- FIX: Process results and append NEW visualizations ---
        new_viz_added_count = 0
        for result in subtask_results:
            if not isinstance(result, dict):  # Skip non-dict results
                continue

            # Skip results with errors
            if "error" in result:
                self.logger.warning(
                    f"[ResultCombiner] Skipping result with error: {result.get('error')}"
                )
                continue

            # Check if this is a visualization result
            # Structure expected: {'search_result': {...}, 'code_data': {...}, 'results': {'results': [...] or 'error': ...}}
            is_visualization_result = (
                "results" in result
                and isinstance(result["results"], dict)
                and "results" in result["results"]
                and isinstance(result["results"]["results"], list)
            )
            visualization_failed = (
                "results" in result
                and isinstance(result["results"], dict)
                and "error" in result["results"]
            )

            if visualization_failed:
                # Use single quotes inside the f-string expression
                self.logger.warning(
                    f"[ResultCombiner] Visualization subtask failed: {result['results'].get('error')}"
                )
                # Still process the underlying search result if it exists
                if "search_result" in result and isinstance(
                    result["search_result"], dict
                ):
                    search_result_data = result["search_result"]
                    if "subtask" in search_result_data:
                        # Process search result components
                        if isinstance(
                            search_result_data.get("formatted_sources"), list
                        ):
                            all_formatted_sources.extend(
                                search_result_data.get("formatted_sources", [])
                            )
                        else:
                            all_formatted_sources.append(
                                search_result_data.get("formatted_sources", "")
                            )
                        all_search_strings.append(
                            search_result_data.get("search_string", "")
                        )
                        all_tools_used.add(
                            search_result_data.get(
                                "tool_used",
                                search_result_data.get("subtask", {}).get(
                                    "tool", "general_search"
                                ),
                            )
                        )
                        subtopic_results.append(
                            {
                                "subtopic": search_result_data.get("subtask", {}),
                                "search_result": search_result_data,
                            }
                        )
                continue  # Skip adding visualization parts for failed task

            if is_visualization_result:
                viz_list = result["results"].get("results", [])
                viz_files_from_this_task = []
                for viz_item in viz_list:
                    filepath = None
                    filename = None
                    description = None  # Initialize description
                    if isinstance(viz_item, dict) and viz_item.get("type") == "image":
                        filepath = viz_item.get("filepath")
                        description = viz_item.get("description")  # Get description
                        if filepath and os.path.exists(filepath):
                            filename = viz_item.get(
                                "filename", os.path.basename(filepath)
                            )
                        else:
                            # Use repr() for safer logging
                            self.logger.warning(
                                f"[ResultCombiner] Invalid/missing viz filepath: {repr(filepath)}"
                            )
                            filepath = None  # Ensure invalid path isn't used

                    if filepath and filename:
                        # Check if this visualization path is already tracked
                        if filepath not in seen_viz_paths:
                            viz_files_from_this_task.append(
                                {
                                    "filepath": filepath,
                                    "filename": filename,
                                    "subtask_name": result.get("code_data", {}).get(
                                        "subtask_name", "Visualization"
                                    ),
                                    "description": description,  # Store description
                                }
                            )
                            seen_viz_paths.add(filepath)  # Track the path
                            new_viz_added_count += 1
                        else:
                            # Use repr() for safer logging
                            self.logger.info(
                                f"[ResultCombiner] Skipping duplicate viz filepath: {repr(filepath)}"
                            )
                    else:
                        # Use repr() for safer logging
                        self.logger.warning(
                            f"[ResultCombiner] Skipping invalid/missing visualization item: {repr(viz_item)}"
                        )

                # Extend the main list with new, unique visualizations from this task
                all_visualizations.extend(viz_files_from_this_task)
                self.logger.info(
                    f"[ResultCombiner] Added {len(viz_files_from_this_task)} NEW unique visualizations from this task."
                )
                # Also handle the base64 encoded images if they exist for this viz task
                # (assuming format_visualizations_html populates self._base64_encoded_images)
                # We need to call format_visualizations_html *for this task's visualizations* temporarily
                # to potentially populate the base64 cache within the combiner instance for this call
                self.format_visualizations_html(
                    viz_files_from_this_task
                )  # This populates self._base64_encoded_images
                if (
                    hasattr(self, "_base64_encoded_images")
                    and self._base64_encoded_images
                ):
                    for img_data in self._base64_encoded_images:
                        if (
                            isinstance(img_data, dict)
                            and img_data.get("filename")
                            and img_data["filename"] not in seen_base64_filenames
                        ):
                            current_loop_base64_images.append(img_data)
                            seen_base64_filenames.add(img_data["filename"])
                    self._base64_encoded_images = (
                        []
                    )  # Clear instance cache for next task

            # Process the underlying search result data (even if part of a viz result)
            search_result_data = None
            if "search_result" in result and isinstance(result["search_result"], dict):
                search_result_data = result["search_result"]
            elif (
                "subtask" in result
            ):  # Handle regular search results not nested under 'search_result'
                search_result_data = result

            if search_result_data and "subtask" in search_result_data:
                subtask = search_result_data.get("subtask", {})
                # Extract components for UI state
                if isinstance(search_result_data.get("formatted_sources"), list):
                    all_formatted_sources.extend(
                        search_result_data.get("formatted_sources", [])
                    )
                else:
                    all_formatted_sources.append(
                        search_result_data.get("formatted_sources", "")
                    )
                all_search_strings.append(search_result_data.get("search_string", ""))
                all_tools_used.add(
                    search_result_data.get(
                        "tool_used", subtask.get("tool", "general_search")
                    )
                )
                subtopic_results.append(
                    {"subtopic": subtask, "search_result": search_result_data}
                )

                # Collect raw_contents
                if "raw_contents" in search_result_data and isinstance(
                    search_result_data["raw_contents"], list
                ):
                    for rc in search_result_data["raw_contents"]:
                        if rc:  # Ensure we don't append empty/None items
                            all_raw_contents.append(rc)
            elif not is_visualization_result and not visualization_failed:
                self.logger.warning(
                    f"[ResultCombiner] Skipping result with unrecognized structure: {result.keys() if isinstance(result, dict) else type(result)}"
                )

            # --- START: Aggregate Code Snippets ---
            # Check direct top-level code_snippets (our new addition for visualizations)
            if (
                isinstance(result.get("code_snippets"), list)
                and result["code_snippets"]
            ):
                all_code_snippets.extend(result["code_snippets"])
                self.logger.info(
                    f"[ResultCombiner] Found and added {len(result['code_snippets'])} code snippets from top-level code_snippets."
                )
            # Check for code_snippets in results object (also for visualizations)
            elif isinstance(result.get("results"), dict) and isinstance(
                result["results"].get("code_snippets"), list
            ):
                snippets = result["results"]["code_snippets"]
                if snippets:
                    all_code_snippets.extend(snippets)
                    self.logger.info(
                        f"[ResultCombiner] Found and added {len(snippets)} code snippets from results.code_snippets."
                    )
            # Check top-level enriched_data (for non-visualization results)
            elif isinstance(result.get("enriched_data"), dict):
                snippets = result["enriched_data"].get("code_snippets")
                if isinstance(snippets, list) and snippets:
                    all_code_snippets.extend(snippets)
                    self.logger.info(
                        f"[ResultCombiner] Found and added {len(snippets)} code snippets from top-level enriched_data."
                    )
            # Check nested enriched_data within the 'results' key (for visualization results)
            elif isinstance(result.get("results"), dict) and isinstance(
                result["results"].get("enriched_data"), dict
            ):
                nested_enriched_data = result["results"]["enriched_data"]
                snippets = nested_enriched_data.get("code_snippets")
                if isinstance(snippets, list) and snippets:
                    all_code_snippets.extend(snippets)
                    self.logger.info(
                        f"[ResultCombiner] Found and added {len(snippets)} code snippets from nested results.enriched_data."
                    )
            # --- END: Aggregate Code Snippets ---

        # --- END FIX ---

        self.logger.info(
            f"[ResultCombiner] Total NEW unique visualizations added in this loop: {new_viz_added_count}"
        )
        self.logger.info(
            f"[ResultCombiner] Accumulated total unique visualizations: {len(all_visualizations)}"
        )
        final_base64_images = existing_base64 + current_loop_base64_images
        self.logger.info(
            f"[ResultCombiner] Accumulated total base64 images: {len(final_base64_images)}"
        )

        # --- EXISTING LOGIC TO PARSE FORMATTED SOURCES ---
        parsed_sources = []
        seen_urls = set()  # To handle potential duplicates from formatted_sources
        for source_str in all_formatted_sources:
            if not isinstance(source_str, str):  # Handle potential non-string items
                self.logger.warning(
                    f"[ResultCombiner] Skipping non-string item in all_formatted_sources: {type(source_str)}"
                )
                continue

            # Attempt to parse the string - adjust regex if format is different
            # Common formats: "1. Title: [Actual Title] (Source: [Actual URL])", "Title: ... (Source: ...)", "[Title](URL)"
            # Regex tries to capture title and URL from common patterns
            match = None
            # Pattern 1: Explicit Title and Source labels
            pattern1 = r"(?:\d+\.\s*)?(?:Title|Name|headline):\s*(.*?)\s*(?:\(Source:|\(URL:|URL:|Source:)\s*(https?://.*?)(?:\)|$)"
            # Pattern 2: Markdown link style
            pattern2 = r"\s*\[(.*?)\]\((https?://.*?)\)"
            # Pattern 3: Simpler Title : URL
            pattern3 = r"\s*(.*?)\s*:\s*(https?://\S+)"

            match = re.search(pattern1, source_str.strip(), re.IGNORECASE)
            if not match:
                match = re.search(pattern2, source_str.strip())
            if not match:
                match = re.search(pattern3, source_str.strip())

            if match:
                # Group 1 is usually title, Group 2 is usually URL across patterns
                title = match.group(1).strip() if match.group(1) else "No Title Found"
                url = (
                    match.group(2).strip()
                    if len(match.groups()) > 1 and match.group(2)
                    else None
                )

                if not url and len(match.groups()) > 0:
                    # Check if URL might be the only capture group in some patterns
                    potential_url = match.group(1).strip()
                    if potential_url.startswith("http"):
                        url = potential_url
                        title = "No Title Found"  # Reset title if URL was in group 1

                if url and url not in seen_urls:
                    parsed_sources.append({"title": title, "url": url})
                    seen_urls.add(url)
                elif url:
                    self.logger.info(
                        f"[ResultCombiner] Skipping duplicate URL found in formatted_sources: {url}"
                    )
                else:
                    self.logger.warning(
                        f"[ResultCombiner] Could not extract URL from formatted source: {source_str}"
                    )

            else:
                self.logger.warning(
                    f"[ResultCombiner] Could not parse formatted source string: {source_str}"
                )

        # Create combined results structure based on topic complexity
        if research_plan.get("topic_complexity") == "complex":
            combined_results = {
                "topic_complexity": "complex",
                "main_query": research_plan.get("main_query", query),
                "main_tool": research_plan.get("main_tool", "general_search"),
                "subtopic_results": subtopic_results,
                "visualizations": all_visualizations,
            }
        else:
            # For simple topics, use a simpler structure
            combined_results = {
                "topic_complexity": "simple",
                "query": query,
                "aspect": (
                    subtask_results[0]
                    .get("subtask", {})
                    .get("aspect", "general information")
                    if subtask_results
                    else "general information"
                ),
                "rationale": "Direct search",
                "suggested_tool": (
                    list(all_tools_used)[0] if all_tools_used else "general_search"
                ),
                "search_result": subtask_results[0] if subtask_results else {},
                "visualizations": all_visualizations,
            }

        # --- BUILD OUTPUTS FROM PARSED SOURCES ---
        # Get existing citations if available
        existing_source_citations = {}
        if current_state is not None:
            existing_source_citations = getattr(current_state, "source_citations", {})

        # Create new citations dictionary, starting with existing citations
        citations = existing_source_citations.copy()
        sources_gathered = []
        final_domains = []  # Use parsed_sources for the 'domains' output field

        # Get existing sources to check for duplicates
        existing_sources = []
        if current_state is not None:
            existing_sources = getattr(current_state, "sources_gathered", [])
        existing_urls = set()

        # Extract URLs from existing sources for faster duplicate checking
        for source in existing_sources:
            if " : " in source:
                url = source.split(" : ", 1)[1].strip()
                existing_urls.add(url)

        self.logger.info(
            f"[ResultCombiner] Found {len(existing_sources)} existing sources with {len(existing_urls)} unique URLs for deduplication"
        )
        self.logger.info(
            f"[ResultCombiner] Starting with {len(existing_source_citations)} existing citations"
        )

        # Determine the next available citation number
        next_citation_num = 1
        if citations:
            # Extract highest existing citation number
            existing_nums = [int(num) for num in citations.keys() if num.isdigit()]
            if existing_nums:
                next_citation_num = max(existing_nums) + 1

        # Add new sources that aren't duplicates
        for source_info in parsed_sources:
            title = source_info.get("title", "N/A")
            url = source_info.get("url", "")

            # Check if URL already exists in existing citations to avoid duplicates
            url_already_in_citations = False
            for citation in citations.values():
                if citation.get("url") == url:
                    url_already_in_citations = True
                    break

            if url and not url_already_in_citations and url not in existing_urls:
                # Add as a new citation with incrementing number
                citations[str(next_citation_num)] = {"title": title, "url": url}
                sources_gathered.append(f"{title} : {url}")
                final_domains.append(
                    {"title": title, "url": url}
                )  # Populate domains with parsed info
                existing_urls.add(
                    url
                )  # Add to existing URLs to prevent duplicates within this batch
                next_citation_num += 1

        self.logger.info(
            f"[ResultCombiner] Finished with {len(citations)} total citations after adding new ones"
        )

        # Format visualizations for inclusion in the report
        # Ensure all_visualizations contains unique items before formatting
        unique_visualizations = []
        seen_paths = set()
        for viz_data in all_visualizations:
            if (
                isinstance(viz_data, dict)
                and viz_data.get("filepath") not in seen_paths
            ):
                unique_visualizations.append(viz_data)
                seen_paths.add(viz_data.get("filepath"))
            elif isinstance(viz_data, str) and viz_data not in seen_paths:
                unique_visualizations.append(
                    viz_data
                )  # Keep strings if they represent valid paths
                seen_paths.add(viz_data)

        self.logger.info(
            f"[ResultCombiner] Processing {len(unique_visualizations)} unique visualizations for HTML embedding."
        )
        visualization_html = self.format_visualizations_html(unique_visualizations)

        # --- START MODIFICATION: Create visualization_paths list ---
        visualization_paths = []
        for viz_data in unique_visualizations:  # Use unique list
            if isinstance(viz_data, str):
                # Direct filepath string
                if os.path.exists(viz_data):
                    visualization_paths.append(viz_data)
            elif isinstance(viz_data, dict) and viz_data.get("filepath"):
                # Dictionary with filepath attribute
                if os.path.exists(viz_data.get("filepath")):
                    visualization_paths.append(viz_data.get("filepath"))

        # DEBUG: Log information about extracted visualization paths
        self.logger.info(
            f"[ResultCombiner] Extracted {len(visualization_paths)} visualization paths from {len(all_visualizations)} visualization entries"
        )
        for i, path in enumerate(visualization_paths):
            self.logger.info(f"[ResultCombiner] Visualization path {i+1}: {path}")
        if not visualization_paths and all_visualizations:
            self.logger.warning(
                f"[ResultCombiner] Found {len(all_visualizations)} visualizations but extracted 0 paths!"
            )
            for i, viz in enumerate(all_visualizations):
                if isinstance(viz, dict):
                    self.logger.warning(
                        f"[ResultCombiner] Visualization {i+1} keys: {viz.keys() if hasattr(viz, 'keys') else 'No keys (not a dict)'}"
                    )
                    self.logger.warning(
                        f"[ResultCombiner] Visualization {i+1} filepath: {viz.get('filepath') if hasattr(viz, 'get') else 'No get method'}"
                    )
                else:
                    self.logger.warning(
                        f"[ResultCombiner] Visualization {i+1} type: {type(viz)}, value: {viz}"
                    )
        # --- END MODIFICATION ---

        # Return both new and old format results for backward compatibility
        final_result = {
            "research_results": combined_results,
            "citations": citations,  # New citations from parsed sources
            "source_citations": citations,  # Use the same merged citations dictionary for source_citations
            "formatted_sources": all_formatted_sources,  # Keep original formatted strings
            "search_string": " | ".join(all_search_strings),
            "research_topic": original_research_topic,  # Use the original research topic instead of the query
            "research_loop_count": research_loop_count,
            "tools": list(all_tools_used),
            "domains": final_domains,  # New domains list from parsed sources
            # We store the raw contents in web_research_results
            "web_research_results": all_raw_contents,
            "sources_gathered": sources_gathered,  # New sources_gathered from parsed sources
            "selected_search_tool": (
                list(all_tools_used)[-1] if all_tools_used else "general_search"
            ),
            "visualizations": all_visualizations,  # Keep original visualization dicts for potential other uses
            "visualization_paths": visualization_paths,  # <-- ADDED: List of path strings for finalize_report
            "visualization_html": visualization_html,  # HTML string for visualization embedding
            "base64_encoded_images": final_base64_images,  # Include base64 encoded images if they exist
            "code_snippets": all_code_snippets,  # Add aggregated code snippets
        }

        # --- Add debug logging before return ---
        self.logger.info(f"[ResultCombiner] COMBINE FINAL CHECK:")
        self.logger.info(
            f"[ResultCombiner]   - Returning combined_summary length: {len(final_result.get('research_results', {}))}"
        )
        self.logger.info(
            f"[ResultCombiner]   - Returning sources_gathered count: {len(final_result.get('sources_gathered', []))}"
        )
        self.logger.info(
            f"[ResultCombiner]   - Returning source_citations count: {len(final_result.get('source_citations', {}))}"
        )
        self.logger.info(
            f"[ResultCombiner]   - Returning visualization_html length: {len(final_result.get('visualization_html', ''))}"
        )
        self.logger.info(
            f"[ResultCombiner]   - Returning visualizations count: {len(final_result.get('visualizations', []))}"
        )
        self.logger.info(
            f"[ResultCombiner]   - Returning visualization_paths count: {len(final_result.get('visualization_paths', []))}"
        )
        self.logger.info(
            f"[ResultCombiner]   - Returning base64_encoded_images count: {len(final_result.get('base64_encoded_images', []))}"
        )
        self.logger.info(
            f"[ResultCombiner]   - Returning code_snippets count: {len(final_result.get('code_snippets', []))}"
        )  # Log snippet count
        if len(final_result.get("base64_encoded_images", [])) > 0:
            self.logger.info(
                f"[ResultCombiner]   - First base64 image src start: {final_result['base64_encoded_images'][0].get('src', '')[:100]}..."
            )

        return final_result

    def format_visualizations_html(self, visualizations):
        """
        Format visualizations as HTML for inclusion in the research report.

        Args:
            visualizations: List of visualization metadata dictionaries or strings

        Returns:
            HTML string with embedded visualizations
        """
        if not visualizations:
            print(
                "WORKING@@@@@@@@[format_visualizations_html] No visualizations to format"
            )
            return ""

        html_parts = ["<div class='visualizations'>"]
        base64_encoded_images = []

        for viz in visualizations:
            # Handle different types of visualization items
            if isinstance(viz, str):
                # Direct string filepath
                filepath = viz
                filename = os.path.basename(filepath)
                title = "Visualization"
            else:
                # Dictionary with metadata
                filename = viz.get("filename", "")
                filepath = viz.get("filepath", "")
                title = (
                    viz.get("subtask_name", "Visualization").replace("_", " ").title()
                )

            # Check if filepath exists and is not empty
            if not filepath or not os.path.exists(filepath):
                self.logger.warning(
                    f"[ResultCombiner] Visualization file not found: {filepath}"
                )
                continue

            # Convert image to base64 for embedding
            try:
                with open(filepath, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode("utf-8")

                # Determine image format from filename
                img_format = os.path.splitext(filename)[1][1:].lower()
                if img_format not in ["png", "jpg", "jpeg", "gif", "svg"]:
                    img_format = "png"  # Default to png

                # Store base64 data for this image
                base64_encoded_images.append(
                    {
                        "filename": filename,
                        "title": title,
                        "base64_data": img_data,
                        "format": img_format,
                    }
                )

                # Debug: Log base64 data length to help diagnose issues
                self.logger.info(
                    f"[ResultCombiner] Embedding visualization {filepath} with base64 data length: {len(img_data)}"
                )

                # Use data URI for reliable embedding
                html_parts.append(
                    f"""
                <div class='visualization-container'>
                    <h3>{title}</h3>
                    <img src="data:image/{img_format};base64,{img_data}" alt="{title}" style="max-width:100%; height:auto;" />
                </div>
                """
                )
                self.logger.info(
                    f"[ResultCombiner] Successfully embedded visualization: {filepath}"
                )
            except Exception as e:
                self.logger.error(
                    f"[ResultCombiner] Error embedding visualization {filepath}: {str(e)}"
                )
                self.logger.error(traceback.format_exc())

        html_parts.append("</div>")

        # Add CSS for visualization styling
        css = """
        <style>
        .visualizations {
            margin: 20px 0;
        }
        .visualization-container {
            margin-bottom: 30px;
        }
        .visualization-container h3 {
            margin-bottom: 10px;
            font-weight: bold;
        }
        .visualization-container img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            height: auto;
            display: block;
        }
        </style>
        """

        # Store base64 data in a custom attribute for access by finalize_report
        self._base64_encoded_images = base64_encoded_images

        return css + "\n".join(html_parts)
