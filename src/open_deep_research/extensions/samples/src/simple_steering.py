"""
Simple Steering System for Deep Research
Like Claude Code's todo.md but for research steering.

User sends messages during research -> Messages queued -> Converted to todo tasks -> Agent reads before next loop
"""

import asyncio
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import re

# Create a dedicated logger for steering events
logger = logging.getLogger(__name__)
steering_logger = logging.getLogger("steering")  # Dedicated steering logger


# Helper function for important steering events
def log_steering(message: str, level: str = "info"):
    """Log steering-specific events that can be filtered"""
    log_func = getattr(steering_logger, level, steering_logger.info)
    log_func(f"[STEERING] {message}")


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class SteeringTask:
    """Simple task derived from user steering message"""

    id: str
    description: str
    status: TaskStatus
    priority: int = 5  # 1-10
    source: str = (
        "knowledge_gap"  # "original_query", "knowledge_gap", "steering_message"
    )
    created_from_message: str = ""
    created_at: datetime = None
    search_queries: List[str] = field(default_factory=list)  # Related search queries
    completed_note: str = ""  # Note about how/why task was completed or cancelled
    completed_at: Optional[datetime] = None  # When task was completed

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ResearchTodoManager:
    """
    Simple todo.md manager for research steering.
    Like Claude Code's todo system but for research tasks.
    """

    def __init__(self, research_topic: str):
        self.research_topic = research_topic
        self.tasks: Dict[str, SteeringTask] = {}
        self.pending_messages: List[str] = []  # User messages queue
        self.all_user_messages: List[str] = []  # ALL user messages for final report
        self.todo_version = 1
        self.last_updated = datetime.now()
        self.initial_plan_created = False  # Track if initial plan was created
        self.research_loop_count = 0  # Track current research loop
        self.executed_queries: set = (
            set()
        )  # Track queries we've already executed to avoid duplicates

    async def create_initial_plan(self, initial_query: str, research_context: str = ""):
        """Create initial todo.md plan from the research query"""
        if self.initial_plan_created:
            logger.info("[STEERING] Initial plan already created, skipping")
            return

        try:
            from llm_clients import get_async_llm_client
            from src.configuration import Configuration

            # Use configured LLM provider instead of hardcoded Google
            config = Configuration.from_runnable_config()
            provider = config.activity_llm_provider.value if hasattr(config.activity_llm_provider, 'value') else config.activity_llm_provider
            model = config.activity_llm_model

            llm = await get_async_llm_client(provider, model)
            # Set max_output_tokens - increased for JSON response
            if hasattr(llm, "max_output_tokens"):
                llm.max_output_tokens = 2048
            elif hasattr(llm, "max_tokens"):
                llm.max_tokens = 2048

            planning_prompt = f"""
You are creating an initial research plan for the topic: "{self.research_topic}"

Initial Query: "{initial_query}"
Research Context: {research_context if research_context else "Starting fresh research"}

Create 3-5 initial research tasks that break down this query into actionable research steps. Return a JSON array where each task is an object with:
- "description": Clear, actionable research task (string)
- "priority": 1-10 (integer, higher = more important, default=5)
- "type": "research" (string, always "research" for initial tasks)

Focus on:
1. Understanding the core topic
2. Gathering comprehensive information 
3. Identifying key aspects to explore
4. Building foundational knowledge

Example for "Silvio Savarese":

<answer>
[
  {{"description": "Research Silvio Savarese's academic background and current position", "priority": 8, "type": "research"}},
  {{"description": "Investigate his key research contributions and publications", "priority": 7, "type": "research"}},
  {{"description": "Explore his industry experience and leadership roles", "priority": 6, "type": "research"}},
  {{"description": "Analyze his impact on computer vision and AI fields", "priority": 5, "type": "research"}}
]
</answer>

CRITICAL: You MUST wrap your JSON array in <answer></answer> tags. Output ONLY the <answer> tags with valid JSON inside. No explanatory text before or after.
"""

            messages = [{"role": "user", "content": planning_prompt}]
            response = await llm.ainvoke(messages)

            import json

            try:
                response_text = response.content.strip()
                logger.info(
                    f"[STEERING] LLM response for initial plan: {response_text[:300]}..."
                )

                # Helper function to extract JSON from <answer> tags
                def parse_wrapped_response(reg_exp, text_phrase):
                    match = re.search(reg_exp, text_phrase, re.DOTALL)
                    if match:
                        return match.group(1)
                    return ""

                # First try to extract from <answer> tags
                json_text = parse_wrapped_response(
                    r"<answer>\s*(.*?)\s*</answer>", response_text
                )

                if json_text:
                    # Clean up the JSON string
                    json_text = json_text.strip()
                else:
                    # Fallback: Remove markdown code blocks if present
                    if "```json" in response_text:
                        response_text = re.sub(r"```json\s*", "", response_text)
                        response_text = re.sub(r"```\s*$", "", response_text)
                        response_text = response_text.strip()
                    elif "```" in response_text:
                        response_text = re.sub(r"```\s*", "", response_text)
                        response_text = response_text.strip()

                    # Extract JSON array
                    json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group()
                    else:
                        raise json.JSONDecodeError(
                            "No JSON array found", response_text, 0
                        )

                # Fix trailing commas
                for _ in range(5):
                    json_text = re.sub(r",(\s*[\]}])", r"\1", json_text)

                initial_tasks = json.loads(json_text)

                if not isinstance(initial_tasks, list):
                    initial_tasks = [initial_tasks]

                # Convert string items to objects if needed
                converted_tasks = []
                for task in initial_tasks:
                    if isinstance(task, str):
                        # LLM returned strings instead of objects
                        converted_tasks.append(
                            {"description": task, "priority": 5, "type": "research"}
                        )
                    elif isinstance(task, dict):
                        converted_tasks.append(task)
                    else:
                        logger.warning(
                            f"[STEERING] Skipping invalid task format: {type(task)}"
                        )

                initial_tasks = converted_tasks

            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(
                    f"[STEERING] Failed to parse initial plan: {str(e)[:200]}"
                )
                # Fallback to basic plan
                initial_tasks = [
                    {
                        "description": f"Research comprehensive information about {self.research_topic}",
                        "type": "research",
                    }
                ]

            # Create SteeringTask objects for initial plan
            for i, task_data in enumerate(initial_tasks):
                task_id = f"initial_{i+1}_{str(uuid.uuid4())[:8]}"

                task = SteeringTask(
                    id=task_id,
                    description=task_data.get(
                        "description", f"Research {self.research_topic}"
                    ),
                    status=TaskStatus.PENDING,
                    priority=9,  # Fixed: original_query tasks always priority 9
                    source="original_query",
                    created_from_message=f"Initial query: {initial_query}",
                    search_queries=[],
                )

                self.tasks[task_id] = task
                log_steering(f"[STEERING] Created initial task: {task.description}")

            self.initial_plan_created = True
            self.todo_version += 1
            self.last_updated = datetime.now()

            logger.info(
                f"[STEERING] Created initial plan with {len(initial_tasks)} tasks"
            )

        except Exception as e:
            logger.error(f"[STEERING] Error creating initial plan: {e}")
            # Create fallback task
            fallback_task = SteeringTask(
                id=f"initial_fallback_{str(uuid.uuid4())[:8]}",
                description=f"Research comprehensive information about {self.research_topic}",
                status=TaskStatus.PENDING,
                priority=9,  # Fixed: original_query tasks always priority 9
                source="original_query",
                created_from_message=f"Initial query: {initial_query}",
                search_queries=[],
            )
            self.tasks[fallback_task.id] = fallback_task
            self.initial_plan_created = True
            self.todo_version += 1
            self.last_updated = datetime.now()

    async def add_user_message(self, message: str):
        """Add user steering message to queue (like Cursor's message queuing)"""
        self.pending_messages.append(message)
        self.all_user_messages.append(message)  # Track ALL messages for final report
        log_steering(
            f"[STEERING] Queued user message: '{message}' (queue size: {len(self.pending_messages)})"
        )

        # Don't immediately process - let the research loop handle it
        # This mimics Cursor's behavior of queuing messages

    async def _process_pending_messages(self):
        """Convert pending user messages into actionable tasks with smart summarization"""
        if not self.pending_messages:
            return

        logger.info(
            f"[STEERING] Processing {len(self.pending_messages)} pending messages"
        )

        # If we have multiple messages, first summarize and consolidate them
        if len(self.pending_messages) > 1:
            await self._summarize_and_consolidate_messages()
        else:
            # Single message - process directly
            message = self.pending_messages[0]
            tasks = await self._message_to_tasks(message)
            for task in tasks:
                self.tasks[task.id] = task
                log_steering(f"[STEERING] Created new task: {task.description}")

        log_steering(
            f"[STEERING] Cleared message queue ({len(self.pending_messages)} messages processed)"
        )
        self.pending_messages.clear()
        self.todo_version += 1
        self.last_updated = datetime.now()

    async def _summarize_and_consolidate_messages(self):
        """Summarize multiple messages and create consolidated tasks"""
        try:
            from llm_clients import get_async_llm_client
            from src.configuration import Configuration

            # Use configured LLM provider instead of hardcoded Google
            config = Configuration.from_runnable_config()
            provider = config.activity_llm_provider.value if hasattr(config.activity_llm_provider, 'value') else config.activity_llm_provider
            model = config.activity_llm_model

            llm = await get_async_llm_client(provider, model)
            # Set max_output_tokens
            if hasattr(llm, "max_output_tokens"):
                llm.max_output_tokens = 800
            elif hasattr(llm, "max_tokens"):
                llm.max_tokens = 800

            # Prepare messages for summarization
            messages_text = "\n".join([f"- {msg}" for msg in self.pending_messages])

            current_tasks_text = ""
            if self.tasks:
                current_tasks = [
                    f"- {t.description}"
                    for t in self.tasks.values()
                    if t.status == TaskStatus.PENDING
                ]
                current_tasks_text = f"\n\nCURRENT PENDING TASKS:\n" + "\n".join(
                    current_tasks[:10]
                )

            consolidation_prompt = f"""
You are managing a research todo.md system. The user has sent multiple steering messages that need to be consolidated into actionable tasks.

RESEARCH TOPIC: {self.research_topic}

USER MESSAGES TO PROCESS:
{messages_text}
{current_tasks_text}

Your job is to:
1. Summarize the user's intent across all messages
2. Identify any conflicting instructions and resolve them
3. Update existing tasks or create new ones as needed
4. Mark any existing tasks as cancelled if the user changed their mind

Return a JSON object with:
- "summary": Brief summary of what the user wants
- "actions": Array of actions to take, each with:
  - "type": "create_task", "update_task", "cancel_task", or "no_action"
  - "task_id": (for update/cancel actions, use existing task descriptions to match)
  - "description": Task description (for create/update)
  - "priority": 1-10 priority (for create/update)
  - "reasoning": Why this action is needed

Example response:
{{
  "summary": "User wants to focus only on academic work and exclude personal information",
  "actions": [
    {{"type": "create_task", "description": "Focus research only on academic work and publications", "priority": 9, "reasoning": "User emphasized academic focus"}},
    {{"type": "create_task", "description": "Exclude personal life and biographical information", "priority": 8, "reasoning": "User wants to avoid personal details"}},
    {{"type": "cancel_task", "task_id": "Research comprehensive information about X", "reasoning": "Too broad given new focus constraints"}}
  ]
}}

Return ONLY the JSON object, no other text.
"""

            messages = [{"role": "user", "content": consolidation_prompt}]
            response = await llm.ainvoke(messages)

            import json

            try:
                response_text = response.content.strip()
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]

                consolidation_result = json.loads(response_text)

                logger.info(
                    f"[STEERING] Message consolidation summary: {consolidation_result.get('summary', 'No summary')}"
                )

                # Execute the actions
                for action in consolidation_result.get("actions", []):
                    await self._execute_consolidation_action(action)

            except json.JSONDecodeError:
                logger.warning(
                    f"[STEERING] Failed to parse consolidation response: {response.content}"
                )
                # Fallback - process messages individually
                for message in self.pending_messages:
                    tasks = await self._message_to_tasks(message)
                    for task in tasks:
                        self.tasks[task.id] = task
                        logger.info(
                            f"[STEERING] Created fallback task: {task.description}"
                        )

        except Exception as e:
            logger.error(f"[STEERING] Error consolidating messages: {e}")
            # Fallback - process messages individually
            for message in self.pending_messages:
                tasks = await self._message_to_tasks(message)
                for task in tasks:
                    self.tasks[task.id] = task
                    logger.info(f"[STEERING] Created fallback task: {task.description}")

    async def _execute_consolidation_action(self, action: Dict[str, Any]):
        """Execute a consolidation action from message summarization"""
        action_type = action.get("type")

        if action_type == "create_task":
            task_id = f"consolidated_{str(uuid.uuid4())[:8]}"
            task = SteeringTask(
                id=task_id,
                description=action.get(
                    "description", "Task from consolidated messages"
                ),
                status=TaskStatus.PENDING,
                priority=min(10, max(1, action.get("priority", 5))),
                created_from_message="Consolidated from multiple messages",
                search_queries=[],
            )
            self.tasks[task_id] = task
            logger.info(f"[STEERING] Created consolidated task: {task.description}")

        elif action_type == "cancel_task":
            # Find task to cancel by matching description
            task_to_cancel = action.get("task_id", "")
            for task_id, task in self.tasks.items():
                if (
                    task_to_cancel.lower() in task.description.lower()
                    and task.status == TaskStatus.PENDING
                ):
                    task.status = TaskStatus.CANCELLED
                    logger.info(
                        f"[STEERING] Cancelled task: {task.description} - {action.get('reasoning', '')}"
                    )
                    break

        elif action_type == "update_task":
            # Find and update existing task
            task_to_update = action.get("task_id", "")
            for task_id, task in self.tasks.items():
                if (
                    task_to_update.lower() in task.description.lower()
                    and task.status == TaskStatus.PENDING
                ):
                    task.description = action.get("description", task.description)
                    task.priority = min(
                        10, max(1, action.get("priority", task.priority))
                    )
                    logger.info(f"[STEERING] Updated task: {task.description}")
                    break

    async def prepare_for_next_loop(
        self,
        research_loop_count: int,
        previous_summary: str = "",
        current_results: str = "",
    ):
        """
        Process queued messages and update todo.md before next research loop.
        This is the key method that mimics Cursor's behavior.
        """
        self.research_loop_count = research_loop_count

        logger.info(f"[STEERING] Preparing for research loop {research_loop_count}")
        logger.info(
            f"[STEERING] Pending messages to process: {len(self.pending_messages)}"
        )

        # Process any queued steering messages
        if self.pending_messages:
            logger.info(
                f"[STEERING] Processing {len(self.pending_messages)} queued messages..."
            )
            old_version = self.todo_version
            await self._process_pending_messages()
            logger.info(
                f"[STEERING] Messages processed, todo.md updated from version {old_version} to {self.todo_version}"
            )
            logger.info(
                f"[STEERING] Current tasks: {len(self.get_pending_tasks())} pending, {len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])} completed"
            )

        # Mark completed tasks based on research progress
        if previous_summary and research_loop_count > 1:
            await self._mark_completed_tasks_based_on_progress(
                previous_summary, current_results
            )

        # Log current todo state
        pending_tasks = self.get_pending_tasks()
        completed_tasks = [
            t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED
        ]

        logger.info(f"[STEERING] Todo state for loop {research_loop_count}:")
        logger.info(f"  - Pending tasks: {len(pending_tasks)}")
        logger.info(f"  - Completed tasks: {len(completed_tasks)}")
        logger.info(f"  - Todo version: {self.todo_version}")

        return self.get_todo_md()

    async def _mark_completed_tasks_based_on_progress(
        self, previous_summary: str, current_results: str
    ):
        """Mark tasks as completed based on research progress"""
        try:
            from llm_clients import get_async_llm_client
            from src.configuration import Configuration

            # Use configured LLM provider instead of hardcoded Google
            config = Configuration.from_runnable_config()
            provider = config.activity_llm_provider.value if hasattr(config.activity_llm_provider, 'value') else config.activity_llm_provider
            model = config.activity_llm_model

            llm = await get_async_llm_client(provider, model)
            # Set max_output_tokens
            if hasattr(llm, "max_output_tokens"):
                llm.max_output_tokens = 600
            elif hasattr(llm, "max_tokens"):
                llm.max_tokens = 600

            pending_tasks = self.get_pending_tasks()
            if not pending_tasks:
                return

            # Create task completion analysis prompt
            tasks_text = "\n".join(
                [f"- {t.id}: {t.description}" for t in pending_tasks[:10]]
            )  # Limit to 10 tasks

            completion_prompt = f"""
Based on the research progress, determine which tasks have been completed.

PENDING TASKS:
{tasks_text}

RESEARCH PROGRESS:
Previous Summary: {previous_summary[:1000]}...

Current Results: {current_results[:1000]}...

Return a JSON array of task IDs that have been completed based on the research progress:
["task_id_1", "task_id_2", ...]

If no tasks are completed, return an empty array: []

Only mark tasks as completed if there's clear evidence they've been addressed in the research.
Return ONLY the JSON array, no other text.
"""

            messages = [{"role": "user", "content": completion_prompt}]
            response = await llm.ainvoke(messages)

            import json

            try:
                completed_task_ids = json.loads(response.content.strip())
                if not isinstance(completed_task_ids, list):
                    completed_task_ids = []

                for task_id in completed_task_ids:
                    if task_id in self.tasks:
                        self.mark_task_completed(task_id)
                        logger.info(
                            f"[STEERING] Marked task as completed: {self.tasks[task_id].description}"
                        )

            except json.JSONDecodeError:
                logger.warning(
                    f"[STEERING] Failed to parse task completion response: {response.content}"
                )

        except Exception as e:
            logger.error(f"[STEERING] Error marking completed tasks: {e}")

    def get_current_loop_guidance(self) -> str:
        """Get guidance for the current research loop based on active tasks"""
        pending_tasks = self.get_pending_tasks()
        if not pending_tasks:
            return "Continue comprehensive research on the topic."

        # Sort by priority
        pending_tasks.sort(key=lambda t: t.priority, reverse=True)

        guidance_parts = []
        guidance_parts.append(
            f"**Research Loop {self.research_loop_count} Objectives:**"
        )

        for i, task in enumerate(pending_tasks[:5], 1):  # Top 5 tasks
            guidance_parts.append(
                f"{i}. {task.description} (Priority: {task.priority})"
            )

        if len(pending_tasks) > 5:
            guidance_parts.append(f"... and {len(pending_tasks) - 5} more tasks")

        return "\n".join(guidance_parts)

    async def _message_to_tasks(self, message: str) -> List[SteeringTask]:
        """Convert user message to actionable research tasks using LLM parsing"""
        try:
            # Use LLM to intelligently parse the user message into structured tasks
            from llm_clients import get_async_llm_client
            from src.configuration import Configuration

            # Use configured LLM provider instead of hardcoded Google
            config = Configuration.from_runnable_config()
            provider = config.activity_llm_provider.value if hasattr(config.activity_llm_provider, 'value') else config.activity_llm_provider
            model = config.activity_llm_model

            # Get LLM client (use a lightweight model for parsing)
            llm = await get_async_llm_client(provider, model)

            # Set max_output_tokens
            if hasattr(llm, "max_output_tokens"):
                llm.max_output_tokens = 500
            elif hasattr(llm, "max_tokens"):
                llm.max_tokens = 500

            parsing_prompt = f"""
You are helping to parse a user's steering message for a research system. The user has sent a message to guide ongoing research about "{self.research_topic}".

User Message: "{message}"

Parse this message and create appropriate research steering tasks. Return a JSON array of tasks, where each task has:
- "type": one of ["focus", "exclude", "prioritize", "stop_searching", "guidance"]  
- "description": a clear, actionable description of what the research should do
- "priority": integer 1-10 (higher = more important)
- "subject": the main subject/topic this task relates to (lowercase)

Guidelines for task types:
- "focus" = narrow research to specific areas/topics only
- "exclude" = avoid certain topics/areas completely
- "prioritize" = give higher priority to certain topics (but don't exclude others)
- "stop_searching" = halt searches for specific topics immediately
- "guidance" = general research direction/advice that doesn't fit other categories

Examples:
- "Focus on Stanford University work" → [{{"type": "focus", "description": "Focus research only on Stanford University work", "priority": 8, "subject": "stanford university work"}}]
- "Exclude personal life information" → [{{"type": "exclude", "description": "Exclude personal life information from research", "priority": 7, "subject": "personal life"}}]
- "Prioritize AI research" → [{{"type": "prioritize", "description": "Prioritize AI research topics", "priority": 8, "subject": "ai research"}}]
- "Stop looking at entertainment stuff" → [{{"type": "stop_searching", "description": "Stop searching for entertainment-related information", "priority": 9, "subject": "entertainment"}}]

IMPORTANT: Return ONLY the JSON array, no other text or explanation.
"""

            # Get LLM response
            messages = [{"role": "user", "content": parsing_prompt}]
            response = await llm.ainvoke(messages)

            # Parse the response
            import json

            try:
                response_text = response.content.strip()
                # Clean up response in case there's extra text
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]

                parsed_tasks = json.loads(response_text)
                if not isinstance(parsed_tasks, list):
                    parsed_tasks = [parsed_tasks]

            except json.JSONDecodeError as e:
                logger.warning(
                    f"[STEERING] Failed to parse LLM response as JSON: {response.content}"
                )
                # Fallback to simple task
                parsed_tasks = [
                    {
                        "type": "guidance",
                        "description": f"Research guidance: {message}",
                        "priority": 5,
                        "subject": message.lower(),
                    }
                ]

            # Convert parsed tasks to SteeringTask objects
            tasks = []
            for i, task_data in enumerate(parsed_tasks):
                task_id = f"{task_data.get('type', 'task')}_{str(uuid.uuid4())[:8]}"

                task = SteeringTask(
                    id=task_id,
                    description=task_data.get("description", f"Task: {message}"),
                    status=TaskStatus.PENDING,
                    priority=min(
                        10, max(1, task_data.get("priority", 5))
                    ),  # Clamp to 1-10
                    created_from_message=message,
                    search_queries=[],
                )

                tasks.append(task)
                logger.info(
                    f"[STEERING] Created {task_data.get('type', 'task')} task: {task.description}"
                )

            return tasks

        except Exception as e:
            logger.error(f"[STEERING] Error parsing message with LLM: {e}")
            # Fallback to simple task creation
            fallback_task = SteeringTask(
                id=f"guidance_{str(uuid.uuid4())[:8]}",
                description=f"Research guidance: {message}",
                status=TaskStatus.PENDING,
                priority=5,
                created_from_message=message,
                search_queries=[],
            )
            return [fallback_task]

    def _extract_focus_target(self, message: str) -> str:
        """Extract what user wants to focus on"""
        # Simple extraction - could be enhanced
        patterns = [
            r"focus.*?on\s+(.+?)(?:\.|$|,)",
            r"only\s+(.+?)(?:\.|$|,)",
            r"specifically\s+(.+?)(?:\.|$|,)",
        ]

        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                return match.group(1).strip()

        # Fallback - take everything after focus/only
        for word in ["focus on", "only", "specifically"]:
            if word in message.lower():
                parts = message.lower().split(word, 1)
                if len(parts) > 1:
                    return parts[1].strip(" .,")

        return ""

    def _extract_exclude_target(self, message: str) -> str:
        """Extract what user wants to exclude"""
        patterns = [
            r"exclude\s+(.+?)(?:\.|$|,)",
            r"ignore\s+(.+?)(?:\.|$|,)",
            r"not\s+(.+?)(?:\.|$|,)",
            r"skip\s+(.+?)(?:\.|$|,)",
        ]

        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                return match.group(1).strip()

        return ""

    def _extract_priority_target(self, message: str) -> str:
        """Extract what user wants to prioritize"""
        patterns = [
            r"prioritize\s+(.+?)(?:\.|$|,)",
            r"emphasize\s+(.+?)(?:\.|$|,)",
            r"important\s+(.+?)(?:\.|$|,)",
        ]

        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                return match.group(1).strip()

        return ""

    def _extract_stop_target(self, message: str) -> str:
        """Extract what user wants to stop"""
        patterns = [
            r"stop.*?(?:searching for|looking for|finding)\s+(.+?)(?:\.|$|,)",
            r"halt\s+(.+?)(?:\.|$|,)",
            r"cancel\s+(.+?)(?:\.|$|,)",
        ]

        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                return match.group(1).strip()

        return ""

    def get_pending_tasks(self, sorted_by_priority=True) -> List[SteeringTask]:
        """Get tasks that need to be executed, sorted by priority (highest first)"""
        pending = [
            task for task in self.tasks.values() if task.status == TaskStatus.PENDING
        ]
        if sorted_by_priority:
            pending.sort(key=lambda t: t.priority, reverse=True)
        return pending

    def mark_task_completed(self, task_id: str, completion_note: str = ""):
        """Mark task as completed (idempotent - safe to call multiple times)"""
        if task_id in self.tasks:
            task = self.tasks[task_id]

            # If already completed, just update the note if provided
            if task.status == TaskStatus.COMPLETED:
                if completion_note and completion_note != task.completed_note:
                    task.completed_note = completion_note
                    logger.debug(
                        f"[STEERING] Task {task_id} already completed, updated note"
                    )
                return

            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.completed_note = completion_note
            self.todo_version += 1
            self.last_updated = datetime.now()
            log_steering(f"[STEERING] Completed task: {task.description}")

    def mark_task_cancelled(self, task_id: str, reason: str = ""):
        """Mark task as cancelled"""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.CANCELLED
            self.tasks[task_id].completed_at = datetime.now()
            self.tasks[task_id].completed_note = (
                f"Cancelled: {reason}" if reason else "Cancelled"
            )
            self.todo_version += 1
            self.last_updated = datetime.now()
            log_steering(f"[STEERING] Cancelled task: {self.tasks[task_id].description}")

    def mark_task_in_progress(self, task_id: str):
        """Mark a task as in progress (only if not already completed/cancelled)"""
        if task_id in self.tasks:
            task = self.tasks[task_id]

            if task.status == TaskStatus.COMPLETED:
                logger.debug(
                    f"[STEERING] Task {task_id} already completed, skipping IN_PROGRESS"
                )
                return

            if task.status == TaskStatus.CANCELLED:
                logger.debug(
                    f"[STEERING] Task {task_id} was cancelled, skipping IN_PROGRESS"
                )
                return

            # Only mark as in-progress if it's PENDING
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.IN_PROGRESS
                log_steering(f"[STEERING] Task started: {task.description}")

    def create_task(
        self,
        description: str,
        priority: int = 5,
        source: str = "knowledge_gap",
        search_queries: List[str] = None,
        created_from_message: str = "",
    ):
        """Create a new steering task (with FUZZY duplicate detection)"""

        # Normalize description for comparison
        normalized_desc = description.lower().strip()

        # Remove common prefixes to improve matching
        # (e.g., "Research: X" and "Complete research on: X" should match)
        normalized_desc = re.sub(
            r"^(research:|complete research on:|analyze:|investigate:)\s*",
            "",
            normalized_desc,
        )

        # Check for duplicates (exact + fuzzy matching)
        import difflib

        for existing_task in self.tasks.values():
            if existing_task.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
                # Normalize existing task description
                existing_normalized = existing_task.description.lower().strip()
                existing_normalized = re.sub(
                    r"^(research:|complete research on:|analyze:|investigate:)\s*",
                    "",
                    existing_normalized,
                )

                # Calculate similarity
                similarity = difflib.SequenceMatcher(
                    None, normalized_desc, existing_normalized
                ).ratio()

                # Exact match check
                if normalized_desc == existing_normalized:
                    log_steering(
                        f"[WARN] Exact duplicate: '{description}' → Merged with {existing_task.id}"
                    )
                    # Update priority if the new one is higher
                    if priority > existing_task.priority:
                        existing_task.priority = priority
                    return existing_task.id

                # Fuzzy similarity check (70% threshold)
                if similarity > 0.70:
                    log_steering(
                        f"[WARN] Similar task ({similarity:.0%}): '{description}' → Merged with {existing_task.id}"
                    )
                    # Update priority if the new one is higher
                    if priority > existing_task.priority:
                        existing_task.priority = priority
                    return existing_task.id

        # No duplicate found - create new task
        task_id = f"task_{len(self.tasks) + 1}_{int(datetime.now().timestamp())}"
        task = SteeringTask(
            id=task_id,
            description=description,
            priority=priority,
            source=source,
            search_queries=search_queries or [],
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            created_from_message=created_from_message,
        )
        self.tasks[task_id] = task
        log_steering(f"[STEERING] Created task: {description} (source: {source})")
        return task_id

    def get_completed_tasks(self, limit: int = None) -> List[SteeringTask]:
        """Get all completed tasks, sorted by completion time (most recent first)"""
        completed = [
            task for task in self.tasks.values() if task.status == TaskStatus.COMPLETED
        ]
        # Sort by completion time, most recent first
        completed.sort(key=lambda t: t.completed_at or t.created_at, reverse=True)
        return completed[:limit] if limit else completed

    def get_pending_tasks_for_llm(self) -> str:
        """
        Format ONLY pending tasks for LLM reflection to decide completion.
        Clean, focused format - only what needs evaluation.
        """
        pending = self.get_pending_tasks(sorted_by_priority=True)
        if not pending:
            return "No pending tasks - all tasks completed or cancelled."

        lines = ["PENDING TASKS TO EVALUATE FOR COMPLETION:"]
        lines.append(
            "(Only mark as completed if research clearly addressed this task)\n"
        )
        for task in pending:
            lines.append(f"- [{task.id}] (P{task.priority}) {task.description}")

        return "\n".join(lines)

    def get_completed_tasks_for_llm(self, limit: int = 10) -> str:
        """
        Format completed AND cancelled tasks for LLM when creating NEW tasks.
        Shows what's already done/cancelled to avoid duplicates.
        """
        completed = self.get_completed_tasks(limit=limit)
        cancelled = [t for t in self.tasks.values() if t.status == TaskStatus.CANCELLED]
        cancelled.sort(key=lambda t: t.completed_at or t.created_at, reverse=True)
        if limit:
            cancelled = cancelled[:limit]

        if not completed and not cancelled:
            return "No tasks completed or cancelled yet - just starting research."

        lines = ["ALREADY COMPLETED OR CANCELLED (do NOT create duplicate tasks):"]

        for task in completed:
            lines.append(f"[done] COMPLETED: {task.description}")

        for task in cancelled:
            lines.append(f"[cancelled] CANCELLED: {task.description}")

        return "\n".join(lines)

    def update_task_priority(self, task_id: str, new_priority: int):
        """Update task priority"""
        if task_id in self.tasks:
            self.tasks[task_id].priority = new_priority
            self.last_updated = datetime.now()
            logger.info(
                f"[STEERING] Updated priority for {self.tasks[task_id].description}: P{new_priority}"
            )

    def extract_search_query_from_task(self, task_description: str) -> str:
        """Extract a searchable query from a task description"""
        # Remove common task prefixes
        desc = task_description.lower()
        prefixes = [
            "research:",
            "investigate:",
            "find:",
            "search for:",
            "look into:",
            "explore:",
            "study:",
        ]
        for prefix in prefixes:
            if desc.startswith(prefix):
                desc = desc[len(prefix) :].strip()
                break

        # Remove emoji and special markdown
        desc = re.sub(r"[^\w\s-]", " ", desc)
        desc = " ".join(desc.split())  # Normalize whitespace

        return desc.strip()

    def get_todo_md(self) -> str:
        """Generate todo.md content like Claude Code"""
        md_content = f"# Research Steering Plan (v{self.todo_version})\n\n"
        md_content += f"**Topic:** {self.research_topic}\n"
        md_content += (
            f"**Last Updated:** {self.last_updated.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        # Pending tasks (high priority first)
        pending_tasks = [
            t for t in self.tasks.values() if t.status == TaskStatus.PENDING
        ]
        if pending_tasks:
            md_content += "## Pending\n\n"
            pending_tasks.sort(key=lambda t: t.priority, reverse=True)
            for idx, task in enumerate(pending_tasks, 1):
                # Add source emoji
                source_emoji = {
                    "steering_message": "[STEER]",
                    "original_query": "[QUERY]",
                    "knowledge_gap": "[GAP]",
                }.get(task.source, "")

                md_content += f"- [ ] **[{idx}]** {source_emoji} {task.description}\n"
                md_content += f'  - *From user:* "{task.created_from_message}"\n'
                md_content += (
                    f"  - *Created:* {task.created_at.strftime('%H:%M:%S')}\n\n"
                )

        # In progress tasks
        active_tasks = [
            t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS
        ]
        if active_tasks:
            md_content += "## Currently Processing\n\n"
            for idx, task in enumerate(active_tasks, 1):
                # Add source emoji
                source_emoji = {
                    "steering_message": "[STEER]",
                    "original_query": "[QUERY]",
                    "knowledge_gap": "[GAP]",
                }.get(task.source, "")

                md_content += f"- [ ] **[{idx}]** {source_emoji} {task.description}\n"
                md_content += f'  - *From user:* "{task.created_from_message}"\n\n'

        # Completed tasks (most recent 5)
        completed_tasks = self.get_completed_tasks(
            limit=5
        )  # Already sorted by completion time
        if completed_tasks:
            md_content += "## Recently Completed\n\n"
            for idx, task in enumerate(completed_tasks, 1):
                # Add source emoji
                source_emoji = {
                    "steering_message": "[STEER]",
                    "original_query": "[QUERY]",
                    "knowledge_gap": "[GAP]",
                }.get(task.source, "")

                md_content += f"- [x] **[{idx}]** {source_emoji} {task.description}\n"
                if task.completed_note:
                    md_content += f"  - *{task.completed_note}*\n"
                if task.completed_at:
                    md_content += f"  - *Completed at:* {task.completed_at.strftime('%H:%M:%S')}\n"
                md_content += "\n"

        # Cancelled tasks (last 3)
        cancelled_tasks = [
            t for t in self.tasks.values() if t.status == TaskStatus.CANCELLED
        ]
        if cancelled_tasks:
            md_content += "## Cancelled\n\n"
            for idx, task in enumerate(cancelled_tasks[-3:], 1):  # Last 3
                # Add source emoji
                source_emoji = {
                    "steering_message": "[STEER]",
                    "original_query": "[QUERY]",
                    "knowledge_gap": "[GAP]",
                }.get(task.source, "")

                md_content += f"- [~] **[{idx}]** {source_emoji} {task.description}\n"
                if task.completed_note:
                    md_content += f"  - *Reason:* {task.completed_note}\n\n"

        if not any([pending_tasks, active_tasks, completed_tasks, cancelled_tasks]):
            md_content += "## No steering instructions yet\n\n"
            md_content += (
                "User can send messages during research to guide the process.\n\n"
            )

        return md_content

    def get_current_constraints(self) -> Dict[str, List[str]]:
        """Get current active constraints for the research agent"""
        constraints = {
            "focus_on": [],
            "exclude": [],
            "prioritize": [],
            "stop_searching": [],
        }

        for task in self.get_pending_tasks():
            desc_lower = task.description.lower()
            task_id_lower = task.id.lower()

            # Use both task ID and description to determine constraint type
            if (
                task_id_lower.startswith("focus_")
                or "focus research only on:" in desc_lower
                or "focus" in desc_lower
                and ("only" in desc_lower or "specifically" in desc_lower)
            ):

                if ":" in task.description:
                    target = task.description.split(":", 1)[1].strip()
                else:
                    # Extract focus target from natural language
                    target = (
                        desc_lower.replace("focus research", "")
                        .replace("only on", "")
                        .replace("specifically on", "")
                        .strip()
                    )

                if target:
                    constraints["focus_on"].append(target)

            elif (
                task_id_lower.startswith("exclude_")
                or "exclude from research:" in desc_lower
                or "exclude" in desc_lower
            ):

                if ":" in task.description:
                    target = task.description.split(":", 1)[1].strip()
                else:
                    # Extract exclusion target from natural language
                    target = (
                        desc_lower.replace("exclude", "")
                        .replace("from research", "")
                        .strip()
                    )

                if target:
                    constraints["exclude"].append(target)

            elif (
                task_id_lower.startswith("prioritize_")
                or "prioritize research on:" in desc_lower
                or "prioritize" in desc_lower
            ):

                if ":" in task.description:
                    target = task.description.split(":", 1)[1].strip()
                else:
                    # Extract priority target from natural language
                    target = (
                        desc_lower.replace("prioritize", "")
                        .replace("research on", "")
                        .strip()
                    )

                if target:
                    constraints["prioritize"].append(target)

            elif (
                task_id_lower.startswith("stop_searching_")
                or "stop searching for:" in desc_lower
                or ("stop" in desc_lower and "search" in desc_lower)
            ):

                if ":" in task.description:
                    target = task.description.split(":", 1)[1].strip()
                else:
                    # Extract stop target from natural language
                    target = (
                        desc_lower.replace("stop searching", "")
                        .replace("for", "")
                        .strip()
                    )

                if target:
                    constraints["stop_searching"].append(target)

        return constraints

    def should_cancel_search(self, search_query: str) -> bool:
        """Check if a search query should be cancelled based on constraints"""
        constraints = self.get_current_constraints()
        search_lower = search_query.lower()

        # Check exclusions
        for exclude_term in constraints["exclude"]:
            if exclude_term.lower() in search_lower:
                logger.info(
                    f"[STEERING] Cancelling search '{search_query}' - matches exclusion: {exclude_term}"
                )
                return True

        # Check focus constraints
        if constraints["focus_on"]:
            # If there are focus constraints, cancel searches that don't match any
            matches_focus = any(
                focus_term.lower() in search_lower
                for focus_term in constraints["focus_on"]
            )
            if not matches_focus:
                logger.info(
                    f"[STEERING] Cancelling search '{search_query}' - doesn't match focus constraints"
                )
                return True

        # Check stop instructions
        for stop_term in constraints["stop_searching"]:
            if stop_term.lower() in search_lower:
                logger.info(
                    f"[STEERING] Cancelling search '{search_query}' - matches stop instruction: {stop_term}"
                )
                return True

        return False

    def get_search_priority_boost(self, search_query: str) -> int:
        """Get priority boost for search query based on constraints"""
        constraints = self.get_current_constraints()
        search_lower = search_query.lower()
        boost = 0

        # Boost for priority terms
        for priority_term in constraints["prioritize"]:
            if priority_term.lower() in search_lower:
                boost += 3
                logger.info(
                    f"[STEERING] Boosting priority for '{search_query}' - matches priority: {priority_term}"
                )

        # Boost for focus terms
        for focus_term in constraints["focus_on"]:
            if focus_term.lower() in search_lower:
                boost += 2
                logger.info(
                    f"[STEERING] Boosting priority for '{search_query}' - matches focus: {focus_term}"
                )

        return boost

    def is_query_duplicate(self, query: str) -> bool:
        """
        Check if a query is a duplicate or very similar to previously executed queries.
        Uses fuzzy matching to catch semantic duplicates.
        """
        if not query or not query.strip():
            return True

        query_normalized = query.lower().strip()

        # Exact match check
        if query_normalized in self.executed_queries:
            log_steering(f"[DEDUP] Skipping duplicate query: '{query}'", level="info")
            return True

        # Fuzzy similarity check for near-duplicates
        import difflib

        for executed_query in self.executed_queries:
            similarity = difflib.SequenceMatcher(
                None, query_normalized, executed_query
            ).ratio()
            if similarity > 0.85:  # 85% similarity threshold
                log_steering(
                    f"[DEDUP] Skipping similar query: '{query}' (similar to: '{executed_query}')",
                    level="info",
                )
                return True

        return False

    def mark_query_executed(self, query: str):
        """Mark a query as executed to prevent future duplicates"""
        if query and query.strip():
            query_normalized = query.lower().strip()
            self.executed_queries.add(query_normalized)
            logger.debug(f"[STEERING] Marked query as executed: '{query}'")

    def filter_duplicate_queries(self, queries: List[str]) -> List[str]:
        """
        Filter out duplicate queries from a list.
        Returns only unique queries that haven't been executed before.
        """
        unique_queries = []
        for query in queries:
            if not self.is_query_duplicate(query):
                unique_queries.append(query)
                self.mark_query_executed(query)
            else:
                log_steering(f"[DEDUP] Filtered out duplicate: '{query}'")

        if len(unique_queries) < len(queries):
            log_steering(
                f"[DEDUP] Query deduplication: {len(queries)} → {len(unique_queries)} (removed {len(queries) - len(unique_queries)} duplicates)"
            )

        return unique_queries


# Integration with existing SummaryState
def extend_summary_state_with_steering(state_class):
    """Add steering capability to existing SummaryState"""

    def __init__(self, **data):
        # Call original init
        original_init = state_class.__init__
        original_init(self, **data)

        # Add steering manager
        if data.get("steering_enabled", False):
            topic = data.get("research_topic", "Research")
            self.steering_todo = ResearchTodoManager(topic)
            logger.info(f"[STEERING] Initialized todo manager for: {topic}")
        else:
            self.steering_todo = None

    async def add_steering_message(self, message: str) -> Dict[str, Any]:
        """Add steering message from user"""
        if not self.steering_todo:
            raise ValueError("Steering not enabled for this research session")

        await self.steering_todo.add_user_message(message)

        return {
            "message_processed": True,
            "todo_version": self.steering_todo.todo_version,
            "pending_tasks": len(self.steering_todo.get_pending_tasks()),
            "current_plan": self.steering_todo.get_todo_md(),
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

    # Add methods to state class
    state_class.__init__ = __init__
    state_class.add_steering_message = add_steering_message
    state_class.get_steering_plan = get_steering_plan
    state_class.should_cancel_search = should_cancel_search
    state_class.get_search_priority_boost = get_search_priority_boost

    return state_class
