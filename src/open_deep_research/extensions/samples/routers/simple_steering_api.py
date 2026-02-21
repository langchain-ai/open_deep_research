"""
Simplified Steering API - User just sends natural messages
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/steering", tags=["steering"])

# Simple session registry
active_research_sessions: Dict[str, Dict[str, Any]] = {}

# File-based session store for cross-process sharing
try:
    from session_store import session_store

    USE_FILE_STORE = True
    logger.info("[STEERING] Using file-based session store")
except ImportError:
    USE_FILE_STORE = False
    logger.warning("[STEERING] File-based session store not available")


class SteeringMessage(BaseModel):
    """Simple steering message - user just sends natural text"""

    session_id: str = Field(description="Research session ID")
    message: str = Field(description="Natural language steering message from user")


class SteeringResponse(BaseModel):
    """Response to steering message"""

    success: bool
    message: str
    session_id: str
    todo_updated: bool = False
    current_plan: Optional[str] = None
    tasks_created: int = 0


class SessionPlan(BaseModel):
    """Current research plan"""

    session_id: str
    plan: str
    version: int
    last_updated: str


class PlanStatus(BaseModel):
    """Real-time plan status"""

    session_id: str
    todo_version: int
    pending_tasks: int
    completed_tasks: int
    queued_messages: int
    queued_message_list: Optional[List[str]] = (
        None  # Actual queued message content for UI
    )
    last_updated: str
    research_loop_count: int
    plan_summary: str
    current_plan: Optional[str] = None  # Full markdown plan with task statuses


@router.post("/message", response_model=SteeringResponse)
async def send_steering_message(request: SteeringMessage) -> SteeringResponse:
    """
    Send a natural language steering message during research.

    User just sends what they want:
    - "Focus on the computer scientist"
    - "Exclude entertainment stuff"
    - "Stop looking for movies"
    - "Prioritize recent work"
    """
    # logger.info(
    #     f"[STEERING] Received message for session {request.session_id}: {request.message}"
    # )

    # Check if session exists
    if request.session_id not in active_research_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Research session {request.session_id} not found or not active",
        )

    session_info = active_research_sessions[request.session_id]

    # Get the research state first
    state = session_info.get("state")

    if not state or not hasattr(state, "steering_todo"):
        raise HTTPException(status_code=400, detail="Session does not support steering")

    # Check if research is actually complete
    is_complete = getattr(state, "research_complete", False)
    current_node = session_info.get("current_node", "")
    is_active = session_info.get("active", False)

    try:
        # Reject ONLY if research is fully complete AND session is inactive
        if is_complete and not is_active:
            logger.info(
                f"[STEERING] Message received after research completion. Skipping."
            )
            return SteeringResponse(
                success=False,
                message=f"Research complete, skipping your message. Please start a new research session.",
                session_id=request.session_id,
                todo_updated=False,
                current_plan=None,
                tasks_created=0,
            )

        # If in finalize_report node, queue message but don't interrupt (just log)
        if current_node == "finalize_report":
            logger.info(
                f"[STEERING] Message received during finalization. Queueing but won't interrupt finalization."
            )

        # Add message ONLY to the steering todo manager (single source of truth)
        if state and hasattr(state, "steering_todo") and state.steering_todo:
            await state.steering_todo.add_user_message(request.message)
            logger.info(f"[STEERING] Queued message: {request.message[:50]}...")
        else:
            logger.warning(
                f"[STEERING] Could not add message to todo manager - state: {type(state)}, has_steering_todo: {hasattr(state, 'steering_todo') if state else False}"
            )
            raise HTTPException(
                status_code=500,
                detail="Could not queue message - steering manager not available",
            )

        # Get current plan info
        current_plan = state.get_steering_plan() if state.steering_todo else None
        pending_tasks = (
            len(state.steering_todo.get_pending_tasks()) if state.steering_todo else 0
        )

        return SteeringResponse(
            success=True,
            message=f"Steering message queued for processing (queue size: {len(state.steering_todo.pending_messages)})",
            session_id=request.session_id,
            todo_updated=False,  # Will be updated when processed at loop boundary
            current_plan=current_plan,
            tasks_created=1,  # Message was queued and will create tasks when processed
        )

    except Exception as e:
        logger.error(f"[STEERING] Error processing message: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing steering message: {str(e)}"
        )


@router.get("/plan/{session_id}", response_model=SessionPlan)
async def get_research_plan(session_id: str) -> SessionPlan:
    """Get current research steering plan in todo.md format"""
    # logger.info(f"[STEERING] Getting plan for session {session_id}")

    if session_id not in active_research_sessions:
        raise HTTPException(
            status_code=404, detail=f"Research session {session_id} not found"
        )

    session_info = active_research_sessions[session_id]
    state = session_info.get("state")

    if not state or not hasattr(state, "steering_todo"):
        return SessionPlan(
            session_id=session_id,
            plan="# Steering not enabled\n\nThis research session does not support steering.",
            version=0,
            last_updated=datetime.now().isoformat(),
        )

    plan = state.get_steering_plan()

    return SessionPlan(
        session_id=session_id,
        plan=plan,
        version=state.steering_todo.todo_version,
        last_updated=state.steering_todo.last_updated.isoformat(),
    )


@router.get("/status/{session_id}", response_model=PlanStatus)
async def get_plan_status(session_id: str) -> PlanStatus:
    """Get real-time plan status for UI updates"""
    # logger.info(f"[STEERING] Getting plan status for session {session_id}")

    if session_id not in active_research_sessions:
        raise HTTPException(
            status_code=404, detail=f"Research session {session_id} not found"
        )

    session_info = active_research_sessions[session_id]
    state = session_info.get("state")

    if not state or not hasattr(state, "steering_todo") or not state.steering_todo:
        return PlanStatus(
            session_id=session_id,
            todo_version=0,
            pending_tasks=0,
            completed_tasks=0,
            queued_messages=0,
            last_updated=datetime.now().isoformat(),
            research_loop_count=getattr(state, "research_loop_count", 0),
            plan_summary="Steering not enabled for this session",
        )

    todo_manager = state.steering_todo
    pending_tasks = todo_manager.get_pending_tasks()
    completed_tasks = [
        t for t in todo_manager.tasks.values() if t.status.name == "COMPLETED"
    ]

    # Create a brief plan summary
    plan_summary = f"Research: {todo_manager.research_topic}"
    if pending_tasks:
        plan_summary += f" | {len(pending_tasks)} pending tasks"
    if completed_tasks:
        plan_summary += f" | {len(completed_tasks)} completed"

    # Get the full markdown plan with updated task statuses
    current_plan_markdown = todo_manager.get_markdown()

    # Get list of queued messages for UI display
    queued_message_texts = []
    for msg in todo_manager.pending_messages:
        # Messages can be strings or dicts depending on how they were added
        if isinstance(msg, str):
            queued_message_texts.append(msg)
        elif isinstance(msg, dict):
            queued_message_texts.append(msg.get("message", msg.get("content", "")))
        else:
            queued_message_texts.append(str(msg))

    # Debug logging to track queue clearing
    logger.info(
        f"[STEERING STATUS] Session {session_id}: "
        f"raw_queue_size={len(todo_manager.pending_messages)}, "
        f"parsed_messages={len(queued_message_texts)}, "
        f"todo_version={todo_manager.todo_version}"
    )
    if len(queued_message_texts) > 0:
        logger.info(f"[STEERING STATUS] Queued messages: {queued_message_texts}")

    return PlanStatus(
        session_id=session_id,
        todo_version=todo_manager.todo_version,
        pending_tasks=len(pending_tasks),
        completed_tasks=len(completed_tasks),
        queued_messages=len(todo_manager.pending_messages),
        queued_message_list=queued_message_texts,  # Include actual messages for UI (empty list if cleared)
        last_updated=todo_manager.last_updated.isoformat(),
        research_loop_count=getattr(state, "research_loop_count", 0),
        plan_summary=plan_summary,
        current_plan=current_plan_markdown,  # Include full plan for UI updates
    )


# Alias endpoint for frontend compatibility - frontend expects /interactive/session/{session_id}
@router.get("/interactive/session/{session_id}")
async def get_interactive_session_status(session_id: str) -> Dict[str, Any]:
    """
    Alias endpoint for frontend polling compatibility.
    Frontend expects: /interactive/session/{sessionId}
    Returns todo plan and status for real-time UI updates.
    """
    # logger.info(f"[STEERING] Getting interactive session status for {session_id}")

    if session_id not in active_research_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found or steering not enabled",
        )

    session_info = active_research_sessions[session_id]
    state = session_info.get("state")

    if not state:
        raise HTTPException(
            status_code=404,
            detail=f"Session state not available for {session_id}",
        )

    # Determine session status
    status = "active"
    if hasattr(state, "research_complete") and state.research_complete:
        status = "completed"

    # Get the current plan
    current_plan = None
    if hasattr(state, "get_steering_plan"):
        current_plan = state.get_steering_plan()
        # logger.info(
        #     f"[STEERING] Retrieved plan, length: {len(current_plan) if current_plan else 0}"
        # )
    else:
        logger.warning(f"[STEERING] State has no get_steering_plan method")

    todo_version = 0
    queued_message_list = []
    if hasattr(state, "steering_todo") and state.steering_todo:
        todo_version = state.steering_todo.todo_version
        # Get queued messages for UI display - handle both dict and string formats
        queued_message_list = []
        for msg in state.steering_todo.pending_messages:
            if isinstance(msg, dict):
                queued_message_list.append(msg.get("message", msg.get("content", "")))
            elif isinstance(msg, str):
                queued_message_list.append(msg)
            else:
                queued_message_list.append(str(msg))
        # logger.info(
        #     f"[STEERING] Todo version: {todo_version}, tasks: {len(state.steering_todo.tasks)}"
        # )
    else:
        logger.warning(f"[STEERING] State has no steering_todo")

    # Return format expected by frontend polling (matches researchService.js expectations)
    result = {
        "status": status,
        "current_plan": current_plan,
        "todo_version": todo_version,
        "session_id": session_id,
        "queued_message_list": queued_message_list,  # Add queued messages for UI
    }

    # logger.info(
    #     f"[STEERING] Returning: status={status}, plan_length={len(current_plan) if current_plan else 0}, version={todo_version}"
    # )

    return result


@router.get("/sessions")
async def list_steerable_sessions() -> Dict[str, Any]:
    """List all research sessions that support steering"""
    steerable_sessions = []

    # Get sessions from both memory and file store
    # logger.info(
    #     f"[STEERING] Checking sessions - memory has {len(active_research_sessions)} sessions"
    # )
    all_sessions = active_research_sessions.copy()
    if USE_FILE_STORE:
        file_sessions = session_store.get_all_sessions()
        # logger.info(f"[STEERING] File store has {len(file_sessions)} sessions")
        all_sessions.update(file_sessions)
    # else:
    # logger.info("[STEERING] File store not available")

    # logger.info(f"[STEERING] Total sessions to check: {len(all_sessions)}")

    for session_id, session_info in all_sessions.items():
        state = session_info.get("state")
        steering_enabled = session_info.get("steering_enabled", False)

        # Check if this session supports steering
        # For in-memory sessions: check if state has steering_todo
        # For file store sessions: check steering_enabled flag
        supports_steering = False

        # Initialize variables with defaults
        research_topic = "Unknown"
        todo_version = 1
        pending_tasks = 0
        last_updated = datetime.now().isoformat()

        if steering_enabled:
            if (
                state
                and hasattr(state, "steering_todo")
                and state.steering_todo is not None
            ):
                # In-memory session with full state object
                supports_steering = True
                research_topic = state.steering_todo.research_topic
                todo_version = state.steering_todo.todo_version
                pending_tasks = len(state.steering_todo.get_pending_tasks())
                last_updated = state.steering_todo.last_updated.isoformat()
            else:
                # File store session - use session info or defaults
                supports_steering = True
                research_topic = session_info.get("research_topic", "Unknown")
                todo_version = 1
                pending_tasks = 0
                last_updated = session_info.get(
                    "created_at", datetime.now().isoformat()
                )

        logger.info(
            f"[STEERING] Session {session_id}: steering_enabled={steering_enabled}, supports_steering={supports_steering}"
        )

        if supports_steering:
            steerable_sessions.append(
                {
                    "session_id": session_id,
                    "active": session_info.get("active", False),
                    "research_topic": research_topic,
                    "todo_version": todo_version,
                    "pending_tasks": pending_tasks,
                    "last_updated": last_updated,
                }
            )

    return {
        "steerable_sessions": steerable_sessions,
        "total_sessions": len(steerable_sessions),
        "timestamp": datetime.now().isoformat(),
    }


# Session management functions
def register_research_session(session_id: str, state, steering_enabled: bool = False):
    """Register a research session for potential steering"""
    logger.info(
        f"[STEERING] REGISTERING session {session_id} with steering_enabled={steering_enabled}"
    )
    logger.info(f"[STEERING] State object type: {type(state)}")
    logger.info(
        f"[STEERING] State has steering_todo: {hasattr(state, 'steering_todo') if state else 'No state'}"
    )

    session_info = {
        "session_id": session_id,
        "active": True,
        "state": state,
        "steering_enabled": steering_enabled,
        "research_topic": (
            getattr(state, "research_topic", "Unknown") if state else "Unknown"
        ),
        "created_at": datetime.now().isoformat(),
        "last_activity": datetime.now().isoformat(),
    }

    # Store in memory
    active_research_sessions[session_id] = session_info
    logger.info(
        f"[STEERING] Added to active_research_sessions. Total sessions: {len(active_research_sessions)}"
    )

    # Also store in file for cross-process access
    if USE_FILE_STORE:
        try:
            session_store.add_session(session_id, session_info)
            logger.info(f"[STEERING] Added to file store successfully")
        except Exception as e:
            logger.error(f"[STEERING] Failed to add to file store: {e}")

    # Create initial plan IMMEDIATELY if steering is enabled
    if (
        steering_enabled
        and state
        and hasattr(state, "steering_todo")
        and state.steering_todo
    ):
        try:
            import asyncio

            logger.info(
                f"[STEERING] Creating initial plan immediately for session {session_id}"
            )

            # Use nest_asyncio to handle nested event loops
            try:
                import nest_asyncio

                nest_asyncio.apply()
            except ImportError:
                logger.warning(
                    "[STEERING] nest_asyncio not available - may have issues in nested loops"
                )

            # Try to run in current loop or create new one
            try:
                # Check if we're in an async context
                running_loop = asyncio.get_running_loop()
                # Use run_until_complete with nest_asyncio applied
                logger.info("[STEERING] Running in existing event loop")
                running_loop.run_until_complete(
                    state.steering_todo.create_initial_plan(
                        initial_query=getattr(state, "search_query", "")
                        or getattr(state, "research_topic", ""),
                        research_context="Initial research planning",
                    )
                )
                logger.info(
                    f"[STEERING] Initial plan created with {len(state.steering_todo.tasks)} tasks"
                )
            except RuntimeError:
                # No event loop running - create one
                logger.info("[STEERING] No event loop - creating new one")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        state.steering_todo.create_initial_plan(
                            initial_query=getattr(state, "search_query", "")
                            or getattr(state, "research_topic", ""),
                            research_context="Initial research planning",
                        )
                    )
                    logger.info(
                        f"[STEERING] Initial plan created with {len(state.steering_todo.tasks)} tasks"
                    )
                finally:
                    loop.close()
        except Exception as e:
            logger.error(f"[STEERING] Failed to create initial plan: {e}")
            import traceback

            traceback.print_exc()

    logger.info(
        f"[STEERING] Successfully registered research session {session_id} (steering: {steering_enabled})"
    )


def unregister_research_session(session_id: str):
    """Unregister a research session"""
    if session_id in active_research_sessions:
        active_research_sessions[session_id]["active"] = False
        logger.info(f"[STEERING] Unregistered research session {session_id}")


def update_session_activity(session_id: str):
    """Update last activity time for session"""
    if session_id in active_research_sessions:
        active_research_sessions[session_id][
            "last_activity"
        ] = datetime.now().isoformat()


# Example usage for documentation
EXAMPLE_MESSAGES = [
    "Focus only on John Smith the computer scientist",
    "Exclude any entertainment or movie information",
    "Prioritize his recent research papers",
    "Stop searching for biographical details",
    "Look specifically at his work at Stanford",
    "Ignore anything about acting or Hollywood",
    "Emphasize his AI and machine learning contributions",
    "Cancel searches about his personal life",
]


@router.get("/examples")
async def get_steering_examples():
    """Get examples of natural language steering messages"""
    return {
        "description": "Send natural language messages to steer research in real-time",
        "examples": EXAMPLE_MESSAGES,
        "usage_tips": [
            "Just send what you want in natural language",
            "No need to specify message types or categories",
            "Messages are automatically converted to actionable tasks",
            "Use 'focus on' to narrow research scope",
            "Use 'exclude' or 'ignore' to remove topics",
            "Use 'prioritize' or 'emphasize' to boost importance",
            "Use 'stop' to halt specific searches",
        ],
    }
