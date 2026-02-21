from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends
from sse_starlette.sse import EventSourceResponse
import json
import asyncio
from asyncio import Queue
import logging
from typing import Dict, Any, List, Union, Optional
import datetime
import uuid
import time

from models.research import (
    ResearchRequest,
    ResearchResponse,
    ResearchEvent,
    StreamResponse,
)
from services.research import ResearchService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Research"])


@router.post(
    "/deep-research",
    response_model=Union[ResearchResponse, StreamResponse],
    summary="Perform deep research on a topic",
    description="Conducts comprehensive research on the provided query using LLMs and web search capabilities",
)
async def deep_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Perform deep research on a topic.

    This endpoint takes a research query and performs comprehensive research using
    web search, LLMs, and other tools to generate a detailed summary. The process
    typically takes around 500 seconds.

    If streaming is enabled, it returns a stream URL and starts the research
    in the background. Otherwise, it returns the complete research results.
    """
    logger.info(
        f"Received research request: {request.query[:50]}{'...' if len(request.query) > 50 else ''}"
    )
    logger.info(
        f"Extra effort: {request.extra_effort} (type: {type(request.extra_effort)})"
    )
    logger.info(
        f"Minimum effort: {request.minimum_effort} (type: {type(request.minimum_effort)})"
    )
    # CRITICAL DEBUG: Log steering_enabled
    logger.info(
        f"STEERING_ENABLED: {request.steering_enabled} (type: {type(request.steering_enabled)})"
    )
    logger.info(
        f"Benchmark mode: {request.benchmark_mode} (type: {type(request.benchmark_mode)})"
    )

    # Enhanced logging for uploaded content
    if request.uploaded_data_content:
        logger.info(f"[UPLOAD_TRACE] Uploaded data content received in router")
        logger.info(
            f"[UPLOAD_TRACE] Content length: {len(request.uploaded_data_content)} characters"
        )
        logger.info(
            f"[UPLOAD_TRACE] Content preview (first 100 chars): {request.uploaded_data_content[:100]}..."
        )
        logger.info(
            f"[UPLOAD_TRACE] Content type: {type(request.uploaded_data_content)}"
        )
    else:
        logger.info(
            f"[UPLOAD_TRACE] No uploaded data content in request (value: {request.uploaded_data_content})"
        )

    if request.streaming:
        # Generate a unique stream ID
        stream_id = str(uuid.uuid4())
        stream_url = f"/stream/{stream_id}"

        # Create a new queue for this stream
        queue = Queue()
        await ResearchService.add_queue(stream_id, queue)

        logger.info(
            f"[UPLOAD_TRACE] Passing uploaded_data_content to streaming research service"
        )

        # Start research in the background, passing the queue
        background_tasks.add_task(
            ResearchService.conduct_research,
            query=request.query,
            extra_effort=request.extra_effort,
            minimum_effort=request.minimum_effort,
            benchmark_mode=request.benchmark_mode,
            streaming=True,
            stream_id=stream_id,
            queue=queue,  # Pass the queue instance
            provider=request.provider,
            model=request.model,
            uploaded_data_content=request.uploaded_data_content,  # Pass uploaded content
            database_info=request.database_info,  # Pass database information
            uploaded_files=request.uploaded_files,  # Pass uploaded file IDs
            steering_enabled=request.steering_enabled,  # Pass steering flag
            parallel_search_enabled=request.parallel_search_enabled,
            parallel_search_max_concurrency=request.parallel_search_max_concurrency,
        )

        # Return the stream URL
        return StreamResponse(
            stream_url=stream_url,
            message="Research started. Connect to the stream URL to receive updates.",
        )

    # Handle non-streaming case (synchronous execution)
    else:
        # Non-streaming research (kept separate for clarity, no queue needed)
        try:
            logger.info(
                f"[UPLOAD_TRACE] Passing uploaded_data_content to non-streaming research service"
            )

            result = await ResearchService.conduct_research(
                query=request.query,
                extra_effort=request.extra_effort,
                minimum_effort=request.minimum_effort,
                benchmark_mode=request.benchmark_mode,
                streaming=False,
                stream_id=None,  # No stream_id needed
                queue=None,  # No queue needed
                provider=request.provider,
                model=request.model,
                uploaded_data_content=request.uploaded_data_content,  # Pass uploaded content
                database_info=request.database_info,  # Pass database information
                uploaded_files=request.uploaded_files,  # Pass uploaded file IDs
                steering_enabled=request.steering_enabled,  # Pass steering flag
                parallel_search_enabled=request.parallel_search_enabled,
                parallel_search_max_concurrency=request.parallel_search_max_concurrency,
            )

            logger.info(
                f"Research completed for: {request.query[:50]}{'...' if len(request.query) > 50 else ''}"
            )

            # Ensure result is ResearchResponse model before returning
            if isinstance(result, ResearchResponse):
                return result
            else:
                # Log unexpected result type and raise internal server error
                logger.error(
                    f"Unexpected result type for non-streaming research: {type(result)}"
                )
                raise HTTPException(
                    status_code=500,
                    detail="Internal server error: Unexpected research result format.",
                )

        except Exception as e:
            logger.error(f"Error performing non-streaming research: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error performing research: {str(e)}"
            )


@router.get(
    "/research-status",
    summary="Check API status",
    description="Returns the current status of the research API",
)
async def research_status():
    """Check if the research API is operational."""
    return {
        "status": "operational",
        "message": "Research API is running and ready to accept requests",
    }


@router.post(
    "/cleanup-all",
    summary="Cleanup all active sessions",
    description="Stops all active research and clears all sessions for fresh start",
)
async def cleanup_all_sessions():
    """Cleanup all active research sessions for fresh start."""
    logger.info("Received cleanup request for all active sessions")

    try:
        # Cancel all active research streams
        cancelled_count = 0
        for stream_id in list(ResearchService._cancellation_flags.keys()):
            await ResearchService.request_cancellation(stream_id)
            cancelled_count += 1

        # Cleanup all steering sessions
        try:
            from routers.simple_steering_api import active_research_sessions

            session_count = len(active_research_sessions)
            active_research_sessions.clear()
            logger.info(f"Cleared {session_count} steering sessions")
        except Exception as steering_err:
            logger.warning(f"Error cleaning up steering sessions: {steering_err}")

        # Clear all session-specific database info
        try:
            from src.graph import clear_database_info

            clear_database_info()  # Clear all sessions
            logger.info("Cleared all session database info")
        except Exception as db_err:
            logger.warning(f"Error clearing database info: {db_err}")

        return {
            "status": "cleanup_complete",
            "cancelled_streams": cancelled_count,
            "message": "All active sessions have been cleaned up. Ready for fresh research.",
        }
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")


@router.post(
    "/stop/{stream_id}",
    summary="Stop research execution",
    description="Requests cancellation of an active research session",
)
async def stop_research(stream_id: str):
    """Stop/cancel an active research session."""
    logger.info(f"Received stop request for stream {stream_id}")

    try:
        await ResearchService.request_cancellation(stream_id)

        # Also cleanup any associated steering sessions
        try:
            from routers.simple_steering_api import active_research_sessions

            sessions_to_remove = []
            for session_id, session_info in active_research_sessions.items():
                # Mark any active sessions as inactive
                if session_info.get("active", False):
                    session_info["active"] = False
                    sessions_to_remove.append(session_id)
                    logger.info(
                        f"Marked steering session {session_id} as inactive due to stop request"
                    )

            # Remove inactive sessions
            for session_id in sessions_to_remove:
                if session_id in active_research_sessions:
                    del active_research_sessions[session_id]
                    logger.info(f"Removed steering session {session_id}")
        except Exception as steering_err:
            logger.warning(f"Error cleaning up steering sessions: {steering_err}")

        # Clear database info for this specific stream
        try:
            from src.graph import clear_database_info

            clear_database_info(session_id=stream_id)
            logger.info(f"Cleared database info for stream {stream_id}")
        except Exception as db_err:
            logger.warning(f"Error clearing database info for stream: {db_err}")

        return {
            "status": "cancellation_requested",
            "stream_id": stream_id,
            "message": "Research cancellation requested. All active sessions will be stopped.",
        }
    except Exception as e:
        logger.error(f"Error stopping research {stream_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error stopping research: {str(e)}"
        )


@router.get(
    "/stream/{stream_id}",
    summary="Stream research updates",
    description="Streams research updates for a specific research session",
)
async def stream_research(stream_id: str, request: Request):
    """Stream research updates via Server-Sent Events."""

    logger.info(f"Received connection for stream {stream_id}")

    # Get the queue for this stream
    queue = await ResearchService.get_queue(stream_id)

    if not queue:
        logger.error(f"No queue found for stream {stream_id}")
        raise HTTPException(
            status_code=404, detail=f"Stream {stream_id} not found or expired"
        )

    async def event_generator():
        logger.info(f"Event generator started for {stream_id}")
        heartbeat_interval = 5.0  # Send heartbeat every 5 seconds when idle
        last_heartbeat_time = time.time()

        try:
            # Send an initial connection event
            connection_event = {
                "event_type": "connected",
                "data": {
                    "stream_id": stream_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
            }
            yield {"event": "connected", "data": json.dumps(connection_event)}
            logger.info(f"Sent initial connection event for {stream_id}")

            # Process events from the queue
            while True:
                # Check for client disconnect
                if await request.is_disconnected():
                    logger.warning(f"Client disconnected from stream {stream_id}")
                    break

                # Check if it's time to send a heartbeat
                current_time = time.time()
                time_since_last_heartbeat = current_time - last_heartbeat_time

                # Get event from queue with timeout
                try:
                    # Reduce timeout to ensure we send heartbeats on schedule
                    timeout = min(
                        heartbeat_interval,
                        max(0.1, heartbeat_interval - time_since_last_heartbeat),
                    )
                    event = await asyncio.wait_for(queue.get(), timeout=timeout)

                    # None is the sentinel value indicating end of stream
                    if event is None:
                        logger.info(f"Received end-of-stream signal for {stream_id}")
                        break

                    # Format and yield the event for SSE
                    event_type = event.get("event_type", "update")
                    event_data = json.dumps(event)
                    yield {"event": event_type, "data": event_data}
                    if event_type != "heartbeat":
                        logger.info(
                            f"Sent regular event type {event_type} for {stream_id}"
                        )
                    else:
                        logger.info(
                            f"Sent heartbeat event type {event_type} for {stream_id}"
                        )

                except asyncio.TimeoutError:
                    # Check if it's time to send a heartbeat
                    if time_since_last_heartbeat >= heartbeat_interval:
                        # Send a heartbeat event on timeout
                        heartbeat = {
                            "event_type": "heartbeat",
                            "data": {"timestamp": datetime.datetime.now().isoformat()},
                        }
                        yield {"event": "heartbeat", "data": json.dumps(heartbeat)}
                        # logger.info(f"Sent heartbeat for {stream_id}")
                        last_heartbeat_time = time.time()
                except Exception as e:
                    logger.error(f"Error processing event for {stream_id}: {e}")
                    error_event = {"event_type": "error", "data": {"error": str(e)}}
                    yield {"event": "error", "data": json.dumps(error_event)}

        except asyncio.CancelledError:
            logger.info(f"Event generator for {stream_id} cancelled")
        except Exception as e:
            logger.error(
                f"Error in event generator for {stream_id}: {e}", exc_info=True
            )
        finally:
            logger.info(f"Event generator for {stream_id} finishing")
            # Don't remove the queue here, as the research might still be running

    return EventSourceResponse(event_generator())
