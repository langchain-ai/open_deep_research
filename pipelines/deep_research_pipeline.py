"""
title: Deep Research Pipeline
author: Open Deep Research
author_url: https://github.com/langchain-ai/open_deep_research
description: Integrates the Open Deep Research LangGraph agent with OpenWebUI for comprehensive AI-powered research
required_open_webui_version: 0.4.3
requirements: requests, pydantic
version: 1.0.0
licence: MIT
"""

import json
import random
import time
from typing import Any, Dict, Generator, Iterator, List, Union

import requests
from pydantic import BaseModel, Field


class Pipeline:
    """OpenWebUI Pipeline for Deep Research integration."""

    class Valves(BaseModel):
        """Configuration options exposed in OpenWebUI admin panel."""

        # ===================
        # Connection Settings
        # ===================
        LANGGRAPH_URL: str = Field(
            default="http://open-deep-research:2024",
            description="URL of the LangGraph Deep Research server (use container name for Docker network)",
        )

        # ================
        # Polling Settings
        # ================
        POLL_INTERVAL: int = Field(
            default=3,
            description="Seconds between status checks during research (lower = more responsive, higher = less load)",
        )
        MAX_WAIT_TIME: int = Field(
            default=600,
            description="Maximum seconds to wait for research completion (default: 10 minutes)",
        )

        # =======================
        # Research Configuration
        # =======================
        SEARCH_API: str = Field(
            default="tavily",
            description="Search API to use: 'tavily', 'openai', 'anthropic', or 'none'",
        )
        RESEARCH_MODEL: str = Field(
            default="anthropic:claude-sonnet-4-20250514",
            description="Model for conducting research (format: provider:model-name)",
        )
        RESEARCH_MODEL_MAX_TOKENS: int = Field(
            default=16000, description="Maximum output tokens for research model"
        )
        COMPRESSION_MODEL: str = Field(
            default="anthropic:claude-sonnet-4-20250514",
            description="Model for compressing research findings from sub-agents",
        )
        COMPRESSION_MODEL_MAX_TOKENS: int = Field(
            default=8192, description="Maximum output tokens for compression model"
        )
        FINAL_REPORT_MODEL: str = Field(
            default="anthropic:claude-sonnet-4-20250514",
            description="Model for writing the final comprehensive research report",
        )
        FINAL_REPORT_MODEL_MAX_TOKENS: int = Field(
            default=16000, description="Maximum output tokens for final report model"
        )
        SUMMARIZATION_MODEL: str = Field(
            default="anthropic:claude-3-5-haiku-latest",
            description="Model for summarizing search results (fast/cheap model recommended)",
        )
        SUMMARIZATION_MODEL_MAX_TOKENS: int = Field(
            default=4096, description="Maximum output tokens for summarization model"
        )

        # ==================
        # Research Behavior
        # ==================
        ALLOW_CLARIFICATION: bool = Field(
            default=True,
            description="Allow the researcher to ask clarifying questions before starting research",
        )
        MAX_CONCURRENT_RESEARCH_UNITS: int = Field(
            default=5,
            description="Maximum parallel research sub-agents (higher = faster but may hit rate limits)",
        )
        MAX_RESEARCHER_ITERATIONS: int = Field(
            default=6,
            description="Maximum research iterations for the supervisor (more = deeper research)",
        )
        MAX_REACT_TOOL_CALLS: int = Field(
            default=10, description="Maximum tool calls per researcher step"
        )
        MAX_CONTENT_LENGTH: int = Field(
            default=50000,
            description="Maximum character length for webpage content before summarization",
        )

    # Static constant - must match the graph name in langgraph.json
    ASSISTANT_ID = "Deep Researcher"

    def __init__(self):
        """Initialize the pipeline with default configuration."""
        # Pipeline identification - simple pipe type (single model endpoint)
        # Note: Do NOT set self.id - let it be inferred from filename
        # This avoids routing issues with OpenWebUI's pipeline. prefix
        self.name = "Deep Research"

        # Initialize valves (configuration)
        self.valves = self.Valves()

        # HTTP headers for LangGraph API
        self.headers = {"Content-Type": "application/json"}

        # Store thread data per conversation
        self.conversations: Dict[str, Dict[str, Any]] = {}

        # Status messages to show during long-running research
        self.status_messages = [
            "Analyzing your research question...",
            "Diving deep into the data ocean...",
            "Gathering insights from multiple sources...",
            "Cross-referencing findings...",
            "Synthesizing research results...",
            "Compiling comprehensive analysis...",
            "Consulting the knowledge base...",
            "Exploring related topics...",
            "Verifying information accuracy...",
            "Building your research report...",
            "Connecting the dots across sources...",
            "Extracting key insights...",
            "Organizing findings systematically...",
            "Preparing detailed analysis...",
            "Almost ready with your report...",
        ]

    async def on_startup(self):
        """Called when the server is started."""
        print(f"on_startup:{__name__}")
        print(f"LangGraph URL: {self.valves.LANGGRAPH_URL}")
        print(f"Assistant ID: {self.ASSISTANT_ID}")

    async def on_shutdown(self):
        """Called when the server is stopped."""
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        """Called when the valves are updated."""
        print(f"on_valves_updated:{__name__}")
        print(f"New LangGraph URL: {self.valves.LANGGRAPH_URL}")

    def get_conversation_id(self, messages: List[Dict[str, Any]]) -> str:
        """Generate a conversation ID based on the first message."""
        if messages:
            first_msg = str(messages[0].get("content", ""))[:100]
            return str(hash(first_msg))
        return "default"

    def get_random_status_message(self) -> str:
        """Returns a random status message for the research process."""
        return random.choice(self.status_messages)

    def create_thread(self) -> str:
        """Creates a new LangGraph thread and returns the thread ID."""
        try:
            url = f"{self.valves.LANGGRAPH_URL}/threads"
            payload = json.dumps(
                {"thread_id": "", "metadata": {}, "if_exists": "raise"}
            )
            response = requests.post(
                url, headers=self.headers, data=payload, timeout=30
            )
            response.raise_for_status()
            return response.json()["thread_id"]
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to LangGraph server at {self.valves.LANGGRAPH_URL}. "
                f"Ensure the open-deep-research container is running and on the same Docker network. "
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Error creating thread: {e}")

    def get_configurable_params(self) -> Dict[str, Any]:
        """Build the configurable parameters to pass to LangGraph.

        Only include parameters that have non-default values to avoid
        validation errors with the LangGraph API.
        """
        # Build config with all parameters - LangGraph will use defaults for any missing
        config = {}

        # Only add non-empty/non-default values
        if self.valves.SEARCH_API:
            config["search_api"] = self.valves.SEARCH_API
        if self.valves.RESEARCH_MODEL:
            config["research_model"] = self.valves.RESEARCH_MODEL
        if self.valves.RESEARCH_MODEL_MAX_TOKENS:
            config["research_model_max_tokens"] = self.valves.RESEARCH_MODEL_MAX_TOKENS
        if self.valves.COMPRESSION_MODEL:
            config["compression_model"] = self.valves.COMPRESSION_MODEL
        if self.valves.COMPRESSION_MODEL_MAX_TOKENS:
            config["compression_model_max_tokens"] = (
                self.valves.COMPRESSION_MODEL_MAX_TOKENS
            )
        if self.valves.FINAL_REPORT_MODEL:
            config["final_report_model"] = self.valves.FINAL_REPORT_MODEL
        if self.valves.FINAL_REPORT_MODEL_MAX_TOKENS:
            config["final_report_model_max_tokens"] = (
                self.valves.FINAL_REPORT_MODEL_MAX_TOKENS
            )
        if self.valves.SUMMARIZATION_MODEL:
            config["summarization_model"] = self.valves.SUMMARIZATION_MODEL
        if self.valves.SUMMARIZATION_MODEL_MAX_TOKENS:
            config["summarization_model_max_tokens"] = (
                self.valves.SUMMARIZATION_MODEL_MAX_TOKENS
            )

        # Boolean and integer configs
        config["allow_clarification"] = self.valves.ALLOW_CLARIFICATION
        config["max_concurrent_research_units"] = (
            self.valves.MAX_CONCURRENT_RESEARCH_UNITS
        )
        config["max_researcher_iterations"] = self.valves.MAX_RESEARCHER_ITERATIONS
        config["max_react_tool_calls"] = self.valves.MAX_REACT_TOOL_CALLS
        config["max_content_length"] = self.valves.MAX_CONTENT_LENGTH

        return config

    def emit_status(self, description: str, done: bool = False) -> Dict[str, Any]:
        """Emit a status event for OpenWebUI to display."""
        return {
            "event": {
                "type": "status",
                "data": {
                    "description": description,
                    "done": done,
                },
            }
        }

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        """
        Main pipeline method that processes the user message and returns response.

        This method:
        1. Creates/reuses a LangGraph thread for the conversation
        2. Starts a research run with the user's query
        3. Polls for completion while showing status updates
        4. Returns the final research report
        """
        print(f"pipe:{__name__}")

        # Check if this is a title generation request
        if body.get("title", False):
            print("Title Generation Request - returning simple response")
            return "Deep Research Report"

        # Emit initial status
        yield self.emit_status("Initializing deep research...")

        try:
            print(f"[Deep Research] Processing query: {user_message[:100]}...")

            if not messages:
                yield self.emit_status("Error: No messages provided", done=True)
                yield "Error: No messages provided"
                return

            # Get or create conversation context
            conv_id = self.get_conversation_id(messages)
            conv_data = self.conversations.get(
                conv_id, {"thread_id": None, "message_count": 0}
            )

            # Create new thread for new conversations
            if not conv_data["thread_id"]:
                print("[Deep Research] Creating new thread...")
                yield self.emit_status("Creating research session...")
                try:
                    conv_data["thread_id"] = self.create_thread()
                    conv_data["message_count"] = 0
                    print(f"[Deep Research] Created thread: {conv_data['thread_id']}")
                except ConnectionError as e:
                    yield self.emit_status(f"Connection Error", done=True)
                    yield f"Connection Error: {e}"
                    return
                except Exception as e:
                    yield self.emit_status("Error creating session", done=True)
                    yield f"Error creating research session: {e}"
                    return

            # Store conversation data
            self.conversations[conv_id] = conv_data

            # Prepare the run request
            thread_id = conv_data["thread_id"]
            run_url = f"{self.valves.LANGGRAPH_URL}/threads/{thread_id}/runs"

            # Get username from body if available
            username = "unknown"
            if "user" in body and isinstance(body["user"], dict):
                username = body["user"].get("name", "unknown")

            # Build the payload for LangGraph API
            # Note: LangGraph uses "role": "human" for user messages
            request_payload = {
                "assistant_id": self.ASSISTANT_ID,
                "input": {"messages": [{"role": "human", "content": user_message}]},
                "metadata": {"openwebui_username": username},
            }

            # Add configurable parameters if any are set
            config_params = self.get_configurable_params()
            if config_params:
                request_payload["config"] = {"configurable": config_params}

            payload = json.dumps(request_payload)

            # Debug: Log the payload being sent
            print(f"[Deep Research] Sending payload: {payload[:500]}...")

            # Start the research run
            print(f"[Deep Research] Starting run at: {run_url}")
            yield self.emit_status("Starting deep research...")

            try:
                run_response = requests.post(
                    run_url, headers=self.headers, data=payload, timeout=60
                )
                # Check for errors and get detailed error message
                if run_response.status_code >= 400:
                    error_detail = "Unknown error"
                    try:
                        error_json = run_response.json()
                        error_detail = error_json.get("detail", str(error_json))
                    except:
                        error_detail = run_response.text[:500]
                    print(
                        f"[Deep Research] API Error {run_response.status_code}: {error_detail}"
                    )
                    yield self.emit_status(
                        f"API Error ({run_response.status_code})", done=True
                    )
                    yield f"Error starting research ({run_response.status_code}): {error_detail}"
                    return
                run_response.raise_for_status()
            except requests.exceptions.ConnectionError as e:
                yield self.emit_status("Connection Error", done=True)
                yield f"Connection Error: Cannot reach LangGraph server. {e}"
                return
            except requests.exceptions.Timeout:
                yield self.emit_status("Request Timeout", done=True)
                yield "Error: Request timed out while starting research"
                return
            except requests.exceptions.HTTPError as e:
                yield self.emit_status("HTTP Error", done=True)
                yield f"Error starting research: {e}"
                return

            run_data = run_response.json()
            run_id = run_data.get("run_id")

            if not run_id:
                yield self.emit_status("Failed to start research", done=True)
                yield "Error: Failed to start research - No run ID received"
                return

            print(f"[Deep Research] Started run: {run_id}")
            yield self.emit_status(f"Research started (ID: {run_id[:8]}...)")

            # Poll for completion
            check_run_url = (
                f"{self.valves.LANGGRAPH_URL}/threads/{thread_id}/runs/{run_id}"
            )
            start_time = time.time()
            last_status_time = time.time()
            status_interval = 8  # Show status message every 8 seconds

            print("[Deep Research] Polling for completion...")

            while time.time() - start_time <= self.valves.MAX_WAIT_TIME:
                try:
                    status_response = requests.get(
                        check_run_url, headers=self.headers, timeout=30
                    )
                    status_response.raise_for_status()
                    status_data = status_response.json()
                    current_status = status_data.get("status", "unknown")

                    print(f"[Deep Research] Status: {current_status}")

                    if current_status == "success":
                        print("[Deep Research] Research completed successfully")
                        yield self.emit_status("Finalizing research report...")
                        break
                    elif current_status in ["error", "timeout", "interrupted"]:
                        error_detail = status_data.get("error", "Unknown error")
                        print(
                            f"[Deep Research] Run failed: {current_status} - {error_detail}"
                        )
                        yield self.emit_status(
                            f"Research failed: {current_status}", done=True
                        )
                        yield f"Research failed with status: {current_status}\nDetails: {error_detail}"
                        return

                    # Show periodic status updates with elapsed time
                    current_time = time.time()
                    if current_time - last_status_time >= status_interval:
                        elapsed = int(current_time - start_time)
                        status_msg = self.get_random_status_message()
                        yield self.emit_status(f"[{elapsed}s] {status_msg}")
                        last_status_time = current_time

                    time.sleep(self.valves.POLL_INTERVAL)

                except requests.exceptions.RequestException as e:
                    print(f"[Deep Research] Polling error: {e}")
                    yield self.emit_status("Temporary connection issue, retrying...")
                    time.sleep(self.valves.POLL_INTERVAL)
                    continue
            else:
                elapsed = int(time.time() - start_time)
                yield self.emit_status(
                    f"Research timed out after {elapsed}s", done=True
                )
                yield f"Research timed out after {elapsed} seconds. Try increasing MAX_WAIT_TIME in pipeline settings."
                return

            # Fetch the final result
            join_url = f"{check_run_url}/join"
            print(f"[Deep Research] Fetching results from: {join_url}")
            yield self.emit_status("Retrieving research results...")

            try:
                result_response = requests.get(
                    join_url, headers=self.headers, timeout=60
                )
                result_response.raise_for_status()
            except requests.exceptions.RequestException as e:
                yield self.emit_status("Error fetching results", done=True)
                yield f"Error fetching research results: {e}"
                return

            # Mark status as done before returning the report
            yield self.emit_status("", done=True)

            # Extract and return the response
            result_data = result_response.json()
            messages_data = result_data.get("messages", [])

            if not messages_data:
                yield "No research results received. The query may have been too complex or unclear."
                return

            # Get the last assistant message (the final report)
            latest_message = messages_data[-1]
            output_content = latest_message.get("content", "")

            if not output_content:
                yield "Research completed but no report was generated."
                return

            # Return the research report
            yield output_content

            # Update conversation state
            conv_data["message_count"] = len(messages)
            self.conversations[conv_id] = conv_data

            print("[Deep Research] Response delivered successfully")

        except requests.exceptions.RequestException as e:
            error_msg = f"Network error: {e}"
            print(f"[Deep Research] {error_msg}")
            yield self.emit_status("Network Error", done=True)
            yield error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            print(f"[Deep Research] {error_msg}")
            import traceback

            traceback.print_exc()
            yield self.emit_status("Unexpected Error", done=True)
            yield error_msg
