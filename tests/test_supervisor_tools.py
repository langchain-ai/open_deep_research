"""Unit tests for `supervisor_tools` research-phase error handling.

Regression tests for the bug where any child `ConductResearch` exception was
treated as a successful end of the research phase (see issue #283). A genuine
child-researcher failure must be surfaced, while a token-limit error should
still end the phase gracefully with whatever notes were gathered.
"""

import asyncio
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END

from open_deep_research import deep_researcher
from open_deep_research.deep_researcher import researcher_subgraph, supervisor_tools


def _supervisor_state():
    """Build minimal supervisor state with a single ConductResearch tool call."""
    return {
        "supervisor_messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ConductResearch",
                        "args": {"research_topic": "test topic"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        "research_iterations": 1,
        "research_brief": "brief",
    }


_CONFIG = {"configurable": {"research_model": "openai:gpt-4.1"}}


async def _boom(*args, **kwargs):
    raise RuntimeError("boom from child researcher")


def test_child_research_error_is_surfaced():
    """A non-token-limit child failure must propagate, not end as success."""
    with patch.object(researcher_subgraph, "ainvoke", side_effect=_boom):
        with pytest.raises(RuntimeError, match="boom from child researcher"):
            asyncio.run(supervisor_tools(_supervisor_state(), _CONFIG))


def test_token_limit_ends_research_phase_gracefully():
    """A token-limit error should end the research phase with gathered notes."""
    with (
        patch.object(researcher_subgraph, "ainvoke", side_effect=_boom),
        patch.object(deep_researcher, "is_token_limit_exceeded", return_value=True),
    ):
        command = asyncio.run(supervisor_tools(_supervisor_state(), _CONFIG))

    assert command.goto == END
    assert "research_brief" in command.update
