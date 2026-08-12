"""Unit tests for open_deep_research/state.py.

Tests cover the custom override_reducer function and state model structures.
"""

import operator

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearchQuestion,
    Summary,
    SupervisorState,
    override_reducer,
)


class TestOverrideReducer:
    """Tests for the custom override_reducer used in state annotations."""

    def test_normal_addition(self):
        """Regular values should be combined with operator.add."""
        result = override_reducer(["a", "b"], ["c"])
        assert result == ["a", "b", "c"]

    def test_override_type(self):
        """Override dicts should replace the current value entirely."""
        result = override_reducer(
            ["old_value"],
            {"type": "override", "value": ["new_value"]}
        )
        assert result == ["new_value"]

    def test_override_with_empty_value(self):
        """Override with empty list should clear the state."""
        result = override_reducer(
            ["existing"],
            {"type": "override", "value": []}
        )
        assert result == []

    def test_regular_dict_not_override(self):
        """Regular dicts without 'type': 'override' should be added normally."""
        result = override_reducer([{"a": 1}], [{"b": 2}])
        assert result == [{"a": 1}, {"b": 2}]

    def test_override_missing_value_key(self):
        """Override dict missing 'value' key should fall back to the dict itself."""
        result = override_reducer(
            ["existing"],
            {"type": "override"}
        )
        # When "value" key is missing, get returns the new_value dict
        assert result == {"type": "override"}


# ===================================================================
# Structured Output Models
# ===================================================================

class TestStructuredOutputModels:
    """Tests for Pydantic models used as structured outputs."""

    def test_conduct_research_fields(self):
        cr = ConductResearch(research_topic="quantum computing applications")
        assert cr.research_topic == "quantum computing applications"

    def test_research_complete_has_no_required_fields(self):
        rc = ResearchComplete()
        assert rc is not None

    def test_summary_fields(self):
        s = Summary(summary="Key findings...", key_excerpts="Quote 1, Quote 2")
        assert s.summary == "Key findings..."
        assert s.key_excerpts == "Quote 1, Quote 2"

    def test_clarify_with_user_fields(self):
        cwu = ClarifyWithUser(
            need_clarification=True,
            question="What do you mean by AI?",
            verification=""
        )
        assert cwu.need_clarification is True
        assert cwu.question == "What do you mean by AI?"

    def test_research_question_fields(self):
        rq = ResearchQuestion(research_brief="Investigate the impact of...")
        assert rq.research_brief == "Investigate the impact of..."
