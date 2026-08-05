"""Unit tests for utility functions in open_deep_research.utils."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from open_deep_research.utils import remove_oldest_research_turn


def _research_transcript() -> list:
    """Build a representative researcher message history."""
    topic = HumanMessage(content="Research topic")
    first_turn_ai = AIMessage(
        content="Searching for sources",
        tool_calls=[{"name": "web_search", "args": {"queries": ["q1"]}, "id": "call_1"}],
    )
    first_turn_tool = ToolMessage(content="Result A", tool_call_id="call_1")
    second_turn_ai = AIMessage(
        content="Searching for more sources",
        tool_calls=[{"name": "web_search", "args": {"queries": ["q2"]}, "id": "call_2"}],
    )
    second_turn_tool = ToolMessage(content="Result B", tool_call_id="call_2")
    final_ai = AIMessage(content="Research complete")
    compress_instruction = HumanMessage(content="Now compress the findings")
    return [
        topic,
        first_turn_ai,
        first_turn_tool,
        second_turn_ai,
        second_turn_tool,
        final_ai,
        compress_instruction,
    ]


def test_removes_oldest_research_turn() -> None:
    """The oldest AI turn and its tool results are dropped."""
    messages = _research_transcript()

    result = remove_oldest_research_turn(messages)

    assert result == [
        messages[0],  # topic is preserved
        messages[3],  # second turn AI
        messages[4],  # second turn tool result
        messages[5],  # final AI
        messages[6],  # compress instruction is preserved
    ]


def test_removes_turn_without_tool_results() -> None:
    """An AI message without tool results is still removable."""
    messages = [
        HumanMessage(content="Research topic"),
        AIMessage(content="Research complete"),
        HumanMessage(content="Now compress the findings"),
    ]

    result = remove_oldest_research_turn(messages)

    assert result == [
        messages[0],
        messages[2],
    ]


def test_returns_original_list_without_ai_messages() -> None:
    """A transcript without AI messages is returned unchanged."""
    messages = [HumanMessage(content="Research topic")]

    result = remove_oldest_research_turn(messages)

    assert result == messages


def test_keeps_tool_results_paired_with_their_ai_message() -> None:
    """Removal never leaves an orphaned ToolMessage behind."""
    messages = _research_transcript()

    result = remove_oldest_research_turn(messages)

    for i, message in enumerate(result):
        if isinstance(message, ToolMessage):
            assert isinstance(result[i - 1], AIMessage)


def test_repeated_calls_remove_turns_one_at_a_time() -> None:
    """Each call drops the next-oldest turn until nothing is left to drop."""
    messages = _research_transcript()

    result = remove_oldest_research_turn(messages)
    assert len(result) == 5

    result = remove_oldest_research_turn(result)
    assert len(result) == 3

    result = remove_oldest_research_turn(result)
    assert result == [
        messages[0],
        messages[6],
    ]

    # No AI messages remain, so the transcript is returned unchanged.
    assert remove_oldest_research_turn(result) == result
