from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from open_deep_research.utils import remove_up_to_last_ai_message


def test_remove_up_to_last_ai_message_keeps_recent_tail_after_last_ai():
    messages = [
        HumanMessage(content="old user"),
        AIMessage(content="old ai"),
        ToolMessage(content="recent tool", tool_call_id="tool_1"),
        HumanMessage(content="recent user"),
    ]

    result = remove_up_to_last_ai_message(messages)

    assert result == messages[2:]


def test_remove_up_to_last_ai_message_no_ai_returns_original():
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="user"),
        ToolMessage(content="tool", tool_call_id="tool_2"),
    ]

    result = remove_up_to_last_ai_message(messages)

    assert result == messages


def test_remove_up_to_last_ai_message_last_message_is_ai_returns_empty():
    messages = [
        HumanMessage(content="user"),
        AIMessage(content="latest ai"),
    ]

    result = remove_up_to_last_ai_message(messages)

    assert result == []


def test_remove_up_to_last_ai_message_multiple_ai_uses_last_ai_boundary():
    messages = [
        HumanMessage(content="u1"),
        AIMessage(content="a1"),
        HumanMessage(content="u2"),
        AIMessage(content="a2"),
        ToolMessage(content="t1", tool_call_id="tool_3"),
    ]

    result = remove_up_to_last_ai_message(messages)

    assert result == messages[4:]
