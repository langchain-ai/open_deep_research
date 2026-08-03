from types import SimpleNamespace

import pytest

from legacy import multi_agent


class FakeSupervisorModel:
    def __init__(self):
        self.messages = None

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return "response"


class SectionsTool:
    name = "Sections"
    metadata = None

    async def ainvoke(self, args, config):
        return multi_agent.Sections(sections=["First section", "Second section"])


async def no_supervisor_tools(config):
    return []


async def sections_tool(config):
    return [SectionsTool()]


@pytest.mark.asyncio
async def test_supervisor_tools_records_planned_sections(monkeypatch):
    monkeypatch.setattr(multi_agent, "get_supervisor_tools", sections_tool)
    state = {
        "messages": [
            SimpleNamespace(
                tool_calls=[{"name": "Sections", "args": {}, "id": "sections-call"}]
            )
        ]
    }

    command = await multi_agent.supervisor_tools(state, {"configurable": {}})

    assert command.update["sections"] == ["First section", "Second section"]


@pytest.mark.asyncio
async def test_supervisor_waits_for_every_planned_section(monkeypatch):
    model = FakeSupervisorModel()
    monkeypatch.setattr(multi_agent, "init_chat_model", lambda **kwargs: model)
    monkeypatch.setattr(multi_agent, "get_supervisor_tools", no_supervisor_tools)
    state = {
        "messages": [{"role": "user", "content": "Research this topic."}],
        "sections": ["First section", "Second section"],
        "completed_sections": [
            multi_agent.Section(
                name="First section", description="First", content="First findings"
            )
        ],
    }

    await multi_agent.supervisor(state, {"configurable": {"search_api": "none"}})

    assert model.messages[-1] == state["messages"][0]


@pytest.mark.asyncio
async def test_supervisor_starts_final_writing_after_every_planned_section(monkeypatch):
    model = FakeSupervisorModel()
    monkeypatch.setattr(multi_agent, "init_chat_model", lambda **kwargs: model)
    monkeypatch.setattr(multi_agent, "get_supervisor_tools", no_supervisor_tools)
    state = {
        "messages": [{"role": "user", "content": "Research this topic."}],
        "sections": ["First section", "Second section"],
        "completed_sections": [
            multi_agent.Section(
                name="First section", description="First", content="First findings"
            ),
            multi_agent.Section(
                name="Second section", description="Second", content="Second findings"
            ),
        ],
    }

    await multi_agent.supervisor(state, {"configurable": {"search_api": "none"}})

    assert model.messages[-1]["content"].startswith("Research is complete")
