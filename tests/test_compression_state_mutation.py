import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace


class Message:
    def __init__(self, content: str):
        self.content = content


class FakeConfiguration:
    compression_model = "compression-model"
    compression_model_max_tokens = 1000
    research_model = "research-model"

    @classmethod
    def from_runnable_config(cls, config):
        return cls()


class FakeSynthesizer:
    async def ainvoke(self, messages):
        return SimpleNamespace(content="compressed")


class FakeConfigurableModel:
    def with_config(self, config):
        return FakeSynthesizer()


def _load_compress_research():
    path = (
        Path(__file__).parents[1]
        / "src"
        / "open_deep_research"
        / "deep_researcher.py"
    )
    module = ast.parse(path.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "compress_research"
    )
    namespace = {
        "ResearcherState": dict,
        "RunnableConfig": dict,
        "Configuration": FakeConfiguration,
        "configurable_model": FakeConfigurableModel(),
        "get_api_key_for_model": lambda model, config: None,
        "HumanMessage": Message,
        "SystemMessage": Message,
        "compress_research_simple_human_message": "compress now",
        "compress_research_system_prompt": "Date: {date}",
        "get_today_str": lambda: "Mon Aug 3, 2026",
        "filter_messages": lambda messages, include_types: [],
        "is_token_limit_exceeded": lambda error, model: False,
        "remove_up_to_last_ai_message": lambda messages: messages,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[]),
            ),
            filename=str(path),
            mode="exec",
        ),
        namespace,
    )
    return namespace["compress_research"]


def test_compression_does_not_mutate_input_state() -> None:
    compress_research = _load_compress_research()
    original_message = Message("research result")
    state = {"researcher_messages": [original_message]}

    result = asyncio.run(compress_research(state, {}))

    assert state["researcher_messages"] == [original_message]
    assert result["compressed_research"] == "compressed"
