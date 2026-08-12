import runpy
import sys
import types
from pathlib import Path


def _module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def test_run_evaluate_loads_repository_env(monkeypatch, tmp_path) -> None:
    repo_root = Path(__file__).parents[1]
    loaded_paths = []

    def capture_dotenv_path(path) -> None:
        loaded_paths.append(Path(path).resolve())

    evaluator_names = (
        "eval_completeness",
        "eval_correctness",
        "eval_groundedness",
        "eval_overall_quality",
        "eval_relevance",
        "eval_structure",
    )
    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        _module("dotenv", load_dotenv=capture_dotenv_path),
    )
    monkeypatch.setitem(
        sys.modules,
        "langsmith",
        _module("langsmith", Client=lambda: object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "tests.evaluators",
        _module("tests.evaluators", **dict.fromkeys(evaluator_names)),
    )
    monkeypatch.setitem(
        sys.modules,
        "open_deep_research.deep_researcher",
        _module(
            "open_deep_research.deep_researcher",
            deep_researcher_builder=object(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "langgraph.checkpoint.memory",
        _module("langgraph.checkpoint.memory", MemorySaver=object),
    )
    monkeypatch.chdir(tmp_path)

    runpy.run_path(
        str(repo_root / "tests" / "run_evaluate.py"),
        run_name="run_evaluate_for_test",
    )

    assert loaded_paths == [repo_root / ".env"]
