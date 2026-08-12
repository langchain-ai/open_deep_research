import importlib.util
import sys
import types
from pathlib import Path


def test_pairwise_evaluation_runs_only_from_main(monkeypatch) -> None:
    calls = []

    def evaluate_comparative(*args, **kwargs):
        calls.append((args, kwargs))

    anthropic = types.ModuleType("langchain_anthropic")
    anthropic.ChatAnthropic = object
    evaluation = types.ModuleType("langsmith.evaluation")
    evaluation.evaluate_comparative = evaluate_comparative
    monkeypatch.setitem(sys.modules, "langchain_anthropic", anthropic)
    monkeypatch.setitem(sys.modules, "langsmith.evaluation", evaluation)

    module_name = "_pairwise_evaluation_import_test"
    path = Path(__file__).parent / "pairwise_evaluation.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    assert calls == []

    module.main()

    assert len(calls) == 1
    assert calls[0][0][0] == (
        module.single_agent,
        module.multi_agent_supervisor_v2,
    )
