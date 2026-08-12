import importlib.util
import sys
import types
from pathlib import Path


class EmptyClaimsModel:
    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def with_retry(self, **kwargs):
        return self

    def invoke(self, messages):
        return self.schema(claims=[])


def test_groundedness_without_claims_returns_zero(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    utils = types.ModuleType("open_deep_research.utils")
    utils.get_today_str = lambda: "Mon Aug 3, 2026"
    monkeypatch.setitem(sys.modules, "open_deep_research.utils", utils)

    module_name = "_evaluators_empty_claims_test"
    path = Path(__file__).parent / "evaluators.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    evaluators = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, evaluators)
    spec.loader.exec_module(evaluators)
    evaluators.eval_model = EmptyClaimsModel()

    result = evaluators.eval_groundedness(
        {},
        {
            "final_report": "No factual claims.",
            "raw_notes": [],
        },
    )

    assert result == {
        "key": "groundedness_score",
        "score": 0.0,
        "comment": "[]",
    }
