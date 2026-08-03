import importlib.util
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

SCORE_MODELS = {
    "OverallQualityScore": (
        "research_depth",
        "source_quality",
        "analytical_rigor",
        "practical_value",
        "balance_and_objectivity",
        "writing_quality",
    ),
    "RelevanceScore": ("score",),
    "StructureScore": ("score",),
    "CorrectnessScore": ("score",),
    "CompletenessScore": ("score",),
}


def _load_evaluators(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    utils = types.ModuleType("open_deep_research.utils")
    utils.get_today_str = lambda: "Mon Aug 3, 2026"
    monkeypatch.setitem(sys.modules, "open_deep_research.utils", utils)

    module_name = "_evaluators_score_bounds_test"
    path = Path(__file__).parent / "evaluators.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _valid_payload(model, score_fields):
    payload = {}
    for name, field in model.model_fields.items():
        if name in score_fields:
            payload[name] = 3
        elif field.annotation is str:
            payload[name] = "reason"
        else:
            raise AssertionError(f"Unsupported field {model.__name__}.{name}")
    return payload


@pytest.mark.parametrize(
    ("model_name", "score_fields"),
    SCORE_MODELS.items(),
)
def test_evaluator_scores_are_bounded(
    monkeypatch,
    model_name,
    score_fields,
) -> None:
    evaluators = _load_evaluators(monkeypatch)
    model = getattr(evaluators, model_name)
    schema = model.model_json_schema()["properties"]

    for field_name in score_fields:
        assert schema[field_name]["minimum"] == 1
        assert schema[field_name]["maximum"] == 5

        payload = _valid_payload(model, score_fields)
        payload[field_name] = 0
        with pytest.raises(ValidationError):
            model(**payload)

        payload[field_name] = 6
        with pytest.raises(ValidationError):
            model(**payload)
