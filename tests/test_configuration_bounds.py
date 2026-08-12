import pytest
from pydantic import ValidationError

from open_deep_research.configuration import Configuration

BOUNDED_FIELDS = [
    ("max_structured_output_retries", 1, 10),
    ("max_concurrent_research_units", 1, 20),
    ("max_researcher_iterations", 1, 10),
    ("max_react_tool_calls", 1, 30),
    ("max_content_length", 1000, 200000),
]


@pytest.mark.parametrize(("field_name", "minimum", "maximum"), BOUNDED_FIELDS)
def test_configuration_enforces_declared_bounds(
    field_name: str,
    minimum: int,
    maximum: int,
):
    field_schema = Configuration.model_json_schema()["properties"][field_name]
    assert field_schema["minimum"] == minimum
    assert field_schema["maximum"] == maximum

    assert getattr(Configuration(**{field_name: minimum}), field_name) == minimum
    assert getattr(Configuration(**{field_name: maximum}), field_name) == maximum

    with pytest.raises(ValidationError):
        Configuration(**{field_name: minimum - 1})

    with pytest.raises(ValidationError):
        Configuration(**{field_name: maximum + 1})
