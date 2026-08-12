import pytest
from pydantic import ValidationError

from legacy.state import SearchQuery


def test_search_query_is_required():
    schema = SearchQuery.model_json_schema()
    assert schema["required"] == ["search_query"]

    with pytest.raises(ValidationError):
        SearchQuery()

    with pytest.raises(ValidationError):
        SearchQuery(search_query=None)

    query = SearchQuery(search_query="LangGraph state reducers")
    assert query.search_query == "LangGraph state reducers"
