from legacy.configuration import (
    Configuration,
    MultiAgentConfiguration,
    SearchAPI,
)


def test_workflow_environment_values_use_declared_types(monkeypatch) -> None:
    monkeypatch.setenv("INCLUDE_SOURCE_STR", "false")
    monkeypatch.setenv("MAX_SEARCH_DEPTH", "4")
    monkeypatch.setenv("NUMBER_OF_QUERIES", "7")
    monkeypatch.setenv("SEARCH_API", "none")
    monkeypatch.setenv("SEARCH_API_CONFIG", '{"max_results": 3}')

    config = Configuration.from_runnable_config()

    assert config.include_source_str is False
    assert config.max_search_depth == 4
    assert config.number_of_queries == 7
    assert config.search_api is SearchAPI.NONE
    assert config.search_api_config == {"max_results": 3}


def test_multi_agent_environment_values_use_declared_types(monkeypatch) -> None:
    monkeypatch.setenv("ASK_FOR_CLARIFICATION", "false")
    monkeypatch.setenv("INCLUDE_SOURCE_STR", "false")
    monkeypatch.setenv("MCP_SERVER_CONFIG", '{"server": {"url": "http://localhost"}}')
    monkeypatch.setenv("MCP_TOOLS_TO_INCLUDE", '["search", "fetch"]')
    monkeypatch.setenv("NUMBER_OF_QUERIES", "7")

    config = MultiAgentConfiguration.from_runnable_config()

    assert config.ask_for_clarification is False
    assert config.include_source_str is False
    assert config.mcp_server_config == {
        "server": {"url": "http://localhost"}
    }
    assert config.mcp_tools_to_include == ["search", "fetch"]
    assert config.number_of_queries == 7
