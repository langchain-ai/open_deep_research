"""Tests for runtime configuration loading."""

import pytest
from pydantic import ValidationError

from open_deep_research.configuration import Configuration


def test_parses_mcp_config_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "MCP_CONFIG",
        '{"url":"http://localhost:8000/mcp","auth_required":"false"}',
    )

    config = Configuration.from_runnable_config(
        {"configurable": {"mcp_config": {"url": "http://fallback.example/mcp"}}}
    )

    assert config.mcp_config is not None
    assert config.mcp_config.url == "http://localhost:8000/mcp"
    assert config.mcp_config.auth_required is False


def test_accepts_mcp_config_from_runnable_config() -> None:
    config = Configuration.from_runnable_config(
        {
            "configurable": {
                "mcp_config": {
                    "url": "http://localhost:8000/mcp",
                    "auth_required": False,
                }
            }
        }
    )

    assert config.mcp_config is not None
    assert config.mcp_config.url == "http://localhost:8000/mcp"
    assert config.mcp_config.auth_required is False


def test_rejects_invalid_mcp_config_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_CONFIG", "{not-valid-json")

    with pytest.raises(ValidationError, match="Invalid JSON"):
        Configuration.from_runnable_config()
