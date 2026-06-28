"""Unit tests for the fastCRW search provider (mocked HTTP, no network)."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch

from open_deep_research.configuration import SearchAPI
from open_deep_research.utils import (
    CRW_DEFAULT_BASE_URL,
    crw_search_async,
    get_crw_api_key,
    get_crw_base_url,
    get_search_tool,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    async def json(self):
        return self._payload


class _FakeSession:
    """Minimal aiohttp.ClientSession stand-in capturing calls and returning a payload."""

    def __init__(self, payload, recorder):
        self._payload = payload
        self._recorder = recorder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @asynccontextmanager
    async def post(self, url, json=None, headers=None):
        self._recorder.append({"url": url, "json": json, "headers": headers})
        yield _FakeResponse(self._payload)


def _make_session_factory(payload, recorder):
    def factory(*args, **kwargs):
        return _FakeSession(payload, recorder)

    return factory


def test_crw_search_async_normalizes_envelope(monkeypatch):
    """crw_search_async maps the fastCRW envelope to a Tavily-like shape."""
    monkeypatch.setenv("CRW_API_KEY", "test-key")
    monkeypatch.delenv("CRW_API_URL", raising=False)
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "false")

    payload = {
        "success": True,
        "data": [
            {
                "title": "Example",
                "url": "https://example.com",
                "description": "An example result",
                "markdown": "# Example\n\nBody content",
            }
        ],
    }
    recorder = []

    with patch(
        "open_deep_research.utils.aiohttp.ClientSession",
        _make_session_factory(payload, recorder),
    ):
        results = asyncio.run(
            crw_search_async(["hello world"], max_results=3, config=None)
        )

    assert len(results) == 1
    assert results[0]["query"] == "hello world"
    assert results[0]["results"] == [
        {
            "title": "Example",
            "url": "https://example.com",
            "content": "An example result",
            "raw_content": "# Example\n\nBody content",
        }
    ]

    # Verify the request hit /v1/search with Bearer auth and the right body
    assert len(recorder) == 1
    call = recorder[0]
    assert call["url"] == f"{CRW_DEFAULT_BASE_URL}/v1/search"
    assert call["json"]["query"] == "hello world"
    assert call["json"]["limit"] == 3
    assert call["headers"]["Authorization"] == "Bearer test-key"


def test_crw_search_async_omits_auth_when_no_key(monkeypatch):
    """Self-hosted fastCRW without a key sends no Authorization header."""
    monkeypatch.delenv("CRW_API_KEY", raising=False)
    monkeypatch.setenv("CRW_API_URL", "http://localhost:3000")
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "false")

    payload = {"success": True, "data": []}
    recorder = []

    with patch(
        "open_deep_research.utils.aiohttp.ClientSession",
        _make_session_factory(payload, recorder),
    ):
        results = asyncio.run(crw_search_async(["q"], config=None))

    assert results[0]["results"] == []
    call = recorder[0]
    assert call["url"] == "http://localhost:3000/v1/search"
    assert "Authorization" not in call["headers"]


def test_get_crw_api_key_from_env(monkeypatch):
    """The fastCRW key is read from CRW_API_KEY by default."""
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "false")
    monkeypatch.setenv("CRW_API_KEY", "abc123")
    assert get_crw_api_key(None) == "abc123"


def test_get_crw_base_url_defaults_to_cloud(monkeypatch):
    """The base URL defaults to the managed cloud when unset."""
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "false")
    monkeypatch.delenv("CRW_API_URL", raising=False)
    assert get_crw_base_url(None) == CRW_DEFAULT_BASE_URL


def test_get_search_tool_returns_crw_tool():
    """The dispatch returns the crw_search tool for SearchAPI.CRW."""
    tools = asyncio.run(get_search_tool(SearchAPI.CRW))
    assert len(tools) == 1
    assert tools[0].name == "crw_search"
    assert tools[0].metadata["name"] == "web_search"
