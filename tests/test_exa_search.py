"""Unit tests for the Exa search provider."""

from unittest.mock import MagicMock, patch

import pytest

from open_deep_research.configuration import SearchAPI
from open_deep_research.utils import (
    _format_exa_response,
    exa_search,
    exa_search_async,
    get_exa_api_key,
    get_search_tool,
)


def _make_result(**kwargs):
    """Build an attribute-style result object similar to the Exa SDK's typed model."""
    return MagicMock(spec_set=list(kwargs.keys()), **kwargs)


class TestFormatExaResponse:
    """Tests for _format_exa_response normalization and content fallback."""

    def test_full_content_uses_summary_as_snippet(self):
        response = MagicMock()
        response.results = [
            _make_result(
                title="Hello",
                url="https://example.com/a",
                text="Long body text",
                summary="One-line summary",
                highlights=["highlight one", "highlight two"],
                score=0.9,
            )
        ]

        out = _format_exa_response("hello", response)

        assert out["query"] == "hello"
        assert len(out["results"]) == 1
        result = out["results"][0]
        assert result["title"] == "Hello"
        assert result["url"] == "https://example.com/a"
        assert result["content"] == "One-line summary"
        assert "Long body text" in result["raw_content"]
        assert "highlight one" in result["raw_content"]
        assert result["score"] == 0.9

    def test_only_text_falls_back_to_text_excerpt(self):
        response = MagicMock()
        response.results = [
            _make_result(
                title="Title",
                url="https://example.com/b",
                text="x" * 2000,
                summary=None,
                highlights=None,
                score=None,
            )
        ]

        out = _format_exa_response("q", response)
        result = out["results"][0]

        # Snippet uses first 1000 chars of text when summary/highlights are absent
        assert result["content"] == "x" * 1000
        assert result["raw_content"] == "x" * 2000

    def test_only_highlights_used_as_snippet(self):
        response = MagicMock()
        response.results = [
            _make_result(
                title="t",
                url="https://example.com/c",
                text=None,
                summary=None,
                highlights=["only highlight"],
                score=None,
            )
        ]

        out = _format_exa_response("q", response)
        assert out["results"][0]["content"] == "only highlight"
        assert out["results"][0]["raw_content"] == "only highlight"

    def test_no_content_yields_empty_snippet_and_none_raw(self):
        response = MagicMock()
        response.results = [
            _make_result(
                title="t",
                url="https://example.com/d",
                text=None,
                summary=None,
                highlights=None,
                score=None,
            )
        ]

        out = _format_exa_response("q", response)
        assert out["results"][0]["content"] == ""
        assert out["results"][0]["raw_content"] is None

    def test_dict_results_are_supported(self):
        response = {
            "results": [
                {
                    "title": "Dict Title",
                    "url": "https://example.com/e",
                    "text": "body",
                    "summary": "sum",
                    "highlights": ["h"],
                    "score": 0.42,
                }
            ]
        }

        out = _format_exa_response("q", response)
        result = out["results"][0]
        assert result["title"] == "Dict Title"
        assert result["content"] == "sum"

    def test_duplicate_urls_are_skipped(self):
        response = MagicMock()
        response.results = [
            _make_result(
                title="first",
                url="https://example.com/dup",
                text="t1",
                summary="s1",
                highlights=[],
                score=None,
            ),
            _make_result(
                title="second",
                url="https://example.com/dup",
                text="t2",
                summary="s2",
                highlights=[],
                score=None,
            ),
        ]

        out = _format_exa_response("q", response)
        assert len(out["results"]) == 1
        assert out["results"][0]["title"] == "first"

    def test_results_without_url_are_skipped(self):
        response = MagicMock()
        response.results = [
            _make_result(
                title="no url",
                url=None,
                text="t",
                summary="s",
                highlights=[],
                score=None,
            )
        ]

        out = _format_exa_response("q", response)
        assert out["results"] == []


class TestGetExaApiKey:
    """Tests for get_exa_api_key env vs config-driven lookup."""

    def test_returns_value_from_env(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "env-key")
        monkeypatch.delenv("GET_API_KEYS_FROM_CONFIG", raising=False)
        assert get_exa_api_key(config=None) == "env-key"

    def test_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("GET_API_KEYS_FROM_CONFIG", raising=False)
        assert get_exa_api_key(config=None) is None

    def test_returns_value_from_config_when_flag_set(self, monkeypatch):
        monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "true")
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        cfg = {"configurable": {"apiKeys": {"EXA_API_KEY": "cfg-key"}}}
        assert get_exa_api_key(cfg) == "cfg-key"


class TestExaSearchAsync:
    """Tests for the async wrapper around the Exa SDK."""

    @pytest.mark.asyncio
    async def test_raises_when_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("GET_API_KEYS_FROM_CONFIG", raising=False)
        with pytest.raises(ValueError, match="EXA_API_KEY"):
            await exa_search_async(["q"], config=None)

    @pytest.mark.asyncio
    async def test_rejects_conflicting_domain_filters(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "k")
        with pytest.raises(ValueError, match="include_domains and exclude_domains"):
            await exa_search_async(
                ["q"],
                include_domains=["a.com"],
                exclude_domains=["b.com"],
                config=None,
            )

    @pytest.mark.asyncio
    async def test_calls_sdk_and_sets_tracking_header(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "k")
        fake_response = MagicMock()
        fake_response.results = []

        fake_client = MagicMock()
        fake_client.headers = {}
        fake_client.search_and_contents.return_value = fake_response

        with patch("open_deep_research.utils.Exa", return_value=fake_client) as ctor:
            results = await exa_search_async(
                ["alpha", "beta"],
                max_results=3,
                search_type="neural",
                category="research paper",
                include_domains=["arxiv.org"],
                start_published_date="2025-01-01",
                config=None,
            )

        ctor.assert_called_once_with(api_key="k")
        assert fake_client.headers["x-exa-integration"] == "open-deep-research"
        assert fake_client.search_and_contents.call_count == 2

        _, kwargs = fake_client.search_and_contents.call_args_list[0]
        assert kwargs["num_results"] == 3
        assert kwargs["type"] == "neural"
        assert kwargs["text"] is True
        assert kwargs["highlights"] is True
        assert kwargs["summary"] is True
        assert kwargs["category"] == "research paper"
        assert kwargs["include_domains"] == ["arxiv.org"]
        assert kwargs["start_published_date"] == "2025-01-01"
        assert "exclude_domains" not in kwargs
        assert "end_published_date" not in kwargs

        assert len(results) == 2
        assert results[0]["query"] == "alpha"
        assert results[1]["query"] == "beta"


class TestGetSearchTool:
    """Tests for the search tool dispatcher."""

    @pytest.mark.asyncio
    async def test_exa_returns_exa_tool(self):
        tools = await get_search_tool(SearchAPI.EXA)
        assert len(tools) == 1
        assert tools[0] is exa_search
        assert tools[0].metadata.get("name") == "web_search"

    @pytest.mark.asyncio
    async def test_none_returns_empty(self):
        assert await get_search_tool(SearchAPI.NONE) == []
