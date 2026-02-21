"""Unit tests for llm_factory.py - get_extensions_llm function."""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Mock the llm_clients module before importing get_extensions_llm.
# get_extensions_llm does `from llm_clients import get_llm_client, MODEL_CONFIGS`
# inside its function body, so we mock the entire llm_clients module.

MOCK_MODEL_CONFIGS = {
    "azure": {"default_model": "gpt-4o"},
    "gemini": {"default_model": "gemini-2.5-flash"},
}


def _call_get_extensions_llm(mock_get_client, **kwargs):
    """Helper that patches llm_clients and calls get_extensions_llm."""
    mock_module = MagicMock()
    mock_module.get_llm_client = mock_get_client
    mock_module.MODEL_CONFIGS = MOCK_MODEL_CONFIGS

    with patch.dict("sys.modules", {"llm_clients": mock_module}):
        # Re-import to ensure the lazy import inside the function uses our mock
        import importlib
        import extensions.utils.llm_factory as factory_mod
        importlib.reload(factory_mod)
        return factory_mod.get_extensions_llm(**kwargs)


class TestGetExtensionsLLM:
    """Test the get_extensions_llm factory function."""

    def test_explicit_provider_and_model(self):
        mock_get_client = MagicMock(return_value=MagicMock())
        _call_get_extensions_llm(mock_get_client, provider="gemini", model="gemini-2.5-pro")
        mock_get_client.assert_called_once_with("gemini", "gemini-2.5-pro")

    def test_falls_back_to_env_provider(self):
        mock_get_client = MagicMock(return_value=MagicMock())
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "LLM_MODEL": "gemini-2.5-flash"}):
            _call_get_extensions_llm(mock_get_client)
            mock_get_client.assert_called_once_with("gemini", "gemini-2.5-flash")

    def test_falls_back_to_azure_default(self):
        mock_get_client = MagicMock(return_value=MagicMock())
        # Remove env vars to force default
        env_backup = {}
        for key in ["LLM_PROVIDER", "LLM_MODEL"]:
            env_backup[key] = os.environ.pop(key, None)
        try:
            _call_get_extensions_llm(mock_get_client)
            call_args = mock_get_client.call_args
            # Provider should default to "azure"
            assert call_args[0][0] == "azure"
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val

    def test_model_falls_back_to_provider_default(self):
        mock_get_client = MagicMock(return_value=MagicMock())
        old_model = os.environ.pop("LLM_MODEL", None)
        try:
            _call_get_extensions_llm(mock_get_client, provider="gemini")
            call_args = mock_get_client.call_args
            # Model should default to gemini-2.5-flash
            assert call_args[0][1] == "gemini-2.5-flash"
        finally:
            if old_model is not None:
                os.environ["LLM_MODEL"] = old_model

    def test_explicit_params_override_env(self):
        mock_get_client = MagicMock(return_value=MagicMock())
        with patch.dict(os.environ, {"LLM_PROVIDER": "azure", "LLM_MODEL": "gpt-4o"}):
            _call_get_extensions_llm(mock_get_client, provider="gemini", model="gemini-2.5-pro")
            mock_get_client.assert_called_once_with("gemini", "gemini-2.5-pro")

    def test_returns_llm_client(self):
        mock_llm = MagicMock()
        mock_get_client = MagicMock(return_value=mock_llm)
        result = _call_get_extensions_llm(mock_get_client, provider="azure", model="gpt-4o")
        assert result is mock_llm
