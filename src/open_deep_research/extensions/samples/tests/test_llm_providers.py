"""Tests for LLM provider configuration and client creation."""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestLLMProviderEnum:
    """Test LLMProvider enum has only Azure and Gemini."""

    def test_enum_has_gemini(self):
        from src.configuration import LLMProvider
        assert LLMProvider.GEMINI.value == "gemini"

    def test_enum_has_azure(self):
        from src.configuration import LLMProvider
        assert LLMProvider.AZURE.value == "azure"

    def test_enum_no_google(self):
        """Ensure 'google' is not a provider."""
        from src.configuration import LLMProvider
        values = [e.value for e in LLMProvider]
        assert "google" not in values

    def test_all_providers_count(self):
        from src.configuration import LLMProvider
        assert len(LLMProvider) == 2

    def test_no_removed_providers(self):
        """Ensure removed providers are gone."""
        from src.configuration import LLMProvider
        values = [e.value for e in LLMProvider]
        for removed in ["openai", "anthropic", "groq", "mistral", "huggingface", "ollama"]:
            assert removed not in values


class TestConfigurationDefaults:
    """Test Configuration default model mappings."""

    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}, clear=False)
    def test_gemini_default_model(self):
        from src.configuration import Configuration
        config = Configuration()
        assert config.llm_model == "gemini-2.5-flash"

    @patch.dict(os.environ, {"LLM_PROVIDER": "azure"}, clear=False)
    def test_default_provider_is_azure(self):
        from src.configuration import Configuration, LLMProvider
        config = Configuration()
        assert config.llm_provider == LLMProvider.AZURE


class TestModelConfigs:
    """Test MODEL_CONFIGS dictionary in llm_clients."""

    def test_gemini_in_model_configs(self):
        from llm_clients import MODEL_CONFIGS
        assert "gemini" in MODEL_CONFIGS

    def test_azure_in_model_configs(self):
        from llm_clients import MODEL_CONFIGS
        assert "azure" in MODEL_CONFIGS

    def test_only_two_providers_in_model_configs(self):
        from llm_clients import MODEL_CONFIGS
        assert len(MODEL_CONFIGS) == 2

    def test_no_removed_providers_in_model_configs(self):
        from llm_clients import MODEL_CONFIGS
        for removed in ["openai", "anthropic", "groq", "mistral", "huggingface", "ollama", "sfrgateway", "sambnova"]:
            assert removed not in MODEL_CONFIGS

    def test_google_not_in_model_configs(self):
        """Ensure 'google' key was replaced with 'gemini'."""
        from llm_clients import MODEL_CONFIGS
        assert "google" not in MODEL_CONFIGS

    def test_gemini_default_model(self):
        from llm_clients import MODEL_CONFIGS
        assert MODEL_CONFIGS["gemini"]["default_model"] == "gemini-2.5-flash"

    def test_each_config_has_required_keys(self):
        from llm_clients import MODEL_CONFIGS
        for provider, config in MODEL_CONFIGS.items():
            assert "default_model" in config, f"{provider} missing default_model"
            assert "available_models" in config, f"{provider} missing available_models"


class TestLLMClientCreation:
    """Test get_llm_client creates valid clients for supported providers."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False)
    def test_gemini_client_creation(self):
        from llm_clients import get_llm_client
        client = get_llm_client("gemini", "gemini-2.5-flash")
        assert client is not None

    def test_missing_gemini_key_raises(self):
        from llm_clients import get_llm_client
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                get_llm_client("gemini", "gemini-2.5-flash")

    def test_unsupported_provider_raises(self):
        from llm_clients import get_llm_client
        with pytest.raises(ValueError, match="Unsupported provider"):
            get_llm_client("nonexistent_provider", "some-model")

    def test_removed_provider_raises(self):
        """Removed providers should raise ValueError."""
        from llm_clients import get_llm_client
        for removed in ["openai", "anthropic", "groq", "mistral"]:
            with pytest.raises(ValueError, match="Unsupported provider"):
                get_llm_client(removed, "some-model")


class TestExtensionsLLMFactory:
    """Test the extensions LLM factory."""

    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}, clear=False)
    def test_factory_creates_client(self):
        from extensions.utils.llm_factory import get_extensions_llm
        client = get_extensions_llm()
        assert client is not None

    @patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}, clear=False)
    def test_factory_reads_provider_from_env(self):
        from extensions.utils.llm_factory import get_extensions_llm
        client = get_extensions_llm()
        assert client is not None
