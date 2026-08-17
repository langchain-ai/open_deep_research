from open_deep_research.utils import get_api_key_for_model, get_base_url_for_model


def test_get_api_key_for_deepseek_model_from_environment(monkeypatch):
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "false")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")

    api_key = get_api_key_for_model("deepseek:deepseek-v4-flash", {})

    assert api_key == "deepseek-test-key"


def test_get_api_key_for_deepseek_model_from_config(monkeypatch):
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "true")
    config = {
        "configurable": {
            "apiKeys": {
                "DEEPSEEK_API_KEY": "deepseek-config-key",
            }
        }
    }

    api_key = get_api_key_for_model("deepseek:deepseek-v4-pro", config)

    assert api_key == "deepseek-config-key"


def test_get_base_url_for_deepseek_model_from_environment(monkeypatch):
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "false")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    base_url = get_base_url_for_model("deepseek:deepseek-v4-flash", {})

    assert base_url == "https://api.deepseek.com"


def test_get_base_url_for_non_deepseek_model_is_none(monkeypatch):
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "false")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    base_url = get_base_url_for_model("openai:gpt-4.1-mini", {})

    assert base_url is None
