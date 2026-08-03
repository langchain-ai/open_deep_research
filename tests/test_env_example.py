from pathlib import Path

BOOLEAN_ENV_VARS = {"GET_API_KEYS_FROM_CONFIG", "LANGSMITH_TRACING"}


def test_boolean_environment_examples_have_valid_defaults() -> None:
    env_path = Path(__file__).parents[1] / ".env.example"
    values = dict(
        line.split("=", maxsplit=1)
        for line in env_path.read_text().splitlines()
        if line and not line.startswith("#")
    )

    for variable in BOOLEAN_ENV_VARS:
        assert values[variable].lower() in {"true", "false"}
