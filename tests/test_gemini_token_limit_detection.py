import ast
from pathlib import Path


def _load_gemini_classifier():
    path = (
        Path(__file__).parents[1]
        / "src"
        / "open_deep_research"
        / "utils.py"
    )
    module = ast.parse(path.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_check_gemini_token_limit"
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[]),
            ),
            filename=str(path),
            mode="exec",
        ),
        namespace,
    )
    return namespace["_check_gemini_token_limit"]


def _exception(name: str, module: str, message: str) -> Exception:
    exception_type = type(name, (Exception,), {"__module__": module})
    return exception_type(message)


def _classify(exception: Exception) -> bool:
    classifier = _load_gemini_classifier()
    return classifier(exception, str(exception).lower())


def test_gemini_quota_exhaustion_is_not_token_overflow() -> None:
    error = _exception(
        "ResourceExhausted",
        "google.api_core.exceptions",
        "429 Resource exhausted. Please check your quota.",
    )

    assert _classify(error) is False


def test_gemini_resource_exhausted_with_context_message_is_overflow() -> None:
    error = _exception(
        "ResourceExhausted",
        "google.api_core.exceptions",
        "The input token count exceeds the maximum context window.",
    )

    assert _classify(error) is True


def test_gemini_invalid_argument_with_context_message_is_overflow() -> None:
    error = _exception(
        "InvalidArgument",
        "google.api_core.exceptions",
        "The input token count exceeds the maximum context window.",
    )

    assert _classify(error) is True


def test_non_google_context_error_is_not_gemini_overflow() -> None:
    error = _exception(
        "InvalidArgument",
        "other_provider",
        "The input token count exceeds the maximum context window.",
    )

    assert _classify(error) is False
