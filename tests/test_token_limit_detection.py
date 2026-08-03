"""Regression tests for token-limit exception classification."""

import ast
import unittest
from pathlib import Path


def load_openai_token_limit_checker():
    """Load the pure checker without importing optional search dependencies."""
    source_path = (
        Path(__file__).resolve().parents[1] / "src" / "open_deep_research" / "utils.py"
    )
    tree = ast.parse(source_path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_check_openai_token_limit"
    )
    namespace = {}
    exec(
        compile(ast.Module([function], type_ignores=[]), source_path, "exec"), namespace
    )
    return namespace["_check_openai_token_limit"]


def make_openai_error(message: str, *, code: str, error_type: str) -> Exception:
    """Create an OpenAI-shaped exception without importing the SDK."""
    error_class = type("BadRequestError", (Exception,), {"__module__": "openai"})
    error = error_class(message)
    error.code = code
    error.type = error_type
    return error


class OpenAITokenLimitDetectionTest(unittest.TestCase):
    """Verify only token-related invalid requests trigger truncation logic."""

    def setUp(self) -> None:
        self.check = load_openai_token_limit_checker()

    def test_invalid_value_is_not_a_token_limit(self) -> None:
        error = make_openai_error(
            "Invalid value: 'function_call'",
            code="invalid_value",
            error_type="invalid_request_error",
        )

        self.assertFalse(self.check(error, str(error).lower()))

    def test_context_length_code_is_a_token_limit(self) -> None:
        error = make_openai_error(
            "Request exceeds the model context window",
            code="context_length_exceeded",
            error_type="invalid_request_error",
        )

        self.assertTrue(self.check(error, str(error).lower()))

    def test_token_message_is_a_token_limit(self) -> None:
        error = make_openai_error(
            "Maximum context length exceeded",
            code="invalid_request_error",
            error_type="invalid_request_error",
        )

        self.assertTrue(self.check(error, str(error).lower()))


if __name__ == "__main__":
    unittest.main()
