"""Regression tests for the research compression prompt."""

import unittest

from open_deep_research.prompts import compress_research_system_prompt


class CompressionPromptTest(unittest.TestCase):
    """Verify that the prompt specifies one canonical sources format."""

    def test_output_format_uses_sources_heading(self) -> None:
        output_format = compress_research_system_prompt.split(
            "<Output Format>", maxsplit=1
        )[1].split("</Output Format>", maxsplit=1)[0]

        self.assertIn("### Sources", output_format)
        self.assertNotIn("**List of All Relevant Sources", output_format)


if __name__ == "__main__":
    unittest.main()
