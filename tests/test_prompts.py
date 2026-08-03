"""Regression tests for evaluation prompt templates."""

import unittest

from tests.prompts import CORRECTNESS_PROMPT


class CorrectnessPromptTest(unittest.TestCase):
    """Verify that correctness evaluations receive all comparison inputs."""

    def test_includes_question_report_answer_and_date(self) -> None:
        rendered = CORRECTNESS_PROMPT.format(
            user_question="QUESTION_SENTINEL",
            report="REPORT_SENTINEL",
            answer="ANSWER_SENTINEL",
            today="DATE_SENTINEL",
        )

        self.assertIn("QUESTION_SENTINEL", rendered)
        self.assertIn("REPORT_SENTINEL", rendered)
        self.assertIn("ANSWER_SENTINEL", rendered)
        self.assertIn("DATE_SENTINEL", rendered)


if __name__ == "__main__":
    unittest.main()
