"""Regression tests for runtime packages included in distribution builds."""

import ast
import re
import unittest
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


class WheelContentsTest(unittest.TestCase):
    """Verify runtime modules referenced by deployment config are packaged."""

    def test_includes_security_auth_module(self) -> None:
        pyproject = PYPROJECT.read_text()
        setuptools_section = re.search(
            r"(?ms)^\[tool\.setuptools\]\n(.*?)(?=^\[)",
            pyproject,
        )
        package_dir_section = re.search(
            r"(?ms)^\[tool\.setuptools\.package-dir\]\n(.*?)(?=^\[)",
            pyproject,
        )

        self.assertIsNotNone(setuptools_section)
        self.assertIsNotNone(package_dir_section)

        packages_match = re.search(
            r"(?m)^packages\s*=\s*(\[.*\])$",
            setuptools_section.group(1),
        )
        self.assertIsNotNone(packages_match)
        packages = ast.literal_eval(packages_match.group(1))

        package_dirs = dict(
            re.findall(
                r'(?m)^"([^"]+)"\s*=\s*"([^"]+)"$',
                package_dir_section.group(1),
            )
        )

        self.assertIn("security", packages)
        self.assertEqual(package_dirs.get("security"), "src/security")


if __name__ == "__main__":
    unittest.main()
