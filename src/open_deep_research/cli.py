"""Command line interface for Open Deep Research."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from .agent import run_multi_agent


def main() -> None:
    """Entry point for the ``odr`` command."""
    parser = argparse.ArgumentParser(description="Run the multi-agent assistant from the terminal")
    parser.add_argument("prompt_file", help="Path to a file containing the prompt")
    parser.add_argument(
        "--auto-accept-plan",
        action="store_true",
        help="Automatically accept the generated plan without confirmation",
    )
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text()
    report = run_multi_agent(prompt, auto_accept_plan=args.auto_accept_plan)
    console = Console()
    console.print(report)


if __name__ == "__main__":  # pragma: no cover
    main()
