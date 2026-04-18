"""Batch runner for Turn-3 multimodal probe.

Reads v2 reports from draco_eval/reports_<slug>/ and writes outputs to
draco_eval/turn3_outputs_<slug>/. The slug is derived from --reports-model.

Usage:
    # GPT-4.1 (default) — single task
    python draco_eval/scripts/run_turn3.py --task task_001

    # GPT-4.1 — all tasks
    python draco_eval/scripts/run_turn3.py

    # GPT-4.1-mini
    python draco_eval/scripts/run_turn3.py --reports-model openai:gpt-4.1-mini

    # Gemini 2.5 Pro
    python draco_eval/scripts/run_turn3.py --reports-model google_vertexai:gemini-2.5-pro

    # Overwrite existing outputs
    python draco_eval/scripts/run_turn3.py --overwrite

    # Override directories explicitly
    python draco_eval/scripts/run_turn3.py \\
        --reports-dir draco_eval/reports_gpt4.1 \\
        --output-dir  draco_eval/turn3_outputs_gpt4.1

Output file naming: {task_id}_turn3.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Make sure turn3_probe is importable when running from repo root
_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from turn3_probe import run_turn3_probe  # noqa: E402

DRACO_DIR = _SCRIPTS_DIR.parent


def _model_slug(model: str) -> str:
    """'openai:gpt-4.1' → 'gpt4.1', 'google_genai:gemini-2.5-pro' → 'gemini2.5pro'"""
    return model.split(":")[-1].replace("-", "")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Turn-3 multimodal probe runner")
    parser.add_argument(
        "--reports-model",
        default="openai:gpt-4.1",
        help="Agent model that generated the v2 reports (e.g. 'openai:gpt-4.1'). "
             "Controls which reports_<slug>/ and turn3_outputs_<slug>/ folders are used.",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Override: directory containing *_v2.md report files (default: reports_<slug>/)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override: directory to write {task_id}_turn3.json outputs (default: turn3_outputs_<slug>/)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Run a single task by ID (e.g. task_001). Omit to run all.",
    )
    parser.add_argument(
        "--model",
        default="openai:gpt-4.1",
        help="Vision/probe model for Turn-3 (default: openai:gpt-4.1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run tasks that already have output files",
    )
    return parser.parse_args()


async def run_all(
    report_files: list[Path],
    output_dir: Path,
    log_path: Path,
    model: str,
    overwrite: bool,
    image_save_dir: Path,
) -> None:
    total = len(report_files)
    done = skipped = failed = 0

    for i, report_file in enumerate(report_files, 1):
        task_id = report_file.stem.replace("_v2", "")
        output_path = output_dir / f"{task_id}_turn3.json"

        if output_path.exists() and not overwrite:
            print(f"[{i}/{total}] skip   {task_id}  (output exists, use --overwrite to re-run)")
            skipped += 1
            continue

        print(f"[{i}/{total}] running {task_id}...", end="  ", flush=True)
        t0 = time.monotonic()

        try:
            report_text = report_file.read_text(encoding="utf-8")

            task_file = DRACO_DIR / "tasks" / f"{task_id}.json"
            task_prompt = ""
            if task_file.exists():
                task_data = json.loads(task_file.read_text(encoding="utf-8"))
                task_prompt = task_data.get("prompt", "")

            result = await run_turn3_probe(
                report_text=report_text,
                task_id=task_id,
                task_prompt=task_prompt,
                report_path=str(report_file.resolve()),
                log_path=log_path,
                image_save_dir=image_save_dir,
                model=model,
            )
            output_path.write_text(
                json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            elapsed = time.monotonic() - t0
            vision_calls = result.metadata.get("vision_calls", 0)

            if result.no_image_found:
                print(f"done ({elapsed:.1f}s)  no_image_found=True  vision_calls={vision_calls}")
                print(f"         reason: {result.reason_if_none}")
            else:
                print(f"done ({elapsed:.1f}s)  image found  vision_calls={vision_calls}")
                print(f"         image  : {result.metadata.get('image_url', '')}")
                print(f"         saved  : {result.local_image_path}")
                print(f"         shows  : {result.what_it_shows[:120]}...")

            done += 1

        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(f"ERROR ({elapsed:.1f}s)  {exc}")
            # Write a minimal failure record so the batch doesn't silently lose tasks
            failure = {
                "task_id": task_id,
                "error": str(exc),
                "no_image_found": True,
                "reason_if_none": f"Probe raised an exception: {exc}",
            }
            output_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
            failed += 1

    print(f"\n{'='*55}")
    print(f"Batch complete: {done} done, {skipped} skipped, {failed} failed")
    print(f"Outputs → {output_dir}")
    print(f"Log     → {log_path}")


def main() -> None:
    args = _parse_args()

    reports_slug = _model_slug(args.reports_model)
    reports_dir = Path(args.reports_dir) if args.reports_dir else DRACO_DIR / f"reports_{reports_slug}"
    output_dir = Path(args.output_dir) if args.output_dir else DRACO_DIR / f"turn3_outputs_{reports_slug}"
    log_dir = DRACO_DIR / f"logs_{reports_slug}"
    image_save_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "turn3_probe.log"

    if args.task:
        report_files = sorted(reports_dir.glob(f"{args.task}_v2.md"))
        if not report_files:
            print(f"Error: no file matching '{args.task}_v2.md' in {reports_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        report_files = sorted(reports_dir.glob("*_v2.md"))

    if not report_files:
        print(f"No *_v2.md files found in {reports_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Turn-3 multimodal probe")
    print(f"  Reports dir : {reports_dir}  ({len(report_files)} files)")
    print(f"  Output dir  : {output_dir}")
    print(f"  Images dir  : {image_save_dir}")
    print(f"  Model       : {args.model}")
    print(f"  Log         : {log_path}")
    print()

    asyncio.run(
        run_all(
            report_files=report_files,
            output_dir=output_dir,
            log_path=log_path,
            model=args.model,
            overwrite=args.overwrite,
            image_save_dir=image_save_dir,
        )
    )


if __name__ == "__main__":
    main()
