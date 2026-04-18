"""Evaluate Turn-3 multimodal probe outputs using an LLM judge.

For each task, loads the turn3 JSON + saved image + relevant report section,
then evaluates two criteria scored as +1 / 0:

  1. description_accuracy  — does what_it_shows accurately describe the image?
  2. section_fit           — does the image genuinely add value to the named section?

Both criteria are vision calls (gpt-4.1, multimodal).
section_fit uses a two-step process: independent assessment first, then agent
explanation evaluated in light of that assessment.

Scoring per task:
  task_score = (description_accuracy + section_fit) / 2   → 0.0, 0.5, or 1.0
  no_image_found tasks receive task_score = 0.0

Aggregate metrics across all tasks (computed by aggregate_scores()):
  coverage  = tasks with image found / total tasks
  quality   = mean task_score over tasks where image was found
  overall   = mean task_score over ALL tasks  (coverage x quality, comparable across models)

Reads from draco_eval/turn3_outputs_<slug>/ and writes to
draco_eval/turn3_evaluations_<slug>/. The slug is derived from --model.

Usage:
    # GPT-4.1 (default) — all tasks
    uv run python draco_eval/scripts/evaluate_turn3.py

    # GPT-4.1 — single task
    uv run python draco_eval/scripts/evaluate_turn3.py --task task_001

    # GPT-4.1-mini
    uv run python draco_eval/scripts/evaluate_turn3.py --model openai:gpt-4.1-mini

    # Gemini 2.5 Pro
    uv run python draco_eval/scripts/evaluate_turn3.py --model google_vertexai:gemini-2.5-pro

    # Overwrite existing evaluations
    uv run python draco_eval/scripts/evaluate_turn3.py --overwrite

    # Print aggregate metrics only (no re-evaluation)
    uv run python draco_eval/scripts/evaluate_turn3.py --aggregate
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from turn3_probe import extract_section  # noqa: E402

DRACO_DIR = _SCRIPTS_DIR.parent

JUDGE_MODEL = "gpt-5.2-2025-12-11"   # uses Responses API for vision support


def _model_slug(model: str) -> str:
    """'openai:gpt-4.1' → 'gpt4.1', 'google_genai:gemini-2.5-pro' → 'gemini2.5pro'"""
    return model.split(":")[-1].replace("-", "")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are evaluating a multimodal AI probe that was given a research report and \
tasked with finding a relevant image from the report's cited sources.

You will receive one criterion at a time and must return a JSON verdict.

OUTPUT FORMAT — return exactly:
{
  "explanation": "<your reasoning, under 80 words>",
  "score": 1 or 0
}

SCORING:
- 1: the criterion IS satisfied
- 0: the criterion IS NOT satisfied

Be strict about factual accuracy. Be flexible about exact wording.
Return only raw JSON starting with {, no backticks.
"""

# ---------------------------------------------------------------------------
# Criterion 1 — description accuracy
# Vision call: does what_it_shows accurately describe the image?
# ---------------------------------------------------------------------------

_DESC_ACCURACY_PROMPT = """\
<criterion>
The description in what_it_shows accurately describes what the image actually \
shows. It should not hallucinate content that is absent from the image, and \
should not omit the most visually prominent elements.
</criterion>

<what_it_shows_claim>
{what_it_shows}
</what_it_shows_claim>

The image to evaluate is attached. Does the claim above accurately describe it?
Return your verdict as JSON.
"""

# ---------------------------------------------------------------------------
# Criterion 2 — section fit (two-step)
#
# Step 1: Independent assessment — judge sees image + section content + task
#         query. No agent explanation. Judge forms its own view first.
#
# Step 2: Agent explanation assessment — judge sees Step 1 result + agent's
#         where_it_fits. Validates or overrides based on argument quality.
# ---------------------------------------------------------------------------

_SECTION_FIT_INDEPENDENT_PROMPT = """\
You are independently assessing whether an image retrieved from a cited webpage \
is genuinely useful to a reader of a specific section of a research report.

ORIGINAL RESEARCH QUESTION:
{task_prompt}

SECTION CONTENT:
{section_text}

---
The image is attached. Assess it WITHOUT reference to any agent explanation.

The core question is: does this image show something a reader could not \
adequately get from reading the section text alone?

The right visual type depends on what the section is claiming:
- For quantitative claims (metrics, margins, benchmarks, statistical results): \
charts, plots, tables, or dashboards qualify. A decorative photo does not.
- For software or integration claims (APIs, protocols, configuration, interfaces): \
real interface screenshots or configuration screens qualify, even without numbers.
- For physical specification or hardware claims (machine dimensions, tooling, \
mechanical features): technical product photos showing the relevant feature \
qualify. Generic exterior shots without visible technical detail do not.
- For process or workflow claims: flow diagrams, architecture diagrams, or \
annotated screenshots qualify.
- For location or property claims in a financial context: exterior photos of \
a building or property do NOT qualify — they add no analytical value.

Apply whichever category fits the section. An image can be useful even if it \
contains no numbers, as long as it shows something specific and relevant that \
the text cannot fully convey.

OUTPUT FORMAT — return exactly:
{{"independently_useful": true or false, "reasoning": "<your assessment, under 80 words>"}}

Return only raw JSON starting with {{, no backticks.
"""

_SECTION_FIT_VERDICT_PROMPT = """\
You have already independently assessed whether an image is useful for a \
research report section. Now evaluate the agent's explanation for why it \
selected this image.

YOUR INDEPENDENT ASSESSMENT:
  Useful: {independently_useful}
  Reasoning: {independent_reasoning}

AGENT'S EXPLANATION (where_it_fits):
{where_it_fits}

Decision rules:

If your independent assessment found the image IS useful:
- Score 1 if the agent's explanation correctly identifies what the image \
shows and why it fits the section. Be flexible about exact wording.
- Score 0 if the agent's explanation is factually wrong about the image \
content or names a section clearly unrelated to what the image shows.

If your independent assessment found the image is NOT useful:
- Read the agent's explanation critically. Does it point to something \
specific and non-obvious in the image that you may have missed — a concrete \
feature, a labelled detail, or a specific interface element that directly \
addresses the section's claim?
- Score 1 ONLY if the explanation reveals something substantive that genuinely \
changes your assessment. The bar is high.
- Score 0 if the explanation is vague or post-hoc — using language like \
"illustrates", "represents", or "visually supports" without pointing to \
specific content in the image that the text does not already convey.

OUTPUT FORMAT — return exactly:
{{"explanation": "<your reasoning, under 80 words>", "score": 1 or 0}}

Return only raw JSON starting with {{, no backticks.
"""


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------

def _encode_image(image_path: str) -> tuple[str, str]:
    """Return (base64_data, mime_type) for a local image file."""
    path = Path(image_path)
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return data, mime


def _parse_scored_verdict(raw: str) -> dict:
    """Parse a {score, explanation} JSON response from the judge."""
    parsed = json.loads(raw)
    score = parsed.get("score")
    if score not in (0, 1):
        raise ValueError(f"Unexpected score {score!r}. Raw: {raw}")
    return {"score": int(score), "explanation": parsed.get("explanation", "")}


def _judge_with_image(client: OpenAI, prompt: str, image_path: str) -> dict:
    """Vision call with system prompt via Responses API. Returns {score, explanation}."""
    b64, mime = _encode_image(image_path)
    response = client.responses.create(
        model=JUDGE_MODEL,
        instructions=_SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
                ],
            }
        ],
        reasoning={"effort": "none"},
        temperature=0,
        text={"format": {"type": "json_object"}},
    )
    return _parse_scored_verdict(response.output_text)


def _judge_section_fit_two_step(
    client: OpenAI,
    image_path: str,
    section_text: str,
    task_prompt: str,
    where_it_fits: str,
) -> dict:
    """Two-step section fit evaluation.

    Step 1: Independent vision assessment — no agent explanation visible.
    Step 2: Agent explanation evaluated against Step 1 result — text only.

    Returns {score, explanation, independent_assessment}.
    independent_assessment carries Step 1 result for transparency in output.
    """
    b64, mime = _encode_image(image_path)

    # Step 1 — independent vision assessment, no agent explanation visible
    step1_prompt = _SECTION_FIT_INDEPENDENT_PROMPT.format(
        task_prompt=task_prompt,
        section_text=section_text,
    )
    step1_response = client.responses.create(
        model=JUDGE_MODEL,
        instructions=_SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": step1_prompt},
                    {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
                ],
            }
        ],
        reasoning={"effort": "none"},
        temperature=0,
        text={"format": {"type": "json_object"}},
    )
    step1_raw = json.loads(step1_response.output_text)
    independently_useful: bool = bool(step1_raw.get("independently_useful", False))
    independent_reasoning: str = step1_raw.get("reasoning", "")

    # Step 2 — agent explanation verdict, text only, no image needed
    step2_prompt = _SECTION_FIT_VERDICT_PROMPT.format(
        independently_useful=independently_useful,
        independent_reasoning=independent_reasoning,
        where_it_fits=where_it_fits,
    )
    step2_response = client.responses.create(
        model=JUDGE_MODEL,
        instructions=_SYSTEM_PROMPT,
        input=step2_prompt,
        reasoning={"effort": "none"},
        temperature=0,
        text={"format": {"type": "json_object"}},
    )
    verdict = _parse_scored_verdict(step2_response.output_text)

    # If independent assessment said NOT useful but agent explanation convinced
    # the judge, cap score at 0.5 — the image passes but is weaker than one that
    # is independently obvious. Prevents agent explanation from fully overriding
    # a clear independent "not useful" verdict.
    independent_override = not independently_useful and verdict["score"] == 1
    final_score = 0.5 if independent_override else verdict["score"]

    return {
        "score": final_score,
        "explanation": verdict["explanation"],
        "independent_override": independent_override,
        "independent_assessment": {
            "independently_useful": independently_useful,
            "reasoning": independent_reasoning,
        },
    }


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_turn3(client: OpenAI, turn3_result: dict) -> dict:
    """Evaluate a single Turn-3 result. Returns structured evaluation dict.

    task_score is normalised 0.0-1.0 (sum of criterion scores / n_criteria).
    Tasks where no image was found receive task_score = 0.0.
    """
    task_id = turn3_result["task_id"]
    no_image_found = turn3_result.get("no_image_found", True)

    if no_image_found:
        zero = {"score": 0, "explanation": "No image was found by the probe."}
        return {
            "task_id": task_id,
            "image_found": False,
            "criteria": {
                "description_accuracy": zero,
                "section_fit": zero,
            },
            "task_score": 0.0,
        }

    image_path = turn3_result.get("local_image_path", "")
    report_path = turn3_result.get("report_path", "")
    task_prompt = turn3_result.get("task_prompt", "")
    what_it_shows = turn3_result.get("what_it_shows", "")
    where_it_fits = turn3_result.get("where_it_fits", "")
    section_heading = turn3_result.get("section_heading", "")

    section_text = "(section not available)"
    if report_path and Path(report_path).exists():
        report_text = Path(report_path).read_text(encoding="utf-8")
        if section_heading:
            section_text = extract_section(report_text, section_heading)
            if not section_text:
                section_text = "(section heading not found in report)"

    criteria: dict = {}

    # Criterion 1 — description accuracy (single vision call)
    prompt1 = _DESC_ACCURACY_PROMPT.format(what_it_shows=what_it_shows)
    criteria["description_accuracy"] = _judge_with_image(client, prompt1, image_path)

    # Criterion 2 — section fit (two-step)
    criteria["section_fit"] = _judge_section_fit_two_step(
        client=client,
        image_path=image_path,
        section_text=section_text,
        task_prompt=task_prompt,
        where_it_fits=where_it_fits,
    )

    n_criteria = len(criteria)
    total_score = sum(c["score"] for c in criteria.values())
    task_score = round(total_score / n_criteria, 4)

    return {
        "task_id": task_id,
        "image_found": True,
        "image_path": image_path,
        "section_heading": section_heading,
        "criteria": criteria,
        "task_score": task_score,
    }


# ---------------------------------------------------------------------------
# Aggregation across all eval files
# ---------------------------------------------------------------------------

def aggregate_scores(eval_dir: Path) -> dict:
    """Compute coverage, quality, and overall score across all eval files.

    coverage  = tasks with image found / total tasks
    quality   = mean task_score over tasks where image was found
                (pure signal on image usefulness, unaffected by coverage gaps)
    overall   = mean task_score over ALL tasks, no_image_found treated as 0
                (single comparable number across models; penalises missing images)
    """
    files = sorted(eval_dir.glob("*_turn3_eval.json"))
    if not files:
        return {}

    all_scores: list[float] = []
    found_scores: list[float] = []

    for f in files:
        result = json.loads(f.read_text(encoding="utf-8"))
        score = result.get("task_score", 0.0)
        image_found = result.get("image_found", False)
        all_scores.append(score)
        if image_found:
            found_scores.append(score)

    n_total = len(all_scores)
    n_found = len(found_scores)

    return {
        "n_tasks": n_total,
        "n_with_image": n_found,
        "coverage": round(n_found / n_total, 4) if n_total else 0.0,
        "quality": round(sum(found_scores) / n_found, 4) if n_found else 0.0,
        "overall": round(sum(all_scores) / n_total, 4) if n_total else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge Turn-3 multimodal probe outputs")
    parser.add_argument("--task", default=None, help="Single task ID (e.g. task_001)")
    parser.add_argument("--overwrite", action="store_true", help="Re-evaluate existing outputs")
    parser.add_argument("--aggregate", action="store_true",
                        help="Print aggregate metrics (also printed automatically after a full run)")
    parser.add_argument(
        "--model", default="openai:gpt-4.1",
        help="Agent model whose turn3 outputs to evaluate. "
             "Controls which turn3_outputs_<slug>/ and turn3_evaluations_<slug>/ folders are used."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    slug = _model_slug(args.model)
    TURN3_OUTPUT_DIR = DRACO_DIR / f"turn3_outputs_{slug}"
    EVAL_OUTPUT_DIR = DRACO_DIR / f"turn3_evaluations_{slug}"
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if args.task:
        input_files = sorted(TURN3_OUTPUT_DIR.glob(f"{args.task}_turn3.json"))
    else:
        input_files = sorted(TURN3_OUTPUT_DIR.glob("*_turn3.json"))

    if not input_files:
        print("No turn3 JSON files found.", file=sys.stderr)
        sys.exit(1)

    total = len(input_files)
    done = skipped = failed = 0

    for i, f in enumerate(input_files, 1):
        task_id = f.stem.replace("_turn3", "")
        out_path = EVAL_OUTPUT_DIR / f"{task_id}_turn3_eval.json"

        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{total}] skip   {task_id}  (exists, use --overwrite)")
            skipped += 1
            continue

        print(f"[{i}/{total}] judging {task_id}...", end="  ", flush=True)
        t0 = time.monotonic()

        try:
            turn3_result = json.loads(f.read_text(encoding="utf-8"))
            eval_result = evaluate_turn3(client, turn3_result)
            out_path.write_text(
                json.dumps(eval_result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            elapsed = time.monotonic() - t0
            print(f"done ({elapsed:.1f}s)  task_score={eval_result['task_score']:.2f}")
            for criterion, r in eval_result["criteria"].items():
                ind = ""
                if "independent_assessment" in r:
                    useful = r["independent_assessment"]["independently_useful"]
                    override = r.get("independent_override", False)
                    ind = f"  [independent: {'useful' if useful else 'not useful'}{'  OVERRIDE→0.5' if override else ''}]"
                print(f"    {criterion:<24} {r['score']}  — {r['explanation'][:80]}{ind}")
            done += 1

        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(f"ERROR ({elapsed:.1f}s)  {exc}")
            failed += 1

    print(f"\n{'='*55}")
    print(f"Done: {done}  skipped: {skipped}  failed: {failed}")
    print(f"Outputs -> {EVAL_OUTPUT_DIR}")  # type: ignore[union-attr]

    # Print aggregate automatically after a full run, or when --aggregate is passed
    if args.aggregate or (done > 0 and not args.task):
        agg = aggregate_scores(EVAL_OUTPUT_DIR)  # type: ignore[arg-type]
        if agg:
            print(f"\n{'='*55}")
            print("Aggregate metrics")
            print(f"  Tasks evaluated : {agg['n_tasks']}")
            print(f"  With image      : {agg['n_with_image']}")
            print(f"  Coverage        : {agg['coverage']:.3f}")
            print(f"  Quality         : {agg['quality']:.3f}  (mean score | image found)")
            print(f"  Overall         : {agg['overall']:.3f}  (mean score | all tasks)")


if __name__ == "__main__":
    main()
    