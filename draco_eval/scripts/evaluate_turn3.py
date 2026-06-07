"""Evaluate Turn-3 multimodal probe outputs using an LLM judge.

For each task, loads the turn3 JSON + saved images + relevant report section,
then produces three binary scores:

  1. description_score  — does what_it_shows accurately describe the selected image?
  2. selection_score    — does the judge independently prefer the same image the agent chose?
  3. explanation_score  — is the agent's section placement explanation accurate?

Output per task (flat JSON):
  {
    "task_id": "task_001",
    "description_score": 1,
    "selection_score": 0,
    "explanation_score": 1,
    "candidates_evaluated": 3,
    "judge_preferred_index": 2,
    "agent_selected_index": 0
  }
  no_image_found tasks receive all scores = 0 and candidates_evaluated = 0.

Aggregate metrics (computed by aggregate_scores()):
  coverage          = tasks with image found / total tasks
  description_mean  = mean description_score over tasks with image found
  selection_mean    = mean selection_score over tasks with image found
  explanation_mean  = mean explanation_score over tasks with image found
  selection_by_n    = selection_mean stratified by candidate count (1, 2, 3)
  candidate_dist    = distribution of candidate counts

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

from turn3_probe import extract_section, extract_headings  # noqa: E402

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
Look at this image. The agent described it as:

"{what_it_shows}"

Is this description accurate? Does it correctly capture what is visually \
present in the image?

Score 1 if accurate, 0 if not.
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
# Score 2 — selection score
# ---------------------------------------------------------------------------

_SELECTION_RANK_PROMPT = """\
You are evaluating whether any of the candidate images below would genuinely \
add value to a research report, and if so, which is best.

ORIGINAL RESEARCH QUESTION:
{task_prompt}

REPORT SUMMARY:
{report_summary}

SECTION HEADINGS FROM THE REPORT:
{headings_list}

---
Below are {n_candidates} candidate images retrieved from the report's cited sources.
They are labelled Candidate 0, Candidate 1, etc.

First decide: does ANY candidate genuinely add value to a reader of this report? \
An image adds value only if it conveys substantive information that the report \
text alone cannot provide and is relevant to the research question's analytical intent.

If NO candidate adds genuine value, return best_index = -1.
If at least one does, return the 0-based index of the best one.

OUTPUT FORMAT — return exactly:
{{"best_index": <0-based integer, or -1 if none are worth including>, "explanation": "<your reasoning, under 80 words>"}}

Return only raw JSON starting with {{, no backticks.
"""

_SELECTION_SINGLE_PROMPT = """\
You are evaluating whether an image would add genuine value to a research report.

ORIGINAL RESEARCH QUESTION:
{task_prompt}

REPORT SUMMARY:
{report_summary}

---
The image below was retrieved from one of the report's cited sources.

Would this image add genuine value to a reader of this report? \
Score 1 if it conveys information that meaningfully complements the report, \
0 if it does not.

OUTPUT FORMAT — return exactly:
{{"score": 1 or 0, "explanation": "<your reasoning, under 80 words>"}}

Return only raw JSON starting with {{, no backticks.
"""

# ---------------------------------------------------------------------------
# Score 3 — explanation score
# ---------------------------------------------------------------------------

_EXPLANATION_PROMPT = """\
Here is an image selected for a research report. The agent claims it fits in \
the following section:

SECTION HEADING: {section_heading}

AGENT'S EXPLANATION:
{where_it_fits}

FULL SECTION CONTENT:
{section_text}

---
The image is attached. Does the image genuinely support or illustrate content \
in this section, and is the agent's explanation accurate?

Score 1 if the explanation is accurate and the image fits the section, 0 if not.
Return your verdict as JSON.
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


def _judge_selection_score(
    client: OpenAI,
    all_candidates: list[dict],
    selected_candidate_index: int,
    task_prompt: str,
    report_summary: str,
    report_headings: list[str],
) -> dict:
    """Score 2: does the judge independently prefer the same image the agent chose?

    Multi-candidate path: show all images (no agent labels/reasoning), ask judge
    to pick the best. Compare judge's pick to selected_candidate_index.
    Match → score 1, no match → score 0.

    Single-candidate fallback: ask judge whether the one image adds genuine value.
    Score 1 if yes, 0 if no.

    Returns {score, explanation, judge_best_index} (judge_best_index is None for
    the single-candidate path).
    """
    headings_text = "\n".join(report_headings) if report_headings else "  (none)"

    if len(all_candidates) == 1:
        # Single-candidate fallback — binary evaluation
        b64, mime = _encode_image(all_candidates[0]["local_path"])
        prompt = _SELECTION_SINGLE_PROMPT.format(
            task_prompt=task_prompt,
            report_summary=report_summary,
        )
        response = client.responses.create(
            model=JUDGE_MODEL,
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
        verdict = _parse_scored_verdict(response.output_text)
        return {
            "score": verdict["score"],
            "explanation": verdict["explanation"],
            "judge_best_index": None,
        }

    # Multi-candidate ranking path
    prompt = _SELECTION_RANK_PROMPT.format(
        task_prompt=task_prompt,
        report_summary=report_summary,
        headings_list=headings_text,
        n_candidates=len(all_candidates),
    )
    # Build content: prompt text, then labelled image blocks (no agent metadata)
    content: list[dict] = [{"type": "input_text", "text": prompt}]
    for i, c in enumerate(all_candidates):
        b64, mime = _encode_image(c["local_path"])
        content.append({"type": "input_text", "text": f"--- Candidate {i} ---"})
        content.append({"type": "input_image", "image_url": f"data:{mime};base64,{b64}"})

    response = client.responses.create(
        model=JUDGE_MODEL,
        input=[{"role": "user", "content": content}],
        reasoning={"effort": "none"},
        temperature=0,
        text={"format": {"type": "json_object"}},
    )
    parsed = json.loads(response.output_text)
    judge_idx = parsed.get("best_index")
    explanation = parsed.get("explanation", "")

    # -1 means the judge rejected all candidates — score 0
    if judge_idx == -1:
        return {
            "score": 0,
            "explanation": explanation,
            "judge_best_index": -1,
        }

    if not isinstance(judge_idx, int) or not (0 <= judge_idx < len(all_candidates)):
        # Malformed index — treat as no match
        return {
            "score": 0,
            "explanation": f"Judge returned invalid best_index {judge_idx!r}. {explanation}",
            "judge_best_index": judge_idx,
        }

    score = 1 if judge_idx == selected_candidate_index else 0
    return {
        "score": score,
        "explanation": explanation,
        "judge_best_index": judge_idx,
    }


def _judge_explanation_score(
    client: OpenAI,
    image_path: str,
    section_heading: str,
    where_it_fits: str,
    section_text: str,
) -> dict:
    """Score 3: is the agent's section placement explanation accurate?

    Single vision call: judge sees image + section heading + agent's explanation
    + full section text. Returns {score, explanation}.
    """
    prompt = _EXPLANATION_PROMPT.format(
        section_heading=section_heading,
        where_it_fits=where_it_fits,
        section_text=section_text,
    )
    return _judge_with_image(client, prompt, image_path)


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
    """Evaluate a single Turn-3 result. Returns flat evaluation dict.

    Produces three binary scores. Tasks where no image was found receive
    all scores = 0 and candidates_evaluated = 0.
    """
    task_id = turn3_result["task_id"]
    no_image_found = turn3_result.get("no_image_found", True)

    if no_image_found:
        return {
            "task_id": task_id,
            "description_score": 0,
            "selection_score": 0,
            "explanation_score": 0,
            "candidates_evaluated": 0,
            "judge_preferred_index": None,
            "agent_selected_index": None,
        }

    image_path = turn3_result.get("local_image_path", "")
    report_path = turn3_result.get("report_path", "")
    task_prompt = turn3_result.get("task_prompt", "")
    what_it_shows = turn3_result.get("what_it_shows", "")
    where_it_fits = turn3_result.get("where_it_fits", "")
    section_heading = turn3_result.get("section_heading", "")
    report_summary = turn3_result.get("report_summary", "")
    all_candidates: list[dict] = turn3_result.get("all_candidates", [])
    selected_candidate_index: int = turn3_result.get("selected_candidate_index", 0)

    section_text = "(section not available)"
    report_headings: list[str] = []
    if report_path and Path(report_path).exists():
        report_text = Path(report_path).read_text(encoding="utf-8")
        report_headings = extract_headings(report_text, max_level=2)
        if section_heading:
            section_text = extract_section(report_text, section_heading)
            if not section_text:
                section_text = "(section heading not found in report)"

    # Score 1 — description accuracy (image + what_it_shows only)
    desc_result = _judge_with_image(
        client,
        _DESC_ACCURACY_PROMPT.format(what_it_shows=what_it_shows),
        image_path,
    )

    # Score 2 — selection score (all candidates, NO agent reasoning visible)
    sel_result = _judge_selection_score(
        client=client,
        all_candidates=all_candidates,
        selected_candidate_index=selected_candidate_index,
        task_prompt=task_prompt,
        report_summary=report_summary,
        report_headings=report_headings,
    )

    # Score 3 — explanation score (image + section heading + where_it_fits + section text)
    expl_result = _judge_explanation_score(
        client=client,
        image_path=image_path,
        section_heading=section_heading,
        where_it_fits=where_it_fits,
        section_text=section_text,
    )

    return {
        "task_id": task_id,
        "description_score": desc_result["score"],
        "selection_score": sel_result["score"],
        "explanation_score": expl_result["score"],
        "candidates_evaluated": len(all_candidates),
        "judge_preferred_index": sel_result.get("judge_best_index"),
        "agent_selected_index": selected_candidate_index,
        # Verbose fields for inspection
        "description_explanation": desc_result.get("explanation", ""),
        "selection_explanation": sel_result.get("explanation", ""),
        "explanation_explanation": expl_result.get("explanation", ""),
    }


# ---------------------------------------------------------------------------
# Aggregation across all eval files
# ---------------------------------------------------------------------------

def aggregate_scores(eval_dir: Path) -> dict:
    """Compute per-score means, coverage, and candidate count distribution.

    Scores are only averaged over tasks where an image was found (candidates_evaluated > 0).
    selection_by_n breaks down selection accuracy by candidate count (1, 2, 3).
    """
    files = sorted(eval_dir.glob("*_turn3_eval.json"))
    if not files:
        return {}

    n_total = len(files)
    n_with_image = 0
    desc_scores: list[int] = []
    sel_scores: list[int] = []
    expl_scores: list[int] = []
    sel_by_n: dict[int, list[int]] = {}
    candidate_dist: dict[int, int] = {}

    for f in files:
        result = json.loads(f.read_text(encoding="utf-8"))
        n_cands = result.get("candidates_evaluated", 0)
        candidate_dist[n_cands] = candidate_dist.get(n_cands, 0) + 1
        if n_cands == 0:
            continue
        n_with_image += 1
        desc_scores.append(result.get("description_score", 0))
        sel_scores.append(result.get("selection_score", 0))
        expl_scores.append(result.get("explanation_score", 0))
        sel_by_n.setdefault(n_cands, []).append(result.get("selection_score", 0))

    def _mean(xs: list) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "n_tasks": n_total,
        "n_with_image": n_with_image,
        "coverage": round(n_with_image / n_total, 4) if n_total else 0.0,
        "description_mean": _mean(desc_scores),
        "selection_mean": _mean(sel_scores),
        "explanation_mean": _mean(expl_scores),
        "selection_by_n": {k: _mean(v) for k, v in sorted(sel_by_n.items())},
        "candidate_dist": {k: v for k, v in sorted(candidate_dist.items())},
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
            d = eval_result.get("description_score", 0)
            s = eval_result.get("selection_score", 0)
            e = eval_result.get("explanation_score", 0)
            n = eval_result.get("candidates_evaluated", 0)
            j = eval_result.get("judge_preferred_index")
            a = eval_result.get("agent_selected_index")
            print(f"done ({elapsed:.1f}s)  desc={d}  sel={s}  expl={e}  cands={n}")
            if n > 0:
                print(f"    judge_pick={j}  agent_pick={a}  "
                      f"{'MATCH' if j == a else 'MISMATCH'}")
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
            print("Aggregate metrics  (scores over tasks with image found)")
            print(f"  Tasks evaluated  : {agg['n_tasks']}")
            print(f"  With image       : {agg['n_with_image']}  "
                  f"(coverage={agg['coverage']:.3f})")
            print(f"  Description mean : {agg['description_mean']:.3f}")
            print(f"  Selection mean   : {agg['selection_mean']:.3f}")
            print(f"  Explanation mean : {agg['explanation_mean']:.3f}")
            print(f"  Selection by N candidates:")
            for n, v in agg.get("selection_by_n", {}).items():
                print(f"    N={n}: {v:.3f}")
            print(f"  Candidate dist   : {agg.get('candidate_dist', {})}")


if __name__ == "__main__":
    main()
    