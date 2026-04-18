"""Turn-3 multimodal capability probe.

This module implements the Turn-3 image-retrieval probe as a plain async
function (run_turn3_probe). It is integrated into the ODR LangGraph as the
`multimodal_probe` node in deep_researcher.py — i.e. it runs as a node in
the existing graph after `final_report_generation`, not as a separate graph.

Internally it contains no LangGraph nodes, state machines, or supervisor
loops. The pipeline is linear and self-contained:
  1. Extract all cited URLs from the v2 report.
  2. Analyse the report once to identify visual opportunities (Phase 1).
  3. For each cited URL, fetch the page and find candidate images.
  4. For each candidate, run a vision LLM call to judge whether the image
     genuinely supports a claim in the report.
  5. Return the first image that passes all filters as a Turn3Result.

Keeping the probe as a plain async function (rather than a nested graph)
avoids state-management conflicts from running a LangGraph inside another
LangGraph node, while still benefiting from full integration into the ODR
framework (correct model routing, slug-namespaced output paths, task_id
tracking, and LangSmith tracing via the parent node).

Two httpx helpers are used internally:
  fetch_url(url)        → fetches HTML of a cited page, 5s timeout
  download_image(url)   → HEAD first, then GET + base64 encode if it passes heuristics

Usage (standalone test, bypasses ODR graph):
    python draco_eval/scripts/turn3_probe.py \
        --report draco_eval/reports/task_001_v2.md \
        --task-id task_001
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI  # kept for type hints only
from pydantic import BaseModel, Field

load_dotenv()

DRACO_DIR = Path(__file__).parent.parent
LOG_DIR = DRACO_DIR / "logs"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("turn3_probe")
    logger.setLevel(logging.INFO)
    # Avoid adding duplicate handlers on re-import
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
               for h in logger.handlers):
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
        logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class Turn3Result(BaseModel):
    """Judge-facing output for a single turn-3 multimodal probe.

    Contains exactly what an LLM judge needs to evaluate whether the probe
    correctly described the image and correctly identified where it fits in
    the report. Operational metadata (URLs, call counts) lives in `metadata`.
    """

    # --- Identity ---
    task_id: str = Field(description="Task identifier, e.g. task_001")
    task_prompt: str = Field(description="Original research question the v2 report answered")
    report_path: str = Field(description="Path to the v2 report markdown file")

    # --- Image ---
    local_image_path: str = Field(
        default="",
        description="Absolute path to the saved image file on disk (empty if no image found)",
    )

    # --- Claims to judge ---
    what_it_shows: str = Field(
        default="",
        description="Max 3 sentences describing what the image shows",
    )
    where_it_fits: str = Field(
        default="",
        description="Which section/claim in the report this image supports, max 2 sentences",
    )
    section_heading: str = Field(
        default="",
        description="Exact section heading from the report this image best supports (verbatim copy for section lookup)",
    )

    # --- Outcome ---
    no_image_found: bool = Field(
        description="True if no valid image was found after exhausting all cited URLs"
    )
    reason_if_none: str = Field(
        default="",
        description="Explanation of why no image was found, if no_image_found is true",
    )

    # --- Phase 1 context (for debugging and transparency) ---
    report_summary: str = Field(
        default="",
        description="Summary of the report generated by Phase 1 analysis",
    )
    visual_opportunities: list[str] = Field(
        default_factory=list,
        description="Visual opportunities identified by Phase 1 analysis",
    )

    # --- Operational metadata (not for judging) ---
    metadata: dict = Field(
        default_factory=dict,
        description="Operational details: image_url, visited_url, vision_calls",
    )


# ---------------------------------------------------------------------------
# Section extraction (used by judge scripts)
# ---------------------------------------------------------------------------


def _normalise_heading(s: str) -> str:
    """Normalise typographic punctuation to ASCII for fuzzy heading matching."""
    return (
        s.replace('\u2019', "'").replace('\u2018', "'")   # curly single quotes
         .replace('\u201c', '"').replace('\u201d', '"')   # curly double quotes
         .replace('\u2013', '-').replace('\u2014', '-')   # en/em dash
         .strip()
    )


def extract_section(report_text: str, heading: str) -> str:
    """Extract the full text of a section by its heading.

    Handles typographic characters (curly quotes, em-dashes) by normalising
    both sides before comparing, so a model-output heading like "Sant'Anna"
    (ASCII apostrophe) will match "Sant\u2019Anna" (curly quote) in the report.

    Collects all lines from the heading until the next heading at the same or
    higher level (same or fewer # characters).

    Returns empty string if the heading is not found.
    """
    heading_norm = _normalise_heading(heading)
    level_match = re.match(r'^(#+)', heading_norm)
    target_level = len(level_match.group(1)) if level_match else 0

    lines = report_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if _normalise_heading(line) == heading_norm:
            start_idx = i
            break

    if start_idx is None:
        return ""

    section_lines = [lines[start_idx]]
    for line in lines[start_idx + 1:]:
        m = re.match(r'^(#+)\s', line)
        if m:
            next_level = len(m.group(1))
            if target_level == 0 or next_level <= target_level:
                break
        section_lines.append(line)

    return "\n".join(section_lines).strip()


def extract_headings(report_text: str) -> list[str]:
    """Extract all markdown headings from the report, preserving # prefix and order."""
    return [
        line.rstrip()
        for line in report_text.splitlines()
        if re.match(r'^#{1,6}\s', line)
    ]


# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------

_INLINE_LINK_RE = re.compile(r'\[([^\]]*)\]\((https?://[^\s)]+)\)')
_BARE_URL_RE = re.compile(r'(?<!\()(https?://[^\s\)>\]]+)')


def extract_cited_urls(report_text: str) -> list[str]:
    """Extract all cited URLs from a v2 markdown report in document order.

    Captures both inline markdown links [text](url) and bare URLs in the
    sources section. Deduplicates while preserving first-seen order.
    """
    urls: list[str] = []
    seen: set[str] = set()

    for m in _INLINE_LINK_RE.finditer(report_text):
        url = m.group(2).rstrip(".,)")
        if url not in seen:
            urls.append(url)
            seen.add(url)

    for m in _BARE_URL_RE.finditer(report_text):
        url = m.group(1).rstrip(".,)")
        if url not in seen:
            urls.append(url)
            seen.add(url)

    return urls


# ---------------------------------------------------------------------------
# Image filtering heuristics
# ---------------------------------------------------------------------------

_BAD_URL_FRAGMENTS = [
    "/icon/", "/logo/", "/avatar/", "/badge/", "/pixel/",
    "/tracker/", "/ads/", "/banner/", "/spacer/",
    "doubleclick.net", "googletagmanager", "facebook.com/tr",
]
_BAD_EXTENSIONS = {".gif", ".ico", ".bmp"}
# Filename stems (without extension) that are almost always decorative assets,
# regardless of which directory they live in.
_BAD_FILENAME_STEMS = {
    "logo", "favicon", "icon", "avatar", "banner", "badge",
    "spinner", "loading", "placeholder", "spacer", "pixel",
    "arrow", "close", "menu", "hamburger", "search", "share",
}
_IMAGE_LIKE_PATTERN = re.compile(
    r"(image|img|photo|picture|figure|fig|chart|graph|plot)", re.IGNORECASE
)
_BAD_ALT_EXACT = {
    "image", "photo", "picture", "thumbnail", "banner",
    "logo", "icon", "button", "loading", "arrow", "close",
}
_ALT_FILENAME_RE = re.compile(r"\.(jpg|jpeg|png|gif)\s*$", re.IGNORECASE)


def _passes_url_filter(src: str) -> bool:
    """URL-pattern checks — no HTTP request needed. Return True to keep."""
    src_lower = src.lower()
    for fragment in _BAD_URL_FRAGMENTS:
        if fragment in src_lower:
            return False

    path = urlparse(src_lower).path
    ext = Path(path).suffix.lower()

    if ext in _BAD_EXTENSIONS:
        return False

    # Filename-stem check: catch logo.png, favicon.png, avatar.jpg, etc.
    # that live at the root or in non-flagged directories.
    stem = Path(path).stem.lower()
    if stem in _BAD_FILENAME_STEMS:
        return False

    # No extension and no image-like keyword → discard
    if not ext and not _IMAGE_LIKE_PATTERN.search(src_lower):
        return False

    return True


def _passes_alt_filter(alt: Optional[str]) -> bool:
    """Alt-text checks — no HTTP request needed. Return True to keep.

    Images with no alt attribute (alt=None) are allowed through — academic
    figures rarely have alt text. Only reject images whose alt text is
    explicitly bad (logo, icon, banner, etc.).
    """
    if alt is None:
        # No alt attribute at all — let URL and HEAD filters decide
        return True
    if not alt.strip():
        # Empty alt string — also allow through
        return True
    alt_norm = alt.strip().lower()
    if alt_norm in _BAD_ALT_EXACT:
        return False
    if _ALT_FILENAME_RE.search(alt_norm):
        return False
    return True


async def _passes_head_filter(
    client: httpx.AsyncClient, src: str, logger: logging.Logger
) -> tuple[bool, str]:
    """HEAD request checks. Returns (passes, content_type)."""
    try:
        resp = await client.head(src, timeout=5.0, follow_redirects=True)
        ct = resp.headers.get("content-type", "")
        if "image/jpeg" not in ct and "image/png" not in ct:
            logger.info(f"    HEAD: bad content-type {ct!r} → {src}")
            return False, ct
        cl = resp.headers.get("content-length")
        if cl and int(cl) < 5 * 1024:
            logger.info(f"    HEAD: content-length {cl} bytes < 5 KB → {src}")
            return False, ct
        return True, ct
    except Exception as exc:
        logger.info(f"    HEAD: request failed ({exc}) → {src}")
        return False, ""


# ---------------------------------------------------------------------------
# httpx tools
# ---------------------------------------------------------------------------

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


async def fetch_url(
    client: httpx.AsyncClient, url: str, logger: logging.Logger
) -> Optional[str]:
    """Fetch HTML of a cited page. Returns None on any error or non-HTML response."""
    try:
        resp = await client.get(url, timeout=5.0, follow_redirects=True)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        # Skip binary responses (PDFs, gzip, etc.) — BeautifulSoup can't parse them
        if ct and "html" not in ct and "text" not in ct:
            logger.info(f"  fetch_url: skipping non-HTML content-type {ct!r} → {url}")
            return None
        return resp.text
    except Exception as exc:
        logger.info(f"  fetch_url failed ({exc}) → {url}")
        return None


_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG"


def _detect_mime(data: bytes) -> Optional[str]:
    """Detect image MIME type from magic bytes. Returns None if not a valid image."""
    if data[:3] == _JPEG_MAGIC:
        return "image/jpeg"
    if data[:4] == _PNG_MAGIC:
        return "image/png"
    return None  # not a real image — caller should discard


async def download_image(
    client: httpx.AsyncClient,
    image_url: str,
    logger: logging.Logger,
    max_bytes: int = 5 * 1024 * 1024,
) -> Optional[tuple[bytes, str]]:
    """HEAD filter then GET raw bytes.

    Returns (raw_bytes, mime_type) or None if skipped.
    Raw bytes are kept so the caller can both base64-encode for vision
    and save to disk without re-downloading.
    """
    passes, _ct = await _passes_head_filter(client, image_url, logger)
    if not passes:
        return None

    try:
        resp = await client.get(image_url, timeout=5.0, follow_redirects=True)
        resp.raise_for_status()
        content = resp.content
        if len(content) < 5 * 1024:
            logger.info(
                f"    download: {image_url} is {len(content)} bytes < 5 KB, skipping"
            )
            return None
        if len(content) > max_bytes:
            logger.info(
                f"    download: {image_url} is {len(content)//1024}KB > 5MB, skipping"
            )
            return None
        # Validate magic bytes — server may have lied about content-type in HEAD
        mime = _detect_mime(content)
        if mime is None:
            logger.info(f"    download: magic bytes not JPEG/PNG, discarding → {image_url}")
            return None
        return content, mime
    except Exception as exc:
        logger.info(f"    download failed ({exc}) → {image_url}")
        return None


# ---------------------------------------------------------------------------
# Image extraction from HTML
# ---------------------------------------------------------------------------


def extract_images(html: str, base_url: str) -> list[dict]:
    """Parse all <img> tags from HTML, resolving relative src URLs.

    Deduplicates by src within the page — if the same image URL appears in
    multiple tags (e.g. header + footer both use the site logo), only the
    first occurrence is kept.

    Returns empty list if the content can't be parsed as HTML.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []
    images = []
    seen_srcs: set[str] = set()
    for img in soup.find_all("img"):
        src = img.get("src", "").strip()
        # Fall back to lazy-load attributes if src is missing or a placeholder
        if not src or src.startswith("data:"):
            src = (
                img.get("data-src", "")
                or img.get("data-lazy-src", "")
                or img.get("data-original", "")
            ).strip()
        if not src or src.startswith("data:"):
            continue
        alt: Optional[str] = img.get("alt", "").strip() or None
        if not src.startswith("http"):
            src = urljoin(base_url, src)
        if src in seen_srcs:
            continue
        seen_srcs.add(src)
        images.append({"src": src, "alt": alt})
    return images


# ---------------------------------------------------------------------------
# Phase 1 — one-time report analysis (runs before image loop)
# ---------------------------------------------------------------------------


class _ReportContext(BaseModel):
    """Compact report understanding produced once before any vision calls."""

    summary: str = Field(
        description=(
            "A thorough summary covering the report's topic, scope, all key entities "
            "or methods compared or analysed, and main conclusions. Cover every "
            "significant section — a vision model will use this to judge whether any "
            "image is relevant to ANY part of the report, so completeness matters "
            "more than brevity."
        )
    )
    visual_opportunities: list[str] = Field(
        description=(
            "Deduplicated list of sections or claims in the report where a visual "
            "would add genuine value. Each entry names the SECTION OR CLAIM (as it "
            "appears in the report) and the TYPE of visual that would help. "
            "If a parent section and its subsections share the same visual type, "
            "list only the parent. No two entries should describe the same visual "
            "type for overlapping content. Stay abstract: describe image TYPE and "
            "TOPIC, not predicted content. Do not fabricate sections."
        )
    )


_ANALYSE_REPORT_PROMPT = """\
You are preparing a compact reference that will be used to judge whether images \
found on cited webpages are relevant to the following research task and report.

ORIGINAL RESEARCH QUESTION:
{task_prompt}

FULL REPORT:
{report_text}

---
Your tasks:

1. Write a thorough SUMMARY of the report covering its topic, scope, ALL key \
entities or methods compared or analysed, and main conclusions. Cover every \
significant section — completeness matters more than brevity here. A vision \
model will use this summary to judge whether any found image is relevant to any \
part of the report, so do not omit topics.

2. List the sections or claims in the report where a visual would genuinely add \
value for a reader. For each, state:
   - the section name or claim (as it appears in the report)
   - the type of visual that would be useful (e.g. bar chart, system diagram, \
product photo, event-study plot)

DEDUPLICATION RULES — apply strictly:
   - If a parent section and its subsections would benefit from the SAME type \
of visual, list only the parent section — do not repeat it for every subsection.
   - If multiple sections would benefit from different visual types, list each once.
   - No two entries in the list should describe the same visual type for \
overlapping content.

CRITICAL: Let the research question drive what counts as a useful visual type. \
If the task is primarily quantitative or analytical (financial metrics, \
performance benchmarks, statistical comparisons), only data-bearing visuals \
qualify — charts, tables, plots, technical diagrams, interface screenshots. \
A photograph of a named entity (a building, a person, a product exterior) does \
NOT qualify for a quantitative task, even if the entity is directly discussed. \
If the task is descriptive or comparative of physical objects or places, \
real photographs may qualify. Match visual type to analytical intent.

Keep visual opportunities at the level of image type + topic. Do NOT predict \
specific image content or bias toward particular sources.
"""


async def _analyse_report(
    llm: ChatOpenAI, report_text: str, task_prompt: str, logger: logging.Logger
) -> tuple[_ReportContext, bool]:
    """Phase 1: read the full report once, produce compact context for vision calls.

    Returns (context, phase1_failed). phase1_failed=True means the LLM call
    failed and the fallback summary is being used — callers should record this
    in metadata so the failure is visible in the output.
    """
    logger.info("  [phase-1] Analysing report to extract summary and visual opportunities...")
    try:
        structured_llm = llm.with_structured_output(_ReportContext)
        raw = await structured_llm.ainvoke(
            [HumanMessage(content=_ANALYSE_REPORT_PROMPT.format(
                task_prompt=task_prompt,
                report_text=report_text,
            ))]
        )
        ctx = _ReportContext.model_validate(raw) if isinstance(raw, dict) else raw
        assert isinstance(ctx, _ReportContext)
        logger.info(f"  [phase-1] Summary: {ctx.summary[:120]!r}")
        logger.info(f"  [phase-1] Visual opportunities ({len(ctx.visual_opportunities)}): "
                    + " | ".join(ctx.visual_opportunities[:3]))
        return ctx, False
    except Exception as exc:
        logger.warning(f"  [phase-1] FAILED ({exc}) — falling back to first 2000 chars of report")
        return _ReportContext(
            summary=report_text[:2000],
            visual_opportunities=[],
        ), True


# ---------------------------------------------------------------------------
# Phase 2 — per-image fit evaluation (uses compact context, not full report)
# ---------------------------------------------------------------------------


class _ImageFitResult(BaseModel):
    """Internal schema for the per-image vision evaluation step."""

    fits_report: bool = Field(
        description=(
            "True ONLY if this image meaningfully illustrates or supports a specific "
            "claim, finding, method, or data point described in the report context. "
            "False for logos, branding, site chrome, author photos, generic stock "
            "images, or anything with no clear connection to the report content."
        )
    )
    what_it_shows: str = Field(
        description="Max 3 sentences describing what the image actually shows."
    )
    where_it_fits: str = Field(
        description=(
            "If fits_report is True: max 2 sentences naming the specific section or "
            "claim from the report context this image supports. "
            "If fits_report is False: empty string."
        ),
        default="",
    )
    section_heading: str = Field(
        description=(
            "If fits_report is True: the EXACT section heading from the report "
            "(e.g. '## Comparative Technical Analysis') that this image best supports. "
            "Copy it verbatim — do not paraphrase. Empty string if fits_report is False."
        ),
        default="",
    )


_FIT_PROMPT_TEMPLATE = """\
You are evaluating whether an image found on a cited webpage is genuinely \
useful to a reader of a research report.

ORIGINAL RESEARCH QUESTION:
{task_prompt}

REPORT SUMMARY:
{summary}

SECTIONS / CLAIMS WHERE A VISUAL WOULD ADD VALUE:
{visual_opportunities}

EXACT SECTION HEADINGS FROM THE REPORT (copy verbatim for section_heading):
{headings_list}

---
The image below was retrieved from:
  Page URL  : {visited_url}
  Image URL : {image_url}

Your task:
1. Describe what the image shows (what_it_shows) in max 3 sentences. \
State only what is visually present — do not explain how it relates to the \
report or justify its inclusion. No filler phrases like "this image supports" \
or "relevant to the report's analysis of". Just describe the image content.
2. Decide whether the image would genuinely help a reader of this report \
(fits_report).

Set fits_report = True ONLY if ALL of the following hold:
1. The image contains SUBSTANTIVE VISUAL CONTENT — actual data, a real diagram, \
a real interface screenshot, or a real photograph. It must convey information \
a reader could not get from reading the report text alone.
2. It directly illustrates something covered in the report summary.
3. The IMAGE TYPE matches what the research question actually needs. \
If the research question asks for quantitative analysis (metrics, margins, \
benchmarks, financial figures, statistical comparisons), only data-bearing \
visuals qualify: charts, plots, tables, technical diagrams, dashboards, \
interface screenshots. A photograph of a named subject — a building, a person, \
a product exterior — does NOT pass this condition for a quantitative task, \
even if the subject is discussed in the report. If the task is descriptive \
or comparative of physical objects or places, real photographs may qualify.
{condition_4}

Examples of True: a chart comparing the entities the report analyses, a diagram \
of a process the report describes, a real product screenshot, an event-study plot, \
a flow diagram explaining a business model, a real interface showing a workflow, \
a technical spec sheet, a performance benchmark table.

Set fits_report = False (and leave where_it_fits and section_heading empty) for ANY of:
  - Photographs of buildings, properties, or locations when the task is financial \
or quantitative
  - Images containing only text (title slides, quote cards, heading-only slides)
  - AI-generated illustrations or generic stock artwork with no real data
  - Marketing/decorative images that carry no information beyond the report text
  - Platform logos, site branding, favicons
  - Author headshots or profile photos
  - UI chrome, navigation elements, unrelated illustrations

If fits_report is True:
- Fill where_it_fits: name the specific section or visual opportunity this image \
best supports (max 2 sentences). Do not pad with justification — be direct.
- Fill section_heading: pick the single best matching heading from the \
"EXACT SECTION HEADINGS FROM THE REPORT" list above and copy it CHARACTER FOR \
CHARACTER including the leading # symbols (e.g. "## Comparative Technical Analysis"). \
Do NOT paraphrase or invent a heading — only use headings from that list.
If fits_report is False, leave both where_it_fits and section_heading empty.
"""


async def _evaluate_image_fit(
    llm: ChatOpenAI,
    report_context: _ReportContext,
    report_headings: list[str],
    task_prompt: str,
    image_bytes: bytes,
    mime: str,
    image_url: str,
    visited_url: str,
) -> _ImageFitResult:
    """Phase 2: ask gpt-4.1 vision whether this image fits the report context."""
    b64_data = base64.b64encode(image_bytes).decode("utf-8")
    if report_context.visual_opportunities:
        opportunities_text = "\n".join(f"- {v}" for v in report_context.visual_opportunities)
        condition_4 = (
            "4. It matches (or is closely related to) one of the listed visual opportunities."
        )
    else:
        opportunities_text = "  (not available — Phase 1 analysis failed)"
        condition_4 = (
            "4. (Visual opportunities unavailable — apply conditions 1, 2, and 3 only.)"
        )
    headings_text = "\n".join(report_headings) if report_headings else "  (none)"
    prompt = _FIT_PROMPT_TEMPLATE.format(
        task_prompt=task_prompt,
        summary=report_context.summary,
        visual_opportunities=opportunities_text,
        headings_list=headings_text,
        visited_url=visited_url,
        image_url=image_url,
        condition_4=condition_4,
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{b64_data}",
                    "detail": "high",
                },
            },
        ]
    )

    def _clean(result: _ImageFitResult) -> _ImageFitResult:
        """Ensure where_it_fits and section_heading are empty when fits_report is False."""
        if not result.fits_report and (result.where_it_fits or result.section_heading):
            return result.model_copy(update={"where_it_fits": "", "section_heading": ""})
        return result

    # Primary path: with_structured_output + multimodal input.
    # gpt-4.1 supports tool-calling and vision simultaneously.
    try:
        structured_llm = llm.with_structured_output(_ImageFitResult)
        raw = await structured_llm.ainvoke([message])
        ctx = _ImageFitResult.model_validate(raw) if isinstance(raw, dict) else raw
        return _clean(ctx)
    except Exception as primary_exc:
        primary_msg = str(primary_exc)
        # API-level image rejection (invalid base64, unsupported format, etc.)
        # The image itself is bad — retrying with a fallback prompt won't help.
        # Discard this image and let the probe continue to the next one.
        if "invalid_base64" in primary_msg or "invalid_image" in primary_msg or "400" in primary_msg:
            return _ImageFitResult(
                fits_report=False,
                what_it_shows="Image rejected by vision API (invalid or unsupported format).",
                where_it_fits="",
            )
        # Other errors (network, timeout, JSON parse) — try raw fallback once
        try:
            fallback_prompt = prompt + "\n\nRespond ONLY with a valid JSON object."
            raw = await llm.ainvoke(
                [HumanMessage(content=[
                    {"type": "text", "text": fallback_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime};base64,{b64_data}", "detail": "high"
                    }},
                ])]
            )
            text: str = raw.content if isinstance(raw.content, str) else str(raw.content)
            text = re.sub(r"^```(?:json)?\s*", "", text.strip())
            text = re.sub(r"\s*```$", "", text.strip())
            data = json.loads(text)
            return _clean(_ImageFitResult(**data))
        except Exception:
            # Fallback also failed — discard image, continue probe
            return _ImageFitResult(
                fits_report=False,
                what_it_shows="Vision call failed after retry.",
                where_it_fits="",
            )


# ---------------------------------------------------------------------------
# Core probe entry point
# ---------------------------------------------------------------------------


async def run_turn3_probe(
    report_text: str,
    task_id: str,
    task_prompt: str,
    report_path: str,
    log_path: Optional[Path] = None,
    image_save_dir: Optional[Path] = None,
    model: str = "openai:gpt-4.1",
) -> Turn3Result:
    """Run the Turn-3 multimodal probe on a single v2 report.

    Iterates over ALL cited URLs and ALL images within each URL. For every image
    that passes the heuristic filters it calls gpt-4.1 vision to judge whether
    the image genuinely fits the report content. Continues to the next image (or
    next URL) if the model says fits_report=False. Returns as soon as one image
    is confirmed to fit. Returns no_image_found=True only after all URLs and all
    images within them are exhausted.

    When a fitting image is found its raw bytes are saved to image_save_dir
    (defaults to draco_eval/turn3_outputs/images/) so the judge LLM can load
    it locally without re-fetching a potentially expired URL.

    Args:
        report_text: Full markdown text of the v2 report.
        task_id: Identifier used for logging.
        log_path: File to append skip/debug log lines. Defaults to
                  draco_eval/logs/turn3_probe.log.
        image_save_dir: Directory to save the selected image. Defaults to
                        draco_eval/turn3_outputs/images/.
        model: Full model string (e.g. 'openai:gpt-4.1', 'google_vertexai:gemini-2.5-pro').

    Returns:
        A Turn3Result with the structured analysis, including local_image_path.
    """
    if log_path is None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / "turn3_probe.log"

    if image_save_dir is None:
        image_save_dir = DRACO_DIR / "turn3_outputs" / "images"
    image_save_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(log_path)
    logger.info(f"=== Turn-3 probe starting: {task_id} ===")

    llm = init_chat_model(model=model, temperature=0)

    cited_urls = extract_cited_urls(report_text)
    logger.info(f"Extracted {len(cited_urls)} cited URLs from report")

    if not cited_urls:
        return Turn3Result(
            task_id=task_id,
            task_prompt=task_prompt,
            report_path=report_path,
            no_image_found=True,
            reason_if_none="No cited URLs found in the report.",
            metadata={"vision_calls": 0},
        )

    # Phase 1 — analyse the full report once before visiting any URLs
    report_context, phase1_failed = await _analyse_report(llm, report_text, task_prompt, logger)
    # Extract all headings from the report once — passed to every vision call
    # so the model can copy a heading verbatim rather than paraphrase from memory
    report_headings = extract_headings(report_text)
    logger.info(f"  Extracted {len(report_headings)} section headings from report")

    images_evaluated = 0
    # Global dedup: track every image src already evaluated across all pages.
    # The same site logo or shared asset often appears on multiple cited pages;
    # evaluating it twice would produce the same result and waste an API call.
    seen_image_srcs: set[str] = set()

    async with httpx.AsyncClient(headers=_DEFAULT_HEADERS, follow_redirects=True) as client:
        for url in cited_urls:
            logger.info(f"Visiting: {url}")
            html = await fetch_url(client, url, logger)
            if html is None:
                continue

            images = extract_images(html, url)
            logger.info(f"  Found {len(images)} <img> tags")

            for img in images:
                src: str = img["src"]
                alt: Optional[str] = img["alt"]

                if src in seen_image_srcs:
                    logger.info(f"  [dedup]       skip {src}  (already evaluated)")
                    continue
                seen_image_srcs.add(src)

                if not _passes_url_filter(src):
                    logger.info(f"  [url-filter]  skip {src}")
                    continue

                if not _passes_alt_filter(alt):
                    logger.info(f"  [alt-filter]  skip {src}  alt={alt!r}")
                    continue

                result_dl = await download_image(client, src, logger)
                if result_dl is None:
                    continue

                image_bytes, mime = result_dl
                images_evaluated += 1
                logger.info(
                    f"  [evaluating #{images_evaluated}] {src}  mime={mime}  "
                    f"size={len(image_bytes)//1024}KB"
                )

                fit = await _evaluate_image_fit(llm, report_context, report_headings, task_prompt, image_bytes, mime, src, url)
                logger.info(
                    f"  [vision]  fits_report={fit.fits_report}  "
                    f"what_it_shows={fit.what_it_shows[:80]!r}"
                )

                if not fit.fits_report:
                    logger.info("  [no fit]  discarding, continuing to next image")
                    continue

                # Save image bytes to disk immediately — URL may expire
                ext = "jpg" if mime == "image/jpeg" else "png"
                image_file = image_save_dir / f"{task_id}_turn3.{ext}"
                image_file.write_bytes(image_bytes)
                logger.info(f"  [saved]  {image_file}")
                logger.info(
                    f"=== Turn-3 probe complete: {task_id} — fitting image found "
                    f"({images_evaluated} vision calls) ==="
                )
                return Turn3Result(
                    task_id=task_id,
                    task_prompt=task_prompt,
                    report_path=report_path,
                    local_image_path=str(image_file.resolve()),
                    what_it_shows=fit.what_it_shows,
                    where_it_fits=fit.where_it_fits,
                    section_heading=fit.section_heading,
                    no_image_found=False,
                    reason_if_none="",
                    report_summary=report_context.summary,
                    visual_opportunities=report_context.visual_opportunities,
                    metadata={
                        "image_url": src,
                        "visited_url": url,
                        "vision_calls": images_evaluated,
                        "phase1_failed": phase1_failed,
                    },
                )

            logger.info("  No fitting image on this page — moving to next cited URL")

    logger.info(
        f"=== Turn-3 probe complete: {task_id} — no fitting image found "
        f"({images_evaluated} vision calls) ==="
    )
    return Turn3Result(
        task_id=task_id,
        task_prompt=task_prompt,
        report_path=report_path,
        no_image_found=True,
        reason_if_none=(
            f"Visited all {len(cited_urls)} cited URLs and evaluated "
            f"{images_evaluated} image(s) that passed heuristic filters. "
            "None were judged by the vision model to meaningfully support "
            "a section or claim in the report."
        ),
        report_summary=report_context.summary,
        visual_opportunities=report_context.visual_opportunities,
        metadata={
            "vision_calls": images_evaluated,
            "phase1_failed": phase1_failed,
        },
    )


# ---------------------------------------------------------------------------
# CLI for standalone testing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Turn-3 multimodal probe on one report")
    parser.add_argument(
        "--report",
        required=True,
        help="Path to v2 report markdown file (e.g. draco_eval/reports/task_001_v2.md)",
    )
    parser.add_argument(
        "--task-file",
        default=None,
        help="Path to task JSON file to read task_prompt from. "
             "Defaults to draco_eval/tasks/{task_id}.json",
    )
    parser.add_argument(
        "--task-id",
        required=True,
        help="Task identifier (e.g. task_001)",
    )
    parser.add_argument("--model", default="gpt-4.1")
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Error: report file not found: {report_path}", file=sys.stderr)
        sys.exit(1)

    task_file = Path(args.task_file) if args.task_file else (
        DRACO_DIR / "tasks" / f"{args.task_id}.json"
    )
    task_prompt = ""
    if task_file.exists():
        task_data = json.loads(task_file.read_text(encoding="utf-8"))
        task_prompt = task_data.get("prompt", "")
    else:
        print(f"Warning: task file not found: {task_file}, task_prompt will be empty", file=sys.stderr)

    report_text = report_path.read_text(encoding="utf-8")
    result = await run_turn3_probe(
        report_text=report_text,
        task_id=args.task_id,
        task_prompt=task_prompt,
        report_path=str(report_path.resolve()),
        model=args.model,
    )

    output_dir = DRACO_DIR / "turn3_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.task_id}_turn3.json"
    output_path.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    print(f"\nJSON saved to : {output_path}", file=sys.stderr)
    if result.local_image_path:
        print(f"Image saved to: {result.local_image_path}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(_main())