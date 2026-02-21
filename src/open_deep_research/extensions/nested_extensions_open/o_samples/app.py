"""
Chainlit frontend for Research & Analysis.

Run from the src/ directory:
    cd src
    chainlit run app.py

Features:
  - User message displayed (Chainlit native)
  - Initial clarification message before pipeline runs
  - Status pill: "Research complete -- 47s"
  - Agent activity as collapsible Steps (research, data_analysis, report_generation)
  - Nested sub-steps inside research (web_search queries shown)
  - Per-step execution time display
  - Chart details inside data_analysis step body
  - Sub-queries display (enhanced research mode)
  - Sources with clickable links + count + "more" overflow
  - Inline Plotly chart rendering with title + explanation
  - HTML report download (cl.File)
  - CSV extracted data download (generated from analysis output)
  - Extracted data table and data profile display
  - Final output as formatted markdown
  - Feedback buttons (Chainlit native)
  - Input bar with follow-up (Chainlit native)
"""

import os
import sys
import re
import json
import csv
import io
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime

import chainlit as cl
from chainlit.input_widget import Switch
from dotenv import load_dotenv

# -- Environment --
# Load .env from project root (one level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Ensure src/ is on sys.path so extension imports work
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# -- Imports from existing codebase (zero modifications to core) --
from extensions.agents.master_agent import MasterAgent
from extensions.utils.plotly_utils import load_plotly_figure, figure_to_html
from extensions.utils.report_builder import build_html_report


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _resolve_path(path: str) -> Path:
    """Resolve a path that may be relative to CWD or src/."""
    p = Path(path)
    if p.exists():
        return p.resolve()
    candidate = SRC_DIR / path
    if candidate.exists():
        return candidate.resolve()
    return p


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def _extract_csv_from_charts(chart_paths: list, chart_explanations: dict) -> str | None:
    """Extract CSV data from Plotly chart HTML files for download.

    Reads each chart's data traces and builds a combined CSV with columns:
    chart_title, trace_name, x, y.  Returns the CSV string or None.
    """

    rows = []
    for chart_path in chart_paths:
        fig = load_plotly_figure(chart_path, src_dir=SRC_DIR)
        if fig is None:
            continue

        info = chart_explanations.get(chart_path, {})
        chart_title = info.get("title", Path(chart_path).stem)

        for trace in fig.data:
            try:
                trace_name = getattr(trace, "name", "") or ""
                x_vals = getattr(trace, "x", None)
                y_vals = getattr(trace, "y", None)

                if x_vals is None and y_vals is None:
                    continue
                if x_vals is not None and not isinstance(x_vals, (list, tuple)):
                    continue
                if y_vals is not None and not isinstance(y_vals, (list, tuple)):
                    continue

                x_vals = x_vals or ()
                y_vals = y_vals or ()
                length = max(len(x_vals), len(y_vals))
                for i in range(length):
                    x = x_vals[i] if i < len(x_vals) else ""
                    y = y_vals[i] if i < len(y_vals) else ""
                    rows.append({
                        "chart": chart_title,
                        "trace": trace_name,
                        "x": str(x),
                        "y": str(y),
                    })
            except Exception:
                continue

    if not rows:
        return None

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["chart", "trace", "x", "y"])
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


AGENT_LABELS = {
    "research": "deep_research",
    "data_analysis": "data_analysis",
    "report_generation": "generate_report",
}


# --------------------------------------------------------------------------
# Chainlit lifecycle
# --------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """Initialize MasterAgent once per session and greet the user."""
    settings = await cl.ChatSettings(
        [
            Switch(
                id="enhanced_research",
                label="Enhanced Research (deeper, 3-4x slower)",
                initial=False,
                description="Breaks query into sub-queries for more comprehensive results.",
            ),
        ]
    ).send()

    cl.user_session.set("enhanced_research", settings.get("enhanced_research", False))

    # Generate a session_id for grouping conversations in this browser session
    import uuid as _uuid
    session_id = str(_uuid.uuid4())
    cl.user_session.set("session_id", session_id)

    init_msg = cl.Message(content="Initializing Research Agent...")
    await init_msg.send()

    try:
        agent = MasterAgent(
            enable_state_persistence=True,
            storage_type="sqlite",
            use_enhanced_research=False,
        )
        cl.user_session.set("agent", agent)
        init_msg.content = (
            "**Welcome to Research & Analysis!**\n\n"
            "I can help you with:\n"
            "- **Deep research** on any topic\n"
            "- **Data analysis** with interactive Plotly charts\n"
            "- **Comprehensive reports** (HTML download)\n\n"
            "Toggle **Enhanced Research** in the settings panel "
            "for deeper, multi-query research.\n\n"
            "Just type your question to get started."
        )
        await init_msg.update()
    except Exception as e:
        init_msg.content = f"Failed to initialize agent: {e}"
        await init_msg.update()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    """Re-create MasterAgent when user toggles Enhanced Research."""
    enhanced = settings.get("enhanced_research", False)
    prev = cl.user_session.get("enhanced_research", False)

    if enhanced == prev:
        return

    cl.user_session.set("enhanced_research", enhanced)

    mode_label = "Enhanced" if enhanced else "Standard"
    status_msg = cl.Message(content=f"Switching to **{mode_label}** research mode...")
    await status_msg.send()

    try:
        agent = MasterAgent(
            enable_state_persistence=True,
            storage_type="sqlite",
            use_enhanced_research=enhanced,
        )
        cl.user_session.set("agent", agent)
        status_msg.content = f"Switched to **{mode_label}** research mode."
        await status_msg.update()
    except Exception as e:
        status_msg.content = f"Failed to switch mode: {e}"
        await status_msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user queries."""
    agent: MasterAgent = cl.user_session.get("agent")
    if agent is None:
        await cl.Message(content="Agent not initialized. Please refresh the page.").send()
        return

    query = message.content
    start_time = datetime.now()
    enhanced = cl.user_session.get("enhanced_research", False)
    mode_label = "Enhanced" if enhanced else "Standard"

    # 1. Initial clarification message
    await cl.Message(
        content=(
            f"Great question! I'll research this thoroughly for you. "
            f"**Mode: {mode_label}**\n\n"
            f"This may take a few minutes -- you can watch the progress below."
        ),
    ).send()

    # 2. Status pill: running
    status_msg = cl.Message(content="**Research in progress...**")
    await status_msg.send()

    # 3. Run the agent
    session_id = cl.user_session.get("session_id")
    try:
        result = await agent.run_async(query, session_id=session_id)
    except Exception as e:
        status_msg.content = f"**Research failed:** {e}"
        await status_msg.update()
        return

    state = result.get("state", {})
    output = result.get("output", "")
    status = result.get("status", "unknown")
    error = result.get("error")
    agents_used = result.get("agents_used", [])
    exec_time = result.get("execution_time", 0)

    # 4. Handle errors
    if status == "error":
        status_msg.content = f"**Research failed:** {error}"
        await status_msg.update()
        return

    # 5. Status pill: done
    status_msg.content = f"**Research complete -- {_format_time(exec_time)}**"
    await status_msg.update()

    # 6. Agent activity Steps
    sources = state.get("sources", [])
    sub_queries = state.get("sub_queries", [])
    charts = state.get("charts", [])
    chart_explanations = state.get("chart_explanations", {})
    extracted_data = state.get("extracted_data", "")
    data_profile = state.get("data_profile", "")

    for agent_name in agents_used:
        label = AGENT_LABELS.get(agent_name, agent_name)

        if agent_name == "research":
            async with cl.Step(name=f"{label} -- {query[:60]}", type="tool") as research_step:
                if sub_queries:
                    for sq in sub_queries:
                        async with cl.Step(name=f"web_search -- \"{sq}\"", type="tool") as sub_step:
                            sub_step.output = f"Searched: {sq}"

                    async with cl.Step(name=f"synthesize -- combining {len(sub_queries)} research threads", type="tool") as synth_step:
                        synth_step.output = f"Merged {len(sub_queries)} sub-query results into final report"

                research_step.output = (
                    f"**Status:** Completed\n"
                    f"**Sources found:** {len(sources)}\n"
                    f"**Sub-queries:** {len(sub_queries) if sub_queries else 'N/A (standard mode)'}"
                )

        elif agent_name == "data_analysis":
            async with cl.Step(name=f"{label} -- Creating {len(charts)} visualizations", type="tool") as analysis_step:
                body_lines = []
                if extracted_data:
                    table_count = extracted_data.count("--- Table")
                    body_lines.append(f"Extracted {table_count} data table(s) from research findings.")
                if data_profile:
                    body_lines.append("Profiled data: statistics, distributions, column analysis.")
                body_lines.append(f"Created {len(charts)} visualization(s):")
                for cp in charts:
                    info = chart_explanations.get(cp, {})
                    chart_title = info.get("title", Path(cp).stem)
                    chart_type = Path(cp).stem.split("_")[0] if "_" in Path(cp).stem else "chart"
                    body_lines.append(f"  - `{chart_type}` -- {chart_title}")
                analysis_step.output = "\n".join(body_lines)

    # 7. Final output message (research findings)
    full_text = state.get("final_report", "") or output
    chat_text = full_text
    if chat_text and len(chat_text) > 8000:
        chat_text = chat_text[:8000] + "\n\n---\n*[Output truncated -- see downloadable report for complete results]*"
    if chat_text:
        await cl.Message(content=chat_text).send()

    # 7b. Extracted data summary (if pipeline produced it)
    if extracted_data:
        # Show a brief preview
        lines = [l for l in extracted_data.strip().split("\n") if l.strip()]
        preview_lines = lines[:15]  # first 15 lines
        preview = "\n".join(preview_lines)
        if len(lines) > 15:
            preview += f"\n\n*... {len(lines) - 15} more lines. See full data in downloaded CSV.*"
        await cl.Message(
            content=f"**Extracted Data Preview:**\n\n```\n{preview}\n```"
        ).send()

    # 7c. Data profile highlights (if pipeline produced it)
    if data_profile:
        # Truncate if very long
        profile_text = data_profile
        if len(profile_text) > 3000:
            profile_text = profile_text[:3000] + "\n\n*[Profile truncated]*"
        await cl.Message(
            content=f"**Data Profile:**\n\n{profile_text}"
        ).send()

    # 8. Plotly charts inline
    for chart_path in charts:
        info = chart_explanations.get(chart_path, {})
        title = info.get("title", "Chart")
        explanation = info.get("explanation", "")

        fig = load_plotly_figure(chart_path, src_dir=SRC_DIR)
        if fig is not None:
            elements = [cl.Plotly(name=title, figure=fig, display="inline")]
            caption = f"**{title}**"
            if explanation:
                caption += f"\n\n{explanation}"
            await cl.Message(content=caption, elements=elements).send()
        else:
            resolved = _resolve_path(chart_path)
            if resolved.exists():
                elements = [cl.File(name=f"{title}.html", path=str(resolved), display="inline")]
                await cl.Message(
                    content=f"**{title}** *(interactive chart file)*\n\n{explanation}",
                    elements=elements,
                ).send()

    # 9. Sources block
    if sources:
        source_lines = []
        for i, src in enumerate(sources[:15], 1):
            source_lines.append(f"{i}. [{src}]({src})")
        if len(sources) > 15:
            source_lines.append(f"*+ {len(sources) - 15} more sources...*")
        await cl.Message(
            content=f"**Sources ({len(sources)} references):**\n\n" + "\n".join(source_lines),
        ).send()

    # 10. Downloadable files (CSV + HTML report)
    download_elements = []

    # CSV data export
    if charts:
        csv_content = _extract_csv_from_charts(charts, chart_explanations)
        if csv_content:
            csv_dir = SRC_DIR / "outputs" / "data"
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = csv_dir / f"data_extracted_{state.get('conversation_id', 'unknown')[:8]}.csv"
            csv_path.write_text(csv_content, encoding="utf-8")
            download_elements.append(
                cl.File(name=csv_path.name, path=str(csv_path), display="inline")
            )

    # HTML report generation
    try:
        report_path = build_html_report(
            display_text=full_text,
            analysis_output=state.get("analysis_output", ""),
            figures=charts,
            chart_explanations=chart_explanations,
            sources=sources,
            query=state.get("query", ""),
            sub_queries=sub_queries,
            conversation_id=state.get("conversation_id", ""),
            src_dir=SRC_DIR,
            extracted_data_summary=extracted_data,
            data_profile_summary=data_profile,
        )
        download_elements.append(
            cl.File(name=Path(report_path).name, path=report_path, display="inline")
        )
    except Exception as e:
        print(f"Error generating HTML report: {e}")

    if download_elements:
        await cl.Message(
            content="**Downloads:**",
            elements=download_elements,
        ).send()

    # 11. Sub-queries (if enhanced research was used)
    if sub_queries:
        sq_lines = "\n".join(f"{i}. {sq}" for i, sq in enumerate(sub_queries, 1))
        await cl.Message(
            content=f"**Sub-queries explored ({len(sub_queries)}):**\n\n{sq_lines}",
        ).send()

    # 12. Empty output fallback
    if not output and not charts:
        await cl.Message(content="No output was generated. Please try a different query.").send()
