"""Shared Plotly utilities for consistent chart rendering across Chainlit and reports.

This module provides the SINGLE SOURCE OF TRUTH for loading and rendering Plotly charts.
All chart loading should use these functions to ensure visual consistency.
"""
import re
import json
from pathlib import Path
from typing import Optional
import plotly.graph_objects as go


def _extract_json_value(text: str, start: int):
    """Extract a complete JSON value (array or object) using bracket counting.

    Handles nested brackets, strings with escaped quotes, etc.
    Returns (json_string, next_position) or (None, start) on failure.
    """
    if start >= len(text):
        return None, start
    ch = text[start]
    if ch == "[":
        open_c, close_c = "[", "]"
    elif ch == "{":
        open_c, close_c = "{", "}"
    else:
        return None, start

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return text[start : i + 1], i + 1
    return None, start


def load_plotly_figure(html_path: str, src_dir: Optional[Path] = None) -> Optional[go.Figure]:
    """Load a Plotly Figure from a saved Plotly HTML file.

    This is the SINGLE SOURCE OF TRUTH for loading charts.
    Used by both Chainlit UI and HTML report generation to ensure consistency.

    Plotly's write_html() creates files containing a call like:
        Plotly.newPlot("uuid", [<data>], {<layout>}, {<config>})
    We locate that call, extract the data and layout JSON using robust
    bracket-counting (not regex), and reconstruct the Figure.

    Args:
        html_path: Path to the Plotly HTML file (relative or absolute)
        src_dir: Optional source directory for resolving relative paths

    Returns:
        go.Figure object or None if loading fails
    """
    abs_path = Path(html_path)
    if not abs_path.is_absolute():
        if src_dir:
            abs_path = src_dir / html_path
        else:
            # Try current working directory
            abs_path = Path.cwd() / html_path

    if not abs_path.exists():
        return None

    try:
        html = abs_path.read_text(encoding="utf-8")
    except Exception:
        return None

    # Find Plotly.newPlot("uuid",  — skip the UUID to reach the data array
    match = re.search(r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*', html)
    if not match:
        return None

    pos = match.end()

    # Extract data (JSON array)
    data_json, next_pos = _extract_json_value(html, pos)
    if data_json is None:
        return None

    try:
        data = json.loads(data_json)
    except json.JSONDecodeError:
        return None

    # Skip comma + whitespace to reach layout object
    while next_pos < len(html) and html[next_pos] in ", \n\r\t":
        next_pos += 1

    # Extract layout (JSON object) — optional, chart still works without it
    layout = {}
    layout_json, _ = _extract_json_value(html, next_pos)
    if layout_json:
        try:
            layout = json.loads(layout_json)
        except json.JSONDecodeError:
            pass

    return go.Figure(data=data, layout=layout)


def figure_to_html(fig: go.Figure, include_plotlyjs: bool = False) -> str:
    """Convert a Plotly Figure to HTML string for embedding in reports.

    Args:
        fig: Plotly Figure object
        include_plotlyjs: Whether to include Plotly.js (False = use CDN)

    Returns:
        HTML string for embedding
    """
    return fig.to_html(
        include_plotlyjs='cdn' if not include_plotlyjs else True,
        config={
            'responsive': True,
            'displayModeBar': True,
            'displaylogo': False,
        },
        div_id=None,  # Auto-generate unique ID
        full_html=False  # Just the div, not full HTML page
    )


__all__ = ['load_plotly_figure', 'figure_to_html']
