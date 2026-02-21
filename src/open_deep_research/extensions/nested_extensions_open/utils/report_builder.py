"""HTML report builder - extracted from app.py for shared use.

Used by both:
  - Chainlit app (app.py)
  - Jupyter notebooks (test_research.ipynb)
"""

import uuid
import html as html_lib
from pathlib import Path
from datetime import datetime

from extensions.utils.plotly_utils import load_plotly_figure, figure_to_html

import markdown

# Default src directory (parent of extensions/)
_DEFAULT_SRC_DIR = Path(__file__).resolve().parent.parent.parent  # src/

# Markdown extensions for rich rendering
_MD_EXTENSIONS = [
    'extra',          # tables, fenced_code, footnotes, attr_list, etc.
    'sane_lists',     # proper list handling
    'toc',            # table-of-contents anchor links
    'tables',         # pipe-style markdown tables
]


def _md_to_html(text: str) -> str:
    """Convert markdown text to HTML using Python markdown library."""
    if not text:
        return ""
    return markdown.markdown(text, extensions=_MD_EXTENSIONS)


def build_html_report(
    display_text: str,
    analysis_output: str,
    figures: list,
    chart_explanations: dict,
    sources: list,
    query: str = "",
    sub_queries: list = None,
    conversation_id: str = "",
    src_dir: Path = None,
    extracted_data_summary: str = "",
    data_profile_summary: str = "",
) -> str:
    """Build HTML report using same data displayed in Chainlit.

    Args:
        display_text: Research findings (markdown formatted)
        analysis_output: Data analysis text (markdown formatted)
        figures: List of chart file paths
        chart_explanations: Dict of {chart_path: {title, explanation}}
        sources: List of source URLs
        query: Original user query
        sub_queries: List of sub-queries if enhanced research
        conversation_id: Unique conversation ID
        src_dir: Base src/ directory for resolving relative paths
        extracted_data_summary: CSV data extracted from research (for Analysis Results section)
        data_profile_summary: Data profile output (for Analysis Results section)

    Returns:
        Path to generated HTML report file
    """
    if src_dir is None:
        src_dir = _DEFAULT_SRC_DIR
    src_dir = Path(src_dir)

    sub_queries = sub_queries or []
    report_id = conversation_id[:8] if conversation_id else str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert markdown to rendered HTML server-side
    research_html = _md_to_html(display_text)
    analysis_html = _md_to_html(analysis_output)

    # Build HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research & Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.8;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background-color: white;
            padding: 50px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 4px solid #667eea;
            padding-bottom: 15px;
            margin-bottom: 30px;
            font-size: 2.2em;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
            font-size: 1.8em;
        }}
        h3 {{
            color: #4a5568;
            margin-top: 25px;
            font-size: 1.4em;
        }}
        .metadata {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metadata strong {{
            color: #fff;
            font-weight: 600;
        }}
        .section {{
            margin-bottom: 50px;
        }}
        .content {{
            color: #2d3748;
            line-height: 1.8;
        }}
        .content p {{
            margin-bottom: 15px;
        }}
        .content ul, .content ol {{
            margin-left: 25px;
            margin-bottom: 15px;
        }}
        .content li {{
            margin-bottom: 8px;
        }}
        .content strong {{
            color: #2c3e50;
            font-weight: 600;
        }}
        .content em {{
            font-style: italic;
            color: #4a5568;
        }}
        .content code {{
            background-color: #f7fafc;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            color: #e53e3e;
        }}
        .content pre {{
            background-color: #1e1e2e;
            color: #cdd6f4;
            padding: 18px 22px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.9em;
            line-height: 1.6;
            margin-bottom: 20px;
        }}
        .content pre code {{
            background: none;
            color: inherit;
            padding: 0;
        }}
        .content a {{
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
        }}
        .content a:hover {{
            text-decoration: underline;
            color: #764ba2;
        }}
        .content h1 {{
            font-size: 1.8em;
            color: #2c3e50;
            margin-top: 35px;
            margin-bottom: 15px;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 8px;
        }}
        .content h2 {{
            font-size: 1.5em;
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 12px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 6px;
        }}
        .content h3 {{
            font-size: 1.25em;
            color: #4a5568;
            margin-top: 25px;
            margin-bottom: 10px;
        }}
        .content h4 {{
            font-size: 1.1em;
            color: #555;
            margin-top: 20px;
            margin-bottom: 8px;
        }}
        .content blockquote {{
            border-left: 4px solid #667eea;
            padding: 12px 20px;
            margin: 15px 0;
            background-color: #f0f4ff;
            color: #4a5568;
            border-radius: 0 8px 8px 0;
        }}
        .content table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95em;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            border-radius: 8px;
            overflow: hidden;
        }}
        .content thead {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }}
        .content th {{
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
        }}
        .content td {{
            padding: 10px 16px;
            border-bottom: 1px solid #e2e8f0;
        }}
        .content tbody tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .content tbody tr:hover {{
            background-color: #edf2f7;
        }}
        .content hr {{
            border: none;
            height: 2px;
            background: linear-gradient(to right, #667eea, #764ba2, #667eea);
            margin: 30px 0;
            border-radius: 1px;
        }}
        .chart-block {{
            margin: 40px 0;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
            background-color: #f7fafc;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        .chart-block-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 25px;
            font-size: 1.2em;
            font-weight: 600;
        }}
        .chart-block-visual {{
            padding: 20px;
            background-color: white;
            min-height: 450px;
        }}
        .chart-block-explanation {{
            padding: 20px 25px;
            background: linear-gradient(to right, #edf2f7, #e6fffa);
            border-top: 3px solid #667eea;
            color: #2d3748;
            font-size: 0.95em;
            line-height: 1.8;
        }}
        .chart-block-explanation strong {{
            color: #2c3e50;
            font-weight: 600;
        }}
        .sources {{
            background: linear-gradient(to right, #edf2f7, #e6fffa);
            padding: 25px;
            border-left: 5px solid #667eea;
            margin-top: 40px;
            border-radius: 8px;
        }}
        .sources ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .sources li {{
            margin-bottom: 12px;
            padding-left: 25px;
            position: relative;
        }}
        .sources li:before {{
            content: "#";
            position: absolute;
            left: 0;
        }}
        .sources a {{
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
        }}
        .sources a:hover {{
            text-decoration: underline;
            color: #764ba2;
        }}
        .sub-queries {{
            background-color: #fff5f5;
            padding: 20px;
            border-left: 5px solid #fc8181;
            margin-bottom: 30px;
            border-radius: 8px;
        }}
        .sub-queries ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .sub-queries li {{
            margin-bottom: 10px;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            border-left: 3px solid #fc8181;
        }}
        .error {{
            color: #e53e3e;
            font-style: italic;
            padding: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Research & Analysis Report</h1>

        <div class="metadata">
            <strong>Report ID:</strong> {report_id}<br>
            <strong>Generated:</strong> {timestamp}"""

    if query:
        safe_query = html_lib.escape(query)
        html_content += f"""<br>
            <strong>Query:</strong> {safe_query}"""

    if sub_queries:
        html_content += f"""<br>
            <strong>Research Mode:</strong> Enhanced ({len(sub_queries)} sub-queries)"""

    html_content += """
        </div>"""

    if sub_queries:
        html_content += """
        <div class="sub-queries">
            <h2>Sub-Queries Explored</h2>
            <ul>"""
        for i, sq in enumerate(sub_queries, 1):
            safe_sq = html_lib.escape(sq)
            html_content += f"""
                <li><strong>{i}.</strong> {safe_sq}</li>"""
        html_content += """
            </ul>
        </div>"""

    html_content += f"""
        <div class="section">
            <h2>Research Findings</h2>
            <div class="content">{research_html}</div>
        </div>"""

    # ── Analysis Results section (extracted data + profile + charts + outliers) ──
    has_analysis_results = extracted_data_summary or data_profile_summary or figures
    if has_analysis_results:
        html_content += """
        <div class="section">
            <h2>Analysis Results</h2>"""

        # Extracted Data Summary (table preview)
        if extracted_data_summary:
            # Show first 10 rows as an HTML table
            csv_lines = [l for l in extracted_data_summary.strip().split("\n") if l.strip()]
            if csv_lines:
                html_content += """
            <h3>Extracted Data Summary</h3>
            <div class="content">
                <table>
                    <thead><tr>"""
                headers = csv_lines[0].split(",")
                for h in headers:
                    html_content += f"<th>{html_lib.escape(h.strip())}</th>"
                html_content += """</tr></thead>
                    <tbody>"""
                for row in csv_lines[1:11]:  # First 10 data rows
                    cols = row.split(",")
                    html_content += "<tr>"
                    for c in cols:
                        html_content += f"<td>{html_lib.escape(c.strip())}</td>"
                    html_content += "</tr>"
                html_content += """</tbody>
                </table>"""
                if len(csv_lines) > 11:
                    html_content += f"""<p><em>Showing first 10 of {len(csv_lines) - 1} rows</em></p>"""
                html_content += """
            </div>"""

        # Data Profile Highlights
        if data_profile_summary:
            profile_html = _md_to_html(data_profile_summary)
            html_content += f"""
            <h3>Data Profile Highlights</h3>
            <div class="content">{profile_html}</div>"""

        # Visualizations + Explanations
        if figures:
            html_content += """
            <h3>Visualizations</h3>"""

            for chart_path in figures:
                if not Path(chart_path).exists():
                    candidate = src_dir / chart_path
                    if candidate.exists():
                        chart_path = str(candidate)

                info = chart_explanations.get(chart_path, {})
                title = info.get("title", "Chart")
                explanation = info.get("explanation", "")

                fig = load_plotly_figure(chart_path, src_dir=src_dir)
                if fig:
                    chart_html = figure_to_html(fig, include_plotlyjs=False)
                    html_content += f"""
            <div class="chart-block">
                <div class="chart-block-header">{html_lib.escape(title)}</div>
                <div class="chart-block-visual">
                    {chart_html}
                </div>"""
                    if explanation:
                        html_content += f"""
                <div class="chart-block-explanation">
                    <strong>Interpretation:</strong> {html_lib.escape(explanation)}
                </div>"""
                    html_content += """
            </div>"""

        html_content += """
        </div>"""

    # Legacy: standalone analysis text (when no pipeline data available)
    elif analysis_html:
        html_content += f"""
        <div class="section">
            <h2>Data Analysis</h2>
            <div class="content">{analysis_html}</div>
        </div>"""

    if not has_analysis_results and figures:
        html_content += """
        <div class="section">
            <h2>Visualizations</h2>"""

        for chart_path in figures:
            if not Path(chart_path).exists():
                candidate = src_dir / chart_path
                if candidate.exists():
                    chart_path = str(candidate)

            info = chart_explanations.get(chart_path, {})
            title = info.get("title", "Chart")
            explanation = info.get("explanation", "")

            fig = load_plotly_figure(chart_path, src_dir=src_dir)
            if fig:
                chart_html = figure_to_html(fig, include_plotlyjs=False)
                html_content += f"""
            <div class="chart-block">
                <div class="chart-block-header">{html_lib.escape(title)}</div>
                <div class="chart-block-visual">
                    {chart_html}
                </div>"""
                if explanation:
                    html_content += f"""
                <div class="chart-block-explanation">
                    <strong>Interpretation:</strong> {html_lib.escape(explanation)}
                </div>"""
                html_content += """
            </div>"""

        html_content += """
        </div>"""

    if sources:
        html_content += """
        <div class="sources">
            <h2>Sources</h2>
            <ul>"""
        for source in sources:
            safe_source = html_lib.escape(source)
            html_content += f'                <li><a href="{safe_source}" target="_blank">{safe_source}</a></li>\n'
        html_content += """
            </ul>
        </div>"""

    html_content += """
    </div>
</body>
</html>"""

    # Save report
    report_dir = src_dir / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"report_{report_id}.html"
    report_path.write_text(html_content, encoding='utf-8')

    return str(report_path)


__all__ = ['build_html_report']
