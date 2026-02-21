"""Data Analysis Agent - handles data profiling, visualization, and math."""
from typing import Dict, Any, List
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
import csv
from io import StringIO
import os
import json
import re
import logging
import time

# Import tools from extensions
from extensions.tools.math_tools import MATH_TOOLS
from extensions.tools.data_profiling import profile_data_tool, profile_data
from extensions.tools.data_extraction import extract_data_tool, extract_data
from extensions.tools.visualization import create_chart_tool, create_chart, detect_outliers_tool, detect_outliers

# Import chart explanation and path extraction schemas
from extensions.models.tool_schemas import ChartExplanation, ChartExplanations, ExtractedChartPaths

# Import structured data extraction schema
from extensions.models.extracted_data_schema import ExtractedDataset

# Import LLM factory
from extensions.utils.llm_factory import get_extensions_llm

logger = logging.getLogger(__name__)


class DataAnalysisAgent:
    """Agent for data analysis, visualization, and calculations.

    Combines:
    - Data profiling
    - Data extraction
    - Visualization (Plotly)
    - Math operations

    Two modes:
    - run(): Free-form agent (LLM decides tool order) for standalone queries
    - run_pipeline(): Enforced sequential chain (extract→profile→visualize) for research output
    """

    def __init__(self, provider: str = None, model: str = None):
        """Initialize data analysis agent.

        Args:
            provider: LLM provider ('azure' or 'gemini'). Uses env fallback if None.
            model:    Model name. Uses env/provider default if None.
        """
        self.name = "data_analysis"

        # Collect all tools
        self.tools = [
            profile_data_tool,
            extract_data_tool,
            create_chart_tool,
            detect_outliers_tool,
            *MATH_TOOLS
        ]

        # Create LLM using unified factory — pass through user-selected provider/model
        logger.info(f"[AGENT] DataAnalysisAgent: Initializing with provider={provider}, model={model}")
        self.llm = get_extensions_llm(provider=provider, model=model)

        # Structured LLM for chart explanation extraction
        try:
            self.explanation_llm = self.llm.with_structured_output(ChartExplanations)
            self.path_extractor_llm = self.llm.with_structured_output(ExtractedChartPaths)
        except Exception as e:
            logger.warning(f"[AGENT] DataAnalysisAgent: with_structured_output not supported: {e}. Using fallback.")
            self.explanation_llm = None
            self.path_extractor_llm = None

        # Structured LLM for data extraction (Pydantic schema → guaranteed format)
        try:
            self.extraction_llm = self.llm.with_structured_output(ExtractedDataset)
        except Exception as e:
            logger.warning(f"[AGENT] DataAnalysisAgent: with_structured_output for extraction not supported: {e}.")
            self.extraction_llm = None
        
        self._system_prompt = """You are a data analysis assistant with advanced visualization and outlier detection capabilities.

You have access to the following tools:
- profile_data: Analyze and profile data from any source (statistics, distributions, patterns)
- extract_data: Extract structured data from text
- create_chart: Create interactive Plotly visualizations
- detect_outliers: Detect outliers using IQR, Z-score, or Isolation Forest methods
- add, subtract, multiply, divide, calculate: Perform mathematical operations

**CRITICAL RULES:**
- ALWAYS use the create_chart tool to create visualizations.
- Charts are saved automatically to outputs/charts/ as interactive Plotly HTML files.
- NEVER invent or fabricate file paths. Only reference paths returned by create_chart.
- Aim to create 2-4 different charts covering different aspects of the data.

Always provide clear, concise analysis with actionable insights."""
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run data analysis agent (free-form mode).

        Args:
            query: Analysis task/query

        Returns:
            Dictionary with analysis results including chart explanations
        """
        from langgraph.prebuilt import create_react_agent
        from langchain_core.messages import HumanMessage

        start_time = datetime.now()

        try:
            agent = create_react_agent(self.llm, self.tools, prompt=self._system_prompt)
            result = agent.invoke({"messages": [HumanMessage(content=query)]})

            messages = result.get("messages", [])
            output = messages[-1].content if messages else ""

            # Extract chart paths from tool messages
            intermediate_steps = []
            for msg in messages:
                tool_name = getattr(msg, "name", None)
                if tool_name == "create_chart":
                    intermediate_steps.append((type("A", (), {"tool": "create_chart"})(), msg.content))

            charts = self._extract_chart_paths(intermediate_steps, output)
            chart_explanations = self._extract_chart_explanations(output, charts) if charts else {}

            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                "agent_name": self.name,
                "status": "completed",
                "output": output,
                "charts": charts,
                "chart_explanations": chart_explanations,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "error": None,
            }

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                "agent_name": self.name,
                "status": "error",
                "output": "",
                "charts": [],
                "chart_explanations": {},
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }
    
    def run_pipeline(self, research_text: str) -> Dict[str, Any]:
        """Run enforced sequential analysis pipeline on research output.

        Extraction: LLM (primary) → regex (fallback)
        Then for each extracted table: profile → plan charts → create charts → outliers
        Finally: extract chart explanations across all charts.

        Args:
            research_text: Research report text to analyze

        Returns:
            Dictionary with extracted data, profile, charts, explanations
        """
        pipeline_start = time.time()
        logger.info("[PIPELINE] ========================================")
        logger.info("[PIPELINE] Starting enforced analysis pipeline")
        logger.info(f"[PIPELINE] Input: {len(research_text)} chars of research text")

        all_extracted_data = ""
        all_profiles = ""
        charts = []
        chart_explanations = {}
        outlier_analyses = []
        outlier_chart_meta = {}  # {path: {"title": ..., "explanation": ...}}
        total_chart_cap = 6  # max charts across all tables    #### number of plots generated.

        try:
            # ── Step 1 PRIMARY: LLM-powered extraction (multiple tables) ──
            logger.info("[PIPELINE] Step 1/5: Extracting data (LLM primary)...")
            step1_start = time.time()
            csv_tables = self._llm_extract_data(research_text)
            step1_time = time.time() - step1_start

            if csv_tables:
                logger.info(
                    f"[PIPELINE] Step 1/5: LLM extracted {len(csv_tables)} tables ({step1_time:.1f}s)"
                )
            else:
                # ── Step 1 FALLBACK: Regex extraction (single table) ──
                logger.info("[PIPELINE] Step 1/5: LLM extraction returned nothing, trying regex fallback...")
                step1b_start = time.time()
                regex_data = extract_data(research_text, format="csv")
                step1b_time = time.time() - step1b_start

                regex_failed = (
                    not regex_data
                    or "[ERROR]" in regex_data
                    or "No structured data found" in regex_data
                )

                if regex_failed:
                    logger.warning(
                        f"[PIPELINE] Step 1/5: No data found by LLM ({step1_time:.1f}s) "
                        f"or regex ({step1b_time:.1f}s)"
                    )
                    return {
                        "extracted_data": "",
                        "data_profile": "",
                        "charts": [],
                        "chart_explanations": {},
                        "outlier_analysis": [],
                        "status": "completed",
                        "output": "No structured data could be extracted from the research material for visualization.",
                        "execution_time": time.time() - pipeline_start,
                        "error": None
                    }
                else:
                    csv_tables = [regex_data]
                    logger.info(
                        f"[PIPELINE] Step 1/5: Regex fallback extracted 1 table ({step1b_time:.1f}s)"
                    )

            # ── Steps 2-5: Process each table ──
            for table_idx, csv_data in enumerate(csv_tables, 1):
                if len(charts) >= total_chart_cap:
                    logger.info(f"[PIPELINE] Chart cap ({total_chart_cap}) reached, skipping remaining tables")
                    break

                lines = [l for l in csv_data.strip().split("\n") if l.strip()]
                row_count = max(0, len(lines) - 1)
                # Use csv.reader to correctly count columns (handles quoted commas)
                col_count = len(next(csv.reader([lines[0]]))) if lines else 0
                logger.info(
                    f"[PIPELINE] Table {table_idx}/{len(csv_tables)}: "
                    f"{row_count} rows, {col_count} columns"
                )

                # Accumulate extracted data for the return value
                all_extracted_data += f"\n--- Table {table_idx} ---\n{csv_data}\n"

                # ── Step 2: Profile data ──
                logger.info(f"[PIPELINE] Step 2/5: Profiling table {table_idx}...")
                step2_start = time.time()
                table_profile = profile_data(csv_data)
                step2_time = time.time() - step2_start
                all_profiles += f"\n--- Table {table_idx} ---\n{table_profile}\n"
                logger.info(f"[PIPELINE] Step 2/5: Profile complete ({step2_time:.1f}s)")

                # ── Step 3: Plan charts for this table ──
                remaining_cap = total_chart_cap - len(charts)
                logger.info(f"[PIPELINE] Step 3/5: Planning charts for table {table_idx} (cap: {remaining_cap} remaining)...")
                step3_start = time.time()
                chart_plan = self._plan_charts(csv_data, table_profile)
                # Limit per table: 1-2 charts, but also respect total cap
                chart_plan = chart_plan[:min(1, remaining_cap)]    ### change make min 2 for more charts per table
                step3_time = time.time() - step3_start
                chart_types = [c.get("chart_type", "unknown") for c in chart_plan]
                logger.info(
                    f"[PIPELINE] Step 3/5: Planned {len(chart_plan)} charts: "
                    f"{', '.join(chart_types)} ({step3_time:.1f}s)"
                )

                # ── Step 4: Create charts ──
                step4_start = time.time()
                for i, spec in enumerate(chart_plan, 1):
                    logger.info(
                        f"[PIPELINE] Step 4/5: Table {table_idx}, chart {i}/{len(chart_plan)} "
                        f"({spec.get('chart_type', 'unknown')})..."
                    )
                    try:
                        result = create_chart(
                            data=csv_data,
                            chart_type=spec.get("chart_type", "bar"),
                            title=spec.get("title", "Chart"),
                            x_column=spec.get("x_column", ""),
                            y_column=spec.get("y_column", ""),
                        )
                        if "File:" in result:
                            path = result.split("File:")[1].split("\n")[0].strip()
                            if path and os.path.exists(path):
                                charts.append(path)
                                logger.info(f"[PIPELINE] Step 4/5: Chart saved: {path}")
                            else:
                                logger.warning(f"[PIPELINE] Step 4/5: Chart path not found: {path}")
                        else:
                            logger.warning(f"[PIPELINE] Step 4/5: Chart creation returned unexpected format")
                    except Exception as e:
                        logger.error(f"[PIPELINE] Step 4/5: Chart failed: {e}")
                step4_time = time.time() - step4_start
                logger.info(f"[PIPELINE] Step 4/5: Created charts for table {table_idx} ({step4_time:.1f}s)")

                # ── Step 5: Outlier detection (up to 2 numeric columns per table) ──
                ### to diable outliers put: if False:
                
                if col_count >= 2 and row_count >= 5 and len(charts) < total_chart_cap:
                #if False:
                    try:
                        # Use csv.reader to correctly parse headers/cells (handles quoted commas)
                        header_cols = next(csv.reader([lines[0]])) if lines else []

                        # Find numeric columns by sampling first 3 data rows
                        numeric_cols = []
                        for col_idx, col_name in enumerate(header_cols[1:], start=1):
                            col_name = col_name.strip()
                            sample_values = []
                            for data_line in lines[1:4]:
                                cells = next(csv.reader([data_line]))
                                if col_idx < len(cells):
                                    sample_values.append(cells[col_idx].strip())
                            numeric_count = sum(
                                1 for v in sample_values
                                if v.replace('.', '', 1).replace('-', '', 1).isdigit()
                            )
                            if sample_values and numeric_count >= len(sample_values) / 2:
                                numeric_cols.append(col_name)

                        # Run outlier detection on up to 2 numeric columns
                        for numeric_col in numeric_cols[:2]:
                            if len(charts) >= total_chart_cap:
                                break
                            try:
                                outlier_result = detect_outliers(
                                    data=csv_data,
                                    column=numeric_col,
                                    method="iqr"
                                )
                                # detect_outliers returns "Visualization:" (not "File:")
                                if "File:" in outlier_result or "Visualization:" in outlier_result:
                                    if "File:" in outlier_result:
                                        outlier_path = outlier_result.split("File:")[1].split("\n")[0].strip()
                                    else:
                                        outlier_path = outlier_result.split("Visualization:")[1].split("\n")[0].strip()
                                    if outlier_path and os.path.exists(outlier_path):
                                        charts.append(outlier_path)
                                        outlier_analyses.append(outlier_result)
                                        # Store column-aware title (numeric_col is available here)
                                        explanation_text = outlier_result.split("Interpretation:")[0].strip() if "Interpretation:" in outlier_result else ""
                                        outlier_chart_meta[outlier_path] = {
                                            "title": f"Outlier Detection: {numeric_col}",
                                            "explanation": explanation_text,
                                        }
                                        logger.info(f"[PIPELINE] Step 5/5: Outlier chart saved: {outlier_path}")
                                else:
                                    outlier_analyses.append(outlier_result)
                            except Exception as e:
                                logger.info(f"[PIPELINE] Step 5/5: Outlier detection failed for column '{numeric_col}': {e}")
                    except Exception as e:
                        logger.info(f"[PIPELINE] Step 5/5: Outlier detection skipped for table {table_idx}: {e}")

            # ── Extract chart explanations (across all charts) ──
            if charts:
                chart_explanations = self._extract_chart_explanations(
                    f"Data profiles:\n{all_profiles}\n\nCharts created from extracted research data.",
                    charts
                )
                # Override outlier chart titles with column-aware names
                # (LLM may not know which column the outlier chart is for)
                chart_explanations.update(outlier_chart_meta)

            total_time = time.time() - pipeline_start
            total_rows = sum(
                max(0, len([l for l in csv_tbl.strip().split("\n") if l.strip()]) - 1)
                for csv_tbl in csv_tables
            )
            logger.info(
                f"[PIPELINE] Complete: {len(csv_tables)} tables, {total_rows} total rows, "
                f"{len(charts)} charts, {total_time:.1f}s"
            )
            logger.info("[PIPELINE] ========================================")

            return {
                "extracted_data": all_extracted_data.strip(),
                "data_profile": all_profiles.strip(),
                "charts": charts,
                "chart_explanations": chart_explanations,
                "outlier_analysis": outlier_analyses if outlier_analyses else None,
                "status": "completed",
                "output": (
                    f"Pipeline complete: {len(csv_tables)} tables, "
                    f"{total_rows} total rows, created {len(charts)} charts."
                ),
                "execution_time": total_time,
                "error": None
            }

        except Exception as e:
            total_time = time.time() - pipeline_start
            logger.error(f"[PIPELINE] Failed: {e}")
            return {
                "extracted_data": all_extracted_data.strip(),
                "data_profile": all_profiles.strip(),
                "charts": charts,
                "chart_explanations": chart_explanations,
                "outlier_analysis": outlier_analyses if outlier_analyses else None,
                "status": "error",
                "output": f"Pipeline error: {e}",
                "execution_time": total_time,
                "error": str(e)
            }

    def _llm_extract_data(self, research_text: str) -> List[str]:
        """Use LLM with Pydantic structured output to extract multiple tables from research text.

        Primary extraction method. Finds both existing tables and numerical data
        embedded in prose, organizing them into separate structured tables.

        Args:
            research_text: Research report text (markdown or plain text)

        Returns:
            List of CSV strings (one per extracted table). Empty list on failure.
        """
        if not self.extraction_llm:
            logger.info("[PIPELINE] extraction_llm not available, skipping LLM extraction")
            return []

        # Truncate to stay within token limits (~7500 tokens for 30k chars)
        max_chars = 30000
        truncated = research_text[:max_chars]
        if len(research_text) > max_chars:
            logger.warning(
                f"[PIPELINE] Research text truncated from {len(research_text)} to {max_chars} chars "
                f"for LLM extraction. Data beyond this limit will not be extracted."
            )

        prompt = (
            "You are a data extraction specialist. Extract ALL quantitative data from the "
            "following research text into structured tables.\n\n"
            "WHAT TO EXTRACT:\n"
            "- Tables already present in the text (preserve them exactly)\n"
            "- Numbers, percentages, monetary values, statistics embedded in prose\n"
            "- Comparisons, rankings, counts, rates, ratios\n"
            "- Time-series data points mentioned across paragraphs\n\n"
            "RULES:\n"
            "- Extract ONLY data explicitly stated in the text\n"
            "- Do NOT estimate, invent, infer, or fabricate any numbers\n"
            "- Every value must come directly from the text\n"
            "- Group related data points into the same table\n"
            "- Give each table a descriptive name\n"
            "- Use clean numeric values (no currency symbols or units in value columns)\n"
            "- Add a units column if values have different units\n"
            "- Each table must have at least 2 columns and at least 2 data rows\n"
            "- Skip a table if fewer than 2 data points are available for it\n\n"
            f"RESEARCH TEXT:\n{truncated}"
        )

        try:
            result = self.extraction_llm.invoke(prompt)

            csv_tables = []
            for table in result.tables:
                # Validate: at least 2 columns, at least 1 data row
                if len(table.headers) < 2 or len(table.rows) < 1:
                    logger.info(
                        f"[PIPELINE] LLM extraction: skipping table '{table.table_name}' "
                        f"(headers={len(table.headers)}, rows={len(table.rows)})"
                    )
                    continue

                # Convert to CSV string (csv.writer handles quoting for commas, quotes, etc.)
                buf = StringIO()
                writer = csv.writer(buf)
                writer.writerow(table.headers)
                for row in table.rows:
                    # Pad or trim row to match header count
                    padded = row[:len(table.headers)]
                    while len(padded) < len(table.headers):
                        padded.append("")
                    writer.writerow(padded)
                csv_str = buf.getvalue().strip()
                csv_tables.append(csv_str)
                logger.info(
                    f"[PIPELINE] LLM extraction: table '{table.table_name}' "
                    f"({len(table.rows)} rows, {len(table.headers)} cols)"
                )

            logger.info(f"[PIPELINE] LLM extraction: {len(csv_tables)} valid tables extracted")
            return csv_tables

        except Exception as e:
            logger.warning(f"[PIPELINE] LLM data extraction failed: {e}")
            return []

    def _plan_charts(self, extracted_data: str, data_profile: str) -> List[Dict[str, str]]:
        """Use LLM to decide which charts to create based on the data profile.

        Args:
            extracted_data: CSV data
            data_profile: Profile report from profile_data

        Returns:
            List of chart specifications [{chart_type, title, x_column, y_column}]
        """
        prompt = f"""Based on this data profile, decide 2-4 appropriate charts to create.

Data (first 500 chars):
{extracted_data[:500]}

Profile:
{data_profile[:2000]}

Return a JSON array of chart specs. Each spec must have:
- chart_type: one of bar, line, scatter, pie, histogram, box, violin, heatmap, density, bubble
- title: descriptive title
- x_column: column name for x-axis
- y_column: column name for y-axis

Choose chart types that best fit the data:
- Bar for comparisons (categories vs values)
- Line for trends over time
- Pie for proportions/distributions
- Scatter for correlations
- Histogram for distributions of a single variable

Return ONLY a valid JSON array, no other text."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Try to parse full content as JSON array first (handles clean responses)
            chart_plan = None
            content_stripped = content.strip().strip('`').strip()
            if content_stripped.startswith('json'):
                content_stripped = content_stripped[4:].strip()
            if content_stripped.startswith('['):
                try:
                    chart_plan = json.loads(content_stripped)
                except json.JSONDecodeError:
                    pass

            # Fallback: regex extract JSON array from mixed text
            if chart_plan is None:
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    chart_plan = json.loads(json_match.group())

            if chart_plan:
                # Validate and limit to 4 charts
                valid = []
                for spec in chart_plan[:4]:
                    if isinstance(spec, dict) and "chart_type" in spec:
                        valid.append(spec)
                return valid if valid else [{"chart_type": "bar", "title": "Data Overview", "x_column": "", "y_column": ""}]
        except Exception as e:
            logger.warning(f"[PIPELINE] Chart planning failed: {e}. Using default bar chart.")

        return [{"chart_type": "bar", "title": "Data Overview", "x_column": "", "y_column": ""}]

    def _extract_chart_paths(self, intermediate_steps: list, output: str) -> List[str]:
        """Extract chart file paths using a layered approach.
        
        Primary: Read paths directly from create_chart tool results in
                 intermediate_steps (deterministic, zero parsing of LLM text).
        Fallback: Use structured LLM output to extract paths from the agent's
                  text output (handles cases where tool wasn't used or
                  intermediate_steps format changes).
        
        Both layers validate paths with os.path.exists before returning.
        
        Args:
            intermediate_steps: List of (AgentAction, tool_result) tuples from AgentExecutor
            output: Agent's final text output
            
        Returns:
            List of validated chart file paths
        """
        # ── Primary: extract from intermediate_steps ──
        charts = self._extract_from_intermediate_steps(intermediate_steps)
        if charts:
            return charts
        
        # ── Fallback: LLM structured extraction from output text ──
        charts = self._extract_from_output_llm(output)
        if charts:
            return charts
        
        return []
    
    def _extract_from_intermediate_steps(self, intermediate_steps: list) -> List[str]:
        """Extract chart paths from AgentExecutor intermediate tool results.
        
        Looks for create_chart tool calls and extracts the file path from
        our own controlled return format ('File: <path>').
        
        Args:
            intermediate_steps: List of (AgentAction, tool_result) tuples
            
        Returns:
            List of validated chart file paths
        """
        charts = []
        try:
            for action, result in intermediate_steps:
                # Check if this was a create_chart tool call
                tool_name = getattr(action, 'tool', '')
                if tool_name == 'create_chart' and isinstance(result, str):
                    if 'File:' in result:
                        # Parse our own controlled format
                        path = result.split('File:')[1].split('\n')[0].strip()
                        if path and os.path.exists(path):
                            charts.append(path)
        except (TypeError, ValueError, AttributeError) as e:
            # intermediate_steps format might have changed — fail silently
            print(f"Primary chart extraction from intermediate_steps failed: {e}")
            return []
        
        return list(set(charts))
    
    def _extract_from_output_llm(self, output: str) -> List[str]:
        """Fallback: Extract chart paths from output text using structured LLM.
        
        Uses with_structured_output(ExtractedChartPaths) to reliably find
        file paths in the agent's free-form text output. Validates each
        path with os.path.exists to prevent hallucinated paths.
        
        Args:
            output: Agent's final text output
            
        Returns:
            List of validated chart file paths
        """
        if not output or not output.strip():
            return []
        
        try:
            prompt = (
                "Extract all chart or visualization file paths from this analysis output.\n"
                "Return ONLY paths that are explicitly mentioned as created or saved files.\n"
                "Do NOT invent or guess any paths.\n\n"
                f"{output}"
            )
            
            result = self.path_extractor_llm.invoke(prompt)
            
            # Safety net: only keep paths that actually exist on disk
            valid_paths = [p for p in result.paths if os.path.exists(p)]
            
            if not valid_paths and result.paths:
                print(
                    f"LLM fallback found {len(result.paths)} paths but none exist on disk: "
                    f"{result.paths}"
                )
            
            return valid_paths
            
        except Exception as e:
            print(f"Fallback LLM chart path extraction failed: {e}")
            return []
    
    def _extract_chart_explanations(
        self, analysis_output: str, chart_paths: List[str]
    ) -> Dict[str, Dict[str, str]]:
        """Extract per-chart explanations using Pydantic structured LLM output.
        
        Uses the ChartExplanations schema to get structured {path, title, explanation}
        for each chart mentioned in the analysis output.
        
        Args:
            analysis_output: Full text output from the data analysis agent
            chart_paths: List of chart file paths found in the output
            
        Returns:
            Dict mapping chart_path -> {"title": ..., "explanation": ...}
        """
        # Guard: if explanation_llm is not available, return generic explanations
        if not self.explanation_llm:
            return {
                path: {
                    "title": self._title_from_path(path),
                    "explanation": "Visualization generated from the data analysis."
                }
                for path in chart_paths
            }

        # Build the prompt with chart paths for context
        chart_list = "\n".join(f"  - {path}" for path in chart_paths)

        prompt = f"""Analyze the following data analysis output and extract a title and explanation for each chart that was created.

For each chart, provide:
- chart_path: The exact file path of the chart
- title: A short, descriptive title (e.g., "Revenue by Quarter", "Correlation Heatmap")
- explanation: A clear interpretation of what the chart shows, key insights, trends, or patterns visible in the data (2-4 sentences)

Charts found:
{chart_list}

Analysis output:
{analysis_output}

Extract explanations for each chart listed above."""
        
        try:
            result = self.explanation_llm.invoke(prompt)
            
            # Convert to dict: {chart_path: {title, explanation}}
            explanations = {}
            for chart in result.charts:
                explanations[chart.chart_path] = {
                    "title": chart.title,
                    "explanation": chart.explanation
                }
            
            # Ensure all chart paths have an entry (fallback for any missed)
            for path in chart_paths:
                if path not in explanations:
                    explanations[path] = {
                        "title": self._title_from_path(path),
                        "explanation": "Visualization generated from the data analysis."
                    }
            
            return explanations
            
        except Exception as e:
            print(f"Chart explanation extraction failed: {e}. Using fallback.")
            # Fallback: generate generic explanations from file names
            return {
                path: {
                    "title": self._title_from_path(path),
                    "explanation": "Visualization generated from the data analysis."
                }
                for path in chart_paths
            }
    
    @staticmethod
    def _title_from_path(path: str) -> str:
        """Generate a readable title from a chart file path.

        Args:
            path: Chart file path like 'outputs/charts/bar_abc123.html'

        Returns:
            Human-readable title like 'Bar Chart'
        """
        filename = os.path.basename(path)
        # Extract chart type from filename (e.g., 'bar' from 'bar_abc123.html')
        match = re.match(r'([a-z_]+)_[a-f0-9]+\.html', filename)
        if match:
            chart_type = match.group(1).replace('_', ' ').title()
            return f"{chart_type} Chart"
        return "Chart"


__all__ = ['DataAnalysisAgent']
