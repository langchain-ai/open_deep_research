"""Data profiling and analysis tools."""
import json
import pandas as pd
from io import StringIO
from langchain_core.tools import StructuredTool
from extensions.models.tool_schemas import DataProfilingInput


def parse_data(data: str) -> pd.DataFrame:
    """Parse data from various formats into DataFrame."""
    try:
        data = data.strip()

        # Try JSON
        if data.startswith('{') or data.startswith('['):
            json_data = json.loads(data)
            if isinstance(json_data, list):
                return pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                return pd.DataFrame([json_data])

        # Try CSV (with commas)
        if ',' in data:
            return pd.read_csv(StringIO(data))

        # Try tab-separated
        if '\t' in data:
            return pd.read_csv(StringIO(data), sep='\t')

        # Try pipe-separated table
        if '|' in data:
            df = pd.read_csv(StringIO(data), sep='|')
            # Remove empty columns (common in markdown tables)
            df = df.dropna(axis=1, how='all')
            df.columns = df.columns.str.strip()
            return df

        # Try single-column CSV (newline-separated values, no commas)
        # This handles data like "Value\n10\n20\n30"
        lines = data.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        if len(lines) >= 2:
            try:
                df = pd.read_csv(StringIO(data))
                if not df.empty:
                    return df
            except Exception:
                pass

        return None
    except Exception as e:
        print(f"Parse error: {e}")
        return None


def profile_data(data: str, analysis_type: str = "comprehensive") -> str:
    """Profile and analyze data from research or any source.

    Args:
        data: Raw data in CSV, JSON, or table format
        analysis_type: Type of analysis ('comprehensive', 'statistical', 'patterns')

    Returns:
        Formatted profiling report with statistics and insights
    """
    try:
        df = parse_data(data)

        if df is None or df.empty:
            return "[ERROR] Could not parse data. Please provide data in CSV, JSON, or table format."

        missing = df.isnull().sum()

        # -- Section 1: Shape --
        report = f"**Shape:** {df.shape[0]} rows x {df.shape[1]} columns\n\n"

        # -- Section 2: Columns table (type + missing merged) --
        report += "### Columns\n\n"
        report += "| Column | Type | Missing |\n"
        report += "|--------|------|--------|\n"
        for col in df.columns:
            dtype = df[col].dtype
            dtype_label = "numeric" if pd.api.types.is_numeric_dtype(dtype) else "text"
            miss_count = int(missing[col])
            if miss_count > 0:
                miss_pct = (miss_count / len(df)) * 100
                miss_str = f"{miss_count} ({miss_pct:.1f}%)"
            else:
                miss_str = "0"
            report += f"| {col} | {dtype_label} | {miss_str} |\n"
        report += "\n"

        # -- Section 3: Summary Statistics (pipe table) --
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            report += "### Summary Statistics\n\n"
            stats = df[numeric_cols].describe().round(2)
            # Build pipe table header
            report += "| Statistic | " + " | ".join(str(c) for c in stats.columns) + " |\n"
            report += "|---" * (len(stats.columns) + 1) + "|\n"
            # Build pipe table rows
            for idx, row in stats.iterrows():
                vals = " | ".join(str(v) for v in row.values)
                report += f"| {idx} | {vals} |\n"
            report += "\n"

        # -- Section 4: Column Details --
        report += "### Column Details\n\n"
        for col in df.columns:
            unique_count = df[col].nunique()
            unique_pct = (unique_count / len(df)) * 100

            if pd.api.types.is_numeric_dtype(df[col]):
                col_min = df[col].min()
                col_max = df[col].max()
                col_mean = df[col].mean()
                col_median = df[col].median()
                report += f"- **{col}**: {col_min:.2f} -> {col_max:.2f}, mean {col_mean:.2f}, median {col_median:.2f}\n"
            else:
                top_str = ""
                if unique_count <= 10:
                    top_values = df[col].value_counts().head(3)
                    top_str = " -- " + ", ".join(str(v) for v in top_values.index)
                report += f"- **{col}**: {unique_count} unique values ({unique_pct:.1f}% unique){top_str}\n"

        return report

    except Exception as e:
        return f"[ERROR] Error profiling data: {str(e)}"


# Create LangChain tool
profile_data_tool = StructuredTool.from_function(
    func=profile_data,
    name="profile_data",
    description="Profile and analyze data from research or any source. Provides statistics, patterns, missing values, and insights. Accepts CSV, JSON, or table format.",
    args_schema=DataProfilingInput
)


__all__ = ['profile_data_tool', 'profile_data', 'parse_data']
