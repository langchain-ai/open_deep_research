"""Pydantic schemas for tool inputs."""
from pydantic import BaseModel, Field
from typing import Dict, Literal, List


# =====================================
# MATH TOOL SCHEMAS
# =====================================
class AddInput(BaseModel):
    """Schema for addition operation."""
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


class SubtractInput(BaseModel):
    """Schema for subtraction operation."""
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


class MultiplyInput(BaseModel):
    """Schema for multiplication operation."""
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


class DivideInput(BaseModel):
    """Schema for division operation."""
    a: float = Field(description="Numerator")
    b: float = Field(description="Denominator (cannot be zero)")


class CalculateInput(BaseModel):
    """Schema for expression calculation."""
    expression: str = Field(
        description="Mathematical expression to evaluate (supports +, -, *, /, sqrt, sin, cos, tan, pi, e, log, exp)"
    )


# =====================================
# DATA TOOL SCHEMAS
# =====================================
class DataProfilingInput(BaseModel):
    """Schema for data profiling."""
    data: str = Field(description="Raw data to profile (CSV, JSON, or table format)")
    analysis_type: str = Field(
        default="comprehensive",
        description="Type of analysis: 'comprehensive', 'statistical', 'patterns'"
    )


class DataExtractionInput(BaseModel):
    """Schema for data extraction."""
    text: str = Field(description="Text containing data to extract (tables, lists, structured info)")
    format: str = Field(
        default="json",
        description="Output format: 'json', 'csv', 'table'"
    )


# =====================================
# VISUALIZATION TOOL SCHEMAS
# =====================================
class PlotlyVisualizationInput(BaseModel):
    """Schema for Plotly chart creation."""
    data: str = Field(description="Data to visualize (CSV, JSON, or table format)")
    chart_type: Literal[
        "bar", "line", "scatter", "pie", "histogram", "box", 
        "heatmap", "density", "bubble", "violin", "boxplot"
    ] = Field(
        description="Type of chart to create. Options: bar (categorical comparison), line (trends), scatter (relationships), pie (proportions), histogram (distribution), box/boxplot (distribution with outliers), heatmap (correlations), density (continuous distribution), bubble (3D relationships), violin (distribution shape)"
    )
    title: str = Field(default="Chart", description="Chart title")
    x_column: str = Field(default="", description="Column for X-axis (optional, will auto-detect)")
    y_column: str = Field(default="", description="Column for Y-axis (optional, will auto-detect)")
    z_column: str = Field(default="", description="Column for Z-axis or size (for bubble/heatmap charts, optional)")
    color_column: str = Field(default="", description="Column for color grouping (optional)")


class OutlierDetectionInput(BaseModel):
    """Schema for outlier detection."""
    data: str = Field(description="Data to analyze for outliers (CSV, JSON, or table format)")
    column: str = Field(description="Column name to check for outliers")
    method: Literal["iqr", "zscore", "isolation_forest"] = Field(
        default="iqr",
        description="Outlier detection method: 'iqr' (Interquartile Range), 'zscore' (Z-score), 'isolation_forest' (ML-based)"
    )
    threshold: float = Field(
        default=1.5,
        description="Threshold for outlier detection (1.5 for IQR, 3.0 for Z-score)"
    )


# =====================================
# CHART PATH EXTRACTION SCHEMA
# =====================================
class ExtractedChartPaths(BaseModel):
    """Chart/visualization file paths extracted from analysis output."""
    paths: List[str] = Field(
        default_factory=list,
        description="All chart or visualization file paths found in the text. "
                    "Include only paths that reference actual created files. "
                    "Examples: outputs/charts/bar_abc123.html, /mnt/data/chart.png"
    )


# =====================================
# CHART EXPLANATION SCHEMAS
# =====================================
class ChartExplanation(BaseModel):
    """Structured explanation for a single chart."""
    chart_path: str = Field(description="Path to the chart file")
    title: str = Field(description="Short descriptive title for the chart")
    explanation: str = Field(description="What this chart shows and key insights")


class ChartExplanations(BaseModel):
    """Collection of chart explanations extracted from analysis output."""
    charts: List[ChartExplanation] = Field(
        default_factory=list,
        description="List of chart explanations"
    )


__all__ = [
    'AddInput', 'SubtractInput', 'MultiplyInput', 'DivideInput', 'CalculateInput',
    'DataProfilingInput', 'DataExtractionInput',
    'PlotlyVisualizationInput', 'OutlierDetectionInput',
    'ExtractedChartPaths',
    'ChartExplanation', 'ChartExplanations'
]
