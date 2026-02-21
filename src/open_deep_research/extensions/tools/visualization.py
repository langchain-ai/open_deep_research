"""Plotly visualization tools for creating interactive charts."""
import os
import uuid
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.tools import StructuredTool
from extensions.models.tool_schemas import PlotlyVisualizationInput, OutlierDetectionInput
from extensions.tools.data_profiling import parse_data


def create_chart(
    data: str,
    chart_type: str,
    title: str = "Chart",
    x_column: str = "",
    y_column: str = "",
    z_column: str = "",
    color_column: str = ""
) -> str:
    """Create interactive Plotly chart from data.
    
    Args:
        data: Data in CSV, JSON, or table format
        chart_type: Type of chart (bar, line, scatter, pie, histogram, box, heatmap, density, bubble, violin, boxplot)
        title: Chart title
        x_column: Column name for X-axis (optional, will auto-detect)
        y_column: Column name for Y-axis (optional, will auto-detect)
        z_column: Column for Z-axis/size (for bubble/heatmap)
        color_column: Column for color grouping
        
    Returns:
        Path to saved HTML file and preview of chart
    """
    try:
        # Parse data
        df = parse_data(data)
        
        if df is None or df.empty:
            return "[ERROR] Could not parse data. Please provide data in CSV, JSON, or table format."
        
        # Auto-detect columns if not provided
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not x_column and len(df.columns) > 0:
            # Prefer categorical for x-axis, else first column
            x_column = categorical_cols[0] if categorical_cols else df.columns[0]
        
        if not y_column:
            # Find first numeric column for y-axis
            if len(numeric_cols) > 0:
                y_column = numeric_cols[0]
            elif len(df.columns) > 1:
                y_column = df.columns[1]
        
        # Create chart based on type
        fig = None
        chart_info = ""
        
        if chart_type == "bar":
            fig = px.bar(df, x=x_column, y=y_column, title=title, 
                        color=color_column if color_column else None)
            chart_info = f"X: {x_column}, Y: {y_column}"
        
        elif chart_type == "line":
            fig = px.line(df, x=x_column, y=y_column, title=title,
                         color=color_column if color_column else None)
            chart_info = f"X: {x_column}, Y: {y_column}"
        
        elif chart_type == "scatter":
            size_col = z_column if z_column and z_column in df.columns else None
            fig = px.scatter(df, x=x_column, y=y_column, title=title,
                           color=color_column if color_column else None,
                           size=size_col)
            chart_info = f"X: {x_column}, Y: {y_column}"
            if size_col:
                chart_info += f", Size: {size_col}"
        
        elif chart_type == "pie":
            fig = px.pie(df, names=x_column, values=y_column, title=title)
            chart_info = f"Labels: {x_column}, Values: {y_column}"
        
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x_column, title=title,
                             color=color_column if color_column else None)
            chart_info = f"X: {x_column}"
        
        elif chart_type in ["box", "boxplot"]:
            fig = px.box(df, y=y_column, x=x_column if x_column != y_column else None, 
                        title=title, color=color_column if color_column else None)
            chart_info = f"Y: {y_column}"
            if x_column and x_column != y_column:
                chart_info = f"X: {x_column}, " + chart_info
        
        elif chart_type == "violin":
            fig = px.violin(df, y=y_column, x=x_column if x_column != y_column else None,
                          title=title, color=color_column if color_column else None,
                          box=True, points="all")
            chart_info = f"Y: {y_column}"
            if x_column and x_column != y_column:
                chart_info = f"X: {x_column}, " + chart_info
        
        elif chart_type == "heatmap":
            # For heatmap, we need numeric data
            if len(numeric_cols) < 2:
                # Calculate correlation matrix
                corr_df = df[numeric_cols].corr() if numeric_cols else df.select_dtypes(include='number').corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.columns,
                    colorscale='RdBu',
                    zmid=0
                ))
                fig.update_layout(title=title or "Correlation Heatmap")
                chart_info = f"Correlation matrix of {len(corr_df.columns)} variables"
            else:
                # Pivot table heatmap
                if x_column and y_column and z_column:
                    pivot_df = df.pivot_table(values=z_column, index=y_column, columns=x_column, aggfunc='mean')
                    fig = go.Figure(data=go.Heatmap(
                        z=pivot_df.values,
                        x=pivot_df.columns,
                        y=pivot_df.index,
                        colorscale='Viridis'
                    ))
                    fig.update_layout(title=title)
                    chart_info = f"X: {x_column}, Y: {y_column}, Values: {z_column}"
                else:
                    # Correlation matrix as fallback
                    corr_df = df[numeric_cols].corr()
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_df.values,
                        x=corr_df.columns,
                        y=corr_df.columns,
                        colorscale='RdBu',
                        zmid=0
                    ))
                    fig.update_layout(title=title or "Correlation Heatmap")
                    chart_info = f"Correlation matrix of {len(corr_df.columns)} variables"
        
        elif chart_type == "density":
            # 2D density contour plot
            if len(numeric_cols) >= 2:
                x_col = x_column if x_column in numeric_cols else numeric_cols[0]
                y_col = y_column if y_column in numeric_cols else numeric_cols[1]
                fig = px.density_contour(df, x=x_col, y=y_col, title=title)
                chart_info = f"X: {x_col}, Y: {y_col}"
            else:
                # Fallback to histogram for 1D
                fig = px.histogram(df, x=x_column or numeric_cols[0], 
                                 marginal="box", title=title)
                chart_info = f"Distribution of {x_column or numeric_cols[0]}"
        
        elif chart_type == "bubble":
            # Bubble chart (scatter with size)
            if not z_column and len(numeric_cols) >= 3:
                z_column = numeric_cols[2]
            
            size_col = z_column if z_column and z_column in df.columns else None
            
            fig = px.scatter(df, x=x_column, y=y_column, size=size_col,
                           color=color_column if color_column else None,
                           title=title, size_max=60)
            chart_info = f"X: {x_column}, Y: {y_column}"
            if size_col:
                chart_info += f", Size: {size_col}"
        
        else:
            return f"[ERROR] Unsupported chart type: {chart_type}. Use: bar, line, scatter, pie, histogram, box, boxplot, violin, heatmap, density, bubble"
        
        if fig is None:
            return "[ERROR] Could not create chart with provided data and parameters."
        
        # Improve layout
        fig.update_layout(
            template="plotly_white",
            hovermode='closest',
            showlegend=True
        )
        
        # Save chart (use CDN to avoid 4.7MB embedded Plotly.js)
        os.makedirs("outputs/charts", exist_ok=True)
        chart_id = str(uuid.uuid4())[:8]
        filename = f"outputs/charts/{chart_type}_{chart_id}.html"

        # Save with CDN reference instead of embedding full Plotly.js
        # This reduces file size from 4.7MB to ~10KB per chart
        fig.write_html(filename, include_plotlyjs='cdn')
        
        # Return success message
        return f"""
Chart created successfully!

Chart Type: {chart_type.capitalize()}
File: {filename}
Data: {len(df)} rows, {len(df.columns)} columns
Parameters: {chart_info}

Chart saved and ready to view!
"""
        
    except Exception as e:
        return f"[ERROR] Error creating chart: {str(e)}"


# Create LangChain tool
create_chart_tool = StructuredTool.from_function(
    func=create_chart,
    name="create_chart",
    description="Create interactive Plotly charts from data. Supports: bar, line, scatter, pie, histogram, box/boxplot (outlier visualization), violin (distribution shape), heatmap (correlations/matrices), density (2D distributions), bubble (3D relationships). Auto-detects columns if not specified. Perfect for univariate, bivariate, and multivariate analysis.",
    args_schema=PlotlyVisualizationInput
)


def detect_outliers(
    data: str,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5
) -> str:
    """Detect outliers in a dataset column using various methods.
    
    Args:
        data: Data in CSV, JSON, or table format
        column: Column name to check for outliers
        method: Detection method ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold value (1.5 for IQR, 3.0 for Z-score)
        
    Returns:
        Analysis of outliers with visualization
    """
    try:
        # Parse data
        df = parse_data(data)
        
        if df is None or df.empty:
            return "[ERROR] Could not parse data. Please provide data in CSV, JSON, or table format."
        
        if column not in df.columns:
            return f"[ERROR] Column '{column}' not found. Available columns: {', '.join(df.columns)}"
        
        # Get column data
        col_data = df[column].dropna()
        
        if not np.issubdtype(col_data.dtype, np.number):
            return f"[ERROR] Column '{column}' must be numeric for outlier detection."
        
        outlier_mask = np.zeros(len(col_data), dtype=bool)
        method_info = ""
        
        # Apply outlier detection method
        if method == "iqr":
            # Interquartile Range method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            method_info = f"IQR Method (threshold={threshold})\nLower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}"
        
        elif method == "zscore":
            # Z-score method
            mean = col_data.mean()
            std = col_data.std()
            z_scores = np.abs((col_data - mean) / std)
            outlier_mask = z_scores > threshold
            method_info = f"Z-Score Method (threshold={threshold})\nMean: {mean:.2f}, Std: {std:.2f}"
        
        elif method == "isolation_forest":
            # Isolation Forest (ML-based)
            try:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                predictions = iso_forest.fit_predict(col_data.values.reshape(-1, 1))
                outlier_mask = predictions == -1
                method_info = f"Isolation Forest (ML-based)\nContamination: 10%"
            except ImportError:
                return "[ERROR] Isolation Forest requires scikit-learn. Install with: pip install scikit-learn"
        
        else:
            return f"[ERROR] Unknown method: {method}. Use 'iqr', 'zscore', or 'isolation_forest'"
        
        # Count outliers
        n_outliers = outlier_mask.sum()
        outlier_pct = (n_outliers / len(col_data)) * 100
        
        # Create visualization
        fig = go.Figure()
        
        # Add normal points
        fig.add_trace(go.Scatter(
            y=col_data[~outlier_mask],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))
        
        # Add outlier points
        if n_outliers > 0:
            fig.add_trace(go.Scatter(
                y=col_data[outlier_mask],
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=12, symbol='x', line=dict(width=2))
            ))
        
        fig.update_layout(
            title=f"Outlier Detection: {column}",
            yaxis_title=column,
            template="plotly_white",
            showlegend=True,
            hovermode='closest'
        )
        
        # Save chart (use CDN to avoid 4.7MB embedded Plotly.js)
        os.makedirs("outputs/charts", exist_ok=True)
        chart_id = str(uuid.uuid4())[:8]
        filename = f"outputs/charts/outliers_{chart_id}.html"
        fig.write_html(filename, include_plotlyjs='cdn')
        
        # Prepare outlier list
        outlier_values = col_data[outlier_mask].tolist()
        outlier_summary = ", ".join([f"{v:.2f}" for v in outlier_values[:10]])
        if len(outlier_values) > 10:
            outlier_summary += f"... ({len(outlier_values)-10} more)"
        
        # Return analysis
        return f"""
Outlier Detection Complete!

Column: {column}
Method: {method_info}

Results:
   * Total values: {len(col_data)}
   * Outliers found: {n_outliers} ({outlier_pct:.1f}%)
   * Normal values: {len(col_data) - n_outliers} ({100-outlier_pct:.1f}%)

Outlier Values:
   {outlier_summary}

Visualization: {filename}

Interpretation:
   - IQR method: Values beyond Q1-1.5×IQR or Q3+1.5×IQR
   - Z-score: Values with |z| > threshold standard deviations
   - Isolation Forest: ML-based anomaly detection
"""
        
    except Exception as e:
        return f"[ERROR] Error detecting outliers: {str(e)}"


# Create outlier detection tool
detect_outliers_tool = StructuredTool.from_function(
    func=detect_outliers,
    name="detect_outliers",
    description="Detect and visualize outliers in numeric data using IQR (Interquartile Range), Z-score, or Isolation Forest methods. Returns analysis with outlier count, percentage, and visualization.",
    args_schema=OutlierDetectionInput
)


__all__ = ['create_chart_tool', 'create_chart', 'detect_outliers_tool', 'detect_outliers']
