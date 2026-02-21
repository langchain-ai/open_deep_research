#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
bert_annotation_chart.py

MAIN OBJECTIVE:
---------------
Generate publication-quality annotation metrics charts for BERT Annotation Studio.
Provides comprehensive visualizations of annotation probabilities, confidence scores,
per-model performance, and label distributions for BERT-based annotation workflows.

Dependencies:
-------------
- os
- json
- logging
- pathlib
- typing
- datetime
- numpy
- pandas
- matplotlib

MAIN FEATURES:
--------------
1) Generate per-model annotation distribution charts
2) Visualize probability/confidence score distributions
3) Display label distribution with confidence breakdown
4) Show per-language annotation metrics
5) Track annotation coverage and completeness
6) Support multiple models in a single session (refreshed per model)
7) Export metrics as JSON for further analysis

Author:
-------
Antoine Lemor
"""

# Force non-interactive backend for multiprocessing compatibility on macOS
# Must be set BEFORE any pyplot import to avoid "Cannot create a GUI FigureManager
# outside the main thread using the MacOS backend" error in worker processes
import matplotlib
matplotlib.use('Agg')

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

# Configure logging
logger = logging.getLogger(__name__)

# Suppress matplotlib debug logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Publication-quality style settings
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette for consistent styling
COLORS = {
    'high_conf': '#22c55e',       # Green (high confidence)
    'medium_conf': '#f59e0b',     # Amber (medium confidence)
    'low_conf': '#ef4444',        # Red (low confidence)
    'primary': '#6366f1',         # Indigo
    'secondary': '#8b5cf6',       # Violet
    'info': '#3b82f6',            # Blue
    'neutral': '#6b7280',         # Gray
    'dark': '#1f2937',
    'light': '#f3f4f6',
}

# Label distribution colors
LABEL_COLORS = [
    '#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899',
    '#f43f5e', '#ef4444', '#f97316', '#f59e0b', '#eab308',
    '#84cc16', '#22c55e', '#10b981', '#14b8a6', '#06b6d4',
    '#0ea5e9', '#3b82f6', '#2563eb', '#4f46e5', '#7c3aed',
]

# Language colors
LANGUAGE_COLORS = {
    'en': '#3b82f6',
    'fr': '#ef4444',
    'de': '#22c55e',
    'es': '#f59e0b',
    'it': '#8b5cf6',
    'pt': '#06b6d4',
    'nl': '#ec4899',
    'unknown': '#6b7280',
    'multi': '#6366f1',
}


class BERTAnnotationChart:
    """
    Generates publication-quality annotation metrics charts for BERT Annotation Studio.

    This class creates comprehensive visualizations of BERT model annotation results
    including probability distributions, confidence scores, and per-model performance.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        session_id: str,
        model_name: str,
        label_names: Optional[List[str]] = None,
        category_name: Optional[str] = None,
    ):
        """
        Initialize the BERT annotation chart generator.

        Args:
            output_dir: Directory to save charts
            session_id: Unique session identifier
            model_name: Name of the BERT model used
            label_names: List of label names for the model
            category_name: Name of the annotation category
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id
        self.model_name = model_name
        self.label_names = label_names or []
        self.category_name = category_name or "annotations"
        self.timestamp = datetime.now()

        # Metrics storage
        self.metrics: Dict[str, Any] = {
            'total_rows': 0,
            'annotated_rows': 0,
            'label_distribution': {},
            'probability_stats': {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
            },
            'confidence_distribution': {
                'high': 0,      # > 0.8
                'medium': 0,    # 0.5 - 0.8
                'low': 0,       # < 0.5
            },
            'language_distribution': {},
            'per_label_confidence': {},
            'probabilities': [],  # Raw probabilities for histogram
        }

        logger.info(f"BERTAnnotationChart initialized for session {session_id}, model {model_name}")

    def update_from_dataframe(
        self,
        df: pd.DataFrame,
        label_column: str,
        probability_column: Optional[str] = None,
        language_column: Optional[str] = None,
    ) -> None:
        """
        Update metrics from an annotated DataFrame.

        Args:
            df: DataFrame with annotation results
            label_column: Name of the column containing predicted labels
            probability_column: Optional name of the probability column
            language_column: Optional name of the language column
        """
        try:
            self.metrics['total_rows'] = len(df)

            # Count annotated rows (non-null labels)
            annotated_mask = df[label_column].notna()
            if annotated_mask.sum() > 0:
                # Check for empty strings
                annotated_mask = annotated_mask & (df[label_column].astype(str).str.strip() != '')

            self.metrics['annotated_rows'] = int(annotated_mask.sum())
            annotated_df = df[annotated_mask]

            # Label distribution
            label_counts = annotated_df[label_column].value_counts().to_dict()
            self.metrics['label_distribution'] = {str(k): int(v) for k, v in label_counts.items()}

            # Probability statistics
            if probability_column and probability_column in df.columns:
                probabilities = annotated_df[probability_column].dropna()
                # Handle string representations of probabilities
                if probabilities.dtype == 'object':
                    try:
                        probabilities = probabilities.astype(float)
                    except (ValueError, TypeError):
                        # Try to extract from JSON or list format
                        def extract_max_prob(val):
                            try:
                                if isinstance(val, str):
                                    val = json.loads(val)
                                if isinstance(val, (list, tuple)):
                                    return max(val)
                                elif isinstance(val, dict):
                                    return max(val.values())
                                return float(val)
                            except:
                                return None
                        probabilities = probabilities.apply(extract_max_prob).dropna()

                if len(probabilities) > 0:
                    probs_array = probabilities.values.astype(float)
                    self.metrics['probabilities'] = probs_array.tolist()
                    self.metrics['probability_stats'] = {
                        'mean': float(np.mean(probs_array)),
                        'std': float(np.std(probs_array)),
                        'min': float(np.min(probs_array)),
                        'max': float(np.max(probs_array)),
                        'median': float(np.median(probs_array)),
                    }

                    # Confidence distribution
                    self.metrics['confidence_distribution'] = {
                        'high': int(np.sum(probs_array >= 0.8)),
                        'medium': int(np.sum((probs_array >= 0.5) & (probs_array < 0.8))),
                        'low': int(np.sum(probs_array < 0.5)),
                    }

                    # Per-label confidence
                    if label_column in annotated_df.columns:
                        per_label_conf = {}
                        for label in annotated_df[label_column].unique():
                            label_mask = annotated_df[label_column] == label
                            label_probs = annotated_df.loc[label_mask, probability_column].dropna()
                            if len(label_probs) > 0:
                                try:
                                    label_probs_float = label_probs.astype(float)
                                    per_label_conf[str(label)] = {
                                        'mean': float(np.mean(label_probs_float)),
                                        'count': int(len(label_probs_float)),
                                    }
                                except:
                                    pass
                        self.metrics['per_label_confidence'] = per_label_conf

            # Language distribution
            if language_column and language_column in df.columns:
                lang_counts = annotated_df[language_column].value_counts().to_dict()
                self.metrics['language_distribution'] = {
                    str(k): int(v) for k, v in lang_counts.items() if pd.notna(k)
                }

            logger.info(f"Updated BERT metrics: {self.metrics['annotated_rows']} annotated rows")

        except Exception as e:
            logger.error(f"Error updating BERT metrics from DataFrame: {e}")

    def update_from_model_results(
        self,
        model_info: Dict[str, Any],
        annotation_stats: Dict[str, Any],
    ) -> None:
        """
        Update metrics from model results dictionary.

        Args:
            model_info: Dictionary with model information
            annotation_stats: Dictionary with annotation statistics
        """
        try:
            if 'rows_annotated' in annotation_stats:
                self.metrics['annotated_rows'] = annotation_stats['rows_annotated']

            if 'rows_scheduled' in annotation_stats:
                self.metrics['total_rows'] = annotation_stats['rows_scheduled']

            if 'label_distribution' in annotation_stats:
                self.metrics['label_distribution'] = annotation_stats['label_distribution']

            if 'language' in model_info:
                self.metrics['model_language'] = model_info['language']

            logger.info(f"Updated BERT metrics from model results")

        except Exception as e:
            logger.error(f"Error updating BERT metrics from results: {e}")

    def generate_chart(self) -> Optional[Path]:
        """
        Generate the comprehensive BERT annotation metrics chart.

        Returns:
            Path to the generated chart, or None if generation failed
        """
        try:
            # Determine layout based on available data
            has_probabilities = len(self.metrics.get('probabilities', [])) > 0
            has_language_data = bool(self.metrics.get('language_distribution'))
            has_per_label_conf = bool(self.metrics.get('per_label_confidence'))

            # Create figure with appropriate size
            fig = plt.figure(figsize=(16, 12))

            # Create grid layout
            if has_probabilities and has_language_data:
                gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1.2, 1], hspace=0.35, wspace=0.3)
            elif has_probabilities:
                gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1.2, 1], hspace=0.35, wspace=0.3)
            else:
                gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

            # Title
            title_parts = [
                f"BERT Annotation Metrics - {self.category_name}",
                f"Session: {self.session_id}",
                f"Model: {self.model_name}",
                f"Date: {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
            ]
            fig.suptitle('\n'.join(title_parts), fontsize=14, fontweight='bold', y=0.98)

            # 1. Summary statistics (top left)
            ax_summary = fig.add_subplot(gs[0, 0])
            self._plot_summary_stats(ax_summary)

            # 2. Confidence distribution pie chart (top right)
            if has_probabilities:
                ax_conf = fig.add_subplot(gs[0, 1])
                self._plot_confidence_distribution(ax_conf)
            else:
                ax_conf = fig.add_subplot(gs[0, 1])
                ax_conf.axis('off')
                ax_conf.text(0.5, 0.5, 'No probability data available',
                           transform=ax_conf.transAxes, ha='center', va='center')

            # 3. Label distribution (middle row, full width or partial)
            label_dist = self.metrics.get('label_distribution', {})
            if label_dist:
                if has_per_label_conf:
                    ax_labels = fig.add_subplot(gs[1, :2])
                else:
                    ax_labels = fig.add_subplot(gs[1, :])
                self._plot_label_distribution(ax_labels)

            # 4. Probability histogram (if available)
            if has_probabilities:
                ax_hist = fig.add_subplot(gs[2, 0])
                self._plot_probability_histogram(ax_hist)

            # 5. Per-label confidence (if available)
            if has_per_label_conf:
                ax_per_label = fig.add_subplot(gs[1, 2] if has_language_data else gs[2, 1])
                self._plot_per_label_confidence(ax_per_label)

            # 6. Language distribution (if available)
            if has_language_data:
                ax_lang = fig.add_subplot(gs[2, 1] if has_probabilities else gs[1, 1])
                self._plot_language_distribution(ax_lang)

            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # Generate filename
            safe_model_name = self.model_name.replace('/', '_').replace(':', '_').replace('\\', '_')[:40]
            safe_category = self.category_name.replace('/', '_').replace(':', '_')[:20]
            chart_filename = (
                f"bert_annotation_metrics_{self.session_id}_"
                f"{safe_category}_{safe_model_name}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            )
            chart_path = self.output_dir / chart_filename

            # Save chart
            fig.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            logger.info(f"BERT annotation metrics chart saved to {chart_path}")
            return chart_path

        except Exception as e:
            logger.error(f"Error generating BERT annotation metrics chart: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _plot_summary_stats(self, ax: plt.Axes) -> None:
        """Plot summary statistics as a text box."""
        ax.axis('off')

        total = self.metrics.get('total_rows', 0)
        annotated = self.metrics.get('annotated_rows', 0)
        coverage = (annotated / total * 100) if total > 0 else 0

        prob_stats = self.metrics.get('probability_stats', {})
        mean_conf = prob_stats.get('mean', 0)
        median_conf = prob_stats.get('median', 0)

        n_labels = len(self.metrics.get('label_distribution', {}))

        # Build stats text
        stats_lines = [
            f"Total Rows: {total:,}",
            f"Annotated: {annotated:,}",
            f"Coverage: {coverage:.1f}%",
            f"Unique Labels: {n_labels}",
        ]

        if mean_conf > 0:
            stats_lines.extend([
                f"Mean Confidence: {mean_conf:.3f}",
                f"Median Confidence: {median_conf:.3f}",
            ])

        # Draw text box
        props = dict(boxstyle='round,pad=0.5', facecolor=COLORS['light'], alpha=0.8)
        ax.text(0.5, 0.5, '\n'.join(stats_lines),
                transform=ax.transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='center',
                bbox=props, family='monospace')

        ax.set_title('Summary Statistics', fontsize=11, fontweight='bold', pad=10)

    def _plot_confidence_distribution(self, ax: plt.Axes) -> None:
        """Plot confidence distribution as a pie chart."""
        conf_dist = self.metrics.get('confidence_distribution', {})

        if not conf_dist or sum(conf_dist.values()) == 0:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No confidence data available',
                   transform=ax.transAxes, ha='center', va='center')
            return

        labels = ['High (>0.8)', 'Medium (0.5-0.8)', 'Low (<0.5)']
        sizes = [conf_dist.get('high', 0), conf_dist.get('medium', 0), conf_dist.get('low', 0)]
        colors = [COLORS['high_conf'], COLORS['medium_conf'], COLORS['low_conf']]

        # Filter out zero values
        non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
        if not non_zero:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No confidence data available',
                   transform=ax.transAxes, ha='center', va='center')
            return

        labels, sizes, colors = zip(*non_zero)

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            pctdistance=0.75, labeldistance=1.1
        )

        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')

        ax.set_title('Confidence Distribution', fontsize=11, fontweight='bold', pad=10)

    def _plot_label_distribution(self, ax: plt.Axes) -> None:
        """Plot label distribution as a horizontal bar chart."""
        label_dist = self.metrics.get('label_distribution', {})

        if not label_dist:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No label data available',
                   transform=ax.transAxes, ha='center', va='center')
            return

        # Sort by count and limit to top values
        sorted_items = sorted(label_dist.items(), key=lambda x: x[1], reverse=True)
        max_labels = 15

        if len(sorted_items) > max_labels:
            top_items = sorted_items[:max_labels-1]
            other_count = sum(count for _, count in sorted_items[max_labels-1:])
            top_items.append(('Other', other_count))
            sorted_items = top_items

        labels = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]

        # Truncate long label names
        display_labels = []
        for label in labels:
            if len(str(label)) > 25:
                display_labels.append(str(label)[:22] + '...')
            else:
                display_labels.append(str(label))

        # Create horizontal bar chart
        y_pos = np.arange(len(labels))
        colors = [LABEL_COLORS[i % len(LABEL_COLORS)] for i in range(len(labels))]

        bars = ax.barh(y_pos, counts, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_labels, fontsize=9)
        ax.invert_yaxis()

        # Add count labels
        max_count = max(counts) if counts else 1
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width + max_count * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{count:,}', va='center', fontsize=9)

        ax.set_xlabel('Count', fontsize=10)
        ax.set_title(f'Label Distribution: {self.category_name}', fontsize=11, fontweight='bold', pad=10)
        ax.set_xlim(0, max_count * 1.15)

    def _plot_probability_histogram(self, ax: plt.Axes) -> None:
        """Plot probability distribution histogram."""
        probabilities = self.metrics.get('probabilities', [])

        if not probabilities:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No probability data available',
                   transform=ax.transAxes, ha='center', va='center')
            return

        probs_array = np.array(probabilities)

        # Create histogram with color-coded bins
        bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
        n, bins_edges, patches = ax.hist(probs_array, bins=bins, edgecolor='white', alpha=0.8)

        # Color bins based on confidence level
        for patch, left_edge in zip(patches, bins_edges[:-1]):
            if left_edge >= 0.8:
                patch.set_facecolor(COLORS['high_conf'])
            elif left_edge >= 0.5:
                patch.set_facecolor(COLORS['medium_conf'])
            else:
                patch.set_facecolor(COLORS['low_conf'])

        # Add vertical lines for thresholds
        ax.axvline(x=0.5, color=COLORS['neutral'], linestyle='--', linewidth=1.5, alpha=0.7, label='0.5 threshold')
        ax.axvline(x=0.8, color=COLORS['neutral'], linestyle=':', linewidth=1.5, alpha=0.7, label='0.8 threshold')

        # Add mean line
        mean_prob = np.mean(probs_array)
        ax.axvline(x=mean_prob, color=COLORS['primary'], linestyle='-', linewidth=2, alpha=0.8, label=f'Mean: {mean_prob:.3f}')

        ax.set_xlabel('Probability', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Probability Distribution', fontsize=11, fontweight='bold', pad=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(0, 1)

    def _plot_per_label_confidence(self, ax: plt.Axes) -> None:
        """Plot per-label average confidence."""
        per_label_conf = self.metrics.get('per_label_confidence', {})

        if not per_label_conf:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No per-label confidence data',
                   transform=ax.transAxes, ha='center', va='center')
            return

        # Sort by mean confidence
        sorted_items = sorted(
            [(label, data['mean'], data['count']) for label, data in per_label_conf.items()],
            key=lambda x: x[1],
            reverse=True
        )

        # Limit to top 10
        if len(sorted_items) > 10:
            sorted_items = sorted_items[:10]

        labels = [item[0] for item in sorted_items]
        means = [item[1] for item in sorted_items]
        counts = [item[2] for item in sorted_items]

        # Truncate long labels
        display_labels = [l[:15] + '...' if len(str(l)) > 15 else str(l) for l in labels]

        y_pos = np.arange(len(labels))

        # Color based on confidence level
        colors = []
        for m in means:
            if m >= 0.8:
                colors.append(COLORS['high_conf'])
            elif m >= 0.5:
                colors.append(COLORS['medium_conf'])
            else:
                colors.append(COLORS['low_conf'])

        bars = ax.barh(y_pos, means, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_labels, fontsize=8)
        ax.invert_yaxis()

        # Add confidence labels
        for bar, mean, count in zip(bars, means, counts):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{mean:.2f} (n={count})', va='center', fontsize=8)

        ax.set_xlabel('Mean Confidence', fontsize=9)
        ax.set_title('Per-Label Confidence', fontsize=10, fontweight='bold', pad=10)
        ax.set_xlim(0, 1.15)

        # Add threshold lines
        ax.axvline(x=0.5, color=COLORS['neutral'], linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0.8, color=COLORS['neutral'], linestyle=':', linewidth=1, alpha=0.5)

    def _plot_language_distribution(self, ax: plt.Axes) -> None:
        """Plot language distribution."""
        lang_dist = self.metrics.get('language_distribution', {})

        if not lang_dist:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No language data available',
                   transform=ax.transAxes, ha='center', va='center')
            return

        # Sort by count
        sorted_items = sorted(lang_dist.items(), key=lambda x: x[1], reverse=True)
        languages = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]

        # Get colors
        colors = [LANGUAGE_COLORS.get(str(lang).lower(), COLORS['neutral']) for lang in languages]

        # Create bar chart
        x_pos = np.arange(len(languages))
        bars = ax.bar(x_pos, counts, color=colors, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(lang).upper() for lang in languages], fontsize=9)

        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{count:,}', ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('Count', fontsize=9)
        ax.set_title('Language Distribution', fontsize=10, fontweight='bold', pad=10)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def save_metrics_json(self) -> Optional[Path]:
        """
        Save metrics to a JSON file for further analysis.

        Returns:
            Path to the JSON file, or None if saving failed
        """
        try:
            # Remove raw probabilities from JSON (too large)
            metrics_for_json = {k: v for k, v in self.metrics.items() if k != 'probabilities'}
            metrics_for_json['probabilities_count'] = len(self.metrics.get('probabilities', []))

            metrics_data = {
                'session_id': self.session_id,
                'model_name': self.model_name,
                'category_name': self.category_name,
                'timestamp': self.timestamp.isoformat(),
                'label_names': self.label_names,
                'metrics': metrics_for_json,
            }

            safe_model_name = self.model_name.replace('/', '_').replace(':', '_').replace('\\', '_')[:40]
            safe_category = self.category_name.replace('/', '_').replace(':', '_')[:20]
            json_filename = (
                f"bert_annotation_metrics_{self.session_id}_"
                f"{safe_category}_{safe_model_name}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            )
            json_path = self.output_dir / json_filename

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"BERT annotation metrics JSON saved to {json_path}")
            return json_path

        except Exception as e:
            logger.error(f"Error saving BERT metrics JSON: {e}")
            return None


def generate_bert_annotation_chart(
    output_dir: Union[str, Path],
    session_id: str,
    model_name: str,
    df: Optional[pd.DataFrame] = None,
    label_column: Optional[str] = None,
    probability_column: Optional[str] = None,
    language_column: Optional[str] = None,
    category_name: Optional[str] = None,
    label_names: Optional[List[str]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    annotation_stats: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Convenience function to generate a BERT annotation metrics chart.

    Args:
        output_dir: Directory to save the chart
        session_id: Session identifier
        model_name: Name of the BERT model used
        df: Optional DataFrame with annotations
        label_column: Name of the label column
        probability_column: Name of the probability column
        language_column: Optional language column name
        category_name: Name of the annotation category
        label_names: List of label names
        model_info: Optional model information dictionary
        annotation_stats: Optional annotation statistics dictionary

    Returns:
        Path to the generated chart, or None if generation failed
    """
    try:
        chart = BERTAnnotationChart(
            output_dir=output_dir,
            session_id=session_id,
            model_name=model_name,
            label_names=label_names,
            category_name=category_name,
        )

        if df is not None and label_column:
            chart.update_from_dataframe(
                df=df,
                label_column=label_column,
                probability_column=probability_column,
                language_column=language_column,
            )

        if model_info and annotation_stats:
            chart.update_from_model_results(model_info, annotation_stats)

        chart_path = chart.generate_chart()
        chart.save_metrics_json()

        return chart_path

    except Exception as e:
        logger.error(f"Error generating BERT annotation chart: {e}")
        return None
