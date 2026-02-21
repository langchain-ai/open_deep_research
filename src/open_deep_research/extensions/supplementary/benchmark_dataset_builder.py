#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
benchmark_dataset_builder.py

MAIN OBJECTIVE:
---------------
Prepare balanced benchmark datasets with train/test splits, optional class
balancing, language tracking, and CSV logging for the Training Arena.

Dependencies:
-------------
- csv
- json
- random
- collections
- dataclasses
- pathlib
- typing
- numpy
- sklearn.model_selection

MAIN FEATURES:
--------------
1) Load JSON and JSONL sources into balanced train/test splits
2) Apply undersampling or oversampling strategies per configuration
3) Track language distributions and dataset metadata alongside splits
4) Persist split files and optional CSV logs for downstream benchmarking
5) Provide helper structures summarising class balance and dataset stats

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.model_selection import train_test_split

from llm_tool.utils.logging_utils import get_logger


@dataclass
class BenchmarkDataset:
    """Container for benchmark dataset with metadata."""

    train_texts: List[str]
    train_labels: List[int]
    test_texts: List[str]
    test_labels: List[int]
    train_languages: Optional[List[str]] = None  # Per-sample language info
    test_languages: Optional[List[str]] = None   # Per-sample language info
    language: Optional[str] = None  # Primary/dominant language
    category: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        # Calculate statistics
        self.metadata['train_size'] = len(self.train_texts)
        self.metadata['test_size'] = len(self.test_texts)
        self.metadata['train_class_distribution'] = Counter(self.train_labels)
        self.metadata['test_class_distribution'] = Counter(self.test_labels)
        self.metadata['train_class_balance'] = self._calculate_balance(self.train_labels)
        self.metadata['test_class_balance'] = self._calculate_balance(self.test_labels)

        # Track language distribution if available
        if self.train_languages:
            self.metadata['train_language_distribution'] = Counter(self.train_languages)
        if self.test_languages:
            self.metadata['test_language_distribution'] = Counter(self.test_languages)

        if self.language:
            self.metadata['language'] = self.language
        if self.category:
            self.metadata['category'] = self.category

    def _calculate_balance(self, labels: List[int]) -> float:
        """Calculate class balance ratio (minority/majority)."""
        counts = Counter(labels)
        if len(counts) < 2:
            return 1.0
        values = sorted(counts.values())
        return values[0] / values[-1] if values[-1] > 0 else 0.0

    def get_summary(self) -> str:
        """Return human-readable summary of dataset."""
        train_dist = self.metadata['train_class_distribution']
        test_dist = self.metadata['test_class_distribution']

        summary = f"📊 Dataset Summary:\n"
        summary += f"   - Language: {self.language or 'Not specified'}\n"

        # Show language distribution if available
        if 'test_language_distribution' in self.metadata:
            lang_dist = self.metadata['test_language_distribution']
            lang_str = ', '.join(f"{lang}: {count}" for lang, count in lang_dist.items())
            summary += f"   - Language distribution: {lang_str}\n"

        summary += f"   - Category: {self.category or 'Not specified'}\n"
        summary += f"   - Train: {self.metadata['train_size']} samples "
        summary += f"(Class 0: {train_dist.get(0, 0)}, Class 1: {train_dist.get(1, 0)})\n"
        summary += f"   - Test: {self.metadata['test_size']} samples "
        summary += f"(Class 0: {test_dist.get(0, 0)}, Class 1: {test_dist.get(1, 0)})\n"
        summary += f"   - Train balance: {self.metadata['train_class_balance']:.2%}\n"
        summary += f"   - Test balance: {self.metadata['test_class_balance']:.2%}"

        return summary


class BenchmarkDatasetBuilder:
    """Build benchmark datasets from various data sources."""

    def __init__(
        self,
        logger_name: str = "BenchmarkDatasetBuilder",
        random_state: int = 42,
        test_size: float = 0.2,
        stratify: bool = True,
        balance_classes: bool = False,
        balance_method: str = "undersample",  # "undersample", "oversample", or "hybrid"
        min_samples_per_class: int = 10,
        save_splits: bool = True,
        csv_log_path: Optional[Path] = None
    ):
        """Initialize dataset builder.

        Args:
            logger_name: Name for logger
            random_state: Random seed for reproducibility
            test_size: Proportion of data to use for test set
            stratify: Whether to stratify splits by class
            balance_classes: Whether to balance classes (optional)
            balance_method: Method for balancing ("undersample", "oversample", "hybrid")
            min_samples_per_class: Minimum samples required per class
            save_splits: Whether to save train/test splits to disk
            csv_log_path: Path for CSV logging of dataset statistics
        """
        self.logger = get_logger(logger_name)
        self.random_state = random_state
        self.test_size = test_size
        self.stratify = stratify
        self.balance_classes = balance_classes
        self.balance_method = balance_method
        self.min_samples_per_class = min_samples_per_class
        self.save_splits = save_splits
        self.csv_log_path = csv_log_path

        random.seed(random_state)
        np.random.seed(random_state)

        self.dataset_stats: List[Dict[str, Any]] = []

    def build_from_jsonl(
        self,
        data_path: Path,
        text_field: str = "text",
        label_field: str = "label",
        language_field: Optional[str] = "lang",
        category: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> Optional[BenchmarkDataset]:
        """Build dataset from JSONL file.

        Args:
            data_path: Path to JSONL file
            text_field: Field name for text
            label_field: Field name for label
            language_field: Field name for language (optional)
            category: Category name for this dataset
            max_samples: Maximum number of samples to use (None = all)

        Returns:
            BenchmarkDataset or None if building fails
        """
        if not data_path.exists():
            self.logger.error(f"Data file not found: {data_path}")
            return None

        texts = []
        labels = []
        languages = []

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if max_samples and len(texts) >= max_samples:
                        break

                    try:
                        record = json.loads(line)

                        # Extract text
                        if text_field not in record:
                            self.logger.warning(f"Line {line_num}: missing {text_field}")
                            continue

                        # Extract label
                        if label_field not in record:
                            self.logger.warning(f"Line {line_num}: missing {label_field}")
                            continue

                        texts.append(str(record[text_field]))
                        labels.append(int(record[label_field]))

                        # Extract language if available
                        if language_field and language_field in record:
                            languages.append(record[language_field])

                    except (json.JSONDecodeError, ValueError) as e:
                        self.logger.warning(f"Line {line_num}: {e}")
                        continue

            if not texts:
                self.logger.error(f"No valid samples found in {data_path}")
                return None

            # Detect language from data if available
            detected_language = None
            if languages:
                lang_counts = Counter(languages)
                detected_language = lang_counts.most_common(1)[0][0]

            return self._create_splits(
                texts, labels, languages if languages else None, detected_language, category
            )

        except Exception as e:
            self.logger.error(f"Failed to build dataset from {data_path}: {e}")
            return None

    def build_from_csv(
        self,
        data_path: Path,
        text_column: str = "text",
        label_column: str = "label",
        language_column: Optional[str] = None,
        category: Optional[str] = None,
        delimiter: str = ",",
        max_samples: Optional[int] = None
    ) -> Optional[BenchmarkDataset]:
        """Build dataset from CSV file."""
        if not data_path.exists():
            self.logger.error(f"Data file not found: {data_path}")
            return None

        texts = []
        labels = []
        languages = []

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=delimiter)

                for row_num, row in enumerate(reader, 1):
                    if max_samples and len(texts) >= max_samples:
                        break

                    if text_column not in row or label_column not in row:
                        self.logger.warning(f"Row {row_num}: missing required columns")
                        continue

                    texts.append(row[text_column])
                    labels.append(int(row[label_column]))

                    if language_column and language_column in row:
                        languages.append(row[language_column])

            if not texts:
                self.logger.error(f"No valid samples found in {data_path}")
                return None

            detected_language = None
            if languages:
                lang_counts = Counter(languages)
                detected_language = lang_counts.most_common(1)[0][0]

            return self._create_splits(
                texts, labels, languages if languages else None, detected_language, category
            )

        except Exception as e:
            self.logger.error(f"Failed to build dataset from {data_path}: {e}")
            return None

    def build_from_lists(
        self,
        texts: List[str],
        labels: List[int],
        language: Optional[str] = None,
        category: Optional[str] = None
    ) -> Optional[BenchmarkDataset]:
        """Build dataset from lists of texts and labels."""
        if len(texts) != len(labels):
            self.logger.error("Texts and labels must have same length")
            return None

        if not texts:
            self.logger.error("Empty dataset provided")
            return None

        return self._create_splits(texts, labels, None, language, category)

    def _create_splits(
        self,
        texts: List[str],
        labels: List[int],
        languages: Optional[List[str]],  # Per-sample languages
        primary_language: Optional[str],  # Primary/dominant language
        category: Optional[str]
    ) -> Optional[BenchmarkDataset]:
        """Create train/test splits with optional balancing."""

        # Check minimum samples
        from rich.prompt import Confirm
        from rich.console import Console
        from ..utils.data_filter_logger import get_filter_logger

        label_counts = Counter(labels)

        # Check if we have at least 2 instances per class for stratification
        min_count = min(label_counts.values()) if label_counts else 0
        if min_count < 2:
            # Find which classes have insufficient instances
            insufficient_classes = [cls for cls, count in label_counts.items() if count < 2]

            # Display warning to user
            console = Console()
            console.print(f"\n[yellow]⚠️  Found {len(insufficient_classes)} label(s) with insufficient samples:[/yellow]")
            for cls in insufficient_classes:
                count = label_counts[cls]
                console.print(f"  • [red]Label {cls}[/red]: {count} sample(s) - need at least 2 for train/test split")

            console.print(f"\n[bold]What would you like to do?[/bold]")
            console.print(f"  [cyan]1.[/cyan] [green]Remove[/green] these {len(insufficient_classes)} label(s) and continue")
            console.print(f"  [cyan]2.[/cyan] [red]Cancel[/red] to add more samples manually\n")

            remove_labels = Confirm.ask(
                f"[bold yellow]Remove insufficient labels and continue?[/bold yellow]",
                default=True
            )

            if remove_labels:
                # Filter out samples with insufficient labels
                filter_logger = get_filter_logger()

                # Find indices to keep (NOT insufficient)
                indices_to_keep = [i for i, label in enumerate(labels) if label not in insufficient_classes]
                indices_to_remove = [i for i, label in enumerate(labels) if label in insufficient_classes]

                # Log filtered samples
                if indices_to_remove:
                    # Create temporary dataframe for logging
                    import pandas as pd
                    temp_df_before = pd.DataFrame({
                        'text': texts,
                        'label': labels,
                        'language': languages if languages else [None] * len(texts)
                    })
                    temp_df_after = temp_df_before.iloc[indices_to_keep]

                    filter_logger.log_dataframe_filtering(
                        df_before=temp_df_before,
                        df_after=temp_df_after,
                        reason="insufficient_samples_per_class",
                        location=f"benchmark_dataset_builder._create_splits.{category if category else 'unknown'}",
                        text_column='text',
                        log_filtered_samples=min(5, len(indices_to_remove))
                    )

                    console.print(f"\n[green]✓ Removing {len(indices_to_remove)} sample(s) with insufficient labels[/green]")

                # Filter arrays
                texts = [texts[i] for i in indices_to_keep]
                labels = [labels[i] for i in indices_to_keep]
                if languages:
                    languages = [languages[i] for i in indices_to_keep]

                # Recompute label_counts
                label_counts = Counter(labels)

                console.print(f"[green]✓ Continuing with {len(texts)} samples and {len(set(labels))} unique labels[/green]\n")
            else:
                error_msg = (
                    f"Benchmark cancelled by user.\n"
                    f"Please add more samples for label(s): {insufficient_classes}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        for label, count in label_counts.items():
            if count < self.min_samples_per_class:
                self.logger.warning(
                    f"Class {label} has only {count} samples "
                    f"(minimum {self.min_samples_per_class} required)"
                )

        # Log original distribution
        self.logger.info(f"Original distribution: {dict(label_counts)}")
        self.logger.info(f"Total samples: {len(texts)}")

        # Apply balancing if requested (before splitting)
        if self.balance_classes:
            if languages:
                texts, labels, languages = self._balance_dataset_with_languages(texts, labels, languages)
            else:
                texts, labels = self._balance_dataset(texts, labels)
            self.logger.info(f"After balancing: {dict(Counter(labels))}")

        # Create splits
        stratify_labels = labels if self.stratify else None

        try:
            if languages:
                # Split with languages
                train_texts, test_texts, train_labels, test_labels, train_languages, test_languages = train_test_split(
                    texts, labels, languages,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=stratify_labels
                )
            else:
                # Split without languages
                train_texts, test_texts, train_labels, test_labels = train_test_split(
                    texts, labels,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=stratify_labels
                )
                train_languages = None
                test_languages = None
        except ValueError as e:
            self.logger.error(f"Failed to create splits: {e}")
            # Try without stratification
            if self.stratify:
                self.logger.info("Retrying without stratification...")
                if languages:
                    train_texts, test_texts, train_labels, test_labels, train_languages, test_languages = train_test_split(
                        texts, labels, languages,
                        test_size=self.test_size,
                        random_state=self.random_state,
                        stratify=None
                    )
                else:
                    train_texts, test_texts, train_labels, test_labels = train_test_split(
                        texts, labels,
                        test_size=self.test_size,
                        random_state=self.random_state,
                        stratify=None
                    )
                    train_languages = None
                    test_languages = None
            else:
                return None

        # Create dataset
        dataset = BenchmarkDataset(
            train_texts=train_texts,
            train_labels=train_labels,
            test_texts=test_texts,
            test_labels=test_labels,
            train_languages=train_languages,
            test_languages=test_languages,
            language=primary_language,
            category=category
        )

        # Log statistics
        self.logger.info(dataset.get_summary())

        # Save to CSV log if configured
        if self.csv_log_path:
            self._log_to_csv(dataset)

        # Save splits if configured
        if self.save_splits:
            self._save_splits(dataset)

        return dataset

    def _balance_dataset(
        self,
        texts: List[str],
        labels: List[int]
    ) -> Tuple[List[str], List[int]]:
        """Balance dataset classes using configured method."""

        # Group samples by class
        class_samples = {}
        for text, label in zip(texts, labels):
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(text)

        # Find target size
        class_sizes = {label: len(samples) for label, samples in class_samples.items()}

        if self.balance_method == "undersample":
            target_size = min(class_sizes.values())
        elif self.balance_method == "oversample":
            target_size = max(class_sizes.values())
        else:  # hybrid
            target_size = int(np.median(list(class_sizes.values())))

        # Balance each class
        balanced_texts = []
        balanced_labels = []

        for label, samples in class_samples.items():
            current_size = len(samples)

            if current_size < target_size:
                # Oversample
                additional = target_size - current_size
                extra_samples = random.choices(samples, k=additional)
                samples = samples + extra_samples
            elif current_size > target_size:
                # Undersample
                samples = random.sample(samples, k=target_size)

            balanced_texts.extend(samples)
            balanced_labels.extend([label] * len(samples))

        # Shuffle
        combined = list(zip(balanced_texts, balanced_labels))
        random.shuffle(combined)
        balanced_texts, balanced_labels = zip(*combined)

        return list(balanced_texts), list(balanced_labels)

    def _balance_dataset_with_languages(
        self,
        texts: List[str],
        labels: List[int],
        languages: List[str]
    ) -> Tuple[List[str], List[int], List[str]]:
        """Balance dataset classes while preserving language information."""

        # Group samples by class with their languages
        class_samples = {}
        for text, label, lang in zip(texts, labels, languages):
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append((text, lang))

        # Find target size
        class_sizes = {label: len(samples) for label, samples in class_samples.items()}

        if self.balance_method == "undersample":
            target_size = min(class_sizes.values())
        elif self.balance_method == "oversample":
            target_size = max(class_sizes.values())
        else:  # hybrid
            target_size = int(np.median(list(class_sizes.values())))

        # Balance each class
        balanced_texts = []
        balanced_labels = []
        balanced_languages = []

        for label, samples in class_samples.items():
            current_size = len(samples)

            if current_size < target_size:
                # Oversample
                additional = target_size - current_size
                extra_samples = random.choices(samples, k=additional)
                samples = samples + extra_samples
            elif current_size > target_size:
                # Undersample
                samples = random.sample(samples, k=target_size)

            for text, lang in samples:
                balanced_texts.append(text)
                balanced_labels.append(label)
                balanced_languages.append(lang)

        # Shuffle
        combined = list(zip(balanced_texts, balanced_labels, balanced_languages))
        random.shuffle(combined)
        balanced_texts, balanced_labels, balanced_languages = zip(*combined)

        return list(balanced_texts), list(balanced_labels), list(balanced_languages)

    def _save_splits(self, dataset: BenchmarkDataset) -> None:
        """Save train/test splits to JSONL files."""
        if not dataset.category:
            return

        output_dir = Path("data/benchmark") / dataset.category
        if dataset.language:
            output_dir = output_dir / dataset.language

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save train split
        train_file = output_dir / "train.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for text, label in zip(dataset.train_texts, dataset.train_labels):
                record = {
                    "text": text,
                    "label": label,
                    "lang": dataset.language
                } if dataset.language else {
                    "text": text,
                    "label": label
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        # Save test split
        test_file = output_dir / "test.jsonl"
        with open(test_file, 'w', encoding='utf-8') as f:
            for text, label in zip(dataset.test_texts, dataset.test_labels):
                record = {
                    "text": text,
                    "label": label,
                    "lang": dataset.language
                } if dataset.language else {
                    "text": text,
                    "label": label
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        self.logger.info(f"Splits saved to {output_dir}")

    def _log_to_csv(self, dataset: BenchmarkDataset) -> None:
        """Log dataset statistics to CSV."""
        if not self.csv_log_path:
            return

        # Prepare record
        record = {
            'timestamp': np.datetime64('now'),
            'category': dataset.category or 'unknown',
            'language': dataset.language or 'unknown',
            'train_size': dataset.metadata['train_size'],
            'test_size': dataset.metadata['test_size'],
            'train_class_0': dataset.metadata['train_class_distribution'].get(0, 0),
            'train_class_1': dataset.metadata['train_class_distribution'].get(1, 0),
            'test_class_0': dataset.metadata['test_class_distribution'].get(0, 0),
            'test_class_1': dataset.metadata['test_class_distribution'].get(1, 0),
            'train_balance': f"{dataset.metadata['train_class_balance']:.2%}",
            'test_balance': f"{dataset.metadata['test_class_balance']:.2%}",
            'balanced': 'Yes' if self.balance_classes else 'No',
            'balance_method': self.balance_method if self.balance_classes else 'N/A'
        }

        # Write to CSV
        file_exists = self.csv_log_path.exists()

        with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(record)

        self.logger.info(f"Statistics logged to {self.csv_log_path}")
