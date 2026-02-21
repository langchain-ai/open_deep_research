#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
data_splitter.py

MAIN OBJECTIVE:
---------------
Split datasets into train/validation/test partitions while maintaining
label and language balance for both single and multi-label workflows.

Dependencies:
-------------
- random
- typing
- collections
- numpy
- dataclasses
- logging

MAIN FEATURES:
--------------
1) Configure stratification behaviour through the SplitConfig dataclass
2) Perform label-aware splits with optional language stratification
3) Support multi-label inputs by deriving composite stratification keys
4) Guarantee minimum sample counts per stratum and normalise ratios
5) Provide verbose logging and helper utilities for downstream inspection

Author:
-------
Antoine Lemor
"""

import random
from typing import List, Tuple, Dict, Any, Optional, Union
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for data splitting."""
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    stratify_by_label: bool = True
    stratify_by_language: bool = False
    ensure_min_samples: int = 1
    random_seed: int = 42
    verbose: bool = True


class DataSplitter:
    """
    Advanced data splitter with stratification support.

    Ensures balanced distribution of labels across splits.
    """

    def __init__(self, config: Optional[SplitConfig] = None):
        """
        Initialize the data splitter.

        Args:
            config: SplitConfig object with splitting parameters
        """
        self.config = config or SplitConfig()
        self._validate_config()

        # Set random seed for reproducibility
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

    def _validate_config(self):
        """Validate configuration parameters."""
        total = self.config.train_ratio + self.config.val_ratio + self.config.test_ratio

        if abs(total - 1.0) > 0.001:
            # Normalize ratios
            self.config.train_ratio /= total
            self.config.val_ratio /= total
            self.config.test_ratio /= total

            if self.config.verbose:
                logger.info(f"Normalized ratios: train={self.config.train_ratio:.2f}, "
                          f"val={self.config.val_ratio:.2f}, test={self.config.test_ratio:.2f}")

    def split_data(self,
                   samples: List[Any],
                   labels: Optional[List[Any]] = None,
                   languages: Optional[List[str]] = None) -> Union[Tuple[List, List], Tuple[List, List, List]]:
        """
        Split data with stratification.

        Args:
            samples: List of samples (can be dicts, DataSample objects, etc.)
            labels: Optional list of labels for stratification
            languages: Optional list of language codes for stratification

        Returns:
            Tuple of (train, val) or (train, val, test) depending on config
        """
        if not samples:
            raise ValueError("No samples provided")

        # Extract labels and languages if samples are dicts/objects
        if labels is None and hasattr(samples[0], '__dict__'):
            labels = self._extract_labels(samples)

        if languages is None and hasattr(samples[0], '__dict__'):
            languages = self._extract_languages(samples)

        # Determine stratification strategy
        if self.config.stratify_by_label and labels:
            return self._stratified_split_by_label(samples, labels, languages)
        elif self.config.stratify_by_language and languages:
            return self._stratified_split_by_language(samples, languages)
        else:
            return self._random_split(samples)

    def _extract_labels(self, samples: List[Any]) -> Optional[List[Any]]:
        """Extract labels from samples."""
        labels = []
        for sample in samples:
            if hasattr(sample, 'label'):
                labels.append(sample.label)
            elif hasattr(sample, 'labels'):
                # For multi-label, use first label or create composite key
                if isinstance(sample.labels, dict) and sample.labels:
                    # Create a composite label for stratification
                    label_key = tuple(sorted(sample.labels.items()))
                    labels.append(label_key)
                else:
                    labels.append(sample.labels)
            elif isinstance(sample, dict):
                if 'label' in sample:
                    labels.append(sample['label'])
                elif 'labels' in sample:
                    if isinstance(sample['labels'], dict):
                        label_key = tuple(sorted(sample['labels'].items()))
                        labels.append(label_key)
                    elif isinstance(sample['labels'], list):
                        # Convert list to tuple for hashability
                        label_key = tuple(sorted(sample['labels'])) if sample['labels'] else ('__empty__',)
                        labels.append(label_key)
                    else:
                        labels.append(sample['labels'])
                else:
                    return None
            else:
                return None

        return labels if labels else None

    def _extract_languages(self, samples: List[Any]) -> Optional[List[str]]:
        """Extract languages from samples."""
        languages = []
        for sample in samples:
            if hasattr(sample, 'lang'):
                languages.append(sample.lang or 'unknown')
            elif isinstance(sample, dict) and 'lang' in sample:
                languages.append(sample['lang'] or 'unknown')
            else:
                return None

        return languages if languages else None

    def _stratified_split_by_label(self,
                                   samples: List[Any],
                                   labels: List[Any],
                                   languages: Optional[List[str]] = None) -> Union[Tuple[List, List], Tuple[List, List, List]]:
        """
        Perform stratified split by label.

        Ensures each split has the same proportion of each label.
        """
        # Group samples by label
        label_groups = defaultdict(list)
        for idx, (sample, label) in enumerate(zip(samples, labels)):
            # Convert lists to tuples for hashability
            if isinstance(label, list):
                label = tuple(sorted(label)) if label else ('__empty__',)
            elif isinstance(label, dict):
                label = tuple(sorted(label.items()))
            label_groups[label].append(idx)

        # Further stratify by language if requested
        if self.config.stratify_by_language and languages:
            return self._double_stratified_split(samples, labels, languages)

        # Initialize splits
        train_indices = []
        val_indices = []
        test_indices = []

        # Split each label group proportionally
        for label, indices in label_groups.items():
            # Shuffle indices
            random.shuffle(indices)

            n = len(indices)
            if n < self.config.ensure_min_samples:
                # If too few samples, put all in training
                train_indices.extend(indices)
                if self.config.verbose:
                    logger.warning(f"Label {label} has only {n} samples, all going to training set")
                continue

            # Calculate split points
            train_end = int(n * self.config.train_ratio)
            val_end = train_end + int(n * self.config.val_ratio)

            # Ensure at least one sample in validation if possible
            if self.config.val_ratio > 0 and val_end == train_end and n > 1:
                val_end = train_end + 1
                train_end = max(1, train_end - 1)

            # Split indices
            train_indices.extend(indices[:train_end])
            val_indices.extend(indices[train_end:val_end])

            if self.config.test_ratio > 0:
                test_indices.extend(indices[val_end:])

        # Shuffle final indices
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        if test_indices:
            random.shuffle(test_indices)

        # Create final splits
        train = [samples[i] for i in train_indices]
        val = [samples[i] for i in val_indices]

        if self.config.verbose:
            self._log_split_stats(train, val, test_indices, labels, label_groups)

        if self.config.test_ratio > 0:
            test = [samples[i] for i in test_indices]
            return train, val, test
        else:
            return train, val

    def _double_stratified_split(self,
                                 samples: List[Any],
                                 labels: List[Any],
                                 languages: List[str]) -> Union[Tuple[List, List], Tuple[List, List, List]]:
        """
        Perform double stratification by both label and language.
        """
        # Create composite keys
        composite_groups = defaultdict(list)
        for idx, (sample, label, lang) in enumerate(zip(samples, labels, languages)):
            # Convert lists to tuples for hashability
            if isinstance(label, list):
                label = tuple(sorted(label)) if label else ('__empty__',)
            elif isinstance(label, dict):
                label = tuple(sorted(label.items()))
            key = (label, lang)
            composite_groups[key].append(idx)

        # Initialize splits
        train_indices = []
        val_indices = []
        test_indices = []

        # Split each composite group
        for (label, lang), indices in composite_groups.items():
            random.shuffle(indices)

            n = len(indices)
            if n < self.config.ensure_min_samples:
                train_indices.extend(indices)
                continue

            train_end = int(n * self.config.train_ratio)
            val_end = train_end + int(n * self.config.val_ratio)

            train_indices.extend(indices[:train_end])
            val_indices.extend(indices[train_end:val_end])

            if self.config.test_ratio > 0:
                test_indices.extend(indices[val_end:])

        # Shuffle and create splits
        random.shuffle(train_indices)
        random.shuffle(val_indices)

        train = [samples[i] for i in train_indices]
        val = [samples[i] for i in val_indices]

        if self.config.test_ratio > 0:
            random.shuffle(test_indices)
            test = [samples[i] for i in test_indices]
            return train, val, test
        else:
            return train, val

    def _stratified_split_by_language(self,
                                      samples: List[Any],
                                      languages: List[str]) -> Union[Tuple[List, List], Tuple[List, List, List]]:
        """
        Perform stratified split by language only.
        """
        # Group by language
        lang_groups = defaultdict(list)
        for idx, (sample, lang) in enumerate(zip(samples, languages)):
            lang_groups[lang].append(idx)

        # Split each language group
        train_indices = []
        val_indices = []
        test_indices = []

        for lang, indices in lang_groups.items():
            random.shuffle(indices)

            n = len(indices)
            train_end = int(n * self.config.train_ratio)
            val_end = train_end + int(n * self.config.val_ratio)

            train_indices.extend(indices[:train_end])
            val_indices.extend(indices[train_end:val_end])

            if self.config.test_ratio > 0:
                test_indices.extend(indices[val_end:])

        # Create splits
        random.shuffle(train_indices)
        random.shuffle(val_indices)

        train = [samples[i] for i in train_indices]
        val = [samples[i] for i in val_indices]

        if self.config.test_ratio > 0:
            random.shuffle(test_indices)
            test = [samples[i] for i in test_indices]
            return train, val, test
        else:
            return train, val

    def _random_split(self, samples: List[Any]) -> Union[Tuple[List, List], Tuple[List, List, List]]:
        """
        Perform simple random split without stratification.
        """
        # Shuffle samples
        indices = list(range(len(samples)))
        random.shuffle(indices)

        # Calculate split points
        n = len(samples)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)

        # Create splits
        train = [samples[i] for i in indices[:train_end]]
        val = [samples[i] for i in indices[train_end:val_end]]

        if self.config.test_ratio > 0:
            test = [samples[i] for i in indices[val_end:]]
            return train, val, test
        else:
            return train, val

    def _log_split_stats(self, train, val, test_indices, labels, label_groups):
        """Log statistics about the splits."""
        logger.info(f"Split sizes: Train={len(train)}, Val={len(val)}, Test={len(test_indices)}")

        # Calculate label distribution
        for label in label_groups.keys():
            train_count = sum(1 for i, l in enumerate(labels[:len(train)]) if l == label)
            val_count = sum(1 for i, l in enumerate(labels[len(train):len(train)+len(val)]) if l == label)

            logger.info(f"  Label {label}: Train={train_count}, Val={val_count}")


def create_stratified_splits(samples: List[Any],
                            train_ratio: float = 0.8,
                            val_ratio: float = 0.2,
                            test_ratio: float = 0.0,
                            stratify_by_label: bool = True,
                            stratify_by_language: bool = False,
                            random_seed: int = 42) -> Union[Tuple[List, List], Tuple[List, List, List]]:
    """
    Convenience function for creating stratified splits.

    Args:
        samples: List of samples to split
        train_ratio: Proportion for training set (default: 0.8)
        val_ratio: Proportion for validation set (default: 0.2)
        test_ratio: Proportion for test set (default: 0.0)
        stratify_by_label: Whether to stratify by label
        stratify_by_language: Whether to stratify by language
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train, val) or (train, val, test) depending on test_ratio

    Example:
        >>> from LLMTool import create_stratified_splits
        >>> train, val = create_stratified_splits(samples, train_ratio=0.8, val_ratio=0.2)
    """
    config = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify_by_label=stratify_by_label,
        stratify_by_language=stratify_by_language,
        random_seed=random_seed
    )

    splitter = DataSplitter(config)
    return splitter.split_data(samples)


# Export main classes and functions
__all__ = ['DataSplitter', 'SplitConfig', 'create_stratified_splits']
