#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
multilingual_selector.py

MAIN OBJECTIVE:
---------------
Analyse language distributions and recommend multilingual transformer models,
including ensemble strategies and resource-aware guidance.

Dependencies:
-------------
- typing
- dataclasses
- enum
- logging
- collections
- numpy
- llm_tool.trainers.bert_base
- llm_tool.trainers.sota_models
- llm_tool.trainers.models

MAIN FEATURES:
--------------
1) Model language distribution via LanguageDistribution and helper enums
2) Score multilingual backbones using curated performance matrices
3) Recommend single-model or ensemble strategies tailored to datasets
4) Estimate resource requirements and provide deployment-friendly hints
5) Offer utilities to build multilingual ensembles for the training pipeline

Author:
-------
Antoine Lemor
"""

from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import Counter
import numpy as np

from llm_tool.trainers.bert_base import BertBase
from llm_tool.trainers.sota_models import (
    MDeBERTaV3Base,
    XLMRobertaBase,
    XLMRobertaLarge,
)
from llm_tool.trainers.models import XLMRoberta


class ModelSize(Enum):
    """Model size categories for performance-resource trade-offs."""
    XSMALL = "xsmall"  # < 100M parameters
    SMALL = "small"    # 100-200M parameters
    BASE = "base"      # 200-400M parameters
    LARGE = "large"    # 400M+ parameters


class TaskType(Enum):
    """Task types for specialized model selection."""
    CLASSIFICATION = "classification"
    SENTIMENT = "sentiment"
    NER = "ner"
    QA = "qa"
    SIMILARITY = "similarity"


@dataclass
class LanguageDistribution:
    """Represents language distribution in dataset."""
    languages: Dict[str, float]  # language_code -> proportion
    primary_language: str
    is_balanced: bool
    diversity_score: float  # 0-1, higher = more diverse


@dataclass
class ModelRecommendation:
    """Comprehensive model recommendation with rationale."""
    model_class: type
    model_name: str
    score: float
    rationale: Dict[str, Any]
    expected_performance: Dict[str, float]
    resource_requirements: Dict[str, Any]


class MultilingualModelSelector:
    """
    Sophisticated multilingual model selection system.
    Analyzes language distribution, task requirements, and constraints to recommend optimal models.
    """

    # Performance benchmarks (F1 scores) for different models on multilingual tasks
    PERFORMANCE_MATRIX = {
        'mdeberta-v3-base': {
            'avg_score': 0.89,
            'languages': {
                'en': 0.92, 'zh': 0.88, 'es': 0.90, 'fr': 0.91, 'de': 0.90,
                'ja': 0.87, 'ru': 0.88, 'ar': 0.86, 'hi': 0.85, 'pt': 0.89,
                'it': 0.89, 'ko': 0.86, 'nl': 0.89, 'tr': 0.87, 'pl': 0.88
            },
            'memory_gb': 1.2,
            'inference_speed': 0.85,  # relative
            'max_seq_length': 512
        },
        'xlm-roberta-base': {
            'avg_score': 0.86,
            'languages': {
                'en': 0.90, 'zh': 0.85, 'es': 0.88, 'fr': 0.89, 'de': 0.88,
                'ja': 0.84, 'ru': 0.86, 'ar': 0.83, 'hi': 0.82, 'pt': 0.87,
                'it': 0.87, 'ko': 0.83, 'nl': 0.87, 'tr': 0.85, 'pl': 0.86
            },
            'memory_gb': 1.1,
            'inference_speed': 0.90,
            'max_seq_length': 512
        },
        'xlm-roberta-large': {
            'avg_score': 0.91,
            'languages': {
                'en': 0.94, 'zh': 0.90, 'es': 0.92, 'fr': 0.93, 'de': 0.92,
                'ja': 0.89, 'ru': 0.90, 'ar': 0.88, 'hi': 0.87, 'pt': 0.91,
                'it': 0.91, 'ko': 0.88, 'nl': 0.91, 'tr': 0.89, 'pl': 0.90
            },
            'memory_gb': 2.8,
            'inference_speed': 0.45,
            'max_seq_length': 512
        }
    }

    # Language families for similarity scoring
    LANGUAGE_FAMILIES = {
        'germanic': ['en', 'de', 'nl', 'sv', 'da', 'no'],
        'romance': ['fr', 'es', 'it', 'pt', 'ro'],
        'slavic': ['ru', 'pl', 'cs', 'uk', 'bg'],
        'sino_tibetan': ['zh', 'my'],
        'japonic': ['ja'],
        'koreanic': ['ko'],
        'semitic': ['ar', 'he'],
        'indo_aryan': ['hi', 'bn', 'ur'],
        'turkic': ['tr', 'az', 'kk']
    }

    def __init__(self,
                 verbose: bool = True,
                 custom_benchmarks: Optional[Dict] = None):
        """
        Initialize the multilingual model selector.

        Args:
            verbose: Enable detailed logging
            custom_benchmarks: Custom performance benchmarks to override defaults
        """
        self.verbose = verbose
        if custom_benchmarks:
            self.PERFORMANCE_MATRIX.update(custom_benchmarks)

        if verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

    def analyze_language_distribution(self,
                                     texts: List[str],
                                     labels: Optional[List[str]] = None,
                                     sample_size: int = 1000) -> LanguageDistribution:
        """
        Analyze language distribution in dataset.

        Args:
            texts: List of text samples
            labels: Optional labels for stratified analysis
            sample_size: Number of samples to analyze for efficiency

        Returns:
            LanguageDistribution object with detailed statistics
        """
        try:
            from langdetect import detect_langs, LangDetectException
        except ImportError:
            raise ImportError("langdetect required. Install with: pip install langdetect")

        # Sample texts for efficiency
        if len(texts) > sample_size:
            import random
            sampled_texts = random.sample(texts, sample_size)
        else:
            sampled_texts = texts

        language_counts = Counter()

        for text in sampled_texts:
            try:
                # Get language probabilities
                langs = detect_langs(text[:500])  # Limit text length for speed
                if langs:
                    # Use highest probability language
                    language_counts[langs[0].lang] += 1
            except LangDetectException:
                continue

        # Calculate proportions
        total = sum(language_counts.values())
        if total == 0:
            return LanguageDistribution(
                languages={'unknown': 1.0},
                primary_language='unknown',
                is_balanced=False,
                diversity_score=0.0
            )

        languages = {lang: count/total for lang, count in language_counts.items()}
        primary_language = max(languages, key=languages.get)

        # Calculate diversity score (normalized entropy)
        entropy = -sum(p * np.log(p) for p in languages.values() if p > 0)
        max_entropy = np.log(len(languages)) if len(languages) > 1 else 1
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0

        # Check if balanced (no language > 70%)
        is_balanced = max(languages.values()) < 0.7

        return LanguageDistribution(
            languages=languages,
            primary_language=primary_language,
            is_balanced=is_balanced,
            diversity_score=diversity_score
        )

    def calculate_model_score(self,
                            model_key: str,
                            lang_dist: LanguageDistribution,
                            size_constraint: Optional[ModelSize] = None,
                            speed_priority: float = 0.3) -> Tuple[float, Dict]:
        """
        Calculate comprehensive score for a model given requirements.

        Args:
            model_key: Key in PERFORMANCE_MATRIX
            lang_dist: Language distribution analysis
            size_constraint: Maximum model size allowed
            speed_priority: Weight for inference speed (0-1)

        Returns:
            Tuple of (score, detailed_metrics)
        """
        if model_key not in self.PERFORMANCE_MATRIX:
            return 0.0, {}

        model_info = self.PERFORMANCE_MATRIX[model_key]

        # Calculate language coverage score
        lang_coverage = 0.0
        covered_langs = 0

        for lang, proportion in lang_dist.languages.items():
            if lang in model_info['languages']:
                lang_coverage += proportion * model_info['languages'][lang]
                covered_langs += 1
            else:
                # Penalize for unsupported languages
                lang_coverage += proportion * 0.5

        # Diversity handling bonus
        diversity_bonus = 0.0
        if lang_dist.diversity_score > 0.5:
            # Model gets bonus for handling diverse languages well
            diversity_bonus = 0.1 * model_info['avg_score']

        # Speed score
        speed_score = model_info['inference_speed'] * speed_priority

        # Performance score
        perf_score = model_info['avg_score'] * (1 - speed_priority)

        # Size constraint penalty
        size_penalty = 0.0
        if size_constraint:
            if size_constraint == ModelSize.XSMALL and model_info['memory_gb'] > 0.5:
                size_penalty = 0.3
            elif size_constraint == ModelSize.SMALL and model_info['memory_gb'] > 1.0:
                size_penalty = 0.2
            elif size_constraint == ModelSize.BASE and model_info['memory_gb'] > 2.0:
                size_penalty = 0.1

        # Calculate final score
        # NOTE: No speed criteria - selection based on quality only
        final_score = (
            lang_coverage * 0.50 +    # 50% weight on language coverage
            perf_score * 0.40 +       # 40% weight on performance
            diversity_bonus * 0.10 -  # 10% bonus for diversity
            size_penalty
        )

        detailed_metrics = {
            'language_coverage': lang_coverage,
            'performance_score': perf_score,
            'speed_score': speed_score,
            'diversity_bonus': diversity_bonus,
            'size_penalty': size_penalty,
            'covered_languages': covered_langs,
            'memory_requirement': model_info['memory_gb']
        }

        return final_score, detailed_metrics

    def recommend_model(self,
                        texts: Optional[List[str]] = None,
                        lang_dist: Optional[LanguageDistribution] = None,
                        task_type: TaskType = TaskType.CLASSIFICATION,
                        size_constraint: Optional[ModelSize] = None,
                        speed_priority: float = 0.3,
                        min_language_coverage: float = 0.8) -> ModelRecommendation:
        """
        Recommend best multilingual model based on comprehensive analysis.

        Args:
            texts: Text samples for language analysis (if lang_dist not provided)
            lang_dist: Pre-computed language distribution
            task_type: Type of NLP task
            size_constraint: Maximum model size
            speed_priority: Importance of inference speed (0-1)
            min_language_coverage: Minimum required language coverage

        Returns:
            ModelRecommendation with best model and rationale
        """
        # Analyze language distribution if not provided
        if lang_dist is None:
            if texts is None:
                raise ValueError("Either texts or lang_dist must be provided")
            lang_dist = self.analyze_language_distribution(texts)

        # Score all models
        model_scores = {}
        model_details = {}

        model_mapping = {
            'mdeberta-v3-base': (MDeBERTaV3Base, 'microsoft/mdeberta-v3-base'),
            'xlm-roberta-base': (XLMRobertaBase, 'xlm-roberta-base'),
            'xlm-roberta-large': (XLMRobertaLarge, 'xlm-roberta-large')
        }

        for model_key in self.PERFORMANCE_MATRIX.keys():
            score, details = self.calculate_model_score(
                model_key, lang_dist, size_constraint, speed_priority
            )
            model_scores[model_key] = score
            model_details[model_key] = details

        # Select best model
        best_model_key = max(model_scores, key=model_scores.get)
        best_score = model_scores[best_model_key]

        # Check if meets minimum requirements
        if model_details[best_model_key]['language_coverage'] < min_language_coverage:
            if self.verbose:
                self.logger.warning(
                    f"Best model only covers {model_details[best_model_key]['language_coverage']:.1%} "
                    f"of languages (minimum required: {min_language_coverage:.1%})"
                )

        # Prepare recommendation
        model_class, model_name = model_mapping[best_model_key]

        recommendation = ModelRecommendation(
            model_class=model_class,
            model_name=model_name,
            score=best_score,
            rationale={
                'primary_reason': self._get_selection_reason(best_model_key, model_details[best_model_key]),
                'language_distribution': lang_dist.__dict__,
                'detailed_scores': model_details[best_model_key],
                'alternatives': {k: v for k, v in model_scores.items() if k != best_model_key}
            },
            expected_performance={
                'avg_f1': self.PERFORMANCE_MATRIX[best_model_key]['avg_score'],
                'language_specific': {
                    lang: self.PERFORMANCE_MATRIX[best_model_key]['languages'].get(lang, 0.5)
                    for lang in lang_dist.languages.keys()
                }
            },
            resource_requirements={
                'memory_gb': self.PERFORMANCE_MATRIX[best_model_key]['memory_gb'],
                'inference_speed': self.PERFORMANCE_MATRIX[best_model_key]['inference_speed'],
                'max_sequence_length': self.PERFORMANCE_MATRIX[best_model_key]['max_seq_length']
            }
        )

        if self.verbose:
            self._print_recommendation(recommendation)

        return recommendation

    def _get_selection_reason(self, model_key: str, details: Dict) -> str:
        """Generate human-readable selection reason."""
        reasons = []

        if details['language_coverage'] > 0.9:
            reasons.append("excellent language coverage")
        if details['performance_score'] > 0.85:
            reasons.append("superior performance metrics")
        if details['speed_score'] > 0.7:
            reasons.append("fast inference speed")
        if details['diversity_bonus'] > 0.05:
            reasons.append("handles language diversity well")

        if not reasons:
            reasons.append("best overall balance")

        return f"Selected for {', '.join(reasons)}"

    def _print_recommendation(self, rec: ModelRecommendation):
        """Print formatted recommendation."""
        print("\n" + "="*60)
        print("MULTILINGUAL MODEL RECOMMENDATION")
        print("="*60)
        print(f"Recommended Model: {rec.model_name}")
        print(f"Overall Score: {rec.score:.3f}")
        print(f"Rationale: {rec.rationale['primary_reason']}")
        print(f"\nExpected Performance:")
        print(f"  - Average F1: {rec.expected_performance['avg_f1']:.3f}")
        print(f"\nResource Requirements:")
        print(f"  - Memory: {rec.resource_requirements['memory_gb']:.1f} GB")
        print(f"  - Relative Speed: {rec.resource_requirements['inference_speed']:.2f}")
        print("="*60 + "\n")

    def get_model_for_languages(self,
                               languages: List[str],
                               weights: Optional[List[float]] = None) -> type:
        """
        Quick model selection for specific languages.

        Args:
            languages: List of ISO language codes
            weights: Optional weights for each language

        Returns:
            Model class ready for instantiation
        """
        if weights is None:
            weights = [1.0] * len(languages)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]

        # Create language distribution
        lang_dist = LanguageDistribution(
            languages=dict(zip(languages, weights)),
            primary_language=languages[np.argmax(weights)],
            is_balanced=max(weights) < 0.7,
            diversity_score=len(set(languages)) / len(languages)
        )

        rec = self.recommend_model(lang_dist=lang_dist)
        return rec.model_class


def create_multilingual_ensemble(
    models: List[BertBase],
    weights: Optional[List[float]] = None,
    voting: str = 'soft'
) -> 'MultilingualEnsemble':
    """
    Create an ensemble of multilingual models for improved performance.

    Args:
        models: List of initialized model instances
        weights: Optional weights for each model
        voting: 'soft' for probability averaging, 'hard' for majority voting

    Returns:
        MultilingualEnsemble instance
    """
    return MultilingualEnsemble(models, weights, voting)


class MultilingualEnsemble:
    """Ensemble of multilingual models for robust predictions."""

    def __init__(self,
                 models: List[BertBase],
                 weights: Optional[List[float]] = None,
                 voting: str = 'soft'):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.voting = voting

    def predict(self, dataloader) -> np.ndarray:
        """Generate ensemble predictions."""
        predictions = []

        for model, weight in zip(self.models, self.weights):
            preds = model.predict(dataloader, model.model)
            predictions.append(preds * weight)

        if self.voting == 'soft':
            return np.mean(predictions, axis=0)
        else:  # hard voting
            return np.round(np.mean(predictions, axis=0))
