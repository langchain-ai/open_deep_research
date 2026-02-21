#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
llm_annotator.py

MAIN OBJECTIVE:
---------------
This is the core annotation module that provides comprehensive LLM annotation
capabilities including single and multi-prompt processing, parallel execution,
JSON repair, schema validation, and support for multiple data sources.

Dependencies:
-------------
- sys
- os
- json
- pandas
- numpy
- logging
- typing
- concurrent.futures
- time
- math
- random
- pathlib
- tqdm
- sqlalchemy
- pydantic

MAIN FEATURES:
--------------
1) Single and multi-prompt annotation processing
2) Parallel execution with ProcessPoolExecutor
3) JSON repair and validation (5 retry attempts)
4) Schema validation with Pydantic
5) Support for PostgreSQL, CSV, Excel, Parquet, RData/RDS
6) Incremental saving and resume capability
7) Progress tracking with error handling
8) Sample size calculation (95% CI)
9) Warm-up calls for Ollama models
10) Per-prompt status tracking

Author:
-------
Antoine Lemor
"""

import os
import sys
import json
import re
import shutil
import logging
import time
import math
import random
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Iterable, Set
from datetime import datetime
from collections import deque, defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel, create_model

# SQLAlchemy imports for database support
try:
    from sqlalchemy import create_engine, text, JSON, bindparam
    from sqlalchemy.exc import SQLAlchemyError
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    logging.warning("SQLAlchemy not installed. PostgreSQL support disabled.")

# pyreadr for RData/RDS support
try:
    import pyreadr
    HAS_PYREADR = True
except ImportError:
    HAS_PYREADR = False
    logging.warning("pyreadr not installed. RData/RDS support disabled.")

# Import from other modules
from ..annotators.api_clients import create_api_client
from ..annotators.prompt_manager import PromptManager
from ..annotators.json_cleaner import JSONCleaner, clean_json_output
from ..config.settings import Settings
from ..utils.data_filter_logger import get_filter_logger

# Try to import local model support
try:
    from ..annotators.local_models import OllamaClient, LlamaCPPClient
    HAS_LOCAL_MODELS = True
except ImportError:
    HAS_LOCAL_MODELS = False
    logging.warning("Local model support not available")

# OpenAI SDK for batch operations
try:
    from openai import OpenAI, NotFoundError
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None
    NotFoundError = Exception
    logging.warning("OpenAI SDK not installed. Batch API support disabled.")

OPENAI_BATCH_MAX_FILE_BYTES = 512 * 1024 * 1024  # 512 MB per input file
OPENAI_BATCH_MAX_INPUT_TOKENS = 50_000_000  # OpenAI documented soft limit
OPENAI_BATCH_APPROX_CHARS_PER_TOKEN = 4  # heuristic for request sizing
OPENAI_BATCH_APPROX_TOKEN_OVERHEAD = 32  # safety margin for system tokens
OPENAI_BATCH_COMPLETION_OVERHEAD = 64  # padding for completion tokens in batch sizing

# Rich library for enhanced CLI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

# Constants
CSV_APPEND = True
OTHER_FORMAT_SAVE_EVERY = 50
PROMPT_SUFFIXES = ["raw_per_prompt", "cleaned_per_prompt", "status_per_prompt"]

# Global tracking
status_counts = {"success": 0, "error": 0, "cleaning_failed": 0, "decode_error": 0}

# Context Window Labels (default English)
DEFAULT_CONTEXT_LABELS = {
    'context_before': 'Context sentences BEFORE the sentence to annotate:',
    'sentence_to_annotate': 'Sentence to annotate:',
    'context_after': 'Context sentences AFTER:'
}


def _coerce_mapping(value: Any) -> Optional[Dict[str, Any]]:
    """Best-effort conversion of SDK objects into plain dictionaries."""
    if value is None:
        return None

    if isinstance(value, dict):
        return value

    for attr_name in ("model_dump", "to_dict", "dict"):
        attr = getattr(value, attr_name, None)
        if callable(attr):
            try:
                data = attr()
            except TypeError:
                data = None
        else:
            data = attr
        if isinstance(data, dict):
            return data

    if hasattr(value, "__dict__"):
        # Filter out private attributes that Pydantic / SDK objects may include
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}

    try:
        data = dict(value)  # type: ignore[arg-type]
    except Exception:
        data = None
    return data if isinstance(data, dict) else None


def _as_request_counts(raw_counts: Optional[Any]) -> Dict[str, int]:
    """Normalize OpenAI batch request counts into an int-valued dict."""
    mapping = _coerce_mapping(raw_counts) or {}
    normalized: Dict[str, int] = {}

    for key, value in mapping.items():
        if value in (None, "", False):
            continue
        try:
            normalized[str(key)] = int(value)
            continue
        except (TypeError, ValueError):
            pass
        try:
            normalized[str(key)] = int(float(value))
        except (TypeError, ValueError):
            continue

    return normalized


def _stringify_error_detail(detail: Any) -> str:
    """Convert various error payloads into a concise string."""
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail.strip()

    mapping = _coerce_mapping(detail)
    if mapping:
        for key in ("message", "error", "description", "detail"):
            value = mapping.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        try:
            return json.dumps(mapping, ensure_ascii=False)
        except Exception:
            return str(mapping)

    if isinstance(detail, (list, tuple, set)):
        parts = [part for part in (_stringify_error_detail(item) for item in detail) if part]
        return ", ".join(parts)

    try:
        return str(detail)
    except Exception:
        return repr(detail)


def _extract_output_file_id(batch_job: Any) -> Optional[str]:
    """Safely extract the output file id from an OpenAI batch response."""

    def _normalize_candidate(candidate: Any) -> Optional[str]:
        if candidate in (None, "", False):
            return None
        if isinstance(candidate, str):
            candidate = candidate.strip()
            return candidate or None
        if isinstance(candidate, (list, tuple, set)):
            for item in candidate:
                result = _normalize_candidate(item)
                if result:
                    return result
            return None
        mapping = _coerce_mapping(candidate)
        if mapping:
            for key in ("output_file_id", "file_id", "id"):
                result = _normalize_candidate(mapping.get(key))
                if result:
                    return result
            return None
        for attr in ("file_id", "id"):
            if hasattr(candidate, attr):
                result = _normalize_candidate(getattr(candidate, attr))
                if result:
                    return result
        return None

    if batch_job is None:
        return None

    candidates: List[Any] = [
        getattr(batch_job, "output_file_id", None),
        getattr(batch_job, "output_file", None),
        getattr(batch_job, "output_file_ids", None),
        getattr(batch_job, "output_files", None),
    ]

    mapping = _coerce_mapping(batch_job)
    if mapping:
        candidates.extend(
            [
                mapping.get("output_file_id"),
                mapping.get("output_file"),
                mapping.get("output_file_ids"),
                mapping.get("output_files"),
            ]
        )

    for candidate in candidates:
        result = _normalize_candidate(candidate)
        if result:
            return result

    return None


class LLMAnnotator:
    """Main LLM annotation class with comprehensive features"""

    def __init__(self, settings: Optional[Settings] = None, progress_callback=None, progress_manager=None):
        """Initialize the LLM annotator

        Args:
            settings: Optional Settings object
            progress_callback: Optional callback for progress updates (current, total, message)
            progress_manager: Optional progress manager for displaying warnings/errors
        """
        self.settings = settings or Settings()
        self.progress_callback = progress_callback
        self.progress_manager = progress_manager
        self.logger = logging.getLogger(__name__)

        # If we have a progress manager, disable console logging to avoid conflicts
        if self.progress_manager:
            # Remove all console handlers to prevent duplicate output
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    self.logger.removeHandler(handler)
            # Also prevent propagation to root logger
            self.logger.propagate = False

        self.json_cleaner = JSONCleaner()
        self.prompt_manager = PromptManager()
        self.api_client = None
        self.local_client = None
        self.progress_bar = None
        self.last_annotation = None  # Store last successful annotation for display
        self.doccano_sync = None  # DoccanoSyncClient, set externally before annotate()

    def annotate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main annotation entry point.

        Parameters
        ----------
        config : dict
            Configuration including:
            - data_source: 'csv', 'excel', 'parquet', 'rdata', 'rds', 'postgresql'
            - file_path or db_config: Data location
            - prompts: List of prompt configurations
            - model: Model configuration
            - num_processes: Number of parallel processes
            - output_path: Where to save results
            - resume: Whether to resume from existing annotations

        Returns
        -------
        dict
            Annotation results and statistics
        """
        # Reset counters to avoid leaking state across multiple runs
        global status_counts
        status_counts = {"success": 0, "error": 0, "cleaning_failed": 0, "decode_error": 0}

        # Wire Doccano sync client if provided in config
        if config.get('doccano_sync_client'):
            self.doccano_sync = config['doccano_sync_client']

        # Validate configuration
        self.logger.info("[ANNOTATOR] Validating config...")
        self._validate_config(config)

        # Setup model client
        self.logger.info(f"[ANNOTATOR] Setting up model client for {config.get('model')}...")
        if self.progress_callback:
            self.progress_callback(0, 100, f"Setting up {config.get('model', 'model')}...")
        self._setup_model_client(config)
        self.logger.info("[ANNOTATOR] Model client setup complete")

        # Load data
        self.logger.info("[ANNOTATOR] Loading data...")
        data, metadata = self._load_data(config)
        self.logger.info(f"[ANNOTATOR] Loaded {len(data)} rows")

        # In resume mode, deduplicate rows that may have been duplicated by
        # previous append-based incremental saves.  Keep the last occurrence
        # so that the most recent annotation is preserved.
        # IMPORTANT: Use (id, text) pairs for dedup because IDs may not be
        # unique -- different sentences can share the same identifier.
        if config.get('resume', False):
            id_col = config.get('identifier_column')
            text_col = config.get('text_column') or (
                config.get('text_columns', [None])[0] if config.get('text_columns') else None
            )
            if id_col and id_col in data.columns:
                dedup_cols = [id_col]
                if text_col and text_col in data.columns:
                    dedup_cols.append(text_col)
                before_dedup = len(data)
                data = data.drop_duplicates(subset=dedup_cols, keep='last').reset_index(drop=True)
                after_dedup = len(data)
                if before_dedup != after_dedup:
                    self.logger.warning(
                        "Resume dedup: removed %d duplicate rows (kept last occurrence per %s). "
                        "%d → %d rows.",
                        before_dedup - after_dedup, dedup_cols, before_dedup, after_dedup
                    )

        # Prepare prompts
        self.logger.info("[ANNOTATOR] Preparing prompts...")
        prompts = self._prepare_prompts(config)
        self.logger.info(f"[ANNOTATOR] Prepared {len(prompts)} prompt(s)")

        # Perform annotation
        self.logger.info("[ANNOTATOR] Starting annotation process...")
        results = self._annotate_data(data, prompts, config)

        # Save results
        self.logger.info("[ANNOTATOR] Saving results...")
        self._save_results(results, config)

        return self._generate_summary(results, config)

    def _validate_config(self, config: Dict[str, Any]):
        """Validate annotation configuration"""
        required = ['data_source', 'model', 'output_path']
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")

    def _setup_model_client(self, config: Dict[str, Any]):
        """Setup the appropriate model client"""
        # Handle both dict and string model config
        if isinstance(config.get('model'), str):
            # Simple string model name - use provider from config
            model_name = config['model']
            provider = config.get('provider', 'ollama')
            api_key = config.get('api_key')
        else:
            # Dict model config
            model_config = config['model']
            provider = model_config.get('provider', 'ollama')
            model_name = model_config.get('model_name')
            api_key = model_config.get('api_key')

        self.logger.info(f"[SETUP] Provider: {provider}, Model: {model_name}")

        if provider in ['openai', 'anthropic', 'google']:
            self.logger.info(f"[SETUP] Creating API client for {provider}...")
            self.api_client = create_api_client(
                provider=provider,
                api_key=api_key,
                model=model_name,
                progress_manager=self.progress_manager  # Pass progress manager for warnings/errors
            )
            self.logger.info(f"[SETUP] API client created successfully")
        elif provider == 'ollama' and HAS_LOCAL_MODELS:
            self.logger.info(f"[SETUP] Creating OllamaClient for {model_name}...")
            self.local_client = OllamaClient(model_name)
            self.logger.info(f"[SETUP] OllamaClient created successfully")
        elif provider == 'llamacpp' and HAS_LOCAL_MODELS:
            self.logger.info(f"[SETUP] Creating LlamaCPPClient for {model_name}...")
            self.local_client = LlamaCPPClient(model_name)  # Assuming path for llamacpp
            self.logger.info(f"[SETUP] LlamaCPPClient created successfully")
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    def _load_data(self, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
        """Load data from various sources"""
        source = config['data_source']
        metadata = {'source': source}

        if source == 'postgresql':
            if not HAS_SQLALCHEMY:
                raise ImportError("SQLAlchemy required for PostgreSQL support")
            return self._load_postgresql(config['db_config']), metadata

        elif source in ['csv', 'excel', 'parquet', 'rdata', 'rds']:
            return self._load_file(config['file_path'], source), metadata

        else:
            raise ValueError(f"Unsupported data source: {source}")

    def _load_file(self, file_path: str, format: str) -> pd.DataFrame:
        """Load data from file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if format == 'csv':
            return pd.read_csv(path)
        elif format == 'excel':
            return pd.read_excel(path)
        elif format == 'parquet':
            return pd.read_parquet(path)
        elif format == 'json':
            return pd.read_json(path, lines=False)
        elif format == 'jsonl':
            return pd.read_json(path, lines=True)
        elif format in ['rdata', 'rds']:
            if not HAS_PYREADR:
                raise ImportError("pyreadr required for RData/RDS files")
            result = pyreadr.read_r(path)
            return list(result.values())[0]
        else:
            raise ValueError(f"Unsupported file format: {format}")

    def _load_postgresql(self, db_config: Dict) -> pd.DataFrame:
        """Load data from PostgreSQL"""
        engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config.get('port', 5432)}/{db_config['database']}"
        )

        query = db_config.get('query') or f"SELECT * FROM {db_config['table']}"
        with engine.connect() as conn:
            return pd.read_sql_query(query, conn)

    def _prepare_prompts(self, config: Dict[str, Any]) -> List[Dict]:
        """Prepare prompts for annotation"""
        prompts_config = config.get('prompts', [])
        if not prompts_config:
            # Load from prompt files if specified
            prompt_dir = config.get('prompt_dir')
            if prompt_dir:
                prompts_config = self.prompt_manager.load_prompts_from_directory(prompt_dir)
            else:
                raise ValueError("No prompts specified")

        return prompts_config

    def _annotate_data(
        self,
        data: pd.DataFrame,
        prompts: List[Dict],
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Perform annotation on the data.
        """
        # Setup columns
        text_columns = config.get('text_columns', [])
        if not text_columns:
            # Auto-detect text columns
            text_columns = self._detect_text_columns(data)

        identifier_column = self._resolve_identifier_column(
            data,
            config.get('identifier_column')
        )
        config['identifier_column'] = identifier_column

        config.setdefault('export_to_doccano', True)
        annotation_column = config.get('annotation_column', 'annotation')
        resume = config.get('resume', False)

        total_loaded = len(data)
        already_annotated_count = 0

        # Filter data for annotation
        filter_logger = get_filter_logger()
        if resume and annotation_column in data.columns:
            data_before_filter = data.copy()
            data_to_annotate = data[data[annotation_column].isna()].copy()
            already_annotated_count = len(data_before_filter) - len(data_to_annotate)

            # Log already-annotated rows that are being skipped
            if already_annotated_count > 0:
                filter_logger.log_dataframe_filtering(
                    df_before=data_before_filter,
                    df_after=data_to_annotate,
                    reason="already_annotated",
                    location="llm_annotator._annotate_data.resume_mode",
                    text_column=text_columns[0] if text_columns else None,
                    log_filtered_samples=3
                )

            self.logger.info(
                "Resume mode: %d/%d rows already annotated, %d remaining",
                already_annotated_count, total_loaded, len(data_to_annotate)
            )
        else:
            data_to_annotate = data.copy()

        # Allow explicit skip lists (e.g. when continuing from checkpoints).
        # Prefer (id, text) pairs for precise matching when IDs aren't unique.
        skip_pairs = config.get('skip_annotation_pairs') or []
        skip_text_col = config.get('skip_annotation_text_column') or (text_columns[0] if text_columns else None)
        skip_ids = config.get('skip_annotation_ids') or []

        if skip_pairs and identifier_column in data_to_annotate.columns and skip_text_col in data_to_annotate.columns:
            _sep = '|||'
            skip_keys = set(str(p[0]) + _sep + str(p[1]) for p in skip_pairs)
            data_before_skip = data_to_annotate.copy()
            before_skip = len(data_before_skip)
            composite = data_before_skip[identifier_column].astype(str) + _sep + data_before_skip[skip_text_col].astype(str)
            keep_mask = ~composite.isin(skip_keys)
            data_to_annotate = data_before_skip[keep_mask].copy()
            skipped = before_skip - len(data_to_annotate)
            if skipped > 0:
                already_annotated_count += skipped
                filter_logger.log_dataframe_filtering(
                    df_before=data_before_skip,
                    df_after=data_to_annotate,
                    reason="resume_skip_pairs",
                    location="llm_annotator._annotate_data.skip_pairs",
                    text_column=text_columns[0] if text_columns else None,
                    log_filtered_samples=3
                )
                self.logger.info("Skipping %s additional row(s) via (id, text) pair skip list", skipped)
        elif skip_ids and identifier_column in data_to_annotate.columns:
            # Fallback to ID-only skip (for backward compatibility)
            skip_ids_set = set(skip_ids)
            data_before_skip = data_to_annotate.copy()
            before_skip = len(data_before_skip)
            data_to_annotate = data_before_skip[
                ~data_before_skip[identifier_column].isin(skip_ids_set)
            ].copy()
            skipped = before_skip - len(data_to_annotate)
            if skipped > 0:
                already_annotated_count += skipped
                filter_logger.log_dataframe_filtering(
                    df_before=data_before_skip,
                    df_after=data_to_annotate,
                    reason="resume_skip_ids",
                    location="llm_annotator._annotate_data.skip_ids",
                    text_column=text_columns[0] if text_columns else None,
                    log_filtered_samples=3
                )
                self.logger.info("Skipping %s additional row(s) via ID-only skip list", skipped)

        # Store resume context so downstream progress reporting can use it
        if resume and already_annotated_count > 0:
            config['_resume_already_annotated'] = already_annotated_count
            config['_resume_total_in_file'] = total_loaded

        annotation_limit = config.get('annotation_sample_size') or config.get('annotation_limit')
        # Ensure annotation_limit is an integer (may come as string from JSON metadata)
        if annotation_limit is not None:
            try:
                annotation_limit = int(annotation_limit)
            except (ValueError, TypeError):
                annotation_limit = None
        if annotation_limit and len(data_to_annotate) > annotation_limit:
            data_before_limit = data_to_annotate.copy()
            strategy = config.get('annotation_sampling_strategy', 'head')
            sample_seed = config.get('annotation_sample_seed', 42)
            if strategy == 'random':
                data_to_annotate = data_to_annotate.sample(annotation_limit, random_state=sample_seed)
            else:
                data_to_annotate = data_to_annotate.head(annotation_limit)

            # Log filtered rows
            filter_logger.log_dataframe_filtering(
                df_before=data_before_limit,
                df_after=data_to_annotate,
                reason=f"annotation_limit_{strategy}",
                location="llm_annotator._annotate_data.sampling",
                text_column=text_columns[0] if text_columns else None,
                log_filtered_samples=3
            )

            self.logger.info(
                "Limiting annotation to %s rows using '%s' sampling strategy",
                len(data_to_annotate),
                strategy
            )

        # Calculate sample size if requested
        if config.get('calculate_sample_size', False):
            sample_size = self.calculate_sample_size(len(data_to_annotate))
            if config.get('use_sample', False):
                data_to_annotate = data_to_annotate.sample(n=sample_size, random_state=42)
                self.logger.info(f"Using sample of {sample_size} rows")

        # Doccano rewrite filter: restrict to matched rows only
        doccano_rewrite_ids = config.get('doccano_rewrite_ids')
        if doccano_rewrite_ids is not None:
            rewrite_set = set(str(x) for x in doccano_rewrite_ids)
            id_col = config.get('identifier_column', 'identifier')
            before = len(data_to_annotate)
            data_to_annotate = data_to_annotate[
                data_to_annotate[id_col].astype(str).isin(rewrite_set)
            ].copy()
            self.logger.info("Doccano rewrite filter: %d -> %d rows", before, len(data_to_annotate))

        # Prepare for parallel processing
        use_parallel = config.get('use_parallel', True)
        num_processes = config.get('num_processes', 1)
        if not use_parallel:
            num_processes = 1
        multiple_prompts = len(prompts) > 1

        # Add necessary columns
        if annotation_column not in data.columns:
            data[annotation_column] = pd.NA
        if f"{annotation_column}_inference_time" not in data.columns:
            data[f"{annotation_column}_inference_time"] = pd.NA

        # Special handling for OpenAI batch mode
        if config.get('annotation_mode') == 'openai_batch':
            return self._execute_openai_batch_annotation(
                full_data=data,
                data_subset=data_to_annotate,
                prompts=prompts,
                text_columns=text_columns,
                identifier_column=identifier_column,
                config=config
            )

        # Warm up model if using local
        if self.local_client and config.get('warmup', True):
            self._warmup_model()

        # Prepare tasks
        tasks = self._prepare_annotation_tasks(
            data_to_annotate,
            prompts,
            text_columns,
            identifier_column,
            config
        )

        # Execute annotation (sequential fallback when only one process requested)
        if num_processes <= 1:
            annotated_data = self._execute_sequential_annotation(
                data,
                tasks,
                annotation_column,
                identifier_column,
                config
            )
        else:
            annotated_data = self._execute_parallel_annotation(
                data,
                tasks,
                num_processes,
                annotation_column,
                identifier_column,
                config
            )

        return annotated_data

    def _detect_text_columns(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect text columns in dataframe"""
        text_columns = []
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if it contains text-like data
                sample = data[col].dropna().head(10)
                if len(sample) > 0:
                    avg_length = sample.astype(str).str.len().mean()
                    if avg_length > 20:  # Likely text content
                        text_columns.append(col)
        return text_columns

    def _create_unique_id(self, data: pd.DataFrame) -> str:
        """Create unique identifier column"""
        id_column = 'llm_annotation_id'
        if id_column not in data.columns:
            data[id_column] = range(1, len(data) + 1)
            self.logger.info(f"Created unique identifier column: {id_column}")
        return id_column

    def _resolve_identifier_column(
        self,
        data: pd.DataFrame,
        identifier_column: Optional[str]
    ) -> str:
        """
        Ensure a usable identifier column exists on the dataframe.

        Handles user-selected composite identifiers of the form "col_a+col_b"
        by creating a new column that joins the component values.
        """
        if not identifier_column:
            return self._create_unique_id(data)

        if identifier_column in data.columns:
            return identifier_column

        if '+' in identifier_column:
            component_cols = [col.strip() for col in identifier_column.split('+') if col.strip()]
            if len(component_cols) <= 1:
                self.logger.warning(
                    "Composite identifier '%s' does not reference multiple columns; "
                    "falling back to auto-generated IDs.",
                    identifier_column,
                )
                return self._create_unique_id(data)

            missing = [col for col in component_cols if col not in data.columns]
            if missing:
                self.logger.warning(
                    "Composite identifier columns missing in data: %s; "
                    "falling back to auto-generated IDs.",
                    ', '.join(missing),
                )
                return self._create_unique_id(data)

            # Build a stable combined identifier using a separator that will not
            # collide with numeric inputs.
            separator = '|'
            placeholder = '__MISSING__'
            combined = data[component_cols[0]].astype(str).where(
                data[component_cols[0]].notna(),
                placeholder,
            )
            for col in component_cols[1:]:
                part = data[col].astype(str).where(
                    data[col].notna(),
                    placeholder,
                )
                combined = combined + separator + part

            data[identifier_column] = combined
            self.logger.info(
                "Created composite identifier column '%s' from [%s]",
                identifier_column,
                ', '.join(component_cols),
            )
            return identifier_column

        self.logger.warning(
            "Identifier column '%s' not found in data; "
            "falling back to auto-generated IDs.",
            identifier_column,
        )
        return self._create_unique_id(data)

    def _warmup_model(self):
        """Warm up local model with test call"""
        try:
            self.logger.info("Warming up model (this may take a few minutes for large models)...")
            if self.progress_callback:
                self.progress_callback(0, 100, "Warming up model (loading into GPU memory)...")

            test_prompt = 'Return a simple JSON: {"test": true}'
            if self.local_client:
                result = self.local_client.generate(test_prompt, timeout=600)  # 10 min timeout for warmup
                if result:
                    self.logger.info("Model warmed up successfully")
                    if self.progress_callback:
                        self.progress_callback(0, 100, "Model ready!")
                else:
                    self.logger.warning("Warm-up returned no result")
        except Exception as e:
            self.logger.warning(f"Warm-up failed: {e}")

    def _prepare_annotation_tasks(
        self,
        data: pd.DataFrame,
        prompts: List[Dict],
        text_columns: List[str],
        identifier_column: str,
        config: Dict[str, Any],
        full_data: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
        """Prepare tasks for parallel annotation

        Parameters
        ----------
        data : pd.DataFrame
            Data to annotate (may be filtered subset)
        prompts : List[Dict]
            List of prompt configurations
        text_columns : List[str]
            Columns containing text to annotate
        identifier_column : str
            Column used as row identifier
        config : Dict[str, Any]
            Annotation configuration
        full_data : pd.DataFrame, optional
            Full dataset for context window (if enabled)
        """
        tasks = []
        context_config = config.get('context_window_config')
        context_enabled = context_config and context_config.get('enabled', False)

        # Use full_data for context, or data if full_data not provided
        context_source = full_data if full_data is not None else data

        for idx, row in data.iterrows():
            # Build model config dict for the task
            if isinstance(config.get('model'), str):
                model_config = {
                    'provider': config.get('provider', 'ollama'),
                    'model_name': config.get('model'),
                    'api_key': config.get('api_key'),
                    'temperature': config.get('temperature', 0.7),
                    'max_tokens': config.get('max_tokens', 1000)
                }
            else:
                model_config = config.get('model', {})

            task = {
                'index': idx,
                'row': row,
                'prompts': prompts,
                'text_columns': text_columns,
                'identifier_column': identifier_column,
                'identifier': row[identifier_column],
                'model_config': model_config,
                'options': config.get('options', {}),
                'context_window_config': context_config if context_enabled else None
            }

            # Pre-compute context for this row if context window is enabled
            if context_enabled and text_columns:
                primary_column = text_columns[0]  # Use first text column for context
                context_before, _, context_after = _build_context_text(
                    data=context_source,
                    current_idx=idx,
                    text_column=primary_column,
                    context_config=context_config
                )
                task['context_before'] = context_before
                task['context_after'] = context_after

            tasks.append(task)

        return tasks

    def _execute_parallel_annotation(
        self,
        full_data: pd.DataFrame,
        tasks: List[Dict],
        num_processes: int,
        annotation_column: str,
        identifier_column: str,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Execute annotation tasks in parallel.
        """
        total_tasks = len(tasks)
        output_path = config.get('output_path')
        save_incrementally = config.get('save_incrementally', True)
        log_enabled = config.get('enable_logging', False)
        log_path = config.get('log_path')
        output_format = config.get('output_format', config.get('data_source', 'csv'))

        # In resume mode, disable CSV append to avoid duplicating rows.
        is_resume = config.get('resume', False)
        use_csv_append = CSV_APPEND and not is_resume

        # Resume context for progress display
        resume_already = config.get('_resume_already_annotated', 0)

        if not tasks:
            message = "[ANNOTATOR] No tasks to process after filtering; returning original dataset."
            if self.progress_manager:
                self.progress_manager.show_warning(message)
            else:
                self.logger.warning(message)
            return full_data

        # Report initial progress - DEBUG with file logging
        import sys
        try:
            with open('/tmp/llmtool_debug.log', 'a') as f:
                f.write(f"[ANNOTATOR] Starting with {total_tasks} tasks, callback={self.progress_callback is not None}\n")
        except:
            pass

        if self.progress_callback:
            try:
                with open('/tmp/llmtool_debug.log', 'a') as f:
                    f.write(f"[ANNOTATOR] Calling progress_callback(0, {total_tasks}, ...)\n")
            except:
                pass
            if resume_already > 0:
                self.progress_callback(
                    0, total_tasks,
                    f"Resuming: {resume_already:,} already done, annotating {total_tasks:,} remaining"
                )
            else:
                self.progress_callback(0, total_tasks, f"Starting annotation of {total_tasks} items")
        else:
            try:
                with open('/tmp/llmtool_debug.log', 'a') as f:
                    f.write(f"[ANNOTATOR] ERROR: No progress_callback!\n")
            except:
                pass

        # Initialize progress bar with position lock to prevent line jumps
        disable_pbar = config.get('disable_tqdm', False)
        pbar_desc = '🤖 LLM Annotation'
        if resume_already > 0:
            pbar_desc = f'🤖 Annotation (resume, {resume_already:,} already done)'

        with tqdm(total=total_tasks, desc=pbar_desc, unit='items',
                  position=0, leave=True, dynamic_ncols=True, disable=disable_pbar) as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Submit all tasks – track (identifier, index) for precise row updates
                if len(tasks[0]['prompts']) > 1:
                    futures = {
                        executor.submit(process_multiple_prompts, task): (task['identifier'], task['index'])
                        for task in tasks
                    }
                else:
                    futures = {
                        executor.submit(process_single_prompt, task): (task['identifier'], task['index'])
                        for task in tasks
                    }

                pending_save = 0
                batch_results = []
                completed_count = 0

                # Process completed tasks
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        identifier = result['identifier']
                        final_json = result['final_json']
                        inference_time = result['inference_time']
                        status = result.get('status', 'unknown')
                        _, row_idx = futures[future]

                        # Update the exact row using its DataFrame index to
                        # avoid overwriting other rows that share the same
                        # identifier but contain different text.
                        if row_idx in full_data.index:
                            full_data.loc[row_idx, annotation_column] = final_json
                            full_data.loc[row_idx, f"{annotation_column}_inference_time"] = inference_time
                        else:
                            mask = full_data[identifier_column] == identifier
                            if mask.any():
                                first_match = mask.idxmax()
                                full_data.loc[first_match, annotation_column] = final_json
                                full_data.loc[first_match, f"{annotation_column}_inference_time"] = inference_time

                        # Track status
                        if final_json:
                            status_counts['success'] += 1
                            batch_results.append((identifier, final_json, inference_time))
                            # Store last successful annotation for display
                            try:
                                self.last_annotation = json.loads(final_json) if isinstance(final_json, str) else final_json
                            except:
                                pass
                        else:
                            status_counts['error'] += 1

                        # Incremental saving
                        if save_incrementally and output_path:
                            if output_format == 'csv' and use_csv_append:
                                self._append_to_csv(full_data, identifier, identifier_column, annotation_column, output_path, row_index=row_idx)
                            else:
                                pending_save += 1
                                if pending_save >= OTHER_FORMAT_SAVE_EVERY:
                                    self._save_data(full_data, output_path, output_format)
                                    pending_save = 0

                        # Log if enabled
                        if log_enabled and log_path:
                            self._write_log_entry(
                                log_path,
                                {
                                    'id': identifier,
                                    'final_json': final_json,
                                    'inference_time': inference_time,
                                    'status': status
                                }
                            )

                        # Display sample results only if progress bar is enabled
                        if not disable_pbar and len(batch_results) >= 10:
                            sample = random.choice(batch_results)
                            tqdm.write(f"✨ Sample annotation for ID {sample[0]}: {sample[1][:100]}...")
                            batch_results = []

                        pbar.update(1)

                        # Report progress via callback if available
                        completed_count += 1
                        if self.progress_callback:
                            if resume_already > 0:
                                self.progress_callback(completed_count, total_tasks,
                                    f"Annotated {completed_count}/{total_tasks} (+ {resume_already:,} previously done)")
                            else:
                                self.progress_callback(completed_count, total_tasks,
                                    f"Annotated {completed_count}/{total_tasks} items")

                    except Exception as e:
                        import traceback
                        self.logger.error(f"Task failed: {e}")
                        self.logger.debug(f"Traceback: {traceback.format_exc()}")
                        status_counts['error'] += 1
                        pbar.update(1)

                        # Report error progress too
                        completed_count += 1
                        if self.progress_callback:
                            self.progress_callback(completed_count, total_tasks,
                                f"Annotated {completed_count}/{total_tasks} items ({status_counts['error']} errors)")

        # Final save if needed
        if save_incrementally and output_path and pending_save > 0:
            self._save_data(full_data, output_path, output_format)

        # Report final progress
        if self.progress_callback:
            self.progress_callback(total_tasks, total_tasks, f"Completed annotation of {total_tasks} items")

        return full_data

    def _execute_sequential_annotation(
        self,
        full_data: pd.DataFrame,
        tasks: List[Dict],
        annotation_column: str,
        identifier_column: str,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Execute annotation tasks sequentially (no process pool)."""
        output_path = config.get('output_path')
        save_incrementally = config.get('save_incrementally', True)
        log_enabled = config.get('enable_logging', False)
        log_path = config.get('log_path')
        output_format = config.get('output_format', config.get('data_source', 'csv'))

        # In resume mode, disable CSV append to avoid duplicating rows.
        # full_data already contains the complete dataset (previously annotated
        # + new rows), so a full overwrite is the correct strategy.
        is_resume = config.get('resume', False)
        use_csv_append = CSV_APPEND and not is_resume

        pending_save = 0
        total_tasks = len(tasks)
        completed_count = 0

        # Resume context for progress display
        resume_already = config.get('_resume_already_annotated', 0)
        resume_total = config.get('_resume_total_in_file', 0)

        # Report initial progress
        if self.progress_callback:
            try:
                with open('/tmp/llmtool_debug.log', 'a') as f:
                    f.write(f"[ANNOTATOR SEQUENTIAL] Starting with {total_tasks} tasks, callback={self.progress_callback is not None}\n")
                    f.flush()
            except:
                pass
            if resume_already > 0:
                self.progress_callback(
                    0, total_tasks,
                    f"Resuming: {resume_already:,} already done, annotating {total_tasks:,} remaining"
                )
            else:
                self.progress_callback(0, total_tasks, f"Starting annotation of {total_tasks} items")

        disable_pbar = config.get('disable_tqdm', False)

        # Retry configuration for failed annotations
        max_retries_per_row = config.get('max_retries_per_row', 3)
        retry_delay = config.get('retry_delay', 2.0)  # Seconds between retries

        # Stagnation detection: track consecutive errors across rows
        consecutive_row_failures = 0
        max_consecutive_failures = config.get('max_consecutive_errors', 10)
        stagnation_threshold = config.get('stagnation_threshold', 5)  # Warn after N consecutive row failures

        # Queue for failed rows to retry at the end
        failed_tasks_queue = []
        max_final_retry_rounds = config.get('max_final_retry_rounds', 3)  # How many times to retry the failed queue

        pbar_desc = '🤖 LLM Annotation'
        if resume_already > 0:
            pbar_desc = f'🤖 Annotation (resume, {resume_already:,} already done)'

        for task in tqdm(tasks, desc=pbar_desc, unit='items',
                         position=0, leave=True, dynamic_ncols=True, disable=disable_pbar,
                         initial=0):

            # Retry loop for each row
            final_json = None
            inference_time = 0
            status = 'error'
            row_attempts = 0

            while row_attempts < max_retries_per_row:
                row_attempts += 1

                if len(task['prompts']) > 1:
                    result = process_multiple_prompts(task)
                else:
                    result = process_single_prompt(task)

                identifier = result['identifier']
                final_json = result['final_json']
                inference_time = result['inference_time']
                raw_json = result.get('raw_json')
                cleaned_json = result.get('cleaned_json')
                status = result.get('status', 'unknown')

                # If we got a valid result, break out of retry loop
                if final_json:
                    if row_attempts > 1:
                        self.logger.info(f"[RETRY SUCCESS] Row {identifier} succeeded on attempt {row_attempts}/{max_retries_per_row}")
                    break

                # Failed attempt - log and potentially retry
                if row_attempts < max_retries_per_row:
                    self.logger.warning(f"[RETRY] Row {identifier} failed (attempt {row_attempts}/{max_retries_per_row}). Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    # Exponential backoff
                    retry_delay_current = retry_delay * row_attempts
                    time.sleep(retry_delay_current)
                else:
                    # Add to failed queue for later retry instead of giving up
                    self.logger.warning(f"[QUEUED] Row {identifier} failed after {max_retries_per_row} attempts. Added to retry queue.")
                    failed_tasks_queue.append(task)

            # Update dataframe with result (success or final failure)
            # Use the exact DataFrame index to avoid updating/duplicating rows
            # that share the same identifier but have different text.
            row_idx = task['index']
            if row_idx in full_data.index:
                full_data.loc[row_idx, annotation_column] = final_json
                full_data.loc[row_idx, f"{annotation_column}_inference_time"] = inference_time
            else:
                # Fallback: match by identifier (should not happen)
                mask = full_data[identifier_column] == identifier
                if mask.any():
                    first_match = mask.idxmax()
                    full_data.loc[first_match, annotation_column] = final_json
                    full_data.loc[first_match, f"{annotation_column}_inference_time"] = inference_time

            if final_json:
                status_counts['success'] += 1
                consecutive_row_failures = 0  # Reset error counter on success
                # Store last successful annotation for display
                try:
                    self.last_annotation = json.loads(final_json) if isinstance(final_json, str) else final_json
                except:
                    pass

                # Live sync to Doccano
                if self.doccano_sync:
                    try:
                        text_cols = task.get('text_columns', [])
                        row = task.get('row')
                        sync_text = str(row[text_cols[0]]) if text_cols and row is not None else str(identifier)
                        self.doccano_sync.push(
                            text=sync_text,
                            annotation=self.last_annotation or {},
                            meta={"identifier": str(identifier), "inference_time": inference_time}
                        )
                    except Exception as sync_err:
                        self.logger.debug(f"Doccano sync error (queued): {sync_err}")
            else:
                status_counts['error'] += 1
                consecutive_row_failures += 1

                # Warn about potential stagnation
                if consecutive_row_failures == stagnation_threshold:
                    warning_msg = f"[STAGNATION WARNING] {consecutive_row_failures} consecutive row failures detected. Model may be unresponsive."
                    if self.progress_manager:
                        self.progress_manager.show_warning(warning_msg)
                    else:
                        self.logger.warning(warning_msg)

                # Attempt recovery if too many consecutive failures
                if consecutive_row_failures >= max_consecutive_failures:
                    error_msg = f"[STAGNATION RECOVERY] {consecutive_row_failures} consecutive row failures. Checking Ollama status..."
                    if self.progress_manager:
                        self.progress_manager.show_warning(error_msg)
                    else:
                        self.logger.warning(error_msg)

                    # Try to check/restart the local client connection
                    try:
                        if self.local_client:
                            # Re-check Ollama service
                            import subprocess
                            check_result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
                            if check_result.returncode != 0:
                                self.logger.error("Ollama service appears to be down. Please restart Ollama.")
                            else:
                                self.logger.info("Ollama service is running. Resetting consecutive failure counter.")
                            consecutive_row_failures = 0  # Reset to avoid infinite loop
                    except Exception as restart_error:
                        self.logger.error(f"Failed to check Ollama status: {restart_error}")

            if save_incrementally and output_path:
                if output_format == 'csv' and use_csv_append:
                    self._append_to_csv(full_data, identifier, identifier_column, annotation_column, output_path, row_index=row_idx)
                else:
                    pending_save += 1
                    if pending_save >= OTHER_FORMAT_SAVE_EVERY:
                        self._save_data(full_data, output_path, output_format)
                        pending_save = 0

            if log_enabled and log_path:
                self._write_log_entry(
                    log_path,
                    {
                        'id': identifier,
                        'final_json': final_json,
                        'inference_time': inference_time,
                        'status': status
                    }
                )

            # Report progress via callback if available
            completed_count += 1
            if self.progress_callback:
                try:
                    with open('/tmp/llmtool_debug.log', 'a') as f:
                        f.write(f"[ANNOTATOR SEQUENTIAL] Progress: {completed_count}/{total_tasks}\n")
                        f.flush()
                except:
                    pass
                if resume_already > 0:
                    self.progress_callback(completed_count, total_tasks,
                        f"Annotated {completed_count}/{total_tasks} (+ {resume_already:,} previously done)")
                else:
                    self.progress_callback(completed_count, total_tasks,
                        f"Annotated {completed_count}/{total_tasks} items")

        if save_incrementally and output_path and (output_format != 'csv' or is_resume) and pending_save > 0:
            self._save_data(full_data, output_path, output_format)
        elif not save_incrementally and output_path:
            self._save_data(full_data, output_path, output_format)

        # ============================================================
        # RETRY FAILED TASKS - Process the failed queue
        # ============================================================
        if failed_tasks_queue:
            self.logger.warning(f"[RETRY QUEUE] {len(failed_tasks_queue)} rows failed. Starting retry rounds...")

            if self.progress_callback:
                self.progress_callback(completed_count, total_tasks,
                    f"Retrying {len(failed_tasks_queue)} failed rows...")

            for retry_round in range(max_final_retry_rounds):
                if not failed_tasks_queue:
                    break

                self.logger.info(f"[RETRY QUEUE] Round {retry_round + 1}/{max_final_retry_rounds}: {len(failed_tasks_queue)} rows to retry")

                # Perform ACTIVE health check and hard reset before retry round
                if self.local_client and hasattr(self.local_client, 'health_check'):
                    self.logger.info("[RETRY QUEUE] Performing ACTIVE health check before retry round...")
                    try:
                        # Use active_test=True to actually verify model responds
                        health = self.local_client.health_check(active_test=True)

                        if health.get('stuck_models'):
                            self.logger.warning(f"[RETRY QUEUE] Found stuck models: {health['stuck_models']}")
                            self.local_client._check_and_recover_stuck_models(max_wait=60)

                        elif health.get('model_loaded') and health.get('responds_to_requests') is False:
                            # Model loaded but unresponsive - hard reset needed
                            self.logger.warning("[RETRY QUEUE] Model loaded but unresponsive - performing HARD RESET...")
                            if hasattr(self.local_client, '_hard_reset_model'):
                                self.local_client._hard_reset_model()
                            else:
                                # Fallback: just wait and hope
                                self.logger.warning("[RETRY QUEUE] Hard reset not available, waiting 30s...")
                                time.sleep(30)

                        elif health.get('healthy') and health.get('responds_to_requests'):
                            self.logger.info("[RETRY QUEUE] Model is healthy and responsive")

                        else:
                            self.logger.warning(f"[RETRY QUEUE] Health check result: {health}")

                    except Exception as e:
                        self.logger.warning(f"[RETRY QUEUE] Health check failed: {e}")

                    # Small delay to let Ollama stabilize after any recovery
                    time.sleep(5)

                still_failed = []

                for task in failed_tasks_queue:
                    # Try each failed task again
                    if len(task['prompts']) > 1:
                        result = process_multiple_prompts(task)
                    else:
                        result = process_single_prompt(task)

                    identifier = result['identifier']
                    final_json = result['final_json']
                    inference_time = result['inference_time']
                    status = result.get('status', 'unknown')

                    retry_row_idx = task['index']

                    if final_json:
                        # Success! Update dataframe using exact row index
                        self.logger.info(f"[RETRY SUCCESS] Row {identifier} succeeded on retry round {retry_round + 1}")
                        if retry_row_idx in full_data.index:
                            full_data.loc[retry_row_idx, annotation_column] = final_json
                            full_data.loc[retry_row_idx, f"{annotation_column}_inference_time"] = inference_time
                        else:
                            mask = full_data[identifier_column] == identifier
                            if mask.any():
                                first_match = mask.idxmax()
                                full_data.loc[first_match, annotation_column] = final_json
                                full_data.loc[first_match, f"{annotation_column}_inference_time"] = inference_time

                        status_counts['success'] += 1
                        status_counts['error'] -= 1  # Correct the earlier error count

                        # Save incrementally
                        if save_incrementally and output_path:
                            if output_format == 'csv' and use_csv_append:
                                self._append_to_csv(full_data, identifier, identifier_column, annotation_column, output_path, row_index=retry_row_idx)
                            else:
                                self._save_data(full_data, output_path, output_format)

                        try:
                            self.last_annotation = json.loads(final_json) if isinstance(final_json, str) else final_json
                        except:
                            pass
                    else:
                        # Still failed
                        still_failed.append(task)
                        self.logger.warning(f"[RETRY FAILED] Row {identifier} still failed on round {retry_round + 1}")

                failed_tasks_queue = still_failed

                if failed_tasks_queue:
                    self.logger.warning(f"[RETRY QUEUE] {len(failed_tasks_queue)} rows still failing after round {retry_round + 1}")
                    # Longer delay between retry rounds
                    time.sleep(10)
                else:
                    self.logger.info("[RETRY QUEUE] All failed rows recovered successfully!")

            # Final report on any remaining failures
            if failed_tasks_queue:
                failed_ids = [task.get('identifier', 'unknown') for task in failed_tasks_queue]
                self.logger.error(f"[FINAL FAILURE] {len(failed_tasks_queue)} rows could not be annotated after all retries: {failed_ids[:10]}...")
                if self.progress_manager:
                    self.progress_manager.show_warning(
                        f"{len(failed_tasks_queue)} rows failed permanently. Check logs for details."
                    )

        # Report final progress
        if self.progress_callback:
            try:
                with open('/tmp/llmtool_debug.log', 'a') as f:
                    f.write(f"[ANNOTATOR SEQUENTIAL] Completed all {total_tasks} tasks\n")
                    f.flush()
            except:
                pass
            self.progress_callback(total_tasks, total_tasks, f"Completed annotation of {total_tasks} items")

        # Finalize Doccano sync
        if self.doccano_sync:
            sync_stats = self.doccano_sync.stop()
            if sync_stats.get('remaining_queue', 0) > 0:
                self.logger.warning(f"Doccano sync: {sync_stats['remaining_queue']} items remain in offline queue")
            else:
                self.logger.info(f"Doccano sync: all {sync_stats['pushed']} items pushed successfully")

        return full_data

    def _execute_openai_batch_annotation(
        self,
        full_data: pd.DataFrame,
        data_subset: pd.DataFrame,
        prompts: List[Dict],
        text_columns: List[str],
        identifier_column: str,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Execute annotation using the OpenAI Batch API."""
        start_time = time.perf_counter()
        global status_counts
        status_counts = {"success": 0, "error": 0, "cleaning_failed": 0, "decode_error": 0}

        if not HAS_OPENAI:
            raise ImportError("OpenAI SDK is required for batch mode. Install the 'openai' package.")

        api_key = config.get('api_key')
        if not api_key:
            raise ValueError("OpenAI API key is required for batch mode.")

        model_name = config.get('model') or config.get('annotation_model')
        if not model_name:
            raise ValueError("Annotation model must be specified for OpenAI batch mode.")

        config.setdefault('model_display_name', config.get('model_display_name') or model_name)

        total_rows = len(data_subset)
        prompt_count = len(prompts)
        if total_rows == 0 or prompt_count == 0:
            self.logger.info("[BATCH] No rows or prompts supplied; skipping batch annotation.")
            return full_data

        annotation_column = config.get('annotation_column', 'annotation')
        output_path = config.get('output_path')
        output_format = config.get('output_format', config.get('data_source', 'csv'))
        save_incrementally = config.get('save_incrementally', True)
        log_enabled = config.get('enable_logging', False)
        log_path = config.get('log_path')

        batch_base = config.get('openai_batch_dir')
        if batch_base:
            batch_dir = Path(batch_base)
        else:
            batch_dir = self.settings.paths.logs_dir / "openai_batches"

        input_dir = batch_dir / "inputs"
        output_dir = batch_dir / "outputs"
        for directory in (batch_dir, input_dir, output_dir):
            directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        input_file_path = input_dir / f"openai_batch_input_{timestamp}.jsonl"
        output_file_path = output_dir / f"openai_batch_output_{timestamp}.jsonl"
        metadata_file_path = batch_dir / f"openai_batch_metadata_{timestamp}.json"

        expected_total_requests = total_rows * prompt_count
        self.logger.info("[BATCH] Preparing %s requests for OpenAI batch job...", expected_total_requests)

        request_entries: List[Tuple[Dict[str, Any], int]] = []
        request_metadata: Dict[str, Dict[str, Any]] = {}
        per_row_custom_ids: Dict[str, List[str]] = defaultdict(list)
        prompts_by_index: Dict[int, Dict[str, Any]] = {
            idx + 1: prompt_cfg for idx, prompt_cfg in enumerate(prompts)
        }
        prefixed_expected_keys: Dict[int, List[str]] = {}
        all_expected_keys: Set[str] = set()
        for prompt_idx, prompt_cfg in prompts_by_index.items():
            expected_keys = prompt_cfg.get('expected_keys') or []
            prefix = prompt_cfg.get('prefix', '') or ''
            normalized_keys: List[str] = []
            for key in expected_keys:
                if not key:
                    continue
                normalized_keys.append(f"{prefix}_{key}" if prefix else key)
            prefixed_expected_keys[prompt_idx] = normalized_keys
            all_expected_keys.update(normalized_keys)
        row_lookup: Dict[str, Dict[str, Any]] = {}
        row_results: Dict[str, Dict[str, Any]] = {}

        usage_keys = (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        )

        def aggregate_usage(usage_values: Iterable[Dict[str, Any]]) -> Dict[str, float]:
            totals = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_prompt_tokens": 0,
                "reasoning_tokens": 0,
            }
            for usage in usage_values:
                if not usage:
                    continue
                for key in usage_keys:
                    totals[key] += usage.get(key) or 0
                prompt_details = usage.get("prompt_tokens_details") or {}
                totals["cached_prompt_tokens"] += prompt_details.get("cached_tokens") or 0
                completion_details = usage.get("completion_tokens_details") or {}
                totals["reasoning_tokens"] += completion_details.get("reasoning_tokens") or 0
            return totals

        def compose_text(row: pd.Series) -> str:
            segments: List[str] = []
            for column in text_columns:
                value = row.get(column)
                if pd.notna(value):
                    segments.append(str(value))
            return "\n\n".join(segments).strip()

        model_lower = str(model_name).lower()

        def build_request_body(prompt_text: str) -> Dict[str, Any]:
            body: Dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt_text}],
            }
            is_o_series = (
                model_lower == 'o1'
                or model_lower.startswith('o1-')
                or model_lower.startswith('o3-')
                or model_lower.startswith('o4-')
            )
            is_2025_model = any(token in model_lower for token in ['2025', 'gpt-5', 'gpt5']) and not model_lower.startswith('gpt-4.1')
            default_max_tokens = 2000 if (is_o_series or is_2025_model) else 1000
            max_tokens = config.get('max_tokens', default_max_tokens)
            if is_o_series or is_2025_model:
                body["max_completion_tokens"] = max_tokens
                body["temperature"] = 1.0
                body["top_p"] = 1.0
                if config.get('temperature') not in (None, 1, 1.0):
                    self.logger.debug(
                        "[BATCH] Overriding temperature to 1.0 for model %s (batch API restriction).",
                        model_name
                    )
            else:
                body["max_tokens"] = max_tokens
                body["temperature"] = config.get('temperature', 0.7)
                body["top_p"] = config.get('top_p', 1.0)
            return body

        def retry_failed_prompts(
            identifier_key: str,
            missing_indices: List[int],
            row_state: Optional[Dict[str, Any]]
        ) -> None:
            """Sequentially retry prompts that failed in the batch job."""
            if not missing_indices:
                return

            row_dict = row_lookup.get(identifier_key)
            if not row_dict:
                self.logger.warning("[BATCH] Unable to retry %s – original row data missing.", identifier_key)
                return

            local_row_state = row_state
            if local_row_state is None:
                local_row_state = {
                    'identifier': row_dict.get(identifier_column),
                    'row_index': None,
                    'raw': {},
                    'cleaned': {},
                    'status': {},
                    'merged': {},
                    'errors': [],
                    'usage': {},
                    'response_info': {}
                }
                row_results[identifier_key] = local_row_state

            retry_max_tokens = config.get(
                'openai_batch_retry_max_tokens',
                max(config.get('max_tokens', 1000) * 2, 2048)
            )
            retry_model_config = {
                'provider': config.get('annotation_provider') or config.get('provider', 'openai'),
                'model_name': model_name,
                'api_key': api_key,
                'temperature': config.get('temperature', 0.7),
                'top_p': config.get('top_p', 1.0),
                'max_tokens': retry_max_tokens
            }

            row_series = pd.Series(row_dict)
            text_payload = compose_text(row_series)

            for prompt_index in missing_indices:
                prompt_cfg = prompts_by_index.get(prompt_index)
                if not prompt_cfg:
                    continue

                prompt_template = (prompt_cfg.get('prompt') or '').strip()
                if not prompt_template:
                    self.logger.warning("[BATCH] Retry skipped – prompt template missing for index %s.", prompt_index)
                    continue

                expected_keys = prompt_cfg.get('expected_keys', []) or []
                prefix = prompt_cfg.get('prefix', '') or ''
                schema = None
                if expected_keys and not config.get('disable_schema', False):
                    try:
                        schema = build_dynamic_schema(expected_keys)
                    except Exception as exc:
                        self.logger.debug("[BATCH] Schema build failed for retry prompt %s: %s", prompt_index, exc)

                local_retry_config = dict(retry_model_config)
                prompt_key = str(prompt_index)
                if 'response_info' not in local_row_state or not isinstance(local_row_state['response_info'], dict):
                    local_row_state['response_info'] = {}
                response_info_map = local_row_state['response_info']
                previous_info = response_info_map.get(prompt_key) or {}
                if previous_info.get('finish_reason') == 'length':
                    length_retry_cap = config.get(
                        'openai_batch_length_retry_tokens',
                        max(retry_max_tokens * 2, 4096)
                    )
                    local_retry_config['max_tokens'] = max(
                        local_retry_config.get('max_tokens', retry_max_tokens),
                        length_retry_cap
                    )
                try:
                    cleaned_json = self.analyze_text_with_model(
                        text=text_payload,
                        prompt=prompt_template,
                        model_config=local_retry_config,
                        schema=schema
                    )
                except Exception as exc:
                    self.logger.warning("[BATCH] Retry failed for %s prompt %s: %s", identifier_key, prompt_index, exc)
                    local_row_state['status'][str(prompt_index)] = 'retry_failed'
                    local_row_state['errors'].append(f"Sequential retry failed for prompt {prompt_index}: {exc}")
                    continue

                previous_status = local_row_state['status'].get(prompt_key)

                if cleaned_json:
                    local_row_state['raw'][prompt_key] = cleaned_json
                    local_row_state['cleaned'][prompt_key] = cleaned_json
                    local_row_state['status'][prompt_key] = 'success'
                    response_info_map[prompt_key] = {
                        'status_code': None,
                        'finish_reason': 'retry_success',
                        'request_id': None,
                        'sequential_retry': True,
                    }
                    try:
                        parsed = json.loads(cleaned_json)
                        if prefix:
                            parsed = {f"{prefix}_{k}": v for k, v in parsed.items()}
                        local_row_state.setdefault('merged', {}).update(parsed)
                        self.last_annotation = parsed
                    except Exception as exc:
                        local_row_state['status'][prompt_key] = 'decode_error'
                        local_row_state['errors'].append(f"Decode error after retry (prompt {prompt_index}): {exc}")
                        status_counts['decode_error'] += 1
                        continue

                    if previous_status == 'parse_error':
                        status_counts['cleaning_failed'] = max(0, status_counts['cleaning_failed'] - 1)
                    elif previous_status == 'decode_error':
                        status_counts['decode_error'] = max(0, status_counts['decode_error'] - 1)
                    elif previous_status == 'error':
                        status_counts['error'] = max(0, status_counts['error'] - 1)

                    status_counts['success'] += 1
                    local_row_state['usage'][prompt_key] = {'fallback': True}
                    info = local_row_state['response_info'].setdefault(prompt_key, {})
                    max_used_tokens = local_retry_config.get('max_tokens')
                    info['fallback'] = True
                    info['max_tokens'] = max_used_tokens
                    info['fallback_max_tokens'] = max_used_tokens
                    if local_row_state['errors']:
                        local_row_state['errors'] = [
                            err for err in local_row_state['errors']
                            if f"prompt {prompt_index}" not in err and "Unable to parse model response" not in err
                        ]
                else:
                    local_row_state['status'][prompt_key] = 'retry_failed'
                    local_row_state['errors'].append(f"Sequential retry returned empty response for prompt {prompt_index}")

        # Context window configuration for batch
        context_config = config.get('context_window_config')
        context_enabled = context_config and context_config.get('enabled', False)
        batch_labels = None
        if context_enabled:
            batch_labels = context_config.get('labels') or DEFAULT_CONTEXT_LABELS.copy()

        for row_index, row in data_subset.iterrows():
            identifier_value = row[identifier_column]
            identifier_key = str(identifier_value)
            current_text = compose_text(row)

            # Apply context window if enabled
            if context_enabled and text_columns:
                primary_column = text_columns[0]
                context_before, _, context_after = _build_context_text(
                    data=full_data,
                    current_idx=row_index,
                    text_column=primary_column,
                    context_config=context_config
                )
                text_payload = _format_context_prompt(
                    context_before=context_before,
                    current_text=current_text,
                    context_after=context_after,
                    labels=batch_labels
                )
            else:
                text_payload = current_text

            if identifier_key not in row_lookup:
                row_dict = row.to_dict()
                row_dict['row_index'] = row_index
                row_lookup[identifier_key] = row_dict

            for prompt_idx, prompt_cfg in enumerate(prompts, 1):
                prompt_payload = prompt_cfg.get('prompt')
                expected_keys = prompt_cfg.get('expected_keys', [])
                prefix = prompt_cfg.get('prefix', '')
                prompt_name = prompt_cfg.get('name')

                prompt_template = ""
                if isinstance(prompt_payload, dict):
                    prompt_template = (
                        prompt_payload.get('content')
                        or prompt_payload.get('template')
                        or prompt_payload.get('prompt')
                        or ''
                    )
                    if not expected_keys:
                        keys_candidate = prompt_payload.get('keys')
                        if isinstance(keys_candidate, list):
                            expected_keys = keys_candidate
                    if not prompt_name:
                        prompt_name = prompt_payload.get('name')
                elif prompt_payload is not None:
                    prompt_template = str(prompt_payload)

                prompt_template = (prompt_template or '').strip()
                if not prompt_template:
                    self.logger.warning(
                        "[BATCH] Prompt %s is empty; skipping row %s.",
                        prompt_idx,
                        identifier_key
                    )
                    continue

                if not isinstance(expected_keys, list):
                    if expected_keys is None:
                        expected_keys = []
                    elif isinstance(expected_keys, (tuple, set)):
                        expected_keys = list(expected_keys)
                    else:
                        expected_keys = [expected_keys]

                prompt_display_name = prompt_name or f"prompt_{prompt_idx}"

                if text_payload:
                    if context_enabled:
                        # Context is already formatted with labels
                        full_prompt = f"{prompt_template}\n\n{text_payload}"
                    else:
                        full_prompt = f"{prompt_template}\n\nText to analyze:\n{text_payload}"
                else:
                    full_prompt = prompt_template

                sanitized_identifier = identifier_key.replace("\n", " ").replace("\r", " ")
                custom_id = f"{sanitized_identifier}|p{prompt_idx}|{row_index}"

                request_body = build_request_body(full_prompt)
                completion_limit = int(
                    request_body.get("max_completion_tokens")
                    or request_body.get("max_tokens")
                    or config.get("max_tokens")
                    or 0
                )

                request_entries.append((
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": request_body,
                    },
                    len(full_prompt),
                    completion_limit,
                ))

                request_metadata[custom_id] = {
                    "identifier": identifier_value,
                    "identifier_key": identifier_key,
                    "prompt_index": prompt_idx,
                    "prefix": prefix,
                    "expected_keys": expected_keys,
                    "row_index": row_index,
                    "prompt_name": prompt_display_name
                }
                per_row_custom_ids[identifier_key].append(custom_id)

        max_file_bytes = int(config.get('openai_batch_max_file_bytes', OPENAI_BATCH_MAX_FILE_BYTES))
        max_input_tokens = int(config.get('openai_batch_max_input_tokens', OPENAI_BATCH_MAX_INPUT_TOKENS))
        approx_chars_per_token = float(config.get('openai_batch_chars_per_token', OPENAI_BATCH_APPROX_CHARS_PER_TOKEN))
        approx_token_overhead = int(config.get('openai_batch_token_overhead', OPENAI_BATCH_APPROX_TOKEN_OVERHEAD))

        chunk_dir = batch_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_infos: List[Dict[str, Any]] = []
        chunk_index = 0
        current_chunk_index = 0
        chunk_path: Optional[Path] = None
        chunk_handle = None
        chunk_size_bytes = 0
        chunk_tokens = 0
        chunk_request_count = 0

        def _close_chunk() -> None:
            nonlocal chunk_handle, chunk_path, chunk_size_bytes, chunk_tokens, chunk_request_count, current_chunk_index, chunk_infos
            if chunk_handle and chunk_path:
                chunk_handle.close()
                chunk_infos.append(
                    {
                        "index": current_chunk_index,
                        "path": chunk_path,
                        "bytes": chunk_size_bytes,
                        "tokens": chunk_tokens,
                        "requests": chunk_request_count,
                    }
                )
            chunk_handle = None
            chunk_path = None

        def _open_new_chunk() -> None:
            nonlocal chunk_index, current_chunk_index, chunk_handle, chunk_path, chunk_size_bytes, chunk_tokens, chunk_request_count
            _close_chunk()
            chunk_index += 1
            current_chunk_index = chunk_index
            chunk_path = chunk_dir / f"openai_batch_input_{timestamp}_chunk{chunk_index:03d}.jsonl"
            chunk_handle = chunk_path.open('w', encoding='utf-8')
            chunk_size_bytes = 0
            chunk_tokens = 0
            chunk_request_count = 0

        completion_overhead = int(config.get('openai_batch_completion_overhead', OPENAI_BATCH_COMPLETION_OVERHEAD))

        for entry_dict, prompt_length, completion_limit in request_entries:
            entry_json = json.dumps(entry_dict, ensure_ascii=False)
            entry_bytes = entry_json.encode('utf-8')
            entry_size = len(entry_bytes) + 1
            approx_tokens = max(1, math.ceil(prompt_length / approx_chars_per_token) + approx_token_overhead)
            completion_allowance = max(0, int(completion_limit or 0))
            approx_tokens += completion_allowance + completion_overhead

            if chunk_handle is None:
                _open_new_chunk()
            elif chunk_request_count > 0 and (
                chunk_size_bytes + entry_size > max_file_bytes
                or chunk_tokens + approx_tokens > max_input_tokens
            ):
                _open_new_chunk()

            if chunk_request_count == 0 and (entry_size > max_file_bytes or approx_tokens > max_input_tokens):
                self.logger.warning(
                    "[BATCH] Single request (custom_id=%s) exceeds configured OpenAI batch limits "
                    "(size=%s bytes, tokens≈%s). It will be submitted alone.",
                    entry_dict.get("custom_id"),
                    entry_size,
                    approx_tokens,
                )

            if chunk_handle is None:
                _open_new_chunk()

            chunk_handle.write(entry_json)
            chunk_handle.write('\n')
            chunk_size_bytes += entry_size
            chunk_tokens += approx_tokens
            chunk_request_count += 1

        _close_chunk()

        if not chunk_infos:
            raise RuntimeError("OpenAI batch preparation failed: no requests were generated.")

        config['openai_batch_input_paths'] = [str(info['path']) for info in chunk_infos]
        config['openai_batch_chunk_count'] = len(chunk_infos)

        total_requests = sum(info['requests'] for info in chunk_infos)
        if total_requests != expected_total_requests:
            self.logger.debug(
                "[BATCH] Request aggregation mismatch: calculated=%s, expected=%s",
                total_requests,
                expected_total_requests,
            )

        if len(chunk_infos) > 1:
            self.logger.info(
                "[BATCH] Split %s request(s) into %s chunk files (≤%s MB, ≤%s tokens per chunk).",
                total_requests,
                len(chunk_infos),
                int(max_file_bytes / (1024 ** 2)),
                max_input_tokens,
            )

        completion_window = config.get('openai_batch_completion_window', '24h')

        def _submit_chunk(chunk_info: Dict[str, Any]) -> Dict[str, Any]:
            chunk_label = f"chunk {chunk_info['index']:03d}"
            local_client = OpenAI(api_key=api_key)
            with chunk_info['path'].open('rb') as handle:
                uploaded_file = local_client.files.create(file=handle, purpose='batch')
            batch_job = local_client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window=completion_window,
            )
            self.logger.info(
                "[BATCH] Submitted %s as job %s (status: %s).",
                chunk_label,
                batch_job.id,
                batch_job.status,
            )
            return {
                "chunk": chunk_info,
                "job_id": batch_job.id,
                "input_file_id": uploaded_file.id,
                "status": batch_job.status,
                "request_counts": _as_request_counts(getattr(batch_job, 'request_counts', None)),
            }

        if len(chunk_infos) == 1:
            submission_results = [_submit_chunk(chunk_infos[0])]
        else:
            parallel_jobs = int(config.get('openai_batch_parallel_jobs', min(len(chunk_infos), 4)))
            parallel_jobs = max(1, parallel_jobs)
            submission_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
                futures = {executor.submit(_submit_chunk, info): info for info in chunk_infos}
                for future in concurrent.futures.as_completed(futures):
                    submission_results.append(future.result())

        submission_results.sort(key=lambda item: item['chunk']['index'])
        config['openai_batch_job_ids'] = [entry['job_id'] for entry in submission_results]

        client = OpenAI(api_key=api_key)
        poll_interval = max(2, int(config.get('openai_batch_poll_interval', 5)))
        ongoing_statuses = {'validating', 'queued', 'in_progress', 'processing', 'finalizing'}

        disable_chunk_pbars = config.get('disable_tqdm', False)
        chunk_progress: Dict[str, Dict[str, Any]] = {}
        if not disable_chunk_pbars and submission_results:
            for idx, job_info in enumerate(submission_results, start=1):
                total_requests_for_chunk = max(1, job_info['chunk']['requests'])
                bar = tqdm(
                    total=total_requests_for_chunk,
                    desc=f"Chunk {job_info['chunk']['index']:03d} submitted",
                    unit="req",
                    position=idx,
                    leave=True,
                    dynamic_ncols=True,
                    disable=disable_chunk_pbars,
                )
                chunk_progress[job_info['job_id']] = {
                    'bar': bar,
                    'last_completed': 0,
                    'total': total_requests_for_chunk,
                }

        latest_status_message = "OpenAI batches: submitted"
        while submission_results:
            all_finalised = True
            completed_total = 0
            status_parts: List[str] = []

            for job_info in submission_results:
                batch_job = client.batches.retrieve(job_info['job_id'])
                job_info['batch_job'] = batch_job
                job_info['status'] = batch_job.status
                request_counts = _as_request_counts(getattr(batch_job, 'request_counts', None))
                job_info['request_counts'] = request_counts
                completed = request_counts.get('completed') or request_counts.get('succeeded') or 0
                clamped_completed = min(completed, job_info['chunk']['requests'])
                completed_total += clamped_completed

                progress_info = chunk_progress.get(job_info['job_id'])
                if progress_info:
                    delta = clamped_completed - progress_info['last_completed']
                    if delta > 0:
                        progress_info['bar'].update(delta)
                        progress_info['last_completed'] = clamped_completed
                    status_label = batch_job.status.replace('_', ' ')
                    progress_info['bar'].set_description(f"Chunk {job_info['chunk']['index']:03d} {status_label}")

                status_parts.append(f"{job_info['job_id'][:8]}:{batch_job.status}")
                if batch_job.status in ongoing_statuses:
                    all_finalised = False

            latest_status_message = f"OpenAI batches: {', '.join(status_parts)}"
            if self.progress_callback:
                self.progress_callback(min(completed_total, total_requests), total_requests, latest_status_message)

            if all_finalised:
                break

            time.sleep(poll_interval)

        for job_info in submission_results:
            progress_info = chunk_progress.get(job_info['job_id'])
            if not progress_info:
                continue
            remaining = progress_info['total'] - progress_info['last_completed']
            if remaining > 0:
                progress_info['bar'].update(remaining)
                progress_info['last_completed'] = progress_info['total']
            final_label = (job_info.get('status') or 'unknown').replace('_', ' ')
            progress_info['bar'].set_description(f"Chunk {job_info['chunk']['index']:03d} {final_label}")
            progress_info['bar'].close()

        if self.progress_callback:
            self.progress_callback(total_requests, total_requests, latest_status_message)

        failed_jobs: List[Dict[str, Any]] = []
        failed_records: List[Dict[str, Optional[str]]] = []
        if submission_results:
            for job_info in submission_results:
                if job_info.get('status') == 'completed':
                    continue

                batch_job = job_info.get('batch_job')
                error_file_path: Optional[Path] = None
                error_file_id = getattr(batch_job, 'error_file_id', None) if batch_job else None
                if error_file_id:
                    error_path = batch_dir / f"openai_batch_errors_{timestamp}_chunk{job_info['chunk']['index']:03d}.jsonl"
                    try:
                        error_response = client.files.content(error_file_id)
                        if hasattr(error_response, 'content'):
                            error_path.write_bytes(error_response.content)
                        else:
                            error_path.write_bytes(error_response.read())
                        error_file_path = error_path
                        self.logger.error("[BATCH] Error details for job %s saved to %s", job_info['job_id'], error_path)
                    except Exception as exc:
                        self.logger.warning("[BATCH] Unable to fetch error file %s: %s", error_file_id, exc)

                error_detail = None
                if batch_job:
                    error_detail = _stringify_error_detail(getattr(batch_job, 'error', None))
                    if not error_detail:
                        error_detail = _stringify_error_detail(getattr(batch_job, 'errors', None))
                    if not error_detail:
                        error_detail = _stringify_error_detail(getattr(batch_job, 'last_error', None))
                if not error_detail and job_info.get('request_counts'):
                    error_detail = f"request_counts={job_info['request_counts']}"

                job_info['error_detail'] = error_detail
                job_info['error_file_path'] = str(error_file_path) if error_file_path else None
                failed_jobs.append(job_info)

                chunk_idx = job_info['chunk']['index']
                status_label = job_info.get('status') or 'failed'
                if error_detail:
                    self.logger.error(
                        "[BATCH] Chunk %03d (job %s) finished with status '%s': %s",
                        chunk_idx,
                        job_info['job_id'],
                        status_label,
                        error_detail,
                    )
                else:
                    self.logger.error(
                        "[BATCH] Chunk %03d (job %s) finished with status '%s'.",
                        chunk_idx,
                        job_info['job_id'],
                        status_label,
                    )

                progress_info = chunk_progress.get(job_info['job_id'])
                if progress_info:
                    progress_info['bar'].set_description(f"Chunk {chunk_idx:03d} {status_label}")

                failed_records.append(
                    {
                        "chunk_index": f"{chunk_idx:03d}",
                        "job_id": job_info.get('job_id'),
                        "status": status_label,
                        "error_detail": error_detail,
                        "error_file": str(error_file_path) if error_file_path else None,
                    }
                )

        if failed_records:
            config['openai_batch_failed_jobs'] = failed_records
            if self.progress_manager:
                self.progress_manager.show_warning(
                    "One or more OpenAI batch chunks failed. Check logs for details."
                )

        output_file_path.write_text("")
        chunk_output_paths: List[Path] = []

        successful_jobs = [info for info in submission_results if info.get('status') == 'completed']
        if not successful_jobs:
            self.logger.error("[BATCH] No OpenAI batch chunks completed successfully; aggregated output will remain empty.")

        for job_info in successful_jobs:
            batch_job = job_info['batch_job']
            output_file_id = _extract_output_file_id(batch_job)
            if not output_file_id:
                raise RuntimeError(
                    f"OpenAI batch job {job_info['job_id']} completed but did not return an output file id."
                )
            try:
                output_response = client.files.content(output_file_id)
            except NotFoundError:
                raise RuntimeError(
                    f"OpenAI returned 404 for output file {output_file_id}. "
                    f"Check batch job {job_info['job_id']} in the OpenAI dashboard."
                )

            chunk_output_path = output_dir / f"openai_batch_output_{timestamp}_chunk{job_info['chunk']['index']:03d}.jsonl"
            if hasattr(output_response, 'content'):
                chunk_output_path.write_bytes(output_response.content)
            else:
                chunk_output_path.write_bytes(output_response.read())

            with chunk_output_path.open('r', encoding='utf-8') as src, output_file_path.open('a', encoding='utf-8') as dest:
                shutil.copyfileobj(src, dest)

            chunk_output_paths.append(chunk_output_path)
            job_info['output_file_id'] = output_file_id
            job_info['output_file_path'] = chunk_output_path
            self.logger.info(
                "[BATCH] Output for chunk %s saved to %s",
                job_info['chunk']['index'],
                chunk_output_path,
            )

        for job_info in failed_jobs:
            job_info.setdefault('output_file_id', None)
            job_info.setdefault('output_file_path', None)

        batch_elapsed = time.perf_counter() - start_time

        parsed_records = 0
        parsing_errors = 0
        if output_file_path.exists():
            with output_file_path.open('r', encoding='utf-8') as aggregated_handle:
                for line_number, raw_line in enumerate(aggregated_handle, 1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception as exc:  # pragma: no cover - defensive
                        parsing_errors += 1
                        self.logger.debug("[BATCH] Skipping malformed JSONL line %s: %s", line_number, exc)
                        continue

                    custom_id = record.get('custom_id')
                    if not custom_id:
                        continue
                    meta = request_metadata.get(custom_id)
                    if not meta:
                        self.logger.debug("[BATCH] No metadata found for custom_id=%s", custom_id)
                        continue

                    identifier_key = meta['identifier_key']
                    prompt_idx = meta['prompt_index']
                    prompt_key = str(prompt_idx)
                    prefix = meta.get('prefix') or ''
                    expected_keys = meta.get('expected_keys') or []

                    row_state = row_results.get(identifier_key)
                    if row_state is None:
                        row_dict = row_lookup.get(identifier_key, {})
                        row_state = {
                            'identifier': meta['identifier'],
                            'row_index': row_dict.get('row_index', meta.get('row_index')),
                            'raw': {},
                            'cleaned': {},
                            'status': {},
                            'merged': {},
                            'errors': [],
                            'usage': {},
                            'response_info': {}
                        }
                        row_results[identifier_key] = row_state

                    if row_state.get('row_index') is None:
                        row_dict = row_lookup.get(identifier_key)
                        if row_dict:
                            row_state['row_index'] = row_dict.get('row_index', meta.get('row_index'))
                        else:
                            row_state['row_index'] = meta.get('row_index')

                    response_payload = record.get('response') or {}
                    error_payload = record.get('error')
                    if error_payload:
                        status_counts['error'] += 1
                        row_state['status'][prompt_key] = 'error'
                        row_state['errors'].append(_stringify_error_detail(error_payload))
                        row_state['usage'][prompt_key] = {}
                        row_state['response_info'][prompt_key] = {
                            'status_code': response_payload.get('status_code'),
                            'finish_reason': None,
                            'request_id': response_payload.get('request_id'),
                            'batch_error': True,
                        }
                        continue

                    body = response_payload.get('body') or {}
                    choices = body.get('choices') or []
                    message_content = None
                    if choices:
                        message = choices[0].get('message') or {}
                        message_content = message.get('content')

                    if not message_content:
                        status_counts['error'] += 1
                        row_state['status'][prompt_key] = 'error'
                        row_state['errors'].append(f"No content returned for prompt {prompt_idx}")
                        row_state['usage'][prompt_key] = body.get('usage') or response_payload.get('usage') or {}
                        row_state['response_info'][prompt_key] = {
                            'status_code': response_payload.get('status_code'),
                            'finish_reason': choices[0].get('finish_reason') if choices else None,
                            'request_id': response_payload.get('request_id'),
                        }
                        continue

                    cleaned_json = message_content.strip()
                    if expected_keys:
                        try:
                            cleaned_candidate = clean_json_output(cleaned_json, expected_keys)
                            if cleaned_candidate:
                                cleaned_json = cleaned_candidate
                        except Exception as exc:  # pragma: no cover - defensive
                            self.logger.debug("[BATCH] clean_json_output failed for prompt %s: %s", prompt_idx, exc)

                    if not cleaned_json:
                        status_counts['cleaning_failed'] += 1
                        row_state['status'][prompt_key] = 'parse_error'
                        row_state['errors'].append(f"Unable to parse model response for prompt {prompt_idx}")
                        continue

                    row_state['raw'][prompt_key] = cleaned_json
                    row_state['cleaned'][prompt_key] = cleaned_json
                    row_state['status'][prompt_key] = 'success'
                    usage_payload = body.get('usage') or response_payload.get('usage') or {}
                    row_state['usage'][prompt_key] = usage_payload
                    finish_reason = choices[0].get('finish_reason') if choices else None
                    row_state['response_info'][prompt_key] = {
                        'status_code': response_payload.get('status_code'),
                        'finish_reason': finish_reason,
                        'request_id': response_payload.get('request_id'),
                    }

                    try:
                        parsed = json.loads(cleaned_json)
                        if prefix:
                            parsed = {f"{prefix}_{k}": v for k, v in parsed.items()}
                        row_state.setdefault('merged', {}).update(parsed)
                        self.last_annotation = parsed
                        status_counts['success'] += 1
                    except Exception as exc:
                        row_state['status'][prompt_key] = 'decode_error'
                        row_state['errors'].append(f"Decode error for prompt {prompt_idx}: {exc}")
                        status_counts['decode_error'] += 1
                        continue

                    parsed_records += 1

        if parsing_errors:
            self.logger.debug("[BATCH] Encountered %s parsing error(s) while reading OpenAI batch output.", parsing_errors)
        self.logger.info("[BATCH] Parsed %s response record(s) from OpenAI batch output.", parsed_records)

        jobs_metadata: List[Dict[str, Any]] = []
        weighted_time = 0.0
        weighted_requests = 0
        for job_info in submission_results:
            batch_job = job_info['batch_job']
            chunk_info = job_info['chunk']
            job_created_at = getattr(batch_job, 'created_at', None) or getattr(batch_job, 'created', None)
            job_started_at = getattr(batch_job, 'started_at', None)
            job_completed_at = getattr(batch_job, 'completed_at', None) or getattr(batch_job, 'finished_at', None)
            final_request_counts = job_info.get('request_counts', {})

            job_elapsed_seconds = None
            if isinstance(job_started_at, (int, float)) and isinstance(job_completed_at, (int, float)):
                job_elapsed_seconds = job_completed_at - job_started_at
            elif isinstance(job_created_at, (int, float)) and isinstance(job_completed_at, (int, float)):
                job_elapsed_seconds = job_completed_at - job_created_at

            if isinstance(job_elapsed_seconds, (int, float)) and job_elapsed_seconds >= 0:
                weighted_time += job_elapsed_seconds
                weighted_requests += max(chunk_info['requests'], 1)
            else:
                job_elapsed_seconds = None

            jobs_metadata.append(
                {
                    "chunk_index": chunk_info['index'],
                    "job_id": getattr(batch_job, 'id', None),
                    "status": getattr(batch_job, 'status', None),
                    "created_at": job_created_at,
                    "started_at": job_started_at,
                    "completed_at": job_completed_at,
                    "request_counts": final_request_counts,
                    "total_requests": chunk_info['requests'],
                    "input_file": str(chunk_info['path']),
                    "output_file": str(job_info['output_file_path']) if job_info.get('output_file_path') else None,
                    "output_file_id": job_info.get('output_file_id'),
                    "job_elapsed_seconds": job_elapsed_seconds,
                    "model": model_name,
                    "provider": config.get('provider') or config.get('annotation_provider'),
                    "error_file": job_info.get('error_file_path'),
                    "error_detail": job_info.get('error_detail'),
                }
            )

        overall_metadata = {
            "jobs": jobs_metadata,
            "chunk_count": len(jobs_metadata),
            "total_requests": total_requests,
            "dataset_rows": total_rows,
            "prompt_count": prompt_count,
            "batch_elapsed_seconds": batch_elapsed,
            "input_paths": [str(info['path']) for info in chunk_infos],
            "output_paths": [str(path) for path in chunk_output_paths],
            "aggregated_output": str(output_file_path),
            "model": model_name,
            "provider": config.get('provider') or config.get('annotation_provider'),
            "failed_jobs": failed_records,
        }

        try:
            metadata_file_path.write_text(json.dumps(overall_metadata, ensure_ascii=False, indent=2))
        except Exception as exc:
            self.logger.warning("[BATCH] Unable to write batch metadata: %s", exc)

        input_file_path = chunk_infos[0]['path']
        config['openai_batch_metadata_path'] = str(metadata_file_path)
        config['openai_batch_input_path'] = str(input_file_path)
        config['openai_batch_input_paths'] = [str(info['path']) for info in chunk_infos]
        config['openai_batch_output_path'] = str(output_file_path)
        config['openai_batch_output_paths'] = [str(path) for path in chunk_output_paths]
        config['openai_batch_job_id'] = jobs_metadata[0].get('job_id') if jobs_metadata else None
        config['openai_batch_job_ids'] = [meta.get('job_id') for meta in jobs_metadata]
        config['openai_batch_status'] = jobs_metadata[0].get('status') if jobs_metadata else None
        config['openai_batch_statuses'] = [meta.get('status') for meta in jobs_metadata]
        config['openai_batch_job_elapsed_seconds'] = jobs_metadata[0].get('job_elapsed_seconds') if jobs_metadata else None
        config['openai_batch_total_requests'] = total_requests
        config['openai_batch_jobs_metadata'] = jobs_metadata
        config['openai_batch_successful_job_ids'] = [
            meta.get('job_id') for meta in jobs_metadata if meta.get('status') == 'completed'
        ]
        config['openai_batch_chunk_output_paths'] = [str(path) for path in chunk_output_paths]
        per_row_time = (
            (weighted_time / weighted_requests) if weighted_requests else batch_elapsed / max(total_rows, 1)
        )

        archive_dir = config.get('openai_batch_archive_dir')
        if archive_dir:
            try:
                archive_path = Path(archive_dir)
                archive_path.mkdir(parents=True, exist_ok=True)
                archive_sources = [metadata_file_path, output_file_path] + [info['path'] for info in chunk_infos] + chunk_output_paths
                for src_path in archive_sources:
                    path_obj = Path(src_path)
                    if path_obj.exists():
                        shutil.copy2(path_obj, archive_path / path_obj.name)
            except Exception as exc:
                self.logger.warning("[BATCH] Unable to archive batch artifacts to %s: %s", archive_dir, exc)

        # Sequential fallback for prompts that did not return usable results
        fallback_candidates: List[Tuple[str, List[int]]] = []
        for identifier_key, custom_ids in per_row_custom_ids.items():
            missing_indices: List[int] = []
            row_state = row_results.get(identifier_key)
            if not row_state:
                for custom_id in custom_ids:
                    meta = request_metadata.get(custom_id)
                    if meta:
                        missing_indices.append(int(meta['prompt_index']))
            else:
                for custom_id in custom_ids:
                    meta = request_metadata.get(custom_id)
                    if not meta:
                        continue
                    prompt_index = int(meta['prompt_index'])
                    status = row_state['status'].get(str(prompt_index))
                    if status != 'success':
                        missing_indices.append(prompt_index)
            if missing_indices:
                fallback_candidates.append((identifier_key, missing_indices))

        if fallback_candidates:
            self.logger.info(
                "[BATCH] Retrying %s row(s) sequentially due to incomplete batch responses.",
                len(fallback_candidates)
            )
            for identifier_key, missing_indices in fallback_candidates:
                retry_failed_prompts(
                    identifier_key=identifier_key,
                    missing_indices=missing_indices,
                    row_state=row_results.get(identifier_key)
                )

        cleanup_enabled = config.get('openai_batch_cleanup_missing', True)
        if cleanup_enabled:
            residual_candidates: List[Tuple[str, List[int]]] = []
            for identifier_key, row_state in row_results.items():
                if not row_state:
                    continue
                missing_indices: List[int] = []
                for prompt_idx, expected_keys in prefixed_expected_keys.items():
                    if not expected_keys:
                        continue
                    status = row_state['status'].get(str(prompt_idx))
                    merged_payload = row_state.get('merged', {})
                    needs_retry = any(
                        merged_payload.get(key) in (None, '', []) for key in expected_keys
                    )
                    if status == 'success' and not needs_retry:
                        continue
                    if needs_retry:
                        missing_indices.append(prompt_idx)
                if missing_indices:
                    residual_candidates.append((identifier_key, missing_indices))
            if residual_candidates:
                self.logger.info(
                    "[BATCH] Performing cleanup retries for %s row(s) with missing prompt keys.",
                    len(residual_candidates)
                )
                for identifier_key, missing_indices in residual_candidates:
                    retry_failed_prompts(
                        identifier_key=identifier_key,
                        missing_indices=missing_indices,
                        row_state=row_results.get(identifier_key)
                    )

        # --- Optimized consolidation: collect in buffers, then bulk-assign ---
        annotation_values = {}       # {row_index: final_json}
        inference_time_values = {}   # {row_index: per_row_time}
        fallback_annotations = []    # [(identifier_value, final_json, per_row_time)]
        log_entries_buffer = []      # Batch log writes
        last_final_payload = None    # For self.last_annotation

        completed_rows = 0
        progress_interval = max(1, total_rows // 100) if total_rows > 100 else 1

        for identifier_key, custom_ids in per_row_custom_ids.items():
            meta = request_metadata[custom_ids[0]]
            identifier_value = meta['identifier']
            row_state = row_results.get(identifier_key)
            if not row_state:
                row_state = {
                    'identifier': identifier_value,
                    'row_index': meta['row_index'],
                    'raw': {},
                    'cleaned': {},
                    'status': {},
                    'merged': {},
                    'errors': ["No response returned by OpenAI."],
                    'usage': {},
                    'response_info': {}
                }

            for custom_id in custom_ids:
                prompt_key = str(request_metadata[custom_id]['prompt_index'])
                if prompt_key not in row_state['status']:
                    row_state['raw'][prompt_key] = None
                    row_state['cleaned'][prompt_key] = None
                    row_state['status'][prompt_key] = 'missing'
                    row_state['errors'].append("No response returned by OpenAI.")
                    row_state['usage'][prompt_key] = {}
                    row_state['response_info'][prompt_key] = {}

            merged_payload = row_state.get('merged', {})
            final_payload: Dict[str, Any] = {}
            if all_expected_keys:
                for key in all_expected_keys:
                    final_payload[key] = merged_payload.get(key)
            for key, value in merged_payload.items():
                if key not in final_payload:
                    final_payload[key] = value
            row_state['merged'] = final_payload
            final_json = json.dumps(final_payload, ensure_ascii=False) if final_payload else None
            usage_totals = aggregate_usage(row_state.get('usage', {}).values())

            row_index = row_state.get('row_index')
            if row_index is not None and row_index in full_data.index:
                annotation_values[row_index] = final_json
                inference_time_values[row_index] = per_row_time
            else:
                fallback_annotations.append((identifier_value, final_json, per_row_time))

            if final_payload:
                last_final_payload = final_payload

            if log_enabled and log_path:
                log_entries_buffer.append({
                    'id': identifier_value,
                    'final_json': final_json,
                    'inference_time': per_row_time,
                    'usage_summary': usage_totals,
                })

            completed_rows += 1
            if self.progress_callback and (completed_rows % progress_interval == 0 or completed_rows == total_rows):
                self.progress_callback(
                    completed_rows,
                    total_rows,
                    f"Annotated {completed_rows}/{total_rows} items via OpenAI batch"
                )

        # Bulk DataFrame assignment (2 calls instead of N*2)
        if annotation_values:
            idx_list = list(annotation_values.keys())
            full_data.loc[idx_list, annotation_column] = [annotation_values[i] for i in idx_list]
            full_data.loc[idx_list, f"{annotation_column}_inference_time"] = [inference_time_values[i] for i in idx_list]

        # Handle fallback rows (identifier-based lookup)
        if fallback_annotations:
            if len(fallback_annotations) > 10:
                # Pre-build index for efficient lookup
                id_to_indices = full_data.groupby(identifier_column).apply(lambda g: g.index.tolist())
                for ident, fjson, rtime in fallback_annotations:
                    matched = id_to_indices.get(ident)
                    if matched:
                        full_data.loc[matched, annotation_column] = fjson
                        full_data.loc[matched, f"{annotation_column}_inference_time"] = rtime
                    else:
                        self.logger.warning(
                            "[BATCH] Unable to apply annotation for identifier '%s' (fallback lookup)",
                            ident,
                        )
            else:
                for ident, fjson, rtime in fallback_annotations:
                    mask = full_data[identifier_column] == ident
                    if mask.any():
                        full_data.loc[mask, annotation_column] = fjson
                        full_data.loc[mask, f"{annotation_column}_inference_time"] = rtime
                    else:
                        self.logger.warning(
                            "[BATCH] Unable to apply annotation for identifier '%s' (fallback lookup)",
                            ident,
                        )

        # Set self.last_annotation without re-parsing JSON
        if last_final_payload:
            self.last_annotation = last_final_payload

        # Batch log write (1 file open instead of N)
        if log_entries_buffer and log_path:
            log_path_obj = Path(log_path)
            log_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path_obj, 'a', encoding='utf-8') as f:
                for entry in log_entries_buffer:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        # Push batch results to Doccano if sync is active
        if self.doccano_sync:
            doccano_items = []
            for identifier_key, custom_ids in per_row_custom_ids.items():
                meta_info = request_metadata[custom_ids[0]]
                row_state = row_results.get(identifier_key, {})
                final_payload = row_state.get('merged', {})
                if not final_payload:
                    continue
                row_idx = row_state.get('row_index')
                if row_idx is not None and row_idx in full_data.index and text_columns:
                    text_val = str(full_data.loc[row_idx, text_columns[0]])
                else:
                    text_val = str(meta_info['identifier'])
                doccano_items.append({
                    "text": text_val,
                    "annotation": final_payload,
                    "meta": {"identifier": str(meta_info['identifier'])},
                })
            if doccano_items:
                try:
                    batch_stats = self.doccano_sync.push_batch(doccano_items)
                    self.logger.info(
                        "[BATCH] Doccano sync: pushed=%s, queued=%s, errors=%s",
                        batch_stats.get('pushed', 0),
                        batch_stats.get('queued', 0),
                        batch_stats.get('errors', 0),
                    )
                except Exception as sync_exc:
                    self.logger.warning("[BATCH] Doccano batch push failed: %s", sync_exc)

        if save_incrementally and output_path:
            self.logger.info("[BATCH] Batch mode overrides incremental saves; writing final dataset once.")

        # Ensure output_path is resolved to absolute path for reliability
        if output_path:
            output_path_obj = Path(output_path)
            if not output_path_obj.is_absolute():
                output_path_obj = Path.cwd() / output_path_obj
            output_path = str(output_path_obj)
            config['output_path'] = output_path  # Update config for downstream use

        # Save the consolidated CSV - this is critical for batch mode
        save_success = False
        if output_path:
            try:
                self.logger.info("[BATCH] Saving consolidated results to %s", output_path)
                self._save_data(full_data, output_path, output_format)
                save_success = Path(output_path).exists()
                if save_success:
                    self.logger.info("[BATCH] Successfully saved consolidated CSV: %s", output_path)
                else:
                    self.logger.error("[BATCH] File was not created after save attempt: %s", output_path)
            except Exception as exc:
                self.logger.error("[BATCH] Failed to save consolidated CSV: %s", exc)
                # Attempt fallback save
                try:
                    fallback_path = batch_dir / f"consolidated_results_{timestamp}.csv"
                    self.logger.info("[BATCH] Attempting fallback save to %s", fallback_path)
                    self._save_data(full_data, str(fallback_path), 'csv')
                    if fallback_path.exists():
                        self.logger.info("[BATCH] Fallback save successful: %s", fallback_path)
                        config['output_path'] = str(fallback_path)
                        save_success = True
                except Exception as fallback_exc:
                    self.logger.error("[BATCH] Fallback save also failed: %s", fallback_exc)

            # Ensure parity with local annotator pipelines by materialising the annotated-only companion file.
            subset_created = None
            subset_flag_original = config.get('create_annotated_subset', False)
            try:
                config['create_annotated_subset'] = True
                subset_created = self._save_annotated_subset(full_data, config)
            except Exception as subset_exc:
                self.logger.warning("[BATCH] Failed to create annotated subset: %s", subset_exc)
            finally:
                config['create_annotated_subset'] = subset_flag_original

            if subset_created:
                state_obj = getattr(self, "state", None)
                if state_obj and isinstance(state_obj.annotation_results, dict):
                    state_obj.annotation_results['annotations_only_path'] = str(subset_created)
                    state_obj.annotation_results['annotated_subset_path'] = str(subset_created)
        else:
            self.logger.warning("[BATCH] No output_path specified; consolidated CSV will not be saved.")

        self.logger.info(
            "[BATCH] Completed OpenAI batch annotation in %.2fs (rows=%s, prompts=%s, saved=%s)",
            batch_elapsed,
            total_rows,
            prompt_count,
            save_success
        )

        # Finalize Doccano sync for batch mode
        if self.doccano_sync:
            sync_stats = self.doccano_sync.stop()
            if sync_stats.get('remaining_queue', 0) > 0:
                self.logger.warning("[BATCH] Doccano sync: %d items remain in offline queue", sync_stats['remaining_queue'])
            else:
                self.logger.info("[BATCH] Doccano sync: all %d items pushed successfully", sync_stats.get('pushed', 0))

        return full_data

    def _save_results(self, data: pd.DataFrame, config: Dict[str, Any]):
        """Save annotation results"""
        output_path = config.get('output_path')
        subset_path = None

        # Resolve to absolute path if needed
        if output_path:
            output_path_obj = Path(output_path)
            if not output_path_obj.is_absolute():
                output_path_obj = Path.cwd() / output_path_obj
            output_path = str(output_path_obj)

        if output_path:
            output_format = config.get('output_format') or 'csv'
            try:
                # Ensure parent directory exists
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                self._save_data(data, output_path, output_format)
                if Path(output_path).exists():
                    self.logger.info(f"Results saved to {output_path}")
                else:
                    self.logger.warning(f"Save called but file not found at {output_path}")
            except Exception as exc:
                self.logger.error(f"Failed to save results to {output_path}: {exc}")

            if config.get('create_annotated_subset'):
                try:
                    subset_path = self._save_annotated_subset(data, config)
                except Exception as exc:
                    self.logger.warning(f"Failed to create annotated subset: {exc}")

        doccano_path: Optional[Path] = None
        state = getattr(self, "state", None)

        if subset_path:
            self.logger.info(f"Annotated subset saved to {subset_path}")

        export_prefs = config.get('export_preferences', {})
        export_doccano = config.get('export_to_doccano') or export_prefs.get('export_to_doccano')
        if export_doccano:
            annotation_column = config.get('annotation_column', 'annotation')
            text_column = config.get('text_column')
            if not text_column:
                text_cols = config.get('text_columns') or []
                if text_cols:
                    text_column = text_cols[0]
            if annotation_column in data.columns and text_column in data.columns:
                mask = (
                    data[annotation_column].notna()
                    & data[annotation_column].astype(str).str.strip().ne('')
                    & data[annotation_column].astype(str).str.lower().ne('nan')
                )
                annotated_df = data.loc[mask].copy()
                if not annotated_df.empty:
                    try:
                        doccano_path = self._export_doccano_jsonl(
                            annotated_df=annotated_df,
                            annotation_column=annotation_column,
                            text_column=text_column,
                            config=config
                        )
                        if doccano_path:
                            self.logger.info(f"Doccano export saved to {doccano_path}")
                            config['doccano_export_path'] = str(doccano_path)
                            if state and isinstance(state.annotation_results, dict):
                                state.annotation_results['doccano_export_path'] = str(doccano_path)
                    except Exception as exc:
                        self.logger.warning(f"Doccano export failed: {exc}")
        elif state and isinstance(state.annotation_results, dict):
            # Ensure stale paths are cleared when export disabled
            state.annotation_results.pop('doccano_export_path', None)
            state.annotation_results.pop('validation_doccano_export_path', None)

    def _save_data(self, data: pd.DataFrame, path: str, format: str):
        """Save data to file"""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        fmt = (format or 'csv').lower()
        if fmt == 'csv':
            data.to_csv(path_obj, index=False)
        elif fmt == 'excel':
            data.to_excel(path_obj, index=False)
        elif fmt == 'parquet':
            data.to_parquet(path_obj, index=False)
        elif fmt == 'json':
            data.to_json(path_obj, orient='records', lines=False, force_ascii=False, indent=2)
        elif fmt == 'jsonl':
            data.to_json(path_obj, orient='records', lines=True, force_ascii=False)
        elif fmt in ['rdata', 'rds']:
            if HAS_PYREADR:
                if fmt == 'rdata':
                    pyreadr.write_rdata({'data': data}, str(path_obj))
                else:
                    pyreadr.write_rds(data, str(path_obj))
            else:
                raise ImportError("pyreadr required for RData/RDS files")
        else:
            raise ValueError(f"Unsupported output format: {format}")

    def _save_annotated_subset(self, data: pd.DataFrame, config: Dict[str, Any]) -> Optional[Path]:
        """Persist a lightweight file containing only annotated rows."""
        if not config.get('create_annotated_subset', True):
            return None

        annotation_column = config.get('annotation_column', 'annotation')
        if annotation_column not in data.columns:
            return None

        mask = (
            data[annotation_column].notna()
            & data[annotation_column].astype(str).str.strip().ne('')
            & data[annotation_column].astype(str).str.lower().ne('nan')
        )
        subset = data.loc[mask].copy()
        if subset.empty:
            return None

        # Drop verbose per-prompt columns from the compact export
        for suffix in PROMPT_SUFFIXES:
            col_name = f"{annotation_column}_{suffix}"
            if col_name in subset.columns:
                subset.drop(columns=[col_name], inplace=True)

        output_path = config.get('output_path')
        output_format = config.get('output_format', 'csv')
        if not output_path:
            return None

        output_path_obj = Path(output_path)
        subset_path = output_path_obj.with_name(f"{output_path_obj.stem}_annotated_only{output_path_obj.suffix}")

        self._save_data(subset, subset_path, output_format)
        config['annotated_subset_path'] = str(subset_path)
        self.logger.info(f"Saved annotated-only subset to {subset_path}")
        return subset_path

    def _export_to_doccano(
        self,
        annotated_df: pd.DataFrame,
        annotation_column: str,
        text_column: Optional[str],
        config: Dict[str, Any]
    ) -> Optional[Path]:
        """Compatibility wrapper that preserves legacy method name."""
        if annotated_df is None or annotated_df.empty:
            self.logger.debug("[DOCCANO] Skipping export; no annotated rows available.")
            return None

        if annotation_column not in annotated_df.columns:
            self.logger.warning(
                "[DOCCANO] Unable to export; annotation column '%s' not found.",
                annotation_column
            )
            return None

        candidate_columns: List[Optional[str]] = []
        if text_column:
            candidate_columns.append(text_column)
        cfg_primary = config.get('text_column')
        if cfg_primary:
            candidate_columns.append(cfg_primary)
        candidate_columns.extend(config.get('text_columns') or [])
        candidate_columns.append('text')

        resolved_text_column = next(
            (col for col in candidate_columns if col and col in annotated_df.columns),
            None
        )
        if not resolved_text_column:
            self.logger.warning(
                "[DOCCANO] Unable to export; none of the candidate text columns %s exist.",
                candidate_columns
            )
            return None

        return self._export_doccano_jsonl(
            annotated_df=annotated_df,
            annotation_column=annotation_column,
            text_column=resolved_text_column,
            config=config
        )

    def _export_doccano_jsonl(
        self,
        *,
        annotated_df: pd.DataFrame,
        annotation_column: str,
        text_column: str,
        config: Dict[str, Any]
    ) -> Optional[Path]:
        """Export annotated samples to a Doccano-compatible JSONL file."""
        if annotated_df.empty:
            return None

        session_dirs = config.get('session_dirs') or {}
        default_dir = self.settings.paths.logs_dir / "doccano_exports"

        doccano_base: Optional[Path] = None
        session_root: Optional[Path] = None

        if isinstance(session_dirs, dict):
            doccano_dir_hint = session_dirs.get('doccano')
            if doccano_dir_hint:
                doccano_base = Path(doccano_dir_hint)
                try:
                    session_root = doccano_base.parent.parent
                except IndexError:
                    session_root = None

            if doccano_base is None:
                session_root_hint = session_dirs.get('session_root') or session_dirs.get('base')
                if session_root_hint:
                    session_root = Path(session_root_hint)
                    doccano_base = session_root / 'validation_exports' / 'doccano'

        if doccano_base is None:
            output_path = config.get('output_path')
            if output_path:
                try:
                    output_path_obj = Path(output_path)
                    for parent in output_path_obj.parents:
                        grandparent = parent.parent
                        if grandparent and grandparent.name in {'annotator', 'annotator_factory'}:
                            session_root = parent
                            doccano_base = session_root / 'validation_exports' / 'doccano'
                            break
                        parent_lower = parent.name.lower()
                        if parent_lower.startswith(('annotator_session_', 'factory_session_', 'test_', 'session_')):
                            session_root = parent
                            doccano_base = session_root / 'validation_exports' / 'doccano'
                            break
                except Exception:
                    session_root = None
                    doccano_base = None

        if doccano_base is None:
            session_id_hint = config.get('session_id')
            if session_id_hint:
                mode_hint = (config.get('annotation_mode') or '').lower()
                mode_folder = 'annotator_factory' if 'factory' in mode_hint else 'annotator'
                session_root = (
                    self.settings.paths.logs_dir
                    / mode_folder
                    / session_id_hint
                )
                doccano_base = session_root / 'validation_exports' / 'doccano'

        if doccano_base is None:
            doccano_base = default_dir

        doccano_base = Path(doccano_base)
        if session_root is None:
            try:
                if (
                    doccano_base.parent.name == 'doccano'
                    and doccano_base.parent.parent.name == 'validation_exports'
                ):
                    session_root = doccano_base.parent.parent.parent
            except Exception:
                session_root = None

        provider_folder_raw = (
            config.get('provider_folder')
            or config.get('annotation_provider')
            or config.get('provider')
        )
        provider_folder = self._sanitize_path_component(provider_folder_raw, 'provider')

        model_reference = (
            config.get('model_folder')
            or config.get('annotation_model')
            or config.get('model')
        )
        if isinstance(model_reference, dict):
            model_reference = (
                model_reference.get('model_name')
                or model_reference.get('name')
            )
        model_folder = self._sanitize_path_component(model_reference, 'model')

        dataset_reference = (
            config.get('dataset_name')
            or config.get('file_path')
            or 'dataset'
        )
        dataset_name = self._sanitize_path_component(Path(str(dataset_reference)).stem, 'dataset')

        provider_root = doccano_base / provider_folder
        provider_root.mkdir(parents=True, exist_ok=True)

        for alt_model_dir in provider_root.iterdir():
            if not alt_model_dir.is_dir():
                continue
            normalized = self._sanitize_path_component(alt_model_dir.name, 'model')
            if normalized == model_folder and alt_model_dir.name != model_folder:
                try:
                    shutil.rmtree(alt_model_dir, ignore_errors=True)
                except Exception as exc:
                    self.logger.debug(f"[DOCCANO] Unable to clean legacy model directory {alt_model_dir}: {exc}")

        doccano_dir = provider_root / model_folder / dataset_name
        doccano_dir.mkdir(parents=True, exist_ok=True)

        # Keep only the latest export for this dataset
        for existing_export in doccano_dir.glob("*.jsonl"):
            try:
                existing_export.unlink()
            except Exception as exc:
                self.logger.debug(f"[DOCCANO] Unable to remove legacy export {existing_export}: {exc}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        doccano_path = doccano_dir / f"{dataset_name}_doccano_{timestamp}.jsonl"

        label_keys: Set[str] = set()
        for prompt_cfg in config.get('prompts', []):
            expected_keys = prompt_cfg.get('expected_keys') or []
            prefix = prompt_cfg.get('prefix', '') or ''
            for key in expected_keys:
                label_keys.add(f"{prefix}_{key}" if prefix else key)

        total_written = 0
        with doccano_path.open('w', encoding='utf-8') as handle:
            for counter, (row_index, row) in enumerate(annotated_df.iterrows(), start=1):
                annotation_str = row.get(annotation_column)
                if not annotation_str or not str(annotation_str).strip():
                    continue

                try:
                    annotation_data = json.loads(annotation_str)
                except Exception:
                    self.logger.debug("[DOCCANO] Skipping row with invalid annotation JSON.")
                    continue

                entry_labels: List[str] = []

                def _normalize_component(value: Any) -> Optional[str]:
                    """Return a lowercase token stripped of punctuation for Doccano labels."""
                    if value is None:
                        return None
                    if isinstance(value, bool):
                        token = "true" if value else "false"
                    elif isinstance(value, (int, float)):
                        token = str(value)
                    else:
                        token = str(value).strip()
                    token = token.strip(",;")
                    if not token:
                        return None
                    lowered = token.lower()
                    if lowered in {"null", "none", "nan"}:
                        return None
                    sanitized = re.sub(r"[^\w]+", "_", lowered).strip("_")
                    return sanitized or None

                if label_keys:
                    label_keys_to_process: Set[str] = {
                        key for key in label_keys if key in annotation_data
                    }
                else:
                    label_keys_to_process = set(annotation_data.keys())

                for label_key in label_keys_to_process:
                    label_value = annotation_data.get(label_key)
                    if label_value is None:
                        continue

                    normalized_key = _normalize_component(label_key)
                    if not normalized_key:
                        normalized_key = _normalize_component(str(label_key))
                    if not normalized_key:
                        normalized_key = str(label_key).strip().replace(" ", "_")

                    def add_combined_label(raw_value: Any, key_prefix: str = normalized_key) -> None:
                        normalized_value = _normalize_component(raw_value)
                        if normalized_value:
                            entry_labels.append(f"{key_prefix}_{normalized_value}")

                    if isinstance(label_value, list):
                        for candidate in label_value:
                            add_combined_label(candidate)
                    elif isinstance(label_value, dict):
                        for sub_key, sub_value in label_value.items():
                            nested_key = _normalize_component(sub_key)
                            combined_key = (
                                f"{normalized_key}_{nested_key}"
                                if nested_key
                                else normalized_key
                            )
                            add_combined_label(sub_value, key_prefix=combined_key)
                    else:
                        add_combined_label(label_value)

                meta: Dict[str, Any] = {}
                for column in annotated_df.columns:
                    if column in (text_column, annotation_column):
                        continue
                    meta[column] = self._serialize_meta_value(row[column])
                meta['annotation_json'] = annotation_data
                cleaned_meta = {k: v for k, v in meta.items() if v not in (None, '')}

                entry_id = row.get('id')
                if entry_id is not None:
                    entry_id = self._serialize_meta_value(entry_id)

                if isinstance(entry_id, str):
                    stripped_id = entry_id.strip()
                    if stripped_id.isdigit():
                        entry_id = int(stripped_id)
                    elif stripped_id:
                        entry_id = stripped_id
                    else:
                        entry_id = None

                if entry_id is None:
                    if isinstance(row_index, (int, np.integer)):
                        entry_id = int(row_index)
                    elif isinstance(row_index, str) and row_index.isdigit():
                        entry_id = int(row_index)
                    else:
                        entry_id = counter

                entry: Dict[str, Any] = {
                    'id': entry_id,
                    'text': str(row[text_column]),
                    'label': [],
                    'Comments': []
                }
                if entry_labels:
                    seen: Set[str] = set()
                    unique_labels: List[str] = []
                    for label_token in entry_labels:
                        if not label_token or label_token in seen:
                            continue
                        unique_labels.append(label_token)
                        seen.add(label_token)
                    entry['label'] = unique_labels
                if cleaned_meta:
                    entry['meta'] = cleaned_meta

                handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
                total_written += 1

        if total_written == 0:
            try:
                doccano_path.unlink(missing_ok=True)
            except Exception:
                pass
            self.logger.warning("[DOCCANO] Export produced zero entries; no file generated.")
            return None

        if session_root:
            try:
                session_id_for_validation = session_root.name
                validation_provider_root = (
                    self.settings.paths.validation_dir
                    / session_id_for_validation
                    / "doccano"
                    / provider_folder
                )
                validation_provider_root.mkdir(parents=True, exist_ok=True)

                for alt_model_dir in validation_provider_root.iterdir():
                    if not alt_model_dir.is_dir():
                        continue
                    normalized = self._sanitize_path_component(alt_model_dir.name, 'model')
                    if normalized == model_folder and alt_model_dir.name != model_folder:
                        try:
                            shutil.rmtree(alt_model_dir, ignore_errors=True)
                        except Exception as exc:
                            self.logger.debug(f"[DOCCANO] Unable to clean legacy validation directory {alt_model_dir}: {exc}")

                validation_model_root = validation_provider_root / model_folder
                validation_model_root.mkdir(parents=True, exist_ok=True)

                validation_dir = validation_model_root / dataset_name
                validation_dir.mkdir(parents=True, exist_ok=True)
                for legacy_validation in validation_dir.glob("*.jsonl"):
                    try:
                        legacy_validation.unlink()
                    except Exception as exc:
                        self.logger.debug(f"[DOCCANO] Unable to remove legacy validation export {legacy_validation}: {exc}")
                validation_copy_path = validation_dir / doccano_path.name
                shutil.copy2(doccano_path, validation_copy_path)

                legacy_plain = (
                    self.settings.paths.validation_dir
                    / session_id_for_validation
                    / "doccano"
                    / doccano_path.name
                )
                if legacy_plain.exists() and legacy_plain != validation_copy_path:
                    try:
                        legacy_plain.unlink()
                    except Exception as exc:
                        self.logger.debug(f"[DOCCANO] Unable to remove legacy validation export {legacy_plain}: {exc}")
                config['validation_doccano_export_path'] = str(validation_copy_path)
                if state and isinstance(state.annotation_results, dict):
                    state.annotation_results['validation_doccano_export_path'] = str(validation_copy_path)
            except Exception as exc:
                self.logger.debug(f"[DOCCANO] Unable to mirror export to validation directory: {exc}")

        return doccano_path

    @staticmethod
    def _sanitize_path_component(value: Optional[str], fallback: str) -> str:
        """Return filesystem-friendly component names (letters, digits, ._-)."""
        if not value:
            return fallback
        sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value))
        sanitized = sanitized.strip('_')
        return sanitized or fallback

    @staticmethod
    def _serialize_meta_value(value: Any) -> Optional[Any]:
        """Ensure metadata values are JSON-serializable."""
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (datetime, pd.Timestamp)):
            return str(value)
        if isinstance(value, (str, bool, list, dict)):
            return value
        return str(value)

    def _store_annotation_payload(
        self,
        dataframe: pd.DataFrame,
        mask: pd.Series,
        payload: Optional[Dict[str, Any]],
        annotation_column: str
    ):
        """No-op placeholder retained for compatibility with legacy workflows."""
        return

    def _append_to_csv(self, data: pd.DataFrame, identifier, identifier_column: str,
                      annotation_column: str, path: str, row_index=None):
        """Append single row to CSV for incremental saving.

        Parameters
        ----------
        row_index : optional
            Exact DataFrame index of the row to append.  When provided the
            method writes only that single row, avoiding duplication when
            multiple rows share the same *identifier*.
        """
        if row_index is not None and row_index in data.index:
            row_df = data.loc[[row_index]].copy()
        else:
            # Fallback: match by identifier (legacy behaviour)
            row_df = data[data[identifier_column] == identifier].copy()

        # Remove per-prompt columns for cleaner output
        for suffix in PROMPT_SUFFIXES:
            col = f"{annotation_column}_{suffix}"
            if col in row_df.columns:
                row_df = row_df.drop(columns=[col])

        header_needed = not os.path.exists(path)
        row_df.to_csv(path, mode='a', index=False, header=header_needed)

    def _write_log_entry(self, log_path: str, entry: Dict[str, Any]):
        """Write log entry for annotation"""
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def _generate_summary(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate annotation summary statistics"""
        total = len(data)
        success = status_counts.get('success', 0)
        errors = status_counts.get('error', 0)
        annotation_column = config.get('annotation_column', 'annotation')

        annotated_rows = 0
        if annotation_column in data.columns:
            annotated_rows = data[annotation_column].dropna().shape[0]

        usage_columns = {
            'prompt_tokens': f"{annotation_column}_prompt_tokens",
            'completion_tokens': f"{annotation_column}_completion_tokens",
            'total_tokens': f"{annotation_column}_total_tokens",
            'cached_prompt_tokens': f"{annotation_column}_cached_prompt_tokens",
            'reasoning_tokens': f"{annotation_column}_reasoning_tokens",
        }
        usage_summary: Dict[str, float] = {}
        for key, column in usage_columns.items():
            if column in data.columns:
                try:
                    usage_summary[key] = float(pd.to_numeric(data[column], errors='coerce').fillna(0).sum())
                except Exception:
                    usage_summary[key] = 0.0

        row_success_count = int(annotated_rows)

        summary: Dict[str, Any] = {
            'total_processed': total,
            'successful': success,
            'errors': errors,
            'success_rate': (row_success_count / total * 100) if total > 0 else 0,
            'annotated_rows': int(annotated_rows),
            'total_annotated': int(annotated_rows),
            'success_count': row_success_count,
            'prompt_success_count': success,
            'error_count': errors,
            'annotation_column': annotation_column,
            'output_file': config.get('output_path'),
            'output_format': config.get('output_format', 'csv'),
            'model': config.get('model'),
            'provider': config.get('provider'),
            'annotation_sample_size': config.get('annotation_sample_size') or config.get('annotation_limit'),
            'usage_summary': usage_summary,
            'status_breakdown': dict(status_counts),
            'timestamp': datetime.now().isoformat()
        }

        model_display_name = config.get('model_display_name')
        if model_display_name:
            summary['model_display_name'] = model_display_name

        inference_column = f"{annotation_column}_inference_time"
        if inference_column in data.columns:
            try:
                inference_series = pd.to_numeric(data[inference_column], errors='coerce').dropna()
            except Exception:
                inference_series = pd.Series(dtype=float)
            if not inference_series.empty:
                summary['inference_time_column'] = inference_column
                summary['mean_inference_time'] = float(inference_series.mean())
                summary['median_inference_time'] = float(inference_series.median())
                summary['total_inference_time'] = float(inference_series.sum())

        subset_path = config.get('annotated_subset_path')
        if subset_path:
            summary['annotated_subset_path'] = subset_path

        batch_keys = (
            'openai_batch_dir',
            'openai_batch_metadata_path',
            'openai_batch_input_path',
            'openai_batch_output_path',
            'openai_batch_job_id',
            'openai_batch_status',
            'openai_batch_job_elapsed_seconds',
            'openai_batch_total_requests',
        )
        for key in batch_keys:
            value = config.get(key)
            if value is not None:
                summary[key] = value

        preview_samples: List[Dict[str, Any]] = []
        identifier_column = config.get('identifier_column')
        if identifier_column and identifier_column in data.columns and annotation_column in data.columns:
            annotated_df = data[
                data[annotation_column].notna()
                & data[annotation_column].astype(str).str.strip().ne('')
                & data[annotation_column].astype(str).str.lower().ne('nan')
            ]
            if not annotated_df.empty:
                preview_cols = [identifier_column, annotation_column]
                if inference_column in annotated_df.columns:
                    preview_cols.append(inference_column)

                sample_size = int(config.get('preview_sample_size', 1))
                sample_size = max(1, sample_size)
                actual_size = min(sample_size, len(annotated_df))
                sampled = annotated_df[preview_cols].sample(
                    n=actual_size,
                    random_state=random.randint(0, 10**6)
                )
                max_chars = int(config.get('preview_max_chars', 320))

                for _, row in sampled.iterrows():
                    preview_entry: Dict[str, Any] = {}
                    for col in preview_cols:
                        value = row[col]
                        if isinstance(value, pd.Timestamp):
                            preview_entry[col] = value.isoformat()
                        elif pd.isna(value):
                            preview_entry[col] = None
                        else:
                            preview_entry[col] = value

                    annotation_payload = preview_entry.get(annotation_column)
                    preview_text = None
                    if isinstance(annotation_payload, str):
                        try:
                            annotation_json = json.loads(annotation_payload)
                        except Exception:
                            annotation_json = None
                        if isinstance(annotation_json, dict):
                            fragments: List[str] = []
                            for key in sorted(annotation_json.keys()):
                                value = annotation_json[key]
                                if value in (None, '', [], {}):
                                    continue
                                if isinstance(value, list):
                                    value_repr = ", ".join(str(item) for item in value if str(item).strip())
                                elif isinstance(value, dict):
                                    value_repr = json.dumps(value, ensure_ascii=False)
                                else:
                                    value_repr = str(value)
                                if value_repr:
                                    fragments.append(f"{key}: {value_repr}")
                            preview_text = " | ".join(fragments) if fragments else annotation_payload
                        else:
                            preview_text = annotation_payload
                    if preview_text:
                        preview_entry['annotation_preview'] = preview_text[:max_chars]
                    preview_samples.append(preview_entry)

        summary['preview_samples'] = preview_samples

        return summary

    async def annotate_async(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async wrapper for annotation (required by pipeline controller).

        Parameters
        ----------
        config : dict
            Annotation configuration

        Returns
        -------
        dict
            Annotation results
        """
        # Run the sync annotate method in executor to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.annotate, config)

    def calculate_sample_size(self, total: int, confidence: float = 0.95,
                            margin_error: float = 0.05, proportion: float = 0.5) -> int:
        """
        Calculate sample size for given confidence interval.
        
        Parameters
        ----------
        total : int
            Total population size
        confidence : float
            Confidence level (default 0.95 for 95% CI)
        margin_error : float
            Margin of error (default 0.05 for 5%)
        proportion : float
            Expected proportion (default 0.5 for maximum variability)
        
        Returns
        -------
        int
            Required sample size
        """
        # Z-score for confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)
        
        # Sample size formula
        numerator = (z**2) * proportion * (1 - proportion)
        denominator = margin_error**2
        
        sample_size = (numerator / denominator) / (
            1 + ((numerator / denominator - 1) / total)
        )
        
        return min(math.ceil(sample_size), total)

    def analyze_text_with_model(
        self,
        text: str,
        prompt: str,
        model_config: Dict[str, Any],
        schema: Optional[BaseModel] = None,
        context_formatted: bool = False
    ) -> Optional[str]:
        """
        Analyze text using configured model.

        Parameters
        ----------
        text : str
            Text to analyze (may include context window formatting)
        prompt : str
            Prompt template
        model_config : dict
            Model configuration
        schema : BaseModel, optional
            Pydantic schema for validation
        context_formatted : bool, optional
            If True, text already contains context window formatting with labels.
            The "Text to analyze:" header will not be added. Default: False.

        Returns
        -------
        str or None
            JSON string response or None on failure
        """
        # Build full prompt
        if context_formatted:
            # Text already contains context labels formatting
            full_prompt = f"{prompt}\n\n{text}"
        else:
            full_prompt = f"{prompt}\n\nText to analyze:\n{text}"

        self.logger.debug(f"[ANALYZE] Calling model with prompt length: {len(full_prompt)}")
        self.logger.debug(f"[ANALYZE] Text to analyze (first 200 chars): {text[:200]}")

        # Call appropriate model
        provider = model_config.get('provider')

        if provider in ['openai', 'anthropic', 'google'] and self.api_client:
            self.logger.debug(f"[ANALYZE] Using API client for provider: {provider}")
            response = self.api_client.generate(
                prompt=full_prompt,
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens', 1000)
            )
        elif provider in ['ollama', 'llamacpp'] and self.local_client:
            self.logger.debug(f"[ANALYZE] Using local client for provider: {provider}")
            response = self.local_client.generate(
                prompt=full_prompt,
                options=model_config.get('options', {})
            )
        else:
            self.logger.error(f"No client configured for provider: {provider}")
            return None

        self.logger.debug(f"[ANALYZE] Raw response from model: {response}")

        if not response:
            warning_msg = "[ANALYZE] Model returned empty response"
            # Only show via progress manager if available, otherwise log
            if self.progress_manager:
                self.progress_manager.show_warning(warning_msg)
            else:
                self.logger.warning(warning_msg)
            return None

        # Clean and validate response
        # Extract expected keys from schema if available
        expected_keys = []
        if schema:
            # Use model_fields for Pydantic V2, fallback to __fields__ for V1
            if hasattr(schema, 'model_fields'):
                expected_keys = list(schema.model_fields.keys())
            elif hasattr(schema, '__fields__'):
                expected_keys = list(schema.__fields__.keys())

        self.logger.debug(f"[ANALYZE] Cleaning JSON with expected keys: {expected_keys}")
        cleaned = clean_json_output(response, expected_keys)
        self.logger.debug(f"[ANALYZE] Cleaned JSON: {cleaned}")
        
        # Validate with schema if provided
        if cleaned and schema:
            self.logger.debug("[ANALYZE] Attempting schema validation")
            try:
                validated = schema.model_validate_json(cleaned)
                final_result = validated.model_dump_json()
                self.logger.debug(f"[ANALYZE] Schema validated successfully: {final_result}")
                return final_result
            except Exception as e:
                self.logger.warning(f"[ANALYZE] Schema validation failed: {e}, returning cleaned JSON")
                return cleaned

        self.logger.debug(f"[ANALYZE] Returning final result: {cleaned}")
        return cleaned


# =============================================================================
# Context Window Helper Functions
# =============================================================================

def _build_context_text(
    data: pd.DataFrame,
    current_idx: Any,
    text_column: str,
    context_config: Dict[str, Any]
) -> Tuple[str, str, str]:
    """
    Build context text from neighboring rows in the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Full DataFrame containing all rows
    current_idx : Any
        Index of current row being annotated
    text_column : str
        Column name containing text to extract
    context_config : dict
        Context window configuration with 'sentences_before' and 'sentences_after'

    Returns
    -------
    tuple
        (context_before_text, current_text, context_after_text)
    """
    sentences_before = context_config.get('sentences_before', 0)
    sentences_after = context_config.get('sentences_after', 0)

    # Get row indices as a list to find positions
    all_indices = list(data.index)
    try:
        current_position = all_indices.index(current_idx)
    except ValueError:
        # Current index not found, return just the current text
        current_text = ""
        if current_idx in data.index:
            val = data.loc[current_idx, text_column]
            current_text = str(val) if pd.notna(val) else ""
        return "", current_text, ""

    # Get context BEFORE (from oldest to most recent)
    context_before_texts = []
    for i in range(sentences_before, 0, -1):
        prev_position = current_position - i
        if prev_position >= 0:
            prev_idx = all_indices[prev_position]
            prev_text = data.loc[prev_idx, text_column]
            if pd.notna(prev_text):
                context_before_texts.append(str(prev_text))

    # Get current text
    current_val = data.loc[current_idx, text_column]
    current_text = str(current_val) if pd.notna(current_val) else ""

    # Get context AFTER
    context_after_texts = []
    for i in range(1, sentences_after + 1):
        next_position = current_position + i
        if next_position < len(all_indices):
            next_idx = all_indices[next_position]
            next_text = data.loc[next_idx, text_column]
            if pd.notna(next_text):
                context_after_texts.append(str(next_text))

    return (
        "\n".join(context_before_texts),
        current_text,
        "\n".join(context_after_texts)
    )


def _translate_context_labels(
    prompt_text: str,
    default_labels: Dict[str, str],
    model_config: Dict[str, Any],
    api_client: Any,
    local_client: Any,
    logger: logging.Logger
) -> Tuple[Dict[str, str], str]:
    """
    Detect prompt language and translate context labels if needed.

    Uses the same LLM as annotation for translation to maintain consistency.

    Parameters
    ----------
    prompt_text : str
        The annotation prompt content (used for language detection)
    default_labels : dict
        Default English labels dictionary
    model_config : dict
        Model configuration for LLM calls
    api_client : Any
        API client instance (or None)
    local_client : Any
        Local client instance (or None)
    logger : logging.Logger
        Logger instance

    Returns
    -------
    tuple
        (translated_labels_dict, detected_language_code)
    """
    # Import language detector
    try:
        from ..utils.language_detector import LanguageDetector
    except ImportError:
        logger.warning("[CONTEXT] Language detector not available, using default labels")
        return default_labels.copy(), 'en'

    # Detect prompt language (use first 500 chars for speed)
    detector = LanguageDetector()
    detection_result = detector.detect(prompt_text[:500] if len(prompt_text) > 500 else prompt_text)
    detected_lang = detection_result.get('language', 'en')
    confidence = detection_result.get('confidence', 0.0)

    logger.debug(f"[CONTEXT] Detected prompt language: {detected_lang} (confidence: {confidence:.2f})")

    # If English or low confidence, use defaults
    if detected_lang == 'en' or confidence < 0.7:
        return default_labels.copy(), 'en'

    # Language names for translation prompt
    lang_names = {
        'fr': 'French', 'es': 'Spanish', 'de': 'German',
        'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch',
        'ru': 'Russian', 'zh': 'Chinese', 'ja': 'Japanese',
        'ar': 'Arabic', 'ko': 'Korean', 'pl': 'Polish',
        'sv': 'Swedish', 'da': 'Danish', 'no': 'Norwegian',
        'fi': 'Finnish', 'cs': 'Czech', 'hu': 'Hungarian',
        'ro': 'Romanian', 'tr': 'Turkish', 'uk': 'Ukrainian'
    }
    target_language = lang_names.get(detected_lang, 'the same language as the prompt')

    # Translate each label using the annotation LLM
    translated_labels = {}
    provider = model_config.get('provider')

    for key, english_text in default_labels.items():
        translation_prompt = f"""Translate the following text to {target_language}.
If it's already in {target_language}, return it unchanged.
Return ONLY the translation, nothing else. Do not add quotes or explanations.

Text: {english_text}

{target_language} translation:"""

        try:
            response = None
            if provider in ['openai', 'anthropic', 'google'] and api_client:
                response = api_client.generate(
                    prompt=translation_prompt,
                    temperature=0.3,
                    max_tokens=100
                )
            elif provider in ['ollama', 'llamacpp'] and local_client:
                response = local_client.generate(
                    prompt=translation_prompt,
                    options={'temperature': 0.3, 'num_predict': 100}
                )

            if response:
                # Clean the response
                cleaned = response.strip().strip('"\'')
                translated_labels[key] = cleaned if cleaned else english_text
            else:
                translated_labels[key] = english_text

        except Exception as e:
            logger.warning(f"[CONTEXT] Translation failed for '{key}': {e}")
            translated_labels[key] = english_text

    logger.debug(f"[CONTEXT] Translated labels to {target_language}: {translated_labels}")
    return translated_labels, detected_lang


def _format_context_prompt(
    context_before: str,
    current_text: str,
    context_after: str,
    labels: Dict[str, str]
) -> str:
    """
    Format the context block with translated labels.

    Creates the structured context section to append to the prompt.

    Parameters
    ----------
    context_before : str
        Text from preceding rows (newline-separated)
    current_text : str
        Current row text to annotate
    context_after : str
        Text from following rows (newline-separated)
    labels : dict
        Translated (or default) labels dictionary

    Returns
    -------
    str
        Formatted context block ready to append to prompt
    """
    sections = []

    # Add context BEFORE section if present
    if context_before.strip():
        sections.append(f"{labels['context_before']}\n\n{context_before}")

    # Add main sentence to annotate (always present)
    sections.append(f"{labels['sentence_to_annotate']}\n\n{current_text}")

    # Add context AFTER section if present
    if context_after.strip():
        sections.append(f"{labels['context_after']}\n\n{context_after}")

    return "\n\n".join(sections)


# =============================================================================
# End of Context Window Helper Functions
# =============================================================================


def process_single_prompt(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single prompt for annotation.

    Parameters
    ----------
    task : dict
        Task configuration including row data, prompt, and options

    Returns
    -------
    dict
        Annotation result with identifier, JSON, and timing
    """
    # Setup logging for this function
    logger = logging.getLogger(__name__)

    start_time = time.perf_counter()

    # Extract task parameters
    row = task['row']
    prompt_config = task['prompts'][0]  # Single prompt
    text_columns = task['text_columns']
    identifier = task['identifier']
    model_config = task['model_config']
    options = task.get('options', {})

    logger.debug(f"[PROCESS] Starting annotation for identifier: {identifier}")

    # Check for context window configuration
    context_config = task.get('context_window_config')
    context_enabled = context_config and context_config.get('enabled', False)

    # Build text from columns
    text_parts = []
    for col in text_columns:
        if pd.notna(row[col]):
            text_parts.append(str(row[col]))
    current_text = "\n\n".join(text_parts)

    logger.debug(f"[PROCESS] Built text from {len(text_columns)} columns, length: {len(current_text)}")

    # Apply context window if enabled
    if context_enabled:
        context_before = task.get('context_before', '')
        context_after = task.get('context_after', '')

        # Get or translate labels (cached in context_config)
        labels = context_config.get('labels')
        if not labels:
            labels = DEFAULT_CONTEXT_LABELS.copy()
            logger.debug("[PROCESS] Using default context labels")

        # Format the text with context
        text = _format_context_prompt(
            context_before=context_before,
            current_text=current_text,
            context_after=context_after,
            labels=labels
        )
        logger.debug(f"[PROCESS] Context window applied: {len(context_before)} chars before, {len(context_after)} chars after")
    else:
        text = current_text

    logger.debug(f"[PROCESS] Final text length: {len(text)}")

    # Get prompt details (handle both 'prompt' and 'template' keys)
    prompt_text = prompt_config.get('prompt') or prompt_config.get('template', '')
    expected_keys = prompt_config.get('expected_keys', [])
    prefix = prompt_config.get('prefix', '')

    logger.debug(f"[PROCESS] Prompt length: {len(prompt_text)}, expected keys: {expected_keys}, prefix: {prefix}")

    # Build schema if expected keys provided
    schema = None
    if expected_keys and not options.get('disable_schema', False):
        schema = build_dynamic_schema(expected_keys)
        logger.debug(f"[PROCESS] Built dynamic schema for keys: {expected_keys}")

    # Create annotator instance for this process
    annotator = LLMAnnotator()

    # Setup the model client for this process
    config_for_setup = {
        'model': model_config.get('model_name'),
        'provider': model_config.get('provider', 'ollama'),
        'api_key': model_config.get('api_key')
    }
    logger.debug(f"[PROCESS] Setting up model client: {config_for_setup}")
    annotator._setup_model_client(config_for_setup)

    # Analyze text
    logger.debug(f"[PROCESS] Calling analyze_text_with_model (context_formatted={context_enabled})")
    result = annotator.analyze_text_with_model(
        text=text,
        prompt=prompt_text,
        model_config=model_config,
        schema=schema,
        context_formatted=context_enabled
    )

    logger.debug(f"[PROCESS] Got result from analyze_text_with_model: {result}")

    # Apply prefix if specified
    if result and prefix:
        logger.debug(f"[PROCESS] Applying prefix '{prefix}' to result")
        try:
            parsed = json.loads(result)
            prefixed = {f"{prefix}_{k}": v for k, v in parsed.items()}
            result = json.dumps(prefixed, ensure_ascii=False)
            logger.debug(f"[PROCESS] Prefixed result: {result}")
        except Exception as e:
            logger.warning(f"[PROCESS] Failed to apply prefix: {e}")

    elapsed = time.perf_counter() - start_time
    logger.debug(f"[PROCESS] Completed annotation for {identifier} in {elapsed:.2f}s, result: {result}")
    
    return {
        'identifier': identifier,
        'final_json': result,
        'inference_time': elapsed,
        'status': 'success' if result else 'error'
    }


def process_multiple_prompts(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process multiple prompts and merge results.
    
    Parameters
    ----------
    task : dict
        Task configuration with multiple prompts
    
    Returns
    -------
    dict
        Merged annotation results
    """
    start_time = time.perf_counter()
    
    # Extract task parameters
    row = task['row']
    prompts = task['prompts']
    text_columns = task['text_columns']
    identifier = task['identifier']
    model_config = task['model_config']
    options = task.get('options', {})

    # Check for context window configuration
    context_config = task.get('context_window_config')
    context_enabled = context_config and context_config.get('enabled', False)

    # Build text from columns
    text_parts = []
    for col in text_columns:
        if pd.notna(row[col]):
            text_parts.append(str(row[col]))
    current_text = "\n\n".join(text_parts)

    # Apply context window if enabled
    if context_enabled:
        context_before = task.get('context_before', '')
        context_after = task.get('context_after', '')

        labels = context_config.get('labels')
        if not labels:
            labels = DEFAULT_CONTEXT_LABELS.copy()

        text = _format_context_prompt(
            context_before=context_before,
            current_text=current_text,
            context_after=context_after,
            labels=labels
        )
    else:
        text = current_text
    
    # Process each prompt
    raw_dict = {}
    cleaned_dict = {}
    status_dict = {}
    collected_json_objects = []
    
    annotator = LLMAnnotator()

    # Setup the model client for this process
    config_for_setup = {
        'model': model_config.get('model_name'),
        'provider': model_config.get('provider', 'ollama'),
        'api_key': model_config.get('api_key')
    }
    annotator._setup_model_client(config_for_setup)

    for idx, prompt_config in enumerate(prompts, 1):
        prompt_text = prompt_config.get('prompt') or prompt_config.get('template', '')
        expected_keys = prompt_config.get('expected_keys', [])
        prefix = prompt_config.get('prefix', '')
        
        # Build schema
        schema = None
        if expected_keys and not options.get('disable_schema', False):
            schema = build_dynamic_schema(expected_keys)
        
        # Analyze with this prompt
        result = annotator.analyze_text_with_model(
            text=text,
            prompt=prompt_text,
            model_config=model_config,
            schema=schema,
            context_formatted=context_enabled
        )
        
        raw_dict[str(idx)] = result
        
        if result:
            cleaned_dict[str(idx)] = result
            status_dict[str(idx)] = 'success'
            
            # Apply prefix and collect
            try:
                parsed = json.loads(result)
                if prefix:
                    parsed = {f"{prefix}_{k}": v for k, v in parsed.items()}
                collected_json_objects.append(parsed)
            except:
                status_dict[str(idx)] = 'parse_error'
        else:
            cleaned_dict[str(idx)] = None
            status_dict[str(idx)] = 'error'
    
    # Merge all JSON objects
    merged = {}
    for obj in collected_json_objects:
        if isinstance(obj, dict):
            merged.update(obj)
    
    final_json = json.dumps(merged, ensure_ascii=False) if merged else None
    elapsed = time.perf_counter() - start_time
    
    return {
        'identifier': identifier,
        'final_json': final_json,
        'inference_time': elapsed,
        'raw_json': json.dumps(raw_dict, ensure_ascii=False),
        'cleaned_json': json.dumps(cleaned_dict, ensure_ascii=False),
        'status': json.dumps(status_dict, ensure_ascii=False)
    }


def build_dynamic_schema(expected_keys: List[str]) -> BaseModel:
    """
    Build dynamic Pydantic schema from expected keys.
    
    Parameters
    ----------
    expected_keys : list
        List of expected JSON keys
    
    Returns
    -------
    BaseModel
        Dynamic Pydantic model
    """
    fields = {}
    for key in expected_keys:
        # Make all fields optional strings for flexibility
        fields[key] = (Optional[Union[str, int, float, bool, list, dict]], None)
    
    return create_model('DynamicAnnotationSchema', **fields)
