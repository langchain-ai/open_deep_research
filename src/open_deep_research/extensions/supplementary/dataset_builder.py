#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
dataset_builder.py

MAIN OBJECTIVE:
---------------
Maintain backward compatibility by re-exporting the training dataset builder
abstractions under their historical module path.

Dependencies:
-------------
- llm_tool.trainers.training_data_builder

MAIN FEATURES:
--------------
1) Expose TrainingDatasetBuilder without forcing import refactors
2) Re-export TrainingDataBundle for downstream configuration helpers
3) Provide TrainingDataRequest in legacy locations for CLI consumers
4) Keep __all__ aligned with the canonical builder module
5) Simplify gradual migration by centralising the compatibility shim

Author:
-------
Antoine Lemor
"""

from llm_tool.trainers.training_data_builder import (
    TrainingDatasetBuilder,
    TrainingDataBundle,
    TrainingDataRequest,
)

__all__ = [
    "TrainingDatasetBuilder",
    "TrainingDataBundle",
    "TrainingDataRequest",
]
