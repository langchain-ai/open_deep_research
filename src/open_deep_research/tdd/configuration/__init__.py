"""
Configuration for the Technical Due Diligence (TDD) process.

This package contains the configuration classes used to customize
the behavior of the TDD agents and tools.
"""

from open_deep_research.tdd.configuration.config import (
    TDDConfiguration, AgentConfiguration, ToolConfiguration
)

__all__ = [
    "TDDConfiguration",
    "AgentConfiguration",
    "ToolConfiguration",
]
