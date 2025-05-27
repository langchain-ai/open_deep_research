"""
Logging configuration for the Open Deep Research project.

This module provides a centralized configuration for logging across the project.
It defines log formats, levels, and handlers that can be used by all modules.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Define default log directory
DEFAULT_LOG_DIR = Path.home() / ".open_deep_research" / "logs"

def configure_logging(
    log_level: str = "INFO",
    log_file: bool = True,
    log_dir: Path = DEFAULT_LOG_DIR,
    max_file_size_mb: int = 10,
    backup_count: int = 5
) -> None:
    """
    Configure logging for the Open Deep Research project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Whether to log to a file in addition to console
        log_dir: Directory to store log files
        max_file_size_mb: Maximum size of log file in MB before rotation
        backup_count: Number of backup log files to keep
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler if requested
    if log_file:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handler
        log_file_path = log_dir / "open_deep_research.log"
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_file_path}")

def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging with a simpler interface for the TDD module.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    configure_logging(log_level=level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name for the logger
        
    Returns:
        A logger instance
    """
    return logging.getLogger(name)


# Configure module-level logger
logger = logging.getLogger(__name__)
