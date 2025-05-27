"""Planning, research, and report generation with Technical Due Diligence capabilities."""

__version__ = "0.1.0"

# Core functionality
from open_deep_research.multi_agent import run_research
from open_deep_research.configuration import Configuration

# TDD functionality
try:
    from open_deep_research.tdd.run import run_tdd
    from open_deep_research.tdd.configuration import TDDConfiguration
    from open_deep_research.tdd.integration import TDDResearchAdapter
    
    # Flag to indicate TDD is available
    TDD_AVAILABLE = True
except ImportError:
    # TDD module might not be available
    TDD_AVAILABLE = False