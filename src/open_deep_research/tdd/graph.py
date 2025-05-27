"""
Re-export of graph components for backward compatibility with LangGraph Studio.
"""

from open_deep_research.tdd.graph.builder import create_tdd_graph
from open_deep_research.tdd.graph.nodes import (
    initialize, run_tdd_planner, run_domain_agent,
    run_reflection, run_writer
)
from open_deep_research.tdd.graph.edges import (
    should_return_to_planner, should_continue_to_next_domain,
    should_proceed_to_writer
)

# Create the TDD graph for LangGraph Studio
from open_deep_research.tdd.configuration import TDDConfiguration
graph = create_tdd_graph(TDDConfiguration())
