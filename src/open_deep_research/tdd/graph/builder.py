"""
Graph builder for the Technical Due Diligence (TDD) Agent System.

This module defines the function to create the TDD graph, which is used
to coordinate the execution of the TDD process.
"""

import logging
from typing import Dict, Any, Optional

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END

from open_deep_research.tdd.state.graph_state import TDDGraphState
from open_deep_research.tdd.graph.nodes import (
    initialize, run_tdd_planner, run_domain_agent, run_reflection, run_writer
)
from open_deep_research.tdd.graph.edges import (
    should_return_to_planner, should_continue_to_next_domain, should_proceed_to_writer
)

logger = logging.getLogger(__name__)

def create_tdd_graph(config: Optional[Dict[str, Any]] = None) -> StateGraph:
    """
    Create the TDD graph.
    
    This function creates a StateGraph for the TDD process, defining the
    nodes and edges that determine the flow of execution.
    
    Args:
        config: Optional configuration for the TDD process
        
    Returns:
        A StateGraph for the TDD process
    """
    logger.info("Creating TDD graph")
    
    # Create a new graph
    graph = StateGraph(TDDGraphState)
    
    # Add nodes to the graph
    
    # Initialization node
    graph.add_node("initialize", initialize)
    
    # TDD Planner node
    graph.add_node("tdd_planner", run_tdd_planner)
    
    # Domain agent node
    graph.add_node("domain_agent", run_domain_agent)
    
    # Reflection node to evaluate progress
    graph.add_node("reflection", run_reflection)
    
    # Final report writer
    graph.add_node("writer", run_writer)
    
    # End node
    graph.add_node("end", lambda state: state)
    
    # Add edges to the graph
    
    # Start -> Initialize -> TDD Planner
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "tdd_planner")
    
    # TDD Planner -> Domain Agent
    graph.add_edge("tdd_planner", "domain_agent")
    
    # Domain Agent -> Reflection
    graph.add_edge("domain_agent", "reflection")
    
    # Reflection -> TDD Planner, Domain Agent, or Writer
    graph.add_conditional_edges(
        "reflection",
        should_return_to_planner,
        {
            "tdd_planner": "tdd_planner",
            "domain_agent": "domain_agent",
            "writer": "writer"
        }
    )
    
    # Writer -> End
    graph.add_edge("writer", "end")
    
    # Set the entry point
    graph.set_entry_point("initialize")
    
    logger.info("TDD graph created successfully")
    return graph
