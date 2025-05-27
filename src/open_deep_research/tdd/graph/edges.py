"""
Edge conditions for the Technical Due Diligence (TDD) Agent System.

This module defines the edge conditions used in the TDD graph to determine
the flow of execution between nodes.
"""

import logging
from typing import Dict, List, Any, Literal

from open_deep_research.tdd.state.graph_state import TDDGraphState

logger = logging.getLogger(__name__)

def should_return_to_planner(state: TDDGraphState) -> Literal["tdd_planner", "domain_agent", "writer"]:
    """
    Determine whether to return to the TDD Planner, continue with Domain Agent, or proceed to the Writer.
    
    This function checks the reflection results to determine the next step in the TDD process.
    If replanning is needed, it returns "tdd_planner".
    If there are remaining domains to investigate, it returns "domain_agent".
    Otherwise, it returns "writer".
    
    Args:
        state: The current state of the TDD investigation
        
    Returns:
        "tdd_planner" if replanning is needed, "domain_agent" if there are remaining domains, "writer" otherwise
    """
    # Check if replanning is needed
    if state.get("replanning_needed", False):
        logger.info("Returning to TDD Planner for replanning")
        return "tdd_planner"
    
    # Check if there are remaining domains to investigate
    domains = state.get("domains", [])
    domain_results = state.get("domain_results", {})
    remaining_domains = [d for d in domains if d not in domain_results]
    
    if remaining_domains:
        logger.info(f"Continuing with domain agent for remaining domains: {remaining_domains}")
        return "domain_agent"
    else:
        logger.info("All domains completed, proceeding to Writer for final report")
        return "writer"

def should_continue_to_next_domain(state: TDDGraphState) -> Literal["domain_agent", "reflection"]:
    """
    Determine whether to continue to the next domain or proceed to reflection.
    
    This function checks if there are more domains to investigate. If there
    are, it returns "domain_agent", otherwise it returns "reflection".
    
    Args:
        state: The current state of the TDD investigation
        
    Returns:
        "domain_agent" if there are more domains to investigate, "reflection" otherwise
    """
    domains = state.get("domains", [])
    current_domain = state.get("current_domain", "")
    domain_results = state.get("domain_results", {})
    
    # If we've completed all domains, proceed to reflection
    if len(domain_results) >= len(domains):
        logger.info("All domains completed, proceeding to reflection")
        return "reflection"
    
    # If we're not on a valid domain, proceed to reflection
    if current_domain not in domains:
        logger.info(f"Current domain {current_domain} not in domains list, proceeding to reflection")
        return "reflection"
    
    # If we've already investigated the current domain, proceed to reflection
    if current_domain in domain_results:
        logger.info(f"Domain {current_domain} already investigated, proceeding to reflection")
        return "reflection"
    
    # Otherwise, continue with the current domain
    logger.info(f"Continuing with domain {current_domain}")
    return "domain_agent"

def should_proceed_to_writer(state: TDDGraphState) -> Literal["writer", "domain_agent"]:
    """
    Determine whether to proceed to the Writer or continue with domain agents.
    
    This function checks if all domains have been investigated and there are
    no significant issues requiring further investigation. If so, it returns
    "writer", otherwise it returns "domain_agent".
    
    Args:
        state: The current state of the TDD investigation
        
    Returns:
        "writer" if all domains have been investigated and there are no significant issues,
        "domain_agent" otherwise
    """
    domains = state.get("domains", [])
    domain_results = state.get("domain_results", {})
    reflection_summary = state.get("reflection_summary", {})
    
    # Check the recommendation from reflection
    recommendation = reflection_summary.get("recommendation", "")
    
    if "final report" in recommendation.lower() or "writer" in recommendation.lower():
        logger.info("Reflection recommends proceeding to Writer")
        return "writer"
    
    # If we've completed all domains, proceed to Writer
    if len(domain_results) >= len(domains):
        logger.info("All domains completed, proceeding to Writer")
        return "writer"
    
    # Otherwise, continue with domain agents
    logger.info("Continuing with domain agents")
    return "domain_agent"
