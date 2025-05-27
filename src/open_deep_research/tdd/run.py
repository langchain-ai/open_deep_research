"""
Run functions for the Technical Due Diligence (TDD) Agent System.

This module provides functions for running the TDD process.
"""

import logging
import asyncio
from typing import Dict, Any, Optional

from langchain_core.messages import HumanMessage

from open_deep_research.tdd.configuration import TDDConfiguration
from open_deep_research.tdd.graph import create_tdd_graph
from open_deep_research.tdd.vdr import VirtualDataRoom

logger = logging.getLogger(__name__)

async def run_tdd(query: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the TDD process with the given query and configuration.
    
    Args:
        query: The query to run TDD for
        config: Optional configuration for the TDD process
        
    Returns:
        The results of the TDD process
    """
    logger.info(f"Running TDD for query: {query}")
    
    # Create configuration if not provided
    if config is None:
        config = {}
    
    # Create TDD configuration
    tdd_config = TDDConfiguration(**config)
    
    # Create the TDD graph
    graph = create_tdd_graph(config)
    
    # Initialize the state
    state = {
        "query": query,
        "config": config,
        "messages": [
            HumanMessage(content=f"Conduct a technical due diligence on: {query}")
        ]
    }
    
    # Run the graph
    try:
        result = await graph.ainvoke(state)
        logger.info("TDD process completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error running TDD process: {e}")
        raise
