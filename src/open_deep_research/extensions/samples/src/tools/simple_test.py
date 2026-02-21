#!/usr/bin/env python
"""
Test script for the research agent with tool calling.
This script serves as a simple test to ensure the research agent works correctly.
"""

import os
import sys
import logging
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("research_agent_test")

# Import needed modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.graph import research_agent
from src.state import SummaryState

def run_test():
    """Run a simple test of the research agent."""
    logger.info("Starting research agent test...")
    logger.info("Testing research agent with tool calling...")
    
    # Create a summary state
    summary_state = SummaryState(
        research_topic="Latest advancements in large language models",
        research_loop_count=0
    )
    
    # Call the research agent
    logger.info(f"Calling research agent with topic: {summary_state.research_topic}")
    
    # Create a simple configuration
    config = {
        "callbacks": {
            "on_event": lambda event_type, data: logger.info(f"Event: {event_type} - {json.dumps(data)}")
        }
    }
    
    # Execute the agent
    result = research_agent(summary_state, config)
    
    # Log the results
    logger.info("Research completed. Results summary:")
    logger.info(f"- Sources count: {len(result.get('formatted_sources', []))}")
    logger.info(f"- Tools used: {result.get('tools', [])}")
    logger.info(f"- Domains count: {len(result.get('domains', []))}")
    
    logger.info("Test completed.")

def run_consolidated_test():
    """Run a test of the consolidated tool calling approach."""
    logger.info("Starting consolidated tool calling test...")
    
    # Create a summary state with a biographical query to test various tool types
    summary_state = SummaryState(
        research_topic="Who is Caiming Xiong and what are his contributions to AI research?",
        research_loop_count=0
    )
    
    # Call the research agent
    logger.info(f"Calling research agent with biographical topic: {summary_state.research_topic}")
    
    # Create a simple configuration
    config = {
        "callbacks": {
            "on_event": lambda event_type, data: logger.info(f"Event: {event_type} - {json.dumps(data, default=str)}")
        }
    }
    
    # Execute the agent
    result = research_agent(summary_state, config)
    
    # Log the results
    logger.info("Biographical research completed. Results summary:")
    logger.info(f"- Sources count: {len(result.get('formatted_sources', []))}")
    tools = result.get('tools', [])
    logger.info(f"- Tools used: {tools}")
    logger.info(f"- Domains count: {len(result.get('domains', []))}")
    
    # Verify that the results include appropriate tools
    if 'linkedin_search' in tools:
        logger.info("[PASS] Test passed: linkedin_search tool was used for biographical research")
    else:
        logger.warning("[FAIL] Test warning: linkedin_search tool was not used for biographical research")
        
    if 'academic_search' in tools:
        logger.info("[PASS] Test passed: academic_search tool was used for contributions research")
    else:
        logger.warning("[FAIL] Test warning: academic_search tool was not used for contributions research")
    
    logger.info("Consolidated tool calling test completed.")

if __name__ == "__main__":
    logger.info("Running research agent tests...")
    
    # Run both tests
    run_test()
    run_consolidated_test() 