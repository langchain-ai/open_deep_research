"""
Test script for the tool calling mechanism.

This script tests the search tools, registry, and executor implementations.
"""

import logging
import sys
import json
from src.tools import (
    SearchToolRegistry,
    ToolExecutor,
    GeneralSearchTool,
    AcademicSearchTool,
    GithubSearchTool,
    LinkedinSearchTool
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("test_tools")

def test_tool_registry():
    """Test the tool registry implementation."""
    logger.info("Testing tool registry...")
    
    # Create a registry
    registry = SearchToolRegistry()
    
    # Get tool descriptions
    descriptions = registry.get_tool_descriptions()
    logger.info(f"Registered tools: {json.dumps(descriptions, indent=2)}")
    
    # Get a specific tool
    tool = registry.get_tool("general_search")
    logger.info(f"Retrieved tool: {tool.name} - {tool.description}")
    
    logger.info("Tool registry test completed successfully.")

def test_tool_executor():
    """Test the tool executor implementation."""
    logger.info("Testing tool executor...")
    
    # Create a registry and executor
    registry = SearchToolRegistry()
    executor = ToolExecutor(registry)
    
    # Execute a tool
    query = "python langgraph framework"
    logger.info(f"Executing general_search tool with query: {query}")
    
    result = executor.execute_tool_sync("general_search", {"query": query})
    
    # Print results
    logger.info(f"Tool execution result - Formatted sources count: {len(result.get('formatted_sources', []))}")
    logger.info(f"Tool execution result - Search string length: {len(result.get('search_string', ''))}")
    logger.info(f"Tool execution result - Tools used: {result.get('tools', [])}")
    logger.info(f"Tool execution result - Domains count: {len(result.get('domains', []))}")
    
    logger.info("Tool executor test completed successfully.")

if __name__ == "__main__":
    logger.info("Starting tool tests...")
    
    # Run tests
    test_tool_registry()
    print("\n" + "="*80 + "\n")
    test_tool_executor()
    
    logger.info("All tests completed.") 