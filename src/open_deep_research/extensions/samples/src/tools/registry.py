"""
Tool registry for managing and accessing search tools.

This module provides a registry system for search tools used by the research agent.
It allows tools to be registered, retrieved, and managed in a central location.
"""

import logging
import traceback
from typing import Dict, List, Optional, Any

from src.tools.search_tools import (
    GeneralSearchTool,
    AcademicSearchTool,
    GithubSearchTool,
    LinkedinSearchTool
)
from src.tools.text2sql_tool import Text2SQLTool

logger = logging.getLogger(__name__)

class SearchToolRegistry:
    """Registry for search tools that can be used by the research agent."""
    
    def __init__(self, config=None):
        """
        Initialize the search tool registry.
        
        Args:
            config: Configuration object to pass to tools
        """
        logger.info(f"[SearchToolRegistry.__init__] Initializing registry with config type: {type(config).__name__ if config else 'None'}")
        self.config = config
        self.tools = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register the default set of search tools."""
        logger.info("Registering default search tools")
        
        try:
            # Initialize tools with config
            logger.info(f"[SearchToolRegistry._register_default_tools] Creating tool instances with config")
            
            general_search = GeneralSearchTool()
            general_search.config = self.config
            logger.info(f"[SearchToolRegistry._register_default_tools] Created GeneralSearchTool instance")
            
            github_search = GithubSearchTool()
            github_search.config = self.config
            logger.info(f"[SearchToolRegistry._register_default_tools] Created GithubSearchTool instance")
            
            academic_search = AcademicSearchTool()
            academic_search.config = self.config
            logger.info(f"[SearchToolRegistry._register_default_tools] Created AcademicSearchTool instance")
            
            linkedin_search = LinkedinSearchTool()
            linkedin_search.config = self.config
            logger.info(f"[SearchToolRegistry._register_default_tools] Created LinkedinSearchTool instance")
            
            text2sql = Text2SQLTool()
            text2sql.config = self.config
            logger.info(f"[SearchToolRegistry._register_default_tools] Created Text2SQLTool instance")
            
            # Add tools to registry
            logger.info(f"[SearchToolRegistry._register_default_tools] Registering tools in registry")
            self.register_tool(general_search)
            self.register_tool(github_search)
            self.register_tool(academic_search)
            self.register_tool(linkedin_search)
            self.register_tool(text2sql)
            
            logger.info(f"Registered {len(self.tools)} default search tools")
            logger.info(f"[SearchToolRegistry._register_default_tools] Registered tools: {list(self.tools.keys())}")
        except Exception as e:
            logger.error(f"Error registering default tools: {str(e)}")
            logger.error(f"[SearchToolRegistry._register_default_tools] Traceback: {traceback.format_exc()}")
            # Re-raise the exception to help with debugging
            raise
    
    def register_tool(self, tool):
        """
        Register a new tool with the registry.
        
        Args:
            tool: The tool to register
        """
        logger.info(f"[SearchToolRegistry.register_tool] Registering tool: {tool.name}, class: {tool.__class__.__name__}")
        # Ensure the tool has the config attribute
        if hasattr(tool, 'config') and tool.config is None:
            tool.config = self.config
            logger.info(f"[SearchToolRegistry.register_tool] Set config on tool: {tool.name}")
        self.tools[tool.name] = tool
        logger.info(f"[SearchToolRegistry.register_tool] Tool {tool.name} successfully registered")
        
    def get_tool(self, tool_name):
        """
        Get a tool by name.
        
        Args:
            tool_name: The name of the tool to retrieve
            
        Returns:
            The requested tool or None if not found
        """
        logger.info(f"[SearchToolRegistry.get_tool] Retrieving tool: {tool_name}")
        tool = self.tools.get(tool_name)
        if not tool:
            logger.warning(f"Tool not found: {tool_name}")
            logger.info(f"[SearchToolRegistry.get_tool] Available tools: {list(self.tools.keys())}")
            return None
        
        logger.info(f"[SearchToolRegistry.get_tool] Retrieved tool {tool_name}, class: {tool.__class__.__name__}")
        return tool
    
    def get_all_tools(self):
        """
        Get all registered tools.
        
        Returns:
            List of all registered tools
        """
        logger.info(f"[SearchToolRegistry.get_all_tools] Returning all {len(self.tools)} registered tools")
        return list(self.tools.values())
    
    def get_tool_description(self, tool_name):
        """
        Get the description of a tool.
        
        Args:
            tool_name: The name of the tool
            
        Returns:
            The description of the tool or None if the tool is not found
        """
        logger.info(f"[SearchToolRegistry.get_tool_description] Getting description for tool: {tool_name}")
        tool = self.get_tool(tool_name)
        if tool and hasattr(tool, 'description'):
            logger.info(f"[SearchToolRegistry.get_tool_description] Found description for {tool_name}")
            return tool.description
        logger.info(f"[SearchToolRegistry.get_tool_description] No description found for {tool_name}")
        return None
    
    def get_all_tool_descriptions(self):
        """
        Get descriptions of all registered tools.
        
        Returns:
            Dict mapping tool names to their descriptions
        """
        logger.info(f"[SearchToolRegistry.get_all_tool_descriptions] Getting descriptions for all tools")
        return {
            name: tool.description if hasattr(tool, 'description') else "No description"
            for name, tool in self.tools.items()
        } 