"""
Integration module for connecting the TDD Agent System with Open Deep Research.

This module provides functions and classes for integrating the TDD Agent System
with the core Open Deep Research framework.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import asyncio

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.multi_agent import supervisor, get_search_tool
from open_deep_research.configuration import Configuration
from open_deep_research.tdd.configuration import TDDConfiguration
from open_deep_research.tdd.run import run_tdd
from open_deep_research.tdd.state.graph_state import TDDReportState, DomainState
from open_deep_research.tdd.vdr import VirtualDataRoom

logger = logging.getLogger(__name__)

class TDDResearchAdapter:
    """Adapter for integrating TDD with Open Deep Research.
    
    This class provides methods for running TDD as part of a broader
    research process using the Open Deep Research framework.
    """
    
    def __init__(self, config: Optional[Union[Configuration, TDDConfiguration]] = None):
        """Initialize the adapter.
        
        Args:
            config: Configuration for the adapter. If None, a default
                   TDDConfiguration will be created.
        """
        self.config = config or TDDConfiguration()
        if isinstance(self.config, Configuration) and not isinstance(self.config, TDDConfiguration):
            # Convert base Configuration to TDDConfiguration
            config_dict = self.config.__dict__
            self.config = TDDConfiguration(**config_dict)
        
        self.vdr = VirtualDataRoom()
        logger.info("Initialized TDDResearchAdapter")
    
    async def run_tdd_research(self, query: str) -> Dict[str, Any]:
        """Run TDD research using the Open Deep Research framework.
        
        This method runs the TDD process and integrates the results with
        the broader research process.
        
        Args:
            query: The query to run TDD for
            
        Returns:
            The results of the TDD process
        """
        logger.info(f"Running TDD research for query: {query}")
        
        # Run the TDD process
        tdd_result = await run_tdd(query, self.config)
        
        # Convert TDD result to a format compatible with Open Deep Research
        research_result = {
            "title": f"Technical Due Diligence: {query}",
            "sections": []
        }
        
        # Add domain reports as sections
        for domain, report in tdd_result.get("domain_reports", {}).items():
            research_result["sections"].append({
                "title": report.title,
                "content": report.content
            })
        
        # Add final report as a section
        if "final_report" in tdd_result:
            research_result["sections"].append({
                "title": "Executive Summary",
                "content": tdd_result["final_report"]
            })
        
        logger.info("TDD research completed successfully")
        return research_result
    
    async def integrate_with_supervisor(self, query: str, config: RunnableConfig) -> Dict[str, Any]:
        """Integrate TDD with the Open Deep Research supervisor.
        
        This method allows the TDD process to be run as part of the
        Open Deep Research supervisor workflow.
        
        Args:
            query: The query to run TDD for
            config: The runnable configuration
            
        Returns:
            The results of the supervisor process
        """
        logger.info(f"Integrating TDD with supervisor for query: {query}")
        
        # Create a state for the supervisor
        from open_deep_research.multi_agent import ReportState
        state = ReportState(
            messages=[HumanMessage(content=query)],
            sections=[],
            title=""
        )
        
        # Run the supervisor
        supervisor_result = await supervisor(state, config)
        
        # Run the TDD process
        tdd_result = await self.run_tdd_research(query)
        
        # Merge the results
        merged_result = {
            "title": supervisor_result.get("title", ""),
            "sections": supervisor_result.get("sections", [])
        }
        
        # Add TDD sections
        for section in tdd_result.get("sections", []):
            merged_result["sections"].append(section)
        
        logger.info("Integration with supervisor completed successfully")
        return merged_result

def create_tdd_search_tool(config: TDDConfiguration):
    """Create a search tool that integrates with the TDD system.
    
    This function creates a search tool that can be used by the TDD agents
    to search for information using the Open Deep Research search tools.
    
    Args:
        config: The TDD configuration
        
    Returns:
        A search tool compatible with the TDD system
    """
    # Get the base search tool from Open Deep Research
    base_search_tool = get_search_tool(config)
    
    # Wrap it to make it compatible with the TDD system
    async def tdd_search_tool(query: str) -> List[Dict[str, str]]:
        """Search tool for TDD agents.
        
        Args:
            query: The search query
            
        Returns:
            Search results
        """
        logger.info(f"Running TDD search for query: {query}")
        
        # Run the base search tool
        search_results = await base_search_tool(query)
        
        # Convert to a format compatible with the TDD system
        tdd_results = []
        for result in search_results:
            tdd_results.append({
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "url": result.get("url", "")
            })
        
        logger.info(f"TDD search completed with {len(tdd_results)} results")
        return tdd_results
    
    return tdd_search_tool
