"""
Tool executor for dynamically executing search tools.

This module provides a tool executor that can execute tools from the registry
based on their name and parameters. It handles both synchronous and asynchronous tools.
"""

import logging
import asyncio
import inspect
import traceback
from typing import Dict, Any, Optional

from src.tools.registry import SearchToolRegistry

logger = logging.getLogger(__name__)

class ToolExecutor:
    """Executes tools based on their name and parameters."""
    
    def __init__(self, registry=None):
        """
        Initialize the tool executor.
        
        Args:
            registry: Tool registry to use, or create a new one if None
        """
        self.registry = registry or SearchToolRegistry()
        logger.info(f"ToolExecutor initialized with registry: {type(self.registry).__name__}")
    
    async def execute_tool(self, tool_name: str, params: Dict[str, Any], config: Optional[Dict[str, Any]] = None, raise_exceptions: bool = False) -> Any:
        """
        Execute a tool asynchronously with the given parameters and configuration.
        Handles LangChain tools (_arun, _run), functions, and lc_tool wrappers.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool
            config: Optional execution configuration (e.g., RunnableConfig for LangChain)
            raise_exceptions: Whether to raise exceptions or return them as error messages
            
        Returns:
            The result of the tool execution, or an error message if execution failed
            and raise_exceptions is False
        """
        logger.info(f"[ToolExecutor.execute_tool] Executing tool: {tool_name} with params: {params}, config: {config}")
        config = config or {} # Ensure config is at least an empty dict
        
        try:
            # Get the tool from the registry
            tool = self.registry.get_tool(tool_name)
            if not tool:
                error_msg = f"Tool not found: {tool_name}"
                logger.error(f"[ToolExecutor.execute_tool] {error_msg}")
                if raise_exceptions:
                    raise ValueError(error_msg)
                return {"error": error_msg}
            
            logger.info(f"[ToolExecutor.execute_tool] Tool found: {tool_name}, determining execution method...")

            # --- Execution Logic --- 
            result = None
            executed = False

            # 1. Check for LangChain BaseTool async method (_arun)
            if hasattr(tool, '_arun') and callable(tool._arun):
                logger.info(f"[ToolExecutor.execute_tool] Found LangChain async method _arun for {tool_name}.")
                sig = inspect.signature(tool._arun)
                if 'config' in sig.parameters or 'run_manager' in sig.parameters:
                    result = await tool._arun(**params, config=config) 
                else:
                    result = await tool._arun(**params)
                executed = True
            
            # 2. Check for standard async function attribute
            elif hasattr(tool, 'function') and asyncio.iscoroutinefunction(tool.function):
                logger.info(f"[ToolExecutor.execute_tool] Found standard async function for {tool_name}.")
                sig = inspect.signature(tool.function)
                if 'config' in sig.parameters:
                    result = await tool.function(**params, config=config)
                else:
                    result = await tool.function(**params)
                executed = True

            # 3. Check for LangChain lc_tool attribute (usually a Runnable)
            elif hasattr(tool, 'lc_tool') and tool.lc_tool and hasattr(tool.lc_tool, 'ainvoke'):
                logger.info(f"[ToolExecutor.execute_tool] Found LangChain lc_tool with ainvoke for {tool_name}.")
                result = await tool.lc_tool.ainvoke(params, config=config)
                executed = True

            # 4. Check for LangChain BaseTool sync method (_run) - run in thread
            elif hasattr(tool, '_run') and callable(tool._run):
                logger.info(f"[ToolExecutor.execute_tool] Found LangChain sync method _run for {tool_name}. Running in thread.")
                sig = inspect.signature(tool._run)
                if 'config' in sig.parameters or 'run_manager' in sig.parameters:
                    func_to_run = lambda p=params, c=config: tool._run(**p, config=c)
                else:
                    func_to_run = lambda p=params: tool._run(**p)
                result = await asyncio.to_thread(func_to_run)
                executed = True

            # 5. Check for standard sync function attribute - run in thread
            elif hasattr(tool, 'function') and callable(tool.function):
                logger.info(f"[ToolExecutor.execute_tool] Found standard sync function for {tool_name}. Running in thread.")
                sig = inspect.signature(tool.function)
                if 'config' in sig.parameters:
                    func_to_run = lambda p=params, c=config: tool.function(**p, config=c)
                else:
                    func_to_run = lambda p=params: tool.function(**p)
                result = await asyncio.to_thread(func_to_run)
                executed = True
            
            # 6. Fallback/legacy checks (less common)
            elif asyncio.iscoroutinefunction(tool): # Check if tool itself is an async func
                logger.warning(f"[ToolExecutor.execute_tool] Tool {tool_name} is a direct async function. Executing.")
                result = await tool(**params)
                executed = True
            elif hasattr(tool, 'search') and asyncio.iscoroutinefunction(tool.search):
                logger.warning(f"[ToolExecutor.execute_tool] Tool {tool_name} has async search method. Executing.")
                result = await tool.search(**params)
                executed = True
            elif hasattr(tool, 'search') and callable(tool.search):
                logger.warning(f"[ToolExecutor.execute_tool] Tool {tool_name} has sync search method. Running in thread.")
                func_to_run = lambda p=params: tool.search(**p)
                result = await asyncio.to_thread(func_to_run)
                executed = True
            
            # --- End Execution Logic --- 

            if not executed:
                error_msg = f"Tool {tool_name} could not be executed. No suitable execution method found (_arun, _run, function, lc_tool.ainvoke, search)."
                logger.error(f"[ToolExecutor.execute_tool] {error_msg}")
                if raise_exceptions:
                    raise TypeError(error_msg)
                return {"error": error_msg}
            
            logger.info(f"[ToolExecutor.execute_tool] Tool execution completed for: {tool_name}")
            return result
        
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(f"[ToolExecutor.execute_tool] {error_msg}")
            logger.error(f"[ToolExecutor.execute_tool] Traceback: {traceback.format_exc()}")
            
            if raise_exceptions:
                raise
            
            return {"error": error_msg}
    
    def execute_tool_sync(self, tool_name, parameters):
        """
        Execute a tool synchronously by name with the given parameters.
        This is a convenience method for synchronous execution.
        
        Args:
            tool_name: The name of the tool to execute
            parameters: Dictionary of parameters to pass to the tool
            
        Returns:
            The result of the tool execution
        """
        logger.info(f"Executing tool synchronously: {tool_name} with parameters: {parameters}")
        
        # Get the tool from the registry
        tool = self.registry.get_tool(tool_name)
        if not tool:
            logger.error(f"Unknown tool: {tool_name}")
            return {
                "formatted_sources": [],
                "search_string": parameters.get("query", ""),
                "tools": [tool_name],
                "domains": [],
                "citations": [],
                "error": f"Unknown tool: {tool_name}"
            }
        
        # Log tool details
        logger.info(f"Tool class: {tool.__class__.__name__}, Config present: {hasattr(tool, 'config') and tool.config is not None}")
        if hasattr(tool, 'config') and tool.config is not None:
            logger.info(f"Tool config keys: {list(tool.config.keys()) if isinstance(tool.config, dict) else 'Non-dict config'}")
        
        # Execute the tool
        try:
            # For synchronous tools
            if hasattr(tool, "_run"):
                logger.info(f"Executing synchronous tool: {tool_name} via _run")
                start_time = __import__('time').time()
                result = tool._run(**parameters)
                execution_time = __import__('time').time() - start_time
                logger.info(f"Tool {tool_name} execution completed in {execution_time:.2f}s with result keys: {list(result.keys()) if isinstance(result, dict) else 'Non-dict result'}")
                if isinstance(result, dict) and 'formatted_sources' in result:
                    sources_count = len(result['formatted_sources']) if isinstance(result['formatted_sources'], list) else 'string'
                    logger.info(f"Tool {tool_name} returned {sources_count} formatted sources")
                return result
            # For async tools - this is a fallback and should be avoided
            elif hasattr(tool, "_arun"):
                logger.warning(f"Tool {tool_name} is async but being called synchronously")
                # Create a new event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                start_time = __import__('time').time()
                result = loop.run_until_complete(tool._arun(**parameters))
                execution_time = __import__('time').time() - start_time
                logger.info(f"Async tool {tool_name} execution completed in {execution_time:.2f}s with result type: {type(result).__name__}")
                return result
            else:
                logger.error(f"Tool {tool_name} has no run method")
                return {
                    "formatted_sources": [],
                    "search_string": parameters.get("query", ""),
                    "tools": [tool_name],
                    "domains": [],
                    "citations": [],
                    "error": f"Tool {tool_name} has no run method"
                }
                
        except Exception as e:
            # Log the error
            logger.error(f"Error executing tool {tool_name}: {str(e)}", exc_info=True)
            # Return empty result
            return {
                "formatted_sources": [],
                "search_string": parameters.get("query", ""),
                "tools": [tool_name],
                "domains": [],
                "citations": [],
                "error": str(e)
            }
    
    def get_available_tools(self) -> Dict[str, str]:
        """
        Get all available tools and their descriptions.
        
        Returns:
            Dictionary mapping tool names to their descriptions
        """
        logger.info(f"[ToolExecutor.get_available_tools] Getting available tools")
        return self.registry.get_all_tool_descriptions() 