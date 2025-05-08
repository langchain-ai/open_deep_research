# open_deep_research/mcp_integration.py
import logging
from typing import Dict, Any, Optional, List
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

class MCPClientManager:
    """Manages MCP client lifecycle and tool access."""
    
    def __init__(self):
        self.client = None
        self.tools: List[BaseTool] = []
    
    async def initialize(self, mcp_servers: Dict[str, Dict[str, Any]]):
        """Initialize the MCP client and get all tools."""
        if not mcp_servers:
            logger.warning("No MCP servers provided")
            return
            
        try:
            # Create client with detailed logging
            logger.info(f"Initializing MCP client with servers: {list(mcp_servers.keys())}")
            for server_name, server_config in mcp_servers.items():
                logger.info(f"Server {server_name} config: transport={server_config.get('transport')}")
                if server_config.get('transport') == 'stdio':
                    logger.info(f"  Command: {server_config.get('command')} {' '.join(server_config.get('args', []))}")
                elif server_config.get('transport') == 'sse':
                    logger.info(f"  URL: {server_config.get('url')}")
            
            # Initialize the client
            self.client = MultiServerMCPClient(mcp_servers)
            
            # Use __aenter__ instead of initialize
            await self.client.__aenter__()
            
            # Get all tools
            self.tools = self.client.get_tools()
            
            # Log detailed information about loaded tools
            if self.tools:
                logger.info(f"Successfully loaded {len(self.tools)} tools from MCP servers")
                for i, tool in enumerate(self.tools):
                    logger.info(f"Tool {i+1}: {tool.name} - {tool.description}")
                    logger.info(f"  Arguments: {getattr(tool, 'args', 'Unknown')}")
            else:
                logger.warning("No tools were loaded from MCP servers")
                
        except Exception as e:
            logger.error(f"Error initializing MCP client: {e}", exc_info=True)
            if self.client:
                try:
                    await self.client.__aexit__(None, None, None)
                except Exception as cleanup_err:
                    logger.error(f"Error during cleanup after initialization failure: {cleanup_err}")
                self.client = None
    
    async def cleanup(self):
        """Clean up MCP resources"""
        if self.client:
            try:
                await self.client.__aexit__(None, None, None)
            except RuntimeError as e:
                if "cancel scope in a different task" in str(e):
                    # Log as warning, not error
                    logger.warning("Task context error during MCP client cleanup - this is expected in StateGraph context")
                else:
                    # Re-raise other types of RuntimeError
                    raise
    
    def get_tools(self) -> List[BaseTool]:
        """Get all tools from connected MCP servers."""
        return self.tools

async def create_mcp_manager(mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[MCPClientManager]:
    """Create and initialize an MCP client manager."""
    if not mcp_servers:
        logger.warning("No MCP servers provided to create_mcp_manager")
        return None
        
    logger.info(f"Creating MCP manager with servers: {list(mcp_servers.keys())}")
    manager = MCPClientManager()
    await manager.initialize(mcp_servers)
    
    # Validate that tools were loaded
    if not manager.tools:
        logger.warning("No tools loaded in MCP manager")
        await manager.cleanup()
        return None
        
    return manager