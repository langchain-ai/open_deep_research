"""
MCP (Model Context Protocol) integration for the tools registry.
Allows loading tools from MCP servers and registering them with the tool registry.
"""
import asyncio
import sys
import httpx
from functools import partial
from typing import Dict, List, Optional, Union, Any, Callable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

from .registry import SearchToolRegistry as ToolRegistry
from .tool_schema import Tool, ToolParameter, ToolParameterType

# Define an equivalent HttpServerParameters class
class HttpServerParameters:
    """Parameters for HTTP MCP server connection."""
    
    def __init__(self, base_url: str) -> None:
        """Initialize the HTTP server parameters.
        
        Args:
            base_url: Base URL of the MCP server
        """
        self.base_url = base_url

# Define an HTTP client function
async def http_client(params: HttpServerParameters):
    """Create an HTTP client for an MCP server."""
    
    class HttpRead:
        """Read interface for HTTP client."""
        
        def __init__(self, base_url: str):
            self.base_url = base_url
            self.client = httpx.AsyncClient()
            
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.client.aclose()
            
        async def read(self):
            """Read from the server."""
            # Basic implementation - this would normally set up SSE
            response = await self.client.get(f"{self.base_url}/mcp/v1/events")
            return response.text
    
    class HttpWrite:
        """Write interface for HTTP client."""
        
        def __init__(self, base_url: str):
            self.base_url = base_url
            self.client = httpx.AsyncClient()
            
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.client.aclose()
            
        async def write(self, data: str):
            """Write to the server."""
            response = await self.client.post(
                f"{self.base_url}/mcp/v1/messages", 
                content=data
            )
            return response.status_code
            
    read = HttpRead(params.base_url)
    write = HttpWrite(params.base_url)
    
    return read, write

class MCPToolProvider:
    """Provider for MCP-based tools."""
    
    def __init__(self, name: str):
        """
        Initialize the MCP tool provider.

        Args:
            name: Name of the MCP server for identification
        """
        self.name = name
        self.session: Optional[ClientSession] = None
        self.tools = []
        self._client_context = None # Store the stdio_client context
        
    async def connect_stdio(self, command: str, args: List[str]) -> None:
        """
        Connect to an MCP server using stdio transport.

        Args:
            command: Command to run the MCP server
            args: Arguments to pass to the command
        """
        server_params = StdioServerParameters(command=command, args=args)
        self._client_context = stdio_client(server_params)
        _read, _write = await self._client_context.__aenter__()
        
        session_context = ClientSession(_read, _write)
        self.session = await session_context.__aenter__()
        # Keep the session context around to exit it later if needed, although
        # closing the client_context should ideally handle this.
        self._session_context = session_context 
        
        await self.session.initialize()

    async def connect_http(self, base_url: str) -> None:
        """
        Connect to an MCP server using HTTP transport.

        Args:
            base_url: Base URL of the MCP server
        """
        server_params = HttpServerParameters(base_url=base_url)
        self._client_context = http_client(server_params) # Assuming http_client returns a context manager pair
        _read, _write = await self._client_context # Assuming http_client returns (read, write) pair directly
        
        session_context = ClientSession(_read, _write)
        self.session = await session_context.__aenter__()
        self._session_context = session_context
        
        await self.session.initialize()

    async def load_tools(self) -> List[Tool]:
        """Load tools from the MCP server and convert them to our Tool format.
        
        Returns:
            List of Tool objects
        """
        if self.session is None:
            raise ValueError("Not connected to an MCP server")
        
        # Load tools from the MCP server using langchain-mcp-adapters
        langchain_tools = await load_mcp_tools(self.session)
        
        # Convert LangChain tools to our Tool format
        tools = []
        for lc_tool in langchain_tools:
            # Extract parameter info from the tool's args_schema
            parameters = []
            if hasattr(lc_tool, 'args_schema'):
                schema = None
                if hasattr(lc_tool.args_schema, 'schema'):
                    schema = lc_tool.args_schema.schema()
                elif isinstance(lc_tool.args_schema, dict):
                    schema = lc_tool.args_schema
                    
                if schema and 'properties' in schema:
                    for name, prop in schema['properties'].items():
                        param_type = ToolParameterType.STRING
                        if 'type' in prop:
                            if prop['type'] == 'integer':
                                param_type = ToolParameterType.NUMBER
                            elif prop['type'] == 'number':
                                param_type = ToolParameterType.NUMBER
                            elif prop['type'] == 'boolean':
                                param_type = ToolParameterType.BOOLEAN
                        
                        required = name in schema.get('required', [])
                        description = prop.get('description', '')
                        
                        parameters.append(ToolParameter(
                            name=name,
                            type=param_type,
                            required=required,
                            description=description
                        ))
            
            # Create our Tool object, storing the LangChain tool directly
            tool = Tool(
                name=f"mcp.{self.name}.{lc_tool.name}",
                description=lc_tool.description,
                parameters=parameters,
                lc_tool=lc_tool # Store the LangChain tool instance
            )
            
            tools.append(tool)
        
        self.tools = tools
        return tools
    
    async def close(self) -> None:
        """Close the connection to the MCP server using context managers."""
        if hasattr(self, '_session_context') and self._session_context:
            try:
                await self._session_context.__aexit__(*sys.exc_info())
            except Exception as e:
                print(f"Error closing MCP session context: {e}") # Log error
            finally:
                self.session = None
                self._session_context = None

        if hasattr(self, '_client_context') and self._client_context:
            try:
                await self._client_context.__aexit__(*sys.exc_info())
            except Exception as e:
                print(f"Error closing MCP client context: {e}") # Log error
            finally:
                self._client_context = None


class MCPToolManager:
    """Manager for MCP tool providers."""
    
    def __init__(self, registry: ToolRegistry):
        """Initialize the MCP tool manager.
        
        Args:
            registry: Tool registry to register tools with
        """
        self.registry = registry
        self.providers: Dict[str, MCPToolProvider] = {}
        
    async def register_stdio_server(self, name: str, command: str, args: List[str]) -> List[Tool]:
        """Register an MCP server using stdio transport.
        
        Args:
            name: Name of the MCP server for identification
            command: Command to run the MCP server
            args: Arguments to pass to the command
            
        Returns:
            List of registered tools
        """
        provider = MCPToolProvider(name)
        await provider.connect_stdio(command, args)
        tools = await provider.load_tools()
        
        # Register tools with the registry
        for tool in tools:
            self.registry.register_tool(tool)
        
        self.providers[name] = provider
        return tools
    
    async def register_http_server(self, name: str, base_url: str) -> List[Tool]:
        """Register an MCP server using HTTP transport.
        
        Args:
            name: Name of the MCP server for identification
            base_url: Base URL of the MCP server
            
        Returns:
            List of registered tools
        """
        provider = MCPToolProvider(name)
        await provider.connect_http(base_url)
        tools = await provider.load_tools()
        
        # Register tools with the registry
        for tool in tools:
            self.registry.register_tool(tool)
        
        self.providers[name] = provider
        return tools
    
    async def close_all(self) -> None:
        """Close all MCP connections."""
        for provider in self.providers.values():
            await provider.close()
        
        self.providers = {} 