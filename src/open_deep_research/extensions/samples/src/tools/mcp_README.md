# MCP (Model Context Protocol) Integration

This module provides integration with the Model Context Protocol (MCP), allowing you to connect to MCP servers and use their tools within our tool registry system.

## What is MCP?

The Model Context Protocol (MCP) is an open protocol that standardizes how applications provide context to LLMs. Think of MCP like a USB-C port for AI applications - it provides a standardized way to connect AI models to different data sources and tools.

MCP solves the "N×M problem" where N represents different LLMs and M represents various tools and data sources. Without standardization, each combination requires custom integration work. MCP standardizes the interface, allowing any MCP-compatible tool to work with any MCP-compatible AI model or application.

## Features

- Connect to MCP servers using either HTTP or stdio transport
- Load tools from MCP servers and convert them to our Tool format
- Register MCP tools with our tool registry
- Manage multiple MCP connections

## Installation

To use this module, you need to install the following packages:

```bash
pip install mcp
pip install langchain-mcp-adapters
```

## Usage

### Connecting to an MCP Server via HTTP

```python
import asyncio
from src.tools.registry import ToolRegistry
from src.tools.mcp_tools import MCPToolManager

async def main():
    # Create a tool registry
    registry = ToolRegistry()
    
    # Create an MCP tool manager
    mcp_manager = MCPToolManager(registry)
    
    # Connect to an MCP server over HTTP
    tools = await mcp_manager.register_http_server(
        name="my_server",
        base_url="http://localhost:3000"
    )
    
    print(f"Registered {len(tools)} tools from MCP server")
    
    # Don't forget to close connections when done
    await mcp_manager.close_all()

if __name__ == "__main__":
    asyncio.run(main())
```

### Connecting to an MCP Server via Stdio

```python
import asyncio
from src.tools.registry import ToolRegistry
from src.tools.mcp_tools import MCPToolManager

async def main():
    # Create a tool registry
    registry = ToolRegistry()
    
    # Create an MCP tool manager
    mcp_manager = MCPToolManager(registry)
    
    # Start an MCP server as a subprocess and connect to it
    tools = await mcp_manager.register_stdio_server(
        name="math_tools",
        command="python",
        args=["math_server.py"]
    )
    
    print(f"Registered {len(tools)} tools from MCP server")
    
    # Don't forget to close connections when done
    await mcp_manager.close_all()

if __name__ == "__main__":
    asyncio.run(main())
```

### Using MCP Tools with the Tool Executor

```python
import asyncio
from src.tools.registry import ToolRegistry
from src.tools.executor import ToolExecutor
from src.tools.mcp_tools import MCPToolManager

async def main():
    # Create a tool registry and executor
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    
    # Create an MCP tool manager
    mcp_manager = MCPToolManager(registry)
    
    # Connect to a math MCP server
    await mcp_manager.register_stdio_server(
        name="math",
        command="python",
        args=["math_server.py"]
    )
    
    # Execute a tool
    result = await executor.execute_tool(
        "mcp.math.add",
        {"a": 5, "b": 3}
    )
    
    print(f"Result: {result}")  # Output: Result: 8
    
    # Close all MCP connections
    await mcp_manager.close_all()

if __name__ == "__main__":
    asyncio.run(main())
```

## Available MCP Servers

There are several MCP servers available that you can use with this integration:

1. **Puppeteer MCP Server**: Provides web browsing capabilities
   - GitHub: https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer
   - Installation: `npm install -g @modelcontextprotocol/server-puppeteer`
   - Usage: `npx -y @modelcontextprotocol/server-puppeteer`

2. **Filesystem MCP Server**: File system operations
   - GitHub: https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem
   - Installation: `npm install -g @modelcontextprotocol/server-filesystem`
   - Usage: `npx -y @modelcontextprotocol/server-filesystem`

3. **Fetch MCP Server**: HTTP requests and responses
   - GitHub: https://github.com/modelcontextprotocol/servers/tree/main/src/fetch
   - Installation: `npm install -g @modelcontextprotocol/server-fetch`
   - Usage: `npx -y @modelcontextprotocol/server-fetch`

4. **Math MCP Server**: Simple example providing math operations
   - See the `math_server.py` example in this repository

5. **Other MCP Servers**: More servers are available in the [MCP servers repository](https://github.com/modelcontextprotocol/servers/tree/main/src)

## Creating Your Own MCP Server

You can create your own MCP server using the `fastmcp` library. Here's a simple example:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MyTools")

@mcp.tool()
def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run(transport="stdio")  # or "http"
```

For more information, see the [Model Context Protocol documentation](https://modelcontextprotocol.io/). 