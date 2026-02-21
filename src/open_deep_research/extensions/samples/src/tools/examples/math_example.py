"""
Example script demonstrating how to use the Math MCP server with our tool registry.

This example shows how to:
1. Start the Math MCP server
2. Register it with our tool registry
3. Execute some math operations using the tools

Prerequisites:
- The math_server.py file should be in the src/tools directory
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.tools.registry import SearchToolRegistry
from src.tools.executor import ToolExecutor
from src.tools.mcp_tools import MCPToolManager

async def main():
    # Create a tool registry
    registry = SearchToolRegistry()
    executor = ToolExecutor(registry)
    
    # Create an MCP tool manager
    mcp_manager = MCPToolManager(registry)
    
    try:
        # Register the Math MCP server as a stdio subprocess
        math_server_path = Path(__file__).parent.parent / "math_server.py"
        
        if not math_server_path.exists():
            print(f"Error: {math_server_path} not found")
            return
            
        print(f"Starting Math MCP server from: {math_server_path}")
        
        tools = await mcp_manager.register_stdio_server(
            name="math",
            command=sys.executable,
            args=[str(math_server_path)]
        )
        
        print(f"Registered {len(tools)} tools from Math MCP server:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
            params = [f"{p.name}: {p.type}" for p in tool.parameters]
            print(f"    Parameters: {', '.join(params)}")
        
        # Execute some math operations
        operations = [
            ("add", {"a": 10, "b": 5}),
            ("subtract", {"a": 10, "b": 5}),
            ("multiply", {"a": 10, "b": 5}),
            ("divide", {"a": 10, "b": 5})
        ]
        
        for op_name, params in operations:
            print(f"\nExecuting {op_name} with parameters: {params}")
            result = await executor.execute_tool(
                f"mcp.math.{op_name}",
                params
            )
            print(f"Result: {result}")
        
    finally:
        # Close all MCP connections
        await mcp_manager.close_all()

if __name__ == "__main__":
    asyncio.run(main()) 