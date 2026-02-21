"""
Example demonstrating how to integrate a custom MCP client with our tool registry.

This script shows how to:
1. Create a custom MCP client for our math server
2. Adapt it to work with our tool registry
3. Execute math operations using our tool executor

Prerequisites:
- The math_server.py file should be in the src/tools directory
"""
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.tools.registry import SearchToolRegistry
from src.tools.executor import ToolExecutor
from src.tools.tool_schema import Tool, ToolParameter, ToolParameterType
from src.tools.examples.simple_math_client import SimpleMathClient

class MathServerAdapter:
    """Adapter for the math server that works with our tool registry."""
    
    def __init__(self, server_path: str):
        """Initialize the adapter.
        
        Args:
            server_path: Path to the math server script
        """
        self.server_path = server_path
        self.client = None
        self.tools = []
        
    async def start(self) -> List[Tool]:
        """Start the math server and register tools with the registry.
        
        Returns:
            List of tools that were registered
        """
        # Create and start the client
        self.client = SimpleMathClient(self.server_path)
        tool_definitions = await self.client.start()
        
        # Convert MCP tools to our Tool format
        self.tools = []
        for tool_def in tool_definitions:
            # Extract tool information
            name = tool_def.get("name")
            description = tool_def.get("description", "")
            
            # Extract parameter information
            parameters = []
            for param_name, param_info in tool_def.get("parameters", {}).items():
                # Determine parameter type
                param_type = ToolParameterType.STRING
                if param_info.get("type") == "number":
                    param_type = ToolParameterType.NUMBER
                elif param_info.get("type") == "boolean":
                    param_type = ToolParameterType.BOOLEAN
                
                # Create parameter object
                parameters.append(ToolParameter(
                    name=param_name,
                    type=param_type,
                    required=True,
                    description=param_info.get("description", "")
                ))
            
            # Create a tool wrapper function that captures the current name
            tool_name = name  # Create a variable to capture in the closure
            
            # Define a specific async function for each tool to ensure closure captures correctly
            async def make_tool_function(tool_name):
                async def execute_wrapper(**kwargs):
                    return await self.client.execute_tool(tool_name, kwargs)
                return execute_wrapper
            
            # Create our Tool object with the dynamic function
            tool = Tool(
                name=f"math.{name}",
                description=description,
                parameters=parameters,
                function=await make_tool_function(name)
            )
            
            self.tools.append(tool)
        
        return self.tools
    
    async def close(self):
        """Close the connection to the server."""
        if self.client:
            await self.client.close()
            self.client = None

async def main():
    # Create a tool registry
    registry = SearchToolRegistry()
    executor = ToolExecutor(registry)
    
    # Get the path to the math server
    math_server_path = Path(__file__).parent.parent / "math_server.py"
    
    if not math_server_path.exists():
        print(f"Error: Math server not found at {math_server_path}")
        return
    
    # Create the adapter
    adapter = MathServerAdapter(str(math_server_path))
    
    try:
        # Start the adapter and register tools
        tools = await adapter.start()
        
        # Register tools with the registry
        for tool in tools:
            registry.register_tool(tool)
        
        print(f"Registered {len(tools)} tools with the registry:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
            params = [f"{p.name}: {p.type}" for p in tool.parameters]
            print(f"    Parameters: {', '.join(params)}")
        
        # Execute some math operations
        operations = [
            ("math.add", {"a": 10, "b": 5}),
            ("math.subtract", {"a": 10, "b": 5}),
            ("math.multiply", {"a": 10, "b": 5}),
            ("math.divide", {"a": 10, "b": 5})
        ]
        
        for op_name, params in operations:
            print(f"\nExecuting {op_name} with parameters: {params}")
            result = await executor.execute_tool(op_name, params)
            print(f"Result: {result}")
        
    finally:
        # Close the adapter
        await adapter.close()

if __name__ == "__main__":
    asyncio.run(main()) 