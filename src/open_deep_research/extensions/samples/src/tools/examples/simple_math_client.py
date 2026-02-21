"""
A simplified client to communicate with our custom math server.

This example shows how to:
1. Start the Math server as a subprocess
2. Communicate with it directly
3. Execute math operations

Prerequisites:
- The math_server.py file should be in the src/tools directory
"""
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, AsyncGenerator

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class SimpleMathClient:
    """A simple client for the math server."""
    
    def __init__(self, server_path: str):
        """Initialize the client.
        
        Args:
            server_path: Path to the math server script
        """
        self.server_path = server_path
        self.process = None
        
    async def start(self):
        """Start the math server process."""
        # Start the server process
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            self.server_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Initialize the server
        response = await self._send_message({
            "type": "request",
            "method": "initialize"
        })
        
        print(f"Server initialized: {response}")
        
        # Get the list of available tools
        response = await self._send_message({
            "type": "request",
            "method": "getTools"
        })
        
        self.tools = response.get("tools", [])
        print(f"Available tools: {len(self.tools)}")
        for tool in self.tools:
            params = [f"{name}: {param['type']}" for name, param in tool.get('parameters', {}).items()]
            print(f"  - {tool['name']}: {tool.get('description', '')}")
            print(f"    Parameters: {', '.join(params)}")
        
        return self.tools
        
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a tool.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool
            
        Returns:
            The result of the operation
        """
        response = await self._send_message({
            "type": "request",
            "method": "executeTool",
            "tool": tool_name,
            "parameters": parameters
        })
        
        if "error" in response:
            raise ValueError(response["error"])
            
        return response.get("result")
        
    async def _send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to the server and wait for a response.
        
        Args:
            message: The message to send
            
        Returns:
            The response from the server
        """
        if not self.process:
            raise ValueError("Server not started")
            
        # Send the message
        self.process.stdin.write(json.dumps(message).encode() + b"\n")
        await self.process.stdin.drain()
        
        # Read the response
        response_line = await self.process.stdout.readline()
        response = json.loads(response_line.decode())
        
        return response
        
    async def close(self):
        """Close the connection to the server."""
        if self.process:
            # Close pipes
            if self.process.stdin:
                self.process.stdin.close()
                await self.process.stdin.wait_closed()
                
            # Terminate the process
            self.process.terminate()
            await self.process.wait()
            self.process = None

async def main():
    # Get the path to the math server
    math_server_path = Path(__file__).parent.parent / "math_server.py"
    
    if not math_server_path.exists():
        print(f"Error: Math server not found at {math_server_path}")
        return
        
    # Create the client
    client = SimpleMathClient(str(math_server_path))
    
    try:
        # Start the server
        await client.start()
        
        # Execute some math operations
        operations = [
            ("add", {"a": 10, "b": 5}),
            ("subtract", {"a": 10, "b": 5}),
            ("multiply", {"a": 10, "b": 5}),
            ("divide", {"a": 10, "b": 5})
        ]
        
        for op_name, params in operations:
            print(f"\nExecuting {op_name} with parameters: {params}")
            result = await client.execute_tool(op_name, params)
            print(f"Result: {result}")
            
    finally:
        # Close the connection
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 