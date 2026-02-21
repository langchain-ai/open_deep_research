"""
A simple MCP server that provides math tools.
"""
import sys
import json
from typing import Dict, Any, List, Optional, Union

# Simple MCP server that doesn't rely on external libraries
class SimpleMathMCP:
    """A simple MCP server that provides math tools."""
    
    def __init__(self):
        """Initialize the MCP server."""
        self.tools = {
            "add": {
                "name": "add",
                "description": "Add two numbers.",
                "parameters": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                }
            },
            "subtract": {
                "name": "subtract",
                "description": "Subtract b from a.",
                "parameters": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Number to subtract"}
                }
            },
            "multiply": {
                "name": "multiply",
                "description": "Multiply two numbers.",
                "parameters": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                }
            },
            "divide": {
                "name": "divide",
                "description": "Divide a by b. Returns an error if b is 0.",
                "parameters": {
                    "a": {"type": "number", "description": "Dividend"},
                    "b": {"type": "number", "description": "Divisor"}
                }
            }
        }
        
    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP message.
        
        Args:
            message: The MCP message
            
        Returns:
            The response message
        """
        if message.get("type") == "request":
            if message.get("method") == "initialize":
                return self._handle_initialize()
            elif message.get("method") == "getTools":
                return self._handle_get_tools()
            elif message.get("method") == "executeTool":
                return self._handle_execute_tool(message)
            else:
                return {"type": "error", "message": f"Unknown method: {message.get('method')}"}
        else:
            return {"type": "error", "message": f"Unknown message type: {message.get('type')}"}
    
    def _handle_initialize(self) -> Dict[str, Any]:
        """Handle an initialize request."""
        return {
            "type": "response",
            "serverInfo": {
                "name": "MathTools",
                "version": "1.0.0"
            }
        }
    
    def _handle_get_tools(self) -> Dict[str, Any]:
        """Handle a getTools request."""
        return {
            "type": "response",
            "tools": list(self.tools.values())
        }
    
    def _handle_execute_tool(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an executeTool request.
        
        Args:
            message: The MCP message
            
        Returns:
            The response message
        """
        tool_name = message.get("tool")
        params = message.get("parameters", {})
        
        if tool_name not in self.tools:
            return {"type": "error", "message": f"Unknown tool: {tool_name}"}
        
        try:
            # Convert parameters to numbers
            a = float(params.get("a", 0))
            b = float(params.get("b", 0))
            
            # Execute the appropriate operation
            if tool_name == "add":
                result = a + b
            elif tool_name == "subtract":
                result = a - b
            elif tool_name == "multiply":
                result = a * b
            elif tool_name == "divide":
                if b == 0:
                    return {"type": "error", "message": "Cannot divide by zero"}
                result = a / b
            else:
                return {"type": "error", "message": f"Unknown operation: {tool_name}"}
            
            # Return the result
            return {
                "type": "response",
                "result": result
            }
            
        except Exception as e:
            return {"type": "error", "message": str(e)}

def main():
    """Main entry point."""
    server = SimpleMathMCP()
    
    # Read messages from stdin and write responses to stdout
    while True:
        try:
            line = input()
            if not line:
                continue
                
            # Parse the message
            message = json.loads(line)
            
            # Handle the message
            response = server.handle_message(message)
            
            # Write the response
            print(json.dumps(response))
            sys.stdout.flush()
            
        except EOFError:
            # End of input
            break
        except Exception as e:
            # Write an error response
            error_response = {"type": "error", "message": str(e)}
            print(json.dumps(error_response))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
