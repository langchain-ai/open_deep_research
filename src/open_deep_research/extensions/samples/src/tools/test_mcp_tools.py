"""
Tests for the MCP tools integration.
"""
import asyncio
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.tools.mcp_tools import MCPToolProvider, MCPToolManager
from src.tools.registry import ToolRegistry
from src.tools.tool_schema import Tool, ToolParameter, ToolParameterType

# Create a simple test MCP server script for testing
TEST_SERVER_SCRIPT = """
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("TestTools")

@mcp.tool()
def echo(message: str) -> str:
    \"\"\"Echo back the message.\"\"\"
    return message

@mcp.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        mcp.run(transport="http")
    else:
        mcp.run(transport="stdio")
"""

@pytest.fixture
def test_server_path(tmp_path):
    """Create a temporary test server script."""
    server_path = tmp_path / "test_server.py"
    server_path.write_text(TEST_SERVER_SCRIPT)
    return server_path

@pytest.fixture
def tool_registry():
    """Create a tool registry for testing."""
    return ToolRegistry()

@pytest.mark.asyncio
async def test_mcp_tool_provider_stdio(test_server_path):
    """Test that we can connect to an MCP server using stdio transport."""
    provider = MCPToolProvider("test")
    
    # Mock the ClientSession
    with patch("src.tools.mcp_tools.stdio_client") as mock_stdio_client, \
         patch("src.tools.mcp_tools.ClientSession") as mock_session_class, \
         patch("src.tools.mcp_tools.load_mcp_tools") as mock_load_mcp_tools:
        
        # Mock the stdio client
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_stdio_client.return_value.__aenter__.return_value = (mock_read, mock_write)
        
        # Mock the session
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        # Mock the load_mcp_tools function
        mock_tool = MagicMock()
        mock_tool.name = "echo"
        mock_tool.description = "Echo back the message."
        mock_tool._run = lambda **kwargs: kwargs["message"]
        mock_tool.args_schema.schema.return_value = {
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to echo back"
                }
            },
            "required": ["message"]
        }
        mock_load_mcp_tools.return_value = [mock_tool]
        
        # Connect to the server
        await provider.connect_stdio(
            command=sys.executable,
            args=[str(test_server_path)]
        )
        
        # Check that the session was initialized
        mock_session.initialize.assert_called_once()
        
        # Load the tools
        tools = await provider.load_tools()
        
        # Check that we got the expected tools
        assert len(tools) == 1
        assert tools[0].name == "mcp.test.echo"
        assert tools[0].description == "Echo back the message."
        assert len(tools[0].parameters) == 1
        assert tools[0].parameters[0].name == "message"
        assert tools[0].parameters[0].type == ToolParameterType.STRING
        assert tools[0].parameters[0].required == True
        
        # Close the connection
        await provider.close()

@pytest.mark.asyncio
async def test_mcp_tool_manager(tool_registry):
    """Test the MCP tool manager."""
    manager = MCPToolManager(tool_registry)
    
    # Mock the MCPToolProvider
    with patch("src.tools.mcp_tools.MCPToolProvider") as mock_provider_class:
        # Create a mock provider
        mock_provider = AsyncMock()
        mock_provider.name = "test"
        
        # Create a mock tool
        mock_tool = Tool(
            name="mcp.test.echo",
            description="Echo back the message.",
            parameters=[
                ToolParameter(
                    name="message",
                    type=ToolParameterType.STRING,
                    required=True,
                    description="The message to echo back"
                )
            ],
            function=lambda **kwargs: kwargs["message"]
        )
        
        # Set up the mock provider
        mock_provider.load_tools.return_value = [mock_tool]
        mock_provider_class.return_value = mock_provider
        
        # Register a server
        tools = await manager.register_stdio_server(
            name="test",
            command=sys.executable,
            args=["test_server.py"]
        )
        
        # Check that the provider was created and used
        mock_provider_class.assert_called_once_with("test")
        mock_provider.connect_stdio.assert_called_once()
        mock_provider.load_tools.assert_called_once()
        
        # Check that we got the expected tools
        assert len(tools) == 1
        assert tools[0].name == "mcp.test.echo"
        
        # Check that the tool was registered
        assert tool_registry.get_tool("mcp.test.echo") is not None
        
        # Close the connections
        await manager.close_all()
        mock_provider.close.assert_called_once()

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 