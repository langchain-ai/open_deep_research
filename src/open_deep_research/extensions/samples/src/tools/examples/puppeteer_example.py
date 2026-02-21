"""
Example script demonstrating how to use the Puppeteer MCP server with our tool registry.

This example shows how to:
1. Start the Puppeteer MCP server (assuming it's installed)
2. Register it with our tool registry
3. Execute some web browsing commands using the tools

Prerequisites:
- Install the MCP Puppeteer server:
  npm install -g @modelcontextprotocol/server-puppeteer

- Run the server:
  npx -y @modelcontextprotocol/server-puppeteer
  
- Alternatively, you can run it without installing:
  npx -y @modelcontextprotocol/server-puppeteer
"""
import asyncio
import sys
import os
import shutil
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
        # Start the Puppeteer MCP server as a subprocess
        print("Registering Puppeteer MCP server via stdio...")
        tools = await mcp_manager.register_stdio_server(
            name="puppeteer",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-puppeteer"]
        )
        
        print(f"Registered {len(tools)} tools from Puppeteer MCP server:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # Example: Execute some web browsing commands
        
        # First, navigate to a website
        print("\nNavigating to a website...")
        result = await executor.execute_tool(
            "mcp.puppeteer.puppeteer_navigate",
            {
                "url": "https://example.com",
                "launchOptions": {
                    "headless": "new",
                    "args": ["--no-sandbox"],
                },
                "allowDangerous": True
            },
            config={}
        )
        print(f"Navigation result: {result}")
        
        # Take a screenshot
        print("\nTaking screenshot...")
        screenshots_dir = os.path.join(project_root, "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        
        try:
            result = await executor.execute_tool(
                "mcp.puppeteer.puppeteer_screenshot",
                {
                    "name": "example_screenshot",
                    "width": 1024,
                    "height": 768
                },
                config={}
            )
            print(f"Screenshot command executed. Result: {result}")
            
            # Process the result which should contain base64 image data
            if isinstance(result, dict) and 'content' in result:
                for content in result['content']:
                    if content.get('type') == 'image' and content.get('data'):
                        # Save the base64 data
                        screenshot_path = os.path.join(screenshots_dir, "example_screenshot.png")
                        import base64
                        img_data = base64.b64decode(content['data'])
                        with open(screenshot_path, 'wb') as f:
                            f.write(img_data)
                        print(f"Screenshot saved to: {screenshot_path}")
                        break
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            import traceback
            traceback.print_exc()
        
        # Get basic page information
        print("\nGetting page information...")
        try:
            result = await executor.execute_tool(
                "mcp.puppeteer.puppeteer_evaluate",
                {
                    "pageFunction": "() => document.title"
                },
                config={}
            )
            print(f"Page title: {result}")
            
            # Get more page information with a separate call
            result = await executor.execute_tool(
                "mcp.puppeteer.puppeteer_evaluate",
                {
                    "pageFunction": "() => ({ url: window.location.href, h1: document.querySelector('h1')?.textContent })"
                },
                config={}
            )
            print(f"Additional page info: {result}")
        except Exception as e:
            print(f"Error getting page information: {e}")
            import traceback
            traceback.print_exc()
        
    finally:
        # Close all MCP connections
        print("\nClosing MCP connections...")
        await mcp_manager.close_all()
        print("Puppeteer MCP Server closed")

if __name__ == "__main__":
    asyncio.run(main()) 