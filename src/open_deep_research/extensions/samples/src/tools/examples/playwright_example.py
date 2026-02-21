"""
Example script demonstrating how to use the Playwright MCP server with our tool registry.

This example shows how to:
1. Start the Playwright MCP server (via stdio)
2. Register it with our tool registry
3. Execute some web browsing commands using the Playwright tools

Prerequisites:
- Requires Node.js and npx to be installed.
- The Playwright MCP server will be downloaded via npx if not already present.
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
        # Start the Playwright MCP server as a subprocess using stdio
        # Using headless mode by default
        print("Registering Playwright MCP server via stdio (headless)...")
        tools = await mcp_manager.register_stdio_server(
            name="playwright",
            command="npx",
            args=["-y", "@playwright/mcp@latest", "--headless"]
            # args=["-y", "@playwright/mcp@latest"]
        )

        print(f"Registered {len(tools)} tools from Playwright MCP server:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        # Example: Execute some web browsing commands

        # Navigate to a website
        print("\nNavigating to https://example.com...")
        nav_result = await executor.execute_tool(
            "mcp.playwright.browser_navigate",
            {"url": "https://xplorestaging.ieee.org/author/37086453282"},
            config={}
        )
        print(f"Navigation result: {nav_result}")

        # Take a screenshot after navigating (using PDF as workaround)
        print("\nTaking screenshot (using PDF as workaround)...")
        screenshots_dir = os.path.join(project_root, "screenshots")
        output_file_path = os.path.join(screenshots_dir, "example_com.pdf")
        
        try:
            pdf_result = await executor.execute_tool(
                "mcp.playwright.browser_pdf_save",
                {"path": "example_com.pdf"},  # This will be ignored, but we include it anyway
                config={}
            )
            print(f"PDF save command executed. Result: {pdf_result}")
            
            # Extract the actual path from the result
            if isinstance(pdf_result, str) and pdf_result.startswith("Saved as "):
                actual_path = pdf_result.replace("Saved as ", "").strip()
                print(f"Actual PDF file location: {actual_path}")
                
                # Copy the file to our screenshots directory
                import shutil
                os.makedirs(screenshots_dir, exist_ok=True)  # Ensure screenshots directory exists
                shutil.copy2(actual_path, output_file_path)
                
                if os.path.exists(output_file_path):
                    print(f"Successfully saved visual capture to: {output_file_path}")
                else:
                    print(f"Failed to save visual capture to: {output_file_path}")
            else:
                print(f"Unexpected PDF result format: {pdf_result}")
        except Exception as e:
            print(f"Error capturing visual: {e}")
            import traceback
            traceback.print_exc()
            
        # ALSO attempt to take an actual screenshot (as a second approach)
        print("\nAlso attempting to take a PNG screenshot...")
        output_screenshot_path = os.path.join(screenshots_dir, "linkedin_profile.png")
        
        try:
            screenshot_result = await executor.execute_tool(
                "mcp.playwright.browser_take_screenshot",
                {"path": "linkedin_profile.png"},
                config={}
            )
            print(f"Screenshot command executed. Result: {screenshot_result}")
            
            # Check if Playwright saved the screenshot somewhere and reported the location
            if isinstance(screenshot_result, str) and "Saved as" in screenshot_result:
                # Extract the actual path from the result
                actual_path = screenshot_result.replace("Saved as", "").strip()
                print(f"Actual screenshot file location: {actual_path}")
                
                # Copy the file to our screenshots directory
                import shutil
                shutil.copy2(actual_path, output_screenshot_path)
                
                if os.path.exists(output_screenshot_path):
                    print(f"Successfully copied screenshot to: {output_screenshot_path}")
                else:
                    print(f"Failed to copy screenshot to: {output_screenshot_path}")
            else:
                # In case it worked but with a different result format
                print(f"Screenshot result doesn't contain path info. Looking for files...")
                import glob
                # Try to find any recent png files in typical temp directories
                tmp_files = glob.glob("/tmp/*.png") + glob.glob("/var/tmp/*.png")
                # Sort by creation time, newest first
                tmp_files.sort(key=lambda x: os.path.getctime(x), reverse=True)
                
                if tmp_files:
                    newest_file = tmp_files[0]
                    print(f"Found possible screenshot at: {newest_file}")
                    import shutil
                    shutil.copy2(newest_file, output_screenshot_path)
                    if os.path.exists(output_screenshot_path):
                        print(f"Successfully copied most recent PNG to: {output_screenshot_path}")
                    else:
                        print(f"Failed to copy most recent PNG to: {output_screenshot_path}")
                else:
                    print("No recent PNG files found in temp directories")
        except Exception as e:
            print(f"Error processing screenshot: {e}")
            import traceback
            traceback.print_exc()

        # Take an accessibility snapshot - Temporarily commented out for debugging
        # print("\nTaking accessibility snapshot...")
        # snapshot_result = await executor.execute_tool(
        #     "mcp.playwright.browser_snapshot",
        #     {},
        #     config={}
        # )
        # # Snapshot result can be large, print only a part or confirmation
        # if isinstance(snapshot_result, dict) and 'snapshot' in snapshot_result:
        #      print(f"Snapshot captured successfully (first 200 chars):\n{str(snapshot_result['snapshot'])[:200]}...")
        # else:
        #      print(f"Snapshot result: {snapshot_result}")


        # Example: Typing into a non-existent element (will likely fail, demonstrating error handling)
        # On example.com, there isn't an obvious input field without more complex interaction
        # print("\nAttempting to type (expected to fail)...")
        # try:
        #     type_result = await executor.execute_tool(
        #         "mcp.playwright.browser_type",
        #         {"ref": "input#search", "element": "search input", "text": "hello world"}
        #     )
        #     print(f"Type result: {type_result}")
        # except Exception as e:
        #     print(f"Typing failed as expected: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close all MCP connections
        print("\nClosing MCP connections...")
        await mcp_manager.close_all()
        print("Connections closed.")

if __name__ == "__main__":
    asyncio.run(main()) 