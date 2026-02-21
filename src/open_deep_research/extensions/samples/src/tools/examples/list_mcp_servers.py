"""
Utility script to list available MCP servers and installation instructions.

This script provides a list of known MCP servers and instructions
on how to install and use them with our MCP integration.
"""
import os
import sys
from pathlib import Path
import subprocess
import json
from typing import Dict, List, Optional

# List of known MCP servers with installation and usage instructions
MCP_SERVERS = [
    {
        "name": "Puppeteer",
        "description": "Web browsing capabilities using Puppeteer",
        "repo": "https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer",
        "install_command": "npm install -g @modelcontextprotocol/server-puppeteer",
        "run_command": "npx -y @modelcontextprotocol/server-puppeteer",
        "transport_modes": ["http", "stdio"],
        "default_port": 3000,
        "example": """
# Connect via HTTP
tools = await mcp_manager.register_http_server(
    name="puppeteer",
    base_url="http://localhost:3000"
)

# Connect via stdio
tools = await mcp_manager.register_stdio_server(
    name="puppeteer",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-puppeteer", "--transport", "stdio"]
)

# Alternatively, you can use it via Docker
# tools = await mcp_manager.register_stdio_server(
#     name="puppeteer",
#     command="docker",
#     args=["run", "-i", "--rm", "--init", "-e", "DOCKER_CONTAINER=true", "mcp/puppeteer"]
# )
"""
    },
    {
        "name": "Filesystem",
        "description": "Filesystem operations and file management",
        "repo": "https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem",
        "install_command": "npm install -g @modelcontextprotocol/server-filesystem",
        "run_command": "npx -y @modelcontextprotocol/server-filesystem",
        "transport_modes": ["http", "stdio"],
        "default_port": 3000,
        "example": """
# Connect via HTTP
tools = await mcp_manager.register_http_server(
    name="filesystem",
    base_url="http://localhost:3000"
)

# Connect via stdio
tools = await mcp_manager.register_stdio_server(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "--transport", "stdio"]
)
"""
    },
    {
        "name": "Fetch",
        "description": "Make HTTP requests and process responses",
        "repo": "https://github.com/modelcontextprotocol/servers/tree/main/src/fetch",
        "install_command": "npm install -g @modelcontextprotocol/server-fetch",
        "run_command": "npx -y @modelcontextprotocol/server-fetch",
        "transport_modes": ["http", "stdio"],
        "default_port": 3000,
        "example": """
# Connect via HTTP
tools = await mcp_manager.register_http_server(
    name="fetch",
    base_url="http://localhost:3000"
)

# Connect via stdio
tools = await mcp_manager.register_stdio_server(
    name="fetch",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-fetch", "--transport", "stdio"]
)
"""
    },
    {
        "name": "Math",
        "description": "Simple math operations (example server)",
        "repo": "N/A (custom example)",
        "install_command": "N/A (see math_server.py example)",
        "run_command": "python math_server.py",
        "transport_modes": ["stdio"],
        "default_port": None,
        "example": """
# Connect via stdio
tools = await mcp_manager.register_stdio_server(
    name="math",
    command="python",
    args=["math_server.py"]
)
"""
    },
    {
        "name": "Playwright",
        "description": "Browser automation capabilities using Playwright's accessibility tree.",
        "repo": "https://github.com/microsoft/playwright-mcp", # Note: Inferring repo URL, might need adjustment if official URL differs
        "install_command": "npx @playwright/mcp@latest --help", # Use --help to check install/availability
        "run_command": "npx @playwright/mcp@latest",
        "transport_modes": ["stdio", "http"], # Assuming both based on docs
        "default_port": 8931, # Based on SSE example, might not be default for non-SSE HTTP
        "example": """
# Connect via stdio (headless)
tools = await mcp_manager.register_stdio_server(
    name="playwright",
    command="npx",
    args=["@playwright/mcp@latest", "--headless"]
)

# Connect via stdio (headed)
# tools = await mcp_manager.register_stdio_server(
#     name="playwright",
#     command="npx",
#     args=["@playwright/mcp@latest"]
# )

# Connect via HTTP (requires running server separately with --port)
# npx @playwright/mcp@latest --port 8931
# tools = await mcp_manager.register_http_server(
#     name="playwright",
#     url="http://localhost:8931/sse" # Using SSE endpoint from docs
# )
"""
    }
]

def check_if_installed(server: Dict) -> bool:
    """Check if an MCP server is installed."""
    if server["name"] == "Math":
        return True  # Custom example, always available
        
    try:
        if server["install_command"].startswith("npm"):
            # For npm packages, try using npx to check if it's available
            package_name = server["install_command"].split()[-1]
            cmd = f"npx -y {package_name} --version"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)  # Added 5 second timeout
            return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        pass
        
    return False

def check_if_running(server: Dict) -> bool:
    """Check if an MCP server is running."""
    if "http" not in server["transport_modes"] or server["default_port"] is None:
        return False
        
    try:
        import requests
        url = f"http://localhost:{server['default_port']}/mcp/v1/initialize"
        response = requests.get(url, timeout=1)
        return response.status_code == 200
    except Exception:
        return False

def print_server_info(server: Dict) -> None:
    """Print information about an MCP server."""
    installed = check_if_installed(server)
    running = check_if_running(server)
    
    print(f"{'=' * 50}")
    print(f"Server: {server['name']}")
    print(f"Description: {server['description']}")
    print(f"Repository: {server['repo']}")
    print(f"Install Command: {server['install_command']}")
    print(f"Run Command: {server['run_command']}")
    print(f"Transport Modes: {', '.join(server['transport_modes'])}")
    if server["default_port"]:
        print(f"Default Port: {server['default_port']}")
    print(f"Status: {'Installed' if installed else 'Not Installed'}")
    if "http" in server["transport_modes"]:
        print(f"Running: {'Yes' if running else 'No'}")
    print(f"\nUsage Example:\n{server['example']}")
    
def list_all_servers() -> None:
    """List all known MCP servers."""
    print("Available MCP Servers:")
    print("=" * 50)
    
    for server in MCP_SERVERS:
        print_server_info(server)
        print()
        
def install_server(server_name: str) -> None:
    """Install an MCP server."""
    server = next((s for s in MCP_SERVERS if s["name"].lower() == server_name.lower()), None)
    if not server:
        print(f"Server '{server_name}' not found.")
        return
        
    if server["install_command"] == "N/A (see math_server.py example)":
        print("This is a custom example server. No installation needed.")
        return
        
    print(f"Installing {server['name']} server...")
    result = subprocess.run(server["install_command"], shell=True)
    
    if result.returncode == 0:
        print(f"{server['name']} server installed successfully.")
    else:
        print(f"Failed to install {server['name']} server.")
        
def run_server(server_name: str) -> None:
    """Run an MCP server."""
    server = next((s for s in MCP_SERVERS if s["name"].lower() == server_name.lower()), None)
    if not server:
        print(f"Server '{server_name}' not found.")
        return
        
    if server["run_command"] == "N/A":
        print("This server cannot be run directly.")
        return
        
    print(f"Running {server['name']} server...")
    print(f"Command: {server['run_command']}")
    print("Press Ctrl+C to stop the server.")
    
    try:
        subprocess.run(server["run_command"], shell=True)
    except KeyboardInterrupt:
        print("Server stopped.")

def print_usage() -> None:
    """Print usage instructions."""
    print("Usage:")
    print("  python list_mcp_servers.py [command] [server_name]")
    print("\nCommands:")
    print("  list                List all available MCP servers")
    print("  info <server_name>  Show information about a specific server")
    print("  install <server_name>  Install a specific server")
    print("  run <server_name>   Run a specific server")
    print("\nExamples:")
    print("  python list_mcp_servers.py list")
    print("  python list_mcp_servers.py info puppeteer")
    print("  python list_mcp_servers.py install puppeteer")
    print("  python list_mcp_servers.py run puppeteer")
    
def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        return
        
    command = sys.argv[1].lower()
    
    if command == "list":
        list_all_servers()
    elif command == "info" and len(sys.argv) >= 3:
        server = next((s for s in MCP_SERVERS if s["name"].lower() == sys.argv[2].lower()), None)
        if server:
            print_server_info(server)
        else:
            print(f"Server '{sys.argv[2]}' not found.")
    elif command == "install" and len(sys.argv) >= 3:
        install_server(sys.argv[2])
    elif command == "run" and len(sys.argv) >= 3:
        run_server(sys.argv[2])
    else:
        print_usage()
        
if __name__ == "__main__":
    main() 