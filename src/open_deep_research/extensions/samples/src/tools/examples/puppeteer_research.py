"""
Example script demonstrating how to use the Puppeteer MCP server for web research.

This script shows how to:
1. Start the Puppeteer MCP server
2. Register it with our tool registry
3. Use it to perform web research on a topic

Prerequisites:
- Install the MCP Puppeteer server:
  npm install -g @modelcontextprotocol/server-puppeteer

- Run the server (it should be running before running this script):
  npx -y @modelcontextprotocol/server-puppeteer
  
- Alternatively, you can run it without installing:
  npx -y @modelcontextprotocol/server-puppeteer
"""
import asyncio
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.tools.registry import ToolRegistry
from src.tools.executor import ToolExecutor
from src.tools.mcp_tools import MCPToolManager
from src.tools.search_tools import GeneralSearchTool

class WebResearcher:
    """Class for performing web research using MCP Puppeteer tools."""
    
    def __init__(self):
        """Initialize the web researcher."""
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry)
        self.mcp_manager = MCPToolManager(self.registry)
        
        # Register the general search tool
        self.search_tool = GeneralSearchTool()
        self.registry.register_tool(self.search_tool)
    
    async def setup(self):
        """Set up the MCP tools."""
        # Connect to the Puppeteer MCP server
        try:
            self.puppeteer_tools = await self.mcp_manager.register_http_server(
                name="puppeteer",
                base_url="http://localhost:3000"
            )
            print(f"Connected to Puppeteer MCP server with {len(self.puppeteer_tools)} tools")
        except Exception as e:
            print(f"Error connecting to Puppeteer MCP server: {e}")
            print("Make sure the server is running with: npx -y @modelcontextprotocol/server-puppeteer")
            sys.exit(1)
    
    async def teardown(self):
        """Clean up and close connections."""
        await self.mcp_manager.close_all()
    
    async def search_for_topic(self, topic: str) -> List[Dict[str, str]]:
        """Perform a search for a topic and return search results.
        
        Args:
            topic: The topic to search for
            
        Returns:
            List of search results with title and URL
        """
        print(f"Searching for: {topic}")
        search_results = await self.executor.execute_tool(
            "search",
            {"query": topic, "num_results": 5}
        )
        
        return search_results
    
    async def visit_and_extract_info(self, url: str) -> Dict[str, Any]:
        """Visit a URL and extract information from it.
        
        Args:
            url: The URL to visit
            
        Returns:
            Dictionary with extracted information
        """
        print(f"Visiting: {url}")
        
        # Navigate to the URL
        await self.executor.execute_tool(
            "mcp.puppeteer.navigate",
            {"url": url}
        )
        
        # Take a screenshot (optional)
        screenshot_path = f"screenshot_{url.replace('://', '_').replace('/', '_').replace('.', '_')}.png"
        await self.executor.execute_tool(
            "mcp.puppeteer.screenshot",
            {"path": screenshot_path}
        )
        print(f"Screenshot saved to: {screenshot_path}")
        
        # Get the page content
        content = await self.executor.execute_tool(
            "mcp.puppeteer.getPageContent",
            {}
        )
        
        # Get the page title
        title = await self.executor.execute_tool(
            "mcp.puppeteer.getPageTitle",
            {}
        )
        
        # Extract all links
        links = await self.executor.execute_tool(
            "mcp.puppeteer.evaluateScript",
            {"script": """
                () => {
                    const links = Array.from(document.querySelectorAll('a'));
                    return links.map(link => ({
                        text: link.textContent.trim(),
                        href: link.href
                    })).filter(link => link.text && link.href);
                }
            """}
        )
        
        # Extract main text content (simplified approach)
        main_content = await self.executor.execute_tool(
            "mcp.puppeteer.evaluateScript",
            {"script": """
                () => {
                    // Try to find the main content
                    const contentSelectors = [
                        'article', 'main', '.content', '#content',
                        '.article', '.post', '.entry', '.page-content'
                    ];
                    
                    for (const selector of contentSelectors) {
                        const element = document.querySelector(selector);
                        if (element) {
                            return element.textContent.trim();
                        }
                    }
                    
                    // Fallback: Get all paragraphs
                    const paragraphs = Array.from(document.querySelectorAll('p'));
                    return paragraphs.map(p => p.textContent.trim()).join('\\n\\n');
                }
            """}
        )
        
        return {
            "url": url,
            "title": title,
            "links": links,
            "main_content": main_content,
            "full_content": content[:1000] + "..." if len(content) > 1000 else content,
            "screenshot_path": screenshot_path
        }
    
    async def research_topic(self, topic: str, max_urls: int = 3) -> Dict[str, Any]:
        """Research a topic by searching and visiting relevant pages.
        
        Args:
            topic: The topic to research
            max_urls: Maximum number of URLs to visit
            
        Returns:
            Dictionary with research results
        """
        # Search for the topic
        search_results = await self.search_for_topic(topic)
        
        # Visit the top results
        visited_pages = []
        for i, result in enumerate(search_results[:max_urls]):
            try:
                page_info = await self.visit_and_extract_info(result["link"])
                visited_pages.append(page_info)
            except Exception as e:
                print(f"Error visiting {result['link']}: {e}")
        
        # Compile the research results
        research_results = {
            "topic": topic,
            "search_results": search_results,
            "visited_pages": visited_pages,
            "summary": f"Researched {topic} by visiting {len(visited_pages)} pages"
        }
        
        return research_results

async def main():
    # Create the web researcher
    researcher = WebResearcher()
    
    try:
        # Set up the MCP tools
        await researcher.setup()
        
        # Get the research topic from command line or use default
        topic = sys.argv[1] if len(sys.argv) > 1 else "Model Context Protocol (MCP)"
        
        # Perform the research
        research_results = await researcher.research_topic(topic)
        
        # Save the results to a JSON file
        results_file = f"{topic.replace(' ', '_')}_research.json"
        with open(results_file, "w") as f:
            json.dump(research_results, f, indent=2)
        
        print(f"Research completed. Results saved to {results_file}")
        
    finally:
        # Clean up
        await researcher.teardown()

if __name__ == "__main__":
    asyncio.run(main()) 