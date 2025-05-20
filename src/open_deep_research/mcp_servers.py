import os

MCP_SERVERS = {
    "playwright": {
        "command": "npx",
        "args": ["@playwright/mcp"],
        "transport": "stdio",
        "env": {"PATH": os.environ.get("PATH", "")}
    },
    "langgraph-docs-mcp": {
        "command": "uvx",
        "args": [
            "--from",
            "mcpdoc",
            "mcpdoc",
            "--urls",
            "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt",
            "--transport",
            "stdio"
        ],
        "transport": "stdio",
        "env": {"PATH": os.environ.get("PATH", "")}
    }, 
     "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/rlm/Desktop/Code/open_deep_research/src/open_deep_research/files",
      ]
    }
}

# MCP tool usage prompts
MCP_PROMPTS = {
    "playwright": """
Step 1: Use the Playwright browser_navigate tool to navigate to https://langchain-ai.github.io/langgraph/ to gather information about the user's topic.
""",
    "langgraph-docs-mcp": """
Step 1: Call `list_doc_sources` to find the available `llms.txt` file.
Step 2: Use `fetch_docs` to read the contents of `llms.txt`.
Step 3: Review the URLs listed in the file and decide which relate to the user's topic.
Step 4: Reflect again on the user's research topic to refine your understanding.
Step 5: Call `fetch_docs` on any relevant URLs to gather source information.
""",
    "filesystem": """
Step 1: Use the `list_allowed_directories` tool to get the list of allowed directories.
Step 2: Use the `read_file` tool to read files in the allowed directory.
"""
}