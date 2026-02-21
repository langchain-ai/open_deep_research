# Tool Calling Mechanism for Research Agent

This directory contains the implementation of a tool calling mechanism for the research agent. The implementation follows the LangChain tool calling pattern and provides a standardized interface for search tools.

## Components

### Search Tools

- `GeneralSearchTool`: A tool for general web search
- `AcademicSearchTool`: A tool for academic and scholarly search
- `GithubSearchTool`: A tool for GitHub and code-related search
- `LinkedinSearchTool`: A tool for LinkedIn and professional profile search

Each tool follows a standardized interface and provides a consistent output format.

### Tool Registry

The `SearchToolRegistry` provides a central registry for search tools. It allows tools to be registered, retrieved, and managed in a central location.

### Tool Executor

The `ToolExecutor` is responsible for executing tools based on their name and parameters. It handles both synchronous and asynchronous tools.

## Usage

To use the tool calling mechanism, you need to:

1. Create a registry with `registry = SearchToolRegistry(config)`
2. Create an executor with `executor = ToolExecutor(registry, config)`
3. Execute a tool with `result = executor.execute_tool_sync("tool_name", {"param": "value"})`

## Testing

You can test the implementation with:

- `test_tools.py`: Tests the search tools, registry, and executor
- `simple_test.py`: Tests the research agent with the new tool calling mechanism

## Extension

To add a new search tool:

1. Create a new tool class that inherits from `BaseTool`
2. Implement the `_run` method to execute the tool
3. Register the tool with the registry using `registry.register_tool(new_tool)`

## Example

```python
# Create registry and executor
registry = SearchToolRegistry(config)
executor = ToolExecutor(registry, config)

# Execute a tool
result = executor.execute_tool_sync(
    "general_search", 
    {"query": "python langgraph framework"}
)

# Process results
formatted_sources = result.get("formatted_sources", [])
search_string = result.get("search_string", "")
tools_used = result.get("tools", [])
domains = result.get("domains", []) 