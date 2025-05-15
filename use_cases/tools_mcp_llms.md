# Configuring MCP for llms.txt Files in Claude Desktop and Cursor

## Understanding llms.txt and MCP

Before configuring your MCP clients, it's important to understand the two components involved:

1. **llms.txt**: A website index format that provides background information, guidance, and links to detailed documentation for LLMs. As described in the LangChain documentation, llms.txt is "an index file containing links with brief descriptions of the content"[1]. It acts as a structured gateway to a project's documentation.

2. **MCP (Model Context Protocol)**: A protocol enabling communication between AI agents and external tools, allowing LLMs to discover and use various capabilities. As stated by Anthropic, MCP is "an open protocol that standardizes how applications provide context to LLMs"[2].

The mcpdoc server, created by LangChain, "create[s] an open source MCP server to provide MCP host applications (e.g., Cursor, Windsurf, Claude Code/Desktop) with (1) a user-defined list of llms.txt files and (2) a simple fetch_docs tool read URLs within any of the provided llms.txt files"[3]. This bridges llms.txt with MCP, giving developers full control over how documentation is accessed.

References:
1. LangChain LLMS-txt Overview (https://langchain-ai.github.io/langgraph/llms-txt-overview/)
2. Model Context Protocol Introduction (https://modelcontextprotocol.io/introduction)
3. LangChain mcpdoc GitHub Repository (https://github.com/langchain-ai/mcpdoc)

## Claude Desktop Configuration

### Step 1: Install Prerequisites

1. Ensure you have Python installed
2. Install UV (Universal Python Wrapper):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

### Step 2: Configure Claude Desktop

1. Navigate to Claude Desktop settings
2. Open the MCP configuration section
3. Add the following JSON configuration:

```json
{
  "mcpServers": {
    "documentation-server": {
      "command": "uvx",
      "args": [
        "--from",
        "mcpdoc",
        "mcpdoc",
        "--urls",
        "PydanticAI:https://ai.pydantic.dev/llms.txt",
        "PydanticAI Full:https://ai.pydantic.dev/llms-full.txt",
        "MCP Protocol:https://modelcontextprotocol.io/llms.txt",
        "MCP Protocol Full:https://modelcontextprotocol.io/llms-full.txt",
        "Google A2A:https://raw.githubusercontent.com/google/A2A/refs/heads/main/llms.txt",
        "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt",
        "LangChain:https://python.langchain.com/llms.txt",
         "Vercel AI SDK:https://sdk.vercel.ai/llms.txt",
        "--transport",
        "stdio"
      ],
      "description": "Documentation server for multiple AI frameworks"
    }
  }
}
```

This configuration uses the mcpdoc server command with multiple URLs specified via the `--urls` parameter, as documented in the mcpdoc README: "You can specify multiple URLs by using the --urls parameter multiple times"[3].

### Step 3: Security Considerations

The mcpdoc server implements strict domain access controls as documented in the LangChain repository:

"When you specify a remote llms.txt URL (e.g., https://langchain-ai.github.io/langgraph/llms.txt), mcpdoc automatically adds only that specific domain (langchain-ai.github.io) to the allowed domains list. This means the tool can only fetch documentation from URLs on that domain"[3].

For local files, the documentation states: "When using a local file, NO domains are automatically added to the allowed list. You MUST explicitly specify which domains to allow using the --allowed-domains parameter"[3].

Key security guidelines:
- For remote llms.txt files, only the domain of the specified URL is automatically allowed[3]
- For local files, you must explicitly specify allowed domains using `--allowed-domains`[3]
- To allow additional domains, add: `--allowed-domains domain1.com domain2.com`[3]
- Use `--allowed-domains '*'` to allow all domains (use with caution)[3]

## Cursor Configuration

### Step 1: Open Configuration File

According to the mcpdoc documentation, to configure Cursor:
1. "Open Cursor Settings and MCP tab. This will open the ~/.cursor/mcp.json file"[3]
2. Navigate to the MCP tab
3. This opens `~/.cursor/mcp.json`

### Step 2: Add Configuration

The mcpdoc documentation provides a specific format for Cursor configuration[3]. Here's the recommended configuration:

```json
{
  "mcpServers": {
    "ai-docs-server": {
      "command": "uvx",
      "args": [
        "--from",
        "mcpdoc",
        "mcpdoc",
        "--urls",
        "PydanticAI:https://ai.pydantic.dev/llms.txt",
        "MCP Protocol:https://modelcontextprotocol.io/llms.txt",
        "Google A2A:https://raw.githubusercontent.com/google/A2A/refs/heads/main/llms.txt",
        "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt",
        "LangChain:https://python.langchain.com/llms.txt",
        "Vercel AI SDK:https://sdk.vercel.ai/llms.txt",
        "--transport",
        "stdio",
        "--allowed-domains",
        "ai.pydantic.dev",
        "modelcontextprotocol.io",
        "raw.githubusercontent.com",
        "langchain-ai.github.io",
        "python.langchain.com",
        "sdk.vercel.ai"
      ]
    }
  }
}
```

This configuration follows the example provided in the mcpdoc documentation, which shows how to specify multiple URLs and configure domain access[3].

### Step 3: Update Cursor Rules

The mcpdoc documentation recommends updating Cursor Global (User) Rules for optimal usage[3]. According to their guide, "Best practice is to then update Cursor Global (User) rules"[3]:

```
<rules>
for ANY question about LangGraph, use the langgraph-docs-mcp server to help answer --
+ call list_doc_sources tool to get the available llms.txt file
+ call fetch_docs tool to read it
+ reflect on the urls in llms.txt
+ reflect on the input question
+ call fetch_docs on any urls relevant to the question
</rules>
```

This rule structure should be adapted for each framework in your configuration, as demonstrated in the mcpdoc documentation[3].

## Best Practices

### 1. Multiple URLs in Single Entry

As documented in the mcpdoc repository: "You can specify multiple URLs by using the --urls parameter multiple times"[3]. The documentation provides this example: "uvx --from mcpdoc mcpdoc \ --urls \"LangGraph:https://langchain-ai.github.io/langgraph/llms.txt\" \"LangChain:https://python.langchain.com/llms.txt\""[3].

### 2. Using llms.txt vs llms-full.txt

According to the LangChain documentation:
- `llms.txt`: "is an index file containing links with brief descriptions of the content"[1]
- `llms-full.txt`: "includes all the detailed content directly in a single file, eliminating the need for additional navigation"[1]

The documentation notes: "A key consideration when using llms-full.txt is its size. For extensive documentation, this file may become too large to fit into an LLM's context window"[1].

### 3. Test Your Configuration

The mcpdoc documentation provides specific instructions for testing your configuration[3]:

```bash
uvx --from mcpdoc mcpdoc \
    --urls "Test:https://your-test-url.com/llms.txt" \
    --transport sse \
    --port 8082 \
    --host localhost
```

"Run MCP inspector and connect to the running server: npx @modelcontextprotocol/inspector"[3]

The documentation notes: "Here, you can test the tool calls"[3].

## Alternative Configurations

### Using YAML or JSON Files

The mcpdoc documentation states: "You can specify documentation sources in three ways, and these can be combined"[3]:

```yaml
# sample_config.yaml
- name: LangGraph Python
  llms_txt: https://langchain-ai.github.io/langgraph/llms.txt
```

As shown in the documentation: "This will load the LangGraph Python documentation from the sample_config.yaml file in this repo"[3].

Reference in your configuration:
```json
{
  "mcpServers": {
    "docs-from-file": {
      "command": "uvx",
      "args": [
        "--from",
        "mcpdoc",
        "mcpdoc",
        "--yaml",
        "path/to/sample_config.yaml",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

According to the documentation: "Both YAML and JSON configuration files should contain a list of documentation sources"[3].

### Local llms.txt Files

For local files, always specify allowed domains:
```json
{
  "mcpServers": {
    "local-docs": {
      "command": "uvx",
      "args": [
        "--from",
        "mcpdoc",
        "mcpdoc",
        "--urls",
        "Local Docs:/path/to/local/llms.txt",
        "--allowed-domains",
        "docs.example.com",
        "api.example.com",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

## Using the Documentation Server in Claude Desktop

Once configured, you can access documentation through the MCP server using two main tools:

### 1. List Documentation Sources

To see all available documentation sources, use the `list_doc_sources` tool. This will show you the configured llms.txt files and their names:

```
Tool: list_doc_sources
```

This will return a list like:
- PydanticAI: https://ai.pydantic.dev/llms.txt
- MCP Protocol: https://modelcontextprotocol.io/llms.txt
- LangGraph: https://langchain-ai.github.io/langgraph/llms.txt
- LangChain: https://python.langchain.com/llms.txt

### 2. Fetch Documentation Content

To retrieve specific documentation, use the `fetch_docs` tool with the URL of the desired content:

```
Tool: fetch_docs
URL: https://ai.pydantic.dev/agents/index.md
```

### Advanced Use Cases

Here are powerful examples of how to leverage the MCP documentation server:

#### Feature Comparison Across Frameworks

You: "Compare the agent architectures between PydanticAI and LangGraph. How do they handle system prompts, tools, and dependency injection?"

Claude's workflow:
1. Uses the mcpdoc server to access documentation from both frameworks[1]
2. Analyzes system prompt configurations in both frameworks
3. Compares tool registration mechanisms
4. Examines dependency injection patterns
5. Creates comparative analysis as described in the MCP documentation: "gives developers the best way to provide contextual data to LLMs and AI assistants to solve problems"[4]

#### Identifying Integration Opportunities

You: "Can you analyze how I could integrate a LangGraph agent with PydanticAI's MCP client for real-time data fetching?"

Claude's workflow:
1. Accesses documentation via the mcpdoc server[3]
2. Examines integration options as described in MCP protocols: "enables seamless integration between LLM applications and external data sources and tools"[6]
3. Identifies compatible interfaces and data formats
4. Suggests bridge code for message passing
5. Provides implementation examples using MCP's two-way communication feature[7]

#### API Compatibility Analysis

You: "Which model providers are commonly supported across PydanticAI, LangChain, and LangGraph? How do their APIs differ?"

Claude's workflow:
1. Retrieves model provider documentation from all three frameworks
2. Maps common providers (OpenAI, Anthropic, Google, etc.)
3. Analyzes API differences for model initialization
4. Identifies compatibility layers and adapters
5. Creates a unified interface recommendation

#### Dependency Graph Exploration

You: "Map out the dependency relationship between MCP servers, tools, and resources as implemented in PydanticAI versus the Model Context Protocol spec."

Claude's workflow:
1. Fetches MCP specification documentation
2. Analyzes PydanticAI's MCP implementation
3. Creates a hierarchical diagram of components
4. Highlights implementation deviations
5. Suggests standardization improvements

#### Code Migration Guidance

You: "I have a LangChain agent using OpenAI and vector stores. How can I migrate this to PydanticAI while maintaining similar functionality?"

Claude's workflow:
1. Examines LangChain's agent patterns and vector store usage
2. Analyzes PydanticAI's equivalent features
3. Maps concepts between frameworks
4. Provides step-by-step migration guide
5. Highlights potential pitfalls and solutions

#### Architecture Pattern Analysis

You: "Compare how streaming responses are handled in LangGraph, PydanticAI, and the core MCP protocol. Which patterns should I adopt for real-time applications?"

Claude's workflow:
1. Retrieves streaming documentation from all sources
2. Analyzes implementation patterns
3. Evaluates performance implications
4. Recommends architecture based on use case
5. Provides code examples for each pattern

#### Cross-Framework Tool Design

You: "I need to create a tool that works across MCP-compatible frameworks. What's the common interface pattern?"

Claude's workflow:
1. Fetches tool specifications from MCP protocol
2. Analyzes tool implementations in PydanticAI and LangGraph
3. Identifies common interfaces and parameters
4. Suggests a universal tool template
5. Provides validation and testing strategies

### Expert Tips for Maximizing MCP Value

Based on MCP best practices and documentation:

1. **Use Cross-Reference Queries**: As MCP enables "dynamic discovery" of capabilities[7], ask Claude to find references to specific concepts across multiple frameworks simultaneously

2. **Request Compatibility Matrices**: MCP's standardized protocol allows for "comparing model context protocol server frameworks"[8], helping you get detailed compatibility information

3. **Explore Edge Cases**: Ask about framework-specific limitations and workarounds using documentation insights from the llms.txt index[1]

4. **Version-Aware Analysis**: Include version numbers in queries to ensure compatibility, as recommended for documentation access[3]

5. **Performance Comparisons**: Request benchmarking data or performance considerations from framework documentation, utilizing MCP's ability to "connect with 100+ MCP servers"[4]

### Understanding MCP's Advanced Capabilities

MCP enables sophisticated documentation access that goes beyond simple retrieval, as described in various sources:

1. **Dynamic Discovery**: "MCP allows AI models to dynamically discover and interact with available tools without hard-coded knowledge of each integration"[7]
2. **Real-time Updates**: Documentation changes are immediately available without reconfiguration[2]
3. **Contextual Understanding**: "MCP gives developers the best way to provide contextual data to LLMs and AI assistants to solve problems"[4]
4. **Cross-Framework Analysis**: Seamlessly compare features across different ecosystems[8]
5. **Integration Insights**: MCP provides "standardization for connecting LLMs with external tools & data"[9], identifying patterns that may not be obvious from individual documentation

References:
1. LangChain LLMS-txt Overview (https://langchain-ai.github.io/langgraph/llms-txt-overview/)
2. Model Context Protocol Introduction (https://modelcontextprotocol.io/introduction)
3. LangChain mcpdoc GitHub Repository (https://github.com/langchain-ai/mcpdoc)
4. The Top 7 MCP-Supported AI Frameworks (https://getstream.io/blog/mcp-llms-agents/)
5. Google A2A Protocol High-Level Summary (https://raw.githubusercontent.com/google/A2A/refs/heads/main/llms.txt)
6. Model Context Protocol GitHub Organization (https://github.com/modelcontextprotocol)
7. MCP vs API Explained (https://norahsakal.com/blog/mcp-vs-api-model-context-protocol-explained/)
8. Comparing Model Context Protocol Server Frameworks (https://medium.com/@FrankGoortani/comparing-model-context-protocol-mcp-server-frameworks-03df586118fd)
9. What is MCP? (https://addyo.substack.com/p/mcp-what-it-is-and-why-it-matters)

## Troubleshooting

According to the mcpdoc documentation[3]:

1. **Server not starting**: Check that UV is properly installed ("Please see official uv docs for other ways to install uv"[3])
2. **Permission issues**: Ensure the user has access to read the configuration files
3. **Domains blocked**: Verify that required domains are included in `--allowed-domains`[3]
4. **Tool calls failing**: "Confirm that the server is running in your Cursor Settings/MCP tab"[3]
5. **Tool availability**: Ensure MCP is enabled in Claude Desktop settings

By following this guide, you'll have a robust MCP configuration that provides seamless access to documentation across multiple AI frameworks in both Claude Desktop and Cursor, along with practical knowledge of how to effectively use the documentation server in your daily workflow.
