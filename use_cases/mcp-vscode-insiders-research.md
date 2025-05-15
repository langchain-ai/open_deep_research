# Model Context Protocol Integration with Visual Studio Code Insiders

**Date: April 6, 2025**

## Table of Contents
1. [Introduction to Model Context Protocol (MCP)](#introduction-to-model-context-protocol-mcp)
2. [VS Code Insiders and MCP Integration](#vs-code-insiders-and-mcp-integration)
3. [Setting Up MCP Servers in VS Code](#setting-up-mcp-servers-in-vs-code)
4. [Using MCP with GitHub Copilot Agent Mode](#using-mcp-with-github-copilot-agent-mode)
5. [Popular MCP Servers for VS Code](#popular-mcp-servers-for-vs-code)
6. [Real-World Examples and Use Cases](#real-world-examples-and-use-cases)
7. [Future Developments](#future-developments)
8. [Resources and References](#resources-and-references)

## Introduction to Model Context Protocol (MCP)

Model Context Protocol (MCP) is an open standard that enables AI models to interact with external tools and services through a unified interface. In VS Code, MCP support enhances GitHub Copilot's agent mode by allowing you to connect any MCP-compatible server to your agentic coding workflow.

The primary purpose of MCP is to standardize how language models discover and interact with external tools, applications, and data sources. When a user enters a chat prompt in agent mode in VS Code, the model can invoke various tools to perform tasks like file operations, accessing databases, or calling APIs in response to requests.

MCP follows a client-server architecture:
- MCP clients (like VS Code) connect to MCP servers and request actions on behalf of the AI model
- MCP servers provide one or more tools that expose specific functionalities through a well-defined interface
- The Model Context Protocol defines the message format for communication between clients and servers, including tool discovery, invocation, and response handling

By standardizing this interaction, MCP eliminates the need for custom integrations between each AI model and each tool. This allows extending AI assistants' capabilities by simply adding new MCP servers to the workspace.

## VS Code Insiders and MCP Integration

As of March 2025 (version 1.99), [Visual Studio Code - Insiders](https://code.visualstudio.com/insiders/) now officially supports Model Context Protocol (MCP) servers in agent mode. This significant addition enables VS Code and its GitHub Copilot integration to interact with a wide variety of external tools and data sources through a standardized protocol.

The March 2025 release included several key features related to MCP:
- Full MCP server support in agent mode
- Configuration options for MCP servers
- Ability to discover and use MCP servers from other applications (like Claude Desktop)
- Tools to manage MCP servers directly within VS Code

VS Code supports local standard input/output (stdio) and server-sent events (sse) for MCP server transport. Currently, of the three primitives (tools, prompts, resources), servers can only provide tools to Copilot's agent mode. Tool lists and descriptions can be updated dynamically using list changed events.

## Setting Up MCP Servers in VS Code

Setting up an MCP server in VS Code involves several steps:

1. **Configure MCP Servers**: Create a `.vscode/mcp.json` file in your workspace to share configurations with project collaborators. Avoid hardcoding sensitive information like API keys by using input variables or environment files.

2. **Example Configuration**:
   ```json
   {
     "inputs": [
       {
         "type": "promptString",
         "id": "api-key",
         "description": "API Key",
         "password": true
       }
     ],
     "servers": {
       "my-server": {
         "type": "stdio",
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-fetch"],
         "env": {
           "API_KEY": "${input:api-key}"
         }
       }
     }
   }
   ```

3. **Using Command Palette**: Run the `MCP: Add Server` command from the Command Palette and provide the server information to add a new MCP server configuration.

4. **Auto-Discovery**: VS Code can automatically detect and reuse MCP servers defined in other tools, such as Claude Desktop. Enable auto-discovery with the `chat.mcp.discovery.enabled` setting.

5. **Command-line Configuration**: You can use the VS Code command-line interface to add an MCP server to your user profile or to a workspace using the `--add-mcp` command.

## Using MCP with GitHub Copilot Agent Mode

Copilot agent mode is the next evolution in AI-assisted coding. Acting as an autonomous peer programmer, it performs multi-step coding tasks at your command — analyzing your codebase, reading relevant files, proposing file edits, and running terminal commands and tests.

To use MCP tools in agent mode:

1. Open the Chat view (⌃⌘I on Windows/Linux Ctrl+Alt+I), and select Agent mode from the dropdown.

2. Select the Tools button to view the list of available tools. Optionally, select or deselect the tools you want to use. You can search tools by typing in the search box.

3. You can then enter a prompt in the chat input box and notice how tools are automatically invoked as needed. By default, when a tool is invoked, you need to confirm the action before it is run.

When you send a request to Copilot in agent mode, it makes a prompt to the selected LLM that includes:
- Your query
- A summarized structure of the workspace
- Machine context (e.g., what OS you are using)
- Tool descriptions (and optionally tool call results)

The agent mode process operates in loops:
1. Determines the relevant context and files to edit autonomously
2. Offers both code changes and terminal commands to complete tasks
3. Monitors the correctness of code edits and terminal command output
4. Iterates to remediate issues as needed

## Popular MCP Servers for VS Code

The MCP ecosystem offers a growing collection of servers that can be integrated with VS Code. These fall into several categories:

### Reference Servers
These are maintained by the MCP team and demonstrate core capabilities:


- **Brave Search**: Web and local search using Brave's Search API
- **Fetch**: Web content fetching and conversion for efficient LLM usage
- **Filesystem**: Secure file operations with configurable access controls
- **Git**: Tools to read, search, and manipulate Git repositories
- **GitHub**: Repository management, file operations, and GitHub API integration
- **PostgreSQL**: Read-only database access with schema inspection
- **Sequential Thinking**: Dynamic and reflective problem-solving
- **Sqlite**: Database interaction and business intelligence capabilities
- **Time**: Time and timezone conversion capabilities


### Official Integrations
These are maintained by companies building production-ready MCP servers:


- **ClickHouse**: Query your ClickHouse database server
- **Cloudflare**: Deploy, configure & interrogate resources on the Cloudflare developer platform
- **E2B**: Run code in secure sandboxes
- **Grafana**: Search dashboards, investigate incidents and query datasources
- **JetBrains**: Work on your code with JetBrains IDEs
- **Kagi Search**: Search the web using Kagi's search API
- **Neo4j**: Graph database server and database-backed memory
- **Perplexity**: Connects to Perplexity's Sonar API for real-time web research
- **Zapier**: Connect AI Agents to 8,000+ apps instantly


### Community Servers
A growing set of community-developed servers demonstrates various applications:


- **Airflow**: Connects to Apache Airflow using official python client
- **Airtable**: Read and write access to Airtable databases
- **AWS**: Perform operations on AWS resources
- **Azure DevOps**: Bridge to Azure DevOps services
- **BigQuery**: Database schema inspection and query execution
- **Browser-use**: Dockerized playwright for web automation


## Real-World Examples and Use Cases

In a practical example demonstrated by Thang Chung, MCP servers can be set up for various purposes including time conversions, filesystem operations, and database queries. Using a custom MCP server with the C# SDK, developers can create tools like a time service that responds to natural language queries about time in different cities.

Another example shows connecting the Postgres MCP server to enable natural language to SQL queries. This allows the AI to analyze, plan, and generate SQL code to query databases. By configuring the MCP server to connect to a PostgreSQL database, users can ask questions like "Give me top 10 orders with highest price" and the AI will generate and execute the appropriate SQL query.

The integration can be extended to automate repetitive tasks, such as converting SQL queries to LINQ for .NET developers. This demonstrates how MCP can enhance developer experience by bridging the gap between natural language and code.

Jesse Houwing's experience shows practical considerations when running MCP servers that require specific Node.js versions. By using Node version managers like Fast Node Switcher (fnm) or Node Version Switcher (nvs), developers can ensure MCP servers run with the correct Node version even when their primary development environment uses a different version.

## Future Developments

The VS Code team has shared plans for future developments related to MCP and agent mode:
- Fine-grained undo capability
- Simplifying the context UI (working set)
- Notebook support
- Ability to auto-approve specific terminal commands
- Improved terminal tool UI
- Exploring tool extensibility and expanding MCP servers as tools for agent mode
- Unifying the chat and edits experience

The GitHub team has also announced that agent mode and MCP support are rolling out to all VS Code users, highlighting that the GitHub local MCP server equips agent mode with compelling capabilities such as searching across repositories and code, managing issues, and creating PRs.

As more developers contribute to the MCP ecosystem, we can expect a growing collection of specialized servers that enhance AI-assisted development workflows.

## Resources and References

### Official Documentation:
- [VS Code MCP Servers Documentation](https://code.visualstudio.com/docs/copilot/chat/mcp-servers)
- [GitHub Copilot Agent Mode](https://code.visualstudio.com/blogs/2025/02/24/introducing-copilot-agent-mode)
- [VS Code March 2025 Release Notes](https://code.visualstudio.com/updates/v1_99)
- [Model Context Protocol Official Site](https://modelcontextprotocol.io/)
- [MCP Servers Repository](https://github.com/modelcontextprotocol/servers)

### YouTube Videos:
1. [Visual Studio Code + Model Context Protocol (MCP) Servers Getting Started Guide](https://www.youtube.com/watch?v=iS25RFups4A) - A comprehensive guide to getting started with MCP in VS Code, covering setup, configuration, and practical examples.

2. [How MCPs Make Agents Smarter (for non-techies)](https://www.youtube.com/watch?v=m0YrxLnFPzQ) - An accessible explanation of how Model Context Protocol enhances AI agents' capabilities, with a focus on making the concept understandable to non-technical audiences.

### Community Resources:
- [Visual Studio Code + MCP Servers - First Look](https://dev.to/thangchung/visual-studio-code-model-context-protocol-mcp-servers-the-first-look-18nb) by Thang Chung
- [Running MCP using Node Version](https://jessehouwing.net/vscode-running-mcp-using-node-version/) by Jesse Houwing
- [YouTube Transcript Generator Using MCP](https://collabnix.com/youtube-transcript-generator-using-model-context-protocol-in-just-5-lines-of-code/)

### Repositories:
- [Awesome MCP Servers](https://github.com/appcypher/awesome-mcp-servers) - A curated list of MCP servers
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP YouTube Server](https://github.com/anaisbetts/mcp-youtube)

## Appendix: Sample MCP Configuration

Below is a sample MCP configuration from a `settings.json` file that demonstrates how to configure multiple MCP servers:

```json
"mcp": {
    "inputs": [],
    "servers": {
        "mcp-server-time": {
            "command": "uvx",
            "args": [
                "mcp-server-time",
                "--local-timezone=America/Los_Angeles"
            ],
            "env": {}
        },
        "brave-search": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-brave-search"
            ],
            "env": {
                "BRAVE_API_KEY": "BSAx***********************"
            }
        },
        "fetch": {
            "command": "uvx",
            "args": [
                "mcp-server-fetch"
            ]
        },
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "\\\\wsl.localhost\\Ubuntu\\home\\donbr\\aim"
            ],
            "env": {}
        },
        "qdrant": {
            "command": "uvx",
            "args": [
                "mcp-server-qdrant"
            ],
            "env": {
                "QDRANT_URL": "https://********-****-****-****-************.us-east4-0.gcp.cloud.qdrant.io:6333",
                "QDRANT_API_KEY": "eyJhbGci*****************************************",
                "COLLECTION_NAME": "mcp-anthropic-desktop",
                "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
            }
        },
        "sequential-thinking": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-sequential-thinking"
            ]
        }
    }
}
```

This configuration demonstrates several important MCP servers:
- Time conversion tool with local timezone specified
- Brave search integration for web searches
- Fetch tool for retrieving web content
- Filesystem access to a WSL Ubuntu directory
- Qdrant vector database integration for knowledge retrieval
- Sequential thinking for advanced reasoning capabilities
