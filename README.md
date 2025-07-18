# Open Deep Research

<img width="1388" height="298" alt="full_diagram" src="https://github.com/user-attachments/assets/12a2371b-8be2-4219-9b48-90503eb43c69" />

Deep research has broken out as one of the most popular agent applications. This is a simple, configurable, fully open source deep research agent that works across many model providers, search tools, and MCP servers. 

* Read more in our [blog](https://blog.langchain.com/open-deep-research/) 
* See our [video](https://www.youtube.com/watch?v=agGiWUpxkhg) for a quick overview

### 🚀 Quickstart

1. Clone the repository and activate a virtual environment:
```bash
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
uv pip install -r pyproject.toml
```

3. Set up your `.env` file to customize the environment variables (for model selection, search tools, and other configuration settings):
```bash
cp .env.example .env
```

4. Launch the assistant with the LangGraph server locally to open LangGraph Studio in your browser:

```bash
# Install dependencies and start the LangGraph server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

Use this to open the Studio UI:
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```
<img width="817" height="666" alt="Screenshot 2025-07-13 at 11 21 12 PM" src="https://github.com/user-attachments/assets/052f2ed3-c664-4a4f-8ec2-074349dcaa3f" />

Ask a question in the `messages` input field and click `Submit`.

### Configurations

Open Deep Research offers extensive configuration options to customize the research process and model behavior. All configurations can be set via the web UI, environment variables, or by modifying the configuration directly.

#### General Settings

- **Max Structured Output Retries** (default: 3): Maximum number of retries for structured output calls from models when parsing fails
- **Allow Clarification** (default: true): Whether to allow the researcher to ask clarifying questions before starting research
- **Max Concurrent Research Units** (default: 5): Maximum number of research units to run concurrently using sub-agents. Higher values enable faster research but may hit rate limits

#### Research Configuration

- **Search API** (default: Tavily): Choose from Tavily (works with all models), OpenAI Native Web Search, Anthropic Native Web Search, or None
- **Max Researcher Iterations** (default: 3): Number of times the Research Supervisor will reflect on research and ask follow-up questions
- **Max React Tool Calls** (default: 5): Maximum number of tool calling iterations in a single researcher step

#### Models

Open Deep Research uses multiple specialized models for different research tasks:

- **Summarization Model** (default: `openai:gpt-4.1-nano`): Summarizes research results from search APIs
- **Research Model** (default: `openai:gpt-4.1`): Conducts research and analysis 
- **Compression Model** (default: `openai:gpt-4.1-mini`): Compresses research findings from sub-agents
- **Final Report Model** (default: `openai:gpt-4.1`): Writes the final comprehensive report

All models are configured using [init_chat_model() API](https://python.langchain.com/docs/how_to/chat_models_universal_init/) which supports providers like OpenAI, Anthropic, Google Vertex AI, and others.

**Important Model Requirements:**

1. **Structured Outputs**: All models must support structured outputs. Check support [here](https://python.langchain.com/docs/integrations/chat/).

2. **Search API Compatibility**: Research and Compression models must support your selected search API:
   - Anthropic search requires Anthropic models with web search capability
   - OpenAI search requires OpenAI models with web search capability  
   - Tavily works with all models

3. **Tool Calling**: All models must support tool calling functionality

4. **Special Configurations**:
   - For OpenRouter: Follow [this guide](https://github.com/langchain-ai/open_deep_research/issues/75#issuecomment-2811472408)
   - For local models via Ollama: See [setup instructions](https://github.com/langchain-ai/open_deep_research/issues/65#issuecomment-2743586318)

#### Example MCP (Model Context Protocol) Servers

Open Deep Research supports MCP servers to extend research capabilities. 

#### Local MCP Servers

**Filesystem MCP Server** provides secure file system operations with robust access control:
- Read, write, and manage files and directories
- Perform operations like reading file contents, creating directories, moving files, and searching
- Restrict operations to predefined directories for security
- Support for both command-line configuration and dynamic MCP roots

Example usage:
```bash
mcp-server-filesystem /path/to/allowed/dir1 /path/to/allowed/dir2
```

#### Remote MCP Servers  

**Remote MCP servers** enable distributed agent coordination and support streamable HTTP requests. Unlike local servers, they can be multi-tenant and require more complex authentication.

**Arcade MCP Server Example**:
```json
{
  "url": "https://api.arcade.dev/v1/mcps/ms_0ujssxh0cECutqzMgbtXSGnjorm",
  "tools": ["Search_SearchHotels", "Search_SearchOneWayFlights", "Search_SearchRoundtripFlights"]
}
```

Remote servers can be configured as authenticated or unauthenticated and support JWT-based authentication through OAuth endpoints.

### Evaluation

A comprehensive batch evaluation system designed for detailed analysis and comparative studies.

#### **Features:**
- **Multi-dimensional Scoring**: Specialized evaluators with 0-1 scale ratings
- **Dataset-driven Evaluation**: Batch processing across multiple test cases

#### **Usage:**
```bash
# Run comprehensive evaluation on LangSmith datasets
python tests/run_evaluate.py
```
#### **Key Files:**
- `tests/run_evaluate.py`: Main evaluation script
- `tests/evaluators.py`: Specialized evaluator functions
- `tests/prompts.py`: Evaluation prompts for each dimension

### Deployments and Usages

#### LangGraph Studio

Follow the [quickstart](#-quickstart) to start LangGraph server locally and test the agent out on LangGraph Studio.

#### Command-Line Usage

To run the research process programmatically and save the final report to a file, you can use the `run_research.py` script.

```bash
python run_research.py "Your research query here"
```

This will create a `final_report.md` file in the root directory with the output.

#### Hosted deployment
 
You can easily deploy to [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options). 

#### Open Agent Platform

Open Agent Platform (OAP) is a UI from which non-technical users can build and configure their own agents. OAP is great for allowing users to configure the Deep Researcher with different MCP tools and search APIs that are best suited to their needs and the problems that they want to solve.

We've deployed Open Deep Research to our public demo instance of OAP. All you need to do is add your API Keys, and you can test out the Deep Researcher for yourself! Try it out [here](https://oap.langchain.com)

You can also deploy your own instance of OAP, and make your own custom agents (like Deep Researcher) available on it to your users.
1. [Deploy Open Agent Platform](https://docs.oap.langchain.com/quickstart)
2. [Add Deep Researcher to OAP](https://docs.oap.langchain.com/setup/agents)

### Updates 🔥

### Legacy Implementations 🏛️

The `src/legacy/` folder contains two earlier implementations that provide alternative approaches to automated research:

#### 1. Workflow Implementation (`legacy/graph.py`)
- **Plan-and-Execute**: Structured workflow with human-in-the-loop planning
- **Sequential Processing**: Creates sections one by one with reflection
- **Interactive Control**: Allows feedback and approval of report plans
- **Quality Focused**: Emphasizes accuracy through iterative refinement

#### 2. Multi-Agent Implementation (`legacy/multi_agent.py`)  
- **Supervisor-Researcher Architecture**: Coordinated multi-agent system
- **Parallel Processing**: Multiple researchers work simultaneously
- **Speed Optimized**: Faster report generation through concurrency
- **MCP Support**: Extensive Model Context Protocol integration

See `src/legacy/legacy.md` for detailed documentation, configuration options, and usage examples for both legacy implementations.
