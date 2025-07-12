# Open Deep Research

Open Deep Research is an experimental, fully open-source research assistant that automates deep research and produces comprehensive reports on any topic. You can customize the entire research and writing process with specific models, prompts, and search tools.

You can read more about our architecture and decision making in this [blog]()

Two prior implementations - a [workflow](https://langchain-ai.github.io/langgraph/tutorials/workflows/) and our first multi-agent architecture can found in the legacy folder.

### 🚀 Quickstart

Clone the repository:
```bash
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
```

Then edit the `.env` file to customize the environment variables (for model selection, search tools, and other configuration settings):
```bash
cp .env.example .env
```

Launch the assistant with the LangGraph server locally, which will open in your browser:

#### Mac

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and start the LangGraph server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

#### Windows / Linux

```powershell
# Install dependencies 
pip install -e .
pip install -U "langgraph-cli[inmem]" 

# Start the LangGraph server
langgraph dev
```

Use this to open the Studio UI:
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

### Using the package

```bash
pip install open-deep-research
```

## Configuring Open Deep Research

Open deep research uses a supervisor-researcher architecture:

- **Supervisor Agent**: Manages the overall research process, hands off research tasks to Researcher Agents
- **Researcher Agents**: Multiple independent agents work in parallel, each responsible for researching and returning findings on a specific section
- **Search and MCP Support**: Works with Tavily for web search, and also OpenAI or Anthropic native web search, also supports MCP servers for local/external data access, or can operate without search tools using only MCP tools

You can customize the open deep research through several parameters:

- `search_api`: The researcher's web search tool. Tavily by default, can also use OpenAI or Anthropic native web search, or None to use only MCP tools
- `allow_clarification`: If true, the researcher can choose to chat with the user to gain additional context before deep research (default: true)
- `max_concurrent_research_units`: Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits. (default: 5)
- `max_researcher_iterations`: Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions. (default: 3)
- `max_react_tool_calls`: "Maximum number of tool calling iterations to make in a single researcher step. (default: 5)
- `research_model`: Model for supervisor and researcher agents (default: "openai:gpt-4.1") 
- `compression_model`: Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API. (default: "openai:gpt-4.1-mini") 
- `final_report_model`: Model for writing the final report from all research findings. (default: "openai:gpt-4.1") 
- `summarization_model`: Only used if Tavily is the Search API, model used to summarize webpages found with Tavily. (default: "openai:gpt-4.1-nano") 
- `mcp_config`: Configuration for MCP servers (optional)
- `mcp_prompt`: Additional instructions for using MCP tools (optional)


## MCP (Model Context Protocol) Support

Open Deep Research supports MCP servers to extend research capabilities beyond web search. MCP tools are available to research agents alongside traditional search tools, enabling access to local files, databases, APIs, and other data sources. Open Deep Research was built to be compatible with Open Agent Platform, which is the primary means through which we expect folks to connect to their MCP servers and expose tools to end users.

#### Arcade Example

##### Studio
MCP config:
```
{
  "url": "https://api.arcade.dev/v1/mcps/ms_0ujssxh0cECutqzMgbtXSGnjorm"
  "tools": ["Search_SearchHotels", "Search_SearchOneWayFlights", "Search_SearchRoundtripFlights"]
}
```

MCP prompt:
```
Use these tools to look up information about flights and hotels
```
## Open Agent Platform
Follow this [quickstart guide](https://docs.oap.langchain.com/quickstart) for Open Agent Platform to deploy the Deep Researcher on OAP!

Open Agent Platform is a UI from which non-technical users can build and configure their own agents. Each user can configure the Deep Researcher with different MCP tools and search APIs that are best suited to their needs and the problems that they want to solve.

## Model Considerations

(1) You can use models supported with [the `init_chat_model()` API](https://python.langchain.com/docs/how_to/chat_models_universal_init/). See full list of supported integrations [here](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html).

(2) ***All models need to support structured outputs***: Check whether structured outputs are supported by the model you are using [here](https://python.langchain.com/docs/integrations/chat/).

(3) ***The Research and Compression models need to support the Search API***: If you select Anthropic search, you need to use Anthropic models which support web search for these two configurations. If you select OpenAI search, you need to use OpenAI models which support web search for these two configurations. Tavily works with all models.

(4) ***All models need to support tool calling*** Ensure tool calling is well supoorted

(5) Follow [here[(https://github.com/langchain-ai/open_deep_research/issues/75#issuecomment-2811472408) to use with OpenRouter.

(6) For working with local models via Ollama, see [here](https://github.com/langchain-ai/open_deep_research/issues/65#issuecomment-2743586318).

## Evaluation

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

## UX

### Local deployment

Follow the [quickstart](#-quickstart) to start LangGraph server locally.

### Hosted deployment
 
You can easily deploy to [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options). 
