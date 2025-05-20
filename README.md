# Open Deep Research

Open Deep Research is an experimental, fully open-source research assistant that automates deep research and produces comprehensive reports on any topic. It features two implementations - a [workflow](https://langchain-ai.github.io/langgraph/tutorials/workflows/) and a multi-agent architecture - each with distinct advantages. You can customize the entire research and writing process with specific models, prompts, report structure, and search tools.

#### Workflow

![open-deep-research-overview](https://github.com/user-attachments/assets/a171660d-b735-4587-ab2f-cd771f773756)

####  Multi-agent

![multi-agent-researcher](https://github.com/user-attachments/assets/3c734c3c-57aa-4bc0-85dd-74e2ec2c0880)

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

#### Multi-agent

(1) Chat with the agent about your topic of interest, and it will initiate report generation:

<img width="1326" alt="input" src="https://github.com/user-attachments/assets/dc8f59dd-14b3-4a62-ac18-d2f99c8bbe83" />

(2) The report is produced as markdown.

#### Workflow

(1) Provide a `Topic`:

<img width="1326" alt="input" src="https://github.com/user-attachments/assets/de264b1b-8ea5-4090-8e72-e1ef1230262f" />

(2) This will generate a report plan and present it to the user for review.

(3) We can pass a string (`"..."`) with feedback to regenerate the plan based on the feedback.

<img width="1326" alt="feedback" src="https://github.com/user-attachments/assets/c308e888-4642-4c74-bc78-76576a2da919" />

(4) Or, we can just pass `true` to the JSON input box in Studio accept the plan.

<img width="1480" alt="accept" src="https://github.com/user-attachments/assets/ddeeb33b-fdce-494f-af8b-bd2acc1cef06" />

(5) Once accepted, the report sections will be generated.

<img width="1326" alt="report_gen" src="https://github.com/user-attachments/assets/74ff01cc-e7ed-47b8-bd0c-4ef615253c46" />

The report is produced as markdown.

<img width="1326" alt="report" src="https://github.com/user-attachments/assets/92d9f7b7-3aea-4025-be99-7fb0d4b47289" />

### Search Tools

Available search tools:

* [Tavily API](https://tavily.com/) - General web search
* [Perplexity API](https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api) - General web search
* [Exa API](https://exa.ai/) - Powerful neural search for web content
* [ArXiv](https://arxiv.org/) - Academic papers in physics, mathematics, computer science, and more
* [PubMed](https://pubmed.ncbi.nlm.nih.gov/) - Biomedical literature from MEDLINE, life science journals, and online books
* [Linkup API](https://www.linkup.so/) - General web search
* [DuckDuckGo API](https://duckduckgo.com/) - General web search
* [Google Search API/Scrapper](https://google.com/) - Create custom search engine [here](https://programmablesearchengine.google.com/controlpanel/all) and get API key [here](https://developers.google.com/custom-search/v1/introduction)
* [Microsoft Azure AI Search](https://azure.microsoft.com/en-us/products/ai-services/ai-search) - Cloud based vector database solution 

Open Deep Research is compatible with many different LLMs: 

* You can select any model that is integrated [with the `init_chat_model()` API](https://python.langchain.com/docs/how_to/chat_models_universal_init/)
* See full list of supported integrations [here](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html)

### Using the package

```bash
pip install open-deep-research
```

See [src/open_deep_research/graph.ipynb](src/open_deep_research/graph.ipynb) and [src/open_deep_research/multi_agent.ipynb](src/open_deep_research/multi_agent.ipynb) for example usage in a Jupyter notebook:

## Open Deep Research Implementations

Open Deep Research features two distinct implementation approaches, each with its own strengths:

## 1. Graph-based Workflow Implementation (`src/open_deep_research/graph.py`)

The graph-based implementation follows a structured plan-and-execute workflow:

- **Planning Phase**: Uses a planner model to analyze the topic and generate a structured report plan
- **Human-in-the-Loop**: Allows for human feedback and approval of the report plan before proceeding
- **Sequential Research Process**: Creates sections one by one with reflection between search iterations
- **Section-Specific Research**: Each section has dedicated search queries and content retrieval
- **Supports Multiple Search Tools**: Works with all search providers (Tavily, Perplexity, Exa, ArXiv, PubMed, Linkup, etc.)

This implementation provides a more interactive experience with greater control over the report structure, making it ideal for situations where report quality and accuracy are critical.

You can customize the research assistant workflow through several parameters:

- `report_structure`: Define a custom structure for your report (defaults to a standard research report format)
- `number_of_queries`: Number of search queries to generate per section (default: 2)
- `max_search_depth`: Maximum number of reflection and search iterations (default: 2)
- `planner_provider`: Model provider for planning phase (default: "anthropic", but can be any provider from supported integrations with `init_chat_model` as listed [here](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html))
- `planner_model`: Specific model for planning (default: "claude-3-7-sonnet-latest")
- `planner_model_kwargs`: Additional parameter for planner_model
- `writer_provider`: Model provider for writing phase (default: "anthropic", but can be any provider from supported integrations with `init_chat_model` as listed [here](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html))
- `writer_model`: Model for writing the report (default: "claude-3-5-sonnet-latest")
- `writer_model_kwargs`: Additional parameter for writer_model
- `search_api`: API to use for web searches (default: "tavily", options include "perplexity", "exa", "arxiv", "pubmed", "linkup")

## 2. Multi-Agent Implementation (`src/open_deep_research/multi_agent.py`)

The multi-agent implementation uses a supervisor-researcher architecture:

- **Supervisor Agent**: Manages the overall research process, plans sections, and assembles the final report
- **Researcher Agents**: Multiple independent agents work in parallel, each responsible for researching and writing a specific section
- **Parallel Processing**: All sections are researched simultaneously, significantly reducing report generation time
- **Specialized Tool Design**: Each agent has access to specific tools for its role (search for researchers, section planning for supervisors)
- **Currently Limited to Tavily Search**: The multi-agent implementation currently only works with Tavily for search, though the framework is designed to support additional search tools in the future

This implementation focuses on efficiency and parallelization, making it ideal for faster report generation with less direct user involvement.

## Search API Configuration

Not all search APIs support additional configuration parameters. Here are the ones that do:

- **Exa**: `max_characters`, `num_results`, `include_domains`, `exclude_domains`, `subpages`
  - Note: `include_domains` and `exclude_domains` cannot be used together
  - Particularly useful when you need to narrow your research to specific trusted sources, ensure information accuracy, or when your research requires using specified domains (e.g., academic journals, government sites)
  - Provides AI-generated summaries tailored to your specific query, making it easier to extract relevant information from search results
- **ArXiv**: `load_max_docs`, `get_full_documents`, `load_all_available_meta`
- **PubMed**: `top_k_results`, `email`, `api_key`, `doc_content_chars_max`
- **Linkup**: `depth`

Example with Exa configuration:
```python
thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                           "search_api": "exa",
                           "search_api_config": {
                               "num_results": 5,
                               "include_domains": ["nature.com", "sciencedirect.com"]
                           },
                           # Other configuration...
                           }}
```

## Model Considerations

(1) You can use models supported with [the `init_chat_model()` API](https://python.langchain.com/docs/how_to/chat_models_universal_init/). See full list of supported integrations [here](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html).

(2) ***The workflow planner and writer models need to support structured outputs***: Check whether structured outputs are supported by the model you are using [here](https://python.langchain.com/docs/integrations/chat/).

(3) ***The agent models need to support tool calling:*** Ensure tool calling is well supoorted; tests have been done with Claude 3.7, o3, o3-mini, and gpt4.1. See [here](https://smith.langchain.com/public/adc5d60c-97ee-4aa0-8b2c-c776fb0d7bd6/d).

(4) With Groq, there are token per minute (TPM) limits if you are on the `on_demand` service tier:
- The `on_demand` service tier has a limit of `6000 TPM`
- You will want a [paid plan](https://github.com/cline/cline/issues/47#issuecomment-2640992272) for section writing with Groq models

(5) `deepseek-R1` [is not strong at function calling](https://api-docs.deepseek.com/guides/reasoning_model), which the assistant uses to generate structured outputs for report sections and report section grading. See example traces [here](https://smith.langchain.com/public/07d53997-4a6d-4ea8-9a1f-064a85cd6072/r).  
- Consider providers that are strong at function calling such as OpenAI, Anthropic, and certain OSS models like Groq's `llama-3.3-70b-versatile`.
- If you see the following error, it is likely due to the model not being able to produce structured outputs (see [trace](https://smith.langchain.com/public/8a6da065-3b8b-4a92-8df7-5468da336cbe/r)):
```
groq.APIError: Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.
```

(6) Follow [here[(https://github.com/langchain-ai/open_deep_research/issues/75#issuecomment-2811472408) to use with OpenRouter.

(7) For working with local models via Ollama, see [here](https://github.com/langchain-ai/open_deep_research/issues/65#issuecomment-2743586318).

## Testing Report Quality

To compare the quality of reports generated by both implementations:

```bash
# Test with default Anthropic models
python tests/run_test.py --all

# Test with OpenAI o3 models
python tests/run_test.py --all \
  --supervisor-model "openai:o3" \
  --researcher-model "openai:o3" \
  --planner-provider "openai" \
  --planner-model "o3" \
  --writer-provider "openai" \
  --writer-model "o3" \
  --eval-model "openai:o3" \
  --search-api "tavily"
```

The test results will be logged to LangSmith, allowing you to compare the quality of reports generated by each implementation with different model configurations.

## UX

### Local deployment

Follow the [quickstart](#-quickstart) to start LangGraph server locally.

### Hosted deployment
 
You can easily deploy to [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options). 
