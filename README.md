# Open Deep Research
 
Open Deep Research is a web research assistant that generates comprehensive reports on any topic following a workflow similar to [OpenAI](https://openai.com/index/introducing-deep-research/) and [Gemini](https://blog.google/products/gemini/google-gemini-deep-research/) Deep Research. However, it allows you to customize the models, prompts, report structure, search API, and research depth. Specifically, you can customize:

- provide an outline with a desired report structure
- set the planner model (e.g., DeepSeek, OpenAI reasoning model, etc)
- give feedback on the plan of report sections and iterate until user approval 
- set the search API (e.g., Tavily, Perplexity) and # of searches to run for each research iteration
- set the depth of search for each section (# of iterations of writing, reflection, search, re-write)
- customize the writer model (e.g., Anthropic)

![report-generation](https://github.com/user-attachments/assets/6595d5cd-c981-43ec-8e8b-209e4fefc596)

## 🚀 Quickstart

Ensure you have API keys set for your desired tools.

Select a web search tool (by default Open Deep Research uses Tavily):

* [Tavily API](https://tavily.com/)
* [Perplexity API](https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api)

Select a writer model (by default Open Deep Research uses Anthropic Claude 3.5 Sonnet):

* [Anthropic](https://www.anthropic.com/)
* [OpenAI](https://openai.com/)
* [Groq](https://groq.com/)
* [Gemini](https://cloud.google.com/genai)

Select a planner model (by default Open Deep Research uses OpenAI o3-mini):

* [Anthropic](https://www.anthropic.com/)
* [OpenAI](https://openai.com/)
* [Groq](https://groq.com/)
* [Gemini](https://cloud.google.com/genai)

### Using the package

(Recommended: Create a virtual environment):
```
python -m venv open_deep_research
source open_deep_research/bin/activate
```

Install:
```
pip install open-deep-research
```

See [src/open_deep_research/graph.ipynb](src/open_deep_research/graph.ipynb) for an example of usage in a Jupyter notebook.

Import and compile the graph:
```python
from langgraph.checkpoint.memory import MemorySaver
from open_deep_research.graph import builder
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```

View the graph:
```python
from IPython.display import Image, display
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

Run the graph with a desired topic and configuration:
```python
import uuid 
thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                           "search_api": "tavily",
                           "planner_provider": "openai",
                           "planner_model": "o3-mini",
                           "writer_provider": "anthropic",
                           "writer_model": "claude-3-5-sonnet-latest",
                           "max_search_depth": 1,
                           }}

topic = "Overview of the AI inference market with focus on Fireworks, Together.ai, Groq"
async for event in graph.astream({"topic":topic,}, thread, stream_mode="updates"):
    print(event)
    print("\n")
```

The graph will stop when the report plan is generated, and you can pass feedback to update the report plan:
```python
from langgraph.types import Command
async for event in graph.astream(Command(resume="Include a revenue estimate (ARR) in the sections"), thread, stream_mode="updates"):
    print(event)
    print("\n")
```

When you are satisfied with the report plan, you can pass `True` to proceed to report generation:
```
# Pass True to approve the report plan and proceed to report generation
async for event in graph.astream(Command(resume=True), thread, stream_mode="updates"):
    print(event)
    print("\n")
```

### Running LangGraph Studio UI locally

Clone the repository:
```bash
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
```

Edit the `.env` file with your API keys (e.g., the API keys for default selections are shown below):

```bash
cp .env.example .env
```

Set:
```bash
export TAVILY_API_KEY=<your_tavily_api_key>
export ANTHROPIC_API_KEY=<your_anthropic_api_key>
export OPENAI_API_KEY=<your_openai_api_key>
export GEMINI_API_KEY=<your_gemini_api_key>
```

Launch the assistant with the LangGraph server locally, which will open in your browser:

#### Mac

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and start the LangGraph server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```

#### Windows

```powershell
# Install dependencies 
pip install -e .
pip install langgraph-cli[inmem]

# Start the LangGraph server
langgraph dev
```

Use this to open the Studio UI:
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

(1) Provide a `Topic` and hit `Submit`:

<img width="1326" alt="input" src="https://github.com/user-attachments/assets/de264b1b-8ea5-4090-8e72-e1ef1230262f" />

(2) This will generate a report plan and present it to the user for review.

(3) We can pass a string (`"..."`) with feedback to regenerate the plan based on the feedback.

<img width="1326" alt="feedback" src="https://github.com/user-attachments/assets/c308e888-4642-4c74-bc78-76576a2da919" />

(4) Or, we can just pass `true` to accept the plan.

<img width="1480" alt="accept" src="https://github.com/user-attachments/assets/ddeeb33b-fdce-494f-af8b-bd2acc1cef06" />

(5) Once accepted, the report sections will be generated.

<img width="1326" alt="report_gen" src="https://github.com/user-attachments/assets/74ff01cc-e7ed-47b8-bd0c-4ef615253c46" />

The report is produced as markdown.

<img width="1326" alt="report" src="https://github.com/user-attachments/assets/92d9f7b7-3aea-4025-be99-7fb0d4b47289" />

## 📖 Customizing the report

You can customize the research assistant's behavior through several parameters:

- `report_structure`: Define a custom structure for your report (defaults to a standard research report format)
- `number_of_queries`: Number of search queries to generate per section (default: 2)
- `max_search_depth`: Maximum number of reflection and search iterations (default: 2)
- `planner_provider`: Model provider for planning phase (default: "openai", but can be "groq")
- `planner_model`: Specific model for planning (default: "o3-mini", but can be any Groq hosted model such as "deepseek-r1-distill-llama-70b")
- `writer_model`: Model for writing the report (default: "claude-3-5-sonnet-latest")
- `search_api`: API to use for web searches (default: Tavily)

These configurations allow you to fine-tune the research process based on your needs, from adjusting the depth of research to selecting specific AI models for different phases of report generation.

### Model Considerations

(1) With Groq, there are token per minute (TPM) limits if you are on the `on_demand` service tier:
- The `on_demand` service tier has a limit of `6000 TPM`
- You will want a [paid plan](https://github.com/cline/cline/issues/47#issuecomment-2640992272) for section writing with Groq models

(2) `deepseek` [isn't great at function calling](https://api-docs.deepseek.com/guides/reasoning_model). Our assistant uses function calling to generate structured outputs for report sections and search queries within each section.  
- Because, section writing performs a larger number of function calls, OpenAI, Anthropic, and certain OSS models that are stromng at function calling like Groq's `llama-3.3-70b-versatile` are advised.
- If you see the following error, it is likely due to the model not being able to produce structured outputs (see [trace](https://smith.langchain.com/public/8a6da065-3b8b-4a92-8df7-5468da336cbe/r)):
```
groq.APIError: Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.
```

## How it works
   
1. `Plan and Execute` - Open Deep Research follows a [plan-and-execute workflow](https://github.com/assafelovic/gpt-researcher) that separates planning from research, allowing for human-in-the-loop approval of a report plan before the more time-consuming research phase. It uses, by default, a [reasoning model](https://www.youtube.com/watch?v=f0RbwrBcFmc) to plan the report sections. During this phase, it uses web search to gather general information about the report topic to help in planning the report sections. But, it also accepts a report structure from the user to help guide the report sections as well as human feedback on the report plan.
   
2. `Research and Write` - Each section of the report is written in parallel. The research assistant uses web search via [Tavily API](https://tavily.com/) or [Perplexity](https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api) to gather information about each section topic. It will reflect on each report section and suggest follow-up questions for web search. This "depth" of research will proceed for any many iterations as the user wants. Any final sections, such as introductions and conclusions, are written after the main body of the report is written, which helps ensure that the report is cohesive and coherent. The planner determines main body versus final sections during the planning phase.

3. `Managing different types` - Open Deep Research is built on LangGraph, which has native support for configuration management [using assistants](https://langchain-ai.github.io/langgraph/concepts/assistants/). The report `structure` is a field in the graph configuration, which allows users to create different assistants for different types of reports. 

## UX

### Local deployment

Follow the [quickstart](#quickstart) to start LangGraph server locally.

### Hosted deployment
 
You can easily deploy to [LangGraph Platform ](https://langchain-ai.github.io/langgraph/concepts/#deployment-options). 
