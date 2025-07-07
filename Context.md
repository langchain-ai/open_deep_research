```markdown
# CONTEXT.md – Project Rules & Invariants

> **Project codename:** *Directed Deep Research* (DDR)  
> **Tech stack:** Python 3.11 • LangChain + LangGraph • Groq Chat models • Tavily Search • Crawl4AI • Pandas + OpenPyXL

---

## 1  Project Awareness & Context

* **Always read `PLANNING.md`** at the start of a new conversation to understand the architecture, goals, style, and constraints.  
* **Check `TASK.md`** before starting a new task. If the task isn’t listed, add it with a brief description and today’s date.  
* Use the agent catalogue in `AGENTS.md` to understand role boundaries.  
* Prefer **Groq’s `llama-3` 405 B** for planning, but lean on cheaper models (`mixtral-8x7B`, `llama-3-70B`) for high-fan-out execution.

---

## 2  Code Structure & Modularity

```

ddr/
├── agents/          # Planner, Researcher, Scraper, Writer, Cleaner
│   ├── planner.py
│   ├── researcher.py
│   ├── scraper.py
│   ├── writer.py
│   └── cleaner.py
├── tools.py         # Tavily & crawl\_url wrappers
├── prompts.py       # Centralised system prompts / CoT helpers
├── data/
│   └── index/       # Ephemeral vectorstore shards
├── reports/
│   └── \*.xlsx       # Final user-visible spreadsheets
└── tests/           # Pytest suites mirroring src layout

````

* **Hard 500-line cap** per source file – refactor if you get close.  
* Pydantic v2 for all request/response schemas.  
* **LangGraph** orchestrates the DAG; each agent node may mount tools declared in `tools.py`.  
* Secrets live in `.env` — never hard-code keys.

---

## 3  Testing & Reliability

* Write **pytest** for every new function, class, or critical path.  
* Each test file must include: normal-path success, edge-case success, and intentional failure.  
* The CI workflow will run `pytest -q && ruff check . && black --check .`.

---

## 4  Style & Conventions

* **PEP 8 + type hints**; auto-format with **black**; lint with **ruff**.  
* Docstrings in **Google style**.  
* Comment non-obvious blocks with `# Reason:` so future agents understand *why*.

---

## 5  Documentation & Explainability

* Update `README.md` and in-code docstrings whenever APIs, env variables, or usage patterns change.  
* Inline comments wherever logic is opaque.

---

## 6  AI Behaviour Rules

* **Never assume missing context** – ask the user or consult docs.  
* **Never hallucinate libraries or functions**; import only verified packages.  
* **Confirm file paths** exist before referencing.  
* **Do not delete or overwrite existing code** unless the task explicitly requires it.

---

## 7  Environment Variables

| Key                         | Purpose                             |
|-----------------------------|-------------------------------------|
| `GROQ_API_KEY`              | Auth for all Groq chat models       |
| `TAVILY_API_KEY`            | Auth for Tavily web-search tool     |
| `PLAYWRIGHT_BROWSERS_PATH`  | Needed by Crawl4AI in some CI images |

---

## 8  External Services

| Service        | Library                         | Notes                                                                                  |
|----------------|---------------------------------|----------------------------------------------------------------------------------------|
| **Groq LLMs**      | `langchain_groq.ChatGroq`           | OpenAI-compatible; supports structured output                                         |
| **Tavily Search**  | *Custom wrapper in `tools.py`*      | Wrap your Tavily API key → returns JSON search results                                |
| **Crawl4AI**       | `crawl4ai.AsyncWebCrawler`          | Directed scraping tool                                                                |
| **Vector DB**      | `langchain.vectorstores.Chroma`     | In-memory by default                                                                  |

---

## 9  Examples & Patterns

### 9.1  Tavily Search Tool Wrapper
```python
from tools import tavily_search
# tools.py contains:
#   from tavily import TavilyClient
#   client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
#   def tavily_search(query: str) -> List[Document]:
#       return client.search(query, count=5)
````

### 9.2  Crawl a URL to Markdown

```python
from crawl4ai import AsyncWebCrawler

async def crawl_url(url: str) -> str:
    async with AsyncWebCrawler() as crawler:
        return (await crawler.arun(url=url)).markdown
```

([Crawl4AI GitHub repo](https://github.com/unclecode/crawl4ai))

### 9.3  Export Clean Data to Excel

```python
with pd.ExcelWriter("research_report.xlsx") as w:
    numeric_df.to_excel(w, "data_numeric", index=False)
    summary_df.to_excel(w, "report_md",  index=False)
```

Refer to DataFrame.to\_excel docs:
[https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to\_excel.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html)

---

## 10  References & Further Reading

* **Open Deep Research** – baseline workflow & multi-agent blueprint
* **Multi-agent implementation** – dynamic tool factory & state definitions
* **Crawl4AI README – Quick Start**
* **Pandas `read_html` guide** – [https://pandas.pydata.org/docs/reference/api/pandas.read\_html.html](https://pandas.pydata.org/docs/reference/api/pandas.read_html.html)

```
```
