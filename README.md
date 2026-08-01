# 🔬 Open Deep Research（开放深度研究）

<img width="1388" height="298" alt="full_diagram" src="https://github.com/user-attachments/assets/12a2371b-8be2-4219-9b48-90503eb43c69" />

深度研究（Deep Research）已成为最热门的 AI Agent 应用之一。这是一个**简单、可配置、完全开源**的深度研究 Agent，支持多种模型提供商、搜索工具和 MCP 服务器。其性能与许多流行的深度研究 Agent 相当（[参见 Deep Research Bench 排行榜](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard)）。

<img width="817" height="666" alt="Screenshot 2025-07-13 at 11 21 12 PM" src="https://github.com/user-attachments/assets/052f2ed3-c664-4a4f-8ec2-074349dcaa3f" />

### 🔥 最近更新

**2025年8月14日**：查看我们的免费课程[点这里](https://academy.langchain.com/courses/deep-research-with-langgraph)（课程仓库[点这里](https://github.com/langchain-ai/deep_research_from_scratch)），学习如何构建开放深度研究。

**2025年8月7日**：新增 GPT-5 支持，并更新了 Deep Research Bench 评估结果。

**2025年8月2日**：在 [Deep Research Bench 排行榜](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard)上取得第6名，总分 0.4344。

**2025年7月30日**：阅读我们的[博客文章](https://rlancemartin.github.io/2025/07/30/bitter_lesson/)，了解从最初实现到当前版本的演进历程。

**2025年7月16日**：阅读我们的[博客](https://blog.langchain.com/open-deep-research/)并观看[视频](https://www.youtube.com/watch?v=agGiWUpxkhg)获取快速概览。

### 🚀 快速开始

1. 克隆仓库并激活虚拟环境：
```bash
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
uv venv
source .venv/bin/activate  # Windows 系统：.venv\Scripts\activate
```

2. 安装依赖：
```bash
uv sync
# 或者
uv pip install -r pyproject.toml
```

3. 设置 `.env` 文件来自定义环境变量（用于模型选择、搜索工具和其他配置）：
```bash
cp .env.example .env
```

4. 使用 LangGraph 本地服务器启动 Agent：

```bash
# 安装依赖并启动 LangGraph 服务器
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

这将在浏览器中打开 LangGraph Studio 界面。

```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio 界面: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API 文档: http://127.0.0.1:2024/docs
```

在 `messages` 输入框中提问，然后点击 `Submit`（提交）。在 "Manage Assistants"（管理助手）标签页中选择不同的配置。

### ⚙️ 配置说明

#### 大语言模型 (LLM) 🧠

Open Deep Research 通过 [init_chat_model() API](https://python.langchain.com/docs/how_to/chat_models_universal_init/) 支持多种 LLM 提供商。它在以下几个不同任务中使用 LLM。请参阅 [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) 文件中的模型字段了解更多详情。这些配置可以通过 LangGraph Studio 界面访问。

- **摘要模型（Summarization）**（默认：`openai:gpt-4.1-mini`）：对搜索 API 结果进行摘要
- **研究模型（Research）**（默认：`openai:gpt-4.1`）：驱动搜索 Agent
- **压缩模型（Compression）**（默认：`openai:gpt-4.1`）：压缩研究发现
- **最终报告模型（Final Report Model）**（默认：`openai:gpt-4.1`）：撰写最终报告

> 注意：所选模型需要支持[结构化输出（structured outputs）](https://python.langchain.com/docs/integrations/chat/)和[工具调用（tool calling）](https://python.langchain.com/docs/how_to/tool_calling/)。

> 注意：OpenRouter 用户请参考[此指南](https://github.com/langchain-ai/open_deep_research/issues/75#issuecomment-2811472408)；通过 Ollama 使用本地模型的用户请参考[设置说明](https://github.com/langchain-ai/open_deep_research/issues/65#issuecomment-2743586318)。

#### 搜索 API 🔍

Open Deep Research 支持多种搜索工具。默认使用 [Tavily](https://www.tavily.com/) 搜索 API。完全兼容 MCP，并支持 Anthropic 和 OpenAI 的原生网页搜索。请参阅 [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) 文件中的 `search_api` 和 `mcp_config` 字段了解更多详情。这些配置可以通过 LangGraph Studio 界面访问。

#### 其他配置

请参阅 [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) 中的各个字段，了解用于自定义 Open Deep Research 行为的各种其他设置。

### 📊 评估

Open Deep Research 配置了 [Deep Research Bench](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard) 评估。该基准测试包含 100 个博士级别的研究任务（50 个英文，50 个中文），由 22 个领域（如科技、商业与金融）的领域专家精心制作，以模拟真实的深度研究需求。它有 2 个评估指标，排行榜基于 RACE 分数。使用 LLM 作为评判者（Gemini），根据专家编写的黄金标准报告对研究报告进行评估。

#### 使用方法

> 警告：运行全部 100 个示例大约需要花费 $20-$100，具体取决于模型选择。

数据集可在 [LangSmith 上通过此链接获取](https://smith.langchain.com/public/c5e7a6ad-fdba-478c-88e6-3a388459ce8b/d)。要启动评估，请运行以下命令：

```bash
# 在 LangSmith 数据集上运行综合评估
python tests/run_evaluate.py
```

这将提供一个指向 LangSmith 实验的链接，实验名称为 `YOUR_EXPERIMENT_NAME`。完成后，将结果提取为可提交到 Deep Research Bench 的 JSONL 文件。

```bash
python tests/extract_langsmith_data.py --project-name "你的实验名称" --model-name "你的模型名称" --dataset-name "deep_research_bench"
```

这将生成 `tests/expt_results/deep_research_bench_model-name.jsonl` 文件，包含所需的格式。将生成的 JSONL 文件移至 Deep Research Bench 仓库的本地克隆，并按照其[快速入门指南](https://github.com/Ayanami0730/deep_research_bench?tab=readme-ov-file#quick-start)提交评估。

#### 评估结果

| 名称 | 提交版本 | 摘要模型 | 研究模型 | 压缩模型 | 总成本 | 总 Token 数 | RACE 分数 | 实验链接 |
|------|--------|---------------|----------|-------------|------------|--------------|------------|------------|
| GPT-5 | [ca3951d](https://github.com/langchain-ai/open_deep_research/pull/168/commits) | openai:gpt-4.1-mini | openai:gpt-5 | openai:gpt-4.1 |  | 204,640,896 | 0.4943 | [链接](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-613c-4bda-8bde-f64f0422bbf3/compare?selectedSessions=4d5941c8-69ce-4f3d-8b3e-e3c99dfbd4cc&baseline=undefined) |
| 默认配置 | [6532a41](https://github.com/langchain-ai/open_deep_research/commit/6532a4176a93cc9bb2102b3d825dcefa560c85d9) | openai:gpt-4.1-mini | openai:gpt-4.1 | openai:gpt-4.1 | $45.98 | 58,015,332 | 0.4309 | [链接](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=cf4355d7-6347-47e2-a774-484f290e79bc&baseline=undefined) |
| Claude Sonnet 4 | [f877ea9](https://github.com/langchain-ai/open_deep_research/pull/163/commits/f877ea93641680879c420ea991e998b47aab9bcc) | openai:gpt-4.1-mini | anthropic:claude-sonnet-4-20250514 | openai:gpt-4.1 | $187.09 | 138,917,050 | 0.4401 | [链接](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=04f6002d-6080-4759-bcf5-9a52e57449ea&baseline=undefined) |
| Deep Research Bench 提交 | [c0a160b](https://github.com/langchain-ai/open_deep_research/commit/c0a160b57a9b5ecd4b8217c3811a14d8eff97f72) | openai:gpt-4.1-nano | openai:gpt-4.1 | openai:gpt-4.1 | $87.83 | 207,005,549 | 0.4344 | [链接](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=e6647f74-ad2f-4cb9-887e-acb38b5f73c0&baseline=undefined) |

### 🚀 部署与使用

#### LangGraph Studio

按照[快速开始](#-快速开始)的说明在本地启动 LangGraph 服务器，并在 LangGraph Studio 上测试 Agent。

#### 托管部署
 
您可以轻松部署到 [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options)。

#### 开放 Agent 平台（Open Agent Platform）

开放 Agent 平台（OAP）是一个 UI 界面，非技术用户可以在其中构建和配置自己的 Agent。OAP 非常适合让用户配置深度研究器，搭配不同的 MCP 工具和搜索 API，以最适合他们的需求和要解决的问题。

我们已将 Open Deep Research 部署到了我们的公共 OAP 演示实例。您只需添加 API 密钥，即可亲自测试深度研究器！[点击试用](https://oap.langchain.com)

您也可以部署自己的 OAP 实例，并将自己的自定义 Agent（如 Deep Researcher）提供给用户使用。
1. [部署 Open Agent Platform](https://docs.oap.langchain.com/quickstart)
2. [将 Deep Researcher 添加到 OAP](https://docs.oap.langchain.com/setup/agents)

### 旧版实现 🏛️

`src/legacy/` 文件夹包含两个早期实现，提供了自动化研究的替代方法。它们的性能不如当前实现，但提供了理解深度研究不同方法的替代思路。

#### 1. 工作流实现（`legacy/graph.py`）
- **计划-执行模式**：带有"人在回路"（human-in-the-loop）计划功能的结构化工作流
- **顺序处理**：逐个创建章节并进行反思
- **交互控制**：允许对报告计划进行反馈和审批
- **质量优先**：通过迭代优化注重准确性

#### 2. 多 Agent 实现（`legacy/multi_agent.py`）
- **主管-研究员架构**：协调的多 Agent 系统
- **并行处理**：多个研究员同时工作
- **速度优化**：通过并发实现更快的报告生成
- **MCP 支持**：广泛的 Model Context Protocol 集成
