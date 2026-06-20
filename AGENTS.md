# Open Deep Research 仓库概览

## 项目描述

Open Deep Research 是一个可配置、完全开源的深度研究智能体(deep research agent)，支持多个模型提供商、搜索工具以及 MCP（Model Context Protocol，模型上下文协议）服务器。它能够通过并行处理自动执行研究任务，并生成内容全面的研究报告。

## 仓库结构

### 根目录

* `README.md` - 完整的项目文档，包含快速入门指南
* `pyproject.toml` - Python 项目配置与依赖声明
* `langgraph.json` - LangGraph 配置文件，用于定义主图(Graph)的入口点
* `uv.lock` - UV 包管理器的依赖锁定文件
* `LICENSE` - MIT 许可证
* `.env.example` - 环境变量模板（不受版本控制追踪）

### 核心实现（`src/open_deep_research/`）

* `deep_researcher.py` - LangGraph 的主要实现文件（入口点：`deep_researcher`）
* `configuration.py` - 配置管理与设置
* `state.py` - 图状态(Graph state)定义与数据结构
* `prompts.py` - 系统提示词与提示词模板
* `utils.py` - 工具函数与辅助功能
* `files/` - 研究输出文件与示例文件

### 遗留实现（`src/legacy/`）

包含两种早期的研究实现：

* `graph.py` - 带有人机协同(human-in-the-loop)机制的规划与执行(plan-and-execute)工作流
* `multi_agent.py` - 监督者—研究员(supervisor-researcher)多智能体架构
* `legacy.md` - 遗留实现的说明文档
* `CLAUDE.md` - 面向遗留实现的 Claude 专用指令
* `tests/` - 遗留实现专用测试

### 安全模块（`src/security/`）

* `auth.py` - 用于 LangGraph 部署的身份验证处理程序

### 测试（`tests/`）

* `run_evaluate.py` - 主评估脚本，配置为在 Deep Research Bench 上运行
* `evaluators.py` - 专用评估函数
* `prompts.py` - 评估提示词与评估标准
* `pairwise_evaluation.py` - 对比评估工具
* `supervisor_parallel_evaluation.py` - 多线程并行评估

### 示例（`examples/`）

* `arxiv.md` - ArXiv 研究示例
* `pubmed.md` - PubMed 研究示例
* `inference-market.md` - 推理市场分析示例

### 手把手教学（deep_research_from_scratch/）
* 改文件夹下放的该项目进面向人类初学者的教学jupyter notebook内容，可视为与本项目的实际运行和使用完全解耦的模块

## 核心技术

* **LangGraph** - 工作流编排与图执行(Graph execution)
* **LangChain** - 大语言模型集成与工具调用(tool calling)
* **多个 LLM 提供商** - 支持 OpenAI、Anthropic、Google、Groq 和 DeepSeek
* **搜索 API** - 支持 Tavily、OpenAI/Anthropic 原生搜索、DuckDuckGo 和 Exa
* **MCP Servers** - 通过 Model Context Protocol 扩展智能体能力

## 开发命令

* `uvx langgraph dev` - 启动带有 LangGraph Studio 的开发服务器
* `python tests/run_evaluate.py` - 运行完整评估
* `ruff check` - 执行代码 lint 检查
* `mypy` - 执行类型检查

## 配置

所有设置均可通过以下方式进行配置：

* 环境变量（`.env` 文件）
* LangGraph Studio 中的 Web UI
* 直接修改配置文件

关键设置包括模型选择、搜索 API 选择、并发限制以及 MCP server 配置。
