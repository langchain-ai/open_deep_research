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

# Open Deep Research 智能体工作规范

## 启动流程（Startup Workflow）

开始修改前：

1. 确认位于仓库根目录并阅读 `README.md`。
2. 阅读 `feature_list.json`、`progress.md` 和 `session-handoff.md`。
3. 执行 `git status --short`，保留用户已有及无关改动。
4. 一次只处理一个功能（One feature at a time）。
5. 运行 `./init.sh`；若基线失败，先记录证据再修改。

缺少开发依赖时执行 `uv sync --extra dev`。

## 项目范围（Scope）

核心代码位于 `src/open_deep_research/`，认证代码位于 `src/security/`，
评估工具位于 `tests/`，遗留实现位于 `src/legacy/`。
`deep_research_from_scratch/` 是独立教学模块，除非当前功能明确要求，否则不要修改。

- 不得静默扩大功能范围。
- 不得覆盖或清理无关工作树改动。
- 不得提交 `.env`、API 密钥、私有 MCP 配置或敏感报告。

## 状态文件

- `feature_list.json`：功能 status、dependencies 和 evidence 的唯一来源。
- `progress.md`：记录当前状态、决定、阻塞项、文件和下一步。
- `session-handoff.md`：跨会话恢复入口。
- `init.sh`：统一验证入口。

状态仅使用 `not-started`、`in-progress`、`blocked`、`completed`。

## 验证命令（Verification Commands）

```bash
./init.sh
```

脚本执行源码编译、Ruff、mypy 和遗留测试收集。它不会运行
`python tests/run_evaluate.py`，因为完整评估依赖外部服务并可能产生费用。

## 完成标准（Definition of Done）

功能只有同时满足以下条件才算完成：

- 范围内行为已实现。
- 相关验证已运行并通过。
- 命令结果或其他证据已记录。
- 相关文档已更新。
- 状态文件足以让下一会话独立恢复。

## 会话结束（End of Session）

1. 更新功能 status 和 evidence。
2. 在 `progress.md` 记录验证、文件、阻塞项和下一步。
3. 未完成时更新 `session-handoff.md`。
4. 检查 `git status --short`。
5. 留下 clean、可重复验证且不依赖聊天记录的恢复路径。
