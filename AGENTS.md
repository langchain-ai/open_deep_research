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
