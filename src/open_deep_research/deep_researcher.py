"""Main LangGraph implementation for the Deep Research agent."""

# 这个文件实现了 Deep Research 代理的核心工作流。
# 整个流程可以理解成一个“多阶段状态图”：
# 1. 检查用户问题是否清楚。
# 2. 生成研究简报。
# 3. 由 supervisor 拆解和调度研究任务。
# 4. researcher 调用工具完成具体调查。
# 5. 汇总所有研究结果，生成最终报告。

# `asyncio` 是 Python 的异步并发库。
# 这里会用它并行执行多个 researcher 子任务。
import asyncio
import re
from datetime import datetime, timezone

# `Literal` 用来声明“只能返回固定几个字符串值”。
# 这里主要用于给 `Command[...]` 标注合法跳转目标。
from typing import Literal

# `init_chat_model` 用来创建可配置的大语言模型实例。
# 这个项目里所有节点基本都通过它来派生自己的模型。
from langchain.chat_models import init_chat_model

# 导入 LangChain 的消息类型和消息辅助函数。
# 这个项目以“消息列表”作为模型推理的主要输入格式。
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)

# `RunnableConfig` 表示运行时配置对象。
# 模型名、API key、token 限制等信息都会从这里读取。
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store

# `START` 和 `END` 是图的起点和终点。
# `StateGraph` 用来定义节点之间的连接关系。
from langgraph.graph import END, START, StateGraph

# `Command` 是 LangGraph 节点的标准返回对象：
# 它同时描述“下一步去哪”和“要更新哪些状态”。
from langgraph.types import Command

# 导入项目自己的配置类。
# 它负责把外部传入的运行参数转换成统一的 Python 对象。
from open_deep_research.configuration import (
    Configuration,
    MemoryWritePolicy,
)

# 导入各种 prompt 模板。
# 每个节点调用模型前都会用这些模板构造提示词。
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)

# 导入 `state.py` 里定义的状态类型和结构化输出模型。
# 这些类型控制着整个图里“数据长什么样”。
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)

# 导入各种辅助工具函数。
# 它们负责搜索判断、token 限制判断、消息裁剪、提取笔记等工作。
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# 初始化一个“可配置模型工厂”。
# 注意这里不是绑定某个固定模型，而是先创建一个模板；
# 后面每个节点再用 `.with_config(...)` 指定自己要用的模型参数。
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """Deduplicate non-empty strings while preserving original order."""
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _extract_preference_candidates(text: str) -> list[str]:
    """Extract lightweight session-level preference hints from a user message."""
    if not text:
        return []

    candidates: list[str] = []
    lowered = text.lower()

    keyword_patterns = [
        r"请用[^，。\n]+",
        r"希望[^，。\n]+",
        r"尽量[^，。\n]+",
        r"不要[^，。\n]+",
        r"prefer[^\n,.]+",
        r"please[^\n,.]+",
    ]
    for pattern in keyword_patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            cleaned = match.strip().strip("。,.，")
            if len(cleaned) >= 4:
                candidates.append(cleaned)

    if "中文" in text:
        candidates.append("用户偏好中文输出")
    if "英文" in text or "english" in lowered:
        candidates.append("用户偏好英文输出")
    if "简洁" in text or "concise" in lowered:
        candidates.append("用户偏好简洁表达")
    if "详细" in text or "detailed" in lowered:
        candidates.append("用户偏好详细说明")

    return _dedupe_keep_order(candidates)


def _build_session_memory_update(
    messages: list,
    verification_or_question: str,
    existing_preferences: list[str] | None = None,
    existing_key_context: list[str] | None = None,
) -> dict:
    """Build session memory update payload from conversation messages."""
    existing_preferences = existing_preferences or []
    existing_key_context = existing_key_context or []

    human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    latest_user_text = str(human_messages[-1].content) if human_messages else ""

    preference_candidates = _extract_preference_candidates(latest_user_text)
    merged_preferences = _dedupe_keep_order(existing_preferences + preference_candidates)

    context_candidates = [ctx for ctx in existing_key_context if isinstance(ctx, str)]
    if latest_user_text.strip():
        context_candidates.append(latest_user_text.strip())
    merged_context = _dedupe_keep_order(context_candidates)
    merged_context = merged_context[-5:]

    return {
        "session_clarification_summary": verification_or_question.strip() or None,
        "session_temporary_preferences": {
            "type": "override",
            "value": merged_preferences,
        },
        "session_key_context": {
            "type": "override",
            "value": merged_context,
        },
    }


def _build_session_memory_prompt_block(
    clarification_summary: str | None,
    temporary_preferences: list[str] | None,
    key_context: list[str] | None,
) -> str:
    """Format session memory for prompt injection before generation."""
    preferences = [p for p in (temporary_preferences or []) if p and str(p).strip()]
    context_items = [c for c in (key_context or []) if c and str(c).strip()]

    if not clarification_summary and not preferences and not context_items:
        return ""

    lines = ["<SessionMemory>"]
    if clarification_summary:
        lines.append(f"ClarificationSummary: {clarification_summary}")
    if preferences:
        lines.append("TemporaryPreferences:")
        lines.extend([f"- {item}" for item in preferences])
    if context_items:
        lines.append("KeyContext:")
        lines.extend([f"- {item}" for item in context_items[-3:]])
    lines.append("</SessionMemory>")
    return "\n".join(lines)


def _build_session_memory_clear_update() -> dict:
    """Build payload that clears short-term session memory at session end."""
    return {
        "session_clarification_summary": None,
        "session_temporary_preferences": {"type": "override", "value": []},
        "session_key_context": {"type": "override", "value": []},
    }


def _dedupe_memory_candidates(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deduplicate memory candidates by (kind, value) while preserving order."""
    seen: set[tuple[str, str]] = set()
    output: list[dict[str, str]] = []
    for candidate in candidates:
        kind = str(candidate.get("kind", "")).strip().lower()
        value = str(candidate.get("value", "")).strip()
        if not kind or not value:
            continue
        key = (kind, value)
        if key in seen:
            continue
        seen.add(key)
        output.append({"kind": kind, "value": value})
    return output


def _extract_long_term_memory_candidates(text: str, max_candidates: int) -> list[dict[str, str]]:
    """Extract stable long-term preference candidates from a user message."""
    if not text:
        return []

    candidates: list[dict[str, str]] = []
    lowered = text.lower()

    if "中文" in text:
        candidates.append({"kind": "language", "value": "中文"})
    if "英文" in text or "english" in lowered:
        candidates.append({"kind": "language", "value": "英文"})

    if "简洁" in text or "concise" in lowered:
        candidates.append({"kind": "style", "value": "简洁"})
    if "详细" in text or "detailed" in lowered:
        candidates.append({"kind": "style", "value": "详细"})

    focus_matches = re.findall(r"关注([^，。\n]+)", text)
    for match in focus_matches:
        normalized = match.strip()
        if normalized:
            candidates.append({"kind": "topic", "value": normalized})

    taboo_matches = re.findall(r"不要([^，。\n]+)", text)
    for match in taboo_matches:
        normalized = match.strip()
        if normalized:
            candidates.append({"kind": "taboo", "value": f"不要{normalized}"})

    deduped = _dedupe_memory_candidates(candidates)
    return deduped[:max_candidates]


def _is_explicit_memory_confirmation(text: str) -> bool:
    """Detect whether the user explicitly confirms long-term memory write."""
    if not text:
        return False

    confirmation_markers = [
        "确认记忆",
        "确认保存偏好",
        "确认写入记忆",
        "请记住",
        "记住这些偏好",
        "confirm memory",
        "confirm preference",
        "remember these preferences",
    ]
    lowered = text.lower()
    return any(marker in text or marker in lowered for marker in confirmation_markers)


def _resolve_memory_owner(configurable: Configuration, config: RunnableConfig) -> str | None:
    """Resolve memory owner ID from metadata.owner or configuration fallback."""
    owner = config.get("metadata", {}).get("owner") if config else None
    if owner:
        return str(owner)
    if configurable.user_id:
        return str(configurable.user_id)
    return None


def _build_long_term_memory_namespace(configurable: Configuration, owner: str) -> tuple[str, str, str]:
    """Build namespace tuple for long-term memory isolation."""
    return (configurable.memory_namespace_prefix, owner, "long_term")


def _format_long_term_preferences_for_state(preferences: list[dict[str, str]]) -> list[str]:
    """Format persistent preferences into a compact list for state observability."""
    return [f"{item.get('kind', '').strip()}: {item.get('value', '').strip()}" for item in preferences if item.get("kind") and item.get("value")]


def _build_long_term_memory_prompt_block(preferences: list[dict[str, str]]) -> str:
    """Build prompt block from confirmed long-term preferences."""
    if not preferences:
        return ""

    lines = ["<LongTermMemory>"]
    for item in preferences:
        kind = str(item.get("kind", "")).strip()
        value = str(item.get("value", "")).strip()
        if kind and value:
            lines.append(f"- {kind}: {value}")
    lines.append("</LongTermMemory>")
    return "\n".join(lines)


def _build_long_term_memory_clear_update() -> dict:
    """Build payload that clears session-scoped PR-5 fields without touching persisted store."""
    return {
        "memory_candidates_pending_confirmation": {"type": "override", "value": []},
        "confirmed_long_term_preferences": {"type": "override", "value": []},
    }


async def _load_long_term_preferences(
    configurable: Configuration,
    config: RunnableConfig,
    store_override=None,
) -> list[dict[str, str]]:
    """Load persistent long-term preferences for the resolved owner."""
    owner = _resolve_memory_owner(configurable, config)
    if not owner:
        return []

    store = store_override or get_store()
    namespace = _build_long_term_memory_namespace(configurable, owner)
    record = await store.aget(namespace, "profile")
    if not record:
        return []

    preferences = record.value.get("preferences", []) if isinstance(record.value, dict) else []
    if not isinstance(preferences, list):
        return []
    cleaned = [item for item in preferences if isinstance(item, dict)]
    return _dedupe_memory_candidates(cleaned)


async def _persist_long_term_preferences(
    candidates: list[dict[str, str]],
    configurable: Configuration,
    config: RunnableConfig,
    store_override=None,
) -> list[dict[str, str]]:
    """Persist confirmed long-term preferences with owner-level isolation."""
    if not candidates:
        return []

    owner = _resolve_memory_owner(configurable, config)
    if not owner:
        return []

    store = store_override or get_store()
    namespace = _build_long_term_memory_namespace(configurable, owner)

    existing = await _load_long_term_preferences(configurable, config, store_override=store)
    merged = _dedupe_memory_candidates(existing + candidates)

    payload = {
        "owner": owner,
        "preferences": merged,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    await store.aput(namespace, "profile", payload)
    return merged


async def _process_long_term_memory_turn(
    messages: list,
    existing_pending_candidates: list[dict[str, str]],
    existing_confirmed_preferences: list[str],
    configurable: Configuration,
    config: RunnableConfig,
    store_override=None,
) -> dict:
    """Generate candidates, handle explicit confirmation, and manage long-term memory updates."""
    if not configurable.memory_enabled:
        return {}

    existing_pending_candidates = [
        item for item in (existing_pending_candidates or []) if isinstance(item, dict)
    ]
    existing_confirmed_preferences = [
        item for item in (existing_confirmed_preferences or []) if isinstance(item, str)
    ]

    human_messages = [message for message in messages if isinstance(message, HumanMessage)]
    latest_user_text = str(human_messages[-1].content) if human_messages else ""

    extracted_candidates = _extract_long_term_memory_candidates(
        latest_user_text,
        configurable.memory_max_candidates_per_turn,
    )
    merged_pending = _dedupe_memory_candidates(existing_pending_candidates + extracted_candidates)
    merged_pending = merged_pending[: configurable.memory_max_candidates_per_turn]

    update = {
        "memory_candidates_pending_confirmation": {
            "type": "override",
            "value": merged_pending,
        },
        "confirmed_long_term_preferences": {
            "type": "override",
            "value": existing_confirmed_preferences,
        },
    }

    if configurable.memory_write_policy != MemoryWritePolicy.EXPLICIT_CONFIRMATION:
        return update

    if not _is_explicit_memory_confirmation(latest_user_text):
        return update

    candidates_to_persist = existing_pending_candidates or extracted_candidates
    persisted = await _persist_long_term_preferences(
        candidates_to_persist,
        configurable,
        config,
        store_override=store_override,
    )
    update["memory_candidates_pending_confirmation"] = {
        "type": "override",
        "value": [],
    }
    update["confirmed_long_term_preferences"] = {
        "type": "override",
        "value": _format_long_term_preferences_for_state(persisted),
    }
    return update

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # 第 1 步：读取运行时配置，确认是否允许向用户追问澄清问题。
    configurable = Configuration.from_runnable_config(config)
    messages = state.get("messages", [])

    long_term_memory_update = await _process_long_term_memory_turn(
        messages=messages,
        existing_pending_candidates=state.get("memory_candidates_pending_confirmation", []),
        existing_confirmed_preferences=state.get("confirmed_long_term_preferences", []),
        configurable=configurable,
        config=config,
    )

    if not configurable.allow_clarification:
        memory_update = _build_session_memory_update(
            messages,
            "Clarification disabled by configuration; proceeding with available user context.",
            state.get("session_temporary_preferences", []),
            state.get("session_key_context", []),
        )
        # 如果配置里禁止澄清，就直接跳过这一阶段，进入研究简报节点。
        return Command(goto="write_research_brief", update={**memory_update, **long_term_memory_update})

    # 组装本节点使用的模型配置。
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # 第 3 步：创建一个“结构化输出模型”。
    # 也就是说，模型输出必须符合 `ClarifyWithUser` 这个 Pydantic 结构。
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # 第 4 步：构造提示词，让模型判断“当前问题是否需要先澄清”。
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )

    # 发起异步模型调用。
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # 第 5 步：根据结构化结果决定下一跳。
    if response.need_clarification:
        memory_update = _build_session_memory_update(
            messages,
            response.question,
            state.get("session_temporary_preferences", []),
            state.get("session_key_context", []),
        )
        # 如果需要澄清，就把提问写成 AIMessage 并结束本轮流程，等待用户回复。
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)], **memory_update, **long_term_memory_update}
        )
    else:
        memory_update = _build_session_memory_update(
            messages,
            response.verification,
            state.get("session_temporary_preferences", []),
            state.get("session_key_context", []),
        )
        # 如果不需要澄清，就给出确认消息，然后进入研究简报生成阶段。
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)], **memory_update, **long_term_memory_update}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with initialized context
    """
    # 第 1 步：读取运行时配置。
    configurable = Configuration.from_runnable_config(config)

    # 为“研究简报生成”准备模型配置。
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # 第 2 步：把模型包装成结构化输出模式。
    # 这个节点要求模型返回 `ResearchQuestion` 结构，其中最关键的是 `research_brief`。
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # 第 3 步：把消息历史整理成 prompt，请模型生成一份更清晰的研究简报。
    session_memory_block = _build_session_memory_prompt_block(
        state.get("session_clarification_summary"),
        state.get("session_temporary_preferences", []),
        state.get("session_key_context", []),
    )
    confirmed_persistent_preferences: list[dict[str, str]] = []
    if configurable.memory_enabled:
        confirmed_persistent_preferences = await _load_long_term_preferences(
            configurable,
            config,
        )
    long_term_memory_block = _build_long_term_memory_prompt_block(confirmed_persistent_preferences)

    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    if session_memory_block:
        prompt_content = (
            f"{prompt_content}\n\nUse the following in-session memory to preserve user intent and style:\n"
            f"{session_memory_block}"
        )
    if long_term_memory_block:
        prompt_content = (
            f"{prompt_content}\n\nUse the following confirmed long-term preferences (persisted across sessions):\n"
            f"{long_term_memory_block}"
        )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # 第 4 步：构造 supervisor 的系统提示词。
    # supervisor 是“研究负责人”，会基于简报继续拆分任务。
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    # 第 5 步：进入 supervisor 子图。
    # 这里同时更新两个关键状态：
    # - `research_brief`：研究任务说明
    # - `supervisor_messages`：supervisor 的初始消息上下文
    #
    # 注意 `supervisor_messages` 使用了 `{"type": "override"}`，
    # 这意味着不是“追加”旧消息，而是“直接覆盖”成新的初始上下文。
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "confirmed_long_term_preferences": {
                "type": "override",
                "value": _format_long_term_preferences_for_state(confirmed_persistent_preferences),
            },
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # 第 1 步：读取配置。
    configurable = Configuration.from_runnable_config(config)

    # 为 supervisor 节点准备模型配置。
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # 第 2 步：定义 supervisor 允许调用的工具。
    # - `ConductResearch`：派 researcher 去查某个主题
    # - `ResearchComplete`：宣布研究阶段可以结束
    # - `think_tool`：记录自己的中间思考
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # 第 3 步：创建一个绑定工具的 supervisor 模型。
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # 第 4 步：读取 supervisor 目前为止的消息上下文。
    supervisor_messages = state.get("supervisor_messages", [])

    # 让模型基于当前上下文决定下一步操作。
    response = await research_model.ainvoke(supervisor_messages)
    
    # 第 5 步：把本轮 supervisor 的 AI 响应写回状态，
    # 并把研究轮次 `research_iterations` 加 1。
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # 第 1 步：读取 supervisor 当前状态和配置。
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)

    # 最近一条消息通常就是 supervisor 刚刚输出的、包含 tool_calls 的 AIMessage。
    most_recent_message = supervisor_messages[-1]
    
    # 第 2 步：判断研究阶段是否应该结束。
    # 满足以下任一条件都会结束：
    # - 已经超过允许的研究迭代次数
    # - 最近一条消息没有调用任何工具
    # - 最近一条消息调用了 `ResearchComplete`
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # 如果满足结束条件，就停止 supervisor 子图，
    # 同时从 supervisor 的消息历史中提取 `notes`，供最终报告使用。
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # 第 3 步：如果还没结束，就处理最近一条消息中的所有工具调用。
    # `all_tool_messages` 用来收集这轮工具的所有输出结果。
    all_tool_messages = []

    # `update_payload` 是这个节点最终要写回状态的内容。
    update_payload = {"supervisor_messages": []}
    
    # 第 4 步：处理 `think_tool` 调用。
    # 这类调用不会联网搜索，只是把 supervisor 的思考内容记录下来。
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    # 把每条反思结果包装成 ToolMessage，后面会追加回 supervisor 的消息历史。
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # 第 5 步：处理 `ConductResearch` 调用。
    # 这类调用表示 supervisor 要派 researcher 去调查具体子主题。
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            # 为了防止一次启动过多 researcher，这里做并发数量限制。
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            
            # 第 6 步：把每个研究任务变成一个 researcher 子图调用。
            # 每个 researcher 的初始状态都包含：
            # - `researcher_messages`：先塞一个 HumanMessage，内容就是研究主题
            # - `research_topic`：当前 researcher 要研究的具体主题
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]
            
            # 使用 `asyncio.gather` 并行等待所有 researcher 完成。
            tool_results = await asyncio.gather(*research_tasks)
            
            # 第 7 步：把每个 researcher 返回的压缩研究结果包装成 ToolMessage。
            # 这样 supervisor 下一轮推理时就能读到这些结果。
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # 第 8 步：如果有超出并发上限的任务，就为这些任务生成报错消息。
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # 第 9 步：把所有 researcher 返回的原始笔记拼接起来。
            # 这些 `raw_notes` 是底层研究材料的集合。
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            # 如果确实有原始笔记，就把它们写回 supervisor 状态。
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # 如果 researcher 子图执行出错，就提前结束研究阶段。
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # 这里实际上因为 `or True` 的存在，会对任何异常都直接结束。
                # 这是一种保守的容错策略。
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )
    
    # 第 10 步：把本轮所有工具输出写回 `supervisor_messages`，
    # 然后返回到 `supervisor` 节点，让它继续基于新结果做决策。
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# 下面开始构建 supervisor 子图。
# 这个子图负责“管理和协调研究任务”。
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# 把两个核心节点注册到 supervisor 子图中。
supervisor_builder.add_node("supervisor", supervisor)           # 负责思考和调度
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # 负责执行工具调用

# 定义 supervisor 子图入口：一进入子图就先跑 `supervisor`。
supervisor_builder.add_edge(START, "supervisor")

# 编译子图，供主图中的 `research_supervisor` 节点调用。
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # 第 1 步：读取配置，并取出 researcher 当前的消息状态。
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # 第 2 步：加载 researcher 当前可用的所有工具。
    # 这些工具可能包括搜索 API、MCP 工具等。
    tools = await get_all_tools(config)
    if len(tools) == 0:
        # 如果一个工具都没有，就无法做研究，直接抛出错误。
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # 第 3 步：准备 researcher 模型配置。
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # 第 4 步：构造 researcher 的系统提示词。
    # 如果配置里有 `mcp_prompt`，这里会一起注入上下文。
    rag_usage_instructions = ""
    if configurable.rag_enabled:
        rag_usage_instructions = (
            "3. **rag_search**: Search the local knowledge base first when the topic may rely on personal or internal documents"
        )

    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "",
        rag_usage_instructions=rag_usage_instructions,
        date=get_today_str(),
    )
    
    # 第 5 步：创建一个绑定了工具的 researcher 模型。
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # 第 6 步：把系统消息放到最前面，再拼接 researcher 的消息历史。
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages

    # 让模型决定下一步是调用工具、继续思考，还是准备结束研究。
    response = await research_model.ainvoke(messages)
    
    # 第 7 步：把本轮 AI 响应写回 `researcher_messages`，
    # 并把工具调用轮次 `tool_call_iterations` 加 1。
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# 这是一个辅助函数，负责安全执行单个工具。
# 它的作用是：即使工具失败，也尽量把错误转成字符串返回，而不是直接中断整个流程。
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        # 正常情况下，异步执行工具。
        return await tool.ainvoke(args, config)
    except Exception as e:
        # 如果工具报错，就把错误包装成普通文本。
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # 第 1 步：读取 researcher 当前状态和配置。
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # 第 2 步：判断最近一条 researcher 消息里是否真的触发了研究动作。
    # 这里要同时检查两类情况：
    # - 显式工具调用：体现在 `tool_calls` 中
    # - 原生搜索：某些模型会在底层触发 web search，但不出现在 `tool_calls` 里
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    # 如果两种研究动作都没有发生，就说明 researcher 已经没有更多要查的内容，
    # 可以直接进入压缩总结阶段。
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    
    # 第 3 步：重新加载工具，并建立“工具名 -> 工具对象”的映射表。
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # 第 4 步：取出最近一条消息中的所有工具调用，并并发执行。
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # 第 5 步：把每个工具执行结果包装成 ToolMessage。
    # 这些消息会被追加回 researcher 的消息历史，供下一轮推理使用。
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # 第 6 步：执行完工具后，再判断是否该结束 researcher 的循环。
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # 如果达到最大轮数，或者 researcher 主动宣布完成，就进入压缩阶段。
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # 否则，把工具输出加入消息历史，回到 `researcher` 节点继续研究。
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # 第 1 步：读取配置，并准备一个专门负责“压缩总结”的模型。
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    
    # 第 2 步：取出 researcher 在整个研究过程中累积的消息历史。
    researcher_messages = state.get("researcher_messages", [])
    
    # 追加一条 HumanMessage，明确告诉模型：
    # 现在不要继续研究了，而是开始压缩和总结已有材料。
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # 第 3 步：准备带重试的压缩逻辑。
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # 构造压缩任务的系统提示词。
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # 调用模型执行压缩总结。
            response = await synthesizer_model.ainvoke(messages)
            
            # 从消息历史里提取所有 tool/ai 消息，作为 `raw_notes` 返回。
            # 这相当于保留一份较原始的研究过程记录。
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # 压缩成功后，返回 researcher 子图的最终输出。
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            # 如果失败，先记录本次尝试失败。
            synthesis_attempts += 1
            
            # 如果失败原因是 token 超限，就删除一部分较早的 AI 消息，再继续尝试。
            # 这个策略是“牺牲部分旧上下文，换取模型还能继续工作”。
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # 对其他错误，这里也继续重试，尽量提高成功率。
            continue
    
    # 第 4 步：如果多次重试都失败，就返回一个失败结果，
    # 同时尽量保留 `raw_notes`，避免研究过程完全丢失。
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# 下面开始构建 researcher 子图。
# 这个子图对应“单个研究员”的完整工作流。
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# 注册 researcher 子图的节点。
researcher_builder.add_node("researcher", researcher)                 # researcher 决策节点
researcher_builder.add_node("researcher_tools", researcher_tools)     # researcher 工具执行节点
researcher_builder.add_node("compress_research", compress_research)   # researcher 压缩总结节点

# 定义 researcher 子图的边。
researcher_builder.add_edge(START, "researcher")           # 子图入口
researcher_builder.add_edge("compress_research", END)      # 一旦压缩完成就结束子图

# 编译 researcher 子图，供 supervisor 并行调用。
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # 第 1 步：从主状态中读取 supervisor 汇总好的 `notes`。
    notes = state.get("notes", [])

    # 这里预先构造一个“清空 notes”的更新对象。
    # 因为 `notes` 在 state.py 里不是普通赋值，而是通过 reducer 合并，
    # 所以要用 `{"type": "override"}` 明确表示“直接覆盖成空列表”。
    cleared_state = {
        "notes": {"type": "override", "value": []},
        **_build_session_memory_clear_update(),
        **_build_long_term_memory_clear_update(),
    }

    # 把多条笔记拼成一个大字符串，供写报告模型使用。
    findings = "\n".join(notes)
    
    # 第 2 步：读取配置，并准备最终写报告用的模型参数。
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # 第 3 步：带重试逻辑地生成最终报告。
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # 构造最终报告 prompt。
            # 它会同时注入：
            # - `research_brief`：原始研究目标
            # - `messages`：用户对话上下文
            # - `findings`：研究结果摘要
            # - `date`：当前日期
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # 调用模型生成最终报告。
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            
            # 成功后返回：
            # - `final_report`：报告正文
            # - `messages`：把这次生成结果也放回消息历史
            # - `cleared_state`：清空 notes
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state
            }
            
        except Exception as e:
            # 如果报错，并且错误原因是 token 超限，就尝试缩短 `findings` 后重试。
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # 第一次重试时，先获取模型上下文长度上限。
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # 这里用 “token 上限 * 4” 作为字符数近似值来做截断。
                    findings_token_limit = model_token_limit * 4
                else:
                    # 后续每次重试都在上一次基础上再缩短 10%。
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # 截断研究结果文本后继续下一轮尝试。
                findings = findings[:findings_token_limit]
                continue
            else:
                # 如果不是 token 超限，而是别的异常，就直接返回错误结果。
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # 第 4 步：如果所有重试都失败，返回统一的失败结果。
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

# 下面开始构建主图，也就是完整的 Deep Research agent。
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# 把主图中的核心节点注册进去。
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # 判断是否需要澄清
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # 生成研究简报
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # 进入 supervisor 子图
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # 生成最终报告

# 定义主图的固定边：
# - 从 START 进入 `clarify_with_user`
# - supervisor 子图完成后进入最终报告生成
# - 报告生成完成后结束
#
# 另外要注意：
# `clarify_with_user` 和 `write_research_brief` 的下一跳并不完全写死在这里，
# 而是在各自函数里通过 `Command(goto=...)` 动态决定。
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# 把整个主图编译成一个可执行对象。
deep_researcher = deep_researcher_builder.compile()
