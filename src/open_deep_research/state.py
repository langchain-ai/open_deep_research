"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# 下面这个分隔注释只是为了把文件结构分成“结构化输出定义”和“状态定义”两大部分。
###################
# Structured Outputs
###################


# 定义一个结构化输出模型：要求系统去执行一次研究任务。
# 这个类通常会被模型当成“工具调用参数”的 schema 来使用。
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""

    # `research_topic` 是研究主题。
    # 它要求是字符串，并且在描述里明确要求：只能是单个主题，而且要写得足够详细。
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )


# 定义一个结构化输出模型：表示“研究已经完成”。
# 这个类没有字段，本质上是一个“空信号”，只用来表达某个状态。
class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""


# 定义“研究总结”的结构化输出模型。
# 它用来规范研究人员节点最终应该返回什么内容。
class Summary(BaseModel):
    """Research summary with key findings."""

    # `summary` 存放总结后的主要结论。
    summary: str

    # `key_excerpts` 存放关键摘录，通常是从原始资料里抽出的重点内容。
    key_excerpts: str


# 定义一个“向用户澄清需求”的结构化输出模型。
# 当用户问题太模糊时，系统可以生成这个对象来决定是否先追问。
class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""

    # `need_clarification` 是布尔值，表示是否真的需要向用户再问一个澄清问题。
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )

    # `question` 是要问用户的具体问题。
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )

    # `verification` 是一段确认话术。
    # 含义是：等用户补充了必要信息之后，系统会继续开始研究。
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )


# 定义“研究问题/研究简报”的结构化输出模型。
# 这个对象通常作为后续研究阶段的指导输入。
class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""

    # `research_brief` 是整理后的研究任务说明。
    # 它会告诉后续 researcher 节点到底应该围绕什么问题去调查。
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


# 第二部分：状态定义。
# 这里定义的是工作流在运行中如何保存和传递数据。
###################
# State Definitions
###################


# 定义一个自定义 reducer 函数。
# 在 LangGraph 里，如果多个节点同时更新同一个状态字段，就需要 reducer 来决定“怎么合并”。
def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    # 如果 `new_value` 是一个字典，并且其中 `type` 字段等于 `"override"`，
    # 那就说明这次更新不是“追加”，而是“直接覆盖旧值”。
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        # 优先取字典中的 `value` 字段作为真正的新值。
        # 如果没有 `value`，就退回整个 `new_value` 本身。
        return new_value.get("value", new_value)
    else:
        # 否则，默认使用 `operator.add` 来合并。
        # 对列表来说是拼接，对字符串来说是连接。
        return operator.add(current_value, new_value)


# 定义 Agent 的“输入状态”。
# 继承 `MessagesState` 后，这个状态只需要保留消息上下文即可。
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""


# 定义主 Agent 在整个运行过程中的完整状态。
# 这个类会随着流程推进不断累积消息、研究摘要和最终报告。
class AgentState(MessagesState):
    """Main agent state containing messages and research data."""

    # `supervisor_messages` 保存 supervisor 视角下的消息列表。
    # 使用 `Annotated[..., override_reducer]` 表示这个字段在状态合并时使用自定义 reducer。
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]

    # `research_brief` 保存整理后的研究任务说明。
    # `Optional[str]` 表示这个值可能暂时还没有生成，因此允许为 `None`。
    research_brief: str | None

    # `session_clarification_summary` 保存会话内最新澄清结论。
    # 该字段用于在后续轮次中快速复用用户已确认过的约束。
    session_clarification_summary: str | None = None

    # `session_temporary_preferences` 保存会话级临时偏好。
    # 例如语言、输出风格、关注重点等，不会跨会话持久化。
    session_temporary_preferences: Annotated[list[str], override_reducer] = []

    # `session_key_context` 保存会话内关键上下文片段，供后续生成阶段复用。
    session_key_context: Annotated[list[str], override_reducer] = []

    # `memory_candidates_pending_confirmation` 保存待用户确认写入的长期记忆候选。
    # 候选内容仅来自稳定偏好（语言、输出风格、关注主题、禁忌项）。
    memory_candidates_pending_confirmation: Annotated[list[dict[str, str]], override_reducer] = []

    # `confirmed_long_term_preferences` 是本轮已确认或已读取的长期偏好快照。
    # 它用于提示构建和可观测性，不直接作为跨会话存储。
    confirmed_long_term_preferences: Annotated[list[str], override_reducer] = []

    # `raw_notes` 保存原始研究笔记。
    # 默认值是空列表；合并时使用 `override_reducer`，这样既能追加，也能按需整体覆盖。
    raw_notes: Annotated[list[str], override_reducer] = []

    # `notes` 保存进一步整理后的笔记。
    # 它和 `raw_notes` 类似，只是语义上更偏向清洗过、可直接利用的内容。
    notes: Annotated[list[str], override_reducer] = []

    # `final_report` 保存最终生成的完整报告文本。
    final_report: str

    # `api_response` 保存 API 友好的结构化返回体。
    api_response: dict | None = None


# 定义 supervisor 节点使用的状态结构。
# 这里用 `TypedDict`，说明它本质上是一个受类型约束的字典。
class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""

    # `supervisor_messages` 是 supervisor 节点看到的消息历史。
    # 多次更新这个字段时，采用上面定义的 `override_reducer` 来合并。
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]

    # `research_brief` 是当前研究任务的简报。
    # supervisor 会基于它来决定派发哪些研究任务。
    research_brief: str

    # `notes` 保存 supervisor 汇总后的研究笔记。
    # 默认是空列表。
    notes: Annotated[list[str], override_reducer] = []

    # `research_iterations` 记录 supervisor 已经进行了多少轮研究调度。
    # 默认从 0 开始，用于限制循环次数或判断是否该结束。
    research_iterations: int = 0

    # `raw_notes` 保存还没压缩、还没深度整理的原始笔记。
    raw_notes: Annotated[list[str], override_reducer] = []


# 定义单个 researcher 节点使用的状态结构。
# 每个 researcher 通常只负责一个具体研究主题。
class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""

    # `researcher_messages` 保存 researcher 自己这条支线上的消息历史。
    # 这里用 `operator.add`，表示多个消息更新会直接拼接到列表后面。
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]

    # `tool_call_iterations` 表示这个 researcher 已经调用过多少轮工具。
    # 默认从 0 开始，通常用于避免无限调用搜索工具。
    tool_call_iterations: int = 0

    # `research_topic` 是这个 researcher 当前负责调查的具体主题。
    research_topic: str

    # `compressed_research` 保存压缩后的研究结果。
    # 一般是 researcher 对搜集材料做过总结提炼后的文本。
    compressed_research: str

    # `raw_notes` 保存 researcher 收集到的原始笔记。
    raw_notes: Annotated[list[str], override_reducer] = []


# 定义 researcher 节点结束后返回的输出模型。
# 这里使用 `BaseModel`，方便把输出结构标准化。
class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""

    # `compressed_research` 是 researcher 最终交回来的压缩研究结果。
    compressed_research: str

    # `raw_notes` 是附带返回的原始笔记列表。
    raw_notes: Annotated[list[str], override_reducer] = []
