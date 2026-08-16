def plan_paper_queries(question: str, max_queries: int = 8) -> list[str]:
    normalized = question.strip()
    lower = normalized.lower()

    if any(term in normalized for term in ["记忆", "memory"]):
        candidates = [
            "long-term memory for LLM agents",
            "memory mechanisms in autonomous agents",
            "retrieval augmented memory for agent planning",
            "episodic memory in language model agents",
            "multi-agent systems memory planning",
            "agent memory benchmark planning",
            "LLM agent memory systems survey",
            "memory augmented language model agents",
        ]
    elif "tool" in lower or "工具" in normalized:
        candidates = [
            "tool use in language model agents",
            "LLM agents tool learning evaluation",
            "function calling agents benchmark",
            "tool augmented language models",
            "agent tool use planning",
            "API use by autonomous LLM agents",
            "tool use reasoning language models",
            "LLM tool use survey",
        ]
    else:
        topic = _english_fallback_topic(normalized)
        candidates = [
            f"{topic} computer science paper",
            f"{topic} LLM agents",
            f"{topic} survey",
            f"{topic} benchmark",
            f"{topic} method evaluation",
            f"{topic} arXiv",
            f"{topic} empirical study",
            f"{topic} recent advances",
        ]

    return _deduplicate(candidates)[: max(1, min(max_queries, 8))]


def _english_fallback_topic(question: str) -> str:
    if not _contains_cjk(question):
        return question

    translated_terms = [
        english
        for chinese, english in [
            ("强化学习", "reinforcement learning"),
            ("代码生成", "code generation"),
            ("大语言模型", "large language models"),
            ("语言模型", "language models"),
            ("多智能体", "multi-agent systems"),
            ("智能体", "agents"),
            ("检索增强", "retrieval augmented generation"),
            ("任务规划", "task planning"),
            ("规划", "planning"),
            ("评估", "evaluation"),
            ("基准", "benchmark"),
            ("局限", "limitations"),
            ("应用", "applications"),
        ]
        if chinese in question
    ]

    return " ".join(translated_terms) if translated_terms else "computer science research topic"


def _contains_cjk(text: str) -> bool:
    return any("\u3400" <= character <= "\u9fff" for character in text)


def _deduplicate(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        key = query.lower().strip()
        if key not in seen:
            unique.append(query)
            seen.add(key)
    return unique
