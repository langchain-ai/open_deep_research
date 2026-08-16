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
        candidates = [
            f"{normalized} computer science paper",
            f"{normalized} LLM agents",
            f"{normalized} survey",
            f"{normalized} benchmark",
            f"{normalized} method evaluation",
            f"{normalized} arXiv",
            f"{normalized} empirical study",
            f"{normalized} recent advances",
        ]

    return _deduplicate(candidates)[: max(1, min(max_queries, 8))]


def _deduplicate(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        key = query.lower().strip()
        if key not in seen:
            unique.append(query)
            seen.add(key)
    return unique
