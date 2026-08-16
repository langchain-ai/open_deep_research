import re

from open_deep_research.paper_research.query_planner import plan_paper_queries


CJK_RE = re.compile(r"[\u3400-\u9fff]")


def assert_no_cjk(queries: list[str]) -> None:
    assert all(not CJK_RE.search(query) for query in queries)


def test_plan_paper_queries_returns_english_agent_memory_queries():
    queries = plan_paper_queries("多智能体系统中的记忆机制如何影响任务规划能力？")

    assert 5 <= len(queries) <= 8
    assert any("memory" in query.lower() for query in queries)
    assert any("agent" in query.lower() for query in queries)
    assert all(query.strip() for query in queries)
    assert_no_cjk(queries)


def test_plan_paper_queries_returns_english_fallback_queries_for_chinese_question():
    queries = plan_paper_queries("强化学习在代码生成中的应用有哪些局限？")

    assert 5 <= len(queries) <= 8
    assert any("computer science paper" in query.lower() for query in queries)
    assert any("survey" in query.lower() for query in queries)
    assert all(query.strip() for query in queries)
    assert_no_cjk(queries)


def test_plan_paper_queries_limits_count():
    queries = plan_paper_queries("LLM agent tool use evaluation", max_queries=5)

    assert len(queries) == 5
